"""Alicia-D gamepad teleoperation + automatic demonstration collection.

The viewer shows the vendor Alicia-D MJCF. A gamepad moves the absolute TCP
target; recording starts on the first meaningful input and stops/saves after a
short idle period (or with A / Start). B cancels, Back resets the scene.

Windows path uses XInput directly (no extra dependency). Keyboard fallback:
W/S x, A/D y, Q/E z, Z/X yaw, O/C gripper, Space pause, R reset, T record.

Examples:
    python scripts/alicia_teleop_collect.py
    python scripts/alicia_teleop_collect.py --no-auto-record --stop-idle 2.0
    python scripts/alicia_teleop_collect.py --self-test
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure") and not sys.stdout.isatty():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if platform.system() == "Windows":
    os.environ.setdefault("MUJOCO_GL", "wgl")

from kite.robot.alicia import load_config  # noqa: E402
from kite.sim.alicia import AliciaSimEnv  # noqa: E402
from scripts.collect_alicia_episodes import Recorder  # noqa: E402

# XInput button bit masks.
DPAD_UP, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT = 0x0001, 0x0002, 0x0004, 0x0008
START, BACK = 0x0010, 0x0020
LEFT_THUMB, RIGHT_THUMB = 0x0040, 0x0080
LB, RB = 0x0100, 0x0200
A, B, X, Y = 0x1000, 0x2000, 0x4000, 0x8000


@dataclass
class GamepadState:
    buttons: int
    lx: float
    ly: float
    rx: float
    ry: float
    lt: float
    rt: float


class _XInputGamepad(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.c_ushort),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class _XInputState(ctypes.Structure):
    _fields_ = [("dwPacketNumber", ctypes.c_ulong), ("Gamepad", _XInputGamepad)]


class _JoyInfoEx(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("dwXpos", ctypes.c_ulong),
        ("dwYpos", ctypes.c_ulong),
        ("dwZpos", ctypes.c_ulong),
        ("dwRpos", ctypes.c_ulong),
        ("dwUpos", ctypes.c_ulong),
        ("dwVpos", ctypes.c_ulong),
        ("dwButtons", ctypes.c_ulong),
        ("dwButtonNumber", ctypes.c_ulong),
        ("dwPOV", ctypes.c_ulong),
        ("dwReserved1", ctypes.c_ulong),
        ("dwReserved2", ctypes.c_ulong),
    ]


class WinMMGamepad:
    """Fallback for DirectInput-compatible controllers via winmm."""

    JOY_RETURNALL = 0xFF
    JOYSTICKID1 = 0

    def __init__(self, index: int = 0) -> None:
        self.index = index if index else self.JOYSTICKID1
        self.dll: Any | None = None
        if platform.system() != "Windows":
            return
        try:
            self.dll = ctypes.windll.LoadLibrary("winmm.dll")
        except OSError:
            self.dll = None

    @property
    def available(self) -> bool:
        return self.dll is not None

    def read(self) -> GamepadState | None:
        if self.dll is None:
            return None
        info = _JoyInfoEx()
        info.dwSize = ctypes.sizeof(_JoyInfoEx)
        info.dwFlags = self.JOY_RETURNALL
        if self.dll.joyGetPosEx(self.index, ctypes.byref(info)) != 0:
            return None

        def axis(value: int) -> float:
            return max(-1.0, min(1.0, (value - 32767.0) / 32767.0))

        return GamepadState(
            buttons=int(info.dwButtons),
            lx=axis(info.dwXpos),
            ly=axis(info.dwYpos),
            rx=axis(info.dwRpos),
            ry=axis(info.dwZpos),
            lt=max(0.0, min(1.0, (info.dwUpos or 0) / 65535.0)),
            rt=max(0.0, min(1.0, (info.dwVpos or 0) / 65535.0)),
        )


class GamepadReader:
    """Try XInput first, then the winmm/DirectInput fallback."""

    def __init__(self, index: int = 0) -> None:
        self.xinput = XInputGamepad(index)
        self.winmm = WinMMGamepad(index)

    @property
    def available(self) -> bool:
        return self.xinput.available or self.winmm.available

    def read(self) -> GamepadState | None:
        state = self.xinput.read()
        if state is not None:
            return state
        return self.winmm.read()


class XInputGamepad:
    """Minimal XInput reader (xinput1_4 / xinput1_3)."""

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.dll: Any | None = None
        if platform.system() != "Windows":
            return
        for name in ("xinput1_4.dll", "xinput1_3.dll"):
            try:
                self.dll = ctypes.windll.LoadLibrary(name)
                break
            except OSError:
                continue

    @property
    def available(self) -> bool:
        return self.dll is not None

    def read(self) -> GamepadState | None:
        if self.dll is None:
            return None
        state = _XInputState()
        if self.dll.XInputGetState(self.index, ctypes.byref(state)) != 0:
            return None
        pad = state.Gamepad

        def axis(value: int) -> float:
            return max(-1.0, min(1.0, value / 32767.0))

        return GamepadState(
            buttons=int(pad.wButtons),
            lx=axis(pad.sThumbLX),
            ly=axis(pad.sThumbLY),
            rx=axis(pad.sThumbRX),
            ry=axis(pad.sThumbRY),
            lt=pad.bLeftTrigger / 255.0,
            rt=pad.bRightTrigger / 255.0,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alicia-D 手柄遥操作采集")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/alicia_teleop"))
    parser.add_argument("--instruction", default="teleoperated pick and place")
    parser.add_argument("--controller", type=int, default=0)
    parser.add_argument("--speed-mps", type=float, default=0.15, help="平移速度 m/s")
    parser.add_argument("--yaw-rps", type=float, default=0.8, help="偏航速度 rad/s")
    parser.add_argument("--deadzone", type=float, default=0.15)
    parser.add_argument("--stop-idle", type=float, default=2.0, help="无输入多久自动停止")
    parser.add_argument("--frame-every", type=int, default=2)
    parser.add_argument("--no-auto-record", action="store_true")
    parser.add_argument("--no-wrist-inset", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _episode_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root / stamp
    suffix = 0
    while path.exists():
        suffix += 1
        path = root / f"{stamp}-{suffix}"
    return path


class TeleopCollector:
    def __init__(self, env: AliciaSimEnv, args: argparse.Namespace) -> None:
        self.env = env
        self.args = args
        self.gamepad = GamepadReader(args.controller)
        self.prev_buttons = 0
        self.recorder: Recorder | None = None
        self.episode_dir: Path | None = None
        self.last_input = 0.0
        self.keyboard_last = 0.0
        self.grip = 0.0
        self.frames = 0
        self.saved = 0
        self.cancelled = 0
        self.last_meta: dict[str, Any] | None = None

    def _active(self, pad: GamepadState | None) -> bool:
        if pad is None:
            return False
        return (
            max(abs(pad.lx), abs(pad.ly), abs(pad.rx), abs(pad.ry), pad.lt, pad.rt)
            > self.args.deadzone
        )

    def start(self) -> None:
        if self.recorder is not None:
            return
        self.episode_dir = _episode_dir(self.args.out)
        self.recorder = Recorder(self.env, self.episode_dir, self.args.frame_every)
        self.frames = 0
        self.last_input = time.monotonic()
        print(f"REC start {self.episode_dir}", flush=True)

    def stop(self, saved: bool = True) -> None:
        if self.recorder is None:
            return
        recorder = self.recorder
        episode_dir = self.episode_dir
        self.recorder = None
        self.episode_dir = None
        if saved and self.frames >= 5 and episode_dir is not None:
            recorder.save()
            meta = {
                "episode_id": episode_dir.name,
                "started": episode_dir.name,
                "instruction": self.args.instruction,
                "fps": 20,
                "action_layout": "[ee_xyz(3), rot6d(6), grip(1)]",
                "rot6d": "interleaved",
                "units": {"length": "m", "angle": "rad", "gripper": "0_1_closed_is_1"},
                "renderer": "mujoco",
                "depth_unit": "none",
                "camera_intrinsics": {
                    "front": {"width": 224, "height": 224, "fovy_deg": 48},
                    "wrist": {"width": 224, "height": 224, "fovy_deg": 75},
                },
                "camera_extrinsics": {"front": "world fixed", "wrist": "link6 +0.06 x"},
                "source": "gamepad_teleop",
                "controller": self.args.controller,
                "frames": self.frames,
                "split": "train",
                "success": None,
            }
            (episode_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.saved += 1
            self.last_meta = meta
            print(f"SAVE {episode_dir} frames={self.frames}", flush=True)
        else:
            self.cancelled += 1
            print("CANCEL episode", flush=True)

    def toggle(self) -> None:
        if self.recorder is None:
            self.start()
        else:
            self.stop(saved=True)

    def _handle_buttons(self, pad: GamepadState) -> None:
        pressed = pad.buttons & ~self.prev_buttons
        self.prev_buttons = pad.buttons
        if pressed & (A | START):
            self.toggle()
        if pressed & B:
            self.stop(saved=False)
        if pressed & X:
            self.grip = 0.0
        if pressed & Y:
            self.grip = 1.0
        if pressed & BACK:
            self.stop(saved=False)
            self.env.reset()

    def step(self, pad: GamepadState | None) -> None:
        now = time.monotonic()
        dt = self.env.control_period
        if pad is not None:
            self._handle_buttons(pad)
            active = self._active(pad)
            if pad.rt > pad.lt and pad.rt > 0.1:
                self.grip = 1.0
            elif pad.lt > pad.rt and pad.lt > 0.1:
                self.grip = 0.0
            if active:
                self.last_input = now
                dx = -pad.ly * self.args.speed_mps * dt
                dy = -pad.lx * self.args.speed_mps * dt
                dz = -pad.ry * self.args.speed_mps * dt
                dyaw = -pad.rx * self.args.yaw_rps * dt
                if max(abs(pad.lx), abs(pad.ly), abs(pad.rx), abs(pad.ry)) > self.args.deadzone:
                    self.env.nudge(dx=dx, dy=dy, dz=dz, dyaw=dyaw)
        else:
            active = False
        if now - self.keyboard_last < 0.25:
            active = True
        if active:
            self.last_input = now
        auto = not self.args.no_auto_record
        if auto and self.recorder is None and active:
            self.start()
        if (
            auto
            and self.recorder is not None
            and not active
            and now - self.last_input > self.args.stop_idle
        ):
            self.stop(saved=True)
        self.env.command_gripper_norm(self.grip)
        self.env.step()
        if self.recorder is not None:
            self.frames += 1
            self.recorder.record(self.grip)


def _run_self_test(args: argparse.Namespace) -> int:
    pad = GamepadReader(args.controller)
    if not pad.available:
        print("gamepad backend unavailable")
        return 1
    state = pad.read()
    print(
        json.dumps(
            {"connected": state is not None, "state": None if state is None else state.__dict__},
            ensure_ascii=False,
        )
    )
    return 0


def _run_viewer(collector: TeleopCollector, args: argparse.Namespace) -> int:
    import mujoco.viewer

    env = collector.env
    paused = False
    running = True
    started = time.monotonic()

    def key_callback(keycode: int) -> None:
        nonlocal paused, running
        key = chr(keycode).lower() if 32 <= keycode < 127 else ""
        if keycode == 256:
            running = False
        elif key == " ":
            paused = not paused
        elif key == "r":
            collector.stop(saved=False)
            env.reset()
        elif key == "t":
            collector.toggle()
        elif key == "h":
            env.set_target(env.eef_pose()[0])
        elif key in ("w", "s", "a", "d", "q", "e", "z", "x"):
            step = 0.01
            if key == "w":
                env.nudge(dx=-step)
            elif key == "s":
                env.nudge(dx=step)
            elif key == "a":
                env.nudge(dy=step)
            elif key == "d":
                env.nudge(dy=-step)
            elif key == "q":
                env.nudge(dz=step)
            elif key == "e":
                env.nudge(dz=-step)
            elif key == "z":
                env.nudge(dyaw=step)
            else:
                env.nudge(dyaw=-step)
            collector.keyboard_last = time.monotonic()
            collector.last_input = collector.keyboard_last
        elif key == "o":
            collector.grip = 0.0
        elif key == "c":
            collector.grip = 1.0

    viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = env.camera_ids["front"]
    last_hud = 0.0
    try:
        while running and viewer.is_running():
            now = time.monotonic()
            if args.duration > 0.0 and now - started > args.duration:
                break
            pad = collector.gamepad.read()
            if not paused:
                collector.step(pad)
            if now - last_hud > 0.2:
                last_hud = now
                state = env.state()
                mode = "REC" if collector.recorder is not None else "IDLE"
                viewer.set_texts(
                    [
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            "Alicia-D teleop",
                            f"{mode} | saved {collector.saved} | {collector.frames} frames",
                        ),
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                            (
                                f"tcp {np.round(state.position, 3)} | "
                                f"err {state.position_error * 1000:.1f} mm"
                            ),
                            f"grip {collector.grip:.2f} | pad {'on' if pad else 'off'}",
                        ),
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                            "LS move | RS z/yaw | RT close | LT open",
                            "A rec/save  B cancel  Back reset",
                        ),
                    ]
                )
            if not args.no_wrist_inset:
                image = env.render("wrist")
                viewer.set_images([(mujoco.MjrRect(8, 8, image.shape[1], image.shape[0]), image)])
            viewer.sync()
            remaining = env.control_period - (time.monotonic() - now)
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        collector.stop(saved=True)
        viewer.close()
    return 0


def main() -> int:
    args = parse_args()
    if not args.out.is_absolute():
        args.out = REPO_ROOT / args.out
    if args.self_test:
        return _run_self_test(args)
    config = load_config(args.config)
    env = AliciaSimEnv(config=config, with_task=True, seed=args.seed)
    collector = TeleopCollector(env, args)
    try:
        return _run_viewer(collector, args)
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
