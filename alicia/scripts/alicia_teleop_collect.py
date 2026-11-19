"""Alicia-D gamepad teleoperation + automatic demonstration collection.

The viewer shows the vendor Alicia-D MJCF. A gamepad moves the absolute TCP
target; recording starts on the first meaningful input. Press A / Start when
the task is done: a popup asks whether to save. B cancels, Back resets.

Windows path uses XInput directly (no extra dependency). Keyboard fallback:
W/S x, A/D y, Q/E z, Z/X yaw, O/C gripper, Space pause, R reset, T record.

Examples:
    python scripts/alicia_teleop_collect.py
    python scripts/alicia_teleop_collect.py --auto-success
    python scripts/alicia_teleop_collect.py --self-test
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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
from scripts.gamepad_overlay import GamepadWindow, draw_gamepad  # noqa: E402

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

        raw = int(info.dwButtons)
        buttons = 0
        # Map DirectInput button bit order to the XInput masks used by the app.
        direct_to_xinput = (
            (0, A),
            (1, B),
            (2, X),
            (3, Y),
            (4, LB),
            (5, RB),
            (6, BACK),
            (7, START),
            (8, LEFT_THUMB),
            (9, RIGHT_THUMB),
        )
        for bit, mask in direct_to_xinput:
            if raw & (1 << bit):
                buttons |= mask
        pov = int(info.dwPOV)
        if pov != 0xFFFF:
            if pov < 4500 or pov > 31500:
                buttons |= DPAD_UP
            if 4500 <= pov < 13500:
                buttons |= DPAD_RIGHT
            if 13500 <= pov < 22500:
                buttons |= DPAD_DOWN
            if 22500 <= pov <= 31500:
                buttons |= DPAD_LEFT
        return GamepadState(
            buttons=buttons,
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
    parser.add_argument(
        "--task-file", type=Path, default=REPO_ROOT / "configs" / "teleop_tasks.yaml"
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--cycle-tasks", action="store_true", help="保存后自动切换到下一条指令")
    parser.add_argument("--instruction", default=None, help="覆盖预分配标签")
    parser.add_argument("--controller", type=int, default=0)
    parser.add_argument("--speed-mps", type=float, default=0.15, help="平移速度 m/s")
    parser.add_argument("--yaw-rps", type=float, default=0.8, help="偏航速度 rad/s")
    parser.add_argument("--deadzone", type=float, default=0.15)
    parser.add_argument(
        "--stop-idle",
        type=float,
        default=0.0,
        help="0=不自动停止；仅保留兼容参数",
    )
    parser.add_argument("--frame-every", type=int, default=2)
    parser.add_argument("--success-z", type=float, default=0.34)
    parser.add_argument("--auto-success", action="store_true", help="方块抬升后自动完成")
    parser.add_argument(
        "--no-grasp-assist",
        action="store_true",
        help="关闭演示用吸附（默认开启，便于手动采集）",
    )
    parser.add_argument("--no-auto-record", action="store_true")
    parser.add_argument("--no-save-prompt", action="store_true", help="完成时直接保存")
    parser.add_argument("--no-wrist-inset", action="store_true")
    parser.add_argument(
        "--gamepad-panel",
        choices=("window", "hud", "off"),
        default="window",
        help="手柄面板：独立窗口 / 右上叠加 / 关闭",
    )
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_tasks(path: Path, override: str | None) -> list[dict[str, str]]:
    if override:
        return [{"id": "custom", "zh": override, "en": override}]
    if not path.exists():
        return [{"id": "pick_red", "zh": "拿起红色方块", "en": "pick up the red block"}]
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    tasks = [task for task in data.get("tasks", []) if task.get("en")]
    return tasks or [{"id": "pick_red", "zh": "拿起红色方块", "en": "pick up the red block"}]


def _episode_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root / stamp
    suffix = 0
    while path.exists():
        suffix += 1
        path = root / f"{stamp}-{suffix}"
    return path


def _tune_contact(env: AliciaSimEnv) -> None:
    """Increase contact friction for manual grasping in the vendor MJCF."""
    model = env.model
    finger_bodies = {
        model.body("left_gripper").id,
        model.body("right_gripper").id,
    }
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name
        body_id = int(model.geom_bodyid[geom_id])
        if name in ("cube_geom", "table", "pedestal") or (
            body_id in finger_bodies and model.geom_contype[geom_id]
        ):
            model.geom_friction[geom_id] = [3.0, 2.0, 0.1]
            model.geom_condim[geom_id] = 4


class TeleopCollector:
    def __init__(self, env: AliciaSimEnv, args: argparse.Namespace) -> None:
        self.env = env
        self.args = args
        self.gamepad = GamepadReader(args.controller)
        self.assist = not args.no_grasp_assist
        self.tasks = _load_tasks(args.task_file, args.instruction)
        self.task_index = min(max(0, args.task_index), len(self.tasks) - 1)
        self.prev_buttons = 0
        self.recorder: Recorder | None = None
        self.episode_dir: Path | None = None
        self.last_input = 0.0
        self.keyboard_last = 0.0
        self.grip = 0.0
        self.frames = 0
        self.saved = 0
        self.cancelled = 0
        self.armed = True
        self.camera_toggle = False
        self.last_meta: dict[str, Any] | None = None

    @property
    def current_task(self) -> dict[str, str]:
        return self.tasks[self.task_index]

    def next_task(self, delta: int) -> None:
        self.task_index = (self.task_index + delta) % len(self.tasks)
        task = self.current_task
        print(f"TASK {task['id']}: {task['zh']} / {task['en']}", flush=True)

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
        task = self.current_task
        print(
            f"REC start {self.episode_dir} | {task['id']}: {task['zh']} / {task['en']}",
            flush=True,
        )

    def _save(self, recorder: Recorder, episode_dir: Path) -> None:
        recorder.save()
        task = self.current_task
        meta = {
            "episode_id": episode_dir.name,
            "started": episode_dir.name,
            "instruction": task["en"],
            "instruction_zh": task.get("zh", task["en"]),
            "instruction_en": task["en"],
            "task_id": task.get("id", "custom"),
            "task_index": self.task_index,
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
            "grasp_assist": self.assist,
            "frames": self.frames,
            "split": "train",
            "success": "manual",
            "success_z": self.args.success_z,
        }
        (episode_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.saved += 1
        self.last_meta = meta
        print(f"SAVE {episode_dir} frames={self.frames}", flush=True)
        if self.args.cycle_tasks:
            self.next_task(1)

    def _prompt_save(self, episode_dir: Path) -> bool:
        if self.args.no_save_prompt:
            return True
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            result = messagebox.askyesno(
                "示教完成",
                f"保存本次示教？\n\n{self.frames} 帧\n{episode_dir}",
                parent=root,
            )
            root.destroy()
            return bool(result)
        except Exception as exc:  # noqa: BLE001
            print(f"save prompt failed: {exc}; use --no-save-prompt", flush=True)
            return True

    def finish(self) -> None:
        if self.recorder is None:
            return
        recorder = self.recorder
        episode_dir = self.episode_dir
        self.recorder = None
        self.episode_dir = None
        self.armed = False
        self.env.detach_cube()
        if self.frames < 5 or episode_dir is None:
            if episode_dir is not None:
                shutil.rmtree(episode_dir, ignore_errors=True)
            self.cancelled += 1
            print("CANCEL empty episode", flush=True)
            return
        if self._prompt_save(episode_dir):
            self._save(recorder, episode_dir)
        else:
            shutil.rmtree(episode_dir, ignore_errors=True)
            self.cancelled += 1
            print(f"DISCARD {episode_dir}", flush=True)

    def cancel(self) -> None:
        if self.recorder is None:
            return
        episode_dir = self.episode_dir
        self.recorder = None
        self.episode_dir = None
        self.armed = False
        self.env.detach_cube()
        if episode_dir is not None:
            shutil.rmtree(episode_dir, ignore_errors=True)
        self.cancelled += 1
        print("CANCEL episode", flush=True)

    def _handle_buttons(self, pad: GamepadState) -> None:
        pressed = pad.buttons & ~self.prev_buttons
        self.prev_buttons = pad.buttons
        if pressed & (A | START):
            self.finish()
        if pressed & DPAD_RIGHT:
            self.next_task(1)
        if pressed & DPAD_LEFT:
            self.next_task(-1)
        if pressed & B:
            self.cancel()
        if pressed & X:
            self.grip = 0.0
        if pressed & Y:
            self.grip = 1.0
        if pressed & LB:
            self.cancel()
            self.env.reset()
        if pressed & RB:
            self.camera_toggle = True
        if pressed & BACK:
            self.cancel()
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
        else:
            self.armed = True
        auto = not self.args.no_auto_record
        if auto and self.recorder is None and active and self.armed:
            self.start()
            self.armed = False
        self.env.command_gripper_norm(self.grip)
        self.env.step()
        if self.assist:
            cube = self.env.cube_position()
            if self.grip > 0.5 and cube is not None and not self.env.cube_attached:
                if float(np.linalg.norm(cube - self.env.eef_pose()[0])) < 0.07:
                    self.env.attach_cube()
            elif self.grip < 0.2 and self.env.cube_attached:
                self.env.detach_cube()
        if self.recorder is not None:
            self.frames += 1
            self.recorder.record(self.grip)
            if self.args.auto_success:
                cube = self.env.cube_position()
                if (
                    cube is not None
                    and float(cube[2]) > self.args.success_z
                    and self.env.gripper_norm() > 0.5
                ):
                    self.finish()


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
    camera_free = True
    started = time.monotonic()

    def key_callback(keycode: int) -> None:
        nonlocal paused, running, camera_free
        key = chr(keycode).lower() if 32 <= keycode < 127 else ""
        if keycode == 256:
            running = False
        elif key == " ":
            paused = not paused
        elif key == "r":
            collector.cancel()
            env.reset()
        elif key == "t":
            collector.finish()
        elif key == "n":
            collector.next_task(1)
        elif key == "p":
            collector.next_task(-1)
        elif key == "h":
            env.set_target(env.eef_pose()[0])
        elif key == "v":
            camera_free = not camera_free
            if camera_free:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = env.camera_ids["front"]
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

    panel_window = None
    panel_mode = args.gamepad_panel
    if panel_mode == "window":
        try:
            panel_window = GamepadWindow()
        except Exception as exc:  # noqa: BLE001
            print(f"gamepad window unavailable: {exc}; using HUD", flush=True)
            panel_mode = "hud"
    viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat = [0.35, 0.0, 0.3]
    viewer.cam.distance = 1.25
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -25.0
    last_hud = 0.0
    try:
        while running and viewer.is_running():
            now = time.monotonic()
            if args.duration > 0.0 and now - started > args.duration:
                break
            pad = collector.gamepad.read()
            if not paused:
                collector.step(pad)
            if collector.camera_toggle:
                collector.camera_toggle = False
                camera_free = not camera_free
                if camera_free:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                else:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = env.camera_ids["front"]
            if now - last_hud > 0.2:
                last_hud = now
                state = env.state()
                mode = "REC" if collector.recorder is not None else "IDLE"
                task = collector.current_task
                if pad is None:
                    pad_text = "pad: none"
                else:
                    pad_text = (
                        f"LX {pad.lx:+.2f}  LY {pad.ly:+.2f}  "
                        f"RX {pad.rx:+.2f}  RY {pad.ry:+.2f}  "
                        f"LT {pad.lt:.2f}  RT {pad.rt:.2f}  "
                        f"BTN {pad.buttons:#06x}"
                    )
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
                            f"grip {collector.grip:.2f} | {'pad on' if pad else 'pad off'}",
                        ),
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                            f"task[{collector.task_index}] {task['id']}: {task['zh']}",
                            pad_text,
                        ),
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                            "D-pad task | A save | B cancel",
                            "LB reset | RB camera | mouse orbit/zoom",
                        ),
                    ]
                )
                if panel_window is not None:
                    panel_window.update(pad)
            images = []
            if not args.no_wrist_inset:
                image = env.render("wrist")
                images.append((mujoco.MjrRect(8, 8, image.shape[1], image.shape[0]), image))
            if panel_mode == "hud":
                panel = draw_gamepad(pad)
                panel_w, panel_h = panel.shape[1], panel.shape[0]
                view_w = int(getattr(viewer.viewport, "width", 1280))
                view_h = int(getattr(viewer.viewport, "height", 720))
                images.append(
                    (
                        mujoco.MjrRect(
                            max(8, view_w - panel_w - 12),
                            max(8, view_h // 2 - panel_h // 2),
                            panel_w,
                            panel_h,
                        ),
                        panel,
                    )
                )
            if images:
                viewer.set_images(images)
            else:
                viewer.clear_images()
            viewer.sync()
            remaining = env.control_period - (time.monotonic() - now)
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        collector.cancel()
        if panel_window is not None:
            panel_window.close()
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
    _tune_contact(env)
    collector = TeleopCollector(env, args)
    try:
        return _run_viewer(collector, args)
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
