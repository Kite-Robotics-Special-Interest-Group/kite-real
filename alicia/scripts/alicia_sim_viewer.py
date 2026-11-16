"""Alicia-D v5.6 MuJoCo supervisor window.

The window shows the vendor MJCF, a fixed front camera, an optional wrist
camera inset and a yellow TCP target.  A human can pause, jog the target,
open/close the gripper and reset the scene while a policy is running.

Examples:
    python scripts/alicia_sim_viewer.py
    python scripts/alicia_sim_viewer.py --no-auto
    python scripts/alicia_sim_viewer.py --headless --frames 100 \
        --snapshot outputs/alicia_sim.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alicia-D v5.6 仿真监督窗口")
    parser.add_argument("--config", default=None, help="机器人配置 YAML")
    parser.add_argument("--camera", choices=("front", "wrist"), default="front")
    parser.add_argument("--no-task", action="store_true", help="不加载桌面/方块")
    parser.add_argument("--no-auto", action="store_true", help="启动时进入手动模式")
    parser.add_argument(
        "--no-wrist-inset", action="store_true", help="关闭腕部相机小窗"
    )
    parser.add_argument(
        "--jog-step", type=float, default=0.01, help="键盘点动步长（米）"
    )
    parser.add_argument("--frames", type=int, default=0, help="无窗口模式运行帧数")
    parser.add_argument(
        "--duration", type=float, default=0.0, help="自动关闭时间（秒）"
    )
    parser.add_argument("--headless", action="store_true", help="不打开窗口")
    parser.add_argument("--snapshot", default=None, help="保存 front/wrist 拼图 PNG")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_env(args: argparse.Namespace) -> Any:
    from kite.sim.alicia import AliciaSimEnv

    return AliciaSimEnv(config=args.config, with_task=not args.no_task, seed=args.seed)


def _auto_target(env: Any, home: np.ndarray, elapsed: float) -> None:
    phase = elapsed * 0.9
    offset = np.array(
        [
            0.045 * math.cos(phase),
            0.055 * math.sin(phase * 0.7),
            0.035 * math.sin(phase),
        ]
    )
    env.set_target(home + offset)
    grip = 0.5 + 0.5 * math.sin(elapsed * 1.2)
    env.set_gripper_norm(0.0 if grip < 0.5 else 1.0)


def _save_snapshot(env: Any, path: Path) -> None:
    from PIL import Image

    front = env.render("front")
    wrist = env.render("wrist")
    canvas = Image.new("RGB", (front.shape[1] * 2, front.shape[0]))
    canvas.paste(Image.fromarray(front), (0, 0))
    canvas.paste(Image.fromarray(wrist), (front.shape[1], 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _run_headless(env: Any, args: argparse.Namespace) -> int:
    frames = args.frames if args.frames > 0 else int(20 * max(args.duration, 2.0))
    home = env.target_position.copy()
    for index in range(frames):
        if not args.no_auto:
            _auto_target(env, home, index * env.control_period)
        env.step()
    state = env.state()
    print(
        json.dumps(
            {
                "frames": frames,
                "position": state.position.tolist(),
                "target": state.target_position.tolist(),
                "position_error": state.position_error,
                "rotation_error": state.rotation_error,
                "gripper_norm": state.gripper_norm,
            },
            ensure_ascii=False,
        )
    )
    if args.snapshot:
        _save_snapshot(env, Path(args.snapshot))
        print(f"snapshot: {args.snapshot}")
    return 0


def _run_viewer(env: Any, args: argparse.Namespace) -> int:
    import mujoco.viewer

    flags = {"paused": False, "auto": not args.no_auto, "running": True}
    home = env.target_position.copy()
    started = time.monotonic()

    def key_callback(keycode: int) -> None:
        key = chr(keycode).lower() if 32 <= keycode < 127 else ""
        if keycode == 256:
            flags["running"] = False
        elif key == " ":
            flags["paused"] = not flags["paused"]
        elif key == "r":
            env.reset()
        elif key == "h":
            env.set_target(home)
        elif key == "b":
            env.set_target(env.eef_pose()[0])
        elif key == "t":
            flags["auto"] = not flags["auto"]
        elif key == "p":
            print(json.dumps(env.state().__dict__, ensure_ascii=False, default=list))
        elif key == "w":
            env.nudge(dx=-args.jog_step)
        elif key == "s":
            env.nudge(dx=args.jog_step)
        elif key == "a":
            env.nudge(dy=args.jog_step)
        elif key == "d":
            env.nudge(dy=-args.jog_step)
        elif key == "q":
            env.nudge(dz=args.jog_step)
        elif key == "e":
            env.nudge(dz=-args.jog_step)
        elif key == "z":
            env.nudge(dyaw=args.jog_step)
        elif key == "x":
            env.nudge(dyaw=-args.jog_step)
        elif key == "o":
            env.set_gripper_norm(0.0)
        elif key == "c":
            env.set_gripper_norm(1.0)
        elif key == "g":
            env.set_gripper_norm(0.0 if env.gripper_norm() > 0.5 else 1.0)

    viewer = mujoco.viewer.launch_passive(
        env.model, env.data, key_callback=key_callback
    )
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = env.camera_ids[args.camera]
    period = env.control_period
    last_hud = 0.0
    try:
        while flags["running"] and viewer.is_running():
            now = time.monotonic()
            if args.duration > 0.0 and now - started > args.duration:
                break
            if not flags["paused"]:
                if flags["auto"]:
                    _auto_target(env, home, now - started)
                env.step()
            if now - last_hud > 0.2:
                last_hud = now
                state = env.state()
                mode = "AUTO" if flags["auto"] else "MANUAL"
                pause = "PAUSED" if flags["paused"] else "RUN"
                viewer.set_texts(
                    [
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            "Alicia-D v5.6 supervisor",
                            f"{mode} | {pause}",
                        ),
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                            (
                                f"TCP {np.round(state.position, 3)} | "
                                f"target {np.round(state.target_position, 3)}"
                            ),
                            (
                                f"err {state.position_error * 1000:.1f} mm | "
                                f"grip {state.gripper_norm:.2f}"
                            ),
                        ),
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                            "space pause  r reset  h home  b brake",
                            "w/s x  a/d y  q/e z  z/x yaw  o/c grip  t auto",
                        ),
                    ]
                )
            if not args.no_wrist_inset:
                image = env.render("wrist")
                viewer.set_images(
                    [(mujoco.MjrRect(8, 8, image.shape[1], image.shape[0]), image)]
                )
            viewer.sync()
            remaining = period - (time.monotonic() - now)
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        viewer.close()
    return 0


def main() -> int:
    args = parse_args()
    env = _load_env(args)
    try:
        if args.headless:
            return _run_headless(env, args)
        return _run_viewer(env, args)
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
