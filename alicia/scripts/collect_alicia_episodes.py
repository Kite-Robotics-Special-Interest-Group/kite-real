"""Collect scripted Alicia-D pick demonstrations for soft-prompt training.

The scripted expert uses a temporary cube attachment while closing the gripper.
The vendor MJCF's 50 mm finger geometry is narrow at the fingertips, so a purely
physical grasp slips during lift; the attachment keeps the demonstration
kinematically clean.  Metadata records this explicitly.

Examples:
    python scripts/collect_alicia_episodes.py --episodes 120 --out outputs/alicia_episodes
    python scripts/collect_alicia_episodes.py --episodes 5 --out outputs/alicia_try
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure") and not sys.stdout.isatty():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from kite.robot.alicia import load_config, matrix_to_rot6d
from kite.sim.alicia import AliciaSimEnv

INSTRUCTION = "pick up the red block"
DROP_XY = (0.34, -0.14)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集 Alicia-D 脚本示教")
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/alicia_episodes"))
    parser.add_argument("--frame-every", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class Recorder:
    """Collect actions, proprio and sparse JPEG observations."""

    def __init__(self, env: AliciaSimEnv, ep_dir: Path, frame_every: int) -> None:
        self.env = env
        self.ep_dir = ep_dir
        self.frame_every = max(1, frame_every)
        self.actions: list[np.ndarray] = []
        self.proprio: list[np.ndarray] = []
        self.frames: list[int] = []
        (ep_dir / "obs").mkdir(parents=True, exist_ok=True)

    def record(self, grip: float) -> None:
        index = len(self.actions)
        target_position = self.env.target_position.copy()
        target_rotation = self.env.target_rotation.copy()
        self.actions.append(
            np.concatenate(
                [target_position, matrix_to_rot6d(target_rotation), [float(grip)]]
            ).astype(np.float32)
        )
        self.proprio.append(self.env.proprio().astype(np.float32))
        if index % self.frame_every == 0:
            Image.fromarray(self.env.render("front")).save(
                self.ep_dir / "obs" / f"{index:04d}_front.jpg", quality=90
            )
            Image.fromarray(self.env.render("wrist")).save(
                self.ep_dir / "obs" / f"{index:04d}_wrist.jpg", quality=90
            )
            self.frames.append(index)

    def save(self) -> None:
        np.save(self.ep_dir / "action.npy", np.stack(self.actions))
        np.save(self.ep_dir / "proprio.npy", np.stack(self.proprio))


def _go(
    env: AliciaSimEnv,
    recorder: Recorder,
    target: np.ndarray,
    rotation: np.ndarray,
    grip: float,
    max_steps: int = 120,
    tolerance: float = 0.005,
) -> float:
    env.set_target(target, rotation)
    error = float("inf")
    for _ in range(max_steps):
        env.command_gripper_norm(grip)
        env.step()
        error = env.state().position_error
        recorder.record(grip)
        if error < tolerance:
            break
    return error


def collect_episode(seed: int, out: Path, base_config: dict, frame_every: int) -> dict:
    rng = np.random.default_rng(seed)
    cube_xy = (
        0.42 + float(rng.uniform(-0.04, 0.04)),
        0.10 + float(rng.uniform(-0.08, 0.12)),
    )
    config = copy.deepcopy(base_config)
    config["robot"]["scene"]["pedestal"]["pos_m"] = [cube_xy[0], cube_xy[1], 0.25]
    config["robot"]["scene"]["cube"]["pos_m"] = [cube_xy[0], cube_xy[1], 0.30]

    env = AliciaSimEnv(config=config, with_task=True, seed=seed)
    ep_dir = out / f"{seed:04d}"
    recorder = Recorder(env, ep_dir, frame_every)
    metadata = {
        "episode_id": f"{seed:04d}",
        "seed": seed,
        "instruction": INSTRUCTION,
        "fps": 20,
        "action_layout": "[ee_xyz(3), rot6d(6), grip(1)]",
        "units": {
            "length": "m",
            "angle": "rad",
            "gripper": "normalized_0_1_closed_is_1",
        },
        "renderer": "mujoco",
        "depth_unit": "none",
        "camera_intrinsics": {
            "front": {"width": 224, "height": 224, "fovy_deg": 48},
            "wrist": {"width": 224, "height": 224, "fovy_deg": 75},
        },
        "camera_extrinsics": {
            "front": "world fixed",
            "wrist": "link6 +0.06 x",
        },
        "split": "train",
        "grasp_assist": True,
    }
    success = False
    try:
        env.reset()
        rotation = env.eef_pose()[1].copy()
        cube = env.cube_position()
        if cube is None:
            raise RuntimeError("cube scene missing")
        recorder.record(0.0)

        _go(
            env,
            recorder,
            np.array([cube[0], cube[1], 0.34]),
            rotation,
            0.0,
            max_steps=50,
        )
        _go(
            env,
            recorder,
            np.array([cube[0], cube[1], 0.30]),
            rotation,
            0.0,
            max_steps=35,
        )
        env.attach_cube()
        _go(
            env,
            recorder,
            np.array([cube[0], cube[1], 0.30]),
            rotation,
            1.0,
            max_steps=15,
        )
        _go(
            env,
            recorder,
            np.array([cube[0], cube[1], 0.36]),
            rotation,
            1.0,
            max_steps=45,
        )
        lifted = env.cube_position()
        lift_ok = lifted is not None and float(lifted[2]) > 0.30

        _go(
            env,
            recorder,
            np.array([DROP_XY[0], DROP_XY[1], 0.34]),
            rotation,
            1.0,
            max_steps=50,
        )
        env.command_gripper_norm(0.0)
        for _ in range(15):
            env.step()
            recorder.record(0.0)
        env.detach_cube()
        for _ in range(20):
            env.command_gripper_norm(0.0)
            env.step()
            recorder.record(0.0)
        final = env.cube_position()
        drop_ok = (
            final is not None
            and float(final[2]) < 0.30
            and float(np.linalg.norm(final[:2] - np.array(DROP_XY))) < 0.18
        )
        success = bool(lift_ok and drop_ok)
        metadata.update(
            {
                "cube_xy": list(cube_xy),
                "success": success,
                "lift_ok": lift_ok,
                "drop_ok": drop_ok,
                "steps": len(recorder.actions),
                "frames": recorder.frames,
            }
        )
        recorder.save()
    finally:
        env.close()
    (ep_dir / "meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> int:
    args = parse_args()
    base_config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    attempts = args.max_attempts or args.episodes * 2
    collected = 0
    seed = args.start
    while collected < args.episodes and seed < args.start + attempts:
        meta = collect_episode(seed, args.out, base_config, args.frame_every)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "success": meta["success"],
                    "lift": meta["lift_ok"],
                    "drop": meta["drop_ok"],
                    "steps": meta["steps"],
                },
                ensure_ascii=False,
            )
        )
        collected += int(meta["success"])
        seed += 1

    episode_ids = sorted(
        path.name for path in args.out.iterdir() if (path / "meta.json").exists()
    )
    split = int(len(episode_ids) * 0.85)
    (args.out / "splits.json").write_text(
        json.dumps(
            {"train": episode_ids[:split], "val": episode_ids[split:]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"episodes={len(episode_ids)} successful={collected} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
