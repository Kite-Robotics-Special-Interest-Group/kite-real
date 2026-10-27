"""Write episodes in the shared kite-bench format while recording on the real robot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kite_real.schema import ACTION_DIM, PROPRIO_DIM, write_meta


class EpisodeWriter:
    """Collects per-step observations and flushes them as one episode directory."""

    def __init__(self, root: Path, instruction: str, control_hz: float, gripper: str,
                 action_space: str):
        self.root = root
        self.instruction = instruction
        self.control_hz = control_hz
        self.gripper = gripper
        self.action_space = action_space
        self.actions: list[np.ndarray] = []
        self.proprio: list[np.ndarray] = []
        self.frames: list[np.ndarray] = []

    def add(self, action: np.ndarray, proprio: np.ndarray, frame: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        proprio = np.asarray(proprio, dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"action must be {ACTION_DIM}-dim, got {action.shape}")
        if proprio.shape != (PROPRIO_DIM,):
            raise ValueError(f"proprio must be {PROPRIO_DIM}-dim, got {proprio.shape}")
        self.actions.append(action)
        self.proprio.append(proprio)
        self.frames.append(frame)

    def close(self, success: bool, failure_type: str | None = None, **extra) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        obs = self.root / "obs"
        obs.mkdir(exist_ok=True)
        np.save(self.root / "action.npy", np.stack(self.actions))
        np.save(self.root / "proprio.npy", np.stack(self.proprio))

        from PIL import Image

        for i, frame in enumerate(self.frames):
            Image.fromarray(frame).save(obs / f"{i:04d}_front.png")

        write_meta(
            self.root,
            instruction=self.instruction,
            control_hz=self.control_hz,
            gripper=self.gripper,
            action_space=self.action_space,
            success=bool(success),
            failure_type=failure_type,
            steps=len(self.actions),
            **extra,
        )
        return self.root
