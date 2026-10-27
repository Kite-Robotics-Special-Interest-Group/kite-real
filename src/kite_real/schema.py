"""Episode format shared with kite-bench, plus validation for real-robot recordings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REQUIRED_META = ("instruction", "control_hz", "gripper", "action_space", "success")
ACTION_DIM = 10  # ee_xyz(3) + rot6d(6) + grip(1)
PROPRIO_DIM = 7  # joints(6) + gripper


@dataclass
class Episode:
    root: Path
    meta: dict
    actions: np.ndarray
    proprio: np.ndarray
    frames: list[Path]


def load_episode(root: Path) -> Episode:
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    actions = np.load(root / "action.npy")
    proprio = np.load(root / "proprio.npy")
    frames = sorted((root / "obs").glob("*.png"))
    return Episode(root=root, meta=meta, actions=actions, proprio=proprio, frames=frames)


def validate(root: Path) -> list[str]:
    """Return a list of problems; empty means the episode is usable."""
    problems: list[str] = []
    try:
        ep = load_episode(root)
    except Exception as exc:  # noqa: BLE001 - report any load failure as a problem
        return [f"cannot load: {exc}"]

    missing = [k for k in REQUIRED_META if k not in ep.meta]
    if missing:
        problems.append(f"missing meta fields: {', '.join(missing)}")

    if ep.actions.ndim != 2 or ep.actions.shape[1] != ACTION_DIM:
        problems.append(f"action shape {ep.actions.shape}, expected [T, {ACTION_DIM}]")
    if ep.proprio.ndim != 2 or ep.proprio.shape[1] != PROPRIO_DIM:
        problems.append(f"proprio shape {ep.proprio.shape}, expected [T, {PROPRIO_DIM}]")
    if ep.actions.shape[0] != ep.proprio.shape[0]:
        problems.append(
            f"length mismatch: actions {ep.actions.shape[0]} vs proprio {ep.proprio.shape[0]}"
        )
    if np.isnan(ep.actions).any():
        problems.append("actions contain NaN")
    if np.isnan(ep.proprio).any():
        problems.append("proprio contains NaN")
    if len(ep.frames) == 0:
        problems.append("no observation frames")
    if ep.meta.get("success") is not None and ep.meta.get("failure_type") is None \
            and not ep.meta.get("success"):
        problems.append("failed episode without failure_type")
    return problems


def write_meta(root: Path, **fields) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(
        json.dumps(fields, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: python -m kite_real.schema <episode_dir> [...]")
        return 2
    failed = 0
    for path in argv:
        problems = validate(Path(path))
        if problems:
            failed += 1
            print(f"✗ {path}")
            for p in problems:
                print(f"    {p}")
        else:
            print(f"✓ {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
