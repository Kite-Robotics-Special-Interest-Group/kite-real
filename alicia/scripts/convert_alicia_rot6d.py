"""Convert Alicia episodes/features from column-major rot6d to official interleaved.

Old layout: [R00,R10,R20,R01,R11,R21]
New layout: [R00,R01,R10,R11,R20,R21]
Reorder indices: old 3..8 -> 3,6,4,7,5,8.

Example:
    python scripts/convert_alicia_rot6d.py \
        --episodes outputs/alicia_episodes --out-episodes outputs/alicia_episodes_v2 \
        --features outputs/alicia_features --out-features outputs/alicia_features_v2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

REORDER = np.array([0, 3, 1, 4, 2, 5])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rot6d columns -> interleaved")
    parser.add_argument("--episodes", type=Path, default=Path("outputs/alicia_episodes"))
    parser.add_argument("--out-episodes", type=Path, default=Path("outputs/alicia_episodes_v2"))
    parser.add_argument("--features", type=Path, default=Path("outputs/alicia_features"))
    parser.add_argument("--out-features", type=Path, default=Path("outputs/alicia_features_v2"))
    return parser.parse_args()


def convert_labels(array: np.ndarray) -> np.ndarray:
    result = array.copy()
    block = result[..., 3:9]
    result[..., 3:9] = block[..., REORDER]
    return result


def convert_episodes(source: Path, output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output)
    for action_path in output.glob("*/action.npy"):
        action = np.load(action_path)
        np.save(action_path, convert_labels(action))
    for proprio_path in output.glob("*/proprio.npy"):
        proprio = np.load(proprio_path)
        np.save(proprio_path, convert_labels(proprio))
    # copy split manifest
    split = source / "splits.json"
    if split.exists():
        shutil.copy2(split, output / "splits.json")


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def convert_features(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in ("vlm_features.npy", "aux_visual.npy"):
        link_or_copy(source / name, output / name)
    for name in ("action.npy", "proprio.npy"):
        array = np.load(source / name)
        np.save(output / name, convert_labels(array))
    for name in ("meta.json",):
        shutil.copy2(source / name, output / name)


def main() -> int:
    args = parse_args()
    convert_episodes(args.episodes, args.out_episodes)
    convert_features(args.features, args.out_features)
    print(f"episodes: {args.out_episodes}")
    print(f"features: {args.out_features}")
    action = np.load(args.out_features / "action.npy")
    proprio = np.load(args.out_features / "proprio.npy")
    print("action", action.shape, "proprio", proprio.shape)
    print(json.dumps({"rot6d": "[b1x,b2x,b1y,b2y,b1z,b2z]"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
