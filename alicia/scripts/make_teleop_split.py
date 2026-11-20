"""Create an episode-level train/val split for teleop demonstrations.

Singleton tasks stay in train; larger task groups contribute a deterministic
fraction to validation. The split is written to ``splits.json`` and mirrored
into every episode's ``meta.json``.

Example:
    python scripts/make_teleop_split.py --root outputs/alicia_teleop
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="遥操作数据 train/val 划分")
    parser.add_argument("--root", type=Path, default=Path("outputs/alicia_teleop"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-group", type=int, default=5)
    parser.add_argument(
        "--max-val-frames",
        type=int,
        default=1200,
        help="超过该帧数的 episode 不放进 val；0 表示不限制",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    episodes = sorted(path for path in root.iterdir() if (path / "meta.json").exists())
    groups: dict[str, list[Path]] = defaultdict(list)
    frames: dict[str, int] = {}
    for episode in episodes:
        meta = json.loads((episode / "meta.json").read_text(encoding="utf-8"))
        groups[str(meta.get("task_id", "unknown"))].append(episode)
        frames[episode.name] = int(meta.get("frames", 0))

    rng = random.Random(args.seed)
    val: set[str] = set()
    for task_id in sorted(groups):
        group = sorted(groups[task_id])
        rng.shuffle(group)
        if len(group) >= args.min_group:
            count = max(1, round(len(group) * args.val_fraction))
            if args.max_val_frames > 0:
                eligible = [
                    path
                    for path in group
                    if frames.get(path.name, 0) <= args.max_val_frames
                ]
            else:
                eligible = group
            if len(eligible) < count:
                eligible = group
            val.update(path.name for path in eligible[:count])

    train = [episode.name for episode in episodes if episode.name not in val]
    val_list = sorted(val)
    if not val_list:
        raise SystemExit("empty validation split; lower --min-group or add episodes")

    split = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "min_group": args.min_group,
        "train": train,
        "val": val_list,
        "tasks": {
            task_id: {
                "train": [p.name for p in groups[task_id] if p.name in train],
                "val": [p.name for p in groups[task_id] if p.name in val],
            }
            for task_id in sorted(groups)
        },
    }
    (root / "splits.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for episode in episodes:
        meta_path = episode / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["split"] = "val" if episode.name in val else "train"
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"episodes={len(episodes)} train={len(train)} val={len(val_list)}")
    print(f"val={val_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
