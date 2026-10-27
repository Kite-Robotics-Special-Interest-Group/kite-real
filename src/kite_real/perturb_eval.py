"""Perturbation evaluation for real-robot observations (reuses kite-bench tools).

Install the simulator-agnostic pieces of kite-bench (``perturb``/``metrics``) to use
this: ``pip install -e ../kite-bench``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

LEVELS: dict[str, list[dict]] = {
    "brightness": [{"factor": v} for v in (0.7, 0.85, 1.15, 1.3)],
    "contrast": [{"factor": v} for v in (0.7, 0.85, 1.15, 1.3)],
    "blur": [{"radius": v} for v in (1.0, 2.0, 3.0, 4.0)],
    "noise": [{"sigma": v, "seed": 0} for v in (5.0, 10.0, 20.0, 30.0)],
    "occlude_rect": [{"area": a, "seed": i} for i, a in enumerate((0.05, 0.1, 0.2, 0.3))],
    "jpeg": [{"quality": v} for v in (80, 60, 40, 20)],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="sweep perturbations over real frames")
    parser.add_argument("--frames", type=Path, required=True, help="directory of PNG frames")
    parser.add_argument("--out", type=Path, default=Path("out/perturb_real.md"))
    args = parser.parse_args()

    from kitebench import perturb

    registry = perturb.registry()
    images = sorted(args.frames.glob("*.png"))
    if not images:
        print(f"no PNG frames in {args.frames}")
        return 1

    rows = []
    for family, levels in LEVELS.items():
        fn = registry[family]
        for kwargs in levels:
            saved = []
            for img_path in images:
                img = Image.open(img_path).convert("RGB")
                out = args.out.parent / "perturbed" / family
                out.mkdir(parents=True, exist_ok=True)
                path = out / f"{img_path.stem}_{'_'.join(f'{k}{v}' for k, v in kwargs.items())}.png"
                fn(img, **kwargs).save(path)
                saved.append(str(path))
            rows.append({"family": family, "level": kwargs, "frames": saved})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} configs over {len(images)} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

