"""Precompute frozen Florence2 features for Alicia-D soft-prompt training.

Stores fp16 features and 20D action labels under ``--out``.  Training then only
runs the small domain head, which is much faster than re-encoding images.

Example:
    python scripts/precompute_alicia_features.py --limit 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure") and not sys.stdout.isatty():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.train_alicia_softprompt import (
    build_chunk,
    build_samples,
    collate_batch,
    load_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预计算 Alicia-D X-VLA 视觉特征")
    parser.add_argument(
        "--episodes", type=Path, default=REPO_ROOT / "outputs/alicia_episodes"
    )
    parser.add_argument(
        "--model", type=str, default=str(REPO_ROOT / "weights/X-VLA-WidowX")
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "outputs/alicia_features"
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=8000)
    parser.add_argument("--val-limit", type=int, default=1500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from transformers import AutoModel, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    train_ids, val_ids = load_split(args.episodes)
    train_samples = build_samples(args.episodes, train_ids)[: args.train_limit]
    val_samples = build_samples(args.episodes, val_ids)[: args.val_limit]
    samples = train_samples + val_samples
    if args.limit:
        samples = samples[: args.limit]
        train_count = min(len(train_samples), len(samples))
        val_count = max(0, len(samples) - train_count)
    else:
        train_count = len(train_samples)
        val_count = len(val_samples)
    print(f"samples={len(samples)} train={train_count} val={val_count}", flush=True)

    model = (
        AutoModel.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch.float32
        )
        .to("cuda")
        .eval()
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    args.out.mkdir(parents=True, exist_ok=True)

    vlm_mem = None
    aux_mem = None
    proprio_labels = np.empty((len(samples), 20), dtype=np.float32)
    action_labels = np.empty((len(samples), 30, 20), dtype=np.float32)
    started = time.perf_counter()
    for start in range(0, len(samples), args.batch):
        batch = samples[start : start + args.batch]
        input_ids, image_input, image_mask, proprio, _action = collate_batch(
            processor, batch, "cuda"
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            encoded = model.forward_vlm(input_ids, image_input, image_mask)
        vlm = encoded["vlm_features"].float().cpu().numpy().astype(np.float16)
        aux = encoded["aux_visual_inputs"].float().cpu().numpy().astype(np.float16)
        if vlm_mem is None:
            vlm_mem = np.lib.format.open_memmap(
                args.out / "vlm_features.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(samples), *vlm.shape[1:]),
            )
            aux_mem = np.lib.format.open_memmap(
                args.out / "aux_visual.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(samples), *aux.shape[1:]),
            )
        end = start + len(batch)
        vlm_mem[start:end] = vlm
        aux_mem[start:end] = aux
        proprio_labels[start:end] = proprio.detach().cpu().numpy()
        for offset, sample in enumerate(batch):
            action_labels[start + offset] = build_chunk(
                sample["actions"], sample["index"]
            )
        if start and start % (args.batch * 50) == 0:
            elapsed = time.perf_counter() - started
            print(f"  {start}/{len(samples)} ({elapsed:.1f}s)", flush=True)

    vlm_mem.flush()
    aux_mem.flush()
    np.save(args.out / "proprio.npy", proprio_labels)
    np.save(args.out / "action.npy", action_labels)
    metadata = {
        "samples": len(samples),
        "train": train_count,
        "val": val_count,
        "episodes": str(args.episodes),
        "vlm_shape": list(vlm_mem.shape),
        "aux_shape": list(aux_mem.shape),
        "seconds": round(time.perf_counter() - started, 1),
        "units": {
            "length": "m",
            "angle": "rad",
            "gripper": "normalized_0_1_closed_is_1",
        },
    }
    (args.out / "meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved {args.out} in {metadata['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
