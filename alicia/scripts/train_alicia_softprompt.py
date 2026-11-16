"""Train an Alicia-D soft prompt + domain action heads for X-VLA.

Only the new domain's soft prompt and ``DomainAwareLinear`` rows are trainable.
The Florence2 encoder and all shared transformer blocks stay frozen.

Examples:
    python scripts/train_alicia_softprompt.py --limit 200 --epochs 1
    python scripts/train_alicia_softprompt.py --epochs 2 --batch 4 \
        --out weights/X-VLA-WidowX-alicia/softprompt_domain19.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure") and not sys.stdout.isatty():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HORIZON = 30
DIM_ACTION = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Alicia-D X-VLA soft prompt")
    parser.add_argument(
        "--episodes", type=Path, default=REPO_ROOT / "outputs/alicia_episodes"
    )
    parser.add_argument(
        "--model", type=str, default=str(REPO_ROOT / "weights/X-VLA-WidowX")
    )
    parser.add_argument("--domain-id", type=int, default=19)
    parser.add_argument("--init-domain", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gripper-weight", type=float, default=1.0)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="特征训练时顺序读取（显著减少随机内存访问）",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "weights/X-VLA-WidowX-alicia/softprompt_domain19.pt",
    )
    parser.add_argument(
        "--log", type=Path, default=REPO_ROOT / "outputs/alicia_softprompt_log.json"
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--resume", type=Path, default=None, help="继续训练已有 domain 权重"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="预计算特征目录（见 scripts/precompute_alicia_features.py）",
    )
    parser.add_argument(
        "--train-blocks",
        type=int,
        default=0,
        help="额外解冻最后 N 个 transformer block",
    )
    return parser.parse_args()


def load_split(episodes: Path) -> tuple[list[str], list[str]]:
    split_path = episodes / "splits.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        return list(split.get("train", [])), list(split.get("val", []))
    ids = sorted(
        path.name for path in episodes.iterdir() if (path / "meta.json").exists()
    )
    cut = int(len(ids) * 0.85)
    return ids[:cut], ids[cut:]


def build_samples(episodes: Path, episode_ids: list[str]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        ep = episodes / episode_id
        meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
        if not meta.get("success", False):
            continue
        actions = np.load(ep / "action.npy")
        proprio = np.load(ep / "proprio.npy")
        for index in meta.get("frames", []):
            if index >= len(actions):
                continue
            samples.append(
                {
                    "episode": ep,
                    "index": int(index),
                    "instruction": meta["instruction"],
                    "actions": actions,
                    "proprio": proprio[index].astype(np.float32),
                }
            )
    return samples


def build_chunk(actions: np.ndarray, index: int) -> np.ndarray:
    chunk = np.zeros((HORIZON, DIM_ACTION), dtype=np.float32)
    future = actions[index : index + HORIZON, :10]
    if len(future) == 0:
        return chunk
    chunk[: len(future), :10] = future
    if len(future) < HORIZON:
        chunk[len(future) :, :10] = future[-1]
    return chunk


def collate_batch(
    processor: Any, samples: list[dict[str, Any]], device: str
) -> tuple[Any, ...]:
    import torch

    images = []
    instructions = []
    actions = []
    proprio = []
    for sample in samples:
        index = sample["index"]
        front = Image.open(
            sample["episode"] / "obs" / f"{index:04d}_front.jpg"
        ).convert("RGB")
        wrist = Image.open(
            sample["episode"] / "obs" / f"{index:04d}_wrist.jpg"
        ).convert("RGB")
        images.append([front, wrist])
        instructions.append(sample["instruction"])
        actions.append(build_chunk(sample["actions"], index))
        proprio.append(sample["proprio"])
    encoded = processor(images=images, language_instruction=instructions)
    return (
        encoded["input_ids"].to(device),
        encoded["image_input"].to(device),
        encoded["image_mask"].to(device),
        torch.as_tensor(np.stack(proprio), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack(actions), dtype=torch.float32, device=device),
    )


def _mask_rows(parameter: Any, domain_id: int) -> None:
    def hook(grad: Any) -> Any:
        mask = grad.new_zeros(grad.shape)
        mask[domain_id] = 1.0
        return grad * mask

    parameter.register_hook(hook)


def prepare_domain(model: Any, domain_id: int, init_domain: int) -> list[Any]:
    import torch

    transformer = model.transformer
    with torch.no_grad():
        transformer.soft_prompt_hub.weight[domain_id] = (
            transformer.soft_prompt_hub.weight[init_domain].clone()
        )
        for layer in (transformer.action_encoder, transformer.action_decoder):
            layer.fc.weight[domain_id] = layer.fc.weight[init_domain].clone()
            layer.bias.weight[domain_id] = layer.bias.weight[init_domain].clone()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trainable = [
        transformer.soft_prompt_hub.weight,
        transformer.action_encoder.fc.weight,
        transformer.action_encoder.bias.weight,
        transformer.action_decoder.fc.weight,
        transformer.action_decoder.bias.weight,
    ]
    for parameter in trainable:
        parameter.requires_grad_(True)
        _mask_rows(parameter, domain_id)
    return trainable


def run_epoch(
    model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    optimizer: Any,
    device: str,
    args: argparse.Namespace,
    train: bool,
) -> dict[str, float]:
    import torch

    model.eval()
    order = list(range(len(samples)))
    if train:
        random.Random(args.seed + int(time.time())).shuffle(order)
    totals = {
        "loss": 0.0,
        "position_loss": 0.0,
        "rotate6D_loss": 0.0,
        "gripper_loss": 0.0,
    }
    steps = 0
    for start in range(0, len(order), args.batch):
        batch = [samples[index] for index in order[start : start + args.batch]]
        input_ids, image_input, image_mask, proprio, action = collate_batch(
            processor, batch, device
        )
        batch_size = input_ids.shape[0]
        with torch.no_grad():
            if args.no_amp:
                encoded = model.forward_vlm(input_ids, image_input, image_mask)
            else:
                with torch.autocast("cuda", dtype=torch.float16):
                    encoded = model.forward_vlm(input_ids, image_input, image_mask)
            encoded = {key: value.float() for key, value in encoded.items()}
        t = (
            torch.rand(1, device=device)
            + torch.arange(batch_size, device=device) / batch_size
        ) % (1 - 1e-5)
        noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(
            -1, 1, 1
        )
        proprio_m, noisy_m = model.action_space.preprocess(proprio, noisy)
        with torch.set_grad_enabled(train):
            prediction = model.transformer(
                domain_id=torch.full(
                    (batch_size,), args.domain_id, dtype=torch.long, device=device
                ),
                action_with_noise=noisy_m,
                proprio=proprio_m,
                t=t,
                **encoded,
            )
            losses = model.action_space.compute_loss(prediction, action)
            loss = (
                losses["position_loss"]
                + losses["rotate6D_loss"]
                + args.gripper_weight * losses["gripper_loss"]
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
        for key in totals:
            totals[key] += float(
                losses[key].detach() if key in losses else loss.detach()
            )
        steps += 1
        if train and steps % 50 == 0:
            print(
                f"  step {steps} loss={float(loss.detach()):.4f} "
                f"pos={float(losses['position_loss']):.3f} "
                f"rot={float(losses['rotate6D_loss']):.3f} "
                f"grip={float(losses['gripper_loss']):.3f}",
                flush=True,
            )
    return {key: value / max(steps, 1) for key, value in totals.items()}


def run_epoch_features(
    model: Any,
    features: dict[str, Any],
    optimizer: Any,
    device: str,
    args: argparse.Namespace,
    indices: list[int],
    train: bool,
) -> dict[str, float]:
    import torch

    model.eval()
    order = list(indices)
    if train and not args.no_shuffle:
        random.Random(args.seed + int(time.time())).shuffle(order)
    vlm_mem = features["vlm"]
    aux_mem = features["aux"]
    proprio_labels = features["proprio"]
    action_labels = features["action"]
    totals = {
        "loss": 0.0,
        "position_loss": 0.0,
        "rotate6D_loss": 0.0,
        "gripper_loss": 0.0,
    }
    steps = 0
    for start in range(0, len(order), args.batch):
        index = order[start : start + args.batch]
        encoded = {
            "vlm_features": torch.as_tensor(
                np.asarray(vlm_mem[index]), dtype=torch.float32, device=device
            ),
            "aux_visual_inputs": torch.as_tensor(
                np.asarray(aux_mem[index]), dtype=torch.float32, device=device
            ),
        }
        proprio = torch.as_tensor(
            proprio_labels[index], dtype=torch.float32, device=device
        )
        action = torch.as_tensor(
            action_labels[index], dtype=torch.float32, device=device
        )
        batch_size = len(index)
        t = (
            torch.rand(1, device=device)
            + torch.arange(batch_size, device=device) / batch_size
        ) % (1 - 1e-5)
        noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(
            -1, 1, 1
        )
        proprio_m, noisy_m = model.action_space.preprocess(proprio, noisy)
        with torch.set_grad_enabled(train):
            prediction = model.transformer(
                domain_id=torch.full(
                    (batch_size,), args.domain_id, dtype=torch.long, device=device
                ),
                action_with_noise=noisy_m,
                proprio=proprio_m,
                t=t,
                **encoded,
            )
            losses = model.action_space.compute_loss(prediction, action)
            loss = (
                losses["position_loss"]
                + losses["rotate6D_loss"]
                + args.gripper_weight * losses["gripper_loss"]
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
        for key in totals:
            totals[key] += float(
                losses[key].detach() if key in losses else loss.detach()
            )
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


def save_checkpoint(
    model: Any, args: argparse.Namespace, metrics: dict[str, Any]
) -> None:
    import torch

    transformer = model.transformer
    args.out.parent.mkdir(parents=True, exist_ok=True)
    domain = args.domain_id
    checkpoint = {
        "domain_id": domain,
        "init_domain": args.init_domain,
        "model": args.model,
        "soft_prompt": transformer.soft_prompt_hub.weight[domain].detach().cpu(),
        "action_encoder_fc": transformer.action_encoder.fc.weight[domain]
        .detach()
        .cpu(),
        "action_encoder_bias": transformer.action_encoder.bias.weight[domain]
        .detach()
        .cpu(),
        "action_decoder_fc": transformer.action_decoder.fc.weight[domain]
        .detach()
        .cpu(),
        "action_decoder_bias": transformer.action_decoder.bias.weight[domain]
        .detach()
        .cpu(),
        "metrics": metrics,
        "args": {key: str(value) for key, value in vars(args).items()},
    }
    torch.save(checkpoint, args.out)
    print(f"saved {args.out}", flush=True)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    from transformers import AutoModel, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    device = "cuda"
    model = (
        AutoModel.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    trainable = prepare_domain(model, args.domain_id, args.init_domain)
    if args.resume is not None:
        state = torch.load(str(args.resume), map_location=device, weights_only=False)
        domain = args.domain_id
        transformer = model.transformer
        with torch.no_grad():
            transformer.soft_prompt_hub.weight[domain] = state["soft_prompt"]
            transformer.action_encoder.fc.weight[domain] = state["action_encoder_fc"]
            transformer.action_encoder.bias.weight[domain] = state[
                "action_encoder_bias"
            ]
            transformer.action_decoder.fc.weight[domain] = state["action_decoder_fc"]
            transformer.action_decoder.bias.weight[domain] = state[
                "action_decoder_bias"
            ]
        print(f"resumed domain={domain} from {args.resume}", flush=True)
    if args.train_blocks:
        blocks = model.transformer.blocks
        for block in blocks[-args.train_blocks :]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        trainable.extend(
            parameter
            for block in blocks[-args.train_blocks :]
            for parameter in block.parameters()
        )
    print(
        f"domain={args.domain_id} init={args.init_domain} "
        f"trainable={sum(p.numel() for p in trainable) / 1e6:.3f}M",
        flush=True,
    )

    features: dict[str, Any] | None = None
    processor = None
    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []
    train_indices: list[int] = []
    val_indices: list[int] = []
    if args.features:
        metadata = json.loads((args.features / "meta.json").read_text(encoding="utf-8"))
        features = {
            "vlm": np.load(args.features / "vlm_features.npy"),
            "aux": np.load(args.features / "aux_visual.npy"),
            "proprio": np.load(args.features / "proprio.npy"),
            "action": np.load(args.features / "action.npy"),
        }
        train_indices = list(range(int(metadata["train"])))
        val_indices = list(
            range(int(metadata["train"]), int(metadata["train"]) + int(metadata["val"]))
        )
        if args.limit:
            train_indices = train_indices[: args.limit]
        val_indices = val_indices[: args.val_limit]
        print(
            f"features train={len(train_indices)} val={len(val_indices)} "
            f"shape={features['vlm'].shape}",
            flush=True,
        )
    else:
        train_ids, val_ids = load_split(args.episodes)
        train_samples = build_samples(args.episodes, train_ids)
        val_samples = build_samples(args.episodes, val_ids)[: args.val_limit]
        if args.limit:
            train_samples = train_samples[: args.limit]
        print(
            f"episodes train={len(train_ids)} val={len(val_ids)} "
            f"samples train={len(train_samples)} val={len(val_samples)}",
            flush=True,
        )
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        started = time.perf_counter()
        if features is not None:
            train_metrics = run_epoch_features(
                model, features, optimizer, device, args, train_indices, train=True
            )
            val_metrics = (
                run_epoch_features(
                    model, features, optimizer, device, args, val_indices, train=False
                )
                if val_indices
                else {}
            )
        else:
            train_metrics = run_epoch(
                model, processor, train_samples, optimizer, device, args, train=True
            )
            val_metrics = (
                run_epoch(
                    model, processor, val_samples, optimizer, device, args, train=False
                )
                if val_samples
                else {}
            )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "seconds": round(time.perf_counter() - started, 1),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        save_checkpoint(model, args, {"history": history})
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
