"""Run the trained Alicia-D X-VLA policy on the real robot.

The default is read-only: it connects, reads state, captures cameras and prints
the planned actions. Add ``--execute`` to send motion commands.

Example:
    python scripts/alicia_real_policy.py --port COM3 \
      --front-camera 0 --wrist-camera 1 --cycles 1 --exec-steps 5
    python scripts/alicia_real_policy.py --port COM3 --execute ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kite.robot.alicia import (  # noqa: E402
    AliciaSdkClient,
    action10_to_sdk_pose,
    clamp_action10,
    load_config,
)

HORIZON = 30
ACTION_DIM = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alicia-D 真机 X-VLA 策略")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=str(REPO_ROOT / "weights" / "X-VLA-WidowX"))
    parser.add_argument(
        "--checkpoint",
        default=str(
            REPO_ROOT / "weights" / "X-VLA-WidowX-alicia" / "softprompt_domain19.pt"
        ),
    )
    parser.add_argument("--domain-id", type=int, default=19)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--front-camera", type=int, default=0)
    parser.add_argument("--wrist-camera", type=int, default=1)
    parser.add_argument("--instruction", default="pick up the red block")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--exec-steps", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--speed-deg-s", type=float, default=10.0)
    parser.add_argument("--execute", action="store_true", help="真正发送动作")
    parser.add_argument("--log", default=None)
    return parser.parse_args()


def _load_policy(model_path: Path, checkpoint: Path, domain_id: int):
    model = (
        AutoModel.from_pretrained(
            str(model_path), trust_remote_code=True, torch_dtype=torch.float32
        )
        .to("cuda")
        .eval()
    )
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    state = torch.load(str(checkpoint), map_location="cuda", weights_only=False)
    domain = int(state.get("domain_id", domain_id))
    transformer = model.transformer
    with torch.no_grad():
        transformer.soft_prompt_hub.weight[domain] = state["soft_prompt"]
        transformer.action_encoder.fc.weight[domain] = state["action_encoder_fc"]
        transformer.action_encoder.bias.weight[domain] = state["action_encoder_bias"]
        transformer.action_decoder.fc.weight[domain] = state["action_decoder_fc"]
        transformer.action_decoder.bias.weight[domain] = state["action_decoder_bias"]
    domain_tensor = torch.full((1,), domain, dtype=torch.long, device="cuda")
    return model, processor, domain_tensor, domain


def _open_cameras(front_id: int, wrist_id: int):
    cameras = {}
    for name, index in (("front", front_id), ("wrist", wrist_id)):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {name} camera index {index}")
        cameras[name] = capture
    return cameras


def _read_rgb(capture, size: int = 224) -> np.ndarray:
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError("camera read failed")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(f"CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    model, processor, domain_tensor, domain = _load_policy(
        Path(args.model), Path(args.checkpoint), args.domain_id
    )
    print(f"loaded domain={domain}", flush=True)

    client = AliciaSdkClient(config_path=args.config, port=args.port, auto_connect=False)
    client.connect()
    cameras = _open_cameras(args.front_camera, args.wrist_camera)
    log_path = Path(args.log) if args.log else None
    log_file = log_path.open("a", encoding="utf-8") if log_path else None
    try:
        for cycle in range(args.cycles):
            observation = client.read()
            front = _read_rgb(cameras["front"])
            wrist = _read_rgb(cameras["wrist"])
            encoded = processor.encode_image([Image.fromarray(front), Image.fromarray(wrist)])
            input_ids = processor.encode_language(args.instruction)["input_ids"].to("cuda")
            image_input = encoded["image_input"].to("cuda")
            image_mask = encoded["image_mask"].to("cuda")
            proprio = torch.as_tensor(
                observation.proprio, dtype=torch.float32, device="cuda"
            ).unsqueeze(0)
            with torch.inference_mode():
                action = model.generate_actions(
                    input_ids=input_ids,
                    image_input=image_input,
                    image_mask=image_mask,
                    domain_id=domain_tensor,
                    proprio=proprio,
                    steps=args.steps,
                )
            chunk = action[0, :, :10].detach().float().cpu().numpy()
            print(
                f"cycle {cycle} pose={np.round(observation.position, 4).tolist()} "
                f"grip={observation.gripper_norm:.3f}",
                flush=True,
            )
            for index, action10 in enumerate(chunk[: args.exec_steps]):
                clamped = clamp_action10(
                    action10,
                    previous_position=observation.position,
                    previous_rotation=observation.rotation,
                    config=config,
                )
                pose, ticks = action10_to_sdk_pose(clamped.action, config)
                record = {
                    "cycle": cycle,
                    "step": index,
                    "pose": pose,
                    "ticks": ticks,
                    "interventions": clamped.interventions,
                    "execute": bool(args.execute),
                }
                print(json.dumps(record, ensure_ascii=False), flush=True)
                if log_file:
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_file.flush()
                if args.execute:
                    client.send_action10(
                        clamped.action,
                        speed_deg_s=args.speed_deg_s,
                        execute=True,
                        previous=observation,
                    )
                    observation = client.read()
                else:
                    time.sleep(0.05)
    except KeyboardInterrupt:
        print("interrupted; holding", flush=True)
        if args.execute:
            client.hold()
    finally:
        for capture in cameras.values():
            capture.release()
        client.close()
        if log_file:
            log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
