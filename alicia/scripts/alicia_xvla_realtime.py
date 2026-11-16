"""Run the original X-VLA-WidowX weights on Alicia-D in MuJoCo (CUDA).

This is a live, supervisor-friendly loop:
render front/wrist -> X-VLA (CUDA) -> absolute EEF chunk -> Alicia MuJoCo.

Examples:
    python scripts/alicia_xvla_realtime.py --duration 30
    python scripts/alicia_xvla_realtime.py --headless --cycles 2 --exec-steps 5
    python scripts/alicia_xvla_realtime.py --instruction "pick up the red cube"
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure") and not sys.stdout.isatty():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if platform.system() == "Windows":
    os.environ.setdefault("MUJOCO_GL", "wgl")

import torch
from kite.robot.alicia import load_config, matrix_to_rot6d
from kite.sim.alicia import AliciaSimEnv
from PIL import Image
from transformers import AutoModel, AutoProcessor

DOMAIN_ID = 0
MODEL_PATH = REPO_ROOT / "weights" / "X-VLA-WidowX"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="X-VLA 原版权重 + Alicia-D 实时仿真")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None, help="Alicia soft prompt .pt")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument(
        "--grasp-assist", action="store_true", help="演示用：闭合时吸附方块"
    )
    parser.add_argument(
        "--fixed-orientation",
        action="store_true",
        help="演示用：保持示教时的固定 EEF 朝向",
    )
    parser.add_argument(
        "--position-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="部署标定偏移（米）",
    )
    parser.add_argument("--instruction", default="pick up the red cube")
    parser.add_argument("--steps", type=int, default=10, help="流匹配去噪步数")
    parser.add_argument(
        "--exec-steps", type=int, default=10, help="每次推理执行多少步动作"
    )
    parser.add_argument("--cycles", type=int, default=3, help="无窗口模式推理轮数")
    parser.add_argument(
        "--duration", type=float, default=0.0, help="窗口自动关闭时间（秒）"
    )
    parser.add_argument("--camera", choices=("front", "wrist"), default="front")
    parser.add_argument("--no-wrist-inset", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--no-clamp", action="store_true", help="关闭安全限幅（仅调试）"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshot", default=None)
    return parser.parse_args()


class XvlaPolicy:
    """Original X-VLA checkpoint loaded with the stock processor."""

    def __init__(
        self,
        model_path: Path,
        instruction: str,
        steps: int,
        checkpoint: Path | None = None,
        domain_id: int = 0,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.backends.cuda.matmul.allow_tf32 = True
        self.device = torch.device("cuda")
        self.steps = steps
        self.model = (
            AutoModel.from_pretrained(
                str(model_path), trust_remote_code=True, torch_dtype=torch.float32
            )
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        self.input_ids = self.processor.encode_language(instruction)["input_ids"].to(
            self.device
        )
        self.domain = int(domain_id)
        if checkpoint is not None:
            state = torch.load(
                str(checkpoint), map_location=self.device, weights_only=False
            )
            self.domain = int(state.get("domain_id", domain_id))
            transformer = self.model.transformer
            with torch.no_grad():
                transformer.soft_prompt_hub.weight[self.domain] = state["soft_prompt"]
                transformer.action_encoder.fc.weight[self.domain] = state[
                    "action_encoder_fc"
                ]
                transformer.action_encoder.bias.weight[self.domain] = state[
                    "action_encoder_bias"
                ]
                transformer.action_decoder.fc.weight[self.domain] = state[
                    "action_decoder_fc"
                ]
                transformer.action_decoder.bias.weight[self.domain] = state[
                    "action_decoder_bias"
                ]
            print(f"loaded soft prompt domain={self.domain} from {checkpoint}")
        self.domain_id = torch.full(
            (1,), self.domain, dtype=torch.long, device=self.device
        )
        self.fixed_rot6d: np.ndarray | None = None
        self.position_offset = np.zeros(3, dtype=np.float64)
        self.instruction = instruction

    @torch.inference_mode()
    def act(
        self, front: np.ndarray, wrist: np.ndarray, proprio: np.ndarray
    ) -> tuple[np.ndarray, float]:
        images = [Image.fromarray(front), Image.fromarray(wrist)]
        encoded = self.processor.encode_image(images)
        image_input = encoded["image_input"].to(self.device)
        image_mask = encoded["image_mask"].to(self.device)
        proprio_tensor = torch.as_tensor(
            proprio, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        start = time.perf_counter()
        action = self.model.generate_actions(
            input_ids=self.input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=self.domain_id,
            proprio=proprio_tensor,
            steps=self.steps,
        )
        latency = time.perf_counter() - start
        chunk = action[0, :, :10].detach().float().cpu().numpy()
        if self.fixed_rot6d is not None or np.any(self.position_offset):
            chunk = chunk.copy()
            chunk[:, :3] += self.position_offset
            if self.fixed_rot6d is not None:
                chunk[:, 3:9] = self.fixed_rot6d
        return chunk, latency

    @torch.inference_mode()
    def warmup(self, front: np.ndarray, wrist: np.ndarray, proprio: np.ndarray) -> None:
        for _ in range(2):
            self.act(front, wrist, proprio)


def _observation(env: AliciaSimEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return env.render("front"), env.render("wrist"), env.proprio()


def _apply_grasp_assist(env: AliciaSimEnv, action: np.ndarray, enabled: bool) -> None:
    if not enabled:
        return
    grip = float(action[9])
    if grip < 0.2:
        env.detach_cube()
        return
    if grip > 0.6 and not env.cube_attached:
        cube = env.cube_position()
        if cube is not None and np.linalg.norm(cube - env.eef_pose()[0]) < 0.06:
            env.attach_cube()


def _print_chunk(chunk: np.ndarray, latency: float) -> None:
    print(
        json.dumps(
            {
                "latency_ms": round(latency * 1000, 1),
                "chunk_pos_first": np.round(chunk[0, :3], 4).tolist(),
                "chunk_pos_last": np.round(chunk[-1, :3], 4).tolist(),
                "grip_first": round(float(chunk[0, 9]), 3),
                "grip_last": round(float(chunk[-1, 9]), 3),
            },
            ensure_ascii=False,
        )
    )


def _save_snapshot(env: AliciaSimEnv, path: Path) -> None:
    front, wrist, _ = _observation(env)
    canvas = Image.new("RGB", (front.shape[1] * 2, front.shape[0]))
    canvas.paste(Image.fromarray(front), (0, 0))
    canvas.paste(Image.fromarray(wrist), (front.shape[1], 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _run_headless(
    env: AliciaSimEnv, policy: XvlaPolicy, args: argparse.Namespace
) -> int:
    for cycle in range(args.cycles):
        front, wrist, proprio = _observation(env)
        chunk, latency = policy.act(front, wrist, proprio)
        _print_chunk(chunk, latency)
        for action in chunk[: args.exec_steps]:
            _apply_grasp_assist(env, action, args.grasp_assist)
            env.execute_action(action, clamp=not args.no_clamp)
        state = env.state()
        cube = env.cube_position()
        print(
            f"cycle {cycle + 1}/{args.cycles} "
            f"tcp={np.round(state.position, 3).tolist()} "
            f"cube={None if cube is None else np.round(cube, 3).tolist()} "
            f"err={state.position_error * 1000:.1f}mm "
            f"grip={state.gripper_norm:.2f} attached={env.cube_attached}"
        )
    if args.snapshot:
        _save_snapshot(env, Path(args.snapshot))
        print(f"snapshot: {args.snapshot}")
    return 0


def _run_viewer(env: AliciaSimEnv, policy: XvlaPolicy, args: argparse.Namespace) -> int:
    import mujoco.viewer

    flags = {"paused": False, "running": True, "policy": True, "brake": False}
    started = time.monotonic()
    chunk: np.ndarray | None = None
    chunk_index = 0
    latency = 0.0
    interventions: list[str] = []

    def key_callback(keycode: int) -> None:
        nonlocal chunk, chunk_index
        key = chr(keycode).lower() if 32 <= keycode < 127 else ""
        if keycode == 256:
            flags["running"] = False
        elif key == " ":
            flags["paused"] = not flags["paused"]
        elif key == "t":
            flags["policy"] = not flags["policy"]
            flags["brake"] = not flags["policy"]
            chunk = None
        elif key == "b":
            flags["brake"] = True
            flags["policy"] = False
            chunk = None
            env.set_target(env.eef_pose()[0])
        elif key == "r":
            env.reset()
            chunk = None
            chunk_index = 0
        elif key == "p":
            print(json.dumps(env.state().__dict__, ensure_ascii=False, default=list))
        elif key == "g":
            value = 0.0 if env.gripper_norm() > 0.5 else 1.0
            env.set_gripper_norm(value)

    viewer = mujoco.viewer.launch_passive(
        env.model, env.data, key_callback=key_callback
    )
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = env.camera_ids[args.camera]
    last_hud = 0.0
    try:
        while flags["running"] and viewer.is_running():
            now = time.monotonic()
            if args.duration > 0.0 and now - started > args.duration:
                break
            if not flags["paused"]:
                if flags["policy"]:
                    if chunk is None or chunk_index >= min(args.exec_steps, len(chunk)):
                        front, wrist, proprio = _observation(env)
                        chunk, latency = policy.act(front, wrist, proprio)
                        chunk_index = 0
                        _print_chunk(chunk, latency)
                    _apply_grasp_assist(env, chunk[chunk_index], args.grasp_assist)
                    interventions = env.execute_action(
                        chunk[chunk_index], clamp=not args.no_clamp
                    )
                    chunk_index += 1
                else:
                    env.step()
            if now - last_hud > 0.2:
                last_hud = now
                state = env.state()
                mode = (
                    "POLICY"
                    if flags["policy"]
                    else ("BRAKE" if flags["brake"] else "HOLD")
                )
                pause = "PAUSED" if flags["paused"] else "RUN"
                action = "none" if chunk is None else np.round(chunk[0, :3], 3).tolist()
                viewer.set_texts(
                    [
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            "X-VLA → Alicia-D (CUDA)",
                            f"{mode} | {pause}",
                        ),
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                            (
                                f"tcp {np.round(state.position, 3)} | "
                                f"err {state.position_error * 1000:.1f} mm"
                            ),
                            (
                                f"grip {state.gripper_norm:.2f} | "
                                f"infer {latency * 1000:.0f} ms"
                            ),
                        ),
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                            f"instruction: {policy.instruction}",
                            (
                                f"action {action} | clamp "
                                f"{','.join(interventions) if interventions else '-'} "
                                "| space pause  b brake  t policy  r reset"
                            ),
                        ),
                    ]
                )
            if not args.no_wrist_inset:
                image = env.render("wrist")
                viewer.set_images(
                    [(mujoco.MjrRect(8, 8, image.shape[1], image.shape[0]), image)]
                )
            viewer.sync()
            remaining = env.control_period - (time.monotonic() - now)
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        viewer.close()
    return 0


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"model: {args.model}")
    print(f"instruction: {args.instruction!r}")
    env = AliciaSimEnv(config=config, with_task=True, seed=args.seed)
    try:
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        policy = XvlaPolicy(
            Path(args.model),
            args.instruction,
            args.steps,
            checkpoint=checkpoint,
            domain_id=args.domain_id,
        )
        if args.fixed_orientation:
            policy.fixed_rot6d = matrix_to_rot6d(env.eef_pose()[1])
        policy.position_offset = np.asarray(args.position_offset, dtype=np.float64)
        front, wrist, proprio = _observation(env)
        print("warmup: 2 inferences")
        policy.warmup(front, wrist, proprio)
        if args.headless:
            return _run_headless(env, policy, args)
        return _run_viewer(env, policy, args)
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
