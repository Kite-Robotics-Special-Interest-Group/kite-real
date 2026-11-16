"""Read-only Alicia-D SDK smoke check.

Examples:
    python scripts/alicia_real_check.py --port COM3
    python scripts/alicia_real_check.py --port /dev/ttyACM0 --home
    python scripts/alicia_real_check.py --action10 0.2 0 0.3 1 0 0 0 1 0 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure") and not sys.stdout.isatty():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from kite.robot.alicia import AliciaSdkClient, clamp_action10, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alicia-D 实机 SDK 自检")
    parser.add_argument("--config", default=None)
    parser.add_argument("--port", default=None, help="串口，例如 COM3 或 /dev/ttyACM0")
    parser.add_argument("--home", action="store_true", help="读取后回到零位（会动）")
    parser.add_argument(
        "--action10", type=float, nargs=10, default=None, help="校验一条动作"
    )
    parser.add_argument("--backend", default="numpy", choices=("cpp", "numpy", "torch"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    client = AliciaSdkClient(
        config_path=args.config,
        port=args.port,
        auto_connect=False,
        backend=args.backend,
    )
    client.connect()
    try:
        observation = client.read()
        payload = {
            "connected": client.connected,
            "sdk_version": client.version(),
            "joint_rad": observation.joint_positions.tolist(),
            "tcp_position_m": observation.position.tolist(),
            "gripper_ticks": observation.gripper_ticks,
            "gripper_norm_closed_is_1": observation.gripper_norm,
            "action10": observation.action10.tolist(),
            "proprio20": observation.proprio.tolist(),
            "units": config["data"]["required_units"],
        }
        if args.action10 is not None:
            clamped = clamp_action10(
                args.action10,
                previous_position=observation.position,
                previous_rotation=observation.rotation,
                config=config,
            )
            payload["action10_clamped"] = clamped.action.tolist()
            payload["interventions"] = clamped.interventions
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if args.home:
            client.home()
            print("home command sent")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
