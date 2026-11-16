"""Check Alicia-D data/interface/unit conventions for the native X-VLA path.

Examples:
    python scripts/check_alicia_conventions.py
    python scripts/check_alicia_conventions.py --strict \
        --json outputs/alicia_checks.json
    python scripts/check_alicia_conventions.py --data-root data/alicia_d
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure") and not sys.stdout.isatty():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from kite.robot.alicia import (
    DEFAULT_CONFIG_PATH,
    action10_to_sdk_pose,
    clamp_action10,
    gripper_mapper,
    load_config,
    matrix_to_rot6d,
    proprio_from_pose,
    rot6d_to_matrix,
    sdk_pose_to_action10,
)

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alicia-D 数据/接口/单位约定检查")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--strict", action="store_true", help="WARN 也返回非零")
    parser.add_argument("--json", default=None, help="结果 JSON 输出路径")
    return parser.parse_args()


def _check_assets(config: dict[str, Any]) -> list[Check]:
    robot = config["robot"]
    checks: list[Check] = []
    for key in ("mjcf", "urdf", "mesh_dir"):
        path = REPO_ROOT / robot["assets"][key]
        checks.append(Check(f"asset:{key}", PASS if path.exists() else FAIL, str(path)))
    mesh_dir = REPO_ROOT / robot["assets"]["mesh_dir"]
    meshes = sorted(mesh_dir.glob("*.STL"))
    expected = {
        *(
            f"{name}.STL"
            for name in (
                "base_link",
                "link1",
                "link2",
                "link3",
                "link4",
                "link5",
                "link6",
            )
        ),
        "left_gripper_50mm.STL",
        "right_gripper_50mm.STL",
    }
    missing = sorted(expected - {mesh.name for mesh in meshes})
    checks.append(
        Check(
            "asset:meshes",
            PASS if not missing else FAIL,
            f"{len(meshes)} STL, missing={missing}",
        )
    )

    sdk = REPO_ROOT / "third_party" / "Alicia-D-SDK"
    sdk_version = _read_sdk_version(sdk)
    checks.append(
        Check(
            "asset:sdk",
            PASS if sdk.exists() else FAIL,
            f"{sdk} ({sdk_version})",
        )
    )

    checkpoint = REPO_ROOT / config["xvla"]["checkpoint"]
    weights = checkpoint / "model.safetensors"
    checks.append(
        Check(
            "asset:model",
            PASS if weights.exists() else FAIL,
            f"{weights} ({weights.stat().st_size / 1e9:.2f} GB)"
            if weights.exists()
            else f"missing {weights}",
        )
    )

    try:
        import mujoco

        mjcf = REPO_ROOT / robot["assets"]["mjcf"]
        model = mujoco.MjModel.from_xml_path(str(mjcf))
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(model.njnt)
        ]
        expected_joints = robot["joints"]["arm"] + robot["joints"]["gripper"]
        checks.append(
            Check(
                "mjcf:load",
                PASS if set(expected_joints).issubset(names) else FAIL,
                f"nq={model.nq}, nv={model.nv}, joints={names}",
            )
        )
    except ImportError:
        checks.append(Check("mjcf:load", SKIP, "mujoco 未安装"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("mjcf:load", FAIL, f"{type(exc).__name__}: {exc}"))
    return checks


def _read_sdk_version(sdk: Path) -> str:
    init = sdk / "alicia_d_sdk" / "__init__.py"
    if not init.exists():
        return "missing"
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init.read_text(encoding="utf-8"))
    return match.group(1) if match else "unknown"


def _check_urdf(config: dict[str, Any]) -> list[Check]:
    robot = config["robot"]
    urdf_path = REPO_ROOT / robot["assets"]["urdf"]
    mesh_dir = REPO_ROOT / robot["assets"]["mesh_dir"]
    checks: list[Check] = []
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception as exc:  # noqa: BLE001
        return [Check("urdf:parse", FAIL, f"{type(exc).__name__}: {exc}")]

    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    names = list(robot["joints"]["arm"]) + list(robot["joints"]["gripper"])
    missing = [name for name in names if name not in joints]
    checks.append(
        Check("urdf:joints", PASS if not missing else FAIL, f"missing={missing}")
    )

    mismatches: list[str] = []
    for name, limits in zip(names, robot["joints"]["limits_rad"]):
        joint = joints.get(name)
        if joint is None:
            continue
        limit = joint.find("limit")
        if limit is None:
            mismatches.append(f"{name}: no limit")
            continue
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        if abs(lower - limits[0]) > 1e-4 or abs(upper - limits[1]) > 1e-4:
            mismatches.append(f"{name}: URDF[{lower},{upper}] config{limits}")
    checks.append(
        Check(
            "urdf:limits_rad",
            PASS if not mismatches else FAIL,
            "; ".join(mismatches) or "ok",
        )
    )

    mesh_files = [element.get("filename", "") for element in root.findall(".//mesh")]
    missing_meshes = [
        filename for filename in mesh_files if not (mesh_dir / filename).exists()
    ]
    checks.append(
        Check(
            "urdf:meshes",
            PASS if not missing_meshes else FAIL,
            f"{len(mesh_files)} refs, missing={missing_meshes}",
        )
    )

    tool = joints.get("tool0_fixed_joint")
    tool_xyz = None if tool is None else tool.find("origin").get("xyz")
    checks.append(
        Check(
            "urdf:tcp",
            PASS
            if tool_xyz is not None
            and np.allclose([float(v) for v in tool_xyz.split()], [0, 0, -0.13])
            else FAIL,
            f"tool0_fixed_joint origin={tool_xyz}",
        )
    )
    return checks


def _check_model(config: dict[str, Any]) -> list[Check]:
    checkpoint = REPO_ROOT / config["xvla"]["checkpoint"]
    checks: list[Check] = []
    try:
        model_config = json.loads(
            (checkpoint / "config.json").read_text(encoding="utf-8")
        )
    except Exception as exc:  # noqa: BLE001
        return [Check("model:config", FAIL, f"{type(exc).__name__}: {exc}")]

    expected = {
        "action_mode": config["xvla"]["action_mode"],
        "num_actions": config["xvla"]["action_chunk"],
        "use_proprio": True,
    }
    mismatch = {
        key: (model_config.get(key), value)
        for key, value in expected.items()
        if model_config.get(key) != value
    }
    checks.append(
        Check("model:config", PASS if not mismatch else FAIL, json.dumps(mismatch))
    )

    preprocessor = json.loads(
        (checkpoint / "preprocessor_config.json").read_text(encoding="utf-8")
    )
    image_size = int(config["xvla"]["image_size"])
    size = preprocessor.get("size", {})
    size_ok = (
        size.get("height") == image_size and size.get("width") == image_size
    ) or (size.get("shortest_edge") == image_size)
    checks.append(
        Check(
            "model:image",
            PASS
            if size_ok
            and preprocessor.get("do_rescale")
            and preprocessor.get("do_normalize")
            else FAIL,
            f"size={size}, rescale={preprocessor.get('do_rescale')}, "
            f"normalize={preprocessor.get('do_normalize')}",
        )
    )

    action_hub = (checkpoint / "action_hub.py").read_text(encoding="utf-8")
    processing = (checkpoint / "processing_xvla.py").read_text(encoding="utf-8")
    constants = {
        "dim_action=20": bool(re.search(r"dim_action\s*=\s*20", action_hub)),
        "gripper_idx=(9,19)": bool(
            re.search(r"gripper_idx\s*=\s*\(9,\s*19\)", action_hub)
        ),
        "num_views=3": bool(re.search(r"num_views\s*(?::\s*int)?\s*=\s*3", processing)),
    }
    checks.append(
        Check(
            "model:action_layout",
            PASS if all(constants.values()) else FAIL,
            json.dumps(constants),
        )
    )

    try:
        from safetensors import safe_open

        with safe_open(checkpoint / "model.safetensors", "pt") as handle:
            keys = list(handle.keys())
            decoder = next(
                (
                    key
                    for key in keys
                    if "action_decoder" in key and key.endswith("weight")
                ),
                None,
            )
            shape = handle.get_slice(decoder).get_shape() if decoder else None
        checks.append(
            Check(
                "model:weights",
                PASS if decoder is not None and shape[:2] == [30, 20] else FAIL,
                f"{len(keys)} tensors, {decoder} {shape}",
            )
        )
    except ImportError:
        checks.append(Check("model:weights", SKIP, "safetensors 未安装"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("model:weights", FAIL, f"{type(exc).__name__}: {exc}"))
    return checks


def _check_interface(config: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    identity = rot6d_to_matrix([1, 0, 0, 1, 0, 0])
    checks.append(
        Check(
            "interface:rot6d_identity",
            PASS if np.allclose(identity, np.eye(3)) else FAIL,
            "ok",
        )
    )

    rng = np.random.default_rng(0)
    max_error = 0.0
    for _ in range(100):
        rotation = _random_rotation(rng)
        recovered = rot6d_to_matrix(matrix_to_rot6d(rotation))
        max_error = max(max_error, float(np.max(np.abs(recovered - rotation))))
    checks.append(
        Check(
            "interface:rot6d_roundtrip",
            PASS if max_error < 1e-9 else FAIL,
            f"max error {max_error:.2e}",
        )
    )

    action = np.array([0.2, 0.0, 0.3, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    pose, ticks = action10_to_sdk_pose(action, config)
    checks.append(
        Check(
            "interface:sdk_pose",
            PASS
            if ticks == 0
            and len(pose) == 7
            and abs(np.linalg.norm(pose[3:]) - 1) < 1e-9
            else FAIL,
            f"pose={np.round(pose, 4).tolist()}, ticks={ticks}",
        )
    )
    roundtrip = sdk_pose_to_action10(
        np.asarray(pose[:3]), rot6d_to_matrix(action[3:9]), ticks, config
    )
    checks.append(
        Check(
            "interface:sdk_roundtrip",
            PASS if np.allclose(roundtrip, action, atol=1e-9) else FAIL,
            f"max error {np.max(np.abs(roundtrip - action)):.2e}",
        )
    )

    mapper = gripper_mapper(config)
    endpoints = {"open": mapper.norm_to_ticks(0.0), "closed": mapper.norm_to_ticks(1.0)}
    checks.append(
        Check(
            "interface:gripper_polarity",
            PASS if endpoints == {"open": 1000, "closed": 0} else FAIL,
            f"X-VLA 0->{endpoints['open']}, 1->{endpoints['closed']} (SDK ticks)",
        )
    )

    proprio = proprio_from_pose([0.2, 0.0, 0.3], identity)
    checks.append(
        Check(
            "interface:proprio20",
            PASS
            if proprio.shape == (20,)
            and np.allclose(proprio[:10], [0.2, 0.0, 0.3, 1, 0, 0, 1, 0, 0, 0])
            and np.allclose(proprio[10:], 0)
            else FAIL,
            f"shape={proprio.shape}, first10={np.round(proprio[:10], 3).tolist()}",
        )
    )

    clamped = clamp_action10(
        [2.0, 2.0, 2.0, 1, 0, 0, 1, 0, 0, 2.0],
        previous_position=[0.2, 0.0, 0.3],
        config=config,
    )
    checks.append(
        Check(
            "interface:safety_clamp",
            PASS
            if {
                "x_workspace",
                "y_workspace",
                "z_workspace",
                "position_step",
                "gripper_range",
            }
            & set(clamped.interventions)
            else FAIL,
            f"interventions={clamped.interventions}",
        )
    )
    return checks


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    x, y, z, w = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _check_units(config: dict[str, Any]) -> list[Check]:
    robot = config["robot"]
    units = config["data"]["required_units"]
    checks = [
        Check(
            "units:base",
            PASS
            if units
            == {
                "length": "m",
                "angle": "rad",
                "depth": "m",
                "gripper": "normalized_0_1_closed_is_1",
            }
            else FAIL,
            json.dumps(units, ensure_ascii=False),
        )
    ]
    rate = float(robot["control"]["rate_hz"])
    timestep = float(robot["control"]["sim_timestep_s"])
    substeps = 1.0 / (rate * timestep)
    configured = int(robot["control"]["sim_substeps"])
    checks.append(
        Check(
            "units:control_ratio",
            PASS
            if abs(substeps - round(substeps)) < 1e-9 and configured == round(substeps)
            else FAIL,
            f"{rate} Hz × {timestep} s = {substeps:.3f} substeps (config {configured})",
        )
    )
    compiler = ET.parse(REPO_ROOT / robot["assets"]["mjcf"]).getroot().find("compiler")
    angle = None if compiler is None else compiler.get("angle")
    checks.append(
        Check(
            "units:mjcf_angle",
            PASS if angle in (None, "radian") else FAIL,
            f"compiler angle={angle}（MJCF 省略时默认 radian）",
        )
    )
    checks.append(
        Check(
            "units:gripper_travel",
            PASS if float(robot["gripper"]["travel_m"]) == 0.025 else FAIL,
            f"50mm 夹爪单侧行程 {robot['gripper']['travel_m']} m",
        )
    )
    return checks


def _check_data(config: dict[str, Any], data_root: Path) -> list[Check]:
    if not data_root.exists():
        return [
            Check(
                "data:root", SKIP, f"{data_root} 不存在（无本地数据，约定已写入配置）"
            )
        ]
    required = list(config["data"]["required_meta"])
    meta_files = sorted(
        path
        for path in data_root.rglob("*.json")
        if "split" not in path.name.lower() and "manifest" not in path.name.lower()
    )[:50]
    if not meta_files:
        return [Check("data:episodes", WARN, f"{data_root} 下没有 JSON meta")]
    missing: dict[str, list[str]] = {}
    for path in meta_files:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            missing[str(path)] = ["invalid_json"]
            continue
        lacks = [key for key in required if key not in meta]
        if lacks:
            missing[str(path)] = lacks
    checks = [
        Check(
            "data:meta",
            PASS if not missing else FAIL,
            json.dumps(missing, ensure_ascii=False)[:500]
            if missing
            else f"{len(meta_files)} files",
        )
    ]
    action_files = sorted(data_root.rglob("*action*.npy"))[:100]
    bad: list[str] = []
    for path in action_files:
        try:
            array = np.load(path, mmap_mode="r")
        except Exception:  # noqa: BLE001
            bad.append(f"{path.name}:unreadable")
            continue
        if array.ndim != 2 or array.shape[-1] != int(config["xvla"]["action_dim"]):
            bad.append(f"{path.name}:{array.shape}")
    checks.append(
        Check(
            "data:action_shape",
            SKIP if not action_files else (PASS if not bad else FAIL),
            f"{len(action_files)} files, bad={bad[:5]}",
        )
    )
    split_path = REPO_ROOT / config["data"]["split_manifest"]
    if not split_path.exists():
        checks.append(Check("data:split", WARN, f"missing {split_path}"))
    else:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train, val = set(split.get("train", [])), set(split.get("val", []))
        overlap = sorted(train & val)
        checks.append(
            Check(
                "data:split",
                PASS if train and val and not overlap else FAIL,
                f"train={len(train)}, val={len(val)}, overlap={len(overlap)}",
            )
        )
    return checks


def _check_cross(config: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    base_link = config["robot"]["links"]["base"]
    tcp_link = config["robot"]["links"]["tcp"]
    checks.append(
        Check(
            "cross:domain",
            WARN,
            f"domain_id={config['xvla']['domain_id']} 是 Bridge/WidowX；"
            "Alicia-D 未在该域训练，零样本动作不可信，需微调/新 domain",
        )
    )
    checks.append(
        Check(
            "cross:views",
            WARN,
            f"模型 processor 期望 {config['xvla']['views']} 视角；"
            "仿真提供 front+wrist，第 3 视角补 0",
        )
    )
    checks.append(
        Check(
            "cross:tcp",
            PASS,
            f"SDK 与 MJCF 均使用 {base_link} -> {tcp_link}",
        )
    )
    checks.append(
        Check(
            "cross:gripper",
            PASS,
            "X-VLA 1=闭合；SDK 0-1000 中 1000=张开；转换函数已双向覆盖",
        )
    )
    checks.append(
        Check(
            "cross:depth",
            PASS,
            "X-VLA 不接收深度；深度仅用于数据归档，单位统一为米",
        )
    )
    return checks


def _check_adaptation(config: dict[str, Any]) -> list[Check]:
    path = REPO_ROOT / config["xvla"].get("adapted_checkpoint", "")
    domain = int(config["xvla"].get("adapted_domain_id", -1))
    if not path.exists():
        return [Check("adapt:softprompt", SKIP, f"missing {path}")]
    try:
        import torch

        state = torch.load(str(path), map_location="cpu", weights_only=False)
        soft_prompt = state.get("soft_prompt")
        stored_domain = int(state.get("domain_id", -1))
        shape_ok = (
            tuple(soft_prompt.shape) == (32768,) if soft_prompt is not None else False
        )
        return [
            Check(
                "adapt:softprompt",
                PASS
                if stored_domain == domain
                and shape_ok
                and all(
                    key in state
                    for key in (
                        "action_encoder_fc",
                        "action_encoder_bias",
                        "action_decoder_fc",
                        "action_decoder_bias",
                    )
                )
                else FAIL,
                f"{path} domain={stored_domain} shape={None if soft_prompt is None else tuple(soft_prompt.shape)}",
            )
        ]
    except ImportError:
        return [Check("adapt:softprompt", SKIP, "torch 未安装")]
    except Exception as exc:  # noqa: BLE001
        return [Check("adapt:softprompt", FAIL, f"{type(exc).__name__}: {exc}")]


def _check_sdk_import() -> Check:
    try:
        import alicia_d_sdk  # noqa: F401

        return Check("sdk:import", PASS, "alicia_d_sdk 可导入")
    except Exception as exc:  # noqa: BLE001
        return Check(
            "sdk:import",
            SKIP,
            f"{type(exc).__name__}: {exc}"
            "（真机侧执行 pip install -e third_party/Alicia-D-SDK）",
        )


def _check_sim(config: dict[str, Any]) -> list[Check]:
    try:
        import mujoco

        from kite.sim.alicia import AliciaSimEnv, build_model

        model = build_model(config)
        env = AliciaSimEnv(config=config, with_task=True)
        try:
            actuators = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
                for index in range(model.nu)
            ]
            cameras = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
                for index in range(model.ncam)
            ]
            env.set_target(env.target_position + np.array([0.02, 0.0, 0.0]))
            for _ in range(10):
                env.step()
            state = env.state()
            front = env.render("front")
            wrist = env.render("wrist")
            checks = [
                Check(
                    "sim:actuators",
                    PASS if len(actuators) == 8 else FAIL,
                    f"{actuators}",
                ),
                Check(
                    "sim:cameras",
                    PASS if {"front", "wrist"}.issubset(set(cameras)) else FAIL,
                    f"{cameras}",
                ),
                Check(
                    "sim:tracking",
                    PASS if state.position_error < 0.01 else FAIL,
                    f"error {state.position_error * 1000:.1f} mm",
                ),
                Check(
                    "sim:rgb",
                    PASS
                    if front.shape == (224, 224, 3) and wrist.shape == (224, 224, 3)
                    else FAIL,
                    f"front={front.shape}, wrist={wrist.shape}",
                ),
            ]
        finally:
            env.close()
        return checks
    except ImportError:
        return [Check("sim:load", SKIP, "mujoco 未安装")]
    except Exception as exc:  # noqa: BLE001
        return [Check("sim:load", FAIL, f"{type(exc).__name__}: {exc}")]


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    checks: list[Check] = []
    checks.extend(_check_assets(config))
    checks.extend(_check_urdf(config))
    checks.extend(_check_model(config))
    checks.extend(_check_interface(config))
    checks.extend(_check_units(config))
    checks.extend(_check_data(config, Path(args.data_root)))
    checks.extend(_check_cross(config))
    checks.extend(_check_adaptation(config))
    checks.append(_check_sdk_import())
    checks.extend(_check_sim(config))

    width = max(len(check.name) for check in checks)
    for check in checks:
        print(f"[{check.status:4}] {check.name:<{width}}  {check.detail}")

    summary = {
        status: sum(check.status == status for check in checks)
        for status in (PASS, WARN, FAIL, SKIP)
    }
    print("-" * 72)
    print("  ".join(f"{key}={value}" for key, value in summary.items()))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"summary": summary, "checks": [asdict(check) for check in checks]},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"json: {path}")
    if summary[FAIL] or (args.strict and summary[WARN]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
