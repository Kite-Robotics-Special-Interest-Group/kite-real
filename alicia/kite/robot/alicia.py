"""Alicia-D v5.6 adapter for the native X-VLA action interface.

The SDK speaks joint radians, gripper ticks in ``[0, 1000]`` and Cartesian
poses as ``[x, y, z, qx, qy, qz, qw]``.  X-VLA speaks absolute end-effector
``[xyz_m, rot6d, gripper]`` with a normalized gripper where ``1`` is closed.
This module is the single place where those conventions are converted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "robot" / "alicia_d_v5_6.yaml"
EPS = 1e-9


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the Alicia-D interface configuration."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"invalid robot config: {config_path}")
    return data


def _as_vector(value: np.ndarray | list[float], size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size:
        raise ValueError(f"{name} must have {size} elements, got {vector.size}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector


def rot6d_to_matrix(rot6d: np.ndarray | list[float]) -> np.ndarray:
    """Convert interleaved 6D rotation ``[b1x,b2x,b1y,b2y,b1z,b2z]``."""
    values = _as_vector(rot6d, 6, "rot6d")
    first = values[0:6:2]
    second = values[1:6:2]
    norm_first = np.linalg.norm(first)
    if norm_first < EPS:
        raise ValueError("rot6d first basis vector has zero norm")
    basis_x = first / norm_first
    basis_y = second - np.dot(basis_x, second) * basis_x
    norm_second = np.linalg.norm(basis_y)
    if norm_second < EPS:
        raise ValueError("rot6d basis vectors are degenerate")
    basis_y = basis_y / norm_second
    basis_z = np.cross(basis_x, basis_y)
    return np.stack([basis_x, basis_y, basis_z], axis=1)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to interleaved first two columns."""
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {rotation.shape}")
    first, second = rotation[:, 0], rotation[:, 1]
    return np.array(
        [first[0], second[0], first[1], second[1], first[2], second[2]]
    )


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a normalized ``(x, y, z, w)`` quaternion."""
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {rotation.shape}")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < EPS:
        raise ValueError("quaternion has zero norm")
    return quaternion / norm


def quat_xyzw_to_matrix(quaternion: np.ndarray | list[float]) -> np.ndarray:
    """Convert a normalized or unnormalized ``(x, y, z, w)`` quaternion."""
    x, y, z, w = _as_vector(quaternion, 4, "quaternion")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < EPS:
        raise ValueError("quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_matrix(angle_rad: float) -> np.ndarray:
    """Rotation about the base ``+Z`` axis."""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_vector(matrix: np.ndarray) -> np.ndarray:
    """Log map of ``SO(3)`` as an axis-angle vector."""
    rotation = np.asarray(matrix, dtype=np.float64)
    cos_angle = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cos_angle)
    if angle < 1e-8:
        return np.zeros(3)
    if abs(math.pi - angle) < 1e-6:
        column = np.argmax(np.diag(rotation))
        axis = np.zeros(3)
        axis[column] = math.sqrt(max((rotation[column, column] + 1.0) * 0.5, 0.0))
        for index in range(3):
            if index != column:
                axis[index] = rotation[index, column] / (2.0 * axis[column] + EPS)
        return axis / (np.linalg.norm(axis) + EPS) * angle
    factor = angle / (2.0 * math.sin(angle))
    return factor * np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )


def matrix_from_rotation_vector(vector: np.ndarray | list[float]) -> np.ndarray:
    """Exponential map from an axis-angle vector to ``SO(3)``."""
    omega = _as_vector(vector, 3, "rotation vector")
    angle = float(np.linalg.norm(omega))
    if angle < 1e-9:
        return np.eye(3)
    axis = omega / angle
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


@dataclass(frozen=True)
class GripperMapper:
    """Map X-VLA normalized gripper values to Alicia-D SDK ticks."""

    sdk_min: int = 0
    sdk_max: int = 1000
    xvla_closed: float = 1.0
    xvla_open: float = 0.0

    def norm_to_ticks(self, value: float) -> int:
        clipped = float(np.clip(value, 0.0, 1.0))
        open_fraction = (self.xvla_closed - clipped) / (
            self.xvla_closed - self.xvla_open
        )
        ticks = self.sdk_min + open_fraction * (self.sdk_max - self.sdk_min)
        return round(ticks)

    def ticks_to_norm(self, ticks: float) -> float:
        clipped = float(np.clip(ticks, self.sdk_min, self.sdk_max))
        open_fraction = (clipped - self.sdk_min) / (self.sdk_max - self.sdk_min)
        return self.xvla_closed - open_fraction * (self.xvla_closed - self.xvla_open)


def gripper_mapper(config: Mapping[str, Any] | None = None) -> GripperMapper:
    """Build the gripper mapper from the robot config."""
    if config is None:
        config = load_config()
    gripper = config["robot"]["gripper"]
    return GripperMapper(
        sdk_min=int(gripper["sdk"]["range"][0]),
        sdk_max=int(gripper["sdk"]["range"][1]),
        xvla_closed=float(gripper["xvla"]["closed"]),
        xvla_open=float(gripper["xvla"]["open"]),
    )


def action10_to_sdk_pose(
    action: np.ndarray | list[float],
    config: Mapping[str, Any] | None = None,
) -> tuple[list[float], int]:
    """Convert one X-VLA ``[xyz, rot6d, grip]`` action to an SDK pose/ticks.

    Returns ``([x, y, z, qx, qy, qz, qw], gripper_ticks)``.
    """
    values = _as_vector(action, 10, "action")
    rotation = rot6d_to_matrix(values[3:9])
    quaternion = matrix_to_quat_xyzw(rotation)
    grip = gripper_mapper(config).norm_to_ticks(float(values[9]))
    pose = [*values[:3].tolist(), *quaternion.tolist()]
    return pose, grip


def sdk_pose_to_action10(
    position: np.ndarray | list[float],
    rotation: np.ndarray,
    gripper_ticks: float,
    config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Convert SDK pose/ticks to one X-VLA 10D action."""
    xyz = _as_vector(position, 3, "position")
    rot6d = matrix_to_rot6d(rotation)
    grip = gripper_mapper(config).ticks_to_norm(gripper_ticks)
    return np.concatenate([xyz, rot6d, [grip]])


def proprio_from_pose(
    position: np.ndarray | list[float], rotation: np.ndarray
) -> np.ndarray:
    """Build the 20D single-arm proprio vector used by X-VLA.

    Layout is ``[pos(3), rot6d(6), gripper(1)]`` for the active arm plus ten
    zeros for the unused second arm.  The gripper channel is zeroed because
    ``EE6DActionSpace.preprocess`` masks it during inference.
    """
    xyz = _as_vector(position, 3, "position")
    active = np.concatenate([xyz, matrix_to_rot6d(rotation), [0.0]])
    return np.concatenate([active, np.zeros(10, dtype=np.float64)])


@dataclass(frozen=True)
class Workspace:
    """Axis-aligned Cartesian workspace in the robot base frame."""

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> Workspace:
        if config is None:
            config = load_config()
        bounds = config["robot"]["workspace_m"]
        return cls(
            x=tuple(float(value) for value in bounds["x"]),
            y=tuple(float(value) for value in bounds["y"]),
            z=tuple(float(value) for value in bounds["z"]),
        )

    def clamp(self, position: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Clamp a position and report which axes were clipped."""
        clipped = np.asarray(position, dtype=np.float64).copy()
        interventions: list[str] = []
        for index, axis in enumerate(("x", "y", "z")):
            lower, upper = getattr(self, axis)
            value = float(clipped[index])
            if value < lower or value > upper:
                clipped[index] = float(np.clip(value, lower, upper))
                interventions.append(f"{axis}_workspace")
        return clipped, interventions


@dataclass
class ClampResult:
    """Result of a safety clamp on a single absolute action."""

    action: np.ndarray
    interventions: list[str] = field(default_factory=list)


def clamp_action10(
    action: np.ndarray | list[float],
    previous_position: np.ndarray | list[float] | None = None,
    previous_rotation: np.ndarray | None = None,
    config: Mapping[str, Any] | None = None,
) -> ClampResult:
    """Clamp absolute EEF actions to workspace, step, rotation and gripper limits."""
    if config is None:
        config = load_config()
    values = _as_vector(action, 10, "action").copy()
    interventions: list[str] = []

    position, clipped_axes = Workspace.from_config(config).clamp(values[:3])
    if clipped_axes:
        values[:3] = position
        interventions.extend(clipped_axes)

    max_step = float(config["robot"]["control"]["max_step_m"])
    if previous_position is not None and max_step > 0.0:
        previous = _as_vector(previous_position, 3, "previous position")
        delta = values[:3] - previous
        distance = float(np.linalg.norm(delta))
        if distance > max_step:
            values[:3] = previous + delta / distance * max_step
            interventions.append("position_step")

    rotation = rot6d_to_matrix(values[3:9])
    max_rotation = float(config["robot"]["control"]["max_yaw_rad"])
    if previous_rotation is not None and max_rotation > 0.0:
        relative = rotation_vector(np.asarray(previous_rotation).T @ rotation)
        angle = float(np.linalg.norm(relative))
        if angle > max_rotation:
            rotation = np.asarray(previous_rotation) @ matrix_from_rotation_vector(
                relative / angle * max_rotation
            )
            interventions.append("rotation_step")
    values[3:9] = matrix_to_rot6d(rotation)

    grip = float(values[9])
    clipped_grip = float(np.clip(grip, 0.0, 1.0))
    if clipped_grip != grip:
        values[9] = clipped_grip
        interventions.append("gripper_range")

    return ClampResult(action=values, interventions=interventions)


@dataclass
class AliciaObservation:
    """One synchronized real-robot observation."""

    joint_positions: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    gripper_ticks: float
    timestamp: float
    mapper: GripperMapper = field(
        default_factory=gripper_mapper, repr=False, compare=False
    )

    @property
    def gripper_norm(self) -> float:
        return self.mapper.ticks_to_norm(self.gripper_ticks)

    @property
    def proprio(self) -> np.ndarray:
        return proprio_from_pose(self.position, self.rotation)

    @property
    def action10(self) -> np.ndarray:
        return np.concatenate(
            [
                self.position,
                matrix_to_rot6d(self.rotation),
                [self.gripper_norm],
            ]
        )


class AliciaSdkClient:
    """Thin safety-aware wrapper around :mod:`alicia_d_sdk` (lazy import)."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        port: str | None = None,
        auto_connect: bool = False,
        backend: str = "numpy",
    ) -> None:
        self.config = load_config(config_path)
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.port = (
            port
            if port is not None
            else str(self.config["robot"]["sdk"]["serial_port"])
        )
        self.auto_connect = auto_connect
        self.backend = backend
        self.mapper = gripper_mapper(self.config)
        self._robot: Any | None = None
        if auto_connect:
            self.connect()

    @property
    def connected(self) -> bool:
        return self._robot is not None and bool(self._robot.is_connected())

    @property
    def robot(self) -> Any:
        """Underlying vendor robot instance (after :meth:`connect`)."""
        return self._require_robot()

    def version(self) -> dict[str, Any] | None:
        """Firmware/hardware/serial version dictionary."""
        return self.robot.get_robot_state("version")

    def connect(self) -> None:
        """Import and connect the vendor SDK."""
        if self._robot is not None:
            return
        try:
            import alicia_d_sdk
        except ImportError as exc:  # pragma: no cover - depends on vendor install
            raise RuntimeError(
                "alicia_d_sdk is not installed; run "
                "`python -m pip install -e third_party/Alicia-D-SDK`"
            ) from exc
        sdk = self.config["robot"]["sdk"]
        urdf = REPO_ROOT / self.config["robot"]["assets"]["urdf"]
        self._robot = alicia_d_sdk.create_robot(
            port=self.port,
            version=sdk["version"],
            variant=sdk["variant"],
            model_format="urdf",
            model_path=str(urdf),
            base_link=self.config["robot"]["links"]["base"],
            end_link=self.config["robot"]["links"]["tcp"],
            backend=self.backend,
            auto_connect=False,
        )
        if not self._robot.connect():
            self._robot = None
            raise RuntimeError(
                f"failed to connect Alicia-D on port {self.port!r}; "
                "check USB/serial and the vendor SDK"
            )

    def close(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None

    def _require_robot(self) -> Any:
        if self._robot is None:
            raise RuntimeError("call connect() first")
        return self._robot

    def read(self) -> AliciaObservation:
        """Read joints, TCP pose and gripper ticks in one observation."""
        robot = self._require_robot()
        state = robot.get_robot_state("joint_gripper")
        pose = robot.get_pose()
        if state is None or pose is None:
            raise RuntimeError("failed to read robot state or pose")
        return AliciaObservation(
            joint_positions=np.asarray(state.angles, dtype=np.float64),
            position=np.asarray(pose["position"], dtype=np.float64),
            rotation=np.asarray(pose["rotation"], dtype=np.float64),
            gripper_ticks=float(state.gripper),
            timestamp=float(state.timestamp),
            mapper=self.mapper,
        )

    def send_action10(
        self,
        action: np.ndarray | list[float],
        speed_deg_s: float = 10.0,
        execute: bool = True,
        previous: AliciaObservation | None = None,
    ) -> ClampResult:
        """Clamp and execute one absolute X-VLA action on the real robot."""
        if previous is None and execute:
            previous = self.read()
        result = clamp_action10(
            action,
            previous_position=None if previous is None else previous.position,
            previous_rotation=None if previous is None else previous.rotation,
            config=self.config,
        )
        pose, gripper_ticks = action10_to_sdk_pose(result.action, self.config)
        if execute:
            robot = self._require_robot()
            robot.set_pose(pose, speed_deg_s=speed_deg_s, execute=True)
            robot.set_robot_state(
                gripper_value=gripper_ticks,
                wait_for_completion=False,
                gripper_speed_deg_s=100.0,
            )
        return result

    def hold(self) -> None:
        """Hold the current joint configuration (supervisor stop)."""
        robot = self._require_robot()
        joints = robot.get_robot_state("joint")
        if joints is None:
            raise RuntimeError("failed to read joint angles")
        robot.set_robot_state(target_joints=list(joints), joint_format="rad")

    def home(self, speed_deg_s: float = 10.0) -> None:
        """Move to the vendor zero pose with the gripper open."""
        self._require_robot().set_home(speed_deg_s=speed_deg_s)
