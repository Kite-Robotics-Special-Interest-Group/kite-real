"""MuJoCo scene and absolute-EEF controller for Alicia-D v5.6."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from kite.robot.alicia import (
    REPO_ROOT,
    clamp_action10,
    load_config,
    matrix_to_quat_xyzw,
    matrix_to_rot6d,
    proprio_from_pose,
    quat_xyzw_to_matrix,
    rot6d_to_matrix,
    rotation_vector,
    yaw_matrix,
)

if sys.platform == "win32":
    os.environ.setdefault("MUJOCO_GL", "wgl")

import mujoco


def _lookat_quat_wxyz(position: np.ndarray, lookat: np.ndarray) -> list[float]:
    """MuJoCo camera quaternion (w, x, y, z) looking from position to target."""
    forward = lookat - position
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, forward)
    rotation = np.stack([right, up, -forward], axis=1)
    x, y, z, w = matrix_to_quat_xyzw(rotation)
    return [float(w), float(x), float(y), float(z)]


def build_model(
    config: Mapping[str, Any] | None = None,
    with_task: bool = True,
    grasp_pads: bool = False,
) -> mujoco.MjModel:
    """Load the vendor MJCF and add cameras, actuators and a task scene."""
    if config is None:
        config = load_config()
    robot = config["robot"]
    mjcf_path = REPO_ROOT / robot["assets"]["mjcf"]
    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    spec.option.timestep = float(robot["control"]["sim_timestep_s"])

    for name, limits in zip(
        robot["joints"]["arm"], robot["joints"]["limits_rad"], strict=False
    ):
        actuator = spec.add_actuator()
        actuator.name = f"pos_{name}"
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.target = name
        actuator.set_to_position(kp=150.0, kv=25.0)
        actuator.ctrlrange = limits
        actuator.ctrllimited = True
        actuator.forcelimited = True
        actuator.forcerange = [-20.0, 20.0]

    travel = float(robot["gripper"]["travel_m"])
    for name, limits in zip(
        robot["joints"]["gripper"], ([0.0, travel], [-travel, 0.0]), strict=False
    ):
        actuator = spec.add_actuator()
        actuator.name = f"pos_{name}"
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.target = name
        actuator.set_to_position(kp=100.0, kv=20.0)
        actuator.ctrlrange = limits
        actuator.ctrllimited = True
        actuator.forcelimited = True
        actuator.forcerange = [-20.0, 20.0]

    front = robot["cameras"]["front"]
    front_position = np.asarray(front["pos_m"], dtype=np.float64)
    camera = spec.worldbody.add_camera()
    camera.name = "front"
    camera.pos = front_position.tolist()
    camera.quat = _lookat_quat_wxyz(front_position, np.asarray(front["lookat_m"]))
    camera.fovy = float(front["fovy_deg"])

    wrist = robot["cameras"]["wrist"]
    wrist_body = spec.body(wrist["parent"])
    wrist_camera = wrist_body.add_camera()
    wrist_camera.name = "wrist"
    wrist_camera.pos = list(wrist["pos_m"])
    wrist_camera.quat = [float(value) for value in wrist["quat_wxyz"]]
    wrist_camera.fovy = float(wrist["fovy_deg"])

    target = spec.worldbody.add_body()
    target.name = "eef_target"
    target.mocap = True
    target.pos = [0.26, 0.0, 0.61]
    target_geom = target.add_geom()
    target_geom.name = "eef_target_geom"
    target_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    target_geom.size = [0.012, 0.0, 0.0]
    target_geom.rgba = [1.0, 0.78, 0.1, 0.65]
    target_geom.contype = 0
    target_geom.conaffinity = 0

    if with_task:
        table = robot["scene"]["table"]
        table_geom = spec.worldbody.add_geom()
        table_geom.name = "table"
        table_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        table_geom.size = list(table["size_m"])
        table_geom.pos = list(table["pos_m"])
        table_geom.rgba = [0.55, 0.40, 0.28, 1.0]
        if "pedestal" in robot["scene"]:
            pedestal = robot["scene"]["pedestal"]
            pedestal_geom = spec.worldbody.add_geom()
            pedestal_geom.name = "pedestal"
            pedestal_geom.type = mujoco.mjtGeom.mjGEOM_BOX
            pedestal_geom.size = list(pedestal["size_m"])
            pedestal_geom.pos = list(pedestal["pos_m"])
            pedestal_geom.rgba = [0.35, 0.35, 0.38, 1.0]
        cube = robot["scene"]["cube"]
        cube_body = spec.worldbody.add_body()
        cube_body.name = "cube"
        cube_body.pos = list(cube["pos_m"])
        cube_body.add_freejoint()
        cube_geom = cube_body.add_geom()
        cube_geom.name = "cube_geom"
        cube_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        cube_geom.size = list(cube["size_m"])
        cube_geom.rgba = [0.85, 0.18, 0.14, 1.0]
        cube_geom.mass = 0.05

    if grasp_pads:
        for name in ("left_gripper", "right_gripper"):
            finger = spec.body(name)
            for geom in finger.geoms:
                if geom.contype:
                    geom.contype = 0
                    geom.conaffinity = 0
            pad = finger.add_geom()
            pad.name = f"{name}_pad"
            pad.type = mujoco.mjtGeom.mjGEOM_BOX
            pad.size = [0.006, 0.020, 0.020]
            pad.pos = [0.0, 0.011, 0.0]
            pad.rgba = [0.10, 0.10, 0.10, 1.0]
            pad.friction = [2.0, 1.0, 0.01]
            pad.condim = 4

    model = spec.compile()
    robot_dofs = [
        model.jnt_dofadr[model.joint(name).id]
        for name in robot["joints"]["arm"] + robot["joints"]["gripper"]
    ]
    model.dof_armature[robot_dofs] = 0.05
    model.dof_damping[robot_dofs] = 0.5
    return model


def _home_qpos(config: Mapping[str, Any]) -> np.ndarray:
    joints = np.asarray(config["robot"]["scene"]["home_joint_rad"], dtype=np.float64)
    grip = float(config["robot"]["scene"]["home_gripper_norm"])
    travel = float(config["robot"]["gripper"]["travel_m"])
    return np.concatenate([joints, [grip * travel, -grip * travel]])


@dataclass
class SimState:
    """Current simulated robot state."""

    joint_positions: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    gripper_norm: float
    target_position: np.ndarray
    target_rotation: np.ndarray
    position_error: float
    rotation_error: float


class AliciaSimEnv:
    """Absolute end-effector MuJoCo environment with a supervisor target."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        with_task: bool = True,
        seed: int = 0,
        grasp_pads: bool = False,
    ) -> None:
        self.config = load_config() if config is None else config
        self.model = build_model(
            self.config, with_task=with_task, grasp_pads=grasp_pads
        )
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        robot = self.config["robot"]
        self.arm_joint_names = list(robot["joints"]["arm"])
        self.gripper_joint_names = list(robot["joints"]["gripper"])
        self.joint_names = self.arm_joint_names + self.gripper_joint_names
        self.qpos_adr = np.array(
            [
                self.model.jnt_qposadr[self.model.joint(name).id]
                for name in self.joint_names
            ]
        )
        self.arm_dof = np.array(
            [
                self.model.jnt_dofadr[self.model.joint(name).id]
                for name in self.arm_joint_names
            ]
        )
        self.ctrl_ids = np.array(
            [self.model.actuator(f"pos_{name}").id for name in self.joint_names]
        )
        self.site_id = self.model.site(robot["links"]["tcp"] + "_site").id
        self.target_mocap_id = self.model.body("eef_target").mocapid[0]
        self.camera_ids = {
            name: self.model.camera(name).id for name in ("front", "wrist")
        }
        self.substeps = int(robot["control"]["sim_substeps"])
        self.control_period = 1.0 / float(robot["control"]["rate_hz"])
        self.travel = float(robot["gripper"]["travel_m"])
        self._renderers: dict[str, mujoco.Renderer] = {}
        self.home_qpos = _home_qpos(self.config)
        self.cube_qpos_adr: int | None = None
        self.cube_dof_adr: int | None = None
        self._attached_offset: np.ndarray | None = None
        if with_task:
            cube_body = self.model.body("cube")
            cube_joint = self.model.body_jntadr[cube_body.id]
            self.cube_qpos_adr = int(self.model.jnt_qposadr[cube_joint])
            self.cube_dof_adr = int(self.model.jnt_dofadr[cube_joint])
        self.reset()

    def reset(self, cube_position: np.ndarray | None = None) -> SimState:
        """Reset joints, gripper and free cube to their home configuration."""
        mujoco.mj_resetData(self.model, self.data)
        self._attached_offset = None
        self.data.qpos[self.qpos_adr] = self.home_qpos
        if cube_position is not None and self.cube_qpos_adr is not None:
            self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 7] = [
                *np.asarray(cube_position, dtype=np.float64),
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        self.data.ctrl[self.ctrl_ids] = self.home_qpos
        mujoco.mj_forward(self.model, self.data)
        position, rotation = self.eef_pose()
        self._set_target(position, rotation)
        return self.state()

    def eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Current TCP position and rotation matrix in the base frame."""
        position = np.array(self.data.site_xpos[self.site_id], dtype=np.float64)
        rotation = np.array(
            self.data.site_xmat[self.site_id], dtype=np.float64
        ).reshape(3, 3)
        return position, rotation

    def joint_positions(self) -> np.ndarray:
        return np.array([self.data.qpos[address] for address in self.qpos_adr[:6]])

    def gripper_norm(self) -> float:
        left = float(self.data.qpos[self.qpos_adr[6]])
        return float(np.clip(left / self.travel, 0.0, 1.0))

    def _set_target(self, position: np.ndarray, rotation: np.ndarray) -> None:
        self.target_position = np.asarray(position, dtype=np.float64).copy()
        self.target_rotation = np.asarray(rotation, dtype=np.float64).copy()
        quaternion = matrix_to_quat_xyzw(self.target_rotation)
        self.data.mocap_pos[self.target_mocap_id] = self.target_position
        self.data.mocap_quat[self.target_mocap_id] = [
            quaternion[3],
            quaternion[0],
            quaternion[1],
            quaternion[2],
        ]

    def set_target(
        self,
        position: np.ndarray | list[float] | None = None,
        rotation: np.ndarray | None = None,
    ) -> None:
        """Set the supervisor target without moving the arm immediately."""
        current_position, current_rotation = self.eef_pose()
        self._set_target(
            current_position if position is None else np.asarray(position),
            current_rotation if rotation is None else rotation,
        )

    def nudge(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        dyaw: float = 0.0,
    ) -> None:
        """Move the supervisor target in the base frame."""
        self.target_position = self.target_position + np.array([dx, dy, dz])
        if dyaw:
            self.target_rotation = yaw_matrix(dyaw) @ self.target_rotation

    def _gripper_qpos(self, value: float) -> np.ndarray:
        grip = float(np.clip(value, 0.0, 1.0))
        return np.array([grip * self.travel, -grip * self.travel])

    def set_gripper_norm(self, value: float) -> None:
        """Teleport the fingers to a normalized value (1 closed, 0 open)."""
        finger_qpos = self._gripper_qpos(value)
        self.data.qpos[self.qpos_adr[6:]] = finger_qpos
        self.data.ctrl[self.ctrl_ids[6:]] = finger_qpos

    def command_gripper_norm(self, value: float) -> None:
        """Command the fingers smoothly through the position actuators."""
        self.data.ctrl[self.ctrl_ids[6:]] = self._gripper_qpos(value)

    def _solve_ik(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        iterations: int = 40,
        damping: float = 0.02,
    ) -> tuple[float, float]:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        lower = self.model.jnt_range[
            [self.model.joint(name).id for name in self.arm_joint_names], 0
        ]
        upper = self.model.jnt_range[
            [self.model.joint(name).id for name in self.arm_joint_names], 1
        ]
        position_error = 0.0
        rotation_error = 0.0
        for _ in range(iterations):
            mujoco.mj_forward(self.model, self.data)
            position, rotation = self.eef_pose()
            position_delta = target_position - position
            rotation_delta = rotation_vector(target_rotation @ rotation.T)
            position_error = float(np.linalg.norm(position_delta))
            rotation_error = float(np.linalg.norm(rotation_delta))
            if position_error < 2e-4 and rotation_error < 2e-3:
                break
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            jacobian = np.vstack([jacp[:, self.arm_dof], jacr[:, self.arm_dof]])
            error = np.concatenate([position_delta, rotation_delta])
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), error
            )
            arm = np.array(self.data.qpos[self.qpos_adr[:6]])
            self.data.qpos[self.qpos_adr[:6]] = np.clip(
                arm + np.clip(delta, -0.15, 0.15), lower, upper
            )
        return position_error, rotation_error

    def _apply_control(self) -> None:
        arm = np.array(self.data.qpos[self.qpos_adr[:6]])
        self.data.ctrl[self.ctrl_ids[:6]] = arm

    def cube_position(self) -> np.ndarray | None:
        """Current cube position when the task scene is present."""
        if self.cube_qpos_adr is None:
            return None
        return np.array(self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 3])

    def attach_cube(self) -> np.ndarray | None:
        """Attach the cube to the current TCP for scripted demonstrations."""
        cube = self.cube_position()
        if cube is None:
            return None
        self._attached_offset = cube - self.eef_pose()[0]
        return self._attached_offset.copy()

    def detach_cube(self) -> None:
        """Release a scripted demonstration attachment."""
        self._attached_offset = None

    @property
    def cube_attached(self) -> bool:
        """Whether the demonstration cube currently follows the TCP."""
        return self._attached_offset is not None

    def _follow_attached_cube(self) -> None:
        if self._attached_offset is None or self.cube_qpos_adr is None:
            return
        cube = self.eef_pose()[0] + self._attached_offset
        self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 3] = cube
        if self.cube_dof_adr is not None:
            self.data.qvel[self.cube_dof_adr : self.cube_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, substeps: int | None = None) -> None:
        """Solve IK to the current target, apply it and advance physics."""
        self._solve_ik(self.target_position, self.target_rotation)
        self._apply_control()
        for _ in range(self.substeps if substeps is None else substeps):
            mujoco.mj_step(self.model, self.data)
        self._follow_attached_cube()

    def execute_action(
        self,
        action: np.ndarray | list[float],
        clamp: bool = True,
        substeps: int | None = None,
    ) -> list[str]:
        """Execute one absolute X-VLA action ``[xyz, rot6d, grip]``."""
        values = np.asarray(action, dtype=np.float64).reshape(-1)[:10]
        current_position, current_rotation = self.eef_pose()
        interventions: list[str] = []
        if clamp:
            result = clamp_action10(
                values,
                previous_position=current_position,
                previous_rotation=current_rotation,
                config=self.config,
            )
            values = result.action
            interventions = result.interventions
        self._set_target(values[:3], rot6d_to_matrix(values[3:9]))
        self.command_gripper_norm(float(values[9]))
        self.step(substeps=substeps)
        return interventions

    def execute_chunk(
        self,
        actions: np.ndarray,
        clamp: bool = True,
        substeps: int | None = None,
    ) -> list[list[str]]:
        """Execute a ``[T, 10]`` absolute action chunk."""
        chunk = np.asarray(actions, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] < 10:
            raise ValueError(f"actions must be [T, 10], got {chunk.shape}")
        return [
            self.execute_action(action, clamp=clamp, substeps=substeps)
            for action in chunk[:, :10]
        ]

    def state(self) -> SimState:
        position, rotation = self.eef_pose()
        return SimState(
            joint_positions=self.joint_positions(),
            position=position,
            rotation=rotation,
            gripper_norm=self.gripper_norm(),
            target_position=self.target_position.copy(),
            target_rotation=self.target_rotation.copy(),
            position_error=float(np.linalg.norm(self.target_position - position)),
            rotation_error=float(
                np.linalg.norm(rotation_vector(self.target_rotation @ rotation.T))
            ),
        )

    def action10(self) -> np.ndarray:
        position, rotation = self.eef_pose()
        return np.concatenate(
            [position, matrix_to_rot6d(rotation), [self.gripper_norm()]]
        )

    def proprio(self) -> np.ndarray:
        position, rotation = self.eef_pose()
        return proprio_from_pose(position, rotation)

    def render(self, camera: str = "front") -> np.ndarray:
        """Render an RGB observation from ``front`` or ``wrist``."""
        if camera not in self.camera_ids:
            raise ValueError(
                f"unknown camera {camera!r}; choose {sorted(self.camera_ids)}"
            )
        renderer = self._renderers.get(camera)
        if renderer is None:
            resolution = self.config["robot"]["cameras"][camera]["resolution"]
            renderer = mujoco.Renderer(
                self.model, height=int(resolution[0]), width=int(resolution[1])
            )
            self._renderers[camera] = renderer
        renderer.update_scene(self.data, camera=self.camera_ids[camera])
        return renderer.render().copy()

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def observation(self, render: bool = False) -> dict[str, Any]:
        """Return JSON-friendly observation fields plus optional RGB images."""
        position, rotation = self.eef_pose()
        observation: dict[str, Any] = {
            "joint_positions": self.joint_positions(),
            "position": position,
            "rotation": rotation,
            "gripper_norm": self.gripper_norm(),
            "proprio": self.proprio(),
            "action10": np.concatenate(
                [position, matrix_to_rot6d(rotation), [self.gripper_norm()]]
            ),
        }
        if render:
            observation["images"] = {
                "front": self.render("front"),
                "wrist": self.render("wrist"),
            }
        return observation


def make_renderer(
    model: mujoco.MjModel, height: int = 224, width: int = 224
) -> mujoco.Renderer:
    """Create a renderer with the requested RGB resolution."""
    return mujoco.Renderer(model, height=height, width=width)


def cube_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
    """Return the free cube pose when the task scene is present."""
    try:
        joint = model.joint("cube_joint")
    except KeyError:
        return None
    address = model.jnt_qposadr[joint.id]
    return np.array(data.qpos[address : address + 7])


def quat_to_rotation_wxyz(quaternion: list[float]) -> np.ndarray:
    """Convenience for tools that store MuJoCo (w, x, y, z) quaternions."""
    w, x, y, z = quaternion
    return quat_xyzw_to_matrix([x, y, z, w])


def scene_path(config: Mapping[str, Any] | None = None) -> Path:
    """Absolute path to the vendor MJCF."""
    if config is None:
        config = load_config()
    return REPO_ROOT / config["robot"]["assets"]["mjcf"]
