from pathlib import Path

import numpy as np
import pytest
from kite.robot.alicia import (
    DEFAULT_CONFIG_PATH,
    action10_to_sdk_pose,
    clamp_action10,
    gripper_mapper,
    load_config,
    matrix_to_quat_xyzw,
    matrix_to_rot6d,
    proprio_from_pose,
    quat_xyzw_to_matrix,
    rot6d_to_matrix,
    sdk_pose_to_action10,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads():
    config = load_config()
    assert config["robot"]["name"] == "Alicia_D_v5_6"
    assert DEFAULT_CONFIG_PATH.exists()


def test_rot6d_identity_and_roundtrip():
    assert np.allclose(rot6d_to_matrix([1, 0, 0, 1, 0, 0]), np.eye(3))
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    assert np.allclose(rot6d_to_matrix(matrix_to_rot6d(rotation)), rotation)


def test_quaternion_roundtrip():
    rotation = np.array(
        [[-0.68615984, 0.0, -0.72744977], [0.0, -1.0, 0.0], [-0.72744977, 0.0, 0.68615984]]
    )
    quaternion = matrix_to_quat_xyzw(rotation)
    assert abs(np.linalg.norm(quaternion) - 1.0) < 1e-12
    assert np.allclose(quat_xyzw_to_matrix(quaternion), rotation)


def test_gripper_polarity():
    config = load_config()
    mapper = gripper_mapper(config)
    assert mapper.norm_to_ticks(0.0) == 1000
    assert mapper.norm_to_ticks(1.0) == 0
    assert mapper.ticks_to_norm(1000) == pytest.approx(0.0)
    assert mapper.ticks_to_norm(0) == pytest.approx(1.0)


def test_action_roundtrip_with_sdk_pose():
    config = load_config()
    action = np.array([0.21, -0.03, 0.32, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.4])
    pose, ticks = action10_to_sdk_pose(action, config)
    recovered = sdk_pose_to_action10(
        pose[:3], rot6d_to_matrix(action[3:9]), ticks, config
    )
    assert np.allclose(recovered, action, atol=1e-9)


def test_proprio_layout():
    proprio = proprio_from_pose([0.2, 0.0, 0.3], np.eye(3))
    assert proprio.shape == (20,)
    assert np.allclose(proprio[:10], [0.2, 0.0, 0.3, 1, 0, 0, 1, 0, 0, 0])
    assert np.allclose(proprio[10:], 0.0)


def test_safety_clamp_reports_interventions():
    config = load_config()
    result = clamp_action10(
        [2.0, 2.0, 2.0, 1, 0, 0, 1, 0, 0, 2.0],
        previous_position=[0.2, 0.0, 0.3],
        config=config,
    )
    assert result.action[9] == 1.0
    assert np.all(np.abs(result.action[:3]) <= 0.65)
    assert "position_step" in result.interventions
    assert "gripper_range" in result.interventions


def test_simulator_tracks_absolute_action():
    pytest.importorskip("mujoco")
    if not (REPO_ROOT / "assets/synriard/mjcf/Alicia_D_v5_6").exists():
        pytest.skip("Alicia-D assets are not downloaded")
    from kite.sim.alicia import AliciaSimEnv

    env = AliciaSimEnv()
    try:
        action = env.action10()
        if action[9] > 0.5:
            action[9] = 0.0
        env.execute_action(action)
        for _ in range(5):
            env.step()
        state = env.state()
        assert state.position_error < 0.005
    finally:
        env.close()


def test_simulator_gripper_polarity():
    pytest.importorskip("mujoco")
    if not (REPO_ROOT / "assets/synriard/mjcf/Alicia_D_v5_6").exists():
        pytest.skip("Alicia-D assets are not downloaded")
    from kite.sim.alicia import AliciaSimEnv

    env = AliciaSimEnv()
    try:
        env.set_gripper_norm(0.0)
        assert env.gripper_norm() == pytest.approx(0.0)
        env.set_gripper_norm(1.0)
        assert env.gripper_norm() == pytest.approx(1.0)
    finally:
        env.close()
