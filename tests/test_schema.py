from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kite_real import schema
from kite_real.record import EpisodeWriter


def make_episode(root: Path, steps: int = 3, success: bool = True, **meta) -> Path:
    writer = EpisodeWriter(root, "pick the cube", 10.0, "binary", "ee6d")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for _ in range(steps):
        writer.add(np.zeros(10), np.zeros(7), frame)
    return writer.close(success=success, **meta)


def test_roundtrip_and_validate(tmp_path: Path) -> None:
    root = make_episode(tmp_path / "0000")
    assert schema.validate(root) == []
    ep = schema.load_episode(root)
    assert ep.actions.shape == (3, 10)
    assert ep.proprio.shape == (3, 7)
    assert len(ep.frames) == 3
    assert ep.meta["instruction"] == "pick the cube"


def test_failed_episode_requires_failure_type(tmp_path: Path) -> None:
    root = make_episode(tmp_path / "0001", success=False)
    problems = schema.validate(root)
    assert any("failure_type" in p for p in problems)


def test_failure_type_ok(tmp_path: Path) -> None:
    root = make_episode(tmp_path / "0002", success=False, failure_type="slip")
    assert schema.validate(root) == []


def test_wrong_action_shape_rejected(tmp_path: Path) -> None:
    writer = EpisodeWriter(tmp_path / "0003", "x", 10.0, "binary", "ee6d")
    with pytest.raises(ValueError):
        writer.add(np.zeros(9), np.zeros(7), np.zeros((8, 8, 3), dtype=np.uint8))


def test_nan_detected(tmp_path: Path) -> None:
    root = make_episode(tmp_path / "0004")
    actions = np.load(root / "action.npy")
    actions[0, 0] = np.nan
    np.save(root / "action.npy", actions)
    assert any("NaN" in p for p in schema.validate(root))


def test_missing_meta_field(tmp_path: Path) -> None:
    root = make_episode(tmp_path / "0005")
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    meta.pop("control_hz")
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert any("missing meta" in p for p in schema.validate(root))

