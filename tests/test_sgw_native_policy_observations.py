import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1.adapters import AdapterError
from experiments.workshops.spatial_grounding_v1.policy_observations import (
    CAMERAS, dreamzero_observation, nano_observation, native_policy_observation,
)


def observation():
    frame = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(1, 8, 12, 3)
    return {
        "image_obs": {name: frame + index for index, name in enumerate(CAMERAS)},
        "proprio_obs": {"arm_joint_pos": np.zeros((1, 7), dtype=np.float32),
                       "gripper_pos": np.ones((1, 1), dtype=np.float32)},
        "scoring_object_states": {"cube": [1, 2, 3]},
    }


def test_observation_whitelist_preserves_batched_arrays_and_distinct_model_slots():
    raw = observation()
    native = native_policy_observation(raw)
    assert set(native) == {"image_obs", "proprio_obs"}
    native["image_obs"]["wrist_cam"].fill(0)
    assert np.ptp(raw["image_obs"]["wrist_cam"])
    packed = nano_observation(raw)
    assert "observation/image" not in packed
    assert "observation/exterior_image_0_left" not in packed
    np.testing.assert_array_equal(packed["observation/exterior_image_1_left"],
                                  raw["image_obs"]["over_shoulder_left_camera"][0])
    bad = observation()
    bad["proprio_obs"]["arm_joint_pos"][0, 0] = np.nan
    with pytest.raises(AdapterError, match="proprioception"):
        native_policy_observation(bad)
    bad = observation()
    bad["image_obs"]["wrist_cam"] = bad["image_obs"]["wrist_cam"].astype(float)
    with pytest.raises(AdapterError, match="uint8"):
        native_policy_observation(bad)


def native_nano_helpers():
    source_path = os.environ.get("SGW01_NANO_SOURCE_AUDIT")
    if not source_path:
        pytest.skip("set the authorized Nano source export; this test additionally requires CPU torch")
    import torch
    import torch.nn.functional as functional

    source = json.loads(Path(source_path).read_text())
    assert hashlib.sha256(source["text"].encode()).hexdigest() == source["sha256"] == (
        "024a6d19048a6ac732ca0b23e741b5da6d38424223783b944bf539b2b6bd9a24"
    )
    names = {"_ensure_rgb_uint8_image", "_resize_rgb_uint8", "_compose_roboarena_views",
             "_extract_observation_image", "_ensure_2d_float_array", "_ensure_gripper_array"}
    nodes = [node for node in ast.parse(source["text"]).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    namespace = {"np": np, "torch": torch, "F": functional, "Any": Any}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), source["path"], "exec"), namespace)
    return namespace


def test_pinned_nano_source_composes_all_views_with_real_tensor_interpolation():
    namespace = native_nano_helpers()
    torch = namespace["torch"]
    raw = observation()
    wire = json.loads(json.dumps(nano_observation(raw), default=lambda value: value.tolist()))
    composed = namespace["_extract_observation_image"](wire)
    assert composed.shape == (12, 12, 3)
    np.testing.assert_array_equal(composed[:8], raw["image_obs"]["wrist_cam"][0])
    resize = namespace["_resize_rgb_uint8"]
    np.testing.assert_array_equal(composed[8:, :6], resize(raw["image_obs"]["over_shoulder_left_camera"][0], (4, 6)))
    np.testing.assert_array_equal(composed[8:, 6:], resize(raw["image_obs"]["over_shoulder_right_camera"][0], (4, 6)))
    assert namespace["_ensure_2d_float_array"](wire["observation/joint_position"], "joint", 7).shape == (1, 7)
    assert namespace["_ensure_gripper_array"](wire["observation/gripper_position"]).shape == (1, 1)
    restored = dreamzero_observation(raw)
    assert isinstance(restored["image_obs"]["wrist_cam"], torch.Tensor)
    np.testing.assert_array_equal(restored["image_obs"]["wrist_cam"].numpy(), raw["image_obs"]["wrist_cam"])
