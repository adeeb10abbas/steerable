import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1 import nano_fixed_input as fixed
from experiments.workshops.spatial_grounding_v1.paper_engineering import record


def capture(tmp_path):
    result = {"views": {}, "robot_snapshot": {
        "joint_names": [f"panda_joint{i}" for i in range(1, 8)] + ["finger_joint"],
        "joint_position_rad": [0, 0.1, 0.2, -2, 0.3, 1, 0.4, np.pi / 8],
    }, "objects": {"cube": {"position": [10, 20, 30]}}}
    for index, name in enumerate(fixed.CAMERAS):
        path = tmp_path / (name + ".npy")
        fixed.save_array(path, np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3) + index)
        result["views"][name] = {"lossless_array": record(path)}
    return result


def test_retained_observation_has_only_native_images_and_correct_proprioception(tmp_path):
    raw = capture(tmp_path)
    value = fixed.native_observation(raw)
    assert set(value) == {
        "observation/wrist_image_left", "observation/exterior_image_1_left",
        "observation/exterior_image_2_left", "observation/joint_position",
        "observation/gripper_position",
    }
    np.testing.assert_array_equal(value["observation/joint_position"], np.asarray(raw["robot_snapshot"]["joint_position_rad"][:7], dtype=np.float32))
    np.testing.assert_array_equal(value["observation/gripper_position"], np.array([0.5], dtype=np.float32))
    assert value["observation/wrist_image_left"].shape == (8, 12, 3)
    assert "objects" not in value
    path = tmp_path / "wrist_cam.npy"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes differ"):
        fixed.native_observation(raw)


def test_observation_rejects_ambiguous_joint_order(tmp_path):
    raw = capture(tmp_path)
    raw["robot_snapshot"]["joint_names"][:2] = ["panda_joint2", "panda_joint1"]
    with pytest.raises(ValueError, match="order"):
        fixed.native_observation(raw)


def registration(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"candidate_id": "SGW-ENG-008-LAT-057", "status": "verified_all_six_pass"}))
    return {
        "schema_version": fixed.SCHEMA, "model_config": fixed.NANO_CONFIG,
        "request_prompt_ids": fixed.ORDER, "maximum_model_requests": 6,
        "behavioral_episodes": 0, "executed_actions": 0, "sampling_seed": 1140,
        "native_decode_video": True, "release_permitted": False,
        **{key: record(source) for key in ("capture", "native_proprio_source", "native_server_source", "prompts", "engineering_verification")},
    }


@pytest.mark.parametrize("key,value", [
    ("maximum_model_requests", 7), ("behavioral_episodes", 6), ("executed_actions", 32),
    ("sampling_seed", 42), ("native_decode_video", False), ("release_permitted", True),
    ("request_prompt_ids", ["LAT-C-POS"] * 6),
])
def test_unregistered_requests_or_behavior_refused(tmp_path, key, value):
    spec = registration(tmp_path)
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(spec))
    assert fixed.load_registration(path)["maximum_model_requests"] == 6
    spec[key] = value
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="bounded six-request"):
        fixed.load_registration(path)


def test_compare_records_nondeterminism_instead_of_hiding_it():
    left = np.zeros((32, 8), dtype=np.float32)
    right = np.ones((32, 8), dtype=np.float32)
    actions = [left.copy(), left.copy(), right.copy()] * 2
    result = fixed.compare_actions(actions)
    assert result["repeat_equal"] == [True, True]
    assert result["opposite_prompt_distinct"] == [True, True]
    assert result["opposite_prompt_action_rms"] == [1, 1]
    assert result["between_groups_equal"] == [True] * 3
    changed = [value.copy() for value in actions]
    changed[1][0, 0] = 0.1
    assert fixed.compare_actions(changed)["repeat_equal"] == [False, True]
    with pytest.raises(ValueError, match="six finite"):
        fixed.compare_actions(actions[:5])


def test_arrays_and_runtime_output_are_exclusive(tmp_path):
    path = tmp_path / "output.npy"
    fixed.save_array(path, np.ones((2, 3), dtype=np.float32))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        fixed.save_array(path, np.zeros((2, 3), dtype=np.float32))
    assert path.read_bytes() == before
    spec_path = tmp_path / "registration.json"
    spec_path.write_text(json.dumps(registration(tmp_path)))
    with pytest.raises(FileExistsError):
        fixed.run(spec_path, tmp_path)


@pytest.mark.parametrize("fail_index", [None, 2])
def test_six_real_http_boundaries_with_fake_model_and_no_retry(tmp_path, monkeypatch, fail_index):
    import torch
    from experiments.workshops.spatial_grounding_v1 import nano_backend, producer
    from experiments.workshops.spatial_grounding_v1.adapters import AdapterError

    source = tmp_path / "capture.json"
    source.write_text(json.dumps(capture(tmp_path)))
    spec = registration(tmp_path)
    spec["registration_id"] = "unit-test-not-scientific-evidence"
    spec["capture"] = record(source)
    spec["prompts"] = record(Path("experiments/workshops/spatial_grounding_v1/spec/prompts.json"))
    spec_path = tmp_path / "registration.json"
    spec_path.write_text(json.dumps(spec))

    @dataclass
    class Config:
        seed: int = 1140
        deterministic_seed: bool = True

    counts = {"requests": 0, "decodes": 0}

    class Model:
        def decode(self, latent):
            counts["decodes"] += 1
            return latent.unsqueeze(0)

    class Service:
        cfg = Config()
        model = Model()

        def infer(self, observation):
            index = counts["requests"]
            counts["requests"] += 1
            if index == fail_index:
                raise AdapterError("deliberate fixture failure")
            latent = torch.linspace(-1, 1, 3 * 33 * 2 * 2).reshape(3, 33, 2, 2)
            decoded = self.model.decode(latent)
            video = ((decoded[0].clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).numpy()
            action = np.full((32, 8), int("right" in observation["prompt"]), dtype=np.float32)
            return {"action": action, "video": video}

    backend = nano_backend.CosmosNanoBackend(
        Service(), resolved_config=fixed.NANO_CONFIG, source_root="/fake", checkpoint_path="/fake",
    )
    monkeypatch.setattr(nano_backend, "build_pinned_nano_backend", lambda: backend)
    monkeypatch.setattr(producer, "derive_nano_attestation", lambda *a, **k: {
        "config": fixed.NANO_CONFIG, "test_fixture_only": True,
    })
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "CPU unit fixture")
    original_to = torch.Tensor.to

    def cpu_only_to(self, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_only_to)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test-only")
    output = tmp_path / "result"
    if fail_index is None:
        result = fixed.run(spec_path, output)
        assert counts == {"requests": 6, "decodes": 9}
        assert result["model_requests"] == 6 and result["executed_actions"] == 0
        assert all(item["native_decode_equal"] for item in result["offline_redecodes"])
        assert result["comparisons"]["between_groups_equal"] == [True] * 3
        assert len(list(output.glob("request-*/intent.json"))) == 6
        assert len(list(output.glob("request-*/vision-latent.npy"))) == 6
        assert len(list(output.glob("futures/*.npy"))) == 6
    else:
        with pytest.raises(Exception, match="400"):
            fixed.run(spec_path, output)
        failure = json.loads((output / "failure.json").read_text())
        assert failure["model_requests_started"] == counts["requests"] == 3
        assert len(list(output.glob("request-*/result.json"))) == 2
        assert not (output / "result.json").exists()
    with pytest.raises(FileExistsError):
        fixed.run(spec_path, output)
