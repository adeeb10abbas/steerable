from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.request0_replay import (
    AMENDMENT_SCHEMA,
    RESET_CONTRACT_SCHEMA,
    Request0ReplayError,
    canonical_json_sha256,
    capture_left_observation,
    observation_payload_sha256,
    replay_left_observation_for_right,
    sha256_file,
)


def _write_amendment(path: Path) -> str:
    path.write_text(
        json.dumps(
            {
                "schema_version": AMENDMENT_SCHEMA,
                "registered_before_new_request": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def _contract(marker: float = 1.0) -> dict:
    value = {
        "schema_version": RESET_CONTRACT_SCHEMA,
        "robot": {"joint_position": [marker, 2.0]},
        "rigid_objects": {"cube": [0.3, 0.0, 0.08]},
        "cameras": {"head": {"shape": [2, 3, 3], "dtype": "uint8"}},
        "observation_contract": {"version": 1},
    }
    value["reset_contract_sha256"] = canonical_json_sha256(value)
    return value


def _observation(image_offset: int = 0) -> dict:
    return {
        "image_obs": {
            "head": (np.arange(18, dtype=np.uint8).reshape(1, 2, 3, 3) + image_offset),
            "wrist": np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3),
        },
        "proprio_obs": {
            "arm_joint_pos": np.asarray([[0.1, 0.2]], dtype=np.float32),
            "gripper_pos": np.asarray([[1.0]], dtype=np.float32),
        },
        "history": [np.asarray([True, False], dtype=np.bool_)],
    }


def _capture(tmp_path: Path):
    amendment = tmp_path / "amendment.json"
    amendment_sha = _write_amendment(amendment)
    cache = tmp_path / "request0.npz"
    manifest = tmp_path / "request0.manifest.json"
    contract = tmp_path / "reset.json"
    source = _observation()
    result = capture_left_observation(
        observation=source,
        reset_contract=_contract(),
        amendment_path=amendment,
        amendment_sha256=amendment_sha,
        cell_id="v3e004:nano:seed9400:s100:left",
        matched_pair_id="v3e004:nano:seed9400:s100",
        cache_path=cache,
        manifest_path=manifest,
        reset_contract_path=contract,
    )
    return amendment, amendment_sha, cache, manifest, contract, source, result


def test_request0_round_trip_replays_every_non_language_leaf_losslessly(tmp_path: Path) -> None:
    amendment, amendment_sha, cache, manifest, contract, source, captured = _capture(tmp_path)
    native_right = _observation(image_offset=37)
    native_right_before = native_right["image_obs"]["head"].copy()
    attestation = tmp_path / "right.attestation.json"
    replayed, result = replay_left_observation_for_right(
        native_observation=native_right,
        native_reset_contract=_contract(),
        amendment_path=amendment,
        amendment_sha256=amendment_sha,
        cell_id="v3e004:nano:seed9400:s100:right",
        matched_pair_id="v3e004:nano:seed9400:s100",
        cache_path=cache,
        cache_sha256=sha256_file(cache),
        manifest_path=manifest,
        manifest_sha256=sha256_file(manifest),
        reset_contract_path=contract,
        reset_contract_file_sha256=sha256_file(contract),
        native_reset_contract_path=tmp_path / "right.reset.json",
        attestation_path=attestation,
    )
    assert observation_payload_sha256(replayed) == observation_payload_sha256(source)
    assert np.array_equal(replayed["image_obs"]["head"], source["image_obs"]["head"])
    assert np.array_equal(native_right["image_obs"]["head"], native_right_before)
    assert result["physical_state_and_camera_contract_bit_identical"] is True
    assert result["request0_non_language_bytes_bit_identical"] is True
    assert result["model_request_count_at_attestation"] == 0
    assert captured["model_request_count_at_capture"] == 0


def test_request0_replay_fails_before_attestation_on_physical_mismatch(tmp_path: Path) -> None:
    amendment, amendment_sha, cache, manifest, contract, _, _ = _capture(tmp_path)
    attestation = tmp_path / "right.attestation.json"
    with pytest.raises(Request0ReplayError, match="physical state or camera contract"):
        replay_left_observation_for_right(
            native_observation=_observation(image_offset=1),
            native_reset_contract=_contract(marker=9.0),
            amendment_path=amendment,
            amendment_sha256=amendment_sha,
            cell_id="v3e004:nano:seed9400:s100:right",
            matched_pair_id="v3e004:nano:seed9400:s100",
            cache_path=cache,
            cache_sha256=sha256_file(cache),
            manifest_path=manifest,
            manifest_sha256=sha256_file(manifest),
            reset_contract_path=contract,
            reset_contract_file_sha256=sha256_file(contract),
            native_reset_contract_path=tmp_path / "mismatch.reset.json",
            attestation_path=attestation,
        )
    assert not attestation.exists()


def test_request0_replay_fails_closed_on_archive_or_pair_change(tmp_path: Path) -> None:
    amendment, amendment_sha, cache, manifest, contract, _, _ = _capture(tmp_path)
    with pytest.raises(Request0ReplayError, match="different pair"):
        replay_left_observation_for_right(
            native_observation=_observation(),
            native_reset_contract=_contract(),
            amendment_path=amendment,
            amendment_sha256=amendment_sha,
            cell_id="v3e004:nano:seed9401:s100:right",
            matched_pair_id="v3e004:nano:seed9401:s100",
            cache_path=cache,
            cache_sha256=sha256_file(cache),
            manifest_path=manifest,
            manifest_sha256=sha256_file(manifest),
            reset_contract_path=contract,
            reset_contract_file_sha256=sha256_file(contract),
            native_reset_contract_path=tmp_path / "wrong.reset.json",
            attestation_path=tmp_path / "wrong.attestation.json",
        )
    cache.write_bytes(cache.read_bytes() + b"corrupt")
    with pytest.raises(Request0ReplayError, match="cache digest mismatch"):
        replay_left_observation_for_right(
            native_observation=_observation(),
            native_reset_contract=_contract(),
            amendment_path=amendment,
            amendment_sha256=amendment_sha,
            cell_id="v3e004:nano:seed9400:s100:right",
            matched_pair_id="v3e004:nano:seed9400:s100",
            cache_path=cache,
            cache_sha256="0" * 64,
            manifest_path=manifest,
            manifest_sha256=sha256_file(manifest),
            reset_contract_path=contract,
            reset_contract_file_sha256=sha256_file(contract),
            native_reset_contract_path=tmp_path / "corrupt.reset.json",
            attestation_path=tmp_path / "corrupt.attestation.json",
        )


def test_request0_capture_refuses_overwrite(tmp_path: Path) -> None:
    amendment, amendment_sha, cache, manifest, contract, source, _ = _capture(tmp_path)
    with pytest.raises(Request0ReplayError, match="already exists"):
        capture_left_observation(
            observation=source,
            reset_contract=_contract(),
            amendment_path=amendment,
            amendment_sha256=amendment_sha,
            cell_id="v3e004:nano:seed9400:s100:left",
            matched_pair_id="v3e004:nano:seed9400:s100",
            cache_path=cache,
            manifest_path=manifest,
            reset_contract_path=contract,
        )


def test_request0_torch_round_trip_preserves_dtype_and_device(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    amendment = tmp_path / "amendment.json"
    amendment_sha = _write_amendment(amendment)
    cache = tmp_path / "torch.npz"
    manifest = tmp_path / "torch.manifest.json"
    contract = tmp_path / "torch.reset.json"
    source = {
        "image_obs": {"head": torch.arange(18, dtype=torch.uint8).reshape(1, 2, 3, 3)},
        "proprio_obs": {"joint": torch.tensor([[0.1, 0.2]], dtype=torch.float32)},
    }
    capture_left_observation(
        observation=source,
        reset_contract=_contract(),
        amendment_path=amendment,
        amendment_sha256=amendment_sha,
        cell_id="v3e004:nano:seed9400:s100:left",
        matched_pair_id="v3e004:nano:seed9400:s100",
        cache_path=cache,
        manifest_path=manifest,
        reset_contract_path=contract,
    )
    native = {
        "image_obs": {"head": torch.full((1, 2, 3, 3), 9, dtype=torch.uint8)},
        "proprio_obs": {"joint": torch.tensor([[7.0, 8.0]], dtype=torch.float32)},
    }
    replayed, _ = replay_left_observation_for_right(
        native_observation=native,
        native_reset_contract=_contract(),
        amendment_path=amendment,
        amendment_sha256=amendment_sha,
        cell_id="v3e004:nano:seed9400:s100:right",
        matched_pair_id="v3e004:nano:seed9400:s100",
        cache_path=cache,
        cache_sha256=sha256_file(cache),
        manifest_path=manifest,
        manifest_sha256=sha256_file(manifest),
        reset_contract_path=contract,
        reset_contract_file_sha256=sha256_file(contract),
        native_reset_contract_path=tmp_path / "torch.right.reset.json",
        attestation_path=tmp_path / "torch.right.attestation.json",
    )
    for group, key in (("image_obs", "head"), ("proprio_obs", "joint")):
        assert replayed[group][key].dtype == source[group][key].dtype
        assert replayed[group][key].device == native[group][key].device
        assert torch.equal(replayed[group][key], source[group][key])
