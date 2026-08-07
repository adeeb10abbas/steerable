"""Focused regressions for the Nano V3-B005 release boundary."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from experiments.v3.cosmos_nano_lateral_sweep.build_release_gate import (
    build_release_gate,
)
from experiments.v3.cosmos_nano_lateral_sweep.fixed_observation_gate import (
    collect_responses,
    evaluate_responses,
)
from experiments.v3.cosmos_nano_lateral_sweep.live_support import (
    PROBE_SEQUENCE,
    authorize_probe_request,
    bind_live_stack_runtime,
    observation_component_hashes,
    validate_fixed_observation_report,
    verify_behavioral_release_gate,
    verify_live_runtime_identity,
)
from experiments.v3.cosmos_nano_lateral_sweep.runtime_adapter import (
    CHECKPOINT_REVISION,
    COSMOS_REPOSITORY_COMMIT,
    EMPTY_SHA256,
    EXPECTED_SHA256,
    LEVELS,
    MODEL_ID,
    MODEL_REPOSITORY,
    ROBOLAB_REPOSITORY_COMMIT,
    RuntimeContractError,
    load_release_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (
    ROOT / "artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b005"
)
MANIFEST = ARTIFACT_ROOT / "nano_lateral_v3b005_manifest.json"


def _base_runtime() -> dict:
    return {
        "study_id": "vla_wam_language_steerability_v3",
        "model_id": MODEL_ID,
        "checkpoint_identifier": MODEL_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": "1" * 64,
        "checkpoint_hash_gate_passed": True,
        "external_repository_commit": COSMOS_REPOSITORY_COMMIT,
        "external_repository_diff_hash": EMPTY_SHA256,
        "simulator_repository_commit": ROBOLAB_REPOSITORY_COMMIT,
        "simulator_repository_diff_hash": EMPTY_SHA256,
        "environment_lock_hash": "2" * 64,
        "runtime_identity_sha256": "3" * 64,
        "adapter_contract_hash": "4" * 64,
        "repository_pins": {"frozen": "5" * 64},
        "simulator_version": "Isaac Sim 5 / RoboLab 0.2.1",
        "renderer_backend": "RTX Vulkan",
    }


def _runtime(tmp_path: Path, release):
    base_path = tmp_path / "phase_a_runtime.json"
    base_path.write_text(json.dumps(_base_runtime()) + "\n")
    with patch(
        "experiments.v3.cosmos_nano_lateral_sweep.live_support.verify_phase_a_runtime_identity",
        return_value=_base_runtime(),
    ):
        runtime = bind_live_stack_runtime(
            study_root=ROOT,
            release=release,
            base_runtime_manifest=base_path,
        )
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    assert verify_live_runtime_identity(
        runtime_path, study_root=ROOT, release=release
    ) == runtime
    return runtime_path, runtime


def _observations():
    return {
        level: {
            "observation/image": np.full((4, 5, 3), level, dtype=np.uint8),
            "observation/joint_position": np.arange(7, dtype=np.float32) + level,
            "observation/gripper_position": np.array([0.25], dtype=np.float32),
        }
        for level in (0, 3, 6)
    }


def _fake_infer(request):
    condition = request["probe_condition"]
    level = request["probe_level_index"]
    value = float(level + (1 if condition == "right" else 0))
    action = np.full((32, 8), value, dtype=np.float32)
    video = np.full((33, 2, 3, 3), int(value), dtype=np.uint8)
    return {
        "action": action,
        "video": video,
        "v3b005_server_mode": "probe_only",
        "amendment_id": "V3-B005",
        "registered_cell_id": request["registered_cell_id"],
        "sampling_seed": 9500,
        "request_index": request["probe_request_index"],
        "probe_request_index": request["probe_request_index"],
        "probe_level_index": level,
        "probe_condition": condition,
        "release_fingerprint_sha256": request["release_fingerprint_sha256"],
        "runtime_identity_sha256": request["runtime_identity_sha256"],
    }


def test_exact_committed_release_and_derived_level_fixture_identity() -> None:
    release = load_release_bundle(MANIFEST)
    assert len(release.cells) == 210
    assert release.hashes == EXPECTED_SHA256
    cell = release.cell("v3b005:nano:seed9500:level3:left")
    assert cell.level_index == 3
    assert cell.arm == "level3"
    assert cell.fixture_id == "v3b005:nano:lateral_level3"
    assert len(cell.fixture_sha256) == 64
    assert release.amendment_sha256 == EXPECTED_SHA256["amendment"]
    assert release.cells_sha256 == EXPECTED_SHA256["cells"]


def test_release_loader_rejects_any_substituted_manifest_hash() -> None:
    with pytest.raises(RuntimeContractError, match="committed SHA-256"):
        load_release_bundle(MANIFEST, expected_manifest_sha256="0" * 64)


def test_probe_authorizer_enforces_order_and_per_level_identical_observations(tmp_path) -> None:
    release = load_release_bundle(MANIFEST)
    _, runtime = _runtime(tmp_path, release)
    observations = _observations()
    remembered = {}
    for index, (level, condition) in enumerate(PROBE_SEQUENCE):
        relation = "left" if condition.startswith("left") else "right"
        cell = release.cell(f"v3b005:nano:seed9500:level{level}:{relation}")
        request = {
            **observations[level],
            "v3b005_server_mode": "probe_only",
            "amendment_id": "V3-B005",
            "probe_request_index": index,
            "probe_level_index": level,
            "probe_condition": condition,
            "registered_cell_id": cell.cell_id,
            "sampling_seed": 9500,
            "prompt": cell.row["prompt"],
            "release_fingerprint_sha256": release.release_fingerprint(cell),
            "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        }
        request["observation_hashes"] = observation_component_hashes(request)
        authorize_probe_request(
            request,
            release=release,
            runtime=runtime,
            expected_request_index=index,
            observation_hashes_by_level=remembered,
        )
    assert set(remembered) == {0, 3, 6}

    level, condition = PROBE_SEQUENCE[1]
    cell = release.cell(f"v3b005:nano:seed9500:level{level}:left")
    changed = {
        **observations[level],
        "observation/image": observations[level]["observation/image"].copy(),
        "v3b005_server_mode": "probe_only",
        "amendment_id": "V3-B005",
        "probe_request_index": 1,
        "probe_level_index": level,
        "probe_condition": condition,
        "registered_cell_id": cell.cell_id,
        "sampling_seed": 9500,
        "prompt": cell.row["prompt"],
        "release_fingerprint_sha256": release.release_fingerprint(cell),
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
    }
    changed["observation/image"][0, 0, 0] = 99
    changed["observation_hashes"] = observation_component_hashes(changed)
    with pytest.raises(RuntimeContractError, match="not byte-identical"):
        authorize_probe_request(
            changed,
            release=release,
            runtime=runtime,
            expected_request_index=1,
            observation_hashes_by_level={0: remembered[0]},
        )


def test_nine_request_report_and_separate_behavioral_release(tmp_path) -> None:
    release = load_release_bundle(MANIFEST)
    runtime_path, runtime = _runtime(tmp_path, release)
    responses = collect_responses(
        release=release,
        runtime=runtime,
        observations=_observations(),
        infer=_fake_infer,
    )
    report = evaluate_responses(release=release, runtime=runtime, responses=responses)
    assert report["status"] == "passed"
    assert report["model_request_count"] == 9
    assert report["behavioral_episode_count"] == 0
    assert all(report["metrics"][f"level{level}"]["passed"] for level in (0, 3, 6))
    report_path = tmp_path / "fixed_observation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    validate_fixed_observation_report(report_path, release=release, runtime=runtime)

    gate = build_release_gate(
        study_root=ROOT,
        manifest=MANIFEST,
        manifest_sha256=EXPECTED_SHA256["manifest"],
        runtime_manifest=runtime_path,
        fixed_observation_report=report_path,
    )
    assert gate["behavioral_release"] is True
    assert gate["model_request_count_before_release"] == 9
    gate_path = tmp_path / "release_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    assert verify_behavioral_release_gate(
        gate_path, release=release, runtime=runtime
    )["authorized_behavioral_cell_count"] == 210


def test_registered_level_coordinates_are_exact() -> None:
    release = load_release_bundle(MANIFEST)
    for cell in release.cells:
        assert cell.row["reference_object_initial_lateral_position_y_m"] == LEVELS[
            cell.level_index
        ]
