from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.offline_recovery import (
    CANDIDATE_SHA256,
    QUEUE_SHA256,
    RECOVERY_BASE_COMMIT,
    REGISTRATION_SHA256,
    RecoverySpec,
    _find_native_output,
    _validate_attempt,
    production_specs,
    select_specs,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.runtime_contract import (
    RuntimeContractError,
    load_runtime_bundle,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"


def _bundle():
    return load_runtime_bundle(
        registration_path=ARTIFACT / "registration.json",
        registration_sha256=REGISTRATION_SHA256,
        queue_path=ARTIFACT / "queue.jsonl",
        queue_sha256=QUEUE_SHA256,
        candidate_path=ARTIFACT / "layout/candidate.json",
        candidate_sha256=CANDIDATE_SHA256,
    )


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_production_recovery_scope_is_exact_and_new_root_is_separate(tmp_path: Path):
    specs = production_specs(tmp_path)
    assert [spec.name for spec in specs] == ["pi05_s100_left", "dreamzero_s100_left", "nano_s100_left"]
    assert all("left" in spec.cell_id and "seed9400:s100" in spec.cell_id for spec in specs)
    assert all("final-69998d6/droid" in str(spec.destination_attempt) for spec in specs)
    assert specs[0].source_attempt != specs[0].destination_attempt
    assert specs[1].source_attempt != specs[1].destination_attempt
    assert specs[2].recovery_mode == "recompile_existing_export"
    assert len(RECOVERY_BASE_COMMIT) == 40
    with pytest.raises(RuntimeContractError, match="outside allowlist"):
        select_specs(specs, ["edge_s100_left"])


def _synthetic_attempt(tmp_path: Path) -> tuple[RecoverySpec, dict, dict]:
    bundle = _bundle()
    cell = bundle.cell("v3e004:pi05:seed9400:s100:left")
    root = tmp_path / "source"
    attempt = {
        "schema_version": "vla-wam-shared-v3e004-droid-attempt-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E004",
        "registered_cell_id": cell.cell_id,
        "registered_cell_sha256": cell.row_sha256,
        "model_id": cell.model_id,
        "registration_sha256": REGISTRATION_SHA256,
        "queue_sha256": QUEUE_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "created_unix_s": 100.0,
        "invocation_argv": ["python", "--expected-study-commit", "a" * 40],
    }
    _write(root / "attempt_manifest.json", attempt)
    infra = {
        "schema_version": "vla-wam-shared-v3e004-infrastructure-attempt-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E004",
        "registered_cell_id": cell.cell_id,
        "registered_cell_sha256": cell.row_sha256,
        "model_id": cell.model_id,
        "behavioral_result_valid": False,
        "denominator_eligible": False,
        "stage": "bridge_or_compile",
        "error_type": "RuntimeContractError",
        "error": "bridge completed without canonical simulator export",
        "attempt_manifest": {
            "path": str(root / "attempt_manifest.json"),
            "bytes": (root / "attempt_manifest.json").stat().st_size,
            "sha256": sha256_file(root / "attempt_manifest.json"),
        },
    }
    _write(root / "infrastructure_invalid.json", infra)
    steps = [
        {"action_step": 0, "object_xyz": [0.30, 0.0, 0.08], "reference_xyz": [0.44, 0.0, 0.08], "grippers_open": True, "object_grabbed": False},
        {"action_step": 1, "object_xyz": [0.44, 0.20, 0.08], "reference_xyz": [0.44, 0.0, 0.08], "grippers_open": True, "object_grabbed": False},
    ]
    capture = {
        "schema_version": "vla-wam-shared-v3e004-droid-state-capture-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E004",
        "registered_cell_id": cell.cell_id,
        "registered_cell_sha256": cell.row_sha256,
        "model_id": cell.model_id,
        "environment_seed": 9400,
        "sampling_seed": 9400,
        "requested_relation": "left",
        "prompt": cell.row["prompt"],
        "action_cap": 450,
        "actions_executed": 1,
        "final_detached_release": True,
        "requested_success": True,
        "right_censored": False,
        "steps": steps,
    }
    _write(root / "state_capture/state_capture.json", capture)
    (root / "state_capture/states.partial.jsonl").write_text(
        "".join(json.dumps(step, sort_keys=True) + "\n" for step in steps), encoding="utf-8"
    )
    spec = RecoverySpec(
        name="test",
        cell_id=cell.cell_id,
        model_id=cell.model_id,
        source_attempt=root,
        destination_attempt=tmp_path / "destination",
        source_code_commit="a" * 40,
        fault_type="RuntimeContractError",
        fault_text="bridge completed without canonical simulator export",
        action_metadata_name="trace.json",
        policy_output_suffix="pi05_v2a010_current",
        recovery_mode="reconstruct_missing_export",
    )
    return spec, attempt, capture


def test_attempt_validation_recomputes_one_step_frozen_predicate(tmp_path: Path):
    spec, _, capture = _synthetic_attempt(tmp_path)
    _, _, observed = _validate_attempt(_bundle(), spec)
    assert observed == capture
    capture["requested_success"] = False
    _write(spec.source_attempt / "state_capture/state_capture.json", capture)
    with pytest.raises(RuntimeContractError, match="recomputed frozen B001"):
        _validate_attempt(_bundle(), spec)


def test_native_viewport_lookup_rejects_ambiguous_policy_outputs(tmp_path: Path):
    spec, attempt, capture = _synthetic_attempt(tmp_path)
    infra = spec.source_attempt / "infrastructure_invalid.json"
    # Enlarge the valid mtime window without changing the exact failure payload.
    infra.touch()
    for stamp in ("one", "two"):
        video = tmp_path / "videos" / f"{stamp}_pi05_v2a010_current/V3E004DroidLeftTask/Put_the_Rubiks_cube_to_the_left_of_the_bowl_0_viewport.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        _write(
            video.parents[1] / "episode_results.jsonl",
            {
                "policy": "pi05_v2a010_current",
                "instruction": "Put the Rubik's cube to the left of the bowl.",
                "episode_step": 1,
            },
        )
        # The synthetic attempt uses epoch 100; make both candidates eligible.
        video.touch()
    attempt["created_unix_s"] = 0.0
    with pytest.raises(RuntimeContractError, match="one unambiguous"):
        _find_native_output(spec, attempt, capture, tmp_path / "videos", probe_video=False)
