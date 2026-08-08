from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from tools.build_v3e004_evidence_manifest import build as build_manifest
from tools.compile_v3e004_results import (
    CompileError,
    _claim_gate,
    _failure_signature,
    compile_outputs,
    load_valid_episodes,
    sha256_file,
)
from tools.render_v3e004_results import render
from tools.validate_v3e004_evidence import validate


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"


def queue_rows() -> list[dict]:
    return [json.loads(line) for line in (SOURCE / "queue.jsonl").read_text().splitlines() if line.strip()]


def raw_row(queue: dict, *, success: bool) -> dict:
    category = "correct" if success else "wrong_side"
    sign = 1.0 if queue["relation"] == "left" else -1.0
    return {
        "schema_version": "vla-wam-shared-v3e004-droid-behavioral-episode-v1",
        "record_type": "behavioral_episode",
        "behavioral_result_valid": True,
        "study_id": queue["study_id"],
        "amendment_id": "V3-E004",
        "registered_cell_id": queue["cell_id"],
        "matched_pair_id": queue["matched_pair_id"],
        "model_id": queue["model_id"],
        "arena": queue["arena"],
        "environment_seed": queue["environment_seed"],
        "sampling_seed": queue["sampling_seed"],
        "requested_relation": queue["relation"],
        "prompt": queue["prompt"],
        "success": success,
        "failure_category": category,
        "signed_final_lateral_offset": 0.12 * sign,
        "requested_side_depth": 0.12,
        "cone_entry_step": 10 if success else None,
        "cone_entry_sustained": success,
        "endpoint_shift": None,
        "action_distinct": None,
        "episode_length": 100,
        "time_to_first_contact": None,
        "grasp_step": 4,
        "cumulative_lateral_path": 0.3,
        "peak_lateral_excursion": 0.15,
        "symmetry_level_s": queue["symmetry_level_s"],
        "asymmetry_metric_A": queue["registered_expected_asymmetry_A"],
        "position_residual": 0.0,
        "orientation_residual": 0.0,
        "midline_residual": 0.0,
        "occlusion_check": {"base_camera": False, "left_wrist_camera": False, "right_wrist_camera": False},
        "realised_object_poses": {"rubiks_cube": {"position_xyz_m": [0.3, 0.0, 0.08]}},
        "arm_reset_pose": {"arm_joint_positions_rad": [0.0] * 7},
        "initial_state_sha256": "1" * 64,
        "registration_sha256": sha256_file(SOURCE / "registration.json"),
        "queue_sha256": sha256_file(SOURCE / "queue.jsonl"),
        "candidate_sha256": sha256_file(SOURCE / "layout/candidate.json"),
    }


def write_raw(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    video = path.with_name("simulator.mp4")
    actions = path.with_name("actions.npy")
    video.write_bytes(b"test-video-evidence")
    np.save(actions, np.zeros((10, 8), dtype=np.float32), allow_pickle=False)
    row["artifacts"] = {
        "viewport_video": {"path": str(video), "bytes": video.stat().st_size, "sha256": sha256_file(video)},
        "executed_action_trace": {"path": str(actions), "bytes": actions.stat().st_size, "sha256": sha256_file(actions)},
    }
    path.write_text(json.dumps(row, sort_keys=True) + "\n")


def prepare_base(tmp_path: Path) -> Path:
    base = tmp_path / "e004"
    for relative in ("registration.json", "queue.jsonl", "layout/candidate.json", "gates/static_layout_gate.json"):
        target = base / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / relative, target)
    return base


def test_partial_pipeline_withholds_claims_and_hash_closes_progress(tmp_path: Path):
    rows = [
        row
        for row in queue_rows()
        if row["model_id"] == "cosmos3_edge_policy_droid"
        and row["environment_seed"] == 9400
        and row["symmetry_level_s"] == 0.0
    ]
    raw_root = tmp_path / "raw"
    for row in rows:
        write_raw(raw_root / row["relation"] / "raw_episode.jsonl", raw_row(row, success=row["relation"] == "right"))
    base = prepare_base(tmp_path)
    report = compile_outputs(
        registration_path=base / "registration.json",
        queue_path=base / "queue.jsonl",
        raw_roots=[raw_root],
        output_root=base,
        resamples=10_000,
        require_complete=False,
    )
    assert report["status"] == "partial_progress_no_publication_claims"
    assert report["valid_behavioral_episodes"] == 2
    assert report["publication_claim_status"] == "withheld_until_all_registered_cells_are_valid"
    assert all(item["claim_gate"]["publication_claims_enabled"] is False for item in report["checkpoints"].values())
    compact = [json.loads(line) for line in (base / "results/episodes.jsonl").read_text().splitlines()]
    assert all(row["pair_fields_status"] == "derived_after_both_hash_bound_directions_exist" for row in compact)
    assert all(row["endpoint_shift"] == pytest.approx(-0.24) for row in compact)
    assert all(row["action_distinct"] is False for row in compact)
    assert len((base / "results/pairs.jsonl").read_text().splitlines()) == 1
    figure_manifest = render(base / "results/results.json", base / "results/figures")
    assert figure_manifest["status"] == "partial_progress_figure_only"
    assert len(figure_manifest["figures"]) == 2
    manifest = build_manifest(base, base / "evidence_manifest.json")
    assert manifest["status"] == "partial_progress_not_publication_evidence"
    checked = validate(base, require_complete=False, verify_raw_sources=True)
    assert checked["status"] == "valid_partial_no_publication_claims"
    with pytest.raises(Exception, match="incomplete"):
        validate(base, require_complete=True, verify_raw_sources=False)


def test_conflicting_valid_duplicate_fails_closed(tmp_path: Path):
    queue = next(
        row
        for row in queue_rows()
        if row["model_id"] == "cosmos3_edge_policy_droid"
        and row["environment_seed"] == 9400
        and row["symmetry_level_s"] == 0.0
        and row["relation"] == "left"
    )
    raw_root = tmp_path / "raw"
    write_raw(raw_root / "first" / "raw_episode.jsonl", raw_row(queue, success=False))
    write_raw(raw_root / "second" / "raw_episode.jsonl", raw_row(queue, success=True))
    registered = {row["cell_id"]: row for row in queue_rows()}
    with pytest.raises(CompileError, match="conflicting valid duplicate"):
        load_valid_episodes(
            [raw_root],
            queue_rows=registered,
            registration_sha256=sha256_file(SOURCE / "registration.json"),
            queue_sha256=sha256_file(SOURCE / "queue.jsonl"),
            candidate_sha256=sha256_file(SOURCE / "layout/candidate.json"),
        )


def test_claim_gate_never_calls_underpowered_null_equivalent():
    analysis = {
        "levels": {"1.00": {"endpoint_redirection_LEFT_minus_RIGHT_m": {"bootstrap_mean95": {"low": 0.01}}}},
        "equivalence_at_s1": {
            "binary_gap": {"power_status": "underpowered_no_equivalence_claim", "equivalent": True, "margin": 0.2},
            "depth_gap_m": {"power_status": "strictly_powered_at_endpoints", "equivalent": True, "margin": 0.05},
        },
    }
    gate = _claim_gate(analysis, globally_complete=True)
    assert gate["equivalence_claims"]["binary_gap"]["publication_equivalence_claim_allowed"] is False
    assert gate["equivalence_claims"]["depth_gap_m"]["publication_equivalence_claim_allowed"] is True


def test_failure_signature_keeps_zero_failure_level_unavailable():
    rows = []
    for relation in ("left", "right"):
        rows.append(
            {
                "environment_seed": 9400,
                "symmetry_level_s": 1.0,
                "relation": relation,
                "success": True,
                "failure_category": "correct",
                "asymmetry_metric_A": 0.0,
            }
        )
    result = _failure_signature(rows, resamples=10_000)
    assert result["levels"]["1.00"]["wrong_side_share_among_failures"] is None
    assert result["levels"]["1.00"]["availability"] == "unavailable_no_failures"
    assert result["trend"]["status"].startswith("unavailable")
