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
    _geometry_summary,
    _attach_power_audit,
    _reset_pose_memo_summary,
    compile_outputs,
    load_infrastructure_invalid,
    load_valid_episodes,
    sha256_file,
)
from tools.render_v3e004_results import render, render_failures, render_geometry_quality
from tools.validate_v3e004_evidence import validate
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.layout_contract import candidate_from_json
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.fastwam_robotwin import (
    asymmetry_A as fastwam_asymmetry_A,
    layout_for_level as fastwam_layout_for_level,
    residuals as fastwam_residuals,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"


def queue_rows() -> list[dict]:
    return [json.loads(line) for line in (SOURCE / "queue.jsonl").read_text().splitlines() if line.strip()]


def raw_row(queue: dict, *, success: bool) -> dict:
    category = "correct" if success else "wrong_side"
    sign = 1.0 if queue["relation"] == "left" else -1.0
    level = float(queue["symmetry_level_s"])
    if queue["arena"] == "droid_robolab":
        candidate = candidate_from_json(json.loads((SOURCE / "layout/candidate.json").read_text()))
        poses = candidate.layout(level)
        realised = {name: pose.to_json() for name, pose in poses.items()}
        residual = candidate.residuals(poses)
        expected_A = candidate.asymmetry_A(poses)
        occlusion = {name: False for name in candidate.expected_cameras}
    else:
        poses = fastwam_layout_for_level(level)
        realised = {name: pose.to_json() for name, pose in poses.items()}
        residual = fastwam_residuals(poses)
        expected_A = fastwam_asymmetry_A(poses)
        occlusion = {"head_camera": False, "left_camera": False, "right_camera": False}
    result = {
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
        "asymmetry_metric_A": expected_A,
        "position_residual": residual["position_residual_m"],
        "orientation_residual": residual["orientation_residual_rad"],
        "midline_residual": residual["midline_residual_m"],
        "occlusion_check": occlusion,
        "realised_object_poses": realised,
        "arm_reset_pose": {"arm_joint_positions_rad": [0.0] * 7},
        "initial_state_sha256": "1" * 64,
        "registration_sha256": sha256_file(SOURCE / "registration.json"),
        "queue_sha256": sha256_file(SOURCE / "queue.jsonl"),
        "candidate_sha256": queue["layout_candidate_sha256"],
    }
    if queue["arena"] == "droid_robolab":
        result.update(
            {
                "request0_pair_identity_sha256": "2" * 64,
                "request0_observation_payload_sha256": "3" * 64,
                "request0_reset_contract_sha256": "4" * 64,
                "request0_replay_mode": "capture_left" if queue["relation"] == "left" else "replay_right",
            }
        )
    return result


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
        and row["symmetry_level_s"] == 1.0
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
    assert report["discovery_only_behavioral_artifacts_by_reason"] == {}
    assert all(item["claim_gate"]["publication_claims_enabled"] is False for item in report["checkpoints"].values())
    compact = [json.loads(line) for line in (base / "results/episodes.jsonl").read_text().splitlines()]
    assert all(row["pair_fields_status"] == "derived_after_both_hash_bound_directions_exist" for row in compact)
    assert all(row["endpoint_shift"] == pytest.approx(-0.24) for row in compact)
    assert all(row["action_distinct"] is False for row in compact)
    assert len((base / "results/pairs.jsonl").read_text().splitlines()) == 1
    assert (base / "results/discovery_only.jsonl").read_text() == ""
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
        and row["symmetry_level_s"] == 1.0
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
            candidate_sha256_by_arena={
                "droid_robolab": sha256_file(SOURCE / "layout/candidate.json"),
                "robotwin": sha256_file(SOURCE / "layout/fastwam_robotwin_candidate.json"),
            },
        )


def test_pre_r002_droid_s0_is_retained_discovery_only(tmp_path: Path):
    queue = next(
        row
        for row in queue_rows()
        if row["model_id"] == "cosmos3_edge_policy_droid"
        and row["environment_seed"] == 9400
        and row["symmetry_level_s"] == 0.0
        and row["relation"] == "left"
    )
    raw_root = tmp_path / "raw"
    write_raw(raw_root / "raw_episode.jsonl", raw_row(queue, success=True))
    registered = {row["cell_id"]: row for row in queue_rows()}
    episodes, ledger, duplicates, discovery = load_valid_episodes(
        [raw_root],
        queue_rows=registered,
        registration_sha256=sha256_file(SOURCE / "registration.json"),
        queue_sha256=sha256_file(SOURCE / "queue.jsonl"),
        candidate_sha256_by_arena={
            "droid_robolab": sha256_file(SOURCE / "layout/candidate.json"),
            "robotwin": sha256_file(SOURCE / "layout/fastwam_robotwin_candidate.json"),
        },
    )
    assert episodes == []
    assert duplicates == []
    assert len(ledger) == len(discovery) == 1
    assert discovery[0]["behavioral_denominator_included"] is False
    assert discovery[0]["reason"] == "pre_r002_s0_missing_prospective_attestation"


def test_pre_r001_droid_non_s0_is_retained_discovery_only(tmp_path: Path):
    queue = next(
        row
        for row in queue_rows()
        if row["model_id"] == "dreamzero_droid_action_cfg"
        and row["environment_seed"] == 9400
        and row["symmetry_level_s"] == 1.0
        and row["relation"] == "left"
    )
    old_row = raw_row(queue, success=True)
    for field in (
        "request0_pair_identity_sha256",
        "request0_observation_payload_sha256",
        "request0_reset_contract_sha256",
        "request0_replay_mode",
    ):
        old_row.pop(field)
    raw_root = tmp_path / "raw"
    write_raw(raw_root / "raw_episode.jsonl", old_row)
    registered = {row["cell_id"]: row for row in queue_rows()}
    episodes, ledger, duplicates, discovery = load_valid_episodes(
        [raw_root],
        queue_rows=registered,
        registration_sha256=sha256_file(SOURCE / "registration.json"),
        queue_sha256=sha256_file(SOURCE / "queue.jsonl"),
        candidate_sha256_by_arena={
            "droid_robolab": sha256_file(SOURCE / "layout/candidate.json"),
            "robotwin": sha256_file(SOURCE / "layout/fastwam_robotwin_candidate.json"),
        },
    )
    assert episodes == []
    assert duplicates == []
    assert len(ledger) == len(discovery) == 1
    assert discovery[0]["reason"] == "pre_r001_missing_request0_pair_identity"


def test_setup_invalid_attempt_is_retained_outside_behavioral_denominator(tmp_path: Path):
    raw_root = tmp_path / "raw"
    setup_invalid = raw_root / "gate/setup_invalid_zero_request.json"
    setup_invalid.parent.mkdir(parents=True)
    setup_invalid.write_text(
        json.dumps(
            {
                "schema_version": "vla-wam-shared-v3e004-setup-invalid-v1",
                "status": "setup_invalid_zero_request",
                "model_request_count": 0,
                "behavioral_episode_count": 0,
            }
        )
        + "\n"
    )
    rows = load_infrastructure_invalid([raw_root])
    assert len(rows) == 1
    assert rows[0]["behavioral_denominator_included"] is False
    assert rows[0]["attempt"]["status"] == "setup_invalid_zero_request"


def test_empty_raw_invalid_marker_is_retained_but_derived_ledgers_are_not(tmp_path: Path):
    raw_root = tmp_path / "raw"
    empty_raw = raw_root / "gate/infrastructure_invalid.json"
    empty_raw.parent.mkdir(parents=True)
    empty_raw.write_bytes(b"")
    derived = raw_root / "old_compile/results/infrastructure_invalid.jsonl"
    derived.parent.mkdir(parents=True)
    derived.write_bytes(b"")
    (derived.parent / "results.json").write_text("{}\n", encoding="utf-8")
    rows = load_infrastructure_invalid([raw_root])
    assert len(rows) == 1
    assert rows[0]["source"]["path"] == str(empty_raw.resolve())
    assert rows[0]["source"]["bytes"] == 0
    assert rows[0]["attempt"]["status"] == "empty_infrastructure_invalid_marker"


def test_bridge_failure_is_retained_outside_behavioral_denominator(tmp_path: Path):
    raw_root = tmp_path / "raw"
    bridge_failure = raw_root / "cell/bridge_failure.json"
    bridge_failure.parent.mkdir(parents=True)
    bridge_failure.write_text(
        json.dumps(
            {
                "schema_version": "vla-wam-shared-v3e004-bridge-failure-v1",
                "record_type": "infrastructure_invalid_attempt",
                "model_request_count": 0,
                "behavioral_episode_count": 0,
            }
        )
        + "\n"
    )
    rows = load_infrastructure_invalid([raw_root])
    assert len(rows) == 1
    assert rows[0]["behavioral_denominator_included"] is False
    assert rows[0]["attempt"]["record_type"] == "infrastructure_invalid_attempt"


def test_fastwam_rows_use_the_registered_robotwin_candidate(tmp_path: Path):
    rows = [
        row
        for row in queue_rows()
        if row["model_id"] == "fastwam_robotwin"
        and row["environment_seed"] == 9400
        and row["symmetry_level_s"] == 0.0
    ]
    assert {row["layout_candidate_sha256"] for row in rows} == {
        sha256_file(SOURCE / "layout/fastwam_robotwin_candidate.json")
    }
    raw_root = tmp_path / "raw"
    for row in rows:
        write_raw(raw_root / row["relation"] / "e004_episode.json", raw_row(row, success=False))
    base = prepare_base(tmp_path)
    report = compile_outputs(
        registration_path=base / "registration.json",
        queue_path=base / "queue.jsonl",
        raw_roots=[raw_root],
        output_root=base,
        resamples=10_000,
        require_complete=False,
    )
    assert report["valid_behavioral_episodes"] == 2
    compact = [json.loads(line) for line in (base / "results/episodes.jsonl").read_text().splitlines()]
    assert {row["arena"] for row in compact} == {"robotwin"}
    assert {row["candidate_sha256"] for row in compact} == {
        sha256_file(SOURCE / "layout/fastwam_robotwin_candidate.json")
    }


def test_realised_pose_tamper_fails_closed(tmp_path: Path):
    queue = next(
        row
        for row in queue_rows()
        if row["model_id"] == "cosmos3_edge_policy_droid"
        and row["environment_seed"] == 9400
        and row["symmetry_level_s"] == 1.0
        and row["relation"] == "left"
    )
    tampered = raw_row(queue, success=True)
    tampered["realised_object_poses"]["rubiks_cube"]["y_m"] += 0.01
    raw_root = tmp_path / "raw"
    write_raw(raw_root / "raw_episode.jsonl", tampered)
    registered = {row["cell_id"]: row for row in queue_rows()}
    with pytest.raises(CompileError, match="realised position differs from requested s"):
        load_valid_episodes(
            [raw_root],
            queue_rows=registered,
            registration_sha256=sha256_file(SOURCE / "registration.json"),
            queue_sha256=sha256_file(SOURCE / "queue.jsonl"),
            candidate_sha256_by_arena={
                "droid_robolab": sha256_file(SOURCE / "layout/candidate.json"),
                "robotwin": sha256_file(SOURCE / "layout/fastwam_robotwin_candidate.json"),
            },
        )


def test_claim_gate_never_calls_underpowered_null_equivalent():
    analysis = {
        "levels": {"1.00": {"endpoint_redirection_LEFT_minus_RIGHT_m": {"bootstrap_mean95": {"low": 0.01}}}},
        "equivalence_at_s1": {
            "binary_gap": {
                "power_status": "underpowered_no_equivalence_claim",
                "equivalent": True,
                "margin": 0.2,
                "registered_power_and_control_audit": {"achieved_mde_within_strict_half_margin_gate": False},
            },
            "depth_gap_m": {
                "power_status": "strictly_powered_at_endpoints",
                "equivalent": True,
                "margin": 0.05,
                "registered_power_and_control_audit": {"achieved_mde_within_strict_half_margin_gate": True},
            },
        },
    }
    gate = _claim_gate(analysis, globally_complete=True)
    assert gate["equivalence_claims"]["binary_gap"]["publication_equivalence_claim_allowed"] is False
    assert gate["equivalence_claims"]["depth_gap_m"]["publication_equivalence_claim_allowed"] is True


def test_power_audit_reports_achieved_mde_and_control_comparison():
    registration = json.loads((SOURCE / "registration.json").read_text())
    analysis = {
        "levels": {
            "1.00": {
                "pairs": 341,
                "binary_gap_R_minus_L": {"mean": 0.2},
                "requested_depth_gap_R_minus_L_m": {"mean": 0.03},
            }
        },
        "equivalence_at_s1": {
            "binary_gap": {"margin": 0.1555555556},
            "depth_gap_m": {"margin": 0.0414940332},
        },
    }
    _attach_power_audit(analysis, registration=registration, model_id="pi05_current_stack_droid")
    binary = analysis["equivalence_at_s1"]["binary_gap"]["registered_power_and_control_audit"]
    assert binary["valid_s1_pairs"] == 341
    assert binary["registered_control_effect"] == pytest.approx(0.7777777778)
    assert binary["s1_minus_registered_control_effect"] == pytest.approx(0.2 - 0.7777777778)
    assert binary["achieved_design_mde80_at_valid_s1_n"] == pytest.approx(
        2.48647 * 0.5773502692 / np.sqrt(341)
    )
    assert binary["achieved_mde_within_strict_half_margin_gate"] is True


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


def test_geometry_summary_reports_every_registered_check_and_reset_pose():
    rows = [
        {
            "cell_id": "left",
            "symmetry_level_s": 1.0,
            "asymmetry_metric_A": 0.01,
            "position_residual": 0.0004,
            "orientation_residual": np.deg2rad(0.2),
            "midline_residual": 0.0003,
            "occlusion_check": {"base_camera": False, "left_wrist_camera": False},
            "arm_reset_pose": {"arm_joint_positions_rad": [0.0] * 7, "gripper_position": [0.0]},
        },
        {
            "cell_id": "right",
            "symmetry_level_s": 1.0,
            "asymmetry_metric_A": 0.02,
            "position_residual": 0.0005,
            "orientation_residual": np.deg2rad(0.3),
            "midline_residual": 0.0002,
            "occlusion_check": {"base_camera": False, "left_wrist_camera": False},
            "arm_reset_pose": {"arm_joint_positions_rad": [0.0] * 7, "gripper_position": [0.0]},
        },
    ]
    result = _geometry_summary(rows)
    assert result["four_registered_layout_quality_checks"] == [
        "position_residual_m",
        "orientation_residual_rad",
        "midline_residual_m",
        "occlusion_check_by_camera",
    ]
    assert result["levels"]["1.00"]["position_residual_m"]["maximum"] == pytest.approx(0.0005)
    assert result["levels"]["1.00"]["orientation_residual_deg_maximum"] == pytest.approx(0.3)
    assert result["levels"]["1.00"]["occlusion_check"]["camera_checks"] == 4
    assert result["levels"]["1.00"]["occlusion_check"]["occluded_camera_checks"] == 0
    assert result["s1_gate"]["all_observed_s1_rows_pass"] is True
    assert result["arm_reset_pose_identity_count"] == 1
    assert result["arm_reset_pose_identities"][0]["episodes"] == 2


def test_geometry_summary_does_not_vacuously_pass_without_s1_rows():
    row = {
        "cell_id": "left",
        "symmetry_level_s": 0.0,
        "asymmetry_metric_A": 3.0,
        "position_residual": 0.1,
        "orientation_residual": 0.2,
        "midline_residual": 0.1,
        "occlusion_check": {"base_camera": False},
        "arm_reset_pose": {"status": "available", "robots": {}},
    }
    result = _geometry_summary([row])
    assert result["s1_gate"]["status"] == "unavailable_no_valid_s1_episodes"
    assert result["s1_gate"]["all_observed_s1_rows_pass"] is None


def test_reset_pose_memo_summary_is_readable_without_dumping_long_vectors():
    droid = _reset_pose_memo_summary(
        {"arm_joint_positions_rad": [0.0, -0.6, 0.0, -2.5, 0.0, 1.9, 0.0], "gripper_position": [0.0]}
    )
    assert "7 joints" in droid and "-2.5000" in droid and "gripper" in droid
    robotwin = _reset_pose_memo_summary(
        {"robots": {"robot.left": {"joint_positions_rad": [0.0] * 38}}}
    )
    assert "38 joints" in robotwin
    assert "exact vectors retained in results.json" in robotwin
    assert robotwin.count("0.0000") < 10


def test_geometry_figure_reports_residuals_occlusion_and_reset_identity(tmp_path: Path):
    summary = _geometry_summary(
        [
            {
                "cell_id": "left",
                "symmetry_level_s": 1.0,
                "asymmetry_metric_A": 0.01,
                "position_residual": 0.0004,
                "orientation_residual": np.deg2rad(0.2),
                "midline_residual": 0.0003,
                "occlusion_check": {"base_camera": False, "left_wrist_camera": False},
                "arm_reset_pose": {"arm_joint_positions_rad": [0.0] * 7, "gripper_position": [0.0]},
            }
        ]
    )
    report = {
        "publication_claim_status": "synthetic_test_only",
        "checkpoints": {
            "pi05_current_stack_droid": {
                "model_id": "pi05_current_stack_droid",
                "geometry_quality": summary,
            }
        },
    }
    records = render_geometry_quality(report, tmp_path)
    assert len(records) == 2
    assert all(Path(record["path"]).stat().st_size > 0 for record in records)
    assert "object-layout gate" in records[0]["caption"]
    assert "embodiment symmetry" in records[0]["caption"]


def test_failure_figure_separates_directions_and_names_exact_prompts(tmp_path: Path):
    taxonomy = {
        "left": {"correct": 1, "pick_failed": 1, "transport_failed": 0, "wrong_side": 0, "release_failed": 0},
        "right": {"correct": 0, "pick_failed": 0, "transport_failed": 1, "wrong_side": 1, "release_failed": 0},
    }
    report = {
        "publication_claim_status": "synthetic_test_only",
        "checkpoints": {
            "fastwam_robotwin": {
                "analysis": {"levels": {"0.00": {"pairs": 2, "failure_taxonomy": taxonomy}}}
            }
        },
    }
    records = render_failures(report, tmp_path)
    assert len(records) == 2
    assert all(Path(record["path"]).stat().st_size > 0 for record in records)
    assert "Put the Rubik's cube to the left of the bowl." in records[0]["caption"]
    assert "Put the Rubik's cube to the right of the bowl." in records[0]["caption"]
