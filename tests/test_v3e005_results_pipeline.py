from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from tools.build_v3e005_evidence_manifest import build as build_manifest
from tools.build_v3e005_publication_decision import build as build_decision
from tools.compile_v3e005_results import (
    CompileError,
    compile_to_directory,
    exact_sign_flip_test,
    sha256,
)
from tools.render_v3e005_results import render
from tools.validate_v3e005_evidence import Invalid, validate


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005"


def queue_rows() -> list[dict]:
    return [json.loads(line) for line in (SOURCE / "queue.jsonl").read_text().splitlines() if line.strip()]


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def synthetic_row(queue: dict, cell_dir: Path, *, h4_effect: float) -> dict:
    relation = queue["relation"]
    level = float(queue["symmetry_level_s"])
    seed = int(queue["environment_seed"])
    # H4 is always derived from the matched signed endpoints.
    signed_endpoint = h4_effect if relation == "left" else 0.0
    # A strong s0 binary gap and a smaller s1 gap exercise the H1 interaction.
    if level == 0.0:
        success = relation == "right" or seed % 4 == 0
        depth = 0.04 if relation == "left" else 0.12
    else:
        success = seed % 3 != 0
        depth = 0.08 if relation == "left" else 0.09
    category = "correct" if success else ("wrong_side" if relation == "left" else "transport_failed")

    cell_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = {
        "result": "result.json",
        "trajectory": "trajectory.json",
        "simulator_viewport_video": "simulator.mp4",
        "executed_action_trace": "action_trace.npz",
        "live_reset_snapshot": "live_reset_snapshot.json",
    }
    artifacts = {}
    for name, filename in artifact_names.items():
        path = cell_dir / filename
        path.write_bytes(f"{queue['cell_id']}:{name}\n".encode())
        artifacts[name] = file_record(path)
    runtime = queue["runtime_identity_requirement"]
    return {
        "schema_version": "vla-wam-shared-v3e005-lingbot-robotwin-episode-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E005",
        "model_id": "lingbot_va_robotwin",
        "arena": "robotwin",
        "cell_id": queue["cell_id"],
        "matched_seed_id": queue["matched_seed_id"],
        "matched_layout_pair_id": queue["matched_layout_pair_id"],
        "scene_id": queue["scene_id"],
        "scene_cluster_id": queue["scene_cluster_id"],
        "anchor_task": queue["anchor_task"],
        "environment_seed": seed,
        "sampling_seed": seed,
        "relation": relation,
        "prompt": queue["prompt"],
        "prompt_sha256": queue["prompt_sha256"],
        "success": success,
        "failure_category": category,
        "signed_final_lateral_offset": signed_endpoint,
        "requested_side_depth": depth,
        "cone_entry_step": 10 if success else None,
        "cone_entry_sustained": success,
        "endpoint_shift": h4_effect,
        "action_distinct": True,
        "episode_length": 80,
        "time_to_first_contact": None,
        "grasp_step": 4 if success else None,
        "cumulative_lateral_path": 0.3,
        "peak_lateral_excursion": 0.15,
        "symmetry_level_s": level,
        "asymmetry_metric_A": 0.0 if level == 1.0 else 0.2,
        "position_residual": 0.0001,
        "orientation_residual": 0.001,
        "midline_residual": 0.0001,
        "occlusion_check": False,
        "realised_object_poses": {"target": {"x": 0.1, "y": 0.0, "z": 0.1}},
        "arm_reset_pose": {"joint_positions": [0.0] * 7},
        "mirrored_asset_identity_verified": level == 1.0,
        "mirrored_yaw_verified": level == 1.0,
        "source_artifacts": artifacts,
        "registration_sha256": sha256(SOURCE / "registration.json"),
        "queue_sha256": sha256(SOURCE / "queue.jsonl"),
        "runtime_identity_sha256": "a" * 64,
        "checkpoint_revision": runtime["checkpoint_revision"],
        "checkpoint_manifest_sha256": runtime["checkpoint_manifest_sha256"],
        "external_repository_commit": runtime["external_repository_commit"],
        "simulator_repository_commit": runtime["simulator_repository_commit"],
    }


def write_cohort(root: Path, *, h4_effect: float, rows: list[dict] | None = None) -> list[Path]:
    paths = []
    selected = rows or queue_rows()
    path_by_cell = {}
    for queue in selected:
        cell_dir = root / queue["cell_id"].replace(":", "__")
        raw_path = cell_dir / "raw_episode.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(synthetic_row(queue, cell_dir, h4_effect=h4_effect), sort_keys=True) + "\n")
        paths.append(raw_path)
        path_by_cell[queue["cell_id"]] = raw_path
    for seed in sorted({int(row["environment_seed"]) for row in selected}):
        seed_rows = [row for row in selected if int(row["environment_seed"]) == seed]
        if len(seed_rows) != 4:
            continue
        cell_ids = [row["cell_id"] for row in seed_rows]
        seed_dir = root / "seeds" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        pair_paths = []
        for level in (0.0, 1.0):
            level_rows = [row for row in seed_rows if float(row["symmetry_level_s"]) == level]
            pair = {
                "schema_version": "vla-wam-shared-v3e005-lingbot-robotwin-pair-v1",
                "study_id": "vla_wam_language_steerability_v3",
                "amendment_id": "V3-E005",
                "model_id": "lingbot_va_robotwin",
                "arena": "robotwin",
                "matched_layout_pair_id": level_rows[0]["matched_layout_pair_id"],
                "environment_seed": seed,
                "sampling_seed": seed,
                "scene_id": level_rows[0]["scene_id"],
                "symmetry_level_s": level,
                "left_cell_id": next(row["cell_id"] for row in level_rows if row["relation"] == "left"),
                "right_cell_id": next(row["cell_id"] for row in level_rows if row["relation"] == "right"),
                "endpoint_shift": h4_effect,
                "action_pair": {"actions_compared": 10, "first_10_action_rms": 0.1, "action_distinct": True},
                "initial_physical_fingerprint_sha256": "d" * 64,
            }
            pair["pair_sha256"] = hashlib.sha256(
                (json.dumps(pair, sort_keys=True, separators=(",", ":")) + "\n").encode()
            ).hexdigest()
            pair_path = seed_dir / f"pair_s{int(level * 100):03d}.json"
            pair_path.write_text(json.dumps(pair, sort_keys=True) + "\n")
            pair_paths.append(str(pair_path.resolve()))
        marker = {
            "schema_version": "vla-wam-shared-v3e005-lingbot-whole-seed-completion-v1",
            "study_id": "vla_wam_language_steerability_v3",
            "amendment_id": "V3-E005",
            "model_id": "lingbot_va_robotwin",
            "arena": "robotwin",
            "seed": seed,
            "status": "complete_four_valid_behavioral_cells",
            "behavioral_episode_count": 4,
            "matched_pair_count": 2,
            "infrastructure_failure_count": 0,
            "cell_ids": cell_ids,
            "compact_episode_paths": [str(path_by_cell[cell_id].resolve()) for cell_id in cell_ids],
            "episode_sha256": {cell_id: sha256(path_by_cell[cell_id]) for cell_id in cell_ids},
            "pair_paths": pair_paths,
            "registration_sha256": sha256(SOURCE / "registration.json"),
            "queue_sha256": sha256(SOURCE / "queue.jsonl"),
            "layout_candidate_sha256": "b" * 64,
            "model_blind_gate_sha256": "c" * 64,
            "runtime_identity_sha256": "a" * 64,
            "completed_at_utc": "2026-08-09T00:00:00Z",
        }
        marker["marker_sha256"] = hashlib.sha256(
            (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
        marker_path = seed_dir / f"seed_{seed}_manifest.json"
        marker_path.write_text(json.dumps(marker, sort_keys=True) + "\n")
    return paths


def prepare_base(tmp_path: Path) -> Path:
    base = tmp_path / "e005"
    base.mkdir()
    shutil.copy2(SOURCE / "registration.json", base / "registration.json")
    shutil.copy2(SOURCE / "queue.jsonl", base / "queue.jsonl")
    return base


def finish_bundle(base: Path, raw_root: Path) -> None:
    render(base / "results/results.json", base / "results/figures")
    build_decision(
        base / "results/results.json",
        base / "DECISION_MEMO.md",
        base / "V3E005_PUBLICATION_DECISION.md",
    )
    build_manifest(base, base / "evidence_manifest.json", raw_roots=[raw_root])


def test_complete_h4_pass_enables_h1_h3_after_gate(tmp_path: Path):
    raw_root = tmp_path / "raw"
    inputs = write_cohort(raw_root, h4_effect=0.08)
    base = prepare_base(tmp_path)
    report = compile_to_directory(inputs=inputs, output_dir=base / "results", require_complete=True)
    assert report["h4_gate"]["outcome"] == "pass"
    assert report["analysis_order"] == ["H4", "H1", "H2", "H3"]
    assert report["hypotheses"]["H1"]["status"] == "reported_after_h4_pass"
    assert report["hypotheses"]["H2"]["binary"]["publication_equivalence_claim_allowed"] is False
    render(base / "results/results.json", base / "results/figures")
    build_decision(
        base / "results/results.json",
        base / "DECISION_MEMO.md",
        base / "V3E005_PUBLICATION_DECISION.md",
    )
    with pytest.raises(FileNotFoundError, match="27 whole-seed manifests"):
        build_manifest(base, base / "evidence_manifest.json")
    build_manifest(base, base / "evidence_manifest.json", raw_roots=[raw_root])
    manifest = json.loads((base / "evidence_manifest.json").read_text())
    compact_names = {Path(item["path"]).name for item in manifest["compact_files"]}
    assert {
        "registration.json",
        "queue.jsonl",
        "results.json",
        "episodes.jsonl",
        "pairs.jsonl",
        "infrastructure_invalid.jsonl",
        "DECISION_MEMO.md",
        "V3E005_PUBLICATION_DECISION.md",
        "figure_manifest.json",
    } <= compact_names
    checked = validate(base, require_complete=True, verify_raw_sources=True)
    assert checked == {
        "status": "valid_complete",
        "valid_behavioral_episodes": 108,
        "complete_matched_pairs": 54,
        "infrastructure_invalid_attempts": 0,
        "h4_outcome": "pass",
        "raw_sources_verified": True,
    }
    figures = json.loads((base / "results/figures/figure_manifest.json").read_text())
    assert figures["h1_h3_rendered"] is True
    assert len(figures["figures"]) == 6


def test_complete_h4_fail_withholds_all_downstream_estimands_and_figures(tmp_path: Path):
    raw_root = tmp_path / "raw"
    inputs = write_cohort(raw_root, h4_effect=0.02)
    base = prepare_base(tmp_path)
    report = compile_to_directory(inputs=inputs, output_dir=base / "results", require_complete=True)
    assert report["h4_gate"]["outcome"] == "fail"
    assert report["analysis_order"] == ["H4"]
    assert report["hypotheses"] == {
        name: {"status": "withheld_due_h4_failure", "estimands_reported": False}
        for name in ("H1", "H2", "H3")
    }
    finish_bundle(base, raw_root)
    checked = validate(base, require_complete=True, verify_raw_sources=False)
    assert checked["h4_outcome"] == "fail"
    figures = json.loads((base / "results/figures/figure_manifest.json").read_text())
    assert figures["h1_h3_rendered"] is False
    assert len(figures["figures"]) == 2
    assert "withhold" in (base / "V3E005_PUBLICATION_DECISION.md").read_text().lower()


def test_partial_progress_has_no_hypothesis_estimates(tmp_path: Path):
    selected = queue_rows()[:2]
    raw_root = tmp_path / "raw"
    inputs = write_cohort(raw_root, h4_effect=0.08, rows=selected)
    base = prepare_base(tmp_path)
    report = compile_to_directory(inputs=inputs, output_dir=base / "results", require_complete=False)
    assert report["status"] == "partial_progress_no_publication_claims"
    assert report["h4_gate"]["outcome"] == "not_evaluable_incomplete"
    assert report["analysis_order"] == ["H4"]
    assert all(item["estimands_reported"] is False for item in report["hypotheses"].values())
    finish_bundle(base, raw_root)
    assert validate(base, require_complete=False, verify_raw_sources=True)["status"] == "valid_partial_no_publication_claims"
    with pytest.raises(Invalid, match="incomplete"):
        validate(base, require_complete=True, verify_raw_sources=False)


def test_raw_root_discovery_fails_closed_on_conflicting_cell(tmp_path: Path):
    queue = queue_rows()[0]
    first = write_cohort(tmp_path / "raw/a", h4_effect=0.08, rows=[queue])[0]
    second = write_cohort(tmp_path / "raw/b", h4_effect=0.08, rows=[queue])[0]
    with pytest.raises(CompileError, match="ambiguous duplicate behavioral evidence"):
        compile_to_directory(inputs=[first, second], output_dir=tmp_path / "results")


def test_droid_row_is_rejected_without_pooling(tmp_path: Path):
    queue = queue_rows()[0]
    path = write_cohort(tmp_path / "raw", h4_effect=0.08, rows=[queue])[0]
    row = json.loads(path.read_text())
    row["arena"] = "droid_robolab"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(CompileError, match="non-RoboTwin"):
        compile_to_directory(inputs=[path], output_dir=tmp_path / "results")


def test_registered_resample_floor_is_enforced_before_analysis(tmp_path: Path):
    queue = queue_rows()[0]
    path = write_cohort(tmp_path / "raw", h4_effect=0.08, rows=[queue])[0]
    with pytest.raises(CompileError, match="at least 20,000"):
        compile_to_directory(inputs=[path], output_dir=tmp_path / "results", resamples=19_999)


def test_exact_sign_flip_known_symmetric_case():
    assert exact_sign_flip_test([1.0, -1.0])["exact_two_sided_p"] == 1.0
    one_sided = exact_sign_flip_test([1.0, 1.0, 1.0])
    assert one_sided["permutations"] == 8
    assert one_sided["exact_two_sided_p"] == pytest.approx(0.25)
