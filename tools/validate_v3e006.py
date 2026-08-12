#!/usr/bin/env python3
"""Validate the hash-closed V3-E006 pre-registration stop package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def verify_binding(binding: Mapping[str, Any], *, local: bool, raw: bool) -> None:
    require(set(("path", "bytes", "sha256")) <= set(binding), "incomplete binding")
    path = (ROOT / str(binding["path"])).resolve() if local else Path(str(binding["path"]))
    if local or raw:
        require(path.is_file(), f"bound file missing: {path}")
        require(path.stat().st_size == binding["bytes"], f"bound bytes differ: {path}")
        require(sha256(path) == binding["sha256"], f"bound digest differs: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-raw", action="store_true")
    args = parser.parse_args()

    results = json.loads((BASE / "results/results.json").read_text(encoding="utf-8"))
    release = json.loads((BASE / "release_gate.json").read_text(encoding="utf-8"))
    manifest = json.loads((BASE / "results/evidence_manifest.json").read_text(encoding="utf-8"))
    require(results["schema_version"] == "vla-wam-shared-v3e006-gate-stop-results-v1", "results schema differs")
    require(results["status"] == "gate_failed_no_valid_candidate_stop_before_registration", "stop status differs")
    require(release["status"] == results["status"], "release/results status differs")
    for payload in (results, release):
        require(payload["model_request_count"] == 0, "nonzero model request count")
        require(payload["behavioral_episode_count"] == 0, "nonzero behavioral count")
        require(payload["state_candidate_count"] == 0, "nonzero accepted candidate count")
    require(results["registration_created"] is False and release["registration_created"] is False, "registration unexpectedly exists")
    require(release["queue_created"] is False and release["release_for_inference"] is False, "inference release unexpectedly exists")
    require(not (BASE / "registration.json").exists(), "registration must be absent after pre-registration stop")
    require(not (BASE / "queue.jsonl").exists(), "queue must be absent after pre-registration stop")
    require((BASE / "results/episodes.jsonl").stat().st_size == 0, "behavioral episode stream is nonempty")
    require((BASE / "results/pairs_or_blocks.jsonl").stat().st_size == 0, "block stream is nonempty")

    grasp = results["canonical_grasp_gate"]
    require(grasp["passed"] is False, "invalid canonical grasp marked passed")
    require(grasp["camera"]["passed"] is True, "camera outcome differs")
    require(grasp["physics"]["passed"] is False, "physics outcome differs")
    require(grasp["ood"]["passed"] is False, "OOD outcome differs")
    require(grasp["companion_pose"]["passed"] is False, "companion outcome differs")
    observed = grasp["physics"]["observed"]
    thresholds = grasp["physics"]["thresholds"]
    require(observed["max_arm_joint_speed_rad_s"] >= thresholds["arm_joint_speed_rad_s_strict"], "arm-speed failure differs")
    require(observed["max_cube_angular_speed_rad_s"] >= thresholds["cube_angular_speed_rad_s_strict"], "angular-speed failure differs")
    require(observed["max_cube_linear_speed_m_s"] >= thresholds["cube_linear_speed_m_s_strict"], "linear-speed failure differs")
    require(observed["max_cube_midline_residual_m"] >= thresholds["cube_midline_residual_m_strict"], "midline failure differs")
    require(observed["rubiks_cube_table_contact_force_n"] > thresholds["unintended_contact_force_n_inclusive"], "contact failure differs")
    require(observed["object_grabbed_all_steps"] is False, "grasp failure differs")
    require(grasp["ood"]["normalized_distance"] > grasp["ood"]["maximum_distance_inclusive"], "OOD failure differs")
    companion = grasp["companion_pose"]
    require(companion["observed"]["bowl_position_error_m"] >= companion["position_tolerance_m_strict"], "bowl position failure differs")
    require(companion["observed"]["bowl_orientation_error_rad"] >= companion["orientation_tolerance_rad_strict"], "bowl orientation failure differs")
    require(all(results["full_reset_gate"].values()), "full-reset gate did not pass")

    candidate_rows = [
        json.loads(line)
        for line in (BASE / "gates/model_blind_candidate_infrastructure_invalid.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(candidate_rows) == 6, "candidate attempt ledger count differs")
    final = candidate_rows[-1]
    require(final["status"] == "retained_gate_failed_no_valid_candidate_stop_before_registration", "final attempt status differs")
    for key in ("model_request_count", "behavioral_episode_count", "state_candidate_count"):
        require(final[key] == 0, f"final attempt has nonzero {key}")
    require(final["behavioral_denominator_included"] is False and final["candidate_gate_passed"] is False, "final attempt entered denominator")
    for binding in (final["construction_source"], final["invocation"]["exact_argv_text"], *final["raw_sources"].values()):
        verify_binding(binding, local=False, raw=args.verify_raw)
    require(final["raw_sources"]["failure_report"] == results["final_candidate_attempt"]["failure_report"], "failure binding differs")
    require(final["raw_sources"]["runtime_log"] == results["final_candidate_attempt"]["runtime_log"], "runtime binding differs")

    summary_rows = [json.loads(line) for line in (BASE / "infrastructure_attempts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    require([row["retained_attempt_count"] for row in summary_rows] == [2, 6], "infrastructure summary counts differ")
    require(all(row["model_request_count"] == row["behavioral_episode_count"] == row["candidate_count"] == 0 for row in summary_rows), "infrastructure summary has nonzero count")

    lineage = json.loads((BASE / "source_lineage.json").read_text(encoding="utf-8"))
    require(lineage["status"] == "closed_before_registration_or_model_request_after_candidate_gate_failure", "lineage status differs")
    require(lineage["commits"][0]["sha"] == "18a2bf0200183647291cc7aeb1fe89997b3fb82f", "required base differs")
    require(lineage["commits"][-1]["sha"] == "59b9fba8e5310b4c2b3cd65e242fd859c1337d0d", "final execution commit differs")
    for row in lineage["commits"]:
        completed = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{row['sha']}^{{commit}}"])
        require(completed.returncode == 0, f"lineage commit missing: {row['sha']}")

    require(manifest["status"] == results["status"], "manifest status differs")
    require(manifest["model_request_count"] == manifest["behavioral_episode_count"] == manifest["state_candidate_count"] == 0, "manifest count differs")
    for binding in manifest["local_files"]:
        verify_binding(binding, local=True, raw=False)
    for binding in manifest["raw_files"]:
        verify_binding(binding, local=False, raw=args.verify_raw)
    memo = (BASE / "results/DECISION_MEMO.md").read_text(encoding="utf-8")
    insert = (BASE / "MANUSCRIPT_INSERT.md").read_text(encoding="utf-8")
    require("gate_failed_no_valid_candidate_stop_before_registration" in memo, "memo status differs")
    require("no alternative candidate" in memo.lower() and "zero" in memo.lower(), "memo stopping/count boundary missing")
    require("No stage-localization plot is authorized" in insert, "manuscript plot boundary differs")
    require("remains unresolved" in insert, "manuscript conclusion differs")
    print(json.dumps({"status": "valid_gate_failed_no_valid_candidate", "model_request_count": 0, "behavioral_episode_count": 0, "candidate_attempts": 6, "raw_verified": args.verify_raw}, sort_keys=True))


if __name__ == "__main__":
    main()
