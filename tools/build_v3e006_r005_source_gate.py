#!/usr/bin/env python3
"""Create the prospective V3-E006-R005 source-push gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


BRANCH = "experiment/v3e006-r005-state-repair"
BASE = "18a2bf0200183647291cc7aeb1fe89997b3fb82f"
R004_CLOSURE_COMMIT = "4775965d721f2c1e8c875bcec566d7436162cb91"
FILES = (
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r005/gates/candidate_schedule.json",
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r005/repair_registration.json",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r005/__init__.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r005/predecessor_contract.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r005/residual_correction.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r005/source_gate_contract.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r005/state_repair_gate.py",
    "tests/test_v3e006_r005.py",
    "tools/build_v3e006_r005_freeze.py",
    "tools/build_v3e006_r005_source_gate.py",
    "tools/close_v3e006_r005.py",
    "tools/run_v3e006_r005_state_repair.py",
    "tools/validate_v3e006_r005.py",
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r005/infrastructure_attempts.jsonl",
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r005/source_push_gate.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def commit_exists(root: Path, commit: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.study_root.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite source gate: {output}")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ):
        parser.error("source-gate checkout must be clean")
    remote = subprocess.check_output(
        ["git", "-C", str(root), "ls-remote", "origin", f"refs/heads/{BRANCH}"],
        text=True,
    ).strip().split()
    if len(remote) != 2 or remote[0] != head:
        parser.error("remote R005 branch is not equal to implementation checkout")
    for commit in (BASE, R004_CLOSURE_COMMIT):
        if not commit_exists(root, commit):
            parser.error(f"required source-lineage commit is absent: {commit}")

    registration = root / FILES[1]
    schedule = root / FILES[0]
    registration_payload = json.loads(registration.read_text(encoding="utf-8"))
    schedule_payload = json.loads(schedule.read_text(encoding="utf-8"))
    if registration_payload.get("counts_at_registration") != {
        "r005_live_diagnostics": 0,
        "r005_live_candidate_evaluations": 0,
        "model_requests": 0,
        "behavioral_episodes": 0,
    }:
        parser.error("R005 registration counts differ")
    if any(
        schedule_payload.get(key) != 0
        for key in (
            "r005_live_diagnostic_count",
            "r005_live_candidate_evaluation_count",
            "model_request_count",
            "behavioral_episode_count",
        )
    ):
        parser.error("R005 schedule counts differ")

    ledger_path = root / FILES[-2]
    ledger_lines = [line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    if len(ledger_lines) != 1:
        parser.error("R005 retry requires exactly one infrastructure-attempt ledger row")
    attempt = json.loads(ledger_lines[0])
    if (
        attempt.get("status")
        != "infrastructure_invalid_after_complete_compute_terminal_serialization_failure"
        or attempt.get("model_request_count") != 0
        or attempt.get("behavioral_episode_count") != 0
        or attempt.get("state_candidate_count") != 0
        or attempt.get("completeness", {}).get("candidate_pair_compute_count") != 4
        or attempt.get("completeness", {}).get("scientifically_completed_candidate_pair_count") != 0
    ):
        parser.error("R005 retained infrastructure attempt differs")
    v1_path = root / FILES[-1]
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    if (
        v1.get("schema_version") != "vla-wam-shared-v3e006-r005-source-push-gate-v1"
        or v1.get("status") != "passed_before_first_r005_live_diagnostic_candidate_or_model_request"
        or any(
            v1.get(key) != 0
            for key in (
                "model_request_count", "behavioral_episode_count", "r005_live_diagnostic_count",
                "r005_live_candidate_evaluation_count", "completed_candidate_pair_count",
                "accepted_state_candidate_count", "infrastructure_invalid_search_attempt_count",
            )
        )
    ):
        parser.error("R005 v1 source-push gate differs")

    inventory = [bind(root, relative) for relative in FILES]
    result = {
        "schema_version": "vla-wam-shared-v3e006-r005-source-push-gate-v2",
        "repair_amendment_id": "V3-E006-R005",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "passed_before_identical_r005_retry_after_terminal_serialization_fix",
        "implementation_commit": head,
        "remote_equality": {
            "remote": "origin",
            "ref": f"refs/heads/{BRANCH}",
            "commit": remote[0],
        },
        "required_repository_base": BASE,
        "r004_timeout_closure_commit": R004_CLOSURE_COMMIT,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "r005_live_diagnostic_count": 4,
        "r005_live_candidate_evaluation_count": 4,
        "completed_candidate_pair_count": 0,
        "raw_candidate_pair_compute_count": 4,
        "accepted_state_candidate_count": 0,
        "infrastructure_invalid_search_attempt_count": 1,
        "retry_scope": "identical_zero_model_search_after_reporting_only_fix",
        "superseded_source_push_gate": bind(root, FILES[-1]),
        "infrastructure_attempts": bind(root, FILES[-2]),
        "repair_registration": bind(root, FILES[1]),
        "candidate_schedule": bind(root, FILES[0]),
        "implementation_files": inventory,
        "implementation_file_count": len(inventory),
        "source_push_assertions": {
            "registration_and_schedule_predate_first_r005_live_diagnostic": True,
            "remote_ref_equals_implementation_commit": True,
            "original_r001_r002_r003_and_r004_predecessors_preserved": True,
            "r004_targets_correction_and_candidate_order_unchanged": True,
            "construction_timeout_only_changed_from_450_to_900_before_any_step": True,
            "behavioral_horizon_unchanged": True,
            "unchanged_scientific_gate_sources_bound": True,
            "behavior_and_model_release_prohibited": True,
            "attempt01_raw_evidence_retained_and_target_validated": True,
            "terminal_selection_rule_now_reads_exact_frozen_candidate_schedule": True,
            "completed_diagnostics_and_attempts_are_atomically_retained_after_each_unit": True,
            "candidate_targets_order_controller_horizon_and_scientific_gates_unchanged": True,
        },
        "release_boundary": (
            "One identical retry of the four frozen zero-model reachability diagnostics followed "
            "conditionally by the finite sequential zero-model R005 candidate search; behavior and "
            "inference remain blocked."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
                "implementation_commit": head,
                "model_request_count": 0,
                "behavioral_episode_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
