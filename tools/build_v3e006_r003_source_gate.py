#!/usr/bin/env python3
"""Create the one-shot prospective V3-E006-R003 source-push gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


BRANCH = "experiment/v3e006-r003-state-repair"
FILES = (
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r003/gates/candidate_schedule.json",
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r003/infrastructure_attempts.json",
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r003/repair_registration.json",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r003/__init__.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r003/predecessor_contract.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r003/state_repair_gate.py",
    "tests/test_v3e006_r003.py",
    "tools/build_v3e006_r003_freeze.py",
    "tools/build_v3e006_r003_source_gate.py",
    "tools/run_v3e006_r003_state_repair.py",
    "tools/validate_v3e006_r003.py",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.study_root.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite source gate: {output}")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True):
        parser.error("source-gate checkout must be clean")
    remote = subprocess.check_output(
        ["git", "-C", str(root), "ls-remote", "origin", f"refs/heads/{BRANCH}"], text=True
    ).strip().split()
    if len(remote) != 2 or remote[0] != head:
        parser.error("remote R003 branch is not equal to implementation checkout")
    inventory = [bind(root, relative) for relative in FILES]
    registration = root / FILES[2]
    schedule = root / FILES[0]
    predecessor_gate = root / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r003/source_push_gate.json"
    infrastructure_attempts = root / FILES[1]
    if not predecessor_gate.is_file():
        parser.error("immutable R003 source-push gate v1 is missing")
    ledger = json.loads(infrastructure_attempts.read_text(encoding="utf-8"))
    if (
        ledger.get("attempt_count") != 1
        or ledger.get("model_request_count") != 0
        or ledger.get("behavioral_episode_count") != 0
        or ledger.get("state_candidate_count") != 0
    ):
        parser.error("R003 invalid-attempt ledger differs")
    result = {
        "schema_version": "vla-wam-shared-v3e006-r003-source-push-gate-v2",
        "repair_amendment_id": "V3-E006-R003",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "passed_before_first_r003_live_diagnostic_candidate_or_model_request",
        "implementation_commit": head,
        "remote_equality": {"remote": "origin", "ref": f"refs/heads/{BRANCH}", "commit": remote[0]},
        "required_repository_base": "18a2bf0200183647291cc7aeb1fe89997b3fb82f",
        "r002_exhaustion_closure_commit": "27d1bfd844808f7f336bbb4e25552a9c859fd08a",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "r003_live_diagnostic_count": 0,
        "r003_live_candidate_evaluation_count": 0,
        "completed_candidate_pair_count": 0,
        "accepted_state_candidate_count": 0,
        "infrastructure_invalid_search_attempt_count": 1,
        "infrastructure_attempts": bind(root, FILES[1]),
        "supersedes_source_push_gate_v1": bind(root, str(predecessor_gate.relative_to(root))),
        "supersession_reason": "The first cleared launch failed before AppLauncher because its loader queried a field absent from the immutable compact R002 closure. V2 binds the retained zero-count failure and the exact compact-schema validation repair; no registered solver, schedule, target, threshold, or scientific gate changed.",
        "repair_registration": bind(root, FILES[2]),
        "candidate_schedule": bind(root, FILES[0]),
        "implementation_files": inventory,
        "implementation_file_count": len(inventory),
        "source_push_assertions": {
            "registration_and_schedule_predate_live_candidate": True,
            "remote_ref_equals_implementation_commit": True,
            "original_r001_and_r002_predecessors_preserved": True,
            "known_reachable_diagnostic_frozen_before_live_evaluation": True,
            "finite_rank_order_and_first_pass_frozen": True,
            "unchanged_scientific_gate_sources_bound": True,
            "behavior_and_model_release_prohibited": True
        },
        "release_boundary": "Four frozen zero-model reachability diagnostics followed conditionally by one finite sequential zero-model R003 candidate search; behavior and inference remain blocked.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size, "sha256": sha256(output),
                      "implementation_commit": head}, sort_keys=True))


if __name__ == "__main__":
    main()
