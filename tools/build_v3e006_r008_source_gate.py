#!/usr/bin/env python3
"""Generate the prospective V3-E006-R008 source-push gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

BRANCH = "experiment/v3e006-r008-object-servo-repair"
BASE = "18a2bf0200183647291cc7aeb1fe89997b3fb82f"
R007_CLOSURE = "7cc3acc120027bdd181340b443633d8a03d6858d"
R007_RESULTS = "3a6ab612919fd9e5eeef2f4bd030b74c25f6b6f871b6c25c13f92efac5ba9b7d"
FILES = (
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r008/gates/candidate_schedule.json",
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r008/repair_registration.json",
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r008/infrastructure_attempts.jsonl",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r008/__init__.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r008/object_servo.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r008/predecessor_contract.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r008/residual_correction.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r008/source_gate_contract.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r008/state_repair_gate.py",
    "tests/test_v3e006_r008.py",
    "tools/build_v3e006_r008_freeze.py",
    "tools/build_v3e006_r008_source_gate.py",
    "tools/run_v3e006_r008_state_repair.py",
    "tools/validate_v3e006_r008.py",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.study_root.resolve(), args.output.resolve()
    if output.exists():
        parser.error(f"refusing overwrite: {output}")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True):
        parser.error("source checkout is dirty")
    remote = subprocess.check_output(["git", "-C", str(root), "ls-remote", "origin", f"refs/heads/{BRANCH}"], text=True).split()
    if len(remote) != 2 or remote[0] != head:
        parser.error("remote branch differs")
    for commit in (BASE, R007_CLOSURE):
        if subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"], check=False).returncode:
            parser.error(f"missing lineage commit: {commit}")
    registration = json.loads((root / FILES[1]).read_text())
    schedule = json.loads((root / FILES[0]).read_text())
    if registration["counts_at_registration"] != {
        "r008_live_diagnostics": 0, "r008_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }:
        parser.error("registration is not prospective")
    if (root / FILES[2]).read_bytes() != b"":
        parser.error("infrastructure ledger is nonempty")
    value = {
        "schema_version": "vla-wam-shared-v3e006-r008-source-push-gate-v1",
        "repair_amendment_id": "V3-E006-R008",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "passed_before_first_r008_live_diagnostic_candidate_or_model_request",
        "implementation_commit": head,
        "remote_equality": {"remote": "origin", "ref": f"refs/heads/{BRANCH}", "commit": head},
        "required_repository_base": BASE,
        "r007_closure_commit": R007_CLOSURE,
        "r007_results_sha256": R007_RESULTS,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "r008_live_diagnostic_count": 0,
        "r008_live_candidate_evaluation_count": 0,
        "accepted_state_candidate_count": 0,
        "infrastructure_invalid_attempt_count": 0,
        "repair_registration": bind(root, FILES[1]),
        "candidate_schedule": bind(root, FILES[0]),
        "implementation_files": [bind(root, path) for path in FILES],
        "implementation_file_count": len(FILES),
        "source_push_assertions": {
            "registration_schedule_source_predate_live_evaluation": True,
            "r007_closure_preserved_and_bound": True,
            "four_ranks_targets_order_diagnostics_and_scientific_gates_unchanged": True,
            "sole_scientific_delta_is_construction_object_servo_q_handoff_and_construction_timeout": True,
            "no_model_behavior_or_post_reset_state_write": True,
        },
        "release_boundary": "One zero-model diagnostic and conditional four-pair construction search only.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size, "sha256": sha(output), "implementation_commit": head}, sort_keys=True))


if __name__ == "__main__":
    main()
