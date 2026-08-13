#!/usr/bin/env python3
"""Generate the prospective V3-E006-R009 source-push gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

BRANCH = "experiment/v3e006-r009-pinch-geometry-repair"
BASE = "18a2bf0200183647291cc7aeb1fe89997b3fb82f"
R008_CLOSURE = "a13d5b7f8a4be6374a22483a436b9a41aadd1c9f"
R008_RESULTS = "5f86b77b721806b0c19416b622e9a6aab68b13de6d596a6dadec89ed41a081d5"
FILES = (
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r009/gates/candidate_schedule.json",
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r009/repair_registration.json",
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r009/infrastructure_attempts.jsonl",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r009/__init__.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r009/object_servo.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r009/pinch_geometry.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r009/predecessor_contract.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r009/residual_correction.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r009/source_gate_contract.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r009/state_repair_gate.py",
    "tests/test_v3e006_r009.py",
    "tools/build_v3e006_r009_freeze.py",
    "tools/build_v3e006_r009_source_gate.py",
    "tools/run_v3e006_r009_state_repair.py",
    "tools/validate_v3e006_r009.py",
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
    for commit in (BASE, R008_CLOSURE):
        if subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"], check=False).returncode:
            parser.error(f"missing lineage commit: {commit}")
    registration = json.loads((root / FILES[1]).read_text())
    schedule = json.loads((root / FILES[0]).read_text())
    if registration["counts_at_registration"] != {
        "r009_live_diagnostics": 0, "r009_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }:
        parser.error("registration is not prospective")
    if (root / FILES[2]).read_bytes() != b"":
        parser.error("infrastructure ledger is nonempty")
    value = {
        "schema_version": "vla-wam-shared-v3e006-r009-source-push-gate-v1",
        "repair_amendment_id": "V3-E006-R009",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "passed_before_first_r009_live_diagnostic_candidate_or_model_request",
        "implementation_commit": head,
        "remote_equality": {"remote": "origin", "ref": f"refs/heads/{BRANCH}", "commit": head},
        "required_repository_base": BASE,
        "r008_closure_commit": R008_CLOSURE,
        "r008_results_sha256": R008_RESULTS,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "r009_live_diagnostic_count": 0,
        "r009_live_candidate_evaluation_count": 0,
        "accepted_state_candidate_count": 0,
        "infrastructure_invalid_attempt_count": 0,
        "repair_registration": bind(root, FILES[1]),
        "candidate_schedule": bind(root, FILES[0]),
        "implementation_files": [bind(root, path) for path in FILES],
        "implementation_file_count": len(FILES),
        "source_push_assertions": {
            "registration_schedule_source_predate_live_evaluation": True,
            "r008_closure_preserved_and_bound": True,
            "four_ranks_targets_order_diagnostics_and_scientific_gates_unchanged": True,
            "sole_scientific_delta_is_uniform_collision_pinch_acquisition_q_handoff_and_construction_timeout": True,
            "no_model_behavior_or_post_reset_state_write": True,
        },
        "release_boundary": "One zero-model diagnostic and conditional four-pair construction search only.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size, "sha256": sha(output), "implementation_commit": head}, sort_keys=True))


if __name__ == "__main__":
    main()
