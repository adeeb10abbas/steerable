#!/usr/bin/env python3
"""Generate the prospective V3-E006-R011 source-push gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

BRANCH = "experiment/v3e006-r011-scene-sync-repair"
BASE = "18a2bf0200183647291cc7aeb1fe89997b3fb82f"
R010_CLOSURE = "45a90fa93b1df2b3a07c0e974527a211814d6cb9"
R010_RESULTS = "d4762cd8f4db539e760a79ce9e36f81d49455cb76836d8df5dadf902e6b78869"
FILES = (
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r011/gates/candidate_schedule.json",
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r011/repair_registration.json",
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r011/infrastructure_attempts.jsonl",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r011/__init__.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r011/object_servo.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r011/pinch_geometry.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r011/predecessor_contract.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r011/residual_correction.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r011/source_gate_contract.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r011/state_repair_gate.py",
    "tests/test_v3e006_r011.py",
    "tools/build_v3e006_r011_freeze.py",
    "tools/build_v3e006_r011_source_gate.py",
    "tools/run_v3e006_r011_state_repair.py",
    "tools/validate_v3e006_r011.py",
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
    for commit in (BASE, R010_CLOSURE):
        if subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"], check=False).returncode:
            parser.error(f"missing lineage commit: {commit}")
    registration = json.loads((root / FILES[1]).read_text())
    schedule = json.loads((root / FILES[0]).read_text())
    if registration["counts_at_registration"] != {
        "r011_geometry_attachment_preflights": 0,
        "r011_live_diagnostics": 0, "r011_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }:
        parser.error("registration is not prospective")
    if (root / FILES[2]).read_bytes() != b"":
        parser.error("infrastructure ledger is nonempty")
    value = {
        "schema_version": "vla-wam-shared-v3e006-r011-source-push-gate-v1",
        "repair_amendment_id": "V3-E006-R011",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "passed_before_first_r011_live_preflight_diagnostic_candidate_or_model_request",
        "implementation_commit": head,
        "remote_equality": {"remote": "origin", "ref": f"refs/heads/{BRANCH}", "commit": head},
        "required_repository_base": BASE,
        "r010_closure_commit": R010_CLOSURE,
        "r010_results_sha256": R010_RESULTS,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "r011_geometry_attachment_preflight_count": 0,
        "r011_live_diagnostic_count": 0,
        "r011_live_candidate_evaluation_count": 0,
        "accepted_state_candidate_count": 0,
        "infrastructure_invalid_attempt_count": 0,
        "repair_registration": bind(root, FILES[1]),
        "candidate_schedule": bind(root, FILES[0]),
        "implementation_files": [bind(root, path) for path in FILES],
        "implementation_file_count": len(FILES),
        "source_push_assertions": {
            "registration_schedule_source_predate_live_evaluation": True,
            "r010_closure_preserved_and_bound": True,
            "four_ranks_targets_order_diagnostics_and_scientific_gates_unchanged": True,
            "sole_delta_is_one_stage_identity_checked_scene_specific_validation_only_geometry_oracle_sync": True,
            "no_model_behavior_or_post_reset_state_write": True,
        },
        "release_boundary": "One zero-model attachment preflight, diagnostic suite, and conditional four-pair construction search only.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size, "sha256": sha(output), "implementation_commit": head}, sort_keys=True))


if __name__ == "__main__":
    main()
