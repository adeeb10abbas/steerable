#!/usr/bin/env python3
"""Create the prospective V3-E006-R006 source-push gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


BRANCH = "experiment/v3e006-r006-state-repair"
BASE = "18a2bf0200183647291cc7aeb1fe89997b3fb82f"
R005_CLOSURE_COMMIT = "040cf75c1d83a2e5f8383d87247fb096e8d2491a"
R005_RESULTS_SHA256 = "550665a234c378cbcb5c8022d16249a980d1a5b5368b08900568c959c51fb9f2"
FILES = (
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r006/gates/candidate_schedule.json",
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r006/repair_registration.json",
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r006/infrastructure_attempts.jsonl",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r006/__init__.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r006/predecessor_contract.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r006/residual_correction.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r006/source_gate_contract.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r006/state_repair_gate.py",
    "tests/test_v3e006_r006.py",
    "tools/build_v3e006_r006_freeze.py",
    "tools/build_v3e006_r006_source_gate.py",
    "tools/run_v3e006_r006_state_repair.py",
    "tools/validate_v3e006_r006.py",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.study_root.resolve(), args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite source gate: {output}")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True):
        parser.error("source-gate checkout must be clean")
    remote = subprocess.check_output(
        ["git", "-C", str(root), "ls-remote", "origin", f"refs/heads/{BRANCH}"], text=True
    ).strip().split()
    if len(remote) != 2 or remote[0] != head:
        parser.error("remote R006 branch is not equal to implementation checkout")
    for commit in (BASE, R005_CLOSURE_COMMIT):
        if subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"], check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode:
            parser.error(f"required lineage commit is absent: {commit}")
    registration = json.loads((root / FILES[1]).read_text(encoding="utf-8"))
    schedule = json.loads((root / FILES[0]).read_text(encoding="utf-8"))
    if registration.get("counts_at_registration") != {
        "r006_live_diagnostics": 0, "r006_live_candidate_evaluations": 0,
        "model_requests": 0, "behavioral_episodes": 0,
    }:
        parser.error("R006 registration counts differ")
    if schedule.get("joint_equilibrium_hold_contract", {}).get("settle_steps") != 780:
        parser.error("R006 equilibrium contract differs")
    if (root / FILES[2]).read_bytes() != b"":
        parser.error("R006 infrastructure ledger is not empty before first attempt")
    value = {
        "schema_version": "vla-wam-shared-v3e006-r006-source-push-gate-v1",
        "repair_amendment_id": "V3-E006-R006",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "passed_before_first_r006_live_diagnostic_candidate_or_model_request",
        "implementation_commit": head,
        "remote_equality": {"remote": "origin", "ref": f"refs/heads/{BRANCH}", "commit": remote[0]},
        "required_repository_base": BASE,
        "r005_closure_commit": R005_CLOSURE_COMMIT,
        "r005_results_sha256": R005_RESULTS_SHA256,
        "model_request_count": 0, "behavioral_episode_count": 0,
        "r006_live_diagnostic_count": 0, "r006_live_candidate_evaluation_count": 0,
        "accepted_state_candidate_count": 0, "infrastructure_invalid_attempt_count": 0,
        "repair_registration": bind(root, FILES[1]),
        "candidate_schedule": bind(root, FILES[0]),
        "implementation_files": [bind(root, relative) for relative in FILES],
        "implementation_file_count": len(FILES),
        "source_push_assertions": {
            "registration_schedule_and_source_predate_first_r006_live_diagnostic": True,
            "remote_ref_equals_implementation_commit": True,
            "r005_closure_preserved_and_bound": True,
            "r005_diagnostics_targets_solver_horizon_candidate_order_and_gates_unchanged": True,
            "sole_candidate_materialization_change_is_uniform_780_step_normal_joint_equilibrium_hold": True,
            "no_cartesian_action_during_settle": True,
            "behavioral_runtime_horizon_model_prompts_and_release_unchanged": True,
        },
        "release_boundary": "One zero-model diagnostic and conditional finite four-pair search only; behavior and inference remain blocked.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size, "sha256": sha256(output), "implementation_commit": head}, sort_keys=True))


if __name__ == "__main__":
    main()
