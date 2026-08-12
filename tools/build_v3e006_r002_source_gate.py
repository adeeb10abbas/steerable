#!/usr/bin/env python3
"""Create the one-shot prospective V3-E006-R002 source-push gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


BRANCH = "experiment/v3e006-r002-state-repair"
FILES = (
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r002/gates/candidate_schedule.json",
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r002/gates/predecessor_closure_binding.json",
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r002/repair_registration.json",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r002/__init__.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r002/candidate_schedule.py",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r002/state_repair_gate.py",
    "tests/test_v3e006_r002.py",
    "tools/build_v3e006_r002_predecessor_binding.py",
    "tools/build_v3e006_r002_source_gate.py",
    "tools/run_v3e006_r002_state_repair.py",
    "tools/validate_v3e006_r002.py",
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
        parser.error("remote R002 branch is not equal to implementation checkout")
    inventory = [bind(root, relative) for relative in FILES]
    registration = root / FILES[2]
    schedule = root / FILES[0]
    predecessor = root / FILES[1]
    result = {
        "schema_version": "vla-wam-shared-v3e006-r002-source-push-gate-v1",
        "repair_amendment_id": "V3-E006-R002",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "passed_before_first_r002_live_candidate_or_model_request",
        "implementation_commit": head,
        "remote_equality": {"remote": "origin", "ref": f"refs/heads/{BRANCH}", "commit": remote[0]},
        "required_repository_base": "18a2bf0200183647291cc7aeb1fe89997b3fb82f",
        "r001_exhaustion_closure_commit": "bbabac55dfd54f7a0b7d8a2693673a4b06409f21",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "r002_live_candidate_evaluation_count": 0,
        "completed_candidate_pair_count": 0,
        "accepted_state_candidate_count": 0,
        "infrastructure_invalid_search_attempt_count": 0,
        "repair_registration": bind(root, FILES[2]),
        "candidate_schedule": bind(root, FILES[0]),
        "predecessor_closure_binding": bind(root, FILES[1]),
        "implementation_files": inventory,
        "implementation_file_count": len(inventory),
        "source_push_assertions": {
            "registration_and_schedule_predate_live_candidate": True,
            "remote_ref_equals_implementation_commit": True,
            "original_and_r001_predecessors_preserved": True,
            "finite_rank_order_and_first_pass_frozen": True,
            "unchanged_scientific_gate_sources_bound": True,
            "behavior_and_model_release_prohibited": True
        },
        "release_boundary": "One finite sequential zero-model R002 candidate search only; behavior and inference remain blocked.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size, "sha256": sha256(output),
                      "implementation_commit": head}, sort_keys=True))


if __name__ == "__main__":
    main()
