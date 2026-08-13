#!/usr/bin/env python3
"""Build the prospective R001 activation-v2 transport correction."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import read_finite_json, sha256_file
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import REPO_ROOT, repo_binding
from tools.build_v3c002r001_registration import IMPLEMENTATION_PATHS


V1 = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/active"
V2 = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/activation_v2"
V1_COMMIT = "2935ad6568ad0034f5f82bfbd83c8d064fdd8331"
EXTRA_SOURCES = (
    "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/request0_replay.py",
    "tools/build_v3c002r001_activation_v2.py",
    "tools/validate_v3c002r001_repeat_fixture.py",
    "tools/validate_v3c002r001_failed_repeat_attempts.py",
)


def _write(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _v1_tree_identity() -> dict:
    rows = subprocess.check_output(
        ["git", "ls-tree", "-r", V1_COMMIT, "--", V1.relative_to(REPO_ROOT).as_posix()],
        cwd=REPO_ROOT,
        text=True,
    ).splitlines()
    paths = [row.split("\t", 1)[1] for row in rows]
    current = [(path.relative_to(REPO_ROOT)).as_posix() for path in sorted(V1.rglob("*")) if path.is_file()]
    if paths != current:
        raise SystemExit("R001 activation-v1 artifact path set changed")
    files = []
    for relative in paths:
        committed = subprocess.check_output(["git", "show", f"{V1_COMMIT}:{relative}"], cwd=REPO_ROOT)
        path = REPO_ROOT / relative
        if hashlib.sha256(committed).hexdigest() != sha256_file(path):
            raise SystemExit(f"R001 activation-v1 artifact changed: {relative}")
        files.append(repo_binding(path))
    return {
        "schema_version": "vla-wam-shared-v3c002r001-activation-v1-identity-v1",
        "status": "activation_v1_preserved_after_excluded_infrastructure_failure",
        "commit": V1_COMMIT,
        "file_count": len(files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-receipt", type=Path, required=True)
    parser.add_argument("--failed-attempt-receipt", type=Path, required=True)
    args = parser.parse_args()
    if V2.exists():
        raise SystemExit(f"refusing to overwrite activation-v2: {V2}")
    fixture_receipt = read_finite_json(args.fixture_receipt)
    failed_receipt = read_finite_json(args.failed_attempt_receipt)
    if (
        not isinstance(fixture_receipt, dict)
        or fixture_receipt.get("status") != "passed_native_tree_reconstruction_and_exact_pi05_request_pack"
        or any(fixture_receipt.get(key) != 0 for key in ("model_request_count", "behavioral_action_count", "behavioral_episode_count"))
    ):
        raise SystemExit("fixture target receipt did not pass at zero request")
    if (
        not isinstance(failed_receipt, dict)
        or failed_receipt.get("status") != "retained_eight_infrastructure_invalid_flat_cache_requests"
        or failed_receipt.get("model_request_count") != 8
        or failed_receipt.get("successful_response_count") != 0
        or failed_receipt.get("action_array_count") != 0
        or failed_receipt.get("behavioral_episode_count") != 0
    ):
        raise SystemExit("failed repeat receipt changed")
    implementation_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    sources = tuple(IMPLEMENTATION_PATHS) + EXTRA_SOURCES
    for relative in sources:
        path = REPO_ROOT / relative
        committed = subprocess.run(["git", "show", f"{implementation_commit}:{relative}"], cwd=REPO_ROOT, capture_output=True)
        if not path.is_file() or committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != sha256_file(path):
            raise SystemExit(f"activation-v2 source is not committed byte-identically: {relative}")

    V2.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(V1 / "queue.jsonl", V2 / "queue.jsonl")
    shutil.copyfile(V1 / "assignment.jsonl", V2 / "assignment.jsonl")
    shutil.copyfile(V1 / "original_c002_tree_manifest.json", V2 / "original_c002_tree_manifest.json")
    gates = V2 / "gates"
    gates.mkdir()
    shutil.copyfile(args.fixture_receipt, gates / "repeat_fixture_target_rehash_receipt.json")
    shutil.copyfile(args.failed_attempt_receipt, gates / "failed_repeat_attempt001_target_rehash_receipt.json")
    _write(V2 / "activation_v1_identity.json", _v1_tree_identity())

    v1_registration = read_finite_json(V1 / "registration.json")
    registration = deepcopy(v1_registration)
    registration.update({
        "activation_id": "V3-C002-R001-A001",
        "title": "Prospective activation-v2 transport correction after retained flat-cache client failure",
        "registered_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repair_model_requests_before_registration": 78,
        "behavioral_episodes_before_registration": 0,
        "status": "registered_prospective_post_gate_repair",
        "activation_v1_identity": repo_binding(V2 / "activation_v1_identity.json"),
        "activation_v1_registration": repo_binding(V1 / "registration.json"),
        "activation_v1_source_push_gate": repo_binding(V1 / "source_push_gate.released.json"),
        "inherited_physical_gates": [repo_binding(path) for path in sorted((V1 / "gates/physical").glob("*.json"))],
        "inherited_excluded_smoke_gate": repo_binding(V1 / "gates/excluded_smoke_gate.json"),
        "inherited_excluded_smoke_target_receipt": repo_binding(V1 / "gates/excluded_smoke_target_raw_rehash_receipt.json"),
        "repeat_fixture_target_receipt": repo_binding(gates / "repeat_fixture_target_rehash_receipt.json"),
        "failed_repeat_attempt001_target_receipt": repo_binding(gates / "failed_repeat_attempt001_target_rehash_receipt.json"),
        "repeat_fixture": fixture_receipt["fixture"],
        "repeat_fixture_manifest": fixture_receipt["fixture_manifest"],
        "transport_correction": {
            "science_changed": False,
            "queue_changed": False,
            "assignment_changed": False,
            "analysis_changed": False,
            "repeat_sequence_changed": False,
            "cause": "activation-v1 repeat client sent flattened cache storage keys instead of the exact E004 native observation tree and frozen Pi0DroidJointposClient packed request",
            "correction": "hash-validate manifest/cache, restore native leaf kinds and tree, then call exact source-bound Pi0DroidJointposClient._extract_observation and _pack_request before dispatch",
            "activation_v1_attempt001_classification": "infrastructure_invalid",
            "activation_v1_attempt001_model_requests": 8,
            "activation_v1_attempt001_successful_responses": 0,
            "activation_v1_attempt001_action_arrays": 0,
            "activation_v1_behavioral_episodes": 0,
            "one_retry_authorized_after_pushed_activation_v2": True,
        },
        "runtime": {**v1_registration["runtime"], "repair_wrapper_implementation_commit": implementation_commit},
        "source_bindings": {relative: repo_binding(REPO_ROOT / relative) for relative in sources},
        "release_status": "blocked_pending_activation_v2_source_push_and_eight_repeat_gates",
    })
    registration["queue"] = repo_binding(V2 / "queue.jsonl")
    registration["assignment_manifest"] = repo_binding(V2 / "assignment.jsonl")
    registration["original_tree_identity"] = repo_binding(V2 / "original_c002_tree_manifest.json")
    _write(V2 / "registration.json", registration)
    _write(V2 / "source_push_gate.json", {
        "schema_version": "vla-wam-shared-v3c002r001-source-push-gate-v1",
        "repair_id": "V3-C002-R001",
        "activation_id": "V3-C002-R001-A001",
        "status": "blocked_pending_activation_v2_registration_commit_and_push",
        "passed": False,
        "pushed": False,
        "repair_registration": repo_binding(V2 / "registration.json"),
        "queue": repo_binding(V2 / "queue.jsonl"),
        "repair_model_requests_before_gate": 78,
        "behavioral_episodes_before_gate": 0,
    })
    _write(V2 / "release_gate.json", {
        "schema_version": "vla-wam-shared-v3c002r001-release-gate-v1",
        "repair_id": "V3-C002-R001",
        "activation_id": "V3-C002-R001-A001",
        "status": "blocked_pending_eight_corrected_repeat_gates",
        "passed": False,
        "behavioral_episodes_authorized": False,
        "full_queue_launched": False,
        "repair_registration": repo_binding(V2 / "registration.json"),
        "queue": repo_binding(V2 / "queue.jsonl"),
        "assignment_manifest": repo_binding(V2 / "assignment.jsonl"),
        "inherited_physical_gate_count": 8,
        "inherited_excluded_smoke_gate_count": 1,
        "corrected_repeat_gate_count": 0,
    })
    print(json.dumps({"status": "registered_activation_v2_blocked_before_retry", "registration_sha256": sha256_file(V2 / "registration.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
