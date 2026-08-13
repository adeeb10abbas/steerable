#!/usr/bin/env python3
"""Freeze the prospective R001 release-accounting activation v3."""

from __future__ import annotations

from copy import deepcopy
import argparse
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


V2 = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/activation_v2"
V3 = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/activation_v3"
V2_COMMIT = "c0d8c143e46d19694f3270908f857d9b313181d7"
EXTRA = (
    "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/request0_replay.py",
    "tools/build_v3c002r001_activation_v2.py",
    "tools/build_v3c002r001_activation_v3.py",
    "tools/validate_v3c002r001_repeat_fixture.py",
    "tools/validate_v3c002r001_failed_repeat_attempts.py",
)


def write(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def v2_identity() -> dict:
    rows = subprocess.check_output(["git", "ls-tree", "-r", V2_COMMIT, "--", V2.relative_to(REPO_ROOT).as_posix()], cwd=REPO_ROOT, text=True).splitlines()
    paths = [row.split("\t", 1)[1] for row in rows]
    current = [(path.relative_to(REPO_ROOT)).as_posix() for path in sorted(V2.rglob("*")) if path.is_file()]
    if paths != current:
        raise SystemExit("activation-v2 artifact path set changed")
    files = []
    for relative in paths:
        committed = subprocess.check_output(["git", "show", f"{V2_COMMIT}:{relative}"], cwd=REPO_ROOT)
        path = REPO_ROOT / relative
        if hashlib.sha256(committed).hexdigest() != sha256_file(path):
            raise SystemExit(f"activation-v2 artifact changed: {relative}")
        files.append(repo_binding(path))
    return {"schema_version": "vla-wam-shared-v3c002r001-activation-v2-identity-v1", "status": "activation_v2_preserved_after_corrected_repeat_before_release", "commit": V2_COMMIT, "file_count": len(files), "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-dir", type=Path, required=True)
    args = parser.parse_args()
    if V3.exists():
        raise SystemExit(f"refusing overwrite: {V3}")
    implementation = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    sources = tuple(IMPLEMENTATION_PATHS) + EXTRA
    for relative in sources:
        path = REPO_ROOT / relative
        committed = subprocess.run(["git", "show", f"{implementation}:{relative}"], cwd=REPO_ROOT, capture_output=True)
        if not path.is_file() or committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != sha256_file(path):
            raise SystemExit(f"activation-v3 source not committed: {relative}")
    V3.mkdir(parents=True)
    for name in ("queue.jsonl", "assignment.jsonl", "original_c002_tree_manifest.json"):
        shutil.copyfile(V2 / name, V3 / name)
    write(V3 / "activation_v2_identity.json", v2_identity())
    lane_root = V3 / "gates/lanes"
    shutil.copytree(args.lane_dir, lane_root)
    lane_manifests = sorted(lane_root.glob("*/lane_manifest.json"))
    if len(lane_manifests) != 8:
        raise SystemExit("activation-v3 requires exactly eight inherited lane manifests")
    v2 = read_finite_json(V2 / "registration.json")
    registration = deepcopy(v2)
    registration.update({
        "activation_id": "V3-C002-R001-A002",
        "title": "Prospective activation-v3 release evidence accounting correction",
        "registered_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repair_model_requests_before_registration": 102,
        "behavioral_episodes_before_registration": 0,
        "activation_v2_identity": repo_binding(V3 / "activation_v2_identity.json"),
        "activation_v2_registration": repo_binding(V2 / "registration.json"),
        "activation_v2_source_push_gate": repo_binding(V2 / "source_push_gate.released.json"),
        "inherited_physical_gates": [repo_binding(path) for path in sorted((REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/active/gates/physical").glob("*.json"))],
        "inherited_excluded_smoke_gate": repo_binding(REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/active/gates/excluded_smoke_gate.json"),
        "inherited_corrected_repeat_gates": [repo_binding(path) for path in sorted((V2 / "gates/repeat").glob("*.json"))],
        "inherited_corrected_repeat_target_receipts": [repo_binding(path) for path in sorted((V2 / "gates/repeat_receipts").glob("*.json"))],
        "inherited_lane_manifests": [repo_binding(path) for path in lane_manifests],
        "failed_repeat_attempt001_target_receipt": v2["failed_repeat_attempt001_target_receipt"],
        "release_accounting_correction": {"science_changed": False, "queue_changed": False, "assignment_changed": False, "analysis_changed": False, "gates_repeated": False, "correct_total_excluded_requests": 102, "breakdown": {"global_smoke": 70, "retained_invalid_attempt001": 8, "corrected_repeat_attempt002": 24}, "reason": "activation-v2 release tool omitted the eight retained infrastructure-invalid attempt001 requests and did not bind all target repeat receipts"},
        "runtime": {**v2["runtime"], "repair_wrapper_implementation_commit": implementation},
        "source_bindings": {relative: repo_binding(REPO_ROOT / relative) for relative in sources},
        "release_status": "blocked_pending_activation_v3_source_push_lane_manifests_and_release",
    })
    registration["queue"] = repo_binding(V3 / "queue.jsonl")
    registration["assignment_manifest"] = repo_binding(V3 / "assignment.jsonl")
    registration["original_tree_identity"] = repo_binding(V3 / "original_c002_tree_manifest.json")
    write(V3 / "registration.json", registration)
    write(V3 / "source_push_gate.json", {"schema_version": "vla-wam-shared-v3c002r001-source-push-gate-v1", "repair_id": "V3-C002-R001", "activation_id": "V3-C002-R001-A002", "status": "blocked_pending_activation_v3_registration_commit_and_push", "passed": False, "pushed": False, "repair_registration": repo_binding(V3 / "registration.json"), "queue": repo_binding(V3 / "queue.jsonl"), "repair_model_requests_before_gate": 102, "behavioral_episodes_before_gate": 0})
    write(V3 / "release_gate.json", {"schema_version": "vla-wam-shared-v3c002r001-release-gate-v1", "repair_id": "V3-C002-R001", "activation_id": "V3-C002-R001-A002", "status": "blocked_pending_lane_manifests_and_corrected_release", "passed": False, "behavioral_episodes_authorized": False, "full_queue_launched": False, "repair_registration": repo_binding(V3 / "registration.json"), "queue": repo_binding(V3 / "queue.jsonl"), "assignment_manifest": repo_binding(V3 / "assignment.jsonl"), "repair_excluded_request_count_before_release": 102})
    print(json.dumps({"status": "registered_activation_v3_blocked_before_behavior", "registration_sha256": sha256_file(V3 / "registration.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
