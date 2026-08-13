#!/usr/bin/env python3
"""Validate the prospective V3-C002-R001 package before technical requests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import sha256_file, validate_file_binding
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import REPO_ROOT, load_repair, require, validate_assignment, verify_pushed_gate


V1_ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/active"
V2_ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/activation_v2"
V3_ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/activation_v3"
ROOT = V3_ROOT if V3_ROOT.exists() else V2_ROOT if V2_ROOT.exists() else V1_ROOT
PARENT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002/active"
PARENT_NAMESPACE = PARENT.parent
ORIGINAL_COMMIT = "4269fd5a17e565beb36beb4e81920af038abc353"
ORIGINAL_TREE_OID = "0c81eda428c2682f3b1cffccc9372289d14f1aee"


def main() -> None:
    repair, cells = load_repair(registration_path=ROOT / "registration.json", queue_path=ROOT / "queue.jsonl")
    require(len(cells) == 1364, "repair cell count changed")
    require(sha256_file(ROOT / "queue.jsonl") == sha256_file(PARENT / "queue.jsonl"), "repair queue is not parent byte-identical")
    rows = validate_assignment(repair["assignment_manifest"])
    for seed in range(12000, 12341):
        require(len([cell for cell in cells if cell.seed == seed]) == 4, f"repair seed {seed} is incomplete")
        require(len([row for row in rows if row["episode_seed"] == seed]) == 1, f"repair seed {seed} assignment changed")
    tree_record = validate_file_binding(repair["original_tree_identity"], "original tree identity")
    tree = json.loads(Path(tree_record["path"]).read_text(encoding="utf-8"))
    require(tree.get("closure_commit") == ORIGINAL_COMMIT and tree.get("git_tree_oid") == ORIGINAL_TREE_OID, "original C002 Git identity changed")
    require(tree.get("file_count") == len(tree.get("files", [])), "original C002 tree file count changed")
    committed_rows = subprocess.run(["git", "ls-tree", "-r", ORIGINAL_COMMIT, "--", "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    committed_paths = [row.split("\t", 1)[1] for row in committed_rows]
    current_paths = [(path.relative_to(REPO_ROOT)).as_posix() for path in sorted(PARENT_NAMESPACE.rglob("*")) if path.is_file()]
    require(committed_paths == current_paths == [record["path"] for record in tree["files"]], "original C002 path set differs from closure commit")
    for record in tree["files"]:
        validate_file_binding(record, "original C002 file")
        committed = subprocess.run(["git", "show", f"{ORIGINAL_COMMIT}:{record['path']}"], cwd=REPO_ROOT, check=True, capture_output=True).stdout
        require(__import__("hashlib").sha256(committed).hexdigest() == record["sha256"], f"original C002 committed bytes differ: {record['path']}")
    require(repair.get("original_c002_excluded_request_count_before_repair") == 30, "original excluded request count changed")
    if ROOT in (V2_ROOT, V3_ROOT):
        expected_activation = "V3-C002-R001-A002" if ROOT == V3_ROOT else "V3-C002-R001-A001"
        expected_requests = 102 if ROOT == V3_ROOT else 78
        require(repair.get("activation_id") == expected_activation, "repair activation ID changed")
        require(repair.get("repair_model_requests_before_registration") == expected_requests and repair.get("behavioral_episodes_before_registration") == 0, "activation preregistration counts changed")
        identity_key = "activation_v2_identity" if ROOT == V3_ROOT else "activation_v1_identity"
        identity_root = V2_ROOT if ROOT == V3_ROOT else V1_ROOT
        identity_record = validate_file_binding(repair.get(identity_key), "prior activation identity")
        identity = json.loads(Path(identity_record["path"]).read_text(encoding="utf-8"))
        committed_rows = subprocess.run(["git", "ls-tree", "-r", identity["commit"], "--", identity_root.relative_to(REPO_ROOT).as_posix()], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
        committed_paths = [row.split("\t", 1)[1] for row in committed_rows]
        current_prior_paths = [(path.relative_to(REPO_ROOT)).as_posix() for path in sorted(identity_root.rglob("*")) if path.is_file()]
        require(committed_paths == current_prior_paths == [record["path"] for record in identity["files"]], "prior activation artifact path set changed")
        for record in identity["files"]:
            validate_file_binding(record, "prior activation artifact")
            committed = subprocess.run(["git", "show", f"{identity['commit']}:{record['path']}"], cwd=REPO_ROOT, check=True, capture_output=True).stdout
            require(__import__("hashlib").sha256(committed).hexdigest() == record["sha256"], f"prior activation committed bytes changed: {record['path']}")
        fixture_receipt = json.loads(Path(validate_file_binding(repair["repeat_fixture_target_receipt"], "repeat fixture receipt")["path"]).read_text(encoding="utf-8"))
        failed_receipt = json.loads(Path(validate_file_binding(repair["failed_repeat_attempt001_target_receipt"], "failed repeat receipt")["path"]).read_text(encoding="utf-8"))
        require(fixture_receipt.get("passed") is True and fixture_receipt.get("model_request_count") == 0 and fixture_receipt.get("behavioral_episode_count") == 0, "repeat fixture target validation changed")
        require(failed_receipt.get("model_request_count") == 8 and failed_receipt.get("successful_response_count") == 0 and failed_receipt.get("action_array_count") == 0 and failed_receipt.get("behavioral_episode_count") == 0, "failed repeat attempt ledger changed")
    release = json.loads((ROOT / "release_gate.json").read_text(encoding="utf-8"))
    require(release.get("passed") is False and release.get("behavioral_episodes_authorized") is False, "repair is prematurely released")
    released_source = ROOT / "source_push_gate.released.json"
    if released_source.exists():
        source = json.loads(released_source.read_text(encoding="utf-8"))
        verify_pushed_gate(source, repair)
        require(source.get("repair_model_requests_before_gate") == expected_requests and source.get("behavioral_episodes_before_gate") == 0, "activation source gate counts changed")
        phase = "pushed_pending_technical_gates"
    else:
        source = json.loads((ROOT / "source_push_gate.json").read_text(encoding="utf-8"))
        require(source.get("passed") is False and source.get("pushed") is False, "pending source gate changed")
        phase = "registered_pending_source_push"
    print(json.dumps({
        "status": "valid_registered_v3c002r001_blocked_before_technical_requests",
        "phase": phase,
        "queue_cells": len(cells),
        "seed_blocks": 341,
        "lane_slots": 8,
        "activation_id": repair.get("activation_id", "V3-C002-R001-v1"),
        "parent_queue_sha256": sha256_file(PARENT / "queue.jsonl"),
        "assignment_sha256": sha256_file(ROOT / "assignment.jsonl"),
        "original_c002_tree_sha256": tree["tree_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
