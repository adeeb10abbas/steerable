#!/usr/bin/env python3
"""Build the prospective V3-C002-R001 repair registration and assignment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import sha256_file
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import REPO_ROOT, repo_binding


ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/active"
PARENT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002/active"
PARENT_NAMESPACE = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002"
IMPLEMENTATION_PATHS = (
    "experiments/v3/phase_c_semantic_equivalence_v3c002r001/__init__.py",
    "experiments/v3/phase_c_semantic_equivalence_v3c002r001/contract.py",
    "experiments/v3/phase_c_semantic_equivalence_v3c002r001/compiler.py",
    "experiments/v3/phase_c_semantic_equivalence_v3c002r001/droid_behavioral_adapter.py",
    "experiments/v3/phase_c_semantic_equivalence_v3c002r001/runner.py",
    "experiments/v3/phase_c_semantic_equivalence_v3c002r001/single_server_repeat.py",
    "tools/build_v3c002r001_registration.py",
    "tools/finalize_v3c002r001_source_gate.py",
    "tools/compile_v3c002r001_physical_gate.py",
    "tools/authorize_v3c002r001_smoke.py",
    "tools/compile_v3c002r001_smoke_gate.py",
    "tools/compile_v3c002r001_repeat_gate.py",
    "tools/build_v3c002r001_lane_manifest.py",
    "tools/release_v3c002r001_behavior.py",
    "tools/validate_v3c002r001.py",
    "tools/validate_v3c002r001_results.py",
    "tools/validate_v3c002r001_publication_bundle.py",
    "tests/test_v3c002r001_repair.py",
)


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, allow_nan=False, sort_keys=True) + "\n" for value in values), encoding="utf-8")


def original_tree_manifest() -> dict[str, Any]:
    committed_rows = subprocess.run(
        ["git", "ls-tree", "-r", "4269fd5a17e565beb36beb4e81920af038abc353", "--", "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    committed_paths = [row.split("\t", 1)[1] for row in committed_rows]
    current_paths = [(path.relative_to(REPO_ROOT)).as_posix() for path in sorted(PARENT_NAMESPACE.rglob("*")) if path.is_file()]
    if committed_paths != current_paths:
        raise SystemExit("original C002 namespace path set differs from closure commit")
    files = []
    for relative in committed_paths:
        path = REPO_ROOT / relative
        committed = subprocess.run(["git", "show", f"4269fd5a17e565beb36beb4e81920af038abc353:{relative}"], cwd=REPO_ROOT, check=True, capture_output=True).stdout
        if hashlib.sha256(committed).hexdigest() != sha256_file(path):
            raise SystemExit(f"original C002 file differs from closure commit: {relative}")
        files.append(repo_binding(path))
    tree_payload = "".join(f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n" for row in files).encode()
    return {
        "schema_version": "vla-wam-shared-v3c002-original-tree-identity-v1",
        "status": "original_c002_preserved_byte_identically_before_repair",
        "closure_commit": "4269fd5a17e565beb36beb4e81920af038abc353",
        "git_tree_oid": "0c81eda428c2682f3b1cffccc9372289d14f1aee",
        "file_count": len(files),
        "tree_sha256": hashlib.sha256(tree_payload).hexdigest(),
        "files": files,
    }


def assignment_rows() -> list[dict[str, Any]]:
    ranked = sorted(
        range(12000, 12341),
        key=lambda seed: hashlib.sha256(f"V3-C002-R001|balanced-lane-rank-v1|{seed}".encode()).hexdigest(),
    )
    rows = []
    for rank, seed in enumerate(ranked):
        slot = rank % 8
        rows.append({
            "schema_version": "vla-wam-shared-v3c002r001-block-assignment-v1",
            "repair_id": "V3-C002-R001",
            "episode_seed": seed,
            "seed_block_id": f"v3c002:seed{seed}",
            "lane_slot": f"repair-lane-{slot:02d}",
            "rank": rank,
            "conditions": ["canonical_left", "inverse_reference_left", "canonical_right", "inverse_reference_right"],
            "block_indivisible": True,
            "within_block_request0_bytes_matched": True,
            "incomplete_block_retry_same_lane_only": True,
        })
    return sorted(rows, key=lambda row: row["episode_seed"])


def main() -> None:
    if ROOT.exists():
        raise SystemExit(f"refusing to overwrite repair registration: {ROOT}")
    implementation_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    original_tree = subprocess.run(["git", "rev-parse", "4269fd5a17e565beb36beb4e81920af038abc353:artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if original_tree != "0c81eda428c2682f3b1cffccc9372289d14f1aee":
        raise SystemExit("original C002 Git tree identity changed")
    if subprocess.run(["git", "diff", "--quiet", "4269fd5a17e565beb36beb4e81920af038abc353", "--", "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002", "experiments/v3/phase_c_semantic_equivalence_v3c002", "tools/validate_v3c002_isolation_closure.py", "tools/validate_v3c002_publication_bundle.py"], cwd=REPO_ROOT).returncode != 0:
        raise SystemExit("original C002 closure/source namespace is not byte-identical")
    for relative in IMPLEMENTATION_PATHS:
        if not (REPO_ROOT / relative).is_file():
            raise SystemExit(f"missing repair implementation source: {relative}")
        committed = subprocess.run(["git", "show", f"{implementation_commit}:{relative}"], cwd=REPO_ROOT, capture_output=True)
        if committed.returncode != 0 or hashlib.sha256(committed.stdout).hexdigest() != sha256_file(REPO_ROOT / relative):
            raise SystemExit(f"repair implementation is not committed byte-identically: {relative}")
    tree_path = ROOT / "original_c002_tree_manifest.json"
    write_json(tree_path, original_tree_manifest())
    queue_path = ROOT / "queue.jsonl"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PARENT / "queue.jsonl", queue_path)
    assignment_path = ROOT / "assignment.jsonl"
    write_jsonl(assignment_path, assignment_rows())
    parent_registration = json.loads((PARENT / "registration.json").read_text(encoding="utf-8"))
    registration = {
        "schema_version": "vla-wam-shared-v3c002r001-repair-registration-v1",
        "repair_id": "V3-C002-R001",
        "parent_experiment_id": "V3-C002",
        "title": "Prospective block-local homogeneous-lane repair after failed cross-server isolation",
        "status": "registered_prospective_post_gate_repair",
        "registered_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z"),
        "user_authorized_protocol_override": True,
        "override_record": "Codex user message after immutable C002 closure: solve the issues and run the full experiment",
        "behavioral_episodes_before_registration": 0,
        "repair_model_requests_before_registration": 0,
        "original_c002_excluded_request_count_before_repair": 30,
        "parent_behavioral_episode_count": 0,
        "parent_registration": repo_binding(PARENT / "registration.json"),
        "parent_queue": repo_binding(PARENT / "queue.jsonl"),
        "queue": repo_binding(queue_path),
        "assignment_manifest": repo_binding(assignment_path),
        "wording_gate_inherited_unchanged": repo_binding(PARENT / "wording_gate.json"),
        "human_attestation_receipt_inherited_unchanged": repo_binding(PARENT / "attestation_receipt_order.json"),
        "original_tree_identity": repo_binding(tree_path),
        "original_failed_isolation_closure": {
            "evidence_manifest": repo_binding(PARENT / "closure/evidence_manifest.json"),
            "failure_report": repo_binding(PARENT / "gates/isolation_failure_report.json"),
            "target_raw_rehash_receipt": repo_binding(PARENT / "gates/isolation_target_raw_rehash_receipt.json"),
        },
        "diagnosis": {
            "same_fixture_prompt_and_seed": True,
            "different_policy_server_processes": True,
            "actions_exactly_equal": False,
            "max_absolute_action_difference": 0.0013794898986816406,
            "mean_absolute_action_difference": 0.0002258223103126511,
            "cause_identified": False,
            "supported_scope": "cross-server policy execution numerical variation or hidden process-state difference; retained evidence does not distinguish causes",
        },
        "prospective_change": {
            "original_two_lane_gate_remains_failed": True,
            "no_numerical_tolerance_added": True,
            "rationale": "The four-condition estimands are paired wholly within seed blocks. Cross-server byte identity is not an estimand and no prior protocol supplies a cross-GPU tolerance. R001 therefore requires exact repeatability within every actual server process and keeps each complete block on one homogeneous lane.",
            "scientific_estimands_changed": False,
            "sample_size_changed": False,
            "prompts_changed": False,
            "seeds_changed": False,
            "analysis_changed": False,
        },
        "execution_topology": {
            "policy": "eight_homogeneous_block_local_lanes_with_exact_within_server_repeat",
            "lane_count": 8,
            "lane_slots": [f"repair-lane-{index:02d}" for index in range(8)],
            "assignment": "SHA-256 rank all 341 registered seed blocks under V3-C002-R001|balanced-lane-rank-v1, then round-robin ranks across sorted lane slots",
            "assignment_count_imbalance_max": 1,
            "one_serial_client_and_one_policy_server_per_lane": True,
            "all_four_conditions_same_lane": True,
            "within_block_request0_bytes_matched": True,
            "exact_identical_input_prompt_seed_repeat_within_each_server": True,
            "cross_lane_action_equality_required": False,
            "cross_server_tolerance": None,
            "no_failover_within_block": True,
            "incomplete_block_retry_same_lane_only": True,
            "completed_block_rerun_prohibited": True,
        },
        "homogeneity_contract": {
            "equal_across_lanes": ["simulator_gpu_model", "simulator_driver", "policy_gpu_model", "policy_driver", "runtime_stack_sha256", "container_image_digest", "checkpoint_digest", "renderer_backend"],
            "distinct_across_lanes": ["simulator_pod_uid", "simulator_gpu_uuid", "policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "raw_root", "server_process_identity", "server_lock_identity"],
            "expected": {
                "simulator_gpu_model": "NVIDIA A40",
                "simulator_driver": "580.95.05",
                "policy_gpu_model": "NVIDIA A100-SXM4-80GB",
                "policy_driver": "580.95.05",
                "checkpoint_digest": parent_registration["exact_e004_pi05_runtime"]["identity_values"]["checkpoint_digest"],
                "renderer_backend": parent_registration["exact_e004_pi05_runtime"]["identity_values"]["renderer_backend"],
            },
        },
        "analysis_plan": parent_registration["analysis_plan"],
        "diagnostic_analysis_additions": {
            "confirmatory_estimands_unchanged": True,
            "lane_wise_paired_differences": True,
            "leave_one_lane_out_paired_differences": True,
            "diagnostic_only_not_claim_gates": True,
        },
        "technical_gate_plan": {
            "global_excluded_smoke_seed": 12000,
            "global_excluded_smoke_lane_slot": "repair-lane-00",
            "smoke_assignment_scope": "technical_excluded_only; behavioral assignment manifest remains unchanged",
            "per_lane_physical_gate_count": 8,
            "per_lane_repeat_gate_count": 8,
            "repeat_sequence": ["canonical_left", "canonical_right", "canonical_left"],
            "repeat_fixture_source": "exact retained request0 observation_cache.npz from the global excluded smoke",
        },
        "runtime": {
            "exact_parent_e004_pi05_contract_sha256": parent_registration["exact_e004_pi05_runtime"]["contract_sha256"],
            "exact_behavior_adapter_source_commit": "e2d9ae3904b4a08e549c784903c167a4213d3d47",
            "repair_wrapper_implementation_commit": implementation_commit,
        },
        "source_bindings": {relative: repo_binding(REPO_ROOT / relative) for relative in IMPLEMENTATION_PATHS},
        "release_sequence": [
            "pushed repair implementation and registration",
            "eight fresh same-process zero-request physical/runtime gates",
            "one global excluded complete four-cell smoke block on repair-lane-00",
            "eight fixed-observation interleaved exact-repeat and prompt-sensitivity gates",
            "eight homogeneous lane manifests and frozen assignment binding",
            "behavioral release before any denominator-eligible request",
        ],
        "release_status": "blocked_pending_pushed_source_and_technical_gates",
        "publication_routing": "Report as V3-C002-R001 post-gate operational repair; preserve original C002 failed isolation as methods/limitations evidence; never state the original gate passed.",
    }
    registration_path = ROOT / "registration.json"
    write_json(registration_path, registration)
    write_json(ROOT / "source_push_gate.json", {
        "schema_version": "vla-wam-shared-v3c002r001-source-push-gate-v1",
        "repair_id": "V3-C002-R001",
        "status": "blocked_pending_repair_registration_commit_and_push",
        "passed": False,
        "pushed": False,
        "repair_registration": repo_binding(registration_path),
        "queue": repo_binding(queue_path),
    })
    write_json(ROOT / "release_gate.json", {
        "schema_version": "vla-wam-shared-v3c002r001-release-gate-v1",
        "repair_id": "V3-C002-R001",
        "status": "blocked_pending_eight_homogeneous_lane_gates",
        "passed": False,
        "repair_registration": repo_binding(registration_path),
        "queue": repo_binding(queue_path),
        "assignment_manifest": repo_binding(assignment_path),
        "required": ["pushed source gate", "8 per-lane physical gates", "1 global excluded four-cell smoke gate on repair-lane-00", "8 per-lane single-server repeat gates", "8 homogeneous lane manifests"],
        "behavioral_episodes_authorized": False,
    })
    (ROOT / "infrastructure_attempts.jsonl").write_text("", encoding="utf-8")
    print(json.dumps({"status": registration["status"], "implementation_commit": implementation_commit, "registration_sha256": sha256_file(registration_path), "queue_sha256": sha256_file(queue_path), "assignment_sha256": sha256_file(assignment_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
