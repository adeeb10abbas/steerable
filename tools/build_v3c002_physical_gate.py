#!/usr/bin/env python3
"""Bind a fresh zero-request E004 s=1 preflight as the C002 physical gate."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError, file_binding, read_finite_json, repo_file_binding, sha256_file, validate_exact_runtime_contract

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True); parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--source-push-gate", type=Path, required=True); parser.add_argument("--standalone-report", type=Path, required=True)
    parser.add_argument("--invocation-record", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ContractError(f"refusing to overwrite physical gate: {args.output}")
    registration = read_finite_json(args.registration); source = read_finite_json(args.source_push_gate); report = read_finite_json(args.standalone_report)
    if not isinstance(registration, dict) or not isinstance(source, dict) or not isinstance(report, dict): raise ContractError("physical gate inputs must be JSON objects")
    exact_sha = validate_exact_runtime_contract(registration.get("exact_e004_pi05_runtime")); commit = registration["source_lineage"]["replacement_commit"]
    expected = {
        "schema_version": "vla-wam-shared-v3e004-standalone-model-blind-droid-gate-v2",
        "status": "passed_model_blind_preflight_not_a_behavioral_release", "passed": True,
        "model_request_count": 0, "behavioral_episode_count": 0, "behavioral_action_count": 0,
        "model_id": "pi05_current_stack_droid", "study_commit": commit,
        "candidate_sha256": registration["e004_s1_layout"]["candidate_sha256"],
    }
    for key, value in expected.items():
        if report.get(key) != value: raise ContractError(f"standalone physical report differs for {key}")
    if source.get("status") != "passed_source_commit_pushed" or source.get("passed") is not True or source.get("pushed") is not True or source.get("source_commit") != commit: raise ContractError("source push gate does not bind the V8 implementation")
    for name in ("live_scene_gate", "viewport_video"):
        record = report.get(name); path = Path(str(record.get("path", ""))) if isinstance(record, dict) else Path("")
        if not path.is_file() or record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path): raise ContractError(f"standalone report {name} binding changed")
    request0 = report.get("request0_replay")
    if not isinstance(request0, dict) or len(str(request0.get("pair_identity_sha256", ""))) != 64: raise ContractError("standalone report lacks request-zero evidence")
    value = {
        "schema_version": "vla-wam-shared-v3c002-model-blind-physical-gate-v1", "status": "passed_exact_e004_model_blind_physical_preflight", "passed": True,
        "registration": repo_file_binding(args.registration), "queue": repo_file_binding(args.queue), "source_push_gate": repo_file_binding(args.source_push_gate),
        "standalone_report": file_binding(args.standalone_report), "invocation_record": file_binding(args.invocation_record),
        "physical_scene": True, "full_reset": True, "policy_cameras": True, "raw_writer": True, "renderer": True,
        "model_requests": 0, "behavioral_episodes": 0, "behavioral_actions": 0,
        "same_process_gate_must_repeat_before_request_zero": True,
        "exact_runtime_contract_sha256": exact_sha, "source_commit": commit,
        "scope": "zero-request eligibility only; this does not authorize or contain behavior",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "sha256": sha256_file(args.output)}, sort_keys=True))
if __name__ == "__main__": main()
