#!/usr/bin/env python3
"""Compile the immutable post-smoke C002 behavioral release gate."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (
    ContractError, RELEASE_GATE_SCHEMA, read_finite_json, repo_file_binding, require_released_gate, sha256_file,
)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-root", type=Path, required=True)
    parser.add_argument("--physical-gate", type=Path, required=True); parser.add_argument("--excluded-smoke-gate", type=Path, required=True)
    parser.add_argument("--two-lane-isolation-gate", type=Path, required=True); parser.add_argument("--lane-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise ContractError(f"refusing to overwrite behavioral release: {args.output}")
    root = args.active_root.resolve()
    smoke = read_finite_json(args.excluded_smoke_gate); isolation = read_finite_json(args.two_lane_isolation_gate)
    if not isinstance(smoke, dict) or not isinstance(isolation, dict): raise ContractError("smoke/isolation gates are invalid")
    request_count = smoke.get("model_request_count") + isolation.get("model_request_count") if type(smoke.get("model_request_count")) is int and type(isolation.get("model_request_count")) is int else None
    if request_count is None: raise ContractError("smoke/isolation gates lack exact model request counts")
    value = {
        "schema_version": RELEASE_GATE_SCHEMA, "study_id": "vla_wam_language_steerability_v3", "amendment_id": "V3-C002",
        "status": "passed_pre_request_release", "passed": True,
        "registration": repo_file_binding(root / "registration.json"), "queue": repo_file_binding(root / "queue.jsonl"),
        "wording_gate": repo_file_binding(root / "wording_gate.json"), "attestation_receipt_order": repo_file_binding(root / "attestation_receipt_order.json"),
        "source_push_gate": repo_file_binding(root / "source_push_gate.json"), "physical_gate": repo_file_binding(args.physical_gate),
        "excluded_smoke_gate": repo_file_binding(args.excluded_smoke_gate), "two_lane_isolation_gate": repo_file_binding(args.two_lane_isolation_gate),
        "lane_manifests": [repo_file_binding(path) for path in args.lane_manifest],
        "model_requests_before_behavioral_release": request_count,
        "behavioral_episodes_before_release": 0,
        "excluded_smoke_and_isolation_requests_are_not_behavioral_observations": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    require_released_gate(registration_path=root / "registration.json", queue_path=root / "queue.jsonl", release_gate_path=args.output)
    print(json.dumps({"status": value["status"], "sha256": sha256_file(args.output)}, sort_keys=True))
if __name__ == "__main__": main()
