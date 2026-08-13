#!/usr/bin/env python3
"""Bind the pushed source and already-committed replacement registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError, file_binding, read_finite_json, require, sha256_file


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "vla-wam-shared-v3c002r001-activation-v3-lane-replacement-source-gate-v1"
REGISTRATION_SCHEMA = "vla-wam-shared-v3c002r001-activation-v3-lane-replacement-registration-v1"


def _git(*parts: str) -> str:
    return subprocess.check_output(["git", *parts], cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replacement-registration", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"refusing to overwrite replacement source gate: {args.output}")
    registration = read_finite_json(args.replacement_registration)
    require(isinstance(registration, dict) and registration.get("schema_version") == REGISTRATION_SCHEMA and registration.get("status") == "registered_prospective_activation_v3_lane_replacement", "replacement registration is not active")
    registration_commit = _git("log", "-1", "--format=%H", "--", str(args.replacement_registration.resolve().relative_to(ROOT)))
    committed_registration = subprocess.check_output(["git", "show", f"{registration_commit}:{args.replacement_registration.resolve().relative_to(ROOT).as_posix()}"], cwd=ROOT)
    require(committed_registration == args.replacement_registration.read_bytes(), "replacement registration is not committed byte-identically")
    bindings = registration.get("source_bindings")
    require(isinstance(bindings, dict) and bindings, "replacement registration has no source bindings")
    for relative, binding in bindings.items():
        path = ROOT / relative
        require(path.is_file() and binding.get("sha256") == sha256_file(path), f"replacement source changed locally: {relative}")
        require(subprocess.run(["git", "diff", "--quiet", args.implementation_commit, "--", relative], cwd=ROOT).returncode == 0, f"replacement source differs from implementation commit: {relative}")
    remote_rows = _git("ls-remote", "--heads", args.remote, args.branch).splitlines()
    require(len(remote_rows) == 1, "replacement branch is absent or ambiguous on remote")
    remote_head = remote_rows[0].split()[0]
    for commit in (args.implementation_commit, registration_commit):
        require(len(commit) == 40 and subprocess.run(["git", "merge-base", "--is-ancestor", commit, remote_head], cwd=ROOT).returncode == 0, f"replacement commit not pushed: {commit}")
    value = {"schema_version": SCHEMA, "repair_id": "V3-C002-R001", "activation_id": registration.get("activation_id"), "status": "passed_activation_v3_lane_replacement_source_and_registration_pushed", "passed": True, "pushed": True, "replacement_registration": file_binding(args.replacement_registration), "replacement_registration_sha256": sha256_file(args.replacement_registration), "registration_commit": registration_commit, "implementation_commit": args.implementation_commit, "remote": args.remote, "branch": args.branch, "remote_head": remote_head, "source_bindings": bindings, "replacement_gate_model_requests_before_gate": 0, "replacement_gate_behavioral_episodes_before_gate": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
