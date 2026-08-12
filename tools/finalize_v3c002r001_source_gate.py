#!/usr/bin/env python3
"""Write the pushed-source gate after the repair registration commit is pushed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import sha256_file
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import repo_binding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--registration-commit", required=True)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite source gate: {args.output}")
    remote_rows = subprocess.run(["git", "ls-remote", "--heads", args.remote, args.branch], check=True, capture_output=True, text=True).stdout.splitlines()
    heads = [row.split()[0] for row in remote_rows if row.split()]
    if len(heads) != 1:
        raise SystemExit("repair remote branch is missing or ambiguous")
    for commit in (args.implementation_commit, args.registration_commit):
        if subprocess.run(["git", "merge-base", "--is-ancestor", commit, heads[0]]).returncode != 0:
            raise SystemExit(f"repair commit is not pushed: {commit}")
    value = {
        "schema_version": "vla-wam-shared-v3c002r001-source-push-gate-v1",
        "repair_id": "V3-C002-R001",
        "status": "passed_repair_source_and_registration_pushed",
        "passed": True,
        "pushed": True,
        "implementation_commit": args.implementation_commit,
        "registration_commit": args.registration_commit,
        "remote": args.remote,
        "branch": args.branch,
        "repair_registration": repo_binding(args.root / "registration.json"),
        "repair_registration_sha256": sha256_file(args.root / "registration.json"),
        "queue": repo_binding(args.root / "queue.jsonl"),
        "assignment_manifest": repo_binding(args.root / "assignment.jsonl"),
        "behavioral_episodes_before_gate": 0,
        "repair_model_requests_before_gate": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
