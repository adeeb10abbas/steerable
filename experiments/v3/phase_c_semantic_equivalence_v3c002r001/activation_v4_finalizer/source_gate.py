#!/usr/bin/env python3
"""Emit a hash-bound post-push source receipt for the A004 final analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (
    ContractError,
    repo_file_binding,
    require,
    sha256_file,
    validate_file_binding,
)
from .registration import SCHEMA as REGISTRATION_SCHEMA, SOURCE_SCHEMA


REPO_ROOT = Path(__file__).resolve().parents[4]
CANONICAL_REMOTE = "https://github.com/adeeb10abbas/steerable.git"
CANONICAL_BRANCH = "experiment/v3c002-semantic-equivalence"


def _git_remote(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False,
    )
    require(result.returncode == 0, f"git verification failed: {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def verify_pushed_lineage(
    *, registration: Path, implementation_commit: str, registration_commit: str,
    remote_head: str, remote: str, branch: str, inventory: dict[str, Any],
) -> dict[str, Any]:
    """Derive all pushed/ancestry facts; never trust operator-supplied flags."""

    require(remote == CANONICAL_REMOTE and branch == CANONICAL_BRANCH, "final-analysis remote/branch changed")
    require(_git_remote("status", "--porcelain") == "", "final-analysis source-gate checkout is not clean")
    head = _git_remote("rev-parse", "HEAD")
    require(head == registration_commit == remote_head, "final-analysis checkout/registration/remote-head claims differ")
    for label, commit in (("implementation", implementation_commit), ("registration", registration_commit)):
        require(_git_remote("cat-file", "-t", commit) == "commit", f"final-analysis {label} commit is absent")
    _git_remote("merge-base", "--is-ancestor", implementation_commit, registration_commit)
    registration_relative = registration.resolve().relative_to(REPO_ROOT).as_posix()
    committed_registration = subprocess.run(
        ["git", "show", f"{registration_commit}:{registration_relative}"], cwd=REPO_ROOT,
        capture_output=True, check=False,
    )
    require(committed_registration.returncode == 0 and committed_registration.stdout == registration.read_bytes(), "final-analysis registration is not byte-identical in its commit")
    for label, binding in inventory.items():
        relative = str(binding.get("path", ""))
        require(relative and not Path(relative).is_absolute(), f"final-analysis source inventory path is not portable: {label}")
        committed = subprocess.run(
            ["git", "show", f"{implementation_commit}:{relative}"], cwd=REPO_ROOT,
            capture_output=True, check=False,
        )
        require(committed.returncode == 0, f"final-analysis source is absent from implementation commit: {label}")
        require(len(committed.stdout) == binding.get("bytes") and sha256_file_bytes(committed.stdout) == binding.get("sha256"), f"final-analysis implementation source bytes changed: {label}")
    remote_rows = _git_remote("ls-remote", "--heads", remote, branch).splitlines()
    require(remote_rows == [f"{registration_commit}\trefs/heads/{branch}"], "final-analysis remote branch is not the exact registration commit")
    return {"checkout_head": head, "remote_head": registration_commit, "source_inventory_verified_at_implementation_commit": True}


def sha256_file_bytes(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


def _write_new(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite final-analysis source gate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(root: Path, *args: str, check: bool = True) -> str:
    """Run a local deterministic Git object/ancestry check."""
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if check:
        require(result.returncode == 0, f"git verification failed: {' '.join(args)}")
    return result.stdout.strip()


def _verify_commit(root: Path, value: str, label: str) -> str:
    require(len(value) == 40 and all(char in "0123456789abcdef" for char in value), f"{label} is not a full lowercase SHA")
    resolved = _git(root, "rev-parse", "--verify", f"{value}^{{commit}}")
    require(resolved == value, f"{label} is not an exact local commit object")
    return resolved


def _verify_inventory_at_commit(root: Path, commit: str, inventory: dict[str, Any]) -> None:
    for relative, binding in inventory.items():
        require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), "source inventory path is not repository-relative")
        content = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{relative}"],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        require(content.returncode == 0, f"source inventory is absent from registration commit: {relative}")
        import hashlib
        require(len(content.stdout) == binding.get("bytes") and hashlib.sha256(content.stdout).hexdigest() == binding.get("sha256"), f"source inventory differs from registration commit: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--registration-commit", required=True)
    parser.add_argument("--remote-head", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    require(
        isinstance(registration, dict)
        and registration.get("schema_version") == REGISTRATION_SCHEMA
        and registration.get("status") == "registered_prospective_corrected_final_analysis_before_raw_aggregation_or_result_read"
        and all(registration.get(key) == 0 for key in (
            "final_analysis_raw_behavioral_rows_read_before_registration",
            "final_analysis_result_compilations_before_registration",
            "final_analysis_output_files_before_registration",
        )),
        "final-analysis registration was not prospective",
    )
    inventory = registration.get("source_inventory")
    require(isinstance(inventory, dict) and inventory, "final-analysis source inventory missing")
    for label, binding in inventory.items():
        validate_file_binding(binding, f"final-analysis source inventory {label}")
    verified = verify_pushed_lineage(
        registration=args.registration,
        implementation_commit=args.implementation_commit,
        registration_commit=args.registration_commit,
        remote_head=args.remote_head,
        remote=args.remote,
        branch=args.branch,
        inventory=inventory,
    )
    root = Path(__file__).resolve().parents[4]
    implementation_commit = _verify_commit(root, args.implementation_commit, "implementation commit")
    registration_commit = _verify_commit(root, args.registration_commit, "registration commit")
    require(args.remote_head == registration_commit, "pinned remote-head receipt must equal the registration commit")
    _git(root, "merge-base", "--is-ancestor", implementation_commit, registration_commit)
    registration_relative = args.registration.resolve().relative_to(root.resolve())
    registration_bytes = subprocess.run(
        ["git", "-C", str(root), "show", f"{registration_commit}:{registration_relative}"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    require(
        registration_bytes.returncode == 0
        and registration_bytes.stdout == args.registration.read_bytes(),
        "registration bytes are not committed at the stated registration commit",
    )
    _verify_inventory_at_commit(root, registration_commit, inventory)
    value = {
        "schema_version": SOURCE_SCHEMA,
        "status": "passed_final_analysis_source_and_registration_pushed_before_raw_aggregation",
        "passed": True,
        "pushed": True,
        "repair_id": "V3-C002-R001",
        "activation_id": "V3-C002-R001-A004-final-analysis",
        "remote": args.remote,
        "branch": args.branch,
        "implementation_commit": implementation_commit,
        "registration_commit": registration_commit,
        "remote_head_at_gate": registration_commit,
        "remote_head_receipt_kind": "live_ls_remote_exact_branch_head_plus_local_ancestry",
        "final_analysis_registration": repo_file_binding(args.registration),
        "registration_sha256": sha256_file(args.registration),
        "final_analysis_raw_behavioral_rows_read_before_registration": 0,
        "final_analysis_result_compilations_before_registration": 0,
        "final_analysis_output_files_before_registration": 0,
        "source_inventory": inventory,
        "git_verification": verified,
    }
    _write_new(args.output, value)
    print(json.dumps(repo_file_binding(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"A004 final-analysis source gate failed: {exc}") from exc
