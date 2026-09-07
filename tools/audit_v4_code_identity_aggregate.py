#!/usr/bin/env python3
"""Audit whether two study commits may share one V4 model-blind aggregate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "v4-code-identity-aggregate-audit-v1"

# Files whose changes can alter model-blind G2/G3 check behavior or receipts.
MODEL_BLIND_CHECK_PATHS: tuple[str, ...] = (
    "experiments/online_correction_v4/droid_robolab.py",
    "experiments/online_correction_v4/detectors.py",
    "experiments/online_correction_v4/model_blind_g2.py",
    "experiments/online_correction_v4/model_blind_g3.py",
    "experiments/online_correction_v4/droid_task_files/",
    "tools/run_v4_g2_checked.py",
    "tools/run_v4_horizontal_g2_seed.py",
    "tools/run_v4_g3_checked.py",
    "tools/run_v4_horizontal_g3_path_seed.py",
    "tools/run_v4_horizontal_g3_scripted_seed.py",
    "deploy/k8s/v4_lane_bundle/scripts/",
    "artifacts/online_correction_v4/setup/horizontal_reset_registry",
    "artifacts/online_correction_v4/setup/horizontal_g3_plan",
    "artifacts/online_correction_v4/setup/reference_binding",
    "artifacts/online_correction_v4/setup/containment",
    "artifacts/online_correction_v4/setup/vertical",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path_touches_model_blind_check(path: str) -> bool:
    normalized = path.strip("/")
    for prefix in MODEL_BLIND_CHECK_PATHS:
        prefix_norm = prefix.strip("/")
        if normalized == prefix_norm or normalized.startswith(prefix_norm):
            return True
    return False


def changed_files_between_commits(
    repo_root: Path,
    *,
    base_commit: str,
    head_commit: str,
) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", f"{base_commit}..{head_commit}"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def audit_code_identity_aggregate(
    *,
    base_commit: str,
    head_commit: str,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    base_commit = base_commit.lower()
    head_commit = head_commit.lower()
    if base_commit == head_commit:
        return {
            "schema_version": SCHEMA,
            "audited_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "base_commit": base_commit,
            "head_commit": head_commit,
            "changed_files": [],
            "model_blind_check_touched_files": [],
            "aggregate_rule": "homogeneous_runtime_stratum_required",
            "aggregate_mixing_allowed": False,
            "compiler_would_accept_mixed_stratum": False,
            "re_run_required_for_homogeneous_aggregate": False,
            "disclosure_only_model_blind_path_clean": True,
            "rationale": (
                "Identical study commits; compile_v4_*_aggregate requires one runtime_stratum "
                "including study_commit across all seed receipts."
            ),
        }

    changed = changed_files_between_commits(repo_root, base_commit=base_commit, head_commit=head_commit)
    touched = sorted(path for path in changed if _path_touches_model_blind_check(path))
    path_clean = not touched
    payload = {
        "schema_version": SCHEMA,
        "audited_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_commit": base_commit,
        "head_commit": head_commit,
        "changed_files": changed,
        "model_blind_check_touched_files": touched,
        "aggregate_rule": "homogeneous_runtime_stratum_required",
        "aggregate_mixing_allowed": False,
        "compiler_would_accept_mixed_stratum": False,
        "re_run_required_for_homogeneous_aggregate": True,
        "disclosure_only_model_blind_path_clean": path_clean,
        "rationale": (
            "tools/compile_v4_horizontal_g2_aggregate.py and model_blind_g3 aggregate compilation "
            "require every seed receipt to share the exact runtime_stratum dict, including "
            "study_commit, gate_core_sha256, and droid_robolab_sha256. Receipts produced at "
            "different study pins cannot be merged into one aggregate even when the Git diff does "
            "not touch the model-blind check path. When model_blind_check_touched_files is empty, "
            "the diff may be disclosed as non-impacting for interpretation, but homogeneous "
            "re-execution at the target pin is still required for aggregate compilation."
        ),
    }
    if touched:
        payload["rationale"] += (
            f" This diff touches model-blind check paths: {touched[:5]}"
            + (" ..." if len(touched) > 5 else "")
            + ". Prior receipts are scientifically stale for the repaired path; re-run all seeds."
        )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--head-commit", required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = audit_code_identity_aggregate(
        base_commit=args.base_commit,
        head_commit=args.head_commit,
        repo_root=args.repo_root.resolve(),
    )
    body = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(body, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(body.encode("utf-8"))
        sidecar = args.out.with_suffix(".sha256")
        sidecar.write_text(_sha256_bytes(body.encode("utf-8")) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
