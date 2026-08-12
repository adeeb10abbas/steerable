#!/usr/bin/env python3
"""Target-side rehash of a failed V3-C002 two-lane isolation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


class ValidationError(RuntimeError):
    """Raised when retained isolation evidence has changed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise ValidationError(f"target file is absent: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON at {path}: {exc}") from exc


def rehash_bindings(
    value: Any,
    *,
    label: str,
    source_root: Path,
    seen: dict[tuple[str, str], int],
) -> None:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            path = Path(str(value["path"]))
            if not path.is_absolute():
                path = source_root / path
            path = path.resolve()
            if not path.is_file():
                raise ValidationError(f"{label} target file is absent: {path}")
            if type(value["bytes"]) is not int or value["bytes"] != path.stat().st_size:
                raise ValidationError(f"{label} byte count changed: {path}")
            if value["sha256"] != sha256_file(path):
                raise ValidationError(f"{label} digest changed: {path}")
            seen[(str(path), str(value["sha256"]))] = path.stat().st_size
            return
        for key, item in value.items():
            rehash_bindings(
                item,
                label=f"{label}.{key}",
                source_root=source_root,
                seen=seen,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rehash_bindings(
                item,
                label=f"{label}[{index}]",
                source_root=source_root,
                seen=seen,
            )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-report", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--active-root", type=Path, required=True)
    parser.add_argument("--invocation", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise ValidationError(f"refusing to overwrite target receipt: {output}")
    source_root = args.source_root.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    require(commit == args.expected_source_commit, "target source commit changed")
    active_root = args.active_root.resolve()
    require(
        not (active_root / "gates/two_lane_isolation_gate.json").exists(),
        "a passed isolation gate exists after registered exact-output failure",
    )
    release_path = active_root / "release_gate.json"
    release = read_json(release_path)
    require(
        isinstance(release, dict)
        and release.get("passed") is False
        and release.get("status") == "blocked_pending_committed_source_and_runtime_preflight_gates",
        "the retained fail-closed prerelease record became a behavioral release",
    )

    report_path = args.failure_report.resolve()
    report = read_json(report_path)
    require(isinstance(report, dict), "failure report is not an object")
    require(
        report.get("schema_version")
        == "vla-wam-shared-v3c002-two-lane-isolation-failure-v1",
        "failure report schema changed",
    )
    require(
        report.get("status") == "failed_registered_exact_action_equality_gate"
        and report.get("passed") is False,
        "isolation failure is not explicit",
    )
    for key in ("release_authorized", "full_behavior_authorized"):
        require(report.get(key) is False, f"{key} must remain false")
    require(report.get("no_retry_performed") is True, "isolation was retried")
    require(
        report.get("model_request_count") == 2
        and report.get("behavioral_episode_count") == 0
        and report.get("excluded_from_behavioral_denominators") is True,
        "isolation request/denominator counts changed",
    )
    for key in ("fixed_observation_equal", "fixed_prompt_equal", "request_seed_equal"):
        require(report.get(key) is True, f"isolation inputs differed for {key}")
    require(report.get("actions_exactly_equal") is False, "failed actions became equal")
    action_hashes = report.get("action_sha256_by_lane")
    require(
        isinstance(action_hashes, list)
        and len(action_hashes) == 2
        and len(set(action_hashes)) == 2,
        "failed action hashes are absent or equal",
    )
    require(
        report.get("finite_action_shape_by_lane") == [[15, 8], [15, 8]],
        "isolation action shapes changed",
    )

    seen: dict[tuple[str, str], int] = {}
    rehash_bindings(report, label="failure_report", source_root=source_root, seen=seen)
    seen[(str(report_path), sha256_file(report_path))] = report_path.stat().st_size

    for label, binding in (
        ("fixture_manifest", report["fixture_manifest"]),
        *(
            (f"lane_response[{index}]", binding)
            for index, binding in enumerate(report["lane_responses"])
        ),
        *(
            (f"runtime_manifest[{index}]", binding)
            for index, binding in enumerate(report["lane_runtime_manifests"])
        ),
    ):
        path = Path(str(binding["path"]))
        value = read_json(path)
        rehash_bindings(value, label=label, source_root=source_root, seen=seen)

    ledger_path = Path(str(report["infrastructure_invalid_attempts"]["path"]))
    ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    require(len(ledger_rows) == 4, "isolation prelaunch invalid ledger count changed")
    for index, row in enumerate(ledger_rows):
        require(
            row.get("model_request_count") == 0
            and row.get("behavioral_episode_count") == 0
            and row.get("denominator_eligible") is False,
            f"prelaunch invalid row {index} contains behavior",
        )
        rehash_bindings(
            row,
            label=f"prelaunch_invalid[{index}]",
            source_root=source_root,
            seen=seen,
        )

    action_paths = [Path(str(record["path"])) for record in report["lane_actions"]]
    actions = [np.load(path, allow_pickle=False) for path in action_paths]
    require(
        all(array.shape == (15, 8) and np.isfinite(array).all() for array in actions),
        "retained isolation action array is malformed",
    )
    require(not np.array_equal(actions[0], actions[1]), "retained actions became equal")
    max_abs = float(np.max(np.abs(actions[0] - actions[1])))
    mean_abs = float(np.mean(np.abs(actions[0] - actions[1])))
    require(max_abs == report["max_absolute_action_difference"], "max action delta changed")
    require(mean_abs == report["mean_absolute_action_difference"], "mean action delta changed")

    invocation = file_binding(args.invocation)
    environment = file_binding(args.environment)
    validator = file_binding(Path(__file__))
    for binding in (invocation, environment, validator):
        seen[(str(binding["path"]), str(binding["sha256"]))] = int(binding["bytes"])
    receipt = {
        "schema_version": "vla-wam-shared-v3c002-isolation-target-raw-rehash-v1",
        "status": "passed_target_rehash_of_failed_isolation_evidence",
        "passed": True,
        "isolation_gate_passed": False,
        "release_authorized": False,
        "full_behavior_authorized": False,
        "failure_report": file_binding(report_path),
        "source_root": str(source_root),
        "source_commit": commit,
        "validator": validator,
        "invocation": invocation,
        "environment": environment,
        "model_request_count": 2,
        "behavioral_episode_count": 0,
        "excluded_from_behavioral_denominators": True,
        "unique_raw_bindings_rehashed": len(seen),
        "raw_bytes_rehashed": sum(seen.values()),
        "action_sha256_by_lane": action_hashes,
        "max_absolute_action_difference": max_abs,
        "mean_absolute_action_difference": mean_abs,
        "validation_location": (
            "target simulator PVC with direct access to both lane roots; repository-relative "
            "bindings resolved against the exact e2d9ae3 source root"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "sha256": sha256_file(output),
                "bindings": receipt["unique_raw_bindings_rehashed"],
                "bytes": receipt["raw_bytes_rehashed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
