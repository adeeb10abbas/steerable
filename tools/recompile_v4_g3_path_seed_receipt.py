#!/usr/bin/env python3
"""Offline recompile object_pair G3 path-seed receipts after gate contract correction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.model_blind_g3 import (  # noqa: E402
    G3GateError,
    HORIZONTAL_GOALS,
    HORIZONTAL_PATH_CHECKS_PER_SEED,
    canonical_json_bytes,
    compile_path_seed_receipt,
    expected_information_gate_pass,
    path_receipt_schema,
    sha256_bytes,
    validate_path_seed_receipt,
    validate_plan_payload,
    _plan_shrinking_area_fraction_gate_applicable,
)

DERIVATION_REASON = "campaign_goal_area_gate_fixture_scope_correction"

GOAL_AREA_EVIDENCE_KEYS = (
    "relation",
    "original_area_m2",
    "destination_area_m2",
    "shrinking_direction",
    "removed_area_fraction",
    "minimum_shrinking_area_fraction",
    "original_goal_empty",
    "destination_goal_empty",
)

CHECK_OBSERVATION_KEYS = (
    "planned_duration_s",
    "sample_interval_s",
    "sample_count",
    "measured_pose_evidence",
    "reference_pose_evidence",
    "path_conformance",
    "collision_free",
    "support_valid",
    "reachable_workspace",
    "legal_goal_nonempty",
    "reference_robot_contact",
    "unmodeled_collision",
    "reasons",
    "passed",
)


def _load_canonical_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes, str]:
    body = path.read_bytes()
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError(f"{label}: must be a JSON object")
    if body != canonical_json_bytes(value):
        raise ValueError(f"{label}: JSON is not canonical")
    return value, body, sha256_bytes(body)


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(dict(payload))
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def _observation_from_check(check: Mapping[str, Any]) -> dict[str, Any]:
    observation: dict[str, Any] = {}
    for key in CHECK_OBSERVATION_KEYS:
        if key not in check:
            raise ValueError(f"path check lacks {key}")
        value = check[key]
        if key in {"measured_pose_evidence", "reference_pose_evidence"}:
            if not isinstance(value, Mapping):
                raise ValueError(f"path check {key} must be a mapping")
            observation[key] = dict(value)
        elif key == "reasons":
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ValueError("path check reasons must be a string list")
            observation[key] = list(value)
        else:
            observation[key] = value
    return observation


def _goal_area_evidence(case: Mapping[str, Any]) -> dict[str, Any]:
    return {key: case[key] for key in GOAL_AREA_EVIDENCE_KEYS}


def _validate_source_receipt(
    receipt: Mapping[str, Any],
    *,
    source_path: Path,
) -> None:
    fixture_id = receipt.get("fixture_id")
    if fixture_id != "object_pair":
        raise ValueError(
            f"{source_path}: only object_pair path-seed receipts may be recompiled"
        )
    schema = receipt.get("schema_version")
    if schema != path_receipt_schema("object_pair"):
        raise ValueError(f"{source_path}: path seed receipt schema differs")
    if receipt.get("campaign_id") != "online_correction_v4":
        raise ValueError(f"{source_path}: campaign differs")
    if receipt.get("model_request_count") != 0:
        raise ValueError(f"{source_path}: source records model requests")
    if receipt.get("behavioral_episode_count") != 0:
        raise ValueError(f"{source_path}: source records behavioral episodes")
    if receipt.get("check_count") != HORIZONTAL_PATH_CHECKS_PER_SEED:
        raise ValueError(f"{source_path}: path receipt check_count differs")
    checks = receipt.get("checks")
    if not isinstance(checks, list) or len(checks) != HORIZONTAL_PATH_CHECKS_PER_SEED:
        raise ValueError(f"{source_path}: path receipt lacks 24 checks")
    goal_area_cases = receipt.get("goal_area_cases")
    if not isinstance(goal_area_cases, list) or len(goal_area_cases) != len(
        HORIZONTAL_GOALS
    ):
        raise ValueError(f"{source_path}: path receipt lacks goal-area cases")
    if type(receipt.get("environment_seed")) is not int:
        raise ValueError(f"{source_path}: environment_seed must be an integer")


def _recompute_goal_area_cases(
    source_cases: list[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selection = plan.get("scale_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("target plan lacks scale selection")
    minimum_fraction = float(selection["minimum_shrinking_area_fraction"])
    apply_gate = _plan_shrinking_area_fraction_gate_applicable(plan)
    recomputed: list[dict[str, Any]] = []
    for goal, case in zip(HORIZONTAL_GOALS, source_cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"goal-area case {goal} is invalid")
        if case.get("relation") != goal:
            raise ValueError("goal-area cases are out of order")
        evidence = _goal_area_evidence(case)
        passes = expected_information_gate_pass(
            original_empty=bool(evidence["original_goal_empty"]),
            destination_empty=bool(evidence["destination_goal_empty"]),
            shrinking=bool(evidence["shrinking_direction"]),
            removed_fraction=float(evidence["removed_area_fraction"]),
            minimum_fraction=minimum_fraction,
            apply_shrinking_fraction_gate=apply_gate,
        )
        recomputed.append(
            {
                **evidence,
                "passes_information_gate": passes,
            }
        )
    return recomputed


def _checks_match(
    left: list[Mapping[str, Any]],
    right: list[Mapping[str, Any]],
) -> bool:
    if len(left) != len(right):
        return False
    for left_check, right_check in zip(left, right):
        if dict(left_check) != dict(right_check):
            return False
    return True


def recompile_receipt(
    *,
    source_receipt_path: Path,
    target_plan_path: Path,
    target_plan_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    source_receipt_path = source_receipt_path.resolve()
    target_plan_path = target_plan_path.resolve()
    output_path = output_path.resolve()

    source_receipt, source_bytes, source_sha256 = _load_canonical_json(
        source_receipt_path,
        label=str(source_receipt_path),
    )
    _validate_source_receipt(source_receipt, source_path=source_receipt_path)

    plan_body = target_plan_path.read_bytes()
    plan = json.loads(plan_body)
    if not isinstance(plan, dict):
        raise ValueError(f"{target_plan_path}: plan must be a JSON object")
    if plan_body != canonical_json_bytes(plan):
        raise ValueError(f"{target_plan_path}: JSON is not canonical")
    if sha256_bytes(plan_body) != target_plan_sha256:
        raise ValueError(f"{target_plan_path}: plan sha256 differs from binding")
    validate_plan_payload(plan)
    if plan.get("fixture_id") != "object_pair":
        raise ValueError("target plan fixture_id must be object_pair")

    environment_seed = int(source_receipt["environment_seed"])
    scale = float(source_receipt["scale"])
    registered_seeds = {
        int(seed) for seed in plan.get("registered_env_seeds", [])
    }
    if environment_seed not in registered_seeds:
        raise ValueError("source environment_seed is absent from target plan")
    candidates = {
        float(item)
        for item in plan.get("scale_selection", {}).get(
            "candidate_scales_descending", []
        )
    }
    if scale not in candidates:
        raise ValueError("source scale is absent from target plan")

    source_checks = source_receipt.get("checks")
    assert isinstance(source_checks, list)
    observations = [_observation_from_check(check) for check in source_checks]

    source_goal_area_cases = source_receipt.get("goal_area_cases")
    assert isinstance(source_goal_area_cases, list)
    goal_area_cases = _recompute_goal_area_cases(source_goal_area_cases, plan=plan)

    plan_receipt = {
        "path": str(target_plan_path),
        "sha256": target_plan_sha256,
    }
    receipt = compile_path_seed_receipt(
        plan=plan,
        plan_receipt=plan_receipt,
        environment_seed=environment_seed,
        scale=scale,
        check_observations=observations,
        goal_area_cases=goal_area_cases,
    )
    if not _checks_match(source_checks, receipt["checks"]):
        raise G3GateError("recompiled path checks differ from source evidence")

    for source_case, output_case in zip(source_goal_area_cases, receipt["goal_area_cases"]):
        if _goal_area_evidence(source_case) != _goal_area_evidence(output_case):
            raise G3GateError("recompiled goal-area evidence differs from source")

    if "runtime_identity" in source_receipt:
        receipt["runtime_identity"] = dict(source_receipt["runtime_identity"])
    if "artifacts" in source_receipt:
        receipt["artifacts"] = json.loads(json.dumps(source_receipt["artifacts"]))

    receipt["derivation"] = {
        "kind": "offline_recompile",
        "reason": DERIVATION_REASON,
        "simulator_rerun": False,
        "source_receipt": {
            "path": str(source_receipt_path),
            "sha256": source_sha256,
            "bytes": len(source_bytes),
        },
    }

    validate_path_seed_receipt(receipt, plan=plan)
    _write_exclusive(output_path, receipt)

    return {
        "output_path": str(output_path),
        "source_receipt_sha256": source_sha256,
        "target_plan_sha256": target_plan_sha256,
        "environment_seed": environment_seed,
        "scale": scale,
        "information_gate_passed": receipt["information_gate_passed"],
        "passed": receipt["passed"],
        "simulator_rerun": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--target-plan", type=Path, required=True)
    parser.add_argument("--target-plan-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = recompile_receipt(
        source_receipt_path=args.source_receipt.resolve(),
        target_plan_path=args.target_plan.resolve(),
        target_plan_sha256=args.target_plan_sha256,
        output_path=args.out.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
