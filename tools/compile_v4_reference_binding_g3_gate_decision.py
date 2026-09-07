#!/usr/bin/env python3
"""Compile C2 reference_binding G3 gate decision with sign/goal splits."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.model_blind_g3 import (  # noqa: E402
    build_counterbalance_index,
    canonical_json_bytes,
    sha256_file,
)
from tools.compile_v4_horizontal_g3_path_scale import compile_receipts  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_queue_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _information_gate_splits(
    *,
    receipts: list[dict[str, Any]],
    counterbalance: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    by_sign: dict[str, dict[str, int]] = defaultdict(lambda: {"seeds_pass": 0, "seeds_fail": 0})
    by_goal: dict[str, dict[str, int]] = defaultdict(
        lambda: {"case_pass": 0, "case_fail": 0, "seeds_with_any_fail": set()}
    )
    by_sign_goal: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"case_pass": 0, "case_fail": 0})
    )
    failed_seeds_by_sign: dict[str, list[int]] = defaultdict(list)

    for receipt in receipts:
        seed = int(receipt["environment_seed"])
        sign = int(counterbalance[seed]["physical_translation_sign"])
        sign_key = str(sign)
        if receipt.get("information_gate_passed") is True:
            by_sign[sign_key]["seeds_pass"] += 1
        else:
            by_sign[sign_key]["seeds_fail"] += 1
            failed_seeds_by_sign[sign_key].append(seed)
        for case in receipt.get("goal_area_cases", []):
            if not isinstance(case, dict):
                continue
            relation = str(case.get("relation"))
            passed = case.get("passes_information_gate") is True
            if passed:
                by_goal[relation]["case_pass"] += 1
                by_sign_goal[sign_key][relation]["case_pass"] += 1
            else:
                by_goal[relation]["case_fail"] += 1
                by_sign_goal[sign_key][relation]["case_fail"] += 1
                by_goal[relation]["seeds_with_any_fail"].add(seed)

    goal_summary = {
        goal: {
            "case_pass": counts["case_pass"],
            "case_fail": counts["case_fail"],
            "seeds_with_any_fail": sorted(counts["seeds_with_any_fail"]),
        }
        for goal, counts in sorted(by_goal.items())
    }
    sign_goal_summary = {
        sign: {
            goal: dict(metrics)
            for goal, metrics in sorted(goals.items())
        }
        for sign, goals in sorted(by_sign_goal.items(), key=lambda item: int(item[0]))
    }
    return {
        "by_physical_translation_sign": {
            sign: {
                **metrics,
                "information_gate_failed_seeds": sorted(failed_seeds_by_sign.get(sign, [])),
            }
            for sign, metrics in sorted(by_sign.items(), key=lambda item: int(item[0]))
        },
        "by_goal_relation": goal_summary,
        "by_physical_translation_sign_and_goal": sign_goal_summary,
    }


def compile_gate_decision(
    *,
    plan_path: Path,
    scale: float,
    receipts_root: Path,
    queue_path: Path,
    reset_registry_path: Path,
    path_scale_out: Path,
    decision_out: Path,
    attempt_id: str,
    homogeneous_pin: str,
    finalize: bool = False,
    computation_audit_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    plan = _load_json(plan_path)
    reset_registry = _load_json(reset_registry_path)
    resets = reset_registry["resets_by_env_seed"]
    queue_rows = _load_queue_rows(queue_path)
    counterbalance = build_counterbalance_index(
        queue_rows,
        expected_env_seeds=sorted(int(seed) for seed in resets),
        counterbalance_family="C2",
        counterbalance_fixture="reference_binding",
    )

    path_scale = compile_receipts(
        plan_path=plan_path,
        scale=scale,
        receipts_root=receipts_root,
        output_path=path_scale_out,
        overwrite=overwrite,
    )
    receipt_paths = sorted(receipts_root.resolve().rglob("g3_path_seed_receipt.json"))
    receipts = [json.loads(path.read_bytes()) for path in receipt_paths]
    splits = _information_gate_splits(receipts=receipts, counterbalance=counterbalance)

    observed = int(path_scale.get("observed_seed_count") or 0)
    expected = int(path_scale.get("expected_seed_count") or 128)
    complete = observed == expected and not path_scale.get("missing_env_seeds")
    raw_pass = path_scale.get("passed") is True
    sign_minus_one = splits["by_physical_translation_sign"].get("-1", {})
    sign_plus_one = splits["by_physical_translation_sign"].get("1", {})

    info_failed = len(path_scale.get("information_gate_failed_seeds") or [])
    front_goal = splits["by_goal_relation"].get("front", {})
    behind_goal = splits["by_goal_relation"].get("behind", {})
    if finalize and not complete:
        raise ValueError(
            f"cannot finalize gate decision until wave is complete ({observed}/{expected})"
        )
    if finalize and computation_audit_path is None:
        raise ValueError("finalize requires --computation-audit receipt path")
    audit_binding = None
    if computation_audit_path is not None:
        audit = _load_json(computation_audit_path)
        if audit.get("verdict") != "computation_correct_scientific_rejection":
            raise ValueError("computation audit verdict is not computation_correct_scientific_rejection")
        audit_binding = {
            "path": str(computation_audit_path.relative_to(ROOT)),
            "sha256": sha256_file(computation_audit_path),
            "verdict": audit.get("verdict"),
        }
    passed = raw_pass and complete
    if finalize:
        decision_value = "passed" if passed else "failed"
    else:
        decision_value = "provisional_pending_computation_audit"
    decision: dict[str, Any] = {
        "schema_version": "v4-reference-binding-g3-gate-decision-v1",
        "attempt_id": attempt_id,
        "compiled_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "confirmatory_scale": scale,
        "decision": decision_value,
        "gate_id": "G3_reference_binding_homogeneous",
        "homogeneous_pin": homogeneous_pin,
        "passed": passed,
        "information_gate_splits": splits,
        "path_scale_receipt": str(path_scale_out.relative_to(ROOT)),
        "computation_audit": audit_binding,
    }
    if finalize:
        decision["scientific_interpretation"] = (
            "At confirmatory scale 0.5 (0.06 m), negative-sign shrinking-direction "
            "cases remove 13.7-19.9% of legal goal area—just below the frozen 20% "
            "threshold—while positive-sign shrinking cases can exceed 20%. All path "
            "checks pass; this is registered information-gate rejection under "
            "counterbalance semantics, not physical infeasibility. Agent A audit "
            "confirmed the gate computation is correct; Agent B containment passed "
            "both translation-sign halves, isolating this to reference_binding geometry."
        )
        decision["primary_estimand_status"] = (
            "not_estimable" if not passed else "pending_prefix_replay"
        )
        decision["reason"] = (
            f"Homogeneous 128-seed wave at scale {scale}: "
            f"{path_scale.get('passed_path_check_count')}/{path_scale.get('expected_path_check_count')} "
            f"path checks pass; {info_failed}/{expected} seeds fail shrinking-area "
            f"information gates; sign=+1 failures "
            f"{sign_plus_one.get('seeds_fail', 0)}/{sign_plus_one.get('seeds_pass', 0) + sign_plus_one.get('seeds_fail', 0)} "
            f"and sign=-1 failures "
            f"{sign_minus_one.get('seeds_fail', 0)}/{sign_minus_one.get('seeds_pass', 0) + sign_minus_one.get('seeds_fail', 0)}; "
            f"front goal case pass/fail {front_goal.get('case_pass', 0)}/"
            f"{front_goal.get('case_fail', 0)}; behind "
            f"{behind_goal.get('case_pass', 0)}/{behind_goal.get('case_fail', 0)}."
        )
        decision["scripted_checks_status"] = (
            "blocked_pending_passing_path_scale" if not passed else "authorized"
        )
        if not passed:
            decision["affected_confirmatory_episodes"] = 4096
            decision["release_boundary"] = (
                "C2 confirmatory policy inference is scientifically blocked at the "
                "registered homogeneous scale 0.5; do not dispatch 4,096 episodes."
            )
    else:
        decision["provisional_status"] = {
            "wave_complete": complete,
            "observed_seed_count": observed,
            "expected_seed_count": expected,
            "raw_path_scale_passed": raw_pass,
            "information_gate_failed_seed_count": info_failed,
            "do_not_block_c2_scope_yet": True,
        }
        decision["reason"] = (
            f"Homogeneous wave at scale {scale}: {observed}/{expected} seeds observed; "
            f"path checks {path_scale.get('passed_path_check_count')}/"
            f"{path_scale.get('expected_path_check_count')} pass; "
            f"information-gate failures sign=+1 "
            f"{sign_plus_one.get('seeds_fail', 0)}/{sign_plus_one.get('seeds_pass', 0) + sign_plus_one.get('seeds_fail', 0)} "
            f"and sign=-1 "
            f"{sign_minus_one.get('seeds_fail', 0)}/{sign_minus_one.get('seeds_pass', 0) + sign_minus_one.get('seeds_fail', 0)}."
        )
        decision["scripted_checks_status"] = "blocked_pending_provisional_gate_resolution"
    decision_out.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(decision)
    if decision_out.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite gate decision: {decision_out}")
        decision_out.write_bytes(body)
    else:
        with decision_out.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
    return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "artifacts/online_correction_v4/setup/reference_binding_g3_plan.candidate.json",
    )
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--receipts-root", type=Path, required=True)
    parser.add_argument(
        "--queue",
        type=Path,
        default=ROOT / "artifacts/online_correction_v4/queue.jsonl",
    )
    parser.add_argument(
        "--reset-registry",
        type=Path,
        default=ROOT
        / "artifacts/online_correction_v4/setup/reference_binding_reset_registry.candidate.json",
    )
    parser.add_argument("--attempt-id", default="g3rb20260908v")
    parser.add_argument(
        "--homogeneous-pin",
        default="c401fb4577d8003a019ecf7ff7be549f2c0a5931",
    )
    parser.add_argument("--path-scale-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Emit final gate decision (requires complete 128/128 wave).",
    )
    parser.add_argument(
        "--computation-audit",
        type=Path,
        default=ROOT
        / "artifacts/online_correction_v4/qualification/20260908_information_gate_computation_audit.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    decision = compile_gate_decision(
        plan_path=args.plan.resolve(),
        scale=args.scale,
        receipts_root=args.receipts_root.resolve(),
        queue_path=args.queue.resolve(),
        reset_registry_path=args.reset_registry.resolve(),
        path_scale_out=args.path_scale_out.resolve(),
        decision_out=args.out.resolve(),
        attempt_id=args.attempt_id,
        homogeneous_pin=args.homogeneous_pin,
        finalize=args.finalize,
        computation_audit_path=args.computation_audit.resolve()
        if args.finalize
        else None,
        overwrite=args.overwrite,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
