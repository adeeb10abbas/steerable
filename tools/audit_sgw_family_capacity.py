"""Bound remaining SGW fixture capacity without releasing or changing a run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.workshops.spatial_grounding_v1.family_campaign import _validate_finite_plan
from experiments.workshops.spatial_grounding_v1.prospective_family_designs import _digest

ROOT = Path(__file__).resolve().parents[1]
INFRA = Path("artifacts/workshops/spatial_grounding_v1/infrastructure")
CAPTURE = "prospective_design_requires_zero_model_capture"
REJECTED = "prospective_design_rejected_geometrically"


def capacity_bound(plan: Mapping[str, Any], outcomes: Mapping[str, bool]) -> dict[str, Any]:
    """Treat every unknown slot and every physical pass as potentially qualified."""
    _validate_finite_plan(plan)
    if (plan.get("family") not in {"HEIGHT", "DIST"} or plan["design_slot_count"] != 100
            or type(plan.get("seed")) is not int):
        raise ValueError("capacity requires a frozen 100-slot HEIGHT/DIST plan and integer seed")
    designs = plan["designs"]
    if any(row.get("side") not in {"left", "right"} or row.get("status") not in {CAPTURE, REJECTED}
           for row in designs):
        raise ValueError("plan contains an unknown side or proposal status")
    eligible = {row["design_id"]: row for row in designs if row["status"] == CAPTURE}
    if set(outcomes) - set(eligible) or any(type(value) is not bool for value in outcomes.values()):
        raise ValueError("terminal outcomes must be boolean and belong to capture-eligible designs")
    pilot = ("left", "right")[plan["seed"] % 2]
    strata = {}
    for side in ("left", "right"):
        side_rows = [row for row in designs if row["side"] == side]
        ids = {row["design_id"] for row in side_rows if row["status"] == CAPTURE}
        accepted = sum(outcomes.get(key) is True for key in ids)
        rejected = sum(outcomes.get(key) is False for key in ids)
        unknown = len(ids) - accepted - rejected
        required = 2 + 12 + int(side == pilot)
        upper_bound = accepted + unknown
        strata[side] = {
            "proposed": len(side_rows),
            "geometric_rejections": len(side_rows) - len(ids),
            "capture_eligible": len(ids),
            "physical_accepted": accepted,
            "physical_rejected": rejected,
            "unresolved_slots": unknown,
            "required_qualified": required,
            "maximum_qualified_possible": upper_bound,
            "additional_rejections_tolerable": max(0, upper_bound - required),
            "unavoidable_shortfall": max(0, required - upper_bound),
        }
    blocked = any(row["unavoidable_shortfall"] for row in strata.values())
    return {
        "family": plan["family"], "plan_sha256": plan["plan_sha256"], "pilot_side": pilot,
        "by_side": strata,
        "status": "mathematically_blocked_by_frozen_stratum_capacity" if blocked
        else "not_ruled_out_not_a_release",
        "release_permitted": False,
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def _binding(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _check_binding(path: Path, binding: Mapping[str, Any]) -> None:
    raw = path.read_bytes()
    if len(raw) != binding["bytes"] or hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError(f"retained file differs from binding: {path}")


def _verification(path: Path) -> dict[str, Any]:
    value = _load(path)
    if (value.get("schema_version") != "sgw-01-family-campaign-verification-v1"
            or value.get("verification_sha256") != _digest(value, "verification_sha256")
            or value.get("model_request_count") != 0 or value.get("behavioral_episode_count") != 0
            or value.get("release_permitted") is not False
            or value.get("status") != "verified_evidence_not_fixture_release"):
        raise ValueError("worker verification identity or zero-model boundary differs")
    return value


def _full_scores(verification: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    trials = verification["trials"]
    expected = {(goal, reset) for goal in (1, -1) for reset in range(3)}
    if len(trials) != 6 or {(row["goal_sign"], row["reset_index"]) for row in trials} != expected:
        raise ValueError("this retained-prefix audit requires six full trials; never pad a partial")
    scores = {(row["goal_sign"], row["reset_index"]): row["recomputed_score"] for row in trials}
    if any(score.get("status") != "valid_model" or score.get("infrastructure_reason") is not None
           or score.get("safety_censored") is not False or type(score.get("requested_success")) is not bool
           or score.get("terminal_step") != 450 for score in scores.values()):
        raise ValueError("an infrastructure, censored or non-full trial cannot become a physical rejection")
    return scores


def audit(prefix_paths: list[Path]) -> dict[str, Any]:
    plan_paths = [ROOT / INFRA / "family-plan-freeze-20260923bm" / f"{family}-plan.json"
                  for family in ("height", "dist")]
    plans = {value["family"]: value for value in map(_load, plan_paths)}
    summary_path = ROOT / INFRA / "family-collector-20260923bu/initial/summary.json"
    summary = _load(summary_path)
    canonical = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if summary["summary_sha256"] != hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest():
        raise ValueError("initial collector summary digest differs")
    outcomes: dict[str, dict[str, bool]] = {family: {} for family in plans}
    evidence = []

    def add(family: str, design_id: str, side: str, accepted: bool, verified: Mapping[str, Any]) -> None:
        if family not in plans or design_id in outcomes[family]:
            raise ValueError("unknown family or duplicate terminal design")
        design = next((row for row in plans[family]["designs"] if row["design_id"] == design_id), None)
        if (design is None or design["side"] != side or design["status"] != CAPTURE
                or verified["family"] != family or verified["design_id"] != design_id):
            raise ValueError("terminal design differs from frozen side, family or identity")
        scores = _full_scores(verified)
        if type(accepted) is not bool or accepted != all(score["requested_success"] for score in scores.values()):
            raise ValueError("physical classification differs from verified full-trial scores")
        outcomes[family][design_id] = accepted
        evidence.append({
            "family": family, "design_id": design_id, "side": side, "accepted": accepted,
            "verification_sha256": verified["verification_sha256"],
            "failed_trials": [
                {"goal_sign": goal, "reset_index": reset, "failure_stage": score["failure_stage"]}
                for (goal, reset), score in sorted(scores.items()) if not score["requested_success"]
            ],
        })

    for row in summary["smoke"]:
        if row["outcome"] != "full_six":
            raise ValueError("smoke is not a completed six-trial candidate")
        path = summary_path.parent / "smoke-rechecks" / f"{row['family'].lower()}-{row['slot_index']:03d}.json"
        _check_binding(path, row["recheck"])
        verified = _verification(path)
        if (verified["verification_sha256"] != row["verification_sha256"]
                or verified["qualification"] != row["qualification"]):
            raise ValueError("smoke verification or qualification binding differs")
        add(row["family"], row["design_id"], row["side"], row["accepted"], verified)
    if len(evidence) != 4:
        raise ValueError("expected exactly four independently rechecked smoke candidates")

    for path in sorted(prefix_paths):
        manifest = _load(path)
        if (manifest.get("schema_version") != "sgw-01-completed-native-partition-prefix-v1"
                or manifest.get("model_requests") != 0 or manifest.get("behavioral_episodes") != 0
                or manifest.get("release_permitted") is not False):
            raise ValueError("prefix identity or zero-model boundary differs")
        for row in manifest["records"]:
            for binding in row["retained_files"]:
                retained = (path.parent / binding["export_path"]).resolve()
                retained.relative_to(path.parent.resolve())
                _check_binding(retained, binding)
            slot = Path("workers") / str(row["rank"]) / "slots" / row["slot"]
            verified = _verification(path.parent / slot / "family_verification.json")
            candidate_path = path.parent / slot / "candidate.json"
            _check_binding(candidate_path, verified["materialized_candidate"])
            candidate = _load(candidate_path)
            completion = _load(path.parent / "workers" / str(row["rank"]) / "completed" / f"{row['slot']}.json")
            executor = _load(path.parent / slot / "executor-receipt.json")
            if (completion["result"] != executor or executor["verification_sha256"] != verified["verification_sha256"]
                    or row["qualification"] != verified["qualification"]):
                raise ValueError("prefix completion/executor/qualification binding differs")
            scores = _full_scores(verified)
            projected = row["trials"]
            if (len(projected) != 6 or row["trial_count"] != 6
                    or {(trial["goal_sign"], trial["reset_index"]) for trial in projected} != set(scores)):
                raise ValueError("projected trials differ from verification")
            for trial in projected:
                score = scores[(trial["goal_sign"], trial["reset_index"])]
                if (trial["score"] != score or trial["passed"] is not score["requested_success"]
                        or trial["actions_executed"] != 450 or trial["state_count"] != 451):
                    raise ValueError("projected full-trial score or shape differs")
            passed = sum(trial["passed"] for trial in projected)
            expected_status = ("accepted" if passed == 6 else "rejected") + "_model_blind_fixture_candidate"
            if row["passed_trials"] != passed or row["status"] != expected_status:
                raise ValueError("prefix classification differs from full-trial evidence")
            family = candidate["family"]
            metadata = candidate["metadata"]
            if family not in plans or metadata["prospective_plan_sha256"] != plans[family]["plan_sha256"]:
                raise ValueError("candidate plan binding differs")
            side = metadata["upper_support_side" if family == "HEIGHT" else "bowl_side"]
            add(family, metadata["prospective_design_id"], side, passed == 6, verified)

    return {
        "schema_version": "sgw-01-frozen-family-capacity-audit-v1",
        "inputs": [_binding(path) for path in plan_paths + [summary_path] + sorted(prefix_paths)],
        "families": {family: capacity_bound(plan, outcomes[family]) for family, plan in plans.items()},
        "terminal_evidence": sorted(evidence, key=lambda row: row["design_id"]),
        "fixture_release_permitted": False, "model_release_permitted": False,
        "model_requests": 0, "behavioral_episodes": 0, "live_workers_modified": False,
        "claim_boundary": (
            "A necessary capacity bound conditioned on retained worker-verified classifications, including "
            "independently rechecked smoke. Unknown slots and physical passes are optimistically counted "
            "as potentially qualified. No historical-coverage, deduplication, runtime or learned release "
            "is inferred. This is not a rerun of raw verification or the pending final collector."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--prefix", type=Path, action="append", required=True, help="Retained prefix manifest; repeat per wave")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit([path.resolve() for path in args.prefix])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
