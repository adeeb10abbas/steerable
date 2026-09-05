"""Offline V4 confirmatory analysis and compact export builders."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

HORIZONTAL_GOALS: tuple[str, ...] = ("left", "right", "front", "behind")
MAIN_POLICIES: tuple[str, ...] = ("cosmos3_nano_droid", "pi05_droid")
C1_INTERACTION_SCENARIOS: tuple[str, ...] = ("original_sham", "move_stop")
C2_SCENARIOS: tuple[str, ...] = ("original_sham", "move_A")
PRIMARY_CONTRASTS: tuple[str, ...] = (
    "C1_wording_x_motion_success_per_main_policy",
    "C2_reference_x_motion_goal_improvement_per_main_policy",
)
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_ANALYSIS_SEED = 20260905
DEFAULT_ALPHA = 0.05
PRIMARY_TEST_COUNT = 4
DEFAULT_PRIMARY_RESPONSE_HORIZON_S = 2.0
DEFAULT_RESPONSE_ANCHOR = "t_event_planned+2.0s"
VERIFIED_COMMON_PREFIX_MODES: frozenset[str] = frozenset(
    {"deterministic_fresh_session_replay", "qualified_full_state_snapshot"}
)
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class AnalysisError(ValueError):
    """Raised when ledger or estimator inputs violate the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _load_planning_helper():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("online_correction_v4", root / "tools/online_correction_v4.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AnalysisError(f"{path}:{line_no}: JSONL record must be an object")
        rows.append(value)
    return rows


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(value: Any) -> str:
    return digest_bytes(canonical_bytes(value))


def load_campaign_config(config_path: Path) -> tuple[dict[str, Any], str]:
    helper = _load_planning_helper()
    return helper.load_json(config_path)


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    return read_jsonl(manifest_path)


def load_accepted_ledger(results_path: Path) -> list[dict[str, Any]]:
    return read_jsonl(results_path)


def primary_response_horizon_s(config: Mapping[str, Any] | None) -> float:
    if config is None:
        return DEFAULT_PRIMARY_RESPONSE_HORIZON_S
    timing = config.get("timing", {})
    value = timing.get("primary_response_horizon_s", DEFAULT_PRIMARY_RESPONSE_HORIZON_S)
    return float(value)


def primary_response_anchor(config: Mapping[str, Any] | None) -> str:
    horizon = primary_response_horizon_s(config)
    return f"t_event_planned+{horizon:.1f}s"


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(HEX64_RE.fullmatch(value))


def _c2_response_outcome_errors(
    outcome: Mapping[str, Any],
    *,
    label: str,
    config: Mapping[str, Any] | None,
    require_explicit_response: bool,
) -> list[str]:
    errors: list[str] = []
    horizon = primary_response_horizon_s(config)
    anchor = primary_response_anchor(config)
    if require_explicit_response and "response_goal_violation_capped_m" not in outcome:
        errors.append(
            f"{label}: C2 requires explicit outcome.response_goal_violation_capped_m; "
            "terminal goal_violation_capped_m is not accepted for primary response scoring"
        )
        return errors
    value = outcome.get("response_goal_violation_capped_m")
    if value is None:
        errors.append(f"{label}: missing outcome.response_goal_violation_capped_m")
        return errors
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        errors.append(f"{label}: response_goal_violation_capped_m must be finite and nonnegative")
    response_horizon = outcome.get("response_horizon_s")
    if response_horizon != horizon:
        errors.append(
            f"{label}: response_horizon_s must equal configured primary horizon {horizon:g}"
        )
    if outcome.get("response_anchor") != anchor:
        errors.append(f"{label}: response_anchor must equal {anchor!r}")
    if outcome.get("response_goal_set_branch") != "move":
        errors.append(f"{label}: response_goal_set_branch must be 'move'")
    if not _valid_sha256(outcome.get("response_goal_set_hash_sha256")):
        errors.append(f"{label}: response_goal_set_hash_sha256 must be a 64-char lowercase hex digest")
    projection = outcome.get("response_projection", "planar")
    if projection != "planar":
        errors.append(f"{label}: response_projection must be 'planar' for primary C2 scoring")
    if not _valid_sha256(outcome.get("response_scorer_sha256")):
        errors.append(f"{label}: response_scorer_sha256 must be a 64-char lowercase hex digest")
    return errors


def _c2_prefix_verification_errors(record: Mapping[str, Any], *, label: str) -> list[str]:
    errors: list[str] = []
    mode = record.get("common_prefix_verification_mode")
    if mode not in VERIFIED_COMMON_PREFIX_MODES:
        errors.append(
            f"{label}: common_prefix_verification_mode must be one of "
            f"{sorted(VERIFIED_COMMON_PREFIX_MODES)}"
        )
    if not _valid_sha256(record.get("common_prefix_verification_receipt_sha256")):
        errors.append(f"{label}: common_prefix_verification_receipt_sha256 must be a 64-char digest")
    if not _valid_sha256(record.get("common_prefix_identity_hash_sha256")):
        errors.append(f"{label}: common_prefix_identity_hash_sha256 must be a 64-char digest")
    return errors


def validate_c2_result_record(
    record: Mapping[str, Any],
    *,
    label: str,
    config: Mapping[str, Any] | None,
) -> list[str]:
    errors = _c2_prefix_verification_errors(record, label=label)
    errors.extend(
        _c2_response_outcome_errors(
            record.get("outcome", {}),
            label=label,
            config=config,
            require_explicit_response=True,
        )
    )
    if record.get("trigger_eligible") is not True:
        errors.append(f"{label}: C2 primary analysis requires trigger_eligible=true on sham and move branches")
    return errors


def validate_accepted_ledger(
    manifest: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate accepted rows and extend the planning helper audit for analysis fields."""

    helper = _load_planning_helper()
    report = helper.check_results(list(manifest), list(results))
    errors = list(report.get("errors", []))
    manifest_by_id = {row["episode_id"]: row for row in manifest}
    accepted: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(results, 1):
        if row.get("status") != "valid":
            continue
        eid = row.get("episode_id")
        if eid not in manifest_by_id:
            continue
        if eid in accepted:
            continue
        accepted[eid] = row
        family = manifest_by_id[eid]["family"]
        label = f"result {index} ({eid})"
        if family == "C2":
            errors.extend(validate_c2_result_record(row, label=label, config=config))
        if family == "C1" and type(row.get("success")) is not bool:
            errors.append(f"{label}: success must be boolean for C1 analysis")
    report = {
        **report,
        "ok": not errors,
        "errors": errors,
        "analysis_scope": "accepted_ledger_loading_and_coverage_reconciliation",
    }
    if config is not None:
        report["campaign_id"] = config.get("campaign_id")
        report["primary_response_horizon_s"] = primary_response_horizon_s(config)
        report["primary_response_anchor"] = primary_response_anchor(config)
    return report


def robot_stack_for_policy(config: Mapping[str, Any], policy_id: str) -> str:
    policy = config["policies"][policy_id]
    return str(policy["stack"])


def _cell_lookup(
    manifest: Sequence[Mapping[str, Any]],
    accepted: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in manifest:
        factors = row["factors"]
        key = (
            row["family"],
            factors["policy"],
            row["block_id"],
            factors["goal"],
            factors.get("wording"),
            factors.get("scenario"),
            factors.get("named_reference"),
        )
        lookup[key] = {
            "manifest": row,
            "result": accepted.get(row["episode_id"]),
            "episode_id": row["episode_id"],
        }
    return lookup


def _success_value(record: Mapping[str, Any] | None) -> float | None:
    if record is None or record.get("status") != "valid":
        return None
    success = record.get("success")
    if type(success) is not bool:
        return None
    return 1.0 if success else 0.0


def _c2_primary_response_distance(
    record: Mapping[str, Any] | None,
    *,
    config: Mapping[str, Any] | None = None,
) -> float | None:
    if record is None or record.get("status") != "valid":
        return None
    outcome = record.get("outcome", {})
    if validate_c2_result_record(record, label="record", config=config):
        return None
    value = outcome.get("response_goal_violation_capped_m")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return None
    if value < 0:
        return None
    return float(value)


def _c2_pair_prefix_consistent(sham: Mapping[str, Any], move: Mapping[str, Any]) -> bool:
    if sham.get("prefix_group_id") != move.get("prefix_group_id"):
        return False
    for key in (
        "common_prefix_verification_mode",
        "common_prefix_verification_receipt_sha256",
        "common_prefix_identity_hash_sha256",
    ):
        if sham.get(key) != move.get(key):
            return False
    sham_goal_hash = sham.get("outcome", {}).get("response_goal_set_hash_sha256")
    move_goal_hash = move.get("outcome", {}).get("response_goal_set_hash_sha256")
    return sham_goal_hash == move_goal_hash and sham_goal_hash is not None


def _c2_branch_pair_valid(
    lookup: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    policy: str,
    block_id: int,
    goal: str,
    named_reference: str,
    config: Mapping[str, Any] | None = None,
) -> bool:
    sham_cell = lookup.get(("C2", policy, block_id, goal, "direct", "original_sham", named_reference))
    move_cell = lookup.get(("C2", policy, block_id, goal, "direct", "move_A", named_reference))
    if sham_cell is None or move_cell is None:
        return False
    sham = sham_cell.get("result")
    move = move_cell.get("result")
    if sham is None or move is None or sham.get("status") != "valid" or move.get("status") != "valid":
        return False
    if sham.get("trigger_eligible") is not True or move.get("trigger_eligible") is not True:
        return False
    if sham_cell["manifest"]["prefix_group_id"] != move_cell["manifest"]["prefix_group_id"]:
        return False
    if not _c2_pair_prefix_consistent(sham, move):
        return False
    if validate_c2_result_record(sham, label="sham", config=config):
        return False
    if validate_c2_result_record(move, label="move", config=config):
        return False
    if _c2_primary_response_distance(sham, config=config) is None:
        return False
    if _c2_primary_response_distance(move, config=config) is None:
        return False
    return True


def c2_goal_selectivity(
    lookup: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    policy: str,
    block_id: int,
    goal: str,
    config: Mapping[str, Any] | None = None,
) -> float | None:
    if not _c2_branch_pair_valid(
        lookup, policy=policy, block_id=block_id, goal=goal, named_reference="A", config=config
    ):
        return None
    if not _c2_branch_pair_valid(
        lookup, policy=policy, block_id=block_id, goal=goal, named_reference="B", config=config
    ):
        return None
    sham_a = lookup[("C2", policy, block_id, goal, "direct", "original_sham", "A")]["result"]
    move_a = lookup[("C2", policy, block_id, goal, "direct", "move_A", "A")]["result"]
    sham_b = lookup[("C2", policy, block_id, goal, "direct", "original_sham", "B")]["result"]
    move_b = lookup[("C2", policy, block_id, goal, "direct", "move_A", "B")]["result"]
    h_a = _c2_primary_response_distance(sham_a, config=config) - _c2_primary_response_distance(move_a, config=config)
    h_b = _c2_primary_response_distance(sham_b, config=config) - _c2_primary_response_distance(move_b, config=config)
    assert h_a is not None and h_b is not None
    return h_a - h_b


def c1_goal_interaction(
    lookup: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    policy: str,
    block_id: int,
    goal: str,
) -> float | None:
    direct_sham = _success_value(lookup.get(("C1", policy, block_id, goal, "direct", "original_sham", "single"), {}).get("result"))
    direct_move = _success_value(lookup.get(("C1", policy, block_id, goal, "direct", "move_stop", "single"), {}).get("result"))
    inverse_sham = _success_value(lookup.get(("C1", policy, block_id, goal, "inverse", "original_sham", "single"), {}).get("result"))
    inverse_move = _success_value(lookup.get(("C1", policy, block_id, goal, "inverse", "move_stop", "single"), {}).get("result"))
    if None in (direct_sham, direct_move, inverse_sham, inverse_move):
        return None
    return (inverse_move - inverse_sham) - (direct_move - direct_sham)


def c1_block_interaction(
    lookup: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    policy: str,
    block_id: int,
) -> float | None:
    interactions = [c1_goal_interaction(lookup, policy=policy, block_id=block_id, goal=goal) for goal in HORIZONTAL_GOALS]
    if any(value is None for value in interactions):
        return None
    return statistics.fmean(interactions)


def _jackknife_se(values: Callable[[list[int]], float | None], blocks: Sequence[int]) -> float | None:
    blocks = list(blocks)
    n = len(blocks)
    if n < 2:
        return None
    estimates: list[float] = []
    for index in range(n):
        reduced = blocks[:index] + blocks[index + 1 :]
        estimate = values(reduced)
        if estimate is None or not math.isfinite(estimate):
            return None
        estimates.append(float(estimate))
    if not estimates:
        return None
    mean_loo = statistics.fmean(estimates)
    variance = sum((value - mean_loo) ** 2 for value in estimates)
    se = math.sqrt(((n - 1) / n) * variance)
    if se == 0.0:
        return 0.0
    return se


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AnalysisError("percentile requested from empty distribution")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class InferenceResult:
    contrast_key: str
    policy_id: str
    robot_stack: str
    estimand: str
    point_estimate: float | None
    ci_low: float | None
    ci_high: float | None
    standard_error: float | None
    p_value: float | None
    test_status: str
    not_estimable_reason: str | None
    n_blocks: int
    n_effective_blocks: int
    bootstrap_resamples: int
    bootstrap_seed: int
    undefined_bootstrap_resamples: int
    zero_or_undefined_se_resamples: int
    holm_adjusted_p: float | None
    holm_rejected: bool
    descriptive: dict[str, Any]


def _c1_estimator(
    lookup: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    policy: str,
    block_ids: Sequence[int],
) -> tuple[float | None, int]:
    effects = [c1_block_interaction(lookup, policy=policy, block_id=block_id) for block_id in block_ids]
    complete = [value for value in effects if value is not None]
    if not complete:
        return None, 0
    return statistics.fmean(complete), len(complete)


def _c2_estimator(
    lookup: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    policy: str,
    block_ids: Sequence[int],
    config: Mapping[str, Any] | None = None,
) -> tuple[float | None, dict[str, Any]]:
    goal_values: dict[str, list[float]] = {goal: [] for goal in HORIZONTAL_GOALS}
    eligible_blocks: set[int] = set()
    for block_id in block_ids:
        block_eligible = False
        for goal in HORIZONTAL_GOALS:
            contrast = c2_goal_selectivity(
                lookup, policy=policy, block_id=block_id, goal=goal, config=config
            )
            if contrast is not None:
                goal_values[goal].append(contrast)
                block_eligible = True
        if block_eligible:
            eligible_blocks.add(block_id)
    goal_means = {goal: statistics.fmean(values) if values else None for goal, values in goal_values.items()}
    if any(goal_means[goal] is None for goal in HORIZONTAL_GOALS):
        return None, {
            "goal_means": goal_means,
            "goal_eligible_counts": {goal: len(values) for goal, values in goal_values.items()},
            "eligible_blocks": len(eligible_blocks),
        }
    aggregate = statistics.fmean(goal_means[goal] for goal in HORIZONTAL_GOALS)
    return aggregate, {
        "goal_means": goal_means,
        "goal_eligible_counts": {goal: len(values) for goal, values in goal_values.items()},
        "eligible_blocks": len(eligible_blocks),
    }


def studentized_reset_block_inference(
    *,
    contrast_key: str,
    policy_id: str,
    robot_stack: str,
    estimand: str,
    block_ids: Sequence[int],
    estimator: Callable[[Sequence[int]], tuple[float | None, dict[str, Any] | int]],
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_ANALYSIS_SEED,
) -> InferenceResult:
    blocks = sorted(set(block_ids))
    _require(blocks, f"{contrast_key}/{policy_id}: no independent reset blocks")

    def scalar_estimator(sample: Sequence[int]) -> float | None:
        value, _ = estimator(sample)
        return value

    point, descriptive_extra = estimator(blocks)
    descriptive: dict[str, Any]
    if isinstance(descriptive_extra, dict):
        descriptive = descriptive_extra
        n_effective = int(descriptive.get("eligible_blocks", len(blocks)))
    else:
        descriptive = {"complete_blocks": int(descriptive_extra)}
        n_effective = int(descriptive_extra)

    se_obs = _jackknife_se(scalar_estimator, blocks)
    undefined = 0
    bad_se = 0
    bootstrap_points: list[float] = []
    t_stars: list[float] = []

    rng = random.Random(seed)
    n = len(blocks)
    t_obs: float | None = None
    if point is not None and se_obs not in (None, 0.0):
        t_obs = point / se_obs

    for _ in range(bootstrap_resamples):
        sample = [blocks[rng.randrange(n)] for _ in range(n)]
        theta_b, _ = estimator(sample)

        def replicate_estimator(reduced: list[int]) -> float | None:
            value, _ = estimator(reduced)
            return value

        se_b = _jackknife_se(replicate_estimator, sample)
        if theta_b is None or not math.isfinite(theta_b):
            undefined += 1
            continue
        bootstrap_points.append(float(theta_b))
        if se_b in (None, 0.0) or point is None or se_obs in (None, 0.0):
            bad_se += 1
            continue
        t_stars.append((float(theta_b) - float(point)) / se_b)

    not_estimable_reason: str | None = None
    p_value: float | None = None
    test_status = "estimable"
    ci_low: float | None = None
    ci_high: float | None = None

    if point is None:
        test_status = "not_estimable"
        not_estimable_reason = "observed estimator undefined"
    elif se_obs in (None, 0.0):
        test_status = "not_estimable"
        not_estimable_reason = "observed jackknife standard error is zero or undefined"
    elif undefined > 0 or bad_se > 0:
        test_status = "not_estimable"
        not_estimable_reason = (
            f"bootstrap resamples with undefined estimator={undefined}; "
            f"zero_or_undefined_se={bad_se}"
        )
    elif t_obs is None:
        test_status = "not_estimable"
        not_estimable_reason = "studentized test statistic undefined"
    else:
        extreme = sum(1 for value in t_stars if abs(value) >= abs(t_obs))
        p_value = (1 + extreme) / (bootstrap_resamples + 1)
        if bootstrap_points:
            ci_low = _percentile(bootstrap_points, 0.025)
            ci_high = _percentile(bootstrap_points, 0.975)

    return InferenceResult(
        contrast_key=contrast_key,
        policy_id=policy_id,
        robot_stack=robot_stack,
        estimand=estimand,
        point_estimate=point,
        ci_low=ci_low,
        ci_high=ci_high,
        standard_error=se_obs,
        p_value=p_value,
        test_status=test_status,
        not_estimable_reason=not_estimable_reason,
        n_blocks=len(blocks),
        n_effective_blocks=n_effective,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=seed,
        undefined_bootstrap_resamples=undefined,
        zero_or_undefined_se_resamples=bad_se,
        holm_adjusted_p=None,
        holm_rejected=False,
        descriptive=descriptive,
    )


def holm_adjust_primary_tests(results: Sequence[InferenceResult]) -> list[InferenceResult]:
    _require(len(results) == PRIMARY_TEST_COUNT, f"primary family requires {PRIMARY_TEST_COUNT} slots")
    bookkeeping = []
    for item in results:
        if item.test_status != "estimable" or item.p_value is None:
            bookkeeping.append(1.0)
        else:
            bookkeeping.append(float(item.p_value))
    order = sorted(range(len(results)), key=lambda index: bookkeeping[index])
    adjusted = [1.0] * len(results)
    running = 0.0
    for rank, index in enumerate(order):
        multiplier = len(results) - rank
        raw = bookkeeping[index] * multiplier
        running = max(running, raw)
        adjusted[index] = min(1.0, running)
    updated: list[InferenceResult] = []
    for index, item in enumerate(results):
        holm_p = adjusted[index]
        rejected = item.test_status == "estimable" and item.p_value is not None and holm_p <= DEFAULT_ALPHA
        updated.append(
            InferenceResult(
                **{
                    **item.__dict__,
                    "holm_adjusted_p": holm_p if item.test_status == "estimable" and item.p_value is not None else None,
                    "holm_rejected": rejected,
                }
            )
        )
    return updated


def build_c1_paired_contrast_rows(
    lookup: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    policy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block_ids = sorted({key[2] for key in lookup if key[0] == "C1" and key[1] == policy})
    for block_id in block_ids:
        goal_effects: dict[str, float] = {}
        episode_ids: list[str] = []
        complete = True
        for goal in HORIZONTAL_GOALS:
            interaction = c1_goal_interaction(lookup, policy=policy, block_id=block_id, goal=goal)
            if interaction is None:
                complete = False
                break
            goal_effects[goal] = interaction
            for wording in ("direct", "inverse"):
                for scenario in C1_INTERACTION_SCENARIOS:
                    cell = lookup.get(("C1", policy, block_id, goal, wording, scenario, "single"))
                    if cell is not None:
                        episode_ids.append(cell["episode_id"])
        block_effect = statistics.fmean(goal_effects.values()) if complete else None
        rows.append(
            {
                "family": "C1",
                "contrast_key": PRIMARY_CONTRASTS[0],
                "policy_id": policy,
                "block_id": block_id,
                "complete_block": complete,
                "block_interaction_success_pp": block_effect,
                "goal_interactions_success_pp": goal_effects,
                "contributing_episode_ids": sorted(set(episode_ids)),
            }
        )
    return rows


def build_c2_paired_contrast_rows(
    lookup: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    policy: str,
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block_ids = sorted({key[2] for key in lookup if key[0] == "C2" and key[1] == policy})
    for block_id in block_ids:
        per_goal: dict[str, float | None] = {}
        episode_ids: list[str] = []
        for goal in HORIZONTAL_GOALS:
            per_goal[goal] = c2_goal_selectivity(
                lookup, policy=policy, block_id=block_id, goal=goal, config=config
            )
            for named_reference in ("A", "B"):
                for scenario in C2_SCENARIOS:
                    cell = lookup.get(("C2", policy, block_id, goal, "direct", scenario, named_reference))
                    if cell is not None:
                        episode_ids.append(cell["episode_id"])
        rows.append(
            {
                "family": "C2",
                "contrast_key": PRIMARY_CONTRASTS[1],
                "policy_id": policy,
                "block_id": block_id,
                "goal_selectivity_m": per_goal,
                "eligible_for_aggregate": all(value is not None for value in per_goal.values()),
                "contributing_episode_ids": sorted(set(episode_ids)),
            }
        )
    return rows


def reconcile_coverage(
    manifest: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest_by_id = {row["episode_id"]: row for row in manifest}
    accepted: dict[str, dict[str, Any]] = {}
    status_counts = Counter()
    for row in results:
        eid = row.get("episode_id")
        status = row.get("status")
        status_counts[status] += 1
        if status == "valid" and eid in manifest_by_id and eid not in accepted:
            accepted[eid] = row
    by_family = defaultdict(lambda: Counter())
    for row in manifest:
        family = row["family"]
        eid = row["episode_id"]
        by_family[family]["planned"] += 1
        result = accepted.get(eid)
        if result is None:
            by_family[family]["missing_valid"] += 1
            continue
        by_family[family]["accepted_valid"] += 1
        if result.get("trigger_eligible"):
            by_family[family]["trigger_eligible"] += 1
        if result.get("event_delivered"):
            by_family[family]["event_delivered"] += 1
        if result.get("event_observed"):
            by_family[family]["event_observed"] += 1
    return {
        "planned_episodes": len(manifest),
        "accepted_valid_unique": len(accepted),
        "missing_valid": len(manifest_by_id) - len(accepted),
        "status_counts": dict(sorted(status_counts.items())),
        "by_family": {family: dict(counts) for family, counts in sorted(by_family.items())},
    }


def build_coverage_by_cell_rows(
    manifest: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    manifest_by_id = {row["episode_id"]: row for row in manifest}
    accepted: dict[str, dict[str, Any]] = {}
    for row in results:
        if row.get("status") == "valid" and row.get("episode_id") in manifest_by_id:
            accepted.setdefault(row["episode_id"], row)
    rows: list[dict[str, Any]] = []
    for episode_id, manifest_row in sorted(manifest_by_id.items()):
        result = accepted.get(episode_id)
        factors = manifest_row["factors"]
        rows.append(
            {
                "episode_id": episode_id,
                "family": manifest_row["family"],
                "fixture": manifest_row["fixture"],
                "block_id": manifest_row["block_id"],
                "policy_id": factors["policy"],
                "goal": factors["goal"],
                "wording": factors.get("wording"),
                "scenario": factors.get("scenario"),
                "named_reference": factors.get("named_reference"),
                "planned": 1,
                "accepted_valid": int(result is not None),
                "missing_valid": int(result is None),
                "infra_invalid": int(any(r.get("episode_id") == episode_id and r.get("status") == "infra_invalid" for r in results)),
                "blocked": int(any(r.get("episode_id") == episode_id and r.get("status") == "blocked" for r in results)),
                "trigger_eligible": int(bool(result and result.get("trigger_eligible"))),
                "event_delivered": int(bool(result and result.get("event_delivered"))),
                "event_observed": int(bool(result and result.get("event_observed"))),
            }
        )
    return rows


def _not_estimable_inference(
    *,
    contrast_key: str,
    policy_id: str,
    robot_stack: str,
    estimand: str,
    reason: str,
    bootstrap_resamples: int,
    seed: int,
) -> InferenceResult:
    return InferenceResult(
        contrast_key=contrast_key,
        policy_id=policy_id,
        robot_stack=robot_stack,
        estimand=estimand,
        point_estimate=None,
        ci_low=None,
        ci_high=None,
        standard_error=None,
        p_value=None,
        test_status="not_estimable",
        not_estimable_reason=reason,
        n_blocks=0,
        n_effective_blocks=0,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=seed,
        undefined_bootstrap_resamples=0,
        zero_or_undefined_se_resamples=0,
        holm_adjusted_p=None,
        holm_rejected=False,
        descriptive={"reason": reason},
    )


def compile_primary_inference(
    lookup: Mapping[tuple[Any, ...], Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_ANALYSIS_SEED,
) -> list[InferenceResult]:
    primary: list[InferenceResult] = []
    seed_slots = [seed, seed + 1, seed + 2, seed + 3]
    for index, policy in enumerate(MAIN_POLICIES):
        c1_blocks = sorted({key[2] for key in lookup if key[0] == "C1" and key[1] == policy})
        if not c1_blocks:
            primary.append(
                _not_estimable_inference(
                    contrast_key=PRIMARY_CONTRASTS[0],
                    policy_id=policy,
                    robot_stack=robot_stack_for_policy(config, policy),
                    estimand="wording_x_motion_success_interaction_pp",
                    reason="no C1 reset blocks in supplied manifest",
                    bootstrap_resamples=bootstrap_resamples,
                    seed=seed_slots[index],
                )
            )
        else:
            primary.append(
                studentized_reset_block_inference(
                    contrast_key=PRIMARY_CONTRASTS[0],
                    policy_id=policy,
                    robot_stack=robot_stack_for_policy(config, policy),
                    estimand="wording_x_motion_success_interaction_pp",
                    block_ids=c1_blocks,
                    estimator=lambda sample, policy=policy: _c1_estimator(lookup, policy=policy, block_ids=sample),
                    bootstrap_resamples=bootstrap_resamples,
                    seed=seed_slots[index],
                )
            )
    for index, policy in enumerate(MAIN_POLICIES):
        c2_blocks = sorted({key[2] for key in lookup if key[0] == "C2" and key[1] == policy})
        if not c2_blocks:
            primary.append(
                _not_estimable_inference(
                    contrast_key=PRIMARY_CONTRASTS[1],
                    policy_id=policy,
                    robot_stack=robot_stack_for_policy(config, policy),
                    estimand="reference_selectivity_equal_goal_m",
                    reason="no C2 reset blocks in supplied manifest",
                    bootstrap_resamples=bootstrap_resamples,
                    seed=seed_slots[index + len(MAIN_POLICIES)],
                )
            )
            continue
        primary.append(
            studentized_reset_block_inference(
                contrast_key=PRIMARY_CONTRASTS[1],
                policy_id=policy,
                robot_stack=robot_stack_for_policy(config, policy),
                estimand="reference_selectivity_equal_goal_m",
                block_ids=c2_blocks,
                estimator=lambda sample, policy=policy: _c2_estimator(
                    lookup, policy=policy, block_ids=sample, config=config
                ),
                bootstrap_resamples=bootstrap_resamples,
                seed=seed_slots[index + len(MAIN_POLICIES)],
            )
        )
    return holm_adjust_primary_tests(primary)


def inference_to_primary_row(result: InferenceResult) -> dict[str, Any]:
    return {
        "contrast_registry_key": result.contrast_key,
        "policy_id": result.policy_id,
        "robot_stack": result.robot_stack,
        "estimand": result.estimand,
        "point_estimate": result.point_estimate,
        "ci95_low": result.ci_low,
        "ci95_high": result.ci_high,
        "standard_error": result.standard_error,
        "p_value": result.p_value,
        "test_status": result.test_status,
        "not_estimable_reason": result.not_estimable_reason,
        "holm_adjusted_p_value": result.holm_adjusted_p,
        "holm_rejected_alpha_0.05": int(result.holm_rejected),
        "n_reset_blocks": result.n_blocks,
        "n_effective_blocks": result.n_effective_blocks,
        "bootstrap_resamples": result.bootstrap_resamples,
        "bootstrap_seed": result.bootstrap_seed,
        "undefined_bootstrap_resamples": result.undefined_bootstrap_resamples,
        "zero_or_undefined_se_resamples": result.zero_or_undefined_se_resamples,
        "descriptive_json": json.dumps(result.descriptive, sort_keys=True),
    }


def build_scope_replication_rows(
    lookup: Mapping[tuple[Any, ...], Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in ("C5", "C6", "C7", "C8"):
        stacks = sorted(
            {
                robot_stack_for_policy(config, key[1])
                for key in lookup
                if key[0] == family
            }
        )
        for stack in stacks:
            rows.append(
                {
                    "family": family,
                    "robot_stack": stack,
                    "status": "not_run",
                    "note": "scope replication estimators are registered separately; no raw cross-stack pooling",
                }
            )
    return rows


def build_failure_composition_rows(
    manifest: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    manifest_by_id = {row["episode_id"]: row for row in manifest}
    accepted: dict[str, dict[str, Any]] = {}
    for row in results:
        if row.get("status") == "valid" and row.get("episode_id") in manifest_by_id:
            accepted.setdefault(row["episode_id"], row)
    rows: list[dict[str, Any]] = []
    for episode_id, manifest_row in sorted(manifest_by_id.items()):
        result = accepted.get(episode_id)
        factors = manifest_row["factors"]
        outcome = result.get("outcome", {}) if result else {}
        exposure = result.get("intervention_exposure", {}) if result else {}
        rows.append(
            {
                "episode_id": episode_id,
                "family": manifest_row["family"],
                "policy_id": factors["policy"],
                "block_id": manifest_row["block_id"],
                "goal": factors["goal"],
                "scenario": factors.get("scenario"),
                "accepted_valid": int(result is not None),
                "success": int(bool(result and result.get("success"))),
                "failure_stage": outcome.get("failure_stage"),
                "failure_label": outcome.get("failure_label"),
                "trigger_eligible": int(bool(result and result.get("trigger_eligible"))),
                "event_delivered": int(bool(result and result.get("event_delivered"))),
                "event_observed": int(bool(result and result.get("event_observed"))),
                "motion_truncated_by_release": int(bool(result and result.get("motion_truncated_by_release"))),
                "intervention_exposure_label": exposure.get("label") or result.get("intervention_exposure_label"),
            }
        )
    return rows


def build_timing_and_motion_rows(
    manifest: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    manifest_by_id = {row["episode_id"]: row for row in manifest}
    accepted: dict[str, dict[str, Any]] = {}
    for row in results:
        if row.get("status") == "valid" and row.get("episode_id") in manifest_by_id:
            accepted.setdefault(row["episode_id"], row)
    rows: list[dict[str, Any]] = []
    for episode_id, manifest_row in sorted(manifest_by_id.items()):
        result = accepted.get(episode_id)
        timing = result.get("timing", {}) if result else {}
        motion = result.get("motion", {}) if result else {}
        rows.append(
            {
                "episode_id": episode_id,
                "family": manifest_row["family"],
                "policy_id": manifest_row["factors"]["policy"],
                "block_id": manifest_row["block_id"],
                "accepted_valid": int(result is not None),
                "t_event_planned_s": timing.get("t_event_planned_s"),
                "t_motion_actual_onset_s": timing.get("t_motion_actual_onset_s"),
                "motion_fraction_observed": motion.get("fraction_observed"),
                "commanded_peak_speed_m_s": motion.get("commanded_peak_speed_m_s"),
                "achieved_peak_speed_m_s": motion.get("achieved_peak_speed_m_s"),
                "observation_delay_s": timing.get("observation_delay_s"),
                "inference_dispatch_delay_s": timing.get("inference_dispatch_delay_s"),
                "queue_execution_delay_s": timing.get("queue_execution_delay_s"),
                "physical_response_delay_s": timing.get("physical_response_delay_s"),
                "inference_wall_time_s": timing.get("inference_wall_time_s"),
            }
        )
    return rows


def build_wording_result_rows(
    lookup: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    policy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for goal in HORIZONTAL_GOALS:
        interactions = [
            value
            for block_id in sorted({key[2] for key in lookup if key[0] == "C1" and key[1] == policy})
            if (value := c1_goal_interaction(lookup, policy=policy, block_id=block_id, goal=goal)) is not None
        ]
        rows.append(
            {
                "family": "C1",
                "policy_id": policy,
                "goal": goal,
                "wording_motion_interaction_success_pp": statistics.fmean(interactions) if interactions else None,
                "n_complete_blocks": len(interactions),
                "status": "descriptive" if interactions else "incomplete",
                "equivalence_claim": "not_authorized",
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def compile_analysis(
    *,
    manifest: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_ANALYSIS_SEED,
) -> dict[str, Any]:
    validation = validate_accepted_ledger(manifest, results, config=config)
    _require(validation["ok"], "; ".join(validation.get("errors", [])))
    accepted = {
        row["episode_id"]: row
        for row in results
        if row.get("status") == "valid" and row.get("episode_id") is not None
    }
    lookup = _cell_lookup(manifest, accepted)
    coverage = reconcile_coverage(manifest, results)
    primary = compile_primary_inference(
        lookup, config, bootstrap_resamples=bootstrap_resamples, seed=seed
    )
    paired_rows = []
    wording_rows = []
    for policy in MAIN_POLICIES:
        paired_rows.extend(build_c1_paired_contrast_rows(lookup, policy=policy))
        paired_rows.extend(build_c2_paired_contrast_rows(lookup, policy=policy, config=config))
        wording_rows.extend(build_wording_result_rows(lookup, policy=policy))
    return {
        "validation": validation,
        "coverage": coverage,
        "primary_results": [inference_to_primary_row(item) for item in primary],
        "primary_inference": primary,
        "coverage_by_cell": build_coverage_by_cell_rows(manifest, results),
        "paired_contrasts": paired_rows,
        "wording_results": wording_rows,
        "scope_replications": build_scope_replication_rows(lookup, config),
        "failure_composition": build_failure_composition_rows(manifest, results),
        "timing_and_motion": build_timing_and_motion_rows(manifest, results),
    }


def export_analysis_tables(
    compiled: Mapping[str, Any],
    output_dir: Path,
    *,
    manifest_path: Path,
    results_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_paths = {
        "coverage_by_cell.csv": output_dir / "coverage_by_cell.csv",
        "primary_results.csv": output_dir / "primary_results.csv",
        "paired_contrasts.csv": output_dir / "paired_contrasts.csv",
        "wording_results.csv": output_dir / "wording_results.csv",
        "scope_replications.csv": output_dir / "scope_replications.csv",
        "failure_composition.csv": output_dir / "failure_composition.csv",
        "timing_and_motion.csv": output_dir / "timing_and_motion.csv",
        "audit_report.json": output_dir / "audit_report.json",
        "results_manifest.json": output_dir / "results_manifest.json",
    }
    write_csv(table_paths["coverage_by_cell.csv"], compiled["coverage_by_cell"])
    write_csv(table_paths["primary_results.csv"], compiled["primary_results"])
    write_csv(table_paths["paired_contrasts.csv"], compiled["paired_contrasts"])
    write_csv(table_paths["wording_results.csv"], compiled["wording_results"])
    write_csv(table_paths["scope_replications.csv"], compiled["scope_replications"])
    write_csv(table_paths["failure_composition.csv"], compiled["failure_composition"])
    write_csv(table_paths["timing_and_motion.csv"], compiled["timing_and_motion"])
    audit = {
        "analysis_scope": "offline_compiler_without_raw_trajectory_replay",
        "validation": compiled["validation"],
        "coverage_reconciliation": compiled["coverage"],
        "limitations": [
            "episode_outcomes.parquet, event_outcomes.parquet, and paired_contrasts.parquet are not emitted "
            "because this repository has no pinned parquet dependency; compact CSV tables are exported instead",
            "figures are not generated without accepted trajectory evidence",
        ],
        "c2_primary_response_contract": {
            "horizon_s": compiled["validation"].get("primary_response_horizon_s", DEFAULT_PRIMARY_RESPONSE_HORIZON_S),
            "anchor": compiled["validation"].get("primary_response_anchor", DEFAULT_RESPONSE_ANCHOR),
            "goal_set_branch": "move",
            "requires_verified_common_prefix": True,
            "terminal_goal_violation_fallback_forbidden": True,
        },
    }
    write_json(table_paths["audit_report.json"], audit)
    manifest = {
        "inputs": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": digest_bytes(manifest_path.read_bytes()),
            "results_path": str(results_path),
            "results_sha256": digest_bytes(results_path.read_bytes()),
            "config_path": str(config_path),
            "config_sha256": digest_bytes(config_path.read_bytes()),
        },
        "outputs": {name: {"path": str(path), "sha256": digest_bytes(path.read_bytes()) if path.exists() and path.stat().st_size else digest("")} for name, path in table_paths.items()},
        "analysis_seed": DEFAULT_ANALYSIS_SEED,
        "bootstrap_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
        "primary_test_count": PRIMARY_TEST_COUNT,
    }
    write_json(table_paths["results_manifest.json"], manifest)
    return {"output_dir": str(output_dir), "tables": {key: str(path) for key, path in table_paths.items()}, "results_manifest": manifest}
