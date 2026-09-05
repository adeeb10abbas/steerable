#!/usr/bin/env python3
"""Plan and audit V4 allocations. This module never runs a robot or qualifies runtime.

Only the Python standard library is required. ``validate`` checks design structure;
``release-check`` checks a declared release lock, not the truth of remote evidence.
All hashes called config_sha256 and manifest_sha256 bind the exact file bytes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "docs/online_correction_v4/campaign.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_FACTORS = {"policy", "goal", "wording", "scenario", "schedule", "named_reference"}
VALID_FAILURE_STAGES = {
    "none", "pickup", "transport", "wrong_relation", "release", "timeout", "collision", "other"
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(value: Any) -> str:
    return digest_bytes(canonical_bytes(value))


def load_json(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), digest_bytes(raw)


def valid_number(value: Any, minimum: float = 0.0, strictly: bool = False) -> bool:
    return (
        isinstance(value, (float, int)) and not isinstance(value, bool)
        and math.isfinite(value) and (value > minimum if strictly else value >= minimum)
    )


def _family_values(family: dict, key: str) -> list:
    return family.get("factors", {}).get(key, [family.get("fixed", {}).get(key)])


def seed_substitution_map(config: dict) -> dict[tuple[str, int], dict[str, Any]]:
    rows = config.get("seed_reservation", {}).get(
        "post_result_environment_seed_substitutions", []
    )
    if not isinstance(rows, list):
        return {}
    return {
        (row["fixture"], row["block_id"]): row
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("fixture"), str)
        and type(row.get("block_id")) is int
    }


def config_errors(config: dict) -> list[str]:
    """Validate allocation and declared dependency structure before expansion."""
    errors = []
    families = config.get("families", [])
    policies = config.get("policies", {})
    fixtures = config.get("fixtures", {})
    if config.get("schema_version") != 1:
        errors.append("config schema_version must be 1")
    if not families or not policies or not fixtures:
        return errors + ["families, policies and fixtures must be nonempty"]
    family_ids = [f.get("id") for f in families]
    if len(set(family_ids)) != len(family_ids):
        errors.append("family IDs are not unique")
    family_map = {f.get("id"): f for f in families}
    slots = [f.get("seed_slot") for f in fixtures.values()]
    if len(set(slots)) != len(slots) or any(type(s) is not int or s < 0 for s in slots):
        errors.append("fixture seed_slot values must be distinct nonnegative integers")
    total = 0
    policy_fixtures = set()
    for family in families:
        fid = family.get("id")
        factors, fixed = family.get("factors", {}), family.get("fixed", {})
        if set(factors) & set(fixed):
            errors.append(f"{fid}: factors overlap fixed values")
        if set(factors) | set(fixed) != REQUIRED_FACTORS:
            errors.append(f"{fid}: factors/fixed must jointly define exactly {sorted(REQUIRED_FACTORS)}")
        blocks = family.get("blocks")
        if type(blocks) is not int or blocks <= 0:
            errors.append(f"{fid}: blocks must be a positive integer")
            continue
        count = blocks
        for key, values in factors.items():
            if not isinstance(values, list) or not values:
                errors.append(f"{fid}: factor {key} must have nonempty values")
                count = 0
                continue
            if len(set(map(str, values))) != len(values):
                errors.append(f"{fid}: factor {key} has duplicate values")
            count *= len(values)
        if count != family.get("expected_new_episodes"):
            errors.append(f"{fid}: allocation is {count}, expected_new_episodes is {family.get('expected_new_episodes')}")
        total += count
        fixture = family.get("fixture")
        if fixture not in fixtures or family.get("seed_group") != fixture:
            errors.append(f"{fid}: fixture/seed_group must identify the same registered fixture")
        for policy in _family_values(family, "policy"):
            if policy not in policies or fixture not in policies[policy].get("fixture_ids", []):
                errors.append(f"{fid}: unsupported policy/fixture {policy}/{fixture}")
            policy_fixtures.add((policy, fixture))
        for reuse in family.get("reuses", []):
            source = family_map.get(reuse.get("family"))
            if not source or source is family:
                errors.append(f"{fid}: invalid reuse source {reuse.get('family')}")
                continue
            if source.get("fixture") != fixture or source.get("seed_group") != family.get("seed_group"):
                errors.append(f"{fid}: reuse crosses a fixture or seed registry")
            limit = reuse.get("block_limit")
            if type(limit) is not int or limit != blocks or limit > source.get("blocks", 0):
                errors.append(f"{fid}: reuse block_limit must equal recipient blocks and fit source")
            for key, wanted in reuse.get("where", {}).items():
                wanted_values = wanted if isinstance(wanted, list) else [wanted]
                if key not in REQUIRED_FACTORS or not set(wanted_values).issubset(_family_values(source, key)):
                    errors.append(f"{fid}: reuse filter {key}={wanted!r} not supported in {source.get('id')}")
    if total != config.get("expected_confirmatory_episodes"):
        errors.append(f"total allocation is {total}, expected_confirmatory_episodes is {config.get('expected_confirmatory_episodes')}")
    pilots = config.get("engineering_pilots", {})
    expected_pilots = len(policy_fixtures) * (
        pilots.get("stationary_per_policy_fixture", 0) + pilots.get("motion_per_policy_fixture", 0)
    )
    if len(policy_fixtures) != pilots.get("expected_policy_fixture_groups"):
        errors.append("engineering pilot policy/fixture group count does not match allocation")
    if expected_pilots != pilots.get("expected_policy_episodes") or pilots.get("exclude_from_confirmatory") is not True:
        errors.append("engineering pilot total or exclusion rule does not match allocation")
    seed = config.get("seed_reservation", {})
    base, stride = seed.get("environment_base"), seed.get("fixture_stride")
    if type(base) is not int or type(stride) is not int or stride <= 0:
        errors.append("environment seed base/stride must be integers with positive stride")
    elif any(f.get("blocks", stride) > stride for f in families):
        errors.append("fixture seed stride is too short for allocated blocks")
    elif slots and base + max(slots) * stride + max(f["blocks"] for f in families) >= 2**31:
        errors.append("environment seed allocation exceeds signed 32-bit range")
    if not seed.get("policy_seed_namespace"):
        errors.append("policy_seed_namespace is required")
    substitutions = seed.get("post_result_environment_seed_substitutions", [])
    if not isinstance(substitutions, list):
        errors.append("post-result environment seed substitutions must be a list")
        substitutions = []
    required_substitution_keys = {
        "fixture",
        "block_id",
        "retired_seed",
        "replacement_seed",
        "reason",
        "evidence_path",
    }
    blocks_by_fixture = {
        fixture: max(
            (
                family.get("blocks", 0)
                for family in families
                if family.get("fixture") == fixture
                and type(family.get("blocks")) is int
            ),
            default=0,
        )
        for fixture in fixtures
    }
    nominal_seeds = {
        base + fixture["seed_slot"] * stride + block
        for fixture_id, fixture in fixtures.items()
        for block in range(blocks_by_fixture[fixture_id])
    } if type(base) is int and type(stride) is int else set()
    seen_substitution_keys: set[tuple[str, int]] = set()
    seen_replacements: set[int] = set()
    for index, substitution in enumerate(substitutions):
        label = f"post-result environment seed substitution {index}"
        if not isinstance(substitution, dict):
            errors.append(f"{label} must be an object")
            continue
        if set(substitution) != required_substitution_keys:
            errors.append(f"{label} must contain exactly {sorted(required_substitution_keys)}")
            continue
        fixture = substitution["fixture"]
        block = substitution["block_id"]
        retired = substitution["retired_seed"]
        replacement = substitution["replacement_seed"]
        if (
            fixture not in fixtures
            or type(block) is not int
            or block < 0
            or block >= blocks_by_fixture.get(fixture, 0)
        ):
            errors.append(f"{label} identifies an invalid fixture/block")
            continue
        key = (fixture, block)
        if key in seen_substitution_keys:
            errors.append(f"{label} duplicates fixture/block {key}")
        seen_substitution_keys.add(key)
        expected_retired = (
            base + fixtures[fixture]["seed_slot"] * stride + block
            if type(base) is int and type(stride) is int
            else None
        )
        if type(retired) is not int or retired != expected_retired:
            errors.append(f"{label} retired_seed does not match the original allocation")
        if (
            type(replacement) is not int
            or replacement < 0
            or replacement >= 2**31
        ):
            errors.append(f"{label} replacement_seed must be a signed 32-bit integer")
        else:
            fixture_start = base + fixtures[fixture]["seed_slot"] * stride
            if not fixture_start <= replacement < fixture_start + stride:
                errors.append(f"{label} replacement_seed leaves the fixture seed namespace")
            if replacement in nominal_seeds:
                errors.append(f"{label} replacement_seed collides with an original allocation")
            if replacement in seen_replacements:
                errors.append(f"{label} replacement_seed is duplicated")
            seen_replacements.add(replacement)
        if not isinstance(substitution["reason"], str) or not substitution["reason"].strip():
            errors.append(f"{label} requires a nonempty reason")
        if (
            not isinstance(substitution["evidence_path"], str)
            or not substitution["evidence_path"].strip()
        ):
            errors.append(f"{label} requires a nonempty evidence_path")
    cb = config.get("counterbalance", {})
    if cb.get("cycle_blocks") != 16 or cb.get("phase_index") != "block_index % 4" or cb.get("state_index") != "(block_index // 4) % 4":
        errors.append("counterbalance must use the declared 16-block phase/state cycle")
    for key in ("physical_signs", "C2_diagonal_signs", "C2_A_colors", "C2_A_start_sides"):
        if len(cb.get(key, [])) != 4:
            errors.append(f"counterbalance {key} must contain four state values")
    if len(config.get("timing", {}).get("event_phase_fractions", [])) != 4:
        errors.append("event_phase_fractions must contain four phase values")
    if any(f.get("blocks", 0) % 16 for f in families):
        errors.append("all family block counts must complete the 16-block counterbalance cycle")
    return errors


def _cell_key(row: dict) -> tuple:
    return (row["fixture"], row["block_id"], *[row["factors"][k] for k in sorted(REQUIRED_FACTORS)])


def _episode_id(campaign: str, family: str, fixture: str, block: int, factors: dict) -> str:
    suffix = digest([fixture, block, factors])[:16]
    return f"{campaign}-{family}-b{block:03d}-{suffix}"


def _prefix_id(campaign: str, fixture: str, block: int, factors: dict) -> str:
    # Motion and execution schedule are inactive before the natural trigger.
    # Destination-static changes the initial scene and therefore its prefix.
    identity = {k: factors[k] for k in ("policy", "goal", "wording", "named_reference")}
    identity["initial_scene"] = "destination" if factors["scenario"] == "destination_static" else "original"
    return f"{campaign}-prefix-{digest([fixture, block, identity])[:24]}"


def _counterbalance(config: dict, fixture: str, block: int) -> dict:
    state = (block // 4) % 4
    cb = config["counterbalance"]
    result = {
        "phase_index": block % 4,
        "state_index": state,
        "event_phase_fraction": config["timing"]["event_phase_fractions"][block % 4],
        "physical_translation_sign": cb["physical_signs"][state],
    }
    if fixture == "reference_binding":
        result.update({
            "physical_A_diagonal_signs": cb["C2_diagonal_signs"][state],
            "physical_A_color": cb["C2_A_colors"][state],
            "physical_A_start_side": cb["C2_A_start_sides"][state],
        })
    return result


def _prompt_recipe(config: dict, fixture: str, factors: dict, cb: dict) -> dict:
    goal, wording = factors["goal"], factors["wording"]
    relation_group = "containment" if goal == "inside" else "vertical" if goal in ("above", "below") else "horizontal"
    clause = config["wording"][f"{relation_group}_{wording}"][goal]
    prompt = config["wording"]["carrier"].format(object="{object}", clause=clause)
    if relation_group == "horizontal":
        prompt += config["wording"]["frame_suffix_horizontal"]
    recipe = {"template": prompt, "object_role": config["fixtures"][fixture]["moving_object"],
              "reference_role": config["fixtures"][fixture]["reference"],
              "binding_status": "resolve_exact_names_once_in_runtime_lock"}
    if fixture == "reference_binding":
        color_a = cb["physical_A_color"]
        recipe["reference_color"] = color_a if factors["named_reference"] == "A" else ("yellow" if color_a == "blue" else "blue")
    return recipe


def _reuse_candidates(row: dict, family: dict, source_rows: dict) -> list[str]:
    reused = []
    for reuse in family.get("reuses", []):
        if row["block_id"] >= reuse["block_limit"]:
            continue
        candidates = source_rows.get((reuse["family"], row["fixture"], row["block_id"], row["factors"]["policy"]), [])
        selected = []
        for candidate in candidates:
            cf, rf = candidate["factors"], row["factors"]
            # Same physical goal and same utterance/reference binding; scenario
            # and schedule are explicitly allowed to differ in a comparison.
            if any(cf[k] != rf[k] for k in ("goal", "wording", "named_reference")):
                continue
            match = True
            for key, wanted in reuse.get("where", {}).items():
                if cf[key] not in (wanted if isinstance(wanted, list) else [wanted]):
                    match = False
                    break
            if match:
                selected.append(candidate["episode_id"])
        if not selected:
            raise ValueError(f"{row['episode_id']}: reuse from {reuse['family']} has no matching cells")
        reused.extend(selected)
    return sorted(set(reused))


def build_manifest(config: dict, config_sha256: str | None = None) -> list[dict]:
    """Expand only NEW assigned episodes; controls are exact ID references."""
    errors = config_errors(config)
    if errors:
        raise ValueError("; ".join(errors))
    config_sha256 = config_sha256 or digest(config)
    campaign = config["campaign_id"]
    seed = config["seed_reservation"]
    substitutions = seed_substitution_map(config)
    rows = []
    by_source = defaultdict(list)
    families = {f["id"]: f for f in config["families"]}
    for family in config["families"]:
        names = sorted(family["factors"])
        fixture = family["fixture"]
        for block in range(family["blocks"]):
            nominal_env_seed = (
                seed["environment_base"]
                + config["fixtures"][fixture]["seed_slot"] * seed["fixture_stride"]
                + block
            )
            substitution = substitutions.get((fixture, block))
            env_seed = (
                substitution["replacement_seed"]
                if substitution is not None
                else nominal_env_seed
            )
            for values in itertools.product(*(family["factors"][name] for name in names)):
                factors = dict(family["fixed"], **dict(zip(names, values)))
                policy_seed = int(digest([seed["policy_seed_namespace"], factors["policy"], fixture, block])[:16], 16) % (2**31)
                row = {
                    "schema_version": 1,
                    "manifest_type": "planning_manifest",
                    "runtime_bound": False,
                    "episode_id": _episode_id(campaign, family["id"], fixture, block, factors),
                    "campaign": campaign,
                    "family": family["id"],
                    "fixture": fixture,
                    "block_id": block,
                    "block_key": f"{campaign}:{fixture}:{block:03d}",
                    "env_seed": env_seed,
                    "policy_seed": policy_seed,
                    "cohort": "confirmatory",
                    "priority": family["priority"],
                    "factors": factors,
                    "prefix_group_id": _prefix_id(campaign, fixture, block, factors),
                    "execution_group": f"{factors['policy']}:{fixture}",
                    "execution_order_key": digest([seed["policy_seed_namespace"], "execution-order", fixture, block, factors]),
                    "config_sha256": config_sha256,
                    "reuse_episode_ids": [],
                }
                if substitution is not None:
                    row["env_seed_substitution"] = {
                        "retired_seed": substitution["retired_seed"],
                        "replacement_seed": substitution["replacement_seed"],
                        "reason": substitution["reason"],
                        "evidence_path": substitution["evidence_path"],
                    }
                row["counterbalance"] = _counterbalance(config, fixture, block)
                row["prompt_recipe"] = _prompt_recipe(config, fixture, factors, row["counterbalance"])
                rows.append(row)
                by_source[(family["id"], fixture, block, factors["policy"])].append(row)
    for row in rows:
        row["reuse_episode_ids"] = _reuse_candidates(row, families[row["family"]], by_source)
    rows.sort(key=lambda r: (r["execution_group"], r["execution_order_key"]))
    order_counts = Counter()
    for row in rows:
        row["execution_order"] = order_counts[row["execution_group"]]
        order_counts[row["execution_group"]] += 1
    return rows


def manifest_bytes(rows: list[dict]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def manifest_errors(rows: list[dict], config: dict, config_sha256: str | None = None) -> list[str]:
    """Check the complete inventory against independent deterministic regeneration."""
    expected = build_manifest(config, config_sha256)
    expected_by_id = {r["episode_id"]: r for r in expected}
    errors = []
    ids = [r.get("episode_id") for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("manifest contains duplicate episode IDs")
    if len(rows) != len(expected):
        errors.append(f"manifest has {len(rows)} rows; expected {len(expected)}")
    got_ids = set(ids)
    if got_ids != set(expected_by_id):
        errors.append(f"manifest ID set differs: missing={len(set(expected_by_id) - got_ids)}, unknown={len(got_ids - set(expected_by_id))}")
    physical_keys = set()
    for row in rows:
        eid = row.get("episode_id")
        if eid not in expected_by_id:
            continue
        reference = expected_by_id[eid]
        if row != reference:
            changed = sorted(k for k in set(row) | set(reference) if row.get(k) != reference.get(k))
            errors.append(f"{eid}: deterministic inventory mismatch in {','.join(changed)}")
        try:
            key = _cell_key(row)
            if key in physical_keys:
                errors.append(f"{eid}: duplicate physical cell (reuse is not a new trial)")
            physical_keys.add(key)
        except (KeyError, TypeError):
            errors.append(f"{eid}: malformed factor identity")
    return errors


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: JSONL record must be an object")
        rows.append(value)
    return rows


def _unresolved(value: Any, path: str = "lock") -> list[str]:
    errors = []
    if value is None:
        errors.append(f"{path}: null is not released")
    elif isinstance(value, str) and (not value.strip() or re.search(r"(^|[^A-Za-z])(TODO|TBD|PLACEHOLDER|UNQUALIFIED)([^A-Za-z]|$)", value, re.I)):
        errors.append(f"{path}: unresolved text")
    elif isinstance(value, dict):
        for key, child in value.items():
            errors.extend(_unresolved(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_unresolved(child, f"{path}[{index}]"))
    return errors


def release_errors(lock: dict, config: dict, config_sha256: str, manifest_sha256: str) -> list[str]:
    errors = _unresolved(lock)
    for key, expected in (("schema_version", 1), ("campaign_id", config["campaign_id"]),
                          ("config_sha256", config_sha256), ("manifest_sha256", manifest_sha256),
                          ("source_commit", config["source_commit"]), ("release_status", "RELEASED")):
        if lock.get(key) != expected:
            errors.append(f"lock {key} does not match required value")
    families = {f["id"]: f for f in config["families"]}
    released_list = lock.get("released_families", [])
    released = set(released_list)
    blocked = lock.get("blocked_families", {})
    if not released or len(released) != len(released_list) or not released <= set(families):
        errors.append("released_families must be a nonempty unique subset of declared families")
    if not isinstance(blocked, dict):
        errors.append("blocked_families must map IDs to reasons")
        blocked = {}
    if released & set(blocked) or released | set(blocked) != set(families):
        errors.append("released and blocked families must partition the complete fixed allocation")
    if any(not isinstance(reason, str) or not reason.strip() for reason in blocked.values()):
        errors.append("each blocked family needs a nonempty reason")
    for fid in released & set(families):
        dependencies = {r["family"] for r in families[fid].get("reuses", [])}
        if not dependencies <= released:
            errors.append(f"{fid}: reused control families must also be released: {sorted(dependencies - released)}")
    runner = lock.get("runner", {})
    if not HEX40.fullmatch(str(runner.get("commit", ""))):
        errors.append("runner.commit must be a full git commit")
    if not HEX64.fullmatch(str(runner.get("sha256", ""))) or not runner.get("entrypoint"):
        errors.append("runner entrypoint and SHA256 are required")
    needed_policies = set()
    needed_fixtures = set()
    for fid in released & set(families):
        needed_policies.update(_family_values(families[fid], "policy"))
        needed_fixtures.add(families[fid]["fixture"])
    policies = lock.get("policies", {})
    fixtures = lock.get("fixtures", {})
    if set(policies) != needed_policies:
        errors.append("lock policy keys must exactly match released family policies")
    if set(fixtures) != needed_fixtures:
        errors.append("lock fixture keys must exactly match released family fixtures")
    for name, policy in policies.items():
        if not HEX64.fullmatch(str(policy.get("checkpoint_sha256", ""))):
            errors.append(f"policy {name}: checkpoint_sha256 required")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(policy.get("runtime_image_digest", ""))):
            errors.append(f"policy {name}: immutable runtime_image_digest required")
        if not valid_number(policy.get("native_control_dt_s"), strictly=True):
            errors.append(f"policy {name}: positive native_control_dt_s required")
        for key in ("checkpoint_uri", "policy_reset_and_history_contract_uri"):
            if not isinstance(policy.get(key), str) or not policy[key].strip():
                errors.append(f"policy {name}: {key} required")
        if not HEX40.fullmatch(str(policy.get("integration_commit", ""))):
            errors.append(f"policy {name}: full integration_commit required")
        achieved = ("achieved_delay_s", "achieved_standard_query_period_s", "achieved_fast_query_period_s")
        requested = ("emulated_observation_action_delay_s", "standard_query_period_s", "fast_query_period_s")
        dt = policy.get("native_control_dt_s")
        for key, request in zip(achieved, requested):
            if not valid_number(policy.get(key), strictly=True):
                errors.append(f"policy {name}: positive {key} required")
            elif valid_number(dt, strictly=True):
                expected = math.ceil(config["timing"][request] / dt - 1e-10) * dt
                if not math.isclose(policy[key], expected, rel_tol=1e-9, abs_tol=1e-9):
                    errors.append(f"policy {name}: {key} does not equal registered upward tick quantization")
        horizon = policy.get("prediction_horizon_actions")
        if type(horizon) is not int or horizon <= 0:
            errors.append(f"policy {name}: positive prediction_horizon_actions required")
        elif valid_number(dt, strictly=True) and all(valid_number(policy.get(k), strictly=True) for k in achieved):
            coverage = horizon * dt
            if coverage + 1e-9 < policy["achieved_standard_query_period_s"] + policy["achieved_delay_s"]:
                errors.append(f"policy {name}: predicted queue cannot cover registered standard period plus delay")
        if "C4" in released and name in _family_values(families["C4"], "policy"):
            fast, standard = policy.get("achieved_fast_query_period_s"), policy.get("achieved_standard_query_period_s")
            if not (valid_number(fast, strictly=True) and valid_number(standard, strictly=True) and fast < standard):
                errors.append(f"policy {name}: C4 requires an achieved fast cadence strictly shorter than standard")
    for name, fixture in fixtures.items():
        for key in ("geometry_sha256", "scorer_sha256", "reset_registry_sha256"):
            if not HEX64.fullmatch(str(fixture.get(key, ""))):
                errors.append(f"fixture {name}: {key} required")
        for key in ("geometry_uri", "scorer_uri", "reset_registry_uri", "frame_transform_uri",
                    "goal_geometry_and_tolerances_uri", "trigger_release_detector_uri",
                    "intervention_trajectory_registry_uri", "scoring_and_visibility_thresholds_uri"):
            if not isinstance(fixture.get(key), str) or not fixture[key].strip():
                errors.append(f"fixture {name}: {key} required")
        if fixture.get("calibration_scale") not in config["motion"]["calibration_scale_candidates"]:
            errors.append(f"fixture {name}: calibration_scale is not a registered candidate")
        if not valid_number(fixture.get("D_cap_m"), strictly=True):
            errors.append(f"fixture {name}: positive D_cap_m required")
    receipts = lock.get("receipts", {})
    for name in config["required_release_receipts"]:
        receipt = receipts.get(name, {})
        if receipt.get("passed") is not True:
            errors.append(f"receipt {name}: passed must be true")
        if not HEX64.fullmatch(str(receipt.get("sha256", ""))) or not receipt.get("uri"):
            errors.append(f"receipt {name}: evidence URI and SHA256 required")
        if not released <= set(receipt.get("family_ids", [])):
            errors.append(f"receipt {name}: family coverage is incomplete")
    return errors


def check_results(manifest: list[dict], results: list[dict]) -> dict:
    """Audit all assigned cells; valid behavioral failures count as accepted data."""
    errors = []
    known = {r["episode_id"]: r for r in manifest}
    if len(known) != len(manifest):
        errors.append("input manifest has duplicate episode IDs")
    accepted = Counter()
    attempts = set()
    infra = 0
    blocked = Counter()
    successes = 0
    for index, row in enumerate(results, 1):
        eid = row.get("episode_id")
        label = f"result {index} ({eid})"
        if eid not in known:
            errors.append(f"{label}: unknown episode ID")
            continue
        attempt = row.get("attempt_id")
        if not isinstance(attempt, str) or not attempt:
            errors.append(f"{label}: attempt_id required")
        elif (eid, attempt) in attempts:
            errors.append(f"{label}: duplicate attempt_id for cell")
        attempts.add((eid, str(attempt)))
        status = row.get("status")
        if status == "infra_invalid":
            infra += 1
            if not isinstance(row.get("reason"), str) or not row["reason"].strip():
                errors.append(f"{label}: infrastructure invalidity reason required")
            continue
        if status == "blocked":
            blocked[known[eid]["family"]] += 1
            if not isinstance(row.get("reason"), str) or not row["reason"].strip():
                errors.append(f"{label}: blocked cell reason required")
            continue
        if status != "valid":
            errors.append(f"{label}: status must be valid, infra_invalid, or blocked")
            continue
        accepted[eid] += 1
        if row.get("config_sha256") != known[eid]["config_sha256"]:
            errors.append(f"{label}: config_sha256 differs from assigned manifest")
        if row.get("prefix_group_id") != known[eid]["prefix_group_id"]:
            errors.append(f"{label}: prefix_group_id differs from assigned manifest")
        for key in ("success", "trigger_eligible", "event_delivered", "event_observed"):
            if type(row.get(key)) is not bool:
                errors.append(f"{label}: {key} must be boolean")
        if row.get("event_observed") is True and row.get("event_delivered") is not True:
            errors.append(f"{label}: observed event was not delivered")
        if row.get("event_delivered") is True and row.get("trigger_eligible") is not True:
            errors.append(f"{label}: delivered event lacks pre-event eligibility")
        if row.get("success") is True:
            successes += 1
        outcome = row.get("outcome", {})
        if not valid_number(outcome.get("goal_violation_capped_m")):
            errors.append(f"{label}: goal_violation_capped_m must be finite and nonnegative")
        elif row.get("success") is True and outcome["goal_violation_capped_m"] > 1e-9:
            errors.append(f"{label}: successful placement cannot have positive geometric goal violation")
        for key in ("goal_set_empty", "goal_violation_cap_applied"):
            if type(outcome.get(key)) is not bool:
                errors.append(f"{label}: outcome.{key} must be boolean")
        if outcome.get("goal_set_empty") is True and (row.get("success") is not False or outcome.get("goal_violation_cap_applied") is not True):
            errors.append(f"{label}: empty goal set requires failure and capped violation")
        if outcome.get("failure_stage") not in VALID_FAILURE_STAGES:
            errors.append(f"{label}: failure_stage must be one of {sorted(VALID_FAILURE_STAGES)}")
        if row.get("success") is True and outcome.get("failure_stage") != "none":
            errors.append(f"{label}: successful placement cannot have a failure stage")
        if row.get("success") is False and outcome.get("failure_stage") == "none":
            errors.append(f"{label}: behavioral failure needs a failure stage")
        for key in ("trace_uri", "video_uri"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                errors.append(f"{label}: {key} required")
        for key in ("trace_sha256", "video_sha256", "scorer_sha256", "protocol_sha256"):
            if not HEX64.fullmatch(str(row.get(key, ""))):
                errors.append(f"{label}: {key} must be SHA256")
    duplicates = sorted(eid for eid, n in accepted.items() if n > 1)
    missing = sorted(set(known) - set(accepted))
    if duplicates:
        errors.append(f"duplicate accepted cells: {len(duplicates)}")
    if missing:
        errors.append(f"incomplete allocation: {len(missing)} cells have no accepted valid record")
    coverage = {}
    for family in sorted({r["family"] for r in manifest}):
        assigned = [r["episode_id"] for r in manifest if r["family"] == family]
        coverage[family] = {
            "assigned": len(assigned), "accepted_unique": sum(eid in accepted for eid in assigned),
            "missing": sum(eid not in accepted for eid in assigned), "blocked_records": blocked[family],
        }
    return {
        "ok": not errors, "audit_scope": "coverage_and_basic_record_consistency_only",
        "assigned": len(manifest), "accepted_unique": len(accepted), "infrastructure_attempts": infra,
        "valid_success_records": successes, "valid_failure_records": sum(accepted.values()) - successes,
        "coverage_by_family": coverage, "duplicate_accepted_episode_ids": duplicates,
        "missing_episode_ids": missing, "errors": errors,
        "limitations": ["Trace/video contents and remote URIs are not opened.",
                        "Scoring, pairing eligibility, receipt truth and statistical analyses require the documented independent audit."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "manifest", "release-check", "check-results"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if name == "manifest":
            command.add_argument("--out", type=Path, required=True)
        if name in ("release-check", "check-results"):
            command.add_argument("--manifest", type=Path, required=True)
        if name == "release-check":
            command.add_argument("--lock", type=Path, required=True)
        if name == "check-results":
            command.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config, config_sha = load_json(args.config)
        errors = config_errors(config)
        if errors:
            report = {"ok": False, "errors": errors}
        else:
            rows = build_manifest(config, config_sha)
            inventory_errors = manifest_errors(rows, config, config_sha)
            if inventory_errors:
                report = {"ok": False, "errors": inventory_errors}
            elif args.command in ("validate", "manifest"):
                raw = manifest_bytes(rows)
                report = {
                    "ok": True, "validation_scope": "design_structure_only", "runtime_released": False,
                    "config_sha256": config_sha, "manifest_sha256": digest_bytes(raw),
                    "new_episode_count": len(rows),
                    "new_episodes_by_family": dict(sorted(Counter(r["family"] for r in rows).items())),
                    "unique_reused_controls_by_family": {
                        f["id"]: len({eid for r in rows if r["family"] == f["id"] for eid in r["reuse_episode_ids"]})
                        for f in config["families"] if f.get("reuses")
                    },
                    "excluded_engineering_pilots": config["engineering_pilots"]["expected_policy_episodes"],
                }
                if args.command == "manifest":
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_bytes(raw)
                    report["manifest_path"] = str(args.out)
            else:
                supplied = read_jsonl(args.manifest)
                errors = manifest_errors(supplied, config, config_sha)
                if errors:
                    report = {"ok": False, "errors": errors}
                elif args.command == "release-check":
                    lock, _ = load_json(args.lock)
                    errors = release_errors(lock, config, config_sha, digest_bytes(args.manifest.read_bytes()))
                    report = {
                        "ok": not errors, "validation_scope": "release_lock_structure_and_file_identity_only",
                        "released_families": lock.get("released_families", []),
                        "blocked_families": lock.get("blocked_families", {}), "errors": errors,
                        "limitations": ["Receipt hashes and URIs are checked syntactically; remote evidence contents are not retrieved or certified.",
                                        "Passing this check does not independently prove robot-runner correctness or launch any job."],
                    }
                else:
                    report = check_results(supplied, read_jsonl(args.results))
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if report["ok"] else 1
    except (ValueError, KeyError, TypeError, OSError, AttributeError) as exc:
        print(json.dumps({"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
