#!/usr/bin/env python3
"""Build prospective V4 design/freeze artifacts before any policy inference.

Generates protocol, prompt, motion, scoring, seed, queue, analysis, gate, and
continuation artifacts under artifacts/online_correction_v4/. Runtime, geometry,
and checkpoint receipts remain explicitly pending or blocked.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "docs/online_correction_v4/campaign.json"
DEFAULT_OUT = ROOT / "artifacts/online_correction_v4"

SPEC = importlib.util.spec_from_file_location("online_correction_v4", ROOT / "tools/online_correction_v4.py")
v4 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(v4)

FIXTURE_PROMPT_NAMES: dict[str, dict[str, Any]] = {
    "horizontal": {
        "object": "cube",
        "reference": "bowl",
        "physical_resolution": "symbolic_fixture_roles_pending_runtime_lock",
    },
    "reference_binding": {
        "object": "cube",
        "physical_resolution": "symbolic_color_labels_pending_runtime_lock",
    },
    "vertical": {
        "object": "cube",
        "reference": "bowl",
        "physical_resolution": "symbolic_fixture_roles_pending_runtime_lock",
    },
    "containment": {
        "object": "cube",
        "reference": "bowl",
        "physical_resolution": "symbolic_fixture_roles_pending_runtime_lock",
    },
    "object_pair": {
        "object": "sponge",
        "reference": "tray",
        "physical_resolution": "symbolic_fixture_roles_pending_runtime_lock",
    },
    "second_stack": {
        "object": None,
        "reference": None,
        "physical_resolution": "UNRESOLVED_PENDING_BRIDGE_FIXTURE_LOCK",
    },
}

C2_BLOCK_REASON = (
    "Primary reference-selectivity (H) requires verified common-prefix replay within each "
    "prompt; deterministic fresh-session replay and qualified full-state snapshot are not "
    "yet demonstrated on the target runtime."
)

C8_BLOCK_REASON = (
    "BLOCKED_RUNTIME: GR00T N1.7 Bridge/WidowX checkpoint URI, integration commit, adapter, "
    "native control period, and SimplerEnv fixture object names are not verified on cluster "
    "workers; second-stack geometry and scoring receipts remain pending."
)

DEFAULT_FAMILY_PENDING = (
    "NOT_RELEASED: runner, simulator fixtures, geometry receipts, checkpoint identity, "
    "and qualification gates are not yet bound or passed."
)

PENDING_FAMILY_IDS = ("C1", "C3", "C4", "C5", "C6", "C7")
HARD_BLOCKED_FAMILY_IDS = ("C2", "C8")

PILOT_GROUPS = [
    ("cosmos3_nano_droid", "horizontal"),
    ("pi05_droid", "horizontal"),
    ("cosmos3_nano_droid", "reference_binding"),
    ("pi05_droid", "reference_binding"),
    ("cosmos3_nano_droid", "vertical"),
    ("pi05_droid", "vertical"),
    ("cosmos3_nano_droid", "containment"),
    ("pi05_droid", "containment"),
    ("cosmos3_nano_droid", "object_pair"),
    ("groot_bridge_widowx", "second_stack"),
]


def repo_head_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def historical_protocol_ledger() -> dict[str, Any]:
    entries = []
    for label, rel in (
        ("v2_protocol", "artifacts/vla_wam_shared_v2/protocol.json"),
        ("v3_protocol", "artifacts/vla_wam_shared_v3/protocol.json"),
    ):
        path = ROOT / rel
        entries.append(
            {
                "ledger_id": label,
                "path": rel,
                "sha256": file_sha256(path),
                "byte_length": path.stat().st_size,
                "immutable": True,
            }
        )
    return {
        "schema_version": 1,
        "purpose": "fail_closed_historical_protocol_integrity",
        "entries": entries,
    }


def _reference_name(
    fixture: str,
    named_reference: str,
    counterbalance: dict | None = None,
) -> tuple[str, str, dict[str, Any]]:
    spec = FIXTURE_PROMPT_NAMES[fixture]
    extra: dict[str, Any] = {}
    if fixture == "reference_binding":
        if counterbalance is None:
            raise ValueError("reference_binding prompts require counterbalance")
        color_a = counterbalance["physical_A_color"]
        color_b = "yellow" if color_a == "blue" else "blue"
        if named_reference == "A":
            ref = f"{color_a} bowl"
            extra["reference_color"] = color_a
            extra["physical_A_color"] = color_a
        else:
            ref = f"{color_b} bowl"
            extra["reference_color"] = color_b
            extra["physical_A_color"] = color_a
        return spec["object"], ref, extra
    if fixture == "second_stack":
        return "{UNRESOLVED_MANIPULATED_OBJECT}", "{UNRESOLVED_REFERENCE_OBJECT}", extra
    return spec["object"], spec["reference"], extra


def resolve_prompt_text(
    config: dict,
    fixture: str,
    factors: dict,
    counterbalance: dict | None = None,
) -> tuple[str, dict]:
    goal, wording = factors["goal"], factors["wording"]
    relation_group = (
        "containment" if goal == "inside" else "vertical" if goal in ("above", "below") else "horizontal"
    )
    clause_template = config["wording"][f"{relation_group}_{wording}"][goal]
    obj, ref, ref_extra = _reference_name(fixture, factors.get("named_reference", "single"), counterbalance)
    clause = clause_template.format(object=obj, reference=ref)
    prompt = config["wording"]["carrier"].format(object=obj, clause=clause)
    if relation_group == "horizontal":
        prompt += config["wording"]["frame_suffix_horizontal"]
    meta = {
        "object_text": obj,
        "reference_text": ref,
        "relation_group": relation_group,
        "goal": goal,
        "wording": wording,
        "named_reference": factors.get("named_reference"),
        "physical_resolution": FIXTURE_PROMPT_NAMES[fixture]["physical_resolution"],
        "launch_critical_names_resolved": fixture != "second_stack",
        **ref_extra,
    }
    return prompt, meta


def prompt_identity(config: dict, row: dict) -> dict[str, Any]:
    """Stable prompt_id inputs; C2 includes block counterbalance color binding."""
    identity: dict[str, Any] = {
        "fixture": row["fixture"],
        "goal": row["factors"]["goal"],
        "wording": row["factors"]["wording"],
        "named_reference": row["factors"]["named_reference"],
    }
    if row["fixture"] == "reference_binding":
        cb = row["counterbalance"]
        identity["physical_A_color"] = cb["physical_A_color"]
        identity["named_reference"] = row["factors"]["named_reference"]
    return identity


def build_prompt_manifest(config: dict, rows: list[dict]) -> dict[str, Any]:
    prompts: dict[tuple[str, str], dict] = {}
    for row in rows:
        text, meta = resolve_prompt_text(config, row["fixture"], row["factors"], row.get("counterbalance"))
        prompt_id = v4.digest(prompt_identity(config, row))[:24]
        key = (prompt_id, text)
        if key not in prompts:
            prompts[key] = {
                "prompt_id": prompt_id,
                "prompt_text": text,
                "prompt_sha256": v4.digest_bytes(text.encode("utf-8")),
                "fixture_ids": sorted({row["fixture"]}),
                **meta,
            }
        else:
            prompts[key]["fixture_ids"] = sorted(set(prompts[key]["fixture_ids"]) | {row["fixture"]})
    ordered = [prompts[k] for k in sorted(prompts)]
    for item in ordered:
        item["fixture_ids"] = sorted(item["fixture_ids"])
    sha_to_ids: dict[str, list[str]] = defaultdict(list)
    for item in ordered:
        sha_to_ids[item["prompt_sha256"]].append(item["prompt_id"])
    shared_sha = {sha: ids for sha, ids in sha_to_ids.items() if len(ids) > 1}
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "carrier_template": config["wording"]["carrier"],
        "frame_suffix_horizontal": config["wording"]["frame_suffix_horizontal"],
        "fixture_name_binding": FIXTURE_PROMPT_NAMES,
        "name_resolution_rule": "bare_nouns_in_templates; carrier_and_clauses_supply_articles",
        "prompt_identity_semantics": {
            "primary_key": "prompt_id",
            "episode_binding_fields": ["prompt_id", "prompt_text", "prompt_sha256"],
            "c2_counterbalance_in_prompt_id": True,
            "prompt_sha256_rule": "sha256(utf8(prompt_text)); identical iff byte-identical resolved text",
            "prompt_sha256_may_map_to_multiple_prompt_ids": True,
            "prompt_sha256_must_be_unique_when_text_differs": True,
            "analysis_forbidden_primary_keys": ["prompt_sha256"],
            "reason": (
                "Identical UTF-8 prompt text can name different physical A/B bowl identities under C2 "
                "counterbalance. Semantic identity is prompt_id plus queue episode binding, never prompt_sha256 alone. "
                "prompt_sha256 is a content hash only: same resolved bytes share one hash; different bytes must not."
            ),
        },
        "prompt_sha256_shared_by_prompt_id_count": len(shared_sha),
        "prompts": ordered,
        "unique_prompt_count": len(ordered),
        "unresolved_physical_name_fixtures": ["second_stack"],
        "runtime_manifest_binding_required": True,
    }


def build_motion_manifest(config: dict) -> dict[str, Any]:
    motion = config["motion"]
    timing = config["timing"]
    profiles = []
    for scenario, duration_key, label in (
        ("move_stop", "move_stop_duration_s", "Primary finite perturbation"),
        ("slow_drift", "slow_drift_duration_s", "Longer exposure"),
        ("fast_drift", "fast_drift_duration_s", "Intermediate speed-duration"),
        ("reversal", None, "Direction change with non-original endpoint"),
    ):
        entry: dict[str, Any] = {
            "scenario": scenario,
            "primitive": motion["primitive"],
            "interpretation": label,
        }
        if duration_key:
            entry["nominal_duration_s"] = motion[duration_key]
        if scenario == "reversal":
            entry["waypoints"] = motion["reversal_waypoints"]
        profiles.append(entry)
    profiles.extend(
        [
            {
                "scenario": "original_sham",
                "primitive": motion["primitive"],
                "nominal_duration_s": motion["move_stop_duration_s"],
                "translation": "identity_pose_update_with_scheduler_overhead",
            },
            {
                "scenario": "destination_static",
                "primitive": motion["primitive"],
                "translation": "reference_at_planned_destination_from_reset",
            },
            {
                "scenario": "move_A",
                "primitive": motion["primitive"],
                "nominal_duration_s": motion["move_stop_duration_s"],
                "motion_direction": "block_balanced_diagonal_identical_across_named_references",
            },
        ]
    )
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "interpolation": "minimum_jerk_scalar S(u)=10u^3-15u^4+6u^5 clipped to [0,1]",
        "path_equation": "p_ref(t)=p_ref(0)+e*D*S(t/T)",
        "calibration": {
            "scale_candidates": motion["calibration_scale_candidates"],
            "selection_rule": motion["calibration_selection_rule"],
            "policy_dependent_scale_selection_forbidden": motion["policy_dependent_scale_selection_forbidden"],
            "selected_scale_by_fixture": {fid: None for fid in config["fixtures"]},
            "status": "pending_model_blind_geometry_gate",
        },
        "fixture_nominal_translation_m": {
            fid: spec["nominal_translation_m"] for fid, spec in config["fixtures"].items()
        },
        "profiles": profiles,
        "truncation_rule": "first_placement_confirmation_or_active_timeout",
        "event_onset": timing["event_phase_anchor"],
        "release_freeze_rule": timing["release_rule"],
    }


def build_scoring_manifest(config: dict) -> dict[str, Any]:
    timing = config["timing"]
    analysis = config["analysis"]
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "terminal_rule": "first_detachment_after_verified_carry_with_two_tick_confirmation",
        "settling_interval_s": timing["release_settling_s"],
        "episode_active_cap_s": timing["episode_cap_s"],
        "trigger_deadline_s": timing["trigger_deadline_s"],
        "post_event_cap_s": timing["post_event_cap_s"],
        "natural_grasp": {
            "min_lift_m": timing["natural_grasp_min_lift_m"],
            "dwell_s": timing["natural_grasp_dwell_s"],
            "relative_drift_max_m": timing["kinematic_grasp_relative_drift_max_m"],
        },
        "response_detection": {
            "primary_displacement_threshold_m": timing["response_displacement_threshold_m"],
            "sensitivity_thresholds_m": timing["response_sensitivity_thresholds_m"],
            "dwell_control_ticks": timing["release_detection_dwell_ticks"],
        },
        "changed_observation": {
            "reference_displacement_m": timing["changed_observation_reference_displacement_m"],
            "visibility_rule": timing["changed_observation_visibility_rule"],
        },
        "primary_response_anchor": timing["primary_response_anchor"],
        "primary_response_horizon_s": timing["primary_response_horizon_s"],
        "secondary_response_horizons_s": timing["secondary_response_horizons_s"],
        "goal_violation_cap_rule": "min(distance_to_goal_set, D_cap); D_cap pending per-fixture geometry receipt",
        "D_cap_m_by_fixture": {fid: None for fid in config["fixtures"]},
        "failure_taxonomy_source": "docs/online_correction_v4/02_METRICS_AND_ANALYSIS.md section 8",
        "primary_contrasts": analysis["primary_contrasts"],
        "multiplicity": {
            "family": analysis["multiplicity"],
            "alpha": analysis["alpha"],
            "primary_tests": analysis["primary_tests"],
            "bootstrap_resamples": analysis["bootstrap_resamples"],
            "analysis_rng_seed": analysis["analysis_rng_seed"],
            "primary_test": analysis["primary_test"],
        },
        "scorer_implementation_status": "code_present_geometry_receipts_pending",
    }


def _v4_reserved_env_seeds(config: dict) -> set[int]:
    seed = config["seed_reservation"]
    base, stride = seed["environment_base"], seed["fixture_stride"]
    substitutions = v4.seed_substitution_map(config)
    reserved: set[int] = set()
    for fixture_id, fixture in config["fixtures"].items():
        blocks = max(f["blocks"] for f in config["families"] if f["fixture"] == fixture_id)
        for block in range(blocks):
            substitution = substitutions.get((fixture_id, block))
            reserved.add(
                substitution["replacement_seed"]
                if substitution is not None
                else base + fixture["seed_slot"] * stride + block
            )
    return reserved


def _v4_pilot_env_seeds(config: dict) -> list[dict]:
    pilot_base = config["seed_reservation"]["pilot_base"]
    per_group = (
        config["engineering_pilots"]["stationary_per_policy_fixture"]
        + config["engineering_pilots"]["motion_per_policy_fixture"]
    )
    rows = []
    for index, (policy, fixture) in enumerate(PILOT_GROUPS):
        for offset in range(per_group):
            rows.append(
                {
                    "pilot_group_index": index,
                    "policy": policy,
                    "fixture": fixture,
                    "env_seed": pilot_base + index * 100 + offset,
                    "cohort": "engineering_pilot",
                }
            )
    return rows


def scan_historical_seeds(
    artifacts_root: Path,
    *,
    exclude_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Best-effort scan of committed artifact JSON/JSONL under artifacts_root.

    This is not an exhaustive repository-wide seed registry audit. It regex-scans
    selected seed field names in artifact files only, excluding configured prefixes.
    """
    historical_env: dict[int, list[str]] = defaultdict(list)
    historical_policy: dict[int, list[str]] = defaultdict(list)
    env_re = re.compile(r'"(?:environment_seed|env_seed|reset_seed)"\s*:\s*(\d+)')
    policy_re = re.compile(r'"(?:policy_seed|sampling_seed)"\s*:\s*(\d+)')
    files_scanned = 0
    for pattern in ("*.json", "*.jsonl"):
        for path in artifacts_root.rglob(pattern):
            rel = str(path.relative_to(ROOT))
            if any(rel.startswith(prefix) for prefix in exclude_prefixes):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            files_scanned += 1
            for match in env_re.finditer(text):
                value = int(match.group(1))
                if rel not in historical_env[value]:
                    historical_env[value].append(rel)
            for match in policy_re.finditer(text):
                value = int(match.group(1))
                if rel not in historical_policy[value]:
                    historical_policy[value].append(rel)
    return {
        "env": {str(k): v for k, v in sorted(historical_env.items())},
        "policy": {str(k): v for k, v in sorted(historical_policy.items())},
        "files_scanned": files_scanned,
    }


def _v4_reserved_policy_seeds(rows: list[dict]) -> set[int]:
    return {row["policy_seed"] for row in rows}


def build_seed_manifest(config: dict, rows: list[dict]) -> dict[str, Any]:
    seed_cfg = config["seed_reservation"]
    reserved_env = _v4_reserved_env_seeds(config)
    substitutions = seed_cfg.get("post_result_environment_seed_substitutions", [])
    retired_env = {row["retired_seed"] for row in substitutions}
    reserved_policy = _v4_reserved_policy_seeds(rows)
    confirmatory = sorted(
        (
            {
                "env_seed": row["env_seed"],
                "policy_seed": row["policy_seed"],
                "fixture": row["fixture"],
                "block_id": row["block_id"],
                "policy": row["factors"]["policy"],
                "episode_id": row["episode_id"],
            }
            for row in rows
        ),
        key=lambda item: (item["fixture"], item["block_id"], item["policy"], item["episode_id"]),
    )
    pilot_rows = _v4_pilot_env_seeds(config)
    pilot_env = {row["env_seed"] for row in pilot_rows}
    scan = scan_historical_seeds(
        ROOT / "artifacts",
        exclude_prefixes=("artifacts/online_correction_v4/",),
    )
    historical_env = {int(k): v for k, v in scan["env"].items()}
    historical_policy = {int(k): v for k, v in scan["policy"].items()}
    v4_env_all = reserved_env | retired_env | pilot_env

    def _collisions(reserved: set[int], historical: dict[int, list[str]], namespace: str) -> list[dict]:
        out = []
        for value in sorted(reserved):
            if value in historical:
                out.append(
                    {
                        "seed": value,
                        "seed_kind": namespace,
                        "historical_paths": historical[value][:20],
                        "historical_path_count": len(historical[value]),
                    }
                )
        return out

    env_collisions = _collisions(v4_env_all, historical_env, "environment")
    policy_collisions = _collisions(reserved_policy, historical_policy, "policy")
    collisions = env_collisions + policy_collisions
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "reservation": seed_cfg,
        "confirmatory_env_seed_count": len(reserved_env),
        "confirmatory_unique_env_seeds": sorted(reserved_env),
        "retired_model_blind_setup_env_seeds": sorted(retired_env),
        "post_result_environment_seed_substitutions": substitutions,
        "confirmatory_unique_policy_seeds": sorted(reserved_policy),
        "confirmatory_rows": confirmatory,
        "engineering_pilot_groups": PILOT_GROUPS,
        "engineering_pilot_seeds": pilot_rows,
        "policy_seed_derivation": "sha256(policy_seed_namespace, policy, fixture, block) mod 2^31",
        "historical_collision_audit": {
            "required": seed_cfg["historical_collision_audit_required"],
            "method": "regex_scan_of_artifact_json_and_jsonl_under_artifacts/",
            "scope": "committed_artifact_tree_only_not_handoff_or_external_outputs",
            "excluded_prefixes": ["artifacts/online_correction_v4/"],
            "files_scanned": scan["files_scanned"],
            "historical_unique_env_seed_count": len(historical_env),
            "historical_unique_policy_seed_count": len(historical_policy),
            "v4_reserved_env_seed_count": len(v4_env_all),
            "v4_reserved_policy_seed_count": len(reserved_policy),
            "env_collisions": env_collisions,
            "policy_collisions": policy_collisions,
            "collisions": collisions,
            "passed": len(collisions) == 0,
            "limitations": [
                "Scan is best-effort over artifact JSON fields; it is not a complete repository seed registry.",
                "Seeds appearing only in code, handoff bundles, or cluster storage outside artifacts/ are not scanned.",
            ],
        },
    }


def build_protocol(
    config: dict,
    config_sha256: str,
    planning_manifest_sha256: str,
    frozen_queue_sha256: str,
    generation_parent_commit: str,
) -> dict[str, Any]:
    design_validation_path = ROOT / "docs/online_correction_v4/design_validation.json"
    design_validation_sha256 = file_sha256(design_validation_path) if design_validation_path.is_file() else None
    design_validation = json.loads(design_validation_path.read_text()) if design_validation_path.is_file() else {}
    return {
        "schema_version": 1,
        "protocol_id": "online_correction_v4",
        "campaign_id": config["campaign_id"],
        "status": "PROSPECTIVE_FROZEN_DESIGN_NOT_RELEASED",
        "source_commit_design": config["source_commit"],
        "generation_parent_commit": generation_parent_commit,
        "generation_parent_commit_meaning": (
            "Git HEAD when the freeze builder last ran; parent commit of the working tree. "
            "This is NOT the commit containing freeze artifacts and cannot self-reference the freeze commit."
        ),
        "git_receipt": {
            "status": "pending",
            "meaning": "Populate after merge with the commit SHA that contains this freeze directory.",
            "expected_fields": ["freeze_commit", "freeze_commit_message", "parent_commit"],
        },
        "config_sha256": config_sha256,
        "planning_manifest_sha256": planning_manifest_sha256,
        "frozen_queue_sha256": frozen_queue_sha256,
        "hash_semantics": {
            "planning_manifest_sha256": "Pre-enrichment rows from tools/online_correction_v4.py build_manifest (matches docs/online_correction_v4/design_validation.json planning_manifest_sha256 when campaign unchanged).",
            "frozen_queue_sha256": "Enriched queue.jsonl bytes including prompt_text, prompt_sha256, and queue_row_kind.",
        },
        "design_validation_reference": {
            "path": "docs/online_correction_v4/design_validation.json",
            "sha256": design_validation_sha256,
            "planning_manifest_sha256": design_validation.get("planning_manifest_sha256"),
            "planning_manifest_sha256_matches": design_validation.get("planning_manifest_sha256") == planning_manifest_sha256,
        },
        "research_questions": [
            "Does online correction depend on the spatial goal and named reference?",
            "How consistently does behavior survive equivalent wording and focused relation/object/stack changes?",
            "Does shortening the executed action prefix improve correction under the same controlled observation-to-action delay?",
        ],
        "primary_outcomes": ["success", "goal_violation_capped_m", "reference_selectivity_m"],
        "independent_unit": "randomized_reset_block",
        "expected_confirmatory_episodes": config["expected_confirmatory_episodes"],
        "excluded_engineering_policy_pilots": config["engineering_pilots"]["expected_policy_episodes"],
        "families": [
            {
                "id": f["id"],
                "fixture": f["fixture"],
                "blocks": f["blocks"],
                "expected_new_episodes": f["expected_new_episodes"],
                "priority": f["priority"],
                "reuses": f.get("reuses", []),
            }
            for f in config["families"]
        ],
        "shared_control_rules": "C3/C4 reuse explicit C1/C3 episode IDs; no silent reruns",
        "failure_taxonomy": "preserved_valid_failures_distinguish_infra_physics_no_trigger",
        "stopping_rule": "every_planned_episode_accepted_blocked_or_infra_resolved_with_reason",
        "simulator_state_extension": "external_reference_motion_for_trigger_feasibility_scoring_only",
        "policy_input_rule": "static_prompt_and_released_observation_interface_only",
        "historical_protocol_immutability": "artifacts/vla_wam_shared_v2/protocol.json and artifacts/vla_wam_shared_v3/protocol.json byte-for-byte",
    }


def build_frozen_analysis_manifest(config: dict) -> dict[str, Any]:
    analysis = config["analysis"]
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "status": "implementation_present_confirmatory_inputs_pending",
        "analysis_module": "experiments/online_correction_v4/analysis.py",
        "primary_contrasts": analysis["primary_contrasts"],
        "primary_tests": analysis["primary_tests"],
        "alpha": analysis["alpha"],
        "multiplicity": analysis["multiplicity"],
        "bootstrap_resamples": analysis["bootstrap_resamples"],
        "analysis_rng_seed": analysis["analysis_rng_seed"],
        "resampling_unit": analysis["resampling_unit"],
        "primary_null": analysis["primary_null"],
        "primary_test": analysis["primary_test"],
        "standard_error": analysis["standard_error"],
        "confidence_interval": analysis["confidence_interval"],
        "bootstrap_pvalue": analysis["bootstrap_pvalue"],
        "studentized_T_star": analysis["studentized_T_star"],
        "unestimable_rule": analysis["unestimable_rule"],
        "primary_C2_aggregation": analysis["primary_C2_aggregation"],
        "holm_missing_test_rule": analysis["holm_missing_test_rule"],
        "no_equivalence_claims": analysis["no_equivalence_claims"],
        "fixed_sample_size": analysis["fixed_sample_size"],
        "C2_prefix_requirement": {
            "required_modes": ["deterministic_fresh_session_replay", "qualified_full_state_snapshot"],
            "status": "blocked_pending_verification",
            "block_reason": C2_BLOCK_REASON,
        },
        "semantic_prompt_binding": {
            "primary_key": "prompt_id",
            "forbidden_semantic_primary_keys": ["prompt_sha256"],
            "episode_level_fields": ["episode_id", "prompt_id", "prompt_text", "prompt_sha256"],
            "note": (
                "C2 may reuse identical prompt_sha256 across distinct prompt_id values when counterbalance "
                "assigns the same words to different physical A/B identities. Never aggregate or join analysis "
                "rows on prompt_sha256 alone."
            ),
        },
        "required_derived_exports": [
            "coverage_by_cell.csv",
            "episode_outcomes.parquet",
            "event_outcomes.parquet",
            "paired_contrasts.parquet",
            "primary_results.csv",
            "scope_replications.csv",
            "wording_results.csv",
            "failure_composition.csv",
            "timing_and_motion.csv",
            "audit_report.json",
            "results_manifest.json",
        ],
    }


def family_gate_status(family_id: str) -> dict[str, str]:
    if family_id == "C8":
        return {
            "lifecycle_status": "BLOCKED_RUNTIME",
            "release_state": "NOT_RELEASED",
            "block_reason": C8_BLOCK_REASON,
            "disposition": "hard_blocked",
        }
    if family_id == "C2":
        return {
            "lifecycle_status": "BLOCKED_SETUP",
            "release_state": "NOT_RELEASED",
            "block_reason": C2_BLOCK_REASON,
            "disposition": "hard_blocked",
        }
    return {
        "lifecycle_status": "IMPLEMENTING",
        "release_state": "NOT_RELEASED",
        "block_reason": DEFAULT_FAMILY_PENDING,
        "disposition": "pending_qualification",
    }


def build_historical_seed_receipt(seed_manifest: dict, seed_manifest_sha256: str) -> dict[str, Any]:
    audit = seed_manifest.get("historical_collision_audit", {})
    passed = audit.get("passed") is True
    env_n = len(audit.get("env_collisions", []))
    policy_n = len(audit.get("policy_collisions", []))
    if passed:
        reason = "No collision between reserved V4 env/policy seeds and scanned repository artifact seeds."
        status = "passed_at_freeze_build"
    else:
        reason = (
            f"Reserved V4 seeds collide with scanned artifact seeds "
            f"(env={env_n}, policy={policy_n})."
        )
        status = "failed_at_freeze_build"
    return {
        "passed": passed,
        "status": status,
        "family_ids": [],
        "uri": "artifacts/online_correction_v4/seed_manifest.json",
        "sha256": seed_manifest_sha256,
        "reason": reason,
        "derived_from": "seed_manifest.historical_collision_audit",
        "audit_summary": {
            "env_collision_count": env_n,
            "policy_collision_count": policy_n,
            "files_scanned": audit.get("files_scanned"),
        },
    }


def build_gate_report(
    config: dict,
    seed_manifest: dict,
    seed_manifest_sha256: str,
) -> dict[str, Any]:
    receipts = {
        name: {
            "passed": False,
            "status": "pending",
            "family_ids": [],
            "uri": None,
            "sha256": None,
            "reason": "Receipt requires runtime, geometry, or cluster evidence not available before model use.",
        }
        for name in config["required_release_receipts"]
    }
    seed_receipt = build_historical_seed_receipt(seed_manifest, seed_manifest_sha256)
    seed_receipt["family_ids"] = [f["id"] for f in config["families"]]
    receipts["historical_seed_collision_audit"] = seed_receipt
    families = {}
    hard_blocked: dict[str, str] = {}
    pending_not_released: dict[str, str] = {}
    for family in config["families"]:
        fid = family["id"]
        gate = family_gate_status(fid)
        families[fid] = {
            **gate,
            "priority": family["priority"],
            "expected_new_episodes": family["expected_new_episodes"],
            "qualification_gates": {
                "G0_source_and_access": "pending",
                "G1_infrastructure": "pending",
                "G2_state_and_coordinates": "pending",
                "G3_motion_and_feasibility": "pending",
                "G4_policy_session": "pending",
                "G5_trigger_and_branch_replay": "blocked" if fid == "C2" else "pending",
                "G6_measurement": "pending",
                "G7_engineering_pilot": "pending",
                "G8_miniature_campaign": "pending",
            },
        }
        if gate["disposition"] == "hard_blocked":
            hard_blocked[fid] = gate["block_reason"]
        else:
            pending_not_released[fid] = gate["block_reason"]
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "overall_status": "QUALIFYING",
        "release_status": "NOT_RELEASED",
        "released_families": [],
        "hard_blocked_families": hard_blocked,
        "pending_not_released_families": pending_not_released,
        "blocked_families": {**hard_blocked, **pending_not_released},
        "families": families,
        "required_release_receipts": receipts,
        "runtime_lock_status": "template_only_not_released",
    }


def enrich_queue_rows(config: dict, rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        text, meta = resolve_prompt_text(config, row["fixture"], row["factors"], row.get("counterbalance"))
        prompt_id = v4.digest(prompt_identity(config, row))[:24]
        enriched.append(
            {
                **row,
                "prompt_id": prompt_id,
                "prompt_text": text,
                "prompt_sha256": v4.digest_bytes(text.encode("utf-8")),
                "prompt_physical_resolution": meta["physical_resolution"],
                "launch_critical_names_resolved": meta["launch_critical_names_resolved"],
                "reference_color": meta.get("reference_color"),
                "physical_A_color": meta.get("physical_A_color"),
                "queue_row_kind": "new_episode",
                "reuse_episode_ids_meaning": (
                    "comparison_control_links_only; this row remains a registered new episode "
                    "and is never a reuse-only alias."
                ),
            }
        )
    return enriched


def build_queue_manifest(
    config: dict,
    rows: list[dict],
    planning_manifest_sha256: str,
    frozen_queue_sha256: str,
) -> dict[str, Any]:
    by_family = Counter(row["family"] for row in rows)
    by_policy_fixture = Counter(row["execution_group"] for row in rows)
    edge_counts: Counter[str] = Counter()
    unique_refs: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        edge_counts[row["family"]] += len(row.get("reuse_episode_ids", []))
        unique_refs[row["family"]].update(row.get("reuse_episode_ids", []))
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "queue_path": "artifacts/online_correction_v4/queue.jsonl",
        "row_count": len(rows),
        "planning_manifest_sha256": planning_manifest_sha256,
        "frozen_queue_sha256": frozen_queue_sha256,
        "queue_sha256": frozen_queue_sha256,
        "hash_semantics": {
            "planning_manifest_sha256": "Pre-enrichment planning inventory from build_manifest.",
            "frozen_queue_sha256": "Enriched queue.jsonl bytes written by the freeze builder.",
        },
        "expected_confirmatory_episodes": config["expected_confirmatory_episodes"],
        "registered_new_episodes_by_family": dict(sorted(by_family.items())),
        "rows_by_execution_group": dict(sorted(by_policy_fixture.items())),
        "total_control_reference_edges_by_family": dict(sorted(edge_counts.items())),
        "unique_referenced_control_episode_ids_by_family": {
            family: len(ids) for family, ids in sorted(unique_refs.items())
        },
        "reuse_semantics": (
            "Every queue row is a registered new episode. reuse_episode_ids link to prior "
            "control episode IDs for analysis pairing; C3/C4 rows remain new episodes even when "
            "they reference C1/C3 controls. C4 fast-schedule sham/move rows are new because schedule differs."
        ),
        "excluded_engineering_policy_pilots": config["engineering_pilots"]["expected_policy_episodes"],
        "release_status": "NOT_RELEASED",
    }


def build_runtime_manifest_stub(config: dict) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "release_status": "NOT_RELEASED",
        "runner": {
            "entrypoint": None,
            "commit": None,
            "sha256": None,
        },
        "policies": {
            policy_id: {
                "checkpoint_uri": None,
                "checkpoint_sha256": None,
                "runtime_image_digest": None,
                "integration_commit": None,
                "native_control_dt_s": None,
                "achieved_delay_s": None,
                "achieved_standard_query_period_s": None,
                "achieved_fast_query_period_s": None,
                "prediction_horizon_actions": None,
            }
            for policy_id in config["policies"]
        },
        "limitations": [
            "Launch-critical runtime fields remain null until qualification receipts pass.",
        ],
    }


def build_setup_manifest_stub(config: dict) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "release_status": "NOT_RELEASED",
        "fixtures": {
            fixture_id: {
                "geometry_uri": None,
                "geometry_sha256": None,
                "frame_transform_uri": None,
                "reset_registry_uri": None,
                "calibration_scale": None,
                "D_cap_m": None,
            }
            for fixture_id in config["fixtures"]
        },
        "limitations": [
            "Task geometry, transforms, and reset registries remain unbound before model-blind gates.",
        ],
    }


def build_launch_matrix_stub(config: dict) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "release_status": "NOT_RELEASED",
        "cluster_context": None,
        "namespace": None,
        "lane_bundle_identity": None,
        "resource_budget": None,
        "qualified_lanes": [],
        "limitations": [
            "Cluster mapping and immutable lane specs remain pending infrastructure qualification.",
        ],
    }


def discover_qualification_receipts() -> list[dict[str, Any]]:
    receipt_dir = DEFAULT_OUT / "qualification"
    if not receipt_dir.is_dir():
        return []
    receipts: list[dict[str, Any]] = []
    for path in sorted(receipt_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"qualification receipt must be an object: {path}")
        receipts.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": file_sha256(path),
                "compiled_at_utc": payload.get("compiled_at_utc"),
                "qualification_scope": payload.get("qualification_scope"),
                "gate": payload.get("gate"),
                "attempt_id": payload.get("attempt_id"),
                "status": payload.get("status"),
                "behavioral_episode_count": payload.get("behavioral_episode_count"),
                "completed_lane_status": (payload.get("completed_lane") or {}).get("status"),
                "partial_lane_status": (payload.get("partial_b200_lane") or {}).get("status"),
            }
        )
    return receipts


def discover_model_blind_candidates() -> list[dict[str, Any]]:
    setup_dir = DEFAULT_OUT / "setup"
    if not setup_dir.is_dir():
        return []
    candidates: list[dict[str, Any]] = []
    for path in sorted(setup_dir.glob("*.candidate.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"model-blind candidate must be an object: {path}")
        candidates.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": file_sha256(path),
                "schema_version": payload.get("schema_version"),
                "fixture_id": payload.get("fixture_id"),
                "status": payload.get("status"),
                "amendment_status": payload.get("amendment_status"),
                "model_request_count": payload.get("model_request_count"),
                "behavioral_episode_count": payload.get("behavioral_episode_count"),
                "registered_env_seed_count": payload.get("registered_env_seed_count"),
            }
        )
    return candidates


def build_continuation_state(
    config: dict,
    generation_parent_commit: str,
    artifact_hashes: dict[str, str],
    planning_manifest_sha256: str,
    frozen_queue_sha256: str,
    gate_report: dict[str, Any],
    qualification_receipts: list[dict[str, Any]],
    model_blind_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    qualification_paths = [row["path"] for row in qualification_receipts]
    candidate_paths = [row["path"] for row in model_blind_candidates]
    g2_setup_failed = any(
        row.get("gate") == "G2"
        and row.get("status") == "failed_model_blind_setup_gate"
        for row in qualification_receipts
    )
    g2_requalification_failed = any(
        row.get("gate") == "G2"
        and row.get("attempt_id") == "g2q20260905f"
        and row.get("status") == "failed_model_blind_setup_gate"
        for row in qualification_receipts
    )
    g2_requalification_frozen = any(
        row.get("schema_version") == "v4-horizontal-g2-post-result-amendment-v1"
        and row.get("amendment_status") == "frozen_for_model_blind_requalification"
        for row in model_blind_candidates
    )
    if g2_requalification_failed:
        cluster_blocker = (
            "BLOCKED_SETUP: amended complete horizontal G2 attempt "
            "g2q20260905f passed 126/128 seeds, while seeds 2100000052 and "
            "2100000101 reproduced object-stability failures. Freeze a new "
            "model-blind seed-substitution or setup-geometry amendment before "
            "another G2 attempt. Do not weaken stability thresholds; G3 and "
            "policy inference remain prohibited."
        )
    elif g2_setup_failed and g2_requalification_frozen:
        cluster_blocker = (
            "REQUALIFICATION_PENDING: complete horizontal G2 attempt "
            "g2q20260905e remains failed. A disclosed zero-model-request amendment "
            "authorizes only a fresh full G2 attempt at the frozen 5 mm registry "
            "position tolerance; G3 and policy inference remain prohibited."
        )
    elif g2_setup_failed:
        cluster_blocker = (
            "BLOCKED_SETUP: complete horizontal G2 attempt g2q20260905e failed "
            "with 64/128 passing seeds; freeze a disclosed model-blind reset/"
            "settling amendment before any fresh G2 attempt. G3 and policy inference "
            "remain prohibited."
        )
    elif qualification_receipts:
        cluster_blocker = (
            "Additional model-blind and behavioral qualification gates remain pending; "
            "completed infrastructure-only strata are listed in qualification_receipts."
        )
    else:
        cluster_blocker = (
            "Runtime lock, geometry receipts, checkpoint identity, and cluster "
            "qualification remain pending."
        )
    g2_setup_commands = (
        []
        if g2_requalification_failed
        or (g2_setup_failed and not g2_requalification_frozen)
        else [
            "python3 tools/build_v4_horizontal_reset_registry.py",
            "python3 tools/build_v4_horizontal_g3_plan.py",
            "python3 tools/render_v4_horizontal_g2_k8s_jobs.py --spec deploy/k8s/v4_lane_bundle/g2-horizontal-spec.example.json --output-root \"$V4_G2_RENDER_ROOT\"",
            "python3 tools/validate_v4_horizontal_g2_k8s_jobs.py --root \"$V4_G2_BUNDLE_ROOT\"",
        ]
    )
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "status": "QUALIFYING",
        "implementation_status": "IMPLEMENTING",
        "release_status": "NOT_RELEASED",
        "active_branch": "research/online-correction-v4",
        "generation_parent_commit": generation_parent_commit,
        "generation_parent_commit_meaning": (
            "Git HEAD when freeze builder last ran; not the commit containing freeze artifacts."
        ),
        "git_receipt": {
            "status": "pending",
            "meaning": "Record freeze_commit after the commit containing artifacts/online_correction_v4/ lands.",
        },
        "design_source_commit": config["source_commit"],
        "authoritative_files": [
            "docs/online_correction_v4/README.md",
            "docs/online_correction_v4/campaign.json",
            "docs/ONLINE_CORRECTION_V4_CONTINUATION.md",
            "artifacts/online_correction_v4/freeze_manifest.json",
            "artifacts/online_correction_v4/continuation_state.json",
            "artifacts/online_correction_v4/protocol.json",
            "artifacts/online_correction_v4/queue.jsonl",
            "artifacts/online_correction_v4/queue_manifest.json",
            "artifacts/online_correction_v4/prompt_manifest.json",
            "artifacts/online_correction_v4/motion_manifest.json",
            "artifacts/online_correction_v4/scoring_manifest.json",
            "artifacts/online_correction_v4/seed_manifest.json",
            "artifacts/online_correction_v4/frozen_analysis_manifest.json",
            "artifacts/online_correction_v4/gate_report.json",
            "artifacts/online_correction_v4/historical_protocol_ledger.json",
            "artifacts/online_correction_v4/runtime_manifest.json",
            "artifacts/online_correction_v4/setup_manifest.json",
            "artifacts/online_correction_v4/launch_matrix.json",
        ] + qualification_paths + candidate_paths,
        "artifact_sha256": artifact_hashes,
        "qualification_receipts": qualification_receipts,
        "model_blind_candidates": model_blind_candidates,
        "planning_manifest_sha256": planning_manifest_sha256,
        "frozen_queue_sha256": frozen_queue_sha256,
        "policy_episodes_executed": 0,
        "completed_groups": [],
        "hard_blocked_families": gate_report["hard_blocked_families"],
        "pending_not_released_families": gate_report["pending_not_released_families"],
        "active_blockers": [
            C8_BLOCK_REASON,
            C2_BLOCK_REASON,
            cluster_blocker,
        ],
        "next_commands": [
            "python3 tools/online_correction_v4.py validate",
        ] + g2_setup_commands + [
            "python3 tools/build_online_correction_v4_freeze.py --out artifacts/online_correction_v4",
            "python3 tools/validate_online_correction_v4.py",
            "python3 -m unittest discover -s tests -p 'test_online_correction_v4*.py'",
            "python3 tools/validate_vla_wam_v2_protocol.py",
            "python3 tools/validate_vla_wam_v3_protocol.py",
        ],
        "limitations": [
            "Prospective design freeze only; no V4 policy inference has been run.",
            "Launch-critical runtime manifests remain unreleased until qualification receipts pass.",
        ],
    }


# Every artifact emitted by build_freeze; validator fails closed if any name is uncovered.
ALL_GENERATED_FREEZE_ARTIFACTS = (
    "historical_protocol_ledger.json",
    "protocol.json",
    "prompt_manifest.json",
    "motion_manifest.json",
    "scoring_manifest.json",
    "seed_manifest.json",
    "queue.jsonl",
    "queue_manifest.json",
    "frozen_analysis_manifest.json",
    "runtime_manifest.json",
    "setup_manifest.json",
    "launch_matrix.json",
    "gate_report.json",
    "continuation_state.json",
    "freeze_manifest.json",
)

# Artifact file names whose raw bytes must match across deterministic rebuilds.
DETERMINISTIC_ARTIFACT_NAMES = (
    "historical_protocol_ledger.json",
    "prompt_manifest.json",
    "motion_manifest.json",
    "scoring_manifest.json",
    "seed_manifest.json",
    "queue.jsonl",
    "queue_manifest.json",
    "frozen_analysis_manifest.json",
    "runtime_manifest.json",
    "setup_manifest.json",
    "launch_matrix.json",
    "gate_report.json",
)

# JSON artifacts that also embed generation_parent_commit and are compared normalized.
GENERATION_PARENT_COMMIT_ARTIFACTS = (
    "protocol.json",
    "continuation_state.json",
    "freeze_manifest.json",
)

# freeze_manifest.json cannot include its own hash inside artifact_sha256 (self-reference).
FREEZE_MANIFEST_SELF_HASH_EXCLUDED = "freeze_manifest.json"


def build_freeze(
    config_path: Path,
    out_dir: Path,
    *,
    generation_parent_commit: str | None = None,
) -> dict[str, Any]:
    config, config_sha256 = v4.load_json(config_path)
    errors = v4.config_errors(config)
    if errors:
        raise ValueError("; ".join(errors))
    generation_parent_commit = generation_parent_commit or repo_head_commit()
    base_rows = v4.build_manifest(config, config_sha256)
    planning_bytes = v4.manifest_bytes(base_rows)
    planning_manifest_sha256 = v4.digest_bytes(planning_bytes)
    rows = enrich_queue_rows(config, base_rows)
    queue_bytes = v4.manifest_bytes(rows)
    frozen_queue_sha256 = v4.digest_bytes(queue_bytes)

    artifact_hashes: dict[str, str] = {}
    artifact_hashes["historical_protocol_ledger.json"] = write_json(
        out_dir / "historical_protocol_ledger.json", historical_protocol_ledger()
    )
    artifact_hashes["protocol.json"] = write_json(
        out_dir / "protocol.json",
        build_protocol(config, config_sha256, planning_manifest_sha256, frozen_queue_sha256, generation_parent_commit),
    )
    artifact_hashes["prompt_manifest.json"] = write_json(
        out_dir / "prompt_manifest.json", build_prompt_manifest(config, rows)
    )
    artifact_hashes["motion_manifest.json"] = write_json(
        out_dir / "motion_manifest.json", build_motion_manifest(config)
    )
    artifact_hashes["scoring_manifest.json"] = write_json(
        out_dir / "scoring_manifest.json", build_scoring_manifest(config)
    )
    seed_manifest = build_seed_manifest(config, rows)
    artifact_hashes["seed_manifest.json"] = write_json(out_dir / "seed_manifest.json", seed_manifest)
    artifact_hashes["queue.jsonl"] = write_bytes(out_dir / "queue.jsonl", queue_bytes)
    artifact_hashes["queue_manifest.json"] = write_json(
        out_dir / "queue_manifest.json",
        build_queue_manifest(config, rows, planning_manifest_sha256, frozen_queue_sha256),
    )
    artifact_hashes["frozen_analysis_manifest.json"] = write_json(
        out_dir / "frozen_analysis_manifest.json", build_frozen_analysis_manifest(config)
    )
    artifact_hashes["runtime_manifest.json"] = write_json(
        out_dir / "runtime_manifest.json", build_runtime_manifest_stub(config)
    )
    artifact_hashes["setup_manifest.json"] = write_json(
        out_dir / "setup_manifest.json", build_setup_manifest_stub(config)
    )
    artifact_hashes["launch_matrix.json"] = write_json(
        out_dir / "launch_matrix.json", build_launch_matrix_stub(config)
    )
    gate_report = build_gate_report(config, seed_manifest, artifact_hashes["seed_manifest.json"])
    artifact_hashes["gate_report.json"] = write_json(out_dir / "gate_report.json", gate_report)
    artifact_hashes["continuation_state.json"] = write_json(
        out_dir / "continuation_state.json",
        build_continuation_state(
            config,
            generation_parent_commit,
            artifact_hashes,
            planning_manifest_sha256,
            frozen_queue_sha256,
            gate_report,
            discover_qualification_receipts(),
            discover_model_blind_candidates(),
        ),
    )
    freeze_index = {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "generation_parent_commit": generation_parent_commit,
        "generation_parent_commit_meaning": (
            "Git HEAD when builder ran; not the commit containing these freeze artifacts."
        ),
        "git_receipt": {"status": "pending"},
        "config_sha256": config_sha256,
        "planning_manifest_sha256": planning_manifest_sha256,
        "frozen_queue_sha256": frozen_queue_sha256,
        "release_status": "NOT_RELEASED",
        "artifact_sha256": artifact_hashes,
    }
    artifact_hashes["freeze_manifest.json"] = write_json(out_dir / "freeze_manifest.json", freeze_index)
    return {
        "ok": True,
        "out_dir": str(out_dir),
        "generation_parent_commit": generation_parent_commit,
        "config_sha256": config_sha256,
        "planning_manifest_sha256": planning_manifest_sha256,
        "frozen_queue_sha256": frozen_queue_sha256,
        "queue_sha256": frozen_queue_sha256,
        "row_count": len(rows),
        "seed_collision_audit_passed": seed_manifest["historical_collision_audit"]["passed"],
        "artifact_sha256": artifact_hashes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        report = build_freeze(args.config, args.out)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
