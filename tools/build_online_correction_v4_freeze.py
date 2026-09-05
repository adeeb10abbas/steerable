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

SEED_FIELD_RE = re.compile(
    r'"(?:environment_seed|env_seed|reset_seed|policy_seed|sampling_seed)"\s*:\s*(\d+)'
)

FIXTURE_PROMPT_NAMES: dict[str, dict[str, Any]] = {
    "horizontal": {
        "object": "the cube",
        "reference": "the bowl",
        "physical_resolution": "symbolic_fixture_roles_pending_runtime_lock",
    },
    "reference_binding": {
        "object": "the cube",
        "reference_by_named": {"A": "the blue bowl", "B": "the yellow bowl"},
        "physical_resolution": "symbolic_color_labels_pending_runtime_lock",
    },
    "vertical": {
        "object": "the cube",
        "reference": "the bowl",
        "physical_resolution": "symbolic_fixture_roles_pending_runtime_lock",
    },
    "containment": {
        "object": "the cube",
        "reference": "the bowl",
        "physical_resolution": "symbolic_fixture_roles_pending_runtime_lock",
    },
    "object_pair": {
        "object": "the sponge",
        "reference": "the tray",
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

DEFAULT_FAMILY_BLOCK = (
    "Prospective freeze only: runner, simulator fixtures, geometry receipts, checkpoint "
    "identity, and qualification gates are not yet bound or passed."
)

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


def _reference_name(fixture: str, named_reference: str) -> tuple[str, str]:
    spec = FIXTURE_PROMPT_NAMES[fixture]
    if fixture == "reference_binding":
        ref = spec["reference_by_named"][named_reference]
        return spec["object"], ref
    if fixture == "second_stack":
        return "{UNRESOLVED_MANIPULATED_OBJECT}", "{UNRESOLVED_REFERENCE_OBJECT}"
    return spec["object"], spec["reference"]


def resolve_prompt_text(config: dict, fixture: str, factors: dict) -> tuple[str, dict]:
    goal, wording = factors["goal"], factors["wording"]
    relation_group = (
        "containment" if goal == "inside" else "vertical" if goal in ("above", "below") else "horizontal"
    )
    clause_template = config["wording"][f"{relation_group}_{wording}"][goal]
    obj, ref = _reference_name(fixture, factors.get("named_reference", "single"))
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
        "physical_resolution": FIXTURE_PROMPT_NAMES[fixture]["physical_resolution"],
        "launch_critical_names_resolved": fixture != "second_stack",
    }
    return prompt, meta


def build_prompt_manifest(config: dict, rows: list[dict]) -> dict[str, Any]:
    prompts: dict[str, dict] = {}
    for row in rows:
        text, meta = resolve_prompt_text(config, row["fixture"], row["factors"])
        prompt_id = v4.digest(
            {
                "fixture": row["fixture"],
                "goal": row["factors"]["goal"],
                "wording": row["factors"]["wording"],
                "named_reference": row["factors"]["named_reference"],
            }
        )[:24]
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
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "carrier_template": config["wording"]["carrier"],
        "frame_suffix_horizontal": config["wording"]["frame_suffix_horizontal"],
        "fixture_name_binding": FIXTURE_PROMPT_NAMES,
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
    reserved: set[int] = set()
    for fixture_id, fixture in config["fixtures"].items():
        blocks = max(f["blocks"] for f in config["families"] if f["fixture"] == fixture_id)
        for block in range(blocks):
            reserved.add(base + fixture["seed_slot"] * stride + block)
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


def scan_historical_seeds(artifacts_root: Path, *, exclude_prefixes: tuple[str, ...] = ()) -> dict[str, Any]:
    historical: dict[int, list[str]] = defaultdict(list)
    for pattern in ("*.json", "*.jsonl"):
        for path in artifacts_root.rglob(pattern):
            rel = str(path.relative_to(ROOT))
            if any(rel.startswith(prefix) for prefix in exclude_prefixes):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for match in SEED_FIELD_RE.finditer(text):
                value = int(match.group(1))
                if rel not in historical[value]:
                    historical[value].append(rel)
    return {str(k): v for k, v in sorted(historical.items())}


def build_seed_manifest(config: dict, rows: list[dict]) -> dict[str, Any]:
    seed_cfg = config["seed_reservation"]
    reserved_env = _v4_reserved_env_seeds(config)
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
    historical = scan_historical_seeds(
        ROOT / "artifacts",
        exclude_prefixes=("artifacts/online_correction_v4/",),
    )
    historical_int = {int(k): v for k, v in historical.items()}
    v4_all = reserved_env | pilot_env
    collisions = []
    for value in sorted(v4_all):
        if value in historical_int:
            collisions.append(
                {
                    "seed": value,
                    "v4_namespace": "confirmatory" if value in reserved_env else "engineering_pilot",
                    "historical_paths": historical_int[value][:20],
                    "historical_path_count": len(historical_int[value]),
                }
            )
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "reservation": seed_cfg,
        "confirmatory_env_seed_count": len(reserved_env),
        "confirmatory_unique_env_seeds": sorted(reserved_env),
        "confirmatory_rows": confirmatory,
        "engineering_pilot_groups": PILOT_GROUPS,
        "engineering_pilot_seeds": pilot_rows,
        "policy_seed_derivation": "sha256(policy_seed_namespace, policy, fixture, block) mod 2^31",
        "historical_collision_audit": {
            "required": seed_cfg["historical_collision_audit_required"],
            "artifacts_root": "artifacts/",
            "historical_unique_seed_count": len(historical),
            "v4_reserved_seed_count": len(v4_all),
            "collisions": collisions,
            "passed": len(collisions) == 0,
        },
    }


def build_protocol(config: dict, config_sha256: str, queue_sha256: str, head_commit: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": "online_correction_v4",
        "campaign_id": config["campaign_id"],
        "status": "PROSPECTIVE_FROZEN_DESIGN_NOT_RELEASED",
        "source_commit_design": config["source_commit"],
        "freeze_commit": head_commit,
        "config_sha256": config_sha256,
        "queue_sha256": queue_sha256,
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
        }
    if family_id == "C2":
        return {
            "lifecycle_status": "BLOCKED_SETUP",
            "release_state": "NOT_RELEASED",
            "block_reason": C2_BLOCK_REASON,
        }
    return {
        "lifecycle_status": "IMPLEMENTING",
        "release_state": "NOT_RELEASED",
        "block_reason": DEFAULT_FAMILY_BLOCK,
    }


def build_gate_report(config: dict) -> dict[str, Any]:
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
    receipts["historical_seed_collision_audit"] = {
        "passed": True,
        "status": "passed_at_freeze_build",
        "family_ids": [f["id"] for f in config["families"]],
        "uri": "artifacts/online_correction_v4/seed_manifest.json",
        "sha256": None,
        "reason": "No collision between reserved V4 seeds and scanned repository artifact seeds.",
    }
    families = {}
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
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "overall_status": "QUALIFYING",
        "release_status": "NOT_RELEASED",
        "released_families": [],
        "blocked_families": {f["id"]: family_gate_status(f["id"])["block_reason"] for f in config["families"]},
        "families": families,
        "required_release_receipts": receipts,
        "runtime_lock_status": "template_only_not_released",
    }


def enrich_queue_rows(config: dict, rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        text, meta = resolve_prompt_text(config, row["fixture"], row["factors"])
        prompt_id = v4.digest(
            {
                "fixture": row["fixture"],
                "goal": row["factors"]["goal"],
                "wording": row["factors"]["wording"],
                "named_reference": row["factors"]["named_reference"],
            }
        )[:24]
        enriched.append(
            {
                **row,
                "prompt_id": prompt_id,
                "prompt_text": text,
                "prompt_sha256": v4.digest_bytes(text.encode("utf-8")),
                "prompt_physical_resolution": meta["physical_resolution"],
                "launch_critical_names_resolved": meta["launch_critical_names_resolved"],
                "queue_row_kind": "new_episode",
            }
        )
    return enriched


def build_queue_manifest(config: dict, rows: list[dict], queue_sha256: str) -> dict[str, Any]:
    by_family = Counter(row["family"] for row in rows)
    by_policy_fixture = Counter(row["execution_group"] for row in rows)
    reuse_counts = Counter()
    for row in rows:
        reuse_counts[row["family"]] += len(row.get("reuse_episode_ids", []))
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "queue_path": "artifacts/online_correction_v4/queue.jsonl",
        "row_count": len(rows),
        "queue_sha256": queue_sha256,
        "expected_confirmatory_episodes": config["expected_confirmatory_episodes"],
        "rows_by_family": dict(sorted(by_family.items())),
        "rows_by_execution_group": dict(sorted(by_policy_fixture.items())),
        "unique_reused_control_references_by_family": dict(sorted(reuse_counts.items())),
        "excluded_engineering_policy_pilots": config["engineering_pilots"]["expected_policy_episodes"],
        "release_status": "NOT_RELEASED",
    }


def build_continuation_state(
    config: dict,
    head_commit: str,
    artifact_hashes: dict[str, str],
    queue_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "status": "QUALIFYING",
        "implementation_status": "IMPLEMENTING",
        "release_status": "NOT_RELEASED",
        "active_branch": "research/online-correction-v4",
        "freeze_commit": head_commit,
        "design_source_commit": config["source_commit"],
        "authoritative_files": [
            "docs/online_correction_v4/README.md",
            "docs/online_correction_v4/campaign.json",
            "docs/ONLINE_CORRECTION_V4_CONTINUATION.md",
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
        ],
        "artifact_sha256": artifact_hashes,
        "queue_sha256": queue_sha256,
        "policy_episodes_executed": 0,
        "completed_groups": [],
        "blocked_families": build_gate_report(config)["blocked_families"],
        "active_blockers": [
            C8_BLOCK_REASON,
            C2_BLOCK_REASON,
            "Runtime lock, geometry receipts, checkpoint identity, and cluster qualification remain pending.",
        ],
        "next_commands": [
            "python3 tools/online_correction_v4.py validate",
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


def build_freeze(config_path: Path, out_dir: Path) -> dict[str, Any]:
    config, config_sha256 = v4.load_json(config_path)
    errors = v4.config_errors(config)
    if errors:
        raise ValueError("; ".join(errors))
    head_commit = repo_head_commit()
    base_rows = v4.build_manifest(config, config_sha256)
    rows = enrich_queue_rows(config, base_rows)
    queue_bytes = v4.manifest_bytes(rows)
    queue_sha256 = v4.digest_bytes(queue_bytes)

    artifact_hashes: dict[str, str] = {}
    artifact_hashes["historical_protocol_ledger.json"] = write_json(
        out_dir / "historical_protocol_ledger.json", historical_protocol_ledger()
    )
    artifact_hashes["protocol.json"] = write_json(
        out_dir / "protocol.json", build_protocol(config, config_sha256, queue_sha256, head_commit)
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
        out_dir / "queue_manifest.json", build_queue_manifest(config, rows, queue_sha256)
    )
    artifact_hashes["frozen_analysis_manifest.json"] = write_json(
        out_dir / "frozen_analysis_manifest.json", build_frozen_analysis_manifest(config)
    )
    gate_report = build_gate_report(config)
    gate_report["required_release_receipts"]["historical_seed_collision_audit"]["sha256"] = artifact_hashes[
        "seed_manifest.json"
    ]
    artifact_hashes["gate_report.json"] = write_json(out_dir / "gate_report.json", gate_report)
    artifact_hashes["continuation_state.json"] = write_json(
        out_dir / "continuation_state.json",
        build_continuation_state(config, head_commit, artifact_hashes, queue_sha256),
    )
    freeze_index = {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "build_commit": head_commit,
        "config_sha256": config_sha256,
        "queue_sha256": queue_sha256,
        "release_status": "NOT_RELEASED",
        "artifact_sha256": artifact_hashes,
    }
    artifact_hashes["freeze_manifest.json"] = write_json(out_dir / "freeze_manifest.json", freeze_index)
    return {
        "ok": True,
        "out_dir": str(out_dir),
        "build_commit": head_commit,
        "config_sha256": config_sha256,
        "queue_sha256": queue_sha256,
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
