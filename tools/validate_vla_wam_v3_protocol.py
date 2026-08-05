#!/usr/bin/env python3
"""Fail-closed validation for the frozen VLA/WAM steerability v3 registry."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    """Raised when a v3 invariant is absent or altered."""


V3 = "artifacts/vla_wam_shared_v3"
REQUIRED = {
    "protocol": f"{V3}/protocol.json",
    "amendment": f"{V3}/post_result_power_failure_ablation_amendment.json",
    "droid": f"{V3}/droid_direct_registry.json",
    "robotwin": f"{V3}/robotwin_direct_registry.json",
    "wording": f"{V3}/four_phrasings_registry.json",
    "stochastic": f"{V3}/stochastic_rollout_registry.json",
    "confound_calibration": f"{V3}/confound_fixture_calibration_registry.json",
    "phase_a_queue": f"{V3}/phase_a_cells.jsonl",
    "phase_a_manifest": f"{V3}/phase_a_cells_manifest.json",
    "taxonomy": f"{V3}/failure_taxonomy.json",
    "analysis": f"{V3}/analysis_plan.json",
    "document": "docs/VLA_WAM_STEERABILITY_V3_PROTOCOL.md",
}
DROID_MODELS = [
    "pi0_fast_droid_vla",
    "groot_n17_droid_vla",
    "cosmos3_edge_policy_droid",
    "cosmos3_nano_policy_droid",
    "pi05_current_stack_droid",
    "dreamzero_droid_action_cfg",
]
ROBOTWIN_MODELS = [
    "efficient_wam_rt_robotwin",
    "fastwam_robotwin",
    "lingbot_va_robotwin",
]
WORDING_MODELS = [
    "groot_n17_droid_vla",
    "cosmos3_edge_policy_droid",
    "cosmos3_nano_policy_droid",
]
EXACT_V2_WORDINGS = {
    "direct_command": {
        "left": "Put the Rubik's cube to the left of the bowl.",
        "right": "Put the Rubik's cube to the right of the bowl.",
    },
    "short_command": {
        "left": "Put the cube left of the bowl.",
        "right": "Put the cube right of the bowl.",
    },
    "goal_as_outcome": {
        "left": "The Rubik's cube should end up to the left of the bowl.",
        "right": "The Rubik's cube should end up to the right of the bowl.",
    },
    "desired_plus_negated_opposite": {
        "left": "Put the Rubik's cube to the left of the bowl, not to the right of the bowl.",
        "right": "Put the Rubik's cube to the right of the bowl, not to the left of the bowl.",
    },
}


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise ValidationError(message)
    checks.append(message)


def load(path: Path) -> dict[str, Any]:
    try:
        with path.open() as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValidationError(f"{path}:{line_number} must contain a JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read {path}: {exc}") from exc
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_range(start: int, end: int) -> list[int]:
    return list(range(start, end + 1))


def validate(root: Path) -> list[str]:
    checks: list[str] = []
    paths = {name: root / relative for name, relative in REQUIRED.items()}
    for name, path in paths.items():
        require(path.is_file(), f"required v3 {name} artifact exists", checks)

    protocol = load(paths["protocol"])
    amendment = load(paths["amendment"])
    droid = load(paths["droid"])
    robotwin = load(paths["robotwin"])
    wording_registry = load(paths["wording"])
    stochastic_registry = load(paths["stochastic"])
    calibration_registry = load(paths["confound_calibration"])
    phase_a_manifest = load(paths["phase_a_manifest"])
    phase_a_rows = load_jsonl(paths["phase_a_queue"])
    taxonomy = load(paths["taxonomy"])
    analysis = load(paths["analysis"])

    require(protocol["schema_version"] == "vla-wam-shared-v3-protocol-v1", "protocol schema is frozen", checks)
    require(protocol["study_id"] == "vla_wam_language_steerability_v3", "protocol study identifier is frozen", checks)
    require(protocol["status"] == "frozen_before_any_v3_model_request_or_behavioral_inference", "protocol prohibits v3 inference before freeze", checks)
    relation = protocol["relationship_to_v2"]
    require("after all v2 and V2-A015 results" in relation["disclosure"], "post-result disclosure covers all v2 and V2-A015 outcomes", checks)
    require("never rewritten" in relation["immutability"] and "rerun" in relation["immutability"], "valid v2 evidence is immutable and non-rerunnable", checks)
    require("never pooled" in relation["arena_boundary"], "protocol forbids cross-arena pooling", checks)
    require(protocol["required_artifacts"] == [
        f"{V3}/post_result_power_failure_ablation_amendment.json",
        f"{V3}/droid_direct_registry.json",
        f"{V3}/robotwin_direct_registry.json",
        f"{V3}/four_phrasings_registry.json",
        f"{V3}/stochastic_rollout_registry.json",
        f"{V3}/confound_fixture_calibration_registry.json",
        f"{V3}/phase_a_cells.jsonl",
        f"{V3}/phase_a_cells_manifest.json",
        f"{V3}/failure_taxonomy.json",
        f"{V3}/analysis_plan.json",
    ], "protocol references the complete immutable v3 artifact set", checks)
    common = protocol["common_execution_contract"]
    require(common["instruction_controller"] == "static_episode_prompt" and common["oracle_actions"] == 0 and not common["subtask_coach"] and not common["progress_conditioned_instruction"], "common contract fixes static prompts and prohibits oracle/coach/progress control", checks)
    require(common["required_raw_outputs"] == ["viewport_video", "executed_action_trace", "raw_result_jsonl"], "common contract requires video, actions, and raw JSONL", checks)

    require(droid["schema_version"] == "vla-wam-shared-v3-droid-direct-registry-v1", "DROID registry schema is frozen", checks)
    require(droid["priority"] == DROID_MODELS, "DROID priority order is exact", checks)
    target = droid["target"]
    require(target["exact_matched_pair_count_per_checkpoint"] == 30 and target["seed_range"] == [8300, 8329] and target["directions_per_pair"] == ["left", "right"], "DROID target is thirty exact LEFT/RIGHT pairs on seeds 8300-8329", checks)
    prompts = droid["direct_prompts"]
    require(prompts == {
        "left": "Put the Rubik's cube to the left of the bowl.",
        "right": "Put the Rubik's cube to the right of the bowl.",
        "predicate": "official release-inside-the-45-degree-cone requested relation termination",
    }, "DROID static direct prompts and predicate are unchanged", checks)
    require(droid["required_evidence"][:3] == ["viewport_video", "executed_action_trace", "raw_result_jsonl"], "DROID registry requires video/actions/JSONL", checks)
    pi0 = droid["checkpoint_rules"]["pi0_fast_droid_vla"]
    require(pi0["preserved_historical_seeds"] == exact_range(8300, 8309), "pi0-FAST preserves historical seeds 8300-8309", checks)
    require(pi0["blocked_new_seeds"] == exact_range(8310, 8329), "pi0-FAST blocks seeds 8310-8329", checks)
    require("9e46d3aea26417bfb564227734b95d010aa827e5" in pi0["blocker"] and "11142d4319e44401e0464866bb5fedf7ec8a8927" in pi0["blocker"] and "V2-A008" in pi0["blocker"], "pi0-FAST blocker pins missing commits and failed V2-A008 sensitivity", checks)
    other = droid["checkpoint_rules"]["other_checkpoints"]
    require(other["models"] == DROID_MODELS[1:], "other DROID checkpoint set is exact", checks)
    require(other["preserved_candidate_seeds"] == [8300, 8301, 8302] and other["new_addition_seeds"] == exact_range(8303, 8329), "other DROID additions use 8303-8329 with conditional preserved seeds", checks)
    require("exactly" in droid["runtime_reuse_rule"] and "never rerun" in droid["runtime_reuse_rule"], "DROID runtime reuse is identity-pinned and non-rerunnable", checks)

    require(robotwin["schema_version"] == "vla-wam-shared-v3-robotwin-direct-registry-v1", "RoboTwin registry schema is frozen", checks)
    require(robotwin["models"] == ROBOTWIN_MODELS, "RoboTwin core model set is exact", checks)
    require(robotwin["scene_pairs"] == [3, 4, 5, 6, 7, 8, 9], "RoboTwin core scenes are pairs03-09", checks)
    sampling = robotwin["sampling"]
    require(sampling["replicates"] == list(range(10)) and "8400 + pair_number" in sampling["seed_rule"] and "+ 100*r" in sampling["seed_rule"], "RoboTwin replicate seeds use the required r=0 and r=1..9 formula", checks)
    require("preserved" in sampling["r0_policy"] and "never rerun" in sampling["r0_policy"], "RoboTwin r=0 evidence is preserved and non-rerunnable", checks)
    accounting = robotwin["episode_accounting"]
    require(accounting == {"existing_r0_episodes": 42, "new_episodes_per_model": 126, "new_episodes_total": 378, "formula": "3 models * 7 pairs * 9 new replicates * 2 directions"}, "RoboTwin accounting fixes 378 new episodes", checks)
    require("same anchor reset" in robotwin["pairing"] and "Only requested direction" in robotwin["pairing"], "RoboTwin pairing keeps both directions in identical anchors", checks)
    require(robotwin["required_evidence"][:3] == ["simulator_viewport_video", "executed_action_trace", "raw_result_jsonl"], "RoboTwin registry requires video/actions/JSONL", checks)

    require(taxonomy["schema_version"] == "vla-wam-shared-v3-failure-taxonomy-v1", "taxonomy schema is frozen", checks)
    require(taxonomy["required_legacy_field"] == "frozen_failure_stage", "taxonomy preserves the frozen failure stage", checks)
    require(taxonomy["primary_precedence"] == ["correct", "pick_failed", "wrong_side", "release_failed", "transport_failed"], "taxonomy precedence is exact", checks)
    classes = taxonomy["classes"]
    require("frozen requested relation-and-detached-release success" in classes["correct"], "correct class defers to frozen success", checks)
    require("picked object" in classes["wrong_side"] and "three consecutive" in classes["wrong_side"] and "opposite" in classes["wrong_side"], "wrong_side requires pickup and sustained opposite region", checks)
    require("picked object" in classes["release_failed"] and "three consecutive" in classes["release_failed"] and "final_detached_release is false" in classes["release_failed"] and "not an inference" in classes["release_failed"], "release_failed requires pickup, sustained requested region, and independent failed release", checks)
    require("+3 cm" in classes["pick_failed"] and "three consecutive" in classes["pick_failed"], "pick_failed requires the frozen pickup threshold", checks)
    require(taxonomy["state_and_scoring_contract"]["sustained_samples"] == 3 and taxonomy["state_and_scoring_contract"]["pickup_threshold_m"] == 0.03, "taxonomy freezes three-sample and +3cm thresholds", checks)
    require(set(taxonomy["continuous_fields"]) == {"signed_final_lateral_offset_m", "final_requested_signed_margin_m", "final_opposite_signed_margin_m", "maximum_requested_signed_margin_m", "maximum_pickup_height_m", "first_verified_pickup_step", "first_cone_or_native_region_entry_step", "entry_kind", "first_requested_region_step", "final_detached_release", "executed_action_count", "episode_length_steps", "first_contact_status", "first_contact_step", "first_contact_unavailable_reason", "object_path_length_m", "wall_time_s"}, "taxonomy continuous fields are complete", checks)
    fields = taxonomy["continuous_fields"]
    require("sustained, transient, or none" in fields["entry_kind"] and "three consecutive" in fields["entry_kind"], "taxonomy records sustained/transient/none entry kind", checks)
    require("delta_y" in fields["signed_final_lateral_offset_m"] and "positive robot LEFT" in fields["signed_final_lateral_offset_m"] and "not the negated reader-display coordinate" in fields["signed_final_lateral_offset_m"], "taxonomy freezes the raw v2 robot-frame lateral offset", checks)
    require("+delta_y for LEFT" in fields["final_requested_signed_margin_m"] and "-delta_y for RIGHT" in fields["final_requested_signed_margin_m"], "taxonomy derives requested margin from raw lateral offset", checks)
    contact = taxonomy["contact_conditional_rules"]
    require(fields["first_contact_status"] == "one of observed, not_observed, or instrumentation_unavailable" and contact["allowed_statuses"] == ["observed", "not_observed", "instrumentation_unavailable"], "taxonomy freezes observed/not-observed/instrumentation-unavailable contact status", checks)
    require(contact["first_contact_step"] == {"observed": "required integer", "not_observed": "required null", "instrumentation_unavailable": "required null"}, "contact step is integer only for observed contact", checks)
    require(contact["first_contact_unavailable_reason"] == {"observed": "must be absent or null", "not_observed": "must be absent or null", "instrumentation_unavailable": "required nonempty string"}, "contact unavailable reason is required only for instrumentation failure", checks)
    require(contact["all_false_retained_contact_stream"] == "A retained all-false contact stream is not_observed, not instrumentation_unavailable.", "retained all-false contact streams are not mislabeled unavailable", checks)
    scorer_rules = taxonomy["scorer_consistency_rules"]
    require(scorer_rules["release_failed"] == "requires verified pickup, sustained final requested region, and final_detached_release=false" and scorer_rules["requested_success_false_with_sustained_requested_and_detached_true"] == "technical_invalid_scorer_inconsistency; preserve the record and invalidate it from behavioral classification until the frozen scorer/input provenance is repaired" and scorer_rules["prohibited_inference"] == "Never infer final_detached_release from requested_success, gripper command, or action trace.", "taxonomy treats detached release as independent and scorer inconsistency as technical invalid", checks)
    require("outside behavioral denominators" in taxonomy["infrastructure_separation"]["rule"], "taxonomy separates infrastructure attempts from behavior", checks)

    require(amendment["schema_version"] == "vla-wam-shared-v3-post-result-power-failure-ablation-amendment-v1", "amendment schema is frozen", checks)
    require(amendment["status"] == "frozen_after_all_v2_and_v2_a015_results_and_before_any_v3_model_request_or_behavioral_inference", "amendment discloses timing before v3 inference", checks)
    phases = amendment["phases"]
    require(set(phases) == {"A_direct_replication", "B_confound_ablation", "C_four_phrasings", "D_16_rollout_stochastic_block"}, "all four v3 phases are registered", checks)
    confound = phases["B_confound_ablation"]
    require(confound["status"] == "separately_gated_not_released_by_phase_a" and confound["calibration_registry"] == f"{V3}/confound_fixture_calibration_registry.json" and "exactly one named factor" in confound["one_factor_rule"] and "randomized" in confound["randomization"], "confound ablations are one-factor, randomized, and separately gated", checks)
    require(calibration_registry["schema_version"] == "vla-wam-shared-v3-confound-fixture-calibration-registry-v1" and calibration_registry["status"] == "model_blind_fixture_calibration_required_before_any_confound_level_is_released", "confound levels remain unreleased pending model-blind calibration", checks)
    require(calibration_registry["unreleased_fields"] == ["numeric_fixture_coordinates", "numeric_requested_region_margin_m", "numeric_opposite_region_margin_m", "confound_factor_levels", "control_intervention_allocation"] and "new hash-pinned amendment" in calibration_registry["freeze_requirement"], "confound calibration freezes no invented numeric level", checks)
    wording = phases["C_four_phrasings"]
    require(wording["status"] == "separately_gated_not_released_by_phase_a" and wording["registry"] == f"{V3}/four_phrasings_registry.json" and "V2-A008 remains failed" in wording["pi0_fast_boundary"], "four phrasing block is independently randomized/gated and cannot bypass V2-A008", checks)
    require(wording_registry["schema_version"] == "vla-wam-shared-v3-four-phrasings-registry-v1" and wording_registry["eligible_model_ids"] == WORDING_MODELS, "four phrasing registry fixes the first three non-pi0 checkpoints", checks)
    shared = wording_registry["shared_matched_pairs"]
    require(shared["seed_range"] == [8500, 8519] and shared["environment_seed_equals_sampling_seed"] is True and shared["pair_count_per_checkpoint"] == 20 and shared["directions_per_pair"] == ["left", "right"], "four phrasing registry fixes twenty shared matched seeds", checks)
    require(wording_registry["prompt_forms"] == EXACT_V2_WORDINGS, "four phrasing registry preserves exact v2 prompt bytes", checks)
    wording_accounting = wording_registry["episode_accounting"]
    require(wording_accounting == {"checkpoints": 3, "matched_pairs_per_checkpoint": 20, "prompt_forms": 4, "directions_per_pair": 2, "episodes_per_checkpoint": 160, "frozen_total_episodes": 480, "formula": "3 checkpoints * 20 pairs * 4 prompt forms * 2 directions"}, "four phrasing accounting fixes 480 episodes", checks)
    optional_pi0 = wording_registry["optional_model_after_independent_release"]
    optional_pi0_conditions = " ".join(optional_pi0["conditions"])
    require(optional_pi0["model_id"] == "pi0_fast_droid_vla" and "exact historical OpenPI/RoboLab revisions" in optional_pi0_conditions and "sensitivity release" in optional_pi0_conditions, "pi0-FAST wording participation requires exact recovery and new sensitivity release", checks)
    block = phases["D_16_rollout_stochastic_block"]
    require(block["status"] == "separately_gated_not_released_by_phase_a" and block["registry"] == f"{V3}/stochastic_rollout_registry.json", "Phase D is separately gated through its registry", checks)
    require(stochastic_registry["schema_version"] == "vla-wam-shared-v3-stochastic-rollout-registry-v1" and stochastic_registry["shared_sampling_seed_indices"] == list(range(16)), "Phase D freezes sixteen shared sampling-seed indices", checks)
    require("Every released Phase-A registered" in stochastic_registry["scope"] and "exactly one rollout" in stochastic_registry["rollout_contract"], "Phase D applies sixteen rollouts to every eligible released Phase-A condition", checks)
    require("deterministic runtime" in stochastic_registry["eligibility_gate"] and "no fake repeated rollouts" in stochastic_registry["eligibility_gate"], "Phase D makes deterministic or ineffective-seed runtimes ineligible", checks)
    require("not sixteen independent scenes" in stochastic_registry["analysis_unit"] and "not independent scenes" in block["analysis_boundary"], "Phase D rejects pseudoreplication", checks)
    require(amendment["shared_required_evidence"][:3] == ["viewport_video", "executed_action_trace", "raw_result_jsonl"], "amendment requires raw video/actions/JSONL", checks)
    require("never pooled" in amendment["arena_boundary"], "amendment maintains arena separation", checks)

    require(phase_a_manifest["schema_version"] == "vla-wam-shared-v3-phase-a-cells-manifest-v1" and phase_a_manifest["study_id"] == "vla_wam_language_steerability_v3", "Phase-A queue manifest schema and study identity are frozen", checks)
    require(phase_a_manifest["queue_file"] == f"{V3}/phase_a_cells.jsonl" and phase_a_manifest["queue_sha256"] == sha256(paths["phase_a_queue"]), "Phase-A queue manifest hash matches the generated JSONL", checks)
    expected_row_counts = {
        "by_arena": {"droid_robolab": 360, "robotwin": 420},
        "by_model": {
            "cosmos3_edge_policy_droid": 60, "cosmos3_nano_policy_droid": 60,
            "dreamzero_droid_action_cfg": 60, "efficient_wam_rt_robotwin": 140,
            "fastwam_robotwin": 140, "groot_n17_droid_vla": 60,
            "lingbot_va_robotwin": 140, "pi05_current_stack_droid": 60,
            "pi0_fast_droid_vla": 60,
        },
        "by_status": {"authorized_new": 648, "blocked_pi0": 40, "preserved_candidate": 50, "preserved_r0": 42},
        "total": 780,
    }
    require(phase_a_manifest["row_counts"] == expected_row_counts, "Phase-A manifest fixes 780 total rows, 360/420 arena rows, and 648/50/42/40 statuses", checks)
    require(len(phase_a_rows) == 780 and all(row.get("schema_version") == "vla-wam-shared-v3-phase-a-cells-v1" for row in phase_a_rows), "Phase-A JSONL has exactly 780 schema-valid rows", checks)
    queue_arena_counts = Counter(row.get("arena") for row in phase_a_rows)
    queue_status_counts = Counter(row.get("status") for row in phase_a_rows)
    require(dict(queue_arena_counts) == {"droid_robolab": 360, "robotwin": 420} and dict(queue_status_counts) == {"authorized_new": 648, "blocked_pi0": 40, "preserved_candidate": 50, "preserved_r0": 42}, "Phase-A JSONL row counts agree with the manifest", checks)

    require(analysis["schema_version"] == "vla-wam-shared-v3-analysis-plan-v1", "analysis schema is frozen", checks)
    require("Never combine" in analysis["primary_estimands"]["no_pooling"], "analysis forbids pooled arena inference", checks)
    hierarchical = analysis["hierarchical_16_rollout_analysis"]
    require("every eligible released Phase-A" in hierarchical["unit"] and "Deterministic" in hierarchical["eligibility"] and "independent scenes" in hierarchical["prohibition"] and "Do not" in hierarchical["prohibition"], "analysis nests Phase D rollouts within every eligible condition", checks)
    require("outside behavioral denominators" in analysis["infrastructure_and_latency"], "analysis separates infrastructure invalidity", checks)

    doc = paths["document"].read_text()
    for phrase in ("30 exact matched", "378 new episodes", "480", "16 shared", "never pooled"):
        require(phrase in doc, f"human-readable protocol states {phrase!r}", checks)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        checks = validate(args.root.resolve())
    except ValidationError as exc:
        print(f"V3 protocol validation failed: {exc}")
        return 1
    if not args.quiet:
        print(f"V3 protocol validation passed: {len(checks)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
