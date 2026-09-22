#!/usr/bin/env python3
"""Build planning artifacts only. No model, network, cluster, or simulator calls."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUDY = "SGW-01"
PIN = "ce561e66f82e95055e39d3d7711691982f6b2086"


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def strings(family, sign):
    cube = "the Rubik's cube"
    if family == "LAT":
        relation, opposite = ("left", "right") if sign == 1 else ("right", "left")
        return {
            "D": f"Put {cube} to the {relation} of the bowl.",
            "C": f"Place {cube} so that {cube} is to the {relation} of the bowl.",
            "I": f"Place {cube} so that the bowl is to the {opposite} of {cube}.",
        }
    if family == "HEIGHT":
        relation, opposite = ("higher", "lower") if sign == 1 else ("lower", "higher")
        return {
            "D": f"Put {cube} {relation} than the bowl.",
            "C": f"Place {cube} so that {cube} is {relation} than the bowl.",
            "I": f"Place {cube} so that the bowl is {opposite} than {cube}.",
        }
    relation, preposition, opposite, inverse_preposition = (
        ("closer", "to", "farther", "from") if sign == 1
        else ("farther", "from", "closer", "to")
    )
    return {
        "D": f"Put {cube} {relation} {preposition} the bowl than {preposition} the plate.",
        "C": f"Place {cube} so that {cube} is {relation} {preposition} the bowl than {preposition} the plate.",
        "I": f"Place {cube} so that the plate is {opposite} {inverse_preposition} {cube} than the bowl is.",
    }


def build():
    families = ["LAT", "HEIGHT", "DIST"]
    prompts = []
    lookup = {}
    for family in families:
        for sign in [1, -1]:
            for form, prompt in strings(family, sign).items():
                item = {"prompt_id": f"{family}-{form}-{'POS' if sign == 1 else 'NEG'}",
                        "family": family, "form": form, "physical_goal_sign": sign,
                        "text": prompt, "sha256": digest(prompt)}
                prompts.append(item)
                lookup[(family, form, sign)] = item
    write_json("prompts.json", {"study_id": STUDY, "version": "1.1", "prompts": prompts})
    cells = []
    for fi, family in enumerate(families):
        for si, (stage, layouts) in enumerate([("P", 1), ("D", 4), ("C", 24)]):
            for index in range(1, layouts + 1):
                layout_id = f"{family}-{stage}{index:02d}"
                cycle, rotation = divmod(index - 1, 6)
                conditions = [(form, sign) for form in ["D", "C", "I"] for sign in [1, -1]]
                conditions.sort(key=lambda c: digest(f"{STUDY}|{family}|{stage}|{cycle}|{c}"))
                order = conditions[rotation:] + conditions[:rotation]
                for model in ["N3", "D1"]:
                    for position, (form, sign) in enumerate(order, 1):
                        p = lookup[(family, form, sign)]
                        cells.append({
                            "study_id": STUDY, "cell_id": f"{layout_id}-{model}-{form}-{'POS' if sign == 1 else 'NEG'}",
                            "block_id": f"{layout_id}-{model}", "model": model, "family": family,
                            "stage": stage, "layout_id": layout_id, "within_block_order": position,
                            "form": form, "physical_goal_sign": sign,
                            "prompt_id": p["prompt_id"], "prompt": p["text"], "prompt_sha256": p["sha256"],
                            "environment_seed": 20260922 + fi * 10000 + si * 1000 + index,
                            "effective_policy_seed": 2026092200 + fi * 1000 + si * 100 + index if model == "N3" else 1140,
                            "action_cap": 450, "status": "PLANNED_NOT_RELEASED",
                            "fixture_sha256": "", "runtime_sha256": "", "time_map_sha256": "",
                        })
    with (ROOT / "planned_cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    source_base = "artifacts/vla_wam_shared_v3/phase_c/"
    sources = {
        "pi05_reference_inversion": source_base + "semantic_equivalence_v3c002r001/activation_v4/final_analysis_v3/results/results.json",
        "nano_wording": source_base + "four_phrasings_v3c001/results/cosmos3_nano_policy_droid/cosmos3_nano_policy_droid_phase_c_summary.json",
        "edge_wording": source_base + "four_phrasings_v3c001/results/cosmos3_edge_policy_droid/cosmos3_edge_policy_droid_phase_c_summary.json",
        "exact_historical_wording": source_base + "four_phrasings_v3c001/prospective_phase_c_v3c001_registration.json",
        "failed_grasp_fixture": "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/results/DECISION_MEMO.md",
    }
    protocol = {
        "study_id": STUDY, "version": "1.1", "date": "2026-09-22",
        "status": "SPECIFICATION_ONLY_NO_NEW_EXPERIMENTS_RUN",
        "source_evidence_commit": PIN,
        "authoring_base_commit": "e67e6c4f",
        "overleaf_project": "https://www.overleaf.com/project/6ab2bdb39b30df4bbc98a15e",
        "primary_question": "Do generated futures remain reliable under changes in spatial goals and equivalent descriptions?",
        "workshop_focus": {"primary_motion": 6, "secondary_motion": 4, "secondary_scope": "own prediction reliability, not arbitrary policy simulation", "pi05_historical_role": "background_only_not_matched_control", "optional_pi05_lat_baseline": {"queued": False, "episodes": 174, "causal_world_model_ablation": False}, "submission_deadline_as_checked": "2026-10-12", "page_limit_as_checked": 4, "checked_on": "2026-09-22", "source": "https://do-robots-need-world-models.github.io/"},
        "primary_comparison": "I minus C, averaged over both physical goals within a layout",
        "primary_execution_endpoint": "fixed_endpoint_success_risk_difference",
        "families": families, "forms": ["D", "C", "I"], "goal_signs": [1, -1],
        "layouts_per_family": {"P": 1, "D": 4, "C": 24},
        "episodes": {"P": 36, "D": 144, "C": 864, "total": 1044},
        "model_configs": {
            "N3": {"asset": "nvidia/Cosmos3-Nano-Policy-DROID", "revision": "6706d7680581c255ff61e0f3bb49d90eac55c79e", "guidance": 3, "denoising_steps": 4, "shift": 5, "history_length": 1, "conditioning_fps": 15, "resolution_setting": 480, "executed_action_horizon": 32},
            "D1": {"asset_resolution": "read exact official identifier from pinned checkpoint provenance", "revision": "96ad344138c66e82536422432ad742f015784942", "action_path": "official_conditional_no_custom_s2", "video_guidance": 5, "configured_steps": 16, "executed_action_horizon": 8, "effective_noise_seed": 1140},
        },
        "scoring": {"action_cap": 450, "goal_termination": False, "relation_margin_m": 0.03, "initial_neutral_tolerance_m": 0.005, "reference_motion_limit_m": 0.005, "reset_position_tolerance_m": 0.003, "reset_angle_tolerance_degrees": 2, "pickup_height_m": 0.03, "pickup_consecutive_steps": 3, "final_stability_seconds": 0.5, "linear_speed_limit_m_s": 0.02, "angular_speed_limit_rad_s": 0.2},
        "statistics": {"unit": "independent_layout", "bootstrap_draws": 20000, "bootstrap_seed": 20260922, "paired_sign_flip_draws": 100000, "paired_sign_flip_seed": 20260923, "multiplicity": "Holm over six primary model-family tests", "equivalence_success_margin": 0.10, "equivalence_relation_margin_m": 0.02, "equivalence_ci": 0.90},
        "prediction_annotation": {"selected_request_fractions": [0.25, 0.75], "raters": 2, "adjudication_distance_image_diagonals": 0.02, "moving_threshold_image_diagonals": 0.02, "max_confirmation_selected_requests": 1728, "max_initial_frame_judgments": 10368},
        "operations": {"max_concurrent_workers": 2, "max_total_allocated_gpus": 4, "max_attempts_per_cell": 3, "max_possible_behavioral_attempts": 3132, "heartbeat_seconds": 60, "termination_grace_seconds": 180, "resource_budget": None, "resource_budget_status": "resolve_existing_user_allocation_before_launch", "persistent_completion_required": True},
        "technical_qualification": {"base_nonbehavioral_requests": 12, "optional_patched_D1_reference_requests": 3, "max_fixture_candidates_per_family": 100, "scripted_checks_per_candidate": 6, "semantic_label_calibration_states_per_family": 50},
        "launch_prerequisites": ["implemented_and_tested_worker", "qualified_runtime_binding", "qualified_family_fixtures", "persistent_recording_and_recovery_receipts", "physical_time_alignment_receipts", "existing_user_resource_budget", "immutable_release_hashes"],
        "historical_sources": {k: {"path": v, "url": f"https://github.com/adeeb10abbas/steerable/blob/{PIN}/{v}"} for k,v in sources.items()},
        "optional_extensions_queued": [],
    }
    write_json("protocol.json", protocol)
    assert len(prompts) == 18 and len({p["text"] for p in prompts}) == 18
    assert len(cells) == 1044 and len({c["cell_id"] for c in cells}) == 1044
    assert Counter(c["stage"] for c in cells) == {"P": 36, "D": 144, "C": 864}
    blocks = defaultdict(list)
    positions = Counter()
    for c in cells:
        assert digest(c["prompt"]) == c["prompt_sha256"]
        assert 0 <= c["effective_policy_seed"] < 2**31
        blocks[c["block_id"]].append(c)
        if c["stage"] == "C":
            positions[(c["model"], c["family"], c["form"], c["physical_goal_sign"], c["within_block_order"])] += 1
    assert len(blocks) == 174 and all(len(b) == 6 for b in blocks.values())
    assert set(positions.values()) == {4}
    for b in blocks.values():
        assert len({c["environment_seed"] for c in b}) == 1
        assert len({c["effective_policy_seed"] for c in b}) == 1
        assert {c["within_block_order"] for c in b} == set(range(1, 7))
        assert {(c["form"],c["physical_goal_sign"]) for c in b} == {(f,q) for f in ["D","C","I"] for q in [1,-1]}
    receipt = {"status": "PLANNING_REGISTRY_VALIDATED_NOT_RUNTIME_QUALIFIED", "prompts": 18, "cells": 1044, "blocks": 174,
               "stage_counts": dict(Counter(c["stage"] for c in cells)), "confirmation_order_each_condition_each_position": 4,
               "files": {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ["prompts.json", "protocol.json", "planned_cells.csv"]},
               "unverified": protocol["launch_prerequisites"]}
    write_json("registry_validation.json", receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    build()
