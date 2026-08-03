#!/usr/bin/env python3
"""Fail-closed validation for the frozen VLA/WAM steerability v2 protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_MODEL_IDS = {
    "pi05_droid_vla",
    "pi0_fast_droid_vla",
    "groot_n17_droid_vla",
    "cosmos3_edge_droid_wam",
    "lingbot_vla_4b_robotwin",
    "efficient_wam_rt_robotwin",
    "fastwam_robotwin",
    "lingbot_va_robotwin",
}
EXPECTED_EXPANSION_IDS = {
    "pi0_fast_droid_vla",
    "groot_n17_droid_vla",
    "lingbot_vla_4b_robotwin",
    "efficient_wam_rt_robotwin",
    "fastwam_robotwin",
    "lingbot_va_robotwin",
}
EXPECTED_PROMPT_IDS = [
    "direct_command",
    "short_command",
    "goal_as_outcome",
    "desired_plus_negated_opposite",
]
EXPECTED_LEGACY_WORDINGS = {
    "canonical",
    "short_paraphrase",
    "declarative_goal",
    "contrastive_goal",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise RuntimeError(message)
    checks.append(message)


def validate_prompt(prompt: dict[str, Any], checks: list[str]) -> None:
    prompt_id = prompt["id"]
    for direction, opposite in (("left", "right"), ("right", "left")):
        text = prompt[direction].lower()
        require(
            "{movable}" in text or "{movable_short}" in text,
            f"{prompt_id}/{direction} identifies or deliberately shortens the movable object",
            checks,
        )
        require(
            "{reference}" in text,
            f"{prompt_id}/{direction} includes the reference object",
            checks,
        )
        require(
            direction in text,
            f"{prompt_id}/{direction} includes the desired relation",
            checks,
        )
        if prompt_id == "desired_plus_negated_opposite":
            require(
                opposite in text and "not" in text,
                f"{prompt_id}/{direction} includes an explicitly negated opposite",
                checks,
            )
        else:
            require(
                opposite not in text,
                f"{prompt_id}/{direction} does not leak the opposite direction",
                checks,
            )


def validate_v1_disclosure(workspace: Path, checks: list[str]) -> dict[str, Any]:
    episodes_path = workspace / "artifacts/vla_wam_shared_v1/final_evidence/episodes.csv"
    with episodes_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 160, "v1 disclosure population contains exactly 160 episodes", checks)
    require(
        {row["model_id"] for row in rows}
        == {"pi05_droid_vla", "cosmos3_edge_droid_wam"},
        "v1 disclosure population contains the two registered reference models",
        checks,
    )
    require(
        {row["wording"] for row in rows} == EXPECTED_LEGACY_WORDINGS,
        "v1 disclosure population contains all four legacy prompt forms",
        checks,
    )
    direct_seeds = {
        int(row["episode_seed"]) for row in rows if row["wording"] == "canonical"
    }
    contrastive_seeds = {
        int(row["episode_seed"])
        for row in rows
        if row["wording"] == "contrastive_goal"
    }
    require(
        direct_seeds.isdisjoint(contrastive_seeds),
        "v1 direct and contrastive seed sets are disjoint and cannot be called exact-seed pairs",
        checks,
    )
    return {
        "path": str(episodes_path.relative_to(workspace)),
        "sha256": sha256(episodes_path),
        "episode_count": len(rows),
        "direct_seeds": sorted(direct_seeds),
        "contrastive_seeds": sorted(contrastive_seeds),
        "exact_seed_direct_contrastive_pair_count": len(direct_seeds & contrastive_seeds),
    }


def validate(workspace: Path) -> dict[str, Any]:
    protocol_path = workspace / "artifacts/vla_wam_shared_v2/protocol.json"
    media_path = workspace / "artifacts/vla_wam_shared_v2/media_selection_plan.json"
    execution_path = workspace / "artifacts/vla_wam_shared_v2/pilot/execution_configs.json"
    technical_path = workspace / "artifacts/vla_wam_shared_v2/pilot/technical_events.json"
    efficient_result_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/results/efficient_wam_rt_direct_gate.json"
    )
    fastwam_result_path = (
        workspace / "artifacts/vla_wam_shared_v2/pilot/results/fastwam_direct_gate.json"
    )
    protocol = load_json(protocol_path)
    media = load_json(media_path)
    execution = load_json(execution_path)
    technical = load_json(technical_path)
    efficient_result = load_json(efficient_result_path)
    fastwam_result = load_json(fastwam_result_path)
    checks: list[str] = []

    require(
        protocol["status"] == "frozen_before_any_standardized_v2_expansion_inference",
        "protocol is marked frozen before standardized v2 expansion inference",
        checks,
    )
    require(
        media["status"] == protocol["status"],
        "media plan and protocol have the same freeze status",
        checks,
    )
    require(
        media["frozen_at_utc"] == protocol["frozen_at_utc"],
        "media plan and protocol have the same freeze timestamp",
        checks,
    )

    models = protocol["models"]
    model_ids = {model["id"] for model in models}
    expansion_ids = {
        model["id"] for model in models if model["standardized_v2_expansion_required"]
    }
    require(len(models) == 8, "protocol registers exactly eight core models", checks)
    require(model_ids == EXPECTED_MODEL_IDS, "registered model identities match the freeze", checks)
    require(
        expansion_ids == EXPECTED_EXPANSION_IDS,
        "exactly the six frozen expansion models require standardized v2 pilots",
        checks,
    )
    require(
        {model["class"] for model in models} == {"VLA", "WAM"},
        "both VLA and WAM model classes are represented",
        checks,
    )
    for model in models:
        require(
            bool(model["world_model_interface"]),
            f"{model['id']} declares its future interface explicitly",
            checks,
        )

    arenas = {arena["id"]: arena for arena in protocol["design"]["arenas"]}
    require(
        set(arenas) == {"droid_robolab", "robotwin_place_a2b"},
        "protocol contains exactly the two frozen arenas",
        checks,
    )
    require(
        protocol["design"]["oracle_episode_count"] == 0
        and protocol["design"]["dynamic_prompt_episode_count"] == 0,
        "protocol contains zero oracle and zero dynamic-prompt episodes",
        checks,
    )
    require(
        "Never pool DROID and RoboTwin" in protocol["design"]["cross_arena_rule"],
        "cross-arena raw-success pooling is explicitly forbidden",
        checks,
    )

    droid_seeds = arenas["droid_robolab"]["episode_seeds"]["new_v2_paired"]
    require(
        droid_seeds == list(range(8300, 8310)),
        "DROID v2 uses the frozen ten-seed exact-pairing block 8300-8309",
        checks,
    )
    require(
        droid_seeds[: arenas["droid_robolab"]["pilot_seed_count"]]
        == [8300, 8301, 8302],
        "DROID pilot uses paired seeds 8300-8302",
        checks,
    )
    robotwin = arenas["robotwin_place_a2b"]
    paired_scenes = robotwin["paired_scenes"]
    require(len(paired_scenes) == 3, "RoboTwin pilot freezes exactly three paired scenes", checks)
    require(
        [scene["environment_seed"] for scene in paired_scenes] == [4300000, 4300001, 4300002],
        "RoboTwin paired scenes use environment seeds 4300000-4300002",
        checks,
    )
    require(
        [scene["sampling_seed"] for scene in paired_scenes] == [8400, 8401, 8402],
        "RoboTwin paired scenes use sampling seeds 8400-8402",
        checks,
    )
    require(
        [scene["anchor_task"] for scene in paired_scenes]
        == ["place_a2b_left", "place_a2b_right", "place_a2b_left"],
        "RoboTwin anchor-task assignment is frozen and direction-independent",
        checks,
    )
    require(
        "Never compare" in robotwin["native_task_confound_block"],
        "RoboTwin native-task scene confound is explicitly blocked",
        checks,
    )
    require(
        "first entry" in robotwin["object_naming_rule"],
        "RoboTwin object naming source is shared across adapters",
        checks,
    )

    prompt_ids = [prompt["id"] for prompt in protocol["prompt_families"]]
    require(prompt_ids == EXPECTED_PROMPT_IDS, "four prompt forms and their order are frozen", checks)
    require(
        {prompt["legacy_v1_id"] for prompt in protocol["prompt_families"]}
        == EXPECTED_LEGACY_WORDINGS,
        "every v2 prompt form maps to one disclosed v1 form",
        checks,
    )
    for prompt in protocol["prompt_families"]:
        validate_prompt(prompt, checks)

    require(len(protocol["hypotheses"]) == 4, "four physical hypotheses are frozen", checks)
    require(
        {hypothesis["id"] for hypothesis in protocol["hypotheses"]}
        == {
            "H1_mirrored_language_redirects_endpoint",
            "H2_wording_robustness",
            "H3_directional_symmetry",
            "H4_imagination_execution_agreement",
        },
        "hypothesis identities match the reader-facing protocol",
        checks,
    )
    amendments = protocol["pre_inference_amendments"]
    require(
        {amendment["id"] for amendment in amendments}
        == {
            "V2-A001_robotwin_anchor_scene_pairing",
            "V2-A002_shared_robotwin_object_naming",
        },
        "both pre-inference RoboTwin confound corrections are disclosed",
        checks,
    )
    require(
        all(amendment["inference_completed_before_amendment"] == 0 for amendment in amendments),
        "no standardized v2 inference preceded either protocol amendment",
        checks,
    )
    post_amendments = protocol["post_inference_amendments"]
    require(
        [amendment["id"] for amendment in post_amendments]
        == ["V2-A003_robotwin_execution_config_registry"],
        "the post-pilot execution-config amendment is disclosed separately",
        checks,
    )
    require(
        post_amendments[0]["inference_completed_before_amendment"] == 6,
        "the execution-config amendment discloses six preceding Efficient-WAM episodes",
        checks,
    )
    require(
        post_amendments[0]["artifact"]
        == "artifacts/vla_wam_shared_v2/pilot/execution_configs.json",
        "the post-pilot amendment points to its machine-readable registry",
        checks,
    )

    execution_models = {model["model_id"]: model for model in execution["models"]}
    require(
        set(execution_models)
        == {
            "efficient_wam_rt_robotwin",
            "fastwam_robotwin",
            "lingbot_va_robotwin",
        },
        "execution registry covers all three runnable RoboTwin WAMs",
        checks,
    )
    require(
        execution_models["efficient_wam_rt_robotwin"]["new_v2_episodes_completed_before_record"]
        == 6,
        "Efficient-WAM execution settings are labeled retrospective after six episodes",
        checks,
    )
    require(
        all(
            execution_models[model_id]["status"]
            == "frozen_before_any_standardized_v2_inference"
            and execution_models[model_id]["new_v2_episodes_completed_before_record"] == 0
            for model_id in ("fastwam_robotwin", "lingbot_va_robotwin")
        ),
        "FastWAM and LingBot-VA execution settings are prospectively frozen",
        checks,
    )
    require(
        execution["shared"]["oracle_actions"] == 0
        and execution["shared"]["dynamic_prompts"] == 0,
        "RoboTwin execution registry preserves the no-oracle design",
        checks,
    )

    summary = efficient_result["summary"]
    require(
        summary["episode_count"] == 6 and summary["pair_count"] == 3,
        "compiled Efficient-WAM pilot contains six episodes in three exact pairs",
        checks,
    )
    require(
        summary["by_direction"]["left"]["successes"] == 2
        and summary["by_direction"]["right"]["successes"] == 0,
        "compiled Efficient-WAM pilot preserves the observed 2/3 LEFT and 0/3 RIGHT result",
        checks,
    )
    require(
        summary["pilot_gate_decision"] == "expand_direct_directional_bias_only",
        "Efficient-WAM pilot follows the frozen one-direction-only expansion gate",
        checks,
    )
    fastwam_summary = fastwam_result["summary"]
    require(
        fastwam_summary["episode_count"] == 6 and fastwam_summary["pair_count"] == 3,
        "compiled FastWAM pilot contains six episodes in three exact pairs",
        checks,
    )
    require(
        fastwam_summary["by_direction"]["left"]["successes"] == 1
        and fastwam_summary["by_direction"]["right"]["successes"] == 0,
        "compiled FastWAM pilot preserves the observed 1/3 LEFT and 0/3 RIGHT result",
        checks,
    )
    require(
        fastwam_summary["pilot_gate_decision"] == "expand_direct_directional_bias_only",
        "FastWAM pilot follows the frozen one-direction-only expansion gate",
        checks,
    )
    require(
        all(
            episode["imagined_future_video"] is None
            for episode in fastwam_result["episodes"]
        ),
        "FastWAM action-only interface records imagined-video evidence as not applicable",
        checks,
    )
    require(
        len(technical["events"]) == 4
        and technical["events"][-1]["classification"] == "environment_repair",
        "pre-episode failures and the renderer repair remain in a separate technical ledger",
        checks,
    )

    pilot = protocol["pilot"]
    calculated_pilot = (
        len(expansion_ids)
        * len(protocol["prompt_families"])
        * 2
        * pilot["seeds_per_cell"]
    )
    require(calculated_pilot == 144, "pilot arithmetic evaluates to 144 episodes", checks)
    require(
        pilot["expected_episode_count"] == calculated_pilot,
        "registered pilot episode count equals the calculated grid",
        checks,
    )
    require(pilot["record_every_episode"], "every pilot episode must be recorded", checks)
    require(pilot["retain_every_valid_failure"], "every valid pilot failure must be retained", checks)

    selection_roles = {
        role["id"] for role in media["prospective_selection_roles_per_model"]
    }
    require(
        selection_roles
        == {
            "first_success_left",
            "first_success_right",
            "first_post_pick_placement_failure",
            "first_direct_to_contrastive_reversal",
        },
        "media plan freezes success, failure, and same-seed reversal roles",
        checks,
    )
    require(
        "no-qualifying-example" in media["missing_category_policy"],
        "missing media categories must be visible rather than hand substituted",
        checks,
    )

    v1 = validate_v1_disclosure(workspace, checks)
    return {
        "status": "valid",
        "protocol_path": str(protocol_path.relative_to(workspace)),
        "protocol_sha256": sha256(protocol_path),
        "media_plan_path": str(media_path.relative_to(workspace)),
        "media_plan_sha256": sha256(media_path),
        "execution_config_path": str(execution_path.relative_to(workspace)),
        "execution_config_sha256": sha256(execution_path),
        "efficient_result_path": str(efficient_result_path.relative_to(workspace)),
        "efficient_result_sha256": sha256(efficient_result_path),
        "fastwam_result_path": str(fastwam_result_path.relative_to(workspace)),
        "fastwam_result_sha256": sha256(fastwam_result_path),
        "check_count": len(checks),
        "checks": checks,
        "registered_model_count": len(models),
        "expansion_model_count": len(expansion_ids),
        "calculated_pilot_episode_count": calculated_pilot,
        "v1_disclosure": v1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    report = validate(workspace)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.write_report:
        output = args.write_report
        if not output.is_absolute():
            output = workspace / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
