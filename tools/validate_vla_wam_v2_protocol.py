#!/usr/bin/env python3
"""Fail-closed validation for the frozen VLA/WAM steerability v2 protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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


def validate_file_record(
    workspace: Path,
    record: dict[str, Any],
    label: str,
    checks: list[str],
) -> None:
    path = Path(record["path"])
    if not path.is_absolute():
        path = workspace / path
    elif not path.is_file():
        # Compiled evidence can retain the absolute path of the host where it
        # was produced.  For portable handoffs, resolve checked-in sources by
        # their repository-relative suffix without weakening the byte/hash
        # validation below.  External raw-output paths still fail closed.
        for repository_root in ("artifacts", "docs", "handoff", "experiments", "tools"):
            if repository_root not in path.parts:
                continue
            relative_path = Path(*path.parts[path.parts.index(repository_root) :])
            candidate = workspace / relative_path
            if candidate.is_file():
                path = candidate
                break
    require(path.is_file(), f"{label} exists", checks)
    require(path.stat().st_size == record["bytes"], f"{label} byte count matches", checks)
    require(sha256(path) == record["sha256"], f"{label} hash matches", checks)


def validate_efficient_pair03_handoff(
    workspace: Path,
    integration_path: Path,
    registry_path: Path,
    bundle_manifest_path: Path,
    checks: list[str],
) -> dict[str, Any]:
    """Validate the compact first prospective WAM pair and portable repo patches."""
    integration = load_json(integration_path)
    registry = load_json(registry_path)
    require(
        integration["schema_version"]
        == "vla-wam-shared-v2-efficient-wam-pair-integration-v1"
        and integration["status"] == "complete_valid_local_pair_do_not_rerun",
        "Efficient-WAM pair03 is frozen as valid completed evidence that must not be rerun",
        checks,
    )
    require(
        integration["model_id"] == "efficient_wam_rt_robotwin"
        and integration["model_repository_commit"]
        == "b0b6cfabcbd68d18888866e958c677ce640f0412"
        and integration["robotwin_repository_commit"]
        == "0bd8e76fde3afcffa4b30a3e3e8f92a206aa66cc",
        "Efficient-WAM pair03 identifies the exact model and simulator integration commits",
        checks,
    )
    pair = integration["pair"]
    registered = next(
        scene for scene in registry["scenes"]
        if scene["environment_seed"] == pair["environment_seed"]
    )
    require(
        pair["pair_id"] == "robotwin_pair_03"
        and pair["anchor_task"] == registered["anchor_task"] == "place_a2b_right"
        and pair["environment_seed"] == registered["environment_seed"] == 4300003
        and pair["sampling_seed"] == registered["sampling_seed"] == 8403
        and pair["requested_relations"] == ["left", "right"],
        "Efficient-WAM pair03 matches the frozen scene, seeds, and direct-command directions",
        checks,
    )
    require(
        integration["execution"]["instruction_controller"] == "static_episode_prompt"
        and integration["execution"]["oracle_actions"] == 0
        and integration["execution"]["subtask_coach"] is False
        and integration["execution"]["simulator_video_enabled"] is True,
        "Efficient-WAM pair03 used static prompts, no oracle or coach, and simulator video",
        checks,
    )
    cells = integration["cells"]
    require(
        len(cells) == 2
        and {cell["requested_relation"] for cell in cells} == {"left", "right"}
        and all(cell["requested_success"] is False for cell in cells)
        and all(cell["actions_executed"] == 400 for cell in cells),
        "Efficient-WAM pair03 preserves two valid 400-action behavioral failures",
        checks,
    )
    require(
        all(
            cell["files"]["action_trace"]["shape"] == [400, 14]
            and cell["files"]["simulator_video"]["frames"] == 400
            and cell["files"]["predicted_video"]["frames"] == 5
            and all(
                record.get("relative_path")
                and record.get("bytes", 0) > 0
                and len(record.get("sha256", "")) == 64
                for record in cell["files"].values()
            )
            for cell in cells
        ),
        "Efficient-WAM pair03 records action, simulator-video, and decoded-future metadata for both cells",
        checks,
    )
    require(
        set(integration["collection_files"])
        == {"manifest.json", "results.csv", "results.jsonl"}
        and all(
            record["bytes"] > 0 and len(record["sha256"]) == 64
            for record in integration["collection_files"].values()
        ),
        "Efficient-WAM pair03 hashes its collection manifest and tabular summaries",
        checks,
    )
    paired = integration["paired_metrics"]
    require(
        paired["first_ten_executed_action_rms"] > 0
        and paired["steps_used"] == 10
        and paired["right_minus_left_final_object_minus_target_x_m"] < 0
        and paired["endpoint_ordering"] == "anti_aligned",
        "Efficient-WAM pair03 distinguishes action sensitivity from anti-aligned physical steering",
        checks,
    )
    thermal = integration["thermal_evidence"]
    validate_file_record(workspace, thermal, "Efficient-WAM pair03 thermal log", checks)
    require(
        thermal["record_count"] == 273
        and thermal["temperature_sample_count"] == 270
        and thermal["maximum_temperature_c"] == 49
        and thermal["pause_count"] == thermal["emergency_count"] == 0
        and integration["execution"]["wall_latency_valid"] is True,
        "Efficient-WAM pair03 thermal evidence contains no latency-invalidating intervention",
        checks,
    )
    remaining = integration["remaining_scope"]
    require(
        remaining["efficient_wam_rt_robotwin"]["pairs"] == [4, 5, 6, 7, 8, 9]
        and remaining["efficient_wam_rt_robotwin"]["episode_count"] == 12
        and remaining["fastwam_robotwin"]["episode_count"] == 14
        and remaining["lingbot_va_robotwin"]["episode_count"] == 14
        and remaining["total_episode_count"] == 40,
        "Efficient-WAM pair03 handoff leaves exactly forty prospective WAM episodes",
        checks,
    )

    bundle_manifest = load_json(bundle_manifest_path)
    bundles = bundle_manifest["bundles"]
    require(
        bundle_manifest["schema_version"] == "vla-wam-repository-handoff-bundles-v1"
        and len(bundles) == 4,
        "cross-host handoff registers exactly four incremental repository bundles",
        checks,
    )
    for record in bundles:
        validate_file_record(workspace, record, f"handoff bundle {Path(record['path']).name}", checks)
        require(
            len(record["prerequisite_commit"]) == 40
            and len(record["target_commit"]) == 40
            and record["prerequisite_commit"] != record["target_commit"]
            and record["upstream"].startswith("https://github.com/"),
            f"handoff bundle {Path(record['path']).name} pins upstream, prerequisite, and target",
            checks,
        )
    require(
        {record["target_commit"] for record in bundles}
        == {
            "b0b6cfabcbd68d18888866e958c677ce640f0412",
            "068d3fd70c89df3726c09893f47b75a624b20c02",
            "d42efbc04e502057dab4b18bb14770cc48e85131",
            "0bd8e76fde3afcffa4b30a3e3e8f92a206aa66cc",
        },
        "handoff bundles contain the four exact integration commits",
        checks,
    )
    return {
        "path": str(integration_path.relative_to(workspace)),
        "sha256": sha256(integration_path),
        "valid_behavioral_episode_count": 2,
        "remaining_new_episode_count": 40,
        "bundle_manifest": str(bundle_manifest_path.relative_to(workspace)),
        "bundle_manifest_sha256": sha256(bundle_manifest_path),
    }


def validate_efficient_pairs04_09_slice(
    workspace: Path,
    manifest_path: Path,
    checks: list[str],
) -> dict[str, Any]:
    """Validate the compact PVC-resident Efficient-WAM prospective slice."""
    manifest = load_json(manifest_path)
    require(
        manifest["schema_version"]
        == "vla-wam-shared-v2-prospective-slice-evidence-manifest-v1"
        and manifest["model_id"] == "efficient_wam_rt_robotwin"
        and manifest["valid_episode_count"] == 12
        and manifest["requested_success_count"] == 5
        and manifest["invalid_attempt_count"] == 4
        and manifest["runtime_intervention_count"] == 0,
        "Efficient-WAM pairs04-09 manifest fixes the valid, invalid, and intervention counts",
        checks,
    )
    for label, record in manifest["files"].items():
        validate_file_record(workspace, record, f"Efficient-WAM pairs04-09 {label}", checks)

    slice_path = workspace / manifest["files"]["slice_json"]["path"]
    payload = load_json(slice_path)
    episodes = payload["episodes"]
    expected = {
        (seed, direction)
        for seed in range(4300004, 4300010)
        for direction in ("left", "right")
    }
    observed = {(int(row["environment_seed"]), row["requested_relation"]) for row in episodes}
    require(
        payload["schema_version"] == "vla-wam-shared-v2-robotwin-prospective-slice-v1"
        and payload["model_id"] == "efficient_wam_rt_robotwin"
        and payload["expected_environment_seeds"] == list(range(4300004, 4300010))
        and len(episodes) == 12
        and observed == expected,
        "Efficient-WAM prospective slice contains exactly pairs04-09 and both directions",
        checks,
    )
    summary = payload["summary"]
    require(
        summary["episode_count"] == 12
        and summary["pair_count"] == 6
        and summary["successes"] == 5
        and summary["by_direction"]["left"]["successes"] == 3
        and summary["by_direction"]["right"]["successes"] == 2
        and summary["aligned_endpoint_pairs"] == 6
        and summary["invalid_attempt_count"] == 4
        and summary["future_interface_counts"] == {"decoded_future_video": 12},
        "Efficient-WAM prospective slice reports 5/12 success and six aligned endpoint pairs",
        checks,
    )
    require(
        all(
            row["prompt_family"] == "direct_command"
            and row["action_trace"]["count"] == row["actions_executed"]
            and row["action_trace"]["shape"][0] == row["actions_executed"]
            and len(row["action_trace"]["sha256"]) == 64
            and row["operational_wall_latency_valid"] is True
            and row["runtime_intervention_ids"] == []
            for row in episodes
        ),
        "Efficient-WAM pairs04-09 retain static direct prompts, executed traces, and valid latency",
        checks,
    )
    pairs = summary["paired_endpoint_responses"]
    cells = {(int(row["environment_seed"]), row["requested_relation"]): row for row in episodes}
    require(
        len(pairs) == 6
        and all(
            pair["physical_initial_state_sha256"]
            == cells[(int(pair["environment_seed"]), "left")]["physical_initial_state_sha256"]
            == cells[(int(pair["environment_seed"]), "right")]["physical_initial_state_sha256"]
            and pair["first_ten_executed_action_rms"] > 0
            and pair["first_ten_executed_action_rms_steps_used"] == 10
            and pair["action_metric_unavailable_reason"] is None
            for pair in pairs
        ),
        "Efficient-WAM pairs04-09 match recorded initial state and differ in paired executed actions",
        checks,
    )
    invalid = load_json(
        workspace / manifest["files"]["invalid_attempt_ledger"]["path"]
    )["events"]
    runtime = load_json(
        workspace / manifest["files"]["runtime_intervention_ledger"]["path"]
    )["events"]
    require(
        len(invalid) == 4
        and all(
            event["classification"] == "partial"
            and event["behavioral_result_valid"] is False
            and event["wall_latency_valid"] is False
            for event in invalid
        )
        and runtime == [],
        "Efficient-WAM infrastructure-invalid attempts remain outside behavior and no thermal event is invented",
        checks,
    )
    return {
        "manifest": str(manifest_path.relative_to(workspace)),
        "manifest_sha256": sha256(manifest_path),
        "slice": str(slice_path.relative_to(workspace)),
        "slice_sha256": sha256(slice_path),
        "valid_episode_count": 12,
        "requested_success_count": 5,
    }


def validate_wam_pairs03_09_slice(
    workspace: Path,
    manifest_path: Path,
    registry_path: Path,
    fixture_path: Path,
    *,
    model_id: str,
    expected_successes: dict[str, int],
    expected_aligned_pairs: int,
    expected_invalid_attempts: int,
    expected_future_interface: str,
    checks: list[str],
) -> dict[str, Any]:
    """Validate a completed fourteen-cell prospective WAM confirmation slice."""
    manifest = load_json(manifest_path)
    expected_pair_ids = [f"robotwin_pair_{index:02d}" for index in range(3, 10)]
    expected_success_count = sum(expected_successes.values())
    require(
        manifest["schema_version"]
        == "vla-wam-shared-v2-prospective-slice-evidence-manifest-v1"
        and manifest["model_id"] == model_id
        and manifest["pair_ids"] == expected_pair_ids
        and manifest["valid_episode_count"] == 14
        and manifest["requested_success_count"] == expected_success_count
        and manifest["invalid_attempt_count"] == expected_invalid_attempts
        and manifest["runtime_intervention_count"] == 0,
        f"{model_id} pairs03-09 manifest fixes all valid, invalid, success, and intervention counts",
        checks,
    )
    require(
        {
            "slice_json",
            "slice_csv",
            "slice_markdown",
            "invalid_attempt_ledger",
            "runtime_intervention_ledger",
        }
        <= set(manifest["files"]),
        f"{model_id} manifest registers its slice and separate operational ledgers",
        checks,
    )
    for label, record in manifest["files"].items():
        validate_file_record(workspace, record, f"{model_id} pairs03-09 {label}", checks)

    slice_path = workspace / manifest["files"]["slice_json"]["path"]
    payload = load_json(slice_path)
    registry = load_json(registry_path)
    efficient_pair03 = load_json(
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/directional_confirmation/efficient_wam_rt_pair03_integration.json"
    )
    efficient_pairs04_09 = load_json(
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/directional_confirmation/efficient_wam_rt_pairs04_09_slice.json"
    )
    frozen_prompts = {
        (efficient_pair03["pair"]["pair_id"], row["requested_relation"]): row["prompt"]
        for row in efficient_pair03["cells"]
    }
    frozen_prompts.update(
        {
            (row["pair_id"], row["requested_relation"]): row["prompt"]
            for row in efficient_pairs04_09["episodes"]
        }
    )
    registry_scenes = {
        int(scene["environment_seed"]): scene
        for scene in registry["scenes"]
        if 4300003 <= int(scene["environment_seed"]) <= 4300009
    }
    episodes = payload["episodes"]
    expected_cells = {
        (f"robotwin_pair_{index:02d}", 4300000 + index, 8400 + index, direction)
        for index in range(3, 10)
        for direction in ("left", "right")
    }
    observed_cells = {
        (
            row["pair_id"],
            int(row["environment_seed"]),
            int(row["sampling_seed"]),
            row["requested_relation"],
        )
        for row in episodes
    }
    require(
        payload["schema_version"] == "vla-wam-shared-v2-robotwin-prospective-slice-v1"
        and payload["model_id"] == model_id
        and payload["expected_environment_seeds"] == list(range(4300003, 4300010))
        and len(episodes) == 14
        and observed_cells == expected_cells
        and all(
            row["pair_id"] == registry_scenes[int(row["environment_seed"])]["pair_id"]
            and row["sampling_seed"]
            == registry_scenes[int(row["environment_seed"])]["sampling_seed"]
            and row["task"]
            == registry_scenes[int(row["environment_seed"])]["anchor_task"]
            for row in episodes
        ),
        f"{model_id} slice contains exactly both directions for frozen pairs03-09",
        checks,
    )
    validate_file_record(
        workspace, payload["source_registry"], f"{model_id} frozen directional registry", checks
    )
    validate_file_record(
        workspace,
        payload["source_directional_fixtures"],
        f"{model_id} model-blind directional fixtures",
        checks,
    )
    require(
        Path(payload["source_registry"]["path"]).name == registry_path.name
        and payload["source_registry"]["sha256"] == sha256(registry_path)
        and Path(payload["source_directional_fixtures"]["path"]).name == fixture_path.name
        and payload["source_directional_fixtures"]["sha256"] == sha256(fixture_path),
        f"{model_id} slice hashes the frozen registry and fixture report",
        checks,
    )

    summary = payload["summary"]
    require(
        summary["episode_count"] == 14
        and summary["pair_count"] == 7
        and summary["successes"] == expected_success_count
        and summary["by_direction"]["left"]["episodes"] == 7
        and summary["by_direction"]["left"]["successes"]
        == expected_successes["left"]
        and summary["by_direction"]["right"]["episodes"] == 7
        and summary["by_direction"]["right"]["successes"]
        == expected_successes["right"]
        and summary["aligned_endpoint_pairs"] == expected_aligned_pairs
        and summary["invalid_attempt_count"] == expected_invalid_attempts
        and summary["future_interface_counts"] == {expected_future_interface: 14},
        f"{model_id} summary preserves the frozen directional success and endpoint numerators",
        checks,
    )
    require(
        all(
            summary["by_direction"][direction]["successes"]
            == sum(
                bool(row["requested_success"])
                for row in episodes
                if row["requested_relation"] == direction
            )
            for direction in ("left", "right")
        ),
        f"{model_id} directional success numerators match the episode rows",
        checks,
    )

    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in episodes:
        by_pair.setdefault(row["pair_id"], {})[row["requested_relation"]] = row
        direction = row["requested_relation"]
        opposite = "right" if direction == "left" else "left"
        prompt = row["prompt"].lower()
        trace = row["action_trace"]
        executed_array = trace["arrays"]["executed"]
        require(
            row["prompt_family"] == "direct_command"
            and row["prompt"] == frozen_prompts[(row["pair_id"], direction)]
            and direction in prompt
            and opposite not in prompt
            and row["runtime_intervention_ids"] == []
            and row["operational_wall_latency_valid"] is True
            and trace["executed_array"] == "executed"
            and trace["count"] == row["actions_executed"]
            and trace["shape"][0] == row["actions_executed"]
            and executed_array["count"] == row["actions_executed"]
            and executed_array["shape"] == trace["shape"]
            and trace["bytes"] > 0
            and len(trace["sha256"]) == 64,
            f"{model_id} {row['pair_id']}/{direction} retains its static prompt and executed trace",
            checks,
        )
        require(
            all(
                record["bytes"] > 0
                and len(record["sha256"]) == 64
                and bool(record["path"])
                for record in (
                    row["raw_result"],
                    row["raw_trajectory"],
                    row["executed_video"],
                )
            ),
            f"{model_id} {row['pair_id']}/{direction} hashes raw scoring, trajectory, and video evidence",
            checks,
        )
        if expected_future_interface == "action_only_not_applicable":
            future_valid = row["imagined_future_artifact"] is None
        else:
            future = row["imagined_future_artifact"]
            future_valid = (
                isinstance(future, dict)
                and future.get("kind") == "latent_tensor"
                and future.get("bytes", 0) > 0
                and len(future.get("sha256", "")) == 64
                and bool(future.get("path"))
            )
        require(
            row["future_interface"] == expected_future_interface and future_valid,
            f"{model_id} {row['pair_id']}/{direction} preserves its released future-interface boundary",
            checks,
        )

    require(
        payload["measurement"] == {
            "oracle_actions": 0,
            "dynamic_prompts": 0,
            "simulator_state_role": "post_action_scoring_and_visualization_only",
        }
        and all(
            cells["left"]["prompt"].replace(" to the left of ", " to the right of ")
            == cells["right"]["prompt"]
            for cells in by_pair.values()
        ),
        f"{model_id} uses mirrored static direct commands with no oracle or prompt switching",
        checks,
    )

    paired = summary["paired_endpoint_responses"]
    paired_by_id = {pair["pair_id"]: pair for pair in paired}
    require(
        len(paired) == 7
        and set(paired_by_id) == set(expected_pair_ids)
        and sum(pair["endpoint_response_direction"] == "aligned" for pair in paired)
        == expected_aligned_pairs
        and all(
            pair["endpoint_response_direction"] in {"aligned", "anti_directed"}
            and pair["physical_initial_state_sha256"]
            == by_pair[pair["pair_id"]]["left"]["physical_initial_state_sha256"]
            == by_pair[pair["pair_id"]]["right"]["physical_initial_state_sha256"]
            and pair["first_ten_executed_action_rms"] > 0
            and pair["first_ten_executed_action_rms_steps_used"] == 10
            and pair["action_metric_unavailable_reason"] is None
            and by_pair[pair["pair_id"]]["left"]["action_trace"]["sha256"]
            != by_pair[pair["pair_id"]]["right"]["action_trace"]["sha256"]
            for pair in paired
        ),
        f"{model_id} records matched initial state and distinct executed actions for all seven pairs",
        checks,
    )

    invalid_path = workspace / manifest["files"]["invalid_attempt_ledger"]["path"]
    runtime_path = workspace / manifest["files"]["runtime_intervention_ledger"]["path"]
    invalid = load_json(invalid_path)["events"]
    runtime = load_json(runtime_path)["events"]
    require(
        len(invalid) == expected_invalid_attempts
        and len(payload["invalid_attempts"]) == expected_invalid_attempts
        and payload["retained_invalid_attempt_ids"]
        == sorted(event["id"] for event in invalid)
        == sorted(event["id"] for event in payload["invalid_attempts"])
        and all(
            event["model_id"] == model_id
            and event["classification"] in {"technical_invalid", "partial"}
            and event["behavioral_result_valid"] is False
            and event["wall_latency_valid"] is False
            for event in invalid
        ),
        f"{model_id} keeps exactly {expected_invalid_attempts} infrastructure-invalid attempts outside model denominators",
        checks,
    )
    require(
        runtime == []
        and payload["applied_runtime_intervention_ids"] == []
        and all(
            source["total_event_count"]
            == source["selected_model_event_count"]
            == len(source["selected_event_ids"])
            == 0
            and source["ignored_other_model_event_count"] == 0
            and source["applied_event_ids"] == []
            and source["bytes"] == manifest["files"]["runtime_intervention_ledger"]["bytes"]
            and source["sha256"] == manifest["files"]["runtime_intervention_ledger"]["sha256"]
            for source in payload["intervention_ledger_sources"]
        ),
        f"{model_id} records zero runtime interventions without inventing thermal exclusions",
        checks,
    )
    require(
        all(
            source["total_event_count"]
            == source["selected_model_event_count"]
            == len(source["selected_event_ids"])
            == len(source["retained_event_ids"])
            == expected_invalid_attempts
            and source["ignored_other_model_event_count"] == 0
            and source["bytes"] == manifest["files"]["invalid_attempt_ledger"]["bytes"]
            and source["sha256"] == manifest["files"]["invalid_attempt_ledger"]["sha256"]
            for source in payload["invalid_attempt_ledger_sources"]
        ),
        f"{model_id} invalid-attempt sources account for every retained event",
        checks,
    )
    return {
        "manifest": str(manifest_path.relative_to(workspace)),
        "manifest_sha256": sha256(manifest_path),
        "slice": str(slice_path.relative_to(workspace)),
        "slice_sha256": sha256(slice_path),
        "valid_episode_count": 14,
        "requested_success_count": expected_success_count,
        "aligned_endpoint_pair_count": expected_aligned_pairs,
        "invalid_attempt_count": expected_invalid_attempts,
        "runtime_intervention_count": 0,
    }


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


def validate_cfg_ablation_v2a015(
    workspace: Path,
    amendment_path: Path,
    checks: list[str],
) -> dict[str, Any]:
    """Validate the disclosed, post-result Cosmos3/DreamZero CFG ablation freeze."""
    amendment = load_json(amendment_path)
    require(
        amendment["schema_version"]
        == "vla-wam-shared-v2-post-result-cfg-ablation-v1"
        and amendment["study_id"] == "vla_wam_language_steerability_v2"
        and amendment["amendment_id"] == "V2-A015"
        and amendment["status"]
        == "frozen_after_baseline_results_and_before_any_cfg_ablation_model_request_or_behavioral_inference"
        and amendment["recorded_at_git_head"]
        == "22218d01c2301cc6ad28c9c1c53905045d8f2e9c",
        "V2-A015 is explicitly frozen post-result and before any CFG-ablation request or behavior",
        checks,
    )

    disclosure = amendment["known_result_disclosure"]
    require(
        all(
            phrase in disclosure["statement"]
            for phrase in (
                "known when this exploratory ablation was selected",
                "disclosed post-result intervention",
                "not preregistration",
                "not a rewrite of any frozen result",
            )
        ),
        "V2-A015 discloses known baselines and forbids a preregistration or frozen-result claim",
        checks,
    )
    expected_baselines = {
        "cosmos3_nano_baseline": {
            "artifact": "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_direct_gate.json",
            "sha256": "4a6cc1d61593c7ba5272e1707f6bbe51261f7d23438070992bd75fd9e95fdb93",
            "checkpoint": "nvidia/Cosmos3-Nano-Policy-DROID",
            "revision": "6706d7680581c255ff61e0f3bb49d90eac55c79e",
            "source_commit": "411d25b2e35bc441126f48c44a4b93e1c0564274",
        },
        "dreamzero_baseline": {
            "artifact": "artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_droid_direct_gate.json",
            "sha256": "4c76cdc3ca9eaf227d21d160199408f22e1b3dd7a71176a5a5dbe22223714461",
            "checkpoint": "GEAR-Dreams/DreamZero-DROID",
            "revision": "96ad344138c66e82536422432ad742f015784942",
            "source_commit": "ab790c198fbce33503358efbbd4187ce9a89adf3",
        },
    }
    for baseline_id, expected in expected_baselines.items():
        baseline = disclosure[baseline_id]
        artifact_path = workspace / baseline["artifact"]
        require(
            all(baseline[key] == value for key, value in expected.items())
            and artifact_path.is_file()
            and sha256(artifact_path) == baseline["sha256"],
            f"V2-A015 preserves and hash-binds the exact {baseline_id} evidence",
            checks,
        )
    require(
        disclosure["cosmos3_nano_baseline"]["guidance"] == 3.0
        and disclosure["cosmos3_nano_baseline"]["result"]
        == "LEFT 3/3; RIGHT 3/3; 3/3 aligned endpoint pairs"
        and disclosure["dreamzero_baseline"]["video_guidance"] == 5.0
        and disclosure["dreamzero_baseline"]["action_guidance_equivalent"] == 1.0
        and disclosure["dreamzero_baseline"]["result"]
        == "LEFT 2/3; RIGHT 1/3; 3/3 aligned endpoint pairs",
        "V2-A015 records the exact known guidance settings and six-cell baseline outcomes",
        checks,
    )

    arms = amendment["arms"]
    require(
        len(arms) == 2
        and [arm["arm_id"] for arm in arms]
        == ["cosmos3_nano_no_cfg_g1", "dreamzero_action_cfg_s2"]
        and all(arm["behavioral_episode_count"] == 6 for arm in arms)
        and sum(arm["behavioral_episode_count"] for arm in arms) == 12,
        "V2-A015 freezes exactly two six-cell arms and twelve new behavioral cells",
        checks,
    )
    arms_by_id = {arm["arm_id"]: arm for arm in arms}
    cosmos = arms_by_id["cosmos3_nano_no_cfg_g1"]
    require(
        cosmos["model_id"] == "cosmos3_nano_policy_droid"
        and cosmos["arena"] == "droid_robolab"
        and cosmos["checkpoint"] == expected_baselines["cosmos3_nano_baseline"]["checkpoint"]
        and cosmos["checkpoint_revision"]
        == expected_baselines["cosmos3_nano_baseline"]["revision"]
        and cosmos["source_commit"]
        == expected_baselines["cosmos3_nano_baseline"]["source_commit"]
        and cosmos["baseline_guidance"] == 3.0
        and cosmos["guidance"] == 1.0
        and cosmos["num_steps"] == 4
        and cosmos["shift"] == 5.0
        and cosmos["action_chunk_shape"] == [32, 8],
        "V2-A015 changes Cosmos3 Nano only from joint CFG g=3 to g=1 while fixing steps, shift, source, and shape",
        checks,
    )
    require(
        "conditional prediction without a CFG blend" in cosmos["intervention"]
        and "33-frame RGB future" in cosmos["future_contract"],
        "V2-A015 defines Cosmos3 g=1 as no blend and retains its decoded-future contract",
        checks,
    )

    dreamzero = arms_by_id["dreamzero_action_cfg_s2"]
    require(
        dreamzero["model_id"] == "dreamzero_droid_action_cfg"
        and dreamzero["arena"] == "droid_robolab"
        and dreamzero["checkpoint"] == expected_baselines["dreamzero_baseline"]["checkpoint"]
        and dreamzero["checkpoint_revision"]
        == expected_baselines["dreamzero_baseline"]["revision"]
        and dreamzero["source_commit"]
        == expected_baselines["dreamzero_baseline"]["source_commit"]
        and dreamzero["baseline_action_guidance_equivalent"] == 1.0
        and dreamzero["action_guidance"] == 2.0
        and dreamzero["video_guidance"] == 5.0
        and dreamzero["runtime_num_inference_steps"] == 16
        and dreamzero["dit_cache"] is True
        and dreamzero["evaluated_dit_steps"] == 8
        and dreamzero["action_chunk_shape"] == [24, 8]
        and dreamzero["executed_open_loop_horizon"] == 8,
        "V2-A015 changes DreamZero action-equivalent guidance 1 to 2 while fixing video CFG 5, runtime 16, cache 8, source, and control shape",
        checks,
    )
    caveat = dreamzero["negative_branch_caveat"]
    require(
        all(
            phrase in caveat
            for phrase in (
                "fixed visual-quality negative prompt",
                "not a strict empty-text unconditional prompt",
                "CFG-style negative-branch action guidance",
                "rather than an official DreamZero action-CFG feature",
            )
        )
        and "a_uncond + 2*(a_cond-a_uncond)" in dreamzero["intervention"]
        and "joint latent video future" in dreamzero["future_contract"],
        "V2-A015 labels DreamZero's fixed-negative branch accurately and retains its latent-video future",
        checks,
    )

    grid = amendment["behavioral_grid"]
    expected_prompts = {
        "left": "Put the Rubik's cube to the left of the bowl.",
        "right": "Put the Rubik's cube to the right of the bowl.",
    }
    require(
        grid["prompt_family"] == "direct_command"
        and grid["prompts"] == expected_prompts
        and grid["environment_seeds"] == [8300, 8301, 8302]
        and grid["sampling_seed_labels"] == [8300, 8301, 8302]
        and grid["requested_relations"] == ["left", "right"]
        and grid["new_behavioral_episode_count"]
        == len(arms) * len(grid["environment_seeds"]) * len(grid["requested_relations"])
        == 12,
        "V2-A015 fixes the exact direct prompts, paired seeds 8300-8302, relations, and twelve-cell cross-product",
        checks,
    )
    require(
        grid["prompt_controller"] == "episode_static"
        and grid["oracle_actions"] == 0
        and grid["subtask_coach"] is False
        and grid["prompt_switching"] is False
        and grid["progress_conditioned_language"] is False
        and grid["simulator_video_required"] is True
        and grid["executed_action_trace_required"] is True
        and grid["all_exposed_futures_retained"] is True
        and "do not stop early" in grid["completion_rule"],
        "V2-A015 requires static language, no oracle or coach, complete video/actions/futures, and no outcome stopping",
        checks,
    )

    release_gates = amendment["fixed_observation_release_gates"]
    require(
        set(release_gates) == set(arms_by_id)
        and any("bit-identical" in item for item in release_gates["cosmos3_nano_no_cfg_g1"])
        and any("LEFT and RIGHT actions" in item for item in release_gates["cosmos3_nano_no_cfg_g1"])
        and any("[32,8]" in item and "33 RGB frames" in item for item in release_gates["cosmos3_nano_no_cfg_g1"])
        and any("scale-1 overlay" in item and "bit-identical" in item for item in release_gates["dreamzero_action_cfg_s2"])
        and any("Scale-2 LEFT and RIGHT actions differ" in item for item in release_gates["dreamzero_action_cfg_s2"])
        and any("finite" in item and "[24,8]" in item for item in release_gates["dreamzero_action_cfg_s2"]),
        "V2-A015 gates behavior on deterministic repeats, directional response, baseline equivalence, and output contracts",
        checks,
    )

    expected_denominator_policy = [
        "Each model and guidance configuration has its own six-cell DROID denominator.",
        "The preserved baselines are referenced by committed hash and are not rerun or overwritten.",
        "Fixed probes, partial runs, and infrastructure-invalid attempts remain outside behavioral denominators.",
        "Cosmos3 Nano and DreamZero results are never pooled, and no DROID result is pooled with RoboTwin.",
    ]
    require(
        amendment["denominator_policy"] == expected_denominator_policy,
        "V2-A015 keeps guidance arms, models, and arenas in independent denominators and excludes invalid attempts",
        checks,
    )
    followup = amendment["optional_followup_not_yet_authorized"]
    require(
        followup["authorized_behavioral_episode_count"] == 0
        and followup["condition"]
        == "Only after both six-cell V2-A015 arms are compiled without outcome-based selection."
        and followup["candidate_arms"]
        == [
            "Cosmos3 Nano guidance 5.0, six cells",
            "DreamZero action guidance 3.0 with video guidance 5.0, six cells",
        ],
        "V2-A015 authorizes zero optional follow-up episodes before both primary arms are compiled",
        checks,
    )
    require(
        "do not claim a powered improvement" in amendment["analysis"]["inference_boundary"]
        and "missing futures are never zeros" in amendment["analysis"]["future_rule"],
        "V2-A015 keeps its pilot inference descriptive and missing futures outside numeric zeros",
        checks,
    )
    return {
        "path": str(amendment_path.relative_to(workspace)),
        "sha256": sha256(amendment_path),
        "arm_count": len(arms),
        "new_behavioral_episode_count": grid["new_behavioral_episode_count"],
        "optional_followup_authorized_episode_count": followup[
            "authorized_behavioral_episode_count"
        ],
    }


def validate_cfg_ablation_v2a015_media(
    workspace: Path,
    continuation_state: dict[str, Any],
    checks: list[str],
) -> dict[str, Any]:
    """Fail closed on the complete V2-A015 actual and predicted media."""
    expected_prompts = {
        "left": "Put the Rubik's cube to the left of the bowl.",
        "right": "Put the Rubik's cube to the right of the bowl.",
    }
    expected_cells = {
        (seed, relation)
        for seed in (8300, 8301, 8302)
        for relation in ("left", "right")
    }
    arm_specs = {
        "cosmos3_nano_no_cfg_g1": {
            "manifest_path": workspace
            / "artifacts/vla_wam_shared_v2/media/cfg_v2a015/"
            "cosmos3_nano_g1/media_manifest.json",
            "result_path": workspace
            / "artifacts/vla_wam_shared_v2/pilot/expansion/"
            "cosmos3_nano_v2a015_no_cfg_g1_result.json",
            "model_id": "cosmos3_nano_policy_droid",
            "state_future_path_key": "prediction_video_path",
            "state_future_hash_key": "prediction_video_sha256",
            "state_future_count_key": "retained_local_prediction_horizon_count",
            "future_count": 64,
        },
        "dreamzero_action_cfg_s2": {
            "manifest_path": workspace
            / "artifacts/vla_wam_shared_v2/media/cfg_v2a015/"
            "dreamzero_action_cfg_s2/media_manifest.json",
            "result_path": workspace
            / "artifacts/vla_wam_shared_v2/pilot/expansion/"
            "dreamzero_v2a015_action_cfg_s2_result.json",
            "model_id": "dreamzero_droid_action_cfg",
            "state_future_path_key": "imagination_video_path",
            "state_future_hash_key": "imagination_video_sha256",
            "state_future_count_key": "complete_official_decode_count",
            "future_count": 6,
        },
    }

    cfg_state = continuation_state["cfg_ablation_v2a015"]
    require(
        cfg_state["status"] == "complete_compiled_evidence_and_publication_media"
        and cfg_state["authorized_cells_completed"] == 12
        and cfg_state["authorized_cells_remaining"] == 0,
        "V2-A015 continuation state requires complete compiled evidence and publication media",
        checks,
    )
    publication_media = cfg_state["final_compiled_comparison"]["publication_media"]
    require(
        "All six valid intervention cells per arm are shown"
        in publication_media["selection_policy"]
        and "no outcome-based selection" in publication_media["selection_policy"],
        "V2-A015 publication media retains every intervention cell without outcome selection",
        checks,
    )

    reports: dict[str, Any] = {}
    workspace_resolved = workspace.resolve()
    for arm_id, spec in arm_specs.items():
        manifest_path = spec["manifest_path"]
        result_path = spec["result_path"]
        manifest = load_json(manifest_path)
        result = load_json(result_path)
        require(
            manifest["schema_version"]
            == "vla-wam-shared-v2-v2a015-cfg-media-v1"
            and manifest["status"]
            == "complete_all_six_cells_actual_and_prediction_media"
            and manifest["amendment_id"] == "V2-A015"
            and manifest["arm_id"] == result["arm_id"] == arm_id
            and manifest["model_id"] == result["model_id"] == spec["model_id"]
            and manifest["exact_prompts"]
            == result["exact_prompts"]
            == expected_prompts,
            f"V2-A015 {arm_id} media fixes its schema, complete status, model, arm, and exact prompts",
            checks,
        )

        source_result = manifest["source_result"]
        require(
            result["status"] == "complete"
            and result["summary"]["valid_episode_count"] == 6
            and source_result["bytes"] == result_path.stat().st_size
            and source_result["sha256"] == sha256(result_path),
            f"V2-A015 {arm_id} media hash-binds its committed compiled result",
            checks,
        )

        input_cells = manifest["input_cells"]
        observed_cells = {
            (cell["environment_seed"], cell["relation"])
            for cell in input_cells
        }
        require(
            len(input_cells) == 6
            and observed_cells == expected_cells
            and set(manifest["request_or_decode_counts"])
            == {f"seed{seed}_{relation}" for seed, relation in expected_cells},
            f"V2-A015 {arm_id} media contains exactly the six paired seed-direction cells",
            checks,
        )
        episodes = {
            (episode["environment_seed"], episode["requested_relation"]): episode
            for episode in result["episodes"]
        }
        require(
            len(episodes) == 6 and set(episodes) == expected_cells,
            f"V2-A015 {arm_id} committed result has the same six media cells",
            checks,
        )

        total_future_sources = 0
        for cell in input_cells:
            relation = cell["relation"]
            cell_id = f"seed{cell['environment_seed']}_{relation}"
            episode = episodes[(cell["environment_seed"], relation)]
            prediction_sources = cell["prediction_sources_in_order"]
            source_count = cell["prediction_source_count"]
            require(
                cell["prompt"] == expected_prompts[relation]
                and cell["actions_executed"] == episode["actions_executed"]
                and cell["requested_success"] == episode["requested_success"]
                and cell["complete_viewport_video"]
                == episode["simulator_artifacts"]["viewport_video"]
                and source_count == len(prediction_sources)
                == manifest["request_or_decode_counts"][cell_id],
                f"V2-A015 {arm_id} {cell_id} media matches its compiled episode and complete rollout",
                checks,
            )

            if arm_id == "cosmos3_nano_no_cfg_g1":
                requests = episode["imagined_future_requests"]
                require(
                    episode["decoded_future_count"] == len(requests) == source_count
                    and [request["request_index"] for request in requests]
                    == list(range(source_count))
                    and prediction_sources
                    == [request["decoded_future"] for request in requests]
                    and cell["prediction_shapes"]
                    == [request["decoded_future_shape"] for request in requests]
                    and all(
                        shape == [33, 528, 640, 3]
                        for shape in cell["prediction_shapes"]
                    ),
                    f"V2-A015 {arm_id} {cell_id} retains every 33-frame local horizon in request order",
                    checks,
                )
            else:
                official_decodes = episode["official_decoded_futures"]
                require(
                    episode["official_decoded_future_count"]
                    == len(official_decodes)
                    == source_count
                    == 1
                    and prediction_sources == official_decodes
                    and cell["prediction_shapes"] == [],
                    f"V2-A015 {arm_id} {cell_id} retains its complete official reset decode",
                    checks,
                )
            total_future_sources += source_count

        if arm_id == "cosmos3_nano_no_cfg_g1":
            require(
                total_future_sources == spec["future_count"] == 64,
                "V2-A015 Cosmos3 Nano publication media contains all 64 ordered local prediction horizons",
                checks,
            )
        else:
            require(
                total_future_sources == spec["future_count"] == 6
                and result["future_retention_audit"]
                ["behavioral_official_decoded_future_count"]
                == 6,
                "V2-A015 DreamZero publication media contains all six complete official decodes",
                checks,
            )

        outputs = manifest["outputs"]
        require(
            set(outputs)
            == {
                "actual_poster",
                "actual_video",
                "prediction_or_imagination_poster",
                "prediction_or_imagination_video",
            },
            f"V2-A015 {arm_id} media declares both complete videos and their posters",
            checks,
        )
        for output_id, record in outputs.items():
            relative_path = Path(record["path"])
            resolved_path = (workspace / relative_path).resolve()
            try:
                resolved_path.relative_to(workspace_resolved)
                path_inside_repository = True
            except ValueError:
                path_inside_repository = False
            require(
                not relative_path.is_absolute() and path_inside_repository,
                f"V2-A015 {arm_id} {output_id} resolves inside the repository",
                checks,
            )
            validate_file_record(
                workspace,
                record,
                f"V2-A015 {arm_id} {output_id}",
                checks,
            )

        renderer_encoding = manifest["renderer"]["encoding"]
        require(
            renderer_encoding["codec"] == "libx264"
            and renderer_encoding["pixel_format"] == "yuv420p"
            and renderer_encoding["movflags"] == "+faststart",
            f"V2-A015 {arm_id} renderer fixes H.264, yuv420p, and fast-start encoding",
            checks,
        )
        output_validation = manifest["output_validation"]
        require(
            "H.264/yuv420p" in output_validation["policy"]
            and "fast-start" in output_validation["policy"],
            f"V2-A015 {arm_id} records its publication-video validation policy",
            checks,
        )
        for output_id in ("actual", "prediction_or_imagination"):
            validation = output_validation[output_id]
            frame_count = validation["frame_count"]
            expected_frame_indices = [0, frame_count // 2, frame_count - 1]
            offsets = validation["faststart_atom_offsets"]
            require(
                validation["codec_name"] == "h264"
                and validation["pixel_format"] == "yuv420p"
                and validation["width"] == 1280
                and validation["height"] == 480
                and validation["duration_s"] > 0
                and validation["fps"] > 0
                and frame_count > 0
                and offsets["ftyp"] == 0
                and offsets["moov"] < offsets["mdat"]
                and validation["decoded_frame_indices"] == expected_frame_indices
                and [sample["frame_index"] for sample in validation["decoded_frame_samples"]]
                == expected_frame_indices
                and all(
                    len(sample["decoded_bgr_sha256"]) == 64
                    for sample in validation["decoded_frame_samples"]
                ),
                f"V2-A015 {arm_id} {output_id} records H.264/yuv420p, fast-start, and first/middle/last decodes",
                checks,
            )

        claim_boundary = manifest["claim_boundary"].lower()
        if arm_id == "cosmos3_nano_no_cfg_g1":
            require(
                "actual composite contains complete simulator viewport executions"
                in claim_boundary
                and "local model-prediction horizon" in claim_boundary
                and "does not make a continuous imagined rollout" in claim_boundary
                and "simulator execution" in claim_boundary,
                "V2-A015 Cosmos3 Nano media distinguishes local horizons from execution and continuous imagination",
                checks,
            )
        else:
            require(
                "actual composite contains complete simulator viewport executions"
                in claim_boundary
                and "imagination composite" in claim_boundary
                and "official model decodes are not simulator execution" in claim_boundary
                and "not an official dreamzero action-cfg mode" in claim_boundary,
                "V2-A015 DreamZero media distinguishes official imagination from execution and official action CFG",
                checks,
            )

        state_media = publication_media[arm_id]
        future_output = outputs["prediction_or_imagination_video"]
        require(
            state_media["manifest_path"]
            == str(manifest_path.relative_to(workspace))
            and state_media["manifest_sha256"] == sha256(manifest_path)
            and state_media["actual_video_path"] == outputs["actual_video"]["path"]
            and state_media["actual_video_sha256"]
            == outputs["actual_video"]["sha256"]
            and state_media[spec["state_future_path_key"]] == future_output["path"]
            and state_media[spec["state_future_hash_key"]]
            == future_output["sha256"]
            and state_media[spec["state_future_count_key"]] == spec["future_count"],
            f"V2-A015 continuation state hash-binds the {arm_id} manifest and both videos",
            checks,
        )
        reports[arm_id] = {
            "manifest_path": str(manifest_path.relative_to(workspace)),
            "manifest_sha256": sha256(manifest_path),
            "actual_video_sha256": outputs["actual_video"]["sha256"],
            "prediction_or_imagination_video_sha256": future_output["sha256"],
            "input_cell_count": len(input_cells),
            "retained_future_count": total_future_sources,
        }

    return reports


def validate_pi0_fast_confirmation(
    workspace: Path, confirmation_path: Path, expansion_path: Path, checks: list[str]
) -> dict[str, Any] | None:
    """Validate the optional post-pilot evidence slice without changing pilot checks."""
    if not confirmation_path.exists():
        return None
    confirmation = load_json(confirmation_path)
    expansion = load_json(expansion_path)
    require(
        confirmation["schema_version"] == "vla-wam-shared-v2-pi0-fast-direct-confirmation-v1",
        "pi0-FAST confirmation uses the dedicated immutable-follow-up schema",
        checks,
    )
    require(
        confirmation["model_id"] == "pi0_fast_droid_vla",
        "pi0-FAST confirmation identifies the frozen checkpoint",
        checks,
    )
    source = confirmation["source_registry"]
    source_path = Path(source["path"])
    if not source_path.is_absolute():
        source_path = workspace / source_path
    require(
        source_path.resolve() == expansion_path.resolve() and source["sha256"] == sha256(expansion_path),
        "pi0-FAST confirmation hashes the frozen directional registry",
        checks,
    )
    episodes = confirmation["episodes"]
    expected_cells = {(seed, direction) for seed in range(8300, 8310) for direction in ("left", "right")}
    observed_cells = {(int(row["environment_seed"]), row["requested_relation"]) for row in episodes}
    require(len(episodes) == 20 and observed_cells == expected_cells,
            "pi0-FAST confirmation contains exactly twenty registered valid cells", checks)
    prompts = expansion["prompts"]
    require(
        all(row["prompt_family"] == "direct_command" and row["prompt"] == prompts[row["requested_relation"]] for row in episodes),
        "pi0-FAST confirmation preserves the registry-matched static direct prompts",
        checks,
    )
    summary = confirmation["summary"]
    require(summary["episode_count"] == 20 and summary["pair_count"] == 10,
            "pi0-FAST confirmation summary reports ten exact pairs", checks)
    require(
        set(summary["by_direction"]) == {"left", "right"}
        and all(summary["by_direction"][direction]["episodes"] == 10 for direction in ("left", "right")),
        "pi0-FAST confirmation reports ten trials per direction", checks,
    )
    for direction in ("left", "right"):
        metrics = summary["by_direction"][direction]
        successes = sum(row["requested_success"] for row in episodes if row["requested_relation"] == direction)
        require(metrics["successes"] == successes and 0 <= successes <= 10,
                f"pi0-FAST confirmation {direction} numerator matches episode rows", checks)
        interval = metrics["success_wilson_95"]
        require(interval["confidence"] == 0.95 and 0 <= interval["lower"] <= interval["upper"] <= 1,
                f"pi0-FAST confirmation {direction} includes a valid Wilson 95% interval", checks)
        require(
            all(key in metrics for key in ("verified_pickups", "entered_requested_region", "released_in_requested_region")),
            f"pi0-FAST confirmation {direction} includes progression stage counts", checks,
        )
    pairs = summary["paired_endpoint_responses"]
    require(len(pairs) == 10 and {int(pair["environment_seed"]) for pair in pairs} == set(range(8300, 8310)),
            "pi0-FAST confirmation retains ten registered endpoint pairs", checks)
    require(
        all(
            pair["endpoint_response_direction"] in {"aligned", "anti_directed", "none"}
            and pair.get("physical_initial_state_sha256")
            and isinstance(pair.get("first_ten_action_rms_steps_used"), int)
            and 0 <= pair["first_ten_action_rms_steps_used"] <= 10
            and (
                (pair["first_ten_action_rms"] is not None
                 and pair["first_ten_action_rms"] >= 0
                 and pair["first_ten_action_rms_steps_used"] > 0
                 and pair["first_ten_action_rms_unavailable_reason"] is None)
                or (pair["first_ten_action_rms"] is None
                    and pair["first_ten_action_rms_unavailable_reason"] in {
                        "no_common_executed_actions", "paired_action_shape_mismatch"
                    })
            )
            for pair in pairs
        ),
        "pi0-FAST confirmation records a bounded paired action metric or an explicit unavailable reason", checks,
    )
    episode_by_cell = {
        (int(row["environment_seed"]), row["requested_relation"]): row for row in episodes
    }
    require(
        all(
            pair["physical_initial_state_sha256"]
            == episode_by_cell[(int(pair["environment_seed"]), "left")]["physical_initial_state_sha256"]
            == episode_by_cell[(int(pair["environment_seed"]), "right")]["physical_initial_state_sha256"]
            for pair in pairs
        ),
        "pi0-FAST confirmation verifies identical recorded initial state inside every pair", checks,
    )
    require(
        all(
            row["operational_wall_latency_valid"] == (not bool(row["runtime_intervention_ids"]))
            for row in episodes
        ),
        "pi0-FAST confirmation excludes thermally intervened wall latency without invalidating behavior", checks,
    )
    intervention_sources = confirmation["intervention_ledger_sources"]
    for ledger_source in intervention_sources:
        validate_file_record(workspace, ledger_source, "pi0-FAST intervention ledger", checks)
        require(
            ledger_source["total_event_count"]
            == ledger_source["selected_model_event_count"]
            + ledger_source["ignored_other_model_event_count"]
            and set(ledger_source["applied_event_ids"])
            <= set(ledger_source["selected_event_ids"]),
            "pi0-FAST intervention ledger records selected, ignored, and applied events", checks,
        )
    applied_ids = sorted(
        event_id for row in episodes for event_id in row["runtime_intervention_ids"]
    )
    require(
        confirmation["applied_runtime_intervention_ids"] == applied_ids
        == sorted(
            event_id for source_record in intervention_sources
            for event_id in source_record["applied_event_ids"]
        )
        and summary["operational_wall_latency_valid_episodes"]
        + summary["operational_wall_latency_excluded_episodes"] == 20,
        "pi0-FAST confirmation accounts for applied thermal events and all wall-latency rows", checks,
    )
    invalid_attempts = confirmation["invalid_attempts"]
    require(
        all(
            event.get("classification") in {"technical_invalid", "partial"}
            and event.get("behavioral_result_valid") is False
            and event.get("wall_latency_valid") is False
            for event in invalid_attempts
        ),
        "pi0-FAST confirmation keeps invalid or partial attempts outside behavior and latency counts", checks,
    )
    require(summary["invalid_attempt_count"] == len(invalid_attempts),
            "pi0-FAST confirmation invalid-attempt count matches its separate ledger", checks)
    return {
        "path": str(confirmation_path.relative_to(workspace)),
        "sha256": sha256(confirmation_path),
        "episode_count": len(episodes),
        "invalid_attempt_count": len(invalid_attempts),
    }


def validate_robotwin_confirmations(
    workspace: Path, registry_path: Path, checks: list[str]
) -> dict[str, dict[str, Any]]:
    """Validate optional per-model confirmation slices, never replacing pilot checks."""
    registry = load_json(registry_path)
    reports: dict[str, dict[str, Any]] = {}
    for model_id in registry["models"]:
        stem = model_id.removesuffix("_robotwin")
        path = workspace / f"artifacts/vla_wam_shared_v2/pilot/results/{stem}_direct_confirmation.json"
        if not path.exists():
            continue
        compiled = load_json(path)
        require(
            compiled["schema_version"] == "vla-wam-shared-v2-robotwin-direct-confirmation-v1"
            and compiled["model_id"] == model_id,
            f"{model_id} confirmation uses the dedicated ten-scene schema", checks,
        )
        source = compiled["source_registry"]
        require(source["sha256"] == sha256(registry_path),
                f"{model_id} confirmation hashes the frozen directional registry", checks)
        episodes = compiled["episodes"]
        expected = {(int(scene["environment_seed"]), direction) for scene in registry["scenes"] for direction in ("left", "right")}
        observed = {(int(row["environment_seed"]), row["requested_relation"]) for row in episodes}
        require(len(episodes) == 20 and observed == expected,
                f"{model_id} confirmation contains exactly twenty registered cells", checks)
        summary = compiled["summary"]
        require(summary["episode_count"] == 20 and summary["pair_count"] == 10,
                f"{model_id} confirmation reports ten exact pairs", checks)
        for direction in ("left", "right"):
            metric = summary["by_direction"][direction]
            numerator = sum(row["requested_success"] for row in episodes if row["requested_relation"] == direction)
            require(metric["episodes"] == metric["started"] == 10 and metric["successes"] == numerator,
                    f"{model_id} confirmation {direction} has ten stage-accounted trials", checks)
            interval = metric["success_wilson_95"]
            require(interval["confidence"] == 0.95 and 0 <= interval["lower"] <= interval["upper"] <= 1,
                    f"{model_id} confirmation {direction} has a Wilson 95% interval", checks)
        pairs = summary["paired_endpoint_responses"]
        require(len(pairs) == 10 and all(pair["endpoint_response_direction"] in {"aligned", "anti_directed", "none"} for pair in pairs),
                f"{model_id} confirmation records ten endpoint shifts and alignment labels", checks)
        cells = {(int(row["environment_seed"]), row["requested_relation"]): row for row in episodes}
        require(all(
            pair["physical_initial_state_sha256"]
            == cells[(int(pair["environment_seed"]), "left")]["physical_initial_state_sha256"]
            == cells[(int(pair["environment_seed"]), "right")]["physical_initial_state_sha256"]
            and pair["initial_state_coverage"]
            == cells[(int(pair["environment_seed"]), "left")]["initial_state_coverage"]
            == cells[(int(pair["environment_seed"]), "right")]["initial_state_coverage"]
            and bool(pair["initial_state_coverage"].get("hash_input_initial_fields"))
            and bool(pair["initial_state_coverage"].get("adapter_state_limits"))
            for pair in pairs
        ), f"{model_id} confirmation verifies matched recorded initial state with explicit coverage limits", checks)
        phase_by_seed = {
            int(scene["environment_seed"]): scene.get("phase", "new_expansion" if int(scene["environment_seed"]) >= 4300003 else "completed_pilot")
            for scene in registry["scenes"]
        }
        for row in episodes:
            trace = row.get("action_trace")
            prospective = phase_by_seed[int(row["environment_seed"])] == "new_expansion"
            require((trace is not None) if prospective else True,
                    f"{model_id} confirmation requires action traces for prospective cells", checks)
            if trace is not None:
                validate_file_record(workspace, trace, f"{model_id} action trace", checks)
                require(trace.get("executed_array") in {"executed", "denormalized"}
                        and "count" in trace and "shape" in trace
                        and bool(trace.get("arrays"))
                        and all("count" in item and "shape" in item for item in trace["arrays"].values()),
                        f"{model_id} action trace records verified executed count and shape", checks)
            require(row["future_interface"] in {"decoded_future_video", "latent_only_future_not_decodable", "action_only_not_applicable"},
                    f"{model_id} confirmation labels future interface without treating absence as zero", checks)
            require(
                row["operational_wall_latency_valid"] == (not bool(row["runtime_intervention_ids"])),
                f"{model_id} confirmation fails closed when a thermal intervention reaches a latency-valid row", checks,
            )
        coverage = summary["first_ten_executed_action_rms_coverage"]
        available_pairs = sum(pair["first_ten_executed_action_rms"] is not None for pair in pairs)
        require(
            coverage == {
                "available_pairs": available_pairs,
                "prospective_pairs": 7,
                "total_pairs": 10,
                "coverage": f"{available_pairs}/10",
            },
            f"{model_id} confirmation reports exact paired action-RMS coverage", checks,
        )
        require(all(
            (pair["first_ten_executed_action_rms"] is None
             and pair["first_ten_executed_action_rms_steps_used"] == 0
             and pair["action_metric_unavailable_reason"] == "historical_pair00_pair02_action_trace_not_required_by_preclarification")
            if phase_by_seed[int(pair["environment_seed"])] == "completed_pilot"
            else (
                isinstance(pair["first_ten_executed_action_rms_steps_used"], int)
                and 0 <= pair["first_ten_executed_action_rms_steps_used"] <= 10
                and (
                    (pair["first_ten_executed_action_rms"] is not None
                     and pair["first_ten_executed_action_rms"] >= 0
                     and pair["first_ten_executed_action_rms_steps_used"] > 0
                     and pair["action_metric_unavailable_reason"] is None)
                    or (pair["first_ten_executed_action_rms"] is None
                        and pair["action_metric_unavailable_reason"] in {
                            "no_common_executed_actions", "paired_action_shape_mismatch",
                            "action_trace_is_scalar_not_a_sequence"
                        })
                )
            )
            for pair in pairs
        ), f"{model_id} confirmation preserves short action traces with steps-used or an explicit reason", checks)
        intervention_sources = compiled["intervention_ledger_sources"]
        for source in intervention_sources:
            validate_file_record(workspace, source, f"{model_id} intervention ledger", checks)
            require(
                source["total_event_count"]
                == source["selected_model_event_count"] + source["ignored_other_model_event_count"]
                and set(source["applied_event_ids"]) <= set(source["selected_event_ids"]),
                f"{model_id} intervention ledger records selected and ignored model counts", checks,
            )
        applied_ids = sorted(
            event_id for row in episodes for event_id in row["runtime_intervention_ids"]
        )
        require(
            compiled["applied_runtime_intervention_ids"] == applied_ids
            == sorted(event_id for source in intervention_sources for event_id in source["applied_event_ids"]),
            f"{model_id} confirmation accounts for every applied runtime intervention ID", checks,
        )
        invalid = compiled["invalid_attempts"]
        require(
            summary["invalid_attempt_count"] == len(invalid)
            and all(
                event.get("classification") in {"technical_invalid", "partial"}
                and event.get("behavioral_result_valid") is False
                and event.get("wall_latency_valid") is False
                for event in invalid
            ),
            f"{model_id} confirmation keeps invalid attempts outside behavior and latency counts", checks,
        )
        invalid_sources = compiled["invalid_attempt_ledger_sources"]
        for source in invalid_sources:
            validate_file_record(workspace, source, f"{model_id} invalid-attempt ledger", checks)
            require(
                source["total_event_count"]
                == source["selected_model_event_count"] + source["ignored_other_model_event_count"]
                and set(source["retained_event_ids"]) <= set(source["selected_event_ids"]),
                f"{model_id} invalid ledger records selected and ignored model counts", checks,
            )
        require(
            compiled["retained_invalid_attempt_ids"] == sorted(event["id"] for event in invalid)
            == sorted(event_id for source in invalid_sources for event_id in source["retained_event_ids"]),
            f"{model_id} confirmation accounts for every retained invalid-attempt ID", checks,
        )
        reports[model_id] = {"path": str(path.relative_to(workspace)), "sha256": sha256(path), "episode_count": len(episodes)}
    return reports


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
    lingbot_result_path = (
        workspace / "artifacts/vla_wam_shared_v2/pilot/results/lingbot_va_direct_gate.json"
    )
    pi0_fast_result_path = (
        workspace / "artifacts/vla_wam_shared_v2/pilot/results/pi0_fast_direct_gate.json"
    )
    pi0_fast_confirmation_path = (
        workspace / "artifacts/vla_wam_shared_v2/pilot/results/pi0_fast_direct_confirmation.json"
    )
    runtime_interventions_path = (
        workspace / "artifacts/vla_wam_shared_v2/pilot/runtime_interventions.json"
    )
    directional_expansion_path = (
        workspace / "artifacts/vla_wam_shared_v2/pilot/directional_expansion.json"
    )
    action_trace_amendment_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/action_trace_instrumentation_amendment.json"
    )
    directional_fixtures_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/directional_fixture_validation.json"
    )
    pi0_fast_expansion_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/pi0_fast_directional_expansion.json"
    )
    continuation_state_path = (
        workspace / "artifacts/vla_wam_shared_v2/continuation_state.json"
    )
    post_result_amendment_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/post_result_expansion_amendment.json"
    )
    second_wave_amendment_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/post_result_second_wave_amendment.json"
    )
    dreamzero_amendment_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/post_result_dreamzero_amendment.json"
    )
    cfg_ablation_v2a015_amendment_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/"
        "post_result_cfg_ablation_v2a015_amendment.json"
    )
    current_stack_amendment_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/post_result_current_stack_replication_amendment.json"
    )
    current_stack_registry_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "pi0_fast_current_stack_v2a008_registry.json"
    )
    current_stack_release_probe_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "pi0_fast_current_stack_v2a008_release_probe.json"
    )
    lawam_withdrawal_amendment_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/post_result_lawam_withdrawal_amendment.json"
    )
    pi05_current_amendment_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/post_result_pi05_current_stack_media_gate_amendment.json"
    )
    cosmos3_nano_amendment_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/post_result_cosmos3_nano_droid_amendment.json"
    )
    pi05_checkpoint_manifest_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_checkpoint_manifest.json"
    )
    pi05_current_registry_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_registry.json"
    )
    pi05_result_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_direct_gate.json"
    )
    pi05_release_probe_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_release_probe.json"
    )
    pi05_fixed_observation_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_fixed_observation.json"
    )
    pi05_invalid_attempts_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_invalid_attempts.json"
    )
    pi05_provenance_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_provenance.json"
    )
    pi05_media_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/media/pi05_current_stack_v2a010/media_manifest.json"
    )
    cosmos3_nano_registry_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_nano_policy_droid_v2a011_registry.json"
    )
    cosmos3_nano_result_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_nano_policy_droid_direct_gate.json"
    )
    cosmos3_nano_fixed_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_nano_policy_droid_fixed_observation.json"
    )
    cosmos3_nano_invalid_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_nano_policy_droid_invalid_attempts.json"
    )
    cosmos3_nano_runtime_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_nano_policy_droid_runtime_interventions.json"
    )
    cosmos3_nano_layout_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_nano_policy_droid_raw_layout_compatibility.json"
    )
    cosmos3_nano_provenance_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_nano_policy_droid_provenance.json"
    )
    cosmos3_nano_media_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/media/"
        "cosmos3_nano_policy_droid_v2a011/media_manifest.json"
    )
    cosmos3_super_amendment_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/post_result_cosmos3_super_droid_amendment.json"
    )
    cosmos3_super_registry_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_super_droid_v2a012_registry.json"
    )
    cosmos3_super_snapshot_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_super_v2a012_hf_snapshot.json"
    )
    cosmos3_super_runtime_gate_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_super_v2a012_runtime_gate.json"
    )
    cosmos3_super_image_only_amendment_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/"
        "post_result_cosmos3_super_image_only_v2a014_amendment.json"
    )
    cosmos3_super_image_only_overlay_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_super_image_only_v2a014_registry_overlay.json"
    )
    cosmos3_super_image_only_result_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_super_image_only_v2a014_result.json"
    )
    cosmos3_super_image_only_invalid_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_super_image_only_v2a014_invalid_attempts.json"
    )
    cosmos3_super_image_only_provenance_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_super_image_only_v2a014_provenance.json"
    )
    cosmos3_super_image_only_media_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/media/"
        "cosmos3_super_base_v2a014/media_manifest.json"
    )
    cosmos3_super_image_only_probe_path = (
        workspace / "experiments/cosmos/run_cosmos3_super_v2a014_probe.py"
    )
    cosmos3_super_image_only_media_builder_path = (
        workspace / "tools/build_cosmos3_super_v2a014_media.py"
    )
    cosmos3_super_a100_manifest_path = (
        workspace / "handoff/k8s/cosmos3-super-a100-2gpu-256gi-ali.yaml"
    )
    cosmos3_super_runbook_path = workspace / "experiments/cosmos/COSMOS3_SUPER_V2A012.md"
    cosmos3_super_builder_path = workspace / "tools/build_cosmos3_super_checkpoint_manifest.py"
    cosmos3_super_finalizer_path = workspace / "tools/finalize_cosmos3_super_registry.py"
    cosmos3_super_pod_manifest_path = (
        workspace / "handoff/k8s/cosmos3-super-b200-4gpu-256gi-ali.yaml"
    )
    cosmos3_edge_base_amendment_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/post_result_cosmos3_edge_base_amendment.json"
    )
    cosmos3_edge_base_registry_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_edge_base_v2a013_registry.json"
    )
    cosmos3_edge_base_runbook_path = (
        workspace / "experiments/cosmos/COSMOS3_EDGE_BASE_V2A013.md"
    )
    cosmos3_edge_base_builder_path = (
        workspace / "tools/build_v2a013_cosmos3_edge_base_registry.py"
    )
    cosmos3_edge_base_fixed_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_edge_base_v2a013_fixed_observation.json"
    )
    cosmos3_edge_base_invalid_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_edge_base_v2a013_invalid_attempts.json"
    )
    cosmos3_edge_base_provenance_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_edge_base_v2a013_provenance.json"
    )
    cosmos3_edge_base_curobo_audit_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_edge_base_v2a013_curobo_usd_audit.json"
    )
    cosmos3_edge_base_media_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/media/"
        "cosmos3_edge_base_v2a013/media_manifest.json"
    )
    cosmos3_edge_policy_result_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_edge_droid_direct_gate.json"
    )
    dreamzero_readiness_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_droid_readiness.json"
    )
    dreamzero_result_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_droid_direct_gate.json"
    )
    dreamzero_raw_collection_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_droid_raw_collection_manifest.json"
    )
    dreamzero_media_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/media/dreamzero_droid/media_manifest.json"
    )
    dreamzero_imagination_media_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/media/dreamzero_droid/imagination/imagination_media_manifest.json"
    )
    groot_result_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/groot_n17_droid_v2_registry.json"
    )
    groot_readiness_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/groot_n17_droid_readiness.json"
    )
    lingbot_vla_result_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/lingbot_vla_4b_direct_gate.json"
    )
    lingbot_vla_readiness_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/lingbot_vla_4b_robotwin_readiness.json"
    )
    lawam_access_retry_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/lawam_dinov3_authenticated_access_retry.json"
    )
    light_wam_result_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/expansion/light_wam_robotwin_direct_gate.json"
    )
    light_wam_media_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/media/light_wam_robotwin/media_manifest.json"
    )
    video_gallery_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/media/video_first_gallery_manifest.json"
    )
    efficient_pair03_integration_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/directional_confirmation/efficient_wam_rt_pair03_integration.json"
    )
    efficient_pairs04_09_manifest_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/directional_confirmation/efficient_wam_rt_pairs04_09_evidence_manifest.json"
    )
    fastwam_pairs03_09_manifest_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/directional_confirmation/fastwam_pairs03_09_evidence_manifest.json"
    )
    lingbot_pairs03_09_manifest_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/directional_confirmation/lingbot_va_pairs03_09_evidence_manifest.json"
    )
    bundle_manifest_path = workspace / "handoff/repo_bundles/MANIFEST.json"
    paired_media_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/media/robotwin_wam_pairs/media_index.json"
    )
    droid_paired_media_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/media/droid_pi0_fast_pairs/media_index.json"
    )
    figures_manifest_path = (
        workspace / "artifacts/vla_wam_shared_v2/figures/figures_manifest.json"
    )
    protocol = load_json(protocol_path)
    media = load_json(media_path)
    execution = load_json(execution_path)
    technical = load_json(technical_path)
    efficient_result = load_json(efficient_result_path)
    fastwam_result = load_json(fastwam_result_path)
    lingbot_result = load_json(lingbot_result_path)
    pi0_fast_result = load_json(pi0_fast_result_path)
    runtime_interventions = load_json(runtime_interventions_path)
    directional_expansion = load_json(directional_expansion_path)
    action_trace_amendment = load_json(action_trace_amendment_path)
    directional_fixtures = load_json(directional_fixtures_path)
    pi0_fast_expansion = load_json(pi0_fast_expansion_path)
    continuation_state = load_json(continuation_state_path)
    post_result_amendment = load_json(post_result_amendment_path)
    second_wave_amendment = load_json(second_wave_amendment_path)
    dreamzero_amendment = load_json(dreamzero_amendment_path)
    cfg_ablation_v2a015_amendment = load_json(cfg_ablation_v2a015_amendment_path)
    current_stack_amendment = load_json(current_stack_amendment_path)
    current_stack_registry = load_json(current_stack_registry_path)
    current_stack_release_probe = load_json(current_stack_release_probe_path)
    lawam_withdrawal_amendment = load_json(lawam_withdrawal_amendment_path)
    pi05_current_amendment = load_json(pi05_current_amendment_path)
    cosmos3_nano_amendment = load_json(cosmos3_nano_amendment_path)
    pi05_checkpoint_manifest = load_json(pi05_checkpoint_manifest_path)
    pi05_current_registry = load_json(pi05_current_registry_path)
    pi05_result = load_json(pi05_result_path)
    pi05_release_probe = load_json(pi05_release_probe_path)
    pi05_fixed_observation = load_json(pi05_fixed_observation_path)
    pi05_invalid_attempts = load_json(pi05_invalid_attempts_path)
    pi05_provenance = load_json(pi05_provenance_path)
    pi05_media = load_json(pi05_media_path)
    cosmos3_nano_registry = load_json(cosmos3_nano_registry_path)
    cosmos3_super_amendment = load_json(cosmos3_super_amendment_path)
    cosmos3_super_registry = load_json(cosmos3_super_registry_path)
    cosmos3_super_snapshot = load_json(cosmos3_super_snapshot_path)
    cosmos3_super_runtime_gate = load_json(cosmos3_super_runtime_gate_path)
    cosmos3_super_image_only_amendment = load_json(
        cosmos3_super_image_only_amendment_path
    )
    cosmos3_super_image_only_overlay = load_json(
        cosmos3_super_image_only_overlay_path
    )
    cosmos3_super_image_only_result = load_json(
        cosmos3_super_image_only_result_path
    )
    cosmos3_super_image_only_invalid = load_json(
        cosmos3_super_image_only_invalid_path
    )
    cosmos3_super_image_only_provenance = load_json(
        cosmos3_super_image_only_provenance_path
    )
    cosmos3_super_image_only_media = load_json(
        cosmos3_super_image_only_media_path
    )
    cosmos3_edge_base_amendment = load_json(cosmos3_edge_base_amendment_path)
    cosmos3_edge_base_registry = load_json(cosmos3_edge_base_registry_path)
    cosmos3_edge_base_fixed = load_json(cosmos3_edge_base_fixed_path)
    cosmos3_edge_base_invalid = load_json(cosmos3_edge_base_invalid_path)
    cosmos3_edge_base_provenance = load_json(cosmos3_edge_base_provenance_path)
    cosmos3_edge_base_curobo_audit = load_json(
        cosmos3_edge_base_curobo_audit_path
    )
    cosmos3_edge_base_media = load_json(cosmos3_edge_base_media_path)
    dreamzero_readiness_artifact = load_json(dreamzero_readiness_path)
    dreamzero_result = load_json(dreamzero_result_path)
    dreamzero_raw_collection = load_json(dreamzero_raw_collection_path)
    dreamzero_media = load_json(dreamzero_media_path)
    dreamzero_imagination_media = load_json(dreamzero_imagination_media_path)
    groot_result = load_json(groot_result_path)
    groot_readiness_artifact = load_json(groot_readiness_path)
    lingbot_vla_result = load_json(lingbot_vla_result_path)
    lingbot_vla_readiness_artifact = load_json(lingbot_vla_readiness_path)
    lawam_access_retry = load_json(lawam_access_retry_path)
    light_wam_result = load_json(light_wam_result_path)
    light_wam_media = load_json(light_wam_media_path)
    video_gallery = load_json(video_gallery_path)
    bundle_manifest = load_json(bundle_manifest_path)
    paired_media = load_json(paired_media_path)
    droid_paired_media = load_json(droid_paired_media_path)
    figures_manifest = load_json(figures_manifest_path)
    checks: list[str] = []

    # Load above so a missing or malformed amendment fails before any weaker
    # downstream state can be accepted; the dedicated validator re-loads and
    # binds the exact on-disk bytes into the final report.
    require(
        cfg_ablation_v2a015_amendment["amendment_id"] == "V2-A015",
        "V2-A015 CFG-ablation amendment is present and parseable",
        checks,
    )
    cfg_ablation_v2a015 = validate_cfg_ablation_v2a015(
        workspace, cfg_ablation_v2a015_amendment_path, checks
    )
    cfg_ablation_v2a015["publication_media"] = (
        validate_cfg_ablation_v2a015_media(workspace, continuation_state, checks)
    )

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
        == [
            "V2-A003_robotwin_execution_config_registry",
            "V2-A004_robotwin_directional_confirmation",
        ],
        "both post-pilot adaptive amendments are disclosed separately",
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
    require(
        post_amendments[1]["inference_completed_before_amendment"] == 18,
        "the directional-confirmation amendment discloses all 18 preceding WAM pilot episodes",
        checks,
    )
    require(
        post_amendments[1]["artifact"]
        == "artifacts/vla_wam_shared_v2/pilot/directional_expansion.json",
        "the directional-confirmation amendment points to its frozen scene registry",
        checks,
    )

    require(
        action_trace_amendment["status"]
        == "frozen_before_any_pair03_pair09_wam_inference"
        and action_trace_amendment["new_directional_confirmation_episodes_completed_before_amendment"]
        == 0,
        "action-trace instrumentation is frozen before every prospective WAM cell", checks,
    )
    amendment_repositories = action_trace_amendment["repositories"]
    portable_instrumentation_commits = {
        record["target_commit"] for record in bundle_manifest["bundles"]
    }
    require(
        [record["model_id"] for record in amendment_repositories]
        == ["efficient_wam_rt_robotwin", "fastwam_robotwin", "lingbot_va_robotwin"],
        "action-trace amendment covers exactly the three confirmation WAMs", checks,
    )
    require(
        all(
            len(record["commit_before_instrumentation"]) == 40
            and len(record["commit_with_instrumentation"]) == 40
            and record["commit_before_instrumentation"] != record["commit_with_instrumentation"]
            for record in amendment_repositories
        ),
        "action-trace amendment records distinct before/after commits", checks,
    )
    for repository_record in amendment_repositories:
        repository_root = Path(repository_record["repository"])
        require(
            repository_record["commit_with_instrumentation"]
            in portable_instrumentation_commits,
            f"{repository_record['model_id']} instrumentation commit is available in the portable handoff",
            checks,
        )
        for script_record in repository_record["files"]:
            script_path = repository_root / script_record["path"]
            external_checkout_present = repository_root.is_dir()
            require(
                not external_checkout_present or script_path.is_file(),
                f"{repository_record['model_id']} local instrumented script exists when its external checkout is present",
                checks,
            )
            require(
                not external_checkout_present
                or script_path.stat().st_size == script_record["bytes"],
                f"{repository_record['model_id']} local instrumented script byte count matches when present",
                checks,
            )
            require(
                not external_checkout_present or sha256(script_path) == script_record["sha256"],
                f"{repository_record['model_id']} local instrumented script hash matches when present",
                checks,
            )
    measurement_contract = action_trace_amendment["measurement_contract"]
    require(
        measurement_contract["array"] == "executed"
        and "env.take_action" in measurement_contract["definition"]
        and measurement_contract["result_metadata"] == ["path", "sha256", "count", "shape"]
        and "seven of ten" in measurement_contract["historical_limit"],
        "action-trace amendment defines exact executed actions and prospective 7/10 coverage", checks,
    )
    unchanged_text = action_trace_amendment["reason"].lower()
    require(
        all(term in unchanged_text for term in ("prompts", "seeds", "checkpoints", "action horizons", "success predicates")),
        "action-trace amendment changes measurement only, not behavior, prompts, seeds, checkpoints, or scoring", checks,
    )

    require(
        directional_expansion["status"]
        == "frozen_before_directional_expansion_inference",
        "RoboTwin directional expansion is frozen before new expansion inference",
        checks,
    )
    require(
        directional_expansion["trigger"]["completed_episode_count_before_freeze"]
        == 18,
        "directional expansion discloses the 18 known pilot outcomes",
        checks,
    )
    require(
        directional_expansion["trigger"]["wording_grid_authorized"] is False,
        "RoboTwin wording grid remains unauthorized after one-direction-only competence",
        checks,
    )
    require(
        directional_expansion["models"]
        == [
            "efficient_wam_rt_robotwin",
            "fastwam_robotwin",
            "lingbot_va_robotwin",
        ],
        "the same three completed WAM pilots enter directional confirmation",
        checks,
    )
    expansion_scenes = directional_expansion["scenes"]
    require(
        len(expansion_scenes) == 10
        and [scene["environment_seed"] for scene in expansion_scenes]
        == list(range(4300000, 4300010))
        and [scene["sampling_seed"] for scene in expansion_scenes]
        == list(range(8400, 8410)),
        "directional confirmation freezes ten contiguous exact-pair scenes",
        checks,
    )
    require(
        [scene["anchor_task"] for scene in expansion_scenes]
        == [
            "place_a2b_left",
            "place_a2b_right",
            "place_a2b_left",
            "place_a2b_right",
            "place_a2b_left",
            "place_a2b_right",
            "place_a2b_left",
            "place_a2b_right",
            "place_a2b_left",
            "place_a2b_right",
        ],
        "directional confirmation alternates five LEFT and five RIGHT native anchors",
        checks,
    )
    require(
        sum(scene["phase"] == "completed_pilot" for scene in expansion_scenes) == 3
        and sum(scene["phase"] == "new_expansion" for scene in expansion_scenes)
        == 7,
        "directional confirmation distinguishes three observed and seven prospective scenes",
        checks,
    )
    require(
        directional_expansion["episode_accounting"]["new_episode_count"] == 42
        and directional_expansion["episode_accounting"][
            "final_direct_confirmation_episode_count"
        ]
        == 60,
        "directional expansion arithmetic is 42 new and 60 total direct episodes",
        checks,
    )
    require(
        directional_expansion["fixture_validation"]["status"]
        == "completed_valid_before_model_inference"
        and directional_expansion["fixture_validation"]["artifact"]
        == "artifacts/vla_wam_shared_v2/pilot/directional_fixture_validation.json",
        "directional registry points to completed model-blind fixture evidence",
        checks,
    )
    require(
        directional_fixtures["status"] == "valid"
        and directional_fixtures["scene_count"] == 7
        and directional_fixtures["valid_scene_count"] == 7,
        "all seven prospective RoboTwin fixtures initialize successfully",
        checks,
    )
    require(
        directional_fixtures["registry_sha256"] == sha256(directional_expansion_path),
        "fixture report hashes the current frozen directional registry",
        checks,
    )
    require(
        all(
            not scene["initially_in_left_region"]
            and not scene["initially_in_right_region"]
            and scene["movable_model_name"] != scene["reference_model_name"]
            for scene in directional_fixtures["scenes"]
        ),
        "all prospective scenes begin outside both goal regions with distinct objects",
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
    lingbot_summary = lingbot_result["summary"]
    require(
        lingbot_summary["episode_count"] == 6 and lingbot_summary["pair_count"] == 3,
        "compiled LingBot-VA pilot contains six episodes in three exact pairs",
        checks,
    )
    require(
        lingbot_summary["by_direction"]["left"]["successes"] == 3
        and lingbot_summary["by_direction"]["right"]["successes"] == 0,
        "compiled LingBot-VA pilot preserves the observed 3/3 LEFT and 0/3 RIGHT result",
        checks,
    )
    require(
        lingbot_summary["pilot_gate_decision"] == "expand_direct_directional_bias_only",
        "LingBot-VA pilot follows the frozen one-direction-only expansion gate",
        checks,
    )
    require(
        all(
            episode["imagined_future_artifact"] is not None
            and episode["imagined_future_artifact"]["kind"] == "latent_tensor"
            for episode in lingbot_result["episodes"]
        ),
        "all six LingBot-VA cells retain their first predicted latent tensor",
        checks,
    )
    require(
        lingbot_summary["operational_wall_latency_valid_episodes"] == 3
        and len(runtime_interventions["events"]) == 3,
        "three thermally interrupted LingBot wall times are explicitly excluded",
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
            episode["imagined_future_artifact"] is None
            for episode in fastwam_result["episodes"]
        ),
        "FastWAM action-only interface records imagined-video evidence as not applicable",
        checks,
    )
    pi0_fast_summary = pi0_fast_result["summary"]
    require(
        pi0_fast_summary["episode_count"] == 6
        and pi0_fast_summary["pair_count"] == 3,
        "compiled pi0-FAST pilot contains six episodes in three exact pairs",
        checks,
    )
    require(
        pi0_fast_summary["by_direction"]["left"]["successes"] == 0
        and pi0_fast_summary["by_direction"]["right"]["successes"] == 3,
        "compiled pi0-FAST pilot preserves the observed 0/3 LEFT and 3/3 RIGHT result",
        checks,
    )
    require(
        pi0_fast_summary["aligned_endpoint_pairs"] == 3
        and pi0_fast_summary["nonzero_first_chunk_pairs"] == 3,
        "all three pi0-FAST pairs change actions and redirect endpoints toward RIGHT",
        checks,
    )
    require(
        pi0_fast_summary["pilot_gate_decision"]
        == "expand_direct_directional_bias_only",
        "pi0-FAST pilot follows the frozen one-direction-only expansion gate",
        checks,
    )
    require(
        pi0_fast_result["measurement"]["oracle_actions"] == 0
        and pi0_fast_result["measurement"]["dynamic_prompts"] == 0
        and not pi0_fast_result["measurement"]["subtask_progress_checking"],
        "pi0-FAST pilot preserves the no-oracle and no-subtask-coach design",
        checks,
    )
    require(
        pi0_fast_expansion["status"]
        == "frozen_before_new_directional_expansion_inference",
        "pi0-FAST directional confirmation is frozen before new inference",
        checks,
    )
    require(
        pi0_fast_expansion["completed_seeds"] == [8300, 8301, 8302]
        and pi0_fast_expansion["new_seeds"]
        == [8303, 8304, 8305, 8306, 8307, 8308, 8309],
        "pi0-FAST confirmation separates three observed and seven prospective seeds",
        checks,
    )
    require(
        pi0_fast_expansion["episode_accounting"]["new_episode_count"] == 14
        and pi0_fast_expansion["episode_accounting"][
            "final_direct_confirmation_episode_count"
        ]
        == 20,
        "pi0-FAST confirmation arithmetic is 14 new and 20 total direct episodes",
        checks,
    )
    require(
        not pi0_fast_expansion["trigger"]["wording_grid_authorized"]
        and pi0_fast_expansion["trigger"]["known_result"]["left_successes"] == 0
        and pi0_fast_expansion["trigger"]["known_result"]["right_successes"] == 3,
        "pi0-FAST expansion discloses the observed one-direction-only trigger",
        checks,
    )
    require(
        pi0_fast_expansion["execution"]["instruction_controller"] == "static"
        and not pi0_fast_expansion["execution"]["subtask_progress_checking"]
        and pi0_fast_expansion["execution"]["video_mode"] == "viewport",
        "pi0-FAST confirmation preserves static prompts, no coach, and viewport video",
        checks,
    )
    require(
        len(technical["events"]) == 5
        and technical["events"][-1]["id"] == "PI0FAST-TECH-001"
        and technical["events"][-1]["classification"] == "environment_repair",
        "pre-episode failures and both environment repairs remain in a separate technical ledger",
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
    require(
        paired_media["status"] == "complete" and len(paired_media["items"]) == 3,
        "paired RoboTwin gallery contains three completed model videos",
        checks,
    )
    require(
        {item["model_id"] for item in paired_media["items"]}
        == {
            "efficient_wam_rt_robotwin",
            "fastwam_robotwin",
            "lingbot_va_robotwin",
        },
        "paired RoboTwin gallery represents every completed WAM pilot",
        checks,
    )
    require(
        all(item["left"]["success"] and not item["right"]["success"] for item in paired_media["items"]),
        "each paired gallery item is a disclosed LEFT success and matched RIGHT failure",
        checks,
    )
    fastwam_media = next(
        item for item in paired_media["items"] if item["model_id"] == "fastwam_robotwin"
    )
    require(
        fastwam_media["source_video_correction"]["classification"]
        == "source_capture_pixel_layout_reconstruction"
        and all(
            item["source_video_correction"] is None
            for item in paired_media["items"]
            if item["model_id"] != "fastwam_robotwin"
        ),
        "FastWAM pixel-layout reconstruction is explicit and not applied to other models",
        checks,
    )
    for item in paired_media["items"]:
        for artifact_name in (
            "video",
            "poster",
            "square_video",
            "square_poster",
            "captions",
        ):
            validate_file_record(
                workspace,
                item[artifact_name],
                f"{item['model_id']} paired {artifact_name}",
                checks,
            )

    require(
        droid_paired_media["status"] == "complete",
        "paired pi0-FAST DROID media package is complete",
        checks,
    )
    droid_item = droid_paired_media["item"]
    require(
        droid_item["model_id"] == "pi0_fast_droid_vla"
        and not droid_item["left"]["success"]
        and droid_item["right"]["success"],
        "pi0-FAST media is the frozen LEFT-failure and matched RIGHT-success pair",
        checks,
    )
    require(
        droid_item["environment_seed"] == 8300
        and droid_item["sampling_seed"] == 8300,
        "pi0-FAST media preserves the exact seed-8300 pair",
        checks,
    )
    for artifact_name in (
        "video",
        "poster",
        "square_video",
        "square_poster",
        "captions",
    ):
        validate_file_record(
            workspace,
            droid_item[artifact_name],
            f"pi0-FAST paired {artifact_name}",
            checks,
        )

    require(
        figures_manifest["status"] == "complete"
        and len(figures_manifest["figures"]) == 16,
        "reader-first figure manifest contains sixteen completed exports",
        checks,
    )
    require(
        figures_manifest["protocol_sha256"] == sha256(protocol_path),
        "reader-first figure manifest hashes the current protocol",
        checks,
    )
    for figure_id, record in figures_manifest["figures"].items():
        validate_file_record(workspace, record, f"figure {figure_id}", checks)

    require(
        continuation_state["study_status"]
        == "pi0_fast_release_gate_failed_pi05_current_stack_complete_cosmos3_nano_current_stack_complete_cosmos3_edge_base_interface_passed_behavior_blocked_cosmos3_super_image_only_interface_passed_no_behavior_lawam_withdrawn",
        "continuation state names the completed nonbehavioral Edge and Super interface gates",
        checks,
    )
    queue = continuation_state["experiment_queue"]
    require(
        [item["priority"] for item in queue] == [0, 1, 2, 3, 4],
        "continuation queue has one unambiguous priority order",
        checks,
    )
    require(
        [item["id"] for item in queue]
        == [
            "pi0_fast_directional_confirmation",
            "robotwin_three_wam_directional_confirmation",
            "lingbot_vla_4b_robotwin_pilot",
            "groot_n17_droid_pilot",
            "dreamzero_droid_direct_gate",
        ],
        "continuation queue preserves the five authorized or blocked next experiments",
        checks,
    )
    groot_readiness = continuation_state.get("groot_n17_readiness", {})
    lingbot_vla_readiness = continuation_state.get("lingbot_vla_4b_readiness", {})
    lawam_readiness = continuation_state.get("lawam_robotwin_readiness", {})
    dreamzero_readiness = continuation_state.get("dreamzero_droid_readiness", {})
    expected_do_not_rerun = [
        f"{model_id}/robotwin_pair_{pair_index:02d}"
        for model_id in (
            "efficient_wam_rt_robotwin",
            "fastwam_robotwin",
            "lingbot_va_robotwin",
        )
        for pair_index in range(3, 10)
    ]
    post_result_decision = continuation_state.get("post_result_decision", {})
    require(
        queue[0]["status"] == "complete"
        and queue[0].get("result_artifact")
        == "artifacts/vla_wam_shared_v2/pilot/results/pi0_fast_direct_confirmation.json"
        and queue[1]["status"]
        == "pairs03_09_complete_full_compilers_blocked_missing_historical_raw"
        and queue[1]["new_episode_count"] == 42
        and queue[1]["handoff_remaining_episode_count"] == 40
        and queue[1]["completed_new_episode_count"] == 42
        and queue[1]["remaining_new_episode_count"] == 0
        and queue[1]["next_cell"] is None
        and queue[1]["do_not_rerun"] == expected_do_not_rerun
        and queue[2]["status"] == "complete_six_valid_cells_left_only"
        and queue[2].get("artifact_sha256") == sha256(lingbot_vla_result_path)
        and queue[2].get("next_cell") is None
        and queue[3]["status"] == "complete_six_valid_cells_zero_success"
        and queue[3].get("artifact_sha256") == sha256(groot_result_path)
        and queue[3].get("next_cell") is None
        and queue[4]["status"]
        == "complete_six_valid_cells_both_directions_gate"
        and queue[4]["first_episode_count"] == 6
        and queue[4].get("artifact")
        == "artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_droid_direct_gate.json"
        and queue[4].get("artifact_sha256") == sha256(dreamzero_result_path)
        and groot_readiness.get("status") == "complete_6_of_6_valid_cells"
        and groot_readiness.get("server_smoke", {}).get("health_ping") is True
        and groot_readiness.get("server_smoke", {}).get("simulator_episode_started") is False
        and groot_readiness.get("result_sha256") == sha256(groot_result_path)
        and groot_readiness.get("readiness_sha256") == sha256(groot_readiness_path)
        and groot_readiness.get("valid_episode_count") == 6
        and groot_readiness.get("valid_failure_count") == 6
        and groot_readiness.get("aligned_endpoint_pair_count") == 3
        and groot_readiness.get("distinct_executed_action_pair_count") == 3
        and lingbot_vla_readiness.get("status") == "six_cell_gate_complete_left_only"
        and lingbot_vla_readiness.get("result_sha256") == sha256(lingbot_vla_result_path)
        and lingbot_vla_readiness.get("readiness_sha256")
        == sha256(lingbot_vla_readiness_path)
        and lingbot_vla_readiness.get("valid_episode_count") == 6
        and lingbot_vla_readiness.get("competence_gate") == "left_only"
        and lawam_readiness.get("authenticated_access_retry_sha256")
        == sha256(lawam_access_retry_path)
        and dreamzero_readiness.get("status")
        == "complete_six_valid_cells_both_directions_gate"
        and dreamzero_readiness.get("model_action_request_count") == 268
        and dreamzero_readiness.get("behavioral_episode_count") == 6
        and dreamzero_readiness.get("result_sha256") == sha256(dreamzero_result_path)
        and dreamzero_readiness.get("raw_collection_manifest_sha256")
        == sha256(dreamzero_raw_collection_path)
        and dreamzero_readiness.get("media_manifest_sha256")
        == sha256(dreamzero_media_path)
        and dreamzero_readiness.get("imagination_media_manifest_sha256")
        == sha256(dreamzero_imagination_media_path)
        and dreamzero_readiness.get("published_imagination_media", {}).get(
            "official_decode_count"
        )
        == 9
        and dreamzero_readiness.get("published_imagination_media", {}).get(
            "paired_behavioral_comparison_count"
        )
        == 3
        and dreamzero_readiness.get("invalid_attempt_count") == 11
        and dreamzero_readiness.get("runtime_intervention_count") == 0,
        "continuation state freezes every completed bounded gate and exact remaining blockers",
        checks,
    )
    remaining = continuation_state.get("remaining_authorized_work", [])
    require(
        [item.get("priority") for item in remaining] == [0, 1, 2, 3, 4]
        and [item.get("id") for item in remaining]
        == [
            "pi0_fast_current_stack_three_wording_replication",
            "pi05_droid_current_stack_direct_media_gate",
            "cosmos3_nano_policy_droid_direct_gate",
            "cosmos3_super_base_droid_feasibility_probe",
            "cosmos3_edge_base_droid_feasibility_probe",
        ]
        and [item.get("authorized_cells_remaining") for item in remaining]
        == [0, 0, 0, 0, 0]
        and remaining[0].get("status")
        == "release_gate_failed_zero_behavioral_cells_not_runnable_under_frozen_protocol"
        and remaining[0].get("amendment_sha256") == sha256(current_stack_amendment_path)
        and remaining[0].get("release_probe")
        == str(current_stack_release_probe_path.relative_to(workspace))
        and remaining[0].get("release_probe_sha256")
        == sha256(current_stack_release_probe_path)
        and remaining[0].get("model_action_request_count") == 3
        and remaining[0].get("behavioral_episode_count") == 0
        and remaining[1].get("amendment_sha256") == sha256(pi05_current_amendment_path)
        and remaining[1].get("status")
        == "complete_six_valid_behavioral_cells_current_stack_only"
        and remaining[1].get("result") == str(pi05_result_path.relative_to(workspace))
        and remaining[1].get("result_sha256") == sha256(pi05_result_path)
        and remaining[1].get("valid_behavioral_episode_count") == 6
        and remaining[1].get("success_by_relation") == {"left": "1/3", "right": "3/3"}
        and remaining[1].get("aligned_endpoint_pair_count") == 3
        and remaining[1].get("distinct_executed_action_pair_count") == 3
        and remaining[1].get("infrastructure_invalid_attempt_count") == 0
        and remaining[1].get("runtime_intervention_count") == 0
        and remaining[2].get("amendment_sha256") == sha256(cosmos3_nano_amendment_path)
        and remaining[2].get("status") == "complete_six_valid_behavioral_cells_current_stack_only"
        and remaining[2].get("result") == str(cosmos3_nano_result_path.relative_to(workspace))
        and remaining[2].get("result_sha256") == sha256(cosmos3_nano_result_path)
        and remaining[2].get("fixed_observation_sha256") == sha256(cosmos3_nano_fixed_path)
        and remaining[2].get("valid_behavioral_episode_count") == 6
        and remaining[2].get("success_by_relation") == {"left": "3/3", "right": "3/3"}
        and remaining[2].get("aligned_endpoint_pair_count") == 3
        and remaining[2].get("distinct_executed_action_pair_count") == 3
        and remaining[2].get("decoded_future_count") == 37
        and remaining[2].get("invalid_attempt_count") == 2
        and remaining[2].get("runtime_intervention_count") == 0
        and remaining[3].get("status")
        == "image_only_action_and_video_interface_passed_no_behavior"
        and remaining[3].get("amendment_sha256")
        == sha256(cosmos3_super_amendment_path)
        and remaining[3].get("registry_sha256")
        == sha256(cosmos3_super_registry_path)
        and remaining[3].get("checkpoint_snapshot_sha256")
        == sha256(cosmos3_super_snapshot_path)
        and remaining[3].get("authorized_probe_requests_remaining") == 0
        and remaining[3].get("conditional_behavioral_cell_count") == 6
        and remaining[3].get("released_behavioral_cell_count") == 0
        and remaining[3].get("model_load_attempt_count") == 3
        and remaining[3].get("model_action_request_count") == 3
        and remaining[3].get("behavioral_episode_count") == 0
        and remaining[3].get("v2_a014_amendment")
        == str(cosmos3_super_image_only_amendment_path.relative_to(workspace))
        and remaining[3].get("v2_a014_amendment_sha256")
        == sha256(cosmos3_super_image_only_amendment_path)
        and remaining[3].get("v2_a014_registry_overlay")
        == str(cosmos3_super_image_only_overlay_path.relative_to(workspace))
        and remaining[3].get("v2_a014_registry_overlay_sha256")
        == sha256(cosmos3_super_image_only_overlay_path)
        and remaining[3].get("runtime_gate")
        == str(cosmos3_super_runtime_gate_path.relative_to(workspace))
        and remaining[3].get("runtime_gate_sha256")
        == sha256(cosmos3_super_runtime_gate_path)
        and remaining[3].get("result")
        == str(cosmos3_super_image_only_result_path.relative_to(workspace))
        and remaining[3].get("result_sha256")
        == sha256(cosmos3_super_image_only_result_path)
        and remaining[3].get("provenance")
        == str(cosmos3_super_image_only_provenance_path.relative_to(workspace))
        and remaining[3].get("provenance_sha256")
        == sha256(cosmos3_super_image_only_provenance_path)
        and remaining[3].get("invalid_attempt_ledger")
        == str(cosmos3_super_image_only_invalid_path.relative_to(workspace))
        and remaining[3].get("invalid_attempt_ledger_sha256")
        == sha256(cosmos3_super_image_only_invalid_path)
        and remaining[3].get("publication_media")
        == str(cosmos3_super_image_only_media_path.relative_to(workspace))
        and remaining[3].get("publication_media_sha256")
        == sha256(cosmos3_super_image_only_media_path)
        and remaining[4].get("status")
        == "fixed_observation_interface_passed_behavior_blocked_exact_mapping"
        and remaining[4].get("amendment_sha256")
        == sha256(cosmos3_edge_base_amendment_path)
        and remaining[4].get("registry_sha256")
        == sha256(cosmos3_edge_base_registry_path)
        and remaining[4].get("authorized_probe_requests_remaining") == 0
        and remaining[4].get("conditional_behavioral_cell_count") == 6
        and remaining[4].get("released_behavioral_cell_count") == 0
        and remaining[4].get("model_load_attempt_count") == 2
        and remaining[4].get("model_action_request_count") == 3
        and remaining[4].get("behavioral_episode_count") == 0
        and remaining[4].get("fixed_observation_result")
        == str(cosmos3_edge_base_fixed_path.relative_to(workspace))
        and remaining[4].get("fixed_observation_result_sha256")
        == sha256(cosmos3_edge_base_fixed_path)
        and remaining[4].get("provenance")
        == str(cosmos3_edge_base_provenance_path.relative_to(workspace))
        and remaining[4].get("provenance_sha256")
        == sha256(cosmos3_edge_base_provenance_path)
        and remaining[4].get("invalid_attempt_ledger")
        == str(cosmos3_edge_base_invalid_path.relative_to(workspace))
        and remaining[4].get("invalid_attempt_ledger_sha256")
        == sha256(cosmos3_edge_base_invalid_path)
        and remaining[4].get("curobo_mapping_audit")
        == str(cosmos3_edge_base_curobo_audit_path.relative_to(workspace))
        and remaining[4].get("curobo_mapping_audit_sha256")
        == sha256(cosmos3_edge_base_curobo_audit_path)
        and remaining[4].get("publication_media")
        == str(cosmos3_edge_base_media_path.relative_to(workspace))
        and remaining[4].get("publication_media_sha256")
        == sha256(cosmos3_edge_base_media_path)
        and remaining[4].get("preserved_edge_policy_do_not_rerun") is True,
        "continuation state closes and hash-binds the nonbehavioral Super and Edge interface gates",
        checks,
    )
    super_base_state = post_result_decision.get(
        "cosmos3_super_base_feasibility_amendment", {}
    )
    edge_base_state = post_result_decision.get(
        "cosmos3_edge_base_feasibility_amendment", {}
    )
    require(
        super_base_state.get("status")
        == "image_only_action_and_video_interface_passed_no_behavior"
        and super_base_state.get("amendment_id") == "V2-A012"
        and super_base_state.get("sha256") == sha256(cosmos3_super_amendment_path)
        and super_base_state.get("registry_sha256")
        == sha256(cosmos3_super_registry_path)
        and super_base_state.get("authorized_probe_request_count") == 3
        and super_base_state.get("completed_probe_request_count") == 3
        and super_base_state.get("conditional_behavioral_cell_count") == 6
        and super_base_state.get("released_behavioral_cell_count") == 0
        and super_base_state.get("behavioral_episode_count") == 0
        and super_base_state.get("v2_a014_amendment")
        == str(cosmos3_super_image_only_amendment_path.relative_to(workspace))
        and super_base_state.get("v2_a014_amendment_sha256")
        == sha256(cosmos3_super_image_only_amendment_path)
        and super_base_state.get("result")
        == str(cosmos3_super_image_only_result_path.relative_to(workspace))
        and super_base_state.get("result_sha256")
        == sha256(cosmos3_super_image_only_result_path)
        and edge_base_state.get("status")
        == "fixed_observation_interface_passed_behavior_blocked_exact_mapping"
        and edge_base_state.get("amendment_id") == "V2-A013"
        and edge_base_state.get("sha256")
        == sha256(cosmos3_edge_base_amendment_path)
        and edge_base_state.get("registry_sha256")
        == sha256(cosmos3_edge_base_registry_path)
        and edge_base_state.get("authorized_probe_request_count") == 3
        and edge_base_state.get("completed_probe_request_count") == 3
        and edge_base_state.get("conditional_behavioral_cell_count") == 6
        and edge_base_state.get("released_behavioral_cell_count") == 0
        and edge_base_state.get("behavioral_episode_count") == 0
        and edge_base_state.get("fixed_observation_result")
        == str(cosmos3_edge_base_fixed_path.relative_to(workspace))
        and edge_base_state.get("fixed_observation_result_sha256")
        == sha256(cosmos3_edge_base_fixed_path)
        and edge_base_state.get("curobo_mapping_audit")
        == str(cosmos3_edge_base_curobo_audit_path.relative_to(workspace))
        and edge_base_state.get("curobo_mapping_audit_sha256")
        == sha256(cosmos3_edge_base_curobo_audit_path)
        and edge_base_state.get("preserved_edge_policy_do_not_rerun") is True,
        "continuation state hash-binds completed nonbehavioral V2-A012/A014 and V2-A013 evidence",
        checks,
    )
    current_stack_state = post_result_decision.get("current_stack_replication_amendment", {})
    require(
        current_stack_state.get("status")
        == "release_gate_failed_zero_behavioral_cells"
        and current_stack_state.get("amendment_id") == "V2-A008"
        and current_stack_state.get("sha256") == sha256(current_stack_amendment_path)
        and current_stack_state.get("authorized_queue") == []
        and current_stack_state.get("authorized_behavioral_episode_count") == 60
        and current_stack_state.get("behavioral_episode_count") == 0
        and current_stack_state.get("registry")
        == str(current_stack_registry_path.relative_to(workspace))
        and current_stack_state.get("registry_sha256")
        == sha256(current_stack_registry_path)
        and current_stack_state.get("adapter_status")
        == "implemented_release_gate_failed_prompt_sensitivity"
        and current_stack_state.get("release_probe")
        == str(current_stack_release_probe_path.relative_to(workspace))
        and current_stack_state.get("release_probe_sha256")
        == sha256(current_stack_release_probe_path),
        "continuation state binds the V2-A008 release-gate failure without rewriting the frozen amendment",
        checks,
    )
    require(
        current_stack_amendment["schema_version"]
        == "vla-wam-shared-v2-post-result-current-stack-replication-amendment-v1"
        and current_stack_amendment["amendment_id"] == "V2-A008"
        and current_stack_amendment["status"]
        == "frozen_before_current_stack_model_load_or_behavioral_inference"
        and current_stack_amendment["recorded_at_git_head"]
        == "12758797e9e7cbfc28e1a0fb1759c730d72540b8"
        and current_stack_amendment["replication_identity"]["policy_repository"]["commit"]
        == "c23745b5ad24e98f66967ea795a07b2588ed6c79"
        and current_stack_amendment["replication_identity"]["simulator_repository"]["commit"]
        == "0aef241fb088ca21bb4ebd24448940ed56620d17"
        and current_stack_amendment["behavioral_grid"]["episode_count"] == 60
        and current_stack_amendment["claim_boundary"]["historical_wording_queue_status"]
        == "remains blocked and unrun at the historical exact-adapter boundary"
        and len(current_stack_amendment["release_gates"]) == 9,
        "V2-A008 freezes exact current revisions, sixty cells, release gates, and non-comparability",
        checks,
    )
    current_cells = current_stack_registry["cells"]
    current_cell_keys = {
        (
            row["environment_seed"],
            row["prompt_family"],
            row["requested_relation"],
        )
        for row in current_cells
    }
    expected_current_cell_keys = {
        (seed, family, relation)
        for seed in range(8300, 8310)
        for family in (
            "short_command",
            "goal_as_outcome",
            "desired_plus_negated_opposite",
        )
        for relation in ("left", "right")
    }
    require(
        current_stack_registry["schema_version"]
        == "vla-wam-v2a008-pi0-current-stack-registry-v1"
        and current_stack_registry["amendment_id"] == "V2-A008"
        and current_stack_registry["status"]
        == "frozen_before_current_stack_model_load_or_behavioral_inference"
        and current_stack_registry["amendment"]["sha256"]
        == sha256(current_stack_amendment_path)
        and current_stack_registry["protocol"]["sha256"] == sha256(protocol_path)
        and current_stack_registry["replication_identity"]
        == current_stack_amendment["replication_identity"]
        and current_stack_registry["claim_boundary"]
        == current_stack_amendment["claim_boundary"],
        "V2-A008 registry is hash-bound to the amendment, protocol, current revisions, and claim boundary",
        checks,
    )
    require(
        len(current_cells) == 60
        and len({row["cell_id"] for row in current_cells}) == 60
        and current_cell_keys == expected_current_cell_keys
        and current_stack_registry["summary"]["left_right_pair_count"] == 30,
        "V2-A008 registry contains exactly three wording families by ten seeds by LEFT and RIGHT",
        checks,
    )
    protocol_prompts = {row["id"]: row for row in protocol["prompt_families"]}

    def expected_current_prompt(family: str, relation: str) -> str:
        prompt = protocol_prompts[family]
        if family == "short_command":
            return prompt[f"droid_exact_{relation}"]
        return prompt[relation].format(
            movable="Rubik's cube", movable_short="cube", reference="bowl"
        )

    require(
        all(
            row["rendered_prompt"]
            == expected_current_prompt(row["prompt_family"], row["requested_relation"])
            and row["sampling_seed_base"] == row["environment_seed"]
            and row["first_policy_request_sampling_seed"]
            == row["environment_seed"] * 1000
            and row["instruction_controller"] == "static"
            and row["oracle_or_subtask_coach"] is False
            and row["dynamic_prompt_switches"] == 0
            and row["video_mode"] == "viewport"
            and row["executed_action_trace_required"] is True
            and row["valid_failure_retained"] is True
            and row["output_folder_name"]
            == (
                f"v2a008_pi0_current_seed{row['environment_seed']}_"
                f"{row['prompt_family']}_{row['requested_relation']}"
            )
            for row in current_cells
        ),
        "V2-A008 registry renders exact protocol prompts and freezes static seeded video/action-trace cells",
        checks,
    )
    require(
        len(current_stack_registry["adapter_sources"]) == 9,
        "V2-A008 registry enumerates the complete builder, compiler, adapter, preflight, and task-overlay source set",
        checks,
    )
    for source in current_stack_registry["adapter_sources"]:
        validate_file_record(
            workspace,
            source,
            f"V2-A008 adapter source {source['path']}",
            checks,
        )
    probe_requests = current_stack_release_probe.get("requests", [])
    require(
        current_stack_release_probe.get("schema_version")
        == "vla-wam-v2a008-pi0-current-release-probe-v1"
        and current_stack_release_probe.get("amendment_id") == "V2-A008"
        and current_stack_release_probe.get("status")
        == "failed_prompt_sensitivity_zero_behavioral_cells"
        and current_stack_release_probe.get("frozen_sources", {}).get("amendment", {}).get("path")
        == str(current_stack_amendment_path.relative_to(workspace))
        and current_stack_release_probe.get("frozen_sources", {}).get("amendment", {}).get("sha256")
        == sha256(current_stack_amendment_path)
        and current_stack_release_probe.get("frozen_sources", {}).get("registry", {}).get("path")
        == str(current_stack_registry_path.relative_to(workspace))
        and current_stack_release_probe.get("frozen_sources", {}).get("registry", {}).get("sha256")
        == sha256(current_stack_registry_path)
        and current_stack_release_probe.get("replication_identity", {}).get("policy_repository", {}).get("commit")
        == "c23745b5ad24e98f66967ea795a07b2588ed6c79"
        and current_stack_release_probe.get("replication_identity", {}).get("policy_repository", {}).get("config")
        == "pi0_fast_droid_jointpos_polaris"
        and current_stack_release_probe.get("replication_identity", {}).get("simulator_repository", {}).get("commit")
        == "0aef241fb088ca21bb4ebd24448940ed56620d17"
        and current_stack_release_probe.get("replication_identity", {}).get("checkpoint", {}).get("file_count") == 19
        and current_stack_release_probe.get("replication_identity", {}).get("checkpoint", {}).get("payload_bytes") == 10_844_314_410
        and current_stack_release_probe.get("fixed_observation", {}).get("fixture_sha256")
        == "ce8be012347718a162bf0d92ba2fb71a01c570a3462d72ef2c16a86082131778"
        and current_stack_release_probe.get("fixed_observation", {}).get("environment_seed") == 8300
        and current_stack_release_probe.get("fixed_observation", {}).get("sampling_seed") == 8300000
        and current_stack_release_probe.get("fixed_observation", {}).get("neutral_reset")
        == {"left_predicate_at_reset": False, "right_predicate_at_reset": False}
        and len(probe_requests) == 3
        and [request.get("label") for request in probe_requests]
        == ["left_a", "left_b_exact_repeat", "right"]
        and [request.get("requested_relation") for request in probe_requests]
        == ["left", "left", "right"]
        and all(request.get("sampling_seed") == 8300000 for request in probe_requests)
        and all(request.get("shape") == [10, 8] and request.get("dtype") == "float32" for request in probe_requests)
        and {
            request.get("action_sha256") for request in probe_requests
        } == {"570867d7533005a68e3365013f2110a856cbbd09e6e190fcdb1dac5b812072f6"}
        and current_stack_release_probe.get("metrics")
        == {
            "left_exact_repeat_bit_identical": True,
            "left_vs_right_action_rms": 0.0,
            "left_right_action_hashes_equal": True,
        }
        and current_stack_release_probe.get("release_gate", {}).get("passed") is False
        and current_stack_release_probe.get("release_gate", {}).get("failed_condition")
        == "LEFT versus RIGHT prompt-only action output differs"
        and current_stack_release_probe.get("counts")
        == {
            "model_load_count": 1,
            "model_action_request_count": 3,
            "behavioral_episode_count": 0,
            "valid_behavioral_failure_count": 0,
            "infrastructure_invalid_attempt_count": 0,
            "registered_behavioral_cells_unrun": 60,
        },
        "V2-A008 fixed-observation release probe preserves exact repeat, identical LEFT/RIGHT actions, and zero behavioral cells",
        checks,
    )
    require(
        lawam_withdrawal_amendment["schema_version"]
        == "vla-wam-shared-v2-post-result-scope-withdrawal-v1"
        and lawam_withdrawal_amendment["amendment_id"] == "V2-A009"
        and lawam_withdrawal_amendment["status"]
        == "frozen_before_any_lawam_model_inference_or_behavioral_episode"
        and lawam_withdrawal_amendment["withdrawn_experiment"]["behavioral_episode_count_completed"] == 0
        and lawam_withdrawal_amendment["withdrawn_experiment"]["model_action_request_count"] == 0
        and lawam_withdrawal_amendment["withdrawn_experiment"]["authorized_cell_count_after_withdrawal"] == 0,
        "V2-A009 withdraws LaWAM before inference without turning missing evidence into a zero",
        checks,
    )
    require(
        pi05_current_amendment["schema_version"]
        == "vla-wam-shared-v2-post-result-pi05-current-stack-media-gate-v1"
        and pi05_current_amendment["amendment_id"] == "V2-A010"
        and pi05_current_amendment["status"]
        == "frozen_before_current_stack_model_load_or_behavioral_inference"
        and pi05_current_amendment["experiment_identity"]["policy_repository"]["commit"]
        == "c23745b5ad24e98f66967ea795a07b2588ed6c79"
        and pi05_current_amendment["experiment_identity"]["policy_repository"]["config"]
        == "pi05_droid_jointpos_polaris"
        and pi05_current_amendment["experiment_identity"]["simulator_repository"]["commit"]
        == "0aef241fb088ca21bb4ebd24448940ed56620d17"
        and pi05_current_amendment["behavioral_grid"]["episode_count"] == 6
        and len(pi05_current_amendment["release_gates"]) == 9,
        "V2-A010 freezes the exact pi0.5 current-stack identity, six cells, and release gates",
        checks,
    )
    require(
        pi05_checkpoint_manifest["schema_version"]
        == "vla-wam-v2a010-pi05-current-checkpoint-manifest-v1"
        and pi05_checkpoint_manifest["status"]
        == "complete_sha256_hashed_before_model_load"
        and pi05_checkpoint_manifest["file_count"] == 26
        and len(pi05_checkpoint_manifest["files"]) == 26
        and pi05_checkpoint_manifest["payload_bytes"] == 12_434_530_510
        and sum(row["bytes"] for row in pi05_checkpoint_manifest["files"])
        == pi05_checkpoint_manifest["payload_bytes"]
        and len({row["path"] for row in pi05_checkpoint_manifest["files"]}) == 26
        and pi05_checkpoint_manifest["model_load_attempt_count"] == 0,
        "V2-A010 checkpoint manifest retains 26 unique SHA-256 records and zero model loads",
        checks,
    )
    pi05_cells = pi05_current_registry["cells"]
    require(
        pi05_current_registry["schema_version"]
        == "vla-wam-v2a010-pi05-current-stack-registry-v1"
        and pi05_current_registry["amendment"]["sha256"]
        == sha256(pi05_current_amendment_path)
        and pi05_current_registry["checkpoint_manifest"]["sha256"]
        == sha256(pi05_checkpoint_manifest_path)
        and pi05_current_registry["protocol"]["sha256"] == sha256(protocol_path)
        and len(pi05_cells) == 6
        and len({row["cell_id"] for row in pi05_cells}) == 6
        and {
            (row["environment_seed"], row["requested_relation"])
            for row in pi05_cells
        }
        == {(seed, relation) for seed in (8300, 8301, 8302) for relation in ("left", "right")},
        "V2-A010 registry is hash-bound and contains exactly three direct-command LEFT/RIGHT pairs",
        checks,
    )
    require(
        all(
            row["rendered_prompt"]
            == expected_current_prompt("direct_command", row["requested_relation"])
            and row["sampling_seed_base"] == row["environment_seed"]
            and row["first_policy_request_sampling_seed"] == row["environment_seed"] * 1000
            and row["open_loop_horizon"] == 15
            and row["instruction_controller"] == "static"
            and row["oracle_or_subtask_coach"] is False
            and row["dynamic_prompt_switches"] == 0
            and row["video_mode"] == "viewport"
            and row["executed_action_trace_required"] is True
            for row in pi05_cells
        ),
        "V2-A010 registry freezes exact direct prompts, horizon 15, seeded static control, videos, and traces",
        checks,
    )
    require(
        len(pi05_current_registry["adapter_sources"]) == 9,
        "V2-A010 registry enumerates the complete adapter, preflight, compiler, builder, and task sources",
        checks,
    )
    for source in pi05_current_registry["adapter_sources"]:
        validate_file_record(
            workspace,
            source,
            f"V2-A010 adapter source {source['path']}",
            checks,
        )
    require(
        pi05_result["schema_version"] == "vla-wam-v2a010-pi05-current-result-v1"
        and pi05_result["status"] == "complete_6_of_6_valid_current_stack_cells"
        and pi05_result["amendment_id"] == "V2-A010"
        and pi05_result["openpi_commit"] == "c23745b5ad24e98f66967ea795a07b2588ed6c79"
        and pi05_result["openpi_config"] == "pi05_droid_jointpos_polaris"
        and pi05_result["robolab_commit"] == "0aef241fb088ca21bb4ebd24448940ed56620d17"
        and pi05_result["valid_episode_count"] == 6
        and pi05_result["summary"]
        == {
            "left_successes": 1,
            "right_successes": 3,
            "aligned_endpoint_pair_count": 3,
            "distinct_executed_action_pair_count": 3,
        }
        and pi05_result["registry"]["sha256"] == sha256(pi05_current_registry_path)
        and pi05_result["checkpoint_manifest"]["sha256"] == sha256(pi05_checkpoint_manifest_path),
        "V2-A010 compiled result is hash-bound to the frozen current-stack registry and reports the complete six-cell outcome",
        checks,
    )
    require(
        {(row["environment_seed"], row["requested_relation"]) for row in pi05_result["episodes"]}
        == {(seed, relation) for seed in (8300, 8301, 8302) for relation in ("left", "right")}
        and len(pi05_result["episodes"]) == 6
        and all(
            row["prompt"] == expected_current_prompt("direct_command", row["requested_relation"])
            and row["prompt_family"] == "direct_command"
            and row["sampling_seed_base"] == row["environment_seed"]
            and set(row["files"])
            == {
                "action_trace",
                "environment",
                "episode_log",
                "episode_results",
                "executed_actions",
                "returned_action_chunks",
                "trajectory",
                "viewport_video",
            }
            for row in pi05_result["episodes"]
        )
        and pi05_result["infrastructure_invalid_attempts"] == [],
        "V2-A010 retains six static-prompt episodes with hash-bearing raw PVC references and no invalid attempt in the behavioral result",
        checks,
    )
    require(
        len(pi05_result["pairs"]) == 3
        and [row["environment_seed"] for row in pi05_result["pairs"]] == [8300, 8301, 8302]
        and all(
            row["endpoint_ordering_aligned"] is True
            and row["executed_actions_distinct"] is True
            and row["right_minus_left_endpoint_shift_m"] > 0
            and row["first_ten_executed_action_rms"] > 0
            for row in pi05_result["pairs"]
        ),
        "V2-A010 preserves all three aligned endpoint pairs and distinct executed-action pairs",
        checks,
    )
    require(
        pi05_release_probe["schema_version"] == "vla-wam-v2a010-pi05-current-release-probe-v1"
        and pi05_release_probe["passed"] is True
        and pi05_release_probe["fixture_sha256"]
        == "19dc038dc152aa3408439f2161496efa2d83bc25b11741a6b5495a28113daf86"
        and pi05_release_probe["registry_sha256"] == sha256(pi05_current_registry_path)
        and pi05_release_probe["metrics"]["left_exact_repeat_bit_identical"] is True
        and pi05_release_probe["metrics"]["left_vs_right_action_rms"] > 0
        and all(row["shape"] == [15, 8] for row in pi05_release_probe["records"].values()),
        "V2-A010 release probe passes deterministic-repeat and prompt-sensitivity gates without changing the fixed fixture",
        checks,
    )
    require(
        pi05_fixed_observation["schema_version"] == "vla-wam-v2a010-pi05-current-fixed-observation-v1"
        and pi05_fixed_observation["fixture_sha256"]
        == "19dc038dc152aa3408439f2161496efa2d83bc25b11741a6b5495a28113daf86"
        and pi05_fixed_observation["registry_sha256"] == sha256(pi05_current_registry_path)
        and pi05_fixed_observation["robolab_commit"] == "0aef241fb088ca21bb4ebd24448940ed56620d17"
        and pi05_fixed_observation["neutral_reset_contract"]
        == {"left_predicate_at_reset": False, "right_predicate_at_reset": False},
        "V2-A010 fixed observation records the neutral-reset contract and exact current-stack simulator revision",
        checks,
    )
    require(
        pi05_invalid_attempts["schema_version"]
        == "vla-wam-v2a010-pi05-current-stack-invalid-attempt-ledger-v1"
        and pi05_invalid_attempts["status"]
        == "complete_no_invalid_attempts_or_runtime_interventions"
        and pi05_invalid_attempts["attempts"] == []
        and pi05_invalid_attempts["runtime_interventions"] == []
        and pi05_invalid_attempts["counts"]
        == {
            "infrastructure_invalid_attempt_count": 0,
            "partial_attempt_count": 0,
            "runtime_intervention_count": 0,
            "valid_behavioral_episode_count": 6,
        },
        "V2-A010 keeps its empty invalid-attempt and intervention ledgers outside behavioral denominators",
        checks,
    )
    compact = pi05_provenance["compact_evidence"]
    require(
        pi05_provenance["schema_version"] == "vla-wam-v2a010-pi05-current-stack-provenance-v1"
        and pi05_provenance["status"] == "complete_six_valid_behavioral_cells"
        and pi05_provenance["amendment_id"] == "V2-A010"
        and compact["compiled_result"]["sha256"] == sha256(pi05_result_path)
        and compact["release_probe"]["sha256"] == sha256(pi05_release_probe_path)
        and compact["fixed_observation"]["sha256"] == sha256(pi05_fixed_observation_path)
        and compact["invalid_attempt_ledger"]["sha256"] == sha256(pi05_invalid_attempts_path)
        and compact["selected_media"]["sha256"] == sha256(pi05_media_path),
        "V2-A010 provenance fail-closes the compiled result, gates, ledger, and selected-media hashes",
        checks,
    )
    validate_file_record(workspace, compact["media_builder"], "V2-A010 media builder", checks)
    require(
        pi05_media["schema_version"] == "vla-wam-v2a010-pi05-current-stack-media-v1"
        and pi05_media["status"] == "complete_selected_seed8300_actual_rollout"
        and pi05_media["source_result"]["sha256"] == sha256(pi05_result_path)
        and len(pi05_media["gallery_entries"]) == 1
        and pi05_media["gallery_entries"][0]["seed"] == 8300
        and pi05_media["gallery_entries"][0]["category"] == "VLA"
        and pi05_media["gallery_entries"][0]["future_interface"] == "Actions only; no decoded visual future"
        and "has no imagined-future counterpart" in pi05_media["gallery_entries"][0]["selection_note"],
        "V2-A010 selected media is a bounded seed-8300 VLA actual rollout with no imagined-future claim",
        checks,
    )
    validate_file_record(workspace, pi05_media["publication_video"], "V2-A010 selected actual video", checks)
    validate_file_record(workspace, pi05_media["poster"], "V2-A010 selected actual poster", checks)
    nano_identity = cosmos3_nano_amendment["experiment_identity"]
    nano_grid = cosmos3_nano_amendment["behavioral_grid"]
    require(
        cosmos3_nano_amendment["schema_version"]
        == "vla-wam-shared-v2-post-result-cosmos3-nano-droid-gate-v1"
        and cosmos3_nano_amendment["amendment_id"] == "V2-A011"
        and cosmos3_nano_amendment["status"]
        == "frozen_before_model_load_or_behavioral_inference"
        and nano_identity["model_repository"] == "nvidia/Cosmos3-Nano-Policy-DROID"
        and nano_identity["model_revision"]
        == "6706d7680581c255ff61e0f3bb49d90eac55c79e"
        and nano_identity["framework_repository"]["commit"]
        == "411d25b2e35bc441126f48c44a4b93e1c0564274"
        and nano_identity["simulator_repository"]["commit"]
        == "0aef241fb088ca21bb4ebd24448940ed56620d17"
        and nano_grid["episode_count"] == 6
        and nano_grid["environment_seeds"] == [8300, 8301, 8302]
        and nano_grid["sampling_seeds"] == [8300, 8301, 8302]
        and nano_grid["requested_relations"] == ["left", "right"]
        and nano_grid["static_episode_prompt_only"] is True
        and nano_grid["oracle_or_subtask_coach"] is False
        and nano_grid["dynamic_prompt_switches"] == 0
        and nano_grid["viewport_video_required"] is True
        and nano_grid["executed_action_trace_required"] is True
        and nano_grid["exposed_generated_future_required"] is True
        and len(cosmos3_nano_amendment["release_gates"]) == 9,
        "V2-A011 freezes the exact Cosmos3 Nano identity, six seeded direct cells, and release gates",
        checks,
    )
    nano_checkpoint = cosmos3_nano_registry["checkpoint"]
    nano_files = nano_checkpoint["files"]
    require(
        cosmos3_nano_registry["schema_version"]
        == "vla-wam-shared-v2-cosmos3-nano-policy-droid-registry-v1"
        and cosmos3_nano_registry["status"]
        == "checkpoint_hash_verified_fixed_observation_gate_pending"
        and cosmos3_nano_registry["amendment_id"] == "V2-A011"
        and cosmos3_nano_registry["amendment_sha256"]
        == sha256(cosmos3_nano_amendment_path)
        and cosmos3_nano_registry["protocol_sha256"] == sha256(protocol_path)
        and nano_checkpoint["id"] == "nvidia/Cosmos3-Nano-Policy-DROID"
        and nano_checkpoint["revision"]
        == "6706d7680581c255ff61e0f3bb49d90eac55c79e"
        and nano_checkpoint["download_status"] == "complete_exact_revision_and_hashed"
        and nano_checkpoint["exact_revision_confirmed_by_hf_metadata"] is True
        and nano_checkpoint["hash_gate_passed"] is True
        and nano_checkpoint["present_non_cache_file_count"] == 43
        and len(nano_files) == 43
        and nano_checkpoint["present_non_cache_bytes"] == 32_937_432_846
        and sum(row["bytes"] for row in nano_files.values()) == 32_937_432_846
        and all(len(row["sha256"]) == 64 for row in nano_files.values())
        and nano_checkpoint["model_load_attempt_count"] == 0
        and nano_checkpoint["behavioral_episode_count"] == 0,
        "V2-A011 registry is hash-bound and retains all 43 exact-revision checkpoint payload hashes before model load",
        checks,
    )
    require(
        nano_checkpoint["required_indexed_weight_files"]
        == [
            "transformer/diffusion_pytorch_model-00001-of-00007.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00007.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00007.safetensors",
            "transformer/diffusion_pytorch_model-00004-of-00007.safetensors",
            "transformer/diffusion_pytorch_model-00005-of-00007.safetensors",
            "transformer/diffusion_pytorch_model-00006-of-00007.safetensors",
            "transformer/diffusion_pytorch_model-00007-of-00007.safetensors",
            "vision_encoder/model.safetensors",
        ]
        and set(nano_checkpoint["required_indexed_weight_files"]) <= set(nano_files)
        and set(nano_checkpoint["required_component_files"]) <= set(nano_files)
        and cosmos3_nano_registry["software"]["cosmos_framework"]["commit"]
        == "411d25b2e35bc441126f48c44a4b93e1c0564274"
        and cosmos3_nano_registry["software"]["robolab"]["commit"]
        == "0aef241fb088ca21bb4ebd24448940ed56620d17"
        and cosmos3_nano_registry["serving_contract"]["action_chunk_size"] == 32
        and cosmos3_nano_registry["serving_contract"]["action_dim"] == 8
        and cosmos3_nano_registry["serving_contract"]["action_space"] == "joint_pos"
        and cosmos3_nano_registry["serving_contract"]["decode_video"] is True
        and "technical failure"
        in cosmos3_nano_registry["serving_contract"]["generated_future_contract"],
        "V2-A011 registry freezes complete model components, pinned software, 32x8 actions, and fail-closed futures",
        checks,
    )
    nano_fixed = cosmos3_nano_registry["fixed_observation_gate"]
    nano_queue = cosmos3_nano_registry["behavioral_queue"]
    require(
        nano_fixed["status"] == "released_pending_three_request_gate"
        and nano_fixed["conditions"] == ["left", "left_exact_repeat", "right"]
        and nano_fixed["sampling_seed"] == 8300
        and nano_queue["status"]
        == "frozen_not_released_until_checkpoint_hash_and_fixed_observation_gates_pass"
        and nano_queue["environment_seeds"] == [8300, 8301, 8302]
        and nano_queue["sampling_seeds"] == [8300, 8301, 8302]
        and nano_queue["requested_relations"] == ["left", "right"]
        and nano_queue["prompts"]
        == {
            "left": "Put the Rubik's cube to the left of the bowl.",
            "right": "Put the Rubik's cube to the right of the bowl.",
        }
        and nano_queue["prompt_controller"] == "episode_static"
        and nano_queue["behavioral_episode_count"] == 6
        and nano_queue["viewport_video_required"] is True
        and nano_queue["executed_action_trace_required"] is True
        and nano_queue["exposed_generated_future_required"] is True
        and nano_queue["valid_failure_retention"] is True
        and nano_queue["infrastructure_failure_denominator_policy"]
        == "exclude_and_ledger"
        and nano_queue["oracle_or_subtask_coach"] is False
        and "never pooled"
        in cosmos3_nano_registry["claim_boundary"]["checkpoint_identity"]
        and "immutable do-not-rerun evidence"
        in cosmos3_nano_registry["claim_boundary"]["cosmos3_edge"],
        "V2-A011 releases only the fixed gate and freezes six static video/action/future cells while preserving Edge",
        checks,
    )
    nano_result = load_json(cosmos3_nano_result_path)
    nano_fixed_result = load_json(cosmos3_nano_fixed_path)
    nano_invalid = load_json(cosmos3_nano_invalid_path)
    nano_runtime = load_json(cosmos3_nano_runtime_path)
    nano_layout = load_json(cosmos3_nano_layout_path)
    nano_provenance = load_json(cosmos3_nano_provenance_path)
    nano_media = load_json(cosmos3_nano_media_path)
    require(
        nano_result["schema_version"] == "vla-wam-shared-v2-cosmos3-nano-policy-droid-result-v1"
        and nano_result["status"] == "complete"
        and nano_result["amendment_id"] == "V2-A011"
        and nano_result["checkpoint_revision"] == nano_identity["model_revision"]
        and nano_result["summary"]["episode_count"] == 6
        and nano_result["summary"]["successes"] == 6
        and nano_result["summary"]["by_direction"] == {
            "left": {"episodes": 3, "successes": 3, "verified_pickups": 3, "entered_requested_region": 3},
            "right": {"episodes": 3, "successes": 3, "verified_pickups": 3, "entered_requested_region": 3},
        }
        and nano_result["summary"]["aligned_endpoint_pairs"] == 3
        and nano_result["summary"]["nonzero_first_chunk_pairs"] == 3
        and len(nano_result["episodes"]) == 6
        and sum(item["decoded_future_count"] for item in nano_result["episodes"]) == 37
        and all(pair["executed_actions_distinct"] for pair in nano_result["pairs"])
        and all(pair["endpoint_response_direction"] == "aligned" for pair in nano_result["pairs"]),
        "V2-A011 complete result retains six successful valid cells, distinct actions, aligned endpoints, and all 37 exposed futures",
        checks,
    )
    require(
        nano_fixed_result["status"] == "passed"
        and nano_fixed_result["metrics"] == {
            "left_repeat_action_rms": 0.0,
            "left_repeat_future_pixel_mae": 0.0,
            "left_right_action_rms": 0.025650289164499847,
            "left_right_future_pixel_mae": 7.465912737698959,
        }
        and len(nano_fixed_result["records"]) == 3,
        "V2-A011 fixed-observation diagnostic is repeatable and prompt-sensitive before behavior",
        checks,
    )
    require(
        nano_invalid["counts"] == {
            "infrastructure_invalid_attempt_count": 2,
            "partial_attempt_count": 0,
            "runtime_intervention_count": 0,
            "valid_behavioral_episode_count": 6,
        }
        and all(item["request_started"] is False for item in nano_invalid["attempts"])
        and nano_runtime["status"] == "complete_no_runtime_interventions"
        and nano_runtime["events"] == []
        and len(nano_runtime["guarded_pairs"]) == 3,
        "V2-A011 preserves two pre-request infrastructure attempts outside denominators and no runtime intervention",
        checks,
    )
    require(
        nano_layout["status"] == "measurement_only_layout_compatibility"
        and nano_layout["behavioral_data_modified"] is False
        and nano_layout["episodes_rerun_by_resolution"] == 0
        and nano_layout["model_requests_started_by_resolution"] == 0
        and len(nano_layout["records"]) == 6
        and all(item["copied_bytes"] == 0 and item["target_resolves_to_source"] for item in nano_layout["records"]),
        "V2-A011 trace-layout compatibility is a six-link zero-copy measurement event",
        checks,
    )
    require(
        nano_provenance["status"] == "complete_compact_evidence_slice"
        and nano_provenance["compact_evidence"]["compiled_result"]["sha256"] == sha256(cosmos3_nano_result_path)
        and nano_provenance["compact_evidence"]["fixed_observation_gate"]["sha256"] == sha256(cosmos3_nano_fixed_path)
        and nano_provenance["compact_evidence"]["invalid_attempt_ledger"]["sha256"] == sha256(cosmos3_nano_invalid_path)
        and nano_provenance["compact_evidence"]["runtime_intervention_ledger"]["sha256"] == sha256(cosmos3_nano_runtime_path)
        and nano_provenance["compact_evidence"]["raw_layout_compatibility_event"]["sha256"] == sha256(cosmos3_nano_layout_path)
        and nano_provenance["compact_evidence"]["selected_media"]["sha256"] == sha256(cosmos3_nano_media_path),
        "V2-A011 provenance binds all compact result, gate, ledger, layout, and media evidence",
        checks,
    )
    require(
        nano_media["status"] == "complete_selected_seed8300_actual_and_prediction_pair"
        and len(nano_media["gallery_entries"]) == 1
        and nano_media["gallery_entries"][0]["comparison_media"]["kind"] == "model_prediction_not_execution"
        and nano_media["prediction_frame_count"] == 33
        and nano_media["actual_frame_count"] == 332,
        "V2-A011 gallery media separates the selected actual rollout from the same-seed decoded prediction",
        checks,
    )
    for record, label in (
        (nano_media["publication_video"], "V2-A011 actual rollout video"),
        (nano_media["prediction_video"], "V2-A011 model prediction video"),
        (nano_media["poster"], "V2-A011 actual rollout poster"),
    ):
        validate_file_record(workspace, record, label, checks)
    super_identity = cosmos3_super_amendment["experiment_identity"]
    super_grid = cosmos3_super_amendment["conditional_behavioral_grid"]
    require(
        cosmos3_super_amendment["schema_version"]
        == "vla-wam-shared-v2-post-result-cosmos3-super-droid-feasibility-v1"
        and cosmos3_super_amendment["amendment_id"] == "V2-A012"
        and cosmos3_super_amendment["status"]
        == "frozen_pre_inference_feasibility_not_released"
        and super_identity["model_repository"] == "nvidia/Cosmos3-Super"
        and super_identity["model_revision"]
        == "e0262be9d8f7586bc24c069a2aed2b665bdff266"
        and super_identity["parameter_count"] == "64B"
        and super_identity["checkpoint_index_total_bytes"] == 129_230_007_264
        and super_identity["full_snapshot_file_count"] == 88
        and super_identity["full_snapshot_total_bytes"] == 132_710_200_213
        and super_grid["potential_episode_count"] == 6
        and super_grid["released_episode_count"] == 0
        and super_grid["environment_seeds"] == [8300, 8301, 8302]
        and super_grid["sampling_seeds"] == [8300, 8301, 8302]
        and super_grid["requested_relations"] == ["left", "right"]
        and super_grid["static_episode_prompt_only"] is True
        and super_grid["oracle_or_subtask_coach"] is False
        and super_grid["dynamic_prompt_switches"] == 0,
        "V2-A012 freezes Cosmos3-Super base identity and six conditional static DROID cells without releasing behavior",
        checks,
    )
    require(
        cosmos3_super_snapshot["schema_version"]
        == "vla-wam-shared-v2-cosmos3-super-hf-snapshot-v1"
        and cosmos3_super_snapshot["amendment_id"] == "V2-A012"
        and cosmos3_super_snapshot["model_repository"] == "nvidia/Cosmos3-Super"
        and cosmos3_super_snapshot["model_revision"]
        == "e0262be9d8f7586bc24c069a2aed2b665bdff266"
        and cosmos3_super_snapshot["resolved_public"] is True
        and cosmos3_super_snapshot["resolved_gated"] is False
        and cosmos3_super_snapshot["model_index_total_bytes"] == 129_230_007_264
        and cosmos3_super_snapshot["snapshot_file_count"] == 88
        and cosmos3_super_snapshot["snapshot_total_bytes"] == 132_710_200_213
        and cosmos3_super_snapshot["snapshot_lfs_payload_bytes"] == 132_693_725_766
        and len(cosmos3_super_snapshot["files"]) == 88
        and len(cosmos3_super_snapshot["indexed_weight_files"]) == 28
        and sum(
            1
            for record in cosmos3_super_snapshot["files"].values()
            if record["lfs_sha256"]
        )
        == 48
        and all(
            record["bytes"] > 0
            and record["git_blob_oid"]
            and (record["lfs_sha256"] is None or len(record["lfs_sha256"]) == 64)
            for record in cosmos3_super_snapshot["files"].values()
        )
        and cosmos3_super_snapshot["metadata_sha256"]
        == {
            "config.json": "90510e6522fa44b79413076f0cf49b3c1b78241c53cfaf70ea0095bb7ed611a0",
            "model.safetensors.index.json": "cc635ec58da60705cbba5ce92c89d10badb4c460a67131196e5d95895dda229b",
            "model_index.json": "90088a6638deb68418d795971ad48e6d5aa3f7764d69f3c62bbca9e8f3485bcd",
            "transformer/config.json": "7d4ade2f8cc05e8b2b505a0ed6260e8a0fb1027c5ee49bc9f0c854e895af682b",
        },
        "V2-A012 retains a public exact-revision 88-file snapshot plan and hash-bearing model metadata before download",
        checks,
    )
    super_checkpoint = cosmos3_super_registry["checkpoint"]
    super_queue = cosmos3_super_registry["behavioral_queue"]
    super_branches = cosmos3_super_registry["action_execution_branches"]
    require(
        cosmos3_super_registry["schema_version"]
        == "vla-wam-shared-v2-cosmos3-super-droid-registry-v1"
        and cosmos3_super_registry["amendment_id"] == "V2-A012"
        and cosmos3_super_registry["amendment_sha256"]
        == sha256(cosmos3_super_amendment_path)
        and cosmos3_super_registry["protocol_sha256"] == sha256(protocol_path)
        and cosmos3_super_registry["status"]
        == "pre_inference_feasibility_blocked_no_cells_released"
        and super_checkpoint["id"] == "nvidia/Cosmos3-Super"
        and super_checkpoint["revision"]
        == "e0262be9d8f7586bc24c069a2aed2b665bdff266"
        and super_checkpoint["source_snapshot_manifest_sha256"]
        == sha256(cosmos3_super_snapshot_path)
        and super_checkpoint["download_status"] == "not_started"
        and super_checkpoint["hash_gate_passed"] is False
        and super_checkpoint["model_load_attempt_count"] == 0
        and super_queue["released_episode_count"] == 0
        and super_queue["status"] == "unreleased_missing_action_execution_contract"
        and super_queue["prompts"]
        == {
            "left": "Put the Rubik's cube to the left of the bowl.",
            "right": "Put the Rubik's cube to the right of the bowl.",
        }
        and super_queue["viewport_video_required"] is True
        and super_queue["executed_action_trace_required"] is True
        and super_queue["model_returned_action_chunk_required"] is True
        and super_queue["exposed_generated_future_required"] is True,
        "V2-A012 registry keeps the Super checkpoint unloaded and the six-cell queue blocked pending its own action and future gates",
        checks,
    )
    require(
        cosmos3_super_registry["software"]["nvidia_cosmos"]["commit"]
        == "e494d734022ab0610061cdf57fa24c843e18767e"
        and cosmos3_super_registry["software"]["vllm_omni"]["commit"]
        == "900a7f0813d0482811b0e4dfd3cf7deabbe2429f"
        and "mutable" in cosmos3_super_registry["software"]["runtime_provenance_gate"]
        and cosmos3_super_registry["server_topology"]["preferred"]["gpu_requirement"]
        == "4 idle ali-owned B200 GPUs in one pod"
        and cosmos3_super_registry["server_topology"]["preferred"]["memory_limit"] == "256Gi"
        and cosmos3_super_registry["server_topology"]["conditional_fallback"]["gpu_indices"]
        == [1, 2]
        and cosmos3_super_registry["server_topology"]["conditional_fallback"]["tensor_parallel_size"]
        == 2
        and super_branches["direct_interface"]["status"]
        == "blocked_missing_official_super_base_joint_pos_8d_contract"
        and super_branches["derived_control_curobo_ik"]["status"]
        == "blocked_pending_documented_10d_output_and_frozen_controller_contract"
        and "never as native policy execution"
        in super_branches["derived_control_curobo_ik"]["reporting"],
        "V2-A012 pins source topology and keeps direct and separately labeled derived CuRobo control fail-closed",
        checks,
    )
    require(
        cosmos3_super_runbook_path.is_file()
        and cosmos3_super_builder_path.is_file()
        and cosmos3_super_finalizer_path.is_file()
        and cosmos3_super_pod_manifest_path.is_file()
        and cosmos3_super_amendment["claim_boundary"]["current_status"]
        == "No Cosmos3-Super model inference, behavioral rollout, imagined video, or success claim exists yet."
        and "never pool"
        in cosmos3_super_amendment["analysis_branches"]["derived_control_curobo_ik"]["reporting"]
        and "Never substitute imagination for execution"
        in cosmos3_super_amendment["media_contract"]["presentation_rule"],
        "V2-A012 provides the frozen builder, finalizer, runbook, four-B200 manifest, and adjacent actual-versus-imagined media rule",
        checks,
    )
    edge_base_identity = cosmos3_edge_base_amendment["experiment_identity"]
    edge_base_amendment_probe = cosmos3_edge_base_amendment[
        "fixed_observation_feasibility_probe"
    ]
    edge_base_grid = cosmos3_edge_base_amendment["conditional_behavioral_grid"]
    require(
        cosmos3_edge_base_amendment["schema_version"]
        == "vla-wam-shared-v2-post-result-cosmos3-edge-base-feasibility-v1"
        and cosmos3_edge_base_amendment["amendment_id"] == "V2-A013"
        and cosmos3_edge_base_amendment["status"]
        == "frozen_pre_inference_feasibility_not_released"
        and edge_base_identity["model_repository"] == "nvidia/Cosmos3-Edge"
        and edge_base_identity["model_revision"]
        == "ff48d22144de52de296a7b4d3a78914831007212"
        and edge_base_identity["parameter_count_bf16_safetensors"] == 3_858_999_728
        and edge_base_identity["snapshot_file_count"] == 48
        and edge_base_identity["snapshot_total_bytes"] == 9_173_855_122
        and edge_base_identity["snapshot_lfs_payload_bytes"] == 9_173_276_024
        and edge_base_amendment_probe["authorized_request_count"] == 3
        and edge_base_amendment_probe["behavioral_episode_count"] == 0
        and edge_base_amendment_probe["conditions"]
        == ["left", "left_exact_repeat", "right"]
        and edge_base_amendment_probe["sampling_seed"] == 8300
        and edge_base_grid["potential_episode_count"] == 6
        and edge_base_grid["released_episode_count"] == 0
        and edge_base_grid["environment_seeds"] == [8300, 8301, 8302]
        and edge_base_grid["requested_relations"] == ["left", "right"]
        and edge_base_grid["static_episode_prompt_only"] is True
        and edge_base_grid["oracle_or_subtask_coach"] is False
        and edge_base_grid["dynamic_prompt_switches"] == 0,
        "V2-A013 freezes the exact Edge base identity, three feasibility requests, and six unreleased static cells",
        checks,
    )
    edge_base_checkpoint = cosmos3_edge_base_registry["checkpoint"]
    edge_base_files = edge_base_checkpoint["files"]
    require(
        cosmos3_edge_base_registry["schema_version"]
        == "vla-wam-shared-v2-cosmos3-edge-base-v2a013-registry-v1"
        and cosmos3_edge_base_registry["amendment_id"] == "V2-A013"
        and cosmos3_edge_base_registry["status"]
        == "pre_inference_probe_frozen_behavioral_cells_unreleased"
        and cosmos3_edge_base_registry["amendment"]["sha256"]
        == sha256(cosmos3_edge_base_amendment_path)
        and cosmos3_edge_base_registry["protocol"]["sha256"]
        == sha256(protocol_path)
        and edge_base_checkpoint["id"] == "nvidia/Cosmos3-Edge"
        and edge_base_checkpoint["revision"]
        == "ff48d22144de52de296a7b4d3a78914831007212"
        and edge_base_checkpoint["public"] is True
        and edge_base_checkpoint["gated"] is False
        and edge_base_checkpoint["download_status"] == "not_started"
        and edge_base_checkpoint["hash_gate_passed"] is False
        and edge_base_checkpoint["model_load_attempt_count"] == 0
        and edge_base_checkpoint["model_action_request_count"] == 0
        and edge_base_checkpoint["behavioral_episode_count"] == 0
        and edge_base_checkpoint["snapshot_file_count"] == 48
        and edge_base_checkpoint["snapshot_total_bytes"] == 9_173_855_122
        and edge_base_checkpoint["snapshot_lfs_payload_bytes"] == 9_173_276_024
        and edge_base_checkpoint["safetensors_parameter_count"] == 3_858_999_728
        and len(edge_base_files) == 48
        and sum(record["bytes"] for record in edge_base_files.values())
        == 9_173_855_122
        and sum(1 for record in edge_base_files.values() if record["lfs_sha256"])
        == 16
        and sum(
            record["bytes"]
            for record in edge_base_files.values()
            if record["lfs_sha256"]
        )
        == 9_173_276_024
        and all(
            record["bytes"] > 0
            and record["git_blob_oid"]
            and (record["lfs_sha256"] is None or len(record["lfs_sha256"]) == 64)
            for record in edge_base_files.values()
        ),
        "V2-A013 registry hash-binds the public exact-revision 48-file Edge base snapshot before download or load",
        checks,
    )
    edge_base_probe = cosmos3_edge_base_registry["fixed_observation_probe"]
    edge_base_queue = cosmos3_edge_base_registry["behavioral_queue"]
    require(
        edge_base_probe["status"] == "frozen_not_run"
        and edge_base_probe["authorized_request_count"] == 3
        and edge_base_probe["released_request_count"] == 0
        and edge_base_probe["behavioral_episode_count"] == 0
        and edge_base_probe["conditions"]
        == ["left", "left_exact_repeat", "right"]
        and edge_base_probe["sampling_seed"] == 8300
        and edge_base_probe["request_contract"]["num_frames"] == 17
        and edge_base_probe["request_contract"]["extra_params"]
        == {
            "action_chunk_size": 16,
            "action_mode": "policy",
            "domain_name": "droid_lerobot",
            "raw_action_dim": 10,
        }
        and any("shape [16,10]" in item for item in edge_base_probe["pass_requirements"])
        and "Missing future is unavailable evidence, not zero"
        in edge_base_probe["failure_policy"]
        and edge_base_queue["status"] == "six_cells_frozen_zero_released"
        and edge_base_queue["potential_episode_count"] == 6
        and edge_base_queue["released_episode_count"] == 0
        and len(edge_base_queue["cells"]) == 6
        and {
            (cell["environment_seed"], cell["requested_relation"])
            for cell in edge_base_queue["cells"]
        }
        == {(seed, relation) for seed in (8300, 8301, 8302) for relation in ("left", "right")}
        and all(cell["status"] == "frozen_unreleased" for cell in edge_base_queue["cells"])
        and edge_base_queue["static_episode_prompt_only"] is True
        and edge_base_queue["oracle_or_subtask_coach"] is False
        and edge_base_queue["exposed_generated_future_required"] is True,
        "V2-A013 permits only three non-behavioral action-plus-future probes and releases zero of six behavioral cells",
        checks,
    )
    edge_base_interface = cosmos3_edge_base_registry["interface"]
    edge_base_derived = cosmos3_edge_base_registry["derived_control_curobo"]
    edge_base_blocker = edge_base_derived["asset_mismatch_blocker"]
    require(
        edge_base_interface["direct_joint_pos_8d"]["status"]
        == "blocked_no_official_base_to_droid_native_8d_contract"
        and edge_base_interface["generic_droid_10d"]["status"]
        == "source_backed_probe_interface_only_not_execution_released"
        and len(edge_base_interface["generic_droid_10d"]["raw_layout"]) == 10
        and edge_base_interface["generic_droid_10d"]["raw_layout"][-1] == "gripper"
        and "backward_framewise"
        in edge_base_interface["generic_droid_10d"]["pose_convention"]
        and edge_base_derived["status"]
        == "blocked_exact_franka_robotiq_urdf_and_control_frame_mapping_absent"
        and edge_base_blocker["robolab_sim_asset_lfs_sha256"]
        == "f555695465687548a1bd31b5e3f30385182d476a67c17080b7820ad0ef747e41"
        and edge_base_blocker["robolab_control_body"]
        == "Gripper/Robotiq_2F_85/base_link"
        and edge_base_blocker["candidate_curobo_urdf_sha256"]
        == "6a0044e6e72ee667927f17d1871ec3e2615a8bc5fe978882fc909e4094667967"
        and "must not be substituted" in edge_base_blocker["candidate_rejection"],
        "V2-A013 keeps native 8D and CuRobo-derived execution blocked at the exact Robotiq asset and frame boundary",
        checks,
    )
    edge_base_software = cosmos3_edge_base_registry["software"]
    edge_base_preserved = cosmos3_edge_base_registry[
        "preserved_completed_edge_policy"
    ]
    require(
        edge_base_software["nvidia_cosmos_commit"]
        == "e494d734022ab0610061cdf57fa24c843e18767e"
        and edge_base_software["nvidia_cosmos_framework_commit"]
        == "a904d2d36b774a51dd06ff9ff906816b1a04f579"
        and edge_base_software["vllm_omni_commit"]
        == "900a7f0813d0482811b0e4dfd3cf7deabbe2429f"
        and edge_base_software["robolab_commit"]
        == "0aef241fb088ca21bb4ebd24448940ed56620d17"
        and edge_base_software["curobo_commit"]
        == "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
        and edge_base_preserved["checkpoint"] == "nvidia/Cosmos3-Edge-Policy-DROID"
        and edge_base_preserved["do_not_rerun"] is True
        and edge_base_preserved["pool_with_edge_base"] is False
        and edge_base_preserved["valid_behavioral_episode_count"] == 6
        and edge_base_preserved["result"]["sha256"]
        == sha256(cosmos3_edge_policy_result_path)
        and cosmos3_edge_base_runbook_path.is_file()
        and cosmos3_edge_base_builder_path.is_file(),
        "V2-A013 pins official software and preserves the completed native Edge Policy result without rerun or pooling",
        checks,
    )
    validate_file_record(
        workspace,
        cosmos3_edge_base_registry["amendment"],
        "V2-A013 amendment",
        checks,
    )
    validate_file_record(
        workspace,
        cosmos3_edge_base_registry["protocol"],
        "V2-A013 protocol",
        checks,
    )
    validate_file_record(
        workspace,
        edge_base_preserved["result"],
        "V2-A013 preserved Edge Policy result",
        checks,
    )
    for index, record in enumerate(cosmos3_edge_base_registry["local_sources"]):
        validate_file_record(workspace, record, f"V2-A013 local source {index}", checks)

    edge_base_records = {
        record["condition"]: record for record in cosmos3_edge_base_fixed["records"]
    }
    edge_base_left = edge_base_records["left"]
    edge_base_repeat = edge_base_records["left_exact_repeat"]
    edge_base_right = edge_base_records["right"]
    require(
        cosmos3_edge_base_fixed["schema_version"]
        == "vla-wam-shared-v2-cosmos3-edge-base-v2a013-fixed-observation-v1"
        and cosmos3_edge_base_fixed["amendment_id"] == "V2-A013"
        and cosmos3_edge_base_fixed["model_id"] == "cosmos3_edge_base_droid"
        and cosmos3_edge_base_fixed["status"]
        == "fixed_observation_passed_behavior_blocked"
        and cosmos3_edge_base_fixed["behavioral_episode_count"] == 0
        and cosmos3_edge_base_fixed["probe"]["status"] == "passed"
        and cosmos3_edge_base_fixed["probe"]["authorized_request_count"] == 3
        and cosmos3_edge_base_fixed["probe"]["conditions"]
        == ["left", "left_exact_repeat", "right"]
        and len(cosmos3_edge_base_fixed["records"]) == 3
        and set(edge_base_records) == {"left", "left_exact_repeat", "right"}
        and "No simulator action was sent"
        in cosmos3_edge_base_fixed["claim_boundary"]
        and "no behavioral denominator"
        in cosmos3_edge_base_fixed["claim_boundary"],
        "V2-A013 compact result contains exactly three fixed-observation requests and zero behavior",
        checks,
    )
    require(
        cosmos3_edge_base_fixed["checks"]["authorized_request_count_exact"] is True
        and cosmos3_edge_base_fixed["checks"]["left_repeat_action_bit_identical"]
        is True
        and cosmos3_edge_base_fixed["checks"]["left_repeat_video_bit_identical"]
        is True
        and cosmos3_edge_base_fixed["checks"]
        ["left_repeat_multipart_fields_identical"]
        is True
        and edge_base_left["prompt"] == edge_base_repeat["prompt"]
        and edge_base_left["action_npy_sha256"]
        == edge_base_repeat["action_npy_sha256"]
        and edge_base_left["action_storage_sha256"]
        == edge_base_repeat["action_storage_sha256"]
        and edge_base_left["future_mp4_sha256"]
        == edge_base_repeat["future_mp4_sha256"]
        and edge_base_left["future_mp4_bytes"]
        == edge_base_repeat["future_mp4_bytes"]
        and edge_base_left["multipart_fields_sha256"]
        == edge_base_repeat["multipart_fields_sha256"],
        "V2-A013 LEFT exact repeat is bit-identical in request fields, action storage, and decoded future",
        checks,
    )
    require(
        cosmos3_edge_base_fixed["checks"]["left_right_action_different"] is True
        and cosmos3_edge_base_fixed["checks"]["left_right_video_different"] is True
        and edge_base_left["prompt"] != edge_base_right["prompt"]
        and edge_base_left["action_npy_sha256"]
        != edge_base_right["action_npy_sha256"]
        and edge_base_left["action_storage_sha256"]
        != edge_base_right["action_storage_sha256"]
        and edge_base_left["future_mp4_sha256"]
        != edge_base_right["future_mp4_sha256"]
        and edge_base_left["multipart_fields_sha256"]
        != edge_base_right["multipart_fields_sha256"]
        and cosmos3_edge_base_fixed["checks"]["action_contract_all_three"]
        == "torch.bfloat16 [16,10], raw_action_dim=10, domain_id=8, finite"
        and cosmos3_edge_base_fixed["checks"]["video_contract_all_three"]
        == "H.264 640x480, 17 decoded frames, 5 fps",
        "V2-A013 RIGHT differs from LEFT while all three outputs retain the frozen action and future contracts",
        checks,
    )
    require(
        cosmos3_edge_base_fixed["registry_sha256"]
        == sha256(cosmos3_edge_base_registry_path)
        and cosmos3_edge_base_fixed["checkpoint"]["revision"]
        == edge_base_checkpoint["revision"]
        and cosmos3_edge_base_fixed["checkpoint"]["file_count"]
        == edge_base_checkpoint["snapshot_file_count"]
        and cosmos3_edge_base_fixed["checkpoint"]["total_bytes"]
        == edge_base_checkpoint["snapshot_total_bytes"]
        and cosmos3_edge_base_fixed["checkpoint"]["hash_gate"] == "passed"
        and cosmos3_edge_base_fixed["probe"]["request_contract"]
        ["raw_action_dim"]
        == 10
        and cosmos3_edge_base_fixed["probe"]["request_contract"]
        ["action_chunk_size"]
        == 16
        and cosmos3_edge_base_fixed["probe"]["request_contract"]["num_frames"]
        == 17
        and cosmos3_edge_base_fixed["probe"]["request_contract"]["sampling_seed"]
        == 8300,
        "V2-A013 compact result hash-binds the registry and exact checkpoint/request contract",
        checks,
    )

    edge_base_invalid_counts = cosmos3_edge_base_invalid["counts"]
    require(
        cosmos3_edge_base_invalid["schema_version"]
        == "vla-wam-shared-v2-cosmos3-edge-base-v2a013-invalid-attempts-v1"
        and cosmos3_edge_base_invalid["status"]
        == "complete_with_retained_pre_request_infrastructure_attempts"
        and cosmos3_edge_base_invalid["model_id"]
        == cosmos3_edge_base_fixed["model_id"]
        and cosmos3_edge_base_invalid["checkpoint_revision"]
        == cosmos3_edge_base_fixed["checkpoint"]["revision"]
        and len(cosmos3_edge_base_invalid["attempts"])
        == edge_base_invalid_counts["infrastructure_invalid_attempt_count"]
        == 3
        and all(
            attempt["result"] == "infrastructure_invalid"
            and attempt["model_request_started"] is False
            for attempt in cosmos3_edge_base_invalid["attempts"]
        )
        and edge_base_invalid_counts["authorized_fixed_observation_request_count"]
        == 3
        and edge_base_invalid_counts["partial_behavioral_attempt_count"] == 0
        and edge_base_invalid_counts["runtime_model_intervention_count"] == 0
        and edge_base_invalid_counts["valid_behavioral_episode_count"] == 0
        and cosmos3_edge_base_invalid["runtime_interventions"] == []
        and "outside every model and behavioral denominator"
        in cosmos3_edge_base_invalid["denominator_policy"],
        "V2-A013 ledger keeps three pre-request infrastructure attempts and zero behavior outside denominators",
        checks,
    )
    edge_base_capture_events = cosmos3_edge_base_invalid["client_capture_events"]
    require(
        len(edge_base_capture_events) == 1
        and edge_base_capture_events[0]["new_model_request_for_recovery"] is False
        and edge_base_capture_events[0]["model_output_changed"] is False
        and "LEFT was not rerun" in edge_base_capture_events[0]["effect"]
        and len(cosmos3_edge_base_invalid["operator_setup_events"]) == 1
        and all(
            event["model_load_started"] is False
            and event["model_request_started"] is False
            for event in cosmos3_edge_base_invalid["operator_setup_events"]
        ),
        "V2-A013 retains the client-capture and operator events without consuming a fourth request",
        checks,
    )
    edge_base_invalid_raw_records: list[dict[str, Any]] = []
    for attempt in cosmos3_edge_base_invalid["attempts"]:
        if "raw_log" in attempt:
            edge_base_invalid_raw_records.append(attempt["raw_log"])
        else:
            edge_base_invalid_raw_records.extend(attempt["raw_logs"])
    require(
        len(edge_base_invalid_raw_records) == 4
        and all(
            record["bytes"] > 0
            and len(record["sha256"]) == 64
            and record["path"].startswith(
                "/data/users/ali/vla_wam/raw/cosmos3_edge_base/v2_a013/"
            )
            for record in edge_base_invalid_raw_records
        ),
        "V2-A013 infrastructure ledger preserves four bounded raw-log hashes on the ali PVC",
        checks,
    )

    edge_base_mapping = cosmos3_edge_base_fixed["mapping_gate"]
    edge_base_audit_sources = cosmos3_edge_base_curobo_audit["source_provenance"]
    edge_base_frame = cosmos3_edge_base_curobo_audit[
        "zero_state_frame_comparison"
    ]
    edge_base_parser_decision = cosmos3_edge_base_curobo_audit[
        "parser_and_collision_decision"
    ]
    require(
        cosmos3_edge_base_curobo_audit["artifact"]
        == "cosmos3_edge_base_v2a013_curobo_usd_audit"
        and cosmos3_edge_base_curobo_audit["status"] == "blocked_no_behavior"
        and cosmos3_edge_base_curobo_audit["scope"]
        == "Static source-and-USD audit only; no Isaac startup, EULA acceptance, GPU initialization, controller construction, simulator action, or behavioral denominator."
        and edge_base_mapping["audit_artifact"]
        == str(cosmos3_edge_base_curobo_audit_path.relative_to(workspace))
        and edge_base_mapping["audit_sha256"]
        == sha256(cosmos3_edge_base_curobo_audit_path)
        and edge_base_mapping["behavior_released"] is False
        and edge_base_mapping["status"]
        == "blocked_missing_exact_franka_robotiq_urdf_collision_and_control_frame_parity",
        "V2-A013 fixed result hash-binds a static CuRobo audit that releases no behavior",
        checks,
    )
    require(
        edge_base_audit_sources["robolab_commit"]
        == cosmos3_edge_base_fixed["source_commits"]["robolab"]
        == edge_base_software["robolab_commit"]
        and edge_base_audit_sources["curobo_commit"]
        == cosmos3_edge_base_fixed["source_commits"]["curobo"]
        == edge_base_software["curobo_commit"]
        and edge_base_audit_sources["usd_sha256"]
        == edge_base_blocker["robolab_sim_asset_lfs_sha256"]
        and edge_base_audit_sources["usd_bytes"] == 14_156_362
        and cosmos3_edge_base_curobo_audit["usd_facts"]["control_body_path"]
        == "/panda/Gripper/Robotiq_2F_85/base_link"
        and cosmos3_edge_base_curobo_audit["usd_facts"]
        ["collision_prim_count_robot_only"]
        == 20
        and edge_base_frame["translation_mismatch_m"]
        == 0.26441704182710424
        and edge_base_frame["rotation_mismatch_degrees"]
        == 89.99734223411144
        and edge_base_frame["result"]
        == "The direct USD parser is frame-wrong at zero state; it is not eligible for controller use."
        and "no collision meshes or spheres"
        in edge_base_parser_decision["collision"]
        and "No 10D-to-8D CuRobo action is authorized"
        in edge_base_parser_decision["decision"]
        and "no Panda" in edge_base_parser_decision["decision"]
        and len(cosmos3_edge_base_curobo_audit["future_release_requirements"])
        == 5,
        "V2-A013 CuRobo audit proves the exact frame/collision mismatch and forbids substitute execution",
        checks,
    )

    edge_base_provenance_tools = cosmos3_edge_base_provenance["tools"]
    edge_base_probe_client_path = (
        workspace / edge_base_provenance_tools["probe_client"]["path"]
    )
    edge_base_media_builder_path = (
        workspace / edge_base_provenance_tools["media_builder"]["path"]
    )
    require(
        cosmos3_edge_base_provenance["schema_version"]
        == "vla-wam-shared-v2-cosmos3-edge-base-v2a013-provenance-v1"
        and cosmos3_edge_base_provenance["status"]
        == "complete_fixed_probe_passed_behavior_blocked"
        and cosmos3_edge_base_provenance["amendment"]["path"]
        == str(cosmos3_edge_base_amendment_path.relative_to(workspace))
        and cosmos3_edge_base_provenance["amendment"]["sha256"]
        == sha256(cosmos3_edge_base_amendment_path)
        and cosmos3_edge_base_provenance["registry"]["path"]
        == str(cosmos3_edge_base_registry_path.relative_to(workspace))
        and cosmos3_edge_base_provenance["registry"]["sha256"]
        == sha256(cosmos3_edge_base_registry_path)
        and cosmos3_edge_base_provenance["runtime"]
        ["fixed_observation_request_count"]
        == 3
        and cosmos3_edge_base_provenance["runtime"]["behavioral_episode_count"]
        == 0
        and cosmos3_edge_base_provenance["checkpoint"]["revision"]
        == cosmos3_edge_base_fixed["checkpoint"]["revision"]
        and cosmos3_edge_base_provenance["source_commits"]
        == cosmos3_edge_base_fixed["source_commits"]
        and cosmos3_edge_base_provenance["raw_evidence"]["manifest_sha256"]
        == cosmos3_edge_base_fixed["probe"]["manifest"]["sha256"]
        and cosmos3_edge_base_provenance["raw_evidence"]["server_log_sha256"]
        == cosmos3_edge_base_fixed["server"]["log_sha256"]
        and cosmos3_edge_base_provenance["preflight"]
        ["successful_runtime_home_override"]
        is False
        and "no action reached a simulator"
        in cosmos3_edge_base_provenance["claim_boundary"],
        "V2-A013 provenance hash-links the amendment, registry, raw result, sources, and zero-behavior boundary",
        checks,
    )
    require(
        edge_base_probe_client_path.is_file()
        and sha256(edge_base_probe_client_path)
        == edge_base_provenance_tools["probe_client"]["sha256"]
        and edge_base_media_builder_path.is_file()
        and sha256(edge_base_media_builder_path)
        == edge_base_provenance_tools["media_builder"]["sha256"],
        "V2-A013 provenance hash-binds the exact probe and publication-media builders",
        checks,
    )

    edge_base_media_entry = cosmos3_edge_base_media["entries"][0]
    edge_base_media_builder_text = edge_base_media_builder_path.read_text()
    require(
        cosmos3_edge_base_media["schema_version"]
        == "vla-wam-shared-v2-cosmos3-edge-base-v2a013-media-v1"
        and cosmos3_edge_base_media["status"]
        == "complete_model_prediction_only"
        and cosmos3_edge_base_media["amendment_id"] == "V2-A013"
        and cosmos3_edge_base_media["model_id"]
        == cosmos3_edge_base_fixed["model_id"]
        and cosmos3_edge_base_media["fixed_observation_result"]
        == str(cosmos3_edge_base_fixed_path.relative_to(workspace))
        and "model prediction only"
        in cosmos3_edge_base_media["claim_boundary"].lower()
        and "no simulator rollout"
        in cosmos3_edge_base_media["claim_boundary"].lower()
        and len(cosmos3_edge_base_media["entries"]) == 1
        and edge_base_media_entry["type"] == "side_by_side_model_prediction"
        and edge_base_media_entry["conditions"] == ["left", "right"]
        and edge_base_media_entry["sampling_seed"] == 8300
        and edge_base_media_entry["actual_rollout"] is None
        and edge_base_media_entry["actual_rollout_unavailable_reason"]
        == "behavior_blocked_exact_franka_robotiq_mapping_not_verified",
        "V2-A013 media manifest labels prediction-only evidence and keeps actual rollout null",
        checks,
    )
    require(
        edge_base_media_entry["source_predictions"]["left_mp4_sha256"]
        == edge_base_left["future_mp4_sha256"]
        and edge_base_media_entry["source_predictions"]["right_mp4_sha256"]
        == edge_base_right["future_mp4_sha256"]
        and edge_base_media_entry["model_prediction"]["frames"] == 17
        and edge_base_media_entry["model_prediction"]["width"] == 1280
        and edge_base_media_entry["model_prediction"]["height"] == 534
        and "model_prediction"
        in edge_base_media_entry["model_prediction"]["path"]
        and "model_prediction_poster"
        in edge_base_media_entry["poster"]["path"]
        and "LEFT: cube left of bowl" in edge_base_media_builder_text
        and "RIGHT: cube right of bowl" in edge_base_media_builder_text
        and "MODEL PREDICTION ONLY | NO SIMULATOR ROLLOUT"
        in edge_base_media_builder_text,
        "V2-A013 paired media hash-links LEFT/RIGHT futures and retains explicit non-rollout labels",
        checks,
    )
    validate_file_record(
        workspace,
        edge_base_media_entry["model_prediction"],
        "V2-A013 paired model-prediction video",
        checks,
    )
    validate_file_record(
        workspace,
        edge_base_media_entry["poster"],
        "V2-A013 paired model-prediction poster",
        checks,
    )

    super_runtime_claim = cosmos3_super_runtime_gate["claim_boundary"]
    super_runtime_checkpoint = cosmos3_super_runtime_gate[
        "checkpoint_verification"
    ]
    super_runtime_attempts = cosmos3_super_runtime_gate["attempts"]
    require(
        cosmos3_super_runtime_gate["schema_version"]
        == "vla-wam-shared-v2-cosmos3-super-v2a012-runtime-gate-v1"
        and cosmos3_super_runtime_gate["amendment_id"] == "V2-A012"
        and cosmos3_super_runtime_gate["status"]
        == "pre_request_infrastructure_blocked_64gib_cgroup_and_pending_256gib_pods"
        and super_runtime_claim["model_request_count"] == 0
        and super_runtime_claim["behavioral_episode_count"] == 0
        and super_runtime_claim["behavioral_video_count"] == 0
        and super_runtime_claim["imagined_future_video_count"] == 0
        and super_runtime_claim["behavioral_release_status"] == "locked"
        and "outside every model denominator"
        in super_runtime_claim["denominator_policy"],
        "V2-A012 runtime gate preserves two pre-request infrastructure attempts and zero model or behavior evidence",
        checks,
    )
    require(
        super_runtime_checkpoint["model"] == super_checkpoint["id"]
        and super_runtime_checkpoint["revision"] == super_checkpoint["revision"]
        and super_runtime_checkpoint["file_count"]
        == cosmos3_super_snapshot["snapshot_file_count"]
        and super_runtime_checkpoint["total_bytes"]
        == cosmos3_super_snapshot["snapshot_total_bytes"]
        and super_runtime_checkpoint["hash_gate_passed"] is True
        and super_runtime_checkpoint["verified_registry"]["bytes"] > 0
        and len(super_runtime_checkpoint["verified_registry"]["sha256"]) == 64
        and cosmos3_super_runtime_gate["software"]["nvidia_cosmos"]["commit"]
        == cosmos3_super_registry["software"]["nvidia_cosmos"]["commit"]
        and cosmos3_super_runtime_gate["software"]["vllm_omni"]["commit"]
        == cosmos3_super_registry["software"]["vllm_omni"]["commit"]
        and cosmos3_super_runtime_gate["software"]["environment"]
        ["isolated_no_system_site_packages"]
        is True
        and cosmos3_super_runtime_gate["software"]["environment"]["pip_check"]
        == "clean",
        "V2-A012 runtime gate advances the exact Super checkpoint from snapshot plan to verified PVC bytes",
        checks,
    )
    require(
        len(super_runtime_attempts) == 2
        and all(attempt["model_request_count"] == 0 for attempt in super_runtime_attempts)
        and super_runtime_attempts[0]["status"]
        == "terminated_pre_request_for_known_runtime_storage_prerequisite"
        and super_runtime_attempts[1]["status"]
        == "model_load_failed_before_listener"
        and super_runtime_attempts[1]["failure_class"]
        == "infrastructure_host_memory"
        and super_runtime_attempts[1]["rank_0_exit_code"] == -9
        and cosmos3_super_runtime_gate["topology"]["namespace"] == "211247-prod"
        and cosmos3_super_runtime_gate["topology"]["fallback_pod"]
        == "lerobot-b200-4gpu-1-ali"
        and cosmos3_super_runtime_gate["topology"]
        ["container_memory_limit_bytes"]
        == 68_719_476_736
        and cosmos3_super_runtime_gate["topology"]["selected_gpu_indices"]
        == [1, 2]
        and len(cosmos3_super_runtime_gate["pending_capacity"]) == 2
        and all(
            item["pod"].endswith("-ali") and item["phase"] == "Pending"
            for item in cosmos3_super_runtime_gate["pending_capacity"]
        )
        and "Do not retry this 64Gi fallback"
        in cosmos3_super_runtime_gate["next_step"]["gate"],
        "V2-A012 runtime gate records the 64Gi host-memory block and waits only for named ali-owned 256Gi pods",
        checks,
    )
    super_runtime_raw_records = [
        attempt[key]
        for attempt in super_runtime_attempts
        for key in ("stdout", "thermal_guard")
    ]
    require(
        all(
            record["bytes"] > 0
            and len(record["sha256"]) == 64
            and record["path"].startswith(
                "/data/users/ali/vla_wam/raw/cosmos3_super_droid/v2_a012/"
            )
            for record in super_runtime_raw_records
        ),
        "V2-A012 runtime gate preserves bounded raw stdout and thermal-guard hashes for both attempts",
        checks,
    )

    super_image_input = cosmos3_super_image_only_amendment[
        "frozen_input_and_requests"
    ]
    super_image_requests = super_image_input["requests"]
    require(
        cosmos3_super_image_only_amendment["schema_version"]
        == "vla-wam-shared-v2-post-result-cosmos3-super-image-only-interface-diagnostic-v1"
        and cosmos3_super_image_only_amendment["amendment_id"] == "V2-A014"
        and cosmos3_super_image_only_amendment["status"]
        == "frozen_pre_inference_nonbehavioral_probe_only"
        and cosmos3_super_image_only_amendment["amends"]["v2_a012_amendment"]
        == str(cosmos3_super_amendment_path.relative_to(workspace))
        and cosmos3_super_image_only_amendment["amends"]["v2_a012_registry"]
        == str(cosmos3_super_registry_path.relative_to(workspace))
        and cosmos3_super_image_only_amendment["experiment_identity"]
        ["model_revision"]
        == super_checkpoint["revision"]
        and cosmos3_super_image_only_amendment["experiment_identity"]
        ["not_a_droid_policy_claim"]
        is True
        and cosmos3_super_image_only_amendment["experiment_identity"]
        ["not_a_behavioral_result"]
        is True
        and super_image_input["parameter_provenance"]["sha256"]
        == sha256(cosmos3_edge_base_amendment_path)
        and super_image_input["fixed_rgb"]["raw_rgb_sha256"]
        == cosmos3_edge_base_fixed["conditioning"]["decoded_rgb_sha256"]
        and super_image_input["fixed_rgb"]["transport_file_sha256"]
        == cosmos3_edge_base_fixed["conditioning"]["png_sha256"]
        and super_image_input["robot_state"]["present"] is False,
        "V2-A014 hash-binds the Super identity and exact V2-A013 image-only diagnostic input without robot state",
        checks,
    )
    require(
        len(super_image_requests) == 3
        and [request["id"] for request in super_image_requests]
        == ["V2-A014-P00", "V2-A014-P01", "V2-A014-P02"]
        and [request["condition"] for request in super_image_requests]
        == ["LEFT", "LEFT_exact_repeat", "RIGHT"]
        and all(request["sampling_seed"] == 8300 for request in super_image_requests)
        and super_image_requests[0]["prompt"] == super_image_requests[1]["prompt"]
        and super_image_requests[0]["prompt"] != super_image_requests[2]["prompt"]
        and super_image_requests[1]["byte_identical_to"]
        == "V2-A014-P00 except request identifier and transport timestamps"
        and super_image_input["request_fields"]["num_frames"] == 17
        and super_image_input["request_fields"]["extra_params"]
        == {
            "action_mode": "policy",
            "domain_name": "droid_lerobot",
            "raw_action_dim": 10,
            "action_chunk_size": 16,
        }
        and cosmos3_super_image_only_amendment["acceptance_and_retention"]
        ["shape_and_media_checks"]["action_shape"]
        == [16, 10]
        and cosmos3_super_image_only_amendment["acceptance_and_retention"]
        ["shape_and_media_checks"]["video_frame_count"]
        == 17
        and "bit-identical"
        in cosmos3_super_image_only_amendment["acceptance_and_retention"]
        ["shape_and_media_checks"]["repeat_requirement"],
        "V2-A014 freezes exactly LEFT, identical LEFT repeat, and RIGHT under the inherited three-request budget",
        checks,
    )
    require(
        cosmos3_super_image_only_amendment["acceptance_and_retention"]
        ["interface_evidence_only"]
        is True
        and any(
            item.startswith("No simulator launch")
            for item in cosmos3_super_image_only_amendment["hard_prohibitions"]
        )
        and any(
            "No request may exceed" in item
            for item in cosmos3_super_image_only_amendment["hard_prohibitions"]
        )
        and "No Cosmos3-Super inference"
        in cosmos3_super_image_only_amendment["claim_boundary"]["current_status"]
        and super_runtime_claim["model_request_count"] == 0
        and super_runtime_claim["behavioral_episode_count"] == 0,
        "V2-A014 amendment was frozen before inference and authorizes no simulator, controller, or behavioral denominator",
        checks,
    )

    super_image_overlay_contract = cosmos3_super_image_only_overlay[
        "diagnostic_contract"
    ]
    super_image_overlay_media = cosmos3_super_image_only_overlay[
        "denominator_and_media_rules"
    ]
    require(
        cosmos3_super_image_only_overlay["schema_version"]
        == "vla-wam-shared-v2-cosmos3-super-image-only-interface-overlay-v1"
        and cosmos3_super_image_only_overlay["amendment_id"] == "V2-A014"
        and cosmos3_super_image_only_overlay["overlay_status"]
        == "frozen_pre_inference_no_behavior"
        and cosmos3_super_image_only_overlay["amendment_path"]
        == str(cosmos3_super_image_only_amendment_path.relative_to(workspace))
        and cosmos3_super_image_only_overlay["base_registry_path"]
        == str(cosmos3_super_registry_path.relative_to(workspace))
        and super_image_overlay_contract["requests"]
        == ["LEFT", "LEFT_exact_repeat", "RIGHT"]
        and super_image_overlay_contract["params"]["model"]
        == super_image_input["request_fields"]["model"]
        and super_image_overlay_contract["params"]["raw_action_dim"] == 10
        and super_image_overlay_contract["params"]["action_chunk_size"] == 16
        and "no robot state" in super_image_overlay_contract["input"]
        and super_image_overlay_contract["state_bearing_route"].startswith(
            "prohibited"
        )
        and super_image_overlay_media["behavioral_cells_released"] == 0
        and super_image_overlay_media["behavioral_denominator"] == "none"
        and super_image_overlay_media["simulator_media"] == "none"
        and super_image_overlay_media["permitted_media_label_if_generated"]
        == "IMAGE-ONLY MODEL-GENERATED INTERFACE VIDEO — NOT EXECUTION"
        and "do not invoke simulator or controller"
        in cosmos3_super_image_only_overlay[
            "next_command_after_all_existing_v2a012_gates_pass"
        ],
        "V2-A014 overlay preserves zero behavior and the explicit image-only non-execution media label",
        checks,
    )

    super_image_result_records = {
        record["condition"]: record
        for record in cosmos3_super_image_only_result["records"]
    }
    super_image_left = super_image_result_records["LEFT"]
    super_image_repeat = super_image_result_records["LEFT_exact_repeat"]
    super_image_right = super_image_result_records["RIGHT"]
    require(
        cosmos3_super_image_only_result["schema_version"]
        == "vla-wam-shared-v2-cosmos3-super-image-only-v2a014-result-v1"
        and cosmos3_super_image_only_result["amendment_id"] == "V2-A014"
        and cosmos3_super_image_only_result["status"]
        == "passed_image_only_action_and_video_interface"
        and cosmos3_super_image_only_result["model_id"]
        == "cosmos3_super_base_image_only_interface_diagnostic"
        and cosmos3_super_image_only_result["behavioral_episode_count"] == 0
        and cosmos3_super_image_only_result["robot_state_present"] is False
        and cosmos3_super_image_only_result["simulator_or_controller_invoked"]
        is False
        and cosmos3_super_image_only_result["probe"]["status"] == "passed"
        and cosmos3_super_image_only_result["probe"]["authorized_request_count"]
        == 3
        and len(cosmos3_super_image_only_result["records"]) == 3
        and set(super_image_result_records)
        == {"LEFT", "LEFT_exact_repeat", "RIGHT"}
        and "No robot state, simulator, controller"
        in cosmos3_super_image_only_result["claim_boundary"]
        and "behavioral denominator"
        in cosmos3_super_image_only_result["claim_boundary"],
        "V2-A014 result contains exactly three image-only requests and zero robot-state or behavioral execution",
        checks,
    )
    super_result_checks = cosmos3_super_image_only_result["checks"]
    require(
        super_result_checks
        == {
            "all_actions_finite_16x10": True,
            "all_videos_decode_to_17_frames": True,
            "authorized_request_count_exactly_three": True,
            "left_repeat_action_bit_identical": True,
            "left_repeat_action_payload_identical": True,
            "left_repeat_decoded_video_identical": True,
            "left_repeat_request_body_identical": True,
            "left_right_action_different": True,
            "left_right_decoded_video_different": True,
        }
        and [record["request_id"] for record in cosmos3_super_image_only_result["records"]]
        == ["V2-A014-P00", "V2-A014-P01", "V2-A014-P02"]
        and super_image_left["prompt"] == super_image_repeat["prompt"]
        and super_image_left["request_body_sha256"]
        == super_image_repeat["request_body_sha256"]
        and super_image_left["action"] == super_image_repeat["action"]
        and {
            key: value
            for key, value in super_image_left["video"].items()
        }
        == super_image_repeat["video"],
        "V2-A014 LEFT repeat is bit-identical in request body, action payload/storage, and decoded video",
        checks,
    )
    require(
        super_image_left["prompt"] != super_image_right["prompt"]
        and super_image_left["request_body_sha256"]
        != super_image_right["request_body_sha256"]
        and super_image_left["action"]["data_sha256"]
        != super_image_right["action"]["data_sha256"]
        and super_image_left["action"]["payload_sha256"]
        != super_image_right["action"]["payload_sha256"]
        and super_image_left["action"]["npy_sha256"]
        != super_image_right["action"]["npy_sha256"]
        and super_image_left["video"]["decoded_rgb_sha256"]
        != super_image_right["video"]["decoded_rgb_sha256"]
        and super_image_left["video"]["mp4_sha256"]
        != super_image_right["video"]["mp4_sha256"],
        "V2-A014 RIGHT differs from LEFT in request, returned bfloat16 action, and decoded future",
        checks,
    )
    require(
        all(
            record["action"]["shape"] == [16, 10]
            and record["action"]["dtype"] == "torch.bfloat16"
            and record["action"]["storage_dtype"]
            == "uint16_bfloat16_bits"
            and record["action"]["raw_action_dim"] == 10
            and record["action"]["domain_id"] == 8
            and record["action"]["mode"] == "policy"
            and record["action"]["finite"] is True
            and record["video"]["codec"] == "h264"
            and record["video"]["decoded_frame_count"] == 17
            and record["video"]["decoded_rgb_bytes"] == 15_667_200
            and record["video"]["width"] == 640
            and record["video"]["height"] == 480
            and record["video"]["fps"] == 5
            for record in cosmos3_super_image_only_result["records"]
        )
        and cosmos3_super_image_only_result["probe"]["request_contract"]
        == {
            "action_chunk_size": 16,
            "action_mode": "policy",
            "domain_name": "droid_lerobot",
            "flow_shift": 5.0,
            "fps": 5,
            "guidance_scale": 1.0,
            "num_frames": 17,
            "num_inference_steps": 30,
            "raw_action_dim": 10,
            "sampling_seed": 8300,
            "size": "640x480",
        },
        "V2-A014 all three responses are finite bfloat16 [16,10] actions with 17-frame H.264 futures",
        checks,
    )
    require(
        cosmos3_super_image_only_result["registry_overlay"]["path"]
        == str(cosmos3_super_image_only_overlay_path.relative_to(workspace))
        and cosmos3_super_image_only_result["registry_overlay"]["sha256"]
        == sha256(cosmos3_super_image_only_overlay_path)
        and cosmos3_super_image_only_result["conditioning"]["decoded_rgb_sha256"]
        == super_image_input["fixed_rgb"]["raw_rgb_sha256"]
        and cosmos3_super_image_only_result["conditioning"]["png_sha256"]
        == super_image_input["fixed_rgb"]["transport_file_sha256"]
        and cosmos3_super_image_only_result["checkpoint"]["repository"]
        == "nvidia/Cosmos3-Super"
        and cosmos3_super_image_only_result["checkpoint"]["revision"]
        == super_checkpoint["revision"]
        and cosmos3_super_image_only_result["checkpoint"]["file_count"]
        == super_runtime_checkpoint["file_count"]
        and cosmos3_super_image_only_result["checkpoint"]["total_bytes"]
        == super_runtime_checkpoint["total_bytes"]
        and cosmos3_super_image_only_result["checkpoint"]["hash_gate"]
        == "passed"
        and cosmos3_super_image_only_result["checkpoint"]["verified_registry"]
        ["sha256"]
        == super_runtime_checkpoint["verified_registry"]["sha256"]
        and cosmos3_super_image_only_result["source_commits"]
        == {
            "cosmos": cosmos3_super_registry["software"]["nvidia_cosmos"]["commit"],
            "vllm_omni": cosmos3_super_registry["software"]["vllm_omni"]["commit"],
        },
        "V2-A014 result hash-links the frozen overlay, conditioning image, verified Super checkpoint, and sources",
        checks,
    )

    super_image_invalid_counts = cosmos3_super_image_only_invalid["counts"]
    require(
        cosmos3_super_image_only_invalid["schema_version"]
        == "vla-wam-shared-v2-cosmos3-super-image-only-v2a014-invalid-attempts-v1"
        and cosmos3_super_image_only_invalid["status"]
        == "complete_with_two_pre_request_infrastructure_attempts"
        and cosmos3_super_image_only_invalid["model_id"]
        == cosmos3_super_image_only_result["model_id"]
        and cosmos3_super_image_only_invalid["checkpoint_revision"]
        == cosmos3_super_image_only_result["checkpoint"]["revision"]
        and len(cosmos3_super_image_only_invalid["attempts"])
        == super_image_invalid_counts["infrastructure_invalid_attempt_count"]
        == 2
        and all(
            attempt["model_request_count"] == 0
            and attempt["result"] == "infrastructure_invalid"
            for attempt in cosmos3_super_image_only_invalid["attempts"]
        )
        and super_image_invalid_counts["authorized_v2_a014_request_count"] == 3
        and super_image_invalid_counts["valid_image_only_interface_request_count"]
        == 3
        and super_image_invalid_counts["partial_behavioral_attempt_count"] == 0
        and super_image_invalid_counts["runtime_model_intervention_count"] == 0
        and super_image_invalid_counts["valid_behavioral_episode_count"] == 0
        and cosmos3_super_image_only_invalid["runtime_interventions"] == []
        and "outside all model and behavioral denominators"
        in cosmos3_super_image_only_invalid["denominator_policy"],
        "V2-A014 ledger keeps two pre-request infrastructure attempts outside all denominators",
        checks,
    )
    validate_file_record(
        workspace,
        cosmos3_super_image_only_invalid["source_runtime_gate"],
        "V2-A014 source V2-A012 runtime gate",
        checks,
    )
    super_invalid_raw_records = [
        record
        for attempt in cosmos3_super_image_only_invalid["attempts"]
        for record in attempt["raw_logs"].values()
    ]
    require(
        len(super_invalid_raw_records) == 4
        and {
            record["sha256"] for record in super_invalid_raw_records
        }
        == {
            record["sha256"] for record in super_runtime_raw_records
        }
        and all(record["bytes"] > 0 for record in super_invalid_raw_records)
        and len(cosmos3_super_image_only_invalid["scheduler_events"]) == 2
        and all(
            event["model_load_started"] is False
            and event["model_request_started"] is False
            for event in cosmos3_super_image_only_invalid["scheduler_events"]
        ),
        "V2-A014 ledger exactly carries forward the V2-A012 raw failures and non-request scheduler events",
        checks,
    )

    super_image_provenance_tools = cosmos3_super_image_only_provenance["tools"]
    require(
        cosmos3_super_image_only_provenance["schema_version"]
        == "vla-wam-shared-v2-cosmos3-super-image-only-v2a014-provenance-v1"
        and cosmos3_super_image_only_provenance["status"]
        == "complete_image_only_interface_probe_passed"
        and cosmos3_super_image_only_provenance["amendment"]["path"]
        == str(cosmos3_super_image_only_amendment_path.relative_to(workspace))
        and cosmos3_super_image_only_provenance["amendment"]["sha256"]
        == sha256(cosmos3_super_image_only_amendment_path)
        and cosmos3_super_image_only_provenance["registry_overlay"]["path"]
        == str(cosmos3_super_image_only_overlay_path.relative_to(workspace))
        and cosmos3_super_image_only_provenance["registry_overlay"]["sha256"]
        == sha256(cosmos3_super_image_only_overlay_path)
        and cosmos3_super_image_only_provenance["result"]["path"]
        == str(cosmos3_super_image_only_result_path.relative_to(workspace))
        and cosmos3_super_image_only_provenance["result"]["sha256"]
        == sha256(cosmos3_super_image_only_result_path)
        and cosmos3_super_image_only_provenance["checkpoint"]["revision"]
        == cosmos3_super_image_only_result["checkpoint"]["revision"]
        and cosmos3_super_image_only_provenance["checkpoint"]
        ["verified_registry_sha256"]
        == cosmos3_super_image_only_result["checkpoint"]["verified_registry"]
        ["sha256"]
        and cosmos3_super_image_only_provenance["source_commits"]
        == cosmos3_super_image_only_result["source_commits"]
        and cosmos3_super_image_only_provenance["raw_evidence"]
        ["manifest_sha256"]
        == cosmos3_super_image_only_result["probe"]["manifest"]["sha256"]
        and cosmos3_super_image_only_provenance["runtime"]
        ["fixed_image_only_request_count"]
        == 3
        and cosmos3_super_image_only_provenance["runtime"]
        ["behavioral_episode_count"]
        == 0
        and "no action reached a simulator or controller"
        in cosmos3_super_image_only_provenance["claim_boundary"],
        "V2-A014 provenance hash-binds amendment, overlay, result, checkpoint, sources, and zero behavior",
        checks,
    )
    require(
        super_image_provenance_tools["probe_client"]["path"]
        == str(cosmos3_super_image_only_probe_path.relative_to(workspace))
        and cosmos3_super_image_only_probe_path.is_file()
        and super_image_provenance_tools["probe_client"]["sha256"]
        == sha256(cosmos3_super_image_only_probe_path)
        and super_image_provenance_tools["media_builder"]["path"]
        == str(cosmos3_super_image_only_media_builder_path.relative_to(workspace))
        and cosmos3_super_image_only_media_builder_path.is_file()
        and super_image_provenance_tools["media_builder"]["sha256"]
        == sha256(cosmos3_super_image_only_media_builder_path),
        "V2-A014 provenance hash-binds the exact probe and media builders",
        checks,
    )
    super_image_kubernetes = cosmos3_super_image_only_provenance["kubernetes"]
    require(
        super_image_kubernetes["ownership_scope"] == "ali_only"
        and super_image_kubernetes["pod"]
        == "cosmos3-super-a100-2gpu-256gi-ali"
        and super_image_kubernetes["namespace"] == "211247-prod"
        and super_image_kubernetes["container_memory_limit_bytes"]
        == 274_877_906_944
        and len(super_image_kubernetes["gpus"]) == 2
        and all(
            gpu["name"] == "NVIDIA A100-SXM4-80GB"
            and gpu["memory_total_mib"] == 81_920
            and gpu["memory_used_mib_after_shutdown"] == 0
            for gpu in super_image_kubernetes["gpus"]
        )
        and super_image_kubernetes["final_memory_events"]["oom"] == 0
        and super_image_kubernetes["final_memory_events"]["oom_kill"] == 0
        and cosmos3_super_image_only_provenance["runtime"]["model_load_shards"]
        == "27/27"
        and cosmos3_super_image_only_provenance["runtime"]["server_shutdown"]
        == "clean_exact_process_group_terminated_both_gpus_zero_mib"
        and cosmos3_super_image_only_provenance["preflight"]["renderer"]
        == "not_applicable_to_v2_a014_image_only_no_simulator_probe",
        "V2-A014 provenance records an ali-owned two-A100 clean run with no OOM or residual GPU process",
        checks,
    )
    super_a100_manifest_text = cosmos3_super_a100_manifest_path.read_text()
    require(
        cosmos3_super_a100_manifest_path.is_file()
        and "name: cosmos3-super-a100-2gpu-256gi-ali"
        in super_a100_manifest_text
        and "namespace: 211247-prod" in super_a100_manifest_text
        and "research-owner: ali" in super_a100_manifest_text
        and "research-task: vla-wam-v2a014" in super_a100_manifest_text
        and "nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB"
        in super_a100_manifest_text
        and super_a100_manifest_text.count('nvidia.com/gpu: "2"') == 2
        and "memory: 256Gi" in super_a100_manifest_text
        and "claimName: 211247-prod-pvc" in super_a100_manifest_text,
        "V2-A014 A100 manifest pins the ali-owned two-GPU 256Gi pod and study PVC",
        checks,
    )
    require(
        cosmos3_super_image_only_provenance["raw_evidence"]["thermal_guard"]
        ["pause_or_emergency_count"]
        == 0
        and cosmos3_super_image_only_provenance["raw_evidence"]
        ["thermal_guard"]["monitor_exit_code"]
        == 0
        and cosmos3_super_image_only_provenance["raw_evidence"]
        ["thermal_guard"]["sample_count"]
        > 0
        and all(
            cosmos3_super_image_only_provenance["raw_evidence"][key]["bytes"]
            > 0
            and len(
                cosmos3_super_image_only_provenance["raw_evidence"][key]
                ["sha256"]
            )
            == 64
            for key in ("probe_driver_log", "server_log", "thermal_guard")
        ),
        "V2-A014 provenance retains bounded raw server, probe, and intervention-free thermal hashes",
        checks,
    )

    super_image_media_entry = cosmos3_super_image_only_media["entries"][0]
    super_image_media_builder_text = (
        cosmos3_super_image_only_media_builder_path.read_text()
    )
    require(
        cosmos3_super_image_only_media["schema_version"]
        == "vla-wam-shared-v2-cosmos3-super-image-only-v2a014-media-v1"
        and cosmos3_super_image_only_media["status"]
        == "complete_prediction_and_unexecuted_actions_only"
        and cosmos3_super_image_only_media["amendment_id"] == "V2-A014"
        and cosmos3_super_image_only_media["model_id"]
        == cosmos3_super_image_only_result["model_id"]
        and cosmos3_super_image_only_media["image_only_result"]
        == str(cosmos3_super_image_only_result_path.relative_to(workspace))
        and cosmos3_super_image_only_media["media_builder"]["path"]
        == str(cosmos3_super_image_only_media_builder_path.relative_to(workspace))
        and cosmos3_super_image_only_media["media_builder"]["sha256"]
        == sha256(cosmos3_super_image_only_media_builder_path)
        and "actions were not executed"
        in cosmos3_super_image_only_media["claim_boundary"]
        and "no actual simulator rollout"
        in cosmos3_super_image_only_media["claim_boundary"]
        and len(cosmos3_super_image_only_media["entries"]) == 1
        and super_image_media_entry["actual_rollout"] is None
        and super_image_media_entry["actual_rollout_unavailable_reason"]
        == "v2_a014_authorized_image_only_interface_evidence_no_simulator_or_controller"
        and super_image_media_entry["type"]
        == "side_by_side_model_prediction_with_unexecuted_action_trajectories"
        and super_image_media_entry["conditions"] == ["LEFT", "RIGHT"]
        and super_image_media_entry["sampling_seed"] == 8300,
        "V2-A014 media manifest labels generated futures and keeps actual rollout explicitly null",
        checks,
    )
    require(
        super_image_media_entry["source_predictions"]["left_mp4_sha256"]
        == super_image_left["video"]["mp4_sha256"]
        and super_image_media_entry["source_predictions"]["left_mp4_bytes"]
        == super_image_left["video"]["bytes"]
        and super_image_media_entry["source_predictions"]["right_mp4_sha256"]
        == super_image_right["video"]["mp4_sha256"]
        and super_image_media_entry["source_predictions"]["right_mp4_bytes"]
        == super_image_right["video"]["bytes"]
        and super_image_media_entry["model_prediction_and_actions"]["frames"]
        == 17
        and super_image_media_entry["model_prediction_and_actions"]["width"]
        == 1280
        and super_image_media_entry["model_prediction_and_actions"]["height"]
        == 720
        and super_image_media_entry["poster"]["width"] == 1280
        and super_image_media_entry["poster"]["height"] == 720
        and super_image_media_entry["action_trajectories"]["representation"]
        == "finite torch.bfloat16 [16,10], stored in NPY as canonical uint16 bfloat16 bits"
        and super_image_media_entry["action_trajectories"]
        ["visualized_in_prediction_clip"]
        is True
        and "NOT EXECUTED" in super_image_media_builder_text
        and "COSMOS3-SUPER IMAGE-ONLY PROBE"
        in super_image_media_builder_text
        and "NO ACTUAL ROLLOUT" in super_image_media_builder_text,
        "V2-A014 selected media links both 17-frame futures and carries explicit image-only non-execution labels",
        checks,
    )
    validate_file_record(
        workspace,
        super_image_media_entry["model_prediction_and_actions"],
        "V2-A014 paired prediction-and-actions video",
        checks,
    )
    validate_file_record(
        workspace,
        super_image_media_entry["poster"],
        "V2-A014 paired prediction-and-actions poster",
        checks,
    )
    for side, result_record in (
        ("left", super_image_left),
        ("right", super_image_right),
    ):
        selected_action = super_image_media_entry["action_trajectories"][side]
        validate_file_record(
            workspace,
            selected_action["json"],
            f"V2-A014 selected {side} action JSON",
            checks,
        )
        validate_file_record(
            workspace,
            selected_action["npy"],
            f"V2-A014 selected {side} canonical bfloat16-bit NPY",
            checks,
        )
        require(
            selected_action["json"]["sha256"]
            == result_record["action"]["json_sha256"]
            and selected_action["npy"]["sha256"]
            == result_record["action"]["npy_sha256"],
            f"V2-A014 selected {side} action files hash-link to the returned action",
            checks,
        )
        selected_action_json = load_json(workspace / selected_action["json"]["path"])
        require(
            selected_action_json["action_mode"] == "policy"
            and selected_action_json["domain_id"] == 8
            and selected_action_json["dtype"] == "torch.bfloat16"
            and selected_action_json["raw_action_dim"] == 10
            and selected_action_json["shape"] == [16, 10]
            and len(selected_action_json["data"]) == 16
            and all(len(row) == 10 for row in selected_action_json["data"])
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for row in selected_action_json["data"]
                for value in row
            ),
            f"V2-A014 selected {side} action JSON independently contains 160 finite bfloat16-decoded values",
            checks,
        )
    require(
        groot_result["schema_version"] == "vla-wam-shared-v2-groot-droid-result-v1"
        and groot_result["status"] == "complete"
        and groot_result["design"]["valid_episode_count"] == 6
        and groot_result["direction_summary"]["LEFT"]["successes"] == 0
        and groot_result["direction_summary"]["RIGHT"]["successes"] == 0
        and groot_result["paired_directional_evidence"]["pair_count"] == 3
        and groot_result["paired_directional_evidence"]["action_different_pair_count"] == 3
        and groot_result["paired_directional_evidence"]["endpoint_requested_ordering_aligned_pair_count"] == 3,
        "GR00T result retains six valid failures with three action-distinct aligned endpoint pairs",
        checks,
    )
    require(
        groot_readiness_artifact["status"] == "complete_6_of_6_valid_cells"
        and groot_readiness_artifact["behavior_queue"]["completed_valid_cells"] == 6
        and groot_readiness_artifact["behavior_queue"]["completed_successes"] == 0
        and groot_readiness_artifact["behavior_queue"]["next_cell"] is None
        and groot_readiness_artifact["behavior_queue"]["result_registry"]["sha256"]
        == sha256(groot_result_path),
        "GR00T readiness closes the six-cell queue and hashes the result",
        checks,
    )
    require(
        lingbot_vla_result["schema_version"] == "vla-wam-shared-v2-robotwin-pilot-v1"
        and len(lingbot_vla_result["episodes"]) == 6
        and lingbot_vla_result["summary"]["by_direction"]["left"]["successes"] == 1
        and lingbot_vla_result["summary"]["by_direction"]["right"]["successes"] == 0
        and lingbot_vla_readiness_artifact["status"] == "six_cell_gate_complete"
        and lingbot_vla_readiness_artifact["interface_contract"]["future_interface"] == "none",
        "LingBot-VLA result closes six action-only cells with LEFT-only competence",
        checks,
    )
    require(
        light_wam_result["schema_version"]
        == "vla-wam-shared-v2-light-wam-robotwin-slice-v1"
        and light_wam_result["status"] == "complete"
        and light_wam_result["valid_episode_count"] == 6
        and light_wam_result["requested_success_count"] == 1
        and light_wam_result["success_by_relation"]
        == {"left": {"successes": 1, "trials": 3}, "right": {"successes": 0, "trials": 3}}
        and light_wam_result["competence_gate"] == "left_only"
        and light_wam_result["future_interface"] == "action_only_infer_action",
        "Light-WAM result retains six action-only cells and five valid failures",
        checks,
    )
    light_evidence = {item["path"]: item for item in light_wam_result["evidence_files"]}
    require(
        light_wam_media["schema_version"] == "vla-wam-shared-v2-light-wam-media-v1"
        and light_wam_media["status"] == "complete_selected_pair00"
        and light_wam_media["behavioral_denominator_change"] == 0
        and light_wam_media["source_result"]["sha256"] == sha256(light_wam_result_path)
        and all(
            light_evidence.get(item["path"], {}).get("sha256") == item["sha256"]
            for item in light_wam_media["source_videos"].values()
        ),
        "Light-WAM publication media is post-hoc and bound to hash-locked pair00 sources",
        checks,
    )
    validate_file_record(workspace, light_wam_media["source_result"], "Light-WAM media source result", checks)
    validate_file_record(workspace, light_wam_media["publication_video"], "Light-WAM paired video", checks)
    require(
        lawam_access_retry["schema_version"]
        == "vla-wam-shared-v2-lawam-dinov3-access-retry-v1"
        and lawam_access_retry["status"]
        == "blocked_authenticated_credential_lacks_gated_model_access"
        and lawam_access_retry["attempt"]["http_status"] == 401
        and lawam_access_retry["attempt"]["model_payload_downloaded"] is False
        and lawam_access_retry["attempt"]["model_load_attempt_count"] == 0
        and lawam_access_retry["attempt"]["model_action_request_count"] == 0
        and lawam_access_retry["attempt"]["behavioral_episode_count"] == 0,
        "LaWAM authenticated retry remains a pre-inference access blocker outside denominators",
        checks,
    )
    light_gallery_entries = [
        item for item in video_gallery["entries"] if item["id"] == "light_wam_pair00"
    ]
    require(
        len(light_gallery_entries) == 1
        and light_gallery_entries[0]["video"]["sha256"]
        == light_wam_media["publication_video"]["sha256"]
        and not any(
            item["model_id"] in {"dreamzero_droid", "light_wam_robotwin"}
            for item in video_gallery["missing_publication_media"]
        ),
        "video gallery publishes Light-WAM and no longer lists completed DreamZero or Light-WAM media as missing",
        checks,
    )
    require(
        dreamzero_readiness_artifact["schema_version"]
        == "vla-wam-shared-v2-dreamzero-droid-readiness-v1"
        and dreamzero_readiness_artifact["status"]
        == "complete_six_valid_cells_both_directions_gate"
        and dreamzero_readiness_artifact["behavior_queue"]["completed_valid_cells"] == 6
        and dreamzero_readiness_artifact["behavior_queue"]["next_cell"] is None
        and dreamzero_readiness_artifact["behavior_queue"]["do_not_rerun"] is True
        and dreamzero_readiness_artifact["compiled_result"]["sha256"]
        == sha256(dreamzero_result_path),
        "DreamZero readiness freezes the completed six-cell gate and exact compiled result",
        checks,
    )
    require(
        dreamzero_result["schema_version"]
        == "vla-wam-shared-v2-dreamzero-droid-direct-gate-v1"
        and dreamzero_result["status"] == "complete"
        and dreamzero_result["model_id"] == "dreamzero_droid"
        and dreamzero_result["amendment_id"] == "V2-A007"
        and dreamzero_result["valid_episode_count"] == 6
        and dreamzero_result["valid_failure_count"] == 3
        and dreamzero_result["requested_success_count"] == 3,
        "DreamZero result preserves all six valid cells, including three valid failures",
        checks,
    )
    require(
        dreamzero_result["success_by_relation"]
        == {
            "left": {"successes": 2, "trials": 3},
            "right": {"successes": 1, "trials": 3},
        }
        and dreamzero_result["aligned_endpoint_pair_count"] == 3
        and dreamzero_result["distinct_executed_action_pair_count"] == 3
        and dreamzero_result["competence_gate"] == "both_directions"
        and dreamzero_result["wording_grid_eligible"] is True,
        "DreamZero result records bidirectional competence, prompt sensitivity, and aligned endpoints",
        checks,
    )
    dreamzero_episodes = dreamzero_result["episodes"]
    require(
        len(dreamzero_episodes) == 6
        and {(item["environment_seed"], item["requested_relation"]) for item in dreamzero_episodes}
        == {(seed, relation) for seed in (8300, 8301, 8302) for relation in ("left", "right")}
        and all(item["sampling_seed"] == item["environment_seed"] for item in dreamzero_episodes)
        and all(item["prompt_family"] == "direct_command" for item in dreamzero_episodes)
        and all(item["dynamic_prompt_switches"] == 0 for item in dreamzero_episodes),
        "DreamZero episodes use the frozen matched seeds and static direct-command conditions",
        checks,
    )
    future_audit = dreamzero_result["future_retention_audit"]
    require(
        future_audit["behavioral_episode_count"] == 6
        and future_audit["behavioral_latent_future_count"] == 265
        and future_audit["fixed_observation_probe_request_count"] == 3
        and future_audit["fixed_observation_probe_latent_future_count"] == 3
        and future_audit["total_retained_latent_future_count"] == 268
        and future_audit["total_official_reset_decode_count"] == 9
        and dreamzero_result["missing_or_unexposed_future_evidence_scored_as_zero"] is False,
        "DreamZero result retains exposed future evidence without scoring missing futures as zero",
        checks,
    )
    require(
        dreamzero_raw_collection["schema_version"]
        == "vla-wam-shared-v2-dreamzero-raw-collection-v1"
        and dreamzero_raw_collection["status"] == "complete"
        and len(dreamzero_raw_collection["cells"]) == 6
        and dreamzero_raw_collection["invalid_attempt_count"] == 11
        and dreamzero_raw_collection["runtime_intervention_count"] == 0,
        "DreamZero raw collection separates six valid cells from eleven invalid attempts and zero interventions",
        checks,
    )
    require(
        dreamzero_media["schema_version"]
        == "vla-wam-shared-v2-dreamzero-droid-media-v1"
        and dreamzero_media["source_result"]["sha256"] == sha256(dreamzero_result_path)
        and len(dreamzero_media["gallery_entries"]) == 3
        and {item["seed"] for item in dreamzero_media["gallery_entries"]}
        == {8300, 8301, 8302},
        "DreamZero media manifest publishes all three matched pairs without outcome selection",
        checks,
    )
    validate_file_record(
        workspace,
        dreamzero_media["source_result"],
        "DreamZero media source result",
        checks,
    )
    for item in dreamzero_media["gallery_entries"]:
        validate_file_record(
            workspace,
            item["video"],
            f"DreamZero seed {item['seed']} paired video",
            checks,
        )
    require(
        dreamzero_imagination_media["schema_version"]
        == "vla-wam-shared-v2-dreamzero-imagination-media-v1"
        and dreamzero_imagination_media["status"]
        == "complete_all_official_decodes_archived"
        and dreamzero_imagination_media["model_id"] == "dreamzero_droid"
        and dreamzero_imagination_media["amendment_id"] == "V2-A007"
        and dreamzero_imagination_media["source_result"]["sha256"]
        == sha256(dreamzero_result_path)
        and dreamzero_imagination_media["official_decode_count"] == 9
        and dreamzero_imagination_media["behavioral_decode_count"] == 6
        and dreamzero_imagination_media["fixed_observation_probe_decode_count"] == 3
        and len(dreamzero_imagination_media["official_decodes"]) == 9
        and len(dreamzero_imagination_media["gallery_entries"]) == 3,
        "DreamZero imagination archive includes all nine official decodes and three paired views",
        checks,
    )
    require(
        {item["scope"] for item in dreamzero_imagination_media["official_decodes"]}
        == {"valid_behavioral_episode", "fixed_observation_diagnostic"}
        and sum(
            item["scope"] == "valid_behavioral_episode"
            for item in dreamzero_imagination_media["official_decodes"]
        )
        == 6
        and sum(
            item["scope"] == "fixed_observation_diagnostic"
            for item in dreamzero_imagination_media["official_decodes"]
        )
        == 3
        and all(
            item.get("media_kind") == "model_prediction_not_execution"
            for item in dreamzero_imagination_media["gallery_entries"]
        ),
        "DreamZero imagination media separates behavioral-session and diagnostic predictions from execution",
        checks,
    )
    validate_file_record(
        workspace,
        dreamzero_imagination_media["source_result"],
        "DreamZero imagination source result",
        checks,
    )
    for item in dreamzero_imagination_media["official_decodes"]:
        validate_file_record(
            workspace,
            item["archived_video"],
            f"DreamZero {item['id']} official decode",
            checks,
        )
    for item in dreamzero_imagination_media["gallery_entries"]:
        validate_file_record(
            workspace,
            item["video"],
            f"DreamZero seed {item['seed']} paired imagination video",
            checks,
        )
    require(
        post_result_decision.get("status")
        == "recorded_and_frozen_before_new_inference"
        and post_result_decision.get("amendment_id") == "V2-A005"
        and post_result_decision.get("amendment")
        == "artifacts/vla_wam_shared_v2/pilot/post_result_expansion_amendment.json"
        and post_result_decision.get("amendment_sha256")
        == sha256(post_result_amendment_path)
        and post_result_decision.get("authorized_queue")
        == [item["id"] for item in post_result_amendment["authorized_queue"]]
        and post_result_decision.get("conditional_second_wave")
        == post_result_amendment["conditional_second_wave"]["candidate_ids"],
        "continuation state binds the frozen V2-A005 authorized queue and conditional audit",
        checks,
    )
    authorized_queue = post_result_amendment["authorized_queue"]
    require(
        post_result_amendment["schema_version"]
        == "vla-wam-shared-v2-post-result-expansion-amendment-v1"
        and post_result_amendment["amendment_id"] == "V2-A005"
        and post_result_amendment["status"]
        == "frozen_after_completed_three_wam_results_before_any_newly_authorized_inference"
        and post_result_amendment["recorded_at_git_head"]
        == "961a6e2c9713af1f377105dae96b11a7ab8bc419"
        and [item["priority"] for item in authorized_queue] == [0, 1, 2, 3, 4]
        and [item["id"] for item in authorized_queue]
        == [
            "groot_n17_droid_direct_gate",
            "cosmos3_edge_droid_v2_direct_replication",
            "lingbot_vla_4b_robotwin_direct_gate",
            "cosmos_reason2_static_language_diagnostic",
            "pi0_fast_three_wording_expansion",
        ],
        "V2-A005 is disclosed, ordered, and frozen against the completed evidence head",
        checks,
    )
    known = post_result_amendment["known_evidence_at_decision"]
    require(
        known["pi0_fast_direct_confirmation_sha256"]
        == sha256(pi0_fast_confirmation_path)
        and known["efficient_wam_rt_pairs04_09_slice_sha256"]
        == sha256(
            workspace
            / "artifacts/vla_wam_shared_v2/pilot/directional_confirmation/efficient_wam_rt_pairs04_09_slice.json"
        )
        and known["fastwam_pairs03_09_slice_sha256"]
        == sha256(
            workspace
            / "artifacts/vla_wam_shared_v2/pilot/directional_confirmation/fastwam_pairs03_09_slice.json"
        )
        and known["lingbot_va_pairs03_09_slice_sha256"]
        == sha256(
            workspace
            / "artifacts/vla_wam_shared_v2/pilot/directional_confirmation/lingbot_va_pairs03_09_slice.json"
        ),
        "V2-A005 hashes the completed evidence known when the expansion was selected",
        checks,
    )
    by_id = {item["id"]: item for item in authorized_queue}
    require(
        by_id["groot_n17_droid_direct_gate"]["behavioral_episode_count"] == 6
        and by_id["cosmos3_edge_droid_v2_direct_replication"]["behavioral_episode_count"]
        == 6
        and by_id["lingbot_vla_4b_robotwin_direct_gate"]["behavioral_episode_count"]
        == 6
        and by_id["cosmos_reason2_static_language_diagnostic"]["behavioral_episode_count"]
        == 0
        and by_id["cosmos_reason2_static_language_diagnostic"]["diagnostic_input_count"]
        == 12
        and by_id["pi0_fast_three_wording_expansion"]["behavioral_episode_count"]
        == 60
        and by_id["pi0_fast_three_wording_expansion"]["prompt_families"]
        == [
            "short_command",
            "goal_as_outcome",
            "desired_plus_negated_opposite",
        ],
        "V2-A005 bounds behavioral spend and keeps Cosmos-Reason2 diagnostic-only",
        checks,
    )
    second_wave_state = post_result_decision.get("second_wave_amendment", {})
    require(
        second_wave_state.get("status")
        == "light_wam_and_lawam_selected_before_download_or_inference"
        and second_wave_state.get("amendment_id") == "V2-A006"
        and second_wave_state.get("path")
        == "artifacts/vla_wam_shared_v2/pilot/post_result_second_wave_amendment.json"
        and second_wave_state.get("sha256") == sha256(second_wave_amendment_path)
        and second_wave_state.get("selected_queue")
        == [item["id"] for item in second_wave_amendment["selected_models"]]
        and second_wave_state.get("deferred_not_zero")
        == [
            item["model_id"]
            for item in second_wave_amendment["audited_not_selected_for_this_wave"]
        ],
        "continuation state binds the V2-A006 selected and deferred second wave",
        checks,
    )
    selected_second_wave = second_wave_amendment["selected_models"]
    require(
        second_wave_amendment["schema_version"]
        == "vla-wam-shared-v2-post-result-second-wave-amendment-v1"
        and second_wave_amendment["amendment_id"] == "V2-A006"
        and second_wave_amendment["status"]
        == "frozen_after_official_release_audit_before_selected_second_wave_download_or_inference"
        and second_wave_amendment["recorded_at_git_head"]
        == "9c0bf0d44cfec7eeb35ab358d6bc38866e5005ab"
        and second_wave_amendment["parent_amendment"]["sha256"]
        == sha256(post_result_amendment_path)
        and [item["priority"] for item in selected_second_wave] == [0, 1]
        and [item["id"] for item in selected_second_wave]
        == ["light_wam_robotwin_direct_gate", "lawam_robotwin_direct_gate"]
        and all(item["behavioral_episode_count"] == 6 for item in selected_second_wave)
        and all(
            item["pair_ids"]
            == ["robotwin_pair_00", "robotwin_pair_01", "robotwin_pair_02"]
            and item["requested_relations"] == ["left", "right"]
            and item["prompt_family"] == "direct_command"
            for item in selected_second_wave
        ),
        "V2-A006 freezes exactly two bounded RoboTwin direct gates before selected-model setup",
        checks,
    )
    require(
        {
            item["model_id"]: item["status"]
            for item in second_wave_amendment["audited_not_selected_for_this_wave"]
        }
        == {
            "dreamzero_droid": "deferred_adapter_and_cost_gate",
            "pi0_droid_vla": "deferred_family_redundancy",
        },
        "V2-A006 keeps DreamZero and pi0 DROID deferred rather than assigning zeros",
        checks,
    )
    dreamzero_state = post_result_decision.get("dreamzero_amendment", {})
    require(
        dreamzero_state.get("status")
        == "frozen_before_any_dreamzero_study_model_request_or_behavioral_inference"
        and dreamzero_state.get("amendment_id") == "V2-A007"
        and dreamzero_state.get("path")
        == "artifacts/vla_wam_shared_v2/pilot/post_result_dreamzero_amendment.json"
        and dreamzero_state.get("sha256") == sha256(dreamzero_amendment_path)
        and dreamzero_state.get("authorized_queue")
        == [dreamzero_amendment["selected_model"]["id"]]
        and dreamzero_state.get("behavioral_episode_count") == 6
        and dreamzero_state.get("simulator_video_lane")
        == "raytrace-rtxpro6000-ali",
        "continuation state binds the frozen V2-A007 DreamZero gate",
        checks,
    )
    dreamzero = dreamzero_amendment["selected_model"]
    require(
        dreamzero_amendment["schema_version"]
        == "vla-wam-shared-v2-post-result-dreamzero-amendment-v1"
        and dreamzero_amendment["amendment_id"] == "V2-A007"
        and dreamzero_amendment["status"]
        == "frozen_after_existing_results_and_before_any_dreamzero_study_model_request_or_behavioral_inference"
        and dreamzero_amendment["recorded_at_git_head"]
        == "ca91aeb7a05b45a911c164638273be237c37b78b"
        and dreamzero_amendment["parent_amendment"]["sha256"]
        == sha256(second_wave_amendment_path)
        and dreamzero["id"] == "dreamzero_droid_direct_gate"
        and dreamzero["repository_commit"]
        == "ab790c198fbce33503358efbbd4187ce9a89adf3"
        and dreamzero["checkpoint_revision"]
        == "96ad344138c66e82536422432ad742f015784942"
        and dreamzero["behavioral_episode_count"] == 6
        and dreamzero["environment_seeds"] == [8300, 8301, 8302]
        and dreamzero["sampling_seeds"] == [8300, 8301, 8302]
        and dreamzero["requested_relations"] == ["left", "right"]
        and dreamzero["prompt_family"] == "direct_command",
        "V2-A007 freezes the exact DreamZero source and bounded six-cell direct gate",
        checks,
    )
    require(
        dreamzero_amendment["runtime_topology"]["simulator_and_video"].find(
            "raytrace-rtxpro6000-ali"
        )
        >= 0
        and dreamzero_amendment["preexisting_process_disclosure"]["study_use"]
        == "prohibited"
        and len(dreamzero_amendment["release_gates"]) == 9
        and "Missing or unexposed futures are not zeros"
        in dreamzero["future_rule"],
        "V2-A007 requires RTX video, excludes the old server, and preserves future-evidence boundaries",
        checks,
    )
    for relative_doc in continuation_state["authoritative_docs"]:
        require(
            (workspace / relative_doc).is_file(),
            f"continuation document {relative_doc} exists",
            checks,
        )

    pair03_handoff = validate_efficient_pair03_handoff(
        workspace,
        efficient_pair03_integration_path,
        directional_expansion_path,
        bundle_manifest_path,
        checks,
    )
    efficient_pairs04_09 = validate_efficient_pairs04_09_slice(
        workspace, efficient_pairs04_09_manifest_path, checks
    )
    fastwam_pairs03_09 = validate_wam_pairs03_09_slice(
        workspace,
        fastwam_pairs03_09_manifest_path,
        directional_expansion_path,
        directional_fixtures_path,
        model_id="fastwam_robotwin",
        expected_successes={"left": 1, "right": 1},
        expected_aligned_pairs=3,
        expected_invalid_attempts=18,
        expected_future_interface="action_only_not_applicable",
        checks=checks,
    )
    lingbot_pairs03_09 = validate_wam_pairs03_09_slice(
        workspace,
        lingbot_pairs03_09_manifest_path,
        directional_expansion_path,
        directional_fixtures_path,
        model_id="lingbot_va_robotwin",
        expected_successes={"left": 3, "right": 4},
        expected_aligned_pairs=6,
        expected_invalid_attempts=5,
        expected_future_interface="latent_only_future_not_decodable",
        checks=checks,
    )
    confirmation = validate_pi0_fast_confirmation(
        workspace, pi0_fast_confirmation_path, pi0_fast_expansion_path, checks
    )
    robotwin_confirmations = validate_robotwin_confirmations(
        workspace, directional_expansion_path, checks
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
        "lingbot_result_path": str(lingbot_result_path.relative_to(workspace)),
        "lingbot_result_sha256": sha256(lingbot_result_path),
        "pi0_fast_result_path": str(pi0_fast_result_path.relative_to(workspace)),
        "pi0_fast_result_sha256": sha256(pi0_fast_result_path),
        "groot_result_path": str(groot_result_path.relative_to(workspace)),
        "groot_result_sha256": sha256(groot_result_path),
        "lingbot_vla_result_path": str(lingbot_vla_result_path.relative_to(workspace)),
        "lingbot_vla_result_sha256": sha256(lingbot_vla_result_path),
        "light_wam_result_path": str(light_wam_result_path.relative_to(workspace)),
        "light_wam_result_sha256": sha256(light_wam_result_path),
        "light_wam_media_path": str(light_wam_media_path.relative_to(workspace)),
        "light_wam_media_sha256": sha256(light_wam_media_path),
        "lawam_access_retry_path": str(lawam_access_retry_path.relative_to(workspace)),
        "lawam_access_retry_sha256": sha256(lawam_access_retry_path),
        "video_gallery_path": str(video_gallery_path.relative_to(workspace)),
        "video_gallery_sha256": sha256(video_gallery_path),
        "pi0_fast_confirmation": confirmation,
        "robotwin_confirmations": robotwin_confirmations,
        "runtime_interventions_path": str(runtime_interventions_path.relative_to(workspace)),
        "runtime_interventions_sha256": sha256(runtime_interventions_path),
        "directional_expansion_path": str(directional_expansion_path.relative_to(workspace)),
        "directional_expansion_sha256": sha256(directional_expansion_path),
        "action_trace_amendment_path": str(action_trace_amendment_path.relative_to(workspace)),
        "action_trace_amendment_sha256": sha256(action_trace_amendment_path),
        "directional_fixtures_path": str(directional_fixtures_path.relative_to(workspace)),
        "directional_fixtures_sha256": sha256(directional_fixtures_path),
        "pi0_fast_expansion_path": str(pi0_fast_expansion_path.relative_to(workspace)),
        "pi0_fast_expansion_sha256": sha256(pi0_fast_expansion_path),
        "continuation_state_path": str(continuation_state_path.relative_to(workspace)),
        "continuation_state_sha256": sha256(continuation_state_path),
        "post_result_expansion_amendment_path": str(
            post_result_amendment_path.relative_to(workspace)
        ),
        "post_result_expansion_amendment_sha256": sha256(
            post_result_amendment_path
        ),
        "post_result_second_wave_amendment_path": str(
            second_wave_amendment_path.relative_to(workspace)
        ),
        "post_result_second_wave_amendment_sha256": sha256(
            second_wave_amendment_path
        ),
        "post_result_dreamzero_amendment_path": str(
            dreamzero_amendment_path.relative_to(workspace)
        ),
        "post_result_dreamzero_amendment_sha256": sha256(
            dreamzero_amendment_path
        ),
        "post_result_cfg_ablation_v2a015": cfg_ablation_v2a015,
        "post_result_current_stack_replication_amendment_path": str(
            current_stack_amendment_path.relative_to(workspace)
        ),
        "post_result_current_stack_replication_amendment_sha256": sha256(
            current_stack_amendment_path
        ),
        "pi0_fast_current_stack_v2a008_registry_path": str(
            current_stack_registry_path.relative_to(workspace)
        ),
        "pi0_fast_current_stack_v2a008_registry_sha256": sha256(
            current_stack_registry_path
        ),
        "pi0_fast_current_stack_v2a008_release_probe_path": str(
            current_stack_release_probe_path.relative_to(workspace)
        ),
        "pi0_fast_current_stack_v2a008_release_probe_sha256": sha256(
            current_stack_release_probe_path
        ),
        "lawam_withdrawal_amendment_path": str(
            lawam_withdrawal_amendment_path.relative_to(workspace)
        ),
        "lawam_withdrawal_amendment_sha256": sha256(
            lawam_withdrawal_amendment_path
        ),
        "pi05_current_stack_amendment_path": str(
            pi05_current_amendment_path.relative_to(workspace)
        ),
        "pi05_current_stack_amendment_sha256": sha256(
            pi05_current_amendment_path
        ),
        "pi05_current_stack_checkpoint_manifest_path": str(
            pi05_checkpoint_manifest_path.relative_to(workspace)
        ),
        "pi05_current_stack_checkpoint_manifest_sha256": sha256(
            pi05_checkpoint_manifest_path
        ),
        "pi05_current_stack_registry_path": str(
            pi05_current_registry_path.relative_to(workspace)
        ),
        "pi05_current_stack_registry_sha256": sha256(
            pi05_current_registry_path
        ),
        "pi05_current_stack_result_path": str(pi05_result_path.relative_to(workspace)),
        "pi05_current_stack_result_sha256": sha256(pi05_result_path),
        "pi05_current_stack_release_probe_path": str(pi05_release_probe_path.relative_to(workspace)),
        "pi05_current_stack_release_probe_sha256": sha256(pi05_release_probe_path),
        "pi05_current_stack_fixed_observation_path": str(pi05_fixed_observation_path.relative_to(workspace)),
        "pi05_current_stack_fixed_observation_sha256": sha256(pi05_fixed_observation_path),
        "pi05_current_stack_invalid_attempts_path": str(pi05_invalid_attempts_path.relative_to(workspace)),
        "pi05_current_stack_invalid_attempts_sha256": sha256(pi05_invalid_attempts_path),
        "pi05_current_stack_provenance_path": str(pi05_provenance_path.relative_to(workspace)),
        "pi05_current_stack_provenance_sha256": sha256(pi05_provenance_path),
        "pi05_current_stack_media_path": str(pi05_media_path.relative_to(workspace)),
        "pi05_current_stack_media_sha256": sha256(pi05_media_path),
        "cosmos3_nano_policy_droid_registry_path": str(
            cosmos3_nano_registry_path.relative_to(workspace)
        ),
        "cosmos3_nano_policy_droid_registry_sha256": sha256(
            cosmos3_nano_registry_path
        ),
        "cosmos3_super_droid_amendment_path": str(
            cosmos3_super_amendment_path.relative_to(workspace)
        ),
        "cosmos3_super_droid_amendment_sha256": sha256(
            cosmos3_super_amendment_path
        ),
        "cosmos3_super_droid_registry_path": str(
            cosmos3_super_registry_path.relative_to(workspace)
        ),
        "cosmos3_super_droid_registry_sha256": sha256(
            cosmos3_super_registry_path
        ),
        "cosmos3_super_hf_snapshot_path": str(
            cosmos3_super_snapshot_path.relative_to(workspace)
        ),
        "cosmos3_super_hf_snapshot_sha256": sha256(cosmos3_super_snapshot_path),
        "cosmos3_super_v2a012_runtime_gate_path": str(
            cosmos3_super_runtime_gate_path.relative_to(workspace)
        ),
        "cosmos3_super_v2a012_runtime_gate_sha256": sha256(
            cosmos3_super_runtime_gate_path
        ),
        "cosmos3_super_image_only_v2a014_amendment_path": str(
            cosmos3_super_image_only_amendment_path.relative_to(workspace)
        ),
        "cosmos3_super_image_only_v2a014_amendment_sha256": sha256(
            cosmos3_super_image_only_amendment_path
        ),
        "cosmos3_super_image_only_v2a014_overlay_path": str(
            cosmos3_super_image_only_overlay_path.relative_to(workspace)
        ),
        "cosmos3_super_image_only_v2a014_overlay_sha256": sha256(
            cosmos3_super_image_only_overlay_path
        ),
        "cosmos3_super_image_only_v2a014_result_path": str(
            cosmos3_super_image_only_result_path.relative_to(workspace)
        ),
        "cosmos3_super_image_only_v2a014_result_sha256": sha256(
            cosmos3_super_image_only_result_path
        ),
        "cosmos3_super_image_only_v2a014_invalid_attempts_path": str(
            cosmos3_super_image_only_invalid_path.relative_to(workspace)
        ),
        "cosmos3_super_image_only_v2a014_invalid_attempts_sha256": sha256(
            cosmos3_super_image_only_invalid_path
        ),
        "cosmos3_super_image_only_v2a014_provenance_path": str(
            cosmos3_super_image_only_provenance_path.relative_to(workspace)
        ),
        "cosmos3_super_image_only_v2a014_provenance_sha256": sha256(
            cosmos3_super_image_only_provenance_path
        ),
        "cosmos3_super_image_only_v2a014_media_path": str(
            cosmos3_super_image_only_media_path.relative_to(workspace)
        ),
        "cosmos3_super_image_only_v2a014_media_sha256": sha256(
            cosmos3_super_image_only_media_path
        ),
        "cosmos3_super_image_only_v2a014_probe_sha256": sha256(
            cosmos3_super_image_only_probe_path
        ),
        "cosmos3_super_image_only_v2a014_media_builder_sha256": sha256(
            cosmos3_super_image_only_media_builder_path
        ),
        "cosmos3_super_a100_manifest_sha256": sha256(
            cosmos3_super_a100_manifest_path
        ),
        "cosmos3_edge_base_amendment_path": str(
            cosmos3_edge_base_amendment_path.relative_to(workspace)
        ),
        "cosmos3_edge_base_amendment_sha256": sha256(
            cosmos3_edge_base_amendment_path
        ),
        "cosmos3_edge_base_registry_path": str(
            cosmos3_edge_base_registry_path.relative_to(workspace)
        ),
        "cosmos3_edge_base_registry_sha256": sha256(
            cosmos3_edge_base_registry_path
        ),
        "cosmos3_edge_base_fixed_observation_path": str(
            cosmos3_edge_base_fixed_path.relative_to(workspace)
        ),
        "cosmos3_edge_base_fixed_observation_sha256": sha256(
            cosmos3_edge_base_fixed_path
        ),
        "cosmos3_edge_base_invalid_attempts_path": str(
            cosmos3_edge_base_invalid_path.relative_to(workspace)
        ),
        "cosmos3_edge_base_invalid_attempts_sha256": sha256(
            cosmos3_edge_base_invalid_path
        ),
        "cosmos3_edge_base_provenance_path": str(
            cosmos3_edge_base_provenance_path.relative_to(workspace)
        ),
        "cosmos3_edge_base_provenance_sha256": sha256(
            cosmos3_edge_base_provenance_path
        ),
        "cosmos3_edge_base_curobo_audit_path": str(
            cosmos3_edge_base_curobo_audit_path.relative_to(workspace)
        ),
        "cosmos3_edge_base_curobo_audit_sha256": sha256(
            cosmos3_edge_base_curobo_audit_path
        ),
        "cosmos3_edge_base_media_path": str(
            cosmos3_edge_base_media_path.relative_to(workspace)
        ),
        "cosmos3_edge_base_media_sha256": sha256(
            cosmos3_edge_base_media_path
        ),
        "efficient_wam_pair03_handoff": pair03_handoff,
        "efficient_wam_pairs04_09": efficient_pairs04_09,
        "fastwam_pairs03_09": fastwam_pairs03_09,
        "lingbot_va_pairs03_09": lingbot_pairs03_09,
        "paired_media_path": str(paired_media_path.relative_to(workspace)),
        "paired_media_sha256": sha256(paired_media_path),
        "droid_paired_media_path": str(droid_paired_media_path.relative_to(workspace)),
        "droid_paired_media_sha256": sha256(droid_paired_media_path),
        "figures_manifest_path": str(figures_manifest_path.relative_to(workspace)),
        "figures_manifest_sha256": sha256(figures_manifest_path),
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
