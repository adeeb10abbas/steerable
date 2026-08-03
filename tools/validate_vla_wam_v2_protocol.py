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
    bundle_manifest = load_json(bundle_manifest_path)
    paired_media = load_json(paired_media_path)
    droid_paired_media = load_json(droid_paired_media_path)
    figures_manifest = load_json(figures_manifest_path)
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
        == "post_result_expansions_v2_a005_a006_frozen_active_setup",
        "continuation state names the frozen V2-A005/A006 active-setup boundary",
        checks,
    )
    queue = continuation_state["experiment_queue"]
    require(
        [item["priority"] for item in queue] == [0, 1, 2, 3],
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
        ],
        "continuation queue preserves the four authorized or blocked next experiments",
        checks,
    )
    groot_readiness = continuation_state.get("groot_n17_readiness", {})
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
        and queue[2]["status"]
        == "authorized_onboarding_blocked_checkpoint_and_repository_missing"
        and queue[3]["status"]
        == "authorized_ready_for_exact_repeat_probe_and_direct_gate"
        and groot_readiness.get("status")
        == "assets_downloaded_and_server_contract_smoke_complete"
        and groot_readiness.get("server_smoke", {}).get("health_ping") is True
        and groot_readiness.get("server_smoke", {}).get("simulator_episode_started") is False,
        "continuation state freezes all 42 WAM cells and records authorized GR00T and LingBot-VLA gates",
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
