#!/usr/bin/env python3
"""Compile one frozen 10-scene RoboTwin directional confirmation.

The output is deliberately separate from each six-cell ``*_direct_gate``
artifact.  It uses recorded simulator state only after policy execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from compile_vla_wam_v2_robotwin_pilot import (
    MODEL_LABELS, PICKUP_CONSECUTIVE_STEPS, PICKUP_LIFT_M, file_record,
    max_consecutive, predicted_artifact, predicted_video,
)


SCHEMA_VERSION = "vla-wam-shared-v2-robotwin-direct-confirmation-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_95(successes: int, total: int) -> dict[str, float]:
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return {"confidence": 0.95, "lower": center - half, "upper": center + half}


def initial_state_record(result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Hash every initial-state field the adapter actually recorded.

    The current adapters expose object/target positions and aggregate gripper
    state, but not full object orientation, robot joint state, or individual
    gripper poses.  Keep that limitation visible rather than calling this a
    byte-identical simulator-state proof.
    """
    initial = result.get("initial")
    if not isinstance(initial, dict):
        raise RuntimeError("Result is missing an initial-state mapping")
    recorded_initial_fields = {
        key: result[key]
        for key in sorted(result)
        if key == "initial" or key.startswith("initial_")
    }
    identity = {
        "task": result["task"],
        "environment_seed": int(result["environment_seed"]),
        "sampling_seed": int(result["sampling_seed"]),
        "object_model_id": result["object_model_id"],
        "target_model_id": result["target_model_id"],
        "object_name": result["object_name"],
        "target_name": result["target_name"],
        "recorded_initial_fields": recorded_initial_fields,
    }
    coverage = {
        "hash_input_top_level_fields": sorted(recorded_initial_fields),
        "hash_input_initial_fields": sorted(initial),
        "adapter_state_limits": [
            "No full object or target orientation is recorded by the current adapter.",
            "No robot joint state or end-effector pose is recorded by the current adapter.",
            "Only aggregate grippers_open is recorded; individual gripper state is unavailable.",
        ],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return digest, coverage


def action_trace_record(result_path: Path, result: dict[str, Any], required: bool) -> dict[str, Any] | None:
    declared = result.get("action_trace")
    if declared is None:
        if required:
            raise RuntimeError(f"Prospective cell is missing result.json action_trace metadata: {result_path}")
        return None
    if set(declared) != {"path", "sha256", "count", "shape"}:
        raise RuntimeError(f"Invalid action_trace metadata schema: {result_path}")
    path = Path(declared["path"])
    if not path.is_absolute():
        path = result_path.parent / path
    if not path.is_file():
        raise RuntimeError(f"Declared action trace is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    executed_name = "executed" if "executed" in arrays else "denormalized" if "denormalized" in arrays else None
    if executed_name is None:
        raise RuntimeError(f"Action trace lacks executed/denormalized actions: {path}")
    executed = arrays[executed_name]
    observed = file_record(path)
    if (observed["sha256"] != declared["sha256"] or int(declared["count"]) != int(executed.shape[0])
            or list(declared["shape"]) != list(executed.shape)):
        raise RuntimeError(f"Action-trace metadata does not match recomputed file/count/shape: {path}")
    return {**observed, "count": int(executed.shape[0]), "shape": list(executed.shape),
            "executed_array": executed_name,
            "arrays": {name: {"shape": list(array.shape), "count": int(array.shape[0])} for name, array in arrays.items()}}


def first_ten_executed_action_rms(left: dict[str, Any], right: dict[str, Any]) -> tuple[float | None, int, str | None]:
    def load(record: dict[str, Any]) -> np.ndarray:
        with np.load(record["path"], allow_pickle=False) as archive:
            return np.asarray(archive[record["executed_array"]])
    left_actions, right_actions = load(left), load(right)
    if left_actions.ndim < 1 or right_actions.ndim < 1:
        return None, 0, "action_trace_is_scalar_not_a_sequence"
    steps_used = min(len(left_actions), len(right_actions), 10)
    if steps_used == 0:
        return None, 0, "no_common_executed_actions"
    if left_actions[:steps_used].shape != right_actions[:steps_used].shape:
        return None, steps_used, "paired_action_shape_mismatch"
    rms = float(np.sqrt(np.mean(np.square(left_actions[:steps_used] - right_actions[:steps_used]))))
    return rms, steps_used, None


def load_registry(path: Path, model_id: str) -> dict[int, dict[str, Any]]:
    registry = json.loads(path.read_text())
    if registry.get("models") != list(MODEL_LABELS) or model_id not in registry["models"]:
        raise RuntimeError(f"Model is not in frozen directional registry: {model_id}")
    scenes = registry.get("scenes", [])
    if len(scenes) != 10 or {scene["environment_seed"] for scene in scenes} != set(range(4300000, 4300010)):
        raise RuntimeError("Directional registry must contain exactly the ten frozen scenes")
    return {int(scene["environment_seed"]): scene for scene in scenes}


def is_prospective_scene(scene: dict[str, Any]) -> bool:
    """Use the frozen phase when present; seed split remains the schema fallback."""
    phase = scene.get("phase")
    if phase is not None and phase not in {"completed_pilot", "new_expansion"}:
        raise RuntimeError(f"Unknown directional scene phase: {phase}")
    return phase == "new_expansion" if phase is not None else int(scene["environment_seed"]) >= 4300003


def _ledger_paths(paths: Path | list[Path] | tuple[Path, ...] | None) -> list[Path]:
    if paths is None:
        return []
    return [paths] if isinstance(paths, Path) else list(paths)


def _load_ledger(path: Path, model_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"Supplied ledger does not exist: {path}")
    payload = json.loads(path.read_text())
    events = payload.get("events")
    if not isinstance(events, list):
        raise RuntimeError(f"Ledger must contain an events list: {path}")
    selected = [event for event in events if event.get("model_id") == model_id]
    unknown_model = [event for event in events if "model_id" not in event]
    if unknown_model:
        raise RuntimeError(f"Cannot model-filter ledger events without model_id: {path}")
    record = {
        **file_record(path),
        "total_event_count": len(events),
        "selected_model_event_count": len(selected),
        "ignored_other_model_event_count": len(events) - len(selected),
        "selected_event_ids": [event.get("id") for event in selected],
    }
    return selected, record


def load_interventions(
    paths: Path | list[Path] | tuple[Path, ...] | None, model_id: str
) -> tuple[dict[tuple[int, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    result: dict[tuple[int, str], list[dict[str, Any]]] = {}
    sources = []
    seen_ids: set[str] = set()
    for path in _ledger_paths(paths):
        events, source = _load_ledger(path, model_id)
        sources.append(source)
        for event in events:
            required = {"id", "environment_seed", "requested_relation", "wall_latency_valid"}
            if not required <= set(event):
                raise RuntimeError(f"Runtime intervention lacks required provenance fields: {event}")
            if event["id"] in seen_ids:
                raise RuntimeError(f"Duplicate runtime intervention ID across ledgers: {event['id']}")
            seen_ids.add(event["id"])
            behavioral_values = [
                event[key] for key in ("behavioral_result_valid", "behavioral_valid") if key in event
            ]
            if not behavioral_values or any(value is not True for value in behavioral_values):
                raise RuntimeError(f"Runtime intervention lacks a valid-behavior declaration: {event}")
            if event["wall_latency_valid"] is not False:
                raise RuntimeError(f"Runtime intervention contradicts valid behavior/invalid wall latency: {event}")
            result.setdefault((int(event["environment_seed"]), event["requested_relation"]), []).append(event)
    return result, sources


def load_invalid_attempts(
    paths: Path | list[Path] | tuple[Path, ...] | None, model_id: str
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    selected = []
    by_cell: dict[tuple[int, str], list[dict[str, Any]]] = {}
    sources = []
    seen_ids: set[str] = set()
    for path in _ledger_paths(paths):
        events, source = _load_ledger(path, model_id)
        sources.append(source)
        for event in events:
            if event.get("classification") not in {"technical_invalid", "partial"}:
                raise RuntimeError(f"Invalid confirmation ledger event: {event}")
            required = {"id", "environment_seed", "requested_relation"}
            if not required <= set(event):
                raise RuntimeError(f"Invalid attempt lacks cell provenance: {event}")
            if event["id"] in seen_ids:
                raise RuntimeError(f"Duplicate invalid-attempt ID across ledgers: {event['id']}")
            seen_ids.add(event["id"])
            if event.get("behavioral_result_valid", False) is not False or event.get("wall_latency_valid", False) is not False:
                raise RuntimeError(f"Invalid/partial attempt must exclude behavior and wall latency: {event}")
            selected.append(event)
            by_cell.setdefault((int(event["environment_seed"]), event["requested_relation"]), []).append(event)
    return selected, by_cell, sources


def validate_result_ledger_reconciliation(
    result_path: Path, result: dict[str, Any], runtime_events: list[dict[str, Any]],
    invalid_events: list[dict[str, Any]],
) -> None:
    reported_runtime_ids = set()
    for key in ("runtime_intervention_ids", "thermal_intervention_ids", "intervention_ids"):
        value = result.get(key, [])
        if value:
            if not isinstance(value, list):
                raise RuntimeError(f"Result {key} must be a list: {result_path}")
            reported_runtime_ids.update(value)
    applied_runtime_ids = {event["id"] for event in runtime_events}
    if not reported_runtime_ids <= applied_runtime_ids:
        raise RuntimeError(f"Result self-reports unrepresented thermal intervention IDs: {result_path}")
    reported_invalid_ids = set()
    for key in ("invalid_attempt_ids", "partial_attempt_ids"):
        value = result.get(key, [])
        if value:
            if not isinstance(value, list):
                raise RuntimeError(f"Result {key} must be a list: {result_path}")
            reported_invalid_ids.update(value)
    represented_invalid_ids = {event["id"] for event in invalid_events}
    if not reported_invalid_ids <= represented_invalid_ids:
        raise RuntimeError(f"Result self-reports unrepresented partial/invalid attempt IDs: {result_path}")
    reports_thermal = any(result.get(key) is True for key in ("thermally_intervened", "thermal_intervention", "thermal_paused"))
    reports_invalid_latency = result.get("wall_latency_valid") is False
    if (reports_thermal or reports_invalid_latency) and not runtime_events:
        raise RuntimeError(f"Result self-reports thermal/latency intervention not represented in ledgers: {result_path}")
    status = str(result.get("status", "")).lower()
    reports_partial = result.get("partial") is True or result.get("behavioral_result_valid") is False or status in {
        "partial", "technical_invalid", "emergency_hold", "worker_exit_nonzero", "monitor_error"
    }
    if reports_partial and not invalid_events:
        raise RuntimeError(f"Result self-reports partial/invalid state not represented in ledgers: {result_path}")
    if reports_partial:
        raise RuntimeError(f"Partial/invalid result cannot enter the 20 valid confirmation cells: {result_path}")


def classify(
    result_path: Path, model_id: str, scenes: dict[int, dict[str, Any]],
    interventions: dict[tuple[int, str], list[dict[str, Any]]],
    invalid_attempts: dict[tuple[int, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    seed, direction = int(result["environment_seed"]), result["requested_relation"]
    scene = scenes.get(seed)
    if scene is None or direction not in {"left", "right"}:
        raise RuntimeError(f"Unregistered confirmation cell: {result_path}")
    events = interventions.get((seed, direction), [])
    cell_invalid_attempts = invalid_attempts.get((seed, direction), [])
    validate_result_ledger_reconciliation(result_path, result, events, cell_invalid_attempts)
    if (result["task"], int(result["sampling_seed"])) != (scene["anchor_task"], int(scene["sampling_seed"])):
        raise RuntimeError(f"Frozen task/seed mismatch: {result_path}")
    if result.get("prompt_family") != "direct_command":
        raise RuntimeError(f"Non-direct prompt family: {result_path}")
    expected_prompt = f"Put the {result['object_name']} to the {direction} of the {result['target_name']}."
    if result.get("prompt") != expected_prompt:
        raise RuntimeError(f"Prompt bytes do not match frozen direct template: {result_path}")
    trajectory_path = Path(result["trajectory_path"])
    trajectory = json.loads(trajectory_path.read_text())
    if not trajectory:
        raise RuntimeError(f"Empty trajectory: {trajectory_path}")
    initial_z = float(trajectory[0]["object_xyz"][2])
    closed = [not bool(step["grippers_open"]) for step in trajectory]
    pickup = max_consecutive([
        float(step["object_xyz"][2]) - initial_z >= PICKUP_LIFT_M and closed_flag
        for step, closed_flag in zip(trajectory, closed, strict=True)
    ]) >= PICKUP_CONSECUTIVE_STEPS
    entered = [int(step["action_step"]) for step in trajectory if step["relation_region"]]
    released = [int(step["action_step"]) for step in trajectory if step["relation_region"] and step["grippers_open"]]
    success = bool(result["requested_success"])
    if success:
        failure_stage = "success"
    elif entered:
        failure_stage = "entered_requested_region_without_verified_completion"
    elif pickup:
        failure_stage = "picked_never_entered_requested_region"
    elif any(closed):
        failure_stage = "closed_gripper_no_verified_pickup"
    else:
        failure_stage = "no_verified_interaction"
    future_video = predicted_video(result)
    future = predicted_artifact(result_path, result, future_video)
    if future and future["kind"] == "decoded_video":
        future_interface = "decoded_future_video"
    elif future and future["kind"] == "latent_tensor":
        future_interface = "latent_only_future_not_decodable"
    else:
        future_interface = "action_only_not_applicable"
    initial_sha256, initial_coverage = initial_state_record(result)
    return {
        "model_id": model_id, "pair_id": scene["pair_id"], "task": result["task"],
        "environment_seed": seed, "sampling_seed": int(result["sampling_seed"]),
        "prompt_family": "direct_command", "requested_relation": direction, "prompt": result["prompt"],
        "movable_model_name": result["object_name"], "reference_model_name": result["target_name"],
        "requested_success": success, "actions_executed": int(result["actions_executed"]),
        "wall_seconds": float(result["wall_seconds"]),
        "operational_wall_latency_valid": not bool(events),
        "runtime_intervention_ids": [event["id"] for event in events],
        "initial_dx_m": float(result["initial"]["object_minus_target_x"]),
        "initial_dy_m": float(result["initial"]["object_minus_target_y"]),
        "final_dx_m": float(result["final"]["object_minus_target_x"]),
        "final_dy_m": float(result["final"]["object_minus_target_y"]),
        "verified_pickup_proxy": pickup, "ever_entered_requested_region": bool(entered),
        "ever_released_in_requested_region": bool(released), "failure_stage": failure_stage,
        "physical_initial_state_sha256": initial_sha256,
        "initial_state_coverage": initial_coverage,
        "raw_result": file_record(result_path), "raw_trajectory": file_record(trajectory_path),
        "executed_video": file_record(Path(result["simulator_video"])),
        "imagined_future_artifact": future, "future_interface": future_interface,
        "action_trace": action_trace_record(result_path, result, is_prospective_scene(scene)),
    }


def compile_confirmation(
    input_root: Path, model_id: str, registry_path: Path,
    interventions_paths: Path | list[Path] | tuple[Path, ...] | None = None,
    invalid_attempts_paths: Path | list[Path] | tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    scenes = load_registry(registry_path, model_id)
    interventions, intervention_sources = load_interventions(interventions_paths, model_id)
    invalid_attempts, invalid_attempts_by_cell, invalid_sources = load_invalid_attempts(
        invalid_attempts_paths, model_id
    )
    paths = sorted(input_root.rglob("result.json"))
    if len(paths) != 20:
        raise RuntimeError(f"Expected exactly 20 confirmation result files, found {len(paths)}")
    episodes = [classify(path, model_id, scenes, interventions, invalid_attempts_by_cell) for path in paths]
    cells = {(row["environment_seed"], row["requested_relation"]) for row in episodes}
    expected_cells = {(seed, direction) for seed in scenes for direction in ("left", "right")}
    if cells != expected_cells:
        raise RuntimeError(f"Confirmation cell mismatch: missing={sorted(expected_cells - cells)}")
    pairs = []
    for seed, scene in sorted(scenes.items()):
        pair = {row["requested_relation"]: row for row in episodes if row["environment_seed"] == seed}
        if pair["left"]["physical_initial_state_sha256"] != pair["right"]["physical_initial_state_sha256"]:
            raise RuntimeError(f"Initial-state mismatch in {scene['pair_id']}")
        if pair["left"]["initial_state_coverage"] != pair["right"]["initial_state_coverage"]:
            raise RuntimeError(f"Initial-state coverage mismatch in {scene['pair_id']}")
        shift = pair["right"]["final_dx_m"] - pair["left"]["final_dx_m"]
        if is_prospective_scene(scene):
            action_metric, action_steps_used, action_metric_reason = first_ten_executed_action_rms(
                pair["left"]["action_trace"], pair["right"]["action_trace"]
            )
        else:
            action_metric = None
            action_steps_used = 0
            action_metric_reason = "historical_pair00_pair02_action_trace_not_required_by_preclarification"
        pairs.append({"pair_id": scene["pair_id"], "environment_seed": seed,
            "left_success": pair["left"]["requested_success"], "right_success": pair["right"]["requested_success"],
            "left_final_dx_m": pair["left"]["final_dx_m"], "right_final_dx_m": pair["right"]["final_dx_m"],
            "right_minus_left_endpoint_dx_m": shift,
            "endpoint_response_direction": "aligned" if shift > 0 else "anti_directed" if shift < 0 else "none",
            "physical_initial_state_sha256": pair["left"]["physical_initial_state_sha256"],
            "initial_state_coverage": pair["left"]["initial_state_coverage"],
            "first_ten_executed_action_rms": action_metric,
            "first_ten_executed_action_rms_steps_used": action_steps_used,
            "action_metric_unavailable_reason": action_metric_reason})
    by_direction = {}
    for direction in ("left", "right"):
        rows = [row for row in episodes if row["requested_relation"] == direction]
        successes = sum(row["requested_success"] for row in rows)
        by_direction[direction] = {"episodes": len(rows), "successes": successes,
            "success_wilson_95": wilson_95(successes, len(rows)), "started": len(rows),
            "verified_pickups": sum(row["verified_pickup_proxy"] for row in rows),
            "entered_requested_region": sum(row["ever_entered_requested_region"] for row in rows),
            "released_in_requested_region": sum(row["ever_released_in_requested_region"] for row in rows),
            "success_stage_count": successes}
    applied_runtime_ids = sorted(
        event_id for row in episodes for event_id in row["runtime_intervention_ids"]
    )
    selected_runtime_ids = sorted(
        event_id for source in intervention_sources for event_id in source["selected_event_ids"]
    )
    if applied_runtime_ids != selected_runtime_ids:
        raise RuntimeError(
            "Selected runtime intervention events do not map exactly to retained confirmation cells"
        )
    for source in intervention_sources:
        source["applied_event_ids"] = sorted(
            set(source["selected_event_ids"]) & set(applied_runtime_ids)
        )
    retained_invalid_ids = sorted(event["id"] for event in invalid_attempts)
    for source in invalid_sources:
        source["retained_event_ids"] = sorted(
            set(source["selected_event_ids"]) & set(retained_invalid_ids)
        )
    return {"schema_version": SCHEMA_VERSION, "compiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id, "model_label": MODEL_LABELS[model_id],
        "source_registry": {"path": str(registry_path), "sha256": sha256(registry_path)},
        "measurement": {"oracle_actions": 0, "dynamic_prompts": 0,
            "simulator_state_role": "post_action_scoring_and_visualization_only"},
        "intervention_ledger_sources": intervention_sources,
        "invalid_attempts": invalid_attempts, "invalid_attempt_ledger_sources": invalid_sources,
        "applied_runtime_intervention_ids": applied_runtime_ids,
        "retained_invalid_attempt_ids": retained_invalid_ids,
        "summary": {"episode_count": len(episodes), "pair_count": len(pairs), "successes": sum(row["requested_success"] for row in episodes),
            "by_direction": by_direction, "failure_stage_counts": dict(sorted(Counter(row["failure_stage"] for row in episodes).items())),
            "paired_endpoint_responses": pairs, "aligned_endpoint_pairs": sum(p["endpoint_response_direction"] == "aligned" for p in pairs),
            "operational_wall_latency_valid_episodes": sum(row["operational_wall_latency_valid"] for row in episodes),
            "operational_wall_latency_excluded_episodes": sum(not row["operational_wall_latency_valid"] for row in episodes),
            "invalid_attempt_count": len(invalid_attempts),
            "future_interface_counts": dict(Counter(row["future_interface"] for row in episodes)),
            "first_ten_executed_action_rms_coverage": {
                "available_pairs": sum(p["first_ten_executed_action_rms"] is not None for p in pairs),
                "prospective_pairs": 7,
                "total_pairs": 10,
                "coverage": f"{sum(p['first_ten_executed_action_rms'] is not None for p in pairs)}/10",
            }},
        "episodes": episodes}


def write_outputs(output: Path, compiled: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(compiled, indent=2, sort_keys=True) + "\n")
    columns = ["pair_id", "environment_seed", "sampling_seed", "requested_relation", "requested_success", "verified_pickup_proxy", "ever_entered_requested_region", "ever_released_in_requested_region", "final_dx_m", "failure_stage", "future_interface", "operational_wall_latency_valid"]
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n"); writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in compiled["episodes"])
    summary = compiled["summary"]
    lines = [f"# {compiled['model_label']} RoboTwin direct-command confirmation", "", "This 20-episode confirmation preserves the earlier six-episode direct gate as a separate artifact.", "", "| Direction | Success | Wilson 95% | Started | Pickup | Entry | Release |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for direction in ("left", "right"):
        row = summary["by_direction"][direction]; interval = row["success_wilson_95"]
        lines.append(f"| {direction.upper()} | {row['successes']}/10 | [{interval['lower']:.3f}, {interval['upper']:.3f}] | {row['started']} | {row['verified_pickups']} | {row['entered_requested_region']} | {row['released_in_requested_region']} |")
    lines.extend(["", "## Paired first-10 executed-action RMS", "", "| Pair | RMS | Availability |", "| --- | ---: | --- |"])
    for pair in summary["paired_endpoint_responses"]:
        rms = pair["first_ten_executed_action_rms"]
        lines.append(
            f"| {pair['pair_id']} | {rms:.6f} | {pair['first_ten_executed_action_rms_steps_used']} shared steps |"
            if rms is not None
            else f"| {pair['pair_id']} | N/A | {pair['action_metric_unavailable_reason']} |"
        )
    lines.extend(["", f"Invalid/partial attempts retained separately: {summary['invalid_attempt_count']}. Thermal latency exclusions: {summary['operational_wall_latency_excluded_episodes']}.", "Future interface counts: " + json.dumps(summary["future_interface_counts"], sort_keys=True)])
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-id", choices=sorted(MODEL_LABELS), required=True)
    parser.add_argument("--registry", type=Path, default=Path("artifacts/vla_wam_shared_v2/pilot/directional_expansion.json"))
    parser.add_argument(
        "--interventions", "--runtime-interventions", dest="interventions",
        action="append", type=Path,
        help="Runtime intervention ledger; repeat for shared/pilot and model-specific confirmation ledgers.",
    )
    parser.add_argument(
        "--invalid-attempts", action="append", type=Path,
        help="Invalid/partial-attempt ledger; repeat for every supplied model-aware source.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/vla_wam_shared_v2/pilot/results"))
    args = parser.parse_args()
    interventions = list(args.interventions or [])
    historical = Path("artifacts/vla_wam_shared_v2/pilot/runtime_interventions.json")
    if historical.is_file() and historical.resolve() not in {path.resolve() for path in interventions}:
        interventions.insert(0, historical)
    compiled = compile_confirmation(
        args.input_root, args.model_id, args.registry, interventions, args.invalid_attempts
    )
    write_outputs(args.output_dir / f"{args.model_id.removesuffix('_robotwin')}_direct_confirmation.json", compiled)


if __name__ == "__main__":
    main()
