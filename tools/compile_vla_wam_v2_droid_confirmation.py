#!/usr/bin/env python3
"""Compile the frozen ten-seed pi0-FAST direct-command confirmation.

This deliberately writes a *new* confirmation evidence slice.  The six-cell
``pi0_fast_direct_gate`` artifacts remain an immutable record of the adaptive
gate that selected this follow-up.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from compile_vla_wam_v2_droid_pilot import (
    MODEL_ID,
    MODEL_LABEL,
    TASKS,
    file_record,
    load_episode,
    sha256,
)


SCHEMA_VERSION = "vla-wam-shared-v2-pi0-fast-direct-confirmation-v1"
EXPECTED_SEEDS = tuple(range(8300, 8310))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def wilson_95(successes: int, total: int) -> dict[str, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError(f"Invalid binomial count: {successes}/{total}")
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return {"confidence": 0.95, "lower": center - half, "upper": center + half}


def load_registry(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    registry = json.loads(path.read_text())
    if registry.get("model_id") != MODEL_ID:
        raise RuntimeError(f"Unexpected confirmation registry model: {registry.get('model_id')}")
    if registry.get("prompt_family") != "direct_command":
        raise RuntimeError("Confirmation registry is not direct-command only")
    if tuple(registry.get("completed_seeds", [])) != (8300, 8301, 8302) or tuple(
        registry.get("new_seeds", [])
    ) != tuple(range(8303, 8310)):
        raise RuntimeError("Confirmation registry does not preserve the frozen 3+7 seed split")
    prompts = registry.get("prompts", {})
    if set(prompts) != set(TASKS):
        raise RuntimeError("Confirmation registry must provide exactly LEFT and RIGHT prompts")
    cells = {
        (seed, direction): {
            "rendered_prompt": prompts[direction],
            "environment_seed": seed,
            "requested_relation": direction,
        }
        for seed in EXPECTED_SEEDS
        for direction in TASKS
    }
    return cells


def load_invalid_attempts(path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if path is None or not path.exists():
        return [], None
    payload = json.loads(path.read_text())
    events = payload.get("events")
    if not isinstance(events, list):
        raise RuntimeError(f"Invalid-attempt ledger must contain an events list: {path}")
    for event in events:
        if event.get("classification") not in {"technical_invalid", "partial"}:
            raise RuntimeError(f"Invalid-attempt ledger contains non-invalid event: {event}")
        if event.get("model_id", MODEL_ID) != MODEL_ID:
            raise RuntimeError(f"Invalid-attempt ledger contains another model: {event}")
        if event.get("behavioral_result_valid", False) is not False or event.get("wall_latency_valid", False) is not False:
            raise RuntimeError(f"Invalid/partial attempt must exclude behavior and wall latency: {event}")
    return events, file_record(path)


def _validate_source_log_hashes(event: dict[str, Any], ledger_path: Path) -> None:
    records = []
    if isinstance(event.get("source_log"), dict):
        records.append(event["source_log"])
    if isinstance(event.get("source_logs"), list):
        records.extend(event["source_logs"])
    if not records and event.get("raw_event_log") and event.get("raw_event_log_sha256"):
        records.append({"path": event["raw_event_log"], "sha256": event["raw_event_log_sha256"]})
    if not records:
        raise RuntimeError(f"Runtime intervention lacks source log hash provenance: {event}")
    for record in records:
        if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
            raise RuntimeError(f"Invalid source log record: {record}")
        path = Path(record["path"])
        if not path.is_absolute():
            path = ledger_path.parent / path
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Source thermal log path/hash mismatch: {path}")


def load_interventions(
    paths: list[Path] | tuple[Path, ...] | None,
) -> tuple[dict[tuple[int, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    by_cell: dict[tuple[int, str], list[dict[str, Any]]] = {}
    sources = []
    seen_ids: set[str] = set()
    for path in list(paths or []):
        if not path.is_file():
            raise RuntimeError(f"Supplied intervention ledger does not exist: {path}")
        payload = json.loads(path.read_text())
        events = payload.get("events")
        if not isinstance(events, list):
            raise RuntimeError(f"Intervention ledger must contain an events list: {path}")
        selected = [event for event in events if event.get("model_id") == MODEL_ID]
        if any("model_id" not in event for event in events):
            raise RuntimeError(f"Cannot model-filter intervention without model_id: {path}")
        source = {
            **file_record(path), "total_event_count": len(events),
            "selected_model_event_count": len(selected),
            "ignored_other_model_event_count": len(events) - len(selected),
            "selected_event_ids": [event.get("id") for event in selected],
        }
        sources.append(source)
        for event in selected:
            required = {
                "id", "model_id", "environment_seed", "requested_relation",
                "behavioral_result_valid", "wall_latency_valid", "events",
                "started_at_utc", "completed_at_utc", "max_temperature_c",
            }
            if not required <= set(event):
                raise RuntimeError(f"Runtime intervention lacks required thermal provenance: {event}")
            _validate_source_log_hashes(event, path)
            if event["id"] in seen_ids:
                raise RuntimeError(f"Duplicate runtime intervention ID across ledgers: {event['id']}")
            seen_ids.add(event["id"])
            if event["behavioral_result_valid"] is not True or event["wall_latency_valid"] is not False:
                raise RuntimeError(f"Runtime intervention must retain behavior and exclude wall latency: {event}")
            key = (int(event["environment_seed"]), event["requested_relation"])
            if key not in {(seed, direction) for seed in EXPECTED_SEEDS for direction in TASKS}:
                raise RuntimeError(f"Runtime intervention references an unregistered confirmation cell: {event}")
            by_cell.setdefault(key, []).append(event)
    return by_cell, sources


def validate_raw_intervention_flags(
    root: Path, direction: str, events: list[dict[str, Any]],
) -> None:
    task = TASKS[direction]
    result_rows = [json.loads(line) for line in (root / "episode_results.jsonl").read_text().splitlines() if line.strip()]
    result = next(row for row in result_rows if row["env_name"] == task)
    log = json.loads((root / task / "log_0_env0.json").read_text())
    applied_ids = {event["id"] for event in events}
    for record in (result, log):
        reported_ids = set(record.get("runtime_intervention_ids", [])) | set(record.get("thermal_intervention_ids", []))
        if not reported_ids <= applied_ids:
            raise RuntimeError(f"Raw result self-reports unrepresented runtime intervention IDs: {root}/{task}")
        reports_thermal = any(record.get(key) is True for key in ("thermally_intervened", "thermal_intervention", "thermal_paused"))
        if (reports_thermal or record.get("wall_latency_valid") is False) and not events:
            raise RuntimeError(f"Raw result self-reports thermal state without a ledger event: {root}/{task}")
        if events and record.get("wall_latency_valid") is True:
            raise RuntimeError(f"Raw result marks thermally affected wall latency valid: {root}/{task}")
        if record.get("behavioral_result_valid") is False or record.get("partial") is True:
            raise RuntimeError(f"Raw result contradicts valid-behavior intervention ledger: {root}/{task}")


def write_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    columns = [
        "model_id", "pair_id", "environment_seed", "sampling_seed", "requested_relation",
        "prompt", "requested_success", "verified_pickup_proxy",
        "ever_entered_requested_region", "ever_released_in_requested_region",
        "final_lateral_display_m", "requested_signed_final_offset_m", "failure_stage",
        "actions_executed", "wall_seconds", "operational_wall_latency_valid",
        "runtime_intervention_ids",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for episode in episodes:
            row = {key: episode[key] for key in columns}
            row["runtime_intervention_ids"] = ";".join(episode["runtime_intervention_ids"])
            writer.writerow(row)


def write_markdown(path: Path, compiled: dict[str, Any]) -> None:
    summary = compiled["summary"]
    lines = [
        "# pi0-FAST DROID direct-command directional confirmation",
        "",
        "This is the 20-episode confirmation selected after the preserved six-episode gate; it does not replace that pilot.",
        "",
        "## Directional results",
        "",
        "| Asked side | Released requested placement | Wilson 95% | Pickup | Entered requested region |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for direction in ("left", "right"):
        row = summary["by_direction"][direction]
        interval = row["success_wilson_95"]
        lines.append(
            f"| {direction.upper()} | {row['successes']}/{row['episodes']} | "
            f"[{interval['lower']:.3f}, {interval['upper']:.3f}] | "
            f"{row['verified_pickups']}/10 | {row['entered_requested_region']}/10 |"
        )
    lines.extend(["", "## Exact same-seed endpoint pairs", "", "| Seed | LEFT end | RIGHT end | RIGHT − LEFT | Alignment | First-10 action RMS |", "| ---: | ---: | ---: | ---: | --- | ---: |"])
    for pair in summary["paired_endpoint_responses"]:
        prefix = (
            f"| {pair['environment_seed']} | {pair['left_final_lateral_display_m']:+.3f} m | "
            f"{pair['right_final_lateral_display_m']:+.3f} m | "
            f"{pair['right_minus_left_endpoint_lateral_m']:+.3f} m | "
            f"{pair['endpoint_response_direction'].replace('_', ' ')} | "
        )
        lines.append(
            prefix + f"{pair['first_ten_action_rms']:.6f} ({pair['first_ten_action_rms_steps_used']} steps) |"
            if pair["first_ten_action_rms"] is not None
            else prefix + f"N/A ({pair['first_ten_action_rms_unavailable_reason']}) |"
        )
    lines.extend([
        "",
        f"Valid behavioral episodes: **{summary['episode_count']}/20**. "
        f"Invalid/partial attempts recorded separately: **{summary['invalid_attempt_count']}**.",
        f"Thermally affected wall-latency exclusions: **{summary['operational_wall_latency_excluded_episodes']}**; behavioral outcomes remain valid.",
        "Simulator state was used only after action execution for scoring and visualization.",
    ])
    path.write_text("\n".join(lines) + "\n")


def compile_confirmation(
    registry_path: Path,
    robolab_output: Path,
    trajectory_dir: Path,
    invalid_attempts_path: Path | None = None,
    interventions_paths: list[Path] | tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    cells = load_registry(registry_path)
    invalid_attempts, invalid_attempt_record = load_invalid_attempts(invalid_attempts_path)
    interventions, intervention_sources = load_interventions(interventions_paths)
    episodes: list[dict[str, Any]] = []
    actions: dict[tuple[int, str], np.ndarray] = {}
    fingerprints: dict[tuple[int, str], str] = {}
    for seed in EXPECTED_SEEDS:
        root = robolab_output / f"v2_pi0_fast_direct_seed{seed}"
        for direction in TASKS:
            cell_events = interventions.get((seed, direction), [])
            validate_raw_intervention_flags(root, direction, cell_events)
            episode, action, fingerprint = load_episode(
                seed=seed, direction=direction, root=root,
                cell=cells[(seed, direction)], trajectory_dir=trajectory_dir,
            )
            episode["runtime_intervention_ids"] = [event["id"] for event in cell_events]
            episode["operational_wall_latency_valid"] = not bool(cell_events)
            episodes.append(episode)
            actions[(seed, direction)] = action
            fingerprints[(seed, direction)] = fingerprint
    if len(episodes) != 20:
        raise RuntimeError(f"Expected 20 valid confirmation cells, found {len(episodes)}")

    pairs = []
    for seed in EXPECTED_SEEDS:
        left = next(row for row in episodes if row["environment_seed"] == seed and row["requested_relation"] == "left")
        right = next(row for row in episodes if row["environment_seed"] == seed and row["requested_relation"] == "right")
        if fingerprints[(seed, "left")] != fingerprints[(seed, "right")]:
            raise RuntimeError(f"Physical initial state mismatch inside confirmation pair {seed}")
        left_actions, right_actions = actions[(seed, "left")], actions[(seed, "right")]
        steps_used = min(len(left_actions), len(right_actions), 10)
        if steps_used == 0:
            action_rms, action_rms_reason = None, "no_common_executed_actions"
        elif left_actions[:steps_used].shape != right_actions[:steps_used].shape:
            action_rms, action_rms_reason = None, "paired_action_shape_mismatch"
        else:
            action_rms = float(np.sqrt(np.mean(np.square(left_actions[:steps_used] - right_actions[:steps_used]))))
            action_rms_reason = None
        shift = right["final_lateral_display_m"] - left["final_lateral_display_m"]
        pairs.append({
            "pair_id": f"droid_pair_seed_{seed}", "environment_seed": seed,
            "left_success": left["requested_success"], "right_success": right["requested_success"],
            "left_final_lateral_display_m": left["final_lateral_display_m"],
            "right_final_lateral_display_m": right["final_lateral_display_m"],
            "right_minus_left_endpoint_lateral_m": shift,
            "endpoint_response_direction": "aligned" if shift > 0 else "anti_directed" if shift < 0 else "none",
            "first_ten_action_rms": action_rms,
            "first_ten_action_rms_steps_used": steps_used,
            "first_ten_action_rms_unavailable_reason": action_rms_reason,
            "physical_initial_state_sha256": fingerprints[(seed, "left")],
        })

    by_direction: dict[str, dict[str, Any]] = {}
    for direction in TASKS:
        rows = [row for row in episodes if row["requested_relation"] == direction]
        successes = sum(row["requested_success"] for row in rows)
        by_direction[direction] = {
            "episodes": len(rows), "successes": successes,
            "success_wilson_95": wilson_95(successes, len(rows)),
            "verified_pickups": sum(row["verified_pickup_proxy"] for row in rows),
            "entered_requested_region": sum(row["ever_entered_requested_region"] for row in rows),
            "released_in_requested_region": sum(row["ever_released_in_requested_region"] for row in rows),
            "mean_requested_signed_final_offset_m": float(np.mean([row["requested_signed_final_offset_m"] for row in rows])),
        }
    summary = {
        "episode_count": len(episodes), "pair_count": len(pairs),
        "successes": sum(row["requested_success"] for row in episodes),
        "by_direction": by_direction,
        "failure_stage_counts": dict(sorted(Counter(row["failure_stage"] for row in episodes).items())),
        "paired_endpoint_responses": pairs,
        "aligned_endpoint_pairs": sum(pair["endpoint_response_direction"] == "aligned" for pair in pairs),
        "nonzero_first_chunk_pairs": sum((pair["first_ten_action_rms"] or 0) > 0 for pair in pairs),
        "invalid_attempt_count": len(invalid_attempts),
        "operational_wall_latency_valid_episodes": sum(row["operational_wall_latency_valid"] for row in episodes),
        "operational_wall_latency_excluded_episodes": sum(not row["operational_wall_latency_valid"] for row in episodes),
    }
    applied_ids = sorted(event_id for row in episodes for event_id in row["runtime_intervention_ids"])
    selected_ids = sorted(event_id for source in intervention_sources for event_id in source["selected_event_ids"])
    if applied_ids != selected_ids:
        raise RuntimeError("Selected runtime intervention events do not map exactly to retained cells")
    for source in intervention_sources:
        source["applied_event_ids"] = sorted(set(source["selected_event_ids"]) & set(applied_ids))
    return {
        "schema_version": SCHEMA_VERSION,
        "compiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID, "model_label": MODEL_LABEL,
        "source_registry": {"path": str(registry_path), "sha256": sha256(registry_path)},
        "measurement": {
            "oracle_actions": 0, "dynamic_prompts": 0, "subtask_progress_checking": False,
            "simulator_state_role": "post_action_scoring_and_visualization_only",
            "first_action_chunk_steps": 10, "future_interface": "none",
        },
        "invalid_attempts": invalid_attempts,
        "invalid_attempt_ledger": invalid_attempt_record,
        "intervention_ledger_sources": intervention_sources,
        "applied_runtime_intervention_ids": applied_ids,
        "summary": summary, "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("artifacts/vla_wam_shared_v2/pilot/pi0_fast_directional_expansion.json"))
    parser.add_argument("--robolab-output", type=Path, default=Path("/home/ali/projects/RoboLab/output"))
    parser.add_argument("--trajectory-dir", type=Path, default=Path("artifacts/vla_wam_shared_v2/pilot/pi0_fast_direct_confirmation/trajectories"))
    parser.add_argument("--invalid-attempts", type=Path)
    parser.add_argument(
        "--interventions", action="append", type=Path,
        help="Valid-behavior runtime intervention ledger; repeat for every supplied source.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/vla_wam_shared_v2/pilot/results/pi0_fast_direct_confirmation.json"))
    args = parser.parse_args()
    compiled = compile_confirmation(
        args.registry, args.robolab_output, args.trajectory_dir,
        args.invalid_attempts, args.interventions,
    )
    dump_json(args.output, compiled)
    write_csv(args.output.with_suffix(".csv"), compiled["episodes"])
    write_markdown(args.output.with_suffix(".md"), compiled)
    print(json.dumps(compiled["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
