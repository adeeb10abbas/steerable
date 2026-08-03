#!/usr/bin/env python3
"""Compile a hash-bearing subset of prospective RoboTwin confirmation pairs.

This is a handoff-safe intermediate artifact, not a replacement for the frozen
20-episode confirmation compiler.  It is useful when all newly authorized raw
episodes are present on a PVC but older raw pilot episodes remain on the source
host.  Missing historical evidence is never synthesized or counted as zero.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compile_vla_wam_v2_robotwin_confirmation import (
    MODEL_LABELS,
    classify,
    file_record,
    first_ten_executed_action_rms,
    load_directional_fixtures,
    load_interventions,
    load_invalid_attempts,
    load_registry,
    wilson_95,
)


SCHEMA_VERSION = "vla-wam-shared-v2-robotwin-prospective-slice-v1"


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("--expected-seeds must contain unique comma-separated integers")
    return seeds


def compile_slice(
    input_root: Path,
    model_id: str,
    registry_path: Path,
    expected_seeds: list[int],
    interventions_paths: list[Path] | None = None,
    invalid_attempts_paths: list[Path] | None = None,
    fixtures_path: Path | None = None,
    robotwin_root: Path | None = None,
) -> dict[str, Any]:
    scenes = load_registry(registry_path, model_id)
    fixtures = load_directional_fixtures(fixtures_path)
    unknown = sorted(set(expected_seeds) - set(scenes))
    if unknown:
        raise RuntimeError(f"Expected seeds are not in the frozen registry: {unknown}")
    if any(int(seed) < 4300003 for seed in expected_seeds):
        raise RuntimeError("A prospective slice may not absorb completed-pilot seeds")

    interventions, intervention_sources = load_interventions(interventions_paths, model_id)
    invalid_attempts, invalid_by_cell, invalid_sources = load_invalid_attempts(
        invalid_attempts_paths, model_id
    )
    paths = sorted(input_root.rglob("result.json"))
    expected_cells = {(seed, direction) for seed in expected_seeds for direction in ("left", "right")}
    episodes = [
        classify(
            path, model_id, scenes, interventions, invalid_by_cell,
            fixtures, robotwin_root,
        )
        for path in paths
        if json.loads(path.read_text()).get("environment_seed") in expected_seeds
    ]
    observed_cells = {(row["environment_seed"], row["requested_relation"]) for row in episodes}
    if len(episodes) != len(expected_cells) or observed_cells != expected_cells:
        raise RuntimeError(
            f"Prospective slice mismatch: missing={sorted(expected_cells - observed_cells)} "
            f"extra={sorted(observed_cells - expected_cells)}"
        )

    pairs = []
    for seed in sorted(expected_seeds):
        scene = scenes[seed]
        pair = {row["requested_relation"]: row for row in episodes if row["environment_seed"] == seed}
        if pair["left"]["physical_initial_state_sha256"] != pair["right"]["physical_initial_state_sha256"]:
            raise RuntimeError(f"Initial-state mismatch in {scene['pair_id']}")
        if pair["left"]["initial_state_coverage"] != pair["right"]["initial_state_coverage"]:
            raise RuntimeError(f"Initial-state coverage mismatch in {scene['pair_id']}")
        action_rms, action_steps, unavailable_reason = first_ten_executed_action_rms(
            pair["left"]["action_trace"], pair["right"]["action_trace"]
        )
        endpoint_shift = pair["right"]["final_dx_m"] - pair["left"]["final_dx_m"]
        pairs.append(
            {
                "pair_id": scene["pair_id"],
                "environment_seed": seed,
                "left_success": pair["left"]["requested_success"],
                "right_success": pair["right"]["requested_success"],
                "left_final_dx_m": pair["left"]["final_dx_m"],
                "right_final_dx_m": pair["right"]["final_dx_m"],
                "right_minus_left_endpoint_dx_m": endpoint_shift,
                "endpoint_response_direction": (
                    "aligned" if endpoint_shift > 0 else "anti_directed" if endpoint_shift < 0 else "none"
                ),
                "physical_initial_state_sha256": pair["left"]["physical_initial_state_sha256"],
                "initial_state_coverage": pair["left"]["initial_state_coverage"],
                "first_ten_executed_action_rms": action_rms,
                "first_ten_executed_action_rms_steps_used": action_steps,
                "action_metric_unavailable_reason": unavailable_reason,
            }
        )

    by_direction: dict[str, dict[str, Any]] = {}
    for direction in ("left", "right"):
        rows = [row for row in episodes if row["requested_relation"] == direction]
        successes = sum(row["requested_success"] for row in rows)
        by_direction[direction] = {
            "episodes": len(rows),
            "successes": successes,
            "success_wilson_95": wilson_95(successes, len(rows)),
            "verified_pickups": sum(row["verified_pickup_proxy"] for row in rows),
            "entered_requested_region": sum(row["ever_entered_requested_region"] for row in rows),
            "released_in_requested_region": sum(row["ever_released_in_requested_region"] for row in rows),
        }

    applied_runtime_ids = sorted(
        event_id for row in episodes for event_id in row["runtime_intervention_ids"]
    )
    selected_runtime_ids = sorted(
        event_id for source in intervention_sources for event_id in source["selected_event_ids"]
    )
    if applied_runtime_ids != selected_runtime_ids:
        raise RuntimeError("Selected runtime events do not map exactly to retained slice cells")
    for source in intervention_sources:
        source["applied_event_ids"] = sorted(
            set(source["selected_event_ids"]) & set(applied_runtime_ids)
        )
    retained_invalid_ids = sorted(event["id"] for event in invalid_attempts)
    for source in invalid_sources:
        source["retained_event_ids"] = sorted(
            set(source["selected_event_ids"]) & set(retained_invalid_ids)
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "compiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "model_label": MODEL_LABELS[model_id],
        "source_registry": file_record(registry_path),
        "source_directional_fixtures": file_record(fixtures_path) if fixtures_path is not None else None,
        "robotwin_root": str(robotwin_root) if robotwin_root is not None else None,
        "input_root": str(input_root),
        "expected_environment_seeds": sorted(expected_seeds),
        "measurement": {
            "oracle_actions": 0,
            "dynamic_prompts": 0,
            "simulator_state_role": "post_action_scoring_and_visualization_only",
        },
        "intervention_ledger_sources": intervention_sources,
        "applied_runtime_intervention_ids": applied_runtime_ids,
        "invalid_attempts": invalid_attempts,
        "invalid_attempt_ledger_sources": invalid_sources,
        "retained_invalid_attempt_ids": retained_invalid_ids,
        "summary": {
            "episode_count": len(episodes),
            "pair_count": len(pairs),
            "successes": sum(row["requested_success"] for row in episodes),
            "by_direction": by_direction,
            "failure_stage_counts": dict(sorted(Counter(row["failure_stage"] for row in episodes).items())),
            "paired_endpoint_responses": pairs,
            "aligned_endpoint_pairs": sum(pair["endpoint_response_direction"] == "aligned" for pair in pairs),
            "invalid_attempt_count": len(invalid_attempts),
            "future_interface_counts": dict(Counter(row["future_interface"] for row in episodes)),
        },
        "episodes": episodes,
        "claim_boundary": (
            "This artifact compiles only the listed prospective PVC-resident pairs. It does not replace "
            "the frozen twenty-episode confirmation and authorizes no model-level ten-scene claim."
        ),
    }


def write_outputs(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    columns = [
        "pair_id", "environment_seed", "requested_relation", "requested_success",
        "actions_executed", "final_dx_m", "failure_stage", "future_interface",
    ]
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in payload["episodes"])
    summary = payload["summary"]
    lines = [
        f"# {payload['model_label']} prospective RoboTwin slice",
        "",
        payload["claim_boundary"],
        "",
        f"Episodes: {summary['episode_count']}; pairs: {summary['pair_count']}; successes: {summary['successes']}.",
        "",
        "| Direction | Success | Wilson 95% |",
        "| --- | ---: | --- |",
    ]
    for direction in ("left", "right"):
        row = summary["by_direction"][direction]
        interval = row["success_wilson_95"]
        lines.append(
            f"| {direction.upper()} | {row['successes']}/{row['episodes']} | "
            f"[{interval['lower']:.3f}, {interval['upper']:.3f}] |"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-id", choices=sorted(MODEL_LABELS), required=True)
    parser.add_argument("--expected-seeds", type=parse_seeds, required=True)
    parser.add_argument(
        "--registry", type=Path,
        default=Path("artifacts/vla_wam_shared_v2/pilot/directional_expansion.json"),
    )
    parser.add_argument(
        "--fixtures", type=Path,
        default=Path("artifacts/vla_wam_shared_v2/pilot/directional_fixture_validation.json"),
    )
    parser.add_argument("--robotwin-root", type=Path)
    parser.add_argument("--runtime-interventions", action="append", type=Path)
    parser.add_argument("--invalid-attempts", action="append", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compile_slice(
        args.input_root,
        args.model_id,
        args.registry,
        args.expected_seeds,
        args.runtime_interventions,
        args.invalid_attempts,
        args.fixtures,
        args.robotwin_root,
    )
    write_outputs(args.output, payload)


if __name__ == "__main__":
    main()
