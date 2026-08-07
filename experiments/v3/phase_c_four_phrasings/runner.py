#!/usr/bin/env python3
"""Create a released, whole-seed execution plan; never performs inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .contract import (
    EXPERIMENT_ID,
    MODEL_CONTRACTS,
    canonical_json_bytes,
    load_jsonl,
    select_whole_seed_blocks,
    sha256_file,
    validate_release_manifest,
)


def build_plan(
    *,
    cells_path: Path,
    registration_manifest_path: Path,
    release_manifest_path: Path,
    raw_root: Path,
    model_id: str,
    lane_index: int,
    lane_count: int,
) -> dict[str, Any]:
    registration_manifest_sha256 = sha256_file(registration_manifest_path)
    released = validate_release_manifest(
        release_manifest_path,
        model_id=model_id,
        registration_manifest_sha256=registration_manifest_sha256,
    )
    rows = select_whole_seed_blocks(
        load_jsonl(cells_path), model_id=model_id, lane_index=lane_index, lane_count=lane_count
    )
    cells: list[dict[str, Any]] = []
    for row in rows:
        cell_dir = raw_root / EXPERIMENT_ID.lower() / model_id / f"seed{row['seed']}" / f"order{row['within_seed_execution_order']:02d}_{row['prompt_family']}_{row['relation']}"
        cells.append(
            {
                "registered_cell_id": row["registered_cell_id"],
                "seed_block_id": row["seed_block_id"],
                "within_seed_execution_order": row["within_seed_execution_order"],
                "prompt": row["prompt"],
                "prompt_sha256": row["prompt_sha256"],
                "environment_seed": row["environment_seed"],
                "sampling_seed": row["sampling_seed"],
                "raw_cell_directory": str(cell_dir),
                "required_outputs": {
                    "behavioral_jsonl": str(cell_dir / "episode.jsonl"),
                    "executed_actions": str(cell_dir / "executed_actions.npy"),
                    "simulator_viewport_video": str(cell_dir / "viewport.mp4"),
                    "state_trace": str(cell_dir / "state_trace.jsonl"),
                    "decoded_future": "required_when_exposed_by_runtime",
                },
            }
        )
    return {
        "schema_version": "vla-wam-shared-v3c-four-phrasings-execution-plan-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": model_id,
        "checkpoint": MODEL_CONTRACTS[model_id]["checkpoint"],
        "checkpoint_revision": MODEL_CONTRACTS[model_id]["checkpoint_revision"],
        "runtime_identity_sha256": released.runtime_identity_sha256,
        "registration_manifest_sha256": registration_manifest_sha256,
        "release_manifest_sha256": released.release_manifest_sha256,
        "lane": {"index": lane_index, "count": lane_count},
        "seed_blocks": sorted({row["seed"] for row in rows}),
        "cell_count": len(cells),
        "cells": cells,
        "execution_status": "plan_only_model_specific_bridge_required",
        "inference_launched": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--registration-manifest", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--model-id", choices=tuple(MODEL_CONTRACTS), required=True)
    parser.add_argument("--lane-index", type=int, default=0)
    parser.add_argument("--lane-count", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(
        cells_path=args.cells,
        registration_manifest_path=args.registration_manifest,
        release_manifest_path=args.release_manifest,
        raw_root=args.raw_root,
        model_id=args.model_id,
        lane_index=args.lane_index,
        lane_count=args.lane_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(plan))
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

