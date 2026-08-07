#!/usr/bin/env python3
"""Compile one complete V3-B005 LEFT/RIGHT pair from retained raw evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.v3.cosmos_nano_lateral_sweep.runtime_adapter import (
    AMENDMENT_ID,
    MODEL_ID,
    RuntimeContractError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from tools.vla_wam_v3_episode_schema import parse_jsonl_record


SCHEMA = "vla-wam-shared-v3b005-nano-lateral-pair-diagnostics-v1"


def _load_one(path: Path, relation: str) -> dict[str, Any]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise RuntimeContractError(f"pair input must contain one episode: {path}")
    row = parse_jsonl_record(lines[0])
    if (
        row.get("behavioral_result_valid") is not True
        or row.get("requested_relation") != relation
        or row.get("model_id") != MODEL_ID
        or row.get("amendment_id") != AMENDMENT_ID
    ):
        raise RuntimeContractError(f"pair input is not the exact {relation.upper()} episode")
    return row


def _action(row: Mapping[str, Any]) -> np.ndarray:
    record = row.get("artifacts", {}).get("executed_action_trace", {})
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise RuntimeContractError("executed-action evidence hash changed")
    value = np.load(path, allow_pickle=False)
    if value.ndim != 2 or value.shape[1] != 8 or not np.isfinite(value).all():
        raise RuntimeContractError("executed-action evidence must be finite [N,8]")
    return value


def build_pair(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = _load_one(left_path, "left")
    right = _load_one(right_path, "right")
    for key in ("pair_id", "environment_seed", "level_index", "initial_state_sha256"):
        if left.get(key) != right.get(key):
            raise RuntimeContractError(f"matched pair differs for {key}")
    left_diag = left.get("nano_v3b005_diagnostics")
    right_diag = right.get("nano_v3b005_diagnostics")
    if not isinstance(left_diag, Mapping) or not isinstance(right_diag, Mapping):
        raise RuntimeContractError("matched pair lacks V3-B005 episode diagnostics")
    left_action, right_action = _action(left), _action(right)
    common = min(len(left_action), len(right_action))
    delta = left_action[:common].astype(np.float64) - right_action[:common].astype(np.float64)
    endpoint_d = float(left_diag["signed_final_lateral_offset_m"]) - float(
        right_diag["signed_final_lateral_offset_m"]
    )
    depth_b = float(right_diag["requested_side_depth_m"]) - float(
        left_diag["requested_side_depth_m"]
    )
    return {
        "schema_version": SCHEMA,
        "study_id": left["study_id"],
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "seed": left["environment_seed"],
        "level_index": left["level_index"],
        "reference_object_initial_lateral_position_y_m": left[
            "reference_object_initial_lateral_position_y_m"
        ],
        "matched_block_id": left["pair_id"],
        "left_registered_cell_id": left["registered_cell_id"],
        "right_registered_cell_id": right["registered_cell_id"],
        "initial_state_sha256": left["initial_state_sha256"],
        "endpoint_redirection_D_m": endpoint_d,
        "endpoint_shift_m": endpoint_d,
        "requested_side_depth_contrast_B_m": depth_b,
        "left_success": bool(left_diag["success"]),
        "right_success": bool(right_diag["success"]),
        "right_minus_left_success": int(bool(right_diag["success"])) - int(bool(left_diag["success"])),
        "executed_actions_distinct": not np.array_equal(left_action[:common], right_action[:common]),
        "action_distinct": not np.array_equal(left_action[:common], right_action[:common]),
        "action_distinct_definition": "bitwise inequality on the complete common executed prefix",
        "left_executed_action_count": int(len(left_action)),
        "right_executed_action_count": int(len(right_action)),
        "common_prefix_action_count": int(common),
        "common_prefix_action_rms": float(math.sqrt(float(np.mean(delta * delta)))),
        "source": {
            "left_raw_jsonl_sha256": sha256_file(left_path),
            "right_raw_jsonl_sha256": sha256_file(right_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-jsonl", type=Path, required=True)
    parser.add_argument("--right-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    value = build_pair(args.left_jsonl, args.right_jsonl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "path": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "canonical_sha256": sha256_bytes(canonical_json_bytes(value)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
