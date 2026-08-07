#!/usr/bin/env python3
"""Derive immutable matched-pair diagnostics after both V3-D001 cells exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.v3.pi05_stochastic_v3d001.contract import (
    QUEUE_SHA256, RELEASE_MANIFEST_SHA256, ContractError, load_release,
    sha256_file,
)
from tools.vla_wam_v3_episode_schema import parse_jsonl_record


def _one(path: Path, cell_id: str) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ContractError(f"matched cell must contain one JSONL row: {path}")
    record = parse_jsonl_record(lines[0])
    manifest = json.loads(path.with_name(path.name+".manifest.json").read_text(encoding="utf-8"))
    if record.get("registered_cell_id") != cell_id or manifest.get("row_count") != 1 or manifest.get("jsonl_sha256") != sha256_file(path):
        raise ContractError("matched cell identity or post-close manifest changed")
    return record


def compile_pair(*, repo_root: Path, release_manifest: Path, block_id: str,
                 left_jsonl: Path, right_jsonl: Path, output: Path) -> dict[str, Any]:
    release = load_release(repo_root, release_manifest)
    cells = [cell for cell in release.cells if cell.block_id == block_id]
    if len(cells) != 2 or {cell.relation for cell in cells} != {"left", "right"}:
        raise ContractError("block is not one exact released LEFT/RIGHT pair")
    by_relation = {cell.relation: cell for cell in cells}
    left = _one(left_jsonl.resolve(), by_relation["left"].cell_id)
    right = _one(right_jsonl.resolve(), by_relation["right"].cell_id)
    if left["initial_state_sha256"] != right["initial_state_sha256"]:
        raise ContractError("matched V3-D001 pair does not share an identical reset")
    left_actions = np.load(left["artifacts"]["executed_action_trace"]["path"], allow_pickle=False)
    right_actions = np.load(right["artifacts"]["executed_action_trace"]["path"], allow_pickle=False)
    for value, record in ((left_actions, left), (right_actions, right)):
        if value.shape != (record["actions_executed"], 8) or not np.isfinite(value).all():
            raise ContractError("matched action artifact changed")
    shared = min(10, len(left_actions), len(right_actions))
    delta = left_actions[:shared].astype(np.float64) - right_actions[:shared].astype(np.float64)
    left_offset = float(left["measurements"]["signed_final_lateral_offset_m"])
    right_offset = float(right["measurements"]["signed_final_lateral_offset_m"])
    row = {
        "schema_version": "vla-wam-shared-v3d001-pi05-matched-pair-diagnostics-v1",
        "study_id": left["study_id"], "registration_id": "V3-D001",
        "matched_stochastic_block_id": block_id,
        "environment_seed": left["environment_seed"],
        "shared_policy_sampling_seed_index": left["shared_policy_sampling_seed_index"],
        "left_registered_cell_id": left["registered_cell_id"],
        "right_registered_cell_id": right["registered_cell_id"],
        "left_raw_episode_jsonl": {"path": str(left_jsonl.resolve()), "sha256": sha256_file(left_jsonl.resolve()), "bytes": left_jsonl.stat().st_size},
        "right_raw_episode_jsonl": {"path": str(right_jsonl.resolve()), "sha256": sha256_file(right_jsonl.resolve()), "bytes": right_jsonl.stat().st_size},
        "identical_reset": True, "initial_state_sha256": left["initial_state_sha256"],
        "left_signed_final_lateral_offset_m": left_offset,
        "right_signed_final_lateral_offset_m": right_offset,
        "endpoint_shift_right_minus_left_m": right_offset - left_offset,
        "action_distinct": bool(np.any(delta != 0.0)),
        "action_distinct_prefix_steps": shared,
        "action_prefix_l2": float(np.linalg.norm(delta)),
        "action_prefix_max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "release_manifest_sha256": RELEASE_MANIFEST_SHA256,
        "queue_sha256": QUEUE_SHA256,
        "analysis_unit": "matched LEFT/RIGHT policy-sampling block nested within one environment seed",
    }
    output = output.resolve()
    manifest = output.with_name(output.name+".manifest.json")
    if output.exists() or manifest.exists():
        raise FileExistsError(f"refusing to overwrite matched V3-D001 evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":"))+"\n", encoding="utf-8")
    manifest.write_text(json.dumps({
        "schema_version": "vla-wam-shared-v3d001-pi05-matched-pair-manifest-v1",
        "matched_stochastic_block_id": block_id, "row_count": 1,
        "json_sha256": sha256_file(output), "json_bytes": output.stat().st_size,
    }, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return {"pair": str(output), "manifest": str(manifest), "action_distinct": row["action_distinct"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--block-id", required=True)
    parser.add_argument("--left-jsonl", type=Path, required=True)
    parser.add_argument("--right-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compile_pair(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
