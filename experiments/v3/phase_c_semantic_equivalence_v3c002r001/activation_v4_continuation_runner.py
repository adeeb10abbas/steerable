"""Resume frozen slots 00/01 after their A003 whole-block retries complete."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from experiments.v3.phase_c_semantic_equivalence_v3c002 import runner as base_runner
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import read_finite_json, require, validate_file_binding
from experiments.v3.phase_c_semantic_equivalence_v3c002r001 import runner as r001_runner  # noqa: F401
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v3_lane_replacement_runner import (
    RETRY,
    require_released_replacement_gate,
)


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def _remaining(cells, *, shard_index: int, shard_count: int):
    require(shard_count == 8 and shard_index in (0, 1), "A004 continuation permits only slots 00/01")
    gate = read_finite_json(Path(_argument("--authorization-gate")))
    slot = f"repair-lane-{shard_index:02d}"
    replacement = gate["replacements"][slot]
    lane_records = []
    for binding in gate["lane_manifests"]:
        value = read_finite_json(Path(validate_file_binding(binding, "A004 lane manifest")["path"]))
        if value.get("lane_slot") == slot:
            lane_records.append(value)
    require(len(lane_records) == 1, "A004 continuation lane is not unique")
    seeds = set(lane_records[0]["assigned_seed_blocks"])
    retry_seed = RETRY[slot]
    marker_path = Path(lane_records[0]["raw_root"]) / "behavioral" / f"seed{retry_seed}" / "completed_block.json"
    marker = read_finite_json(marker_path)
    require(
        isinstance(marker, dict)
        and marker.get("status") == "completed_behavioral_block"
        and marker.get("attempt_root", "").endswith("/attempt002")
        and len(marker.get("raw_episodes", [])) == 4,
        "A004 continuation requires the complete A003 retry marker",
    )
    require(replacement.get("authorized_retry_seed") == retry_seed, "A004 retry lineage changed")
    selected = [cell for cell in cells if cell.seed in seeds]
    require(len(selected) == 4 * len(seeds), "A004 continuation split a frozen block")
    return selected


base_runner.require_released_gate = require_released_replacement_gate
base_runner.grouped_shard = _remaining


def main() -> None:
    base_runner.main()


if __name__ == "__main__":
    main()
