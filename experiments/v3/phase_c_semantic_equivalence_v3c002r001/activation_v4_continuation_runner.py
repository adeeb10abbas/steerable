"""Resume frozen slots 00/01 after their A003 whole-block retries complete."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from experiments.v3.phase_c_semantic_equivalence_v3c002 import runner as base_runner
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import read_finite_json, require, validate_file_binding
from experiments.v3.phase_c_semantic_equivalence_v3c002r001 import runner as r001_runner  # noqa: F401
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_contract import require_a004_gate
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_retry_runner import _require_adapter


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def _remaining(cells, *, shard_index: int, shard_count: int):
    require(shard_count == 8 and 0 <= shard_index < 8, "A004 continuation requires exactly eight frozen slots")
    gate = read_finite_json(Path(_argument("--authorization-gate")))
    slot = f"repair-lane-{shard_index:02d}"
    remaining = gate.get("remaining_seed_blocks_by_lane", {}).get(slot)
    require(isinstance(remaining, list) and len(remaining) == len(set(remaining)), "A004 remaining seeds invalid")
    selected = [cell for cell in cells if cell.seed in set(remaining)]
    require(len(selected) == 4 * len(remaining), "A004 continuation split a frozen block")
    return selected


base_runner.require_released_gate = require_a004_gate
base_runner.grouped_shard = _remaining


def main() -> None:
    from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_pinned_push import install_contract_monkeypatches
    install_contract_monkeypatches()
    _require_adapter()
    base_runner.main()


if __name__ == "__main__":
    main()
