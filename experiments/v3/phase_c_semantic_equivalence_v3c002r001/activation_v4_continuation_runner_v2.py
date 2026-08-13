"""A004 continuation runner without retry-runner import side effects."""

from __future__ import annotations

from pathlib import Path
import sys

from experiments.v3.phase_c_semantic_equivalence_v3c002 import runner as base_runner
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import read_finite_json, require, sha256_file
from experiments.v3.phase_c_semantic_equivalence_v3c002r001 import runner as r001_runner  # noqa: F401
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_contract import require_a004_gate


ADAPTER_MODULE = "experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_replacement_adapter"
ADAPTER_SHA = "ca3bed4c272533e69b3264c6219577dd76a2de1e9d16ac56753c826c24e078f3"


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def _require_adapter() -> None:
    tail = sys.argv[sys.argv.index("--adapter-command") + 1:]
    require(len(tail) >= 3 and tail[1:3] == ["-m", ADAPTER_MODULE], "A004 continuation adapter command changed")
    adapter = Path(__file__).with_name("activation_v4_replacement_adapter.py")
    require(sha256_file(adapter) == ADAPTER_SHA, "A004 continuation adapter source changed")


def _remaining(cells, *, shard_index: int, shard_count: int):
    require(shard_count == 8 and 0 <= shard_index < 8, "A004 continuation requires exactly eight frozen slots")
    gate = read_finite_json(Path(_argument("--authorization-gate")))
    slot = f"repair-lane-{shard_index:02d}"
    remaining = gate.get("remaining_seed_blocks_by_lane", {}).get(slot)
    require(isinstance(remaining, list) and len(remaining) == len(set(remaining)), "A004 remaining seeds invalid")
    selected = [cell for cell in cells if cell.seed in set(remaining)]
    require(len(selected) == 4 * len(remaining), "A004 continuation split a frozen block")
    return selected


# Importing the R001 runner installs the frozen block-local provenance writer and
# leaves the parent runner's ordinary attempt allocator intact.  This module
# deliberately never imports activation_v4_retry_runner.
base_runner.require_released_gate = require_a004_gate
base_runner.grouped_shard = _remaining
require(base_runner._next_attempt_root.__module__ == base_runner.__name__, "A004 continuation attempt allocator was replaced")
require(base_runner._dispatch_block.__module__ == r001_runner.__name__, "A004 continuation provenance dispatch was replaced")


def main() -> None:
    from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_pinned_push import install_contract_monkeypatches
    install_contract_monkeypatches()
    _require_adapter()
    base_runner.main()


if __name__ == "__main__":
    main()
