"""Universal A004 whole-block retry runner with exact adapter enforcement."""

from __future__ import annotations

from pathlib import Path
import sys

from experiments.v3.phase_c_semantic_equivalence_v3c002 import runner as base_runner
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import read_finite_json, require, sha256_file
from experiments.v3.phase_c_semantic_equivalence_v3c002r001 import runner as r001_runner  # noqa: F401
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_contract import ALL_SLOTS, admitted_slots, require_a004_gate
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_pinned_push import install_contract_monkeypatches


ADAPTER_MODULE = "experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_replacement_adapter"
ADAPTER_SHA = "30b5cbb15035ff362a14b7ecf28df3ec9b5f7c675262c2f4b383c490d192135f"
RETRY = {
    "repair-lane-00": 12060, "repair-lane-01": 12101,
    "repair-lane-02": 12128, "repair-lane-03": 12177,
    "repair-lane-04": 12156, "repair-lane-05": 12107,
    "repair-lane-06": 12176, "repair-lane-07": 12112,
}


def _require_adapter() -> None:
    tail = sys.argv[sys.argv.index("--adapter-command") + 1:]
    require(len(tail) >= 3 and tail[1:3] == ["-m", ADAPTER_MODULE], "A004 retry adapter command changed")
    adapter = Path(__file__).with_name("activation_v4_replacement_adapter.py")
    require(sha256_file(adapter) == ADAPTER_SHA, "A004 retry adapter source changed")


def _retry_only(cells, *, shard_index: int, shard_count: int):
    require(shard_count == 8 and 0 <= shard_index < 8, "A004 retry requires one frozen slot of eight")
    slot = ALL_SLOTS[shard_index]
    gate_path = Path(sys.argv[sys.argv.index("--authorization-gate") + 1]).resolve()
    require(slot in admitted_slots(gate_path), "A004 retry gate does not admit this slot")
    selected = [cell for cell in cells if cell.seed == RETRY[slot]]
    require(len(selected) == 4, "A004 retry split the frozen block")
    return selected


def _attempt002(root: Path, seed: int) -> Path:
    require(seed in RETRY.values(), "A004 retry seed changed")
    seed_root = Path(root) / f"seed{seed}"
    require((seed_root / "attempt001").is_dir(), "A004 retained attempt001 is absent")
    require(not (seed_root / "completed_block.json").exists(), "A004 refuses a completed retry seed")
    require(not (seed_root / "attempt002").exists(), "A004 retry attempt002 already exists")
    return seed_root / "attempt002"


_dispatch = base_runner._dispatch_block


def _dispatch_retry(*, block, args, registration_sha, queue_sha):
    seed = block[0].seed
    expected = _attempt002(Path(args.raw_root) / args.authorization_mode, seed)
    _dispatch(block=block, args=args, registration_sha=registration_sha, queue_sha=queue_sha)
    marker_path = Path(args.raw_root) / args.authorization_mode / f"seed{seed}" / "completed_block.json"
    marker = read_finite_json(marker_path)
    require(isinstance(marker, dict) and marker.get("attempt_root") == str(expected.resolve()), "A004 marker does not bind attempt002")
    raws = marker.get("raw_episodes")
    require(isinstance(raws, list) and len(raws) == 4 and all(Path(str(row.get("path", ""))).resolve().is_relative_to(expected.resolve()) for row in raws), "A004 retry marker reuses partial data")


base_runner.require_released_gate = require_a004_gate
base_runner.grouped_shard = _retry_only
base_runner._next_attempt_root = _attempt002
base_runner._dispatch_block = _dispatch_retry


def main() -> None:
    install_contract_monkeypatches()
    _require_adapter()
    base_runner.main()


if __name__ == "__main__":
    main()
