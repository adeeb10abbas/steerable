"""A003 retry runner with exact A004 child-adapter enforcement."""

from __future__ import annotations

from pathlib import Path
import sys

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import require, sha256_file
from experiments.v3.phase_c_semantic_equivalence_v3c002r001 import activation_v3_lane_replacement_runner as a003


ADAPTER_MODULE = "experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_replacement_adapter"
ADAPTER_SHA = "87010f58728363cae86d7a8317b57a151a7bb12509f7fb30ea438b5fe47e92a4"


def _require_adapter() -> None:
    tail = sys.argv[sys.argv.index("--adapter-command") + 1:]
    require(len(tail) >= 3 and tail[1:3] == ["-m", ADAPTER_MODULE], "A004 retry adapter command changed")
    adapter = Path(__file__).with_name("activation_v4_replacement_adapter.py")
    require(sha256_file(adapter) == ADAPTER_SHA, "A004 retry adapter source changed")


def main() -> None:
    _require_adapter()
    a003.main()


if __name__ == "__main__":
    main()
