#!/usr/bin/env python3
"""Run the exact e2d9 C002 adapter after A004 gate admission."""

from __future__ import annotations

import importlib
import runpy
from pathlib import Path
import subprocess
import sys

from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_contract import admitted_slots, lane_records, require_a004_gate
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_pinned_push import install_contract_monkeypatches


E2_COMMIT = "e2d9ae3904b4a08e549c784903c167a4213d3d47"
E2_ADAPTER_SHA256 = "d1c85641b060bccd9267e2e12516b43f0576519aabd281b83f5e1c923777c47d"


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


study_root = Path(_value("--study-root")).resolve()
target = study_root / "experiments/v3/phase_c_semantic_equivalence_v3c002/droid_behavioral_adapter.py"
if subprocess.check_output(["git", "-C", str(study_root), "rev-parse", "HEAD"], text=True).strip() != E2_COMMIT:
    raise RuntimeError("A004 study root is not exact e2d9")
if not target.is_file() or _sha256(target) != E2_ADAPTER_SHA256:
    raise RuntimeError("A004 exact e2d9 adapter bytes changed")
install_contract_monkeypatches()
gate_path = Path(_value("--authorization-gate")).resolve()
lane_id = _value("--lane-id")
lane_matches = [(slot, value) for slot, (_, value) in lane_records(__import__("json").loads(gate_path.read_text())).items() if value.get("lane_id") == lane_id]
if len(lane_matches) != 1 or lane_matches[0][0] not in admitted_slots(gate_path):
    raise RuntimeError("A004 request-zero lane identity is not admitted by its exact gate")
for name in list(sys.modules):
    if name.startswith("experiments.v3.phase_c_semantic_equivalence_v3c002") and "v3c002r001" not in name:
        del sys.modules[name]
sys.path.insert(0, str(study_root))
parent_contract = importlib.import_module("experiments.v3.phase_c_semantic_equivalence_v3c002.contract")
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_pinned_push import verify_parent_pushed_source
parent_contract._verify_pushed_source_commit = verify_parent_pushed_source
parent_contract.require_released_gate = require_a004_gate
runpy.run_path(str(target), run_name="__main__")
