#!/usr/bin/env python3
"""Run the exact e2d9 C002 adapter after A004 gate admission."""

from __future__ import annotations

import importlib
import runpy
from pathlib import Path
import subprocess
import sys

from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_contract import require_a004_gate


def _value(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


study_root = Path(_value("--study-root")).resolve()
target = study_root / "experiments/v3/phase_c_semantic_equivalence_v3c002/droid_behavioral_adapter.py"
if subprocess.check_output(["git", "-C", str(study_root), "rev-parse", "HEAD"], text=True).strip() != "e2d9ae3904b4a08e549c784903c167a4213d3d47":
    raise RuntimeError("A004 study root is not exact e2d9")
if not target.is_file():
    raise RuntimeError("A004 exact e2d9 adapter is absent")
for name in list(sys.modules):
    if name.startswith("experiments.v3.phase_c_semantic_equivalence_v3c002") and "v3c002r001" not in name:
        del sys.modules[name]
sys.path.insert(0, str(study_root))
parent_contract = importlib.import_module("experiments.v3.phase_c_semantic_equivalence_v3c002.contract")
parent_contract.require_released_gate = require_a004_gate
runpy.run_path(str(target), run_name="__main__")
