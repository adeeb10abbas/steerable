#!/usr/bin/env python3
"""Exact C002 behavioral adapter admitted only by the A003 replacement gate."""

from __future__ import annotations

import runpy

import experiments.v3.phase_c_semantic_equivalence_v3c002.contract as parent_contract
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v3_lane_replacement_runner import (
    require_released_replacement_gate,
)


parent_contract.require_released_gate = require_released_replacement_gate
runpy.run_module(
    "experiments.v3.phase_c_semantic_equivalence_v3c002.droid_behavioral_adapter",
    run_name="__main__",
)
