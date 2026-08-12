#!/usr/bin/env python3
"""Run the byte-identical C002 adapter under the prospective R001 gate."""

from __future__ import annotations

import runpy

import experiments.v3.phase_c_semantic_equivalence_v3c002.contract as parent_contract
from .contract import (
    require_model_blind_preflight_authorization,
    require_released_gate,
    require_smoke_authorization,
)


parent_contract.require_model_blind_preflight_authorization = require_model_blind_preflight_authorization
parent_contract.require_smoke_authorization = require_smoke_authorization
parent_contract.require_released_gate = require_released_gate
runpy.run_module(
    "experiments.v3.phase_c_semantic_equivalence_v3c002.droid_behavioral_adapter",
    run_name="__main__",
)
