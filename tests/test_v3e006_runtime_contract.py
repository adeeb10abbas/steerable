from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.v3.phase_e.canonical_stage_localization_v3e006.runtime_contract import (
    RuntimeContractError,
    assert_observed_runtime,
    expected_observation,
    load_runtime_contract,
)


PATH = Path(
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/gates/"
    "exact_pi05_runtime_contract.json"
)


def test_frozen_runtime_contract_round_trip() -> None:
    value = load_runtime_contract(PATH, hashlib.sha256(PATH.read_bytes()).hexdigest())
    assert value["checkpoint_payload_sha256"] == "b193b28b05f9755e24d44a6f5cf3185ca23c2ad3da6c5913370379c82570fbf6"
    assert value["components"]["policy_server"]["contract"]["launcher_sha256"] == "cd415e3a98da977f395242c24bb8f3d3187eb4cc3bf53c5dc659d190e6934051"


def test_observed_runtime_fails_closed() -> None:
    value = json.loads(PATH.read_text())
    observed = expected_observation(value)
    assert_observed_runtime(observed, value)
    observed["open_loop_horizon"] = 14
    with pytest.raises(RuntimeContractError):
        assert_observed_runtime(observed, value)
