from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.validate_v3e006_r010_closure import validate_compact_contract


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r010/results/results.json"


def test_r010_closure_contract_and_mutations() -> None:
    value = json.loads(RESULTS.read_text(encoding="utf-8"))
    validate_compact_contract(value)
    for key, replacement in (
        ("geometry_attachment_preflight_count", 0),
        ("diagnostic_evaluation_count", 1),
        ("candidate_pair_evaluation_count", 1),
        ("accepted_candidate_rank", 1),
        ("model_request_count", 1),
        ("relative_bound_attachment_validated", True),
        ("relative_bound_controller_evaluated", True),
        ("intended_r010_construction_scientifically_exhausted", True),
        ("behavioral_activation_released", True),
    ):
        bad = deepcopy(value)
        bad[key] = replacement
        with pytest.raises(ValueError):
            validate_compact_contract(bad)
