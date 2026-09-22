from __future__ import annotations

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1.adapters import (
    AdapterError,
    DreamZeroPolicyAdapter,
    NanoPolicyAdapter,
)


PROMPT = "Move the object to the left of the reference."


def reset_evidence():
    return {"reset_id": "reset-1", "camera_id": "cam-1", "fingerprint": "a" * 64}


def make_transport(model: str, *, bad: dict | None = None):
    def transport(request):
        horizon = 32 if model == "N3" else 24
        response = {
            "request_id": request["request_id"],
            "registered_cell_id": request["registered_cell_id"],
            "request_index": request["request_index"],
            "reset_id": request["reset_id"],
            "camera_id": request["camera_id"],
            "camera_name": request["camera_name"],
            "reset_fingerprint": request["reset_fingerprint"],
            "actions": np.zeros((horizon, 8), dtype=np.float32),
            "future": {"frames": 1},
            "effective_noise_seed": 1140,
            "action_guidance": 1,
        }
        response.update(bad or {})
        return response
    return transport


def test_n3_exact_config_and_contiguous_execution():
    adapter = NanoPolicyAdapter(
        cell_id="cell-1", prompt=PROMPT, transport=make_transport("N3")
    )
    adapter.reset(reset_fn=reset_evidence, reset_id="reset-1", camera_id="cam-1")
    prediction = adapter.predict({}, PROMPT, action_step_start=0)
    assert prediction.returned_actions.shape == (32, 8)
    assert prediction.executable_actions.shape == (32, 8)
    assert prediction.returned_horizon == prediction.executed_horizon == 32
    assert adapter.predictions[0].future_status == "exposed_and_retained"


def test_d1_executes_only_first_eight_and_requires_official_path():
    adapter = DreamZeroPolicyAdapter(
        cell_id="cell-1", prompt=PROMPT, transport=make_transport("D1")
    )
    adapter.reset(reset_fn=reset_evidence, reset_id="reset-1", camera_id="cam-1")
    prediction = adapter.predict({}, PROMPT, action_step_start=0)
    assert prediction.returned_actions.shape == (24, 8)
    assert prediction.executable_actions.shape == (8, 8)
    with pytest.raises(AdapterError):
        DreamZeroPolicyAdapter(
            cell_id="cell-1",
            prompt=PROMPT,
            transport=make_transport("D1"),
            action_guidance=2,
        )


def test_request_identity_and_static_prompt_fail_closed():
    adapter = NanoPolicyAdapter(
        cell_id="cell-1", prompt=PROMPT, transport=make_transport("N3")
    )
    with pytest.raises(AdapterError):
        adapter.predict({}, PROMPT, action_step_start=0)
    adapter.reset(reset_fn=reset_evidence, reset_id="reset-1", camera_id="cam-1")
    with pytest.raises(AdapterError):
        adapter.predict({}, "changed", action_step_start=0)
    with pytest.raises(AdapterError):
        adapter.predict({}, PROMPT, action_step_start=1)


def test_d1_rejects_non_official_response():
    adapter = DreamZeroPolicyAdapter(
        cell_id="cell-1",
        prompt=PROMPT,
        transport=make_transport("D1", bad={"action_guidance": 2}),
    )
    adapter.reset(reset_fn=reset_evidence, reset_id="reset-1", camera_id="cam-1")
    with pytest.raises(AdapterError):
        adapter.predict({}, PROMPT, action_step_start=0)
