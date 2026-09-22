from __future__ import annotations

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1.adapters import AdapterError, NanoPolicyAdapter


def test_prediction_cannot_cross_reset_or_camera_identity():
    def transport(request):
        response = dict(request)
        response["actions"] = np.zeros((32, 8), dtype=np.float32)
        response["future"] = None
        return response

    adapter = NanoPolicyAdapter(
        cell_id="cell", prompt="static", transport=transport
    )
    adapter.reset(
        reset_fn=lambda: {"reset_id": "r1", "camera_id": "c1", "fingerprint": "b" * 64},
        reset_id="r1",
        camera_id="c1",
    )
    adapter.transport = lambda request: {
        **request,
        "actions": np.zeros((32, 8), dtype=np.float32),
        "future": None,
        "camera_id": "wrong-camera",
    }
    with pytest.raises(AdapterError):
        adapter.predict({}, "static", action_step_start=0)


def test_action_cap_truncates_executed_prefix_without_padding():
    adapter = NanoPolicyAdapter(
        cell_id="cell", prompt="static",
        transport=lambda request: {
            **request,
            "actions": np.ones((32, 8), dtype=np.float32),
            "future": None,
        },
    )
    adapter.reset(
        reset_fn=lambda: {"reset_id": "r", "camera_id": "c", "fingerprint": "c" * 64},
        reset_id="r",
        camera_id="c",
    )
    adapter.executed_steps = 448
    prediction = adapter.predict({}, "static", action_step_start=448)
    assert prediction.returned_horizon == 32
    assert prediction.executed_horizon == 2
    assert prediction.executable_actions.shape == (2, 8)
    adapter.commit_executed(2)
    assert adapter.executed_steps == 450


def test_missing_future_is_not_scored_as_zero():
    adapter = NanoPolicyAdapter(
        cell_id="cell", prompt="static",
        transport=lambda request: {
            **request,
            "actions": np.zeros((32, 8), dtype=np.float32),
        },
    )
    adapter.reset(
        reset_fn=lambda: {"reset_id": "r", "camera_id": "c", "fingerprint": "d" * 64},
        reset_id="r",
        camera_id="c",
    )
    prediction = adapter.predict({}, "static", action_step_start=0)
    assert prediction.future is None
    assert prediction.future_status == "not_exposed"
