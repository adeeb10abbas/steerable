"""Synthetic-hardware integration for the real adapter, recorder, and scorer."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from experiments.workshops.spatial_grounding_v1.adapters import NanoPolicyAdapter, ProductionAdapter
from experiments.workshops.spatial_grounding_v1.compile import compile_manifests
from experiments.workshops.spatial_grounding_v1.contract import Cell, load_release, verify_completion_pointer
from experiments.workshops.spatial_grounding_v1.recorder import AttemptRecorder
from experiments.workshops.spatial_grounding_v1.worker import _canonical_outcome, _load_scorer
from experiments.workshops.spatial_grounding_v1.worker import run_partition
from tests.test_sgw_contract import make_release


@dataclass
class Reset:
    receipt: dict
    snapshot: dict


class SyntheticEnvironment:
    def __init__(self) -> None:
        self.step_index = 0

    def _state(self) -> dict:
        return {
            "cube": (0.0, 0.0, 0.1), "bowl": (0.0, 0.0, 0.1), "plate": (0.0, 0.2, 0.1),
            "gripper_holding": False, "cube_height_lift_m": 0.0, "final_detached_release": True,
            "supported": True, "linear_speed_m_s": 0.0, "angular_speed_rad_s": 0.0,
            "sim_time": self.step_index * 0.01, "sim_time_s": self.step_index * 0.01,
        }

    def reset(self):
        self.step_index = 0
        return Reset(
            {"reset_id": "reset-1", "camera_id": "cam-1", "camera_name": "cam-1",
             "fingerprint": "a" * 64, "temporal_cache_reset": True},
            self._state(),
        )

    def policy_observation(self):
        return {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}

    def snapshot(self):
        return self._state()

    def render_viewport(self):
        return np.full((16, 16, 3), self.step_index % 255, dtype=np.uint8)

    def step(self, _action):
        self.step_index += 1
        return {"safety_terminated": False}

    def close(self):
        pass


TRANSPORT_CALLS = 0


def _transport(request):
    global TRANSPORT_CALLS
    TRANSPORT_CALLS += 1
    return {
        "request_id": request["request_id"], "registered_cell_id": request["registered_cell_id"],
        "request_index": request["request_index"], "reset_id": request["reset_id"],
        "camera_id": request["camera_id"], "camera_name": request["camera_name"],
        "reset_fingerprint": request["reset_fingerprint"],
        "actions": np.zeros((32, 8), dtype=np.float32),
        "future": np.zeros((2, 8, 8, 3), dtype=np.uint8),
    }


def test_production_adapter_recorder_scorer_wrong_side_publishes_real_evidence(tmp_path: Path) -> None:
    global TRANSPORT_CALLS
    TRANSPORT_CALLS = 0
    release = load_release(make_release(tmp_path))
    pointers = []
    for number, original in enumerate(release.partition("N3", "LAT", "P"), 1):
        row = dict(original.row)
        row.update({"sampling_seed": number, "physical_goal_sign": 1, "form": "D"})
        cell = Cell(row)
        recorder = AttemptRecorder(release, cell, "attempt-001")
        recorder.begin()
        adapter = ProductionAdapter(NanoPolicyAdapter, transport=_transport, transport_factory=lambda **_: _transport,
                                    environment_factory=lambda **_: SyntheticEnvironment())
        reset = adapter.reset(cell, recorder)
        raw = adapter.run_episode(cell, recorder, reset)
        outcome = _canonical_outcome({**raw, "attempt_id": recorder.attempt_id}, cell, _load_scorer())
        assert outcome["status"] == "valid_model_failure"
        assert outcome["terminal_step"] == 450
        assert recorder.complete(outcome)
        pointer = release.root.parent / "cells" / f"{cell.cell_id}.complete.json"
        verify_completion_pointer(release, pointer)
        pointers.append(pointer)
        assert (recorder.path / "videos" / "viewport.mp4").is_file()
        assert len(list((recorder.path / "actions").glob("*.npy"))) == 450
        assert len(list((recorder.path / "observations").glob("*.npy"))) == 451
    assert TRANSPORT_CALLS == 90
    compiled = compile_manifests(pointers, expected_release_hashes=release.hashes)
    assert len(compiled.valid_rows) == 6
    assert all(row["status"] == "valid_model_failure" for row in compiled.rows)

    class MustNotRun:
        def reset(self, *_args):
            raise AssertionError("resume should skip every complete cell")
        def run_episode(self, *_args):
            raise AssertionError("resume should issue no requests")
        def close(self):
            pass

    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6,
                         max_attempts=3, worker_id="resume", adapter=MustNotRun(),
                         scorer=_load_scorer()) == 0
    assert TRANSPORT_CALLS == 90
