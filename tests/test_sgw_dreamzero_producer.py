from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.request
import urllib.error

import numpy as np

from experiments.workshops.spatial_grounding_v1.adapters import DREAMZERO_CONFIG
from experiments.workshops.spatial_grounding_v1 import dreamzero_backend
from experiments.workshops.spatial_grounding_v1.dreamzero_producer import (
    DreamZeroEvidenceProducer,
    make_dreamzero_http_server,
)


class _Backend:
    source_root = "/pinned/dreamzero"
    checkpoint_path = "/pinned/checkpoint"
    resolved_config = DREAMZERO_CONFIG

    def __init__(self) -> None:
        self.predict_calls = []
        self.reset_calls = []

    def reset(self, session_id):
        self.reset_calls.append(session_id)
        return {"evicted": session_id}

    def predict(self, observation, prompt, sampling_seed, **kwargs):
        self.predict_calls.append((observation, prompt, sampling_seed, kwargs))
        return {
            "actions": np.full((24, 8), len(self.predict_calls), dtype=np.float32),
            "future": np.arange(6, dtype=np.float32),
            "session_id": "native-session",
        }


def _packet(reset_id: str, index: int, request_id: str) -> dict:
    return {
        "request_id": request_id,
        "request_index": index,
        "reset_id": reset_id,
        "camera_name": "over_shoulder_left_camera",
        "registered_cell_id": "cell-1",
        "reset_fingerprint": "fingerprint-1",
        "prompt": "static prompt",
        "sampling_seed": 1140,
        "observation": {"image": [[[0]]]},
    }


def test_dreamzero_owned_http_producer_records_actions_future_and_native_reset(tmp_path: Path) -> None:
    backend = _Backend()
    producer = DreamZeroEvidenceProducer(
        backend,
        trace_path=tmp_path / "trace.jsonl",
        future_dir=tmp_path / "future",
        attestation_path=tmp_path / "attestation.json",
    )
    reset = producer.reset({"camera_name": "over_shoulder_left_camera"})
    first = producer.predict(_packet(reset["reset_id"], 0, "r0"))
    second = producer.predict(_packet(reset["reset_id"], 1, "r1"))
    assert np.asarray(first["actions"]).shape == (24, 8)
    assert second["future_status"] == "exposed_and_retained"
    assert backend.predict_calls[0][3] == {
        "action_guidance": 1,
        "video_guidance": 5,
        "steps": 16,
        "session_id": None,
    }
    assert backend.predict_calls[1][3]["session_id"] == "native-session"
    records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["actions_shape"] == [24, 8]
    assert records[0]["future_status"] == "exposed_and_retained"
    assert (tmp_path / "future" / f"{records[0]['wrapper_request_id']}.npy").is_file()
    producer.reset({"camera_name": "over_shoulder_left_camera"})
    assert backend.reset_calls == [None, "native-session"]


def test_dreamzero_owned_http_boundary_rejects_request_before_reset(tmp_path: Path) -> None:
    backend = _Backend()
    producer = DreamZeroEvidenceProducer(
        backend,
        trace_path=tmp_path / "trace.jsonl",
        future_dir=tmp_path / "future",
        attestation_path=tmp_path / "attestation.json",
    )
    server = make_dreamzero_http_server(producer, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        packet = _packet("wrong", 0, "r0")
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/predict",
            data=json.dumps(packet).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(request)
        except urllib.error.HTTPError as error:
            assert error.code == 400
        assert backend.predict_calls == []
    finally:
        server.shutdown()
        server.server_close()


def test_dreamzero_native_binding_requires_exported_server_surface(tmp_path: Path) -> None:
    try:
        dreamzero_backend._verify_exported_server_surface(tmp_path)
    except ValueError as error:
        assert "source file is missing" in str(error)
    else:
        raise AssertionError("missing native server source must fail closed")
