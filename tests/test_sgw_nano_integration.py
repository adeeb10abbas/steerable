from __future__ import annotations

import json
from pathlib import Path
import threading

import numpy as np

from experiments.workshops.spatial_grounding_v1 import producer, runtime
from experiments.workshops.spatial_grounding_v1.adapters import NanoPolicyAdapter


class _Backend:
    resolved_config = dict(producer.NANO_CONFIG)
    source_root = "/pinned/source"
    checkpoint_path = "/pinned/checkpoint"

    def predict(self, observation, prompt, sampling_seed):
        return {"action": np.zeros((32, 8), dtype=np.float32)}


def test_transport_producer_backend_adapter_two_resets_and_450_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(producer, "_git_revision", lambda _: producer.NANO_CONFIG["source_commit"])
    monkeypatch.setattr(producer, "_checkpoint_revision", lambda _: producer.NANO_CONFIG["revision"])
    monkeypatch.setattr(producer, "_proc_start_identity", lambda _: "start-1")
    trace_path = tmp_path / "trace.jsonl"
    evidence = producer.NanoEvidenceProducer(
        _Backend(),
        trace_path=trace_path,
        future_dir=tmp_path / "future",
        attestation_path=tmp_path / "attestation.json",
    )
    server = producer.make_nano_http_server(evidence, host="127.0.0.1", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("SGW01_CAMERA_NAME", "wrist")
    monkeypatch.setenv("SGW01_TRACE_SIDECAR", str(trace_path))
    transport = runtime._NanoHttpTransport(
        "127.0.0.1",
        server.server_port,
        runtime.read_trace_sidecar if hasattr(runtime, "read_trace_sidecar") else __import__(
            "experiments.workshops.spatial_grounding_v1.trace", fromlist=["read_trace_sidecar"]
        ).read_trace_sidecar,
        {"config": {"history_length": 1}},
    )
    adapter = NanoPolicyAdapter(
        cell_id="cell",
        prompt="static",
        transport=transport,
        runtime=transport,
        sampling_seed=8300,
    )
    snapshot = {"reset_id": "reset-0", "camera_name": "wrist", "fingerprint": "a" * 64}
    reset_fn = lambda: snapshot
    try:
        adapter.reset(reset_fn=reset_fn, reset_id="reset-0", camera_id="wrist")
        observation = {
            "observation/wrist_image_left": np.zeros((720, 1280, 3), dtype=np.uint8),
            "observation/exterior_image_1_left": np.zeros((720, 1280, 3), dtype=np.uint8),
            "observation/exterior_image_2_left": np.zeros((720, 1280, 3), dtype=np.uint8),
        }
        while adapter.executed_steps < 450:
            prediction = adapter._request(observation, "static", adapter.executed_steps)
            count = min(prediction.executed_horizon, 450 - adapter.executed_steps)
            adapter.commit_executed(count)
        assert adapter.executed_steps == 450
        assert len(adapter.predictions) == 15

        trace_path.write_text("", encoding="utf-8")
        snapshot = {"reset_id": "reset-1", "camera_name": "wrist", "fingerprint": "b" * 64}
        adapter.reset(reset_fn=lambda: snapshot, reset_id="reset-1", camera_id="wrist")
        prediction = adapter._request(observation, "static", 0)
        assert prediction.returned_horizon == 32
        assert len(json.loads(trace_path.read_text().splitlines()[0])["request_id"]) > 0
    finally:
        server.shutdown()
        server.server_close()
