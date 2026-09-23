from __future__ import annotations

import json
from pathlib import Path
import threading
from dataclasses import dataclass

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


def test_native_environment_through_http_backend_and_real_source_packing(tmp_path, monkeypatch):
    from tests.test_sgw_native_policy_observations import native_nano_helpers
    from tests.test_sgw_jointpos_environment import candidate, install_native_boundary
    from tests.test_sgw_contract import make_release
    from experiments.workshops.spatial_grounding_v1.adapters import NANO_CONFIG, ProductionAdapter
    from experiments.workshops.spatial_grounding_v1.contract import Cell, load_release
    from experiments.workshops.spatial_grounding_v1.nano_backend import CosmosNanoBackend
    from experiments.workshops.spatial_grounding_v1.robolab_jointpos_environment import JointPositionEnvironment
    from experiments.workshops.spatial_grounding_v1.recorder import AttemptRecorder
    from experiments.workshops.spatial_grounding_v1.trace import read_trace_sidecar
    from experiments.workshops.spatial_grounding_v1.worker import _canonical_outcome, _load_scorer

    helpers = native_nano_helpers()
    native = install_native_boundary(monkeypatch, tensor_module=helpers["torch"])()
    @dataclass(frozen=True)
    class Config:
        seed: int = 1140
        deterministic_seed: bool = True
    class Service:
        cfg = Config()
        calls = 0

        def infer(self, observation):
            self.calls += 1
            assert "image_obs" not in observation and "observation/image" not in observation
            assert helpers["_extract_observation_image"](observation).shape == (24, 16, 3)
            assert helpers["_ensure_2d_float_array"](observation["observation/joint_position"], "joint", 7).shape == (1, 7)
            assert helpers["_ensure_gripper_array"](observation["observation/gripper_position"]).shape == (1, 1)
            return {"action": np.zeros((32, 8), dtype=np.float32)}
    backend = CosmosNanoBackend(Service(), resolved_config=NANO_CONFIG,
                               source_root="/pinned/source", checkpoint_path="/pinned/checkpoint")
    monkeypatch.setattr(producer, "_git_revision", lambda _: NANO_CONFIG["source_commit"])
    monkeypatch.setattr(producer, "_checkpoint_revision", lambda _: NANO_CONFIG["revision"])
    monkeypatch.setattr(producer, "_proc_start_identity", lambda _: "start-1")
    trace = tmp_path / "trace.jsonl"
    evidence = producer.NanoEvidenceProducer(backend, trace_path=trace, future_dir=tmp_path / "future",
                                            attestation_path=tmp_path / "attestation.json")
    server = producer.make_nano_http_server(evidence, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("SGW01_CAMERA_NAME", "over_shoulder_left_camera")
    monkeypatch.setenv("SGW01_TRACE_SIDECAR", str(trace))
    transport = runtime._NanoHttpTransport("127.0.0.1", server.server_port, read_trace_sidecar,
                                          {"config": {"history_length": 1}})
    release = load_release(make_release(tmp_path))
    cell = Cell({**release.cells[0].row, "sampling_seed": 13, "physical_goal_sign": 1, "form": "D"})
    recorder = AttemptRecorder(release, cell, "attempt-001")
    recorder.begin()
    adapter = ProductionAdapter(
        NanoPolicyAdapter, transport=transport, transport_factory=lambda **_: transport,
        environment_factory=lambda cell, evidence_root: JointPositionEnvironment(
            native, candidate=candidate(), cell_id=cell.cell_id, evidence_root=evidence_root,
        ),
    )
    try:
        reset = adapter.reset(cell, recorder)
        outcome = adapter.run_episode(cell, recorder, reset)
        assert len(native.actions) == 450 and backend.service.calls == 15
        assert outcome["viewport_artifact"]["fps"] == 15
        assert _canonical_outcome(outcome, cell, _load_scorer())["status"] == "valid_success"
        assert len(list((recorder.path / "predictions").glob("request-*.json"))) == 15
    finally:
        adapter.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
