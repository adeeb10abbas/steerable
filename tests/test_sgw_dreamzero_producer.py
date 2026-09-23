from __future__ import annotations

import ast
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


def test_dreamzero_14b_binding_executes_policy_boundary_and_reset() -> None:
    class Policy:
        resolved_config = DREAMZERO_CONFIG

        def __init__(self) -> None:
            self.calls = []
            self.resets = []
            self.video_across_time = [np.ones((2, 3), dtype=np.float32)]

        def infer(self, observation):
            self.calls.append(observation)
            return np.ones((24, 8), dtype=np.float32)

        def reset(self, info):
            self.resets.append(info)

    policy = Policy()
    backend = dreamzero_backend.OfficialDreamZero14BBackend(
        policy,
        source_root="/pinned/server",
        checkpoint_path="/pinned/checkpoint",
        resolved_config=DREAMZERO_CONFIG,
    )
    result = backend.predict({"session_id": "s1"}, "static", 1140)
    assert np.asarray(result["actions"]).shape == (24, 8)
    assert "future" in result
    reset = backend.reset("s1")
    assert reset["evicted_session_id"] == "s1"
    assert policy.resets == [{"session_id": "s1"}]


def test_dreamzero_native_sampler_fields_are_observed_not_checkpoint_overlaid() -> None:
    head = type(
        "ActionHead",
        (),
        {"num_inference_steps": 16, "seed": 1140, "cfg_scale": 5.0},
    )()
    assert dreamzero_backend._observe_action_head(head) == {
        "num_inference_steps": 16,
        "seed": 1140,
        "cfg_scale": 5.0,
    }


def test_exported_14b_ar_policy_executes_infer_and_session_reset() -> None:
    export = Path(
        "/Users/SZ5VJY/.copilot/session-state/"
        "c230f3cd-3fe9-4f1f-9ee4-e8b857151a60/files/"
        "sgw-native-d1-ar-server-ak.json"
    )
    if not export.exists():
        return
    source = json.loads(export.read_text())["files"]["socket_test_optimized_AR.py"]["text"]
    class_node = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "ARDroidRoboarenaPolicy")

    class FakeTensor:
        def cuda(self):
            return self

    class FakeTorch:
        int32 = object()
        int64 = object()
        uint8 = object()

        class Tensor:
            pass

        @staticmethod
        def zeros(*args, **kwargs):
            return FakeTensor()

        @staticmethod
        def tensor(*args, **kwargs):
            return FakeTensor()

        @staticmethod
        def frombuffer(*args, **kwargs):
            return FakeTensor()

        class _NoGrad:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        @staticmethod
        def no_grad():
            return FakeTorch._NoGrad()

    class FakeDist:
        ProcessGroup = object

        @staticmethod
        def broadcast(*args, **kwargs):
            return None

        @staticmethod
        def barrier(*args, **kwargs):
            return None

    class FakeBasePolicy:
        pass

    class FakeBatch:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakePolicy:
        def __init__(self):
            self.calls = 0

        def lazy_joint_forward_causal(self, batch):
            self.calls += 1
            actions = type("Actions", (), {})()
            setattr(actions, "action.joint_position", np.ones((24, 7), dtype=np.float32))
            setattr(actions, "action.gripper_position", np.zeros((24, 1), dtype=np.float32))
            return type("Result", (), {"act": actions})(), np.ones((2, 3), dtype=np.float32)

    namespace = {
        "np": np,
        "torch": FakeTorch,
        "dist": FakeDist,
        "_base_policy": type("BaseModule", (), {"BasePolicy": FakeBasePolicy}),
        "GrootSimPolicy": object,
        "logger": type("Logger", (), {"info": lambda *args, **kwargs: None})(),
        "os": __import__("os"),
        "datetime": __import__("datetime"),
        "rearrange": lambda value, pattern: value,
        "Batch": FakeBatch,
    }
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), "<exported-ar>", "exec"), namespace)
    wrapper = namespace["ARDroidRoboarenaPolicy"].__new__(namespace["ARDroidRoboarenaPolicy"])
    wrapper._policy = FakePolicy()
    wrapper._signal_group = object()
    wrapper._output_dir = None
    wrapper._frame_buffers = {
        "video.exterior_image_1_left": [],
        "video.exterior_image_2_left": [],
        "video.wrist_image_left": [],
    }
    wrapper._call_count = 0
    wrapper._is_first_call = True
    wrapper._current_session_id = None
    wrapper.video_across_time = []
    wrapper._msg_index = 0
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    observation = {
        "observation/exterior_image_0_left": frame,
        "observation/exterior_image_1_left": frame,
        "observation/wrist_image_left": frame,
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "static",
        "session_id": "s1",
    }
    first = wrapper.infer(observation)
    second = wrapper.infer(observation)
    assert first.shape == (24, 8)
    assert second.shape == (24, 8)
    assert wrapper._policy.calls == 2
    observation["session_id"] = "s2"
    wrapper.infer(observation)
    assert wrapper._is_first_call is False
    assert wrapper._current_session_id == "s2"
    assert wrapper._call_count == 1
