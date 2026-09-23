from __future__ import annotations

import ast
from abc import ABC, abstractmethod
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import types
import urllib.request
import urllib.error

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1.adapters import (
    AdapterError, DREAMZERO_CONFIG, DreamZeroPolicyAdapter,
)
from experiments.workshops.spatial_grounding_v1 import dreamzero_backend, runtime
from experiments.workshops.spatial_grounding_v1.trace import read_trace_sidecar
from experiments.workshops.spatial_grounding_v1.recorder import AttemptRecorder
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
            "session_id": observation["session_id"],
        }


def _packet(reset_id: str, index: int, request_id: str) -> dict:
    return {
        "request_id": request_id,
        "request_index": index,
        "reset_id": "physical-reset",
        "wrapper_reset_id": reset_id,
        "camera_id": "over_shoulder_left_camera",
        "camera_name": "over_shoulder_left_camera",
        "registered_cell_id": "cell-1",
        "reset_fingerprint": "fingerprint-1",
        "prompt": "static prompt",
        "sampling_seed": 1140,
        "observation": {"image": [[[0]]], "session_id": "native-session"},
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
    assert second["future_status"] == "latent_only_retained"
    assert backend.predict_calls[0][3] == {
        "action_guidance": 1,
        "video_guidance": 5,
        "steps": 16,
        "session_id": "native-session",
    }
    assert backend.predict_calls[1][3]["session_id"] == "native-session"
    records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["actions_shape"] == [24, 8]
    assert records[0]["future_status"] == "latent_only_retained"
    assert records[0]["reset_id"] == "physical-reset"
    assert records[0]["wrapper_reset_id"] == reset["reset_id"]
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


@pytest.mark.parametrize("field,value", [
    ("reset_id", "another-physical-reset"),
    ("wrapper_reset_id", "another-wrapper-reset"),
    ("camera_id", "another-camera"),
    ("sampling_seed", 8302),
    ("session_id", "another-native-session"),
])
def test_dreamzero_binding_changes_rejected_before_model(
    tmp_path: Path, field: str, value: object,
) -> None:
    backend = _Backend()
    producer = DreamZeroEvidenceProducer(
        backend, trace_path=tmp_path / "trace.jsonl", future_dir=tmp_path / "future",
        attestation_path=tmp_path / "attestation.json",
    )
    reset = producer.reset({"camera_name": "over_shoulder_left_camera"})
    producer.predict(_packet(reset["reset_id"], 0, "r0"))
    packet = _packet(reset["reset_id"], 1, "r1")
    (packet["observation"] if field == "session_id" else packet)[field] = value
    with pytest.raises(AdapterError):
        producer.predict(packet)
    assert len(backend.predict_calls) == 1


def test_latent_trace_stays_undecoded_and_raw_hash_is_checked(tmp_path: Path, monkeypatch) -> None:
    producer = DreamZeroEvidenceProducer(
        _Backend(), trace_path=tmp_path / "trace.jsonl", future_dir=tmp_path / "future",
        attestation_path=tmp_path / "attestation.json",
    )
    monkeypatch.setenv("SGW01_TRACE_SIDECAR", str(tmp_path / "trace.jsonl"))
    reset = producer.reset({"camera_name": "over_shoulder_left_camera"})
    packet = _packet(reset["reset_id"], 0, "r0")
    result = producer.predict(packet)
    response = {"raw_actions": result["actions"], "actions": np.zeros((24, 8))}
    record = read_trace_sidecar(request=packet, response=response)
    assert "future_latents" in record and "future" not in record
    with pytest.raises(AdapterError, match="hash differs"):
        read_trace_sidecar(request=packet, response={**response, "raw_actions": np.zeros((24, 8))})
    saved = json.loads((tmp_path / "trace.jsonl").read_text())
    saved["future_status"] = "exposed_and_retained"
    (tmp_path / "trace.jsonl").write_text(json.dumps(saved) + "\n")
    with pytest.raises(AdapterError, match="latent evidence"):
        read_trace_sidecar(request=packet, response=response)


def test_prediction_array_names_do_not_interpret_observation_keys_as_paths(tmp_path: Path) -> None:
    recorder = AttemptRecorder.__new__(AttemptRecorder)
    recorder.path = tmp_path
    recorder.prediction(types.SimpleNamespace(
        request_index=0,
        raw_request={"observation/exterior_image_0_left": np.zeros((2, 3, 3), dtype=np.uint8)},
        raw_response={"arbitrary/../../key": np.ones((24, 8), dtype=np.float32)},
    ))
    record = json.loads((tmp_path / "predictions/request-0000.json").read_text())
    for envelope in ("raw_request", "raw_response"):
        for item in record[envelope].values():
            path = tmp_path / item["path"]
            assert path.parent == tmp_path / "predictions/request-0000-arrays"
            assert path.is_file()


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
    image = np.ones((3, 4, 3), dtype=np.uint8)
    result = backend.predict({
        "session_id": "s1",
        "observation/exterior_image_0_left": image.tolist(),
        "observation/exterior_image_1_left": image.tolist(),
        "observation/wrist_image_left": image.tolist(),
        "observation/joint_position": [0] * 7,
        "observation/gripper_position": [0],
    }, "static", 1140)
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


def test_official_factory_accepts_owned_factory_config_without_changing_checkpoint_fields(
    tmp_path: Path, monkeypatch,
) -> None:
    identity = {
        "source_root": str(tmp_path), "checkpoint_path": str(tmp_path / "checkpoint"),
        "source_commit": DREAMZERO_CONFIG["source_commit"],
        "checkpoint_revision": DREAMZERO_CONFIG["revision"],
    }
    checkpoint = Path(identity["checkpoint_path"])
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "action_head_cfg": {"config": {"num_inference_timesteps": 4}},
        "action_horizon": 24, "action_dim": 32,
    }))
    monkeypatch.setattr(dreamzero_backend, "_verify_dreamzero_identity", lambda: identity)
    monkeypatch.setattr(dreamzero_backend, "_verify_exported_server_surface", lambda _: None)
    monkeypatch.setattr(dreamzero_backend, "_verify_14b_entrypoint", lambda _: None)
    monkeypatch.setenv("SGW01_D1_MODEL_PATH", str(checkpoint))
    monkeypatch.setenv(
        "SGW01_D1_SERVER_FACTORY",
        "experiments.workshops.spatial_grounding_v1.dreamzero_backend:build_official_14b_dreamzero_backend",
    )
    module = types.ModuleType("socket_test_optimized_AR")
    module.__file__ = str(tmp_path / "socket_test_optimized_AR.py")
    module.init_mesh = lambda: "mock-device-mesh"
    module.dist = types.SimpleNamespace(new_group=lambda **_: "mock-signal-group")
    module.datetime = __import__("datetime")
    module.EmbodimentTag = lambda value: value
    module.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    module.GrootSimPolicy = lambda **_: types.SimpleNamespace(
        model=types.SimpleNamespace(action_head=types.SimpleNamespace(
            num_inference_steps=16, seed=1140, cfg_scale=5.0,
        )),
    )
    module.ARDroidRoboarenaPolicy = lambda **_: types.SimpleNamespace(
        infer=lambda _: np.zeros((24, 8), dtype=np.float32), reset=lambda _: None,
        video_across_time=[],
    )
    monkeypatch.setitem(sys.modules, "socket_test_optimized_AR", module)
    backend = dreamzero_backend.build_pinned_dreamzero_backend()
    assert backend.resolved_config == DREAMZERO_CONFIG
    assert backend.native_metadata["native_checkpoint_num_inference_timesteps"] == 4
    assert backend.native_metadata["native_checkpoint_action_dim"] == 32
    assert backend.native_metadata["native_sampler_steps"] == 16


def test_exported_official_client_runs_owned_http_cache_postprocess_and_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = os.environ.get("SGW01_D1_SOURCE_AUDIT")
    image_export = os.environ.get("SGW01_D1_IMAGE_SOURCE_AUDIT")
    if not export or not image_export:
        pytest.skip("set the authorized D1 client and image-utils source exports")
    sources = json.loads(Path(export).read_text())["native_d1_client_sources"]
    namespace: dict[str, object] = {
        "__name__": __name__,
        "ABC": ABC,
        "abstractmethod": abstractmethod,
        "np": np,
        "os": os,
        "uuid": __import__("uuid"),
        "logging": __import__("logging"),
        "time": __import__("time"),
    }
    for suffix, expected in (
        ("robolab/eval/base_client.py", runtime.D1_BASE_CLIENT_SOURCE_SHA256),
        ("policies/dreamzero/client.py", runtime.D1_CLIENT_SOURCE_SHA256),
    ):
        source = next(value for key, value in sources.items() if key.endswith(suffix))
        assert source["sha256"] == hashlib.sha256(source["text"].encode()).hexdigest() == expected
        nodes = [node for node in ast.parse(source["text"]).body
                 if isinstance(node, (ast.ClassDef, ast.Assign))]
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
            ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), suffix, "exec"), namespace)
    for name in (
        "policies", "policies.dreamzero", "policies.dreamzero.client",
        "robolab", "robolab.core", "robolab.core.utils", "robolab.core.utils.image_utils",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    client_module = sys.modules["policies.dreamzero.client"]
    client_module.__file__ = str(tmp_path / "policies/dreamzero/client.py")
    client_module.DreamZeroClient = namespace["DreamZeroClient"]
    image_source = json.loads(Path(image_export).read_text())
    assert image_source["sha256"] == hashlib.sha256(image_source["text"].encode()).hexdigest() == (
        "aead0c246b696ce5feabbbc63df93f42762365cc6315269fa08ad5c98e1a3d94"
    )
    exec(compile(image_source["text"], image_source["path"], "exec"),
         sys.modules["robolab.core.utils.image_utils"].__dict__)

    class Tensor:
        def __init__(self, value):
            self.value = value

        def clone(self):
            return Tensor(self.value.copy())

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Policy:
        def __init__(self):
            self.calls = []
            self.resets = []
            self.video_across_time = []

        def infer(self, observation):
            self.calls.append(observation)
            self.video_across_time.append(np.ones((2, 3), dtype=np.float32))
            actions = np.full((24, 8), len(self.calls), dtype=np.float32)
            actions[:, -1] = np.tile([0.49, 0.51], 12)
            return actions

        def reset(self, packet):
            self.resets.append(packet)
            self.video_across_time.clear()

    policy = Policy()
    backend = dreamzero_backend.OfficialDreamZero14BBackend(
        policy, source_root="/pinned/source", checkpoint_path="/pinned/checkpoint",
        resolved_config=DREAMZERO_CONFIG,
    )
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
        monkeypatch.setenv("SGW01_D1_HTTP_URL", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setenv("SGW01_D1_CLIENT_SOURCE_ROOT", str(tmp_path))
        monkeypatch.setenv("SGW01_TRACE_SIDECAR", str(tmp_path / "trace.jsonl"))
        monkeypatch.setattr(runtime, "_verify_dreamzero_identity", lambda: {
            "checkpoint_revision": DREAMZERO_CONFIG["revision"],
        })
        transport = runtime._OfficialDreamZeroClient("127.0.0.1", server.server_port, read_trace_sidecar)
        adapter = DreamZeroPolicyAdapter(
            cell_id="cell-1", prompt="static", sampling_seed=8301,
            transport=transport, runtime=transport,
        )
        adapter.reset(
            reset_fn=lambda: {"reset_id": "physical-reset", "camera_id": "over_shoulder_left_camera",
                              "fingerprint": "f" * 64},
            reset_id="physical-reset", camera_id="over_shoulder_left_camera",
        )
        frame = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        observation = {
            "image_obs": {name: [Tensor(frame)] for name in (
                "over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam",
            )},
            "proprio_obs": {
                "arm_joint_pos": [Tensor(np.zeros(7, dtype=np.float32))],
                "gripper_pos": [Tensor(np.zeros(1, dtype=np.float32))],
            },
        }
        first = adapter.predict(observation, "static", action_step_start=0)
        adapter.commit_executed(8)
        second = adapter.predict(observation, "static", action_step_start=8)
        assert len(policy.calls) == 2
        assert transport.client._counters == {0: 8}
        np.testing.assert_array_equal(first.executable_actions[:, -1], [0, 1] * 4)
        assert np.all(second.executable_actions[:, :7] == 2)
        assert first.future is None and first.decoded is False
        assert first.future_status == "latent_only_retained"
        assert first.raw_response["native_trace"]["sampling_seed"] == 8301
        assert first.raw_response["native_trace"]["effective_noise_seed"] == 1140
        assert first.raw_response["native_trace"]["future_latents"].shape == (2, 3)
        session = policy.calls[0]["session_id"]
        assert policy.calls[1]["session_id"] == session
        packed_image = policy.calls[0]["observation/exterior_image_0_left"]
        expected_image = sys.modules["robolab.core.utils.image_utils"].resize_with_pad(frame, 180, 320)
        np.testing.assert_array_equal(packed_image, expected_image)
        assert packed_image.dtype == np.uint8
        recorder = AttemptRecorder.__new__(AttemptRecorder)
        recorder.path = tmp_path / "episode"
        recorder.prediction(first)
        recorded = json.loads((recorder.path / "predictions/request-0000.json").read_text())
        assert recorded["future_status"] == "latent_only_retained"
        assert "future" not in recorded["raw_response"]["native_trace"]
        assert (recorder.path / recorded["raw_response"]["native_trace"]["future_latents"]["path"]).is_file()
        transport.reset()
        assert transport.client._chunks == transport.client._counters == transport.client._env_session_id == {}
        assert policy.resets[-1] == {"session_id": session}

        from tests.test_sgw_jointpos_environment import candidate, install_native_boundary
        from tests.test_sgw_contract import make_release
        from experiments.workshops.spatial_grounding_v1.contract import Cell, load_release
        from experiments.workshops.spatial_grounding_v1.adapters import ProductionAdapter
        from experiments.workshops.spatial_grounding_v1.robolab_jointpos_environment import JointPositionEnvironment
        from experiments.workshops.spatial_grounding_v1.worker import _canonical_outcome, _load_scorer

        native = install_native_boundary(monkeypatch)()
        integration = tmp_path / "production"
        integration.mkdir()
        release = load_release(make_release(integration))
        cell = Cell({**release.cells[0].row, "cell_id": "d1-native-boundary", "model": "D1",
                     "sampling_seed": 17, "physical_goal_sign": 1, "form": "D"})
        episode_recorder = AttemptRecorder(release, cell, "attempt-001")
        episode_recorder.begin()
        production = ProductionAdapter(
            DreamZeroPolicyAdapter, transport=transport, transport_factory=lambda **_: transport,
            environment_factory=lambda cell, evidence_root: JointPositionEnvironment(
                native, candidate=candidate(), cell_id=cell.cell_id, evidence_root=evidence_root,
            ),
        )
        calls_before = len(policy.calls)
        physical_reset = production.reset(cell, episode_recorder)
        outcome = production.run_episode(cell, episode_recorder, physical_reset)
        assert len(policy.calls) - calls_before == 57
        assert len(native.actions) == 450
        assert outcome["viewport_artifact"]["frame_count"] == 451
        assert outcome["viewport_artifact"]["fps"] == 15
        assert _canonical_outcome(outcome, cell, _load_scorer())["status"] == "valid_success"
        for packed in policy.calls[calls_before:]:
            assert packed["observation/exterior_image_0_left"].shape == (180, 320, 3)
            assert packed["observation/joint_position"].shape == (7,)
        production.close()
        assert native.closed == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_exported_14b_ar_policy_executes_infer_and_session_reset() -> None:
    export = os.environ.get("SGW01_D1_AR_SOURCE_AUDIT")
    if not export:
        pytest.skip("set SGW01_D1_AR_SOURCE_AUDIT to the authorized AR source export")
    source = json.loads(Path(export).read_text())["files"]["socket_test_optimized_AR.py"]["text"]
    assert hashlib.sha256(source.encode()).hexdigest() == dreamzero_backend.D1_14B_ENTRYPOINT_SHA256
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
