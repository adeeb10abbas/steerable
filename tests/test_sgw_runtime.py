from __future__ import annotations

import json
import ast
from abc import ABC, abstractmethod
import hashlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1 import runtime
from experiments.workshops.spatial_grounding_v1.adapters import AdapterError


class _ResettableClient:
    def __init__(self) -> None:
        self.calls = 0

    def clear_temporal_cache(self) -> None:
        self.calls += 1


class _NoopClient:
    def reset(self) -> None:
        pass


def test_nano_reset_requires_and_calls_verified_temporal_reset() -> None:
    adapter = runtime._NanoTransport.__new__(runtime._NanoTransport)
    adapter.client = _ResettableClient()
    adapter.reset()
    assert adapter.client.calls == 1

    adapter.client = object()
    with pytest.raises(AdapterError, match="verified temporal reset"):
        adapter.reset()

    adapter.client = _NoopClient()
    with pytest.raises(AdapterError, match="no-op"):
        adapter.reset()


class _DreamZeroClient:
    open_loop_horizon = 8

    def __init__(self, chunks: int) -> None:
        self.returned_chunks: list[np.ndarray] = []
        self.processed_chunks: list[np.ndarray] = []
        self.chunks = chunks
        self.calls = 0
        self.counter = 8

    def infer(self, observation: object, prompt: str) -> dict[str, np.ndarray]:
        if self.counter >= self.open_loop_horizon:
            self.counter = 0
            for _ in range(self.chunks):
                raw = np.zeros((24, 8), dtype=np.float32)
                raw[:, -1] = 0.75
                self.returned_chunks.append(raw)
                processed = raw.copy()
                processed[:, -1] = 1.0
                self.processed_chunks.append(processed)
        self.counter += 1
        self.calls += 1
        return {"action": self.processed_chunks[-1][self.counter - 1].copy()}


class _DreamZeroFutureClient(_DreamZeroClient):
    def infer(self, observation: object, prompt: str) -> dict[str, np.ndarray]:
        self.returned_future = np.ones((5, 2, 2, 3), dtype=np.uint8)
        return super().infer(observation, prompt)


def test_dreamzero_requires_exactly_one_fresh_chunk_per_request() -> None:
    adapter = runtime._OfficialDreamZeroClient.__new__(runtime._OfficialDreamZeroClient)
    adapter.client = _DreamZeroClient(1)
    adapter.trace_reader = lambda **_: {"request_id": "r1"}
    result = adapter({"request_id": "r1", "observation": {}, "prompt": "static"})
    assert np.asarray(result["actions"]).shape == (24, 8)
    assert np.asarray(result["executed_actions"]).shape == (8, 8)
    assert np.array_equal(result["executed_actions"], result["actions"][:8])
    assert np.all(np.asarray(result["actions"])[:, -1] == 1.0)
    assert np.all(np.asarray(result["raw_actions"])[:, -1] == 0.75)
    assert result["returned_horizon"] == 24
    assert result["executed_horizon"] == 8
    assert result["effective_noise_seed"] == 1140
    assert result["request_duration_ns"] >= 0
    assert result["future_status"] == "not_exposed"

    adapter.client = _DreamZeroClient(2)
    with pytest.raises(AdapterError, match="unexpected number"):
        adapter({"request_id": "r1", "observation": {}, "prompt": "static"})

    adapter.client = _DreamZeroFutureClient(1)
    result = adapter({"request_id": "r1", "observation": {}, "prompt": "static"})
    assert np.asarray(result["future"]).shape == (5, 2, 2, 3)
    assert result["future_status"] == "exposed_and_retained"
    assert adapter.client.calls == 8


def test_dreamzero_identity_rejects_wrong_or_dirty_git_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Result:
        stdout = "wrong\n"

    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs: _Result())
    with pytest.raises(AdapterError, match="not pinned"):
        runtime._verify_git_checkout(Path("/wrong"), "expected", "RoboLab client")

    def dirty(*args: object, **kwargs: object) -> object:
        if args and isinstance(args[0], list) and args[0][3] == "diff":
            raise subprocess.CalledProcessError(1, args[0])
        return type("_Result", (), {"stdout": "expected\n"})()

    import subprocess
    monkeypatch.setattr(runtime.subprocess, "run", dirty)
    with pytest.raises(AdapterError, match="dirty"):
        runtime._verify_git_checkout(Path("/dirty"), "expected", "RoboLab client")


def test_dreamzero_source_backed_cache_refreshes_once_per_eight_actions() -> None:
    adapter = runtime._OfficialDreamZeroClient.__new__(runtime._OfficialDreamZeroClient)
    adapter.client = _DreamZeroClient(1)
    adapter.trace_reader = lambda **_: {"request_id": "r1"}
    adapter({"request_id": "r1", "observation": {}, "prompt": "static"})
    assert adapter.client.calls == 8
    assert len(adapter.client.returned_chunks) == 1


def test_dreamzero_processed_chunk_survives_two_requests_and_reset() -> None:
    class CadencedClient(_DreamZeroClient):
        def reset(self, *, env_id: int | None = None) -> None:
            self.calls = 0
            self.counter = 8

    adapter = runtime._OfficialDreamZeroClient.__new__(runtime._OfficialDreamZeroClient)
    adapter.client = CadencedClient(1)
    adapter.trace_reader = lambda **_: {"request_id": "bound"}
    first = adapter({"request_id": "r1", "observation": {}, "prompt": "static"})
    second = adapter({"request_id": "r2", "observation": {}, "prompt": "static"})
    assert adapter.client.calls == 16
    assert np.all(np.asarray(first["actions"])[:, -1] == 1.0)
    assert np.all(np.asarray(second["raw_actions"])[:, -1] == 0.75)
    adapter.reset()
    third = adapter({"request_id": "r3", "observation": {}, "prompt": "static"})
    assert adapter.client.calls == 8
    assert np.all(np.asarray(third["actions"])[:, -1] == 1.0)


def test_dreamzero_wrapper_executes_exported_native_cache_postprocessing_and_reset(tmp_path, monkeypatch) -> None:
    export = os.environ.get("SGW01_D1_SOURCE_AUDIT")
    if not export:
        pytest.skip("set SGW01_D1_SOURCE_AUDIT to the authorized source export")
    sources = json.loads(Path(export).read_text())["native_d1_client_sources"]
    namespace = {"np": np, "ABC": ABC, "abstractmethod": abstractmethod, "__name__": __name__}
    for suffix, expected, class_name in (
        ("robolab/eval/base_client.py", runtime.D1_BASE_CLIENT_SOURCE_SHA256, "InferenceClient"),
        ("policies/dreamzero/client.py", runtime.D1_CLIENT_SOURCE_SHA256, "DreamZeroClient"),
    ):
        source = next(value for key, value in sources.items() if key.endswith(suffix))
        assert source["sha256"] == hashlib.sha256(source["text"].encode()).hexdigest() == expected
        node = next(node for node in ast.parse(source["text"]).body
                    if isinstance(node, ast.ClassDef) and node.name == class_name)
        if class_name == "DreamZeroClient":
            node.body = [method for method in node.body if isinstance(method, ast.FunctionDef)
                         and method.name in {"_unpack_response", "_postprocess_chunk", "reset"}]
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
            ast.alias(name="annotations")], level=0), node], type_ignores=[])
        # Execute hash-verified official methods; only networking and image input
        # are replaced below. No model library or native GPU runtime is imported.
        exec(compile(ast.fix_missing_locations(module), suffix, "exec"), namespace)

    class NetworkBoundary(namespace["DreamZeroClient"]):
        def __init__(self, **kwargs):
            super().__init__()
            self.open_loop_horizon = kwargs["open_loop_horizon"]
            self.binarize_gripper = kwargs["binarize_gripper"]
            self._env_session_id = {}
            self._packer = SimpleNamespace(pack=lambda value: value)
            self.queries = 0
            self.reset_messages = []

        def _extract_observation(self, obs, *, env_id):
            return {"env_id": env_id}

        def _pack_request(self, extracted, instruction):
            self._env_session_id.setdefault(extracted["env_id"], "owned-session")
            return {"instruction": instruction}

        def _query_server(self, request):
            self.queries += 1
            actions = np.full((24, 8), self.queries, dtype=np.float32)
            actions[:, -1] = np.tile([0.49, 0.51], 12)
            return {"actions": actions}

        def _send_recv(self, payload):
            self.reset_messages.append(payload)

    for name in ("policies", "policies.dreamzero", "policies.dreamzero.client"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    client_module = sys.modules["policies.dreamzero.client"]
    client_module.__file__ = str(tmp_path / "policies/dreamzero/client.py")
    client_module.DreamZeroClient = NetworkBoundary
    monkeypatch.setenv("SGW01_D1_CLIENT_SOURCE_ROOT", str(tmp_path))
    monkeypatch.setattr(runtime, "_verify_dreamzero_identity", lambda: {
        "checkpoint_revision": runtime.DREAMZERO_CONFIG["revision"],
    })
    adapter = runtime._OfficialDreamZeroClient("unused", 1, lambda **_: {"request_id": "test-only"})
    request = {"observation": {}, "prompt": "static"}
    first, second = adapter(request), adapter(request)
    assert adapter.client.queries == 2
    assert adapter.client._counters == {0: 8}
    np.testing.assert_array_equal(first["executed_actions"], first["actions"][:8])
    np.testing.assert_array_equal(first["raw_actions"][:, -1], np.tile(np.float32([0.49, 0.51]), 12))
    np.testing.assert_array_equal(second["actions"][:, -1], np.tile([0, 1], 12))
    assert np.all(second["executed_actions"][:, :7] == 2)
    adapter.reset()
    assert adapter.client.reset_messages == [{"endpoint": "reset", "session_ids": ["owned-session"]}]
    assert adapter.client._env_session_id == adapter.client._chunks == adapter.client._counters == {}
    assert adapter(request)["request_count"] == 1
    assert adapter.client.queries == 3


def test_nano_http_transport_uses_native_payload_without_identity_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"request_id": 17, "action": np.zeros((32, 8)).tolist()}).encode()

    captured: list[object] = []

    def fake_urlopen(request: object, timeout: float) -> _Response:
        captured.append(request)
        assert timeout == 180
        return _Response()

    monkeypatch.setattr(runtime.urllib.request, "urlopen", fake_urlopen)
    transport = runtime._NanoHttpTransport(
        "127.0.0.1",
        8123,
        lambda **_: {
            "request_id": "cell:request:0",
            "registered_cell_id": "cell",
            "request_index": 0,
            "reset_id": "reset",
            "camera_id": "cam",
            "camera_name": "cam",
            "reset_fingerprint": "fingerprint",
            "future_status": "not_exposed",
        },
        {"config": {"history_length": 1}},
    )
    response = transport(
        {
            "request_id": "cell:request:0",
            "request_index": 0,
            "registered_cell_id": "cell",
            "camera_id": "cam",
            "camera_name": "cam",
            "reset_id": "reset",
            "reset_fingerprint": "fingerprint",
            "sampling_seed": 1140,
            "prompt": "static",
            "observation": {"observation/image": np.zeros((4, 4, 3), dtype=np.uint8)},
        }
    )
    assert np.asarray(response["actions"]).shape == (32, 8)
    assert "request_id" not in response
    assert "native_trace" in response
    assert len(captured) == 1


def test_nano_http_reset_requires_attested_stateless_history() -> None:
    with pytest.raises(AdapterError, match="history_length=1"):
        runtime._NanoHttpTransport(
            "127.0.0.1",
            8123,
            lambda **_: {},
            {"config": {"history_length": 2}},
        )


def test_receipt_start_identity_is_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not Path("/proc").is_dir():
        pytest.skip("runtime receipt identity is Linux-specific")
    receipt_path = tmp_path / "receipt.json"
    pid = os.getpid()
    if not Path(f"/proc/{pid}/cmdline").is_file():
        pytest.skip("current process identity is unavailable")
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode().strip()
    receipt_path.write_text(
        json.dumps(
            {
                "server_pid": pid,
                "server_start_time": "not-the-current-process",
                "server_cmdline": cmdline.split(),
                "source_commit": "source",
                "checkpoint_revision": "revision",
                "model": "N3",
                "config": {},
            }
        )
    )
    monkeypatch.setenv("SGW01_RUNTIME_RECEIPT", str(receipt_path))
    with pytest.raises(AdapterError, match="start identity"):
        runtime._load_receipt()
