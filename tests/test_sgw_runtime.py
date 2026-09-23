from __future__ import annotations

import json
import os
from pathlib import Path

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
    def __init__(self, chunks: int) -> None:
        self.returned_chunks: list[np.ndarray] = []
        self.chunks = chunks

    def infer(self, observation: object, prompt: str) -> None:
        for _ in range(self.chunks):
            self.returned_chunks.append(np.zeros((24, 8), dtype=np.float32))


def test_dreamzero_requires_exactly_one_fresh_chunk_per_request() -> None:
    adapter = runtime._OfficialDreamZeroClient.__new__(runtime._OfficialDreamZeroClient)
    adapter.client = _DreamZeroClient(1)
    adapter.trace_reader = lambda **_: {"request_id": "r1"}
    result = adapter({"request_id": "r1", "observation": {}, "prompt": "static"})
    assert np.asarray(result["actions"]).shape == (24, 8)

    adapter.client = _DreamZeroClient(2)
    with pytest.raises(AdapterError, match="unexpected number"):
        adapter({"request_id": "r1", "observation": {}, "prompt": "static"})


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
