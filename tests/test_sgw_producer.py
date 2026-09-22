from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading
import urllib.request

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1 import producer
from experiments.workshops.spatial_grounding_v1.adapters import AdapterError


class _Backend:
    resolved_config = dict(producer.NANO_CONFIG)
    source_root = "/pinned/cosmos"
    checkpoint_path = "/pinned/checkpoint"

    def __init__(self) -> None:
        self.calls: list[tuple[object, str, int]] = []

    def predict(self, observation: dict, prompt: str, sampling_seed: int) -> dict:
        self.calls.append((observation, prompt, sampling_seed))
        return {
            "action": np.zeros((32, 8), dtype=np.float32),
            "future": np.ones((33, 2, 2, 3), dtype=np.uint8),
        }


def test_checkpoint_identity_verifies_bytes_not_invented_metadata(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.json").write_bytes(b"{}")
    (checkpoint / "weights.bin").write_bytes(b"synthetic-test-only-payload")
    registry = tmp_path / "registry.json"
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": producer._sha256_bytes(path.read_bytes())}
        for path in checkpoint.iterdir()
    }
    registry.write_text(json.dumps({"checkpoint": {
        "id": producer.NANO_CONFIG["asset"],
        "revision": producer.NANO_CONFIG["revision"],
        "files": files,
    }}))
    monkeypatch.setattr(producer, "CHECKPOINT_REGISTRY", registry)
    assert producer._checkpoint_identity(str(checkpoint)) == (
        producer.NANO_CONFIG["revision"], producer.NANO_CONFIG["asset"]
    )
    (checkpoint / "weights.bin").write_bytes(b"corrupt")
    with pytest.raises(AdapterError, match="hash mismatch"):
        producer._checkpoint_identity(str(checkpoint))
    (checkpoint / "weights.bin").unlink()
    with pytest.raises(AdapterError, match="missing"):
        producer._checkpoint_identity(str(checkpoint))
    registry.write_text(json.dumps({"checkpoint": {"files": {}}}))
    with pytest.raises(AdapterError, match="complete file hashes"):
        producer._checkpoint_identity(str(checkpoint))


def test_source_identity_rejects_tracked_dirt_but_preserves_untracked_files(tmp_path) -> None:
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--quiet")
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    git("add", "source.py")
    git("-c", "user.name=SGW Test", "-c", "user.email=sgw-test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "--quiet", "-m",
        "Synthetic fixture\n\nCo-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>")
    scratch = tmp_path / "unrelated.txt"
    scratch.write_text("preserve\n")
    assert producer._git_revision(str(tmp_path)) == git("rev-parse", "HEAD")
    source.write_text("value = 2\n")
    with pytest.raises(AdapterError, match="tracked modifications"):
        producer._git_revision(str(tmp_path))
    assert scratch.read_text() == "preserve\n"


def test_nano_evidence_producer_writes_wrapper_trace_and_future(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(producer, "_git_revision", lambda _: producer.NANO_CONFIG["source_commit"])
    monkeypatch.setattr(producer, "_checkpoint_revision", lambda _: producer.NANO_CONFIG["revision"])
    monkeypatch.setattr(producer, "_proc_start_identity", lambda _: "start-1")
    backend = _Backend()
    trace_path = tmp_path / "trace.jsonl"
    future_dir = tmp_path / "future"
    attestation_path = tmp_path / "attestation.json"
    wrapper = producer.NanoEvidenceProducer(
        backend,
        trace_path=trace_path,
        future_dir=future_dir,
        attestation_path=attestation_path,
        clock=lambda: 123.5,
    )

    reset = wrapper.reset({"camera_name": "wrist"})
    assert reset["provenance"] == "sgw_wrapper_generated"
    result = wrapper.predict(
        {
            "request_id": "cell:request:0",
            "request_index": 0,
            "registered_cell_id": "cell",
            "camera_id": "camera-0",
            "camera_name": "wrist",
            "reset_id": "adapter-reset-0",
            "reset_fingerprint": "reset-hash",
            "prompt": "static prompt",
            "sampling_seed": 1140,
            "observation": {"native_image": [[1]]},
        }
    )
    assert np.asarray(result["action"]).shape == (32, 8)
    assert backend.calls == [({"native_image": [[1]]}, "static prompt", 1140)]

    record = json.loads(trace_path.read_text().strip())
    assert record["request_id"] == "cell:request:0"
    assert record["wrapper_request_id"].startswith("sgw-n3-request-")
    assert record["reset_id"] == "adapter-reset-0"
    assert record["wrapper_reset_id"].startswith("sgw-reset-")
    assert record["provenance"] == "sgw_wrapper_generated"
    assert record["future_status"] == "exposed_and_retained"
    future_path = Path(record["future_path"])
    assert future_path.is_file()
    assert record["future_sha256"] == producer._sha256_bytes(future_path.read_bytes())
    assert json.loads(attestation_path.read_text())["source_commit"] == producer.NANO_CONFIG["source_commit"]
    with pytest.raises(AdapterError, match="request_index"):
        wrapper.predict(
            {
                "request_id": "cell:request:bad",
                "request_index": 4,
                "registered_cell_id": "cell",
                "camera_id": "camera-0",
                "camera_name": "wrist",
                "reset_id": "adapter-reset-0",
                "reset_fingerprint": "reset-hash",
                "prompt": "static prompt",
                "sampling_seed": 1140,
                "observation": {"native_image": [[1]]},
            }
        )
    assert len(backend.calls) == 1
    with pytest.raises(AdapterError, match="reset binding"):
        wrapper.predict(
            {
                "request_id": "cell:request:1",
                "request_index": 1,
                "registered_cell_id": "cell",
                "camera_id": "camera-0",
                "camera_name": "wrist",
                "reset_id": "adapter-reset-1",
                "reset_fingerprint": "reset-hash",
                "prompt": "static prompt",
                "sampling_seed": 1140,
                "observation": {"native_image": [[1]]},
            }
        )
    assert len(backend.calls) == 1


def test_nano_http_wrapper_routes_reset_and_predict(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(producer, "_git_revision", lambda _: producer.NANO_CONFIG["source_commit"])
    monkeypatch.setattr(producer, "_checkpoint_revision", lambda _: producer.NANO_CONFIG["revision"])
    monkeypatch.setattr(producer, "_proc_start_identity", lambda _: "start-1")
    backend = _Backend()
    evidence = producer.NanoEvidenceProducer(
        backend,
        trace_path=tmp_path / "trace.jsonl",
        future_dir=tmp_path / "future",
        attestation_path=tmp_path / "attestation.json",
    )
    server = producer.make_nano_http_server(evidence, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        def post(path: str, value: dict) -> dict:
            request = urllib.request.Request(
                url + path,
                data=json.dumps(value).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                return json.loads(response.read().decode())

        assert post("/reset", {"camera_name": "wrist"})["status"] == "reset"
        result = post(
            "/predict",
            {
                "request_id": "cell:request:0",
                "request_index": 0,
                "registered_cell_id": "cell",
                "camera_id": "cam",
                "camera_name": "wrist",
                "reset_id": "adapter-reset-0",
                "reset_fingerprint": "reset-hash",
                "prompt": "static",
                "sampling_seed": 1140,
                "observation": {"native_image": [[1]]},
            },
        )
        assert len(result["action"]) == 32
        assert result["provenance"] == "sgw_wrapper_generated"
    finally:
        server.shutdown()
        server.server_close()
