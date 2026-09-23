import multiprocessing
from pathlib import Path
import time
from types import SimpleNamespace
import json
import threading

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1 import simulator_mailbox
from experiments.workshops.spatial_grounding_v1.native_mailbox_receiver import _load_bound_cell, verify_receiver_completion
from experiments.workshops.spatial_grounding_v1.simulator_mailbox import MailboxClient, MailboxError, MailboxReceiver, create_mailbox_environment
from experiments.workshops.spatial_grounding_v1.adapters import NanoPolicyAdapter, ProductionAdapter
from experiments.workshops.spatial_grounding_v1.contract import Cell, load_release
from experiments.workshops.spatial_grounding_v1.recorder import AttemptRecorder
from experiments.workshops.spatial_grounding_v1.worker import _canonical_outcome, _load_scorer
from tests.test_sgw_contract import make_release


IDENTITY = {"release_id": "r", "cell_id": "c", "attempt_id": "a", "channel_nonce": "nonce",
            "candidate_sha256": "a" * 64, "binding_sha256": "b" * 64, "simulator_job_uid": "j", "simulator_pod_uid": "p"}


class FakeEnvironment:
    def __init__(self): self.step_count = 0; self.closed = False; self.close_count = 0
    def _open(self):
        if self.closed: raise RuntimeError("native environment was accessed after close")
    def _state(self):
        self._open()
        return {"cube": (0., 0., .1), "bowl": (0., 0., .1), "plate": (0., .2, .1),
                "gripper_holding": False, "cube_height_lift_m": 0., "final_detached_release": True,
                "supported": True, "linear_speed_m_s": 0., "angular_speed_rad_s": 0.,
                "sim_time": self.step_count / 15, "sim_time_s": self.step_count / 15,
                "action_step": self.step_count}
    def reset(self): return SimpleNamespace(snapshot=self._state(), receipt={"reset_id": "r", "camera_id": "c", "camera_name": "c", "fingerprint": "f" * 64, "temporal_cache_reset": True})
    def step(self, action): self._open(); self.step_count += 1; return {"safety_terminated": False, "count": self.step_count}
    def snapshot(self): return self._state()
    def render_viewport(self): self._open(); return np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    def policy_observation(self):
        self._open()
        return {"image_obs": {"camera/0": self.render_viewport()}, "proprio.obs": np.zeros(7, dtype=np.float32)}
    def close(self):
        if self.closed: raise RuntimeError("double close")
        self.close_count += 1; self.closed = True


def _receiver(root: str):
    receiver = MailboxReceiver(root=Path(root), identity=IDENTITY, environment=FakeEnvironment())
    while not receiver.closed:
        for request in sorted((Path(root) / "requests").glob("*.json")):
            if request.stem.split("-")[0].isdigit() and int(request.stem[:4]) > receiver.last:
                receiver.serve_one(request)
        time.sleep(.001)


def test_450_action_exchange_caches_response_and_rejects_timeout(tmp_path: Path) -> None:
    root = tmp_path / "mail"; (root / "requests").mkdir(parents=True); (root / "responses").mkdir()
    process = multiprocessing.Process(target=_receiver, args=(str(root),)); process.start()
    client = MailboxClient(root=root, identity=IDENTITY, timeout_s=2)
    reset = client.reset()
    assert reset.receipt["reset_id"] == "r"
    assert client.policy_observation()["image_obs"]["camera/0"].shape == (2, 2, 3)
    assert client.policy_observation()["proprio.obs"].shape == (7,)
    for index in range(450):
        assert client.step(np.zeros(8, dtype=np.float32))["count"] == index + 1
    assert client.snapshot()["action_step"] == 450
    assert client.render_viewport().shape == (2, 2, 3)
    client.close(); process.join(2); assert process.exitcode == 0
    with pytest.raises(MailboxError):
        MailboxClient(root=root, identity=IDENTITY, timeout_s=.01).reset()


def test_receiver_rejects_duplicate_command_and_preserves_fault(tmp_path: Path) -> None:
    root = tmp_path / "mail"; (root / "requests").mkdir(parents=True); (root / "responses").mkdir()
    receiver = MailboxReceiver(root=root, identity=IDENTITY, environment=FakeEnvironment())
    request = root / "requests" / "0001-reset.json"
    request.write_text(json.dumps({"command_id": 1, "identity": IDENTITY, "operation": "reset"}))
    receiver.serve_one(request)
    with pytest.raises(MailboxError):
        receiver.serve_one(request)
    assert (root / "faults" / "0001-reset.json").is_file()


def test_production_adapter_recorder_and_strict_scorer_over_mailbox(tmp_path: Path) -> None:
    root = tmp_path / "mail"; (root / "requests").mkdir(parents=True); (root / "responses").mkdir()
    process = multiprocessing.Process(target=_receiver, args=(str(root),)); process.start()
    release = load_release(make_release(tmp_path))
    row = dict(release.partition("N3", "LAT", "P")[0].row)
    row.update({"sampling_seed": 7, "physical_goal_sign": 1, "form": "D"})
    cell = Cell(row)
    recorder = AttemptRecorder(release, cell, "attempt-001"); recorder.begin()
    def transport(request):
        return {"request_id": request["request_id"], "registered_cell_id": request["registered_cell_id"],
                "request_index": request["request_index"], "reset_id": request["reset_id"],
                "camera_id": request["camera_id"], "camera_name": request["camera_name"],
                "reset_fingerprint": request["reset_fingerprint"], "actions": np.zeros((32, 8), dtype=np.float32)}
    adapter = ProductionAdapter(NanoPolicyAdapter, transport=transport, transport_factory=lambda **_: transport,
                                environment_factory=lambda **_: MailboxClient(root=root, identity=IDENTITY, timeout_s=2))
    reset = adapter.reset(cell, recorder)
    raw = adapter.run_episode(cell, recorder, reset)
    outcome = _canonical_outcome({**raw, "attempt_id": recorder.attempt_id}, cell, _load_scorer())
    assert outcome["status"] == "valid_model_failure" and outcome["terminal_step"] == 450
    assert recorder.complete(outcome)
    adapter.close(); process.join(3); assert process.exitcode == 0
    assert len(list((recorder.path / "actions").glob("*.npy"))) == 450
    assert (root / "receiver_complete.json").is_file()
    assert verify_receiver_completion(root, IDENTITY)["command_count"] == 452


def test_explicit_factory_requires_hash_bound_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "mail"; root.mkdir()
    identity = tmp_path / "identity.json"; identity.write_text(json.dumps(IDENTITY))
    monkeypatch.setenv("SGW01_SIMULATOR_MAILBOX_ROOT", str(root))
    monkeypatch.setenv("SGW01_SIMULATOR_MAILBOX_IDENTITY", str(identity))
    import hashlib
    monkeypatch.setenv("SGW01_SIMULATOR_MAILBOX_IDENTITY_SHA256", hashlib.sha256(identity.read_bytes()).hexdigest())
    cell = SimpleNamespace(row={"cell_id": IDENTITY["cell_id"]})
    client = create_mailbox_environment(cell=cell, evidence_root=tmp_path)
    assert client.identity == IDENTITY


def test_atomic_no_overwrite_publication_never_exposes_partial_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "manifest.json"
    entered, release = threading.Event(), threading.Event()
    original = simulator_mailbox._bytes
    def delayed(value):
        entered.set(); assert release.wait(2)
        return original(value)
    monkeypatch.setattr(simulator_mailbox, "_bytes", delayed)
    thread = threading.Thread(target=simulator_mailbox._write, args=(target, {"answer": 42}))
    thread.start(); assert entered.wait(2)
    assert not target.exists()
    release.set(); thread.join(2)
    assert json.loads(target.read_text()) == {"answer": 42}
    with pytest.raises(FileExistsError):
        simulator_mailbox._write(target, {"answer": 43})


def test_corrupt_response_closes_client_and_nested_observations_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "mail"; (root / "requests").mkdir(parents=True); (root / "responses").mkdir()
    response = {"command_id": 1, "identity": IDENTITY, "status": "ok",
                "data": {"viewport": {"path": "missing.npy"}, "policy_observation": {"mapping": {}}}}
    (root / "responses" / "0001-reset.json").write_text(json.dumps(response))
    client = MailboxClient(root=root, identity=IDENTITY, timeout_s=1)
    with pytest.raises(MailboxError):
        client.reset()
    with pytest.raises(MailboxError):
        client.reset()
    assert client._closed


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_client_rejects_nonfinite_or_nonpositive_timeout(tmp_path: Path, timeout: float) -> None:
    with pytest.raises(MailboxError):
        MailboxClient(root=tmp_path, identity=IDENTITY, timeout_s=timeout)


def test_completion_validator_rejects_failure_and_corrupt_receipt(tmp_path: Path) -> None:
    root = tmp_path / "mail"; (root / "responses").mkdir(parents=True)
    response = root / "responses" / "0001-close.json"
    response.write_text(json.dumps({"status": "ok", "identity": IDENTITY}))
    receipt = {"schema": "sgw-01-mailbox-receiver-completion-v1",
               "attempt_scope": "learned_policy_remote_simulator", "identity": IDENTITY,
               "close_command_id": 1, "command_count": 1,
               "close_response_sha256": __import__("hashlib").sha256(response.read_bytes()).hexdigest()}
    (root / "receiver_complete.json").write_text(json.dumps(receipt))
    assert verify_receiver_completion(root, IDENTITY)["command_count"] == 1
    (root / "receiver_failure.json").write_text("{}")
    with pytest.raises(Exception):
        verify_receiver_completion(root, IDENTITY)


def test_receiver_binds_identity_to_hashed_release_cell_bytes(tmp_path: Path) -> None:
    cell = tmp_path / "cell.json"
    cell.write_text(json.dumps({"cell_id": "c", "candidate_sha256": "a" * 64, "binding_sha256": "b" * 64}))
    import hashlib
    assert _load_bound_cell(cell, hashlib.sha256(cell.read_bytes()).hexdigest(), IDENTITY)["cell_id"] == "c"
    cell.write_text(json.dumps({"cell_id": "wrong", "candidate_sha256": "a" * 64, "binding_sha256": "b" * 64}))
    with pytest.raises(Exception):
        _load_bound_cell(cell, hashlib.sha256(cell.read_bytes()).hexdigest(), IDENTITY)
