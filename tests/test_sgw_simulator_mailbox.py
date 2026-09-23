import multiprocessing
from pathlib import Path
import time
from types import SimpleNamespace
import json

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1.simulator_mailbox import MailboxClient, MailboxError, MailboxReceiver, create_mailbox_environment
from experiments.workshops.spatial_grounding_v1.adapters import NanoPolicyAdapter, ProductionAdapter
from experiments.workshops.spatial_grounding_v1.contract import Cell, load_release
from experiments.workshops.spatial_grounding_v1.recorder import AttemptRecorder
from experiments.workshops.spatial_grounding_v1.worker import _canonical_outcome, _load_scorer
from tests.test_sgw_contract import make_release


IDENTITY = {"release_id": "r", "cell_id": "c", "attempt_id": "a", "channel_nonce": "nonce",
            "candidate_sha256": "a" * 64, "binding_sha256": "b" * 64, "simulator_job_uid": "j", "simulator_pod_uid": "p"}


class FakeEnvironment:
    def __init__(self): self.step_count = 0
    def _state(self):
        return {"cube": (0., 0., .1), "bowl": (0., 0., .1), "plate": (0., .2, .1),
                "gripper_holding": False, "cube_height_lift_m": 0., "final_detached_release": True,
                "supported": True, "linear_speed_m_s": 0., "angular_speed_rad_s": 0.,
                "sim_time": self.step_count / 15, "sim_time_s": self.step_count / 15,
                "action_step": self.step_count}
    def reset(self): return SimpleNamespace(snapshot=self._state(), receipt={"reset_id": "r", "camera_id": "c", "camera_name": "c", "fingerprint": "f" * 64, "temporal_cache_reset": True})
    def step(self, action): self.step_count += 1; return {"safety_terminated": False, "count": self.step_count}
    def snapshot(self): return self._state()
    def render_viewport(self): return np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    def policy_observation(self): return {"rgb": self.render_viewport()}
    def close(self): pass


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
