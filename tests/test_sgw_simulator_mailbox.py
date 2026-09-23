import multiprocessing
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1.simulator_mailbox import MailboxClient, MailboxError, MailboxReceiver


IDENTITY = {"release_id": "r", "cell_id": "c", "attempt_id": "a", "channel_nonce": "nonce",
            "candidate_sha256": "a" * 64, "binding_sha256": "b" * 64, "simulator_job_uid": "j", "simulator_pod_uid": "p"}


class FakeEnvironment:
    def __init__(self): self.step_count = 0
    def reset(self): return SimpleNamespace(snapshot={"action_step": 0}, receipt={"reset_id": "r", "camera_id": "c", "camera_name": "c", "fingerprint": "f", "temporal_cache_reset": True})
    def step(self, action): self.step_count += 1; return {"safety_terminated": False, "count": self.step_count}
    def snapshot(self): return {"action_step": self.step_count}
    def render_viewport(self): return np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    def policy_observation(self): return {"observation/image": self.render_viewport(), "observation/joint_position": np.zeros(7, dtype=np.float32)}
    def close(self): pass


def _receiver(root: str):
    receiver = MailboxReceiver(root=Path(root), identity=IDENTITY, environment=FakeEnvironment())
    while not receiver.closed:
        for request in sorted((Path(root) / "requests").glob("*.json")):
            if request.stem.split("-")[0].isdigit() and int(request.stem[:4]) > receiver.last:
                receiver.serve_one(request)
        time.sleep(.001)


def test_exchange_caches_response_and_rejects_timeout(tmp_path: Path) -> None:
    root = tmp_path / "mail"; (root / "requests").mkdir(parents=True); (root / "responses").mkdir()
    process = multiprocessing.Process(target=_receiver, args=(str(root),)); process.start()
    client = MailboxClient(root=root, identity=IDENTITY, timeout_s=2)
    reset = client.reset()
    assert reset.receipt["reset_id"] == "r"
    assert client.step(np.zeros(8, dtype=np.float32))["count"] == 1
    assert client.snapshot()["action_step"] == 1
    assert client.render_viewport().shape == (2, 2, 3)
    client.close(); process.join(2); assert process.exitcode == 0
    with pytest.raises(MailboxError):
        MailboxClient(root=root, identity=IDENTITY, timeout_s=.01).reset()
