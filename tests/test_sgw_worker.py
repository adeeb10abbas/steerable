import fcntl
import os
from pathlib import Path
import subprocess
import sys

from experiments.workshops.spatial_grounding_v1.contract import load_release
from experiments.workshops.spatial_grounding_v1.worker import EXIT_ATTEMPTS_EXHAUSTED, run_partition
from tests.test_sgw_contract import make_release


class FakeAdapter:
    def __init__(self, status="valid_model_failure"):
        self.status = status
        self.resets = 0

    def reset(self, cell, recorder):
        self.resets += 1
        return {"full_reset": True, "reset_number": self.resets}

    def run_episode(self, cell, recorder, reset):
        recorder.request({"request_id": f"r{self.resets}", "request_index": 0,
                          "started_at_utc": "2026-01-01T00:00:00Z",
                          "prompt_sha256": cell.row["prompt_sha256"], "reset_sha256": "reset"})
        return {"status": self.status, "failure_reason": "policy failure"}

    def close(self):
        pass


def test_worker_preserves_model_failures_and_skips_them_on_resume(tmp_path: Path) -> None:
    release = load_release(make_release(tmp_path))
    adapter = FakeAdapter()
    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6,
                         max_attempts=3, worker_id="test", adapter=adapter) == 0
    assert adapter.resets == 6
    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6,
                         max_attempts=3, worker_id="resume", adapter=adapter) == 0
    assert adapter.resets == 6


def test_attempt_limit_survives_restart(tmp_path: Path) -> None:
    release = load_release(make_release(tmp_path))

    class Invalid(FakeAdapter):
        def run_episode(self, cell, recorder, reset):
            return {"status": "technical_invalid", "technical_cause": "renderer lost"}

    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6,
                         max_attempts=3, worker_id="test", adapter=Invalid()) == EXIT_ATTEMPTS_EXHAUSTED
    assert len(list((release.root.parent / "attempts" / "cell-0").iterdir())) == 3


def test_flock_blocks_a_second_process(tmp_path: Path) -> None:
    lock = tmp_path / "model.lock"
    lock.write_text("")
    with lock.open() as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = (
            "import fcntl,sys; f=open(sys.argv[1]);\n"
            "try: fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
            "except BlockingIOError: raise SystemExit(0)\n"
            "raise SystemExit(1)"
        )
        assert subprocess.run([sys.executable, "-c", code, str(lock)], check=False).returncode == 0
