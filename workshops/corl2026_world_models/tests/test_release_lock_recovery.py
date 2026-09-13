"""Focused safety tests for the offline NFS release-lock recovery tool."""
import fcntl
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "scripts/recover_cluster_release_lock.py"


class ReleaseLockRecoveryTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("release_lock_recovery", MODULE)
        self.r = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.r)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name) / "state"
        (self.state / "jobs").mkdir(parents=True)
        (self.state / "control.json").write_text(json.dumps({
            "namespace": self.r.NAMESPACE,
            "control_commit": "a" * 40,
            "control_generation": 7,
            "active_job_ids": [],
            "shutdown": False,
        }))
        (self.state / "release.lock").touch()
        (self.state / "coordinator.lock").touch()
        (self.state / "worker-worker-00.lock").touch()

    def expected(self):
        snap = self.r.snapshot(self.state)
        return dict(
            expected_control_sha256=snap["control"]["sha256"],
            expected_release_device=snap["release_lock"]["device"],
            expected_release_inode=snap["release_lock"]["inode"],
            expected_lifetime_sha256=snap["lifetime_locks"]["sha256"],
            timeout=1.0,
            controllers_offline=True,
        )

    def recover(self, name="receipt.json", **changes):
        values = self.expected()
        values.update(changes)
        return self.r.recover(self.state, self.state / name, **values)

    def add_claim(self, status="succeeded", child_reaped=True):
        job = self.state / "jobs" / "job-001"
        (job / "claim").mkdir(parents=True)
        (job / "result.json").write_text(json.dumps({
            "job_id": "job-001", "status": status, "child_reaped": child_reaped,
        }))

    def test_probe_and_recover_prefer_acquirable_same_inode(self):
        before = self.r.snapshot(self.state)
        probe = self.r.bounded_lock_attempt([self.state / "release.lock"], 1)
        self.assertEqual(probe.response["status"], "acquired")
        probe.close()
        report = self.recover()
        self.assertEqual(report["action"], "same_inode_acquired_no_rotation")
        self.assertEqual(self.r.identity(self.state / "release.lock")["inode"],
                         before["release_lock"]["inode"])
        self.assertFalse(list(self.state.glob("release.lock.quarantine.*")))
        self.assertEqual(json.loads((self.state / "receipt.json").read_text())["action"],
                         "same_inode_acquired_no_rotation")

    def test_two_would_block_results_rotate_and_preserve_quarantine(self):
        self.add_claim()
        old = self.r.identity(self.state / "release.lock")
        held = (self.state / "release.lock").open("r+b")
        self.addCleanup(held.close)
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = self.recover()
        self.assertEqual(report["action"], "rotated_stale_inode")
        self.assertEqual([x["status"] for x in report["release_lock_probes"]],
                         ["would_block", "would_block"])
        quarantine = Path(report["quarantine"]["path"])
        self.assertTrue(quarantine.is_file())
        self.assertEqual((self.r.identity(quarantine)["device"], self.r.identity(quarantine)["inode"]),
                         (old["device"], old["inode"]))
        self.assertNotEqual(self.r.identity(self.state / "release.lock")["inode"], old["inode"])
        self.assertTrue((self.state / "receipt.json").is_file())

    def test_nonterminal_or_unreaped_claim_refuses_without_rotation(self):
        self.add_claim(status="succeeded", child_reaped=False)
        old = self.r.identity(self.state / "release.lock")
        with self.assertRaisesRegex(self.r.RecoveryError, "child_reaped"):
            self.recover()
        self.assertEqual(self.r.identity(self.state / "release.lock")["inode"], old["inode"])
        self.assertFalse((self.state / "receipt.json").exists())

    def test_pinned_identity_mismatch_refuses(self):
        with self.assertRaisesRegex(self.r.RecoveryError, "release inode"):
            self.recover(expected_release_inode=self.expected()["expected_release_inode"] + 1)
        self.assertFalse((self.state / "receipt.json").exists())

    def test_held_lifetime_lock_refuses_before_release_probe(self):
        held = (self.state / "worker-worker-00.lock").open("r+b")
        self.addCleanup(held.close)
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with self.assertRaisesRegex(self.r.RecoveryError, "lifetime lock"):
            self.recover()
        self.assertFalse((self.state / "receipt.json").exists())

    def test_offline_assertion_and_immutable_receipt_are_required(self):
        with self.assertRaisesRegex(self.r.RecoveryError, "controllers-offline"):
            self.recover(controllers_offline=False)
        (self.state / "receipt.json").write_text("existing\n")
        with self.assertRaisesRegex(self.r.RecoveryError, "already exists"):
            self.recover()
        self.assertEqual((self.state / "receipt.json").read_text(), "existing\n")


if __name__ == "__main__":
    unittest.main()
