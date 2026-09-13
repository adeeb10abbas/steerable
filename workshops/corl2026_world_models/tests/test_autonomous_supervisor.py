"""Exercise the detached supervisor with a local fake executable, never a model."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

SUPERVISOR = Path(__file__).resolve().parents[1] / "execution/20260912/autonomy/supervisor.py"
THREAD = "11111111-2222-4333-8444-555555555555"
FAKE_CODE = r'''
import json, os, pathlib, sys, time
root = pathlib.Path(os.environ["FAKE_ROOT"])
calls = root / "calls.jsonl"
prior = calls.read_text().splitlines() if calls.exists() else []
prompt = sys.stdin.read()
with calls.open("a") as f:
    f.write(json.dumps({"args": sys.argv[1:], "prompt": prompt}) + "\n")
plan = json.loads((root / "plan.json").read_text())[len(prior)]
if plan.get("thread", True):
    print(json.dumps({"type": "thread.started", "thread_id": plan.get("uuid", "11111111-2222-4333-8444-555555555555")}), flush=True)
if plan.get("hang"):
    time.sleep(30)
if "stderr" in plan:
    print(plan["stderr"], file=sys.stderr, flush=True)
if "marker" in plan:
    (root / "state" / plan["marker"]).write_text(json.dumps(plan["contents"]))
if plan.get("completed", True):
    print(json.dumps({"type": "turn.completed", "usage": {}}), flush=True)
pathlib.Path(sys.argv[sys.argv.index("-o") + 1]).write_text("fake final output " + str(len(prior)))
sys.exit(plan.get("exit", 0))
'''


class AutonomousSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SUPERVISOR.exists(), "Implement the durable supervisor")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.state = self.root / "state"
        self.state.mkdir()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.fake = self.root / "fake-codex"
        self.fake.write_text("#!" + sys.executable + "\n" + FAKE_CODE)
        self.fake.chmod(0o755)
        self.initial = self.root / "initial.md"
        self.continuation = self.root / "continuation.md"
        self.initial.write_text("Initial authorized work")
        self.continuation.write_text("Continue the same work")
        self.env = {**os.environ, "FAKE_ROOT": str(self.root)}
        self.command = [sys.executable, str(SUPERVISOR), "--codex", str(self.fake),
                        "--repo", str(self.repo), "--state-dir", str(self.state),
                        "--initial-prompt", str(self.initial), "--continuation-prompt", str(self.continuation),
                        "--retry-base", "0.01", "--retry-cap", "0.02",
                        "--attention-poll", "0.02", "--turn-delay", "0"]

    def start_plan(self, plan):
        (self.root / "plan.json").write_text(json.dumps(plan))
        proc = subprocess.Popen(self.command, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        def cleanup():
            if proc.poll() is None:
                proc.terminate()
            proc.communicate(timeout=4)
        self.addCleanup(cleanup)
        return proc

    def wait_state(self, state):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                if self.status().get("state") == state:
                    return self.status()
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(0.02)
        self.fail(f"Supervisor did not enter {state}: {self.status()}")

    def run_plan(self, plan):
        (self.root / "plan.json").write_text(json.dumps(plan))
        return subprocess.run(self.command, env=self.env, capture_output=True, text=True, timeout=8)

    def status(self):
        return json.loads((self.state / "supervisor_status.json").read_text())

    def calls(self):
        p = self.root / "calls.jsonl"
        return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []

    def completion(self):
        return {"marker": "completion.json", "contents": {"status": "complete", "summary": "Verified output", "evidence": ["result.json"], "safe_to_stop": True}}

    def test_completed_turn_resumes_observed_uuid_and_preserves_logs(self):
        result = self.run_plan([{}, self.completion()])
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["prompt"], "Initial authorized work")
        self.assertEqual(calls[1]["prompt"], "Continue the same work")
        self.assertEqual(calls[0]["args"][:7], ["-a", "never", "-s", "danger-full-access", "-C", str(self.repo.resolve()), "exec"])
        self.assertIn("resume", calls[1]["args"])
        self.assertEqual(calls[1]["args"][-2:], [THREAD, "-"])
        self.assertNotIn("--last", calls[1]["args"])
        self.assertNotIn("-m", calls[1]["args"])
        self.assertEqual(self.status()["thread_id"], THREAD)
        self.assertEqual(self.status()["state"], "complete")
        self.assertEqual(len(list(self.state.glob("turn-*.jsonl"))), 2)
        self.assertEqual(len(list(self.state.glob("turn-*.last-message.txt"))), 2)

    def test_restart_resumes_checkpoint_and_does_not_overwrite_old_output(self):
        (self.state / "supervisor_status.json").write_text(json.dumps({"schema_version": 1, "state": "stopped", "turn": 4, "thread_id": THREAD, "child_pid": None}))
        old = self.state / "turn-000004.jsonl"
        old.write_text("old immutable output\n")
        result = self.run_plan([self.completion()])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[0]["args"][-2:], [THREAD, "-"])
        self.assertEqual(old.read_text(), "old immutable output\n")
        self.assertEqual(self.status()["turn"], 5)

    def test_transient_failure_retries_same_uuid_with_bounded_backoff(self):
        result = self.run_plan([{"exit": 1, "completed": False, "stderr": "network connection reset"}, self.completion()])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[1]["args"][-2:], [THREAD, "-"])
        events = [json.loads(x) for x in (self.state / "supervisor_events.jsonl").read_text().splitlines()]
        retries = [e for e in events if e["state"] == "backoff"]
        self.assertEqual(retries[0]["retry_delay_s"], 0.01)
        self.assertEqual(self.status()["consecutive_failures"], 0)

    def test_restart_honors_persisted_future_retry_time(self):
        saved = {"schema_version": 1, "state": "backoff", "turn": 3, "thread_id": THREAD,
                 "child_pid": None, "consecutive_failures": 19, "next_retry_at": time.time() + 0.3}
        (self.state / "supervisor_status.json").write_text(json.dumps(saved))
        started = time.monotonic()
        proc = self.start_plan([self.completion()])
        time.sleep(0.1)
        self.assertEqual(self.calls(), [])
        _, stderr = proc.communicate(timeout=4)
        self.assertEqual(proc.returncode, 0, stderr)
        self.assertGreaterEqual(time.monotonic() - started, 0.25)
        self.assertEqual(self.calls()[0]["args"][-2:], [THREAD, "-"])

    def test_transient_failures_continue_past_former_limit_until_recovery(self):
        failure = {"exit": 1, "completed": False, "stderr": "429 rate limit"}
        result = self.run_plan([failure] * 14 + [self.completion()])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 15)
        self.assertEqual(self.status()["state"], "complete")
        events = [json.loads(x) for x in (self.state / "supervisor_events.jsonl").read_text().splitlines()]
        self.assertEqual([e["retry_delay_s"] for e in events if e["state"] == "backoff"], [0.01] + [0.02] * 13)
        self.assertTrue(all(e["next_retry_at"] > e["updated_at"] for e in events if e["state"] == "backoff"))
        self.assertFalse(any(e["state"] in ("needs_input", "needs_attention") for e in events))

    def test_unknown_resume_id_keeps_watchdog_alive_until_explicit_resolution(self):
        proc = self.start_plan([{}, {"thread": False, "completed": False, "exit": 1, "stderr": "Error: session not found for requested UUID"}, self.completion()])
        self.wait_state("needs_attention")
        time.sleep(0.08)
        self.assertIsNone(proc.poll())
        self.assertEqual(len(self.calls()), 2)
        self.assertEqual(self.status()["thread_id"], THREAD)
        self.assertIn("resume", self.status()["reason"])
        self.assertTrue((self.state / "turn-000001.jsonl").exists())
        (self.state / "operator_resolution.json").write_text(json.dumps({"action": "retry", "thread_id": THREAD}))
        _, stderr = proc.communicate(timeout=4)
        self.assertEqual(proc.returncode, 0, stderr)
        self.assertEqual(self.calls()[2]["args"][-2:], [THREAD, "-"])
        self.assertEqual(self.status()["state"], "complete")

    def test_safe_needs_input_marker_stops_without_launch(self):
        (self.state / "needs_input.json").write_text(json.dumps({"status": "needs_input", "reason": "Missing approved resource", "safe_to_stop": True}))
        result = self.run_plan([])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertEqual(self.status()["state"], "needs_input")

    def test_unsafe_receipts_are_preserved_and_corrected_in_same_thread(self):
        invalid_complete = self.completion()
        invalid_complete["contents"].pop("safe_to_stop")
        invalid_needs = {"marker": "needs_input.json", "contents": {"reason": "Jobs still need monitoring", "safe_to_stop": False}}
        result = self.run_plan([invalid_complete, invalid_needs, self.completion()])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 3)
        self.assertEqual(self.calls()[1]["args"][-2:], [THREAD, "-"])
        self.assertIn("safe_to_stop", self.calls()[1]["prompt"])
        self.assertIn("monitor", self.calls()[2]["prompt"])
        self.assertEqual(len(list(self.state.glob("*.rejected-*.json"))), 2)
        self.assertEqual(self.status()["state"], "complete")

    def test_malformed_completion_is_rejected_without_abandoning_work(self):
        (self.state / "completion.json").write_text(json.dumps({"status": "complete", "summary": "", "evidence": [], "safe_to_stop": True}))
        result = self.run_plan([self.completion()])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.status()["state"], "complete")
        self.assertIn("receipt", self.calls()[0]["prompt"])
        self.assertEqual(len(list(self.state.glob("*.rejected-*.json"))), 1)

    def test_logged_thread_started_recovers_uuid_after_checkpoint_crash(self):
        (self.state / "supervisor_status.json").write_text(json.dumps({"schema_version": 1, "state": "running", "turn": 1, "thread_id": None, "child_pid": None}))
        (self.state / "turn-000001.jsonl").write_text(json.dumps({"type": "thread.started", "thread_id": THREAD}) + "\n")
        result = self.run_plan([self.completion()])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[0]["args"][-2:], [THREAD, "-"])
        self.assertEqual(self.status()["turn"], 2)

    def test_resume_uuid_mismatch_never_replaces_original_identity(self):
        proc = self.start_plan([{}, {"uuid": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"}])
        self.wait_state("needs_attention")
        self.assertIsNone(proc.poll())
        self.assertEqual(self.status()["thread_id"], THREAD)
        self.assertIn("different thread UUID", self.status()["reason"])
        (self.state / "needs_input.json").write_text(json.dumps({"status": "needs_input", "reason": "Monitoring independently reconciled", "safe_to_stop": True}))
        _, stderr = proc.communicate(timeout=4)
        self.assertEqual(proc.returncode, 0, stderr)

    def test_needs_input_without_matching_status_is_rejected(self):
        marker = {"marker": "needs_input.json", "contents": {"reason": "Help needed", "safe_to_stop": True}}
        wrong = {"marker": "needs_input.json", "contents": {"status": "complete", "reason": "Help needed", "safe_to_stop": True}}
        result = self.run_plan([marker, wrong, self.completion()])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 3)
        self.assertEqual(self.status()["state"], "complete")
        self.assertEqual(len(list(self.state.glob("needs_input.rejected-*.json"))), 2)

    def test_unresolved_child_survives_restart_guard_and_explicit_retry(self):
        saved = {"schema_version": 1, "state": "needs_input", "turn": 2,
                 "thread_id": THREAD, "child_pid": None, "previous_child_pid": os.getpid()}
        (self.state / "supervisor_status.json").write_text(json.dumps(saved))
        self.command.append("--continue-after-input")
        proc = self.start_plan([])
        status = self.wait_state("needs_attention")
        self.assertEqual(status["unresolved_child_pid"], os.getpid())
        self.assertEqual(self.calls(), [])
        (self.state / "operator_resolution.json").write_text(json.dumps({"action": "retry", "thread_id": THREAD}))
        time.sleep(0.12)
        self.assertEqual(self.calls(), [])
        self.assertIsNone(proc.poll())
        self.assertEqual(self.status()["unresolved_child_pid"], os.getpid())
        proc.terminate()
        proc.communicate(timeout=4)
        self.assertEqual(self.status()["unresolved_child_pid"], os.getpid())

    def test_valid_completion_marker_remains_stopped_after_restart(self):
        self.run_plan([self.completion()])
        result = self.run_plan([])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(self.status()["state"], "complete")

    def test_singleton_lock_does_not_overwrite_running_status(self):
        expected = {"state": "running", "pid": os.getpid()}
        (self.state / "supervisor_status.json").write_text(json.dumps(expected))
        with (self.state / "supervisor.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_plan([])
        self.assertEqual(result.returncode, 73)
        self.assertEqual(self.status(), expected)
        self.assertEqual(self.calls(), [])

    def test_sigterm_checkpoints_and_only_forwards_to_its_child(self):
        (self.root / "plan.json").write_text(json.dumps([{"hang": True}]))
        proc = subprocess.Popen(self.command, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(lambda: proc.kill() if proc.poll() is None else None)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                if self.status().get("thread_id") == THREAD:
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(0.02)
        self.assertEqual(self.status().get("thread_id"), THREAD)
        proc.send_signal(signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=4)
        self.assertEqual(proc.returncode, 0, stderr)
        self.assertEqual(self.status()["state"], "stopped")
        self.assertEqual(self.status()["thread_id"], THREAD)
        self.assertIsNone(self.status()["child_pid"])


if __name__ == "__main__":
    unittest.main()
