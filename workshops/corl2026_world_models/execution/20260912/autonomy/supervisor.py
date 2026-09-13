#!/usr/bin/env python3
"""Durably continue one explicitly identified Codex task; never run experiments itself.

Codex 0.153.4 flags were checked with exec/resume --help. Prompts arrive on
stdin; no model override, shell, resume --last, or new-session fallback is used.
Stop receipts live in --state-dir: completion.json requires status=complete,
a nonempty summary and nonempty list of evidence strings; needs_input.json
requires a nonempty reason. Both require safe_to_stop=true. Rejected receipts
are retained separately and trigger corrective continuation, not abandonment
of monitored work. Scientific claim validation belongs to the agent.

Transient failures have capped exponential backoff and a finite retry budget.
An operator may resolve a needs-input state, remove its marker, and restart with
--continue-after-input; the recorded thread identity and old outputs are retained.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time
import uuid

TRANSIENT = re.compile(
    r"network|connection (?:reset|refused|aborted)|timed? out|timeout|rate.?limit|"
    r"usage limit|\b(?:429|502|503|504)\b|temporar(?:y|ily)|service unavailable|"
    r"transport error|stream disconnected|failed to (?:connect|resolve)|unexpected eof", re.I)
UNKNOWN_SESSION = re.compile(
    r"(?:session|thread|conversation).{0,120}(?:not found|unknown|does not exist|invalid)|"
    r"(?:unknown|invalid).{0,60}(?:session|thread|uuid)|no (?:session|thread).{0,80}found", re.I)


def atomic_json(path, value):
    """fsync data and directory around replacement so status survives a restart."""
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def valid_uuid(value):
    if not isinstance(value, str):
        raise ValueError("thread.started did not supply a UUID string")
    parsed = str(uuid.UUID(value))
    if parsed != value.lower():
        raise ValueError("Thread identity must use canonical UUID syntax")
    return parsed


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def retry_delay(failures, base, cap):
    # Bounded exponent also avoids overflow after a damaged or very old counter.
    return min(cap, base * 2 ** min(max(0, failures - 1), 30))


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.directory = args.state_dir
        self.status_path = self.directory / "supervisor_status.json"
        self.child = None
        self.stop_signal = None
        self.stop_time = None
        self.status = {"schema_version": 1, "state": "starting", "turn": 0,
                       "thread_id": None, "child_pid": None, "consecutive_failures": 0}

    def checkpoint(self, state, **fields):
        self.status.update(fields)
        self.status.update(state=state, pid=os.getpid(), updated_at=time.time(), repo=str(self.args.repo))
        atomic_json(self.status_path, self.status)
        with (self.directory / "supervisor_events.jsonl").open("a", encoding="utf-8") as events:
            events.write(json.dumps(self.status, sort_keys=True, allow_nan=False) + "\n")
            events.flush()
            os.fsync(events.fileno())

    def park(self, reason, **fields):
        self.checkpoint("needs_input", reason=reason, child_pid=None, **fields)
        return 0

    def on_signal(self, signum, _frame):
        if self.stop_signal is None:
            self.stop_signal, self.stop_time = signum, time.monotonic()
        if self.child is not None and self.child.poll() is None:
            # Signal only the child we created; never a process group or cluster job.
            try:
                self.child.send_signal(signum)
            except ProcessLookupError:
                pass

    def stop_owned_child(self):
        """On supervisor errors, leave no unmanaged agent process behind."""
        if self.child is not None:
            if self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=self.args.signal_grace)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait()
            self.child = None

    def wait(self, seconds):
        end = time.monotonic() + max(0, seconds)
        while not self.stop_signal and time.monotonic() < end:
            time.sleep(min(0.2, end - time.monotonic()))

    def markers(self):
        complete = self.directory / "completion.json"
        needs_input = self.directory / "needs_input.json"
        if complete.exists() and needs_input.exists():
            return "rejected", {"reason": "Conflicting completion.json and needs_input.json receipts", "paths": [complete, needs_input]}
        for path in (needs_input, complete):
            if not path.exists():
                continue
            try:
                value = json.loads(path.read_text())
                if not isinstance(value, dict):
                    raise ValueError("receipt must be an object")
                if value.get("safe_to_stop") is not True:
                    raise ValueError("safe_to_stop must be exactly true after jobs are reconciled or independently monitored")
                if path == needs_input:
                    if not nonempty(value.get("reason")):
                        raise ValueError("needs_input reason must be nonempty")
                    return "needs_input", {"reason": value["reason"], "receipt": value}
                if (value.get("status") != "complete" or not nonempty(value.get("summary"))
                        or not isinstance(value.get("evidence"), list) or not value["evidence"]
                        or not all(nonempty(item) for item in value["evidence"])):
                    raise ValueError("completion requires status=complete, summary, and evidence strings")
                return "complete", {"reason": None, "receipt": value}
            except (OSError, ValueError) as error:
                return "rejected", {"reason": f"Malformed stop receipt {path.name}: {error}", "paths": [path]}
        return None

    def apply_markers(self):
        marker = self.markers()
        if not marker:
            return False
        state, fields = marker
        if state != "rejected":
            self.checkpoint(state, child_pid=None, consecutive_failures=0, **fields)
            return True
        retained = []
        for path in fields["paths"]:
            destination = path.with_name(f"{path.stem}.rejected-{time.time_ns()}.json")
            os.replace(path, destination)
            retained.append(str(destination))
        correction = (
            "Supervisor rejected the stop receipt: " + fields["reason"] + ". "
            "Continue the same authorized work and monitor/reconcile active jobs. "
            "Do not abandon compute monitoring. Write a new valid receipt only after "
            "all jobs are finished or independently durably monitored and reconciled; "
            "include safe_to_stop: true. Completion also requires status: complete, "
            "a nonempty summary and a nonempty list of evidence strings; needs_input "
            "requires a nonempty reason. Prior rejected receipts remain preserved."
        )
        self.checkpoint("receipt_rejected", rejected_receipts=retained,
                        receipt_correction=correction, reason=fields["reason"])
        return False

    def restore(self):
        if self.status_path.exists():
            raw = self.status_path.read_text()
            try:
                saved = json.loads(raw)
                if (not isinstance(saved, dict) or saved.get("schema_version") != 1
                        or type(saved.get("turn")) is not int or saved["turn"] < 0):
                    raise ValueError("Unsupported or malformed supervisor checkpoint")
                self.status.update(saved)
                if self.status["thread_id"] is not None:
                    self.status["thread_id"] = valid_uuid(self.status["thread_id"])
                failures = self.status.get("consecutive_failures", 0)
                if type(failures) is not int or failures < 0:
                    raise ValueError("Malformed retry counter")
            except (ValueError, KeyError) as error:
                return self.park(f"Cannot restore checkpoint: {error}", preserved_invalid_status=raw)
        previous_child = self.status.get("child_pid")
        if previous_child is not None:
            try:
                os.kill(previous_child, 0)
            except ProcessLookupError:
                pass
            except (PermissionError, TypeError, ValueError):
                return self.park("Cannot rule out a surviving child; inspect before resuming", previous_child_pid=previous_child)
            else:
                return self.park("A prior child PID is still alive; inspect before resuming", previous_child_pid=previous_child)
        self.status["child_pid"] = None
        # A crash can occur after fsync of thread.started but before the status replace.
        if self.status["thread_id"] is None:
            observed = set()
            for path in sorted(self.directory.glob("turn-*.jsonl")):
                for line in path.read_text(errors="replace").splitlines():
                    try:
                        event = json.loads(line)
                        if isinstance(event, dict) and event.get("type") == "thread.started":
                            observed.add(valid_uuid(event.get("thread_id")))
                    except (ValueError, AttributeError):
                        continue
            if len(observed) > 1:
                return self.park("Multiple prior thread UUIDs in retained output; choose explicitly")
            if observed:
                self.status["thread_id"] = observed.pop()
            elif self.status["turn"] and self.status["state"] in ("running", "starting", "stopped"):
                return self.park("Interrupted request has no recoverable thread UUID; inspect retained output")
        # Preserve any output emitted after the last successfully persisted counter.
        for path in self.directory.glob("turn-*.jsonl"):
            match = re.fullmatch(r"turn-(\d+)\.jsonl", path.name)
            if match:
                self.status["turn"] = max(self.status["turn"], int(match.group(1)))
        if self.apply_markers():
            return 0
        if self.status["state"] == "complete":
            self.checkpoint("complete", child_pid=None)
            return 0
        if self.status["state"] == "needs_input":
            if not self.args.continue_after_input:
                self.checkpoint("needs_input", child_pid=None)
                return 0
            self.status.update(consecutive_failures=0, next_retry_at=None)
        return None

    def command(self, final_path):
        command = [str(self.args.codex), "-a", "never", "-s", "danger-full-access", "-C", str(self.args.repo), "exec"]
        if self.status["thread_id"]:
            return [*command, "resume", "--json", "-o", str(final_path), self.status["thread_id"], "-"]
        return [*command, "--json", "--color", "never", "-o", str(final_path), "-"]

    def run_turn(self):
        self.status["turn"] += 1
        prefix = self.directory / f"turn-{self.status['turn']:06d}"
        stdout_path = prefix.with_suffix(".jsonl")
        stderr_path = prefix.with_suffix(".stderr.log")
        final_path = prefix.with_suffix(".last-message.txt")
        prompt_path = self.args.continuation_prompt if self.status["thread_id"] else self.args.initial_prompt
        prompt = prompt_path.read_bytes()
        if not prompt.strip():
            raise ValueError(f"Prompt is empty: {prompt_path}")
        if self.status.get("receipt_correction"):
            prompt += ("\n\n" + self.status["receipt_correction"]).encode()
        final_path.touch(exist_ok=True)
        self.checkpoint("starting", child_pid=None, reason=None, stdout_log=str(stdout_path),
                        stderr_log=str(stderr_path), last_message=str(final_path), next_retry_at=None)
        self.child = subprocess.Popen(self.command(final_path), cwd=self.args.repo,
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      start_new_session=True)
        if self.stop_signal:
            self.child.send_signal(self.stop_signal)
        self.checkpoint("running", child_pid=self.child.pid)
        try:
            self.child.stdin.write(prompt)
            self.child.stdin.close()
        except BrokenPipeError:
            pass
        completed = False
        protocol_error = None
        failed_event = False
        tail = b""
        pending = b""
        exited_at = None
        with stdout_path.open("ab") as output, stderr_path.open("ab") as errors, selectors.DefaultSelector() as selector:
            selector.register(self.child.stdout, selectors.EVENT_READ, output)
            selector.register(self.child.stderr, selectors.EVENT_READ, errors)
            while selector.get_map() or self.child.poll() is None:
                if self.stop_signal and time.monotonic() - self.stop_time > self.args.signal_grace:
                    if self.child.poll() is None:
                        self.child.kill()  # Only the exact child process, never its descendants.
                for key, _ in selector.select(timeout=0.2):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    key.data.write(chunk)
                    key.data.flush()
                    os.fsync(key.data.fileno())
                    tail = (tail + chunk)[-32768:]
                    if key.data is not output:
                        continue
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        try:
                            event = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            continue
                        if not isinstance(event, dict):
                            continue
                        if event.get("type") == "thread.started":
                            try:
                                identity = valid_uuid(event.get("thread_id"))
                                if self.status["thread_id"] and identity != self.status["thread_id"]:
                                    raise ValueError("Resume returned a different thread UUID")
                                self.checkpoint("running", thread_id=identity, child_pid=self.child.pid)
                            except ValueError as error:
                                protocol_error = str(error)
                                if self.child.poll() is None:
                                    self.child.terminate()
                        completed |= event.get("type") == "turn.completed"
                        failed_event |= event.get("type") == "turn.failed"
                if self.child.poll() is not None:
                    exited_at = exited_at or time.monotonic()
                    # Detached descendants may retain stdout; they do not own this turn.
                    if time.monotonic() - exited_at > 1:
                        break
        for stream in (self.child.stdout, self.child.stderr):
            stream.close()
        returncode = self.child.wait()
        self.child = None
        self.checkpoint("turn_finished", child_pid=None, returncode=returncode)
        return returncode, completed and not failed_event, protocol_error, tail.decode(errors="replace")

    def run(self):
        restored = self.restore()
        if restored is not None:
            return restored
        if self.status.get("next_retry_at"):
            self.wait(self.status["next_retry_at"] - time.time())
        while not self.stop_signal:
            if self.apply_markers():
                return 0
            try:
                code, completed, protocol_error, tail = self.run_turn()
            except (OSError, ValueError) as error:
                self.stop_owned_child()
                return self.park(f"Supervisor launch or state error: {error}")
            if self.stop_signal:
                break
            if protocol_error:
                return self.park(protocol_error)
            if self.apply_markers():
                return 0
            if code == 0 and completed:
                if self.status["thread_id"] is None:
                    return self.park("Completed turn had no observed thread UUID; cannot resume explicitly")
                self.checkpoint("continuing", consecutive_failures=0, reason=None)
                self.wait(self.args.turn_delay)
                continue
            if UNKNOWN_SESSION.search(tail):
                return self.park("Explicit resume identity failed; prior UUID and output retained for inspection", error_tail=tail)
            if not TRANSIENT.search(tail):
                return self.park("Codex failed without a recognized transient error; inspect retained logs", error_tail=tail)
            failures = self.status["consecutive_failures"] + 1
            if failures >= self.args.max_transient_failures:
                return self.park("Transient retry limit reached; inspect service/network/usage state", consecutive_failures=failures, error_tail=tail)
            delay = retry_delay(failures, self.args.retry_base, self.args.retry_cap)
            self.checkpoint("backoff", consecutive_failures=failures, retry_delay_s=delay,
                            next_retry_at=time.time() + delay, error_tail=tail)
            self.wait(delay)
        self.checkpoint("stopped", child_pid=None, signal=self.stop_signal, reason="Supervisor received a stop signal")
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--initial-prompt", type=Path, required=True)
    parser.add_argument("--continuation-prompt", type=Path, required=True)
    parser.add_argument("--retry-base", type=float, default=10)
    parser.add_argument("--retry-cap", type=float, default=300)
    parser.add_argument("--max-transient-failures", type=int, default=12)
    parser.add_argument("--turn-delay", type=float, default=2)
    parser.add_argument("--signal-grace", type=float, default=20)
    parser.add_argument("--continue-after-input", action="store_true")
    args = parser.parse_args()
    if not args.codex.is_absolute():
        parser.error("--codex must be an absolute executable path")
    if (not all(math.isfinite(v) for v in (args.retry_base, args.retry_cap, args.turn_delay, args.signal_grace))
            or not 0 < args.retry_base <= args.retry_cap <= 300 or args.max_transient_failures < 1
            or args.turn_delay < 0 or args.signal_grace <= 0):
        parser.error("Retry bounds must satisfy 0 < base <= cap <= 300; counts/grace positive, turn delay nonnegative")
    for name in ("repo", "state_dir", "initial_prompt", "continuation_prompt"):
        setattr(args, name, getattr(args, name).resolve())
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another supervisor owns this state directory", file=sys.stderr)
            return 73
        supervisor = Supervisor(args)
        signal.signal(signal.SIGTERM, supervisor.on_signal)
        signal.signal(signal.SIGINT, supervisor.on_signal)
        try:
            return supervisor.run()
        except Exception as error:
            supervisor.stop_owned_child()
            print(f"Unexpected supervisor error: {error}", file=sys.stderr)
            try:
                return supervisor.park(f"Unexpected supervisor error: {error}")
            except OSError as checkpoint_error:
                print(f"Cannot publish durable error status: {checkpoint_error}", file=sys.stderr)
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
