from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "native_process_group_thermal_guard.py"
sys.path.insert(0, str(ROOT / "tools"))
from compile_vla_wam_v2_robotwin_confirmation import (  # noqa: E402
    load_interventions,
    load_invalid_attempts,
)


def _worker(seconds: str = "1") -> subprocess.Popen[str]:
    return subprocess.Popen(["sleep", seconds], start_new_session=True, text=True)


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _process_state(pid: int) -> str:
    proc_status = Path(f"/proc/{pid}/status")
    if proc_status.is_file():
        return next(
            line for line in proc_status.read_text().splitlines() if line.startswith("State:")
        )
    result = subprocess.run(
        ["ps", "-o", "state=", "-p", str(pid)],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _cell_args(tmp_path: Path) -> list[str]:
    return [
        "--ledger-output",
        str(tmp_path / "runtime_interventions_lingbot_va_robotwin.json"),
        "--invalid-attempts-output",
        str(tmp_path / "invalid_attempts_lingbot_va_robotwin.json"),
        "--model-id",
        "lingbot_va_robotwin",
        "--pair-id",
        "robotwin_pair_03",
        "--environment-seed",
        "4300003",
        "--sampling-seed",
        "8403",
        "--requested-relation",
        "left",
        "--requested-relation",
        "right",
    ]


def test_launch_mode_isolates_group_and_writes_compiler_ledger(tmp_path: Path) -> None:
    unrelated_marker = tmp_path / "unrelated-finished"
    unrelated = subprocess.Popen(
        ["sh", "-c", f"sleep 0.08; printf done > {unrelated_marker}; sleep 5"],
        start_new_session=True,
        text=True,
    )
    temperature_script = tmp_path / "temperatures.py"
    temperature_script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "path = Path(sys.argv[1])\n"
        "count = int(path.read_text()) + 1 if path.exists() else 1\n"
        "path.write_text(str(count))\n"
        "print(88 if count == 1 else 79 if count == 2 else 75)\n"
    )
    temperature_command = f"{sys.executable} {temperature_script} {tmp_path / 'count'}"
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--launch",
                "--gpu-index",
                "0",
                "--output",
                str(tmp_path / "thermal.jsonl"),
                "--poll-seconds",
                "0.01",
                "--temperature-command",
                temperature_command,
                *_cell_args(tmp_path),
                "--",
                sys.executable,
                "-c",
                "import time; time.sleep(0.25)",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        events = _events(tmp_path / "thermal.jsonl")
        worker_started = next(event for event in events if event["event"] == "worker_started")
        assert worker_started["target_pid"] == worker_started["pgid"]
        assert any(event["event"] == "cooldown_started" and event["action"] == "sigstop" for event in events)
        assert any(event["event"] == "cooldown_completed" and event["action"] == "sigcont" for event in events)
        assert events[-1]["event"] == "monitor_completed"

        ledger_path = tmp_path / "runtime_interventions_lingbot_va_robotwin.json"
        ledger = json.loads(ledger_path.read_text())
        assert len(ledger["events"]) == 2
        cell = ledger["events"][0]
        assert cell["model_id"] == "lingbot_va_robotwin"
        assert cell["pair_id"] == "robotwin_pair_03"
        assert cell["environment_seed"] == 4300003
        assert cell["sampling_seed"] == 8403
        assert cell["requested_relation"] == "left"
        assert cell["behavioral_result_valid"] is True
        assert cell["wall_latency_valid"] is False
        assert cell["classification"] == "thermal_intervention"
        compiler_view, ledger_sources = load_interventions(ledger_path, "lingbot_va_robotwin")
        assert compiler_view[(4300003, "left")][0]["id"] == cell["id"]
        assert (4300003, "right") in compiler_view
        assert ledger_sources[0]["selected_model_event_count"] == 2

        unrelated.wait(timeout=0.01) if unrelated.poll() is not None else None
        assert unrelated.poll() is None
        assert unrelated_marker.read_text() == "done"
        state = _process_state(unrelated.pid)
        assert "T (stopped)" not in state
        assert not state.startswith("T")
    finally:
        if unrelated.poll() is None:
            os.killpg(os.getpgid(unrelated.pid), signal.SIGTERM)
        unrelated.wait(timeout=2)


def test_attach_requires_acknowledgement_and_rejects_caller_group(tmp_path: Path) -> None:
    base = [
        sys.executable,
        str(TOOL),
        "--pgid",
        str(os.getpgrp()),
        "--gpu-index",
        "0",
        "--output",
        str(tmp_path / "thermal.jsonl"),
        "--temperature-command",
        "printf 75",
        "--dry-run",
    ]
    missing_ack = subprocess.run(base, text=True, capture_output=True, check=False)
    assert missing_ack.returncode != 0
    assert "requires --acknowledge-attach-risk" in missing_ack.stderr
    own_group = subprocess.run(
        [*base, "--acknowledge-attach-risk"], text=True, capture_output=True, check=False
    )
    assert own_group.returncode != 0
    assert "own process group" in own_group.stderr


def test_guard_rejects_cross_model_shared_ledger_filenames(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--launch",
            "--gpu-index",
            "0",
            "--output",
            str(tmp_path / "thermal.jsonl"),
            "--ledger-output",
            str(tmp_path / "runtime_interventions.json"),
            "--invalid-attempts-output",
            str(tmp_path / "invalid_attempts.json"),
            "--model-id",
            "fastwam_robotwin",
            "--pair-id",
            "robotwin_pair_03",
            "--environment-seed",
            "4300003",
            "--sampling-seed",
            "8403",
            "--requested-relation",
            "left",
            "--requested-relation",
            "right",
            "--",
            "true",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Model-specific ledger filename" in result.stderr


def test_dry_run_emergency_holds_without_signalling_worker(tmp_path: Path) -> None:
    worker = _worker("5")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--pid",
                str(worker.pid),
                "--acknowledge-attach-risk",
                "--gpu-index",
                "0",
                "--output",
                str(tmp_path / "thermal.jsonl"),
                "--temperature-command",
                "printf 90",
                "--dry-run",
                *_cell_args(tmp_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 90
        emergency = _events(tmp_path / "thermal.jsonl")[-1]
        assert emergency["event"] == "emergency_hold"
        assert emergency["action"] == "would_sigstop"
        assert worker.poll() is None
        invalid, _, ledger_sources = load_invalid_attempts(
            tmp_path / "invalid_attempts_lingbot_va_robotwin.json", "lingbot_va_robotwin"
        )
        assert len(invalid) == 2
        assert ledger_sources[0]["selected_model_event_count"] == 2
        assert invalid[0]["behavioral_result_valid"] is False
        assert invalid[0]["wall_latency_valid"] is False
    finally:
        os.killpg(os.getpgid(worker.pid), signal.SIGTERM)
        worker.wait(timeout=2)


def test_temperature_query_failure_is_distinct(tmp_path: Path) -> None:
    worker = _worker("5")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--pid",
                str(worker.pid),
                "--acknowledge-attach-risk",
                "--gpu-index",
                "0",
                "--output",
                str(tmp_path / "thermal.jsonl"),
                "--temperature-command",
                "sh -c 'exit 7'",
                "--dry-run",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 2
        failure = _events(tmp_path / "thermal.jsonl")[-1]
        assert failure["event"] == "temperature_query_failed"
        assert failure["action"] == "would_sigstop"
        assert worker.poll() is None
    finally:
        os.killpg(os.getpgid(worker.pid), signal.SIGTERM)
        worker.wait(timeout=2)
