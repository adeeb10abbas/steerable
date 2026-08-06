#!/usr/bin/env python3
"""Run one complete V3-B003 model-blind RTX lane, one reset per process."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


CONDITIONS = (
    "control:left",
    "control:right",
    "position_mirrored:left",
    "position_mirrored:right",
)
CANDIDATE_SHA256 = "e1799b815da41f9a08a4000a360c4958003269fed27e2abe75b273519e4d1c88"


def _append(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--admission-lock", type=Path, required=True)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--pod-uid", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--environment-seed", type=int, default=9400)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--ld-library-path", required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=False)
    args.admission_lock.parent.mkdir(parents=True, exist_ok=True)
    event_path = args.output_root / "launcher_events.jsonl"
    inputs: list[Path] = []
    for condition in CONDITIONS:
        for repeat in range(3):
            slug = condition.replace(":", "_") + f"_repeat{repeat}"
            attempt = args.output_root / slug
            output = attempt / "live_gate"
            report = output / "model_blind_preflight.json"
            attempt.mkdir(parents=True, exist_ok=False)
            caches = {
                "XDG_CACHE_HOME": attempt / "cache/xdg",
                "WARP_CACHE_PATH": attempt / "cache/warp",
                "MPLCONFIGDIR": attempt / "cache/matplotlib",
                "TMPDIR": Path("/tmp") / "v3b003-preflight" / args.pod_uid / slug,
            }
            for path in caches.values():
                path.mkdir(parents=True, exist_ok=False)
            command = [
                str(args.python), "-m", "experiments.v3.pi05_phase_b.model_blind_preflight",
                "--study-root", str(args.repo_root.resolve()),
                "--robolab-root", str(args.robolab_root.resolve()),
                "--candidate", str(args.candidate.resolve()),
                "--candidate-sha256", CANDIDATE_SHA256,
                "--output-dir", str(output.resolve()),
                "--amendment-id", "V3-B003",
                "--condition", condition,
                "--repeat-index", str(repeat),
                "--environment-seed", str(args.environment_seed),
                "--pod", args.pod,
                "--pod-uid", args.pod_uid,
                "--gpu-uuid", args.gpu_uuid,
                "--num-envs", "1", "--headless",
                "--renderer", "realtime", "--rendering-type", "balanced",
                "--device", "cuda:0",
                "--kit_args=--/rtx/verifyDriverVersion/enabled=false",
            ]
            environment = dict(os.environ)
            environment.update({key: str(value) for key, value in caches.items()})
            environment.update({
                "PYTHONPATH": f"{args.repo_root.resolve()}:{args.robolab_root.resolve()}",
                "LD_LIBRARY_PATH": args.ld_library_path,
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "NVIDIA_DRIVER_CAPABILITIES": "all",
                "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
            })
            with args.admission_lock.open("a+") as lock, (attempt / "launch.log").open("wb") as log:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                _append(event_path, {"event": "started", "condition": condition, "repeat": repeat})
                completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            if not report.is_file():
                _append(event_path, {
                    "event": "failed", "condition": condition, "repeat": repeat,
                    "process_returncode": completed.returncode,
                })
                raise RuntimeError(f"physical gate did not emit report: {slug}")
            value = json.loads(report.read_text(encoding="utf-8"))
            if (
                value.get("passed") is not True
                or value.get("condition_scope") != condition
                or value.get("repeat_scope") != repeat
                or value.get("model_request_count") != 0
                or value.get("behavioral_episode_count") != 0
            ):
                raise RuntimeError(f"physical gate report failed validation: {slug}")
            _append(event_path, {
                "event": "passed", "condition": condition, "repeat": repeat,
                "process_returncode": completed.returncode,
            })
            inputs.append(report)

    merged = args.output_root / "merged_model_blind_preflight.json"
    command = [
        str(args.python), str(args.repo_root / "tools/merge_dreamzero_v3b003_preflight.py"),
        *[item for path in inputs for item in ("--input", str(path.resolve()))],
        "--output", str(merged.resolve()),
    ]
    subprocess.run(command, check=True)
    print(json.dumps({"merged_report": str(merged.resolve()), "partial_reports": len(inputs)}, indent=2))


if __name__ == "__main__":
    main()
