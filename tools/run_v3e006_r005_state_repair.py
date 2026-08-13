#!/usr/bin/env python3
"""Failure-retaining outer launcher for the zero-request R005 state repair."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


E004_APP_LAUNCHER_ARGV = (
    "--headless",
    "--device", "cuda:0",
    "--num-envs", "1",
    "--num-runs", "1",
    "--renderer", "realtime",
    "--rendering-type", "balanced",
    "--video-mode", "viewport",
    "--instruction-type", "default",
    "--disable-subtask",
    "--kit_args=--/rtx/verifyDriverVersion/enabled=false",
)
COMPLETED_CHILD_STATUSES = {
    "passed_r005_state_repair_not_released_for_behavior": True,
    "r005_candidate_budget_exhausted_no_valid_state_pair": False,
    "r005_known_reachable_diagnostic_failed_candidates_not_evaluated": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = Path(path).absolute()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def require(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"hash-bound input is missing or changed: {path}")
    return binding(path)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def child_process_completed(returncode: int, report_path: Path | None, payload: Any) -> bool:
    """Accept only an exact normal R005 terminal report, never a failure report."""

    if returncode != 0 or report_path is None or report_path.name != "state_repair_result.json":
        return False
    if not isinstance(payload, dict):
        return False
    expected_passed = COMPLETED_CHILD_STATUSES.get(str(payload.get("status")))
    if expected_passed is None or payload.get("passed") is not expected_passed:
        return False
    if payload.get("model_request_count") != 0 or payload.get("behavioral_episode_count") != 0:
        return False
    diagnostic_count = payload.get("r005_live_diagnostic_count")
    if not isinstance(diagnostic_count, int) or isinstance(diagnostic_count, bool):
        return False
    count = payload.get("repair_candidate_evaluation_count")
    if payload.get("status") == "r005_known_reachable_diagnostic_failed_candidates_not_evaluated":
        return 1 <= diagnostic_count <= 4 and count == 0
    return (
        diagnostic_count == 4
        and isinstance(count, int)
        and not isinstance(count, bool)
        and 1 <= count <= 4
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("--expected-study-commit", required=True)
    parser.add_argument("--expected-robolab-commit", required=True)
    for name in (
        "repair-source", "repair-registration", "candidate-schedule", "original-closure-binding",
        "predecessor-closure-binding",
        "source-push-gate", "e004-candidate", "ood-freeze", "e004-reset-reference", "runtime-contract",
        "control-scene-asset", "paired-scene-asset",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--health-preflight-root", type=Path, required=True)
    parser.add_argument("--health-harness-sha256", required=True)
    parser.add_argument("--health-launch-sha256", required=True)
    parser.add_argument("--health-child-sha256", required=True)
    parser.add_argument("--health-runtime-log-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-log", type=Path, required=True)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--pod-uid", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--driver-version", required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()

    study_root = args.study_root.resolve()
    robolab_root = args.robolab_root.resolve()
    output_root = args.output_root.absolute()
    if output_root.exists():
        parser.error(f"refusing to overwrite R005 attempt: {output_root}")
    if subprocess.check_output(["git", "-C", str(study_root), "rev-parse", "HEAD"], text=True).strip() != args.expected_study_commit:
        parser.error("study checkout differs from expected commit")
    if subprocess.check_output(["git", "-C", str(robolab_root), "rev-parse", "HEAD"], text=True).strip() != args.expected_robolab_commit:
        parser.error("RoboLab checkout differs from expected commit")
    if subprocess.check_output(["git", "-C", str(study_root), "status", "--porcelain"], text=True):
        parser.error("study checkout is not clean")
    remote_head = subprocess.check_output(
        ["git", "-C", str(study_root), "ls-remote", "origin", "refs/heads/experiment/v3e006-r005-state-repair"],
        text=True,
    ).strip().split()
    if len(remote_head) != 2 or remote_head[0] != args.expected_study_commit:
        parser.error("remote repair branch does not equal the immutable runtime checkout")
    if not args.python.is_absolute() or not args.python.is_file() or not os.access(args.python, os.X_OK):
        parser.error("Python interpreter must be an existing executable absolute venv path")

    bound: dict[str, Any] = {}
    for name in (
        "repair_source", "repair_registration", "candidate_schedule", "original_closure_binding",
        "predecessor_closure_binding",
        "source_push_gate", "e004_candidate", "ood_freeze", "e004_reset_reference", "runtime_contract",
        "control_scene_asset", "paired_scene_asset",
    ):
        bound[name] = require(getattr(args, name), getattr(args, f"{name}_sha256"))
    health = {
        "harness_result": require(args.health_preflight_root / "harness_result.json", args.health_harness_sha256),
        "preflight_launch": require(args.health_preflight_root / "preflight_launch.json", args.health_launch_sha256),
        "preflight_result": require(args.health_preflight_root / "preflight_result.json", args.health_child_sha256),
        "runtime_log": require(args.health_preflight_root / "runtime.log", args.health_runtime_log_sha256),
    }
    health_harness = json.loads((args.health_preflight_root / "harness_result.json").read_text(encoding="utf-8"))
    if health_harness.get("status") != "passed_generic_zero_model_health_preflight" or health_harness.get("passed") is not True:
        parser.error("bound formal health preflight did not pass")
    source_gate = json.loads(args.source_push_gate.read_text(encoding="utf-8"))
    if source_gate.get("status") != "passed_before_first_r005_live_diagnostic_candidate_or_model_request":
        parser.error("R005 source-push gate did not pass prospectively")
    if (
        source_gate.get("model_request_count") != 0
        or source_gate.get("behavioral_episode_count") != 0
        or source_gate.get("r005_live_diagnostic_count") != 0
        or source_gate.get("r005_live_candidate_evaluation_count") != 0
        or source_gate.get("completed_candidate_pair_count") != 0
        or source_gate.get("accepted_state_candidate_count") != 0
        or source_gate.get("infrastructure_invalid_search_attempt_count") != 0
    ):
        parser.error("lifecycle-repair source-push history counts differ")
    implementation_commit = str(source_gate.get("implementation_commit", ""))
    if not implementation_commit or subprocess.run(
        ["git", "-C", str(study_root), "merge-base", "--is-ancestor", implementation_commit, args.expected_study_commit],
        check=False,
    ).returncode:
        parser.error("source-push implementation commit is not an ancestor of runtime checkout")
    implementation_files = source_gate.get("implementation_files")
    if not isinstance(implementation_files, list) or not implementation_files:
        parser.error("source-push gate has no implementation inventory")
    for row in implementation_files:
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            parser.error(f"unsafe source-push path: {relative}")
        actual = study_root / relative
        if not actual.is_file() or actual.stat().st_size != row.get("bytes") or sha256(actual) != row.get("sha256"):
            parser.error(f"source-push implementation file changed: {relative}")

    output_root.mkdir(parents=True)
    child_output = output_root / "raw"
    child_log = output_root / "runtime.log"
    cache = {name: output_root / "cache" / name for name in ("home", "xdg", "warp", "matplotlib", "tmp")}
    for path in cache.values():
        path.mkdir(parents=True)
    child_argv = [
        str(args.python), "-m", "experiments.v3.phase_e.canonical_stage_localization_v3e006_r005.state_repair_gate",
        "--study-root", str(study_root), "--robolab-root", str(robolab_root),
        "--expected-study-commit", args.expected_study_commit,
        "--expected-robolab-commit", args.expected_robolab_commit,
        "--e004-candidate", str(args.e004_candidate.absolute()), "--e004-candidate-sha256", args.e004_candidate_sha256,
        "--ood-freeze", str(args.ood_freeze.absolute()), "--ood-freeze-sha256", args.ood_freeze_sha256,
        "--e004-reset-reference", str(args.e004_reset_reference.absolute()), "--e004-reset-reference-sha256", args.e004_reset_reference_sha256,
        "--runtime-bindings", str(args.runtime_contract.absolute()), "--runtime-bindings-sha256", args.runtime_contract_sha256,
        "--repair-registration", str(args.repair_registration.absolute()), "--repair-registration-sha256", args.repair_registration_sha256,
        "--candidate-schedule", str(args.candidate_schedule.absolute()), "--candidate-schedule-sha256", args.candidate_schedule_sha256,
        "--original-closure-binding", str(args.original_closure_binding.absolute()), "--original-closure-binding-sha256", args.original_closure_binding_sha256,
        "--predecessor-closure-binding", str(args.predecessor_closure_binding.absolute()), "--predecessor-closure-binding-sha256", args.predecessor_closure_binding_sha256,
        "--source-push-gate", str(args.source_push_gate.absolute()), "--source-push-gate-sha256", args.source_push_gate_sha256,
        "--control-scene-asset", str(args.control_scene_asset.absolute()),
        "--paired-scene-asset", str(args.paired_scene_asset.absolute()),
        "--output-dir", str(child_output), "--runtime-log", str(child_log),
        "--health-preflight-root", str(args.health_preflight_root.absolute()),
        "--health-harness-sha256", args.health_harness_sha256,
        "--health-launch-sha256", args.health_launch_sha256,
        "--health-child-sha256", args.health_child_sha256,
        "--health-runtime-log-sha256", args.health_runtime_log_sha256,
        "--pod", args.pod, "--pod-uid", args.pod_uid, "--gpu-uuid", args.gpu_uuid,
        "--container-image", args.container_image, "--container-id", args.container_id,
        "--driver-version", args.driver_version,
        *E004_APP_LAUNCHER_ARGV,
    ]
    native = (
        "/data/users/ali/vla_wam/envs/robolab-native-libs-ubuntu2204/usr/lib/x86_64-linux-gnu:"
        "/data/users/ali/glvnd/lib:/data/users/ali/vla_wam/envs/fastwam-native-libs/lib:"
        "/usr/lib/x86_64-linux-gnu"
    )
    environment = dict(os.environ)
    for name in ("DISPLAY", "LD_PRELOAD", "CUDA_VISIBLE_DEVICES"):
        environment.pop(name, None)
    environment.update(
        {
            "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
            "LD_LIBRARY_PATH": native, "HOME": str(cache["home"]), "XDG_CACHE_HOME": str(cache["xdg"]),
            "WARP_CACHE_PATH": str(cache["warp"]), "MPLCONFIGDIR": str(cache["matplotlib"]),
            "TMPDIR": str(cache["tmp"]), "PYTHONPATH": f"{study_root}:{robolab_root}",
        }
    )
    retained_environment = {
        key: environment.get(key)
        for key in (
            "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "OMNI_KIT_ACCEPT_EULA",
            "PYTHONNOUSERSITE", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE", "VK_ICD_FILENAMES",
            "LD_LIBRARY_PATH", "HOME", "XDG_CACHE_HOME", "WARP_CACHE_PATH", "MPLCONFIGDIR", "TMPDIR", "PYTHONPATH",
        )
    }
    launch = {
        "schema_version": "vla-wam-shared-v3e006-r005-state-repair-launch-v1",
        "created_at_utc": utcnow(), "status": "launched_after_prospective_source_push_gate",
        "model_request_count_before_launch": 0, "behavioral_episode_count_before_launch": 0,
        "r005_live_diagnostic_count_before_launch": 0,
        "r005_live_candidate_evaluation_count_before_launch": 0,
        "completed_candidate_pair_count_before_launch": 0,
        "accepted_state_candidate_count_before_launch": 0,
        "infrastructure_invalid_search_attempt_count_before_launch": 0,
        "study_commit": args.expected_study_commit, "robolab_commit": args.expected_robolab_commit,
        "remote_branch_equality": {"remote": "origin", "ref": "refs/heads/experiment/v3e006-r005-state-repair", "commit": remote_head[0]},
        "harness_source": binding(Path(__file__)), "outer_argv": sys.argv,
        "child_argv": child_argv, "input_bindings": bound, "formal_health_preflight": health,
        "python_interpreter": {"path": str(args.python), "resolved_path": str(args.python.resolve())},
        "environment": retained_environment,
        "lane": {"pod": args.pod, "pod_uid": args.pod_uid, "gpu_uuid": args.gpu_uuid,
                 "container_image": args.container_image, "container_id": args.container_id,
                 "driver_version": args.driver_version},
    }
    launch_path = output_root / "launch.json"
    launch_path.write_text(json.dumps(launch, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    launch_error = None
    try:
        with child_log.open("wb") as handle:
            completed = subprocess.run(child_argv, cwd=study_root, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        returncode = completed.returncode
    except BaseException as exc:
        launch_error = f"{type(exc).__name__}: {exc}"
        child_log.write_text(launch_error + "\n", encoding="utf-8")
        returncode = 127
    result_path = child_output / "state_repair_result.json"
    failure_path = child_output / "state_construction_failure.json"
    child_report = result_path if result_path.is_file() else failure_path if failure_path.is_file() else None
    child_payload = json.loads(child_report.read_text(encoding="utf-8")) if child_report else None
    passed_process = child_process_completed(returncode, child_report, child_payload)
    result = {
        "schema_version": "vla-wam-shared-v3e006-r005-state-repair-harness-result-v1",
        "created_at_utc": utcnow(),
        "status": "completed_r005_candidate_search" if passed_process else "infrastructure_invalid_r005_state_repair",
        "process_completed": passed_process, "scientific_gate_passed": bool(child_payload and child_payload.get("passed") is True),
        "behavioral_denominator_included": False, "model_request_count": 0, "behavioral_episode_count": 0,
        "repair_candidate_evaluation_count": int(child_payload.get("repair_candidate_evaluation_count", 0)) if child_payload else 0,
        "r005_live_diagnostic_count": int(child_payload.get("r005_live_diagnostic_count", 0)) if child_payload else 0,
        "process_exit_code": returncode, "launch_error": launch_error, "harness_source": launch["harness_source"],
        "launch": binding(launch_path), "runtime_log": binding(child_log),
        "child_report": binding(child_report) if child_report else None,
        "child_status": child_payload.get("status") if child_payload else None,
        "failure_log_tail": None if passed_process else child_log.read_text(encoding="utf-8", errors="replace")[-8192:],
    }
    harness_path = output_root / "harness_result.json"
    harness_path.write_text(json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(harness_path), "sha256": sha256(harness_path), **result}, sort_keys=True))
    if not passed_process:
        raise SystemExit(returncode or 1)


if __name__ == "__main__":
    main()
