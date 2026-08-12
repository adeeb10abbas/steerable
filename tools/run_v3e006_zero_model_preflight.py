#!/usr/bin/env python3
"""Failure-retaining outer harness for the V3-E006 zero-model health preflight."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def require(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"hash-bound input is missing or changed: {path}")
    return binding(path)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("--expected-study-commit", required=True)
    parser.add_argument("--expected-robolab-commit", required=True)
    parser.add_argument("--preflight-source", type=Path, required=True)
    parser.add_argument("--preflight-source-sha256", required=True)
    parser.add_argument("--e004-candidate", type=Path, required=True)
    parser.add_argument("--e004-candidate-sha256", required=True)
    parser.add_argument("--ood-freeze", type=Path, required=True)
    parser.add_argument("--ood-freeze-sha256", required=True)
    parser.add_argument("--e004-reset-reference", type=Path, required=True)
    parser.add_argument("--e004-reset-reference-sha256", required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--runtime-contract-sha256", required=True)
    parser.add_argument("--diagnostic-input", action="append", type=Path, default=[])
    parser.add_argument("--output-root", type=Path, required=True)
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
    output_root = args.output_root.resolve()
    if output_root.exists():
        parser.error(f"refusing to overwrite preflight attempt: {output_root}")
    if subprocess.check_output(["git", "-C", str(study_root), "rev-parse", "HEAD"], text=True).strip() != args.expected_study_commit:
        parser.error("study checkout differs from expected commit")
    if subprocess.check_output(["git", "-C", str(robolab_root), "rev-parse", "HEAD"], text=True).strip() != args.expected_robolab_commit:
        parser.error("RoboLab checkout differs from expected commit")
    if subprocess.check_output(["git", "-C", str(study_root), "status", "--porcelain"], text=True):
        parser.error("study checkout is not clean")

    inputs = {
        "preflight_source": require(args.preflight_source, args.preflight_source_sha256),
        "e004_candidate": require(args.e004_candidate, args.e004_candidate_sha256),
        "ood_freeze": require(args.ood_freeze, args.ood_freeze_sha256),
        "e004_reset_reference": require(args.e004_reset_reference, args.e004_reset_reference_sha256),
        "runtime_contract": require(args.runtime_contract, args.runtime_contract_sha256),
    }
    diagnostics = []
    for path in args.diagnostic_input:
        if not path.is_file():
            parser.error(f"diagnostic input is missing: {path}")
        diagnostics.append(binding(path))

    output_root.mkdir(parents=True)
    cache = {name: output_root / "cache" / name for name in ("xdg", "warp", "matplotlib", "tmp")}
    for path in cache.values():
        path.mkdir(parents=True)
    child_output = output_root / "preflight_result.json"
    child_argv = [
        str(args.python.resolve()),
        str(args.preflight_source.resolve()),
        "--study-root", str(study_root),
        "--robolab-root", str(robolab_root),
        "--expected-study-commit", args.expected_study_commit,
        "--expected-robolab-commit", args.expected_robolab_commit,
        "--e004-candidate", str(args.e004_candidate.resolve()),
        "--e004-candidate-sha256", args.e004_candidate_sha256,
        "--ood-freeze", str(args.ood_freeze.resolve()),
        "--ood-freeze-sha256", args.ood_freeze_sha256,
        "--e004-reset-reference", str(args.e004_reset_reference.resolve()),
        "--e004-reset-reference-sha256", args.e004_reset_reference_sha256,
        "--runtime-contract", str(args.runtime_contract.resolve()),
        "--runtime-contract-sha256", args.runtime_contract_sha256,
        *[item for path in args.diagnostic_input for item in ("--diagnostic-input", str(path.resolve()))],
        "--output", str(child_output),
        "--pod", args.pod,
        "--pod-uid", args.pod_uid,
        "--gpu-uuid", args.gpu_uuid,
        "--container-image", args.container_image,
        "--container-id", args.container_id,
        "--driver-version", args.driver_version,
        "--num-envs", "1",
        "--headless",
        "--renderer", "realtime",
        "--rendering-type", "balanced",
        "--device", "cuda:0",
        "--kit_args=--/rtx/verifyDriverVersion/enabled=false",
    ]
    native = (
        "/data/users/ali/vla_wam/envs/robolab-native-libs-ubuntu2204/usr/lib/x86_64-linux-gnu:"
        "/data/users/ali/glvnd/lib:"
        "/data/users/ali/vla_wam/envs/fastwam-native-libs/lib:"
        "/usr/lib/x86_64-linux-gnu"
    )
    environment = dict(os.environ)
    for name in ("DISPLAY", "LD_PRELOAD", "CUDA_VISIBLE_DEVICES"):
        environment.pop(name, None)
    environment.update(
        {
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
            "LD_LIBRARY_PATH": native,
            "XDG_CACHE_HOME": str(cache["xdg"]),
            "WARP_CACHE_PATH": str(cache["warp"]),
            "MPLCONFIGDIR": str(cache["matplotlib"]),
            "TMPDIR": str(cache["tmp"]),
            "PYTHONPATH": f"{study_root}:{robolab_root}",
        }
    )
    retained_environment = {
        key: environment.get(key)
        for key in (
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
            "OMNI_KIT_ACCEPT_EULA",
            "PYTHONNOUSERSITE",
            "PYTHONUNBUFFERED",
            "VK_ICD_FILENAMES",
            "LD_LIBRARY_PATH",
            "XDG_CACHE_HOME",
            "WARP_CACHE_PATH",
            "MPLCONFIGDIR",
            "TMPDIR",
            "PYTHONPATH",
        )
    }
    launch = {
        "schema_version": "vla-wam-shared-v3e006-zero-model-preflight-launch-v1",
        "created_at_utc": utcnow(),
        "study_commit": args.expected_study_commit,
        "robolab_commit": args.expected_robolab_commit,
        "model_request_count_before_launch": 0,
        "behavioral_episode_count_before_launch": 0,
        "state_candidate_count_before_launch": 0,
        "scope": "generic zero-model runtime health only",
        "input_bindings": inputs,
        "diagnostic_inputs": diagnostics,
        "child_argv": child_argv,
        "environment": retained_environment,
        "lane": {
            "pod": args.pod,
            "pod_uid": args.pod_uid,
            "gpu_uuid": args.gpu_uuid,
            "container_image": args.container_image,
            "container_id": args.container_id,
            "driver_version": args.driver_version,
        },
    }
    launch_path = output_root / "preflight_launch.json"
    launch_path.write_text(json.dumps(launch, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log_path = output_root / "runtime.log"
    launch_error = None
    try:
        with log_path.open("wb") as log:
            completed = subprocess.run(child_argv, cwd=study_root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        returncode = completed.returncode
    except BaseException as exc:
        launch_error = f"{type(exc).__name__}: {exc}"
        with log_path.open("ab") as log:
            log.write((launch_error + "\n").encode("utf-8", errors="replace"))
        returncode = 127
    child_binding = binding(child_output) if child_output.is_file() else None
    passed = returncode == 0 and child_binding is not None
    if passed:
        child = json.loads(child_output.read_text(encoding="utf-8"))
        passed = (
            child.get("status") == "passed_generic_zero_model_cuda_vulkan_isaac_physics_render_health_preflight"
            and child.get("model_request_count") == 0
            and child.get("behavioral_episode_count") == 0
            and child.get("state_candidate_count") == 0
        )
    result = {
        "schema_version": "vla-wam-shared-v3e006-zero-model-preflight-harness-result-v1",
        "created_at_utc": utcnow(),
        "status": "passed_generic_zero_model_health_preflight" if passed else "infrastructure_invalid_zero_model_health_preflight",
        "passed": passed,
        "behavioral_denominator_included": False,
        "candidate_denominator_included": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "state_candidate_count": 0,
        "process_exit_code": returncode,
        "launch_error": launch_error,
        "launch": binding(launch_path),
        "runtime_log": binding(log_path),
        "child_report": child_binding,
        "failure_log_tail": None if passed else log_path.read_text(encoding="utf-8", errors="replace")[-8192:],
    }
    result_path = output_root / "harness_result.json"
    result_path.write_text(json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "result": str(result_path), "sha256": sha256(result_path)}, sort_keys=True))
    if not passed:
        raise SystemExit(returncode or 1)


if __name__ == "__main__":
    main()
