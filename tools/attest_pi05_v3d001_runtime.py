#!/usr/bin/env python3
"""Attest the exact current-stack pi0.5 server before V3-D001 model load."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from experiments.pi05_current_stack.v2a010_serve_policy import verify_checkpoint
from experiments.v3.pi05_stochastic_probe import (
    MODEL_ID,
    PHASE_D_REGISTRY_SHA256,
    REGISTRATION_ID,
    REGISTRATION_SHA256,
    RUNTIME_SCHEMA,
    SCOPE_CORRECTION_SHA256,
    STUDY_ID,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--pod-uid", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study = args.study_root.resolve()
    openpi = args.openpi_root.resolve()
    if command("git", "rev-parse", "HEAD", cwd=openpi) != "c23745b5ad24e98f66967ea795a07b2588ed6c79":
        raise ValueError("OpenPI commit mismatch")
    if command("git", "status", "--porcelain=v1", "--untracked-files=no", cwd=openpi):
        raise ValueError("OpenPI tracked checkout is dirty")
    registration = study / "artifacts/vla_wam_shared_v3/prospective_tier_b/pi05_stochastic_eligibility_v3d001.json"
    phase_d = study / "artifacts/vla_wam_shared_v3/stochastic_rollout_registry.json"
    correction = study / "artifacts/vla_wam_shared_v3/prospective_tier_b/pi05_stochastic_v3d001_eight_repeat_correction.json"
    server_source = study / "experiments/pi05_current_stack/v2a010_serve_policy.py"
    if (
        sha256_file(registration) != REGISTRATION_SHA256
        or sha256_file(phase_d) != PHASE_D_REGISTRY_SHA256
        or sha256_file(correction) != SCOPE_CORRECTION_SHA256
    ):
        raise ValueError("prospective V3-D001 binding changed")
    if sha256_file(server_source) != "cd415e3a98da977f395242c24bb8f3d3187eb4cc3bf53c5dc659d190e6934051":
        raise ValueError("pi0.5 seeded server source changed")
    if sha256_file(args.checkpoint_manifest) != "f5a56d9565f9381ccdeeaa165b0495dab6d17a81836cc7b01c5fbc6ab89e74ca":
        raise ValueError("checkpoint manifest hash changed")
    verify_checkpoint(args.checkpoint.resolve(), args.checkpoint_manifest.resolve())
    gpu_query = command(
        "nvidia-smi", f"--id={args.gpu_index}",
        "--query-gpu=uuid,name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    )
    gpu_uuid, gpu_model, driver, memory_total, memory_free = [part.strip() for part in gpu_query.split(",")]
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("GPU UUID discovery failed")
    environment_inventory = []
    for name in ("uv.lock", "pyproject.toml"):
        path = openpi / name
        if not path.is_file():
            raise ValueError(f"OpenPI environment lock source is missing: {path}")
        environment_inventory.append({
            "path": name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    payload = {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "registration_id": REGISTRATION_ID,
        "model_id": MODEL_ID,
        "openpi_commit": "c23745b5ad24e98f66967ea795a07b2588ed6c79",
        "openpi_config": "pi05_droid_jointpos_polaris",
        "openpi_tracked_diff_empty": True,
        "server_source_sha256": sha256_file(server_source),
        "checkpoint_manifest_sha256": sha256_file(args.checkpoint_manifest),
        "checkpoint_hash_gate_passed": True,
        "registration_sha256": REGISTRATION_SHA256,
        "phase_d_registry_sha256": PHASE_D_REGISTRY_SHA256,
        "scope_correction_sha256": SCOPE_CORRECTION_SHA256,
        "pod": args.pod,
        "pod_uid": args.pod_uid,
        "owner": "ali",
        "gpu_index": args.gpu_index,
        "gpu_uuid": gpu_uuid,
        "gpu_model": gpu_model,
        "driver_version": driver,
        "gpu_memory_total_mib": int(memory_total),
        "gpu_memory_free_before_model_load_mib": int(memory_free),
        "python": platform.python_version(),
        "environment_lock_sources": environment_inventory,
        "environment_lock_sha256": sha256_bytes(canonical_json_bytes(environment_inventory)),
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }
    payload["runtime_attestation_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(args.output.resolve()), "sha256": sha256_file(args.output), "runtime_attestation_sha256": payload["runtime_attestation_sha256"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
