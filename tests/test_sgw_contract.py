import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from experiments.workshops.spatial_grounding_v1.contract import ContractError, load_release


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def make_release(tmp_path: Path) -> Path:
    os.environ["JOB_UID"] = "test-job-uid"
    os.environ["POD_UID"] = "test-pod-uid"
    root = tmp_path / "release"
    root.mkdir()
    authorizations = {}
    for name in ("direct_command_fixed_input_gate", "P", "D", "C"):
        receipt = tmp_path / f"{name}.json"
        _write(receipt, {"status": "passed", "receipt": name})
        authorizations[name] = {"path": str(receipt), "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}
    source_queue_sha = "q" * 64
    binding = {
        "context": "ali", "namespace": "ali-ns", "resource_owner": "ali",
        "worker_image_digest": "registry.example/worker@sha256:" + "a" * 64,
        "pvc_name": "pvc", "pvc_mount_path": str(tmp_path), "pvc_access_mode": "RWX",
        "lock_test_receipt": "passed", "model_gpu_counts": {"N3": 1, "D1": 2},
        "cpu_memory_limits": {}, "node_gpu_type": "B200", "cluster_version": "v1",
        "source_commit": "a" * 40, "model_code_commits": {}, "simulator_commit": "b" * 40,
        "renderer_receipt": "passed", "persistent_write_receipt": "passed", "checkpoint_hashes": {},
        "user_resource_budget": "approved", "budget_source": "owner", "policy_ports": {},
        "cache_reset_receipt": "passed", "frame_time_mapping_hashes": {},
        "stage_authorizations": authorizations,
        "source_root": "/data/users/ali/sgw-01",
        "max_concurrent_model_workers": 2, "max_total_allocated_gpus": 4,
    }
    runtime_identity = hashlib.sha256(json.dumps({
        key: binding[key] for key in (
            "worker_image_digest", "source_commit", "simulator_commit", "model_code_commits",
            "checkpoint_hashes", "node_gpu_type",
        )
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    budget_receipt = tmp_path / "budget.json"
    _write(budget_receipt, {
        "status": "approved", "release_id": "r1", "source_queue_sha256": source_queue_sha,
        "source_queue_episode_count": 1044, "pvc_name": binding["pvc_name"],
        "pvc_mount_path": binding["pvc_mount_path"], "study_root": binding["source_root"],
        "runtime_identity_sha256": runtime_identity, "model": "all_models",
        "approved_gpu_hours": 20.0, "estimated_remaining_gpu_hours": 10.0,
        "expires_at_utc": "2030-01-01T00:00:00Z",
    })
    binding["resource_budget_receipt"] = {
        "path": str(budget_receipt), "sha256": hashlib.sha256(budget_receipt.read_bytes()).hexdigest(),
    }
    _write(root / "protocol.json", {"study_id": "SGW-01"})
    authorization_receipt = tmp_path / "operational_authorization.json"
    receipt_identity = {
        "release_id": "r1", "source_queue_sha256": source_queue_sha, "source_queue_episode_count": 1044,
        "pvc_name": binding["pvc_name"], "pvc_mount_path": binding["pvc_mount_path"],
        "study_root": binding["source_root"], "runtime_identity_sha256": runtime_identity,
    }
    _write(authorization_receipt, {
        "schema_version": "sgw-01-operational-authorization-v1", "status": "approved",
        "owner_approval_reference": {"kind": "synthetic-test"}, "budget_mode": "existing_idle_capacity_no_aggregate_hour_cap",
        "source_protocol_sha256": hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest(),
        "source_queue_sha256": source_queue_sha,
        "scope": {
            "context": binding["context"], "namespace": binding["namespace"], "pvc": binding["pvc_name"],
            "persistent_study_root": binding["source_root"], "models": ["N3"],
            "maximum_registered_behavioral_episodes": 1044, "maximum_attempts_per_behavioral_cell": 3,
        },
        "constraints": {
            "max_concurrent_model_workers": 2, "max_total_allocated_gpus": 4,
            "existing_authorized_cluster_capacity_only": True, "fresh_idle_allocation_check_required": True,
            "new_paid_capacity_allowed": False, "new_cluster_provisioning_allowed": False,
            "preempt_or_stop_unowned_workloads_allowed": False, "bounded_job_deadline_required": True,
        },
    })
    binding["operational_authorization_receipt"] = {
        "path": str(authorization_receipt), "sha256": hashlib.sha256(authorization_receipt.read_bytes()).hexdigest(),
    }
    idle_receipt = tmp_path / "idle.json"
    observed = datetime.now(timezone.utc)
    job_start = observed - timedelta(seconds=10)
    _write(idle_receipt, {
        "study_id": "SGW-01", "status": "passed_idle_snapshot_only", "observed_at_unix": observed.timestamp(),
        "expected_visible_gpu_count": 1, "model_requests": 0, "compute_occupied_uuids": [],
        "gpus": [{"index": 0, "uuid": "GPU-test", "memory_used_mib": 0, "memory_free_mib": 32768,
                  "utilization_percent": 0}],
        "selected_gpu": {"index": 0, "uuid": "GPU-test", "memory_used_mib": 0, "memory_free_mib": 32768,
                         "utilization_percent": 0},
    })
    allocation_receipt = tmp_path / "allocation.json"
    active_deadline_seconds = 7200
    _write(allocation_receipt, {
        "schema": "sgw-01-external-allocation-v1", "status": "approved",
        "owner_approval_reference": "owner-budget-approval", "reservation_id": "reservation-1",
        "context": binding["context"], "namespace": binding["namespace"],
        "job_name": "sgw-job", "job_uid": "test-job-uid", "pod_uid": "test-pod-uid",
        **receipt_identity,
        "model": "N3", "allocated_gpu_count": 1, "startTime": job_start.isoformat(),
        "activeDeadlineSeconds": active_deadline_seconds,
        "deadline_utc": (job_start + timedelta(seconds=active_deadline_seconds)).isoformat(),
        "budget_mode": "existing_idle_capacity_no_aggregate_hour_cap",
        "allocated_gpu_uuids": ["GPU-test"],
        "gpu_idle_probe_receipt": {"path": str(idle_receipt), "sha256": hashlib.sha256(idle_receipt.read_bytes()).hexdigest()},
        "reservation_gpu_hours": active_deadline_seconds / 3600,
    })
    binding["external_allocation_receipt"] = {
        "path": str(allocation_receipt), "sha256": hashlib.sha256(allocation_receipt.read_bytes()).hexdigest(),
    }
    prompts = {"prompts": []}
    _write(root / "protocol.json", {"study_id": "SGW-01"})
    _write(root / "prompts.json", prompts)
    _write(root / "fixtures.json", {"status": "qualified"})
    _write(root / "runtime_binding.json", binding)
    rows = []
    for index in range(6):
        prompt = f"prompt {index}"
        rows.append({
            "cell_id": f"cell-{index}", "block_id": "block-1", "model": "N3", "family": "LAT",
            "stage": "P", "layout_id": "LAT-P01", "prompt_id": f"p{index}", "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "within_block_order": index + 1,
            "fixture_sha256": "f", "runtime_sha256": "r", "time_map_sha256": "t",
            "release_id": "r1", "status": "RELEASED",
        })
    (root / "queue.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    _write(root / "release_receipt.json", {"release_id": "r1", "resource_owner": "ali", "source_queue_sha256": source_queue_sha,
                                            "stage_authorizations": binding["stage_authorizations"]})
    names = ("protocol.json", "prompts.json", "queue.jsonl", "fixtures.json", "runtime_binding.json", "release_receipt.json")
    _write(root / "hashes.json", {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names})
    return root


def test_release_rejects_prompt_or_hash_drift(tmp_path: Path) -> None:
    release = make_release(tmp_path)
    loaded = load_release(release)
    assert len(loaded.partition("N3", "LAT", "P")) == 6
    (release / "queue.jsonl").write_text("tampered\n")
    with pytest.raises(ContractError, match="hash mismatch"):
        load_release(release)


def test_release_rejects_mutable_image(tmp_path: Path) -> None:
    release = make_release(tmp_path)
    binding = json.loads((release / "runtime_binding.json").read_text())
    binding["worker_image_digest"] = "registry.example/worker:latest"
    _write(release / "runtime_binding.json", binding)
    names = ("protocol.json", "prompts.json", "queue.jsonl", "fixtures.json", "runtime_binding.json", "release_receipt.json")
    _write(release / "hashes.json", {name: hashlib.sha256((release / name).read_bytes()).hexdigest() for name in names})
    with pytest.raises(ContractError, match="immutable by digest"):
        load_release(release)
