import hashlib
import json
from pathlib import Path

import pytest

from experiments.workshops.spatial_grounding_v1.contract import ContractError, load_release


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def make_release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    root.mkdir()
    authorizations = {}
    for name in ("direct_command_fixed_input_gate", "P", "D", "C"):
        receipt = tmp_path / f"{name}.json"
        _write(receipt, {"status": "passed", "receipt": name})
        authorizations[name] = {"path": str(receipt), "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}
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
    _write(root / "release_receipt.json", {"release_id": "r1", "resource_owner": "ali",
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
