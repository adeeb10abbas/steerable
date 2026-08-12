#!/usr/bin/env python3
"""Collect one lane's excluded same-process L-R-L exact-repeat probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
from openpi_client import websocket_client_policy

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (
    ContractError,
    file_binding,
    registered_prompts,
    sha256_file,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002.runtime import validate_runtime_manifest
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.request0_replay import (
    CACHE_MANIFEST_SCHEMA,
    canonical_json_sha256,
    observation_payload_sha256,
)
from .contract import SMOKE_GATE_SCHEMA, read_finite_json


PROBE_SEED = 13_000_000
SEQUENCE = ("canonical_left", "canonical_right", "canonical_left")
REQUIRED_PACKED_KEYS = {
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
    "observation/joint_position",
    "observation/gripper_position",
    "prompt",
}
FROZEN_ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
FROZEN_PI05_CLIENT_SHA256 = "2386e6230ca4e2bbf163159ff0692780f5027a6b158b510b206700d54a4a29a3"


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _native_leaf(array: np.ndarray, kind: str) -> object:
    if kind == "torch_tensor":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - production runtime owns torch
            raise ContractError("torch is unavailable for native fixture reconstruction") from exc
        return torch.as_tensor(array.copy())
    if kind == "numpy_array":
        return array.copy()
    if kind == "python_scalar":
        return array.item()
    raise ContractError(f"unknown fixture native leaf kind: {kind}")


def reconstruct_native_fixture(cache_path: Path, manifest_path: Path) -> tuple[object, dict]:
    """Rebuild and byte-validate the exact frozen E004 request-zero tree."""

    manifest = read_finite_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != CACHE_MANIFEST_SCHEMA:
        raise ContractError("repeat fixture manifest schema changed")
    cache_binding = manifest.get("observation_cache", {})
    if (
        cache_binding.get("sha256") != sha256_file(cache_path)
        or cache_binding.get("bytes") != cache_path.stat().st_size
    ):
        raise ContractError("repeat cache differs from its frozen E004 manifest")
    structure = manifest.get("structure")
    if manifest.get("observation_structure_sha256") != canonical_json_sha256(structure):
        raise ContractError("repeat fixture structure digest changed")
    rows = manifest.get("leaves")
    if not isinstance(rows, list) or not rows:
        raise ContractError("repeat fixture manifest has no leaves")
    values: dict[str, object] = {}
    with np.load(cache_path, allow_pickle=False) as archive:
        expected = {row.get("storage_key") for row in rows if isinstance(row, dict)}
        if set(archive.files) != expected or None in expected:
            raise ContractError("repeat fixture archive inventory changed")
        for row in rows:
            storage_key = row["storage_key"]
            array = np.ascontiguousarray(archive[storage_key])
            raw = array.tobytes(order="C")
            if (
                array.dtype.str != row.get("dtype")
                or list(array.shape) != row.get("shape")
                or len(raw) != row.get("byte_length")
                or hashlib.sha256(raw).hexdigest() != row.get("data_sha256")
            ):
                raise ContractError(f"repeat fixture leaf changed: {storage_key}")
            values[storage_key] = _native_leaf(array, row.get("native_kind"))

    used: set[str] = set()

    def rebuild(node: object) -> object:
        if not isinstance(node, dict):
            raise ContractError("repeat fixture structure node is invalid")
        if set(node) == {"leaf"}:
            key = node["leaf"]
            if key not in values or key in used:
                raise ContractError("repeat fixture leaf reference changed")
            used.add(key)
            return values[key]
        container = node.get("container")
        children = node.get("children")
        if container == "mapping" and isinstance(children, dict):
            return {key: rebuild(value) for key, value in children.items()}
        if container in {"tuple", "list"} and isinstance(children, list):
            rebuilt = [rebuild(value) for value in children]
            return tuple(rebuilt) if container == "tuple" else rebuilt
        raise ContractError("repeat fixture container structure changed")

    observation = rebuild(structure)
    if used != set(values):
        raise ContractError("repeat fixture contains unreferenced leaves")
    if observation_payload_sha256(observation) != manifest.get("observation_payload_sha256"):
        raise ContractError("reconstructed repeat observation payload changed")
    return observation, manifest


def exact_pi05_request(
    observation: object,
    prompt: str,
    *,
    robolab_root: Path,
    robolab_commit: str,
    client_sha256: str,
) -> dict:
    """Run the exact frozen RoboLab π0.5 extraction and packing methods."""

    root = robolab_root.resolve()
    client_path = root / "policies/pi0_family/client.py"
    if (
        subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        != robolab_commit
        or not client_path.is_file()
        or sha256_file(client_path) != client_sha256
    ):
        raise ContractError("frozen RoboLab π0.5 client source changed")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from policies.pi0_family.client import Pi0DroidJointposClient

    adapter = Pi0DroidJointposClient.__new__(Pi0DroidJointposClient)
    extracted = Pi0DroidJointposClient._extract_observation(adapter, observation, env_id=0)
    packed = Pi0DroidJointposClient._pack_request(adapter, extracted, prompt)
    if set(packed) != REQUIRED_PACKED_KEYS:
        raise ContractError("frozen π0.5 packed request keys changed")
    return packed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--excluded-smoke-gate", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--fixture-manifest-sha256", required=True)
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("--robolab-commit", required=True)
    parser.add_argument("--robolab-client-sha256", required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    parser.add_argument("--lane-slot", required=True)
    parser.add_argument("--lane-pod-uid", required=True)
    parser.add_argument("--lane-gpu-uuid", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ContractError(f"refusing to overwrite repeat probe: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "repeat_response.json"
    records: list[dict] = []
    dispatched = 0
    successful = 0
    runtime: dict = {}
    base = {
        "schema_version": "vla-wam-shared-v3c002r001-single-server-repeat-response-v1",
        "repair_id": "V3-C002-R001",
        "lane_slot": args.lane_slot,
        "behavioral_episode_count": 0,
        "excluded_from_behavioral_denominators": True,
        "probe_seed": PROBE_SEED,
        "sequence": list(SEQUENCE),
        "records": records,
    }
    try:
        smoke = read_finite_json(args.excluded_smoke_gate)
        if not isinstance(smoke, dict) or smoke.get("schema_version") != SMOKE_GATE_SCHEMA or smoke.get("status") != "passed_repair_excluded_four_cell_smoke" or smoke.get("passed") is not True:
            raise ContractError("repair excluded smoke did not pass")
        smoke_fixture = smoke.get("repeat_fixture", {})
        if smoke_fixture.get("sha256") != args.fixture_sha256 or smoke_fixture.get("bytes") != args.fixture.stat().st_size:
            raise ContractError("repeat fixture is not the retained global-smoke request-zero fixture")
        runtime = validate_runtime_manifest(
            args.runtime_manifest,
            args.runtime_manifest_sha256,
            registration_path=args.parent_registration,
            queue_path=args.queue,
            pod_uid=args.lane_pod_uid,
            gpu_uuid=args.lane_gpu_uuid,
        )["runtime_identity"]
        if runtime.get("server_port") != args.remote_port:
            raise ContractError("repeat endpoint differs from runtime")
        if (
            args.robolab_commit != FROZEN_ROBOLAB_COMMIT
            or args.robolab_client_sha256 != FROZEN_PI05_CLIENT_SHA256
            or runtime.get("robolab_commit") != args.robolab_commit
        ):
            raise ContractError("repeat RoboLab client differs from the registered runtime")
        if not args.fixture.is_file() or sha256_file(args.fixture) != args.fixture_sha256:
            raise ContractError("repeat fixture changed")
        if not args.fixture_manifest.is_file() or sha256_file(args.fixture_manifest) != args.fixture_manifest_sha256:
            raise ContractError("repeat fixture manifest changed")
        observation, fixture_manifest = reconstruct_native_fixture(args.fixture, args.fixture_manifest)
        prompts = registered_prompts()
        client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
        for ordinal, condition in enumerate(SEQUENCE):
            prompt = prompts[condition]["prompt"]
            request = exact_pi05_request(
                observation,
                prompt,
                robolab_root=args.robolab_root,
                robolab_commit=args.robolab_commit,
                client_sha256=args.robolab_client_sha256,
            )
            if set(request) != REQUIRED_PACKED_KEYS:
                raise ContractError("π0.5 request packing changed before dispatch")
            dispatched += 1
            response = client.infer({**request, "sampling_seed": PROBE_SEED})
            successful += 1
            if response.get("v2a010_sampling_seed") != PROBE_SEED:
                raise ContractError("repeat seed echo changed")
            actions = np.asarray(response.get("actions"), dtype=np.float32)
            if actions.shape != (15, 8) or not np.isfinite(actions).all():
                raise ContractError("repeat response is not finite [15,8]")
            path = args.output_dir / f"request{ordinal}_{condition}.npy"
            np.save(path, actions, allow_pickle=False)
            records.append({
                "ordinal": ordinal,
                "condition": condition,
                "actions": file_binding(path),
                "seed_echo": PROBE_SEED,
                "packed_request_keys": sorted(request),
                "prompt_utf8_hex": prompt.encode("utf-8").hex(),
            })
        arrays = [np.load(record["actions"]["path"], allow_pickle=False) for record in records]
        exact = bool(np.array_equal(arrays[0], arrays[2]))
        sensitive = bool(not np.array_equal(arrays[0], arrays[1]))
        if not exact or not sensitive:
            raise ContractError("single-server repeat/sensitivity gate failed")
        value = {
            **base,
            "status": "completed_excluded_single_server_interleaved_repeat",
            "passed": True,
            "model_request_count": dispatched,
            "successful_response_count": successful,
            "fixture_sha256": args.fixture_sha256,
            "fixture": file_binding(args.fixture),
            "fixture_manifest_sha256": args.fixture_manifest_sha256,
            "fixture_manifest": file_binding(args.fixture_manifest),
            "fixture_observation_payload_sha256": fixture_manifest["observation_payload_sha256"],
            "robolab_client": file_binding(args.robolab_root / "policies/pi0_family/client.py"),
            "robolab_commit": args.robolab_commit,
            "first_final_repeat_exact": exact,
            "prompt_sensitivity_distinct": sensitive,
            "runtime_manifest": file_binding(args.runtime_manifest),
            "excluded_smoke_gate": file_binding(args.excluded_smoke_gate),
        }
        for key in ("policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity"):
            value[key] = runtime[key]
        _write(output, value)
        print(json.dumps({"status": value["status"], "sha256": sha256_file(output)}, sort_keys=True))
    except BaseException as exc:
        failure = {
            **base,
            "status": "failed_excluded_single_server_repeat_retained",
            "passed": False,
            "model_request_count": dispatched,
            "successful_response_count": successful,
            "first_final_repeat_exact": False,
            "prompt_sensitivity_distinct": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        for key in ("policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity"):
            if key in runtime:
                failure[key] = runtime[key]
        _write(output, failure)
        raise


if __name__ == "__main__":
    main()
