"""Lossless, model-agnostic request-zero observation pairing for V3-E004.

Realtime RTX is not byte deterministic across independent Isaac processes.  A
matched behavioral pair therefore retains the LEFT reset observation once and
replays those exact non-language bytes only for the RIGHT policy's first
request.  The native RIGHT reset must first match the LEFT physical and camera
contract bit-for-bit.  All observations after the first executed action remain
native.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


AMENDMENT_SCHEMA = "vla-wam-shared-v3e004-request0-observation-replay-amendment-v1"
CACHE_MANIFEST_SCHEMA = "vla-wam-shared-v3e004-request0-observation-cache-v1"
RESET_CONTRACT_SCHEMA = "vla-wam-shared-v3e004-request0-reset-contract-v1"
REPLAY_ATTESTATION_SCHEMA = "vla-wam-shared-v3e004-request0-replay-attestation-v1"
CAPTURE_ATTESTATION_SCHEMA = "vla-wam-shared-v3e004-request0-capture-attestation-v1"
EVIDENCE_ENVELOPE_SCHEMA = "vla-wam-shared-v3e004-request0-evidence-envelope-v1"
LANE_PREFLIGHT_SCHEMA = "vla-wam-shared-v3e004-request0-lane-preflight-v1"
MODEL_BLIND_PREFLIGHT_SCHEMA = "vla-wam-shared-v3e004-standalone-model-blind-droid-gate-v2"


class Request0ReplayError(RuntimeError):
    """A request-zero cache, reset, or replay binding failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Request0ReplayError(message)


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    _require(path.is_file() and path.stat().st_size > 0, f"missing retained artifact: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _bound_artifact(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} artifact binding is missing")
    path = Path(str(value.get("path"))).resolve()
    artifact = file_record(path)
    for key in ("path", "bytes", "sha256"):
        _require(value.get(key) == artifact[key], f"{label} artifact binding changed for {key}")
    return artifact


def _bound_json_artifact(value: Any, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = _bound_artifact(value, label)
    path = Path(artifact["path"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Request0ReplayError(f"{label} is not UTF-8 JSON: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must be an object")
    return artifact, payload


def validate_lane_preflight(
    lane_release: Mapping[str, Any],
    *,
    amendment_sha256: str,
    model_id: str,
    lane_pod_uid: str,
    lane_gpu_uuid: str,
) -> dict[str, Any]:
    """Verify that a lane release binds the prospective zero-request round trip."""

    value = lane_release.get("request0_replay_preflight")
    _require(isinstance(value, Mapping) and value.get("schema_version") == LANE_PREFLIGHT_SCHEMA, "lane lacks R001 preflight")
    amendment_artifact, amendment = _bound_json_artifact(value.get("amendment"), "R001 lane amendment")
    _require(amendment_artifact["sha256"] == amendment_sha256, "lane R001 amendment digest changed")
    _require(amendment.get("schema_version") == AMENDMENT_SCHEMA, "lane R001 amendment schema changed")
    reports: dict[str, dict[str, Any]] = {}
    identities: set[str] = set()
    pair_ids: set[str] = set()
    shared_artifact_hashes: dict[str, set[str]] = {
        "cache_manifest": set(),
        "observation_cache": set(),
        "reset_contract": set(),
    }
    for relation, mode in (("left", "capture_left"), ("right", "replay_right")):
        artifact, report = _bound_json_artifact(value.get(f"{relation}_report"), f"R001 lane {relation} report")
        _require(report.get("schema_version") == MODEL_BLIND_PREFLIGHT_SCHEMA, f"R001 lane {relation} report schema changed")
        _require(report.get("passed") is True, f"R001 lane {relation} preflight did not pass")
        _require(report.get("model_request_count") == 0, f"R001 lane {relation} preflight made a model request")
        _require(report.get("behavioral_action_count") == 0, f"R001 lane {relation} preflight made a behavioral action")
        _require(report.get("behavioral_episode_count") == 0, f"R001 lane {relation} preflight ran behavior")
        _require(report.get("model_id") == model_id, f"R001 lane {relation} model differs")
        _require(report.get("pod_uid") == lane_pod_uid, f"R001 lane {relation} pod differs")
        _require(report.get("gpu_uuid") == lane_gpu_uuid, f"R001 lane {relation} GPU differs")
        evidence = report.get("request0_replay")
        _require(isinstance(evidence, Mapping) and evidence.get("schema_version") == EVIDENCE_ENVELOPE_SCHEMA, f"R001 lane {relation} evidence is missing")
        _require(evidence.get("mode") == mode, f"R001 lane {relation} mode changed")
        _require(evidence.get("amendment", {}).get("sha256") == amendment_sha256, f"R001 lane {relation} amendment changed")
        for name in (
            "amendment",
            "cache_manifest",
            "observation_cache",
            "reset_contract",
            "native_reset_contract",
            "attestation",
        ):
            bound = _bound_artifact(evidence.get(name), f"R001 lane {relation} {name}")
            if name in shared_artifact_hashes:
                shared_artifact_hashes[name].add(bound["sha256"])
        _, attestation = _bound_json_artifact(evidence.get("attestation"), f"R001 lane {relation} attestation")
        expected_attestation_schema = CAPTURE_ATTESTATION_SCHEMA if relation == "left" else REPLAY_ATTESTATION_SCHEMA
        _require(attestation.get("schema_version") == expected_attestation_schema, f"R001 lane {relation} attestation schema changed")
        _require(attestation.get("model_request_count_at_attestation") == 0, f"R001 lane {relation} attestation followed a request")
        _require(attestation.get("behavioral_action_count_at_attestation") == 0, f"R001 lane {relation} attestation followed an action")
        identities.add(str(evidence.get("pair_identity_sha256")))
        pair_ids.add(str(report.get("matched_pair_id")))
        reports[relation] = {"artifact": artifact, "report": report}
    _require(len(identities) == 1 and all(len(value) == 64 for value in identities), "R001 lane pair identity differs")
    _require(len(pair_ids) == 1, "R001 lane reports are not one matched pair")
    _require(all(len(values) == 1 for values in shared_artifact_hashes.values()), "R001 lane reports do not share one LEFT cache/reset")
    _require(value.get("pair_identity_sha256") == next(iter(identities)), "R001 lane release pair identity changed")
    return {
        "schema_version": LANE_PREFLIGHT_SCHEMA,
        "amendment": amendment_artifact,
        "left_report": reports["left"]["artifact"],
        "right_report": reports["right"]["artifact"],
        "pair_identity_sha256": next(iter(identities)),
        "matched_pair_id": next(iter(pair_ids)),
    }


def _write_new_json(path: Path, value: Any) -> None:
    path = Path(path).resolve()
    _require(not path.exists(), f"refusing to overwrite retained artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_amendment(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = Path(path).resolve()
    _require(path.is_file() and sha256_file(path) == expected_sha256, "request-zero amendment digest mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Request0ReplayError(f"request-zero amendment is not UTF-8 JSON: {exc}") from exc
    _require(value.get("schema_version") == AMENDMENT_SCHEMA, "request-zero amendment schema mismatch")
    _require(value.get("registered_before_new_request") is True, "request-zero amendment was not preregistered")
    return value


def _native_array(value: Any) -> tuple[np.ndarray, str]:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        array = value.detach().cpu().numpy()
        kind = "torch_tensor"
    elif isinstance(value, np.ndarray):
        array = value
        kind = "numpy_array"
    elif isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        array = np.asarray(value)
        kind = "python_scalar"
    else:
        raise Request0ReplayError(f"unsupported observation leaf: {type(value).__name__}")
    array = np.ascontiguousarray(array)
    _require(array.dtype.kind in "biufc", f"unsupported observation dtype: {array.dtype}")
    if array.dtype.kind in "fc":
        _require(bool(np.isfinite(array).all()), "non-finite observation leaf")
    return array, kind


def _flatten_observation(value: Any) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], Any]:
    leaves: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}

    def visit(child: Any, path: tuple[Any, ...]) -> Any:
        if isinstance(child, Mapping):
            _require(all(isinstance(key, str) for key in child), "observation mapping keys must be strings")
            return {
                "container": "mapping",
                "children": {key: visit(child[key], (*path, key)) for key in sorted(child)},
            }
        if isinstance(child, tuple):
            return {
                "container": "tuple",
                "children": [visit(item, (*path, index)) for index, item in enumerate(child)],
            }
        if isinstance(child, list):
            return {
                "container": "list",
                "children": [visit(item, (*path, index)) for index, item in enumerate(child)],
            }
        array, kind = _native_array(child)
        storage_key = f"leaf{len(leaves):04d}"
        raw = array.tobytes(order="C")
        row = {
            "path": list(path),
            "storage_key": storage_key,
            "native_kind": kind,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "byte_length": len(raw),
            "data_sha256": hashlib.sha256(raw).hexdigest(),
        }
        leaves.append(row)
        arrays[storage_key] = array.copy(order="C")
        return {"leaf": storage_key}

    structure = visit(value, ())
    _require(leaves, "observation contains no array leaves")
    return leaves, arrays, structure


def observation_descriptor(value: Any) -> dict[str, Any]:
    leaves, _, structure = _flatten_observation(value)
    contract_leaves = [
        {
            "path": row["path"],
            "native_kind": row["native_kind"],
            "dtype": row["dtype"],
            "shape": row["shape"],
        }
        for row in leaves
    ]
    return {"structure": structure, "leaves": contract_leaves}


def observation_payload_sha256(value: Any) -> str:
    leaves, _, structure = _flatten_observation(value)
    return canonical_json_sha256({"structure": structure, "leaves": leaves})


def _tensor_row(value: Any) -> dict[str, Any]:
    array, _ = _native_array(value)
    raw = array.tobytes(order="C")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "values": array.tolist(),
        "data_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _first_env(value: Any) -> Any:
    try:
        return value[0]
    except (IndexError, KeyError, TypeError) as exc:
        raise Request0ReplayError("live state tensor lacks environment zero") from exc


def build_reset_contract(
    *,
    env: Any,
    physical_object_names: Sequence[str],
    camera_rows: Mapping[str, Mapping[str, Any]],
    observation: Any,
) -> dict[str, Any]:
    """Capture the full live physical state and non-pixel camera contract."""

    robot = env.scene["robot"].data
    robot_fields = {
        name: _tensor_row(_first_env(getattr(robot, attribute)))
        for name, attribute in (
            ("joint_position", "joint_pos"),
            ("joint_velocity", "joint_vel"),
            ("root_position", "root_pos_w"),
            ("root_quaternion_wxyz", "root_quat_w"),
            ("root_linear_velocity", "root_lin_vel_w"),
            ("root_angular_velocity", "root_ang_vel_w"),
        )
    }
    objects: dict[str, Any] = {}
    for name in sorted(set(str(item) for item in physical_object_names)):
        data = env.scene[name].data
        objects[name] = {
            field: _tensor_row(_first_env(getattr(data, attribute)))
            for field, attribute in (
                ("root_position", "root_pos_w"),
                ("root_quaternion_wxyz", "root_quat_w"),
                ("root_linear_velocity", "root_lin_vel_w"),
                ("root_angular_velocity", "root_ang_vel_w"),
            )
        }
    cameras: dict[str, Any] = {}
    for name, raw in sorted(camera_rows.items()):
        row = dict(raw)
        row.pop("rgb_source_sha256", None)
        cameras[str(name)] = row
    payload = {
        "schema_version": RESET_CONTRACT_SCHEMA,
        "robot": robot_fields,
        "rigid_objects": objects,
        "cameras": cameras,
        "observation_contract": observation_descriptor(observation),
    }
    payload["reset_contract_sha256"] = canonical_json_sha256(payload)
    return payload


def capture_left_observation(
    *,
    observation: Any,
    reset_contract: Mapping[str, Any],
    amendment_path: Path,
    amendment_sha256: str,
    cell_id: str,
    matched_pair_id: str,
    cache_path: Path,
    manifest_path: Path,
    reset_contract_path: Path,
) -> dict[str, Any]:
    load_amendment(amendment_path, amendment_sha256)
    _require(cell_id.endswith(":left"), "request-zero cache source must be the registered LEFT cell")
    cache_path = Path(cache_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    reset_contract_path = Path(reset_contract_path).resolve()
    _require(not cache_path.exists() and not manifest_path.exists() and not reset_contract_path.exists(), "request-zero cache output already exists")
    leaves, arrays, structure = _flatten_observation(observation)
    reset_value = dict(reset_contract)
    _require(reset_value.get("schema_version") == RESET_CONTRACT_SCHEMA, "reset contract schema mismatch")
    expected_contract_sha = reset_value.get("reset_contract_sha256")
    unsigned_reset = {key: value for key, value in reset_value.items() if key != "reset_contract_sha256"}
    _require(expected_contract_sha == canonical_json_sha256(unsigned_reset), "reset contract self-digest mismatch")
    _write_new_json(reset_contract_path, reset_value)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("xb") as handle:
        np.savez(handle, **arrays)
    payload_sha = canonical_json_sha256({"structure": structure, "leaves": leaves})
    manifest = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "source_relation": "left",
        "source_cell_id": cell_id,
        "matched_pair_id": matched_pair_id,
        "amendment": file_record(amendment_path),
        "observation_cache": file_record(cache_path),
        "reset_contract": {
            **file_record(reset_contract_path),
            "payload_sha256": expected_contract_sha,
        },
        "observation_payload_sha256": payload_sha,
        "observation_structure_sha256": canonical_json_sha256(structure),
        "leaves": leaves,
        "structure": structure,
        "model_request_count_at_capture": 0,
        "behavioral_action_count_at_capture": 0,
    }
    _require(manifest["amendment"]["sha256"] == amendment_sha256, "amendment binding changed")
    _write_new_json(manifest_path, manifest)
    return {**manifest, "manifest": file_record(manifest_path)}


def write_capture_attestation(
    *,
    amendment_path: Path,
    amendment_sha256: str,
    cell_id: str,
    matched_pair_id: str,
    cache_path: Path,
    manifest_path: Path,
    reset_contract_path: Path,
    observation_payload_sha256: str,
    reset_contract_payload_sha256: str,
    attestation_path: Path,
) -> dict[str, Any]:
    """Bind a completed LEFT cache before any model request or action."""

    load_amendment(amendment_path, amendment_sha256)
    _require(cell_id.endswith(":left"), "request-zero capture attestation must bind LEFT")
    attestation = {
        "schema_version": CAPTURE_ATTESTATION_SCHEMA,
        "registered_cell_id": cell_id,
        "matched_pair_id": matched_pair_id,
        "mode": "capture_left",
        "amendment": file_record(amendment_path),
        "cache_manifest": file_record(manifest_path),
        "observation_cache": file_record(cache_path),
        "reset_contract": file_record(reset_contract_path),
        "observation_payload_sha256": observation_payload_sha256,
        "reset_contract_payload_sha256": reset_contract_payload_sha256,
        "model_request_count_at_attestation": 0,
        "behavioral_action_count_at_attestation": 0,
        "closed_loop_observation_policy": "native_after_first_executed_action",
    }
    _require(attestation["amendment"]["sha256"] == amendment_sha256, "capture amendment binding changed")
    _write_new_json(attestation_path, attestation)
    return {**attestation, "attestation": file_record(attestation_path)}


def pair_identity_sha256(*, observation_payload_sha256: str, reset_contract_payload_sha256: str) -> str:
    _require(
        len(observation_payload_sha256) == 64 and len(reset_contract_payload_sha256) == 64,
        "request-zero pair identity requires SHA-256 payloads",
    )
    return canonical_json_sha256(
        {
            "observation_payload_sha256": observation_payload_sha256,
            "reset_contract_sha256": reset_contract_payload_sha256,
        }
    )


def evidence_envelope(
    *,
    mode: str,
    amendment_path: Path,
    cache_path: Path,
    manifest_path: Path,
    reset_contract_path: Path,
    native_reset_contract_path: Path,
    attestation_path: Path,
    observation_payload_sha256: str,
    reset_contract_payload_sha256: str,
) -> dict[str, Any]:
    _require(mode in {"capture_left", "replay_right"}, "invalid request-zero evidence mode")
    identity = pair_identity_sha256(
        observation_payload_sha256=observation_payload_sha256,
        reset_contract_payload_sha256=reset_contract_payload_sha256,
    )
    return {
        "schema_version": EVIDENCE_ENVELOPE_SCHEMA,
        "mode": mode,
        "amendment": file_record(amendment_path),
        "cache_manifest": file_record(manifest_path),
        "observation_cache": file_record(cache_path),
        "reset_contract": file_record(reset_contract_path),
        "native_reset_contract": file_record(native_reset_contract_path),
        "attestation": file_record(attestation_path),
        "observation_payload_sha256": observation_payload_sha256,
        "reset_contract_payload_sha256": reset_contract_payload_sha256,
        "pair_identity_sha256": identity,
        "closed_loop_observation_policy": "native_after_first_executed_action",
    }


def _load_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    path = Path(path).resolve()
    _require(path.is_file() and sha256_file(path) == expected_sha256, f"{label} digest mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Request0ReplayError(f"{label} is not UTF-8 JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _value_at_path(value: Any, path: Sequence[Any]) -> Any:
    current = value
    for token in path:
        current = current[token]
    return current


def _convert_like(array: np.ndarray, native: Any, kind: str) -> Any:
    if kind == "torch_tensor":
        _require(hasattr(native, "device") and hasattr(native, "dtype"), "native torch leaf contract changed")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - production runtime owns torch
            raise Request0ReplayError("torch is unavailable for tensor replay") from exc
        return torch.as_tensor(array.copy(), dtype=native.dtype, device=native.device)
    if kind == "numpy_array":
        return array.copy()
    if kind == "python_scalar":
        return array.item()
    raise Request0ReplayError(f"unknown observation leaf kind: {kind}")


def _rebuild_like(native: Any, path: tuple[Any, ...], arrays: Mapping[tuple[Any, ...], tuple[np.ndarray, str]]) -> Any:
    if isinstance(native, Mapping):
        return {key: _rebuild_like(native[key], (*path, key), arrays) for key in native}
    if isinstance(native, tuple):
        return tuple(_rebuild_like(item, (*path, index), arrays) for index, item in enumerate(native))
    if isinstance(native, list):
        return [_rebuild_like(item, (*path, index), arrays) for index, item in enumerate(native)]
    _require(path in arrays, f"cache has no observation leaf at {path}")
    array, kind = arrays[path]
    return _convert_like(array, native, kind)


def replay_left_observation_for_right(
    *,
    native_observation: Any,
    native_reset_contract: Mapping[str, Any],
    amendment_path: Path,
    amendment_sha256: str,
    cell_id: str,
    matched_pair_id: str,
    cache_path: Path,
    cache_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    reset_contract_path: Path,
    reset_contract_file_sha256: str,
    native_reset_contract_path: Path,
    attestation_path: Path,
) -> tuple[Any, dict[str, Any]]:
    load_amendment(amendment_path, amendment_sha256)
    _require(cell_id.endswith(":right"), "request-zero replay target must be the registered RIGHT cell")
    manifest = _load_json(manifest_path, manifest_sha256, "request-zero cache manifest")
    _require(manifest.get("schema_version") == CACHE_MANIFEST_SCHEMA, "request-zero cache manifest schema mismatch")
    _require(manifest.get("matched_pair_id") == matched_pair_id, "request-zero cache belongs to a different pair")
    _require(manifest.get("source_relation") == "left", "request-zero cache source is not LEFT")
    _require(manifest.get("amendment", {}).get("sha256") == amendment_sha256, "request-zero cache amendment changed")
    cache_path = Path(cache_path).resolve()
    _require(cache_path.is_file() and sha256_file(cache_path) == cache_sha256, "request-zero observation cache digest mismatch")
    _require(manifest.get("observation_cache", {}).get("sha256") == cache_sha256, "cache manifest binds another observation archive")
    left_contract = _load_json(reset_contract_path, reset_contract_file_sha256, "LEFT reset contract")
    _require(manifest.get("reset_contract", {}).get("sha256") == reset_contract_file_sha256, "cache manifest binds another reset contract")
    native_contract_value = dict(native_reset_contract)
    _require(native_contract_value.get("schema_version") == RESET_CONTRACT_SCHEMA, "native RIGHT reset contract schema mismatch")
    native_contract_sha = native_contract_value.get("reset_contract_sha256")
    unsigned_native = {key: value for key, value in native_contract_value.items() if key != "reset_contract_sha256"}
    _require(native_contract_sha == canonical_json_sha256(unsigned_native), "native RIGHT reset contract self-digest mismatch")
    _write_new_json(native_reset_contract_path, native_contract_value)
    _require(left_contract == native_contract_value, "RIGHT physical state or camera contract differs from LEFT")
    descriptor = observation_descriptor(native_observation)
    cached_descriptor = {
        "structure": manifest.get("structure"),
        "leaves": [
            {
                "path": row["path"],
                "native_kind": row["native_kind"],
                "dtype": row["dtype"],
                "shape": row["shape"],
            }
            for row in manifest.get("leaves", [])
        ],
    }
    _require(descriptor == cached_descriptor, "RIGHT observation structure/dtype/shape differs from LEFT")
    arrays: dict[tuple[Any, ...], tuple[np.ndarray, str]] = {}
    with np.load(cache_path, allow_pickle=False) as archive:
        expected_keys = {row["storage_key"] for row in manifest["leaves"]}
        _require(set(archive.files) == expected_keys, "observation cache leaf inventory changed")
        for row in manifest["leaves"]:
            array = np.ascontiguousarray(archive[row["storage_key"]])
            raw = array.tobytes(order="C")
            _require(array.dtype.str == row["dtype"] and list(array.shape) == row["shape"], "observation cache dtype/shape changed")
            _require(len(raw) == row["byte_length"] and hashlib.sha256(raw).hexdigest() == row["data_sha256"], "observation cache leaf bytes changed")
            arrays[tuple(row["path"])] = (array.copy(), row["native_kind"])
    replayed = _rebuild_like(native_observation, (), arrays)
    replay_sha = observation_payload_sha256(replayed)
    _require(replay_sha == manifest.get("observation_payload_sha256"), "replayed request-zero observation differs from LEFT")
    attestation = {
        "schema_version": REPLAY_ATTESTATION_SCHEMA,
        "target_relation": "right",
        "target_cell_id": cell_id,
        "matched_pair_id": matched_pair_id,
        "amendment": file_record(amendment_path),
        "cache_manifest": file_record(manifest_path),
        "observation_cache": file_record(cache_path),
        "left_reset_contract": file_record(reset_contract_path),
        "right_native_reset_contract": {
            **file_record(native_reset_contract_path),
            "payload_sha256": native_contract_sha,
        },
        "right_reset_contract_sha256": native_contract_sha,
        "request0_observation_payload_sha256": replay_sha,
        "physical_state_and_camera_contract_bit_identical": True,
        "request0_non_language_bytes_bit_identical": True,
        "model_request_count_at_attestation": 0,
        "behavioral_action_count_at_attestation": 0,
        "closed_loop_observation_policy": "native_after_first_executed_action",
    }
    _write_new_json(attestation_path, attestation)
    return replayed, {**attestation, "attestation": file_record(attestation_path)}
