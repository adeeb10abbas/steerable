#!/usr/bin/env python3
"""Qualify generated-frame timing without using video FPS or policy motion.

The source audit in this module fixes a decoded-frame -> executed-control-boundary
mapping from exact files in the two pinned upstream repositories.  That static
mapping is deliberately insufficient for physical-time qualification.  A
separate generation probe must demonstrate the real returned/latent/decoded
tensor shapes, and a recorder-only RoboLab attempt must supply native control,
physics and original-camera clocks while executing joint-position holds.

No model-returned action is executed by this path.  Presentation FPS,
conditioning FPS, ordinal equality and DreamZero's action/block ratio are not
accepted as evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence
import uuid


STUDY_ID = "WMF-ABLATION-001"
CONTRACT_SCHEMA = "wmf-forecast-timing-lineage-contract-v1"
SOURCE_AUDIT_SCHEMA = "wmf-forecast-source-timing-lineage-v1"
GENERATION_PROBE_SCHEMA = "wmf-forecast-zero-policy-generation-probe-v1"
AUTHORITY_SCHEMA = "wmf-forecast-native-target-authority-v1"
REQUEST_INVENTORY_SCHEMA = "wmf-development-timing-request-inventory-v1"
DEVELOPMENT_TIMING_SCHEMA = "wmf-native-generated-target-timing-v1"
RECORDER_QUEUE_SCHEMA = "wmf-forecast-recorder-qualification-job-v1"
RECORDER_CHILD_SCHEMA = "wmf-forecast-recorder-qualification-child-v1"
RECORDER_ATTEMPT_SCHEMA = "wmf-forecast-recording-attempt-v1"
FIXED_CAPTURE_SCHEMA = "wmf-forecast-layout-fixed-observation-capture-v1"
N3_QUALIFICATION_SCHEMA = "wmf-n3-runtime-qualification-v1"
D1_JOB_SCHEMA = "wmf-d1-qualification-job-receipt-v1"
D1_REPORT_SCHEMA = "wmf-d1-six-request-qualification-v1"
D1_REQUEST_SCHEMA = "wmf-d1-request-receipt-v1"
N3_REQUEST_SCHEMA = "wmf-n3-behavioral-server-request-v1"
N3_LIVE_INPUT_SCHEMA = "wmf-n3-live-zero-policy-input-v1"
CONTRACT_REPOSITORY_RELATIVE = Path(
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "forecast_timing_lineage_contract.json"
)

NATIVE_RUNTIME_FIELD = "forecast_timing.generated_targets"
SIDECAR_RUNTIME_FIELD = "request_timing_sidecar.generated_targets"
CAMERAS = (
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")

EXPECTED_MODELS: dict[str, dict[str, Any]] = {
    "N3": {
        "commit": "411d25b2e35bc441126f48c44a4b93e1c0564274",
        "tree": "1e77852f2240d6e00312342ab29164745042b1ae",
        "checkpoint_revision": "6706d7680581c255ff61e0f3bb49d90eac55c79e",
        "checkpoint_aggregate": "55895125805b12635ece5dd2e88453a1aca6a684bac556f114337ae448ffdd83",
        "decoded_frames": 33,
        "raw_action_rows": 33,
        "returned_actions": 32,
        "action_dimensions": 8,
        "prefix": 32,
        "decoded_frame_indices": list(range(33)),
        "control_boundaries": list(range(33)),
        "source_files": {
            "cosmos_framework/data/vfm/action/datasets/droid_lerobot_dataset.py": "5ba98de83a2cb13dba5e52a2d3d605444af1fa635657558b13bae3b013139a76",
            "cosmos_framework/data/vfm/action/transforms.py": "1a0577cf481c8cadd2dda9cc344f1f5dd60b16b020f4636221c8bb71403cfea4",
            "cosmos_framework/data/vfm/sequence_packing.py": "84cf39f1e2adba885f593cd621509c985b5c17db55d8147501f763dbdfd2edd7",
            "cosmos_framework/model/vfm/mot/unified_3dmrope_utils.py": "8388b7071565ef28b86f271b15bdb99f858b081269f84987802b9f9527af1741",
            "cosmos_framework/inference/action.py": "19ff6b3c3a7ca99628b694978cae0eff6b6fb760e2bfb126ab69cc7cbbb3de23",
            "cosmos_framework/scripts/action_policy_server_robolab.py": "024a6d19048a6ac732ca0b23e741b5da6d38424223783b944bf539b2b6bd9a24",
        },
    },
    "D1": {
        "commit": "ab790c198fbce33503358efbbd4187ce9a89adf3",
        "tree": "6b7ba27f1af81e963a6507f1204c05c65a94098c",
        "checkpoint_revision": "96ad344138c66e82536422432ad742f015784942",
        "checkpoint_aggregate": "b4af0ac93474c3295c1ba841a34a8f2f91a5c3ec3c6aac1431b97689a6618c56",
        "decoded_frames": 9,
        "latent_frames": 3,
        "returned_actions": 24,
        "action_dimensions": 8,
        "prefix": 8,
        "decoded_frame_indices": list(range(9)),
        "control_boundaries": [0, 3, 6, 9, 12, 15, 18, 21, 24],
        "source_files": {
            "groot/vla/data/dataset/lerobot.py": "9ec2bae810e385936ca30c885be41e27242bf4ed7a8e4e276f2fbc548d24da46",
            "groot/vla/data/dataset/lerobot_sharded.py": "e075de07675c77818e8b75ba62b74357284dde2e4cfa7235b225bb59584fbf80",
            "groot/vla/model/dreamzero/transform/dreamzero_cotrain.py": "8b4544b0f1c52a1ba78f764cd1882132b212e4be0a78a86e67ca39ccb3902151",
            "groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py": "7193cd73423472aa252bee73bd80e0d673c89d773ec852e90f50154729b50845",
            "groot/vla/model/dreamzero/modules/wan_video_vae.py": "1f65f8fd915c1e6703da9d320e764df107dbb988b12623a7e087554281c84edc",
            "groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py": "efc5120f73cbce2e00b67d78e8bf71a561e943a909791f039b435abc27ccd605",
        },
    },
}


class TimingQualificationError(RuntimeError):
    """Evidence cannot support a physical generated-target mapping."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TimingQualificationError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any, *, ensure_ascii: bool = False) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TimingQualificationError("value is not finite canonical JSON") from error


def pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def sign_document(value: Mapping[str, Any]) -> dict[str, Any]:
    require("payload_sha256" not in value, "document is already signed")
    result = dict(value)
    result["payload_sha256"] = sha256_bytes(canonical_bytes(result))
    return result


def verify_signed(value: Mapping[str, Any], label: str) -> None:
    observed = value.get("payload_sha256")
    require(_valid_sha(observed), f"{label} lacks a valid payload SHA-256")
    unsigned = dict(value)
    unsigned.pop("payload_sha256")
    require(sha256_bytes(canonical_bytes(unsigned)) == observed, f"{label} payload hash mismatch")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any) -> None:
    supplied = Path(path)
    require(not supplied.is_symlink(), f"output is a symlink: {supplied}")
    path = supplied.resolve()
    require(not path.exists(), f"refusing to overwrite immutable output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(pretty_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in output, f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise TimingQualificationError(f"non-finite JSON token: {value}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    supplied = Path(path)
    require(not supplied.is_symlink(), f"{label} is a symlink")
    path = supplied.resolve()
    require(path.is_file(), f"{label} is missing: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except TimingQualificationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TimingQualificationError(f"{label} is not readable JSON: {path}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _resolve_path(raw: Any, *, base: Path, label: str) -> Path:
    require(isinstance(raw, str) and raw, f"{label} path is missing")
    supplied = Path(raw)
    if not supplied.is_absolute():
        supplied = base / supplied
    require(not supplied.is_symlink(), f"{label} is a symlink")
    return supplied.resolve()


def verify_descriptor(
    descriptor: Any,
    *,
    base: Path,
    label: str,
    require_bytes: bool = False,
) -> tuple[dict[str, Any], Path]:
    require(isinstance(descriptor, Mapping), f"{label} descriptor is missing")
    require(_valid_sha(descriptor.get("sha256")), f"{label} descriptor SHA-256 is invalid")
    path = _resolve_path(descriptor.get("path"), base=base, label=label)
    require(path.is_file(), f"{label} file is missing: {path}")
    require(sha256_file(path) == descriptor["sha256"], f"{label} file hash mismatch")
    if require_bytes or "bytes" in descriptor:
        require(type(descriptor.get("bytes")) is int, f"{label} byte count is missing")
        require(path.stat().st_size == descriptor["bytes"], f"{label} byte count mismatch")
    return dict(descriptor), path


def file_descriptor(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def require_descriptor_matches(
    descriptor: Any,
    expected_path: Path,
    *,
    base: Path,
    label: str,
) -> None:
    observed, path = verify_descriptor(descriptor, base=base, label=label, require_bytes=True)
    expected = file_descriptor(expected_path)
    require(path == Path(expected_path).resolve(), f"{label} points to another file")
    require(observed.get("sha256") == expected["sha256"]
            and observed.get("bytes") == expected["bytes"], f"{label} identity changed")


def require_contract_descriptor_matches(
    descriptor: Any,
    expected_path: Path,
    *,
    base: Path,
    label: str,
) -> None:
    """Match immutable contract bytes across different staged source commits.

    Queue staging gives the same repository file a different absolute prefix
    for every source commit.  Contract provenance is therefore its frozen
    repository-relative identity plus exact bytes/hash, not the lexical path
    of one otherwise equivalent staged checkout.
    """

    observed, path = verify_descriptor(
        descriptor, base=base, label=label, require_bytes=True
    )
    expected_path = Path(expected_path).resolve()
    expected = file_descriptor(expected_path)
    suffix = CONTRACT_REPOSITORY_RELATIVE.parts
    require(
        tuple(path.parts[-len(suffix):]) == suffix
        and tuple(expected_path.parts[-len(suffix):]) == suffix,
        f"{label} repository-relative identity changed",
    )
    require(
        observed.get("sha256") == expected["sha256"]
        and observed.get("bytes") == expected["bytes"],
        f"{label} identity changed",
    )


def _read_hashed_json(path: Path, expected_sha256: str, label: str) -> tuple[dict[str, Any], Path]:
    require(_valid_sha(expected_sha256), f"{label} expected SHA-256 is invalid")
    supplied = Path(path)
    require(not supplied.is_symlink(), f"{label} is a symlink")
    resolved = supplied.resolve()
    require(resolved.is_file(), f"{label} is missing")
    require(sha256_file(resolved) == expected_sha256, f"{label} file hash mismatch")
    return load_json(resolved, label), resolved


def _contract_mapping(model: Mapping[str, Any]) -> list[dict[str, int]]:
    lineage = model.get("lineage")
    require(isinstance(lineage, Mapping), "contract model lineage is missing")
    frames = lineage.get("decoded_frame_indices")
    boundaries = lineage.get("executed_control_boundaries")
    require(isinstance(frames, list) and isinstance(boundaries, list), "contract mapping arrays are missing")
    require(len(frames) == len(boundaries) and frames, "contract mapping arrays differ in length")
    return [
        {"generated_frame_index": frame, "executed_control_boundary": boundary}
        for frame, boundary in zip(frames, boundaries)
    ]


def load_contract(path: Path, expected_sha256: str) -> tuple[dict[str, Any], Path]:
    value, resolved = _read_hashed_json(path, expected_sha256, "timing lineage contract")
    require(value.get("schema_version") == CONTRACT_SCHEMA, "timing lineage contract schema changed")
    require(value.get("study_id") == STUDY_ID, "timing lineage contract study changed")
    require(
        value.get("status") == "source_mapping_frozen_live_clock_probe_required",
        "timing lineage contract improperly claims live qualification",
    )
    prohibited = value.get("prohibited_inferences")
    require(isinstance(prohibited, Mapping) and prohibited and all(item is True for item in prohibited.values()),
            "timing lineage contract weakened a prohibited-inference gate")
    require(value.get("original_camera_ids") == list(CAMERAS), "original-camera inventory changed")
    models = value.get("models")
    require(isinstance(models, Mapping) and set(models) == set(EXPECTED_MODELS), "timing model inventory changed")
    for model_id, expected in EXPECTED_MODELS.items():
        model = models[model_id]
        require(isinstance(model, Mapping) and model.get("configuration_id") == model_id,
                f"{model_id} contract identity changed")
        source = model.get("source")
        checkpoint = model.get("checkpoint")
        require(isinstance(source, Mapping) and isinstance(checkpoint, Mapping), f"{model_id} source identity is missing")
        require(source.get("commit") == expected["commit"] and source.get("git_tree") == expected["tree"],
                f"{model_id} source pin changed")
        require(checkpoint.get("revision") == expected["checkpoint_revision"], f"{model_id} checkpoint revision changed")
        require(checkpoint.get("payload_aggregate_sha256") == expected["checkpoint_aggregate"],
                f"{model_id} checkpoint aggregate changed")
        files = model.get("source_files")
        require(isinstance(files, list), f"{model_id} source file inventory is missing")
        observed = {row.get("path"): row.get("sha256") for row in files if isinstance(row, Mapping)}
        require(observed == expected["source_files"], f"{model_id} source file inventory changed")
        mapping = _contract_mapping(model)
        require([row["generated_frame_index"] for row in mapping] == expected["decoded_frame_indices"],
                f"{model_id} decoded-frame mapping changed")
        require([row["executed_control_boundary"] for row in mapping] == expected["control_boundaries"],
                f"{model_id} control-boundary mapping changed")
        lineage = model["lineage"]
        require(lineage.get("returned_action_horizon") == expected["returned_actions"],
                f"{model_id} returned action horizon changed")
        require(lineage.get("unchanged_executed_prefix_horizon") == expected["prefix"],
                f"{model_id} unchanged prefix changed")
        require(isinstance(lineage.get("mapping_basis"), str) and lineage["mapping_basis"],
                f"{model_id} mapping basis is missing")
        live_generation = model.get("live_generation_receipt")
        require(isinstance(live_generation, Mapping), f"{model_id} live generation contract is missing")
        if model_id == "N3":
            require(live_generation.get("required_source_capture_provenance")
                    == "live_fixed_observation_zero_policy_input"
                    and live_generation.get("historical_generation_receipt_eligible") is False
                    and live_generation.get("input_preparation_schema") == N3_LIVE_INPUT_SCHEMA,
                    "N3 live generation requirement changed or admits historical timing")
    clock = value.get("live_clock_probe")
    require(isinstance(clock, Mapping), "live clock probe contract is missing")
    require(clock.get("required_actions") == 450 and clock.get("required_observations") == 451,
            "zero-policy clock trace length changed")
    require(clock.get("required_model_requests") == 0, "clock trace permits model requests")
    return value, resolved


def _run_git(root: Path, *arguments: str, binary: bool = False) -> bytes | str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise TimingQualificationError(f"git {' '.join(arguments)} failed") from error
    if result.returncode:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise TimingQualificationError(f"git {' '.join(arguments)} failed: {detail}")
    if binary:
        return result.stdout
    return result.stdout.decode("utf-8", "surrogateescape")


def audit_source(
    *,
    model_id: str,
    source_root: Path,
    contract_path: Path,
    contract_sha256: str,
) -> dict[str, Any]:
    contract, resolved_contract = load_contract(contract_path, contract_sha256)
    require(model_id in EXPECTED_MODELS, f"unsupported model: {model_id}")
    supplied = Path(source_root)
    require(not supplied.is_symlink(), "source root is a symlink")
    root = supplied.resolve()
    require(root.is_dir(), f"source root is missing: {root}")
    expected = EXPECTED_MODELS[model_id]
    head = str(_run_git(root, "rev-parse", "HEAD")).strip()
    tree = str(_run_git(root, "rev-parse", "HEAD^{tree}")).strip()
    require(COMMIT_RE.fullmatch(head) is not None and head == expected["commit"],
            f"{model_id} source checkout is not at the pinned commit")
    require(COMMIT_RE.fullmatch(tree) is not None and tree == expected["tree"],
            f"{model_id} source tree changed")
    file_rows: list[dict[str, Any]] = []
    for relative, expected_hash in expected["source_files"].items():
        path = root / relative
        require(not path.is_symlink(), f"{model_id} source file is a symlink: {relative}")
        require(path.is_file(), f"{model_id} source file is missing: {relative}")
        working_hash = sha256_file(path)
        blob = _run_git(root, "show", f"{expected['commit']}:{relative}", binary=True)
        assert isinstance(blob, bytes)
        blob_hash = sha256_bytes(blob)
        require(blob_hash == expected_hash, f"{model_id} pinned Git blob hash changed: {relative}")
        require(working_hash == expected_hash, f"{model_id} imported source bytes differ from the pin: {relative}")
        file_rows.append({
            "path": relative,
            "sha256": expected_hash,
            "bytes": path.stat().st_size,
            "working_copy_equals_pinned_blob": True,
        })
    model_contract = contract["models"][model_id]
    mapping = _contract_mapping(model_contract)
    result = sign_document({
        "schema_version": SOURCE_AUDIT_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": model_id,
        "status": "source_mapping_passed_live_probe_required",
        "audited_at_utc": utc_now(),
        "contract": file_descriptor(resolved_contract),
        "source": {
            "path": str(root),
            "commit": head,
            "git_tree": tree,
            "required_files": file_rows,
        },
        "checkpoint_expected": dict(model_contract["checkpoint"]),
        "source_mapping": mapping,
        "source_mapping_sha256": sha256_bytes(canonical_bytes(mapping)),
        "mapping_basis": model_contract["lineage"]["mapping_basis"],
        "dataset_timestamp_rows": dict(model_contract["lineage"]["dataset_timestamp_rows"]),
        "inference_tensor_witness_expected": dict(model_contract["lineage"]["inference_tensor_witness"]),
        "physical_time_qualified": False,
        "prohibited_inferences_used": [],
        "claim_boundary": "Exact pinned source lineage only; no physical seconds or forecast accuracy are qualified.",
    })
    return result


def validate_source_audit(
    path: Path,
    expected_sha256: str,
    *,
    expected_model: str | None = None,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value, _ = _read_hashed_json(path, expected_sha256, "source timing audit")
    require(value.get("schema_version") == SOURCE_AUDIT_SCHEMA, "source timing audit schema changed")
    verify_signed(value, "source timing audit")
    model_id = value.get("model_id")
    require(model_id in EXPECTED_MODELS, "source timing audit model is invalid")
    if expected_model is not None:
        require(model_id == expected_model, "source timing audit model mismatch")
    expected = EXPECTED_MODELS[model_id]
    require(value.get("study_id") == STUDY_ID, "source timing audit study changed")
    require(value.get("status") == "source_mapping_passed_live_probe_required", "source audit did not pass")
    require(value.get("physical_time_qualified") is False, "source audit improperly claims physical timing")
    source = value.get("source")
    require(isinstance(source, Mapping), "source audit identity is missing")
    require(source.get("commit") == expected["commit"] and source.get("git_tree") == expected["tree"],
            "source audit pin changed")
    mapping = [
        {"generated_frame_index": frame, "executed_control_boundary": boundary}
        for frame, boundary in zip(expected["decoded_frame_indices"], expected["control_boundaries"])
    ]
    require(value.get("source_mapping") == mapping, "source audit mapping changed")
    require(value.get("source_mapping_sha256") == sha256_bytes(canonical_bytes(mapping)),
            "source audit mapping hash changed")
    require(value.get("prohibited_inferences_used") == [], "source audit used a prohibited inference")
    if contract is not None:
        require(value.get("checkpoint_expected") == contract["models"][model_id]["checkpoint"],
                "source audit checkpoint expectation differs from contract")
    return value


def _verify_capture_receipt(path: Path, expected_sha256: str, *, model_id: str) -> dict[str, Any]:
    capture, resolved = _read_hashed_json(path, expected_sha256, "live fixed-observation capture")
    require(capture.get("schema_version") == FIXED_CAPTURE_SCHEMA, "live capture schema changed")
    require(capture.get("status") == "passed", "live fixed-observation capture did not pass")
    require(capture.get("model_request_count") == 0 and capture.get("behavioral_action_count") == 0,
            "fixed-observation capture contains model/behavioral activity")
    source = capture.get("source_capture")
    require(isinstance(source, Mapping), "live capture lacks source camera identities")
    frames = source.get("camera_frame_ids")
    times = source.get("camera_capture_time_ns")
    sources = source.get("camera_timestamp_source")
    require(isinstance(frames, Mapping) and isinstance(times, Mapping) and isinstance(sources, Mapping),
            "live capture camera clocks are incomplete")
    for camera in CAMERAS:
        require(camera in frames and camera in times and camera in sources,
                f"live capture lacks original camera {camera}")
        require(type(times[camera]) is int and times[camera] >= 0,
                f"live capture timestamp is invalid for {camera}")
        require(isinstance(sources[camera], str) and sources[camera],
                f"live capture timestamp source is invalid for {camera}")
    artifacts = capture.get("artifacts")
    fixtures = capture.get("model_fixtures")
    require(isinstance(artifacts, Mapping) and isinstance(fixtures, Mapping), "live capture artifact chain is missing")
    require(model_id in fixtures, f"live capture lacks the {model_id} official input fixture")
    fixture = fixtures[model_id]
    require(isinstance(fixture, Mapping) and isinstance(fixture.get("fixture"), Mapping),
            f"live capture {model_id} fixture descriptor is missing")
    descriptor = fixture["fixture"]
    _, fixture_path = verify_descriptor(descriptor, base=resolved.parent, label=f"{model_id} live fixture", require_bytes=True)
    result = {
        "capture": file_descriptor(resolved),
        "capture_id": capture.get("capture_id"),
        "settled_reset_identity": capture.get("settled_reset_identity"),
        "camera_frame_ids": {camera: frames[camera] for camera in CAMERAS},
        "camera_capture_time_ns": {camera: times[camera] for camera in CAMERAS},
        "camera_timestamp_source": {camera: sources[camera] for camera in CAMERAS},
        "model_fixture": file_descriptor(fixture_path),
    }
    if model_id == "N3":
        result["model_fixture_wire_sha256"] = _n3_fixture_wire_sha256(fixture_path)
    return result


def _numpy_array_identity(array: Any) -> dict[str, Any]:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    require(not contiguous.dtype.hasobject, "object array is prohibited")
    header = {"kind": "numpy", "dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
    digest = hashlib.sha256()
    digest.update(canonical_bytes(header, ensure_ascii=True))
    digest.update(contiguous.tobytes(order="C"))
    return {**header, "value_sha256": digest.hexdigest()}


def _n3_fixture_arrays(path: Path) -> dict[str, Any]:
    import numpy as np

    expected = {
        "observation/image": ((540, 640, 3), np.dtype("uint8")),
        "observation/joint_position": ((7,), np.dtype("float32")),
        "observation/gripper_position": ((1,), np.dtype("float32")),
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == set(expected), "N3 live fixture array inventory changed")
            arrays = {key: np.ascontiguousarray(archive[key]).copy() for key in expected}
    except TimingQualificationError:
        raise
    except Exception as error:
        raise TimingQualificationError(f"N3 live fixture is not a safe readable NPZ: {path}") from error
    for key, (shape, dtype) in expected.items():
        value = arrays[key]
        require(value.shape == shape and value.dtype == dtype,
                f"N3 live fixture array changed: {key}")
        if value.dtype.kind in "fc":
            require(bool(np.isfinite(value).all()), f"N3 live fixture array is non-finite: {key}")
    return arrays


def _n3_fixture_wire_sha256(path: Path) -> str:
    arrays = _n3_fixture_arrays(path)
    identities = {key: _numpy_array_identity(value) for key, value in arrays.items()}
    return sha256_bytes(canonical_bytes(identities, ensure_ascii=True))


def prepare_n3_live_input(
    *,
    capture_path: Path,
    capture_sha256: str,
    camera_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Build an immutable live N3 qualification input; issue no model request."""

    require(camera_id in CAMERAS, "selected camera is not an original recorder camera")
    capture = _verify_capture_receipt(capture_path, capture_sha256, model_id="N3")
    capture_value, resolved_capture = _read_hashed_json(
        capture_path, capture_sha256, "N3 live fixed-observation capture"
    )
    source_capture = capture_value.get("source_capture")
    require(isinstance(source_capture, Mapping)
            and isinstance(source_capture.get("simulator_observation_id"), str)
            and source_capture["simulator_observation_id"],
            "N3 live capture lacks its simulator observation identity")
    supplied = Path(output_dir)
    require(not supplied.exists() and not supplied.is_symlink(),
            "N3 live input output directory already exists or is a symlink")
    root = supplied.resolve()
    root.mkdir(parents=True)
    fixture_path = Path(capture["model_fixture"]["path"])
    arrays = _n3_fixture_arrays(fixture_path)
    archive_keys = {key: f"wire_{index:02d}" for index, key in enumerate(arrays)}
    payload_path = root / "observation.npz"
    try:
        import numpy as np

        with payload_path.open("xb") as handle:
            np.savez(handle, **{archive_keys[key]: arrays[key] for key in arrays})
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if payload_path.exists():
            payload_path.unlink()
        raise
    _fsync_directory(root)
    source = {
        "provenance_kind": "live_fixed_observation_zero_policy_input",
        "capture_id": capture["capture_id"],
        "settled_reset_identity": capture["settled_reset_identity"],
        "simulator_observation_id": source_capture["simulator_observation_id"],
        "selected_original_camera_id": camera_id,
        "camera_frame_ids": {"observation/image": capture["camera_frame_ids"][camera_id]},
        "camera_capture_time_ns": {"observation/image": capture["camera_capture_time_ns"][camera_id]},
        "camera_timestamp_source": {"observation/image": capture["camera_timestamp_source"][camera_id]},
        "capture_receipt": file_descriptor(resolved_capture),
        "model_request_count": 0,
        "behavioral_action_count": 0,
    }
    manifest = {
        "schema_version": "wmf-n3-fixed-observation-v1",
        "observation_id": source_capture["simulator_observation_id"],
        "payload": {
            "path": payload_path.name,
            "sha256": sha256_file(payload_path),
            "arrays": {
                key: {"archive_key": archive_keys[key], **_numpy_array_identity(arrays[key])}
                for key in arrays
            },
        },
        "source_capture": source,
    }
    manifest_path = root / "observation_manifest.json"
    atomic_json(manifest_path, manifest)
    wire_sha256 = _n3_fixture_wire_sha256(fixture_path)
    receipt = sign_document({
        "schema_version": N3_LIVE_INPUT_SCHEMA,
        "study_id": STUDY_ID,
        "status": "live_zero_policy_input_prepared_generation_not_yet_run",
        "prepared_at_utc": utc_now(),
        "camera_id": camera_id,
        "capture": file_descriptor(resolved_capture),
        "source_fixture": file_descriptor(fixture_path),
        "observation_manifest": file_descriptor(manifest_path),
        "observation_payload": file_descriptor(payload_path),
        "wire_observation_sha256": wire_sha256,
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "physical_time_qualified": False,
        "claim_boundary": "Prepared a live hash-bound N3 input only; generation, physical timing and policy skill remain unqualified.",
    })
    receipt_path = root / "preparation_receipt.json"
    atomic_json(receipt_path, receipt)
    return {**receipt, "preparation_receipt": file_descriptor(receipt_path)}


def _walk_and_verify_n3_artifacts(value: Any, *, root: Path, label: str) -> None:
    if isinstance(value, list):
        for index, child in enumerate(value):
            _walk_and_verify_n3_artifacts(child, root=root, label=f"{label}[{index}]")
        return
    if not isinstance(value, Mapping):
        return
    if value.get("__type__") in {"numpy", "torch"}:
        _, path = verify_descriptor(value.get("artifact"), base=root, label=f"{label} payload", require_bytes=True)
        require(path.parent == root.resolve(), f"{label} payload escaped its manifest directory")
    for key, child in value.items():
        if key != "artifact":
            _walk_and_verify_n3_artifacts(child, root=root, label=f"{label}.{key}")


def _verify_n3_artifact(value: Any, *, base: Path, label: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    require(isinstance(value, Mapping), f"{label} descriptor is missing")
    descriptor = {"path": value.get("manifest_path"), "sha256": value.get("manifest_sha256")}
    _, path = verify_descriptor(descriptor, base=base, label=label)
    manifest = load_json(path, label)
    require(manifest.get("schema_version") == "wmf-lossless-nested-payload-v1", f"{label} manifest schema changed")
    require(manifest.get("logical_sha256") == value.get("logical_sha256"), f"{label} logical identity changed")
    _walk_and_verify_n3_artifacts(manifest.get("structure"), root=path.parent, label=label)
    return dict(value), manifest, path


def _n3_generation_evidence(qualification: Mapping[str, Any], qualification_path: Path) -> dict[str, Any]:
    compact = qualification
    raw = qualification
    raw_path = qualification_path
    if isinstance(compact.get("raw_receipt"), Mapping):
        _, raw_path = verify_descriptor(compact["raw_receipt"], base=qualification_path.parent,
                                        label="N3 raw qualification", require_bytes=True)
        raw = load_json(raw_path, "N3 raw qualification")
    for value, label in ((compact, "N3 qualification"), (raw, "N3 raw qualification")):
        require(value.get("schema_version") == N3_QUALIFICATION_SCHEMA, f"{label} schema changed")
        require(value.get("status") == "passed" and value.get("qualified") is True, f"{label} did not pass")
        require(value.get("generation_request_count") == 6 and value.get("robot_episode_count") == 0,
                f"{label} is not a six-request zero-robot-episode probe")
    require(raw.get("physical_time_mapping_qualified") is False,
            "N3 generation receipt improperly pre-claims physical timing")
    source = raw.get("source")
    checkpoint = raw.get("checkpoint")
    expected = EXPECTED_MODELS["N3"]
    require(isinstance(source, Mapping) and source.get("commit") == expected["commit"]
            and source.get("git_tree") == expected["tree"], "N3 generation source identity changed")
    require(isinstance(checkpoint, Mapping) and checkpoint.get("revision") == expected["checkpoint_revision"]
            and checkpoint.get("payload_aggregate_sha256") == expected["checkpoint_aggregate"]
            and checkpoint.get("full_payload_rehash_performed") is True,
            "N3 generation checkpoint identity changed")
    records = raw.get("requests")
    require(isinstance(records, list) and len(records) == 6, "N3 generation request inventory changed")
    decoded = [row for row in records if isinstance(row, Mapping) and row.get("decode_requested") is True]
    require(len(decoded) == 3, "N3 generation probe lacks exactly three decoded requests")
    request = decoded[0]
    require(request.get("decode_mode") == "offline_exact_retained_latent_after_action_capture",
            "N3 request decode path changed")
    require(request.get("official_joint_generation_calls") == 1
            and request.get("official_infer_decode_calls") == 0
            and request.get("offline_decode_calls") == 1,
            "N3 request call counts changed")
    require(request.get("action_captured_before_offline_decode") is True,
            "N3 action was not captured before measurement-only decode")
    returned_identity = request.get("returned_action_identity")
    latent_identity = request.get("retained_latent_identity")
    decoded_identity = request.get("decoded_future_identity")
    require(isinstance(returned_identity, Mapping) and returned_identity.get("shape") == [32, 8],
            "N3 returned action shape changed")
    require(isinstance(latent_identity, Mapping) and isinstance(latent_identity.get("shape"), list),
            "N3 retained latent identity is missing")
    require(isinstance(decoded_identity, Mapping) and len(decoded_identity.get("shape", [])) == 4
            and decoded_identity["shape"][0] == 33 and decoded_identity["shape"][-1] == 3,
            "N3 decoded future is not 33 ordered RGB frames")
    _, raw_action_manifest, _ = _verify_n3_artifact(
        request.get("raw_generated_action_artifact"), base=raw_path.parent, label="N3 raw generated action")
    raw_action_structure = raw_action_manifest.get("structure")
    require(isinstance(raw_action_structure, Mapping)
            and raw_action_structure.get("shape", [None])[0] == expected["raw_action_rows"],
            "N3 raw generated tensor does not retain the conditioning row")
    for key, label in (
        ("returned_action_artifact", "N3 returned action"),
        ("retained_latent_artifact", "N3 retained latent"),
        ("decoded_future_artifact", "N3 decoded future"),
    ):
        _verify_n3_artifact(request.get(key), base=raw_path.parent, label=label)
    fixed = raw.get("fixed_observation")
    require(isinstance(fixed, Mapping), "N3 fixed-observation receipt is missing")
    source_capture = fixed.get("source_capture")
    require(isinstance(source_capture, Mapping), "N3 fixed-observation source capture is missing")
    require(source_capture.get("provenance_kind") != "archived_policy_input",
            "N3 historical qualification cannot close live timing; run the zero-policy model on a live fixed capture")
    require(source_capture.get("provenance_kind") == "live_fixed_observation_zero_policy_input",
            "N3 generation request is not bound to the live zero-policy input preparation")
    selected_camera = source_capture.get("selected_original_camera_id")
    require(selected_camera in CAMERAS, "N3 live generation lacks its selected original camera")
    capture_desc = source_capture.get("capture_receipt")
    _, capture_path = verify_descriptor(capture_desc, base=raw_path.parent,
                                        label="N3 fixed-observation capture")
    capture = _verify_capture_receipt(capture_path, sha256_file(capture_path), model_id="N3")
    require(source_capture.get("capture_id") == capture["capture_id"]
            and source_capture.get("settled_reset_identity") == capture["settled_reset_identity"],
            "N3 request source capture identity changed")
    require(source_capture.get("camera_frame_ids") == {"observation/image": capture["camera_frame_ids"][selected_camera]}
            and source_capture.get("camera_capture_time_ns")
            == {"observation/image": capture["camera_capture_time_ns"][selected_camera]}
            and source_capture.get("camera_timestamp_source")
            == {"observation/image": capture["camera_timestamp_source"][selected_camera]},
            "N3 request packed-image clock differs from its selected original camera")
    require(fixed.get("wire_observation_sha256") == capture["model_fixture_wire_sha256"],
            "N3 live request arrays differ from the capture's official fixture")
    capture["selected_camera_id"] = selected_camera
    return {
        "qualification": file_descriptor(qualification_path),
        "raw_qualification": file_descriptor(raw_path),
        "source": {"commit": expected["commit"], "git_tree": expected["tree"]},
        "checkpoint": {
            "revision": expected["checkpoint_revision"],
            "payload_aggregate_sha256": expected["checkpoint_aggregate"],
            "full_payload_rehash_performed": True,
        },
        "selected_request": {
            "request_id": request.get("request_id"),
            "request_index": request.get("request_index"),
            "record_sha256": sha256_bytes(canonical_bytes(request)),
            "returned_action_shape": list(returned_identity["shape"]),
            "returned_action_value_sha256": returned_identity.get("value_sha256"),
            "raw_generated_action_rows": raw_action_structure["shape"][0],
            "latent_shape": list(latent_identity["shape"]),
            "latent_value_sha256": latent_identity.get("value_sha256"),
            "decoded_rgb_shape": list(decoded_identity["shape"]),
            "decoded_value_sha256": decoded_identity.get("value_sha256"),
            "returned_action_executed": False,
            "generation_robot_episode_count": 0,
        },
        "live_capture": capture,
    }


def _verified_artifact_file(value: Any, *, base: Path, label: str) -> Path:
    require(isinstance(value, Mapping), f"{label} descriptor is missing")
    descriptor = {
        "path": value.get("path"),
        "sha256": value.get("file_sha256", value.get("sha256")),
        "bytes": value.get("bytes"),
    }
    _, path = verify_descriptor(descriptor, base=base, label=label, require_bytes=True)
    return path


def _d1_generation_evidence(job: Mapping[str, Any], job_path: Path) -> dict[str, Any]:
    expected = EXPECTED_MODELS["D1"]
    require(job.get("schema_version") == D1_JOB_SCHEMA, "D1 qualification job schema changed")
    require(job.get("status") == "finished" and job.get("decision") == "qualified" and job.get("exit_code") == 0,
            "D1 qualification job did not qualify")
    require(job.get("generation_request_count") == 6 and job.get("behavioral_episode_count") == 0,
            "D1 qualification is not a six-request zero-behavioral-episode probe")
    server = job.get("server_contract")
    require(isinstance(server, Mapping), "D1 server identity is missing")
    require(server.get("official_repository_commit") == expected["commit"]
            and server.get("official_repository_tree") == expected["tree"],
            "D1 generation source identity changed")
    require(server.get("checkpoint_aggregate_sha256") == expected["checkpoint_aggregate"],
            "D1 generation checkpoint identity changed")
    require(server.get("official_action_path") == "GrootSimPolicy.lazy_joint_forward_causal"
            and server.get("custom_s2_used") is False and server.get("patched_s1_used") is False,
            "D1 generation did not use the official conditional path")
    probe = job.get("probe")
    require(isinstance(probe, Mapping) and probe.get("passed") is True and probe.get("status") == "passed",
            "D1 six-request report did not pass")
    _, report_path = verify_descriptor(probe.get("artifact"), base=job_path.parent,
                                       label="D1 six-request report", require_bytes=True)
    report = load_json(report_path, "D1 six-request report")
    require(report.get("schema_version") == D1_REPORT_SCHEMA and report.get("passed") is True
            and report.get("status") == "passed", "D1 six-request report schema/status changed")
    require(report.get("generation_request_count") == 6 and report.get("behavioral_episode_count") == 0,
            "D1 six-request report contains behavioral activity")
    records = report.get("records")
    require(isinstance(records, Mapping), "D1 compact request records are missing")
    decoded_ids = sorted(
        key for key, value in records.items()
        if isinstance(value, Mapping) and value.get("offline_decode_performed") is True
    )
    require(len(decoded_ids) == 3, "D1 report lacks exactly three decoded requests")
    request_id = decoded_ids[0]
    compact = records[request_id]
    manifest_desc = {
        "path": compact.get("episode_manifest"),
        "sha256": compact.get("episode_manifest_sha256"),
    }
    _, manifest_path = verify_descriptor(manifest_desc, base=report_path.parent,
                                         label="D1 decoded episode manifest")
    manifest = load_json(manifest_path, "D1 decoded episode manifest")
    require(manifest.get("schema_version") == "wmf-d1-episode-manifest-v1"
            and manifest.get("status") == "complete" and manifest.get("request_count") == 1,
            "D1 decoded episode manifest is incomplete")
    requests = manifest.get("requests")
    require(isinstance(requests, list) and len(requests) == 1, "D1 decoded episode request inventory changed")
    request = requests[0]
    require(isinstance(request, Mapping) and request.get("schema_version") == D1_REQUEST_SCHEMA,
            "D1 decoded request schema changed")
    require(request.get("official_action_path") == "GrootSimPolicy.lazy_joint_forward_causal"
            and request.get("custom_s2_used") is False and request.get("patched_s1_used") is False,
            "D1 decoded request path changed")
    action = request.get("official_returned_action")
    latent = request.get("latent_video")
    decode = request.get("offline_decode")
    require(isinstance(action, Mapping) and action.get("shape") == [24, 8], "D1 returned action shape changed")
    require(isinstance(latent, Mapping) and len(latent.get("shape", [])) == 5
            and latent["shape"][2] == 3, "D1 returned latent does not contain conditioning plus two future frames")
    require(isinstance(decode, Mapping) and decode.get("requested") is True and decode.get("performed") is True,
            "D1 offline measurement decode was not performed")
    decoded_rgb = decode.get("decoded_rgb")
    decoded_tensor = decode.get("decoded_tensor")
    require(isinstance(decoded_rgb, Mapping) and decoded_rgb.get("shape", [None])[0] == 9
            and len(decoded_rgb.get("shape", [])) == 4 and decoded_rgb["shape"][-1] == 3,
            "D1 decoded RGB does not contain nine ordered frames")
    require(isinstance(decoded_tensor, Mapping) and len(decoded_tensor.get("shape", [])) == 5
            and decoded_tensor["shape"][2] == 9, "D1 decoded tensor temporal shape changed")
    require(decode.get("latent_data_sha256_before") == latent.get("data_sha256")
            and decode.get("latent_data_sha256_after") == latent.get("data_sha256"),
            "D1 measurement decode changed or did not bind the retained latent")
    for value, label in ((action, "D1 returned action"), (latent, "D1 retained latent"),
                         (decoded_rgb, "D1 decoded RGB"), (decoded_tensor, "D1 decoded tensor")):
        _verified_artifact_file(value, base=manifest_path.parent, label=label)
    capture_validation = job.get("capture_manifest")
    require(isinstance(capture_validation, Mapping)
            and capture_validation.get("schema_version") == "wmf-d1-capture-manifest-validation-v1"
            and capture_validation.get("status") == "passed",
            "D1 fixed-observation validation did not pass")
    capture_desc = capture_validation.get("manifest")
    _, capture_path = verify_descriptor(capture_desc, base=job_path.parent,
                                        label="D1 live capture", require_bytes=True)
    capture = _verify_capture_receipt(capture_path, sha256_file(capture_path), model_id="D1")
    require(capture.get("capture_id") == capture_validation.get("capture_id"),
            "D1 live capture identity differs from its validation")
    return {
        "qualification": file_descriptor(job_path),
        "raw_qualification": file_descriptor(report_path),
        "source": {"commit": expected["commit"], "git_tree": expected["tree"]},
        "checkpoint": {
            "revision": expected["checkpoint_revision"],
            "payload_aggregate_sha256": expected["checkpoint_aggregate"],
            "full_payload_rehash_performed": True,
        },
        "selected_request": {
            "request_id": request_id,
            "request_index": request.get("request_index"),
            "record_sha256": sha256_bytes(canonical_bytes(request)),
            "episode_manifest": file_descriptor(manifest_path),
            "returned_action_shape": list(action["shape"]),
            "returned_action_value_sha256": action.get("data_sha256"),
            "latent_shape": list(latent["shape"]),
            "latent_value_sha256": latent.get("data_sha256"),
            "decoded_rgb_shape": list(decoded_rgb["shape"]),
            "decoded_value_sha256": decoded_rgb.get("data_sha256"),
            "returned_action_executed": False,
            "generation_robot_episode_count": 0,
        },
        "live_capture": capture,
    }


def normalize_generation_probe(
    *,
    model_id: str,
    qualification_path: Path,
    qualification_sha256: str,
    contract_path: Path,
    contract_sha256: str,
) -> dict[str, Any]:
    contract, resolved_contract = load_contract(contract_path, contract_sha256)
    qualification, resolved = _read_hashed_json(
        qualification_path, qualification_sha256, f"{model_id} generation qualification"
    )
    require(model_id in EXPECTED_MODELS, f"unsupported model: {model_id}")
    evidence = (
        _n3_generation_evidence(qualification, resolved)
        if model_id == "N3"
        else _d1_generation_evidence(qualification, resolved)
    )
    expected = EXPECTED_MODELS[model_id]
    selected = evidence["selected_request"]
    require(selected["decoded_rgb_shape"][0] == expected["decoded_frames"],
            f"{model_id} decoded frame count differs from source lineage")
    require(selected["returned_action_shape"] == [expected["returned_actions"], expected["action_dimensions"]],
            f"{model_id} returned action tensor differs from source lineage")
    result = sign_document({
        "schema_version": GENERATION_PROBE_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": model_id,
        "status": "live_generation_shape_passed_physical_clock_probe_required",
        "normalized_at_utc": utc_now(),
        "contract": file_descriptor(resolved_contract),
        **evidence,
        "inference_tensor_witness_expected": dict(contract["models"][model_id]["lineage"]["inference_tensor_witness"]),
        "model_returned_action_executed": False,
        "model_output_or_action_modified": False,
        "physical_time_qualified": False,
        "prohibited_inferences_used": [],
        "claim_boundary": "One live generation request and its exact tensor/decode identities; no model action was executed and no physical-time mapping is yet claimed.",
    })
    return result


def validate_generation_probe(
    path: Path,
    expected_sha256: str,
    *,
    expected_model: str | None = None,
) -> dict[str, Any]:
    value, resolved = _read_hashed_json(path, expected_sha256, "zero-policy generation probe")
    require(value.get("schema_version") == GENERATION_PROBE_SCHEMA, "generation probe schema changed")
    verify_signed(value, "zero-policy generation probe")
    model_id = value.get("model_id")
    require(model_id in EXPECTED_MODELS, "generation probe model is invalid")
    if expected_model is not None:
        require(model_id == expected_model, "generation probe model mismatch")
    require(value.get("study_id") == STUDY_ID, "generation probe study changed")
    require(value.get("status") == "live_generation_shape_passed_physical_clock_probe_required",
            "generation probe did not pass")
    require(value.get("model_returned_action_executed") is False
            and value.get("model_output_or_action_modified") is False,
            "generation probe executed or modified model output")
    require(value.get("physical_time_qualified") is False, "generation-only probe pre-claims physical timing")
    require(value.get("prohibited_inferences_used") == [], "generation probe used a prohibited inference")
    expected = EXPECTED_MODELS[model_id]
    source = value.get("source")
    checkpoint = value.get("checkpoint")
    selected = value.get("selected_request")
    require(isinstance(source, Mapping) and source.get("commit") == expected["commit"]
            and source.get("git_tree") == expected["tree"], "generation probe source pin changed")
    require(isinstance(checkpoint, Mapping) and checkpoint.get("revision") == expected["checkpoint_revision"]
            and checkpoint.get("payload_aggregate_sha256") == expected["checkpoint_aggregate"]
            and checkpoint.get("full_payload_rehash_performed") is True,
            "generation probe checkpoint identity changed")
    require(isinstance(selected, Mapping), "generation probe request identity is missing")
    require(selected.get("returned_action_shape") == [expected["returned_actions"], expected["action_dimensions"]],
            "generation probe returned action shape changed")
    require(selected.get("decoded_rgb_shape", [None])[0] == expected["decoded_frames"],
            "generation probe decoded frame count changed")
    require(selected.get("returned_action_executed") is False
            and selected.get("generation_robot_episode_count") == 0,
            "generation request executed a model action")
    for field in ("returned_action_value_sha256", "latent_value_sha256", "decoded_value_sha256", "record_sha256"):
        require(_valid_sha(selected.get(field)), f"generation request lacks {field}")
    qualification = value.get("qualification")
    _, qualification_path = verify_descriptor(
        qualification, base=resolved.parent, label="generation qualification", require_bytes=True
    )
    qualification_value = load_json(qualification_path, "generation qualification")
    regenerated = (
        _n3_generation_evidence(qualification_value, qualification_path)
        if model_id == "N3"
        else _d1_generation_evidence(qualification_value, qualification_path)
    )
    for key in ("qualification", "raw_qualification", "source", "checkpoint", "selected_request", "live_capture"):
        require(value.get(key) == regenerated[key],
                f"generation probe {key} differs from the underlying live qualification")
    live = value.get("live_capture")
    require(isinstance(live, Mapping), "generation probe lacks a live original-camera capture")
    capture_desc = live.get("capture")
    _, capture_path = verify_descriptor(capture_desc, base=resolved.parent,
                                        label="generation live capture", require_bytes=True)
    current_capture = _verify_capture_receipt(capture_path, sha256_file(capture_path), model_id=model_id)
    require(current_capture["capture_id"] == live.get("capture_id"), "generation live capture identity changed")
    return value


def _verify_journal(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for sequence, line in enumerate(handle):
                require(bool(line.strip()), "recorder journal contains a blank line")
                row = json.loads(line)
                require(isinstance(row, dict), "recorder journal row is not an object")
                require(row.get("sequence") == sequence, "recorder journal sequence changed")
                require(row.get("previous_event_sha256") == previous, "recorder journal hash chain changed")
                observed = row.get("event_sha256")
                require(_valid_sha(observed), "recorder journal event hash is invalid")
                unsigned = dict(row)
                unsigned.pop("event_sha256")
                require(sha256_bytes(canonical_bytes(unsigned, ensure_ascii=True)) == observed,
                        "recorder journal event content changed")
                rows.append(row)
                previous = observed
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TimingQualificationError(f"recorder journal is unreadable: {path}") from error
    require(rows, "recorder journal is empty")
    return rows, str(previous)


def _event_payloads(rows: Sequence[Mapping[str, Any]], kind: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row.get("kind") == kind:
            payload = row.get("payload")
            require(isinstance(payload, Mapping), f"recorder {kind} event payload is invalid")
            output.append(dict(payload))
    return output


def _mapping_item(structure: Any, key: str) -> Any:
    require(isinstance(structure, Mapping) and structure.get("__type__") == "mapping",
            "recording payload structure is not a mapping")
    items = structure.get("items")
    require(isinstance(items, Mapping) and key in items, f"recording payload lacks {key}")
    return items[key]


def _array_nodes(value: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if value.get("__type__") == "ndarray":
            output.append(dict(value))
        for child in value.values():
            output.extend(_array_nodes(child))
    elif isinstance(value, list):
        for child in value:
            output.extend(_array_nodes(child))
    return output


def _thaw_scalar_structure(value: Any) -> Any:
    if not isinstance(value, Mapping) or "__type__" not in value:
        return value
    kind = value["__type__"]
    require(kind != "ndarray", "clock receipt unexpectedly contains an array")
    if kind == "mapping":
        items = value.get("items")
        require(isinstance(items, Mapping), "stored scalar mapping is invalid")
        return {key: _thaw_scalar_structure(child) for key, child in items.items()}
    if kind in {"list", "tuple"}:
        items = value.get("items")
        require(isinstance(items, list), "stored scalar sequence is invalid")
        output = [_thaw_scalar_structure(child) for child in items]
        return output if kind == "list" else tuple(output)
    if kind == "path":
        require(isinstance(value.get("value"), str), "stored scalar path is invalid")
        return value["value"]
    raise TimingQualificationError(f"unsupported stored scalar type: {kind}")


def _verify_observation_payload(
    attempt_root: Path,
    descriptor: Any,
    *,
    camera_id: str,
    expected_clock: Mapping[str, Any],
) -> None:
    require(isinstance(descriptor, Mapping), "observation payload descriptor is missing")
    observed = descriptor.get("payload_sha256")
    require(_valid_sha(observed), "observation payload descriptor hash is invalid")
    unsigned = dict(descriptor)
    unsigned.pop("payload_sha256")
    require(sha256_bytes(canonical_bytes(unsigned, ensure_ascii=True)) == observed,
            "observation payload descriptor changed")
    require(type(descriptor.get("array_count")) is int and descriptor["array_count"] > 0,
            "observation payload contains no original arrays")
    artifact = descriptor.get("artifact")
    _, path = verify_descriptor(artifact, base=attempt_root, label="observation NPZ", require_bytes=True)
    require(path.is_relative_to(attempt_root.resolve()), "observation payload escaped recorder attempt")
    structure = descriptor.get("structure")
    stored_clock = _mapping_item(structure, "clock")
    require(_thaw_scalar_structure(stored_clock) == dict(expected_clock),
            "observation payload clock differs from its journal event")
    image_obs = _mapping_item(structure, "image_obs")
    camera = _mapping_item(image_obs, camera_id)
    require(isinstance(camera, Mapping) and camera.get("__type__") == "ndarray",
            f"original camera payload is not retained: {camera_id}")
    require(isinstance(camera.get("shape"), list) and len(camera["shape"]) == 3
            and camera["shape"][-1] in (3, 4), f"original camera payload shape changed: {camera_id}")
    nodes = _array_nodes(structure)
    keys = [node.get("key") for node in nodes]
    require(len(nodes) == descriptor["array_count"] and len(set(keys)) == len(keys)
            and all(isinstance(key, str) and key for key in keys),
            "observation array manifest is incomplete or duplicated")
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == set(keys), "observation NPZ members differ from its manifest")
            for node in nodes:
                array = archive[node["key"]]
                require(list(array.shape) == node.get("shape") and array.dtype.str == node.get("dtype"),
                        f"observation array identity changed: {node['key']}")
    except TimingQualificationError:
        raise
    except Exception as error:
        raise TimingQualificationError(f"observation NPZ is not a safe readable array archive: {path}") from error


def _verify_model_response_request_binding(
    attempt_root: Path,
    descriptor: Any,
    *,
    model_id: str,
    request_path: Path,
    cell_id: str,
    request_index: int,
    action_step_start: int,
) -> dict[str, Any]:
    """Reopen the recorder response payload that transported one server receipt.

    The immutable D1 request receipt intentionally has no cell identifier.  The
    recording adapter nevertheless retained the augmented wire response, which
    includes the exact server-request descriptor and context identifiers.  This
    check prevents a same-index request from another cell/attempt being paired
    with otherwise valid native clock evidence.  N3 is checked through the same
    path so both models use one model-invariant binding rule.
    """

    require(isinstance(descriptor, Mapping), "model response payload descriptor is missing")
    observed = descriptor.get("payload_sha256")
    require(_valid_sha(observed), "model response payload descriptor hash is invalid")
    unsigned = dict(descriptor)
    unsigned.pop("payload_sha256")
    require(sha256_bytes(canonical_bytes(unsigned, ensure_ascii=True)) == observed,
            "model response payload descriptor changed")
    require(descriptor.get("role") == "model_response",
            "model response payload role changed")
    array_count = descriptor.get("array_count")
    require(type(array_count) is int and array_count >= 0,
            "model response payload array count is invalid")
    artifact = descriptor.get("artifact")
    if array_count:
        _, artifact_path = verify_descriptor(
            artifact, base=attempt_root, label="model response NPZ", require_bytes=True
        )
        require(artifact_path.is_relative_to(attempt_root.resolve()),
                "model response payload escaped recorder attempt")
    else:
        require(artifact is None, "array-free model response unexpectedly names an artifact")

    raw_response = _mapping_item(descriptor.get("structure"), "raw_response")
    transported_descriptor = _thaw_scalar_structure(
        _mapping_item(raw_response, "wmf_server_request_receipt")
    )
    require_descriptor_matches(
        transported_descriptor,
        request_path,
        base=attempt_root,
        label="model response transported request receipt",
    )
    transported_index = _thaw_scalar_structure(
        _mapping_item(raw_response, "wmf_request_index")
    )
    require(transported_index == request_index,
            "model response transported request index changed")
    if model_id == "N3":
        require(
            _thaw_scalar_structure(_mapping_item(raw_response, "wmf_cell_id")) == cell_id
            and _thaw_scalar_structure(_mapping_item(raw_response, "wmf_action_step_start"))
            == action_step_start,
            "N3 model response cell/action binding changed",
        )
        context = _thaw_scalar_structure(
            _mapping_item(raw_response, "wmf_server_context_id")
        )
        require(isinstance(context, str) and context,
                "N3 model response context is missing")
        return {
            "payload_sha256": observed,
            "server_context_id": context,
            "transported_request_receipt": file_descriptor(request_path),
        }

    episode_id = _thaw_scalar_structure(
        _mapping_item(raw_response, "wmf_episode_context_id")
    )
    require(isinstance(episode_id, str) and episode_id
            and episode_id == load_json(request_path, "D1 transported request").get("episode_id"),
            "D1 model response episode context changed")
    return {
        "payload_sha256": observed,
        "episode_context_id": episode_id,
        "transported_request_receipt": file_descriptor(request_path),
    }


def _unwrap_recorder_receipt(value: Mapping[str, Any], path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    if value.get("schema_version") == RECORDER_QUEUE_SCHEMA:
        require(value.get("status") == "passed" and value.get("exit_code") == 0,
                "recorder queue job did not pass")
        _, child_path = verify_descriptor(value.get("raw_child_receipt"), base=path.parent,
                                          label="raw recorder child", require_bytes=True)
        child = load_json(child_path, "raw recorder child")
        return child, child_path, {"queue_receipt": file_descriptor(path)}
    require(value.get("schema_version") == RECORDER_CHILD_SCHEMA, "recorder receipt schema changed")
    return dict(value), path, {}


def _positive_intervals(values: Sequence[float], label: str) -> list[float]:
    output = []
    for left, right in zip(values, values[1:]):
        delta = right - left
        require(delta >= 0, f"{label} clock moved backward")
        if delta > 0:
            output.append(delta)
    require(output, f"{label} clock has no positive interval")
    return output


def _load_recorder_clock_trace(
    receipt_path: Path,
    expected_sha256: str,
    *,
    camera_id: str,
    needed_boundaries: Sequence[int],
) -> dict[str, Any]:
    value, path = _read_hashed_json(receipt_path, expected_sha256, "recorder qualification receipt")
    child, child_path, outer = _unwrap_recorder_receipt(value, path)
    require(child.get("schema_version") == RECORDER_CHILD_SCHEMA and child.get("status") == "passed",
            "recorder child did not pass")
    require(child.get("actions_executed") == 450 and child.get("observation_count") == 451,
            "recorder child did not retain the fixed 450-action/451-observation trace")
    require(child.get("model_request_count") == 0 and child.get("behavioral_episode_count") == 0
            and child.get("recording_qualification_count") == 1 and child.get("model_attached") is False,
            "recorder clock trace is not zero-policy/no-model evidence")
    require(child.get("action_source") == "joint_position_hold_from_each_preceding_original_proprioception",
            "recorder action source changed")
    _, completion_path = verify_descriptor(child.get("adapter_completion"), base=child_path.parent,
                                            label="recorder completion", require_bytes=True)
    completion = load_json(completion_path, "recorder completion")
    require(completion.get("schema_version") == RECORDER_ATTEMPT_SCHEMA
            and completion.get("recording_qualification_valid") is True
            and completion.get("behavioral_result_valid") is False
            and completion.get("model_attached") is False,
            "recorder completion is not valid recorder-only evidence")
    require(completion.get("stop_reason") == "action_cap" and completion.get("actions_executed") == 450
            and completion.get("observation_count") == 451 and completion.get("request_count") == 0,
            "recorder completion counts changed")
    _, journal_path = verify_descriptor(child.get("adapter_journal"), base=child_path.parent,
                                         label="recorder journal", require_bytes=True)
    rows, tail = _verify_journal(journal_path)
    journal_desc = child["adapter_journal"]
    require(journal_desc.get("event_count") == len(rows) and journal_desc.get("tail_sha256") == tail,
            "recorder child journal summary changed")
    require(completion.get("event_count") == len(rows) and completion.get("journal_tail_sha256") == tail,
            "recorder completion does not bind the full journal")
    forbidden = {
        "model_request_packed", "model_request_sent", "model_response_received",
        "model_request_completed", "policy_action_returned",
    }
    require(not any(row.get("kind") in forbidden for row in rows),
            "zero-policy recorder journal contains model/policy activity")
    proposed = _event_payloads(rows, "recording_qualification_action_proposed")
    started = _event_payloads(rows, "environment_step_started")
    completed = _event_payloads(rows, "environment_step_completed")
    observations = _event_payloads(rows, "observation_captured")
    require(len(proposed) == len(started) == len(completed) == 450,
            "recorder journal does not contain 450 exact hold-step triplets")
    require(len(observations) == 451, "recorder journal does not contain 451 observations")
    for index, (proposal, start, finish) in enumerate(zip(proposed, started, completed), start=1):
        require(proposal.get("action_step") == index and proposal.get("request_index") is None
                and proposal.get("chunk_offset") is None,
                "recorder hold action order/request binding changed")
        source = proposal.get("action_source")
        require(isinstance(source, Mapping) and source.get("kind") == "joint_position_hold"
                and source.get("policy_model") is None and source.get("model_request") is False,
                "recorder executed a non-hold or policy-provided action")
        require(source.get("source_observation_id") == f"obs_{index - 1:06d}",
                "recorder hold action is not derived from the immediately preceding proprioception")
        require(source.get("action_step") == index
                and source.get("scientific_claim") == "recorder_qualification_only_nonbehavioral",
                "recorder hold provenance changed")
        action_identity = proposal.get("action_identity")
        require(isinstance(action_identity, Mapping)
                and action_identity.get("shape") == [8]
                and action_identity.get("order") == "C"
                and isinstance(action_identity.get("dtype"), str)
                and _valid_sha(action_identity.get("value_sha256")),
                "recorder proposed-action identity is incomplete")
        require(start.get("action_step") == index and start.get("request_index") is None
                and start.get("chunk_offset") is None
                and start.get("executed_action_identity") == action_identity,
                "recorder executed-action identity differs from its exact hold proposal")
        begin_ns = start.get("env_step_start_monotonic_ns")
        require(type(begin_ns) is int and begin_ns >= 0, "recorder environment-step start clock is invalid")
        require(finish.get("action_step") == index and finish.get("request_index") is None
                and finish.get("chunk_offset") is None
                and finish.get("env_step_start_monotonic_ns") == begin_ns,
                "recorder environment-step completion differs from its start")
        require(type(finish.get("env_step_end_monotonic_ns")) is int
                and finish["env_step_end_monotonic_ns"] >= begin_ns,
                "recorder environment-step completion clock is invalid")
        require(type(finish.get("terminated")) is bool and type(finish.get("truncated")) is bool,
                "recorder environment-step termination flags are invalid")
    event_positions: dict[str, dict[int, int]] = {
        "recording_qualification_action_proposed": {},
        "environment_step_started": {},
        "environment_step_completed": {},
        "observation_captured": {},
    }
    for event in rows:
        kind = event.get("kind")
        if kind not in event_positions:
            continue
        payload = event["payload"]
        step = payload.get("control_step") if kind == "observation_captured" else payload.get("action_step")
        require(type(step) is int and step not in event_positions[kind],
                f"recorder {kind} step identity is invalid or duplicated")
        event_positions[kind][step] = event["sequence"]
    require(event_positions["observation_captured"].keys() == set(range(451)),
            "recorder observation event positions are incomplete")
    for kind in (
        "recording_qualification_action_proposed",
        "environment_step_started",
        "environment_step_completed",
    ):
        require(event_positions[kind].keys() == set(range(1, 451)),
                f"recorder {kind} event positions are incomplete")
    for index in range(1, 451):
        require(
            event_positions["observation_captured"][index - 1]
            < event_positions["recording_qualification_action_proposed"][index]
            < event_positions["environment_step_started"][index]
            < event_positions["environment_step_completed"][index]
            < event_positions["observation_captured"][index],
            "recorder hold-step event order changed",
        )
    by_boundary: dict[int, dict[str, Any]] = {}
    for expected_boundary, observation in enumerate(observations):
        require(observation.get("control_step") == expected_boundary
                and observation.get("observation_id") == f"obs_{expected_boundary:06d}",
                "recorder observation/control boundary order changed")
        clock = observation.get("clock")
        require(isinstance(clock, Mapping) and clock.get("control_step") == expected_boundary,
                "recorder native control identity changed")
        require(type(clock.get("physics_step")) is int and clock["physics_step"] >= 0,
                "recorder physics-step identity is missing")
        require(type(clock.get("physics_time_s")) in (int, float)
                and math.isfinite(float(clock["physics_time_s"])), "recorder physics clock is missing")
        cameras = clock.get("cameras")
        camera = cameras.get(camera_id) if isinstance(cameras, Mapping) else None
        require(isinstance(camera, Mapping), f"recorder lacks original camera {camera_id}")
        require(camera.get("frame_id") is not None, "recorder camera frame identity is missing")
        require(type(camera.get("capture_time_ns")) is int and camera["capture_time_ns"] >= 0,
                "recorder camera capture timestamp is missing")
        require(isinstance(camera.get("timestamp_source"), str) and camera["timestamp_source"],
                "recorder camera timestamp source is missing")
        by_boundary[expected_boundary] = {
            "observation_id": observation["observation_id"],
            "artifact": observation.get("artifact"),
            "raw_clock": dict(clock),
            "physics_step": clock["physics_step"],
            "physics_time_s": float(clock["physics_time_s"]),
            "camera_frame_id": camera["frame_id"],
            "camera_capture_time_ns": camera["capture_time_ns"],
            "camera_timestamp_source": camera["timestamp_source"],
        }
    max_needed = max(needed_boundaries)
    require(max_needed <= 450, "source mapping exceeds recorder trace")
    trace = [by_boundary[index] for index in range(max_needed + 1)]
    physics_steps = [row["physics_step"] for row in trace]
    require(all(right > left for left, right in zip(physics_steps, physics_steps[1:])),
            "native physics steps are not strictly increasing")
    physics_times = [row["physics_time_s"] for row in trace]
    camera_times_s = [row["camera_capture_time_ns"] / 1e9 for row in trace]
    physics_intervals = _positive_intervals(physics_times, "physics")
    camera_intervals = _positive_intervals(camera_times_s, "camera")
    tolerance = min(min(physics_intervals) / 2.0, min(camera_intervals) / 2.0)
    require(math.isfinite(tolerance) and tolerance > 0, "native timestamp tolerance is invalid")
    for boundary in sorted(set(needed_boundaries)):
        _verify_observation_payload(
            completion_path.parent,
            by_boundary[boundary]["artifact"],
            camera_id=camera_id,
            expected_clock=by_boundary[boundary]["raw_clock"],
        )
    return {
        **outer,
        "child_receipt": file_descriptor(child_path),
        "completion": file_descriptor(completion_path),
        "journal": {**file_descriptor(journal_path), "event_count": len(rows), "tail_sha256": tail},
        "camera_id": camera_id,
        "observations_by_boundary": by_boundary,
        "minimum_positive_physics_control_interval_s": min(physics_intervals),
        "minimum_positive_camera_capture_interval_s": min(camera_intervals),
        "timestamp_tolerance_s": tolerance,
    }


def qualify_timing(
    *,
    model_id: str,
    contract_path: Path,
    contract_sha256: str,
    source_audit_path: Path,
    source_audit_sha256: str,
    generation_probe_path: Path,
    generation_probe_sha256: str,
    recorder_receipt_path: Path,
    recorder_receipt_sha256: str,
    camera_id: str,
) -> dict[str, Any]:
    contract, resolved_contract = load_contract(contract_path, contract_sha256)
    require(model_id in EXPECTED_MODELS, f"unsupported model: {model_id}")
    require(camera_id in CAMERAS, "selected camera is not an original recorder camera")
    source_audit = validate_source_audit(
        source_audit_path, source_audit_sha256, expected_model=model_id, contract=contract
    )
    generation = validate_generation_probe(generation_probe_path, generation_probe_sha256,
                                           expected_model=model_id)
    require_contract_descriptor_matches(
        source_audit.get("contract"), resolved_contract,
        base=Path(source_audit_path).resolve().parent, label="source-audit timing contract",
    )
    require_contract_descriptor_matches(
        generation.get("contract"), resolved_contract,
        base=Path(generation_probe_path).resolve().parent, label="generation-probe timing contract",
    )
    expected = EXPECTED_MODELS[model_id]
    require(source_audit["source"]["commit"] == generation["source"]["commit"],
            "source audit and live generation use different commits")
    if model_id == "N3":
        require(generation["live_capture"].get("selected_camera_id") == camera_id,
                "N3 timing camera differs from the live request's packed-image clock binding")
    require(source_audit["checkpoint_expected"]["revision"] == generation["checkpoint"]["revision"]
            and source_audit["checkpoint_expected"]["payload_aggregate_sha256"]
            == generation["checkpoint"]["payload_aggregate_sha256"],
            "source audit and live generation use different checkpoints")
    source_mapping = source_audit["source_mapping"]
    eligible = [
        row for row in source_mapping
        if row["generated_frame_index"] > 0
        and row["executed_control_boundary"] > 0
        and row["executed_control_boundary"] <= expected["prefix"]
    ]
    require(eligible, f"{model_id} has no positive source-mapped target inside its executed prefix")
    required_boundaries = [0, *[row["executed_control_boundary"] for row in eligible]]
    clock = _load_recorder_clock_trace(
        recorder_receipt_path,
        recorder_receipt_sha256,
        camera_id=camera_id,
        needed_boundaries=required_boundaries,
    )
    observations = clock.pop("observations_by_boundary")
    initial = observations[0]
    tolerance = clock["timestamp_tolerance_s"]
    rows: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for source_row in eligible:
        boundary = source_row["executed_control_boundary"]
        observed = observations[boundary]
        camera_elapsed = (observed["camera_capture_time_ns"] - initial["camera_capture_time_ns"]) / 1e9
        physics_elapsed = observed["physics_time_s"] - initial["physics_time_s"]
        residual = abs(camera_elapsed - physics_elapsed)
        require(camera_elapsed > 0 and physics_elapsed > 0,
                f"{model_id} boundary {boundary} has no positive physical time")
        require(observed["camera_frame_id"] != initial["camera_frame_id"],
                f"{model_id} boundary {boundary} reused the initial camera frame")
        require(residual <= tolerance + 1e-12,
                f"{model_id} boundary {boundary} camera/physics residual exceeds tolerance")
        targets.append({
            "generated_frame_index": source_row["generated_frame_index"],
            "target_physical_time_s": camera_elapsed,
        })
        rows.append({
            **source_row,
            "observation_id": observed["observation_id"],
            "camera_frame_id": observed["camera_frame_id"],
            "camera_capture_time_ns": observed["camera_capture_time_ns"],
            "physics_step": observed["physics_step"],
            "physics_time_s": observed["physics_time_s"],
            "camera_elapsed_s": camera_elapsed,
            "physics_elapsed_s": physics_elapsed,
            "cross_clock_residual_s": residual,
            "within_timestamp_tolerance": True,
        })
    target_times = [row["target_physical_time_s"] for row in targets]
    require(target_times == sorted(set(target_times)), "qualified physical target times are not strictly increasing")
    source_path = Path(source_audit_path).resolve()
    generation_path = Path(generation_probe_path).resolve()
    result = sign_document({
        "schema_version": AUTHORITY_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": model_id,
        "status": "qualified_from_source_lineage_and_native_zero_policy_clocks",
        "qualified_at_utc": utc_now(),
        "contract": file_descriptor(resolved_contract),
        "source_lineage_audit": file_descriptor(source_path),
        "generation_probe": file_descriptor(generation_path),
        "source_mapping_sha256": source_audit["source_mapping_sha256"],
        "checkpoint_revision": expected["checkpoint_revision"],
        "checkpoint_aggregate_sha256": expected["checkpoint_aggregate"],
        "selected_generation_request": dict(generation["selected_request"]),
        "generation_capture": dict(generation["live_capture"]),
        "zero_policy_clock_probe": clock,
        "camera_id": camera_id,
        "clock_bridge": "elapsed physical seconds from request current original-camera capture",
        "native_runtime_field": NATIVE_RUNTIME_FIELD,
        "returned_action_horizon": expected["returned_actions"],
        "unchanged_executed_prefix_horizon": expected["prefix"],
        "qualified_mapping_rows": rows,
        "generated_targets": targets,
        "timestamp_tolerance_s": tolerance,
        "mapping_did_not_execute_model_returned_actions": True,
        "presentation_video_fps_used": False,
        "conditioning_fps_used_as_target_timing": False,
        "generated_frame_index_interpreted_as_action_index": False,
        "dreamzero_action_block_ratio_used_as_mapping": False,
        "claim_boundary": "Qualifies decoded-frame targets against native physical clocks for later development receipt metadata. It is not a policy episode or forecast-accuracy result.",
    })
    return result


def validate_timing_authority(
    path: Path,
    expected_sha256: str,
    *,
    expected_model: str | None = None,
) -> dict[str, Any]:
    value, resolved = _read_hashed_json(path, expected_sha256, "forecast timing authority")
    require(value.get("schema_version") == AUTHORITY_SCHEMA, "forecast timing authority schema changed")
    verify_signed(value, "forecast timing authority")
    model_id = value.get("model_id")
    require(model_id in EXPECTED_MODELS, "forecast timing authority model is invalid")
    if expected_model is not None:
        require(model_id == expected_model, "forecast timing authority model mismatch")
    expected = EXPECTED_MODELS[model_id]
    contract_descriptor = value.get("contract")
    contract_identity, contract_path = verify_descriptor(
        contract_descriptor, base=resolved.parent, label="authority timing contract", require_bytes=True
    )
    contract, _ = load_contract(contract_path, contract_identity["sha256"])
    source_descriptor = value.get("source_lineage_audit")
    source_identity, source_path = verify_descriptor(
        source_descriptor, base=resolved.parent, label="authority source audit", require_bytes=True
    )
    source_audit = validate_source_audit(
        source_path, source_identity["sha256"], expected_model=model_id, contract=contract
    )
    generation_descriptor = value.get("generation_probe")
    generation_identity, generation_path = verify_descriptor(
        generation_descriptor, base=resolved.parent, label="authority generation probe", require_bytes=True
    )
    generation = validate_generation_probe(
        generation_path, generation_identity["sha256"], expected_model=model_id
    )
    require_contract_descriptor_matches(
        source_audit.get("contract"), contract_path,
        base=source_path.parent, label="authority source-audit timing contract",
    )
    require_contract_descriptor_matches(
        generation.get("contract"), contract_path,
        base=generation_path.parent, label="authority generation-probe timing contract",
    )
    require(value.get("study_id") == STUDY_ID, "forecast timing authority study changed")
    require(value.get("status") == "qualified_from_source_lineage_and_native_zero_policy_clocks",
            "forecast timing authority did not qualify")
    require(value.get("checkpoint_revision") == expected["checkpoint_revision"]
            and value.get("checkpoint_aggregate_sha256") == expected["checkpoint_aggregate"],
            "forecast timing authority checkpoint changed")
    require(value.get("returned_action_horizon") == expected["returned_actions"]
            and value.get("unchanged_executed_prefix_horizon") == expected["prefix"],
            "forecast timing authority action contract changed")
    require(value.get("camera_id") in CAMERAS, "forecast timing authority camera is not original")
    require(value.get("native_runtime_field") == NATIVE_RUNTIME_FIELD, "forecast timing metadata field changed")
    require(value.get("source_mapping_sha256") == source_audit["source_mapping_sha256"],
            "forecast timing authority source mapping changed")
    require(value.get("selected_generation_request") == generation["selected_request"]
            and value.get("generation_capture") == generation["live_capture"],
            "forecast timing authority generation witness changed")
    require(value.get("mapping_did_not_execute_model_returned_actions") is True,
            "forecast timing authority executed model actions")
    for field in (
        "presentation_video_fps_used",
        "conditioning_fps_used_as_target_timing",
        "generated_frame_index_interpreted_as_action_index",
        "dreamzero_action_block_ratio_used_as_mapping",
    ):
        require(value.get(field) is False, f"forecast timing authority used prohibited inference: {field}")
    rows = value.get("qualified_mapping_rows")
    targets = value.get("generated_targets")
    require(isinstance(rows, list) and isinstance(targets, list) and rows and targets,
            "forecast timing authority has no targets")
    expected_pairs = set(zip(expected["decoded_frame_indices"], expected["control_boundaries"]))
    eligible_pairs = [
        pair for pair in zip(expected["decoded_frame_indices"], expected["control_boundaries"])
        if pair[0] > 0 and 0 < pair[1] <= expected["prefix"]
    ]
    require([(row.get("generated_frame_index"), row.get("executed_control_boundary"))
             for row in rows if isinstance(row, Mapping)] == eligible_pairs,
            "forecast timing authority omitted, duplicated, or reordered a source-proven eligible target")
    tolerance = value.get("timestamp_tolerance_s")
    require(type(tolerance) in (int, float) and math.isfinite(float(tolerance)) and float(tolerance) > 0,
            "forecast timing authority timestamp tolerance is invalid")
    stored_clock = value.get("zero_policy_clock_probe")
    require(isinstance(stored_clock, Mapping), "forecast timing authority lacks zero-policy clocks")
    child_descriptor = stored_clock.get("child_receipt")
    child_identity, child_path = verify_descriptor(
        child_descriptor, base=resolved.parent, label="authority recorder child", require_bytes=True
    )
    deep_clock = _load_recorder_clock_trace(
        child_path,
        child_identity["sha256"],
        camera_id=value["camera_id"],
        needed_boundaries=[0, *[pair[1] for pair in eligible_pairs]],
    )
    deep_observations = deep_clock.pop("observations_by_boundary")
    for key in (
        "child_receipt",
        "completion",
        "journal",
        "camera_id",
        "minimum_positive_physics_control_interval_s",
        "minimum_positive_camera_capture_interval_s",
        "timestamp_tolerance_s",
    ):
        require(stored_clock.get(key) == deep_clock.get(key),
                f"forecast timing authority recorder {key} changed")
    require(float(tolerance) == float(deep_clock["timestamp_tolerance_s"]),
            "forecast timing authority tolerance differs from the native trace")
    target_rows = []
    previous_time = -1.0
    for row in rows:
        require(isinstance(row, Mapping), "forecast timing mapping row is invalid")
        pair = (row.get("generated_frame_index"), row.get("executed_control_boundary"))
        require(pair in expected_pairs and type(pair[1]) is int and 0 < pair[1] <= expected["prefix"],
                "forecast timing row is outside the source-proven executed prefix")
        value_s = row.get("camera_elapsed_s")
        require(type(value_s) in (int, float) and math.isfinite(float(value_s)) and float(value_s) > previous_time,
                "forecast timing targets are not finite/strictly increasing")
        require(row.get("within_timestamp_tolerance") is True
                and type(row.get("cross_clock_residual_s")) in (int, float)
                and math.isfinite(float(row["cross_clock_residual_s"]))
                and float(row["cross_clock_residual_s"]) <= float(tolerance) + 1e-12,
                "forecast timing row exceeds timestamp tolerance")
        observed = deep_observations[pair[1]]
        initial = deep_observations[0]
        camera_elapsed = (observed["camera_capture_time_ns"] - initial["camera_capture_time_ns"]) / 1e9
        physics_elapsed = observed["physics_time_s"] - initial["physics_time_s"]
        require(
            row.get("observation_id") == observed["observation_id"]
            and row.get("camera_frame_id") == observed["camera_frame_id"]
            and row.get("camera_capture_time_ns") == observed["camera_capture_time_ns"]
            and row.get("physics_step") == observed["physics_step"]
            and row.get("physics_time_s") == observed["physics_time_s"]
            and row.get("camera_elapsed_s") == camera_elapsed
            and row.get("physics_elapsed_s") == physics_elapsed
            and row.get("cross_clock_residual_s") == abs(camera_elapsed - physics_elapsed),
            "forecast timing row differs from the hash-bound native clock trace",
        )
        previous_time = float(value_s)
        target_rows.append({
            "generated_frame_index": pair[0],
            "target_physical_time_s": value_s,
        })
    require(target_rows == targets, "forecast timing target summary differs from mapping rows")
    return value


def _development_request_binding(
    entry: Any,
    *,
    base: Path,
    authority: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    require(isinstance(entry, Mapping), f"{label} sidecar entry is invalid")
    model_id = authority["model_id"]
    request_descriptor, request_path = verify_descriptor(
        entry.get("request_receipt"), base=base, label=f"{label} request", require_bytes=True
    )
    request = load_json(request_path, f"{label} request")
    expected_schema = N3_REQUEST_SCHEMA if model_id == "N3" else D1_REQUEST_SCHEMA
    require(request.get("schema_version") == expected_schema, f"{label} request schema changed")
    if model_id == "N3":
        require(request.get("study_id") == STUDY_ID and request.get("behavioral_model_request") is True
                and request.get("generation_qualification_request") is False,
                f"{label} is not an N3 behavioral request")
        decoded_shape = request.get("decoded_future_shape")
        require(isinstance(decoded_shape, list) and len(decoded_shape) == 4
                and decoded_shape[0] == EXPECTED_MODELS[model_id]["decoded_frames"],
                f"{label} N3 decoded frame count changed")
    else:
        require(request.get("configuration_id") == "D1"
                and request.get("official_action_path") == "GrootSimPolicy.lazy_joint_forward_causal"
                and request.get("custom_s2_used") is False and request.get("patched_s1_used") is False,
                f"{label} is not an official D1 request")
        decoded = request.get("offline_decode")
        rgb = decoded.get("decoded_rgb") if isinstance(decoded, Mapping) else None
        require(isinstance(decoded, Mapping) and decoded.get("performed") is True
                and isinstance(rgb, Mapping)
                and rgb.get("shape", [None])[0] == EXPECTED_MODELS[model_id]["decoded_frames"],
                f"{label} D1 decoded frame count changed")
    request_index = entry.get("request_index")
    require(type(request_index) is int and request.get("request_index") == request_index,
            f"{label} request index changed")
    completion_descriptor, completion_path = verify_descriptor(
        entry.get("adapter_completion"), base=base, label=f"{label} completion", require_bytes=True
    )
    completion = load_json(completion_path, f"{label} completion")
    identity = completion.get("identity")
    cell_id = entry.get("cell_id")
    require(completion.get("schema_version") == RECORDER_ATTEMPT_SCHEMA
            and completion.get("study_id") == STUDY_ID
            and completion.get("behavioral_result_valid") is True
            and completion.get("technical_invalid") is False,
            f"{label} completion is not a valid behavioral attempt")
    require(isinstance(identity, Mapping) and identity.get("cell_id") == cell_id
            and identity.get("model_config") == model_id
            and identity.get("stage") == "development",
            f"{label} completion identity changed")
    execution = completion.get("request_execution")
    require(isinstance(execution, list) and 0 <= request_index < len(execution),
            f"{label} request execution is absent")
    executed = execution[request_index]
    require(isinstance(executed, Mapping) and executed.get("request_index") == request_index,
            f"{label} request execution index changed")
    action_start = executed.get("action_step_start")
    executed_actions = executed.get("executed_actions")
    require(type(action_start) is int and action_start >= 0
            and type(executed_actions) is int and 0 < executed_actions <= authority["unchanged_executed_prefix_horizon"],
            f"{label} executed prefix is invalid")
    if model_id == "N3":
        require(request.get("cell_id") == cell_id and request.get("action_step_start") == action_start,
                f"{label} N3 request/cell schedule changed")
    journal_descriptor, journal_path = verify_descriptor(
        entry.get("adapter_journal"), base=base, label=f"{label} journal", require_bytes=True
    )
    rows, tail = _verify_journal(journal_path)
    require(journal_descriptor.get("event_count") == len(rows)
            and journal_descriptor.get("tail_sha256") == tail
            and completion.get("event_count") == len(rows)
            and completion.get("journal_tail_sha256") == tail,
            f"{label} journal summary changed")
    observations = _event_payloads(rows, "observation_captured")
    packed = _event_payloads(rows, "model_request_packed")
    responses = _event_payloads(rows, "model_response_received")
    require(len(observations) == 451 and len(packed) == len(execution)
            and len(responses) == len(execution),
            f"{label} journal request/response/observation inventory changed")
    packed_request = packed[request_index]
    require(packed_request.get("request_index") == request_index
            and packed_request.get("action_step_start") == action_start
            and packed_request.get("current_observation_id") == f"obs_{action_start:06d}",
            f"{label} request is not bound to its native current observation")
    response = responses[request_index]
    require(response.get("request_index") == request_index,
            f"{label} response request index changed")
    response_binding = _verify_model_response_request_binding(
        completion_path.parent,
        response.get("response_artifact"),
        model_id=model_id,
        request_path=request_path,
        cell_id=cell_id,
        request_index=request_index,
        action_step_start=action_start,
    )
    camera_id = authority["camera_id"]
    start = observations[action_start]

    def sample(observation: Mapping[str, Any], expected_control: int) -> dict[str, Any]:
        require(observation.get("observation_id") == f"obs_{expected_control:06d}"
                and observation.get("control_step") == expected_control,
                f"{label} observation schedule changed")
        clock = observation.get("clock")
        require(isinstance(clock, Mapping) and clock.get("control_step") == expected_control,
                f"{label} native control clock changed")
        cameras = clock.get("cameras")
        camera = cameras.get(camera_id) if isinstance(cameras, Mapping) else None
        require(type(clock.get("physics_step")) is int
                and type(clock.get("physics_time_s")) in (int, float)
                and math.isfinite(float(clock["physics_time_s"]))
                and isinstance(camera, Mapping)
                and camera.get("frame_id") is not None
                and type(camera.get("capture_time_ns")) is int
                and isinstance(camera.get("timestamp_source"), str)
                and camera["timestamp_source"],
                f"{label} native physics/original-camera receipt is incomplete")
        return {
            "observation_id": observation["observation_id"],
            "physics_step": clock["physics_step"],
            "physics_time_s": float(clock["physics_time_s"]),
            "camera_frame_id": camera["frame_id"],
            "camera_capture_time_ns": camera["capture_time_ns"],
            "camera_timestamp_source": camera["timestamp_source"],
        }

    start_sample = sample(start, action_start)
    target_bindings = []
    tolerance = float(authority["timestamp_tolerance_s"])
    for mapping, target in zip(authority["qualified_mapping_rows"], authority["generated_targets"]):
        boundary = mapping["executed_control_boundary"]
        row = {
            "generated_frame_index": mapping["generated_frame_index"],
            "executed_control_boundary": boundary,
            "authority_target_physical_time_s": target["target_physical_time_s"],
        }
        if boundary > executed_actions:
            target_bindings.append({
                **row,
                "status": "not_executed_in_truncated_prefix",
                "target_observation_id": None,
            })
            continue
        target_sample = sample(observations[action_start + boundary], action_start + boundary)
        camera_elapsed = (
            target_sample["camera_capture_time_ns"] - start_sample["camera_capture_time_ns"]
        ) / 1e9
        physics_elapsed = target_sample["physics_time_s"] - start_sample["physics_time_s"]
        camera_residual = abs(camera_elapsed - float(target["target_physical_time_s"]))
        physics_residual = abs(physics_elapsed - float(target["target_physical_time_s"]))
        require(camera_elapsed > 0 and physics_elapsed > 0
                and target_sample["camera_frame_id"] != start_sample["camera_frame_id"]
                and target_sample["physics_step"] > start_sample["physics_step"],
                f"{label} target clocks did not advance")
        require(camera_residual <= tolerance + 1e-12 and physics_residual <= tolerance + 1e-12,
                f"{label} target differs from the qualified authority")
        target_bindings.append({
            **row,
            "status": "matched_native_request_clocks",
            "target_observation_id": target_sample["observation_id"],
            "camera_elapsed_s": camera_elapsed,
            "physics_elapsed_s": physics_elapsed,
            "camera_authority_residual_s": camera_residual,
            "physics_authority_residual_s": physics_residual,
            "target_camera_frame_id": target_sample["camera_frame_id"],
            "target_camera_capture_time_ns": target_sample["camera_capture_time_ns"],
            "target_physics_step": target_sample["physics_step"],
            "target_physics_time_s": target_sample["physics_time_s"],
        })
    return {
        "cell_id": cell_id,
        "request_index": request_index,
        "source_request_receipt": file_descriptor(request_path),
        "adapter_completion": file_descriptor(completion_path),
        "adapter_journal": {**file_descriptor(journal_path), "event_count": len(rows), "tail_sha256": tail},
        "action_step_start": action_start,
        "executed_actions": executed_actions,
        "request_current_observation_id": start_sample["observation_id"],
        "recorder_response_request_binding": response_binding,
        "camera_id": camera_id,
        "target_bindings": target_bindings,
        "model_output_or_action_modified": False,
    }


def bind_development_requests(
    *,
    authority_path: Path,
    authority_sha256: str,
    inventory_path: Path,
    inventory_sha256: str,
) -> dict[str, Any]:
    authority = validate_timing_authority(authority_path, authority_sha256)
    inventory, inventory_resolved = _read_hashed_json(
        inventory_path, inventory_sha256, "development timing request inventory"
    )
    model_id = authority["model_id"]
    require(inventory.get("schema_version") == REQUEST_INVENTORY_SCHEMA, "request inventory schema changed")
    require(inventory.get("study_id") == STUDY_ID and inventory.get("model_id") == model_id,
            "request inventory identity changed")
    requests = inventory.get("request_receipts")
    require(isinstance(requests, list) and requests, "request inventory is empty")
    require(all(isinstance(entry, Mapping) and "request_receipt" in entry for entry in requests),
            "request inventory must use immutable sidecar entries; old receipts are never rewritten")
    bindings = [
        _development_request_binding(
            entry, base=inventory_resolved.parent, authority=authority,
            label=f"development request {index}",
        )
        for index, entry in enumerate(requests)
    ]
    hashes = [binding["source_request_receipt"]["sha256"] for binding in bindings]
    require(len(set(hashes)) == len(hashes), "development request receipts are duplicated")
    output_targets = [
        {
            "generated_frame_index": row["generated_frame_index"],
            "target_physical_time_s": row["target_physical_time_s"],
            "native_runtime_field": SIDECAR_RUNTIME_FIELD,
        }
        for row in authority["generated_targets"]
    ]
    return sign_document({
        "schema_version": DEVELOPMENT_TIMING_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": model_id,
        "status": "qualified_from_native_runtime_metadata",
        "time_source_kind": "native_runtime_exposed_target_offsets",
        "binding_mode": "immutable_request_receipt_native_clock_sidecar",
        "clock_bridge": "elapsed physical seconds from request current original-camera capture",
        "native_runtime_field": SIDECAR_RUNTIME_FIELD,
        "source_request_receipt_sha256s": hashes,
        "generated_targets": output_targets,
        "timing_authority": file_descriptor(Path(authority_path).resolve()),
        "request_inventory": file_descriptor(inventory_resolved),
        "validator": file_descriptor(Path(__file__).resolve()),
        "request_timing_bindings": bindings,
        "old_request_receipts_modified": False,
        "presentation_video_fps_used": False,
        "conditioning_fps_used_as_target_timing": False,
        "generated_frame_index_interpreted_as_action_index": False,
        "dreamzero_action_block_ratio_used_as_mapping": False,
        "claim_boundary": "Binds qualified native timing to immutable development request and cell-clock receipts through a sidecar; does not release confirmation or claim prediction accuracy.",
    })


def validate_development_timing(
    path: Path,
    expected_sha256: str,
    *,
    expected_model: str | None = None,
    expected_request_hashes: Sequence[str] | None = None,
) -> dict[str, Any]:
    value, resolved = _read_hashed_json(path, expected_sha256, "development timing sidecar")
    require(value.get("schema_version") == DEVELOPMENT_TIMING_SCHEMA, "development timing schema changed")
    verify_signed(value, "development timing sidecar")
    model_id = value.get("model_id")
    require(model_id in EXPECTED_MODELS and value.get("study_id") == STUDY_ID,
            "development timing identity changed")
    if expected_model is not None:
        require(model_id == expected_model, "development timing model mismatch")
    require(value.get("status") == "qualified_from_native_runtime_metadata"
            and value.get("time_source_kind") == "native_runtime_exposed_target_offsets"
            and value.get("binding_mode") == "immutable_request_receipt_native_clock_sidecar"
            and value.get("old_request_receipts_modified") is False,
            "development timing sidecar did not qualify immutable receipts")
    require_descriptor_matches(
        value.get("validator"), Path(__file__).resolve(),
        base=resolved.parent, label="development timing validator",
    )
    for field in (
        "presentation_video_fps_used",
        "conditioning_fps_used_as_target_timing",
        "generated_frame_index_interpreted_as_action_index",
        "dreamzero_action_block_ratio_used_as_mapping",
    ):
        require(value.get(field) is False, f"development timing used prohibited inference: {field}")
    authority_descriptor, authority_path = verify_descriptor(
        value.get("timing_authority"), base=resolved.parent,
        label="development timing authority", require_bytes=True,
    )
    authority = validate_timing_authority(
        authority_path, authority_descriptor["sha256"], expected_model=model_id
    )
    expected_targets = [
        {
            "generated_frame_index": row["generated_frame_index"],
            "target_physical_time_s": row["target_physical_time_s"],
            "native_runtime_field": SIDECAR_RUNTIME_FIELD,
        }
        for row in authority["generated_targets"]
    ]
    require(value.get("generated_targets") == expected_targets
            and value.get("native_runtime_field") == SIDECAR_RUNTIME_FIELD,
            "development timing targets differ from the authority")
    bindings = value.get("request_timing_bindings")
    require(isinstance(bindings, list) and bindings, "development timing sidecar has no request bindings")
    regenerated = [
        _development_request_binding(
            {
                "cell_id": binding.get("cell_id"),
                "request_index": binding.get("request_index"),
                "request_receipt": binding.get("source_request_receipt"),
                "adapter_completion": binding.get("adapter_completion"),
                "adapter_journal": binding.get("adapter_journal"),
            },
            base=resolved.parent,
            authority=authority,
            label=f"development sidecar request {index}",
        )
        for index, binding in enumerate(bindings)
    ]
    require(regenerated == bindings, "development timing bindings differ from immutable receipts/clocks")
    hashes = [binding["source_request_receipt"]["sha256"] for binding in bindings]
    require(value.get("source_request_receipt_sha256s") == hashes
            and len(set(hashes)) == len(hashes),
            "development timing request hash inventory changed")
    if expected_request_hashes is not None:
        require(hashes == list(expected_request_hashes),
                "development timing sidecar does not bind the expected request inventory")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_n3 = commands.add_parser(
        "prepare-n3-live-input",
        help="package the exact live fixed-capture N3 fixture without contacting the model",
    )
    prepare_n3.add_argument("--capture", type=Path, required=True)
    prepare_n3.add_argument("--capture-sha256", required=True)
    prepare_n3.add_argument("--camera-id", choices=CAMERAS, required=True)
    prepare_n3.add_argument("--output-dir", type=Path, required=True)

    audit = commands.add_parser("audit-source", help="verify one pinned source lineage")
    audit.add_argument("--model", choices=tuple(EXPECTED_MODELS), required=True)
    audit.add_argument("--source-root", type=Path, required=True)
    audit.add_argument("--contract", type=Path, required=True)
    audit.add_argument("--contract-sha256", required=True)
    audit.add_argument("--output", type=Path, required=True)

    generation = commands.add_parser("normalize-generation", help="normalize a real N3/D1 qualification receipt")
    generation.add_argument("--model", choices=tuple(EXPECTED_MODELS), required=True)
    generation.add_argument("--qualification", type=Path, required=True)
    generation.add_argument("--qualification-sha256", required=True)
    generation.add_argument("--contract", type=Path, required=True)
    generation.add_argument("--contract-sha256", required=True)
    generation.add_argument("--output", type=Path, required=True)

    qualify = commands.add_parser("qualify", help="close source mapping with generation and native clock probes")
    qualify.add_argument("--model", choices=tuple(EXPECTED_MODELS), required=True)
    qualify.add_argument("--contract", type=Path, required=True)
    qualify.add_argument("--contract-sha256", required=True)
    qualify.add_argument("--source-audit", type=Path, required=True)
    qualify.add_argument("--source-audit-sha256", required=True)
    qualify.add_argument("--generation-probe", type=Path, required=True)
    qualify.add_argument("--generation-probe-sha256", required=True)
    qualify.add_argument("--recorder-receipt", type=Path, required=True)
    qualify.add_argument("--recorder-receipt-sha256", required=True)
    qualify.add_argument("--camera-id", choices=CAMERAS, required=True)
    qualify.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate-authority", help="validate an immutable timing authority")
    validate.add_argument("--authority", type=Path, required=True)
    validate.add_argument("--sha256", required=True)
    validate.add_argument("--model", choices=tuple(EXPECTED_MODELS))

    bind = commands.add_parser("bind-development", help="bind exact development request metadata")
    bind.add_argument("--authority", type=Path, required=True)
    bind.add_argument("--authority-sha256", required=True)
    bind.add_argument("--request-inventory", type=Path, required=True)
    bind.add_argument("--request-inventory-sha256", required=True)
    bind.add_argument("--output", type=Path, required=True)

    validate_development = commands.add_parser(
        "validate-development", help="deeply revalidate an immutable-request timing sidecar"
    )
    validate_development.add_argument("--timing", type=Path, required=True)
    validate_development.add_argument("--sha256", required=True)
    validate_development.add_argument("--model", choices=tuple(EXPECTED_MODELS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-n3-live-input":
        result = prepare_n3_live_input(
            capture_path=args.capture,
            capture_sha256=args.capture_sha256,
            camera_id=args.camera_id,
            output_dir=args.output_dir,
        )
        print(json.dumps({
            "status": result["status"],
            "output": result["preparation_receipt"]["path"],
            "sha256": result["preparation_receipt"]["sha256"],
            "observation_manifest": result["observation_manifest"],
        }, sort_keys=True))
        return 0
    if args.command == "audit-source":
        result = audit_source(
            model_id=args.model,
            source_root=args.source_root,
            contract_path=args.contract,
            contract_sha256=args.contract_sha256,
        )
        atomic_json(args.output, result)
        print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve()),
                          "sha256": sha256_file(args.output)}, sort_keys=True))
        return 0
    if args.command == "normalize-generation":
        result = normalize_generation_probe(
            model_id=args.model,
            qualification_path=args.qualification,
            qualification_sha256=args.qualification_sha256,
            contract_path=args.contract,
            contract_sha256=args.contract_sha256,
        )
        atomic_json(args.output, result)
        print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve()),
                          "sha256": sha256_file(args.output)}, sort_keys=True))
        return 0
    if args.command == "qualify":
        result = qualify_timing(
            model_id=args.model,
            contract_path=args.contract,
            contract_sha256=args.contract_sha256,
            source_audit_path=args.source_audit,
            source_audit_sha256=args.source_audit_sha256,
            generation_probe_path=args.generation_probe,
            generation_probe_sha256=args.generation_probe_sha256,
            recorder_receipt_path=args.recorder_receipt,
            recorder_receipt_sha256=args.recorder_receipt_sha256,
            camera_id=args.camera_id,
        )
        atomic_json(args.output, result)
        print(json.dumps({"status": result["status"], "model": result["model_id"],
                          "target_count": len(result["generated_targets"]),
                          "output": str(Path(args.output).resolve()),
                          "sha256": sha256_file(args.output)}, sort_keys=True))
        return 0
    if args.command == "validate-authority":
        result = validate_timing_authority(args.authority, args.sha256, expected_model=args.model)
        print(json.dumps({"status": result["status"], "model": result["model_id"],
                          "target_count": len(result["generated_targets"])}, sort_keys=True))
        return 0
    if args.command == "validate-development":
        result = validate_development_timing(args.timing, args.sha256, expected_model=args.model)
        print(json.dumps({
            "status": result["status"],
            "model": result["model_id"],
            "request_count": len(result["source_request_receipt_sha256s"]),
        }, sort_keys=True))
        return 0
    result = bind_development_requests(
        authority_path=args.authority,
        authority_sha256=args.authority_sha256,
        inventory_path=args.request_inventory,
        inventory_sha256=args.request_inventory_sha256,
    )
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "model": result["model_id"],
                      "request_count": len(result["source_request_receipt_sha256s"]),
                      "output": str(Path(args.output).resolve()),
                      "sha256": sha256_file(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
