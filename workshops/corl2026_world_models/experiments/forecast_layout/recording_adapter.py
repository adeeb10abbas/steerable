#!/usr/bin/env python3
"""Fail-closed fixed-duration recording adapter for WMF-ABLATION-001.

This module is deliberately workshop-local.  It does not import or modify an
old V2/V3 compiler, registry, or task.  The concrete fixture module must expose
a timeout-only task *before* RoboLab constructs the environment.  This module
then verifies that environment and wraps it so one policy episode means 450
actual calls to ``env.step``.  Success is sampled as measurement data only.

Large payloads are written losslessly beneath one immutable attempt directory.
The JSONL journal is fsync'd and hash chained, so an interrupted attempt remains
auditable even when no completion receipt can be written.  Physical forecast
alignment is intentionally not inferred here: a live integration must supply
native simulator/camera clock identities and later qualify the frame mapping.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any
import uuid

import numpy as np


CONTRACT_PATH = Path(__file__).with_name("recording_contract.json")
REQUIRED_CAMERAS = (
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)
REQUIRED_IDENTITY_FIELDS = (
    "attempt_id",
    "cell_id",
    "stage",
    "layout_pair_id",
    "layout_arm",
    "command",
    "prompt",
    "model_config",
    "effective_seed",
    "source_identity",
    "checkpoint_identity",
)
CONTEXT_RESET_SCOPE = "full_episode_temporal_and_cache_context"


class RecordingContractError(RuntimeError):
    """The attempt cannot qualify under the frozen recording contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_recording_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != "wmf-forecast-recording-contract-v1":
        raise RecordingContractError("unexpected recording-contract schema")
    if value.get("study_id") != "WMF-ABLATION-001":
        raise RecordingContractError("recording contract belongs to another study")
    episode = value.get("episode", {})
    if (
        episode.get("action_cap") != 450
        or episode.get("episode_length_s") != 30
        or episode.get("stop_on_success") is not False
        or episode.get("record_first_success") is not True
    ):
        raise RecordingContractError("fixed-duration episode contract changed")
    return value


def _is_array_like(value: Any) -> bool:
    return isinstance(value, np.ndarray) or all(
        hasattr(value, name) for name in ("detach", "cpu", "numpy")
    )


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if all(hasattr(value, name) for name in ("detach", "cpu", "numpy")):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _one_env_array(value: Any, env_id: int, *, field: str) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim == 0 or array.shape[0] <= env_id:
        raise RecordingContractError(f"{field} has no environment axis for env {env_id}")
    return np.ascontiguousarray(array[env_id])


def _boolean_vector_any(value: Any) -> bool:
    array = _as_numpy(value)
    if array.size == 0:
        return False
    return bool(np.asarray(array, dtype=bool).any())


def _validated_success_snapshot(value: Mapping[str, Any]) -> dict[str, bool]:
    snapshot = dict(value)
    if type(snapshot.get("left")) is not bool or type(snapshot.get("right")) is not bool:
        raise RecordingContractError("success sampler must return boolean left/right predicates")
    if type(snapshot.get("released")) is not bool:
        raise RecordingContractError("success sampler must return boolean released state")
    if snapshot["left"] and snapshot["right"]:
        raise RecordingContractError("left and right success predicates cannot both be true")
    if (snapshot["left"] or snapshot["right"]) and not snapshot["released"]:
        raise RecordingContractError("relation-and-release success predicate disagrees with released state")
    return {
        "left": snapshot["left"],
        "right": snapshot["right"],
        "released": snapshot["released"],
    }


def _array_identity(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    header = {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "order": "C",
    }
    digest = hashlib.sha256()
    digest.update(_canonical_bytes(header))
    digest.update(contiguous.tobytes(order="C"))
    return {**header, "value_sha256": digest.hexdigest()}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class _PayloadStore:
    """Lossless nested payload storage with one NPZ artifact per payload."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self._lock = threading.Lock()

    def _freeze(
        self,
        value: Any,
        arrays: dict[str, np.ndarray],
        location: str,
    ) -> Any:
        if _is_array_like(value):
            array = np.ascontiguousarray(_as_numpy(value)).copy()
            if array.dtype.hasobject:
                raise RecordingContractError(f"object array is not losslessly allowed at {location}")
            key = f"array_{len(arrays):04d}"
            arrays[key] = array
            return {
                "__type__": "ndarray",
                "key": key,
                "shape": list(array.shape),
                "dtype": array.dtype.str,
            }
        if isinstance(value, np.generic):
            return self._freeze(np.asarray(value), arrays, location)
        if isinstance(value, Mapping):
            output: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, (str, int, float, bool)):
                    raise RecordingContractError(f"unsupported mapping key at {location}: {type(key)}")
                encoded = str(key)
                if encoded in output:
                    raise RecordingContractError(f"mapping key collision at {location}: {encoded}")
                output[encoded] = self._freeze(item, arrays, f"{location}.{encoded}")
            return {"__type__": "mapping", "items": output}
        if isinstance(value, tuple):
            return {
                "__type__": "tuple",
                "items": [self._freeze(item, arrays, f"{location}[{index}]") for index, item in enumerate(value)],
            }
        if isinstance(value, list):
            return {
                "__type__": "list",
                "items": [self._freeze(item, arrays, f"{location}[{index}]") for index, item in enumerate(value)],
            }
        if isinstance(value, (bytes, bytearray, memoryview)):
            raw = bytes(value)
            return self._freeze(np.frombuffer(raw, dtype=np.uint8).copy(), arrays, location) | {
                "logical_type": "bytes"
            }
        if isinstance(value, Path):
            return {"__type__": "path", "value": str(value)}
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise RecordingContractError(f"non-finite scalar at {location}")
            return value
        raise RecordingContractError(f"unsupported payload type at {location}: {type(value)}")

    def write(self, role: str, value: Any) -> dict[str, Any]:
        if not role or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in role):
            raise RecordingContractError(f"unsafe payload role: {role!r}")
        with self._lock:
            sequence = self._counter
            self._counter += 1
        arrays: dict[str, np.ndarray] = {}
        frozen = self._freeze(value, arrays, role)
        result: dict[str, Any] = {
            "role": role,
            "structure": frozen,
            "array_count": len(arrays),
        }
        if arrays:
            name = f"{sequence:07d}_{role}.npz"
            path = self.root / name
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("xb") as handle:
                    np.savez(handle, **arrays)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                _fsync_directory(path.parent)
            finally:
                if temporary.exists():
                    temporary.unlink()
            result["artifact"] = {
                "path": str(path.relative_to(self.root.parent)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "encoding": "numpy_npz_zip_stored",
            }
        result["payload_sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
        return result


def _thaw_payload(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    if not isinstance(value, Mapping) or "__type__" not in value:
        return value
    kind = value["__type__"]
    if kind == "ndarray":
        array = np.ascontiguousarray(arrays[value["key"]]).copy()
        if list(array.shape) != value["shape"] or array.dtype.str != value["dtype"]:
            raise RecordingContractError("stored array shape/dtype differs from its payload manifest")
        if value.get("logical_type") == "bytes":
            return array.tobytes()
        return array
    if kind == "mapping":
        return {key: _thaw_payload(item, arrays) for key, item in value["items"].items()}
    if kind == "tuple":
        return tuple(_thaw_payload(item, arrays) for item in value["items"])
    if kind == "list":
        return [_thaw_payload(item, arrays) for item in value["items"]]
    if kind == "path":
        return Path(value["value"])
    raise RecordingContractError(f"unknown stored payload type: {kind}")


def load_payload(attempt_dir: Path, descriptor: Mapping[str, Any]) -> Any:
    """Hash-check and reconstruct one payload for qualification/analysis."""

    expected_payload_hash = descriptor.get("payload_sha256")
    unsigned = {key: value for key, value in descriptor.items() if key != "payload_sha256"}
    if hashlib.sha256(_canonical_bytes(unsigned)).hexdigest() != expected_payload_hash:
        raise RecordingContractError("payload descriptor hash mismatch")
    artifact = descriptor.get("artifact")
    arrays: dict[str, np.ndarray] = {}
    if artifact is not None:
        path = Path(attempt_dir).resolve() / artifact["path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RecordingContractError("payload artifact path/hash mismatch")
        if path.stat().st_size != artifact["bytes"]:
            raise RecordingContractError("payload artifact byte count mismatch")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    if len(arrays) != descriptor["array_count"]:
        raise RecordingContractError("payload array count mismatch")
    return _thaw_payload(descriptor["structure"], arrays)


def verify_journal(path: Path) -> dict[str, Any]:
    """Verify sequence continuity and the complete fsync journal hash chain."""

    previous = None
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("sequence") != count:
                raise RecordingContractError("journal sequence discontinuity")
            if record.get("previous_event_sha256") != previous:
                raise RecordingContractError("journal previous-event hash mismatch")
            claimed = record.pop("event_sha256", None)
            observed = hashlib.sha256(_canonical_bytes(record)).hexdigest()
            if claimed != observed:
                raise RecordingContractError("journal event hash mismatch")
            previous = claimed
            count += 1
    if count == 0:
        raise RecordingContractError("attempt journal is empty")
    return {"event_count": count, "tail_sha256": previous}


def _validate_identity(identity: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_IDENTITY_FIELDS if field not in identity]
    if missing:
        raise RecordingContractError(f"attempt identity missing fields: {missing}")
    value = copy.deepcopy(dict(identity))
    if value.get("model_config") not in contract["models"]:
        raise RecordingContractError("attempt model is outside the primary recording contract")
    if value.get("layout_arm") not in {"original", "reflected"}:
        raise RecordingContractError("layout arm must be original or reflected")
    if value.get("command") not in {"left", "right"}:
        raise RecordingContractError("command must be left or right")
    if not all(isinstance(value[field], (str, int)) and str(value[field]) for field in REQUIRED_IDENTITY_FIELDS):
        raise RecordingContractError("attempt identity fields must be nonempty scalar identities")
    value["study_id"] = contract["study_id"]
    return value


class ForecastRecordingAdapter:
    """Durable state machine shared by the environment and client wrappers."""

    def __init__(
        self,
        attempt_dir: Path,
        identity: Mapping[str, Any],
        *,
        contract_path: Path = CONTRACT_PATH,
    ) -> None:
        self.contract_path = Path(contract_path).resolve()
        self.contract = load_recording_contract(self.contract_path)
        self.identity = _validate_identity(identity, self.contract)
        self.attempt_dir = Path(attempt_dir).resolve()
        if self.attempt_dir.exists() and any(self.attempt_dir.iterdir()):
            raise FileExistsError(f"refusing to reuse nonempty attempt directory: {self.attempt_dir}")
        self.attempt_dir.mkdir(parents=True, exist_ok=True)
        self.payloads = _PayloadStore(self.attempt_dir / "payloads")
        self.journal_path = self.attempt_dir / "events.partial.jsonl"
        self.completion_path = self.attempt_dir / "completion.json"
        if self.journal_path.exists() or self.completion_path.exists():
            raise FileExistsError("refusing to overwrite attempt receipts")
        self._journal_lock = threading.Lock()
        self._sequence = 0
        self._previous_event_sha256: str | None = None
        self.context_reset: dict[str, Any] | None = None
        self.environment_contract: dict[str, Any] | None = None
        self.reset_receipt: dict[str, Any] | None = None
        self.runner_reset_calls = 0
        self.physical_reset_calls = 0
        self.observations: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self._pending_request: dict[str, Any] | None = None
        self._proposed_action: dict[str, Any] | None = None
        self.actions_executed = 0
        self.first_success: dict[str, Any] | None = None
        self._final_receipt: dict[str, Any] | None = None
        self.process_identity = {
            "hostname": os.uname().nodename,
            "pid": os.getpid(),
            "process_start_monotonic_ns": time.monotonic_ns(),
        }
        self._append_event(
            "attempt_started",
            {
                "identity": self.identity,
                "contract_path": str(self.contract_path),
                "contract_sha256": sha256_file(self.contract_path),
                "process_identity": self.process_identity,
            },
        )

    @property
    def finalized(self) -> bool:
        return self._final_receipt is not None

    @property
    def action_cap(self) -> int:
        return int(self.contract["episode"]["action_cap"])

    @property
    def current_observation(self) -> dict[str, Any] | None:
        return self.observations[-1] if self.observations else None

    def _append_event(self, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._journal_lock:
            base = {
                "sequence": self._sequence,
                "kind": kind,
                "wall_time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "previous_event_sha256": self._previous_event_sha256,
                "payload": dict(payload),
            }
            digest = hashlib.sha256(_canonical_bytes(base)).hexdigest()
            record = dict(base, event_sha256=digest)
            with self.journal_path.open("ab") as handle:
                handle.write(_canonical_bytes(record) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            if self._sequence == 0:
                _fsync_directory(self.journal_path.parent)
            self._sequence += 1
            self._previous_event_sha256 = digest
            return record

    def _fail(self, component: str, error: BaseException | str) -> None:
        if self.finalized:
            return
        detail = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        self.finalize(
            "technical_failure",
            {"component": component, "error": detail},
        )

    def record_context_reset(self, receipt: Mapping[str, Any]) -> None:
        if self.context_reset is not None:
            raise RecordingContractError("model context reset was already recorded")
        if self.physical_reset_calls or self.actions_executed:
            raise RecordingContractError("model context reset must precede physical reset and actions")
        required = self.contract["reset"]["context_receipt_required_fields"]
        missing = [field for field in required if field not in receipt]
        if missing:
            raise RecordingContractError(f"context reset receipt missing fields: {missing}")
        if receipt.get("passed") is not True:
            raise RecordingContractError("model context reset did not pass")
        if receipt.get("reset_scope") != CONTEXT_RESET_SCOPE:
            raise RecordingContractError("model context reset scope is incomplete")
        if not isinstance(receipt.get("server_context_id"), str) or not receipt["server_context_id"]:
            raise RecordingContractError("model reset lacks a server context identity")
        if not isinstance(receipt.get("cache_reset_evidence"), Mapping) or not receipt["cache_reset_evidence"]:
            raise RecordingContractError("model reset lacks cache reset evidence")
        artifact = self.payloads.write("context_reset", dict(receipt))
        self.context_reset = {"receipt": copy.deepcopy(dict(receipt)), "artifact": artifact}
        self._append_event("model_context_reset", {"artifact": artifact})

    def record_environment_contract(self, receipt: Mapping[str, Any]) -> None:
        if self.environment_contract is not None:
            raise RecordingContractError("environment contract already recorded")
        self.environment_contract = copy.deepcopy(dict(receipt))
        self._append_event("fixed_duration_environment_verified", self.environment_contract)

    def record_physical_reset(
        self,
        raw_obs: Mapping[str, Any],
        *,
        clock: Mapping[str, Any],
        state: Mapping[str, Any],
        attestation: Mapping[str, Any],
        env_id: int = 0,
    ) -> dict[str, Any]:
        if self.context_reset is None:
            raise RecordingContractError("physical reset preceded model context reset")
        if self.physical_reset_calls:
            raise RecordingContractError("more than one physical reset is prohibited")
        required = self.contract["reset"]["settled_reset_receipt_required_fields"]
        missing = [field for field in required if field not in attestation]
        if missing:
            raise RecordingContractError(f"settled reset receipt missing fields: {missing}")
        if attestation.get("passed") is not True or attestation.get("settled") is not True:
            raise RecordingContractError("physical reset did not pass settling qualification")
        if attestation.get("settled_observation_returned") is not True:
            raise RecordingContractError("reset did not return the exact post-settle observation")
        if attestation.get("model_request_count_during_settle") != 0:
            raise RecordingContractError("model inference occurred during model-blind settling")
        if attestation.get("episode_length_buf_reset_to_zero") is not True:
            raise RecordingContractError("behavioral action counter was not zeroed after settling")
        if attestation.get("left_success") is not False or attestation.get("right_success") is not False:
            raise RecordingContractError("physical reset begins in a success state")
        if not isinstance(attestation.get("reset_identity"), str) or not attestation["reset_identity"]:
            raise RecordingContractError("physical reset identity is missing")
        self.physical_reset_calls = 1
        self.runner_reset_calls = 1
        attestation_artifact = self.payloads.write("physical_reset", dict(attestation))
        self.reset_receipt = {
            "attestation": copy.deepcopy(dict(attestation)),
            "artifact": attestation_artifact,
        }
        self._append_event("physical_reset", {"artifact": attestation_artifact})
        return self.record_observation(
            raw_obs,
            clock=clock,
            state=state,
            control_step=0,
            phase="settled_reset",
            env_id=env_id,
        )

    def record_idempotent_runner_reset(self) -> None:
        if self.physical_reset_calls != 1 or self.runner_reset_calls != 1 or self.actions_executed:
            raise RecordingContractError("second runner reset cannot be made idempotent in this state")
        self.runner_reset_calls = 2
        self._append_event(
            "runner_reset_idempotent",
            {"runner_reset_calls": 2, "physical_reset_calls": 1},
        )

    def _validate_clock(self, clock: Mapping[str, Any], control_step: int) -> dict[str, Any]:
        required = ("physics_step", "physics_time_s", "control_step", "cameras")
        missing = [field for field in required if field not in clock]
        if missing:
            raise RecordingContractError(f"clock sample missing fields: {missing}")
        if type(clock["physics_step"]) is not int or clock["physics_step"] < 0:
            raise RecordingContractError("physics_step must be a nonnegative integer")
        if type(clock["control_step"]) is not int or clock["control_step"] != control_step:
            raise RecordingContractError("native control_step does not match executed action count")
        if not isinstance(clock["physics_time_s"], (int, float)) or not math.isfinite(float(clock["physics_time_s"])):
            raise RecordingContractError("physics_time_s must be finite")
        cameras = clock["cameras"]
        if not isinstance(cameras, Mapping):
            raise RecordingContractError("camera clock identities must be a mapping")
        for name in REQUIRED_CAMERAS:
            item = cameras.get(name)
            if not isinstance(item, Mapping):
                raise RecordingContractError(f"camera clock identity missing for {name}")
            if item.get("frame_id") is None:
                raise RecordingContractError(f"camera frame identity missing for {name}")
            capture = item.get("capture_time_ns")
            if type(capture) is not int or capture < 0:
                raise RecordingContractError(f"camera capture timestamp missing for {name}")
            if not isinstance(item.get("timestamp_source"), str) or not item["timestamp_source"]:
                raise RecordingContractError(f"camera timestamp source missing for {name}")
        return copy.deepcopy(dict(clock))

    def record_observation(
        self,
        raw_obs: Mapping[str, Any],
        *,
        clock: Mapping[str, Any],
        state: Mapping[str, Any],
        control_step: int,
        phase: str,
        env_id: int = 0,
    ) -> dict[str, Any]:
        if self.finalized:
            raise RecordingContractError("cannot append an observation to a finalized attempt")
        if control_step != len(self.observations):
            raise RecordingContractError("observation schedule is not reset plus every action")
        if control_step != self.actions_executed:
            raise RecordingContractError("observation/action count mismatch")
        image_obs = raw_obs.get("image_obs")
        proprio_obs = raw_obs.get("proprio_obs")
        if not isinstance(image_obs, Mapping) or not isinstance(proprio_obs, Mapping):
            raise RecordingContractError("raw observation lacks image_obs or proprio_obs")
        for camera in REQUIRED_CAMERAS:
            if camera not in image_obs:
                raise RecordingContractError(f"required original camera missing: {camera}")
        images = {
            str(name): _one_env_array(value, env_id, field=f"image_obs.{name}")
            for name, value in image_obs.items()
        }
        proprio = {
            str(name): _one_env_array(value, env_id, field=f"proprio_obs.{name}")
            for name, value in proprio_obs.items()
        }
        clock_value = self._validate_clock(clock, control_step)
        observation_id = f"obs_{control_step:06d}"
        received_monotonic_ns = time.monotonic_ns()
        artifact = self.payloads.write(
            "observation",
            {
                "observation_id": observation_id,
                "phase": phase,
                "image_obs": images,
                "proprio_obs": proprio,
                "clock": clock_value,
                "state": dict(state),
                "host_observation_received_monotonic_ns": received_monotonic_ns,
            },
        )
        record = {
            "observation_id": observation_id,
            "control_step": control_step,
            "phase": phase,
            "clock": clock_value,
            "artifact": artifact,
        }
        self.observations.append(record)
        self._append_event("observation_captured", record)
        return record

    def begin_request(self, model_input: Any, wire_request: Any) -> int:
        if self.finalized:
            raise RecordingContractError("cannot start a request on a finalized attempt")
        if self._pending_request is not None:
            raise RecordingContractError("model requests overlap within one isolated context")
        if self.current_observation is None:
            raise RecordingContractError("model request preceded the settled original observation")
        model = self.contract["models"][self.identity["model_config"]]
        request_index = len(self.requests)
        current = self.current_observation
        preceding = self.observations[-2] if len(self.observations) >= 2 else None
        artifact = self.payloads.write(
            "model_request",
            {
                "extracted_preprocessing_output": model_input,
                "wire_request": wire_request,
            },
        )
        request = {
            "request_index": request_index,
            "action_step_start": self.actions_executed,
            "current_observation_id": current["observation_id"],
            "preceding_observation_id": preceding["observation_id"] if preceding else None,
            "constant_velocity_baseline": (
                "current_and_preceding_original_observation"
                if preceding
                else "reduces_to_persistence_no_preceding_observation"
            ),
            "returned_action_horizon": int(model["returned_action_horizon"]),
            "executed_prefix_horizon": int(model["executed_prefix_horizon"]),
            "required_future_evidence": list(model["required_future_evidence"]),
            "model_request_artifact": artifact,
            "pack_monotonic_ns": time.monotonic_ns(),
            "send_monotonic_ns": None,
            "receive_monotonic_ns": None,
            "future_kinds": [],
            "executed_offsets": [],
        }
        self.requests.append(request)
        self._pending_request = request
        self._append_event("model_request_packed", request)
        return request_index

    def mark_request_sent(self) -> None:
        if self._pending_request is None or self._pending_request["send_monotonic_ns"] is not None:
            raise RecordingContractError("request send event is out of order")
        timestamp = time.monotonic_ns()
        self._pending_request["send_monotonic_ns"] = timestamp
        self._append_event(
            "model_request_sent",
            {"request_index": self._pending_request["request_index"], "send_monotonic_ns": timestamp},
        )

    def record_raw_response(self, response: Any, future_evidence: Mapping[str, Any]) -> None:
        if self._pending_request is None or self._pending_request["send_monotonic_ns"] is None:
            raise RecordingContractError("response preceded a sent request")
        if self._pending_request["receive_monotonic_ns"] is not None:
            raise RecordingContractError("duplicate response for one request")
        receive = time.monotonic_ns()
        for kind, value in future_evidence.items():
            self._validate_future_evidence(str(kind), value)
        artifact = self.payloads.write(
            "model_response",
            {"raw_response": response, "future_evidence": dict(future_evidence)},
        )
        kinds = sorted(key for key, value in future_evidence.items() if value is not None)
        self._pending_request.update(
            receive_monotonic_ns=receive,
            response_artifact=artifact,
            future_kinds=kinds,
        )
        self._append_event(
            "model_response_received",
            {
                "request_index": self._pending_request["request_index"],
                "receive_monotonic_ns": receive,
                "response_artifact": artifact,
                "future_kinds": kinds,
            },
        )

    def _validate_future_evidence(self, kind: str, value: Any) -> None:
        if value is None:
            return
        if _is_array_like(value):
            if _as_numpy(value).size == 0:
                raise RecordingContractError(f"empty {kind} future evidence")
            return
        if isinstance(value, (bytes, bytearray, memoryview)):
            if not bytes(value):
                raise RecordingContractError(f"empty {kind} future evidence")
            return
        if isinstance(value, Mapping):
            if "path" in value or "sha256" in value:
                if not isinstance(value.get("path"), (str, Path)) or not isinstance(value.get("sha256"), str):
                    raise RecordingContractError(f"{kind} future reference needs path and SHA-256")
                path = Path(value["path"]).resolve()
                if not path.is_file() or sha256_file(path) != value["sha256"]:
                    raise RecordingContractError(f"{kind} future reference path/hash mismatch")
                if "bytes" in value and path.stat().st_size != value["bytes"]:
                    raise RecordingContractError(f"{kind} future reference byte count mismatch")
                return
            if not value:
                raise RecordingContractError(f"empty {kind} future evidence")
            for child_key, child_value in value.items():
                self._validate_future_evidence(f"{kind}.{child_key}", child_value)
            return
        if isinstance(value, (tuple, list)) and value:
            for index, child in enumerate(value):
                self._validate_future_evidence(f"{kind}[{index}]", child)
            return
        raise RecordingContractError(
            f"{kind} future evidence is neither retained bytes/array nor a hash-checked path"
        )

    def complete_request(self, returned_chunk: Any, executable_chunk: Any) -> None:
        request = self._pending_request
        if request is None or request.get("receive_monotonic_ns") is None:
            raise RecordingContractError("cannot complete a request before its response")
        returned = np.ascontiguousarray(_as_numpy(returned_chunk))
        executable = np.ascontiguousarray(_as_numpy(executable_chunk))
        if returned.ndim != 2 or executable.ndim != 2:
            raise RecordingContractError("returned and executable actions must be rank-two chunks")
        if returned.shape[0] != request["returned_action_horizon"]:
            raise RecordingContractError(
                f"returned action horizon changed: {returned.shape[0]} != {request['returned_action_horizon']}"
            )
        if executable.shape[0] < request["executed_prefix_horizon"]:
            raise RecordingContractError("executable action chunk is shorter than the frozen prefix")
        if returned.shape[1:] != executable.shape[1:]:
            raise RecordingContractError("returned and executable action dimensions differ")
        missing = sorted(set(request["required_future_evidence"]) - set(request["future_kinds"]))
        actions_artifact = self.payloads.write(
            "action_chunks",
            {"returned_action_chunk": returned, "executable_action_chunk": executable},
        )
        request.update(
            action_chunks_artifact=actions_artifact,
            returned_action_shape=list(returned.shape),
            executable_action_shape=list(executable.shape),
        )
        self._pending_request = None
        self._append_event(
            "model_request_completed",
            {
                "request_index": request["request_index"],
                "action_chunks_artifact": actions_artifact,
                "returned_action_shape": list(returned.shape),
                "executable_action_shape": list(executable.shape),
                "missing_future_evidence": missing,
            },
        )
        if missing:
            raise RecordingContractError(
                f"request {request['request_index']} lacks required future evidence: {missing}"
            )

    def record_policy_action(self, action: Any) -> dict[str, Any]:
        if self.finalized:
            raise RecordingContractError("policy action followed finalized attempt")
        if self._pending_request is not None:
            raise RecordingContractError("policy action preceded complete response/action chunk")
        if self._proposed_action is not None:
            raise RecordingContractError("a second policy action was returned before env.step")
        if not self.requests:
            raise RecordingContractError("policy returned an action without a model request")
        request = self.requests[-1]
        offset = self.actions_executed - request["action_step_start"]
        if not 0 <= offset < request["executed_prefix_horizon"]:
            raise RecordingContractError("policy reused or overran the frozen executable prefix")
        array = np.ascontiguousarray(_as_numpy(action)).copy()
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise RecordingContractError("policy action must be one finite numeric vector")
        proposal = {
            "action_step": self.actions_executed + 1,
            "request_index": request["request_index"],
            "chunk_offset": offset,
            "action_identity": _array_identity(array),
            "array": array,
        }
        self._proposed_action = proposal
        self._append_event(
            "policy_action_returned",
            {key: value for key, value in proposal.items() if key != "array"},
        )
        return proposal

    def begin_environment_step(self, executed_action: Any) -> tuple[int, int]:
        proposal = self._proposed_action
        if proposal is None:
            raise RecordingContractError("env.step has no corresponding returned policy action")
        executed = np.ascontiguousarray(_as_numpy(executed_action))
        if (
            executed.shape != proposal["array"].shape
            or executed.dtype != proposal["array"].dtype
            or not np.array_equal(executed, proposal["array"], equal_nan=False)
        ):
            raise RecordingContractError("environment action differs from the exact returned policy action")
        executed_identity = _array_identity(executed)
        if executed_identity != proposal["action_identity"]:
            raise RecordingContractError("executed action byte identity changed after equality check")
        start = time.monotonic_ns()
        self._append_event(
            "environment_step_started",
            {
                "action_step": proposal["action_step"],
                "request_index": proposal["request_index"],
                "chunk_offset": proposal["chunk_offset"],
                "env_step_start_monotonic_ns": start,
                "executed_action_identity": executed_identity,
            },
        )
        return proposal["action_step"], start

    def complete_environment_step(
        self,
        *,
        action_step: int,
        start_monotonic_ns: int,
        success: Mapping[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> None:
        proposal = self._proposed_action
        if proposal is None or proposal["action_step"] != action_step:
            raise RecordingContractError("environment completion does not match its proposed action")
        if action_step != self.actions_executed + 1:
            raise RecordingContractError("environment action steps are not contiguous")
        success = _validated_success_snapshot(success)
        self.actions_executed = action_step
        request = self.requests[proposal["request_index"]]
        request["executed_offsets"].append(proposal["chunk_offset"])
        target = self.identity["command"]
        if success[target] and self.first_success is None:
            self.first_success = {
                "action_step": action_step,
                "requested_relation": target,
                "released": bool(success["released"]),
                "observation_id": f"obs_{action_step:06d}",
            }
            self._append_event("first_success", self.first_success)
        end = time.monotonic_ns()
        self._append_event(
            "environment_step_completed",
            {
                "action_step": action_step,
                "request_index": proposal["request_index"],
                "chunk_offset": proposal["chunk_offset"],
                "env_step_start_monotonic_ns": start_monotonic_ns,
                "env_step_end_monotonic_ns": end,
                "success_predicates": dict(success),
                "terminated": terminated,
                "truncated": truncated,
            },
        )
        self._proposed_action = None

    def _request_execution_receipts(self) -> list[dict[str, Any]]:
        output = []
        for request in self.requests:
            offsets = list(request["executed_offsets"])
            expected = list(range(len(offsets)))
            if offsets != expected:
                raise RecordingContractError(
                    f"request {request['request_index']} executed noncontiguous offsets {offsets}"
                )
            prefix = request["executed_prefix_horizon"]
            returned = request["returned_action_horizon"]
            output.append(
                {
                    "request_index": request["request_index"],
                    "action_step_start": request["action_step_start"],
                    "returned_actions": returned,
                    "eligible_executable_prefix_actions": prefix,
                    "executed_actions": len(offsets),
                    "unused_executable_prefix_actions": prefix - len(offsets),
                    "returned_actions_outside_executable_prefix": returned - prefix,
                    "current_observation_id": request["current_observation_id"],
                    "preceding_observation_id": request["preceding_observation_id"],
                    "future_kinds": request["future_kinds"],
                    "action_chunks_artifact": request.get("action_chunks_artifact"),
                }
            )
        return output

    def finalize(self, stop_reason: str, detail: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self._final_receipt is not None:
            return self._final_receipt
        if stop_reason not in {"action_cap", "safety_abort", "technical_failure"}:
            raise RecordingContractError(f"invalid stop reason: {stop_reason}")
        validation_errors: list[str] = []
        if self.context_reset is None:
            validation_errors.append("missing model context reset")
        if self.environment_contract is None:
            validation_errors.append("missing fixed-duration environment verification")
        if self.physical_reset_calls != 1 or self.runner_reset_calls != 2:
            validation_errors.append("reset count contract not satisfied")
        if len(self.observations) != self.actions_executed + 1:
            validation_errors.append("missing original reset/post-action observation")
        if self._pending_request is not None:
            validation_errors.append("model request lacks a complete response")
        if self._proposed_action is not None:
            validation_errors.append("returned policy action was not executed")
        try:
            request_receipts = self._request_execution_receipts()
        except RecordingContractError as error:
            validation_errors.append(str(error))
            request_receipts = []
        if stop_reason == "action_cap" and self.actions_executed != self.action_cap:
            validation_errors.append("normal completion did not execute exactly 450 actions")
        if stop_reason == "action_cap":
            expected_requests = math.ceil(
                self.action_cap
                / int(self.contract["models"][self.identity["model_config"]]["executed_prefix_horizon"])
            )
            if len(self.requests) != expected_requests:
                validation_errors.append("request count does not cover the exact 450-action schedule")
            if not request_receipts or request_receipts[-1]["executed_actions"] != 2:
                validation_errors.append("final two-action chunk truncation was not observed")
        if stop_reason == "action_cap" and validation_errors:
            stop_reason = "technical_failure"
            detail = {
                "component": "final_validation",
                "error": "; ".join(validation_errors),
                "original_stop_reason": "action_cap",
                **dict(detail or {}),
            }
        behavioral_valid = stop_reason == "action_cap" and not validation_errors
        final_chunk = request_receipts[-1] if request_receipts else None
        receipt = {
            "schema_version": "wmf-forecast-recording-attempt-v1",
            "study_id": self.contract["study_id"],
            "identity": self.identity,
            "process_identity": self.process_identity,
            "stop_reason": stop_reason,
            "behavioral_result_valid": behavioral_valid,
            "right_censored": stop_reason == "safety_abort",
            "technical_invalid": stop_reason == "technical_failure",
            "actions_executed": self.actions_executed,
            "action_cap": self.action_cap,
            "observation_count": len(self.observations),
            "request_count": len(self.requests),
            "first_success": self.first_success,
            "success_configured_as_termination": False,
            "actions_after_first_success": (
                None
                if self.first_success is None
                else self.actions_executed - self.first_success["action_step"]
            ),
            "runner_reset_calls": self.runner_reset_calls,
            "physical_reset_calls": self.physical_reset_calls,
            "context_reset_artifact": (
                self.context_reset["artifact"] if self.context_reset is not None else None
            ),
            "physical_reset_artifact": (
                self.reset_receipt["artifact"] if self.reset_receipt is not None else None
            ),
            "environment_contract": self.environment_contract,
            "request_execution": request_receipts,
            "final_chunk": final_chunk,
            "final_two_action_truncation_recorded": bool(
                final_chunk and final_chunk["executed_actions"] == 2
            ),
            "detail": dict(detail or {}),
            "validation_errors": validation_errors,
            "event_count_before_final": self._sequence,
            "journal_path": str(self.journal_path),
            "journal_tail_sha256_before_final": self._previous_event_sha256,
        }
        self._append_event("attempt_finalized", receipt)
        receipt["event_count"] = self._sequence
        receipt["journal_tail_sha256"] = self._previous_event_sha256
        _atomic_json(self.completion_path, receipt)
        self._final_receipt = receipt
        return receipt

    def mark_technical_failure(self, component: str, error: BaseException | str) -> dict[str, Any]:
        self._fail(component, error)
        assert self._final_receipt is not None
        return self._final_receipt


def _public_termination_terms(terminations: Any) -> dict[str, Any]:
    names: set[str] = set()
    for cls in type(terminations).__mro__:
        names.update(name for name in vars(cls) if not name.startswith("_"))
    names.update(name for name in vars(terminations) if not name.startswith("_"))
    return {
        name: getattr(terminations, name)
        for name in sorted(names)
        if not callable(getattr(terminations, name, None)) and getattr(terminations, name, None) is not None
    }


def _default_timeout_only_termination_factory() -> type:
    import isaaclab.envs.mdp as mdp
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.utils import configclass

    @configclass
    class WMFFixedDurationTermination:
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

    return WMFFixedDurationTermination


def configure_fixed_duration_task(
    task_cls: type,
    *,
    termination_cfg_factory: Callable[[], type] = _default_timeout_only_termination_factory,
) -> type:
    """Replace task termination at import time, before RoboLab registration.

    Concrete fixture files should call this decorator/helper on their new
    workshop Task class.  It intentionally does not mutate an environment that
    has already been constructed.
    """

    termination_type = termination_cfg_factory()
    terms = _public_termination_terms(termination_type())
    if set(terms) != {"time_out"}:
        raise RecordingContractError(
            f"fixed-duration task must expose only time_out, got {sorted(terms)}"
        )
    timeout = terms["time_out"]
    if getattr(timeout, "time_out", None) is not True:
        raise RecordingContractError("time_out termination is not marked as truncation")
    task_cls.terminations = termination_type
    task_cls.episode_length_s = 30
    task_cls._wmf_success_is_measurement_only = True
    task_cls._wmf_action_cap = 450
    return task_cls


def assert_fixed_duration_environment(env: Any, env_cfg: Any) -> dict[str, Any]:
    """Verify the concrete environment before its first physical reset."""

    terms = _public_termination_terms(env_cfg.terminations)
    if set(terms) != {"time_out"}:
        raise RecordingContractError(
            f"constructed environment has non-timeout termination terms: {sorted(terms)}"
        )
    if getattr(terms["time_out"], "time_out", None) is not True:
        raise RecordingContractError("constructed timeout term is not a truncation")
    dt = float(env_cfg.sim.dt)
    decimation = int(env_cfg.decimation)
    render_interval = int(env_cfg.sim.render_interval)
    control_period = dt * decimation
    max_steps = int(env.max_episode_length)
    if not math.isclose(dt, 1 / 120, rel_tol=0.0, abs_tol=1e-12):
        raise RecordingContractError(f"physics dt changed: {dt}")
    if decimation != 8 or render_interval != 8:
        raise RecordingContractError("control decimation/render interval changed")
    if not math.isclose(control_period, 1 / 15, rel_tol=0.0, abs_tol=1e-12):
        raise RecordingContractError(f"control period changed: {control_period}")
    if int(env_cfg.episode_length_s) != 30 or max_steps != 450:
        raise RecordingContractError("constructed environment is not exactly 450 actions")
    return {
        "passed": True,
        "termination_terms": ["time_out"],
        "success_termination_present": False,
        "episode_length_s": 30,
        "max_episode_length": max_steps,
        "physics_dt_s": dt,
        "decimation": decimation,
        "render_interval": render_interval,
        "control_period_s": control_period,
        "nominal_control_hz": 1 / control_period,
        "forecast_alignment_inferred": False,
    }


def default_future_extractor(response: Any) -> dict[str, Any]:
    """Extract only explicit future payloads; never synthesize a future."""

    if not isinstance(response, Mapping):
        return {}
    explicit = response.get("future_evidence")
    if isinstance(explicit, Mapping):
        return dict(explicit)
    output: dict[str, Any] = {}
    for key, value in response.items():
        lowered = str(key).lower()
        if "latent" in lowered or lowered == "video_pred":
            output.setdefault("latent", value)
        elif (
            lowered in {"video", "frames", "decoded_future", "future_video"}
            or "decoded" in lowered
        ):
            output.setdefault("decoded", value)
    return output


def _client_state(client: Any) -> dict[str, Any]:
    chunks = getattr(client, "_chunks", {})
    counters = getattr(client, "_counters", {})
    sessions = getattr(client, "_env_session_id", {})
    return {
        "chunk_env_ids": sorted(int(key) for key in chunks),
        "counter_env_ids": sorted(int(key) for key in counters),
        "session_ids": sorted(str(value) for value in sessions.values()),
    }


class RecordingClientMixin:
    """Cooperative mixin that instruments the official model client hooks.

    Use it first in MRO, e.g. ``class RecordedN3(RecordingClientMixin,
    ExactN3Client): pass``.  The normal client code still performs extraction,
    packing, transport, unpacking and postprocessing; this mixin snapshots each
    boundary without recomputing any preprocessing.
    """

    def attach_forecast_recorder(
        self,
        recorder: ForecastRecordingAdapter,
        *,
        future_extractor: Callable[[Any], Mapping[str, Any]] = default_future_extractor,
    ) -> None:
        if hasattr(self, "_wmf_recorder"):
            raise RecordingContractError("client recorder was already attached")
        self._wmf_recorder = recorder
        self._wmf_future_extractor = future_extractor
        self._wmf_extracted: Any = None
        self._wmf_returned_chunk: Any = None

    def reset_for_recorded_episode(
        self,
        reset_attestor: Callable[[], Mapping[str, Any]],
    ) -> None:
        recorder = getattr(self, "_wmf_recorder", None)
        if recorder is None:
            raise RecordingContractError("attach recorder before resetting client context")
        before = _client_state(self)
        try:
            super().reset(env_id=None)
            external = dict(reset_attestor())
            receipt = {
                **external,
                "client_state_before": before,
                "client_state_after": _client_state(self),
            }
            recorder.record_context_reset(receipt)
        except BaseException as error:
            recorder.mark_technical_failure("model_context_reset", error)
            raise

    def _extract_observation(self, raw_obs: Any, *, env_id: int = 0) -> Any:
        extracted = super()._extract_observation(raw_obs, env_id=env_id)
        self._wmf_extracted = extracted
        return extracted

    def _pack_request(self, extracted_obs: Any, instruction: str) -> Any:
        request = super()._pack_request(extracted_obs, instruction)
        if self._wmf_extracted is None:
            raise RecordingContractError("client preprocessing output was not captured")
        self._wmf_recorder.begin_request(self._wmf_extracted, request)
        return request

    def _query_server(self, request: Any) -> Any:
        self._wmf_recorder.mark_request_sent()
        response = super()._query_server(request)
        future = dict(self._wmf_future_extractor(response))
        self._wmf_recorder.record_raw_response(response, future)
        return response

    def _unpack_response(self, response: Any) -> Any:
        returned = super()._unpack_response(response)
        self._wmf_returned_chunk = np.ascontiguousarray(_as_numpy(returned)).copy()
        return returned

    def _postprocess_chunk(self, chunk: Any) -> Any:
        executable = super()._postprocess_chunk(chunk)
        if self._wmf_returned_chunk is None:
            raise RecordingContractError("client postprocessing preceded returned action capture")
        self._wmf_recorder.complete_request(self._wmf_returned_chunk, executable)
        self._wmf_returned_chunk = None
        return executable

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict[str, Any]:
        if env_id != 0:
            raise RecordingContractError("forecast study requires one isolated environment")
        recorder = getattr(self, "_wmf_recorder", None)
        if recorder is None:
            raise RecordingContractError("recorded client used before recorder attachment")
        try:
            result = super().infer(obs, instruction, env_id=env_id)
            recorder.record_policy_action(result["action"])
            self._wmf_extracted = None
            return result
        except BaseException as error:
            recorder.mark_technical_failure("policy_inference", error)
            raise


class FixedDurationEnvProxy:
    """One-environment proxy enforcing reset and exact-action recording."""

    def __init__(
        self,
        env: Any,
        env_cfg: Any,
        recorder: ForecastRecordingAdapter,
        *,
        state_sampler: Callable[[Any], Mapping[str, Any]],
        clock_sampler: Callable[[Any, str, int], Mapping[str, Any]],
        success_sampler: Callable[[Any], Mapping[str, Any]],
        reset_attestor: Callable[
            [Any, Any, Any],
            Mapping[str, Any] | tuple[Any, Any, Mapping[str, Any]],
        ],
        safety_checker: Callable[[Any, Any, Any], Mapping[str, Any] | str | None] | None = None,
        env_id: int = 0,
    ) -> None:
        self._env = env
        self._env_cfg = env_cfg
        self.recorder = recorder
        self.state_sampler = state_sampler
        self.clock_sampler = clock_sampler
        self.success_sampler = success_sampler
        self.reset_attestor = reset_attestor
        self.safety_checker = safety_checker
        self.env_id = env_id
        if env_id != 0 or int(getattr(env, "num_envs", 1)) != 1:
            raise RecordingContractError("forecast recorder requires one isolated env_id=0")
        contract_receipt = assert_fixed_duration_environment(env, env_cfg)
        recorder.record_environment_contract(contract_receipt)
        self._cached_reset: tuple[Any, Any] | None = None
        self._aborted = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def active_env_ids(self) -> list[int]:
        return [] if self._aborted else list(self._env.active_env_ids)

    @property
    def all_terminated(self) -> bool:
        return self._aborted or bool(self._env.all_terminated)

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        if self.recorder.actions_executed or self.recorder.finalized:
            raise RecordingContractError("physical reset after behavioral execution is prohibited")
        if self._cached_reset is not None:
            self.recorder.record_idempotent_runner_reset()
            return self._cached_reset
        try:
            result = self._env.reset(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RecordingContractError("environment reset did not return (observation, info)")
            obs, info = result
            attested = self.reset_attestor(self._env, obs, info)
            if isinstance(attested, tuple):
                if len(attested) != 3 or not isinstance(attested[2], Mapping):
                    raise RecordingContractError(
                        "reset attestor tuple must be (settled_obs, settled_info, receipt)"
                    )
                obs, info, raw_attestation = attested
            elif isinstance(attested, Mapping):
                raw_attestation = attested
            else:
                raise RecordingContractError("reset attestor returned an unsupported value")
            success = _validated_success_snapshot(self.success_sampler(self._env))
            attestation = dict(raw_attestation)
            attestation.setdefault("left_success", success["left"])
            attestation.setdefault("right_success", success["right"])
            if (
                attestation["left_success"] is not success["left"]
                or attestation["right_success"] is not success["right"]
            ):
                raise RecordingContractError("reset attestation disagrees with live success predicates")
            self.recorder.record_physical_reset(
                obs,
                clock=self.clock_sampler(self._env, "settled_reset", 0),
                state=self.state_sampler(self._env),
                attestation=attestation,
                env_id=self.env_id,
            )
            self._cached_reset = (obs, info)
            return self._cached_reset
        except BaseException as error:
            self.recorder.mark_technical_failure("physical_reset", error)
            raise

    def step(self, action: Any) -> Any:
        if self.recorder.runner_reset_calls != 2 or self._cached_reset is None:
            raise RecordingContractError("env.step preceded the two-call idempotent reset contract")
        if self.recorder.actions_executed >= self.recorder.action_cap:
            raise RecordingContractError("env.step attempted beyond action 450")
        executed = _one_env_array(action, self.env_id, field="environment action")
        try:
            action_step, started = self.recorder.begin_environment_step(executed)
            result = self._env.step(action)
            if not isinstance(result, tuple) or len(result) != 5:
                raise RecordingContractError("environment step did not return five values")
            obs, reward, terminated, truncated, info = result
            success = _validated_success_snapshot(self.success_sampler(self._env))
            self.recorder.complete_environment_step(
                action_step=action_step,
                start_monotonic_ns=started,
                success=success,
                terminated=_boolean_vector_any(terminated),
                truncated=_boolean_vector_any(truncated),
            )
            self.recorder.record_observation(
                obs,
                clock=self.clock_sampler(self._env, "post_action", action_step),
                state=self.state_sampler(self._env),
                control_step=action_step,
                phase="post_action",
                env_id=self.env_id,
            )
            safety = self.safety_checker(self._env, obs, info) if self.safety_checker else None
            if safety is not None:
                detail = dict(safety) if isinstance(safety, Mapping) else {"reason": str(safety)}
                self.recorder.finalize("safety_abort", detail)
                self._aborted = True
                return result
            done = _boolean_vector_any(terminated) or _boolean_vector_any(truncated)
            if action_step < self.recorder.action_cap and (done or bool(self._env.all_terminated)):
                error = RecordingContractError(
                    f"environment terminated/froze early at action {action_step}; success may still be registered"
                )
                self.recorder.mark_technical_failure("early_environment_termination", error)
                self._aborted = True
                raise error
            if action_step == self.recorder.action_cap:
                if not done or not bool(self._env.all_terminated):
                    error = RecordingContractError("action 450 did not trigger the sole timeout termination")
                    self.recorder.mark_technical_failure("timeout_termination", error)
                    self._aborted = True
                    raise error
                self.recorder.finalize("action_cap")
                self._aborted = True
            return result
        except BaseException as error:
            self.recorder.mark_technical_failure("environment_step", error)
            self._aborted = True
            raise

    def close(self) -> Any:
        if not self.recorder.finalized:
            self.recorder.mark_technical_failure(
                "environment_close",
                f"environment closed after {self.recorder.actions_executed} of 450 actions",
            )
        return self._env.close()
