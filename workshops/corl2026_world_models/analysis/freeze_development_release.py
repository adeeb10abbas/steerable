#!/usr/bin/env python3
"""Derive and validate the WMF development-to-confirmation release freeze.

This module intentionally has no model or simulator imports.  It consumes the
immutable receipts produced by the workshop runtimes and derives physical
alignment only from three native sources:

* the model runtime's explicit generated-target time offsets;
* the recorder's native physics/control counters; and
* the original camera's native capture timestamps and frame identities.

Generated frame number, action number, conditioning FPS, and presentation
video FPS are never accepted as timing evidence.  The final release fails
closed until all 16 development cells per qualified model, empirical duplicate
labels from two independent raters, final adjudication, an explicit usability
decision, and measured resource receipts are present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence


STUDY_ID = "WMF-ABLATION-001"
EVIDENCE_SCHEMA = "wmf-development-release-evidence-v1"
TIMING_SCHEMA = "wmf-native-generated-target-timing-v1"
RESOURCE_SCHEMA = "wmf-development-resource-measurement-v1"
CROP_SCHEMA = "wmf-camera-crop-contract-v1"
MAPPING_SCHEMA = "wmf-forecast-physical-alignment-receipt-v1"
ALIGNMENT_SCHEMA = "wmf-forecast-alignment-contract-v1"
RELEASE_SCHEMA = "wmf-development-confirmation-release-freeze-v1"
DEVELOPMENT_SUMMARY_SCHEMA = "wmf-forecast-development-label-noise-v1"
FINAL_CONSENSUS_SCHEMA = "wmf-forecast-final-consensus-v2"

REQUEST_SAMPLING_SEED = 2026091302
REQUEST_SAMPLE_CAP = 4
ANALYSIS_SEED = 2026091301
BOOTSTRAP_RESAMPLES = 10_000
REQUEST_SAMPLING_ALGORITHM = (
    "Within each episode, sort eligible source_request_id values by "
    "SHA256(UTF8(seed) || NUL || UTF8(episode_id) || NUL || "
    "UTF8(source_request_id)), breaking digest ties by source_request_id; "
    "select the first min(4, eligible_count)."
)
MOVEMENT_DISAGREEMENT_DEFINITION = (
    "Euclidean distance between the two raters' cube-minus-bowl image-plane "
    "vectors, with each vector divided by that image's pixel diagonal."
)
MOVEMENT_QUANTILE_METHOD = "linear interpolation at p=0.95 (Hyndman-Fan type 7)"
TIMING_QUALIFIER_PATH = Path(__file__).with_name("qualify_forecast_timing.py")

SHA_RE = re.compile(r"[0-9a-f]{64}")
MODEL_LIMITS = {
    "N3": {
        "returned_action_horizon": 32,
        "unchanged_executed_prefix_horizon": 32,
        "request_count": 15,
        "action_space": "joint_pos",
        "seed_semantics": "matched effective policy seed per layout block",
        "temporal_context": "isolated full temporal/cache reset before every episode",
        "cell_schema": "wmf-n3-behavioral-development-cell-v1",
        "request_schema": "wmf-n3-behavioral-server-request-v1",
    },
    "D1": {
        "returned_action_horizon": 24,
        "unchanged_executed_prefix_horizon": 8,
        "request_count": 57,
        "action_space": "joint_pos",
        "seed_semantics": "fixed effective model noise 1140; requests are not independent draws",
        "temporal_context": "isolated official two-rank full temporal/cache reset before every episode",
        "cell_schema": "wmf-d1-behavioral-development-cell-v1",
        "request_schema": "wmf-d1-request-receipt-v1",
    },
}
BRANCH_MODELS = {
    "full_two_model": ("N3", "D1"),
    "reduced_n3": ("N3",),
    "reduced_d1": ("D1",),
}


class FreezeError(RuntimeError):
    """The supplied evidence cannot authorize confirmation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def recorder_canonical_bytes(value: Any) -> bytes:
    """Canonical encoding used by recording_adapter.py's JSONL hash chain."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_sha256(path: Path) -> str:
    """Match forecast_annotation_workflow.py's file/directory identity."""

    candidate = Path(path)
    require(not candidate.is_symlink(), f"artifact is a symlink: {candidate}")
    path = candidate.resolve()
    if path.is_file():
        return sha256_file(path)
    require(path.is_dir(), f"artifact does not exist: {path}")
    candidates = sorted(path.rglob("*"))
    require(not any(item.is_symlink() for item in candidates),
            f"artifact directory contains a symlink: {path}")
    files = {
        str(item.relative_to(path)): sha256_file(item)
        for item in candidates if item.is_file()
    }
    require(bool(files), f"artifact directory is empty: {path}")
    return sha256_bytes(canonical_bytes(files))


def sign_document(value: Mapping[str, Any]) -> dict[str, Any]:
    require("payload_sha256" not in value, "document is already signed")
    result = dict(value)
    result["payload_sha256"] = sha256_bytes(canonical_bytes(result))
    return result


def verify_signed(value: Mapping[str, Any], label: str) -> None:
    observed = value.get("payload_sha256")
    require(isinstance(observed, str) and SHA_RE.fullmatch(observed) is not None,
            f"{label} lacks a valid payload_sha256")
    unsigned = dict(value)
    unsigned.pop("payload_sha256")
    require(sha256_bytes(canonical_bytes(unsigned)) == observed,
            f"{label} payload hash mismatch")


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreezeError(f"{label} is not readable JSON: {path}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    require(type(value) in (int, float) and math.isfinite(float(value)),
            f"{label} must be finite")
    number = float(value)
    require(number > 0 if positive else number >= 0,
            f"{label} must be {'positive' if positive else 'nonnegative'}")
    return number


def _descriptor(value: Any, label: str, *, base: Path | None = None) -> tuple[dict[str, Any], Path]:
    require(isinstance(value, Mapping), f"{label} descriptor is missing")
    raw_path = value.get("path")
    digest = value.get("sha256")
    require(isinstance(raw_path, str) and raw_path, f"{label} path is missing")
    require(isinstance(digest, str) and SHA_RE.fullmatch(digest) is not None,
            f"{label} SHA-256 is invalid")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        require(base is not None, f"{label} relative path has no base")
        candidate = base / candidate
    require(not candidate.is_symlink(), f"{label} is a symlink")
    path = candidate.resolve()
    require(path.is_file(), f"{label} file is missing")
    require(sha256_file(path) == digest, f"{label} file hash mismatch")
    if "bytes" in value:
        require(type(value["bytes"]) is int and value["bytes"] == path.stat().st_size,
                f"{label} byte count mismatch")
    return dict(value), path


def _file_descriptor(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": display_path if display_path is not None else str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _verify_journal(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for sequence, line in enumerate(handle):
                require(bool(line.strip()), "recording journal contains a blank line")
                row = json.loads(line)
                require(isinstance(row, dict), "recording journal row is not an object")
                require(row.get("sequence") == sequence, "recording journal sequence changed")
                require(row.get("previous_event_sha256") == previous,
                        "recording journal hash chain changed")
                observed = row.get("event_sha256")
                require(isinstance(observed, str) and SHA_RE.fullmatch(observed) is not None,
                        "recording journal event hash is invalid")
                base = dict(row)
                base.pop("event_sha256")
                require(sha256_bytes(recorder_canonical_bytes(base)) == observed,
                        "recording journal event content changed")
                rows.append(row)
                previous = observed
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreezeError(f"recording journal is unreadable: {path}") from error
    require(rows, "recording journal is empty")
    return rows, str(previous)


def _events(rows: Sequence[Mapping[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [dict(row["payload"]) for row in rows if row.get("kind") == kind]


def _planned_development_cells(path: Path, models: Sequence[str]) -> dict[str, set[str]]:
    output = {model: set() for model in models}
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                model = row.get("model_config")
                if row.get("phase") == "development" and model in output and row.get("selected") == "true":
                    output[model].add(str(row.get("cell_id")))
    except OSError as error:
        raise FreezeError("planned-cell CSV is unreadable") from error
    for model, cells in output.items():
        require(len(cells) == 16, f"planned-cell CSV does not contain 16 development cells for {model}")
    return output


def _validate_ablation_spec(spec: Mapping[str, Any]) -> None:
    require(spec.get("spec_id") == STUDY_ID, "ablation spec identity changed")
    require(spec.get("core_episode_ceiling") == 232, "ablation core count changed")
    analysis = spec.get("analysis")
    require(isinstance(analysis, Mapping), "ablation analysis section is missing")
    expected = {
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "analysis_seed": ANALYSIS_SEED,
        "request_sample_cap_per_episode": REQUEST_SAMPLE_CAP,
        "request_sampling_seed": REQUEST_SAMPLING_SEED,
    }
    for key, wanted in expected.items():
        require(analysis.get(key) == wanted, f"ablation analysis {key} changed")
    models = spec.get("models")
    require(isinstance(models, Mapping), "ablation model section is missing")
    require(models.get("N3", {}).get("returned_actions") == 32, "N3 action horizon changed")
    require(models.get("N3", {}).get("executed_prefix_cap") == 32, "N3 prefix changed")
    require(models.get("D1", {}).get("executed_prefix_cap") == 8, "D1 prefix changed")


def _validate_server_request(
    path: Path, *, descriptor: Mapping[str, Any], model: str,
    cell: Mapping[str, Any], request_index: int,
) -> tuple[str, dict[str, Any]]:
    receipt = load_json(path, f"{model} server request {request_index}")
    limits = MODEL_LIMITS[model]
    require(receipt.get("schema_version") == limits["request_schema"],
            f"{model} request {request_index} schema changed")
    require(receipt.get("request_index") == request_index,
            f"{model} request indices are not contiguous")
    if model == "N3":
        require(receipt.get("status") == "passed", "N3 request did not pass")
        require(receipt.get("study_id") == STUDY_ID, "N3 request study changed")
        require(receipt.get("cell_id") == cell["cell_id"], "N3 request cell binding changed")
        require(receipt.get("action_step_start") == request_index * 32,
                "N3 request start-action binding changed")
        require(receipt.get("behavioral_model_request") is True,
                "N3 request is not behavioral")
        shape = receipt.get("decoded_future_shape")
        require(isinstance(shape, list) and len(shape) == 4 and shape[0] > 1,
                "N3 decoded future shape is unavailable")
    else:
        require(receipt.get("configuration_id") == "D1", "D1 request configuration changed")
        require(receipt.get("prompt") == cell.get("prompt"), "D1 request prompt changed")
        require(receipt.get("official_action_path") == "GrootSimPolicy.lazy_joint_forward_causal",
                "D1 request is not the official conditional path")
        require(receipt.get("custom_s2_used") is False and receipt.get("patched_s1_used") is False,
                "D1 request silently changed action path")
        require(receipt.get("effective_official_model_noise_seed") == 1140,
                "D1 fixed noise changed")
        offline = receipt.get("offline_decode")
        require(isinstance(offline, Mapping) and offline.get("performed") is True,
                "D1 decoded future is unavailable")
    return str(descriptor["sha256"]), receipt


def _resource_measurement(path: Path, *, model: str, cell_id: str) -> dict[str, Any]:
    receipt = load_json(path, f"resource receipt for {cell_id}")
    require(receipt.get("schema_version") == RESOURCE_SCHEMA, "resource receipt schema changed")
    verify_signed(receipt, f"resource receipt for {cell_id}")
    require(receipt.get("study_id") == STUDY_ID and receipt.get("model_id") == model,
            "resource receipt study/model binding changed")
    require(receipt.get("cell_id") == cell_id and receipt.get("measurement_complete") is True,
            "resource receipt is incomplete or belongs to another cell")
    values = {
        "episode_wall_seconds": _finite(receipt.get("episode_wall_seconds"), "episode wall seconds", positive=True),
        "inference_wall_seconds_total": _finite(receipt.get("inference_wall_seconds_total"), "inference wall seconds", positive=True),
        "raw_recording_bytes": int(_finite(receipt.get("raw_recording_bytes"), "raw recording bytes", positive=True)),
        "peak_gpu_allocated_bytes": int(_finite(receipt.get("peak_gpu_allocated_bytes"), "peak GPU allocated bytes", positive=True)),
        "peak_gpu_reserved_bytes": int(_finite(receipt.get("peak_gpu_reserved_bytes"), "peak GPU reserved bytes", positive=True)),
        "gpu_count": receipt.get("gpu_count"),
    }
    require(type(values["gpu_count"]) is int and values["gpu_count"] > 0,
            "resource receipt gpu_count is invalid")
    require(values["peak_gpu_reserved_bytes"] >= values["peak_gpu_allocated_bytes"],
            "reserved GPU memory is below allocated GPU memory")
    return values


def _validate_cell(
    entry: Mapping[str, Any], *, model: str, expected_cell_id: str,
    evidence_base: Path,
) -> dict[str, Any]:
    require(set(entry) == {"cell_receipt", "server_request_receipts", "resource_receipt"},
            f"{expected_cell_id} evidence keys changed")
    _, cell_path = _descriptor(entry["cell_receipt"], f"{expected_cell_id} cell receipt", base=evidence_base)
    cell = load_json(cell_path, f"{expected_cell_id} cell receipt")
    limits = MODEL_LIMITS[model]
    require(cell.get("schema_version") == limits["cell_schema"], f"{expected_cell_id} schema changed")
    require(cell.get("status") == "passed" and cell.get("study_id") == STUDY_ID,
            f"{expected_cell_id} did not pass")
    require(cell.get("cell_id") == expected_cell_id and cell.get("model_config") == model,
            f"{expected_cell_id} identity changed")
    require(cell.get("actions_executed") == 450 and cell.get("observation_count") == 451,
            f"{expected_cell_id} is not a complete 450-action recording")
    require(cell.get("behavioral_model_request_count") == limits["request_count"]
            and cell.get("behavioral_episode_count") == 1,
            f"{expected_cell_id} behavioral counts changed")
    require(cell.get("generation_qualification_request_count") == 0,
            f"{expected_cell_id} mixes generation qualification with behavior")

    completion_descriptor, completion_path = _descriptor(
        cell.get("adapter_completion"), f"{expected_cell_id} adapter completion"
    )
    completion = load_json(completion_path, f"{expected_cell_id} adapter completion")
    require(completion.get("schema_version") == "wmf-forecast-recording-attempt-v1",
            f"{expected_cell_id} recording schema changed")
    require(completion.get("behavioral_result_valid") is True
            and completion.get("technical_invalid") is False
            and completion.get("right_censored") is False
            and completion.get("stop_reason") == "action_cap",
            f"{expected_cell_id} recording is not valid complete")
    require(completion.get("actions_executed") == 450
            and completion.get("observation_count") == 451
            and completion.get("request_count") == limits["request_count"],
            f"{expected_cell_id} adapter counts changed")
    identity = completion.get("identity")
    require(isinstance(identity, Mapping) and identity.get("cell_id") == expected_cell_id
            and identity.get("stage") == "development" and identity.get("model_config") == model,
            f"{expected_cell_id} adapter identity changed")
    execution = completion.get("request_execution")
    require(isinstance(execution, list) and len(execution) == limits["request_count"],
            f"{expected_cell_id} request execution inventory changed")
    starts = [row.get("action_step_start") for row in execution if isinstance(row, Mapping)]
    require(starts == [index * limits["unchanged_executed_prefix_horizon"] for index in range(limits["request_count"])],
            f"{expected_cell_id} request start actions changed")
    expected_executed = [limits["unchanged_executed_prefix_horizon"]] * limits["request_count"]
    expected_executed[-1] = 2
    require([row.get("executed_actions") for row in execution] == expected_executed,
            f"{expected_cell_id} executed request prefixes changed")

    journal_descriptor, journal_path = _descriptor(
        cell.get("adapter_journal"), f"{expected_cell_id} adapter journal"
    )
    require(Path(str(completion.get("journal_path", ""))).resolve() == journal_path,
            f"{expected_cell_id} completion/journal path binding changed")
    journal, tail = _verify_journal(journal_path)
    require(len(journal) == completion.get("event_count"), f"{expected_cell_id} journal count changed")
    require(tail == completion.get("journal_tail_sha256"), f"{expected_cell_id} journal tail changed")
    if "event_count" in journal_descriptor:
        require(journal_descriptor["event_count"] == len(journal), f"{expected_cell_id} journal descriptor count changed")
    if "tail_sha256" in journal_descriptor:
        require(journal_descriptor["tail_sha256"] == tail, f"{expected_cell_id} journal descriptor tail changed")

    observations = _events(journal, "observation_captured")
    requests = _events(journal, "model_request_packed")
    require(len(observations) == 451 and len(requests) == limits["request_count"],
            f"{expected_cell_id} journal observation/request counts changed")
    for control_step, observation in enumerate(observations):
        require(observation.get("observation_id") == f"obs_{control_step:06d}"
                and observation.get("control_step") == control_step,
                f"{expected_cell_id} observation schedule changed")
    for index, request in enumerate(requests):
        require(request.get("request_index") == index
                and request.get("action_step_start") == starts[index]
                and request.get("current_observation_id") == f"obs_{starts[index]:06d}",
                f"{expected_cell_id} request-to-observation binding changed")

    server_entries = entry.get("server_request_receipts")
    require(isinstance(server_entries, list) and len(server_entries) == limits["request_count"],
            f"{expected_cell_id} lacks every official request receipt")
    if model == "D1":
        require(cell.get("server_request_receipts") == server_entries,
                f"{expected_cell_id} D1 server request inventory differs from cell receipt")
    request_hashes: list[str] = []
    request_receipts: list[dict[str, Any]] = []
    for index, raw_descriptor in enumerate(server_entries):
        descriptor, request_path = _descriptor(
            raw_descriptor, f"{expected_cell_id} request {index}", base=evidence_base
        )
        request_hash, request_receipt = _validate_server_request(
            request_path, descriptor=descriptor, model=model, cell=cell, request_index=index
        )
        request_hashes.append(request_hash)
        request_receipts.append(request_receipt)

    resource_descriptor, resource_path = _descriptor(
        entry.get("resource_receipt"), f"{expected_cell_id} resource receipt", base=evidence_base
    )
    resource = _resource_measurement(resource_path, model=model, cell_id=expected_cell_id)
    return {
        "cell_id": expected_cell_id,
        "cell_receipt": _file_descriptor(cell_path),
        "adapter_completion": completion_descriptor,
        "adapter_journal": journal_descriptor,
        "request_receipt_sha256s": request_hashes,
        "request_receipts": request_receipts,
        "observations": observations,
        "request_execution": [dict(row) for row in execution],
        "resource_receipt": resource_descriptor,
        "resource": resource,
    }


def _validate_crop(model_evidence: Mapping[str, Any], *, model: str, base: Path) -> dict[str, Any]:
    _, path = _descriptor(model_evidence.get("camera_crop_contract"), f"{model} camera crop", base=base)
    value = load_json(path, f"{model} camera crop contract")
    require(value.get("schema_version") == CROP_SCHEMA, f"{model} crop schema changed")
    verify_signed(value, f"{model} camera crop contract")
    require(value.get("study_id") == STUDY_ID and value.get("model_id") == model,
            f"{model} crop contract identity changed")
    require(value.get("status") == "qualified_from_original_camera_pixels",
            f"{model} crop contract is not qualified")
    for key in ("camera_id", "camera_crop_id", "crop_operation"):
        require(isinstance(value.get(key), str) and value[key], f"{model} crop {key} is missing")
    for key in ("image_width_px", "image_height_px"):
        require(type(value.get(key)) is int and value[key] > 0, f"{model} crop {key} is invalid")
    require(value.get("simulator_state_render_used") is False,
            f"{model} crop contract uses a simulator-state render")
    return {**value, "file_sha256": sha256_file(path), "path": str(path)}


def _validate_generated_timing(
    model_evidence: Mapping[str, Any], *, model: str, base: Path,
    request_hashes: Sequence[str], request_receipts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], Path]:
    _, path = _descriptor(model_evidence.get("generated_target_timing_receipt"), f"{model} generated timing", base=base)
    value = load_json(path, f"{model} generated timing receipt")
    require(value.get("schema_version") == TIMING_SCHEMA, f"{model} generated timing schema changed")
    verify_signed(value, f"{model} generated timing receipt")
    require(value.get("study_id") == STUDY_ID and value.get("model_id") == model,
            f"{model} generated timing identity changed")
    require(value.get("status") == "qualified_from_native_runtime_metadata",
            f"{model} generated timing was not natively qualified")
    require(value.get("time_source_kind") == "native_runtime_exposed_target_offsets",
            f"{model} timing does not come from native exposed target offsets")
    require(value.get("presentation_video_fps_used") is False
            and value.get("conditioning_fps_used_as_target_timing") is False
            and value.get("generated_frame_index_interpreted_as_action_index") is False,
            f"{model} timing uses a prohibited FPS/frame/action inference")
    require(value.get("clock_bridge") == "elapsed physical seconds from request current original-camera capture",
            f"{model} timing clock bridge changed")
    source_field = value.get("native_runtime_field")
    require(isinstance(source_field, str)
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", source_field) is not None,
            f"{model} native runtime timing field is missing")
    require(value.get("source_request_receipt_sha256s") == list(request_hashes),
            f"{model} timing receipt does not bind every development request receipt")
    targets = value.get("generated_targets")
    require(isinstance(targets, list) and targets, f"{model} generated targets are empty")
    indexes: list[int] = []
    times: list[float] = []
    for row in targets:
        require(isinstance(row, Mapping), f"{model} generated target row is invalid")
        require(set(row) == {
            "generated_frame_index", "target_physical_time_s", "native_runtime_field"
        }, f"{model} generated target row fields changed")
        index = row.get("generated_frame_index")
        require(type(index) is int and index >= 0, f"{model} generated frame index is invalid")
        target = _finite(row.get("target_physical_time_s"), f"{model} target time")
        require(row.get("native_runtime_field") == source_field,
                f"{model} target row is not bound to the native timing field")
        indexes.append(index)
        times.append(target)
    require(indexes == sorted(set(indexes)), f"{model} generated frame indices are not unique/sorted")
    require(times == sorted(times) and len(set(times)) == len(times),
            f"{model} generated target times are not strictly increasing")
    require(any(value > 0 for value in times), f"{model} has no strictly positive exposed target")
    exposed_targets = [
        {
            "generated_frame_index": row["generated_frame_index"],
            "target_physical_time_s": row["target_physical_time_s"],
        }
        for row in targets
    ]
    require(len(request_receipts) == len(request_hashes),
            f"{model} native timing request inventory is incomplete")
    sidecar_mode = value.get("binding_mode") == "immutable_request_receipt_native_clock_sidecar"
    if sidecar_mode:
        spec = importlib.util.spec_from_file_location(
            "wmf_validate_forecast_timing_sidecar", TIMING_QUALIFIER_PATH
        )
        require(spec is not None and spec.loader is not None,
                "forecast timing sidecar validator is unavailable")
        qualifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qualifier)
        try:
            qualifier.validate_development_timing(
                path,
                sha256_file(path),
                expected_model=model,
                expected_request_hashes=request_hashes,
            )
        except Exception as error:
            raise FreezeError(f"{model} immutable timing sidecar failed deep validation: {error}") from error
        require(value.get("old_request_receipts_modified") is False,
                f"{model} timing sidecar claims old request receipts were modified")
        require(source_field == "request_timing_sidecar.generated_targets",
                f"{model} timing sidecar pretends metadata existed in immutable request receipts")
    for index, receipt in enumerate(request_receipts):
        if model == "N3":
            decoded_shape = receipt.get("decoded_future_shape")
            require(isinstance(decoded_shape, list) and len(decoded_shape) == 4
                    and type(decoded_shape[0]) is int and decoded_shape[0] > 0,
                    f"{model} request {index} decoded frame count is invalid")
            decoded_frame_count = decoded_shape[0]
        else:
            offline = receipt.get("offline_decode")
            decoded_rgb = offline.get("decoded_rgb") if isinstance(offline, Mapping) else None
            decoded_tensor = offline.get("decoded_tensor") if isinstance(offline, Mapping) else None
            rgb_shape = decoded_rgb.get("shape") if isinstance(decoded_rgb, Mapping) else None
            tensor_shape = decoded_tensor.get("shape") if isinstance(decoded_tensor, Mapping) else None
            require(isinstance(rgb_shape, list) and len(rgb_shape) == 4
                    and type(rgb_shape[0]) is int and rgb_shape[0] > 0,
                    f"{model} request {index} decoded RGB frame count is invalid")
            require(isinstance(tensor_shape, list) and len(tensor_shape) == 5
                    and type(tensor_shape[2]) is int and tensor_shape[2] == rgb_shape[0],
                    f"{model} request {index} decoded tensor/RGB frame counts differ")
            decoded_frame_count = rgb_shape[0]
        require(all(row["generated_frame_index"] < decoded_frame_count for row in exposed_targets),
                f"{model} request {index} native target cites a nonexistent decoded frame")
        if not sidecar_mode:
            observed: Any = receipt
            for component in source_field.split("."):
                require(isinstance(observed, Mapping) and component in observed,
                        f"{model} request {index} lacks native timing field {source_field}")
                observed = observed[component]
            require(observed == exposed_targets,
                    f"{model} request {index} native target times differ from the timing receipt")
    return value, path


def _camera_sample(observation: Mapping[str, Any], camera_id: str, label: str) -> tuple[float, int, Any, int, int]:
    clock = observation.get("clock")
    require(isinstance(clock, Mapping), f"{label} native clock is missing")
    physics_time = _finite(clock.get("physics_time_s"), f"{label} physics time")
    physics_step = clock.get("physics_step")
    control_step = clock.get("control_step")
    require(type(physics_step) is int and physics_step >= 0, f"{label} physics step is invalid")
    require(type(control_step) is int and control_step >= 0, f"{label} control step is invalid")
    cameras = clock.get("cameras")
    require(isinstance(cameras, Mapping), f"{label} camera clocks are missing")
    camera = cameras.get(camera_id)
    require(isinstance(camera, Mapping), f"{label} lacks primary camera {camera_id}")
    capture_ns = camera.get("capture_time_ns")
    require(type(capture_ns) is int and capture_ns >= 0, f"{label} capture time is invalid")
    require(camera.get("frame_id") is not None, f"{label} frame identity is missing")
    require(isinstance(camera.get("timestamp_source"), str) and camera["timestamp_source"],
            f"{label} camera timestamp source is missing")
    return physics_time, capture_ns, camera["frame_id"], physics_step, control_step


def _derive_model_alignment(
    model_evidence: Mapping[str, Any], *, model: str, expected_cells: set[str],
    evidence_base: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(model_evidence.get("model_id") == model, f"{model} evidence identity changed")
    raw_cells = model_evidence.get("development_cells")
    require(isinstance(raw_cells, list) and len(raw_cells) == 16,
            f"{model} does not contain exactly 16 development cells")
    by_id: dict[str, Mapping[str, Any]] = {}
    for entry in raw_cells:
        require(isinstance(entry, Mapping), f"{model} development cell entry is invalid")
        receipt = entry.get("cell_receipt")
        require(isinstance(receipt, Mapping), f"{model} cell descriptor is missing")
        _, path = _descriptor(receipt, f"{model} cell receipt identity", base=evidence_base)
        cell_id = load_json(path, f"{model} cell identity").get("cell_id")
        require(isinstance(cell_id, str) and cell_id not in by_id,
                f"{model} cell identity is missing or duplicated")
        by_id[cell_id] = entry
    require(set(by_id) == expected_cells, f"{model} development cohort is incomplete or contains extra cells")
    cells = [
        _validate_cell(by_id[cell_id], model=model, expected_cell_id=cell_id, evidence_base=evidence_base)
        for cell_id in sorted(expected_cells)
    ]
    request_hashes = [digest for cell in cells for digest in cell["request_receipt_sha256s"]]
    request_receipts = [receipt for cell in cells for receipt in cell["request_receipts"]]
    require(len(request_hashes) == len(set(request_hashes)),
            f"{model} development request receipt identities are not unique")
    timing, timing_path = _validate_generated_timing(
        model_evidence, model=model, base=evidence_base,
        request_hashes=request_hashes, request_receipts=request_receipts,
    )
    crop = _validate_crop(model_evidence, model=model, base=evidence_base)
    camera_id = crop["camera_id"]

    control_intervals: list[float] = []
    capture_intervals: list[float] = []
    for cell in cells:
        previous: tuple[float, int, Any, int, int] | None = None
        for index, observation in enumerate(cell["observations"]):
            current = _camera_sample(observation, camera_id, f"{cell['cell_id']} observation {index}")
            require(current[4] == index, f"{cell['cell_id']} native control counter changed")
            if previous is not None:
                require(current[0] > previous[0], f"{cell['cell_id']} physics time did not advance")
                require(current[1] > previous[1], f"{cell['cell_id']} camera time did not advance")
                require(current[2] != previous[2], f"{cell['cell_id']} camera frame identity repeated")
                require(current[3] > previous[3], f"{cell['cell_id']} physics step did not advance")
                control_intervals.append(current[0] - previous[0])
                capture_intervals.append((current[1] - previous[1]) / 1e9)
            previous = current
    control_step_s = min(control_intervals)
    capture_interval_s = min(capture_intervals)
    tolerance_s = min(control_step_s / 2.0, capture_interval_s / 2.0)
    require(tolerance_s > 0, f"{model} derived timestamp tolerance is not positive")

    mapping_rows: list[dict[str, Any]] = []
    prefix = int(MODEL_LIMITS[model]["unchanged_executed_prefix_horizon"])
    for target in timing["generated_targets"]:
        horizon = float(target["target_physical_time_s"])
        if horizon <= 0:
            mapping_rows.append({
                "generated_frame_index": target["generated_frame_index"],
                "target_physical_time_s": horizon,
                "status": "conditioning_or_current_time_not_strictly_positive",
                "target_executed_action_offset": None,
                "eligible_request_count": 0,
                "full_prefix_request_count": 0,
                "max_camera_timestamp_residual_s": None,
                "max_physics_timestamp_residual_s": None,
            })
            continue
        offsets: list[int] = []
        camera_residuals: list[float] = []
        physics_residuals: list[float] = []
        full_prefix_count = 0
        unsupported = False
        for cell in cells:
            observations = cell["observations"]
            for request in cell["request_execution"]:
                start = int(request["action_step_start"])
                executed = int(request["executed_actions"])
                if executed == prefix:
                    full_prefix_count += 1
                start_sample = _camera_sample(observations[start], camera_id, "request start")
                candidates: list[tuple[float, int, float]] = []
                for offset in range(1, executed + 1):
                    sample = _camera_sample(observations[start + offset], camera_id, "request target")
                    camera_elapsed = (sample[1] - start_sample[1]) / 1e9
                    physics_elapsed = sample[0] - start_sample[0]
                    candidates.append((abs(camera_elapsed - horizon), offset, abs(physics_elapsed - horizon)))
                if not candidates:
                    continue
                camera_residual, offset, physics_residual = min(candidates, key=lambda item: (item[0], item[1]))
                if camera_residual <= tolerance_s and physics_residual <= tolerance_s:
                    offsets.append(offset)
                    camera_residuals.append(camera_residual)
                    physics_residuals.append(physics_residual)
                elif executed == prefix:
                    unsupported = True
        unique_offsets = sorted(set(offsets))
        qualified = (
            not unsupported and bool(offsets) and len(unique_offsets) == 1
            and unique_offsets[0] <= prefix
            and len(offsets) >= full_prefix_count > 0
        )
        mapping_rows.append({
            "generated_frame_index": target["generated_frame_index"],
            "target_physical_time_s": horizon,
            "status": "qualified" if qualified else "unsupported",
            "target_executed_action_offset": unique_offsets[0] if qualified else None,
            "eligible_request_count": len(offsets),
            "full_prefix_request_count": full_prefix_count,
            "max_camera_timestamp_residual_s": max(camera_residuals) if camera_residuals else None,
            "max_physics_timestamp_residual_s": max(physics_residuals) if physics_residuals else None,
        })
    qualified = [row for row in mapping_rows if row["status"] == "qualified"]
    require(qualified, f"{model} has no native-time-qualified target within the executed prefix")
    primary = max(qualified, key=lambda row: (row["target_physical_time_s"], row["generated_frame_index"]))
    earlier = [row for row in qualified if row["target_physical_time_s"] < primary["target_physical_time_s"]]
    early = min(earlier, key=lambda row: (row["target_physical_time_s"], row["generated_frame_index"])) if earlier else None

    resources = [cell["resource"] for cell in cells]
    source_receipts = {
        "development_cell_receipts": [cell["cell_receipt"] for cell in cells],
        "adapter_completions": [cell["adapter_completion"] for cell in cells],
        "adapter_journals": [cell["adapter_journal"] for cell in cells],
        "official_request_receipt_sha256s": request_hashes,
        "generated_target_timing": _file_descriptor(timing_path),
        "resource_receipts": [cell["resource_receipt"] for cell in cells],
        "camera_crop_contract": {
            "path": crop["path"], "sha256": crop["file_sha256"],
            "bytes": Path(crop["path"]).stat().st_size,
        },
    }
    mapping_receipt_id = f"wmf1-development-{model.lower()}-physical-alignment-v1"
    mapping = sign_document({
        "schema_version": MAPPING_SCHEMA,
        "receipt_id": mapping_receipt_id,
        "study_id": STUDY_ID,
        "status": "qualified_from_complete_development_native_timing",
        "model_id": model,
        "development_cell_ids": sorted(expected_cells),
        "development_cell_count": 16,
        "development_request_count": len(request_hashes),
        "request_semantics": {
            key: MODEL_LIMITS[model][key]
            for key in (
                "returned_action_horizon", "unchanged_executed_prefix_horizon",
                "action_space", "seed_semantics", "temporal_context",
            )
        },
        "timing_claim_boundary": {
            "generated_target_source": timing["native_runtime_field"],
            "time_source_kind": timing["time_source_kind"],
            "clock_bridge": timing["clock_bridge"],
            "presentation_video_fps_used": False,
            "conditioning_fps_used_as_target_timing": False,
            "generated_frame_index_interpreted_as_action_index": False,
        },
        "camera": {
            "camera_id": camera_id,
            "camera_crop_id": crop["camera_crop_id"],
            "camera_crop_sha256": crop["payload_sha256"],
            "image_width_px": crop["image_width_px"],
            "image_height_px": crop["image_height_px"],
            "crop_operation": crop["crop_operation"],
        },
        "measured_clock_intervals": {
            "control_step_s_min": control_step_s,
            "control_step_s_max": max(control_intervals),
            "captured_frame_interval_s_min": capture_interval_s,
            "captured_frame_interval_s_max": max(capture_intervals),
            "timestamp_tolerance_s": tolerance_s,
            "tolerance_rule": "min(half minimum positive native control interval, half minimum positive original-camera capture interval)",
        },
        "frame_to_physical_time": mapping_rows,
        "primary_target": dict(primary),
        "early_target": dict(early) if early is not None else None,
        "source_receipts": source_receipts,
    })
    alignment_unsigned = {
        "contract_id": f"wmf1-{model.lower()}-alignment-v1",
        "model_id": model,
        "mapping_receipt_id": mapping_receipt_id,
        # Filled with the exact mapping file hash by the bundle writer.
        "mapping_receipt_sha256": None,
        "primary_horizon_s": primary["target_physical_time_s"],
        "generated_frame_index": primary["generated_frame_index"],
        "target_executed_action_offset": primary["target_executed_action_offset"],
        "control_step_s": control_step_s,
        "captured_frame_interval_s": capture_interval_s,
        "timestamp_tolerance_s": tolerance_s,
        "camera_id": camera_id,
        "camera_crop_id": crop["camera_crop_id"],
        "camera_crop_sha256": crop["payload_sha256"],
        "image_width_px": crop["image_width_px"],
        "image_height_px": crop["image_height_px"],
        "early_horizon": None if early is None else {
            "horizon_s": early["target_physical_time_s"],
            "generated_frame_index": early["generated_frame_index"],
            "target_executed_action_offset": early["target_executed_action_offset"],
        },
    }
    return mapping, alignment_unsigned, {
        "resources": resources,
        "request_hashes": request_hashes,
    }


def _annotation_evidence(
    raw: Any, *, base: Path, models: Sequence[str],
    alignment_hashes: Mapping[str, str],
) -> dict[str, Any]:
    require(isinstance(raw, Mapping), "annotation evidence is missing")
    rubric_descriptor, rubric_path = _descriptor(raw.get("rubric"), "annotation rubric", base=base)
    rubric = load_json(rubric_path, "annotation rubric")
    require(rubric.get("schema_version") == "wmf-forecast-annotation-rubric-v1"
            and rubric.get("study_id") == STUDY_ID and rubric.get("status") == "frozen",
            "annotation rubric is not frozen for this study")
    require(rubric.get("freeze_scope") == "development_duplicate_validation_and_confirmation",
            "annotation rubric scope changed")

    summary_descriptor, summary_path = _descriptor(raw.get("development_summary"), "development label summary", base=base)
    summary = load_json(summary_path, "development label summary")
    require(summary.get("schema_version") == DEVELOPMENT_SUMMARY_SCHEMA, "development label summary schema changed")
    verify_signed(summary, "development label summary")
    require(summary.get("study_id") == STUDY_ID, "development label summary study changed")
    require(summary.get("qualified_model_ids") == list(models), "development label model cohort changed")
    require(summary.get("qualified_alignment_contract_sha256_by_model") == dict(alignment_hashes),
            "development labels do not bind the derived alignment contracts")
    require(summary.get("distinct_raters_attested") is True,
            "development labels lack two independent raters")
    require(type(summary.get("eligible_duplicate_count")) is int
            and summary["eligible_duplicate_count"] > 0,
            "development labels contain no jointly resolvable duplicates")
    threshold = _finite(
        summary.get("movement_resolution_threshold_relative_image_diagonal"),
        "movement-resolution threshold",
    )
    require(summary.get("movement_disagreement_definition") == MOVEMENT_DISAGREEMENT_DEFINITION,
            "development movement-disagreement definition changed")
    require(summary.get("quantile_probability") == 0.95
            and summary.get("quantile_method") == MOVEMENT_QUANTILE_METHOD,
            "movement-resolution threshold is not the required development q95")
    require(summary.get("rubric_sha256") == rubric_descriptor["sha256"],
            "development labels use a different rubric")
    source_files = summary.get("source_files")
    require(isinstance(source_files, Mapping)
            and set(source_files) == {"restricted_map", "rater_a_responses", "rater_b_responses"},
            "development label source inventory is incomplete")
    resolved_sources: dict[str, tuple[Path, str]] = {}
    for source_name, reference in source_files.items():
        require(isinstance(reference, Mapping)
                and set(reference) == {"path", "artifact_sha256"},
                f"development label source {source_name} descriptor changed")
        raw_path = reference.get("path")
        expected_hash = reference.get("artifact_sha256")
        require(isinstance(raw_path, str) and raw_path
                and isinstance(expected_hash, str) and SHA_RE.fullmatch(expected_hash) is not None,
                f"development label source {source_name} descriptor is invalid")
        source_candidate = Path(raw_path)
        if not source_candidate.is_absolute():
            source_candidate = summary_path.parent / source_candidate
        require(not source_candidate.is_symlink(),
                f"development label source {source_name} is a symlink")
        source_path = source_candidate.resolve()
        require(source_path.exists(), f"development label source {source_name} is missing")
        require(artifact_sha256(source_path) == expected_hash,
                f"development label source {source_name} hash mismatch")
        resolved_sources[source_name] = (source_path, expected_hash)
    require(summary.get("restricted_map_sha256") == resolved_sources["restricted_map"][1],
            "development restricted-map hashes disagree")
    require(summary.get("rater_a_response_sha256") == resolved_sources["rater_a_responses"][1]
            and summary.get("rater_b_response_sha256") == resolved_sources["rater_b_responses"][1],
            "development first-pass response hashes disagree")

    consensus_descriptor, consensus_path = _descriptor(raw.get("final_consensus"), "development final consensus", base=base)
    consensus = load_json(consensus_path, "development final consensus")
    require(consensus.get("schema_version") == FINAL_CONSENSUS_SCHEMA,
            "development final consensus schema changed")
    verify_signed(consensus, "development final consensus")
    require(consensus.get("study_id") == STUDY_ID, "development consensus study changed")
    require(consensus.get("stage") == "development"
            and consensus.get("status") == "mechanically_merged_from_locked_blind_responses",
            "development consensus is not final")
    require(consensus.get("source_restricted_map_sha256") == summary.get("restricted_map_sha256"),
            "development consensus restricted-map hash differs from label summary")
    require(consensus.get("first_pass_response_sha256_by_slot") == {
        "rater_a": summary.get("rater_a_response_sha256"),
        "rater_b": summary.get("rater_b_response_sha256"),
    }, "development consensus first-pass hashes differ from label summary")
    adjudication_path_raw = consensus.get("adjudication_map_path")
    adjudication_sha = consensus.get("adjudication_map_sha256")
    require(isinstance(adjudication_path_raw, str) and adjudication_path_raw
            and isinstance(adjudication_sha, str) and SHA_RE.fullmatch(adjudication_sha) is not None,
            "development consensus adjudication-map identity is missing")
    adjudication_candidate = Path(adjudication_path_raw)
    if not adjudication_candidate.is_absolute():
        adjudication_candidate = consensus_path.parent / adjudication_candidate
    require(not adjudication_candidate.is_symlink(),
            "development consensus adjudication map is a symlink")
    adjudication_path = adjudication_candidate.resolve()
    require(adjudication_path.is_file() and sha256_file(adjudication_path) == adjudication_sha,
            "development consensus adjudication-map hash mismatch")
    adjudicator_response = consensus.get("adjudicator_response_path")
    adjudicator_hash = consensus.get("adjudicator_response_sha256")
    if adjudicator_response is None:
        require(adjudicator_hash is None, "development consensus has an unbound adjudicator hash")
    else:
        require(isinstance(adjudicator_response, str) and adjudicator_response
                and isinstance(adjudicator_hash, str) and SHA_RE.fullmatch(adjudicator_hash) is not None,
                "development consensus adjudicator response identity is invalid")
        adjudicator_candidate = Path(adjudicator_response)
        if not adjudicator_candidate.is_absolute():
            adjudicator_candidate = consensus_path.parent / adjudicator_candidate
        require(not adjudicator_candidate.is_symlink(),
                "development consensus adjudicator response is a symlink")
        adjudicator_path = adjudicator_candidate.resolve()
        require(adjudicator_path.exists() and artifact_sha256(adjudicator_path) == adjudicator_hash,
                "development consensus adjudicator response hash mismatch")

    decision = raw.get("usability_decision")
    require(isinstance(decision, Mapping), "development usability decision is missing")
    require(set(decision) == {
        "measurement_usable", "decided_by", "decided_at", "basis",
        "development_summary_sha256",
    }, "development usability-decision fields changed")
    require(decision.get("measurement_usable") is True,
            "development measurement was not explicitly approved as usable")
    for key in ("decided_by", "decided_at", "basis"):
        require(isinstance(decision.get(key), str) and decision[key],
                f"development usability decision {key} is missing")
    require(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", decision["decided_at"]) is not None,
            "development usability decision time is not RFC3339 UTC")
    require(decision.get("development_summary_sha256") == summary_descriptor["sha256"],
            "development usability decision does not bind the summary")
    _deep_validate_annotation_evidence(
        rubric=rubric,
        rubric_path=rubric_path,
        summary=summary,
        summary_path=summary_path,
        consensus=consensus,
        consensus_path=consensus_path,
        restricted_mapping_path=resolved_sources["restricted_map"][0],
        rater_a_path=resolved_sources["rater_a_responses"][0],
        rater_b_path=resolved_sources["rater_b_responses"][0],
    )
    return {
        "rubric": rubric_descriptor,
        "development_summary": summary_descriptor,
        "final_consensus": consensus_descriptor,
        "usability_decision": dict(decision),
        "movement_resolution": {
            "status": "frozen_from_duplicate_development_labels",
            "threshold_relative_image_diagonal": threshold,
            "quantile_probability": 0.95,
            "quantile_method": MOVEMENT_QUANTILE_METHOD,
            "eligible_duplicate_count": summary["eligible_duplicate_count"],
        },
        "annotation_seconds": dict(summary.get("annotation_seconds", {})),
    }


def _load_annotation_validator() -> Any:
    path = Path(__file__).with_name("forecast_annotation_workflow.py")
    require(path.is_file(), "forecast annotation validator is missing")
    specification = importlib.util.spec_from_file_location(
        "wmf_development_release_annotation_validator", path
    )
    require(specification is not None and specification.loader is not None,
            "forecast annotation validator cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _deep_validate_annotation_evidence(
    *,
    rubric: Mapping[str, Any],
    rubric_path: Path,
    summary: Mapping[str, Any],
    summary_path: Path,
    consensus: Mapping[str, Any],
    consensus_path: Path,
    restricted_mapping_path: Path,
    rater_a_path: Path,
    rater_b_path: Path,
) -> None:
    """Delegate reproduction to the source-of-truth annotation workflow."""

    try:
        workflow = _load_annotation_validator()
        workflow._validate_rubric(rubric, stage="confirmation")
        examples_path = rubric_path.parent / rubric["illustrated_examples"]["manifest_path"]
        require(examples_path.is_file(), "rubric illustrated-example manifest is missing")
        require(sha256_file(examples_path) == rubric["illustrated_examples"]["manifest_sha256"],
                "rubric illustrated-example manifest hash mismatch")
        workflow._validate_example_manifest(
            workflow.load_json(examples_path), manifest_path=examples_path
        )
        reproduced_summary = workflow.summarize_development_labels(
            restricted_map_path=restricted_mapping_path,
            rater_a_response_path=rater_a_path,
            rater_b_response_path=rater_b_path,
        )
        require(reproduced_summary == summary,
                "development label summary does not reproduce from locked responses")
        workflow.validate_final_consensus(
            consensus,
            consensus_path=consensus_path,
            development_summary=summary,
            development_summary_sha256=sha256_file(summary_path),
            restricted_mapping=workflow.load_json(restricted_mapping_path),
        )
    except FreezeError:
        raise
    except Exception as error:
        raise FreezeError(
            f"annotation workflow deep validation failed: {type(error).__name__}: {error}"
        ) from error


def _resource_budget(
    raw_policy: Any, *, models: Sequence[str], model_aux: Mapping[str, Mapping[str, Any]],
    annotation: Mapping[str, Any],
) -> dict[str, Any]:
    require(isinstance(raw_policy, Mapping), "resource budget policy is missing")
    required = {
        "headroom_multiplier", "selected_execution_host",
        "max_parallel_blocks_by_model", "authorized_gpu_memory_bytes_per_gpu",
        "confirmation_cells_per_model", "confirmation_annotation_judgment_ceiling",
    }
    require(set(raw_policy) == required, "resource budget policy fields changed")
    multiplier = _finite(raw_policy.get("headroom_multiplier"), "resource headroom", positive=True)
    require(multiplier >= 1, "resource headroom multiplier must be at least one")
    host = raw_policy.get("selected_execution_host")
    require(isinstance(host, str) and host, "selected execution host is missing")
    parallel = raw_policy.get("max_parallel_blocks_by_model")
    require(isinstance(parallel, Mapping) and set(parallel) == set(models),
            "parallel resource budget does not cover the qualified models")
    require(all(type(value) is int and value > 0 for value in parallel.values()),
            "parallel block limits must be positive integers")
    memory_per_gpu = raw_policy.get("authorized_gpu_memory_bytes_per_gpu")
    require(type(memory_per_gpu) is int and memory_per_gpu > 0,
            "authorized GPU memory is invalid")
    confirmation_cells = raw_policy.get("confirmation_cells_per_model")
    require(confirmation_cells == 96, "confirmation budget must retain all 96 cells per qualified model")
    judgments = raw_policy.get("confirmation_annotation_judgment_ceiling")
    require(type(judgments) is int and judgments > 0,
            "annotation judgment ceiling is invalid")

    by_model: dict[str, Any] = {}
    for model in models:
        resources = model_aux[model]["resources"]
        require(len(resources) == 16, f"{model} resource inventory is incomplete")
        observed = {
            "development_cell_count": 16,
            "episode_wall_seconds_total": math.fsum(row["episode_wall_seconds"] for row in resources),
            "episode_wall_seconds_max": max(row["episode_wall_seconds"] for row in resources),
            "inference_wall_seconds_total": math.fsum(row["inference_wall_seconds_total"] for row in resources),
            "raw_recording_bytes_total": sum(row["raw_recording_bytes"] for row in resources),
            "raw_recording_bytes_max": max(row["raw_recording_bytes"] for row in resources),
            "peak_gpu_allocated_bytes_max": max(row["peak_gpu_allocated_bytes"] for row in resources),
            "peak_gpu_reserved_bytes_max": max(row["peak_gpu_reserved_bytes"] for row in resources),
            "gpu_count_max": max(row["gpu_count"] for row in resources),
        }
        ceilings = {
            "per_cell_wall_seconds": math.ceil(observed["episode_wall_seconds_max"] * multiplier),
            "per_cell_raw_recording_bytes": math.ceil(observed["raw_recording_bytes_max"] * multiplier),
            "per_block_peak_gpu_reserved_bytes": math.ceil(observed["peak_gpu_reserved_bytes_max"] * multiplier),
            "confirmation_wall_seconds_serial_ceiling": math.ceil(observed["episode_wall_seconds_max"] * multiplier) * confirmation_cells,
            "confirmation_raw_recording_bytes_ceiling": math.ceil(observed["raw_recording_bytes_max"] * multiplier) * confirmation_cells,
        }
        require(ceilings["per_block_peak_gpu_reserved_bytes"]
                <= memory_per_gpu * observed["gpu_count_max"],
                f"{model} measured GPU budget plus headroom exceeds authorized memory")
        by_model[model] = {
            "observed_development": observed,
            "frozen_confirmation_ceilings": ceilings,
            "max_parallel_blocks": parallel[model],
        }
    annotation_seconds = annotation.get("annotation_seconds")
    require(isinstance(annotation_seconds, Mapping)
            and all(_finite(annotation_seconds.get(key), f"{key} annotation seconds", positive=True) > 0
                    for key in ("rater_a_total", "rater_b_total")),
            "measured annotation time is incomplete")
    return {
        "status": "frozen_from_complete_development_measurements",
        "selected_execution_host": host,
        "headroom_multiplier": multiplier,
        "authorized_gpu_memory_bytes_per_gpu": memory_per_gpu,
        "confirmation_cells_per_model": confirmation_cells,
        "confirmation_annotation_judgment_ceiling": judgments,
        "measured_development_annotation_seconds": dict(annotation_seconds),
        "by_model": by_model,
    }


def derive_bundle(
    evidence_path: Path, *, require_annotation: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any] | None]:
    supplied_evidence = Path(evidence_path)
    require(not supplied_evidence.is_symlink(), "development evidence manifest is a symlink")
    evidence_path = supplied_evidence.resolve()
    evidence = load_json(evidence_path, "development release evidence")
    require(evidence.get("schema_version") == EVIDENCE_SCHEMA, "development evidence schema changed")
    require(evidence.get("study_id") == STUDY_ID, "development evidence study changed")
    branch = evidence.get("cohort_branch")
    require(branch in BRANCH_MODELS, "development evidence cohort branch is invalid")
    models = BRANCH_MODELS[str(branch)]
    require(evidence.get("qualified_model_ids") == list(models),
            "qualified models do not exactly match the cohort branch")
    evidence_base = evidence_path.parent
    _, spec_path = _descriptor(evidence.get("ablation_spec"), "machine-readable ablation spec", base=evidence_base)
    _validate_ablation_spec(load_json(spec_path, "machine-readable ablation spec"))
    _, cells_path = _descriptor(evidence.get("planned_cells"), "planned-cell CSV", base=evidence_base)
    expected_cells = _planned_development_cells(cells_path, models)
    raw_models = evidence.get("model_evidence")
    require(isinstance(raw_models, list), "model evidence must be a list")
    by_model = {row.get("model_id"): row for row in raw_models if isinstance(row, Mapping)}
    require(set(by_model) == set(models) and len(raw_models) == len(models),
            "model evidence does not exactly cover the cohort branch")

    mappings: dict[str, dict[str, Any]] = {}
    alignment_unsigned: dict[str, dict[str, Any]] = {}
    model_aux: dict[str, dict[str, Any]] = {}
    for model in models:
        mappings[model], alignment_unsigned[model], model_aux[model] = _derive_model_alignment(
            by_model[model], model=model, expected_cells=expected_cells[model],
            evidence_base=evidence_base,
        )
    if not require_annotation:
        return mappings, alignment_unsigned, None

    # The file hashes are injected by write_bundle.  Contract hashes are
    # deterministic after the mapping file hashes are known, so annotation is
    # validated in write_bundle rather than here.
    context = {
        "evidence": evidence,
        "evidence_path": evidence_path,
        "models": models,
        "model_aux": model_aux,
        "branch": branch,
    }
    return mappings, alignment_unsigned, context


def _write_atomic_directory(target: Path, files: Mapping[str, bytes]) -> None:
    supplied_target = Path(target)
    require(not supplied_target.is_symlink(), "output directory is a symlink")
    target = supplied_target.resolve()
    require(not target.exists(), f"refusing to overwrite output directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for name, payload in files.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def write_bundle(evidence_path: Path, output_dir: Path, *, confirmation_release: bool) -> dict[str, Any]:
    mappings, alignment_unsigned, context = derive_bundle(
        evidence_path, require_annotation=confirmation_release
    )
    files: dict[str, bytes] = {}
    alignments: dict[str, dict[str, Any]] = {}
    mapping_file_hashes: dict[str, str] = {}
    for model, mapping in mappings.items():
        mapping_name = f"{model.lower()}_physical_alignment_receipt.json"
        mapping_payload = pretty_json_bytes(mapping)
        files[mapping_name] = mapping_payload
        mapping_file_hashes[model] = sha256_bytes(mapping_payload)
        unsigned = dict(alignment_unsigned[model])
        unsigned["mapping_receipt_sha256"] = mapping_file_hashes[model]
        unsigned["contract_sha256"] = sha256_bytes(canonical_bytes(unsigned))
        # The schema name is intentionally carried by the enclosing file role;
        # annotation_workflow.py requires this exact compact key set.
        alignment_name = f"{model.lower()}_alignment_contract.json"
        files[alignment_name] = pretty_json_bytes(unsigned)
        alignments[model] = unsigned

    result: dict[str, Any] = {
        "status": "alignment_qualified",
        "models": list(mappings),
        "alignment_contract_sha256_by_model": {
            model: contract["contract_sha256"] for model, contract in alignments.items()
        },
    }
    if confirmation_release:
        assert context is not None
        evidence = context["evidence"]
        evidence_base = Path(context["evidence_path"]).parent
        contract_hashes = result["alignment_contract_sha256_by_model"]
        annotation = _annotation_evidence(
            evidence.get("annotation"), base=evidence_base,
            models=context["models"], alignment_hashes=contract_hashes,
        )
        budget = _resource_budget(
            evidence.get("resource_budget_policy"), models=context["models"],
            model_aux=context["model_aux"], annotation=annotation,
        )
        freeze = sign_document({
            "schema_version": RELEASE_SCHEMA,
            "study_id": STUDY_ID,
            "status": "frozen_for_confirmation",
            "cohort_branch": context["branch"],
            "qualified_model_ids": list(context["models"]),
            "source_evidence": _file_descriptor(Path(context["evidence_path"])),
            "alignment_contracts_by_model": {
                model: {
                    "path": f"{model.lower()}_alignment_contract.json",
                    "sha256": sha256_bytes(files[f"{model.lower()}_alignment_contract.json"]),
                    "contract_id": alignments[model]["contract_id"],
                    "contract_sha256": alignments[model]["contract_sha256"],
                    "mapping_receipt": {
                        "path": f"{model.lower()}_physical_alignment_receipt.json",
                        "sha256": mapping_file_hashes[model],
                    },
                }
                for model in context["models"]
            },
            "request_semantics_by_model": {
                model: mappings[model]["request_semantics"] for model in context["models"]
            },
            "sampling": {
                "request_seed": REQUEST_SAMPLING_SEED,
                "request_cap_per_episode": REQUEST_SAMPLE_CAP,
                "request_algorithm": REQUEST_SAMPLING_ALGORITHM,
                "analysis_seed": ANALYSIS_SEED,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            },
            "annotation": annotation,
            "resource_budget": budget,
            "release_decision": {
                "eligible": True,
                "blockers": [],
                "claim_boundary": (
                    "Confirmation is eligible only for the listed models and frozen contracts; "
                    "this receipt is not a confirmation episode or forecast-accuracy result."
                ),
            },
        })
        files["confirmation_release_freeze.json"] = pretty_json_bytes(freeze)
        result.update(
            status="frozen_for_confirmation",
            release_freeze="confirmation_release_freeze.json",
            release_freeze_sha256=sha256_bytes(files["confirmation_release_freeze.json"]),
        )
    _write_atomic_directory(output_dir, files)
    return result


def _resolve_bundle_descriptor(base: Path, value: Any, label: str) -> tuple[dict[str, Any], Path]:
    return _descriptor(value, label, base=base)


def validate_release_freeze(
    path: Path, expected_sha256: str, *, expected_model: str | None = None,
) -> dict[str, Any]:
    """Validate a confirmation freeze for queue/server/cell admission.

    The evidence manifest was deeply authenticated when the bundle was built.
    Admission revalidates the immutable freeze, evidence-manifest identity,
    model membership, and every local mapping/alignment object without needing
    model imports or access to labels.
    """

    supplied_path = Path(path)
    require(not supplied_path.is_symlink(), "confirmation freeze is a symlink")
    path = supplied_path.resolve()
    require(SHA_RE.fullmatch(str(expected_sha256)) is not None,
            "expected confirmation-freeze SHA-256 is invalid")
    require(path.is_file(), "confirmation freeze is missing")
    require(sha256_file(path) == expected_sha256, "confirmation freeze file hash mismatch")
    freeze = load_json(path, "confirmation release freeze")
    require(set(freeze) == {
        "schema_version", "study_id", "status", "cohort_branch",
        "qualified_model_ids", "source_evidence",
        "alignment_contracts_by_model", "request_semantics_by_model",
        "sampling", "annotation", "resource_budget", "release_decision",
        "payload_sha256",
    }, "confirmation freeze fields changed")
    require(freeze.get("schema_version") == RELEASE_SCHEMA, "confirmation freeze schema changed")
    verify_signed(freeze, "confirmation release freeze")
    require(freeze.get("study_id") == STUDY_ID and freeze.get("status") == "frozen_for_confirmation",
            "confirmation freeze is not active for this study")
    branch = freeze.get("cohort_branch")
    require(branch in BRANCH_MODELS, "confirmation freeze cohort branch is invalid")
    models = BRANCH_MODELS[str(branch)]
    require(freeze.get("qualified_model_ids") == list(models),
            "confirmation freeze model set changed")
    if expected_model is not None:
        require(expected_model in models, f"model {expected_model} is not qualified by the freeze")
    _descriptor(freeze.get("source_evidence"), "confirmation source evidence")
    contracts = freeze.get("alignment_contracts_by_model")
    require(isinstance(contracts, Mapping) and set(contracts) == set(models),
            "confirmation freeze alignment coverage changed")
    for model in models:
        entry = contracts[model]
        require(isinstance(entry, Mapping), f"{model} alignment descriptor is invalid")
        descriptor, alignment_path = _resolve_bundle_descriptor(
            path.parent, {"path": entry.get("path"), "sha256": entry.get("sha256")},
            f"{model} alignment contract",
        )
        alignment = load_json(alignment_path, f"{model} alignment contract")
        require(alignment.get("model_id") == model, f"{model} alignment identity changed")
        unsigned = dict(alignment)
        contract_sha = unsigned.pop("contract_sha256", None)
        require(contract_sha == entry.get("contract_sha256")
                and sha256_bytes(canonical_bytes(unsigned)) == contract_sha,
                f"{model} alignment contract hash changed")
        require(entry.get("contract_id") == alignment.get("contract_id"),
                f"{model} alignment contract ID changed")
        mapping_entry = entry.get("mapping_receipt")
        mapping_descriptor, mapping_path = _resolve_bundle_descriptor(
            path.parent, mapping_entry, f"{model} mapping receipt"
        )
        mapping = load_json(mapping_path, f"{model} mapping receipt")
        require(mapping.get("schema_version") == MAPPING_SCHEMA
                and mapping.get("status") == "qualified_from_complete_development_native_timing"
                and mapping.get("model_id") == model,
                f"{model} physical mapping is not qualified")
        verify_signed(mapping, f"{model} mapping receipt")
        require(alignment.get("mapping_receipt_sha256") == mapping_descriptor["sha256"],
                f"{model} alignment/mapping hash binding changed")
        timing_boundary = mapping.get("timing_claim_boundary")
        require(isinstance(timing_boundary, Mapping), f"{model} timing boundary is missing")
        require(timing_boundary.get("time_source_kind") == "native_runtime_exposed_target_offsets"
                and timing_boundary.get("clock_bridge")
                == "elapsed physical seconds from request current original-camera capture",
                f"{model} mapping does not use the qualified native clock bridge")
        require(timing_boundary == {
            **timing_boundary,
            "presentation_video_fps_used": False,
            "conditioning_fps_used_as_target_timing": False,
            "generated_frame_index_interpreted_as_action_index": False,
        }, f"{model} mapping contains prohibited timing inference")
        require(alignment.get("mapping_receipt_id") == mapping.get("receipt_id"),
                f"{model} alignment/mapping receipt ID changed")
        primary = mapping.get("primary_target")
        require(isinstance(primary, Mapping)
                and primary.get("status") == "qualified"
                and alignment.get("primary_horizon_s") == primary.get("target_physical_time_s")
                and alignment.get("generated_frame_index") == primary.get("generated_frame_index")
                and alignment.get("target_executed_action_offset")
                == primary.get("target_executed_action_offset"),
                f"{model} alignment primary target differs from its mapping")
        early = mapping.get("early_target")
        expected_early = None if early is None else {
            "horizon_s": early.get("target_physical_time_s"),
            "generated_frame_index": early.get("generated_frame_index"),
            "target_executed_action_offset": early.get("target_executed_action_offset"),
        }
        require(alignment.get("early_horizon") == expected_early,
                f"{model} alignment early target differs from its mapping")
        camera = mapping.get("camera")
        clocks = mapping.get("measured_clock_intervals")
        require(isinstance(camera, Mapping) and isinstance(clocks, Mapping),
                f"{model} mapping camera/clock evidence is missing")
        require(
            alignment.get("camera_id") == camera.get("camera_id")
            and alignment.get("camera_crop_id") == camera.get("camera_crop_id")
            and alignment.get("camera_crop_sha256") == camera.get("camera_crop_sha256")
            and alignment.get("image_width_px") == camera.get("image_width_px")
            and alignment.get("image_height_px") == camera.get("image_height_px")
            and alignment.get("control_step_s") == clocks.get("control_step_s_min")
            and alignment.get("captured_frame_interval_s")
            == clocks.get("captured_frame_interval_s_min")
            and alignment.get("timestamp_tolerance_s") == clocks.get("timestamp_tolerance_s"),
            f"{model} alignment camera/clock values differ from its mapping",
        )
        expected_semantics = {
            key: MODEL_LIMITS[model][key]
            for key in (
                "returned_action_horizon", "unchanged_executed_prefix_horizon",
                "action_space", "seed_semantics", "temporal_context",
            )
        }
        require(mapping.get("request_semantics") == expected_semantics,
                f"{model} mapping request semantics changed")
        require(freeze.get("request_semantics_by_model", {}).get(model) == expected_semantics,
                f"{model} release request semantics differ from mapping")
    decision = freeze.get("release_decision")
    require(isinstance(decision, Mapping) and decision.get("eligible") is True
            and decision.get("blockers") == [], "confirmation freeze is blocked")
    sampling = freeze.get("sampling")
    require(isinstance(sampling, Mapping)
            and sampling.get("request_seed") == REQUEST_SAMPLING_SEED
            and sampling.get("request_cap_per_episode") == REQUEST_SAMPLE_CAP
            and sampling.get("request_algorithm") == REQUEST_SAMPLING_ALGORITHM
            and sampling.get("analysis_seed") == ANALYSIS_SEED
            and sampling.get("bootstrap_resamples") == BOOTSTRAP_RESAMPLES,
            "confirmation sampling/statistical freeze changed")
    annotation = freeze.get("annotation")
    require(isinstance(annotation, Mapping)
            and annotation.get("usability_decision", {}).get("measurement_usable") is True
            and annotation.get("movement_resolution", {}).get("status")
            == "frozen_from_duplicate_development_labels",
            "confirmation annotation evidence is incomplete")
    budget = freeze.get("resource_budget")
    require(isinstance(budget, Mapping)
            and budget.get("status") == "frozen_from_complete_development_measurements",
            "confirmation resource budget is incomplete")
    return freeze


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("derive-alignment", "freeze-confirmation"):
        command = subparsers.add_parser(name)
        command.add_argument("--evidence", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate-release")
    validate.add_argument("--freeze", type=Path, required=True)
    validate.add_argument("--sha256", required=True)
    validate.add_argument("--model", choices=sorted(MODEL_LIMITS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command in {"derive-alignment", "freeze-confirmation"}:
            result = write_bundle(
                args.evidence, args.output_dir,
                confirmation_release=args.command == "freeze-confirmation",
            )
        else:
            freeze = validate_release_freeze(args.freeze, args.sha256, expected_model=args.model)
            result = {
                "status": "valid",
                "cohort_branch": freeze["cohort_branch"],
                "qualified_model_ids": freeze["qualified_model_ids"],
            }
    except FreezeError as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
