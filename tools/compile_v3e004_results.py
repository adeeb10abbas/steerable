#!/usr/bin/env python3
"""Compile compact, pairing-aware V3-E004 evidence from retained raw outputs.

The compiler is deliberately post-processing only.  It never imports a model
or simulator package.  Partial compilation is useful for queue monitoring, but
all publication claims remain disabled until every registered behavioral cell
is present exactly once.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.analysis import (  # noqa: E402
    AnalysisError,
    FAILURE_CATEGORIES,
    compile_checkpoint,
    seed_from_label,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.r002_orientation_tolerance import (  # noqa: E402
    AMENDMENT_SHA256 as R002_AMENDMENT_SHA256,
    validate_runtime_attestation as validate_r002_runtime_attestation,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.runtime_contract import (  # noqa: E402
    RuntimeContractError,
)


BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"
EPISODE_PATTERNS = (
    "**/raw_episode.jsonl",
    "**/e004_episode.json",
    "**/behavioral_episodes.jsonl",
)
INVALID_PATTERNS = (
    "**/infrastructure_invalid*.jsonl",
    "**/infrastructure_invalid*.json",
    "**/infrastructure_failures*.jsonl",
    "**/infrastructure_failures*.json",
    "**/setup_invalid*.jsonl",
    "**/setup_invalid*.json",
    "**/bridge_failure*.jsonl",
    "**/bridge_failure*.json",
)
REQUIRED_MEASUREMENTS = (
    "success",
    "failure_category",
    "signed_final_lateral_offset",
    "requested_side_depth",
    "cone_entry_step",
    "cone_entry_sustained",
    "episode_length",
    "time_to_first_contact",
    "grasp_step",
    "cumulative_lateral_path",
    "peak_lateral_excursion",
    "symmetry_level_s",
    "asymmetry_metric_A",
    "position_residual",
    "orientation_residual",
    "midline_residual",
    "occlusion_check",
    "realised_object_poses",
    "arm_reset_pose",
)


class CompileError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CompileError(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> Any:
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
            require(isinstance(value, dict), f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _finite_number(value: Any, label: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{label} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} is not finite")
    return result


def _row_value(row: Mapping[str, Any], name: str) -> Any:
    aliases = {
        "cell_id": ("cell_id", "registered_cell_id"),
        "relation": ("relation", "requested_relation"),
        "signed_final_lateral_offset": ("signed_final_lateral_offset", "signed_final_lateral_offset_m"),
        "requested_side_depth": ("requested_side_depth", "requested_side_depth_m"),
        "episode_length": ("episode_length", "episode_length_steps", "actions_executed"),
        "time_to_first_contact": ("time_to_first_contact", "time_to_first_contact_steps"),
        "cumulative_lateral_path": ("cumulative_lateral_path", "cumulative_lateral_path_m"),
        "peak_lateral_excursion": ("peak_lateral_excursion", "peak_lateral_excursion_m"),
    }
    for candidate in aliases.get(name, (name,)):
        if candidate in row:
            return row[candidate]
    return None


def _is_episode(row: Mapping[str, Any]) -> bool:
    schema = str(row.get("schema_version", ""))
    return (
        row.get("amendment_id") == "V3-E004"
        and type(row.get("success")) is bool
        and ("episode" in schema or row.get("record_type") == "behavioral_episode")
        and row.get("behavioral_result_valid", True) is True
    )


def _source_result_binding(row: Mapping[str, Any]) -> dict[str, Any] | None:
    path_value, digest = row.get("source_result_path"), row.get("source_result_sha256")
    if not path_value and not digest:
        return None
    require(isinstance(path_value, str) and isinstance(digest, str), "partial source_result binding")
    path = Path(path_value)
    require(path.is_file(), f"missing source result: {path}")
    require(sha256_file(path) == digest, f"source result digest changed: {path}")
    source = load_json(path)
    require(isinstance(source, dict), f"source result is not an object: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest, "payload": source}


def _file_binding(value: Any, label: str) -> dict[str, Any]:
    path_value = value.get("path") if isinstance(value, Mapping) else value
    require(isinstance(path_value, str) and path_value, f"missing {label} path")
    path = Path(path_value)
    require(path.is_file() and path.stat().st_size > 0, f"missing or empty {label}: {path}")
    result = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if isinstance(value, Mapping):
        if value.get("bytes") is not None:
            require(value["bytes"] == result["bytes"], f"{label} byte binding changed")
        if value.get("sha256") is not None:
            require(value["sha256"] == result["sha256"], f"{label} digest changed")
    return result


def normalize_episode(
    row: Mapping[str, Any],
    *,
    queue_row: Mapping[str, Any],
    source_path: Path,
    source_sha256: str,
    source_line: int,
    registration_sha256: str,
    queue_sha256: str,
    candidate_sha256: str,
) -> dict[str, Any]:
    cell_id = str(_row_value(row, "cell_id"))
    relation = str(_row_value(row, "relation"))
    require(cell_id == queue_row["cell_id"], f"cell id is not registered: {cell_id}")
    exact = {
        "model_id": queue_row["model_id"],
        "arena": queue_row["arena"],
        "environment_seed": queue_row["environment_seed"],
        "sampling_seed": queue_row["sampling_seed"],
        "prompt": queue_row["prompt"],
    }
    for key, expected in exact.items():
        require(row.get(key) == expected, f"{cell_id}: {key} differs from registered queue")
    require(relation == queue_row["relation"], f"{cell_id}: relation differs from queue")
    level = _finite_number(_row_value(row, "symmetry_level_s"), f"{cell_id}.symmetry_level_s")
    require(math.isclose(level, float(queue_row["symmetry_level_s"]), abs_tol=1e-12), f"{cell_id}: level differs")
    require(type(row.get("success")) is bool, f"{cell_id}: success must be boolean")
    category = str(row.get("failure_category"))
    require(category in FAILURE_CATEGORIES, f"{cell_id}: invalid failure category")
    require((category == "correct") == bool(row["success"]), f"{cell_id}: success/category mismatch")
    for name in (
        "signed_final_lateral_offset",
        "requested_side_depth",
        "asymmetry_metric_A",
        "position_residual",
        "orientation_residual",
        "midline_residual",
        "cumulative_lateral_path",
        "peak_lateral_excursion",
    ):
        _finite_number(_row_value(row, name), f"{cell_id}.{name}")
    for name in ("cone_entry_step", "time_to_first_contact", "grasp_step"):
        value = _row_value(row, name)
        require(value is None or (type(value) is int and value >= 0), f"{cell_id}.{name} must be NR or a nonnegative step")
    require(type(_row_value(row, "cone_entry_sustained")) is bool, f"{cell_id}: invalid sustained-entry field")
    episode_length = _row_value(row, "episode_length")
    require(type(episode_length) is int and episode_length > 0, f"{cell_id}: invalid episode length")
    occlusion = row.get("occlusion_check")
    require(isinstance(occlusion, dict) and occlusion, f"{cell_id}: missing per-camera occlusion record")
    require(all(value is False for value in occlusion.values()), f"{cell_id}: occluded camera retained")
    require(isinstance(row.get("realised_object_poses"), dict) and row["realised_object_poses"], f"{cell_id}: missing realised poses")
    require(isinstance(row.get("arm_reset_pose"), dict) and row["arm_reset_pose"], f"{cell_id}: missing arm reset pose")
    if level == 1.0:
        require(float(row["position_residual"]) < 0.001, f"{cell_id}: s1 position residual fails")
        require(float(row["orientation_residual"]) < math.radians(0.5), f"{cell_id}: s1 orientation residual fails")
        require(float(row["midline_residual"]) < 0.001, f"{cell_id}: s1 midline residual fails")

    for name, expected in (
        ("registration_sha256", registration_sha256),
        ("queue_sha256", queue_sha256),
        ("candidate_sha256", candidate_sha256),
    ):
        if row.get(name) is not None:
            require(row[name] == expected, f"{cell_id}: {name} changed")
    nested_binding = _source_result_binding(row)
    nested_reset_sha256: str | None = None
    if nested_binding is not None:
        nested = nested_binding.pop("payload").get("v3e004", {})
        require(nested.get("cell_id") == cell_id, f"{cell_id}: nested source cell differs")
        require(nested.get("registration_sha256") == registration_sha256, f"{cell_id}: nested registration differs")
        require(nested.get("queue_sha256") == queue_sha256, f"{cell_id}: nested queue differs")
        require(nested.get("candidate_sha256") == candidate_sha256, f"{cell_id}: nested candidate differs")
        nested_reset_sha256 = nested.get("initial_physical_fingerprint_sha256")

    artifact_container = row.get("artifacts") if isinstance(row.get("artifacts"), Mapping) else {}
    video_value = artifact_container.get("viewport_video", row.get("simulator_video"))
    action_value = artifact_container.get("executed_action_trace", row.get("executed_action_trace"))
    raw_artifacts = {
        "simulator_video": _file_binding(video_value, f"{cell_id} simulator video"),
        "executed_action_trace": _file_binding(action_value, f"{cell_id} executed action trace"),
    }

    compact: dict[str, Any] = {
        "schema_version": "vla-wam-shared-v3e004-compact-episode-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E004",
        "cell_id": cell_id,
        "registered_cell_id": cell_id,
        "matched_pair_id": queue_row["matched_pair_id"],
        "model_id": row["model_id"],
        "arena": row["arena"],
        "environment_seed": int(row["environment_seed"]),
        "sampling_seed": int(row["sampling_seed"]),
        "relation": relation,
        "requested_relation": relation,
        "prompt": row["prompt"],
        "prompt_sha256": queue_row["prompt_sha256"],
        "success": bool(row["success"]),
        "failure_category": category,
        "signed_final_lateral_offset": float(_row_value(row, "signed_final_lateral_offset")),
        "requested_side_depth": float(_row_value(row, "requested_side_depth")),
        "cone_entry_step": _row_value(row, "cone_entry_step"),
        "cone_entry_sustained": bool(_row_value(row, "cone_entry_sustained")),
        "endpoint_shift": row.get("endpoint_shift"),
        "action_distinct": row.get("action_distinct"),
        "episode_length": int(episode_length),
        "time_to_first_contact": _row_value(row, "time_to_first_contact"),
        "grasp_step": _row_value(row, "grasp_step"),
        "cumulative_lateral_path": float(_row_value(row, "cumulative_lateral_path")),
        "peak_lateral_excursion": float(_row_value(row, "peak_lateral_excursion")),
        "symmetry_level_s": level,
        "asymmetry_metric_A": float(row["asymmetry_metric_A"]),
        "position_residual": float(row["position_residual"]),
        "orientation_residual": float(row["orientation_residual"]),
        "midline_residual": float(row["midline_residual"]),
        "occlusion_check": dict(sorted(occlusion.items())),
        "realised_object_poses": row["realised_object_poses"],
        "arm_reset_pose": row["arm_reset_pose"],
        "object_layout_symmetric_not_embodiment": True,
        "initial_state_sha256": row.get("initial_state_sha256", nested_reset_sha256),
        "request0_pair_identity_sha256": row.get("request0_pair_identity_sha256"),
        "request0_observation_payload_sha256": row.get("request0_observation_payload_sha256"),
        "request0_reset_contract_sha256": row.get("request0_reset_contract_sha256"),
        "request0_replay_mode": row.get("request0_replay_mode"),
        "registration_sha256": registration_sha256,
        "queue_sha256": queue_sha256,
        "candidate_sha256": candidate_sha256,
        "source_raw_episode": {
            "path": str(source_path),
            "line": source_line,
            "bytes": source_path.stat().st_size,
            "sha256": source_sha256,
        },
        "source_result": nested_binding,
        "missing_measurement_policy": "NR remains null and is never converted to zero",
    }
    compact["raw_artifacts"] = raw_artifacts
    if row.get("future_evidence") is not None:
        compact["future_evidence"] = row["future_evidence"]
    return compact


def _action_array(binding: Mapping[str, Any], label: str) -> np.ndarray:
    path = Path(str(binding["path"]))
    try:
        payload = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CompileError(f"cannot read {label}: {path}: {exc}") from exc
    if isinstance(payload, np.lib.npyio.NpzFile):
        try:
            for key in ("executed", "actions", "action"):
                if key in payload.files:
                    array = np.asarray(payload[key])
                    break
            else:
                raise CompileError(f"{label} NPZ has no executable action array: {path}")
        finally:
            payload.close()
    else:
        array = np.asarray(payload)
    require(array.ndim == 2 and array.shape[0] > 0 and np.isfinite(array).all(), f"{label} is not finite [steps, dim]")
    return array


def materialize_pair_fields(episodes: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output = [dict(row) for row in episodes]
    groups: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in output:
        groups[str(row["matched_pair_id"])][str(row["relation"])] = row
    pairs: list[dict[str, Any]] = []
    for pair_id, pair in sorted(groups.items()):
        if set(pair) != {"left", "right"}:
            for row in pair.values():
                row["pair_fields_status"] = "NR_until_both_registered_directions_are_valid"
            continue
        left, right = pair["left"], pair["right"]
        for key in ("model_id", "arena", "environment_seed", "sampling_seed", "symmetry_level_s", "asymmetry_metric_A"):
            require(left[key] == right[key], f"{pair_id}: matched pair differs for {key}")
        left_reset, right_reset = left.get("initial_state_sha256"), right.get("initial_state_sha256")
        if left["arena"] == "droid_robolab":
            for key in (
                "request0_pair_identity_sha256",
                "request0_observation_payload_sha256",
                "request0_reset_contract_sha256",
            ):
                left_value, right_value = left.get(key), right.get(key)
                require(
                    isinstance(left_value, str)
                    and len(left_value) == 64
                    and left_value == right_value,
                    f"{pair_id}: R001 {key} is absent or differs",
                )
            require(left.get("request0_replay_mode") == "capture_left", f"{pair_id}: LEFT lacks R001 capture")
            require(right.get("request0_replay_mode") == "replay_right", f"{pair_id}: RIGHT lacks R001 replay")
            identical_reset = True
            reset_definition = "R001 identical request0 observation bytes and reset-contract payload"
        else:
            require(
                isinstance(left_reset, str) and len(left_reset) == 64 and left_reset == right_reset,
                f"{pair_id}: identical reset fingerprint is absent or differs",
            )
            identical_reset = True
            reset_definition = "arena-native initial physical fingerprint"
        endpoint_shift = float(right["signed_final_lateral_offset"]) - float(left["signed_final_lateral_offset"])
        reported = (left.get("action_distinct"), right.get("action_distinct"))
        if all(type(value) is bool for value in reported):
            require(reported[0] == reported[1], f"{pair_id}: reported action-distinct values differ")
            action_distinct = bool(reported[0])
            compared_steps = min(10, int(left["episode_length"]), int(right["episode_length"]))
        else:
            left_actions = _action_array(left["raw_artifacts"]["executed_action_trace"], f"{pair_id} LEFT actions")
            right_actions = _action_array(right["raw_artifacts"]["executed_action_trace"], f"{pair_id} RIGHT actions")
            require(left_actions.shape[1:] == right_actions.shape[1:], f"{pair_id}: action dimensions differ")
            compared_steps = min(10, len(left_actions), len(right_actions))
            require(compared_steps > 0, f"{pair_id}: no common action prefix")
            action_distinct = bool(np.any(left_actions[:compared_steps] != right_actions[:compared_steps]))
        for row in (left, right):
            row["endpoint_shift"] = endpoint_shift
            row["action_distinct"] = action_distinct
            row["pair_fields_status"] = "derived_after_both_hash_bound_directions_exist"
        pairs.append(
            {
                "schema_version": "vla-wam-shared-v3e004-compact-pair-v1",
                "amendment_id": "V3-E004",
                "matched_pair_id": pair_id,
                "model_id": left["model_id"],
                "arena": left["arena"],
                "environment_seed": left["environment_seed"],
                "sampling_seed": left["sampling_seed"],
                "symmetry_level_s": left["symmetry_level_s"],
                "asymmetry_metric_A": left["asymmetry_metric_A"],
                "identical_reset": identical_reset,
                "identical_reset_definition": reset_definition,
                "initial_state_sha256": left_reset if left_reset == right_reset else None,
                "native_initial_state_sha256_left": left_reset,
                "native_initial_state_sha256_right": right_reset,
                "request0_pair_identity_sha256": left.get("request0_pair_identity_sha256"),
                "request0_observation_payload_sha256": left.get("request0_observation_payload_sha256"),
                "request0_reset_contract_sha256": left.get("request0_reset_contract_sha256"),
                "left_cell_id": left["cell_id"],
                "right_cell_id": right["cell_id"],
                "left_success": left["success"],
                "right_success": right["success"],
                "endpoint_shift_right_minus_left_m": endpoint_shift,
                "endpoint_redirection_left_minus_right_m": -endpoint_shift,
                "action_distinct": action_distinct,
                "action_prefix_steps_compared": compared_steps,
                "left_source_raw_episode": left["source_raw_episode"],
                "right_source_raw_episode": right["source_raw_episode"],
            }
        )
    return output, pairs


def scientific_fingerprint(row: Mapping[str, Any]) -> str:
    ignored = {"source_raw_episode", "source_result", "raw_artifacts"}
    return hashlib.sha256(canonical_bytes({key: row[key] for key in row if key not in ignored})).hexdigest()


def discover_paths(roots: Sequence[Path], patterns: Sequence[str], excluded: Sequence[Path]) -> list[Path]:
    excluded_resolved = [path.resolve() for path in excluded]
    paths: set[Path] = set()
    for root in roots:
        root = Path(root).resolve()
        require(root.exists(), f"input root does not exist: {root}")
        if root.is_file():
            candidates = [root]
        else:
            candidates = [path for pattern in patterns for path in root.glob(pattern)]
        for path in candidates:
            resolved = path.resolve()
            if any(resolved == item or item in resolved.parents for item in excluded_resolved):
                continue
            if resolved.is_file():
                paths.add(resolved)
    return sorted(paths)


def _rows_from_path(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        value = load_json(path)
        return [value] if isinstance(value, dict) else []
    return load_jsonl(path)


def load_valid_episodes(
    roots: Sequence[Path],
    *,
    queue_rows: Mapping[str, Mapping[str, Any]],
    registration_sha256: str,
    queue_sha256: str,
    candidate_sha256_by_arena: Mapping[str, str],
    excluded: Sequence[Path] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    chosen: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, str] = {}
    source_ledger: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    discovery_only: list[dict[str, Any]] = []
    for path in discover_paths(roots, EPISODE_PATTERNS, excluded):
        digest = sha256_file(path)
        for line_number, raw in enumerate(_rows_from_path(path), 1):
            if not _is_episode(raw):
                continue
            cell_id = str(_row_value(raw, "cell_id"))
            require(cell_id in queue_rows, f"unregistered E004 behavioral result: {cell_id}")
            queue_row = queue_rows[cell_id]
            arena = str(queue_row.get("arena"))
            require(arena in candidate_sha256_by_arena, f"{cell_id}: no registered candidate for arena {arena}")
            candidate_sha256 = candidate_sha256_by_arena[arena]
            require(
                queue_row.get("layout_candidate_sha256") == candidate_sha256,
                f"{cell_id}: queue candidate differs from registered arena candidate",
            )
            record = {
                "path": str(path),
                "line": line_number,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "cell_id": cell_id,
            }
            level = float(queue_row["symmetry_level_s"])
            if arena == "droid_robolab" and math.isclose(level, 0.0, abs_tol=1e-12):
                r002 = raw.get("live_orientation_realisation_tolerance_amendment")
                if r002 is None:
                    record.update(
                        {
                            "disposition": "discovery_only_excluded_from_behavioral_denominator",
                            "reason": "pre_r002_s0_missing_prospective_attestation",
                            "behavioral_denominator_included": False,
                        }
                    )
                    discovery_only.append(dict(record))
                    source_ledger.append(record)
                    continue
                try:
                    validate_r002_runtime_attestation(
                        r002,
                        amendment_sha256=R002_AMENDMENT_SHA256,
                        symmetry_level_s=0.0,
                    )
                except RuntimeContractError as exc:
                    raise CompileError(f"{cell_id}: invalid R002 s=0 attestation: {exc}") from exc
            if arena == "droid_robolab":
                required_r001 = (
                    raw.get("request0_pair_identity_sha256"),
                    raw.get("request0_observation_payload_sha256"),
                    raw.get("request0_reset_contract_sha256"),
                )
                if not all(isinstance(value, str) and len(value) == 64 for value in required_r001):
                    record.update(
                        {
                            "disposition": "discovery_only_excluded_from_behavioral_denominator",
                            "reason": "pre_r001_missing_request0_pair_identity",
                            "behavioral_denominator_included": False,
                        }
                    )
                    discovery_only.append(dict(record))
                    source_ledger.append(record)
                    continue
            compact = normalize_episode(
                raw,
                queue_row=queue_row,
                source_path=path,
                source_sha256=digest,
                source_line=line_number,
                registration_sha256=registration_sha256,
                queue_sha256=queue_sha256,
                candidate_sha256=candidate_sha256,
            )
            fingerprint = scientific_fingerprint(compact)
            if cell_id in chosen:
                require(fingerprints[cell_id] == fingerprint, f"conflicting valid duplicate for {cell_id}")
                record["disposition"] = "duplicate_valid_excluded_from_denominator"
                duplicates.append(record)
            else:
                record["disposition"] = "selected_valid_behavioral_evidence"
                chosen[cell_id] = compact
                fingerprints[cell_id] = fingerprint
            source_ledger.append(record)
    rows = sorted(
        chosen.values(),
        key=lambda row: (row["arena"], row["model_id"], row["symmetry_level_s"], row["environment_seed"], row["relation"]),
    )
    return rows, source_ledger, duplicates, discovery_only


def load_infrastructure_invalid(roots: Sequence[Path], *, excluded: Sequence[Path] = ()) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in discover_paths(roots, INVALID_PATTERNS, excluded):
        # Prior partial compiles live under ``results/`` and contain a derived
        # infrastructure ledger. They are not raw attempts and must never be
        # recursively re-ingested when a broad PVC root is compiled.
        if path.parent.name == "results" and (path.parent / "results.json").is_file():
            continue
        digest = sha256_file(path)
        if path.stat().st_size == 0:
            output.append(
                {
                    "schema_version": "vla-wam-shared-v3e004-infrastructure-invalid-reference-v1",
                    "behavioral_denominator_included": False,
                    "source": {"path": str(path), "line": None, "bytes": 0, "sha256": digest},
                    "attempt": {
                        "status": "empty_infrastructure_invalid_marker",
                        "parse_status": "empty_file_no_machine_readable_payload",
                    },
                }
            )
            continue
        for line_number, row in enumerate(_rows_from_path(path), 1):
            output.append(
                {
                    "schema_version": "vla-wam-shared-v3e004-infrastructure-invalid-reference-v1",
                    "behavioral_denominator_included": False,
                    "source": {"path": str(path), "line": line_number, "bytes": path.stat().st_size, "sha256": digest},
                    "attempt": row,
                }
            )
    return output


def _coverage(queue: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = {row["cell_id"] for row in episodes}
    queue_ids = {row["cell_id"] for row in queue}
    by_cell: dict[tuple[str, float, str], dict[str, int]] = {}
    for row in queue:
        key = (row["model_id"], float(row["symmetry_level_s"]), row["relation"])
        item = by_cell.setdefault(key, {"registered": 0, "valid": 0})
        item["registered"] += 1
        item["valid"] += int(row["cell_id"] in valid)
    cells = [
        {
            "model_id": key[0],
            "symmetry_level_s": key[1],
            "relation": key[2],
            **counts,
            "missing": counts["registered"] - counts["valid"],
            "complete": counts["registered"] == counts["valid"],
        }
        for key, counts in sorted(by_cell.items())
    ]
    return {
        "registered_cells": len(queue_ids),
        "valid_cells": len(valid),
        "missing_cells": len(queue_ids - valid),
        "unexpected_cells": len(valid - queue_ids),
        "complete": valid == queue_ids,
        "by_model_level_direction": cells,
        "missing_cell_ids": sorted(queue_ids - valid),
    }


def _power_rows(registration: Mapping[str, Any], model_id: str) -> dict[str, Mapping[str, Any]]:
    rows = [row for row in registration["power_registration"]["rows"] if row["model_id"] == model_id]
    require(len(rows) == 2, f"power registration is incomplete for {model_id}")
    return {str(row["estimand"]): row for row in rows}


def _power_maps(registration: Mapping[str, Any], model_id: str) -> tuple[dict[str, float], dict[str, str]]:
    by_estimand = _power_rows(registration, model_id)
    binary = by_estimand["binary_R_minus_L"]
    depth = by_estimand["depth_R_minus_L_m"]
    return (
        {"binary_gap": float(binary["margin"]), "depth_gap_m": float(depth["margin"])},
        {"binary_gap": str(binary["status"]), "depth_gap_m": str(depth["status"])},
    )


def _attach_power_audit(
    analysis: dict[str, Any],
    *,
    registration: Mapping[str, Any],
    model_id: str,
) -> dict[str, Any]:
    """Attach the preregistered control, margin, and achieved-MDE audit."""

    power = _power_rows(registration, model_id)
    z_sum = float(registration["power_registration"]["z_sum"])
    s1_pairs = int(analysis["levels"]["1.00"]["pairs"])
    require(s1_pairs > 0, f"{model_id}: cannot compute achieved MDE without s=1 pairs")
    mappings = {
        "binary_gap": ("binary_R_minus_L", "binary_gap_R_minus_L"),
        "depth_gap_m": ("depth_R_minus_L_m", "requested_depth_gap_R_minus_L_m"),
    }
    for result_name, (registered_name, level_name) in mappings.items():
        registered = power[registered_name]
        margin = float(registered["margin"])
        control_effect = float(registered["control_effect"])
        achieved_mde = z_sum * float(registered["sigma_plan"]) / math.sqrt(s1_pairs)
        s1_estimate = float(analysis["levels"]["1.00"][level_name]["mean"])
        result = analysis["equivalence_at_s1"][result_name]
        result["registered_power_and_control_audit"] = {
            "valid_s1_pairs": s1_pairs,
            "registered_control_effect": control_effect,
            "registered_equivalence_margin": margin,
            "registered_margin_fraction_of_absolute_control": (
                None if control_effect == 0.0 else margin / abs(control_effect)
            ),
            "registered_sigma_plan": float(registered["sigma_plan"]),
            "registered_mde_n27": float(registered["mde_n27"]),
            "registered_strict_n": registered["strict_n"],
            "registered_target_n": int(registered["target_n"]),
            "registered_power_status": str(registered["status"]),
            "mde_formula": str(registration["power_registration"]["formula"]),
            "mde_z_sum": z_sum,
            "achieved_design_mde80_at_valid_s1_n": achieved_mde,
            "strict_half_margin_mde_gate": None if margin == 0.0 else 0.5 * margin,
            "achieved_mde_within_strict_half_margin_gate": (
                False if margin == 0.0 else achieved_mde <= 0.5 * margin + 1e-12
            ),
            "s1_estimated_gap": s1_estimate,
            "s1_minus_registered_control_effect": s1_estimate - control_effect,
            "absolute_s1_fraction_of_absolute_control_effect": (
                None if control_effect == 0.0 else abs(s1_estimate) / abs(control_effect)
            ),
        }
    return analysis


def _paired_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[str(row["matched_pair_id"])][str(row["relation"])] = row
    complete = [pair for pair in groups.values() if set(pair) == {"left", "right"}]
    output: list[dict[str, Any]] = []
    for pair in complete:
        left, right = pair["left"], pair["right"]
        require(left["environment_seed"] == right["environment_seed"], "matched-pair seed differs")
        require(left["symmetry_level_s"] == right["symmetry_level_s"], "matched-pair level differs")
        output.extend((dict(left), dict(right)))
    return output


def _descriptive_progress(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: defaultdict[tuple[float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(float(row["symmetry_level_s"]), str(row["relation"]))].append(row)
    return {
        f"{level:.2f}/{relation}": {
            "valid_episodes": len(items),
            "successes": sum(bool(item["success"]) for item in items),
            "descriptive_only": True,
        }
        for (level, relation), items in sorted(groups.items())
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    """Return finite, JSON-safe descriptive statistics without inferential meaning."""

    require(bool(values), "cannot summarize an empty numeric sequence")
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 1 and np.isfinite(array).all(), "geometry summary contains a non-finite value")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def _geometry_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize every registered layout-quality field and reset-pose identity.

    These are quality-control measurements, not behavioral estimands.  The
    complete per-scene values remain in episodes.jsonl; this summary makes the
    preregistered residual, visibility, and embodiment boundary auditable from
    results.json and the decision memo.
    """

    by_level: defaultdict[float, list[Mapping[str, Any]]] = defaultdict(list)
    reset_poses: dict[str, dict[str, Any]] = {}
    reset_counts: Counter[str] = Counter()
    for row in rows:
        by_level[float(row["symmetry_level_s"])].append(row)
        pose = row["arm_reset_pose"]
        pose_sha256 = hashlib.sha256(canonical_bytes(pose)).hexdigest()
        reset_poses.setdefault(pose_sha256, dict(pose))
        reset_counts[pose_sha256] += 1

    level_summaries: dict[str, Any] = {}
    for level, items in sorted(by_level.items()):
        camera_names = sorted({name for row in items for name in row["occlusion_check"]})
        occluded_checks = [
            {"cell_id": row["cell_id"], "camera": camera}
            for row in items
            for camera, occluded in sorted(row["occlusion_check"].items())
            if occluded is True
        ]
        level_pose_hashes = sorted(
            {
                hashlib.sha256(canonical_bytes(row["arm_reset_pose"])).hexdigest()
                for row in items
            }
        )
        level_summaries[f"{level:.2f}"] = {
            "episodes": len(items),
            "realised_asymmetry_A": _numeric_summary([float(row["asymmetry_metric_A"]) for row in items]),
            "position_residual_m": _numeric_summary([float(row["position_residual"]) for row in items]),
            "orientation_residual_rad": _numeric_summary([float(row["orientation_residual"]) for row in items]),
            "orientation_residual_deg_maximum": math.degrees(
                max(float(row["orientation_residual"]) for row in items)
            ),
            "midline_residual_m": _numeric_summary([float(row["midline_residual"]) for row in items]),
            "occlusion_check": {
                "camera_names": camera_names,
                "camera_checks": sum(len(row["occlusion_check"]) for row in items),
                "episodes_with_any_occlusion": sum(any(row["occlusion_check"].values()) for row in items),
                "occluded_camera_checks": len(occluded_checks),
                "occluded_checks": occluded_checks,
                "all_observed_checks_clear": not occluded_checks,
            },
            "arm_reset_pose_sha256": level_pose_hashes,
        }

    s1 = level_summaries.get("1.00")
    if s1 is None:
        s1_gate: dict[str, Any] = {
            "status": "unavailable_no_valid_s1_episodes",
            "episodes": 0,
            "all_observed_s1_rows_pass": None,
        }
    else:
        passed = bool(
            s1["position_residual_m"]["maximum"] < 0.001
            and s1["orientation_residual_rad"]["maximum"] < math.radians(0.5)
            and s1["midline_residual_m"]["maximum"] < 0.001
            and s1["occlusion_check"]["all_observed_checks_clear"]
        )
        s1_gate = {
            "status": "pass_for_all_observed_s1_episodes" if passed else "fail_closed",
            "episodes": s1["episodes"],
            "all_observed_s1_rows_pass": passed,
        }

    return {
        "status": "available" if rows else "unavailable_no_valid_episodes",
        "episodes": len(rows),
        "four_registered_layout_quality_checks": [
            "position_residual_m",
            "orientation_residual_rad",
            "midline_residual_m",
            "occlusion_check_by_camera",
        ],
        "levels": level_summaries,
        "s1_registered_tolerances": {
            "position_residual_m_strict_upper": 0.001,
            "orientation_residual_rad_strict_upper": math.radians(0.5),
            "orientation_residual_deg_strict_upper": 0.5,
            "midline_residual_m_strict_upper": 0.001,
            "occlusion_check_required_false_all_cameras": True,
        },
        "s1_gate": s1_gate,
        "arm_reset_pose_identity_count": len(reset_poses),
        "arm_reset_pose_identities": [
            {
                "sha256": digest,
                "episodes": reset_counts[digest],
                "pose": reset_poses[digest],
            }
            for digest in sorted(reset_poses)
        ],
        "scope_caveat": (
            "The object layout is measured relative to the robot midline; the robot joint configuration, "
            "arm reset pose, camera rig, wrist mounting, and embodiment are not asserted bilaterally symmetric."
        ),
        "per_scene_values_retained_in": "results/episodes.jsonl",
    }


def _reset_pose_memo_summary(pose: Mapping[str, Any]) -> str:
    """Render a readable pose summary while results.json retains exact vectors."""

    if isinstance(pose.get("arm_joint_positions_rad"), list):
        joints = [float(value) for value in pose["arm_joint_positions_rad"]]
        gripper = pose.get("gripper_position")
        return (
            f"arm q ({len(joints)} joints) = "
            f"[{', '.join(f'{value:+.4f}' for value in joints)}] rad; "
            f"gripper = {json.dumps(gripper, allow_nan=False, separators=(',', ':'))}"
        )
    robots = pose.get("robots")
    if isinstance(robots, Mapping) and robots:
        parts = []
        for name, item in sorted(robots.items()):
            joints = np.asarray(item.get("joint_positions_rad", []), dtype=np.float64)
            if joints.size == 0 or not np.isfinite(joints).all():
                parts.append(f"{name}: unavailable")
                continue
            parts.append(
                f"{name}: {joints.size} joints, ||q||₂={np.linalg.norm(joints):.6f} rad, "
                f"range=[{np.min(joints):+.6f}, {np.max(joints):+.6f}] rad"
            )
        return "; ".join(parts) + "; exact vectors retained in results.json"
    return "exact reset-pose object retained in results.json"


def _failure_signature(rows: Sequence[Mapping[str, Any]], *, resamples: int) -> dict[str, Any]:
    by_level: defaultdict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_level[float(row["symmetry_level_s"])].append(row)
    levels: dict[str, Any] = {}
    for level, items in sorted(by_level.items()):
        failures = [row for row in items if not row["success"]]
        counts = Counter(row["failure_category"] for row in items)
        levels[f"{level:.2f}"] = {
            "episodes": len(items),
            "failures": len(failures),
            "failure_taxonomy": {category: counts[category] for category in FAILURE_CATEGORIES},
            "wrong_side_share_among_failures": None if not failures else counts["wrong_side"] / len(failures),
            "availability": "unavailable_no_failures" if not failures else "available",
        }
    seed_values: defaultdict[int, list[tuple[float, float]]] = defaultdict(list)
    for (seed, level), items in _group_seed_level(rows).items():
        failures = [row for row in items if not row["success"]]
        if failures:
            A = float(np.mean([row["asymmetry_metric_A"] for row in items]))
            seed_values[seed].append((A, sum(row["failure_category"] == "wrong_side" for row in failures) / len(failures)))
    slopes: list[float] = []
    usable: dict[int, list[tuple[float, float]]] = {}
    for seed, values in seed_values.items():
        if len(values) >= 2 and len({value[0] for value in values}) >= 2:
            usable[seed] = values
            slopes.append(float(np.polyfit([value[0] for value in values], [value[1] for value in values], 1)[0]))
    if not slopes:
        trend: dict[str, Any] = {"status": "unavailable_insufficient_within_seed_failure_levels", "wrong_side_share_is_never_imputed_as_zero": True}
    else:
        observed = float(np.mean(slopes))
        rng = np.random.default_rng(seed_from_label("V3-E004:H5:failure-signature"))
        # Every seed has at most five levels.  Enumerate that seed's finite
        # label-permutation support once, then sample the registered clustered
        # null in vector form.  This is identical to repeatedly shuffling
        # labels within each seed but remains practical for powered endpoints.
        null_sum = np.zeros(resamples, dtype=np.float64)
        for values in usable.values():
            A = np.asarray([value[0] for value in values], dtype=np.float64)
            share = np.asarray([value[1] for value in values], dtype=np.float64)
            support = []
            for permuted in permutations(A.tolist()):
                x = np.asarray(permuted, dtype=np.float64)
                centered = x - x.mean()
                support.append(float(np.dot(centered, share - share.mean()) / np.dot(centered, centered)))
            support_array = np.asarray(support, dtype=np.float64)
            null_sum += support_array[rng.integers(0, support_array.size, size=resamples)]
        null = null_sum / len(usable)
        exceed = int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-15))
        trend = {
            "status": "available",
            "estimand": "within-seed slope of wrong-side share among failures versus realised A",
            "seed_cluster_count": len(slopes),
            "mean_seed_slope": observed,
            "median_seed_slope": float(np.median(slopes)),
            "seed_slopes": slopes,
            "within_seed_level_label_permutation": {
                "replicates": resamples,
                "seed": seed_from_label("V3-E004:H5:failure-signature"),
                "two_sided_p": (exceed + 1) / (resamples + 1),
            },
            "wrong_side_share_is_never_imputed_as_zero": True,
        }
    return {"levels": levels, "trend": trend}


def _group_seed_level(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, float], list[Mapping[str, Any]]]:
    output: defaultdict[tuple[int, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        output[(int(row["environment_seed"]), float(row["symmetry_level_s"]))].append(row)
    return dict(output)


def _claim_gate(
    checkpoint: Mapping[str, Any],
    *,
    globally_complete: bool,
) -> dict[str, Any]:
    if not globally_complete:
        return {
            "publication_claims_enabled": False,
            "reason": "The complete registered 4,096-cell cohort is not yet available; partial estimates are monitoring diagnostics only.",
            "equivalence_claims": {},
        }
    equivalence: dict[str, Any] = {}
    for estimand, result in checkpoint["equivalence_at_s1"].items():
        registered_powered = result["power_status"] == "strictly_powered_at_endpoints"
        achieved_powered = bool(
            result["registered_power_and_control_audit"]["achieved_mde_within_strict_half_margin_gate"]
        )
        powered = registered_powered and achieved_powered
        defined = result.get("status") != "margin_zero_equivalence_not_defined"
        equivalent = bool(result.get("equivalent", False))
        equivalence[estimand] = {
            "registered_power_status_permits_equivalence": registered_powered,
            "achieved_mde_gate_passed": achieved_powered,
            "registered_power_gate_passed": powered,
            "margin_defined": defined,
            "interval_within_registered_margin": equivalent,
            "publication_equivalence_claim_allowed": bool(powered and defined and equivalent),
            "interpretation": (
                "equivalence_supported_at_registered_margin"
                if powered and defined and equivalent
                else "no_equivalence_claim"
            ),
        }
    positive = {
        level: bool(item["endpoint_redirection_LEFT_minus_RIGHT_m"]["bootstrap_mean95"]["low"] > 0.0)
        for level, item in checkpoint["levels"].items()
    }
    return {
        "publication_claims_enabled": True,
        "equivalence_claims": equivalence,
        "H4_endpoint_positive_control_by_level": positive,
        "equalisation_interpretation_allowed_by_level": positive,
        "scope": "object-layout symmetry only; robot, reset posture, camera rig, and embodiment are not bilaterally symmetric",
    }


def compile_report(
    *,
    registration: Mapping[str, Any],
    queue: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    invalid_attempts: Sequence[Mapping[str, Any]],
    duplicates: Sequence[Mapping[str, Any]],
    discovery_only: Sequence[Mapping[str, Any]],
    source_ledger: Sequence[Mapping[str, Any]],
    resamples: int,
) -> dict[str, Any]:
    require(resamples >= 10_000, "registered bootstrap/permutation compilation requires at least 10,000 resamples")
    coverage = _coverage(queue, episodes)
    checkpoint_reports: dict[str, Any] = {}
    for model_id in registration["design"]["model_levels_and_pairs"]:
        model_rows = [row for row in episodes if row["model_id"] == model_id]
        paired = _paired_rows(model_rows)
        core_cells = {
            (int(row["environment_seed"]), float(row["symmetry_level_s"]), row["relation"])
            for row in paired
            if int(row["environment_seed"]) in registration["design"]["core_seeds"]
            and float(row["symmetry_level_s"]) in {0.0, 1.0}
        }
        expected_core = {
            (seed, level, relation)
            for seed in registration["design"]["core_seeds"]
            for level in (0.0, 1.0)
            for relation in ("left", "right")
        }
        row: dict[str, Any] = {
            "model_id": model_id,
            "arena": next((item["arena"] for item in queue if item["model_id"] == model_id), None),
            "valid_episodes": len(model_rows),
            "complete_pairs": len(paired) // 2,
            "descriptive_progress": _descriptive_progress(model_rows),
            "core_s0_s1_complete": core_cells == expected_core,
            "geometry_quality": _geometry_summary(model_rows),
            "failure_signature": _failure_signature(model_rows, resamples=resamples),
        }
        if core_cells == expected_core:
            margins, statuses = _power_maps(registration, model_id)
            try:
                analysis = compile_checkpoint(
                    paired,
                    model_id=model_id,
                    margins=margins,
                    power_status=statuses,
                    core_seeds=registration["design"]["core_seeds"],
                    resamples=resamples,
                )
            except AnalysisError as exc:
                raise CompileError(f"{model_id} inferential compilation failed: {exc}") from exc
            analysis = _attach_power_audit(analysis, registration=registration, model_id=model_id)
            row["analysis"] = analysis
            row["claim_gate"] = _claim_gate(analysis, globally_complete=coverage["complete"])
        else:
            row["analysis"] = None
            row["claim_gate"] = _claim_gate({}, globally_complete=False)
        checkpoint_reports[model_id] = row

    arenas: dict[str, Any] = {}
    for arena in sorted({row["arena"] for row in queue}):
        arena_rows = [row for row in episodes if row["arena"] == arena]
        arenas[arena] = {
            "valid_behavioral_episodes": len(arena_rows),
            "model_ids": sorted({row["model_id"] for row in arena_rows}),
            "successes": sum(row["success"] for row in arena_rows),
            "pooled_success_rate": None,
            "pooling_status": "prohibited_across_models_and_arenas_for_inference",
        }
    discovery_by_reason = Counter(str(row.get("reason")) for row in discovery_only)
    status = "complete_hash_closed" if coverage["complete"] else "partial_progress_no_publication_claims"
    return {
        "schema_version": "vla-wam-shared-v3e004-results-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E004",
        "status": status,
        "compiled_at_utc": utc_now(),
        "registered_behavioral_cells": len(queue),
        "valid_behavioral_episodes": len(episodes),
        "infrastructure_invalid_attempts": len(invalid_attempts),
        "valid_duplicate_artifacts_excluded_from_denominators": len(duplicates),
        "discovery_only_behavioral_artifacts_excluded_from_denominators": len(discovery_only),
        "discovery_only_behavioral_artifacts_by_reason": {
            reason: discovery_by_reason[reason] for reason in sorted(discovery_by_reason)
        },
        "coverage": coverage,
        "checkpoints": checkpoint_reports,
        "arenas": arenas,
        "source_summary": {
            "source_ledger_rows": len(source_ledger),
            "unique_source_files": len({row["path"] for row in source_ledger}),
        },
        "publication_claim_status": (
            "enabled_subject_to_per-estimand_power_and_positive-control_gates"
            if coverage["complete"]
            else "withheld_until_all_registered_cells_are_valid"
        ),
        "global_claim_boundary": {
            "droid_and_robotwin_never_pooled": True,
            "behavioral_failures_remain_in_denominators": True,
            "infrastructure_invalid_attempts_excluded": True,
            "pre_r002_droid_s0_artifacts_excluded_as_discovery_only": True,
            "missing_measurements_never_encoded_as_zero": True,
            "s0_to_s1_inventory_transition_disclosed": True,
            "object_layout_symmetry_is_not_robot_or_embodiment_symmetry": True,
            "nondetection_is_not_equivalence": True,
        },
    }


def decision_memo(report: Mapping[str, Any]) -> str:
    complete = report["coverage"]["complete"]
    lines = [
        "# V3-E004 decision memo",
        "",
        f"Status: **{report['status']}**.",
        "",
        "## Evidence boundary",
        "",
        f"Valid behavioral evidence: **{report['valid_behavioral_episodes']}/{report['registered_behavioral_cells']}** registered cells. "
        f"Infrastructure-invalid attempts: **{report['infrastructure_invalid_attempts']}**, excluded from behavioral denominators.",
        f"Discovery-only behavioral artifacts: **{report['discovery_only_behavioral_artifacts_excluded_from_denominators']}**, excluded from behavioral denominators.",
        f"- Pre-R002 DROID s=0 artifacts without prospective R002 attestation: **{report['discovery_only_behavioral_artifacts_by_reason'].get('pre_r002_s0_missing_prospective_attestation', 0)}**.",
        f"- Pre-R001 DROID artifacts without fixed-observation pair identity: **{report['discovery_only_behavioral_artifacts_by_reason'].get('pre_r001_missing_request0_pair_identity', 0)}**.",
        "",
        "This experiment manipulates object-layout symmetry. It does not make the robot, reset posture, camera rig, wrist mounting, or embodiment bilaterally symmetric. DROID/RoboLab and RoboTwin remain separate and are never pooled.",
        "",
        "## Geometry and visibility quality control",
        "",
        "The four registered scene checks are position residual, mirrored-orientation residual, midline residual, and the per-camera occlusion check. The complete per-episode values remain in `results/episodes.jsonl`; the maxima below summarize only currently valid s=1 episodes.",
        "",
        "| Checkpoint | Valid s=1 episodes | Max position residual, mm | Max orientation residual, deg | Max midline residual, mm | Occluded camera checks | Reset-pose identities |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["checkpoints"].values():
        geometry = item["geometry_quality"]
        s1 = geometry["levels"].get("1.00")
        if s1 is None:
            values = (0, "NR", "NR", "NR", "NR")
        else:
            values = (
                s1["episodes"],
                f"{1000.0 * s1['position_residual_m']['maximum']:.3f}",
                f"{s1['orientation_residual_deg_maximum']:.3f}",
                f"{1000.0 * s1['midline_residual_m']['maximum']:.3f}",
                str(s1["occlusion_check"]["occluded_camera_checks"]),
            )
        lines.append(
            f"| {item['model_id']} | {values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} | "
            f"{geometry['arm_reset_pose_identity_count']} |"
        )
    lines.extend(
        [
            "",
            "### Recorded arm reset poses",
            "",
            "Each identity below hashes the complete recorded reset-pose object, including its measurement provenance. Multiple identities are retained rather than averaged away; if more than three occur, this memo shows the three most frequent and `results.json` retains the complete list.",
            "",
        ]
    )
    for item in report["checkpoints"].values():
        identities = item["geometry_quality"]["arm_reset_pose_identities"]
        if not identities:
            lines.append(f"- **{item['model_id']}**: NR — no valid episode is available yet.")
            continue
        displayed = sorted(identities, key=lambda value: (-value["episodes"], value["sha256"]))[:3]
        for identity in displayed:
            lines.append(
                f"- **{item['model_id']}**: `{identity['sha256']}` across {identity['episodes']} episodes; "
                f"{_reset_pose_memo_summary(identity['pose'])}."
            )
        if len(identities) > len(displayed):
            lines.append(
                f"- **{item['model_id']}**: {len(identities) - len(displayed)} additional reset-pose identities are retained in `results.json`."
            )
    lines.extend(["", "Passing these object-layout checks does not establish bilateral robot or embodiment symmetry.", ""])
    if not complete:
        lines.extend(
            [
                "## Publication decision",
                "",
                "No publication claim is authorized from this partial compilation. Estimates shown in progress outputs are queue diagnostics only; nondetection is not equivalence.",
                "",
                "## Progress by checkpoint",
                "",
                "| Checkpoint | Arena | Valid episodes | Complete pairs | Core s=0/s=1 complete |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for item in report["checkpoints"].values():
            lines.append(
                f"| {item['model_id']} | {item['arena']} | {item['valid_episodes']} | {item['complete_pairs']} | "
                f"{'yes' if item['core_s0_s1_complete'] else 'no'} |"
            )
    else:
        lines.extend(
            [
                "## Registered estimands",
                "",
                "| Checkpoint | Binary interaction (s1−s0) | Depth interaction, m (s1−s0) | Endpoint positive control at all levels | Equivalence claims |",
                "| --- | ---: | ---: | --- | --- |",
            ]
        )
        for item in report["checkpoints"].values():
            analysis, gate = item["analysis"], item["claim_gate"]
            binary = analysis["interaction_s1_minus_s0_core"]["binary_gap"]
            depth = analysis["interaction_s1_minus_s0_core"]["depth_gap_m"]
            positive = all(gate["H4_endpoint_positive_control_by_level"].values())
            claims = [name for name, value in gate["equivalence_claims"].items() if value["publication_equivalence_claim_allowed"]]
            lines.append(
                f"| {item['model_id']} | {binary['mean']:+.3f} [{binary['bootstrap_mean95']['low']:+.3f}, {binary['bootstrap_mean95']['high']:+.3f}] | "
                f"{depth['mean']:+.3f} [{depth['bootstrap_mean95']['low']:+.3f}, {depth['bootstrap_mean95']['high']:+.3f}] | "
                f"{'pass' if positive else 'fail closed'} | {', '.join(claims) if claims else 'none'} |"
            )
        lines.extend(
            [
                "",
                "## H2 — power and equivalence audit",
                "",
                "The achieved MDE is the preregistered 80%-power design MDE evaluated at the valid s=1 pair count. Equivalence is authorized only when the registered power status permits it and the paired 90% interval lies wholly inside the registered margin; a nonsignificant difference is never treated as equivalence.",
                "",
                "| Checkpoint / estimand | Control effect | s=1 estimate | Margin | Achieved MDE (n) | Paired 90% CI | TOST bootstrap p (lower / upper) | Decision |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in report["checkpoints"].values():
            for estimand, label in (("binary_gap", "binary R−L"), ("depth_gap_m", "depth R−L, m")):
                result = item["analysis"]["equivalence_at_s1"][estimand]
                audit = result["registered_power_and_control_audit"]
                ci = result.get("ci90")
                ci_text = "NR — zero margin" if ci is None else f"[{ci['low']:+.3f}, {ci['high']:+.3f}]"
                tost_text = (
                    "NR"
                    if ci is None
                    else f"{result['tost_bootstrap_p_lower']:.4g} / {result['tost_bootstrap_p_upper']:.4g}"
                )
                gate = item["claim_gate"]["equivalence_claims"][estimand]
                lines.append(
                    f"| {item['model_id']} / {label} | {audit['registered_control_effect']:+.3f} | "
                    f"{audit['s1_estimated_gap']:+.3f} | {audit['registered_equivalence_margin']:.3f} | "
                    f"{audit['achieved_design_mde80_at_valid_s1_n']:.3f} ({audit['valid_s1_pairs']}) | "
                    f"{ci_text} | {tost_text} | "
                    f"{gate['interpretation']} ({audit['registered_power_status']}) |"
                )
        lines.extend(
            [
                "",
                "## H3 — inventory-matched dose response",
                "",
                "A is the realised object-layout asymmetry (0 = symmetric). The registered primary slope excludes s=0 because the s=0→s>0 transition changes companion-object inventory.",
                "",
                "| Checkpoint | Binary-gap slope per A (95% CI) | Depth-gap slope per A, m (95% CI) | Per-seed binary slope signs (+/−/0) |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for item in report["checkpoints"].values():
            dose = item["analysis"]["dose_response_on_realised_A"]
            if dose is None:
                lines.append(f"| {item['model_id']} | NR — only two registered levels | NR | NR |")
                continue
            binary_dose, depth_dose = dose["binary_gap"], dose["depth_gap_m"]
            signs = binary_dose["sign"]
            lines.append(
                f"| {item['model_id']} | {binary_dose['mean_slope']:+.3f} "
                f"[{binary_dose['bootstrap_mean95']['low']:+.3f}, {binary_dose['bootstrap_mean95']['high']:+.3f}] | "
                f"{depth_dose['mean_slope']:+.3f} "
                f"[{depth_dose['bootstrap_mean95']['low']:+.3f}, {depth_dose['bootstrap_mean95']['high']:+.3f}] | "
                f"{signs['positive']}/{signs['negative']}/{signs['zero']} |"
            )
        lines.extend(
            [
                "",
                "## H5 — failure signature",
                "",
                "The preregistered diagnostic is the within-seed slope of wrong-side share among behavioral failures versus realised A. A negative slope means wrong-side failures become more prominent as the object layout approaches symmetry. A level with no failures remains unavailable and is never imputed as zero.",
                "",
                "| Checkpoint | Seed clusters | Mean slope | Median slope | Two-sided permutation p | Status |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in report["checkpoints"].values():
            trend = item["failure_signature"]["trend"]
            if trend["status"] != "available":
                lines.append(f"| {item['model_id']} | NR | NR | NR | NR | {trend['status']} |")
                continue
            lines.append(
                f"| {item['model_id']} | {trend['seed_cluster_count']} | {trend['mean_seed_slope']:+.3f} | "
                f"{trend['median_seed_slope']:+.3f} | "
                f"{trend['within_seed_level_label_permutation']['two_sided_p']:.4g} | available |"
            )
        lines.extend(
            [
                "",
                "## Interpretation rule",
                "",
                "Interaction estimates answer whether the measured directional gap changes between the registered asymmetric and symmetric object layouts. Equivalence wording is permitted only where the preregistered power gate passed and the paired interval lies inside the registered margin. Underpowered or zero-margin rows remain descriptive even when their point estimate is near zero.",
                "",
                "The s=0→s=1 comparison includes the preregistered companion-object inventory transition. The primary graded dose-response for π0.5 and Nano therefore uses inventory-matched s=0.25, 0.50, 0.75, and 1.00; s=0 is an anchored reference.",
            ]
        )
    lines.extend(
        [
            "",
            "## Frozen prompts",
            "",
            '- DROID LEFT: “Put the Rubik\'s cube to the left of the bowl.”',
            '- DROID RIGHT: “Put the Rubik\'s cube to the right of the bowl.”',
            '- RoboTwin LEFT: “Put the small woodenblock to the left of the red playingcards box.”',
            '- RoboTwin RIGHT: “Put the small woodenblock to the right of the red playingcards box.”',
            "",
        ]
    )
    return "\n".join(lines)


def compile_outputs(
    *,
    registration_path: Path,
    queue_path: Path,
    raw_roots: Sequence[Path],
    output_root: Path,
    resamples: int,
    require_complete: bool,
) -> dict[str, Any]:
    registration = load_json(registration_path)
    queue = load_jsonl(queue_path)
    require(registration.get("amendment_id") == "V3-E004", "wrong registration")
    require(registration["queue"]["sha256"] == sha256_file(queue_path), "queue hash differs from registration")
    require(len(queue) == int(registration["design"]["total_evidence_cells"]), "registration queue count differs")
    queue_by_id = {row["cell_id"]: row for row in queue}
    require(len(queue_by_id) == len(queue), "registered queue has duplicate cell ids")
    registration_sha = sha256_file(registration_path)
    queue_sha = sha256_file(queue_path)
    candidate_sha_by_arena = {
        "droid_robolab": registration["layout"]["candidate_sha256"],
        "robotwin": registration["layout"]["robotwin_stretch_candidate_sha256"],
    }
    results_dir = Path(output_root) / "results"
    episodes, ledger, duplicates, discovery_only = load_valid_episodes(
        raw_roots,
        queue_rows=queue_by_id,
        registration_sha256=registration_sha,
        queue_sha256=queue_sha,
        candidate_sha256_by_arena=candidate_sha_by_arena,
        excluded=(results_dir,),
    )
    episodes, pair_rows = materialize_pair_fields(episodes)
    invalid = load_infrastructure_invalid(raw_roots, excluded=(results_dir,))
    report = compile_report(
        registration=registration,
        queue=queue,
        episodes=episodes,
        invalid_attempts=invalid,
        duplicates=duplicates,
        discovery_only=discovery_only,
        source_ledger=ledger,
        resamples=resamples,
    )
    if require_complete:
        require(report["coverage"]["complete"], f"registered cohort incomplete: {len(episodes)}/{len(queue)}")
    atomic_write(results_dir / "episodes.jsonl", b"".join(canonical_bytes(row) for row in episodes))
    atomic_write(results_dir / "pairs.jsonl", b"".join(canonical_bytes(row) for row in pair_rows))
    atomic_write(results_dir / "infrastructure_invalid.jsonl", b"".join(canonical_bytes(row) for row in invalid))
    atomic_write(results_dir / "discovery_only.jsonl", b"".join(canonical_bytes(row) for row in discovery_only))
    atomic_write(results_dir / "source_ledger.jsonl", b"".join(canonical_bytes(row) for row in ledger))
    atomic_write(results_dir / "results.json", json.dumps(report, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    atomic_write(Path(output_root) / "DECISION_MEMO.md", decision_memo(report).encode("utf-8"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, default=BASE / "registration.json")
    parser.add_argument("--queue", type=Path, default=BASE / "queue.jsonl")
    parser.add_argument("--raw-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=BASE)
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = compile_outputs(
        registration_path=args.registration,
        queue_path=args.queue,
        raw_roots=args.raw_root,
        output_root=args.output_root,
        resamples=args.resamples,
        require_complete=args.require_complete,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "valid_behavioral_episodes": report["valid_behavioral_episodes"],
                "registered_behavioral_cells": report["registered_behavioral_cells"],
                "publication_claim_status": report["publication_claim_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
