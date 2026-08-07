#!/usr/bin/env python3
"""Validate and compact the complete V3-D001 nested stochastic block.

Raw simulator evidence remains on the ali PVC.  This compiler emits bounded,
machine-readable records whose hashes bind every behavioral episode, matched
pair, and preserved non-behavioral attempt used for the committed summary.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.v3.pi05_stochastic_v3d001.contract import (
    ACTION_CHUNK_STEPS,
    ACTION_DIM,
    MODEL_ID,
    PROMPTS,
    QUEUE_SHA256,
    REGISTRATION_ID,
    RELATIONS,
    RELEASE_MANIFEST_SHA256,
    SAMPLING_INDICES,
    SEEDS,
    ContractError,
    canonical_json_bytes,
    load_release,
    sha256_file,
)
from tools.vla_wam_v3_episode_schema import parse_jsonl_record


ATTEMPT = 4
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_MASTER_SEED = 3_104_159
FAILURE_CATEGORIES = ("correct", "pick_failed", "transport_failed", "wrong_side", "release_failed")
COMPACT_EPISODES_NAME = "pi05_v3d001_episodes.jsonl"
COMPACT_PAIRS_NAME = "pi05_v3d001_matched_pairs.jsonl"
INVALID_ATTEMPTS_NAME = "pi05_v3d001_invalid_attempts.jsonl"
SUMMARY_NAME = "pi05_v3d001_summary.json"
EVIDENCE_MANIFEST_NAME = "evidence_manifest.json"


def _canonical_line(value: Mapping[str, Any]) -> str:
    return canonical_json_bytes(value).decode("utf-8") + "\n"


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(_canonical_line(row) for row in rows), encoding="utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_for(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_MASTER_SEED}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _bootstrap_mean(values: Sequence[float], *, label: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ContractError(f"invalid bootstrap population: {label}")
    seed = _seed_for(label)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))
    estimates = array[draws].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "confidence": 0.95,
        "lower": float(lower),
        "method": "environment_seed_cluster_nonparametric_percentile_bootstrap",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "statistic": "mean_of_environment_seed_means",
        "unit_of_resampling": "environment_seed",
        "upper": float(upper),
    }


def _exact_sign_test(values: Sequence[float]) -> dict[str, Any]:
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    ties = len(values) - positive - negative
    effective = positive + negative
    if not effective:
        p_value = 1.0
    else:
        tail = sum(math.comb(effective, k) for k in range(0, min(positive, negative) + 1)) / (2**effective)
        p_value = min(1.0, 2.0 * tail)
    return {
        "effective_n": effective,
        "method": "exact_two_sided_paired_sign_test_zero_ties_excluded",
        "negative": negative,
        "p_value": p_value,
        "positive": positive,
        "ties": ties,
    }


def _wilson(successes: int, total: int) -> dict[str, Any]:
    if not total:
        raise ContractError("Wilson interval requires a positive denominator")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "confidence": 0.95,
        "lower": max(0.0, center - radius),
        "method": "Wilson_score_descriptive_episode_level_not_primary_due_to_seed_nesting",
        "upper": min(1.0, center + radius),
    }


def _summary(values: Sequence[float | int | None], *, clusters: Mapping[int, Sequence[float | int | None]], label: str) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    cluster_means = [
        mean(float(value) for value in clusters[seed] if value is not None and math.isfinite(float(value)))
        for seed in sorted(clusters)
        if any(value is not None and math.isfinite(float(value)) for value in clusters[seed])
    ]
    if not finite:
        return {"available": 0, "missing": len(values), "mean": None, "median": None}
    return {
        "available": len(finite),
        "environment_seed_clusters": len(cluster_means),
        "mean": mean(finite),
        "mean_environment_cluster_bootstrap_95": _bootstrap_mean(cluster_means, label=label),
        "median": median(finite),
        "missing": len(values) - len(finite),
        "sample_standard_deviation": stdev(finite) if len(finite) > 1 else 0.0,
    }


def _cell_slug(cell_id: str) -> str:
    return cell_id.replace(":", "__")


def _load_one_jsonl(path: Path) -> tuple[dict[str, Any], str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ContractError(f"expected exactly one JSONL record: {path}")
    return parse_jsonl_record(lines[0]), sha256_file(path)


def _load_one_object_line(path: Path) -> tuple[dict[str, Any], str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ContractError(f"expected exactly one JSON object line: {path}")
    value = json.loads(lines[0])
    if not isinstance(value, dict):
        raise ContractError(f"expected a JSON object: {path}")
    return value, sha256_file(path)


def _validate_hashed_file(entry: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(entry.get("path", "")))
    if not path.is_file():
        raise ContractError(f"missing {label}: {path}")
    if path.stat().st_size != entry.get("bytes") or sha256_file(path) != entry.get("sha256"):
        raise ContractError(f"hash/size changed for {label}: {path}")
    return path


def _validate_episode(cell: Any, raw_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = raw_root / "V3-D001_pi05_nested_stochastic" / _cell_slug(cell.cell_id) / f"attempt{ATTEMPT:02d}"
    raw_path = directory / "raw_episode.jsonl"
    manifest_path = raw_path.with_name(raw_path.name + ".manifest.json")
    if not raw_path.is_file() or not manifest_path.is_file():
        raise ContractError(f"missing exact attempt04 closeout for {cell.cell_id}")
    record, raw_sha = _load_one_jsonl(raw_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("row_count") != 1 or manifest.get("jsonl_sha256") != raw_sha:
        raise ContractError(f"episode post-close manifest changed: {cell.cell_id}")
    expected = {
        "behavioral_result_valid": True,
        "registered_cell_id": cell.cell_id,
        "environment_seed": cell.environment_seed,
        "requested_relation": cell.relation,
        "shared_policy_sampling_seed_index": cell.sampling_index,
        "policy_seed": cell.sampling_seed_base,
        "prompt": PROMPTS[cell.relation],
        "queue_sha256": QUEUE_SHA256,
        "release_manifest_sha256": RELEASE_MANIFEST_SHA256,
        "registration_id": REGISTRATION_ID,
    }
    for key, wanted in expected.items():
        if record.get(key) != wanted:
            raise ContractError(f"episode identity mismatch {cell.cell_id}:{key}")
    request_seeds = record.get("request_sampling_seeds")
    expected_request_seeds = list(range(cell.sampling_seed_base, cell.sampling_seed_base + len(request_seeds or [])))
    if not request_seeds or request_seeds != expected_request_seeds:
        raise ContractError(f"per-request sampling seeds changed: {cell.cell_id}")
    diagnostics = record.get("v3d001_episode_diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("failure_category") not in FAILURE_CATEGORIES:
        raise ContractError(f"missing V3-D001 diagnostics: {cell.cell_id}")
    if bool(diagnostics.get("success")) != bool(record.get("requested_success")):
        raise ContractError(f"success diagnostics disagree: {cell.cell_id}")

    video = record["artifacts"]["viewport_video"]
    action = record["artifacts"]["executed_action_trace"]
    returned = record["source_artifacts"]["returned_action_chunks"]
    state_capture = record["source_artifacts"]["state_capture"]
    action_metadata = record["source_artifacts"]["action_trace_metadata"]
    video_path = _validate_hashed_file(video, label="viewport video")
    action_path = _validate_hashed_file(action, label="executed action trace")
    returned_path = _validate_hashed_file(returned, label="returned action chunks")
    _validate_hashed_file(state_capture, label="state capture")
    _validate_hashed_file(action_metadata, label="action trace metadata")
    attestations = record["source_artifacts"].get("pre_action_reset_attestations")
    if not isinstance(attestations, list) or not attestations:
        raise ContractError(f"missing reset attestation: {cell.cell_id}")
    for entry in attestations:
        _validate_hashed_file(entry, label="pre-action reset attestation")
    actions = np.load(action_path, allow_pickle=False)
    chunks = np.load(returned_path, allow_pickle=False)
    if list(actions.shape) != [record["actions_executed"], ACTION_DIM] or not np.isfinite(actions).all():
        raise ContractError(f"executed action array changed: {cell.cell_id}")
    if list(chunks.shape) != [len(request_seeds), ACTION_CHUNK_STEPS, ACTION_DIM] or not np.isfinite(chunks).all():
        raise ContractError(f"returned action chunks changed: {cell.cell_id}")
    if not video_path.stat().st_size:
        raise ContractError(f"empty viewport video: {cell.cell_id}")

    scalar_keys = (
        "signed_final_lateral_offset_m", "requested_side_depth_m", "cone_entry_step",
        "cone_entry_sustained", "episode_length_steps", "time_to_first_contact_steps",
        "grasp_step", "cumulative_lateral_path_m", "peak_lateral_excursion_m",
    )
    compact = {
        "schema_version": "vla-wam-shared-v3d001-pi05-compact-episode-v1",
        "registered_cell_id": cell.cell_id,
        "matched_stochastic_block_id": cell.block_id,
        "environment_seed": cell.environment_seed,
        "shared_policy_sampling_seed_index": cell.sampling_index,
        "requested_relation": cell.relation,
        "prompt": record["prompt"],
        "policy_sampling_seed_base": cell.sampling_seed_base,
        "request_sampling_seed_count": len(request_seeds),
        "request_sampling_seeds_sha256": hashlib.sha256(canonical_json_bytes(request_seeds)).hexdigest(),
        "success": bool(record["requested_success"]),
        "failure_category": diagnostics["failure_category"],
        **{key: diagnostics.get(key) for key in scalar_keys},
        "actions_executed": record["actions_executed"],
        "initial_state_sha256": record["initial_state_sha256"],
        "lane_pod_uid": record["lane_pod_uid"],
        "lane_gpu_uuid": record["lane_gpu_uuid"],
        "raw_episode_jsonl": {"path": str(raw_path), "sha256": raw_sha, "bytes": raw_path.stat().st_size},
        "raw_episode_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "bytes": manifest_path.stat().st_size},
        "viewport_video": dict(video),
        "executed_action_trace": dict(action),
        "returned_action_chunks": dict(returned),
        "state_capture": dict(state_capture),
        "action_trace_metadata": dict(action_metadata),
        "pre_action_reset_attestations": attestations,
    }
    return compact, record


def _validate_pair(cell_rows: Mapping[str, dict[str, Any]], block_id: str, raw_root: Path) -> dict[str, Any]:
    slug = block_id.replace(":", "__")
    path = raw_root / "V3-D001_pi05_nested_stochastic" / "matched_pairs" / f"attempt{ATTEMPT:02d}" / f"{slug}.json"
    manifest_path = path.with_name(path.name + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ContractError(f"missing matched-pair diagnostics: {block_id}")
    row, pair_sha = _load_one_object_line(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("row_count") != 1 or manifest.get("json_sha256") != pair_sha:
        raise ContractError(f"matched-pair manifest changed: {block_id}")
    if row.get("matched_stochastic_block_id") != block_id or row.get("queue_sha256") != QUEUE_SHA256:
        raise ContractError(f"matched-pair identity changed: {block_id}")
    left = cell_rows[row["left_registered_cell_id"]]
    right = cell_rows[row["right_registered_cell_id"]]
    if left["initial_state_sha256"] != right["initial_state_sha256"] or row.get("initial_state_sha256") != left["initial_state_sha256"]:
        raise ContractError(f"matched reset identity changed: {block_id}")
    if row["left_raw_episode_jsonl"]["sha256"] != left["raw_episode_jsonl"]["sha256"] or row["right_raw_episode_jsonl"]["sha256"] != right["raw_episode_jsonl"]["sha256"]:
        raise ContractError(f"matched raw episode binding changed: {block_id}")
    expected_shift = right["signed_final_lateral_offset_m"] - left["signed_final_lateral_offset_m"]
    if not math.isclose(row["endpoint_shift_right_minus_left_m"], expected_shift, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError(f"matched endpoint shift changed: {block_id}")
    return {
        **row,
        "schema_version": "vla-wam-shared-v3d001-pi05-compact-matched-pair-v1",
        "left_success": left["success"],
        "right_success": right["success"],
        "left_failure_category": left["failure_category"],
        "right_failure_category": right["failure_category"],
        "endpoint_redirection_left_minus_right_m": -row["endpoint_shift_right_minus_left_m"],
        "endpoint_ordering_aligned": row["endpoint_shift_right_minus_left_m"] < 0,
        "pair_json": {"path": str(path), "sha256": pair_sha, "bytes": path.stat().st_size},
        "pair_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "bytes": manifest_path.stat().st_size},
    }


def _invalid_attempts(raw_root: Path) -> list[dict[str, Any]]:
    experiment = raw_root / "V3-D001_pi05_nested_stochastic"
    attempts = sorted(
        path for path in experiment.glob("v3d001__pi05__env*/attempt*")
        if path.is_dir() and path.name != f"attempt{ATTEMPT:02d}"
    )
    rows = []
    for path in attempts:
        files = []
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            files.append({"relative_path": str(item.relative_to(path)), "bytes": item.stat().st_size, "sha256": sha256_file(item)})
        rows.append({
            "schema_version": "vla-wam-shared-v3d001-nonbehavioral-attempt-inventory-v1",
            "attempt_path": str(path),
            "attempt_name": path.name,
            "registered_cell_slug": path.parent.name,
            "behavioral_denominator_included": False,
            "file_count": len(files),
            "files": files,
        })
    return rows


def _success_summary(episodes: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_condition: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in episodes:
        by_condition[(row["environment_seed"], row["requested_relation"])].append(row)
    if set(by_condition) != {(seed, relation) for seed in SEEDS for relation in RELATIONS} or any(len(rows) != len(SAMPLING_INDICES) for rows in by_condition.values()):
        raise ContractError("nested condition coverage is not exact 27x2x8")
    condition_rows = []
    for seed in SEEDS:
        for relation in RELATIONS:
            rows = by_condition[(seed, relation)]
            successes = sum(row["success"] for row in rows)
            condition_rows.append({
                "environment_seed": seed,
                "requested_relation": relation,
                "rollouts": len(rows),
                "successes": successes,
                "failures": len(rows) - successes,
                "estimated_success_probability": successes / len(rows),
            })
    per_seed = []
    for seed in SEEDS:
        left = next(row for row in condition_rows if row["environment_seed"] == seed and row["requested_relation"] == "left")
        right = next(row for row in condition_rows if row["environment_seed"] == seed and row["requested_relation"] == "right")
        per_seed.append({
            "environment_seed": seed,
            "left_successes_of_8": left["successes"],
            "right_successes_of_8": right["successes"],
            "estimated_p_left": left["estimated_success_probability"],
            "estimated_p_right": right["estimated_success_probability"],
            "directional_gap_p_right_minus_p_left": right["estimated_success_probability"] - left["estimated_success_probability"],
        })
    direction = {}
    for relation in RELATIONS:
        rows = [row for row in episodes if row["requested_relation"] == relation]
        successes = sum(row["success"] for row in rows)
        seed_rates = [next(item["estimated_success_probability"] for item in condition_rows if item["environment_seed"] == seed and item["requested_relation"] == relation) for seed in SEEDS]
        direction[relation] = {
            "episodes": len(rows),
            "environment_seed_conditions": len(SEEDS),
            "successes": successes,
            "failures": len(rows) - successes,
            "estimated_success_probability": successes / len(rows),
            "environment_seed_cluster_bootstrap_95": _bootstrap_mean(seed_rates, label=f"success:{relation}"),
            "Wilson_95_descriptive_only": _wilson(successes, len(rows)),
        }
    gaps = [row["directional_gap_p_right_minus_p_left"] for row in per_seed]
    return {
        "by_direction": direction,
        "directional_gap": {
            "definition": "per-seed estimated p(success|RIGHT) minus p(success|LEFT), each from eight stochastic rollouts",
            "mean": mean(gaps),
            "median": median(gaps),
            "environment_seed_cluster_bootstrap_95": _bootstrap_mean(gaps, label="success:directional_gap"),
            "paired_sign_test": _exact_sign_test(gaps),
        },
        "per_environment_seed": per_seed,
    }, condition_rows


def _failure_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for relation in RELATIONS:
        rows = [row for row in episodes if row["requested_relation"] == relation]
        counts = Counter(row["failure_category"] for row in rows)
        result[relation] = {
            "counts": {category: counts.get(category, 0) for category in FAILURE_CATEGORIES},
            "row_normalized": {category: counts.get(category, 0) / len(rows) for category in FAILURE_CATEGORIES},
        }
    return result


def _continuous_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
        "signed_final_lateral_offset_m", "requested_side_depth_m", "episode_length_steps",
        "cone_entry_step", "grasp_step", "time_to_first_contact_steps",
        "cumulative_lateral_path_m", "peak_lateral_excursion_m",
    )
    output = {}
    for metric in metrics:
        output[metric] = {}
        for relation in RELATIONS:
            rows = [row for row in episodes if row["requested_relation"] == relation]
            clusters = {seed: [row.get(metric) for row in rows if row["environment_seed"] == seed] for seed in SEEDS}
            output[metric][relation] = _summary([row.get(metric) for row in rows], clusters=clusters, label=f"continuous:{metric}:{relation}")
    return output


def _matched_summary(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcome_counts = Counter(
        "both_success" if row["left_success"] and row["right_success"] else
        "left_only" if row["left_success"] else
        "right_only" if row["right_success"] else "neither_success"
        for row in pairs
    )
    redirection = [float(row["endpoint_redirection_left_minus_right_m"]) for row in pairs]
    per_seed_redirection = [mean(row["endpoint_redirection_left_minus_right_m"] for row in pairs if row["environment_seed"] == seed) for seed in SEEDS]
    aligned = sum(row["endpoint_ordering_aligned"] for row in pairs)
    distinct = sum(row["action_distinct"] for row in pairs)
    return {
        "pair_count": len(pairs),
        "binary_outcomes": {key: outcome_counts.get(key, 0) for key in ("both_success", "left_only", "right_only", "neither_success")},
        "action_distinct": {"pairs": distinct, "total": len(pairs), "rate": distinct / len(pairs)},
        "endpoint_ordering_aligned": {"pairs": aligned, "total": len(pairs), "rate": aligned / len(pairs)},
        "endpoint_redirection_left_minus_right_m": {
            "definition": "signed final LEFT-condition offset minus signed final RIGHT-condition offset; positive follows requested ordering",
            "episode_pair_mean": mean(redirection),
            "episode_pair_median": median(redirection),
            "mean_of_seed_means": mean(per_seed_redirection),
            "environment_seed_cluster_bootstrap_95": _bootstrap_mean(per_seed_redirection, label="matched:endpoint_redirection"),
            "paired_seed_mean_sign_test": _exact_sign_test(per_seed_redirection),
        },
    }


def analyze(*, repo_root: Path, raw_root: Path, output_dir: Path, release_manifest: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    raw_root = raw_root.resolve()
    output_dir = output_dir.resolve()
    release = load_release(repo_root, release_manifest.resolve())
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty V3-D001 result directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    compact_episodes = []
    raw_records = {}
    for cell in release.cells:
        compact, raw = _validate_episode(cell, raw_root)
        compact_episodes.append(compact)
        raw_records[cell.cell_id] = raw
    compact_by_id = {row["registered_cell_id"]: row for row in compact_episodes}
    if len(compact_by_id) != 432:
        raise ContractError("validated episode IDs are not unique/exact")

    block_ids = list(dict.fromkeys(cell.block_id for cell in release.cells))
    compact_pairs = [_validate_pair(compact_by_id, block_id, raw_root) for block_id in block_ids]
    if len(compact_pairs) != 216:
        raise ContractError("validated matched-pair count is not 216")
    invalid_attempts = _invalid_attempts(raw_root)

    episodes_path = output_dir / COMPACT_EPISODES_NAME
    pairs_path = output_dir / COMPACT_PAIRS_NAME
    invalid_path = output_dir / INVALID_ATTEMPTS_NAME
    _write_jsonl(episodes_path, compact_episodes)
    _write_jsonl(pairs_path, compact_pairs)
    _write_jsonl(invalid_path, invalid_attempts)

    success, conditions = _success_summary(compact_episodes)
    lane_counts = Counter((row["lane_pod_uid"], row["lane_gpu_uuid"]) for row in compact_episodes)
    summary = {
        "schema_version": "vla-wam-shared-v3d001-pi05-stochastic-summary-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "registration_id": REGISTRATION_ID,
        "model_id": MODEL_ID,
        "arena": "droid_robolab",
        "status": "complete_exact_release_analyzed",
        "population": {
            "environment_seed_count": len(SEEDS),
            "policy_sampling_rollouts_per_seed_direction": len(SAMPLING_INDICES),
            "nested_seed_direction_conditions": len(conditions),
            "valid_behavioral_episodes": len(compact_episodes),
            "matched_left_right_policy_sampling_pairs": len(compact_pairs),
            "behavioral_failures_included": True,
            "infrastructure_attempts_in_behavioral_denominators": False,
            "preserved_nonbehavioral_attempts": len(invalid_attempts),
            "analysis_unit": "environment seed; eight stochastic policy rollouts are nested repeated measurements",
        },
        "exact_prompts": dict(PROMPTS),
        "release_identity": {
            "queue_sha256": QUEUE_SHA256,
            "release_manifest_sha256": RELEASE_MANIFEST_SHA256,
            "released_cells": 432,
        },
        "raw_storage": {"root": str(raw_root), "committed_to_git": False},
        "lane_provenance": [
            {"lane_pod_uid": key[0], "lane_gpu_uuid": key[1], "valid_behavioral_episodes": count}
            for key, count in sorted(lane_counts.items())
        ],
        "success": success,
        "nested_condition_success_probabilities": conditions,
        "failure_taxonomy": _failure_summary(compact_episodes),
        "matched_pair_response": _matched_summary(compact_pairs),
        "continuous_measurements": _continuous_summary(compact_episodes),
        "sampling_index_success": {
            str(index): {
                relation: {
                    "successes": sum(row["success"] for row in compact_episodes if row["shared_policy_sampling_seed_index"] == index and row["requested_relation"] == relation),
                    "episodes": len(SEEDS),
                }
                for relation in RELATIONS
            }
            for index in SAMPLING_INDICES
        },
        "uncertainty_contract": {
            "primary_unit": "environment_seed",
            "bootstrap_master_seed": BOOTSTRAP_MASTER_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_method": "environment-seed cluster nonparametric percentile",
            "episode_level_Wilson_intervals": "descriptive only because eight rollouts are nested within each fixed scene/direction",
            "missing_value_imputation": "none",
        },
    }
    summary_path = output_dir / SUMMARY_NAME
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": "vla-wam-shared-v3d001-pi05-evidence-manifest-v1",
        "study_id": summary["study_id"],
        "registration_id": REGISTRATION_ID,
        "source_release": summary["release_identity"],
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (episodes_path, pairs_path, invalid_path, summary_path)
        },
        "counts": {
            "compact_episode_rows": len(compact_episodes),
            "compact_pair_rows": len(compact_pairs),
            "preserved_nonbehavioral_attempt_rows": len(invalid_attempts),
        },
    }
    manifest_path = output_dir / EVIDENCE_MANIFEST_NAME
    _write_json(manifest_path, manifest)
    return {"summary": summary, "manifest": manifest, "output_dir": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(**vars(args))
    print(json.dumps({
        "output_dir": result["output_dir"],
        "status": result["summary"]["status"],
        "counts": result["manifest"]["counts"],
        "summary_sha256": result["manifest"]["files"][SUMMARY_NAME]["sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
