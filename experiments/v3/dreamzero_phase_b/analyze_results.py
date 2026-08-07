#!/usr/bin/env python3
"""Compile the registered DreamZero V3-B003 mirror analysis from 108 cells."""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from experiments.v3.dreamzero_phase_b.contract import (
    AMENDMENT_ID,
    ARMS,
    MODEL_ID,
    PROMPTS,
    RELATIONS,
    SEEDS,
    STUDY_ID,
    load_cells,
    sha256_file,
)
from experiments.v3.pi05_phase_b.compiler import analyze_pairs, canonical_json_bytes
from tools.vla_wam_v3_episode_schema import parse_jsonl_record


REPORT_SCHEMA = "vla-wam-shared-v3b-dreamzero-mirror-results-v1"
EPISODE_SCHEMA = "vla-wam-shared-v3b-dreamzero-mirror-compact-episode-v1"
RESULTS_MANIFEST_SCHEMA = "vla-wam-shared-v3b-dreamzero-mirror-results-manifest-v2"
CELL_MANIFEST_SCHEMA = "vla-wam-shared-v3b-dreamzero-cell-jsonl-manifest-v1"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 3_104_159
FAILURE_CATEGORIES = ("correct", "pick_failed", "transport_failed", "wrong_side", "release_failed")


class AnalysisError(RuntimeError):
    """Raised when retained evidence is incomplete or violates registration."""


def _fail(message: str) -> None:
    raise AnalysisError(message)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{label} must be finite")
    return float(value)


def _load_action(record: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    artifact = record.get("artifacts", {}).get("executed_action_trace", {})
    path_value = artifact.get("path") if isinstance(artifact, Mapping) else None
    if not isinstance(path_value, str):
        _fail("episode lacks executed action path")
    path = Path(path_value).resolve()
    if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
        _fail(f"executed action evidence hash changed: {path}")
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != 8 or not np.isfinite(array).all():
        _fail("executed action trace must be finite [N,8]")
    if array.shape[0] != record.get("actions_executed"):
        _fail("executed action count disagrees with episode")
    return array, {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _validate_episodes(
    repo_root: Path, rows: Iterable[Mapping[str, Any]]
) -> dict[tuple[int, str, str], dict[str, Any]]:
    registered = {cell.cell_id: cell for cell in load_cells(repo_root)}
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        cell_id = row.get("registered_cell_id")
        cell = registered.get(cell_id)
        if cell is None:
            _fail(f"unregistered behavioral cell: {cell_id}")
        expected = {
            "record_type": "behavioral_episode",
            "behavioral_result_valid": True,
            "study_id": STUDY_ID,
            "amendment_id": AMENDMENT_ID,
            "model_id": MODEL_ID,
            "arena": "droid_robolab",
            "environment_seed": cell.seed,
            "requested_relation": cell.relation,
            "prompt": PROMPTS[cell.relation],
            "arm": cell.arm,
        }
        for key, value in expected.items():
            if row.get(key) != value:
                _fail(f"{cell_id} disagrees with registered {key}")
        key = (cell.seed, cell.arm, cell.relation)
        if key in indexed:
            _fail(f"duplicate behavioral cell: {key}")
        if row.get("failure_taxonomy") not in FAILURE_CATEGORIES:
            _fail(f"{cell_id} has invalid failure taxonomy")
        if type(row.get("requested_success")) is not bool:
            _fail(f"{cell_id} success must be boolean")
        _finite(row.get("signed_final_lateral_offset_m"), f"{cell_id}.offset")
        _finite(row.get("requested_side_depth_m"), f"{cell_id}.depth")
        indexed[key] = row
    expected_keys = {
        (seed, arm, relation) for seed in SEEDS for arm in ARMS for relation in RELATIONS
    }
    if set(indexed) != expected_keys:
        _fail("episodes are not the exact 27-seed x 2-layout x 2-direction design")
    return indexed


def analyze(
    *,
    repo_root: Path,
    episode_rows: Iterable[Mapping[str, Any]],
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    episodes = _validate_episodes(repo_root, episode_rows)
    pair_rows: list[dict[str, Any]] = []
    action_sources: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            left = episodes[(seed, arm, "left")]
            right = episodes[(seed, arm, "right")]
            if left.get("initial_state_sha256") != right.get("initial_state_sha256"):
                _fail(f"seed {seed} {arm} does not share an identical reset")
            left_actions, left_source = _load_action(left)
            right_actions, right_source = _load_action(right)
            common = min(len(left_actions), len(right_actions))
            delta = left_actions[:common].astype(np.float64) - right_actions[:common].astype(np.float64)
            left_offset = float(left["signed_final_lateral_offset_m"])
            right_offset = float(right["signed_final_lateral_offset_m"])
            pair_rows.append({
                "schema_version": "vla-wam-shared-v3b-dreamzero-mirror-pair-v1",
                "study_id": STUDY_ID,
                "amendment_id": AMENDMENT_ID,
                "model_id": MODEL_ID,
                "seed": seed,
                "matched_block_id": left["pair_id"],
                "arm": arm,
                "left_registered_cell_id": left["registered_cell_id"],
                "right_registered_cell_id": right["registered_cell_id"],
                "initial_state_sha256": left["initial_state_sha256"],
                "endpoint_redirection_D_m": left_offset - right_offset,
                "endpoint_shift_m": left_offset - right_offset,
                "requested_side_depth_contrast_B_m": float(right["requested_side_depth_m"])
                - float(left["requested_side_depth_m"]),
                "left_success": left["requested_success"],
                "right_success": right["requested_success"],
                "right_minus_left_success": int(right["requested_success"])
                - int(left["requested_success"]),
                "executed_actions_distinct": not np.array_equal(
                    left_actions[:common], right_actions[:common]
                ),
                "action_distinct": not np.array_equal(left_actions[:common], right_actions[:common]),
                "action_distinct_definition": "bitwise inequality on complete common executed prefix",
                "left_executed_action_count": len(left_actions),
                "right_executed_action_count": len(right_actions),
                "common_prefix_action_count": common,
                "common_prefix_action_rms": float(math.sqrt(float(np.mean(delta * delta)))),
            })
            action_sources.extend([
                {"registered_cell_id": left["registered_cell_id"], **left_source},
                {"registered_cell_id": right["registered_cell_id"], **right_source},
            ])

    registered = analyze_pairs(
        pair_rows,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    complete_case_seeds = [
        seed
        for seed in SEEDS
        if all(episodes[(seed, arm, relation)]["requested_success"] for arm in ARMS for relation in RELATIONS)
    ]
    complete_case = []
    for seed in complete_case_seeds:
        row: dict[str, Any] = {"seed": seed}
        for arm in ARMS:
            left = episodes[(seed, arm, "left")]
            right = episodes[(seed, arm, "right")]
            row[arm] = {
                "left_requested_side_depth_m": left["requested_side_depth_m"],
                "right_requested_side_depth_m": right["requested_side_depth_m"],
                "right_minus_left_depth_m": float(right["requested_side_depth_m"])
                - float(left["requested_side_depth_m"]),
            }
        complete_case.append(row)
    failure_counts = {
        arm: {
            relation: dict(sorted(Counter(
                episodes[(seed, arm, relation)]["failure_taxonomy"] for seed in SEEDS
            ).items()))
            for relation in RELATIONS
        }
        for arm in ARMS
    }
    condition_outcomes = {
        f"{arm}:{relation}": {
            "episodes": len(SEEDS),
            "successes": sum(
                int(episodes[(seed, arm, relation)]["requested_success"])
                for seed in SEEDS
            ),
            "failure_taxonomy_counts": failure_counts[arm][relation],
        }
        for arm in ARMS
        for relation in RELATIONS
    }
    overall_failures = Counter(
        episodes[(seed, arm, relation)]["failure_taxonomy"]
        for seed in SEEDS
        for arm in ARMS
        for relation in RELATIONS
    )
    full_sample_primary = {
        "population": registered["population"],
        "formulas": registered["formulas"],
        "D_by_arm": registered["H1_endpoint_redirection"]["paired_contrast_by_layout"],
        "B_by_arm": registered["H2_requested_side_depth"]["paired_contrast_by_layout"],
        "J_redirection_interaction": registered["H1_endpoint_redirection"][
            "reflected_minus_control_interaction"
        ],
        "I_position_reflection_interaction": registered["H2_requested_side_depth"][
            "reflected_minus_control_interaction"
        ],
        "binary_success_DiD": registered["H3_binary_success"],
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": "droid_robolab",
        "exact_prompts": PROMPTS,
        "population": registered["population"],
        "condition_outcomes": condition_outcomes,
        "full_sample_primary": full_sample_primary,
        "registered_analysis": registered,
        "requested_margin_secondary": {
            "subset_definition": "seeds with all four control/reflected LEFT/RIGHT cells correct",
            "realized_seed_n": len(complete_case),
            "rows": complete_case,
            "missing_value_policy": "non-complete seeds omitted, never encoded as zero",
        },
        "failure_taxonomy_counts": {
            category: overall_failures.get(category, 0) for category in FAILURE_CATEGORIES
        },
        "failure_taxonomy_by_condition": failure_counts,
        "pair_rows": pair_rows,
        "action_sources": sorted(action_sources, key=lambda row: row["registered_cell_id"]),
        "uncertainty_contract": {
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_master_seed": bootstrap_seed,
            "bootstrap_unit": "matched_seed",
            "continuous_test": "exact two-sided paired sign test; zero ties excluded",
            "binary_test": "exact within-seed control/reflected layout-label permutation",
        },
        "claim_boundary": (
            "The movable-object center-position reflection tests whether the directional gap changes with layout. "
            "Robot base, cameras, and nonmovable geometry remain fixed, so the result does not isolate training data, "
            "embodiment handedness, or a full-scene symmetry transformation."
        ),
    }


def _file_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        _fail(f"missing or empty evidence file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _load_episode(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        _fail(f"episode JSONL must contain one row: {path}")
    row = parse_jsonl_record(lines[0])
    source = _file_record(path)
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest_source = _file_record(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": CELL_MANIFEST_SCHEMA,
        "registered_cell_id": row.get("registered_cell_id"),
        "jsonl_sha256": source["sha256"],
        "jsonl_bytes": source["bytes"],
        "row_count": 1,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            _fail(f"cell manifest mismatch for {key}: {manifest_path}")
    row["_aggregate_source"] = {
        "behavioral_jsonl": source,
        "batch_manifest": manifest_source,
    }
    return row


def _verified_artifact(record: Mapping[str, Any], key: str) -> dict[str, Any]:
    artifact = record.get("artifacts", {}).get(key, {})
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
        _fail(f"{record.get('registered_cell_id')} lacks artifact {key}")
    source = _file_record(Path(artifact["path"]))
    if source["sha256"] != artifact.get("sha256") or source["bytes"] != artifact.get("bytes"):
        _fail(f"{record.get('registered_cell_id')} artifact binding changed: {key}")
    return source


def _compact_episode(record: Mapping[str, Any]) -> dict[str, Any]:
    source = record.get("_aggregate_source")
    if not isinstance(source, Mapping):
        _fail("compact episode requires verified aggregate source")
    return {
        "schema_version": EPISODE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": "droid_robolab",
        "registered_cell_id": record["registered_cell_id"],
        "seed": record["environment_seed"],
        "arm": record["arm"],
        "requested_relation": record["requested_relation"],
        "prompt": record["prompt"],
        "requested_success": record["requested_success"],
        "failure_category": record["failure_taxonomy"],
        "signed_final_lateral_offset_m": record["signed_final_lateral_offset_m"],
        "requested_side_depth_m": record["requested_side_depth_m"],
        "cone_entry_step": record.get("cone_entry_step"),
        "cone_entry_sustained": record.get("cone_entry_sustained"),
        "episode_length_steps": record.get("episode_length_steps"),
        "time_to_first_contact": record.get("time_to_first_contact"),
        "grasp_step": record.get("grasp_step"),
        "cumulative_lateral_path_m": record.get("cumulative_lateral_path_m"),
        "peak_lateral_excursion_m": record.get("peak_lateral_excursion_m"),
        "object_path_length_m": record.get("object_path_length_m"),
        "actions_executed": record["actions_executed"],
        "right_censored": record["right_censored"],
        "initial_state_sha256": record["initial_state_sha256"],
        "source_behavioral_jsonl": dict(source["behavioral_jsonl"]),
        "source_batch_manifest": dict(source["batch_manifest"]),
        "executed_action_trace": _verified_artifact(record, "executed_action_trace"),
        "viewport_video": _verified_artifact(record, "viewport_video"),
        "exposed_future_manifest": _verified_artifact(record, "exposed_future_manifest"),
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite {path}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--episode-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    episode_rows = [_load_episode(path) for path in args.episode_jsonl]
    report = analyze(
        repo_root=args.repo_root,
        episode_rows=episode_rows,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    summary_path = output / "dreamzero_v3b003_summary.json"
    episodes_path = output / "dreamzero_v3b003_episodes.jsonl"
    pairs_path = output / "dreamzero_v3b003_matched_pairs.jsonl"
    summary_payload = canonical_json_bytes(report)
    compact_rows = sorted(
        (_compact_episode(row) for row in episode_rows),
        key=lambda row: (row["seed"], row["arm"], row["requested_relation"]),
    )
    episode_payload = b"".join(canonical_json_bytes(row) for row in compact_rows)
    pair_payload = b"".join(
        canonical_json_bytes(row)
        for row in sorted(report["pair_rows"], key=lambda row: (row["seed"], row["arm"]))
    )
    _write_exclusive(summary_path, summary_payload)
    _write_exclusive(episodes_path, episode_payload)
    _write_exclusive(pairs_path, pair_payload)
    manifest = {
        "schema_version": RESULTS_MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "files": [
            _file_record(summary_path),
            _file_record(episodes_path),
            _file_record(pairs_path),
        ],
        "episode_sources": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.episode_jsonl
        ],
    }
    manifest["aggregate_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest["files"])
    ).hexdigest()
    manifest_path = output / "evidence_manifest.json"
    _write_exclusive(manifest_path, canonical_json_bytes(manifest))
    print(json.dumps({"output_dir": str(output), "manifest": _file_record(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
