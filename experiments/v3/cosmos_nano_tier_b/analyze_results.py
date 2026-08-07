#!/usr/bin/env python3
"""Compile complete V3-B008/B009 Nano interaction cohorts.

This is an evidence compiler, not a permissive plotting helper.  It accepts
only the exact released cells, verifies every one-row JSONL and its post-close
manifest, rechecks every retained executed-action trace, and refuses partial
matched blocks.  Behavioral failures remain measurements; infrastructure
attempts are never read from the behavioral root.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.v3.cosmos_nano_tier_b.runtime_contract import (
    CHECKPOINT_REVISION,
    CONFIG,
    MODEL_ID,
    MODEL_REPOSITORY,
    STUDY_ID,
    ReleaseBundle,
    load_release,
    sha256_file,
)
from experiments.v3.pi05_phase_b.compiler import (
    continuous_summary,
    exact_layout_swap_permutation,
)
from tools.vla_wam_v3_episode_schema import parse_jsonl_record


REPORT_SCHEMA = "vla-wam-shared-v3b-nano-factor-results-v1"
PAIR_SCHEMA = "vla-wam-shared-v3b-nano-factor-pair-v1"
EPISODE_SCHEMA = "vla-wam-shared-v3b-nano-factor-compact-episode-v1"
MANIFEST_SCHEMA = "vla-wam-shared-v3b-nano-factor-results-manifest-v1"
BATCH_MANIFEST_SCHEMA = "vla-wam-shared-v3-jsonl-batch-manifest-v1"
BEHAVIORAL_SCHEMA = "vla-wam-shared-v3-raw-episode-v1"
FAILURE_CATEGORIES = (
    "correct",
    "pick_failed",
    "transport_failed",
    "wrong_side",
    "release_failed",
)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 3_104_159


class AnalysisError(RuntimeError):
    """Raised when retained evidence is partial or violates its release."""


def _fail(message: str) -> None:
    raise AnalysisError(message)


def _canonical(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"result is not finite canonical JSON: {exc}") from exc


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _file_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        _fail(f"missing or empty evidence file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _load_one_row(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _file_record(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1 or not lines[0].strip():
        _fail(f"behavioral JSONL must contain exactly one non-empty row: {path}")
    row = parse_jsonl_record(lines[0])
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest_source = _file_record(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": BATCH_MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "jsonl_path": str(path.resolve()),
        "jsonl_sha256": source["sha256"],
        "jsonl_bytes": source["bytes"],
        "row_count": 1,
        "record_schema_versions": [BEHAVIORAL_SCHEMA],
    }
    for key, wanted in expected.items():
        if manifest.get(key) != wanted:
            _fail(f"post-close manifest mismatch for {key}: {manifest_path}")
    return row, {"jsonl": source, "batch_manifest": manifest_source}


def _load_actions(record: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    artifact = record.get("artifacts", {}).get("executed_action_trace", {})
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
        _fail(f"{record.get('registered_cell_id')} lacks an executed action trace")
    source = _file_record(Path(artifact["path"]))
    if artifact.get("sha256") != source["sha256"] or artifact.get("bytes") != source["bytes"]:
        _fail(f"executed action trace binding changed: {source['path']}")
    try:
        actions = np.load(source["path"], allow_pickle=False)
    except Exception as exc:
        raise AnalysisError(f"cannot load action trace {source['path']}: {exc}") from exc
    if (
        actions.ndim != 2
        or actions.shape[1] != 8
        or actions.shape[0] != record.get("actions_executed")
        or not np.issubdtype(actions.dtype, np.number)
        or not np.isfinite(actions).all()
    ):
        _fail(f"invalid executed action trace: {source['path']}")
    return actions, source


def _validate_episode(
    row: Mapping[str, Any], *, release: ReleaseBundle, source: Mapping[str, Any]
) -> dict[str, Any]:
    record = dict(row)
    cell_id = record.get("registered_cell_id")
    if not isinstance(cell_id, str):
        _fail("behavioral row lacks registered_cell_id")
    cell = release.cell(cell_id)
    expected = {
        "schema_version": BEHAVIORAL_SCHEMA,
        "record_type": "behavioral_episode",
        "behavioral_result_valid": True,
        "study_id": STUDY_ID,
        "amendment_id": release.amendment_id,
        "model_id": MODEL_ID,
        "arena": "droid_robolab",
        "environment_seed": cell.seed,
        "policy_seed": cell.seed,
        "requested_relation": cell.relation,
        "prompt": cell.row["prompt"],
        "prompt_family": "direct_command",
        "arm": cell.arm,
        "release_manifest_sha256": release.manifest_sha256,
        "release_fingerprint_sha256": release.release_fingerprint(cell),
        "future_interface": "decoded_rgb_uint8_33_frames_per_policy_request",
        "missing_future_policy": "infrastructure_invalid_never_zero",
    }
    for key, wanted in expected.items():
        if record.get(key) != wanted:
            _fail(f"{cell_id} disagrees with released {key}")
    if record.get("checkpoint") != {
        "id": MODEL_REPOSITORY,
        "revision": CHECKPOINT_REVISION,
    }:
        _fail(f"{cell_id} checkpoint identity changed")
    if type(record.get("requested_success")) is not bool:
        _fail(f"{cell_id} success must be boolean")
    taxonomy = record.get("failure_taxonomy")
    if taxonomy not in FAILURE_CATEGORIES:
        _fail(f"{cell_id} has invalid failure taxonomy: {taxonomy}")
    if record["requested_success"] != (taxonomy == "correct"):
        _fail(f"{cell_id} success and failure taxonomy disagree")
    signed = _finite(record.get("signed_final_lateral_offset_m"), f"{cell_id}.offset")
    depth = _finite(record.get("requested_side_depth_m"), f"{cell_id}.depth")
    expected_depth = signed if cell.relation == "left" else -signed
    if not math.isclose(depth, expected_depth, rel_tol=0.0, abs_tol=1e-12):
        _fail(f"{cell_id} requested-side depth is inconsistent with signed offset")
    actions, action_source = _load_actions(record)
    compact = {
        "schema_version": EPISODE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": release.amendment_id,
        "model_id": MODEL_ID,
        "arena": "droid_robolab",
        "registered_cell_id": cell_id,
        "seed": cell.seed,
        "arm": cell.arm,
        "requested_relation": cell.relation,
        "prompt": cell.row["prompt"],
        "target_object": record.get("target_object"),
        "reference_object": record.get("reference_object"),
        "requested_success": record["requested_success"],
        "failure_category": taxonomy,
        "signed_final_lateral_offset_m": signed,
        "requested_side_depth_m": depth,
        "cone_entry_step": record.get("cone_entry_step"),
        "cone_entry_sustained": record.get("cone_entry_sustained"),
        "episode_length_steps": record.get("episode_length_steps"),
        "time_to_first_contact_steps": record.get("time_to_first_contact_steps"),
        "first_contact_status": record.get("first_contact_status"),
        "grasp_step": record.get("grasp_step"),
        "cumulative_lateral_path_m": record.get("cumulative_lateral_path_m"),
        "peak_lateral_excursion_m": record.get("peak_lateral_excursion_m"),
        "object_path_length_m": record.get("measurements", {}).get("object_path_length_m"),
        "actions_executed": record.get("actions_executed"),
        "right_censored": record.get("right_censored"),
        "initial_state_sha256": record.get("initial_state_sha256"),
        "release_fingerprint_sha256": record.get("release_fingerprint_sha256"),
        "source_behavioral_jsonl": dict(source["jsonl"]),
        "source_batch_manifest": dict(source["batch_manifest"]),
        "executed_action_trace": action_source,
    }
    for name in (
        "cumulative_lateral_path_m",
        "peak_lateral_excursion_m",
        "object_path_length_m",
    ):
        _finite(compact[name], f"{cell_id}.{name}")
    return {"record": record, "compact": compact, "actions": actions}


def _summarize(
    values: Sequence[float], *, label: str, replicates: int, seed: int
) -> dict[str, Any]:
    return continuous_summary(
        values,
        label=label,
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
    )


def _binary_test(values: Sequence[int]) -> dict[str, Any]:
    result = exact_layout_swap_permutation(values)
    result["method"] = "exact_two_sided_within_seed_factor_label_sign_flip_permutation"
    result["test_statistic"] = "absolute_sum_of_per_seed_direction_gap_interactions"
    return result


def _condition_tables(
    episodes: Mapping[tuple[int, str, str], Mapping[str, Any]],
    *,
    release: ReleaseBundle,
) -> tuple[dict[str, Any], dict[str, Any]]:
    successes: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    seeds = release.config["seed_range"]
    for arm in release.config["arms"]:
        successes[arm] = {}
        failures[arm] = {}
        for relation in release.config["relations"]:
            rows = [episodes[(seed, arm, relation)]["compact"] for seed in seeds]
            count = sum(int(row["requested_success"]) for row in rows)
            successes[arm][relation] = {
                "successes": count,
                "episodes": len(rows),
                "failures": len(rows) - count,
            }
            taxonomy = Counter(row["failure_category"] for row in rows)
            failures[arm][relation] = {
                category: taxonomy.get(category, 0) for category in FAILURE_CATEGORIES
            }
    return successes, failures


def _factor_contrasts(
    pairs: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    release: ReleaseBundle,
    replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    arms = release.config["arms"]
    seeds = release.config["seed_range"]
    by_arm = {
        arm: {
            "endpoint_redirection_D_m": _summarize(
                [float(pairs[(seed, arm)]["endpoint_redirection_D_m"]) for seed in seeds],
                label=f"{release.amendment_id}:{arm}:D",
                replicates=replicates,
                seed=bootstrap_seed,
            ),
            "requested_side_depth_contrast_B_m": _summarize(
                [float(pairs[(seed, arm)]["requested_side_depth_contrast_B_m"]) for seed in seeds],
                label=f"{release.amendment_id}:{arm}:B",
                replicates=replicates,
                seed=bootstrap_seed,
            ),
        }
        for arm in arms
    }
    contrasts: dict[str, Any] = {}
    for lower_index, lower in enumerate(arms):
        for upper in arms[lower_index + 1 :]:
            label = f"{upper}_minus_{lower}"
            d = [
                float(pairs[(seed, upper)]["endpoint_redirection_D_m"])
                - float(pairs[(seed, lower)]["endpoint_redirection_D_m"])
                for seed in seeds
            ]
            b = [
                float(pairs[(seed, upper)]["requested_side_depth_contrast_B_m"])
                - float(pairs[(seed, lower)]["requested_side_depth_contrast_B_m"])
                for seed in seeds
            ]
            binary = [
                int(pairs[(seed, upper)]["right_minus_left_success"])
                - int(pairs[(seed, lower)]["right_minus_left_success"])
                for seed in seeds
            ]
            contrasts[label] = {
                "definition": f"direction contrast in {upper} minus direction contrast in {lower}",
                "endpoint_redirection_interaction_m": _summarize(
                    d,
                    label=f"{release.amendment_id}:{label}:D",
                    replicates=replicates,
                    seed=bootstrap_seed,
                ),
                "requested_side_depth_interaction_m": _summarize(
                    b,
                    label=f"{release.amendment_id}:{label}:B",
                    replicates=replicates,
                    seed=bootstrap_seed,
                ),
                "binary_success_interaction": {
                    "per_seed_distribution": dict(sorted(Counter(binary).items())),
                    "mean": float(np.mean(binary)),
                    "median": float(np.median(binary)),
                    "exact_permutation_test": _binary_test(binary),
                },
            }
    result: dict[str, Any] = {
        "direction_contrast_by_factor_level": by_arm,
        "pairwise_factor_interactions": contrasts,
    }
    if release.amendment_id == "V3-B008":
        first_seed = release.config["seed_range"][0]
        cell_by_arm = {
            cell.arm: cell
            for cell in release.cells
            if cell.seed == first_seed and cell.relation == "left"
        }
        level_by_arm = {
            arm: float(cell_by_arm[arm].row["fixture_positions_robot_base_m"]["rubiks_cube"][1])
            - float(cell_by_arm[arm].row["fixture_positions_robot_base_m"]["bowl"][1])
            for arm in arms
        }
        ordered_arms = tuple(sorted(arms, key=level_by_arm.__getitem__))
        x = np.asarray([level_by_arm[arm] for arm in ordered_arms], dtype=np.float64)
        centered = x - x.mean()
        denominator = float(np.dot(centered, centered))
        if denominator <= 0.0 or not np.all(np.diff(x) > 0.0):
            _fail("V3-B008 released start-side levels are not strictly ordered")

        def slopes(field: str) -> list[float]:
            return [
                float(np.dot(
                    centered,
                    np.asarray([float(pairs[(seed, arm)][field]) for arm in ordered_arms])
                    - np.mean([float(pairs[(seed, arm)][field]) for arm in ordered_arms]),
                ) / denominator)
                for seed in seeds
            ]

        result["ordered_start_side_trend"] = {
            "factor": "initial target-minus-reference lateral offset in robot-base meters",
            "factor_levels_m": {arm: level_by_arm[arm] for arm in arms},
            "endpoint_redirection_D_slope_per_m": _summarize(
                slopes("endpoint_redirection_D_m"),
                label=f"{release.amendment_id}:ordered_start_side:D_slope",
                replicates=replicates,
                seed=bootstrap_seed,
            ),
            "requested_side_depth_B_slope_per_m": _summarize(
                slopes("requested_side_depth_contrast_B_m"),
                label=f"{release.amendment_id}:ordered_start_side:B_slope",
                replicates=replicates,
                seed=bootstrap_seed,
            ),
            "binary_direction_gap_slope_per_m": _summarize(
                slopes("right_minus_left_success"),
                label=f"{release.amendment_id}:ordered_start_side:success_gap_slope",
                replicates=replicates,
                seed=bootstrap_seed,
            ),
            "interpretation": "Linear three-level trend across the exact released fixture; pairwise contrasts remain reported without assuming linearity.",
        }
    return result


def analyze(
    *,
    repo_root: Path,
    amendment_id: str,
    raw_root: Path,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = CONFIG[amendment_id]
    manifest = repo_root / cfg["release_dir"] / "release_manifest.json"
    release = load_release(repo_root, amendment_id, manifest)
    paths = sorted(raw_root.rglob("raw_episode.jsonl"))
    if len(paths) != cfg["cells"]:
        _fail(f"expected {cfg['cells']} behavioral JSONLs, found {len(paths)}")
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    seen_cells: set[str] = set()
    for path in paths:
        row, source = _load_one_row(path)
        validated = _validate_episode(row, release=release, source=source)
        compact = validated["compact"]
        cell_id = compact["registered_cell_id"]
        key = (compact["seed"], compact["arm"], compact["requested_relation"])
        if cell_id in seen_cells or key in indexed:
            _fail(f"duplicate behavioral evidence: {cell_id}")
        seen_cells.add(cell_id)
        indexed[key] = validated
    if seen_cells != set(release.by_cell_id):
        _fail("behavioral evidence is not the exact released cell set")

    pair_rows: list[dict[str, Any]] = []
    pair_index: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in cfg["seed_range"]:
        for arm in cfg["arms"]:
            left = indexed[(seed, arm, "left")]
            right = indexed[(seed, arm, "right")]
            if left["record"].get("initial_state_sha256") != right["record"].get("initial_state_sha256"):
                _fail(f"LEFT/RIGHT reset mismatch for seed {seed}, arm {arm}")
            common = min(len(left["actions"]), len(right["actions"]))
            if common <= 0:
                _fail(f"empty common action prefix for seed {seed}, arm {arm}")
            delta = left["actions"][:common].astype(np.float64) - right["actions"][:common].astype(np.float64)
            left_offset = float(left["compact"]["signed_final_lateral_offset_m"])
            right_offset = float(right["compact"]["signed_final_lateral_offset_m"])
            row = {
                "schema_version": PAIR_SCHEMA,
                "study_id": STUDY_ID,
                "amendment_id": amendment_id,
                "model_id": MODEL_ID,
                "arena": "droid_robolab",
                "seed": seed,
                "arm": arm,
                "left_registered_cell_id": left["compact"]["registered_cell_id"],
                "right_registered_cell_id": right["compact"]["registered_cell_id"],
                "initial_state_sha256": left["record"]["initial_state_sha256"],
                "endpoint_redirection_D_m": left_offset - right_offset,
                "endpoint_shift_m": left_offset - right_offset,
                "requested_side_depth_contrast_B_m": float(right["compact"]["requested_side_depth_m"])
                - float(left["compact"]["requested_side_depth_m"]),
                "left_success": left["compact"]["requested_success"],
                "right_success": right["compact"]["requested_success"],
                "right_minus_left_success": int(right["compact"]["requested_success"])
                - int(left["compact"]["requested_success"]),
                "action_distinct": not np.array_equal(
                    left["actions"][:common], right["actions"][:common]
                ),
                "action_distinct_definition": "bitwise inequality on complete common executed prefix",
                "common_prefix_action_count": common,
                "common_prefix_action_rms": float(math.sqrt(float(np.mean(delta * delta)))),
            }
            pair_rows.append(row)
            pair_index[(seed, arm)] = row

    success_table, failure_table = _condition_tables(indexed, release=release)
    factor = _factor_contrasts(
        pair_index,
        release=release,
        replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    compact_episodes = [
        indexed[key]["compact"]
        for key in sorted(indexed, key=lambda value: (value[0], value[1], value[2]))
    ]
    report = {
        "schema_version": REPORT_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": amendment_id,
        "model_id": MODEL_ID,
        "arena": "droid_robolab",
        "population": {
            "matched_seed_count": len(cfg["seed_range"]),
            "behavioral_episode_count": len(compact_episodes),
            "matched_left_right_pair_count": len(pair_rows),
            "valid_behavioral_failures_included": True,
            "infrastructure_attempts_included": False,
            "missing_value_imputation": "none",
        },
        "formulas": {
            "signed_offset": "s; positive robot-base Y is robot LEFT",
            "endpoint_redirection_D": "s_LEFT - s_RIGHT; positive follows requested LEFT-to-RIGHT ordering",
            "requested_side_depth_B": "depth_RIGHT - depth_LEFT = -s_RIGHT - s_LEFT",
            "binary_direction_gap": "success_RIGHT - success_LEFT",
            "pairwise_factor_interaction": "direction contrast at second named factor level minus direction contrast at first named factor level",
        },
        "exact_prompts": {
            f"{cell.arm}:{cell.relation}": cell.row["prompt"] for cell in release.cells[: len(cfg["arms"]) * 2]
        },
        "success_table": success_table,
        "failure_taxonomy_counts": failure_table,
        "factor_analysis": factor,
        "action_diagnostics": {
            "distinct_pairs": sum(int(row["action_distinct"]) for row in pair_rows),
            "pairs": len(pair_rows),
        },
        "uncertainty_contract": {
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_master_seed": bootstrap_seed,
            "bootstrap_unit": "matched_seed",
            "continuous_test": "exact two-sided paired sign test; zero ties excluded",
            "binary_test": "exact two-sided within-seed factor-label sign-flip permutation",
        },
        "claim_boundary": (
            "Start-side interactions combine geometry, reachability, and policy state dependence; they do not isolate training data."
            if amendment_id == "V3-B008"
            else "Role swap changes object semantics and physical affordance together; the interaction is not a language-only effect."
        ),
    }
    return report, compact_episodes, pair_rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(_canonical(dict(row)) for row in rows)
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--amendment-id", choices=tuple(CONFIG), required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap_replicates < 10_000:
        parser.error("registered continuous analysis requires at least 10,000 bootstrap resamples")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    report, episodes, pairs = analyze(
        repo_root=args.repo_root.resolve(),
        amendment_id=args.amendment_id,
        raw_root=args.raw_root.resolve(),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    output.mkdir(parents=True)
    stem = args.amendment_id.lower().replace("-", "")
    report_path = output / f"{stem}_summary.json"
    episodes_path = output / f"{stem}_episodes.jsonl"
    pairs_path = output / f"{stem}_matched_pairs.jsonl"
    report_path.write_bytes(_canonical(report))
    _write_jsonl(episodes_path, episodes)
    _write_jsonl(pairs_path, pairs)
    files = [_file_record(path) for path in (report_path, episodes_path, pairs_path)]
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": args.amendment_id,
        "files": files,
        "aggregate_sha256": hashlib.sha256(_canonical(files)).hexdigest(),
    }
    manifest_path = output / "evidence_manifest.json"
    manifest_path.write_bytes(_canonical(manifest))
    print(json.dumps({"output_dir": str(output), "manifest": _file_record(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
