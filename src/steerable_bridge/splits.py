from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from .text import stable_id


def _bin(value: float, edges: tuple[float, float, float]) -> int:
    return int(value > edges[0]) + int(value > edges[1]) + int(value > edges[2])


def _largest_remainder(counts: Mapping[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if target > total:
        raise ValueError(f"Cannot select {target} records from {total}")
    exact = {key: target * count / total for key, count in counts.items()}
    allocated = {key: min(counts[key], math.floor(value)) for key, value in exact.items()}
    remaining = target - sum(allocated.values())
    order = sorted(
        counts,
        key=lambda key: (exact[key] - allocated[key], counts[key], key),
        reverse=True,
    )
    for key in order:
        if remaining == 0:
            break
        if allocated[key] < counts[key]:
            allocated[key] += 1
            remaining -= 1
    if remaining:
        raise AssertionError("Largest-remainder allocation failed to reach target")
    return allocated


def stratified_split(
    rows: Iterable[Mapping[str, Any]],
    split_counts: Mapping[str, int],
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    candidates = [dict(row) for row in rows]
    total_target = sum(split_counts.values())
    if len(candidates) < total_target:
        raise ValueError(
            f"Need {total_target} eligible trajectories, found {len(candidates)}"
        )

    lengths = np.asarray([float(row["episode_length"]) for row in candidates])
    segments = np.asarray([float(row["semantic_segment_count"]) for row in candidates])
    task_lengths = np.asarray(
        [len(str(row.get("normalized_task", "")).split()) for row in candidates],
        dtype=float,
    )
    length_edges = tuple(float(value) for value in np.quantile(lengths, [0.25, 0.5, 0.75]))
    segment_edges = tuple(float(value) for value in np.quantile(segments, [0.25, 0.5, 0.75]))
    task_length_edges = tuple(
        float(value) for value in np.quantile(task_lengths, [0.25, 0.5, 0.75])
    )

    for row in candidates:
        stratum = (
            f"{row['source_collection']}|"
            f"task_length_q{_bin(len(str(row.get('normalized_task', '')).split()), task_length_edges)}|"
            f"length_q{_bin(float(row['episode_length']), length_edges)}|"
            f"segments_q{_bin(float(row['semantic_segment_count']), segment_edges)}"
        )
        row["selection_stratum"] = stratum
        row["task_family_diagnostic"] = str(row.get("task_family", "unknown"))

    # Assign each original Bridge file to exactly one split before sampling
    # trajectories. Multiple episodes from one out.npy share capture context and
    # should never straddle train/evaluation even though their episode IDs differ.
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        group_key = str(
            row.get("original_bridge_path")
            or f"trajectory:{row['steering_trajectory_id']}"
        )
        row["split_group_key"] = group_key
        group_rows[group_key].append(row)

    split_names = list(split_counts)
    proportions = [split_counts[name] / total_target for name in split_names]
    candidates_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in split_names
    }
    groups_by_source: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for group_key, values in group_rows.items():
        source = str(values[0]["source_collection"])
        groups_by_source[source].append((group_key, values))
    for source, source_groups in groups_by_source.items():
        total_source_rows = sum(len(values) for _, values in source_groups)
        desired = {
            split: total_source_rows * proportion
            for split, proportion in zip(split_names, proportions)
        }
        assigned = {split: 0 for split in split_names}
        ordered_groups = sorted(
            source_groups,
            key=lambda item: (
                -len(item[1]),
                stable_id("group_order", seed, source, item[0]),
            ),
        )
        for group_key, values in ordered_groups:
            chosen = max(
                split_names,
                key=lambda split: (
                    desired[split] - assigned[split],
                    desired[split],
                    stable_id("group_tie", seed, source, group_key, split),
                ),
            )
            candidates_by_split[chosen].extend(values)
            assigned[chosen] += len(values)

    result: dict[str, list[dict[str, Any]]] = {split: [] for split in split_names}
    for split, target in split_counts.items():
        available = candidates_by_split[split]
        if len(available) < target:
            raise ValueError(
                f"Group-safe assignment left {len(available)} candidates for "
                f"{split}, below target {target}; change the seed or lower the target"
            )
        available_by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in available:
            available_by_stratum[str(row["selection_stratum"])].append(row)
        quotas = _largest_remainder(
            {key: len(values) for key, values in available_by_stratum.items()},
            target,
        )
        selected: list[dict[str, Any]] = []
        for stratum, values in available_by_stratum.items():
            ranked = sorted(
                values,
                key=lambda row: stable_id(
                    "rank",
                    seed,
                    split,
                    stratum,
                    row["steering_trajectory_id"],
                    row["lerobot_episode_index"],
                ),
            )
            selected.extend(ranked[: quotas[stratum]])
        for row in selected:
            output = dict(row)
            output["split"] = split
            result[split].append(output)

    observed = {split: len(rows) for split, rows in result.items()}
    if observed != dict(split_counts):
        raise AssertionError(f"Split counts differ: expected {split_counts}, got {observed}")
    selected_ids = [
        row["steering_trajectory_id"] for rows in result.values() for row in rows
    ]
    if len(selected_ids) != len(set(selected_ids)):
        raise AssertionError("Trajectory leakage across splits")
    group_membership: dict[str, set[str]] = defaultdict(set)
    for split, rows in result.items():
        for row in rows:
            group_membership[str(row["split_group_key"])].add(split)
    leaked_groups = [key for key, splits in group_membership.items() if len(splits) > 1]
    if leaked_groups:
        raise AssertionError(f"Original Bridge groups leak across splits: {leaked_groups[:3]}")
    return result


def nested_split(
    parent: Mapping[str, list[Mapping[str, Any]]],
    split_counts: Mapping[str, int],
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Choose a deterministic, role-preserving subset of an existing split."""

    if set(parent) != set(split_counts):
        raise ValueError("Nested split names must match the parent split names")
    result: dict[str, list[dict[str, Any]]] = {}
    for split, target in split_counts.items():
        values = [dict(row) for row in parent[split]]
        if len(values) < target:
            raise ValueError(f"Parent {split} has {len(values)} rows, needs {target}")
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in values:
            by_source[str(row["source_collection"])].append(row)
        source_quotas = _largest_remainder(
            {source: len(rows) for source, rows in by_source.items()}, target
        )
        chosen: list[dict[str, Any]] = []
        for source, source_rows in by_source.items():
            by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in source_rows:
                by_stratum[str(row["selection_stratum"])].append(row)
            quotas = _largest_remainder(
                {key: len(rows) for key, rows in by_stratum.items()},
                source_quotas[source],
            )
            for stratum, rows in by_stratum.items():
                ranked = sorted(
                    rows,
                    key=lambda row: stable_id(
                        "nested",
                        seed,
                        split,
                        source,
                        stratum,
                        row["steering_trajectory_id"],
                    ),
                )
                chosen.extend(ranked[: quotas[stratum]])
        result[split] = chosen
    return result


def split_payload(
    split_rows: Mapping[str, list[Mapping[str, Any]]],
    *,
    seed: int,
    eligibility_rules: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "seed": seed,
        "eligibility_rules": eligibility_rules,
        "counts": {split: len(rows) for split, rows in split_rows.items()},
        "splits": {
            split: [
                {
                    "lerobot_episode_index": int(row["lerobot_episode_index"]),
                    "steering_trajectory_id": int(row["steering_trajectory_id"]),
                    "normalized_task": row["normalized_task"],
                    "source_collection": row["source_collection"],
                    "episode_length": int(row["episode_length"]),
                    "semantic_segment_count": int(row["semantic_segment_count"]),
                    "selection_stratum": row["selection_stratum"],
                    "task_family_diagnostic": row.get(
                        "task_family_diagnostic", "unknown"
                    ),
                    "split_group_key": row["split_group_key"],
                }
                for row in rows
            ]
            for split, rows in split_rows.items()
        },
    }


def balance_rows(split_rows: Mapping[str, list[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split, rows in split_rows.items():
        by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_source[str(row["source_collection"])].append(row)
        for source, source_rows in sorted(by_source.items()):
            lengths = [int(row["episode_length"]) for row in source_rows]
            segments = [int(row["semantic_segment_count"]) for row in source_rows]
            output.append(
                {
                    "split": split,
                    "source_collection": source,
                    "trajectory_count": len(source_rows),
                    "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
                    "median_episode_length": float(np.median(lengths)) if lengths else 0.0,
                    "mean_segment_count": float(np.mean(segments)) if segments else 0.0,
                    "median_segment_count": float(np.median(segments)) if segments else 0.0,
                }
            )
    return output
