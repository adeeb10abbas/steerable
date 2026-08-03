from __future__ import annotations

import json
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Iterable, Iterator, Mapping

from .constants import LEROBOT_REPO, LEROBOT_REVISION
from .text import (
    normalize_semantic_text,
    normalize_text,
    source_collection,
    stable_id,
    task_family,
    unique_normalized_strings,
)


def join_normalization_sensitivity(
    commands: Mapping[str, Any],
    episodes: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Recompute the vocabulary-disambiguated join under explicit form rules."""

    def raw_exact(value: str) -> str:
        return value.strip()

    def case_whitespace(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    def ascii_alphanumeric(value: str) -> str:
        return normalize_text(value)

    def remove_words(value: str, removed: set[str]) -> str:
        return " ".join(token for token in normalize_text(value).split() if token not in removed)

    normalizers = {
        "raw_exact_strip": raw_exact,
        "nfkc_case_whitespace": case_whitespace,
        "nfkc_case_ascii_alphanumeric": ascii_alphanumeric,
        "ascii_alphanumeric_remove_articles": lambda value: remove_words(
            value, {"a", "an", "the"}
        ),
        "ascii_alphanumeric_remove_articles_robot": lambda value: remove_words(
            value, {"a", "an", "the", "robot"}
        ),
    }
    episode_list = list(episodes)
    output: list[dict[str, Any]] = []
    for name, normalizer in normalizers.items():
        def normalized_values(raw_values: object) -> set[str]:
            if not isinstance(raw_values, list):
                return set()
            result: set[str] = set()
            for raw_value in raw_values:
                if not isinstance(raw_value, str):
                    continue
                normalized = normalizer(raw_value)
                if normalized:
                    result.add(normalized)
            return result

        episodes_by_task: dict[str, list[int]] = defaultdict(list)
        for episode in episode_list:
            tasks = episode.get("tasks")
            if not isinstance(tasks, list):
                continue
            values = {
                normalizer(value)
                for value in tasks
                if isinstance(value, str) and normalizer(value)
            }
            for value in values:
                episodes_by_task[value].append(int(episode["episode_index"]))

        steering_by_task: dict[str, list[int]] = defaultdict(list)
        for raw_trajectory_id, raw_pools in commands.items():
            if not isinstance(raw_pools, dict) or not raw_pools:
                continue
            pool_sets = [normalized_values(raw_values) for raw_values in raw_pools.values()]
            common = set.intersection(*pool_sets) if pool_sets else set()
            for value in common:
                steering_by_task[value].append(int(raw_trajectory_id))

        pairs = [
            (episodes_by_task[value][0], steering_by_task[value][0])
            for value in set(episodes_by_task) & set(steering_by_task)
            if len(episodes_by_task[value]) == 1 and len(steering_by_task[value]) == 1
        ]
        output.append(
            {
                "normalization": name,
                "one_to_one_pairs": len(pairs),
                "same_raw_index_pairs": sum(episode == trajectory for episode, trajectory in pairs),
                "task_identity_warning": (
                    "stopword deletion is diagnostic only and is not allowed in the selected join"
                    if "remove" in name
                    else ""
                ),
            }
        )
    return output


def _normalized_pool(pool: object) -> dict[str, str]:
    return unique_normalized_strings(pool if isinstance(pool, list) else [])


def derive_steering_task_records(
    commands: Mapping[str, Any],
    trajectory_map: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Recover a task label as the string shared by every subtask pool.

    The release has no explicit trajectory-level task field. For multi-subtask
    trajectories, the original Bridge task instruction is repeated in each
    pool. One-pool and multi-intersection cases are intentionally rejected.
    """

    records: list[dict[str, Any]] = []
    index_to_key = trajectory_map.get("traj_idx_to_key", {})
    for raw_trajectory_id, raw_pools in commands.items():
        trajectory_id = int(raw_trajectory_id)
        path = ""
        original_episode_id: int | None = None
        original_key = index_to_key.get(str(trajectory_id))
        if isinstance(original_key, list) and len(original_key) == 2:
            path = str(original_key[0])
            try:
                original_episode_id = int(original_key[1])
            except (TypeError, ValueError):
                original_episode_id = None

        pools = raw_pools if isinstance(raw_pools, dict) else {}
        pool_maps = [_normalized_pool(values) for values in pools.values()]
        common = set.intersection(*(set(values) for values in pool_maps)) if pool_maps else set()
        originals: dict[str, str] = {}
        for pool in pool_maps:
            originals.update(pool)

        if not isinstance(raw_pools, dict) or not pools:
            extraction_status = "malformed_or_empty_command_map"
        elif len(pools) == 1:
            extraction_status = "one_subtask_pool_task_ambiguous"
        elif len(common) == 0:
            extraction_status = "no_shared_task_instruction"
        elif len(common) > 1:
            extraction_status = "multiple_shared_task_instructions"
        else:
            extraction_status = "clean"

        candidate_norm = sorted(common)
        canonical_norm = candidate_norm[0] if extraction_status == "clean" else ""
        records.append(
            {
                "steering_trajectory_id": trajectory_id,
                "canonical_task": originals.get(canonical_norm, ""),
                "normalized_task": canonical_norm,
                "task_candidate_count": len(common),
                "task_candidates": json.dumps(candidate_norm, ensure_ascii=False),
                "task_extraction_status": extraction_status,
                "subtask_pool_count": len(pools),
                "original_bridge_path": path,
                "original_bridge_episode_id": original_episode_id,
                "source_collection": source_collection(path),
                "task_family": task_family(path),
            }
        )
    return records


def derive_lerobot_episode_records(episodes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        task_values = episode.get("tasks")
        normalized = unique_normalized_strings(task_values if isinstance(task_values, list) else [])
        if not isinstance(task_values, list):
            status = "malformed_tasks_field"
        elif len(normalized) == 0:
            status = "empty_task"
        elif len(normalized) > 1:
            status = "multiple_tasks"
        else:
            status = "clean"
        normalized_task = next(iter(normalized), "") if status == "clean" else ""
        records.append(
            {
                "lerobot_episode_index": episode_index,
                "episode_length": int(episode.get("length", -1)),
                "task_instruction": normalized.get(normalized_task, ""),
                "normalized_task": normalized_task,
                "episode_task_status": status,
            }
        )
    return records


def build_conservative_join(
    steering_records: list[dict[str, Any]],
    episode_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    steering_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    episode_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in steering_records:
        if row["task_extraction_status"] == "clean":
            steering_by_task[row["normalized_task"]].append(row)
    for row in episode_records:
        if row["episode_task_status"] == "clean":
            episode_by_task[row["normalized_task"]].append(row)

    joins: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    matched_steering: set[int] = set()
    matched_episodes: set[int] = set()

    for task in sorted(set(steering_by_task) & set(episode_by_task)):
        steering_matches = steering_by_task[task]
        episode_matches = episode_by_task[task]
        if len(steering_matches) == 1 and len(episode_matches) == 1:
            steering = steering_matches[0]
            episode = episode_matches[0]
            joins.append(
                {
                    **episode,
                    **steering,
                    "canonical_task": episode["task_instruction"],
                    "join_method": "normalized_task_one_to_one",
                    "raw_indices_equal": (
                        episode["lerobot_episode_index"] == steering["steering_trajectory_id"]
                    ),
                }
            )
            matched_steering.add(steering["steering_trajectory_id"])
            matched_episodes.add(episode["lerobot_episode_index"])

    for row in steering_records:
        trajectory_id = row["steering_trajectory_id"]
        if trajectory_id in matched_steering:
            continue
        if row["task_extraction_status"] != "clean":
            reason = row["task_extraction_status"]
        else:
            task = row["normalized_task"]
            steering_count = len(steering_by_task[task])
            episode_count = len(episode_by_task.get(task, []))
            if episode_count == 0:
                reason = "no_lerobot_task_match"
            elif steering_count > 1:
                reason = "steering_task_not_unique"
            elif episode_count > 1:
                reason = "lerobot_task_not_unique"
            else:
                reason = "unclassified_join_rejection"
        issues.append(
            {
                "scope": "steering_trajectory",
                "record_id": trajectory_id,
                "exclusion_reason": reason,
                "details": row.get("canonical_task", ""),
            }
        )

    for row in episode_records:
        episode_id = row["lerobot_episode_index"]
        if episode_id in matched_episodes:
            continue
        if row["episode_task_status"] != "clean":
            reason = row["episode_task_status"]
        else:
            task = row["normalized_task"]
            steering_count = len(steering_by_task.get(task, []))
            episode_count = len(episode_by_task[task])
            if steering_count == 0:
                reason = "no_steering_task_match"
            elif steering_count > 1:
                reason = "steering_task_not_unique"
            elif episode_count > 1:
                reason = "lerobot_task_not_unique"
            else:
                reason = "unclassified_join_rejection"
        issues.append(
            {
                "scope": "lerobot_episode",
                "record_id": episode_id,
                "exclusion_reason": reason,
                "details": row.get("task_instruction", ""),
            }
        )

    counts = {
        "steering_clean_task_records": sum(
            row["task_extraction_status"] == "clean" for row in steering_records
        ),
        "lerobot_clean_task_records": sum(
            row["episode_task_status"] == "clean" for row in episode_records
        ),
        "normalized_task_overlap": len(set(steering_by_task) & set(episode_by_task)),
        "one_to_one_joined_episodes": len(joins),
        "joined_raw_indices_equal": sum(row["raw_indices_equal"] for row in joins),
    }
    return joins, issues, counts


def _resolve_subtask_pool(
    subtask: object, pools: object
) -> tuple[str, list[object] | None, str]:
    if not isinstance(subtask, str) or not subtask.strip():
        return "", None, "malformed_subtask_text"
    if not isinstance(pools, dict):
        return subtask, None, "malformed_command_map"
    if subtask in pools:
        pool = pools[subtask]
        return subtask, pool if isinstance(pool, list) else None, (
            "exact" if isinstance(pool, list) else "command_pool_not_list"
        )
    target = normalize_semantic_text(subtask)
    matches = [
        (key, value)
        for key, value in pools.items()
        if isinstance(key, str) and normalize_semantic_text(key) == target
    ]
    if len(matches) == 1:
        key, pool = matches[0]
        return key, pool if isinstance(pool, list) else None, (
            "normalized" if isinstance(pool, list) else "command_pool_not_list"
        )
    if not matches:
        return subtask, None, "missing_command_pool"
    return subtask, None, "ambiguous_normalized_command_pool"


def validate_joined_trajectories(
    joins: Iterable[Mapping[str, Any]],
    step_to_subtask: Mapping[str, Any],
    commands: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validations: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for join in joins:
        episode_id = int(join["lerobot_episode_index"])
        trajectory_id = int(join["steering_trajectory_id"])
        episode_length = int(join["episode_length"])
        raw_steps = step_to_subtask.get(str(trajectory_id))
        pools = commands.get(str(trajectory_id))
        reasons: list[str] = []
        numeric_steps: dict[int, object] = {}
        if not isinstance(raw_steps, dict):
            reasons.append("missing_or_malformed_step_map")
        else:
            for key, value in raw_steps.items():
                try:
                    numeric_key = int(key)
                except (TypeError, ValueError):
                    reasons.append("non_integer_timestep_key")
                    continue
                if numeric_key in numeric_steps:
                    reasons.append("duplicate_numeric_timestep")
                numeric_steps[numeric_key] = value

        sorted_steps = sorted(numeric_steps)
        annotation_length = len(sorted_steps)
        contiguous = sorted_steps == list(range(annotation_length))
        if not contiguous:
            reasons.append("annotation_timestep_gap")
        if annotation_length - episode_length != 2:
            reasons.append("unexpected_annotation_episode_length_delta")
        if not all(index in numeric_steps for index in range(max(episode_length, 0))):
            reasons.append("direct_frame_mapping_not_covered")

        labels: list[str] = []
        pool_resolution_modes: Counter[str] = Counter()
        if not reasons or set(reasons) <= {"unexpected_annotation_episode_length_delta"}:
            for frame_index in range(episode_length):
                subtask = numeric_steps.get(frame_index)
                resolved_key, pool, resolution = _resolve_subtask_pool(subtask, pools)
                pool_resolution_modes[resolution] += 1
                if pool is None:
                    reasons.append(resolution)
                    continue
                if not pool:
                    reasons.append("empty_command_pool")
                if any(not isinstance(command, str) or not command.strip() for command in pool):
                    reasons.append("non_string_or_empty_command")
                labels.append(normalize_semantic_text(resolved_key))

        segment_lengths: list[int] = []
        segment_labels: list[str] = []
        for label in labels:
            if not segment_labels or label != segment_labels[-1]:
                segment_labels.append(label)
                segment_lengths.append(1)
            else:
                segment_lengths[-1] += 1
        repeated_noncontiguous = len(segment_labels) != len(set(segment_labels))
        direct_vs_plus_one_label_changes = 0
        if all(index in numeric_steps for index in range(max(episode_length + 1, 0))):
            direct_vs_plus_one_label_changes = sum(
                normalize_semantic_text(str(numeric_steps[index]))
                != normalize_semantic_text(str(numeric_steps[index + 1]))
                for index in range(episode_length)
            )
        unused_tail_labels = [
            str(numeric_steps[index]).strip()
            for index in range(episode_length, annotation_length)
            if index in numeric_steps
        ]
        unused_tail_starts_new_segment = bool(
            episode_length > 0
            and episode_length in numeric_steps
            and (episode_length - 1) in numeric_steps
            and normalize_semantic_text(str(numeric_steps[episode_length]))
            != normalize_semantic_text(str(numeric_steps[episode_length - 1]))
        )
        unique_reasons = sorted(set(reasons))
        eligible = not unique_reasons

        validation = {
            **dict(join),
            "annotation_timestep_count": annotation_length,
            "annotation_minus_episode_length": annotation_length - episode_length,
            "annotation_timesteps_contiguous": contiguous,
            "frame_mapping": "lerobot_frame_i_to_annotation_step_i",
            "direct_vs_plus_one_label_change_count": direct_vs_plus_one_label_changes,
            "unused_tail_annotation_labels": json.dumps(unused_tail_labels, ensure_ascii=False),
            "unused_tail_labels_identical": len(
                set(map(normalize_semantic_text, unused_tail_labels))
            )
            <= 1,
            "unused_tail_starts_new_segment": unused_tail_starts_new_segment,
            "pool_resolution_exact_frames": pool_resolution_modes["exact"],
            "pool_resolution_normalized_frames": pool_resolution_modes["normalized"],
            "semantic_segment_count": len(segment_lengths),
            "rho_sem": (len(segment_lengths) / episode_length if episode_length > 0 else None),
            "segment_lengths": json.dumps(segment_lengths),
            "minimum_segment_length": min(segment_lengths) if segment_lengths else 0,
            "one_step_segment_count": sum(length == 1 for length in segment_lengths),
            "repeated_noncontiguous_subtask": repeated_noncontiguous,
            "validation_status": "eligible" if eligible else "rejected",
            "exclusion_reason": "|".join(unique_reasons),
        }
        validations.append(validation)
        if unique_reasons:
            for reason in unique_reasons:
                issues.append(
                    {
                        "scope": "joined_trajectory",
                        "record_id": f"{episode_id}:{trajectory_id}",
                        "exclusion_reason": reason,
                        "details": join.get("normalized_task", ""),
                    }
                )
    return validations, issues


def lerobot_references(episode_index: int, frame_index: int) -> tuple[str, str]:
    chunk = episode_index // 1000
    base = f"hf://datasets/{LEROBOT_REPO}@{LEROBOT_REVISION}"
    observation = (
        f"{base}/videos/chunk-{chunk:03d}/observation.images.image_0/"
        f"episode_{episode_index:06d}.mp4#frame={frame_index}"
    )
    action = (
        f"{base}/data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        f"#row={frame_index}"
    )
    return observation, action


def iter_frame_records(
    eligible_validations: Iterable[Mapping[str, Any]],
    step_to_subtask: Mapping[str, Any],
    commands: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    for trajectory in eligible_validations:
        if trajectory["validation_status"] != "eligible":
            continue
        episode_id = int(trajectory["lerobot_episode_index"])
        trajectory_id = int(trajectory["steering_trajectory_id"])
        raw_steps = step_to_subtask[str(trajectory_id)]
        pools = commands[str(trajectory_id)]
        prior_subtask = ""
        segment_index = -1
        for frame_index in range(int(trajectory["episode_length"])):
            subtask = raw_steps[str(frame_index)]
            resolved_key, pool, resolution = _resolve_subtask_pool(subtask, pools)
            if pool is None:
                raise AssertionError(
                    f"Validated pool disappeared for trajectory {trajectory_id}, frame {frame_index}"
                )
            normalized_subtask = normalize_semantic_text(resolved_key)
            if normalized_subtask != prior_subtask:
                segment_index += 1
                prior_subtask = normalized_subtask
            observation_ref, action_ref = lerobot_references(episode_id, frame_index)
            yield {
                "lerobot_episode_index": episode_id,
                "steering_trajectory_id": trajectory_id,
                "frame_index": frame_index,
                "annotation_timestep": frame_index,
                "raw_bridge_observation_index": frame_index + 1,
                "task_instruction": trajectory["canonical_task"],
                "task_id": stable_id("task", trajectory["normalized_task"]),
                "subtask_id": stable_id("subtask", trajectory_id, normalized_subtask),
                "subtask_text": resolved_key.strip(),
                "semantic_segment_index": segment_index,
                "command_pool": json.dumps(pool, ensure_ascii=False),
                "command_pool_resolution": resolution,
                "source_collection": trajectory["source_collection"],
                "task_family": trajectory["task_family"],
                "observation_ref": observation_ref,
                "action_ref": action_ref,
            }
