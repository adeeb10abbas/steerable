from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .audits import (
    COMMAND_REVIEW_FIELDS,
    PARAPHRASE_REVIEW_FIELDS,
    SEQUENCE_REVIEW_FIELDS,
    build_audit_lock,
    finalize_audits,
)
from .constants import (
    ANNOTATION_REVISION,
    DEFAULT_SEED,
    LEROBOT_REPO,
    LEROBOT_REVISION,
    PILOT_SPLIT_COUNTS,
    TARGET_SPLIT_COUNTS,
)
from .io import (
    download_inputs,
    load_json,
    read_jsonl,
    write_audit_template,
    write_csv,
    write_json,
)
from .join import (
    build_conservative_join,
    derive_lerobot_episode_records,
    derive_steering_task_records,
    iter_frame_records,
    join_normalization_sensitivity,
    validate_joined_trajectories,
)
from .manifests import build_master_manifest, materialize_views, validate_views
from .provenance import implementation_manifest
from .splits import balance_rows, nested_split, split_payload, stratified_split
from .taxonomy import (
    generated_wrapper_paraphrases,
    iter_taxonomy_rows,
    make_manual_pool_audit,
)
from .text import normalize_semantic_text, normalize_text, normalize_whitespace, stable_id


TAXONOMY_FIELDS = [
    "steering_trajectory_id",
    "source_collection",
    "pool_id",
    "pool_index",
    "canonical_task",
    "canonical_subtask",
    "slot_index",
    "command",
    "normalized_command",
    "automatic_class",
    "rule_reason",
    "coordinate_count",
    "has_unresolved_placeholder",
    "has_numeric_angle_coordinate",
    "has_unparsed_square_bracket",
    "same_intent_candidate",
    "needs_semantic_review",
]


def _write_parquet_batches(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    batch_size: int = 50_000,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    batch: list[dict[str, Any]] = []
    count = 0

    def flush(values: list[dict[str, Any]]) -> None:
        nonlocal writer
        if not values:
            return
        table = pa.Table.from_pylist(values)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
        elif table.schema != writer.schema:
            table = table.cast(writer.schema)
        writer.write_table(table)

    for row in rows:
        batch.append(dict(row))
        count += 1
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
    flush(batch)
    if writer is None:
        raise ValueError(f"Refusing to write empty Parquet file: {path}")
    writer.close()
    return count


def _hash_text(value: str) -> bytes:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()


def _write_taxonomy(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    slot_counts: Counter[str] = Counter()
    raw_unique: set[bytes] = set()
    display_unique: set[bytes] = set()
    token_class_masks: dict[bytes, int] = {}
    class_bits: dict[str, int] = {}
    nonstring_slots = 0
    total_slots = 0
    display_duplicate_extras = 0
    token_duplicate_extras = 0
    format_issues: list[dict[str, Any]] = []
    pool_records: list[dict[str, Any]] = []
    current_pool = ""
    current_meta: dict[str, Any] = {}
    current_display: set[str] = set()
    current_token: set[str] = set()
    current_candidates: dict[str, str] = {}
    current_classes: Counter[str] = Counter()

    def finish_pool() -> None:
        if not current_pool:
            return
        canonical_normalized = normalize_semantic_text(
            str(current_meta["canonical_subtask"])
        )
        canonical_present = canonical_normalized in current_candidates
        paraphrase_candidates = [
            current_candidates[key]
            for key in sorted(current_candidates)
            if key != canonical_normalized
        ]
        paraphrase_count = len(paraphrase_candidates)
        pool_records.append(
            {
                "intent_id": current_pool,
                "level": "subtask",
                "steering_trajectory_id": current_meta["steering_trajectory_id"],
                "source_collection": current_meta["source_collection"],
                "canonical_task": current_meta["canonical_task"],
                "canonical_text": current_meta["canonical_subtask"],
                "origin": "released",
                "automatic_candidate_surfaces": json.dumps(
                    paraphrase_candidates, ensure_ascii=False
                ),
                "train_strings": "[]",
                "heldout_strings": "[]",
                "canonical_present": canonical_present,
                "paraphrase_count_excluding_canonical": paraphrase_count,
                "automatic_candidate_count": paraphrase_count,
                "at_least_1": paraphrase_count >= 1,
                "at_least_2": paraphrase_count >= 2,
                "at_least_4": paraphrase_count >= 4,
                "at_least_6": paraphrase_count >= 6,
                "verification_status": "requires_semantic_review",
                "final_eligible": False,
                "exclusion_reason": "semantic_review_pending",
                "automatic_class_counts": json.dumps(dict(current_classes), sort_keys=True),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TAXONOMY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for raw_row in rows:
            row = dict(raw_row)
            pool_id = str(row["pool_id"])
            if pool_id != current_pool:
                finish_pool()
                current_pool = pool_id
                current_meta = row
                current_display = set()
                current_token = set()
                current_candidates = {}
                current_classes = Counter()

            writer.writerow(row)
            total_slots += 1
            class_name = str(row["automatic_class"])
            slot_counts[class_name] += 1
            current_classes[class_name] += 1
            command = str(row.get("command", ""))
            normalized = str(row.get("normalized_command", ""))
            if row.get("rule_reason") == "non_string_command":
                nonstring_slots += 1
            else:
                raw_unique.add(_hash_text(command))
                display = normalize_whitespace(command)
                raw_display_before = len(current_display)
                current_display.add(display)
                display_duplicate_extras += int(len(current_display) == raw_display_before)
                display_unique.add(_hash_text(display))
            if normalized:
                token_before = len(current_token)
                current_token.add(normalized)
                token_duplicate_extras += int(len(current_token) == token_before)
                key = _hash_text(normalized)
                if class_name not in class_bits:
                    class_bits[class_name] = 1 << len(class_bits)
                token_class_masks[key] = token_class_masks.get(key, 0) | class_bits[class_name]
            if row.get("same_intent_candidate") and normalized:
                current_candidates.setdefault(normalized, command)
            if class_name == "malformed_or_unclear" and row.get("rule_reason") != (
                "semantic_class_unclear_requires_manual_review"
            ):
                format_issues.append(
                    {
                        "scope": "command_slot",
                        "record_id": (
                            f"{row['steering_trajectory_id']}:{row['pool_id']}:{row['slot_index']}"
                        ),
                        "exclusion_reason": row["rule_reason"],
                        "details": command,
                    }
                )
        finish_pool()

    inverse_bits = {bit: name for name, bit in class_bits.items()}
    unique_counts: Counter[str] = Counter()
    for mask in token_class_masks.values():
        if mask and mask & (mask - 1) == 0:
            unique_counts[inverse_bits[mask]] += 1
        else:
            unique_counts["mixed_across_contexts"] += 1
    summary = {
        "command_slots": total_slots,
        "non_string_command_slots": nonstring_slots,
        "raw_exact_unique_string_count": len(raw_unique),
        "display_normalized_unique_string_count": len(display_unique),
        "token_normalized_unique_string_count": len(token_class_masks),
        "within_pool_display_normalized_extra_duplicate_slots": display_duplicate_extras,
        "within_pool_token_normalized_extra_duplicate_slots": token_duplicate_extras,
        "within_pool_display_normalized_duplicate_rate": (
            display_duplicate_extras / max(1, total_slots - nonstring_slots)
        ),
        "within_pool_token_normalized_duplicate_rate": (
            token_duplicate_extras / max(1, total_slots - nonstring_slots)
        ),
        "slot_counts": dict(sorted(slot_counts.items())),
        "slot_shares": {
            name: count / total_slots for name, count in sorted(slot_counts.items())
        },
        "token_normalized_unique_string_counts": dict(sorted(unique_counts.items())),
        "token_normalized_unique_string_shares": {
            name: count / max(1, len(token_class_masks))
            for name, count in sorted(unique_counts.items())
        },
        "method_note": (
            "Automatic class 2 entries are candidates only. Context-sensitive strings with multiple "
            "classes are reported as mixed_across_contexts."
        ),
    }
    return summary, pool_records, format_issues


def _trajectory_density_rows(validations: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "lerobot_episode_index",
        "steering_trajectory_id",
        "normalized_task",
        "source_collection",
        "task_family",
        "episode_length",
        "annotation_timestep_count",
        "annotation_minus_episode_length",
        "annotation_timesteps_contiguous",
        "frame_mapping",
        "direct_vs_plus_one_label_change_count",
        "unused_tail_annotation_labels",
        "unused_tail_labels_identical",
        "unused_tail_starts_new_segment",
        "semantic_segment_count",
        "rho_sem",
        "segment_lengths",
        "minimum_segment_length",
        "one_step_segment_count",
        "repeated_noncontiguous_subtask",
        "validation_status",
        "exclusion_reason",
    ]
    return [{field: row.get(field) for field in fields} for row in validations]


def _plot_density(rows: list[Mapping[str, Any]], plot_dir: Path) -> dict[str, str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    eligible = [row for row in rows if row["validation_status"] == "eligible"]
    k_values = [int(row["semantic_segment_count"]) for row in eligible]
    rho_values = [float(row["rho_sem"]) for row in eligible]
    segment_lengths = [
        length for row in eligible for length in json.loads(str(row["segment_lengths"]))
    ]
    outputs: dict[str, str] = {}
    for name, values, xlabel in (
        ("semantic_segments_K", k_values, "Contiguous semantic segments K"),
        ("semantic_density_K_over_T", rho_values, "Semantic update density K / T"),
        ("segment_length", segment_lengths, "Segment length (frames)"),
    ):
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        axis.hist(values, bins=40, color="#315b7d", edgecolor="white")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Count")
        axis.grid(axis="y", alpha=0.2)
        figure.tight_layout()
        output = plot_dir / f"{name}.png"
        figure.savefig(output, dpi=180)
        plt.close(figure)
        outputs[name] = str(output)
    return outputs


def _density_summary(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["validation_status"] == "eligible"]
    groups: list[tuple[str, str, list[Mapping[str, Any]]]] = [
        ("overall", "all", eligible)
    ]
    for source in sorted({str(row["source_collection"]) for row in eligible}):
        groups.append(
            (
                "source_collection",
                source,
                [row for row in eligible if str(row["source_collection"]) == source],
            )
        )
    output: list[dict[str, Any]] = []
    for scope, group, values in groups:
        lengths = np.asarray([int(row["episode_length"]) for row in values])
        segments = np.asarray([int(row["semantic_segment_count"]) for row in values])
        densities = np.asarray([float(row["rho_sem"]) for row in values])
        output.append(
            {
                "scope": scope,
                "group": group,
                "trajectories": len(values),
                "timesteps_T": int(lengths.sum()) if len(lengths) else 0,
                "semantic_segments_K": int(segments.sum()) if len(segments) else 0,
                "median_episode_length": float(np.median(lengths)) if len(lengths) else 0.0,
                "median_segments_K": float(np.median(segments)) if len(segments) else 0.0,
                "median_rho_sem": float(np.median(densities)) if len(densities) else 0.0,
                "rho_sem_p10": float(np.quantile(densities, 0.10)) if len(densities) else 0.0,
                "rho_sem_p90": float(np.quantile(densities, 0.90)) if len(densities) else 0.0,
            }
        )
    return output


def _plot_anatomy(taxonomy_summary: Mapping[str, Any], plot_dir: Path) -> str:
    counts = taxonomy_summary["slot_counts"]
    labels = list(counts)
    values = [counts[label] for label in labels]
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    bars = axis.barh(labels, values, color="#4c78a8")
    axis.invert_yaxis()
    axis.set_xlabel("Command slots")
    axis.set_title("Released Bridge command pools: provisional automatic taxonomy (unaudited)")
    total = sum(values)
    for bar, value in zip(bars, values):
        axis.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:,} ({value / total:.1%})",
            va="center",
            fontsize=8,
        )
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    output = plot_dir / "annotation_anatomy.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return str(output)


def _segment_sequence(
    trajectory_id: int,
    episode_length: int,
    step_to_subtask: Mapping[str, Any],
) -> list[dict[str, Any]]:
    steps = step_to_subtask[str(trajectory_id)]
    segments: list[dict[str, Any]] = []
    for frame_index in range(episode_length):
        subtask = str(steps[str(frame_index)]).strip()
        if not segments or normalize_semantic_text(subtask) != normalize_semantic_text(
            segments[-1]["subtask"]
        ):
            segments.append(
                {"segment_index": len(segments), "start_frame": frame_index, "end_frame": frame_index, "subtask": subtask}
            )
        else:
            segments[-1]["end_frame"] = frame_index
    return segments


def _manual_sequence_audit(
    eligible: list[Mapping[str, Any]],
    step_to_subtask: Mapping[str, Any],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_source[str(row["source_collection"])].append(row)
    sources = sorted(by_source)
    base = sample_size // len(sources)
    remainder = sample_size % len(sources)
    allocation = {
        source: min(len(by_source[source]), base + int(index < remainder))
        for index, source in enumerate(sources)
    }
    while sum(allocation.values()) < min(sample_size, len(eligible)):
        candidates = [
            source
            for source in sources
            if allocation[source] < len(by_source[source])
        ]
        if not candidates:
            break
        source = min(
            candidates,
            key=lambda value: (
                allocation[value],
                stable_id("sequence_allocation", seed, value, allocation[value]),
            ),
        )
        allocation[source] += 1
    selected: list[Mapping[str, Any]] = []
    for source in sources:
        ranked = sorted(
            by_source[source],
            key=lambda row: stable_id("sequence", seed, row["steering_trajectory_id"]),
        )
        selected.extend(ranked[: allocation[source]])
    output: list[dict[str, Any]] = []
    for row in selected:
        episode = int(row["lerobot_episode_index"])
        chunk = episode // 1000
        video_url = (
            f"https://huggingface.co/datasets/{LEROBOT_REPO}/resolve/{LEROBOT_REVISION}/"
            f"videos/chunk-{chunk:03d}/observation.images.image_0/episode_{episode:06d}.mp4"
        )
        output.append(
            {
                "lerobot_episode_index": episode,
                "steering_trajectory_id": int(row["steering_trajectory_id"]),
                "source_collection": row["source_collection"],
                "task_instruction": row["canonical_task"],
                "episode_length": int(row["episode_length"]),
                "segments": json.dumps(
                    _segment_sequence(
                        int(row["steering_trajectory_id"]),
                        int(row["episode_length"]),
                        step_to_subtask,
                    ),
                    ensure_ascii=False,
                ),
                "video_url": video_url,
                "audit_seed": seed,
                "secondary_review_required": False,
                "primary_reviewer": "",
                "primary_task_identity_believable": "",
                "primary_direct_step_i_alignment_believable": "",
                "primary_boundary_alignment_believable": "",
                "primary_review_notes": "",
                "secondary_reviewer": "",
                "secondary_task_identity_believable": "",
                "secondary_direct_step_i_alignment_believable": "",
                "secondary_boundary_alignment_believable": "",
                "secondary_review_notes": "",
                "adjudicated_task_identity_believable": "",
                "adjudicated_direct_step_i_alignment_believable": "",
                "adjudicated_boundary_alignment_believable": "",
                "adjudicator": "",
                "adjudication_notes": "",
            }
        )
    secondary_ids = set(
        sorted(
            (int(row["steering_trajectory_id"]) for row in output),
            key=lambda trajectory_id: stable_id(
                "sequence_secondary_review", seed, trajectory_id
            ),
        )[: max(1, round(0.20 * len(output)))]
    )
    for row in output:
        row["secondary_review_required"] = (
            int(row["steering_trajectory_id"]) in secondary_ids
        )
    return output


def _eligibility_rows(
    pool_records: list[dict[str, Any]],
    validations: list[Mapping[str, Any]],
    step_to_subtask: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = list(pool_records)
    seen_tasks: set[str] = set()
    seen_subtasks: set[tuple[int, str]] = set()

    def add_generated(
        *,
        level: str,
        intent_id: str,
        trajectory_id: int,
        source: str,
        canonical_task: str,
        canonical_text: str,
    ) -> None:
        train, heldout = generated_wrapper_paraphrases(canonical_text)
        rows.append(
            {
                "intent_id": intent_id,
                "level": level,
                "steering_trajectory_id": trajectory_id,
                "source_collection": source,
                "canonical_task": canonical_task,
                "canonical_text": canonical_text,
                "origin": "generated_verbatim_wrapper",
                "automatic_candidate_surfaces": json.dumps(
                    [*train, *heldout], ensure_ascii=False
                ),
                "train_strings": json.dumps(train, ensure_ascii=False),
                "heldout_strings": json.dumps(heldout, ensure_ascii=False),
                "canonical_present": True,
                "paraphrase_count_excluding_canonical": 6,
                "automatic_candidate_count": 6,
                "at_least_1": True,
                "at_least_2": True,
                "at_least_4": True,
                "at_least_6": True,
                "verification_status": (
                    "verbatim_embedding_preserves_literal_constraint_"
                    "pending_human_naturalness_and_treatment_strength"
                ),
                "final_eligible": False,
                "exclusion_reason": "human_acceptability_review_pending",
                "automatic_class_counts": "{}",
            }
        )

    for validation in validations:
        if validation["validation_status"] != "eligible":
            continue
        task = str(validation["canonical_task"])
        normalized = normalize_semantic_text(task)
        trajectory_id = int(validation["steering_trajectory_id"])
        source = str(validation["source_collection"])
        if normalized not in seen_tasks:
            seen_tasks.add(normalized)
            add_generated(
                level="task",
                intent_id=stable_id("task", normalized),
                trajectory_id=trajectory_id,
                source=source,
                canonical_task=task,
                canonical_text=task,
            )

        steps = step_to_subtask[str(trajectory_id)]
        for frame_index in range(int(validation["episode_length"])):
            canonical_subtask = str(steps[str(frame_index)]).strip()
            normalized_subtask = normalize_semantic_text(canonical_subtask)
            key = (trajectory_id, normalized_subtask)
            if not normalized_subtask or key in seen_subtasks:
                continue
            seen_subtasks.add(key)
            add_generated(
                level="subtask",
                intent_id=stable_id("subtask", trajectory_id, normalized_subtask),
                trajectory_id=trajectory_id,
                source=source,
                canonical_task=task,
                canonical_text=canonical_subtask,
            )
    return rows


def _eligibility_summary(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["level"]), str(row["origin"]))].append(row)
    output: list[dict[str, Any]] = []
    for (level, origin), values in sorted(grouped.items()):
        output.append(
            {
                "level": level,
                "origin": origin,
                "intent_groups": len(values),
                "groups_at_least_1": sum(bool(row["at_least_1"]) for row in values),
                "groups_at_least_2": sum(bool(row["at_least_2"]) for row in values),
                "groups_at_least_4": sum(bool(row["at_least_4"]) for row in values),
                "groups_at_least_6": sum(bool(row["at_least_6"]) for row in values),
                "final_verified_groups": sum(bool(row["final_eligible"]) for row in values),
            }
        )
    return output


def _lexical_distance(strings: list[str]) -> float:
    distances: list[float] = []
    token_sets = [set(normalize_semantic_text(value).split()) for value in strings]
    for index, left in enumerate(token_sets):
        for right in token_sets[index + 1 :]:
            union = left | right
            distances.append(1 - len(left & right) / len(union) if union else 0.0)
    return float(np.mean(distances)) if distances else 0.0


def _surface_report(master: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for level, intent_column, canonical_column, train_column, heldout_column in (
        (
            "task",
            "task_id",
            "canonical_task",
            "task_train_paraphrases",
            "task_heldout_paraphrases",
        ),
        (
            "subtask",
            "subtask_id",
            "canonical_subtask",
            "subtask_train_paraphrases",
            "subtask_heldout_paraphrases",
        ),
    ):
        subset = master.drop_duplicates(intent_column)
        canonical_lengths: list[int] = []
        surface_lengths: list[int] = []
        paraphrase_length_deltas: list[int] = []
        distances: list[float] = []
        duplicate_extras = 0
        total_surfaces = 0
        for row in subset.itertuples():
            canonical = getattr(row, canonical_column)
            strings = [
                canonical,
                *json.loads(getattr(row, train_column)),
                *json.loads(getattr(row, heldout_column)),
            ]
            token_lengths = [
                len(normalize_semantic_text(value).split()) for value in strings
            ]
            canonical_lengths.append(token_lengths[0])
            surface_lengths.extend(token_lengths)
            paraphrase_length_deltas.extend(
                value - token_lengths[0] for value in token_lengths[1:]
            )
            distances.append(_lexical_distance(strings))
            normalized_surfaces = [normalize_semantic_text(value) for value in strings]
            duplicate_extras += len(normalized_surfaces) - len(set(normalized_surfaces))
            total_surfaces += len(normalized_surfaces)

        def percentile(values: list[int], quantile: float) -> float:
            return float(np.quantile(values, quantile)) if values else 0.0

        output.append(
            {
                "level": level,
                "origin": "generated_verbatim_wrapper_provisional",
                "intent_groups": len(subset),
                "surfaces_per_group": 7,
                "train_surfaces_per_group": 4,
                "heldout_surfaces_per_group": 2,
                "canonical_token_length_mean": float(np.mean(canonical_lengths)),
                "canonical_token_length_p10": percentile(canonical_lengths, 0.10),
                "canonical_token_length_median": percentile(canonical_lengths, 0.50),
                "canonical_token_length_p90": percentile(canonical_lengths, 0.90),
                "surface_token_length_mean": float(np.mean(surface_lengths)),
                "surface_token_length_p10": percentile(surface_lengths, 0.10),
                "surface_token_length_median": percentile(surface_lengths, 0.50),
                "surface_token_length_p90": percentile(surface_lengths, 0.90),
                "mean_paraphrase_token_delta": float(
                    np.mean(paraphrase_length_deltas)
                ),
                "mean_pairwise_jaccard_distance": float(np.mean(distances)),
                "within_group_normalized_duplicate_rate": (
                    duplicate_extras / total_surfaces if total_surfaces else 0.0
                ),
                "semantic_equivalence_pass_rate": "pending_human_review",
                "embedding_cosine_distance": "not_computed_pending_verified_surfaces",
            }
        )
    return output


def _surface_match_validation(
    report: list[Mapping[str, Any]],
) -> dict[str, Any]:
    by_level = {str(row["level"]): row for row in report}
    task = by_level["task"]
    subtask = by_level["subtask"]
    token_mean_difference = abs(
        float(task["surface_token_length_mean"])
        - float(subtask["surface_token_length_mean"])
    )
    jaccard_difference = abs(
        float(task["mean_pairwise_jaccard_distance"])
        - float(subtask["mean_pairwise_jaccard_distance"])
    )
    added_token_difference = abs(
        float(task["mean_paraphrase_token_delta"])
        - float(subtask["mean_paraphrase_token_delta"])
    )
    checks = {
        "same_surface_count": (
            task["train_surfaces_per_group"] == subtask["train_surfaces_per_group"] == 4
            and task["heldout_surfaces_per_group"]
            == subtask["heldout_surfaces_per_group"]
            == 2
        ),
        "surface_token_length_means_within_2_tokens": token_mean_difference <= 2.0,
        "mean_pairwise_jaccard_within_0_05": jaccard_difference <= 0.05,
        "mean_added_token_count_within_0_25": added_token_difference <= 0.25,
        "embedding_distance_available_and_matched": False,
        "semantic_equivalence_audit_complete": False,
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "surface_token_length_mean_absolute_difference": token_mean_difference,
        "pairwise_jaccard_mean_absolute_difference": jaccard_difference,
        "added_token_mean_absolute_difference": added_token_difference,
        "status": "provisional_unmatched_pending_human_and_embedding_audit",
        "interpretation": (
            "The same wrapper count and added-token treatment do not make B and D "
            "matched: task and subtask base-language distributions differ."
        ),
    }


def _manual_group_audit(master: pd.DataFrame, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level, intent_column, canonical_column, train_column, heldout_column in (
        (
            "task",
            "task_id",
            "canonical_task",
            "task_train_paraphrases",
            "task_heldout_paraphrases",
        ),
        (
            "subtask",
            "subtask_id",
            "canonical_subtask",
            "subtask_train_paraphrases",
            "subtask_heldout_paraphrases",
        ),
    ):
        subset = master.drop_duplicates(intent_column).copy()
        subset["audit_rank"] = subset.apply(
            lambda row: stable_id("group_audit", seed, level, row[intent_column]), axis=1
        )
        subset.sort_values("audit_rank", inplace=True)
        for row in subset.head(50).itertuples():
            canonical = getattr(row, canonical_column)
            rows.append(
                {
                    "level": level,
                    "intent_id": row.task_id if level == "task" else row.subtask_id,
                    "canonical_text": canonical,
                    "train_surfaces": getattr(row, train_column),
                    "heldout_surfaces": getattr(row, heldout_column),
                    "origin": "generated_verbatim_wrapper",
                    "audit_seed": seed,
                    "secondary_review_required": False,
                    "primary_all_surfaces_semantically_equivalent": "",
                    "primary_language_natural_and_acceptable": "",
                    "primary_reviewer": "",
                    "primary_review_notes": "",
                    "secondary_all_surfaces_semantically_equivalent": "",
                    "secondary_language_natural_and_acceptable": "",
                    "secondary_reviewer": "",
                    "secondary_review_notes": "",
                    "adjudicated_semantically_equivalent": "",
                    "adjudicated_language_acceptable": "",
                    "adjudicator": "",
                    "adjudication_notes": "",
                }
            )
    secondary_ids = set(
        sorted(
            (str(row["intent_id"]) for row in rows),
            key=lambda intent_id: stable_id(
                "group_secondary_review", seed, intent_id
            ),
        )[: max(1, round(0.20 * len(rows)))]
    )
    for row in rows:
        row["secondary_review_required"] = str(row["intent_id"]) in secondary_ids
    return rows


def _write_decision_memo(
    path: Path,
    inventory: Mapping[str, Any],
    eligibility_summary: list[Mapping[str, Any]],
    manifest_validation: Mapping[str, Any],
    pilot_counts: Mapping[str, int],
    target_counts: Mapping[str, int],
) -> None:
    joins = inventory["join"]
    quality = inventory["quality"]
    taxonomy = inventory["taxonomy"]
    eligibility_table = "\n".join(
        "| {level} | {origin} | {intent_groups:,} | {groups_at_least_6:,} | "
        "{final_verified_groups:,} |".format(**row)
        for row in eligibility_summary
    )
    structural_pass = bool(manifest_validation["all_structural_assertions_pass"])
    surface_match = manifest_validation["surface_match_validation"]
    memo = f"""# RES-1 decision memo: CONDITIONAL GO TO CURATION / TRAINING NO-GO

Date: 2026-07-23<br>
Pinned annotations: `{ANNOTATION_REVISION}`<br>
Pinned LeRobot data: `{LEROBOT_REVISION}`

## Decision

**Continue curation, but do not train the current B/D manifests.** A common
robot-row set can generate all four structural views, but the bottleneck cell
has **zero verified paraphrase groups**. The generated strings quote the
canonical instruction inside six prompt wrappers; they are useful for testing
the plumbing, not yet evidence of genuine lexical surface diversity.

## Four-gate result

1. **Sidecar join — conditional pass.** The release has 53,192 trajectory
   keys and 38,454 densely annotated trajectories. Recovering the trajectory
   task as the sole normalized string shared across all subtask pools and then
   requiring a one-to-one normalized-task match yields **{joins['one_to_one_joined_episodes']:,}**
   episode pairs. Only **{joins['joined_raw_indices_equal']:,}** pairs retain the
   same raw index, so index equality would be wrong. Every accepted pair has
   contiguous annotation steps `0..L+1`, direct released-code coverage
   `frame i -> annotation step i`, and valid command pools. Physical boundary
   alignment still requires the 20-video review.

2. **True surface paraphrases — fail pending curation.** The sidecar contains
   **{taxonomy['command_slots']:,}** command slots, mixing task text, subtask
   wording, coordinates, paths, strict motion/gripper commands, hybrids, and
   unclear strings. Automatic rules can identify candidates but cannot certify
   semantic equivalence. Blank audit fields remain failures by construction.

3. **Scale — temporal pass, language fail.** Integrity and
   temporal rules retain **{quality['density_eligible_trajectories']:,}**
   trajectories, enough to freeze both `{dict(pilot_counts)}` and
   `{dict(target_counts)}` structurally. Verified language capacity is zero, so
   these are provisional curation cohorts rather than authorized training sets.
   New task-level paraphrases are mandatory; matched subtask paraphrases must
   also be generated or individually verified.

4. **Matched manifests — {'structural pass' if structural_pass else 'fail'}; scientific fail.**
   A/B/C/D have {manifest_validation['master_row_count']:,} rows and the same
   pinned trajectory, frame, split, observation reference, action reference,
   and reference hash. High-diversity training/held-out pools are disjoint and
   deterministically sampled. However, audit status is pending and B/D are not
   distribution-matched: mean total-length difference is
   **{surface_match['surface_token_length_mean_absolute_difference']:.2f} tokens**
   and mean Jaccard-distance difference is
   **{surface_match['pairwise_jaccard_mean_absolute_difference']:.3f}**.

## Eligibility bottleneck

| Level | Origin | Intent groups | At least 6 non-canonical candidates | Verified |
| --- | --- | ---: | ---: | ---: |
{eligibility_table}

Candidate counts exclude the canonical text and remain diagnostic, not training
eligibility.

## Frozen exclusions and split rules

- Reject one-pool or multi-intersection task recovery, missing/ambiguous task
  matches, malformed/non-contiguous steps, absent command pools, non-string or
  empty used commands, fewer than three semantic segments, any one-frame
  segment, and unknown source collection.
- Assign each original Bridge `out.npy` group to exactly one split, then sample
  the 704-trajectory target with seed `{DEFAULT_SEED + 2}`. Select the nested,
  role-preserving 192-trajectory pilot with seed `{DEFAULT_SEED + 1}`. Language
  surfaces, paraphrase review, and visual review use base seed `{DEFAULT_SEED}`;
  the broader sequence audit uses `{DEFAULT_SEED + 30}`, while command-pool
  membership is deterministic without a separate runtime seed. Stratification uses
  source, instruction length, trajectory length, and segment count; task family
  is retained as a diagnostic because exact task labels are globally unique.
- Coordinates, traces, unresolved placeholders, strict atomic commands,
  hybrids, or any candidate that changes an object, relation, direction,
  gripper state, temporal order, or other constraint cannot enter the primary
  surface-diversity treatment.
- Exact joined task strings are globally unique. Therefore val/test trajectory
  metrics also test new tasks. To isolate wording, evaluate each row's explicit
  `selected_heldout_instruction` against the same immutable robot references.
- Bridge v1 and FLAP each contribute only one density-eligible trajectory and
  are absent from the deterministic 704-trajectory target. Reported cohorts
  cover Bridge v2, RSS, and ICRA only; this is not silently generalized.

## What the diversity number means

The automatic annotation-anatomy figure is a provisional rule audit, not a
claim that residual strings are paraphrases.

![Automatic annotation anatomy](plots/annotation_anatomy.png)

## Tomorrow's first action

Open `visual_audit/index.html`, then complete `visual_alignment_audit.csv`,
`manual_command_audit.csv`, `manual_sequence_audit.csv`, and
`manual_paraphrase_group_audit.csv` with independent second review on the
pre-marked 20 percent. Run `steerable-res1 finalize-audits`. Because the current
B/D language distributions are unmatched and wrappers are low-strength, this
command only reports the fail-closed gate; it does not promote reviewed strings
or rewrite manifests. The expected next step is a genuine matched paraphrase
generation/adjudication and full-regeneration pass, not model training.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(memo, encoding="utf-8")


def run_pipeline(
    raw_root: Path,
    artifact_root: Path,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    def progress(message: str) -> None:
        print(f"[RES-1] {message}", flush=True)

    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = artifact_root / "manifests"
    plot_dir = artifact_root / "plots"

    progress("verifying revision-pinned downloads and content digests")
    input_manifest = download_inputs(raw_root)
    write_json(artifact_root / "input_manifest.json", input_manifest)
    annotation_root = raw_root / "steering_features_bridge"
    lerobot_root = raw_root / "bridge_orig_lerobot" / "meta"
    progress("loading the four annotation sidecars and LeRobot metadata")
    trajectory_map = load_json(annotation_root / "traj_idx_key_map.json")
    step_to_subtask = load_json(annotation_root / "step_to_subtask_dict.json")
    commands = load_json(annotation_root / "subtask_level_commands.json")
    rationales = load_json(annotation_root / "rationales.json")
    episodes = read_jsonl(lerobot_root / "episodes.jsonl")
    lerobot_info = load_json(lerobot_root / "info.json")

    progress("recovering steering-side tasks and building the conservative one-to-one join")
    steering_records = derive_steering_task_records(commands, trajectory_map)
    steering_by_id = {row["steering_trajectory_id"]: row for row in steering_records}
    episode_records = derive_lerobot_episode_records(episodes)
    joins, quality_issues, join_counts = build_conservative_join(
        steering_records, episode_records
    )
    join_sensitivity = join_normalization_sensitivity(commands, episodes)
    selected_sensitivity = next(
        row
        for row in join_sensitivity
        if row["normalization"] == "nfkc_case_ascii_alphanumeric"
    )
    join_counts["selected_join_tier"] = "A_internal_single_shared_task_only"
    join_counts["tier_b_vocabulary_disambiguated_pairs"] = selected_sensitivity[
        "one_to_one_pairs"
    ]
    join_counts["shared_chat_16945_estimate_reproduced"] = False
    write_csv(
        artifact_root / "join_normalization_sensitivity.csv",
        join_sensitivity,
        list(join_sensitivity[0]),
    )
    validations, validation_issues = validate_joined_trajectories(
        joins, step_to_subtask, commands
    )
    quality_issues.extend(validation_issues)

    progress("validating timestep coverage, command-pool joins, and temporal segments")
    pd.DataFrame(joins).sort_values("lerobot_episode_index").to_csv(
        artifact_root / "episode_join.csv", index=False
    )
    density_rows = _trajectory_density_rows(validations)
    pd.DataFrame(density_rows).sort_values("steering_trajectory_id").to_csv(
        artifact_root / "trajectory_density.csv", index=False
    )
    density_summary = _density_summary(density_rows)
    write_csv(
        artifact_root / "temporal_density_summary.csv",
        density_summary,
        list(density_summary[0]),
    )
    density_plots = _plot_density(density_rows, plot_dir)
    eligible_integrity = [
        row for row in validations if row["validation_status"] == "eligible"
    ]
    progress("writing the complete frame-to-language episode annotation manifest")
    episode_manifest_rows = _write_parquet_batches(
        manifest_dir / "episode_annotation_manifest.parquet",
        iter_frame_records(eligible_integrity, step_to_subtask, commands),
    )

    progress("classifying 1.7M command slots and producing the locked manual-audit sample")
    taxonomy_summary, pool_records, format_issues = _write_taxonomy(
        artifact_root / "command_taxonomy.csv",
        iter_taxonomy_rows(commands, steering_by_id),
    )
    quality_issues.extend(format_issues)
    anatomy_plot = _plot_anatomy(taxonomy_summary, plot_dir)
    manual_pool_rows = make_manual_pool_audit(commands, steering_by_id, sample_size=100)
    command_audit_write_status = write_audit_template(
        artifact_root / "manual_command_audit.csv",
        manual_pool_rows,
        list(manual_pool_rows[0]),
        review_fields=COMMAND_REVIEW_FIELDS,
    )

    density_eligible = [
        row
        for row in eligible_integrity
        if int(row["semantic_segment_count"]) >= 3
        and int(row["minimum_segment_length"]) >= 2
        and row["source_collection"] != "unknown"
        and len(str(row["normalized_task"]).split()) >= 2
    ]
    eligibility_rows = _eligibility_rows(
        pool_records, density_eligible, step_to_subtask
    )
    eligibility_frame = pd.DataFrame(eligibility_rows)
    eligibility_frame.to_csv(artifact_root / "eligible_intents.csv", index=False)
    eligibility_summary = _eligibility_summary(eligibility_rows)
    write_csv(
        artifact_root / "paraphrase_eligibility_table.csv",
        eligibility_summary,
        list(eligibility_summary[0]),
    )

    eligibility_rules = [
        "one-to-one normalized task join",
        "direct frame coverage and annotation length equals LeRobot length plus two",
        "all used frames resolve to nonempty all-string command pools",
        "at least three contiguous semantic segments",
        "minimum semantic segment length at least two frames",
        "known Bridge source collection",
        "nonempty normalized task with at least two tokens",
        "provisional generated verbatim-wrapper surface pools pending human review",
    ]
    progress("freezing leakage-safe pilot and target splits")
    target = stratified_split(density_eligible, TARGET_SPLIT_COUNTS, seed=seed + 2)
    pilot = nested_split(target, PILOT_SPLIT_COUNTS, seed=seed + 1)
    write_json(
        artifact_root / "pilot_split.json",
        split_payload(pilot, seed=seed + 1, eligibility_rules=eligibility_rules),
    )
    write_json(
        artifact_root / "target_split.json",
        split_payload(target, seed=seed + 2, eligibility_rules=eligibility_rules),
    )
    pilot_balance = balance_rows(pilot)
    target_balance = balance_rows(target)
    balance = pilot_balance + target_balance
    for row in balance[: len(pilot_balance)]:
        row["cohort"] = "pilot"
    for row in balance[len(pilot_balance) :]:
        row["cohort"] = "target"
    write_csv(artifact_root / "split_balance.csv", balance, list(balance[0]))

    target_ids = {
        int(row["steering_trajectory_id"]): split
        for split, rows in target.items()
        for row in rows
    }
    pilot_ids = {
        int(row["steering_trajectory_id"]): split
        for split, rows in pilot.items()
        for row in rows
    }
    target_group_membership: dict[str, set[str]] = defaultdict(set)
    for split, rows in target.items():
        for row in rows:
            target_group_membership[str(row["split_group_key"])].add(split)
    target_tasks = {
        split: {str(row["normalized_task"]) for row in rows}
        for split, rows in target.items()
    }
    eligible_source_counts = Counter(
        str(row["source_collection"]) for row in density_eligible
    )
    target_source_counts = Counter(
        str(row["source_collection"])
        for rows in target.values()
        for row in rows
    )
    pilot_source_counts = Counter(
        str(row["source_collection"])
        for rows in pilot.values()
        for row in rows
    )
    split_validation = {
        "target_trajectory_ids_disjoint": len(target_ids)
        == sum(len(rows) for rows in target.values()),
        "target_original_bridge_groups_disjoint": all(
            len(splits) == 1 for splits in target_group_membership.values()
        ),
        "pilot_is_subset_of_target": set(pilot_ids) <= set(target_ids),
        "pilot_roles_match_target": all(
            target_ids[trajectory_id] == split
            for trajectory_id, split in pilot_ids.items()
        ),
        "target_exact_task_overlap_counts": {
            f"{left}__{right}": len(target_tasks[left] & target_tasks[right])
            for index, left in enumerate(target_tasks)
            for right in list(target_tasks)[index + 1 :]
        },
        "exact_normalized_task_is_globally_unique_in_conservative_join": len(
            set().union(*target_tasks.values())
        )
        == sum(len(values) for values in target_tasks.values()),
        "density_eligible_source_counts": dict(eligible_source_counts),
        "target_source_counts": dict(target_source_counts),
        "pilot_source_counts": dict(pilot_source_counts),
        "eligible_sources_absent_from_target": {
            source: count
            for source, count in eligible_source_counts.items()
            if target_source_counts[source] == 0
        },
        "absent_source_interpretation": (
            "Bridge v1 and FLAP each have only one density-eligible trajectory; "
            "the deterministic exact-size target does not claim coverage of them."
        ),
        "surface_holdout_protocol": (
            "Use each view's selected_heldout_instruction on the same immutable "
            "row references. Val/test trajectory results are task plus trajectory "
            "generalization, not an isolated wording test."
        ),
    }
    split_validation["all_group_and_role_assertions_pass"] = all(
        split_validation[key]
        for key in (
            "target_trajectory_ids_disjoint",
            "target_original_bridge_groups_disjoint",
            "pilot_is_subset_of_target",
            "pilot_roles_match_target",
        )
    )
    write_json(artifact_root / "split_validation.json", split_validation)

    target_rows = [row for rows in target.values() for row in rows]
    split_by_trajectory = {
        int(row["steering_trajectory_id"]): split
        for split, rows in target.items()
        for row in rows
    }
    target_frame_rows = list(
        iter_frame_records(target_rows, step_to_subtask, commands)
    )
    progress("materializing the master row set and all four matched 2 x 2 views")
    master = build_master_manifest(target_frame_rows, split_by_trajectory, seed=seed)
    master_path = manifest_dir / "master_manifest.parquet"
    master.to_parquet(master_path, index=False, compression="zstd")
    views, view_paths = materialize_views(master, manifest_dir, seed=seed)
    manifest_validation = validate_views(master, views, seed=seed)
    surface_report = _surface_report(master)
    surface_match_validation = _surface_match_validation(surface_report)
    manifest_validation["surface_match_validation"] = surface_match_validation
    manifest_validation["scientific_language_conditions_pass"] = bool(
        manifest_validation["scientific_language_conditions_pass"]
        and surface_match_validation["all_pass"]
    )
    manifest_validation["training_ready"] = bool(
        manifest_validation["all_structural_assertions_pass"]
        and manifest_validation["scientific_language_conditions_pass"]
    )
    manifest_validation["paths"] = {
        "master": str(master_path),
        **view_paths,
    }
    write_json(artifact_root / "manifest_validation.json", manifest_validation)

    write_csv(
        artifact_root / "surface_diversity_report.csv",
        surface_report,
        list(surface_report[0]),
    )
    manual_group_rows = _manual_group_audit(master, seed)
    paraphrase_audit_write_status = write_audit_template(
        artifact_root / "manual_paraphrase_group_audit.csv",
        manual_group_rows,
        list(manual_group_rows[0]),
        review_fields=PARAPHRASE_REVIEW_FIELDS,
    )
    visual_rows = _manual_sequence_audit(
        [row for rows in pilot.values() for row in rows],
        step_to_subtask,
        sample_size=20,
        seed=seed,
    )
    visual_audit_write_status = write_audit_template(
        artifact_root / "visual_alignment_audit.csv",
        visual_rows,
        list(visual_rows[0]),
        review_fields=SEQUENCE_REVIEW_FIELDS,
    )
    sequence_rows = _manual_sequence_audit(
        density_eligible,
        step_to_subtask,
        sample_size=30,
        seed=seed + 30,
    )
    sequence_audit_write_status = write_audit_template(
        artifact_root / "manual_sequence_audit.csv",
        sequence_rows,
        list(sequence_rows[0]),
        review_fields=SEQUENCE_REVIEW_FIELDS,
    )

    pilot_rows = [row for rows in pilot.values() for row in rows]
    pilot_split_by_trajectory = {
        int(row["steering_trajectory_id"]): split
        for split, rows in pilot.items()
        for row in rows
    }
    pilot_frame_rows = list(
        iter_frame_records(pilot_rows, step_to_subtask, commands)
    )
    pilot_master = build_master_manifest(
        pilot_frame_rows, pilot_split_by_trajectory, seed=seed
    )
    pilot_manifest_dir = manifest_dir / "pilot"
    pilot_manifest_dir.mkdir(parents=True, exist_ok=True)
    pilot_master_path = pilot_manifest_dir / "master_manifest.parquet"
    pilot_master.to_parquet(pilot_master_path, index=False, compression="zstd")
    pilot_views, pilot_view_paths = materialize_views(
        pilot_master, pilot_manifest_dir, seed=seed
    )
    pilot_manifest_validation = validate_views(
        pilot_master, pilot_views, seed=seed
    )
    pilot_manifest_validation["paths"] = {
        "master": str(pilot_master_path),
        **pilot_view_paths,
    }
    write_json(
        artifact_root / "pilot_manifest_validation.json",
        pilot_manifest_validation,
    )
    implementation = implementation_manifest()
    write_json(artifact_root / "implementation_manifest.json", implementation)
    audit_write_statuses = {
        "command": command_audit_write_status,
        "paraphrase": paraphrase_audit_write_status,
        "visual_alignment": visual_audit_write_status,
        "sequence": sequence_audit_write_status,
    }
    build_audit_lock(
        artifact_root,
        overwrite=all(status == "written" for status in audit_write_statuses.values()),
        base_seed=seed,
    )
    human_gate = finalize_audits(artifact_root)

    trajectory_sets = {
        "trajectory_map": set(trajectory_map.get("traj_idx_to_key", {})),
        "step_map": set(step_to_subtask),
        "command_map": set(commands),
        "rationale_map": set(rationales),
    }
    intersection_all = set.intersection(*trajectory_sets.values())
    quality_counts = {
        "integrity_eligible_trajectories": len(eligible_integrity),
        "density_eligible_trajectories": len(density_eligible),
        "episode_annotation_manifest_rows": episode_manifest_rows,
        "quality_issue_rows": len(quality_issues),
        "annotation_length_delta_counts": dict(
            Counter(str(row["annotation_minus_episode_length"]) for row in validations)
        ),
        "direct_vs_plus_one_changed_frame_labels": sum(
            int(row["direct_vs_plus_one_label_change_count"]) for row in validations
        ),
        "joined_trajectories_affected_by_plus_one": sum(
            int(row["direct_vs_plus_one_label_change_count"]) > 0 for row in validations
        ),
        "unused_tail_rows": sum(
            int(row["annotation_minus_episode_length"]) for row in validations
        ),
        "unused_tail_starts_new_segment_trajectories": sum(
            bool(row["unused_tail_starts_new_segment"]) for row in validations
        ),
        "retained_pool_resolution_exact_frames": sum(
            int(row["pool_resolution_exact_frames"]) for row in eligible_integrity
        ),
        "retained_pool_resolution_normalized_key_frames": sum(
            int(row["pool_resolution_normalized_frames"]) for row in eligible_integrity
        ),
    }
    progress("assembling the inventory, rejection ledger, validation report, and decision memo")
    inventory = {
        "schema_version": 1,
        "input_manifest": input_manifest,
        "implementation_manifest": implementation,
        "released_files": {
            "trajectory_map_records": len(trajectory_sets["trajectory_map"]),
            "step_map_trajectories": len(trajectory_sets["step_map"]),
            "command_map_trajectories": len(trajectory_sets["command_map"]),
            "rationale_map_trajectories": len(trajectory_sets["rationale_map"]),
            "all_four_sidecar_intersection": len(intersection_all),
            "step_map_timestep_rows": sum(
                len(value) for value in step_to_subtask.values() if isinstance(value, dict)
            ),
            "command_map_semantic_pools": len(pool_records),
            "command_slots": taxonomy_summary["command_slots"],
            "set_differences": {
                name: len(values - intersection_all) for name, values in trajectory_sets.items()
            },
        },
        "lerobot": {
            "episodes_metadata_rows": len(episodes),
            "reported_total_episodes": lerobot_info["total_episodes"],
            "reported_total_frames": lerobot_info["total_frames"],
            "reported_total_tasks": lerobot_info["total_tasks"],
            "fps": lerobot_info["fps"],
        },
        "join": join_counts,
        "join_normalization_sensitivity": join_sensitivity,
        "quality": quality_counts,
        "taxonomy": taxonomy_summary,
        "paraphrase_eligibility": eligibility_summary,
        "split_validation": split_validation,
        "human_audit_gate": human_gate,
        "audit_template_write_status": audit_write_statuses,
        "retention_funnel": [
            {
                "stage": "released trajectory keys",
                "trajectories": len(trajectory_sets["trajectory_map"]),
                "timesteps": None,
                "semantic_segments": None,
            },
            {
                "stage": "has dense step and command maps",
                "trajectories": len(
                    trajectory_sets["step_map"] & trajectory_sets["command_map"]
                ),
                "timesteps": sum(
                    len(value)
                    for key, value in step_to_subtask.items()
                    if key in trajectory_sets["command_map"] and isinstance(value, dict)
                ),
                "semantic_segments": len(pool_records),
            },
            {
                "stage": "clean recovered steering task",
                "trajectories": join_counts["steering_clean_task_records"],
                "timesteps": None,
                "semantic_segments": None,
            },
            {
                "stage": "one-to-one task join",
                "trajectories": join_counts["one_to_one_joined_episodes"],
                "timesteps": sum(int(row["episode_length"]) for row in validations),
                "semantic_segments": sum(
                    int(row["semantic_segment_count"]) for row in validations
                ),
            },
            {
                "stage": "integrity eligible",
                "trajectories": len(eligible_integrity),
                "timesteps": episode_manifest_rows,
                "semantic_segments": sum(
                    int(row["semantic_segment_count"]) for row in eligible_integrity
                ),
            },
            {
                "stage": "temporal density eligible",
                "trajectories": len(density_eligible),
                "timesteps": sum(int(row["episode_length"]) for row in density_eligible),
                "semantic_segments": sum(
                    int(row["semantic_segment_count"]) for row in density_eligible
                ),
            },
            {
                "stage": "provisional pilot",
                "trajectories": sum(PILOT_SPLIT_COUNTS.values()),
                "timesteps": len(pilot_master),
                "semantic_segments": int(pilot_master["subtask_id"].nunique()),
            },
            {
                "stage": "provisional target",
                "trajectories": sum(TARGET_SPLIT_COUNTS.values()),
                "timesteps": len(master),
                "semantic_segments": int(master["subtask_id"].nunique()),
            },
        ],
        "plots": {**density_plots, "annotation_anatomy": anatomy_plot},
        "claim_limits": [
            "one-to-one normalized task identity is conservative but not an original Bridge key join",
            "direct timestep mapping reproduces released loader indexing; visual boundary correctness is pending",
            "automatic taxonomy candidates are not verified semantic paraphrases",
            "generated verbatim wrappers are provisional and provide low-strength surface diversity",
            "verified paraphrase capacity is zero until the locked human audits are finalized",
            "the current B/D total-length and lexical-divergence distributions are not matched",
            "exact task strings are unique in the conservative join, so trajectory test splits also change task identity",
            "reference hashes prove identical immutable references, not locally re-hashed media bytes",
            "the data does not establish same-observation counterfactual behavioral branches",
        ],
    }
    write_json(artifact_root / "annotation_inventory.json", inventory)
    quality_fields = ["scope", "record_id", "exclusion_reason", "details"]
    write_csv(artifact_root / "quality_issues.csv", quality_issues, quality_fields)
    _write_decision_memo(
        artifact_root / "DECISION_MEMO.md",
        inventory,
        eligibility_summary,
        manifest_validation,
        PILOT_SPLIT_COUNTS,
        TARGET_SPLIT_COUNTS,
    )

    summary = {
        "decision": "TRAINING NO-GO; CONDITIONAL GO TO CURATION",
        "implementation_combined_sha256": implementation["combined_sha256"],
        "joined_episodes": join_counts["one_to_one_joined_episodes"],
        "density_eligible_trajectories": len(density_eligible),
        "pilot_counts": dict(PILOT_SPLIT_COUNTS),
        "target_counts": dict(TARGET_SPLIT_COUNTS),
        "master_rows": len(master),
        "pilot_master_rows": len(pilot_master),
        "structural_manifest_assertions_pass": manifest_validation[
            "all_structural_assertions_pass"
        ],
        "scientific_language_conditions_pass": manifest_validation[
            "scientific_language_conditions_pass"
        ],
        "training_ready": human_gate["training_ready"],
        "required_human_artifacts": [
            str(artifact_root / "visual_alignment_audit.csv"),
            str(artifact_root / "manual_command_audit.csv"),
            str(artifact_root / "manual_paraphrase_group_audit.csv"),
            str(artifact_root / "manual_sequence_audit.csv"),
        ],
    }
    write_json(artifact_root / "run_summary.json", summary)
    progress("complete")
    return summary
