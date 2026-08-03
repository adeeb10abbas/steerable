from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .taxonomy import generated_wrapper_paraphrases
from .text import stable_id


CONDITIONS = {
    "A_task_canonical": {"temporal_density": "low", "surface_diversity": "low"},
    "B_task_paraphrases": {"temporal_density": "low", "surface_diversity": "high"},
    "C_subtask_canonical": {"temporal_density": "high", "surface_diversity": "low"},
    "D_subtask_paraphrases": {"temporal_density": "high", "surface_diversity": "high"},
}


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def _choose(pool: list[str], *parts: object) -> str:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode()).digest()
    return pool[int.from_bytes(digest[:8], "big") % len(pool)]


def build_master_manifest(
    frame_rows: Iterable[Mapping[str, Any]],
    split_by_trajectory: Mapping[int, str],
    *,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    task_cache: dict[str, tuple[list[str], list[str]]] = {}
    subtask_cache: dict[str, tuple[list[str], list[str]]] = {}
    for frame in frame_rows:
        trajectory_id = int(frame["steering_trajectory_id"])
        if trajectory_id not in split_by_trajectory:
            continue
        canonical_task = str(frame["task_instruction"])
        canonical_subtask = str(frame["subtask_text"])
        task_cache.setdefault(canonical_task, generated_wrapper_paraphrases(canonical_task))
        subtask_cache.setdefault(canonical_subtask, generated_wrapper_paraphrases(canonical_subtask))
        task_train, task_heldout = task_cache[canonical_task]
        subtask_train, subtask_heldout = subtask_cache[canonical_subtask]
        observation_ref = str(frame["observation_ref"])
        action_ref = str(frame["action_ref"])
        row_key_hash = stable_id(
            "row",
            trajectory_id,
            frame["frame_index"],
            observation_ref,
            action_ref,
            length=32,
        )
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "lerobot_episode_index": int(frame["lerobot_episode_index"]),
                "timestep": int(frame["frame_index"]),
                "annotation_timestep": int(frame["annotation_timestep"]),
                "raw_bridge_observation_index": int(frame["raw_bridge_observation_index"]),
                "split": split_by_trajectory[trajectory_id],
                "source_collection": frame["source_collection"],
                "task_family": frame["task_family"],
                "task_id": frame["task_id"],
                "subtask_id": frame["subtask_id"],
                "semantic_segment_index": int(frame["semantic_segment_index"]),
                "canonical_task": canonical_task,
                "task_train_paraphrases": _json_list(task_train),
                "task_heldout_paraphrases": _json_list(task_heldout),
                "canonical_subtask": canonical_subtask,
                "subtask_train_paraphrases": _json_list(subtask_train),
                "subtask_heldout_paraphrases": _json_list(subtask_heldout),
                "paraphrase_origin": "generated_verbatim_wrapper_provisional",
                "paraphrase_audit_status": "pending_human_review",
                "observation_ref": observation_ref,
                "action_ref": action_ref,
                "row_key_hash": row_key_hash,
                "sampling_seed": seed,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Master manifest would be empty")
    frame.sort_values(["trajectory_id", "timestep"], inplace=True, ignore_index=True)
    if frame["row_key_hash"].duplicated().any():
        raise AssertionError("Duplicate master row keys")
    return frame


def _materialize_view(master: pd.DataFrame, condition: str, seed: int) -> pd.DataFrame:
    spec = CONDITIONS[condition]
    task_level = spec["temporal_density"] == "low"
    high_diversity = spec["surface_diversity"] == "high"
    output = master[
        [
            "trajectory_id",
            "lerobot_episode_index",
            "timestep",
            "annotation_timestep",
            "raw_bridge_observation_index",
            "split",
            "source_collection",
            "task_family",
            "task_id",
            "subtask_id",
            "semantic_segment_index",
            "observation_ref",
            "action_ref",
            "row_key_hash",
        ]
    ].copy()
    output["condition"] = condition
    output["temporal_density"] = spec["temporal_density"]
    output["surface_diversity"] = spec["surface_diversity"]
    output["language_level"] = "task" if task_level else "subtask"
    output["semantic_intent_id"] = master["task_id"] if task_level else master["subtask_id"]
    canonical = master["canonical_task"] if task_level else master["canonical_subtask"]
    train_column = "task_train_paraphrases" if task_level else "subtask_train_paraphrases"
    heldout_column = "task_heldout_paraphrases" if task_level else "subtask_heldout_paraphrases"

    train_pools: list[str] = []
    heldout_pools: list[str] = []
    selected_training: list[str] = []
    selected_heldout: list[str] = []
    for index, row in master.iterrows():
        if high_diversity:
            train_pool = json.loads(row[train_column])
            heldout_pool = json.loads(row[heldout_column])
        else:
            train_pool = [str(canonical.iloc[index])]
            heldout_pool = [str(canonical.iloc[index])]
        train_pools.append(_json_list(train_pool))
        heldout_pools.append(_json_list(heldout_pool))
        selected_training.append(
            _choose(
                train_pool,
                seed,
                condition,
                row["trajectory_id"],
                row["timestep"],
                "training_language",
            )
        )
        selected_heldout.append(
            _choose(
                heldout_pool,
                seed,
                condition,
                row["trajectory_id"],
                row["timestep"],
                "heldout_language",
            )
        )
    output["canonical_instruction"] = canonical.values
    output["training_instruction_pool"] = train_pools
    output["heldout_instruction_pool"] = heldout_pools
    output["selected_training_instruction"] = selected_training
    output["selected_heldout_instruction"] = selected_heldout
    output["selected_instruction"] = selected_training
    output["selected_instruction_partition"] = "training_language"
    output["sampling_policy"] = "uniform_runtime_pool_per_language_partition"
    output["sampling_seed"] = seed
    output["paraphrase_origin"] = (
        "generated_verbatim_wrapper_provisional" if high_diversity else "released_canonical"
    )
    output["paraphrase_audit_status"] = (
        "pending_human_review" if high_diversity else "not_applicable"
    )
    output["surface_holdout_protocol"] = (
        "score selected_heldout_instruction on the same immutable row references; "
        "do not infer wording generalization from trajectory split alone"
    )

    if task_level:
        output["semantic_update"] = ~output["trajectory_id"].duplicated()
    else:
        output["semantic_update"] = (
            output.groupby("trajectory_id")["semantic_intent_id"].transform(
                lambda values: values.ne(values.shift())
            )
        )
    return output


def materialize_views(
    master: pd.DataFrame,
    output_dir: Path,
    *,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    views: dict[str, pd.DataFrame] = {}
    paths: dict[str, str] = {}
    for condition in CONDITIONS:
        view = _materialize_view(master, condition, seed)
        path = output_dir / f"{condition}.parquet"
        view.to_parquet(path, index=False, compression="zstd")
        views[condition] = view
        paths[condition] = str(path)
    return views, paths


def _dataset_reference_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.sort_values(["trajectory_id", "timestep"]).itertuples():
        digest.update(
            f"{row.trajectory_id}\x1f{row.timestep}\x1f{row.observation_ref}\x1f{row.action_ref}\n".encode()
        )
    return digest.hexdigest()


def validate_views(
    master: pd.DataFrame,
    views: Mapping[str, pd.DataFrame],
    *,
    seed: int,
) -> dict[str, Any]:
    master_keys = list(master["row_key_hash"])
    master_splits = dict(zip(master["row_key_hash"], master["split"]))
    expected_hash = _dataset_reference_hash(master)
    assertions: dict[str, dict[str, Any]] = {}
    condition_key_set_exact = set(views) == set(CONDITIONS)
    for condition in CONDITIONS:
        if condition not in views:
            assertions[condition] = {
                "checks": {"view_present": False},
                "all_pass": False,
                "row_count": 0,
                "reference_hash": None,
                "training_pool_sizes": [],
                "heldout_pool_sizes": [],
                "training_heldout_overlap_count": None,
            }
            continue
        view = views[condition]
        spec = CONDITIONS[condition]
        high = spec["surface_diversity"] == "high"
        task_level = spec["temporal_density"] == "low"
        train_overlap = 0
        training_pool_sizes: set[int] = set()
        heldout_pool_sizes: set[int] = set()
        expected_canonical = (
            master["canonical_task"] if task_level else master["canonical_subtask"]
        )
        expected_intent = master["task_id"] if task_level else master["subtask_id"]
        expected_train_column = (
            "task_train_paraphrases" if task_level else "subtask_train_paraphrases"
        )
        expected_heldout_column = (
            "task_heldout_paraphrases" if task_level else "subtask_heldout_paraphrases"
        )
        pools_match_master = True
        selected_training_membership = True
        selected_heldout_membership = True
        deterministic_selection = True
        for position, row in enumerate(view.itertuples()):
            train_pool_values = json.loads(row.training_instruction_pool)
            heldout_pool_values = json.loads(row.heldout_instruction_pool)
            train_pool = set(train_pool_values)
            heldout_pool = set(heldout_pool_values)
            training_pool_sizes.add(len(train_pool))
            heldout_pool_sizes.add(len(heldout_pool))
            train_overlap += len(train_pool & heldout_pool)
            if high:
                expected_train = json.loads(master.iloc[position][expected_train_column])
                expected_heldout = json.loads(master.iloc[position][expected_heldout_column])
            else:
                expected_train = [str(expected_canonical.iloc[position])]
                expected_heldout = [str(expected_canonical.iloc[position])]
            pools_match_master &= (
                train_pool_values == expected_train
                and heldout_pool_values == expected_heldout
            )
            selected_training_membership &= row.selected_training_instruction in train_pool
            selected_heldout_membership &= row.selected_heldout_instruction in heldout_pool
            if train_pool_values:
                deterministic_selection &= row.selected_training_instruction == _choose(
                    train_pool_values,
                    seed,
                    condition,
                    row.trajectory_id,
                    row.timestep,
                    "training_language",
                )
            else:
                deterministic_selection = False
            if heldout_pool_values:
                deterministic_selection &= row.selected_heldout_instruction == _choose(
                    heldout_pool_values,
                    seed,
                    condition,
                    row.trajectory_id,
                    row.timestep,
                    "heldout_language",
                )
            else:
                deterministic_selection = False

        expected_updates = (
            view.groupby("trajectory_id")["semantic_intent_id"]
            .transform(lambda values: values.ne(values.shift()))
            .astype(bool)
        )
        semantic_update_ok = bool(
            (expected_updates.values == view["semantic_update"].values).all()
        )
        if task_level:
            semantic_update_ok &= bool(
                (view.groupby("trajectory_id")["semantic_intent_id"].nunique() == 1).all()
            )
        expected_audit = "pending_human_review" if high else "not_applicable"
        expected_origin = (
            "generated_verbatim_wrapper_provisional" if high else "released_canonical"
        )
        checks = {
            "view_present": True,
            "row_count_equal": len(view) == len(master),
            "ordered_row_keys_equal": list(view["row_key_hash"]) == master_keys,
            "split_membership_equal": dict(zip(view["row_key_hash"], view["split"])) == master_splits,
            "pinned_reference_hash_equal": _dataset_reference_hash(view) == expected_hash,
            "condition_metadata_exact": (
                set(view["condition"]) == {condition}
                and set(view["temporal_density"]) == {spec["temporal_density"]}
                and set(view["surface_diversity"]) == {spec["surface_diversity"]}
                and set(view["language_level"]) == ({"task"} if task_level else {"subtask"})
            ),
            "canonical_instruction_matches_master": bool(
                (view["canonical_instruction"].values == expected_canonical.values).all()
            ),
            "semantic_intent_matches_master": bool(
                (view["semantic_intent_id"].values == expected_intent.values).all()
            ),
            "instruction_pools_match_master": pools_match_master,
            "training_heldout_disjoint": train_overlap == 0 if high else True,
            "expected_training_pool_size": training_pool_sizes == ({4} if high else {1}),
            "expected_heldout_pool_size": heldout_pool_sizes == ({2} if high else {1}),
            "selected_training_instruction_in_pool": selected_training_membership,
            "selected_heldout_instruction_in_pool": selected_heldout_membership,
            "selected_instruction_is_training_partition": bool(
                (view["selected_instruction"] == view["selected_training_instruction"]).all()
                and set(view["selected_instruction_partition"]) == {"training_language"}
            ),
            "deterministic_language_selection": deterministic_selection,
            "semantic_update_rule": semantic_update_ok,
            "fixed_seed": set(view["sampling_seed"]) == {seed},
            "uniform_sampling_policy": set(view["sampling_policy"])
            == {"uniform_runtime_pool_per_language_partition"},
            "paraphrase_metadata_exact": (
                set(view["paraphrase_origin"]) == {expected_origin}
                and set(view["paraphrase_audit_status"]) == {expected_audit}
            ),
        }
        assertions[condition] = {
            "checks": checks,
            "all_pass": all(checks.values()),
            "row_count": len(view),
            "reference_hash": _dataset_reference_hash(view),
            "training_pool_sizes": sorted(training_pool_sizes),
            "heldout_pool_sizes": sorted(heldout_pool_sizes),
            "training_heldout_overlap_count": train_overlap if high else None,
        }
    structural_pass = condition_key_set_exact and all(
        value["all_pass"] for value in assertions.values()
    )
    high_diversity_verified = condition_key_set_exact and all(
        set(views[condition]["paraphrase_audit_status"]) == {"verified"}
        for condition in ("B_task_paraphrases", "D_subtask_paraphrases")
    )
    return {
        "schema_version": 1,
        "sampling_seed": seed,
        "master_row_count": len(master),
        "master_reference_hash": expected_hash,
        "condition_key_set_exact": condition_key_set_exact,
        "unexpected_condition_keys": sorted(set(views) - set(CONDITIONS)),
        "missing_condition_keys": sorted(set(CONDITIONS) - set(views)),
        "reference_hash_definition": (
            "SHA-256 over pinned trajectory/timestep/observation_ref/action_ref records; "
            "this is a reference-identity hash, not a media-byte hash"
        ),
        "conditions": assertions,
        "all_structural_assertions_pass": structural_pass,
        "scientific_language_conditions_pass": high_diversity_verified,
        "training_ready": structural_pass and high_diversity_verified,
        "scientific_status": (
            "structurally_valid_but_language_unverified"
            if structural_pass and not high_diversity_verified
            else "verified"
            if structural_pass
            else "invalid"
        ),
    }
