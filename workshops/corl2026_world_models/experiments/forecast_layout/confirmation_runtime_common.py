#!/usr/bin/env python3
"""Shared fail-closed contracts for WMF confirmation block launchers.

This module does not release confirmation.  It authenticates the immutable
prepared schedule and the post-development release freeze produced by
``analysis/freeze_development_release.py``.  Model-specific launchers must also
authenticate the selected C-layout gate, pose manifest, and fixed observation
before starting a server and again in every simulator cell.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


STUDY_ID = "WMF-ABLATION-001"
NAMESPACE = "wmf_ablation_001_20260912"
SCHEDULE_SCHEMA = "wmf-unreleased-block-schedule-v1"
CONFIRMATION_LAYOUT_IDS = tuple(f"C{index:02d}" for index in range(1, 25))
CONDITION_ALPHABET = (
    "original-left",
    "original-right",
    "reflected-left",
    "reflected-right",
)
REQUIRED_GATES = (
    "original_archive_identity_or_unavailable_result",
    "qualified_runtime_and_seed_behavior",
    "D1_official_or_verified_equivalence",
    "frozen_layout_pose_manifest_and_rejection_ledger",
    "per_model_video_action_physical_time_mapping",
    "primary_horizon_and_camera_tolerance",
    "isolated_DreamZero_temporal_context",
    "annotation_rubric_threshold_and_sampling_algorithm_seed",
    "frozen_execution_order_and_seed_audit",
    "selected_machine_and_bounded_resource_authorization",
    "post_development_confirmation_freeze",
)
EXECUTION_CONTRACT = {
    "full_model_and_simulator_reset_before_each_condition": True,
    "no_global_DreamZero_context_interleaving": True,
    "same_isolated_worker_type_within_job": True,
    "sequential_conditions": True,
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class ConfirmationRuntimeError(RuntimeError):
    """A stable, fail-closed confirmation contract failure."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        text = reason if detail is None else f"{reason}: {detail}"
        super().__init__(text)
        self.reason = reason
        self.detail = detail


def require(condition: bool, reason: str, detail: str | None = None) -> None:
    if not condition:
        raise ConfirmationRuntimeError(reason, detail)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def verify_exact_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    require(
        isinstance(expected_sha256, str) and SHA256_RE.fullmatch(expected_sha256) is not None,
        "confirmation_sha256_invalid",
        label,
    )
    supplied = Path(path)
    require(not supplied.is_symlink(), "confirmation_evidence_symlink", label)
    resolved = supplied.resolve()
    require(resolved.is_file(), "confirmation_evidence_missing", f"{label}:{resolved}")
    identity = file_identity(resolved)
    require(
        identity["sha256"] == expected_sha256,
        "confirmation_evidence_sha256_mismatch",
        label,
    )
    return identity


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfirmationRuntimeError("confirmation_json_unreadable", label) from error
    require(isinstance(value, dict), "confirmation_json_not_object", label)
    return value


@dataclass(frozen=True)
class ConfirmationScheduleBlock:
    """Exact schedule identity shared by the N3 and D1 wrappers."""

    layout_pair_id: str
    model_config: str
    environment_seed: int
    effective_model_seed: int
    condition_order: tuple[str, ...]
    conditions: tuple[tuple[str, str, str], ...]
    cell_ids: tuple[str, ...]
    block_id: str
    schedule_path: Path
    schedule_sha256: str
    schedule_row: Mapping[str, Any]


def condition_tuple(label: str) -> tuple[str, str, str]:
    require(label.count("-") == 1, "confirmation_condition_label_invalid", label)
    arm, command = label.split("-", 1)
    require(arm in {"original", "reflected"}, "confirmation_condition_arm_invalid", arm)
    require(command in {"left", "right"}, "confirmation_condition_command_invalid", command)
    return arm, command, f"WMFForecast{arm.title()}{command.title()}Task"


def _development_dependencies(models: Sequence[str]) -> list[str]:
    return [
        f"{NAMESPACE}__development__D{index:02d}__{model}"
        for index in range(1, 5)
        for model in models
    ]


def _validate_global_confirmation_order(schedule: Mapping[str, Any]) -> None:
    assignment = schedule.get("order_assignment")
    require(isinstance(assignment, Mapping), "confirmation_order_assignment_missing")
    require(
        assignment.get("condition_alphabet") == list(CONDITION_ALPHABET),
        "confirmation_condition_alphabet_changed",
    )
    require(
        assignment.get("same_order_for_both_models") is True
        and assignment.get("assignment_uses_outcomes") is False,
        "confirmation_order_assignment_not_model_blind",
    )
    require(
        assignment.get("confirmation_algorithm")
        == "Sort C01-C24 by digest (block ID tie-break); assign the 24 lexicographic permutations in that order.",
        "confirmation_order_algorithm_changed",
    )
    require(
        assignment.get("input")
        == "ASCII namespace + ':' + layout_pair_id; SHA-256 hexadecimal digest",
        "confirmation_order_hash_input_changed",
    )
    jobs = schedule.get("jobs")
    require(isinstance(jobs, list), "confirmation_schedule_jobs_invalid")
    by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in jobs:
        if not isinstance(row, Mapping) or row.get("phase") != "confirmation":
            continue
        key = (str(row.get("layout_pair_id")), str(row.get("model_config")))
        require(key not in by_identity, "confirmation_schedule_row_duplicate", ":".join(key))
        by_identity[key] = row
    expected_keys = {
        (layout, model)
        for layout in CONFIRMATION_LAYOUT_IDS
        for model in ("N3", "D1")
    }
    require(set(by_identity) == expected_keys, "confirmation_schedule_inventory_changed")

    permutations = tuple(itertools.permutations(CONDITION_ALPHABET))
    require(len(permutations) == 24, "confirmation_permutation_inventory_invalid")
    observed_orders: list[tuple[str, ...]] = []
    block_hashes = assignment.get("block_sha256")
    indices = assignment.get("permutation_index_by_block")
    hash_order = assignment.get("confirmation_blocks_in_hash_order")
    require(isinstance(block_hashes, Mapping), "confirmation_block_hashes_missing")
    require(isinstance(indices, Mapping), "confirmation_permutation_indices_missing")
    require(
        isinstance(hash_order, list)
        and len(hash_order) == 24
        and set(hash_order) == set(CONFIRMATION_LAYOUT_IDS),
        "confirmation_hash_order_invalid",
    )
    for layout in CONFIRMATION_LAYOUT_IDS:
        n3 = by_identity[(layout, "N3")]
        d1 = by_identity[(layout, "D1")]
        order = tuple(n3.get("condition_order", ()))
        require(order == tuple(d1.get("condition_order", ())), "confirmation_model_order_mismatch", layout)
        require(
            len(order) == 4 and set(order) == set(CONDITION_ALPHABET),
            "confirmation_condition_order_not_permutation",
            layout,
        )
        expected_hash = hashlib.sha256(f"{NAMESPACE}:{layout}".encode("ascii")).hexdigest()
        require(block_hashes.get(layout) == expected_hash, "confirmation_block_hash_changed", layout)
        index = indices.get(layout)
        require(type(index) is int and 0 <= index < 24, "confirmation_permutation_index_invalid", layout)
        expected_index = hash_order.index(layout)
        require(index == expected_index, "confirmation_permutation_index_changed", layout)
        require(
            order == permutations[expected_index],
            "confirmation_condition_order_assignment_changed",
            layout,
        )
        observed_orders.append(order)
    require(len(set(observed_orders)) == 24, "confirmation_permutations_not_used_exactly_once")
    computed_hash_order = sorted(CONFIRMATION_LAYOUT_IDS, key=lambda item: (block_hashes[item], item))
    require(computed_hash_order == hash_order, "confirmation_hash_order_changed")


def load_confirmation_schedule_block(
    source_root: Path, layout_pair_id: str, model_config: str
) -> ConfirmationScheduleBlock:
    """Authenticate one of the 48 prepared confirmation rows.

    ``parallel_schedule.json`` deliberately remains unreleased.  A queue job
    becomes eligible only when the separate post-development release freeze is
    supplied and deeply validated by :func:`verify_confirmation_freeze`.
    """

    require(layout_pair_id in CONFIRMATION_LAYOUT_IDS, "confirmation_layout_unsupported", str(layout_pair_id))
    require(model_config in {"N3", "D1"}, "confirmation_model_unsupported", str(model_config))
    source_root = Path(source_root).resolve()
    schedule_path = (
        source_root
        / "workshops/corl2026_world_models/execution/20260912/parallel_schedule.json"
    )
    schedule = load_json(schedule_path, "parallel_schedule")
    require(schedule.get("schema_version") == SCHEDULE_SCHEMA, "confirmation_schedule_schema_changed")
    require(schedule.get("spec_id") == STUDY_ID, "confirmation_schedule_study_changed")
    require(schedule.get("namespace") == NAMESPACE, "confirmation_schedule_namespace_changed")
    require(schedule.get("job_count") == 58 and schedule.get("cell_count") == 232, "confirmation_schedule_counts_changed")
    require(schedule.get("D2_selected") is False, "confirmation_optional_d2_selected")
    require(schedule.get("models") == ["N3", "D1"], "confirmation_schedule_models_changed")
    require(schedule.get("launch_ready") is False, "prepared_schedule_was_retroactively_released")
    require(schedule.get("status") == "ORDER_PREPARED_NOT_RELEASED", "confirmation_schedule_status_changed")
    _validate_global_confirmation_order(schedule)

    block_id = f"{NAMESPACE}__confirmation__{layout_pair_id}__{model_config}"
    jobs = schedule.get("jobs")
    rows = [row for row in jobs if isinstance(row, Mapping) and row.get("job_id") == block_id]
    require(len(rows) == 1, "confirmation_schedule_row_missing_or_duplicate", block_id)
    row = rows[0]
    order = tuple(row.get("condition_order", ()))
    cell_ids = tuple(
        f"wmf1__confirmation__{layout_pair_id}__{model_config}__{label.replace('-', '__')}"
        for label in order
    )
    layout_index = int(layout_pair_id[1:])
    environment_seed = 2026091200 + layout_index
    effective_model_seed = environment_seed if model_config == "N3" else 1140
    expected = {
        "phase": "confirmation",
        "layout_pair_id": layout_pair_id,
        "model_config": model_config,
        "candidate_effective_policy_seed": effective_model_seed,
        "condition_order": list(order),
        "ordered_cell_ids": list(cell_ids),
        "indivisible": True,
        "execution_contract": EXECUTION_CONTRACT,
        "depends_on_jobs": _development_dependencies(("D1", "N3")),
        "required_gates": list(REQUIRED_GATES),
        "released": False,
        "status": "NOT_RELEASED",
    }
    for key, wanted in expected.items():
        require(row.get(key) == wanted, "confirmation_schedule_mismatch", key)
    return ConfirmationScheduleBlock(
        layout_pair_id=layout_pair_id,
        model_config=model_config,
        environment_seed=environment_seed,
        effective_model_seed=effective_model_seed,
        condition_order=order,
        conditions=tuple(condition_tuple(label) for label in order),
        cell_ids=cell_ids,
        block_id=block_id,
        schedule_path=schedule_path,
        schedule_sha256=sha256_file(schedule_path),
        schedule_row=dict(row),
    )


def _load_release_module(source_root: Path):
    module_path = (
        Path(source_root).resolve()
        / "workshops/corl2026_world_models/analysis/freeze_development_release.py"
    )
    require(module_path.is_file(), "confirmation_release_validator_missing", str(module_path))
    name = "wmf_freeze_development_release_runtime"
    spec = importlib.util.spec_from_file_location(name, module_path)
    require(spec is not None and spec.loader is not None, "confirmation_release_validator_unloadable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as error:
        raise ConfirmationRuntimeError("confirmation_release_validator_import_failed", str(error)) from error
    require(callable(getattr(module, "validate_release_freeze", None)), "confirmation_release_validator_api_missing")
    return module


def verify_confirmation_freeze(
    path: Path,
    expected_sha256: str,
    *,
    source_root: Path,
    model_config: str,
) -> dict[str, Any]:
    """Deeply verify the real post-development/annotation release artifact."""

    identity = verify_exact_file(path, expected_sha256, "confirmation_freeze")
    module = _load_release_module(source_root)
    try:
        validation = module.validate_release_freeze(
            Path(identity["path"]), identity["sha256"], expected_model=model_config
        )
    except BaseException as error:
        raise ConfirmationRuntimeError("confirmation_release_freeze_invalid", str(error)) from error
    require(isinstance(validation, Mapping), "confirmation_release_validation_invalid")
    value = load_json(Path(identity["path"]), "confirmation_freeze")
    require(value.get("schema_version") == "wmf-development-confirmation-release-freeze-v1", "confirmation_release_schema_changed")
    require(value.get("status") == "frozen_for_confirmation", "confirmation_release_status_changed")
    qualified = value.get("qualified_model_ids")
    require(isinstance(qualified, list) and model_config in qualified, "confirmation_model_not_released", model_config)
    contracts = value.get("alignment_contracts_by_model")
    require(
        isinstance(contracts, Mapping)
        and isinstance(contracts.get(model_config), Mapping),
        "confirmation_alignment_contract_missing",
        model_config,
    )
    decision = value.get("release_decision")
    require(
        isinstance(decision, Mapping)
        and decision.get("eligible") is True
        and decision.get("blockers") == [],
        "confirmation_release_decision_not_eligible",
    )
    return {
        "confirmation_freeze": identity,
        "cohort_branch": value.get("cohort_branch"),
        "qualified_model_ids": list(qualified),
        "alignment_contract": dict(contracts[model_config]),
        "release_validation": dict(validation),
    }


def descriptor_option(argv: Sequence[Any], option: str) -> str:
    matches = [index for index, item in enumerate(argv) if item == option]
    require(
        len(matches) == 1 and matches[0] + 1 < len(argv),
        "confirmation_queue_option_invalid",
        option,
    )
    value = argv[matches[0] + 1]
    require(isinstance(value, str), "confirmation_queue_option_invalid", option)
    return value
