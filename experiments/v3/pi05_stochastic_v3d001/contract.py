#!/usr/bin/env python3
"""Fail-closed contract for the released V3-D001 π0.5 queue."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.v3.pi05_droid.adapter import (
    FROZEN_CHECKPOINT,
    FROZEN_CONFIG,
    FROZEN_OPENPI_COMMIT,
    FROZEN_ROBOLAB_COMMIT,
    PROMPTS,
    TASKS,
    sha256_file,
    validate_release_gate as validate_phase_a_release_gate,
    validate_runtime_identity as validate_phase_a_runtime_identity,
)


STUDY_ID = "vla_wam_language_steerability_v3"
REGISTRATION_ID = "V3-D001"
MODEL_ID = "pi05_current_stack_droid"
PHASE = "D_16_rollout_stochastic_block"
ARENA = "droid_robolab"
SEEDS = tuple(range(8303, 8330))
SAMPLING_INDICES = tuple(range(8))
RELATIONS = ("left", "right")
ACTION_CAP = 450
ACTION_CHUNK_STEPS = 15
ACTION_DIM = 8
ACTION_SPACE = "joint_position_8d"
SUCCESS_PREDICATE_ID = "v2_frozen_droid_robolab_release_inside_45deg_requested_relation"

RELEASE_ROOT = Path("artifacts/vla_wam_shared_v3/prospective_tier_b/releases/v3d001")
GATE_ROOT = Path("artifacts/vla_wam_shared_v3/prospective_tier_b/gates/v3d001")
QUEUE_NAME = "pi05_v3d001_stochastic_cells.jsonl"
RELEASE_MANIFEST_NAME = "release_manifest.json"
RELEASE_AMENDMENT_NAME = "release_amendment.json"
QUEUE_SHA256 = "50bad0d86c0c1f2ef7204962dbefd20b04f55212aa90efea5870dfe9ef076d22"
RELEASE_MANIFEST_SHA256 = "22855cb1df09fa882059dd520b6aa30028391b0b53f6092d30a8563b1c355c58"
RELEASE_AMENDMENT_SHA256 = "cdf41633f69526ee6fb6f0553ba8fa069d774e29db415f8123c4ebb2aba2982c"
ELIGIBILITY_REPORT_SHA256 = "1108ff2c28c269f9dad307e80d363f93bb4ec7a9482894b74460da4281168b66"
ELIGIBILITY_MANIFEST_SHA256 = "6f192d8c179bd36bc8b4246b8013dbec5ec07ea81ec0ad9a2521776b5ba5cb98"
RAW_SAMPLE_RECORDS_SHA256 = "73695d44b49c74ebfe2e89c3ca8b98ea353851ec6fdf9c209faf1c595205f59a"
REGISTRATION_SHA256 = "899a52c79355919210d56fa8f31d944f8a373e1e184650ee8974d62acfd6c788"
SCOPE_CORRECTION_SHA256 = "b8969639a1c45f5fd8981c5e053f170a8a6ddac5ae7ffd2185e08ff40f751b9e"
PHASE_D_REGISTRY_SHA256 = "e319f8dcaefa6803ca46989313ba737834eef1dd531c1898aeee5fa816a28ad9"
PHASE_A_SUMMARY_SHA256 = "5c6d07fca7a0d20ab8b757f028d469f864c78a1c43ffedc3d257a16caef2a02b"
PHASE_A_RUNTIME_SHA256 = "e73fe7a0cc22db09fa8fdc0babf80dd8ad3280d0502285c6ad1c4d822c7fa532"


class ContractError(ValueError):
    """Raised before inference when any released identity differs."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ContractError(f"blank JSONL row: {path}:{number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ContractError(f"non-object JSONL row: {path}:{number}")
        rows.append(value)
    return rows


@dataclass(frozen=True)
class AuthorizedCell:
    row: dict[str, Any]

    @property
    def cell_id(self) -> str:
        return str(self.row["cell_id"])

    @property
    def environment_seed(self) -> int:
        return int(self.row["environment_seed"])

    @property
    def relation(self) -> str:
        return str(self.row["requested_relation"])

    @property
    def sampling_index(self) -> int:
        return int(self.row["shared_policy_sampling_seed_index"])

    @property
    def sampling_seed_base(self) -> int:
        return int(self.row["policy_sampling_seed_base"])

    @property
    def block_id(self) -> str:
        return str(self.row["matched_stochastic_block_id"])


@dataclass(frozen=True)
class Release:
    manifest_path: Path
    queue_path: Path
    manifest: dict[str, Any]
    amendment: dict[str, Any]
    cells: tuple[AuthorizedCell, ...]

    @property
    def by_id(self) -> dict[str, AuthorizedCell]:
        return {cell.cell_id: cell for cell in self.cells}

    def cell(self, cell_id: str) -> AuthorizedCell:
        try:
            return self.by_id[cell_id]
        except KeyError as exc:
            raise ContractError(f"cell is outside exact V3-D001 release: {cell_id}") from exc

    def fingerprint(self, cell: AuthorizedCell) -> str:
        return sha256_bytes(canonical_json_bytes({
            "release_manifest_sha256": RELEASE_MANIFEST_SHA256,
            "release_amendment_sha256": RELEASE_AMENDMENT_SHA256,
            "queue_sha256": QUEUE_SHA256,
            "cell": cell.row,
        }))


def _verify_cell(row: Mapping[str, Any]) -> None:
    seed = row.get("environment_seed")
    relation = row.get("requested_relation")
    index = row.get("shared_policy_sampling_seed_index")
    if seed not in SEEDS or relation not in RELATIONS or index not in SAMPLING_INDICES:
        raise ContractError("V3-D001 queue contains an unexpected seed/direction/index")
    expected = {
        "schema_version": "vla-wam-shared-v3d001-pi05-stochastic-cell-v1",
        "study_id": STUDY_ID,
        "phase": PHASE,
        "registration_id": REGISTRATION_ID,
        "model_id": MODEL_ID,
        "arena": ARENA,
        "cell_id": f"v3d001:pi05:env{seed}:{relation}:sample{index}",
        "nested_condition_id": f"v3:droid:{MODEL_ID}:seed{seed}:{relation}",
        "matched_stochastic_block_id": f"v3d001:pi05:env{seed}:sample{index}",
        "prompt": PROMPTS[relation],
        "prompt_mode": "static_episode_prompt",
        "policy_sampling_seed_base": seed * 1_000_000 + index * 1_000,
        "per_request_sampling_seed_rule": "policy_sampling_seed_base + zero_based_request_index",
        "source_phase_a_runtime_identity_sha256": PHASE_A_RUNTIME_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "phase_d_registry_sha256": PHASE_D_REGISTRY_SHA256,
        "scope_correction_sha256": SCOPE_CORRECTION_SHA256,
        "phase_a_summary_sha256": PHASE_A_SUMMARY_SHA256,
        "behavioral_status": "authorized_not_launched",
        "analysis_unit": "policy-sampling rollout nested within condition; not an independent scene",
    }
    for key, wanted in expected.items():
        if row.get(key) != wanted:
            raise ContractError(f"V3-D001 row mismatch for {expected['cell_id']}:{key}")
    body = {key: value for key, value in row.items() if key != "cell_sha256"}
    if row.get("cell_sha256") != sha256_bytes(canonical_json_bytes(body)):
        raise ContractError(f"V3-D001 cell hash changed: {expected['cell_id']}")


def load_release(repo_root: Path, manifest_path: Path | None = None) -> Release:
    root = Path(repo_root).resolve()
    manifest_path = (Path(manifest_path).resolve() if manifest_path else root / RELEASE_ROOT / RELEASE_MANIFEST_NAME)
    release_dir = manifest_path.parent
    queue_path = release_dir / QUEUE_NAME
    amendment_path = release_dir / RELEASE_AMENDMENT_NAME
    bound = {
        manifest_path: RELEASE_MANIFEST_SHA256,
        queue_path: QUEUE_SHA256,
        amendment_path: RELEASE_AMENDMENT_SHA256,
        root / GATE_ROOT / "eligibility_report.json": ELIGIBILITY_REPORT_SHA256,
        root / GATE_ROOT / "evidence_manifest.json": ELIGIBILITY_MANIFEST_SHA256,
        root / GATE_ROOT / "raw_sample_records.jsonl": RAW_SAMPLE_RECORDS_SHA256,
        root / "artifacts/vla_wam_shared_v3/results/pi05_current_stack_droid_phase_a_summary.json": PHASE_A_SUMMARY_SHA256,
    }
    for path, wanted in bound.items():
        if not path.is_file() or sha256_file(path) != wanted:
            raise ContractError(f"V3-D001 source binding changed: {path}")
    manifest = _object(manifest_path)
    amendment = _object(amendment_path)
    if manifest.get("status") != "exact_432_cell_queue_released_zero_behavior_launched" or manifest.get("counts") != {
        "cells": 432, "conditions": 54, "environment_seeds": 27,
        "directions": 2, "sampling_seed_indices": 8, "launched": 0,
    }:
        raise ContractError("V3-D001 release manifest identity changed")
    if amendment.get("behavioral_release") is not True or amendment.get("authorized_behavioral_cells") != 432 or amendment.get("launched_behavioral_cells_at_release") != 0:
        raise ContractError("V3-D001 amendment does not release exactly 432 unlaunched cells")
    eligibility = _object(root / GATE_ROOT / "eligibility_report.json")
    if eligibility.get("passed") is not True or eligibility.get("model_request_count") != 32 or eligibility.get("behavioral_episode_count") != 0 or eligibility.get("sampling_seed_indices") != list(SAMPLING_INDICES):
        raise ContractError("V3-D001 effective-seed eligibility changed")
    rows = _jsonl(queue_path)
    if len(rows) != 432:
        raise ContractError("V3-D001 queue must contain 432 rows")
    for row in rows:
        _verify_cell(row)
    ids = [row["cell_id"] for row in rows]
    if len(set(ids)) != 432:
        raise ContractError("V3-D001 queue cell IDs are not unique")
    expected_conditions = {(seed, relation, index) for seed in SEEDS for index in SAMPLING_INDICES for relation in RELATIONS}
    observed = {(row["environment_seed"], row["requested_relation"], row["shared_policy_sampling_seed_index"]) for row in rows}
    if observed != expected_conditions:
        raise ContractError("V3-D001 queue is not the exact 27x2x8 cross-product")
    for offset in range(0, 432, 2):
        block = rows[offset:offset + 2]
        if len({row["matched_stochastic_block_id"] for row in block}) != 1 or [row["execution_order_index_within_matched_stochastic_block"] for row in block] != [0, 1] or {row["requested_relation"] for row in block} != set(RELATIONS):
            raise ContractError("V3-D001 matched block/order changed")
    return Release(manifest_path, queue_path, manifest, amendment, tuple(AuthorizedCell(dict(row)) for row in rows))


def validate_runtime(repo_root: Path, runtime_path: Path, phase_a_gate_path: Path) -> dict[str, Any]:
    runtime_path = Path(runtime_path).resolve()
    if sha256_file(runtime_path) != PHASE_A_RUNTIME_SHA256:
        raise ContractError("V3-D001 requires the exact Phase-A runtime identity")
    runtime = validate_phase_a_runtime_identity(Path(repo_root).resolve(), runtime_path, check_live_repositories=True)
    queue_sha = sha256_file(Path(repo_root).resolve() / "artifacts/vla_wam_shared_v3/phase_a_cells.jsonl")
    validate_phase_a_release_gate(Path(phase_a_gate_path).resolve(), queue_sha256=queue_sha, runtime_identity_sha256=PHASE_A_RUNTIME_SHA256)
    return runtime


def partition_seeds(lane_index: int, lane_count: int) -> tuple[int, ...]:
    if type(lane_count) is not int or not 1 <= lane_count <= len(SEEDS):
        raise ContractError("lane-count must be in 1..27")
    if type(lane_index) is not int or not 0 <= lane_index < lane_count:
        raise ContractError("lane-index must be in 0..lane-count-1")
    return tuple(seed for position, seed in enumerate(SEEDS) if position % lane_count == lane_index)


def cells_for_lane(cells: Sequence[AuthorizedCell], lane_index: int, lane_count: int) -> tuple[AuthorizedCell, ...]:
    selected = set(partition_seeds(lane_index, lane_count))
    output = tuple(cell for cell in cells if cell.environment_seed in selected)
    if len(output) != len(selected) * 16:
        raise ContractError("whole-seed lane partition split or lost V3-D001 cells")
    return output


__all__ = [
    "ACTION_CAP", "ACTION_CHUNK_STEPS", "ACTION_DIM", "ACTION_SPACE", "ARENA",
    "AuthorizedCell", "ContractError", "FROZEN_CHECKPOINT", "FROZEN_CONFIG",
    "FROZEN_OPENPI_COMMIT", "FROZEN_ROBOLAB_COMMIT", "MODEL_ID", "PHASE",
    "PROMPTS", "QUEUE_SHA256", "REGISTRATION_ID", "RELEASE_MANIFEST_SHA256",
    "Release", "SEEDS", "STUDY_ID", "SUCCESS_PREDICATE_ID", "TASKS",
    "canonical_json_bytes", "cells_for_lane", "load_release", "sha256_bytes",
    "sha256_file", "validate_runtime",
]
