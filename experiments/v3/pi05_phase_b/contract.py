#!/usr/bin/env python3
"""Hash-bound V3-B002 design and release loading.

The B002 queue is a checkpoint substitution over the exact V3-B001 matched
blocks.  Seeds, physical fixtures, prompt bytes, and within-seed execution
positions are inherited from the committed B001 queue; only model/runtime
identity and cell IDs change.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STUDY_ID = "vla_wam_language_steerability_v3"
AMENDMENT_ID = "V3-B002"
PHASE = "B_confound_ablation"
MODEL_ID = "pi05_current_stack_droid"
ARENA = "droid_robolab"
FACTOR = "movable_object_center_position_reflection_about_robot_sagittal_plane"
SEEDS = tuple(range(9400, 9427))
ARMS = ("control", "position_mirrored")
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
SUCCESS_PREDICATE_ID = "v2_frozen_droid_robolab_release_inside_45deg_requested_relation"

OPENPI_COMMIT = "c23745b5ad24e98f66967ea795a07b2588ed6c79"
ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
OPENPI_CONFIG = "pi05_droid_jointpos_polaris"
CHECKPOINT_MANIFEST_SHA256 = "f5a56d9565f9381ccdeeaa165b0495dab6d17a81836cc7b01c5fbc6ab89e74ca"
ACTION_CHUNK_STEPS = 15
ACTION_DIM = 8
ACTION_CAP = 450
ACTION_SPACE = "joint_position_8d"

B001_RELATIVE = Path("artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001")
B001_MANIFEST = B001_RELATIVE / "nano_mirror_v3b001_manifest.json"
B001_CELLS = B001_RELATIVE / "nano_mirror_v3b001_cells.jsonl"
B001_AMENDMENT = B001_RELATIVE / "post_result_nano_mirror_v3b001_amendment.json"
B001_MANIFEST_SHA256 = "5c82268739feb41281435a51dcd848b575218cd9fbe5839d9ad130d1a7888830"
B001_CELLS_SHA256 = "018b8b6ae76ac46f2f89eef83c4b16d7a4ff3d1ff15d91527b96fb56b5432c5a"
B001_AMENDMENT_SHA256 = "9d88c29733fa3b24a154977bc25d04d2d77df5be59e3213f0c3a6cfbe3edc6a0"
B001_CANDIDATE_SHA256 = "e1799b815da41f9a08a4000a360c4958003269fed27e2abe75b273519e4d1c88"
B001_FIXTURE_SHA256 = {
    "control": "c5f3c667eda6f512b9e33beb5f7abc91700404feafa8b22279103b809dd238cd",
    "position_mirrored": "5461ab070bc801cff95d2a7437ca2af8bef2e1fac420622dcd7dd4fea1eb6b21",
}

AMENDMENT_SCHEMA = "vla-wam-shared-v3b-pi05-mirror-amendment-v1"
CELL_SCHEMA = "vla-wam-shared-v3b-pi05-mirror-cell-v1"
MANIFEST_SCHEMA = "vla-wam-shared-v3b-pi05-mirror-manifest-v1"
RUNTIME_SCHEMA = "vla-wam-shared-v3b-pi05-runtime-identity-v1"
GATE_SCHEMA = "vla-wam-shared-v3b-pi05-release-gate-v1"


class ContractError(ValueError):
    """Raised before any model request when the registered contract differs."""


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return (payload + ("\n" if newline else "")).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ContractError(f"blank JSONL row at {path}:{number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ContractError(f"non-object JSONL row at {path}:{number}")
        rows.append(row)
    return rows


def load_b001_sources(repo_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the immutable B001 fixture/order source and fail on any drift."""

    root = Path(repo_root).resolve()
    paths = {
        "manifest": root / B001_MANIFEST,
        "cells": root / B001_CELLS,
        "amendment": root / B001_AMENDMENT,
    }
    expected = {
        "manifest": B001_MANIFEST_SHA256,
        "cells": B001_CELLS_SHA256,
        "amendment": B001_AMENDMENT_SHA256,
    }
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected[name]:
            raise ContractError(f"immutable Nano V3-B001 {name} binding changed")
    manifest = _load_json(paths["manifest"])
    if manifest.get("files", {}).get("cells", {}).get("sha256") != B001_CELLS_SHA256:
        raise ContractError("B001 release manifest no longer binds the exact source queue")
    amendment = _load_json(paths["amendment"])
    rows = _load_jsonl(paths["cells"])
    if len(rows) != 108 or amendment.get("design", {}).get("seeds") != list(SEEDS):
        raise ContractError("B001 source is not the exact 27-seed four-cell design")
    fixtures = amendment.get("fixtures", {})
    for arm in ARMS:
        # The fixture hash is the pretty release serialization used by B001.
        payload = (json.dumps(fixtures[arm], allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
        if sha256_bytes(payload) != B001_FIXTURE_SHA256[arm]:
            raise ContractError(f"B001 {arm} fixture bytes changed")
    return amendment, rows


def expected_cells(repo_root: Path) -> tuple[dict[str, Any], ...]:
    """Construct the only allowed B002 cells without writing a release."""

    _, source_rows = load_b001_sources(repo_root)
    output: list[dict[str, Any]] = []
    for source in source_rows:
        seed = source["environment_seed"]
        arm = source["arm"]
        relation = source["relation"]
        if seed not in SEEDS or arm not in ARMS or relation not in RELATIONS:
            raise ContractError("B001 source queue contains an unexpected condition")
        source_id = source["cell_id"]
        output.append(
            {
                "schema_version": CELL_SCHEMA,
                "study_id": STUDY_ID,
                "amendment_id": AMENDMENT_ID,
                "phase": PHASE,
                "arena": ARENA,
                "model_id": MODEL_ID,
                "cell_id": f"v3b002:pi05:seed{seed}:{arm}:{relation}",
                "matched_block_id": f"v3b002:pi05:seed{seed}",
                "arm": arm,
                "relation": relation,
                "environment_seed": seed,
                "sampling_seed": seed,
                "execution_order_index_within_seed": source[
                    "execution_order_index_within_seed"
                ],
                "source_v3b001_cell_id": source_id,
                "source_v3b001_queue_sha256": B001_CELLS_SHA256,
                "source_v3b001_randomization_key_sha256": source["randomization_key_sha256"],
                "source_execution_order_index_within_seed": source[
                    "execution_order_index_within_seed"
                ],
                "factor": FACTOR,
                "fixture_id": f"v3b001_nano_{arm}",
                "fixture_sha256": source["fixture_sha256"],
                "prompt_family": "direct_command",
                "prompt": PROMPTS[relation],
                "prompt_sha256": sha256_bytes(PROMPTS[relation].encode("utf-8")),
                "success_predicate_id": SUCCESS_PREDICATE_ID,
                "runtime_identity_requirement": {
                    "openpi_commit": OPENPI_COMMIT,
                    "robolab_commit": ROBOLAB_COMMIT,
                    "openpi_config": OPENPI_CONFIG,
                    "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
                    "open_loop_horizon": ACTION_CHUNK_STEPS,
                    "action_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
                    "clean_external_repositories_required": True,
                },
                "required_raw_outputs": [
                    "viewport_video",
                    "executed_action_trace",
                    "raw_behavioral_episode_jsonl",
                ],
                "missing_future_policy": "action_only_interface_not_applicable_never_zero",
                "technical_invalidity_policy": (
                    "retain in a separate stream and repair only this identical registered cell"
                ),
                "valid_failure_policy": "retain every valid behavioral failure in all full-sample analyses",
            }
        )
    output.sort(key=lambda row: (row["environment_seed"], row["execution_order_index_within_seed"]))
    return tuple(output)


@dataclass(frozen=True)
class AuthorizedCell:
    row: dict[str, Any]

    @property
    def cell_id(self) -> str:
        return str(self.row["cell_id"])

    @property
    def seed(self) -> int:
        return int(self.row["environment_seed"])

    @property
    def arm(self) -> str:
        return str(self.row["arm"])

    @property
    def relation(self) -> str:
        return str(self.row["relation"])


@dataclass(frozen=True)
class ReleaseBundle:
    manifest_path: Path
    manifest: dict[str, Any]
    amendment: dict[str, Any]
    cells: tuple[AuthorizedCell, ...]
    manifest_sha256: str
    amendment_sha256: str
    cells_sha256: str

    @property
    def by_cell_id(self) -> dict[str, AuthorizedCell]:
        return {cell.cell_id: cell for cell in self.cells}

    def cell(self, cell_id: str) -> AuthorizedCell:
        try:
            return self.by_cell_id[cell_id]
        except KeyError as exc:
            raise ContractError(f"cell is not in the exact B002 release: {cell_id}") from exc

    def release_fingerprint(self, cell: AuthorizedCell) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "study_id": STUDY_ID,
                    "amendment_id": AMENDMENT_ID,
                    "release_manifest_sha256": self.manifest_sha256,
                    "amendment_sha256": self.amendment_sha256,
                    "cells_sha256": self.cells_sha256,
                    "cell": cell.row,
                }
            )
        )


def _file_binding(base: Path, record: Mapping[str, Any], label: str) -> Path:
    name = record.get("path")
    if not isinstance(name, str) or Path(name).name != name:
        raise ContractError(f"{label}.path must name a sibling file")
    path = base / name
    if (
        not path.is_file()
        or record.get("bytes") != path.stat().st_size
        or record.get("sha256") != sha256_file(path)
    ):
        raise ContractError(f"{label} hash/size binding changed")
    return path


def load_release_bundle(
    repo_root: Path, manifest_path: Path, *, expected_manifest_sha256: str
) -> ReleaseBundle:
    path = Path(manifest_path).resolve()
    if sha256_file(path) != expected_manifest_sha256:
        raise ContractError("B002 release manifest does not match the pinned digest")
    manifest = _load_json(path)
    if any(
        manifest.get(key) != value
        for key, value in {
            "schema_version": MANIFEST_SCHEMA,
            "study_id": STUDY_ID,
            "amendment_id": AMENDMENT_ID,
            "status": "hash_bound_registered_not_behaviorally_released",
        }.items()
    ):
        raise ContractError("unexpected B002 release manifest identity")
    files = manifest.get("files", {})
    amendment_path = _file_binding(path.parent, files.get("amendment", {}), "amendment")
    cells_path = _file_binding(path.parent, files.get("cells", {}), "cells")
    amendment = _load_json(amendment_path)
    rows = _load_jsonl(cells_path)
    expected = expected_cells(repo_root)
    if len(rows) != 108 or len(expected) != 108:
        raise ContractError("B002 requires exactly 108 cells")
    # Registration-specific fields may extend each row.  Every scientific and
    # execution field constructed from B001 must match exactly.
    for observed, wanted in zip(rows, expected):
        for key, value in wanted.items():
            if observed.get(key) != value:
                raise ContractError(f"B002 released row mismatch: {wanted['cell_id']} {key}")
    if (
        amendment.get("schema_version") != AMENDMENT_SCHEMA
        or amendment.get("status")
        != "frozen_before_any_v3b002_model_request_or_behavioral_episode"
        or amendment.get("design", {}).get("exact_prompts") != PROMPTS
    ):
        raise ContractError("B002 prompt bytes changed")
    for row in rows:
        if row.get("fixture_sha256") != B001_FIXTURE_SHA256[row["arm"]]:
            raise ContractError("B002 row does not bind the exact B001 fixture")
    return ReleaseBundle(
        manifest_path=path,
        manifest=manifest,
        amendment=amendment,
        cells=tuple(AuthorizedCell(dict(row)) for row in rows),
        manifest_sha256=expected_manifest_sha256,
        amendment_sha256=sha256_file(amendment_path),
        cells_sha256=sha256_file(cells_path),
    )


def partition_seeds(*, lane_index: int, lane_count: int) -> tuple[int, ...]:
    """Partition work without ever splitting a four-cell matched seed block."""

    if type(lane_count) is not int or not 1 <= lane_count <= len(SEEDS):
        raise ContractError("lane_count must be in 1..27")
    if type(lane_index) is not int or not 0 <= lane_index < lane_count:
        raise ContractError("lane_index must be in 0..lane_count-1")
    return tuple(seed for position, seed in enumerate(SEEDS) if position % lane_count == lane_index)


def cells_for_lane(
    cells: Sequence[AuthorizedCell], *, lane_index: int, lane_count: int
) -> tuple[AuthorizedCell, ...]:
    selected = set(partition_seeds(lane_index=lane_index, lane_count=lane_count))
    output = tuple(cell for cell in cells if cell.seed in selected)
    for seed in selected:
        block = [cell for cell in output if cell.seed == seed]
        if len(block) != 4 or {(cell.arm, cell.relation) for cell in block} != {
            (arm, relation) for arm in ARMS for relation in RELATIONS
        }:
            raise ContractError(f"lane partition split or lost seed block {seed}")
    return output


def action_paths_from_trace(trace: Mapping[str, Any]) -> tuple[Path, Path]:
    return (
        Path(str(trace["executed_actions"]["path"])),
        Path(str(trace["returned_action_chunks"]["path"])),
    )
