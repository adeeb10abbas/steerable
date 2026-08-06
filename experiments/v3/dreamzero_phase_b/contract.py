#!/usr/bin/env python3
"""Hash-bound loading for the registered DreamZero V3-B003 cohort."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


STUDY_ID = "vla_wam_language_steerability_v3"
AMENDMENT_ID = "V3-B003"
MODEL_ID = "dreamzero_droid_action_cfg"
IDENTITY_BINDING = "V2-A015:dreamzero_action_cfg_s2"
SEEDS = tuple(range(9400, 9427))
ARMS = ("control", "position_mirrored")
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
AMENDMENT_RELATIVE = Path(
    "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/"
    "post_result_dreamzero_mirror_v3b003_amendment.json"
)
CELLS_RELATIVE = Path(
    "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/"
    "dreamzero_mirror_v3b003_cells.jsonl"
)
MANIFEST_RELATIVE = Path(
    "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/"
    "dreamzero_mirror_v3b003_manifest.json"
)
EXPECTED_SHA256 = {
    "amendment": "ba22681ae4d7f748e375617617d9e130e6f1bd5bc0af1e7a995365b145a470fc",
    "cells": "a6d0f0a5d4c7cdfa5d3de95d44d7b11f42750a76a603ff8c2e44848e34b8f70d",
    "manifest": "efe50df701193e48b981c025ea3b4d27a80e3bdf83216e38a98a63e27061cb23",
}
FIXTURE_CANDIDATE_SHA256 = (
    "e1799b815da41f9a08a4000a360c4958003269fed27e2abe75b273519e4d1c88"
)
RELEASE_GATE_SCHEMA = "vla-wam-shared-v3b-dreamzero-release-gate-v1"


class ContractError(ValueError):
    """Raised before inference when a registered binding differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


@dataclass(frozen=True)
class Cell:
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


def load_cells(repo_root: Path) -> tuple[Cell, ...]:
    root = Path(repo_root).resolve()
    paths = {
        "amendment": root / AMENDMENT_RELATIVE,
        "cells": root / CELLS_RELATIVE,
        "manifest": root / MANIFEST_RELATIVE,
    }
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != EXPECTED_SHA256[name]:
            raise ContractError(f"DreamZero V3-B003 {name} binding changed")
    manifest = load_object(paths["manifest"])
    if (
        manifest.get("status") != "hash_bound_registered_not_behaviorally_released"
        or manifest.get("counts", {}).get("registered_behavioral_cells") != 108
        or manifest.get("files", {}).get("cells", {}).get("sha256")
        != EXPECTED_SHA256["cells"]
    ):
        raise ContractError("DreamZero V3-B003 manifest contract changed")
    rows = []
    for number, line in enumerate(paths["cells"].read_text().splitlines(), 1):
        if not line.strip():
            raise ContractError(f"blank V3-B003 cell row {number}")
        row = json.loads(line)
        if (
            row.get("schema_version")
            != "vla-wam-shared-v3b-dreamzero-mirror-cell-v1"
            or row.get("amendment_id") != AMENDMENT_ID
            or row.get("model_id") != MODEL_ID
            or row.get("environment_seed") not in SEEDS
            or row.get("registered_sampling_seed_label") != row.get("environment_seed")
            or row.get("arm") not in ARMS
            or row.get("relation") not in RELATIONS
            or row.get("prompt") != PROMPTS[row["relation"]]
            or row.get("effective_model_noise_seed") != 1140
        ):
            raise ContractError(f"invalid V3-B003 registered cell row {number}")
        rows.append(Cell(row))
    if len(rows) != 108 or len({row.cell_id for row in rows}) != 108:
        raise ContractError("DreamZero V3-B003 must contain 108 unique cells")
    return tuple(rows)


def load_cell(repo_root: Path, cell_id: str) -> Cell:
    matches = [cell for cell in load_cells(repo_root) if cell.cell_id == cell_id]
    if len(matches) != 1:
        raise ContractError(f"cell is not uniquely registered: {cell_id}")
    return matches[0]


def validate_release_gate(
    path: Path,
    *,
    repo_root: Path,
    runtime_identity: Path,
    lane_pod_uid: str,
    lane_gpu_uuid: str,
) -> dict[str, Any]:
    gate = load_object(path)
    expected = {
        "schema_version": RELEASE_GATE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "identity_binding": IDENTITY_BINDING,
        "manifest_sha256": EXPECTED_SHA256["manifest"],
        "cells_sha256": EXPECTED_SHA256["cells"],
        "runtime_identity_sha256": sha256_file(runtime_identity),
        "fixed_observation_release_passed": True,
        "all_model_blind_lanes_passed": True,
        "behavioral_release": True,
        "model_request_count_before_release": 3,
        "behavioral_episode_count_before_release": 0,
    }
    for key, wanted in expected.items():
        if gate.get(key) != wanted:
            raise ContractError(f"DreamZero V3-B003 release mismatch for {key}")
    lanes = gate.get("model_blind_lanes", [])
    matches = [
        lane for lane in lanes
        if lane.get("pod_uid") == lane_pod_uid and lane.get("gpu_uuid") == lane_gpu_uuid
    ]
    if len(matches) != 1 or matches[0].get("passed") is not True:
        raise ContractError("simulator lane is absent from the passed V3-B003 gate")
    for entry in (gate.get("server_contract"), gate.get("fixed_observation_probe"), *lanes):
        artifact = Path(str(entry.get("path", "")))
        if not artifact.is_file() or entry.get("sha256") != sha256_file(artifact):
            raise ContractError("V3-B003 release artifact path/hash changed")
    return gate
