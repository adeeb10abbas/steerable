"""Additive, retry-only launcher for the two activation-v3 replacement lanes.

It deliberately reuses the frozen R001 dispatch/provenance writer but replaces
only its admission and shard selection: slots 00/01 each receive their one
registered incomplete seed, never an entire original lane or another slot.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from experiments.v3.phase_c_semantic_equivalence_v3c002 import runner as base_runner
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError, load_cells, read_finite_json, require, sha256_file, validate_file_binding
from experiments.v3.phase_c_semantic_equivalence_v3c002r001 import runner as r001_runner
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import LANE_SCHEMA, validate_assignment


SCHEMA = "vla-wam-shared-v3c002r001-activation-v3-lane-replacement-gate-v1"
REGISTRATION_SCHEMA = "vla-wam-shared-v3c002r001-activation-v3-lane-replacement-registration-v1"
SOURCE_SCHEMA = "vla-wam-shared-v3c002r001-activation-v3-lane-replacement-source-gate-v1"
RETRY = {"repair-lane-00": 12060, "repair-lane-01": 12101}


def _argument(name: str) -> str:
    index = sys.argv.index(name)
    return sys.argv[index + 1]


def _gate(path: Path) -> dict:
    gate = read_finite_json(path)
    require(isinstance(gate, dict) and gate.get("schema_version") == SCHEMA and gate.get("status") == "passed_activation_v3_cluster_termination_lane_replacement" and gate.get("passed") is True, "activation-v3 replacement gate did not pass")
    registration_binding = validate_file_binding(gate.get("replacement_registration"), "replacement registration")
    registration = read_finite_json(Path(registration_binding["path"]))
    require(isinstance(registration, dict) and registration.get("schema_version") == REGISTRATION_SCHEMA and registration.get("status") == "registered_prospective_activation_v3_lane_replacement" and registration.get("replacement_gate_model_requests_before_registration") == 0 and registration.get("replacement_gate_behavioral_episodes_before_registration") == 0, "replacement registration changed")
    source_binding = validate_file_binding(gate.get("source_push_gate"), "replacement source gate")
    source = read_finite_json(Path(source_binding["path"]))
    require(isinstance(source, dict) and source.get("schema_version") == SOURCE_SCHEMA and source.get("status") == "passed_activation_v3_lane_replacement_source_and_registration_pushed" and source.get("passed") is True and source.get("pushed") is True and source.get("replacement_registration_sha256") == registration_binding["sha256"], "replacement source gate changed")
    validate_assignment(gate.get("assignment_manifest"))
    replacements = gate.get("replacements")
    require(isinstance(replacements, dict) and set(replacements) == set(RETRY) and gate.get("retry_seed_by_lane") == RETRY and gate.get("no_cross_lane_failover") is True and gate.get("completed_blocks_never_rerun") is True, "replacement retry scope changed")
    return gate


def require_released_replacement_gate(*, registration_path: Path, queue_path: Path, release_gate_path: Path):
    parent, cells = load_cells(registration_path=registration_path, queue_path=queue_path)
    gate = _gate(release_gate_path)
    require(gate.get("queue", {}).get("sha256") == sha256_file(queue_path), "replacement queue changed")
    require(gate.get("activation_v3_registration", {}).get("sha256") == sha256_file(registration_path), "replacement altered the activation-v3 repair registration")
    return parent, cells, gate


def _retry_only(cells, *, shard_index: int, shard_count: int):
    require(shard_count == 8 and shard_index in (0, 1), "replacement launch allows only original slots 00 and 01")
    gate = _gate(Path(_argument("--authorization-gate")).resolve())
    slot = f"repair-lane-{shard_index:02d}"; seed = RETRY[slot]
    replacement = gate["replacements"][slot]
    require(replacement.get("authorized_retry_seed") == seed and replacement.get("authorized_retry_cell_count") == 4 and seed not in replacement.get("completed_seed_blocks_forbidden", []), "replacement retry attempts a completed or wrong block")
    records = [record for record in gate.get("lane_manifests", []) if read_finite_json(Path(validate_file_binding(record, "replacement lane manifest")["path"])).get("lane_slot") == slot]
    require(len(records) == 1, "replacement lane manifest is not unique")
    lane = read_finite_json(Path(validate_file_binding(records[0], "replacement lane manifest")["path"]))
    require(isinstance(lane, dict) and lane.get("schema_version") == LANE_SCHEMA and lane.get("assigned_seed_blocks") and seed in lane["assigned_seed_blocks"], "replacement lane cannot own retry seed")
    selected = [cell for cell in cells if cell.seed == seed]
    require(len(selected) == 4, "replacement retry does not contain exactly four frozen cells")
    return selected


def _replacement_attempt_root(root: Path, seed: int) -> Path:
    """Only attempt002 may execute: retained attempt001 is never resumed."""
    require(seed in set(RETRY.values()), "replacement runner may not choose another seed")
    attempt_root = Path(root) / f"seed{seed}"
    retained, replacement = attempt_root / "attempt001", attempt_root / "attempt002"
    require(retained.is_dir(), "replacement requires retained partial attempt001")
    require(not replacement.exists(), "replacement attempt002 already exists; refusing retry/resume")
    return replacement


def _replacement_preflight(block, args) -> Path:
    require(args.authorization_mode == "behavioral" and len(block) == 4 and len({cell.seed for cell in block}) == 1, "replacement dispatch is not one complete behavioral block")
    slot = f"repair-lane-{args.shard_index:02d}"; seed = block[0].seed
    require(slot in RETRY and seed == RETRY[slot], "replacement dispatch seed/slot changed")
    gate = _gate(Path(args.authorization_gate).resolve())
    replacement = gate["replacements"][slot]
    partial = read_finite_json(Path(validate_file_binding(replacement.get("partial_target_raw_rehash"), "replacement partial raw rehash")["path"]))
    files = partial.get("attempt_files") if isinstance(partial, dict) else None
    require(isinstance(files, list) and files, "replacement partial raw rehash lacks retained attempt files")
    retained = Path(args.raw_root) / "behavioral" / f"seed{seed}" / "attempt001"
    marker = Path(args.raw_root) / "behavioral" / f"seed{seed}" / "completed_block.json"
    next_attempt = retained.with_name("attempt002")
    require(retained.is_dir() and not marker.exists() and not next_attempt.exists(), "replacement retry may not resume a partial cell or overwrite a completed/replacement block")
    for record in files:
        retained_file = Path(validate_file_binding(record, "replacement retained partial file")["path"]).resolve()
        require(retained_file.is_relative_to(retained.resolve()), "replacement partial raw rehash is not bound to retained attempt001")
    return next_attempt


_r001_dispatch_with_provenance = base_runner._dispatch_block


def _dispatch_replacement_block(*, block, args, registration_sha, queue_sha):
    expected_attempt = _replacement_preflight(block, args)
    _r001_dispatch_with_provenance(block=block, args=args, registration_sha=registration_sha, queue_sha=queue_sha)
    marker_path = Path(args.raw_root) / "behavioral" / f"seed{block[0].seed}" / "completed_block.json"
    marker = read_finite_json(marker_path)
    require(isinstance(marker, dict) and marker.get("status") == "completed_behavioral_block" and marker.get("attempt_root") == str(expected_attempt.resolve()), "replacement completion marker did not bind attempt002")
    raws = marker.get("raw_episodes")
    require(isinstance(raws, list) and len(raws) == 4, "replacement completion marker lacks four re-executed cells")
    for record in raws:
        raw = Path(str(record.get("path", ""))).resolve()
        require(raw.is_relative_to(expected_attempt.resolve()), "replacement completion reuses a partial-attempt raw cell")


# R001 has already installed its retained provenance writer into the base runner.
base_runner.require_released_gate = require_released_replacement_gate
base_runner.grouped_shard = _retry_only
base_runner._next_attempt_root = _replacement_attempt_root
base_runner._dispatch_block = _dispatch_replacement_block


def main() -> None:
    base_runner.main()


if __name__ == "__main__":
    main()
