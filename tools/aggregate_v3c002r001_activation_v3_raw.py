#!/usr/bin/env python3
"""Atomically aggregate complete activation-v3 R001 raw evidence without analysis.

This is deliberately a structural collector.  It validates completed blocks,
raw/provenance bindings, released lane identities, and excluded infrastructure
attempts, but never reads an outcome field to calculate a statistic or invoke
the result compiler.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (
    ContractError,
    file_binding,
    read_finite_json,
    require,
    sha256_file,
    validate_file_binding,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import (
    REPO_ROOT,
    load_repair,
    validate_assignment,
)


ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/activation_v3"
LANE_SLOTS = tuple(f"repair-lane-{index:02d}" for index in range(8))
RAW_SCHEMA = "vla-wam-shared-v3c002-raw-episode-v1"
MARKER_SCHEMA = "vla-wam-shared-v3c002-completed-block-v1"
PROVENANCE_SCHEMA = "vla-wam-shared-v3c002r001-raw-provenance-v1"
INFRA_SCHEMA = "vla-wam-shared-v3c002-infrastructure-attempt-v1"


def _parse_lane_roots(values: list[str]) -> dict[str, Path]:
    """Parse exactly one immutable raw root for each registered lane slot."""

    roots: dict[str, Path] = {}
    for value in values:
        slot, separator, path = value.partition("=")
        require(separator == "=" and slot in LANE_SLOTS and path, "--lane-raw-root must be repair-lane-NN=PATH")
        require(slot not in roots, f"--lane-raw-root repeats {slot}")
        roots[slot] = Path(path).resolve()
    require(tuple(sorted(roots)) == LANE_SLOTS, "--lane-raw-root must bind exactly eight repair lanes")
    return roots


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        require(line.strip(), f"{label} has a blank line at {number}")
        try:
            value = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ContractError(f"{label} has invalid JSON at line {number}: {exc}") from exc
        require(isinstance(value, dict), f"{label} row {number} is not an object")
        rows.append(value)
    return rows


def _write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _lane_manifests(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    release_path = root / "release_gate.released.json"
    require(release_path.is_file(), "activation-v3 released behavioral gate is missing")
    gate = read_finite_json(release_path)
    require(isinstance(gate, dict), "activation-v3 released behavioral gate is not an object")
    require(
        gate.get("repair_id") == "V3-C002-R001"
        and gate.get("status") == "passed_homogeneous_block_local_behavioral_release"
        and gate.get("passed") is True,
        "activation-v3 release gate changed",
    )
    records = gate.get("lane_manifests")
    require(isinstance(records, list) and len(records) == 8, "activation-v3 release lacks eight lane manifests")
    values: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        binding = validate_file_binding(record, f"activation-v3 released lane manifest {index}")
        path = Path(binding["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        require(path.is_relative_to(root), "activation-v3 released lane manifest is outside activation_v3")
        value = read_finite_json(path)
        require(isinstance(value, dict), f"activation-v3 lane manifest {index} is not an object")
        slot = value.get("lane_slot")
        require(slot in LANE_SLOTS and slot not in values, "activation-v3 lane manifest slots are incomplete or duplicated")
        values[str(slot)] = value
        bindings[str(slot)] = binding
    require(tuple(sorted(values)) == LANE_SLOTS, "activation-v3 released lane slots changed")
    return values, bindings


def _validate_provenance(
    path: Path,
    *,
    marker_raw: Mapping[str, Any],
    cell: Any,
    slot: str,
    lane: Mapping[str, Any],
    lane_binding: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Validate R001 provenance without inspecting any behavioral measurement."""

    require(path.is_file(), f"R001 provenance sidecar is missing: {cell.cell_id}")
    value = read_finite_json(path)
    require(isinstance(value, dict), f"R001 provenance sidecar is not an object: {cell.cell_id}")
    require(
        value.get("schema_version") == PROVENANCE_SCHEMA
        and value.get("repair_id") == "V3-C002-R001"
        and value.get("cell_id") == cell.cell_id
        and value.get("episode_seed") == cell.seed
        and value.get("lane_slot") == slot
        and value.get("lane_id") == lane.get("lane_id")
        and value.get("block_indivisible") is True,
        f"R001 provenance identity changed: {cell.cell_id}",
    )
    parent_raw = validate_file_binding(value.get("parent_raw_episode"), f"R001 provenance parent raw {cell.cell_id}")
    require(parent_raw == marker_raw, f"R001 provenance parent raw differs from completed marker: {cell.cell_id}")
    registration = validate_file_binding(value.get("repair_registration"), f"R001 provenance registration {cell.cell_id}")
    require(Path(registration["path"]).resolve() == (root / "registration.json").resolve(), f"R001 provenance registration is not activation_v3: {cell.cell_id}")
    assignment = validate_file_binding(value.get("assignment_manifest"), f"R001 provenance assignment {cell.cell_id}")
    require(assignment["sha256"] == sha256_file(root / "assignment.jsonl"), f"R001 provenance assignment changed: {cell.cell_id}")
    authorization = validate_file_binding(value.get("authorization_gate"), f"R001 provenance release {cell.cell_id}")
    require(Path(authorization["path"]).resolve() == (root / "release_gate.released.json").resolve(), f"R001 provenance release is not activation_v3: {cell.cell_id}")
    released_lane = validate_file_binding(value.get("released_lane_manifest"), f"R001 provenance lane manifest {cell.cell_id}")
    require(released_lane == lane_binding, f"R001 provenance lane manifest changed: {cell.cell_id}")
    return file_binding(path)


def _validate_marker(
    path: Path,
    *,
    seed: int,
    slot: str,
    lane: Mapping[str, Any],
    lane_binding: Mapping[str, Any],
    cells_by_id: Mapping[str, Any],
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return four structural raw rows and their provenance bindings for one seed."""

    marker = read_finite_json(path)
    require(isinstance(marker, dict), f"completed marker is not an object: {path}")
    require(
        marker.get("schema_version") == MARKER_SCHEMA
        and marker.get("status") == "completed_behavioral_block"
        and marker.get("authorization_mode") == "behavioral"
        and marker.get("episode_seed") == seed,
        f"completed marker identity changed: {path}",
    )
    records = marker.get("raw_episodes")
    require(isinstance(records, list) and len(records) == 4, f"completed marker lacks four raw records: {path}")
    expected_ids = {cell_id for cell_id, cell in cells_by_id.items() if cell.seed == seed}
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        require(isinstance(record, Mapping), f"completed marker raw binding is invalid: {path}")
        cell_id = record.get("cell_id")
        require(isinstance(cell_id, str) and cell_id in expected_ids and cell_id not in seen, f"completed marker cell set changed: {path}")
        raw_binding = validate_file_binding(record, f"completed marker raw {cell_id}")
        raw_rows = _read_jsonl(Path(raw_binding["path"]), f"completed marker raw {cell_id}")
        require(len(raw_rows) == 1, f"completed marker raw must be one row: {cell_id}")
        row = raw_rows[0]
        cell = cells_by_id[cell_id]
        require(
            row.get("schema_version") == RAW_SCHEMA
            and row.get("cell_id") == cell_id
            and row.get("cell_sha256") == cell.row_sha256
            and row.get("authorization_mode") == "behavioral"
            and row.get("excluded_from_behavioral_denominators") is False,
            f"behavioral raw structural identity changed: {cell_id}",
        )
        runtime = row.get("runtime_identity")
        require(isinstance(runtime, Mapping), f"behavioral raw lacks runtime identity: {cell_id}")
        for key in (
            "lane_id", "simulator_pod_uid", "simulator_gpu_uuid", "policy_server_pod_uid",
            "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity",
        ):
            require(runtime.get(key) == lane.get(key), f"behavioral raw differs from released {slot} for {key}: {cell_id}")
        provenance.append(_validate_provenance(
            Path(raw_binding["path"]).with_name("r001_provenance.json"),
            marker_raw=raw_binding,
            cell=cell,
            slot=slot,
            lane=lane,
            lane_binding=lane_binding,
            root=root,
        ))
        rows.append(row)
        seen.add(cell_id)
    require(seen == expected_ids, f"completed marker is not a registered four-cell block: {path}")
    return rows, provenance, file_binding(path)


def _validate_infrastructure(path: Path, *, slot: str, assignment: Mapping[int, str], cells_by_id: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _read_jsonl(path, f"infrastructure ledger {path}") if path.stat().st_size else []
    for row in rows:
        cell_id = row.get("cell_id")
        require(
            row.get("schema_version") == INFRA_SCHEMA
            and row.get("record_type") == "infrastructure_attempt"
            and row.get("infrastructure_status") == "infrastructure_invalid_excluded"
            and row.get("denominator_eligible") is False
            and row.get("authorization_mode") == "behavioral"
            and isinstance(cell_id, str)
            and cell_id in cells_by_id
            and assignment[cells_by_id[cell_id].seed] == slot,
            f"infrastructure record is not an excluded {slot} behavioral attempt",
        )
    return rows


def collect(root: Path, lane_roots: Mapping[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Collect all source evidence in queue order; no aggregate outcomes are computed."""

    root = Path(root).resolve()
    require(root == ROOT.resolve(), "activation-v3 raw aggregation accepts only activation_v3")
    repair, cells = load_repair(registration_path=root / "registration.json", queue_path=root / "queue.jsonl")
    assignment = {row["episode_seed"]: row["lane_slot"] for row in validate_assignment(repair["assignment_manifest"])}
    cells_by_id = {cell.cell_id: cell for cell in cells}
    lane_values, lane_bindings = _lane_manifests(root)
    source_rows: dict[str, dict[str, Any]] = {}
    lane_receipts: dict[str, dict[str, Any]] = {}
    all_infra: list[dict[str, Any]] = []
    for slot in LANE_SLOTS:
        raw_root = Path(lane_roots[slot]).resolve()
        lane = lane_values[slot]
        require(str(raw_root) == lane.get("raw_root"), f"raw root differs from released {slot}")
        marker_root = raw_root / "behavioral"
        require(marker_root.is_dir(), f"behavioral marker root is missing for {slot}")
        assigned_seeds = {seed for seed, assigned_slot in assignment.items() if assigned_slot == slot}
        marker_paths = sorted(marker_root.glob("seed*/completed_block.json"))
        require(len(marker_paths) == len(assigned_seeds), f"completed marker count is incomplete for {slot}")
        marker_by_seed: dict[int, Path] = {}
        lane_raw_bindings: list[dict[str, Any]] = []
        lane_provenance: list[dict[str, Any]] = []
        for marker_path in marker_paths:
            seed_text = marker_path.parent.name.removeprefix("seed")
            require(seed_text.isdigit(), f"completed marker has malformed seed directory: {marker_path}")
            seed = int(seed_text)
            require(seed in assigned_seeds and seed not in marker_by_seed, f"completed marker belongs to wrong lane: {marker_path}")
            rows, sidecars, marker_binding = _validate_marker(
                marker_path,
                seed=seed,
                slot=slot,
                lane=lane,
                lane_binding=lane_bindings[slot],
                cells_by_id=cells_by_id,
                root=root,
            )
            marker_by_seed[seed] = marker_path
            for row in rows:
                cell_id = str(row["cell_id"])
                require(cell_id not in source_rows, f"duplicate behavioral raw cell: {cell_id}")
                source_rows[cell_id] = row
            lane_raw_bindings.extend(
                validate_file_binding(record, f"completed marker raw binding {record.get('cell_id')}")
                for record in read_finite_json(marker_path)["raw_episodes"]
            )
            lane_provenance.extend(sidecars)
        require(set(marker_by_seed) == assigned_seeds, f"completed seed set is incomplete for {slot}")
        require(len(lane_raw_bindings) == len(assigned_seeds) * 4 and len(lane_provenance) == len(lane_raw_bindings), f"raw/provenance coverage is incomplete for {slot}")
        # The frozen runner owns one append-only ledger at the released lane
        # root.  Do not silently miss it by searching for the final aggregate
        # output name, and do not accept arbitrary nested ledgers.
        infra_path = raw_root / "infrastructure_invalid.jsonl"
        require(not infra_path.exists() or infra_path.is_file(), f"infrastructure ledger is not a file for {slot}")
        infra_paths = [infra_path] if infra_path.is_file() else []
        lane_infra: list[dict[str, Any]] = []
        for infra_path in infra_paths:
            lane_infra.extend(_validate_infrastructure(infra_path, slot=slot, assignment=assignment, cells_by_id=cells_by_id))
        all_infra.extend(lane_infra)
        lane_receipts[slot] = {
            "raw_root": str(raw_root),
            "completed_marker_count": len(marker_by_seed),
            "completed_markers": [file_binding(marker_by_seed[seed]) for seed in sorted(marker_by_seed)],
            "raw_episode_count": len(lane_raw_bindings),
            "raw_episode_bindings": lane_raw_bindings,
            "provenance_sidecar_count": len(lane_provenance),
            "provenance_sidecars": lane_provenance,
            "infrastructure_ledger_count": len(infra_paths),
            "infrastructure_ledgers": [file_binding(path) for path in infra_paths],
            "infrastructure_attempt_count": len(lane_infra),
        }
    require(len(source_rows) == 1364 and set(source_rows) == set(cells_by_id), "all 1,364 registered behavioral cells are required")
    ordered_rows = [source_rows[cell.cell_id] for cell in cells]
    receipt = {
        "schema_version": "vla-wam-shared-v3c002r001-activation-v3-raw-aggregation-receipt-v1",
        "status": "complete_structural_raw_aggregation",
        "repair_id": "V3-C002-R001",
        "activation_id": repair.get("activation_id"),
        "structural_only_no_outcome_aggregate": True,
        "registration": file_binding(root / "registration.json"),
        "queue": file_binding(root / "queue.jsonl"),
        "assignment_manifest": file_binding(root / "assignment.jsonl"),
        "release_gate": file_binding(root / "release_gate.released.json"),
        "lane_evidence": lane_receipts,
        "completed_seed_block_count": 341,
        "behavioral_raw_episode_count": 1364,
        "infrastructure_attempt_count_excluded": len(all_infra),
    }
    return ordered_rows, all_infra, receipt


def _write_outputs(root: Path, rows: list[dict[str, Any]], infrastructure: list[dict[str, Any]], receipt: dict[str, Any]) -> dict[str, str]:
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "episodes": raw_dir / "episodes.jsonl",
        "infrastructure_attempts": root / "infrastructure_attempts.jsonl",
        "receipt": raw_dir / "aggregation_receipt.json",
    }
    require(not any(path.exists() for path in targets.values()), "raw aggregation refuses to overwrite or append to an existing output")
    stage = Path(tempfile.mkdtemp(prefix=".v3c002r001-aggregate-", dir=raw_dir))
    installed: list[Path] = []
    try:
        staged = {name: stage / path.name for name, path in targets.items()}
        _write_jsonl_new(staged["episodes"], rows)
        _write_jsonl_new(staged["infrastructure_attempts"], infrastructure)
        for name in ("episodes", "infrastructure_attempts"):
            # ``link`` is atomic and, unlike rename/replace, refuses to clobber
            # an output another process may have created after our initial
            # no-overwrite check.  Stage and targets share the activation PVC.
            os.link(staged[name], targets[name])
            staged[name].unlink()
            installed.append(targets[name])
        completed = dict(receipt)
        completed["combined_outputs"] = {
            "raw_episodes": file_binding(targets["episodes"]),
            "infrastructure_attempts": file_binding(targets["infrastructure_attempts"]),
        }
        _write_json_new(staged["receipt"], completed)
        os.link(staged["receipt"], targets["receipt"])
        staged["receipt"].unlink()
        installed.append(targets["receipt"])
    except BaseException:
        for path in installed:
            if path.exists():
                path.unlink()
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {name: sha256_file(path) for name, path in targets.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--lane-raw-root", action="append", default=[], metavar="REPAIR-LANE-NN=PATH")
    args = parser.parse_args()
    try:
        root = Path(args.root).resolve()
        roots = _parse_lane_roots(args.lane_raw_root)
        rows, infrastructure, receipt = collect(root, roots)
        output_hashes = _write_outputs(root, rows, infrastructure, receipt)
        print(json.dumps({
            "status": "complete_structural_raw_aggregation",
            "behavioral_raw_episode_count": len(rows),
            "infrastructure_attempt_count_excluded": len(infrastructure),
            "output_sha256": output_hashes,
        }, indent=2, sort_keys=True))
    except ContractError as exc:
        raise SystemExit(f"V3-C002-R001 activation-v3 raw aggregation failed: {exc}") from exc


if __name__ == "__main__":
    main()
