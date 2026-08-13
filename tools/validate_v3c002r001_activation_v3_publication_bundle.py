#!/usr/bin/env python3
"""Additively validate a complete V3-C002-R001 activation-v3 publication bundle.

This validator deliberately has no authority over C002, R001, or the frozen
activation-v1--v3 artifacts.  It is a final-bundle check for the only allowed
result location: ``activation_v3``.  Until a result bundle has been compiled,
it reports a pending state; once any final-result file exists, every expected
file and every one of the eight lane's raw/provenance/infrastructure records
must be present and hash-bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (
    ContractError,
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


def final_paths(root: Path) -> dict[str, Path]:
    """Return every final-result file, all rooted in one activation directory."""

    root = Path(root).resolve()
    results = root / "results"
    return {
        "raw_episodes": root / "raw/episodes.jsonl",
        "aggregation_receipt": root / "raw/aggregation_receipt.json",
        "infrastructure_attempts": root / "infrastructure_attempts.jsonl",
        "episodes": results / "episodes.jsonl",
        "pairs": results / "pairs.jsonl",
        "results": results / "results.json",
        "decision_memo": results / "DECISION_MEMO.md",
        "evidence_manifest": results / "evidence_manifest.json",
        "manuscript_insert": root / "MANUSCRIPT_INSERT.md",
    }


def result_bundle_present(root: Path) -> bool:
    """Fail closed when compiled results exist without the complete bundle.

    ``raw/episodes.jsonl`` and the infrastructure ledger may legitimately be
    assembled during a read-only progress audit.  They become final-publication
    inputs only once any compiled result product exists.
    """

    paths = final_paths(root)
    compiled_names = ("episodes", "pairs", "results", "decision_memo", "evidence_manifest", "manuscript_insert")
    compiled_present = {name: paths[name].exists() for name in compiled_names}
    if not any(compiled_present.values()):
        return False
    present = {name: path.exists() for name, path in paths.items()}
    missing = [name for name, exists in present.items() if not exists]
    require(not missing, f"partial activation-v3 result bundle: missing {', '.join(missing)}")
    for name, path in paths.items():
        require(path.is_file(), f"activation-v3 result {name} is not a regular file")
    return True


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        require(line.strip(), f"{label} has a blank line at {number}")
        try:
            value = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ContractError(f"{label} has invalid JSON at {number}: {exc}") from exc
        require(isinstance(value, dict), f"{label} row {number} is not an object")
        rows.append(value)
    return rows


def _binding_at(record: Any, expected: Path, label: str) -> dict[str, Any]:
    binding = validate_file_binding(record, label)
    require(Path(binding["path"]).resolve() == expected.resolve(), f"{label} is not rooted in activation_v3")
    return binding


def _lane_manifests(root: Path) -> dict[str, dict[str, Any]]:
    gate_path = root / "release_gate.released.json"
    require(gate_path.is_file(), "activation-v3 released gate is missing")
    gate = read_finite_json(gate_path)
    require(isinstance(gate, dict), "activation-v3 released gate is not an object")
    require(
        gate.get("repair_id") == "V3-C002-R001"
        and gate.get("status") == "passed_homogeneous_block_local_behavioral_release"
        and gate.get("passed") is True,
        "activation-v3 released gate changed",
    )
    bindings = gate.get("lane_manifests")
    require(isinstance(bindings, list) and len(bindings) == len(LANE_SLOTS), "activation-v3 release lacks eight lane manifests")
    values: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(bindings):
        binding = validate_file_binding(record, f"activation-v3 lane manifest {index}")
        path = Path(binding["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        require(path.is_relative_to(root), "activation-v3 lane manifest is outside activation_v3")
        value = read_finite_json(path)
        require(isinstance(value, dict), f"activation-v3 lane manifest {index} is not an object")
        slot = value.get("lane_slot")
        require(slot in LANE_SLOTS and slot not in values, "activation-v3 lane slots are incomplete or duplicated")
        values[str(slot)] = value
    require(tuple(sorted(values)) == LANE_SLOTS, "activation-v3 lane manifest set changed")
    return values


def _slot_path_arguments(values: list[str], *, flag: str) -> dict[str, Path]:
    """Parse exactly one ``repair-lane-NN=path`` argument per release lane."""

    parsed: dict[str, Path] = {}
    for value in values:
        slot, separator, raw_path = value.partition("=")
        require(separator == "=" and slot in LANE_SLOTS and raw_path, f"{flag} must be repair-lane-NN=path")
        require(slot not in parsed, f"{flag} repeats {slot}")
        parsed[slot] = Path(raw_path).resolve()
    require(tuple(sorted(parsed)) == LANE_SLOTS, f"{flag} must bind all eight repair lanes")
    return parsed


def _read_live_pid(record_path: Path, *, slot: str, lane_id: str, raw_root: Path) -> int:
    """Read a launch-written, non-outcome PID record for one lane.

    A process lives in its own simulator pod, so a controller cannot reliably
    probe it with ``kill(0)``.  The progress contract instead requires the
    lane's launch-side record to bind that pod's positive PID, lane identity,
    and raw root.  This function intentionally performs no process control.
    """

    require(record_path.is_file(), f"activation-v3 live PID record is missing for {slot}")
    value = read_finite_json(record_path)
    require(isinstance(value, dict), f"activation-v3 live PID record is not an object for {slot}")
    require(
        value.get("lane_slot") == slot
        and value.get("lane_id") == lane_id
        and value.get("raw_root") == str(raw_root),
        f"activation-v3 live PID record identity changed for {slot}",
    )
    pid = value.get("pid")
    require(type(pid) is int and pid > 0, f"activation-v3 live PID is invalid for {slot}")
    return pid


def _validate_live_marker(
    marker_path: Path,
    *,
    slot: str,
    lane: Mapping[str, Any],
    expected_cells: Mapping[str, Any],
    expected_seed: int,
) -> set[str]:
    """Validate one complete four-cell behavioral block without reading outcomes."""

    marker = read_finite_json(marker_path)
    require(isinstance(marker, dict), f"activation-v3 completed marker is not an object: {marker_path}")
    require(
        marker.get("schema_version") == "vla-wam-shared-v3c002-completed-block-v1"
        and marker.get("status") == "completed_behavioral_block"
        and marker.get("authorization_mode") == "behavioral"
        and marker.get("episode_seed") == expected_seed,
        f"activation-v3 completed marker identity changed: {marker_path}",
    )
    records = marker.get("raw_episodes")
    require(isinstance(records, list) and len(records) == 4, f"activation-v3 completed marker lacks four raw episodes: {marker_path}")
    expected = {cell_id for cell_id, cell in expected_cells.items() if cell.seed == expected_seed}
    seen: set[str] = set()
    for record in records:
        require(isinstance(record, Mapping), f"activation-v3 completed marker raw binding is invalid: {marker_path}")
        cell_id = record.get("cell_id")
        require(isinstance(cell_id, str) and cell_id in expected and cell_id not in seen, f"activation-v3 completed marker cell set changed: {marker_path}")
        binding = validate_file_binding(record, f"activation-v3 completed raw episode {cell_id}")
        raw = _read_jsonl(Path(binding["path"]), f"activation-v3 raw episode {cell_id}")
        require(len(raw) == 1, f"activation-v3 raw episode record count changed: {cell_id}")
        row = raw[0]
        require(
            row.get("schema_version") == "vla-wam-shared-v3c002-raw-episode-v1"
            and row.get("cell_id") == cell_id
            and row.get("authorization_mode") == "behavioral"
            and row.get("excluded_from_behavioral_denominators") is False,
            f"activation-v3 raw episode structural binding changed: {cell_id}",
        )
        runtime = row.get("runtime_identity")
        require(isinstance(runtime, Mapping), f"activation-v3 raw episode lacks runtime identity: {cell_id}")
        for key in (
            "lane_id", "simulator_pod_uid", "simulator_gpu_uuid", "policy_server_pod_uid",
            "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity",
        ):
            require(runtime.get(key) == lane.get(key), f"activation-v3 raw episode lane identity changed for {cell_id}: {key}")
        raw_artifacts = row.get("raw_artifacts")
        require(isinstance(raw_artifacts, Mapping), f"activation-v3 raw episode lacks artifact bindings: {cell_id}")
        request_stream = validate_file_binding(raw_artifacts.get("raw_episode_jsonl"), f"activation-v3 raw request stream {cell_id}")
        sidecar_path = Path(request_stream["path"]).with_name("r001_provenance.json")
        provenance = read_finite_json(sidecar_path)
        require(isinstance(provenance, dict), f"activation-v3 raw provenance is not an object: {cell_id}")
        require(
            provenance.get("schema_version") == "vla-wam-shared-v3c002r001-raw-provenance-v1"
            and provenance.get("repair_id") == "V3-C002-R001"
            and provenance.get("cell_id") == cell_id
            and provenance.get("episode_seed") == expected_seed
            and provenance.get("lane_slot") == slot
            and provenance.get("lane_id") == lane.get("lane_id"),
            f"activation-v3 raw provenance identity changed: {cell_id}",
        )
        parent_raw = validate_file_binding(provenance.get("parent_raw_episode"), f"activation-v3 provenance parent raw {cell_id}")
        parent_rows = _read_jsonl(Path(parent_raw["path"]), f"activation-v3 provenance parent raw {cell_id}")
        require(
            len(parent_rows) == 1
            and parent_rows[0].get("cell_id") == cell_id
            and parent_rows[0] == row,
            f"activation-v3 provenance parent raw changed: {cell_id}",
        )
        seen.add(cell_id)
    require(seen == expected, f"activation-v3 completed block is not the registered four-cell block: {marker_path}")
    return seen


def validate_progress(
    root: Path,
    *,
    lane_roots: Mapping[str, Path],
    live_pid_records: Mapping[str, Path],
    infrastructure_ledgers: Mapping[str, Path],
) -> dict[str, Any]:
    """Read-only per-lane progress validation; never compiles or analyzes results."""

    root = Path(root).resolve()
    require(root == ROOT.resolve(), "activation-v3 progress validator accepts only activation_v3")
    repair, cells = load_repair(registration_path=root / "registration.json", queue_path=root / "queue.jsonl")
    assignment = {row["episode_seed"]: row["lane_slot"] for row in validate_assignment(repair["assignment_manifest"])}
    cells_by_id = {cell.cell_id: cell for cell in cells}
    lanes = _lane_manifests(root)
    progress: dict[str, dict[str, Any]] = {}
    total_completed_cells = 0
    for slot in LANE_SLOTS:
        lane = lanes[slot]
        lane_root = Path(lane_roots[slot]).resolve()
        require(str(lane_root) == lane.get("raw_root"), f"activation-v3 progress root differs from released {slot}")
        pid = _read_live_pid(live_pid_records[slot], slot=slot, lane_id=str(lane["lane_id"]), raw_root=lane_root)
        assigned_seeds = {seed for seed, assigned_slot in assignment.items() if assigned_slot == slot}
        marker_root = lane_root / "behavioral"
        completed: dict[int, set[str]] = {}
        if marker_root.is_dir():
            for marker_path in sorted(marker_root.glob("seed*/completed_block.json")):
                seed_text = marker_path.parent.name.removeprefix("seed")
                require(seed_text.isdigit(), f"activation-v3 completed marker has malformed seed directory: {marker_path}")
                seed = int(seed_text)
                require(seed in assigned_seeds and seed not in completed, f"activation-v3 completed marker belongs to the wrong lane: {marker_path}")
                completed[seed] = _validate_live_marker(
                    marker_path,
                    slot=slot,
                    lane=lane,
                    expected_cells=cells_by_id,
                    expected_seed=seed,
                )
        ledger_path = Path(infrastructure_ledgers[slot]).resolve()
        require(not ledger_path.exists() or ledger_path.is_file(), f"activation-v3 infrastructure ledger is not a file for {slot}")
        infra_rows = (
            _read_jsonl(ledger_path, f"activation-v3 infrastructure ledger {slot}")
            if ledger_path.is_file() and ledger_path.stat().st_size
            else []
        )
        for row in infra_rows:
            cell_id = row.get("cell_id")
            require(
                row.get("schema_version") == "vla-wam-shared-v3c002-infrastructure-attempt-v1"
                and row.get("record_type") == "infrastructure_attempt"
                and row.get("infrastructure_status") == "infrastructure_invalid_excluded"
                and row.get("denominator_eligible") is False
                and row.get("authorization_mode") == "behavioral"
                and isinstance(cell_id, str)
                and cell_id in cells_by_id
                and assignment[cells_by_id[cell_id].seed] == slot,
                f"activation-v3 infrastructure record is not an excluded {slot} attempt",
            )
        completed_cells = sum(len(cell_ids) for cell_ids in completed.values())
        total_completed_cells += completed_cells
        progress[slot] = {
            "live_pid": pid,
            "assigned_seed_blocks": len(assigned_seeds),
            "completed_seed_blocks": len(completed),
            "completed_behavioral_cells": completed_cells,
            "infrastructure_attempt_count": len(infra_rows),
            "complete": len(completed) == len(assigned_seeds),
        }
    return {
        "status": "valid_activation_v3_streaming_raw_structure",
        "activation_root": str(root),
        "completed_behavioral_cells": total_completed_cells,
        "complete_seed_blocks": total_completed_cells // 4,
        "all_1364_cells_complete": total_completed_cells == 1364,
        "results_analysis_invoked": False,
        "lanes": progress,
    }


def _aggregate_lane_evidence(root: Path) -> dict[str, dict[str, Any]]:
    """Reconstruct complete raw and infrastructure evidence per frozen lane."""

    paths = final_paths(root)
    repair, cells = load_repair(registration_path=root / "registration.json", queue_path=root / "queue.jsonl")
    assignment = {row["episode_seed"]: row["lane_slot"] for row in validate_assignment(repair["assignment_manifest"])}
    cells_by_id = {cell.cell_id: cell for cell in cells}
    expected_by_slot = {
        slot: {cell.cell_id for cell in cells if assignment[cell.seed] == slot}
        for slot in LANE_SLOTS
    }
    require(all(len(cell_ids) in (168, 172) for cell_ids in expected_by_slot.values()), "activation-v3 lane assignment coverage changed")
    lane_manifests = _lane_manifests(root)

    manifest = read_finite_json(paths["evidence_manifest"])
    require(isinstance(manifest, dict), "activation-v3 evidence manifest is not an object")
    _binding_at(manifest.get("repair_registration"), root / "registration.json", "activation-v3 manifest registration")
    _binding_at(manifest.get("queue"), root / "queue.jsonl", "activation-v3 manifest queue")
    _binding_at(manifest.get("assignment_manifest"), root / "assignment.jsonl", "activation-v3 manifest assignment")
    _binding_at(manifest.get("release_gate"), root / "release_gate.released.json", "activation-v3 manifest release gate")
    _binding_at(manifest.get("raw_episodes"), paths["raw_episodes"], "activation-v3 manifest raw episodes")
    _binding_at(manifest.get("infrastructure_attempts"), paths["infrastructure_attempts"], "activation-v3 manifest infrastructure attempts")

    aggregation = read_finite_json(paths["aggregation_receipt"])
    require(
        isinstance(aggregation, dict)
        and aggregation.get("schema_version") == "vla-wam-shared-v3c002r001-activation-v3-raw-aggregation-receipt-v1"
        and aggregation.get("status") == "complete_structural_raw_aggregation"
        and aggregation.get("structural_only_no_outcome_aggregate") is True
        and aggregation.get("behavioral_raw_episode_count") == 1364
        and aggregation.get("completed_seed_block_count") == 341,
        "activation-v3 raw aggregation receipt changed",
    )
    _binding_at(aggregation.get("registration"), root / "registration.json", "activation-v3 aggregation registration")
    _binding_at(aggregation.get("queue"), root / "queue.jsonl", "activation-v3 aggregation queue")
    _binding_at(aggregation.get("assignment_manifest"), root / "assignment.jsonl", "activation-v3 aggregation assignment")
    _binding_at(aggregation.get("release_gate"), root / "release_gate.released.json", "activation-v3 aggregation release gate")
    combined = aggregation.get("combined_outputs")
    require(isinstance(combined, Mapping), "activation-v3 aggregation combined outputs are missing")
    _binding_at(combined.get("raw_episodes"), paths["raw_episodes"], "activation-v3 aggregation raw episodes")
    _binding_at(combined.get("infrastructure_attempts"), paths["infrastructure_attempts"], "activation-v3 aggregation infrastructure attempts")
    receipt_lanes = aggregation.get("lane_evidence")
    require(isinstance(receipt_lanes, Mapping) and set(receipt_lanes) == set(LANE_SLOTS), "activation-v3 aggregation lane evidence is incomplete")
    compiled = manifest.get("compiled_outputs")
    require(isinstance(compiled, Mapping), "activation-v3 manifest compiled outputs are missing")
    for name in ("episodes", "pairs", "results", "decision_memo"):
        _binding_at(compiled.get(name), paths[name], f"activation-v3 compiled {name}")
    _binding_at(compiled.get("manuscript_insert"), paths["manuscript_insert"], "activation-v3 compiled manuscript insert")

    sidecars = manifest.get("repair_provenance_sidecars")
    require(isinstance(sidecars, list) and len(sidecars) == len(cells), "activation-v3 provenance sidecar coverage is incomplete")
    sidecar_bindings: dict[Path, dict[str, Any]] = {}
    for index, record in enumerate(sidecars):
        binding = validate_file_binding(record, f"activation-v3 provenance sidecar {index}")
        path = Path(binding["path"]).resolve()
        require(path.name == "r001_provenance.json" and path not in sidecar_bindings, "activation-v3 provenance sidecars are duplicated or malformed")
        sidecar_bindings[path] = binding

    aggregates = {
        slot: {
            "assigned_seed_blocks": len({cell.seed for cell in cells if assignment[cell.seed] == slot}),
            "expected_behavioral_cells": len(expected_by_slot[slot]),
            "raw_cell_ids": set(),
            "raw_request_stream_count": 0,
            "raw_request_stream_bytes": 0,
            "provenance_sidecar_count": 0,
            "infrastructure_attempt_count": 0,
            "infrastructure_cell_ids": [],
        }
        for slot in LANE_SLOTS
    }
    raw_rows = _read_jsonl(paths["raw_episodes"], "activation-v3 raw episodes")
    require(len(raw_rows) == len(cells), "activation-v3 raw episode count changed")
    seen_cells: set[str] = set()
    for row in raw_rows:
        cell_id = row.get("cell_id")
        require(isinstance(cell_id, str) and cell_id in cells_by_id and cell_id not in seen_cells, "activation-v3 raw cell coverage changed")
        seen_cells.add(cell_id)
        cell = cells_by_id[cell_id]
        slot = assignment[cell.seed]
        require(row.get("authorization_mode") == "behavioral" and row.get("excluded_from_behavioral_denominators") is False, "activation-v3 raw row is not behavioral")
        runtime = row.get("runtime_identity")
        require(isinstance(runtime, Mapping), "activation-v3 raw row lacks runtime identity")
        lane = lane_manifests[slot]
        for key in (
            "lane_id", "simulator_pod_uid", "simulator_gpu_uuid", "policy_server_pod_uid",
            "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity",
        ):
            require(runtime.get(key) == lane.get(key), f"activation-v3 raw row differs from assigned {slot} for {key}")
        raw_artifacts = row.get("raw_artifacts")
        require(isinstance(raw_artifacts, Mapping), "activation-v3 raw row lacks raw artifacts")
        request_stream = validate_file_binding(raw_artifacts.get("raw_episode_jsonl"), "activation-v3 raw request stream")
        stream_path = Path(request_stream["path"])
        sidecar_path = stream_path.with_name("r001_provenance.json").resolve()
        require(sidecar_path in sidecar_bindings, f"activation-v3 raw provenance is not manifest-bound: {cell_id}")
        provenance = read_finite_json(sidecar_path)
        require(isinstance(provenance, dict), f"activation-v3 raw provenance is not an object: {cell_id}")
        require(
            provenance.get("schema_version") == "vla-wam-shared-v3c002r001-raw-provenance-v1"
            and provenance.get("repair_id") == "V3-C002-R001"
            and provenance.get("cell_id") == cell_id
            and provenance.get("episode_seed") == cell.seed
            and provenance.get("lane_slot") == slot
            and provenance.get("lane_id") == lane.get("lane_id"),
            f"activation-v3 raw provenance identity changed: {cell_id}",
        )
        parent_raw = validate_file_binding(provenance.get("parent_raw_episode"), "activation-v3 provenance parent raw")
        parent_rows = _read_jsonl(Path(parent_raw["path"]), f"activation-v3 provenance parent raw {cell_id}")
        require(
            len(parent_rows) == 1
            and parent_rows[0].get("cell_id") == cell_id
            and parent_rows[0] == row,
            f"activation-v3 provenance parent raw mismatch: {cell_id}",
        )
        aggregates[slot]["raw_cell_ids"].add(cell_id)
        aggregates[slot]["raw_request_stream_count"] += 1
        aggregates[slot]["raw_request_stream_bytes"] += request_stream["bytes"]
        aggregates[slot]["provenance_sidecar_count"] += 1
    require(seen_cells == set(cells_by_id), "activation-v3 raw rows do not cover the registered queue")
    require(set(sidecar_bindings) == {
        Path(validate_file_binding(row["raw_artifacts"]["raw_episode_jsonl"], "activation-v3 raw request stream") ["path"]).with_name("r001_provenance.json").resolve()
        for row in raw_rows
    }, "activation-v3 manifest contains orphaned provenance sidecars")

    infrastructure = _read_jsonl(paths["infrastructure_attempts"], "activation-v3 infrastructure attempts") if paths["infrastructure_attempts"].stat().st_size else []
    for row in infrastructure:
        cell_id = row.get("cell_id")
        require(
            row.get("schema_version") == "vla-wam-shared-v3c002-infrastructure-attempt-v1"
            and row.get("record_type") == "infrastructure_attempt"
            and row.get("infrastructure_status") == "infrastructure_invalid_excluded"
            and row.get("denominator_eligible") is False
            and row.get("authorization_mode") == "behavioral",
            "activation-v3 infrastructure record is not an excluded behavioral attempt",
        )
        require(isinstance(cell_id, str) and cell_id in cells_by_id, "activation-v3 infrastructure record has an unknown cell")
        cell = cells_by_id[cell_id]
        require(row.get("seed_block_id") == cell.block_id, "activation-v3 infrastructure record seed block changed")
        slot = assignment[cell.seed]
        aggregates[slot]["infrastructure_attempt_count"] += 1
        aggregates[slot]["infrastructure_cell_ids"].append(cell_id)

    normalized: dict[str, dict[str, Any]] = {}
    for slot, value in aggregates.items():
        require(value["raw_cell_ids"] == expected_by_slot[slot], f"activation-v3 raw coverage is incomplete for {slot}")
        require(value["provenance_sidecar_count"] == value["expected_behavioral_cells"], f"activation-v3 provenance coverage is incomplete for {slot}")
        normalized[slot] = {
            "assigned_seed_blocks": value["assigned_seed_blocks"],
            "expected_behavioral_cells": value["expected_behavioral_cells"],
            "raw_request_stream_count": value["raw_request_stream_count"],
            "raw_request_stream_bytes": value["raw_request_stream_bytes"],
            "provenance_sidecar_count": value["provenance_sidecar_count"],
            "infrastructure_attempt_count": value["infrastructure_attempt_count"],
            "infrastructure_cell_ids": sorted(value["infrastructure_cell_ids"]),
        }
        receipt_lane = receipt_lanes[slot]
        require(
            isinstance(receipt_lane, Mapping)
            and receipt_lane.get("completed_marker_count") == value["assigned_seed_blocks"]
            and receipt_lane.get("raw_episode_count") == value["expected_behavioral_cells"]
            and receipt_lane.get("provenance_sidecar_count") == value["expected_behavioral_cells"]
            and receipt_lane.get("infrastructure_attempt_count") == value["infrastructure_attempt_count"],
            f"activation-v3 aggregation receipt counts changed for {slot}",
        )
        for label, records in (
            ("completed marker", receipt_lane.get("completed_markers")),
            ("raw episode", receipt_lane.get("raw_episode_bindings")),
            ("provenance sidecar", receipt_lane.get("provenance_sidecars")),
            ("infrastructure ledger", receipt_lane.get("infrastructure_ledgers")),
        ):
            require(isinstance(records, list), f"activation-v3 aggregation {label} bindings missing for {slot}")
            for index, record in enumerate(records):
                validate_file_binding(record, f"activation-v3 aggregation {label} {slot} {index}")
    return normalized


def _run_results_validator(root: Path) -> dict[str, Any]:
    command = [sys.executable, str(REPO_ROOT / "tools/validate_v3c002r001_results.py"), "--root", str(root)]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    if result.returncode:
        raise ContractError(f"activation-v3 results validator failed: {result.stderr or result.stdout}")
    return {"command": command, "stdout": result.stdout.strip()}


def _require_complete_raw_queue_before_analysis(root: Path) -> None:
    """Check only registered IDs/counts before permitting any result analysis."""

    _, cells = load_repair(registration_path=root / "registration.json", queue_path=root / "queue.jsonl")
    rows = _read_jsonl(final_paths(root)["raw_episodes"], "activation-v3 raw episodes")
    cell_ids = [row.get("cell_id") for row in rows]
    require(
        len(rows) == 1364 and len(set(cell_ids)) == 1364 and set(cell_ids) == {cell.cell_id for cell in cells},
        "activation-v3 raw queue is incomplete; result analysis is prohibited",
    )


def validate(root: Path = ROOT, *, run_results_validator: bool = True) -> dict[str, Any]:
    root = Path(root).resolve()
    require(root == ROOT.resolve(), "activation-v3 final validator accepts only activation_v3")
    require((root / "registration.json").is_file() and (root / "queue.jsonl").is_file(), "activation-v3 registration or queue is missing")
    if not result_bundle_present(root):
        return {
            "status": "valid_activation_v3_pending_final_results",
            "activation_root": str(root),
            "final_result_bundle_present": False,
        }
    _require_complete_raw_queue_before_analysis(root)
    result_validation = _run_results_validator(root) if run_results_validator else {"skipped_for_test": True}
    lane_evidence = _aggregate_lane_evidence(root)
    paths = final_paths(root)
    return {
        "status": "valid_complete_activation_v3_publication_bundle",
        "activation_root": str(root),
        "final_result_bundle_present": True,
        "results_validator": result_validation,
        "raw_episodes_sha256": sha256_file(paths["raw_episodes"]),
        "infrastructure_attempts_sha256": sha256_file(paths["infrastructure_attempts"]),
        "lane_evidence": lane_evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--progress-lane-root",
        action="append",
        default=[],
        metavar="REPAIR-LANE-NN=PATH",
        help="Read-only raw root for one live lane; provide all eight with --progress-live-pid-record and --progress-infrastructure-ledger.",
    )
    parser.add_argument(
        "--progress-live-pid-record",
        action="append",
        default=[],
        metavar="REPAIR-LANE-NN=PATH",
        help="Launch-side JSON record containing lane_slot, lane_id, raw_root, and positive pid for one live lane.",
    )
    parser.add_argument(
        "--progress-infrastructure-ledger",
        action="append",
        default=[],
        metavar="REPAIR-LANE-NN=PATH",
        help="Append-only excluded infrastructure JSONL ledger for one live lane.",
    )
    args = parser.parse_args()
    try:
        progress_groups = (
            args.progress_lane_root,
            args.progress_live_pid_record,
            args.progress_infrastructure_ledger,
        )
        if any(progress_groups):
            require(all(progress_groups), "all three --progress-* mappings are required for a streaming raw validation")
            print(json.dumps(validate_progress(
                args.root,
                lane_roots=_slot_path_arguments(args.progress_lane_root, flag="--progress-lane-root"),
                live_pid_records=_slot_path_arguments(args.progress_live_pid_record, flag="--progress-live-pid-record"),
                infrastructure_ledgers=_slot_path_arguments(args.progress_infrastructure_ledger, flag="--progress-infrastructure-ledger"),
            ), indent=2, sort_keys=True))
        else:
            print(json.dumps(validate(args.root), indent=2, sort_keys=True))
    except ContractError as exc:
        raise SystemExit(f"V3-C002-R001 activation-v3 publication validation failed: {exc}") from exc


if __name__ == "__main__":
    main()
