#!/usr/bin/env python3
"""Aggregate the completed A004 cohort across its registered lane epochs.

This collector is outcome blind.  It validates only cell identity, block
completeness, assignment, authorization/lane epoch, raw hashes, provenance,
and excluded infrastructure ledgers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (
    ContractError, file_binding, read_finite_json, require, sha256_file,
    validate_file_binding,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import (
    load_repair, validate_assignment,
)

SLOTS = tuple(f"repair-lane-{index:02d}" for index in range(8))
REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_RELEASE_SHA256 = "28ee96e3fcda4637302aadef6e90233574101a75e5c7af75fa1d9e4b1c060a7d"
A003_RELEASE_SHA256 = "7b0835c2bb76631add47f5e13c6db4d5be40379d234e90c4b401a5214ec2463d"
CONTINUATION_GATE_SHA256 = "f898a52148fd39f6b5178aa7200d3539ec243ce2ed412356e2bf62e3e28139a8"
RETRY = {
    "repair-lane-00": 12060, "repair-lane-01": 12101,
    "repair-lane-02": 12128, "repair-lane-03": 12177,
    "repair-lane-04": 12156, "repair-lane-05": 12107,
    "repair-lane-06": 12176, "repair-lane-07": 12112,
}


def _jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        require(line.strip(), f"{label} blank line {number}")
        value = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        require(isinstance(value, dict), f"{label} row is not an object")
        rows.append(value)
    return rows


def _parse_roots(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        slot, separator, raw = value.partition("=")
        require(separator == "=" and slot in SLOTS and raw and slot not in result, "lane root syntax/coverage changed")
        result[slot] = Path(raw).resolve()
    require(tuple(sorted(result)) == SLOTS, "exactly eight lane roots are required")
    return result


def _same_bound_bytes(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare retained content, not checkout-specific absolute path strings."""

    return left.get("bytes") == right.get("bytes") and left.get("sha256") == right.get("sha256")


def _validate_checkout_binding(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Rehash an immutable binding after an evidence checkout was relocated."""

    path = Path(str(record.get("path", "")))
    if not path.is_file() and path.is_absolute() and "artifacts" in path.parts:
        path = REPO_ROOT.joinpath(*path.parts[path.parts.index("artifacts"):])
    require(path.is_file(), f"{label} artifact does not exist in either bound or current checkout")
    candidate = file_binding(path)
    require(_same_bound_bytes(candidate, record), f"{label} artifact changed after checkout relocation")
    return candidate


def _lane_bindings(
    gate_path: Path,
    *,
    expected_sha256: str,
    expected_schema: str,
    expected_status: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    require(sha256_file(gate_path) == expected_sha256, f"gate bytes changed: {gate_path}")
    gate = read_finite_json(gate_path)
    require(
        isinstance(gate, dict)
        and gate.get("schema_version") == expected_schema
        and gate.get("status") == expected_status
        and gate.get("passed") is True,
        f"gate is not the exact passed epoch: {gate_path}",
    )
    values, bindings = {}, {}
    for record in gate.get("lane_manifests", []):
        binding = _validate_checkout_binding(record, "A004 epoch lane manifest")
        value = read_finite_json(Path(binding["path"]))
        slot = value.get("lane_slot")
        require(slot in SLOTS and slot not in values and value.get("passed") is True, "A004 epoch lane set changed")
        values[slot], bindings[slot] = value, binding
    require(tuple(sorted(values)) == SLOTS, "A004 epoch lacks eight lanes")
    return gate, values, bindings


def _expected_epoch(*, slot: str, seed: int, continuation: Mapping[str, Any], original_path: Path,
                    original_lanes: Mapping[str, Any], original_bindings: Mapping[str, Any],
                    a003_path: Path, a003_lanes: Mapping[str, Any], a003_bindings: Mapping[str, Any],
                    continuation_path: Path, continuation_lanes: Mapping[str, Any], continuation_bindings: Mapping[str, Any]):
    remaining = continuation["remaining_seed_blocks_by_lane"][slot]
    if seed in remaining:
        return "continuation", file_binding(continuation_path), continuation_lanes[slot], continuation_bindings[slot]
    if slot in {"repair-lane-00", "repair-lane-01"} and seed == RETRY[slot]:
        return "a003_replacement_retry", file_binding(a003_path), a003_lanes[slot], a003_bindings[slot]
    return "original_release", file_binding(original_path), original_lanes[slot], original_bindings[slot]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalization-registration", type=Path, required=True)
    parser.add_argument("--finalization-source-gate", type=Path, required=True)
    parser.add_argument("--repair-registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--original-release", type=Path, required=True)
    parser.add_argument("--a003-release", type=Path, required=True)
    parser.add_argument("--continuation-gate", type=Path, required=True)
    parser.add_argument("--lane-raw-root", action="append", required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--infrastructure-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    # Admission is deliberately enforced before any behavioral raw/marker is
    # opened. Import lazily to keep this collector structurally outcome blind.
    from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_finalizer.finalizer import (
        validate_finalization_admission,
    )
    a003_preview = read_finite_json(args.a003_release)
    require(isinstance(a003_preview, dict), "A003 release is not an object")
    repair_registration = _validate_checkout_binding(a003_preview.get("repair_registration"), "A003 repair registration")
    admission_registration = read_finite_json(args.finalization_registration)
    require(isinstance(admission_registration, dict), "finalization registration is not an object")
    v11_registration = _validate_checkout_binding(admission_registration.get("v11_registration"), "finalization v11 registration")
    v11_source_gate = _validate_checkout_binding(admission_registration.get("v11_source_gate"), "finalization v11 source gate")
    parent_registration = _validate_checkout_binding(admission_registration.get("parent_registration"), "finalization parent registration")
    validate_finalization_admission(
        finalization_registration=args.finalization_registration,
        finalization_source_gate=args.finalization_source_gate,
        parent_registration=Path(parent_registration["path"]),
        repair_registration=repair_registration,
        queue=args.queue,
        original_release=args.original_release,
        a003_release=args.a003_release,
        continuation_gate=args.continuation_gate,
        v11_registration=Path(v11_registration["path"]),
        v11_source_gate=Path(v11_source_gate["path"]),
    )
    for output in (args.raw_output, args.infrastructure_output, args.receipt_output):
        require(not output.exists(), f"refusing to overwrite {output}")
    repair, cells = load_repair(registration_path=args.repair_registration, queue_path=args.queue)
    assignment = {int(row["episode_seed"]): str(row["lane_slot"]) for row in validate_assignment(repair["assignment_manifest"])}
    cells_by_id = {cell.cell_id: cell for cell in cells}
    original, original_lanes, original_bindings = _lane_bindings(
        args.original_release,
        expected_sha256=ORIGINAL_RELEASE_SHA256,
        expected_schema="vla-wam-shared-v3c002r001-release-gate-v1",
        expected_status="passed_homogeneous_block_local_behavioral_release",
    )
    a003, a003_lanes, a003_bindings = _lane_bindings(
        args.a003_release,
        expected_sha256=A003_RELEASE_SHA256,
        expected_schema="vla-wam-shared-v3c002r001-activation-v3-lane-replacement-gate-v1",
        expected_status="passed_activation_v3_cluster_termination_lane_replacement",
    )
    continuation, continuation_lanes, continuation_bindings = _lane_bindings(
        args.continuation_gate,
        expected_sha256=CONTINUATION_GATE_SHA256,
        expected_schema="vla-wam-shared-v3c002r001-activation-v4-continuation-gate-v1",
        expected_status="passed_outcome_blind_a004_continuation_release",
    )
    require(continuation.get("outcome_fields_read") is False, "A004 continuation changed")
    require(sum(len(continuation["remaining_seed_blocks_by_lane"][slot]) for slot in SLOTS) == 209, "A004 remaining block count changed")
    roots = _parse_roots(args.lane_raw_root)
    raw_by_id, all_infra, lane_receipts = {}, [], {}
    epoch_counts = {"original_release": 0, "a003_replacement_retry": 0, "continuation": 0}
    for slot in SLOTS:
        assigned = {seed for seed, assigned_slot in assignment.items() if assigned_slot == slot}
        marker_paths = sorted((roots[slot] / "behavioral").glob("seed*/completed_block.json"))
        require(len(marker_paths) == len(assigned), f"{slot} completed marker count is incomplete")
        seen, marker_bindings, sidecar_bindings = set(), [], []
        for marker_path in marker_paths:
            seed_text = marker_path.parent.name.removeprefix("seed")
            require(seed_text.isdigit(), "marker seed directory is malformed")
            seed = int(seed_text)
            marker = read_finite_json(marker_path)
            require(seed in assigned and seed not in seen and marker.get("episode_seed") == seed and marker.get("status") == "completed_behavioral_block" and marker.get("authorization_mode") == "behavioral", "marker identity changed")
            epoch, gate_binding, lane, lane_binding = _expected_epoch(
                slot=slot, seed=seed, continuation=continuation,
                original_path=args.original_release, original_lanes=original_lanes, original_bindings=original_bindings,
                a003_path=args.a003_release, a003_lanes=a003_lanes, a003_bindings=a003_bindings,
                continuation_path=args.continuation_gate, continuation_lanes=continuation_lanes, continuation_bindings=continuation_bindings,
            )
            records = marker.get("raw_episodes")
            require(isinstance(records, list) and len(records) == 4, "marker is not a complete block")
            expected_ids = {cell.cell_id for cell in cells if cell.seed == seed}
            cell_ids = set()
            for record in records:
                raw_binding = validate_file_binding(record, "A004 marker raw")
                rows = _jsonl(Path(raw_binding["path"]), "A004 parent raw")
                require(len(rows) == 1, "A004 parent raw row count changed")
                row = rows[0]; cell_id = row.get("cell_id")
                require(cell_id == record.get("cell_id") and cell_id in expected_ids and cell_id not in raw_by_id, "A004 raw cell coverage changed")
                request_stream = validate_file_binding(row.get("raw_artifacts", {}).get("raw_episode_jsonl"), "A004 request stream")
                sidecar_path = Path(request_stream["path"]).with_name("r001_provenance.json")
                provenance = read_finite_json(sidecar_path)
                require(provenance.get("cell_id") == cell_id and provenance.get("episode_seed") == seed and provenance.get("lane_slot") == slot and provenance.get("lane_id") == lane["lane_id"] and provenance.get("block_indivisible") is True, "A004 provenance identity changed")
                require(_same_bound_bytes(validate_file_binding(provenance.get("parent_raw_episode"), "A004 provenance parent raw"), raw_binding), "A004 provenance parent raw differs from marker")
                require(_same_bound_bytes(validate_file_binding(provenance.get("authorization_gate"), "A004 provenance gate"), gate_binding), "A004 provenance authorization epoch changed")
                require(_same_bound_bytes(validate_file_binding(provenance.get("released_lane_manifest"), "A004 provenance lane"), lane_binding), "A004 provenance lane epoch changed")
                runtime = row.get("runtime_identity", {})
                for key in ("lane_id", "simulator_pod_uid", "simulator_gpu_uuid", "policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity"):
                    require(runtime.get(key) == lane.get(key), f"A004 raw runtime differs for {key}")
                raw_by_id[cell_id] = row; cell_ids.add(cell_id); sidecar_bindings.append(file_binding(sidecar_path)); epoch_counts[epoch] += 1
            require(cell_ids == expected_ids, "A004 marker cell set changed")
            seen.add(seed); marker_bindings.append(file_binding(marker_path))
        require(seen == assigned, f"{slot} completed seeds differ from assignment")
        infra_path = roots[slot] / "infrastructure_invalid.jsonl"
        infra = _jsonl(infra_path, f"{slot} infrastructure") if infra_path.is_file() and infra_path.stat().st_size else []
        for row in infra:
            cell_id = row.get("cell_id")
            require(row.get("infrastructure_status") == "infrastructure_invalid_excluded" and row.get("denominator_eligible") is False and cell_id in cells_by_id and assignment[cells_by_id[cell_id].seed] == slot, "A004 infrastructure row is denominator eligible or misassigned")
        all_infra.extend(infra)
        lane_receipts[slot] = {"raw_root": str(roots[slot]), "completed_markers": marker_bindings, "provenance_sidecars": sidecar_bindings, "infrastructure_ledger": file_binding(infra_path) if infra_path.is_file() else None, "infrastructure_attempt_count": len(infra)}
    require(len(raw_by_id) == 1364 and set(raw_by_id) == set(cells_by_id), "A004 does not contain all 1,364 registered cells")
    require(len(all_infra) == 14, "A004 must retain exactly fourteen excluded infrastructure attempts")
    ordered = [raw_by_id[cell.cell_id] for cell in cells]
    for path in (args.raw_output, args.infrastructure_output, args.receipt_output): path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.raw_output, ordered); _write_jsonl(args.infrastructure_output, all_infra)
    receipt = {
        "schema_version": "vla-wam-shared-v3c002r001-activation-v4-raw-aggregation-v1",
        "status": "complete_outcome_blind_mixed_epoch_raw_aggregation", "passed": True,
        "outcome_fields_read": False, "repair_id": "V3-C002-R001", "behavioral_episode_count": 1364,
        "complete_seed_block_count": 341, "infrastructure_attempt_count_excluded": 14,
        "epoch_episode_counts": epoch_counts, "repair_registration": file_binding(args.repair_registration),
        "queue": file_binding(args.queue), "assignment_manifest": repair["assignment_manifest"],
        "original_release": file_binding(args.original_release), "a003_release": file_binding(args.a003_release),
        "continuation_gate": file_binding(args.continuation_gate), "lane_evidence": lane_receipts,
        "finalization_registration": file_binding(args.finalization_registration),
        "finalization_source_gate": file_binding(args.finalization_source_gate),
        "combined_outputs": {"raw_episodes": file_binding(args.raw_output), "infrastructure_attempts": file_binding(args.infrastructure_output)},
    }
    args.receipt_output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(file_binding(args.receipt_output), indent=2, sort_keys=True))


if __name__ == "__main__":
    try: main()
    except ContractError as exc: raise SystemExit(f"A004 raw aggregation failed: {exc}") from exc
