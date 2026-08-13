#!/usr/bin/env python3
"""Fail-closed finalization of the mixed-epoch V3-C002-R001 A004 cohort.

This is intentionally separate from the frozen C002 and R001 compilers.  Raw
behavior is first compiled with the untouched parent ``compile_episode``.  The
published episode file preserves the actual server identities and hash-bound
authorization epoch for every cell.  A deep, analysis-only identity-normalized
copy is then passed to the parent's ``compile_results`` solely because that
frozen helper requires one identity per logical lane.  Its pair rows must be
byte-for-byte equal to pair rows generated from the actual episodes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import subprocess
from statistics import fmean
import sys
from typing import Any, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.v3.phase_c_semantic_equivalence_v3c002 import compiler as parent
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (
    ContractError,
    file_binding,
    load_cells,
    read_finite_json,
    require,
    sha256_file,
    validate_file_binding,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import (
    LANE_SCHEMA,
    load_repair,
    validate_assignment,
)


ORIGINAL_RELEASE_SHA = "28ee96e3fcda4637302aadef6e90233574101a75e5c7af75fa1d9e4b1c060a7d"
A003_RELEASE_SHA = "7b0835c2bb76631add47f5e13c6db4d5be40379d234e90c4b401a5214ec2463d"
INFRA_COUNT = 14
RAW_COUNT = 1364
SEED_BLOCK_COUNT = 341
PAIR_COUNT = 682
SLOTS = tuple(f"repair-lane-{index:02d}" for index in range(8))
REPLACEMENT_SLOTS = frozenset(SLOTS[:2])
RETRY = {
    "repair-lane-00": 12060, "repair-lane-01": 12101,
    "repair-lane-02": 12128, "repair-lane-03": 12177,
    "repair-lane-04": 12156, "repair-lane-05": 12107,
    "repair-lane-06": 12176, "repair-lane-07": 12112,
}
AGGREGATION_SCHEMA = "vla-wam-shared-v3c002r001-activation-v4-raw-aggregation-v1"
CONTINUATION_SCHEMA = "vla-wam-shared-v3c002r001-activation-v4-continuation-gate-v1"
PROVENANCE_SCHEMA = "vla-wam-shared-v3c002r001-raw-provenance-v1"
FINALIZER_SCHEMA = "vla-wam-shared-v3c002r001-a004-finalizer-manifest-v2"
FINAL_ANALYSIS_REGISTRATION_SCHEMA = "vla-wam-shared-v3c002r001-a004-final-analysis-registration-v2"
FINAL_ANALYSIS_SOURCE_SCHEMA = "vla-wam-shared-v3c002r001-a004-final-analysis-source-gate-v2"
REPO_ROOT = Path(__file__).resolve().parents[4]
CANONICAL_REMOTE = "https://github.com/adeeb10abbas/steerable.git"
CANONICAL_BRANCH = "experiment/v3c002-semantic-equivalence"

# These are the sole values normalized in the private analysis copy.  Raw
# artifacts, full runtime dictionaries, outcome values, and provenance stay
# untouched; the public episode output always retains the actual identities.
IDENTITY_KEYS = (
    "server_port", "raw_root", "simulator_pod_uid", "simulator_gpu_uuid",
    "policy_server_pod_uid", "policy_server_gpu_uuid", "container_identity",
    "runtime_identity_label", "source_commit", "checkpoint_digest",
)
RUNTIME_KEYS = (
    "lane_id", "server_port", "raw_root", "simulator_pod_uid", "simulator_gpu_uuid",
    "policy_server_pod_uid", "policy_server_gpu_uuid", "container_identity",
    "runtime_identity", "server_process_identity", "server_lock_identity",
)
R001_DISCLOSURE = (
    "\n\nR001 disclosure: this is a prospective post-gate operational repair. "
    "Original C002 remains closed after failed cross-server isolation; R001 uses "
    "exact within-server repeatability and block-local homogeneous lanes without "
    "a numerical tolerance.\n"
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = parent._read_jsonl(path)
    require(all(isinstance(row, dict) for row in rows), f"JSONL has a non-object row: {path}")
    return rows


def _validate_checkout_binding(record: Any, label: str) -> dict[str, Any]:
    """Validate a committed artifact after an absolute evidence path moved.

    Only an absent absolute path whose suffix begins at ``artifacts/`` may be
    relocated to this checkout.  Live PVC/raw paths are deliberately not
    relocated: they must exist exactly where their sidecar bound them.
    """
    require(isinstance(record, Mapping), f"{label} binding is missing")
    path = Path(str(record.get("path", "")))
    if not path.is_file() and path.is_absolute() and "artifacts" in path.parts:
        path = Path(__file__).resolve().parents[4].joinpath(*path.parts[path.parts.index("artifacts"):])
    require(path.is_file(), f"{label} committed artifact does not exist in bound or relocated checkout")
    candidate = file_binding(path)
    require(
        candidate["bytes"] == record.get("bytes") and candidate["sha256"] == record.get("sha256"),
        f"{label} committed artifact bytes changed after checkout relocation",
    )
    return candidate


def _same_binding(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> None:
    """Compare validated bindings by immutable content, never machine-local paths."""
    lhs = _validate_checkout_binding(left, f"{label} left")
    rhs = _validate_checkout_binding(right, f"{label} right")
    require(lhs["bytes"] == rhs["bytes"] and lhs["sha256"] == rhs["sha256"], f"{label} binding differs")


def validate_finalization_admission(
    *,
    finalization_registration: Path,
    finalization_source_gate: Path,
    parent_registration: Path,
    repair_registration: Mapping[str, Any],
    queue: Path,
    original_release: Path,
    a003_release: Path,
    continuation_gate: Path,
    v11_registration: Path,
    v11_source_gate: Path,
) -> dict[str, Any]:
    """Require the source-pushed, zero-result-read final-analysis registration.

    This runs before raw aggregation or any behavioral JSONL is opened.  The
    registration binds the finalizer/validator/aggregator source, immutable
    predecessors, and the 14-record infrastructure policy without receiving an
    outcome-file path as an argument.
    """
    registration = read_finite_json(finalization_registration)
    require(
        isinstance(registration, dict)
        and registration.get("schema_version") == FINAL_ANALYSIS_REGISTRATION_SCHEMA
        and registration.get("status") == "registered_prospective_corrected_final_analysis_before_raw_aggregation_or_result_read"
        and all(registration.get(key) == 0 for key in (
            "final_analysis_raw_behavioral_rows_read_before_registration",
            "final_analysis_result_compilations_before_registration",
            "final_analysis_output_files_before_registration",
        )),
        "A004 final analysis was not registered before aggregation/result reading",
    )
    superseded = registration.get("superseded_v1_final_analysis")
    require(
        isinstance(superseded, dict)
        and superseded.get("status") == "superseded_unexecuted_before_any_outcome_aggregation_or_result_read"
        and all(superseded.get(key) == 0 for key in ("raw_behavioral_rows_read", "result_compilations", "output_files")),
        "A004 final-analysis v1 supersession is not a zero-read correction",
    )
    _validate_checkout_binding(superseded.get("registration"), "superseded final-analysis v1 registration")
    _validate_checkout_binding(superseded.get("source_gate"), "superseded final-analysis v1 source gate")
    expected = {
        "parent_registration": parent_registration,
        "queue": queue,
        "original_release28ee": original_release,
        "a003_release7b08": a003_release,
        "v10_continuation_gate": continuation_gate,
        "v11_registration": v11_registration,
        "v11_source_gate": v11_source_gate,
    }
    for label, path in expected.items():
        _same_binding(registration.get(label), file_binding(path), f"final-analysis {label}")
    _same_binding(registration.get("repair_registration"), repair_registration, "final-analysis repair registration")
    frozen = registration.get("frozen_analysis_contract")
    require(
        isinstance(frozen, dict)
        and frozen.get("bootstrap_resamples") == 20_000
        and frozen.get("infrastructure_attempt_count_excluded") == INFRA_COUNT
        and frozen.get("actual_identity_episode_output") is True
        and frozen.get("analysis_only_deep_identity_normalization") is True
        and frozen.get("pair_rows_must_equal_before_after_normalization") is True
        and frozen.get("r001_lane_and_leave_one_out_diagnostics_preserved") is True,
        "A004 final-analysis frozen contract changed",
    )
    inventory = registration.get("source_inventory")
    require(isinstance(inventory, dict) and inventory, "A004 final-analysis source inventory missing")
    for label, binding in inventory.items():
        _validate_checkout_binding(binding, f"A004 final-analysis source {label}")
    v11 = read_finite_json(v11_source_gate)
    require(isinstance(v11, dict) and v11.get("passed") is True and v11.get("pushed") is True, "A004 v11 source gate was not pushed")
    source = read_finite_json(finalization_source_gate)
    require(
        isinstance(source, dict)
        and source.get("schema_version") == FINAL_ANALYSIS_SOURCE_SCHEMA
        and source.get("status") == "passed_final_analysis_source_and_registration_pushed_before_raw_aggregation"
        and source.get("passed") is True
        and source.get("pushed") is True
        and all(source.get(key) == 0 for key in (
            "final_analysis_raw_behavioral_rows_read_before_registration",
            "final_analysis_result_compilations_before_registration",
            "final_analysis_output_files_before_registration",
        )),
        "A004 final-analysis source gate was not pushed prospectively",
    )
    _same_binding(source.get("final_analysis_registration"), file_binding(finalization_registration), "final-analysis source registration")
    require(source.get("source_inventory") == inventory, "A004 final-analysis source inventory differs from registration")
    for label, binding in source["source_inventory"].items():
        _validate_checkout_binding(binding, f"A004 final-analysis source gate {label}")
    implementation = source.get("implementation_commit")
    registration_commit = source.get("registration_commit")
    require(
        source.get("remote") == CANONICAL_REMOTE
        and source.get("branch") == CANONICAL_BRANCH
        and all(isinstance(value, str) and len(value) == 40 for value in (implementation, registration_commit))
        and source.get("remote_head_at_gate") == registration_commit
        and source.get("git_verification", {}).get("remote_head") == registration_commit,
        "A004 final-analysis source-gate Git identity changed",
    )
    runtime_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()
    require(status == "", "A004 final-analysis runtime checkout is not clean")
    for commit in (implementation, registration_commit):
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, runtime_head], cwd=REPO_ROOT, check=False,
        )
        require(ancestor.returncode == 0, "A004 final-analysis recorded source commit is not an ancestor of runtime HEAD")
    remote = subprocess.run(
        ["git", "ls-remote", "--heads", CANONICAL_REMOTE, CANONICAL_BRANCH],
        cwd=REPO_ROOT, text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    require(remote == [f"{runtime_head}\trefs/heads/{CANONICAL_BRANCH}"], "A004 final-analysis runtime/source gate is not the exact pushed branch head")
    return {
        "finalization_registration": file_binding(finalization_registration),
        "finalization_source_gate": file_binding(finalization_source_gate),
    }


def _released_lanes(path: Path, *, expected_sha: str, label: str) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], dict[str, Any]]]]:
    require(sha256_file(path) == expected_sha, f"{label} release bytes changed")
    gate = read_finite_json(path)
    require(isinstance(gate, dict) and gate.get("passed") is True, f"{label} release did not pass")
    records = gate.get("lane_manifests")
    require(isinstance(records, list) and len(records) == 8, f"{label} release lacks eight lanes")
    lanes: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for record in records:
        binding = _validate_checkout_binding(record, f"{label} lane manifest")
        value = read_finite_json(Path(binding["path"]))
        require(
            isinstance(value, dict)
            and value.get("schema_version") == LANE_SCHEMA
            and value.get("passed") is True,
            f"{label} lane manifest invalid",
        )
        slot = value.get("lane_slot")
        require(isinstance(slot, str) and slot in SLOTS and slot not in lanes, f"{label} lane slots invalid")
        lanes[slot] = (binding, value)
    require(tuple(sorted(lanes)) == SLOTS, f"{label} lane coverage changed")
    return gate, lanes


def _continuation_lanes(
    path: Path,
    *,
    assignment: Mapping[int, str],
    original_release: Path,
    original: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
    a003_release: Path,
    replacement: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], dict[str, Any]]], dict[str, set[int]]]:
    """Load the continuation gate and prove its manifest/seed routing is exact."""
    gate = read_finite_json(path)
    require(
        isinstance(gate, dict)
        and gate.get("schema_version") == CONTINUATION_SCHEMA
        and gate.get("status") == "passed_outcome_blind_a004_continuation_release"
        and gate.get("passed") is True
        and gate.get("outcome_fields_read") is False,
        "A004 continuation gate is not a passed outcome-blind release",
    )
    original_binding = _validate_checkout_binding(gate.get("original_release"), "continuation original release")
    a003_binding = _validate_checkout_binding(gate.get("a003_release"), "continuation A003 release")
    _same_binding(original_binding, file_binding(original_release), "continuation original release")
    _same_binding(a003_binding, file_binding(a003_release), "continuation A003 release")
    records = gate.get("lane_manifests")
    require(isinstance(records, list) and len(records) == 8, "A004 continuation lacks eight mixed lane manifests")
    lanes: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for record in records:
        binding = _validate_checkout_binding(record, "A004 continuation lane manifest")
        value = read_finite_json(Path(binding["path"]))
        slot = value.get("lane_slot") if isinstance(value, dict) else None
        require(isinstance(slot, str) and slot in SLOTS and slot not in lanes, "A004 continuation lane invalid")
        expected = replacement[slot][0] if slot in REPLACEMENT_SLOTS else original[slot][0]
        _same_binding(binding, expected, f"A004 continuation exact lane manifest {slot}")
        lanes[slot] = (binding, value)
    require(tuple(sorted(lanes)) == SLOTS, "A004 continuation lane set changed")
    require(
        gate.get("no_cross_lane_failover") is True
        and gate.get("completed_blocks_never_rerun") is True
        and gate.get("science_unchanged") is True,
        "A004 continuation changed the prospective operational contract",
    )
    remaining_value = gate.get("remaining_seed_blocks_by_lane")
    require(isinstance(remaining_value, dict) and set(remaining_value) == set(SLOTS), "A004 remaining seed routing is incomplete")
    remaining: dict[str, set[int]] = {}
    for slot in SLOTS:
        values = remaining_value[slot]
        assigned = {seed for seed, assigned_slot in assignment.items() if assigned_slot == slot}
        require(
            isinstance(values, list)
            and all(type(seed) is int for seed in values)
            and len(values) == len(set(values))
            and set(values) <= assigned,
            f"A004 continuation remaining seeds changed for {slot}",
        )
        # The two A003 retries are a distinct historical epoch, never a
        # continuation cell.  All other slots' completion markers are checked
        # by the gate itself, and the route is reconstructed below.
        if slot in REPLACEMENT_SLOTS:
            require(RETRY[slot] not in values, f"A003 retry seed incorrectly routed as continuation: {slot}")
        remaining[slot] = set(values)
    require(sum(len(values) for values in remaining.values()) == 209, "A004 continuation block count changed")
    return gate, lanes, remaining


def build_exact_routing(
    *,
    assignment: Mapping[int, str],
    original_gate: Mapping[str, Any],
    original_lanes: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
    a003_gate: Mapping[str, Any],
    a003_lanes: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
    continuation_gate: Mapping[str, Any],
    continuation_lanes: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
    continuation_remaining: Mapping[str, set[int]],
) -> dict[int, dict[str, Any]]:
    """Map every registered seed to one, and only one, historical epoch."""
    original_binding = file_binding(Path(str(original_gate["_path"])))
    a003_binding = file_binding(Path(str(a003_gate["_path"])))
    continuation_binding = file_binding(Path(str(continuation_gate["_path"])))
    routes: dict[int, dict[str, Any]] = {}
    for seed, slot in assignment.items():
        if seed in continuation_remaining[slot]:
            epoch, gate, lane = "continuation", continuation_binding, continuation_lanes[slot]
        elif slot in REPLACEMENT_SLOTS and seed == RETRY[slot]:
            epoch, gate, lane = "a003_replacement_retry", a003_binding, a003_lanes[slot]
        else:
            epoch, gate, lane = "original_release", original_binding, original_lanes[slot]
        routes[seed] = {
            "epoch": epoch,
            "lane_slot": slot,
            "authorization_gate": gate,
            "lane_manifest": lane[0],
            "lane": lane[1],
        }
    require(len(routes) == SEED_BLOCK_COUNT, "A004 routing does not cover all registered seed blocks")
    counts = defaultdict(int)
    for route in routes.values():
        counts[str(route["epoch"])] += 1
    require(dict(counts) == {"original_release": 130, "a003_replacement_retry": 2, "continuation": 209}, "A004 exact epoch block counts changed")
    return routes


def _provenance_for(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts = raw.get("raw_artifacts")
    require(isinstance(artifacts, Mapping), "raw episode lacks artifacts")
    request = validate_file_binding(artifacts.get("raw_episode_jsonl"), "finalizer raw episode JSONL")
    sidecar_path = Path(request["path"]).with_name("r001_provenance.json")
    require(sidecar_path.is_file(), f"finalizer provenance missing: {raw.get('cell_id')}")
    sidecar = read_finite_json(sidecar_path)
    require(isinstance(sidecar, dict) and sidecar.get("schema_version") == PROVENANCE_SCHEMA, "finalizer provenance schema changed")
    return file_binding(sidecar_path), sidecar


def validate_mixed_provenance(
    raw_rows: Iterable[Mapping[str, Any]],
    *,
    cells_by_id: Mapping[str, Any],
    routing: Mapping[int, Mapping[str, Any]],
    repair_registration: Mapping[str, Any],
    assignment_binding: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Independently bind each aggregate row back to its parent and epoch."""
    sidecars: list[dict[str, Any]] = []
    parent_raws: list[dict[str, Any]] = []
    diagnostics: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        cell_id = raw.get("cell_id")
        require(isinstance(cell_id, str) and cell_id in cells_by_id, "finalizer raw cell is unregistered")
        cell = cells_by_id[cell_id]
        seed = raw.get("episode_seed")
        require(type(seed) is int and seed == cell.seed and seed in routing, "finalizer raw seed is unassigned")
        route = routing[seed]
        require(route["lane_slot"] == str(route["lane"]["lane_slot"]), "finalizer route lane is malformed")
        sidecar_binding, provenance = _provenance_for(raw)
        require(
            provenance.get("cell_id") == cell_id
            and provenance.get("episode_seed") == seed
            and provenance.get("lane_slot") == route["lane_slot"]
            and provenance.get("lane_id") == route["lane"]["lane_id"]
            and provenance.get("block_indivisible") is True,
            "finalizer provenance cell/block identity changed",
        )
        _same_binding(provenance.get("repair_registration"), repair_registration, "finalizer provenance repair registration")
        _same_binding(provenance.get("assignment_manifest"), assignment_binding, "finalizer provenance assignment")
        parent_raw = validate_file_binding(provenance.get("parent_raw_episode"), "finalizer provenance parent raw")
        parent_rows = _jsonl(Path(parent_raw["path"]))
        require(len(parent_rows) == 1 and parent_rows[0] == raw, "finalizer provenance parent/raw coverage changed")
        _same_binding(provenance.get("authorization_gate"), route["authorization_gate"], "finalizer authorization epoch")
        _same_binding(provenance.get("released_lane_manifest"), route["lane_manifest"], "finalizer lane epoch")
        runtime = raw.get("runtime_identity")
        require(isinstance(runtime, Mapping), "finalizer raw lacks runtime identity")
        for key in RUNTIME_KEYS:
            require(runtime.get(key) == route["lane"].get(key), f"finalizer runtime differs from its hash-bound epoch manifest: {key}")
        diagnostic_key = f"{route['lane_slot']}:{route['epoch']}:{route['authorization_gate']['sha256']}:{route['lane_manifest']['sha256']}"
        actual_identity = {name: runtime[name] for name in RUNTIME_KEYS}
        item = diagnostics.setdefault(
            diagnostic_key,
            {
                "lane_slot": route["lane_slot"],
                "epoch": route["epoch"],
                "authorization_gate": route["authorization_gate"],
                "lane_manifest": route["lane_manifest"],
                "actual_runtime_identity": actual_identity,
                "raw_cell_count": 0,
                "seed_blocks": [],
            },
        )
        require(item["actual_runtime_identity"] == actual_identity, "finalizer epoch identity changed within an epoch")
        item["raw_cell_count"] += 1
        if seed not in item["seed_blocks"]:
            item["seed_blocks"].append(seed)
        sidecars.append(sidecar_binding)
        parent_raws.append(parent_raw)
    require(len(sidecars) == RAW_COUNT and len({record["path"] for record in sidecars}) == RAW_COUNT, "finalizer sidecar coverage is not one-to-one")
    require(len(parent_raws) == RAW_COUNT and len({record["path"] for record in parent_raws}) == RAW_COUNT, "finalizer parent raw coverage is not one-to-one")
    epoch_rows = []
    for key in sorted(diagnostics):
        item = diagnostics[key]
        item["seed_blocks"].sort()
        require(item["raw_cell_count"] == 4 * len(item["seed_blocks"]), "finalizer epoch contains a partial behavioral block")
        epoch_rows.append(item)
    return sidecars, {
        "schema_version": "vla-wam-shared-v3c002r001-a004-actual-epoch-diagnostics-v1",
        "actual_episode_identity_preserved": True,
        "authorization_epoch_routing_exact": True,
        "epochs": epoch_rows,
        "provenance_sidecars": sidecars,
        "parent_raw_episodes": parent_raws,
    }


def validate_infrastructure(
    rows: Iterable[Mapping[str, Any]], *, cells_by_id: Mapping[str, Any], assignment: Mapping[int, str]
) -> list[dict[str, Any]]:
    rows = list(rows)
    require(len(rows) == INFRA_COUNT, "A004 finalization requires all 14 infrastructure-invalid attempts")
    for row in rows:
        cell_id = row.get("cell_id")
        require(
            row.get("schema_version") == "vla-wam-shared-v3c002-infrastructure-attempt-v1"
            and row.get("record_type") == "infrastructure_attempt"
            and row.get("infrastructure_status") == "infrastructure_invalid_excluded"
            and row.get("denominator_eligible") is False
            and row.get("authorization_mode") == "behavioral"
            and isinstance(cell_id, str)
            and cell_id in cells_by_id
            and row.get("seed_block_id") == cells_by_id[cell_id].block_id
            and cells_by_id[cell_id].seed in assignment,
            "A004 infrastructure record is not an excluded registered attempt",
        )
        attempt_root = row.get("attempt_root")
        require(isinstance(attempt_root, str) and Path(attempt_root).is_absolute(), "A004 infrastructure attempt root is not retained as an absolute lane-local path")
    return rows


def validate_aggregation_receipt(
    *,
    receipt_path: Path,
    raw_episodes: Path,
    infrastructure_attempts: Path,
    repair_registration: Mapping[str, Any],
    queue: Path,
    assignment_binding: Mapping[str, Any],
    original_release: Path,
    a003_release: Path,
    continuation_gate: Path,
    routing: Mapping[int, Mapping[str, Any]],
    infrastructure_rows: Iterable[Mapping[str, Any]],
    cells_by_id: Mapping[str, Any],
    assignment: Mapping[int, str],
) -> dict[str, Any]:
    """Bind the finalizer exclusively to an outcome-blind aggregation receipt."""
    receipt = read_finite_json(receipt_path)
    require(
        isinstance(receipt, dict)
        and receipt.get("schema_version") == AGGREGATION_SCHEMA
        and receipt.get("status") == "complete_outcome_blind_mixed_epoch_raw_aggregation"
        and receipt.get("passed") is True
        and receipt.get("outcome_fields_read") is False,
        "A004 aggregation receipt is not a complete outcome-blind collector receipt",
    )
    require(
        receipt.get("repair_id") == "V3-C002-R001"
        and receipt.get("behavioral_episode_count") == RAW_COUNT
        and receipt.get("complete_seed_block_count") == SEED_BLOCK_COUNT
        and receipt.get("infrastructure_attempt_count_excluded") == INFRA_COUNT,
        "A004 aggregation receipt counts changed",
    )
    _same_binding(receipt.get("repair_registration"), repair_registration, "A004 aggregation repair registration")
    _same_binding(receipt.get("queue"), file_binding(queue), "A004 aggregation queue")
    _same_binding(receipt.get("assignment_manifest"), assignment_binding, "A004 aggregation assignment")
    _same_binding(receipt.get("original_release"), file_binding(original_release), "A004 aggregation original release")
    _same_binding(receipt.get("a003_release"), file_binding(a003_release), "A004 aggregation A003 release")
    _same_binding(receipt.get("continuation_gate"), file_binding(continuation_gate), "A004 aggregation continuation gate")
    outputs = receipt.get("combined_outputs")
    require(isinstance(outputs, dict), "A004 aggregation combined outputs missing")
    _same_binding(outputs.get("raw_episodes"), file_binding(raw_episodes), "A004 aggregation raw output")
    _same_binding(outputs.get("infrastructure_attempts"), file_binding(infrastructure_attempts), "A004 aggregation infrastructure output")
    lane_evidence = receipt.get("lane_evidence")
    require(isinstance(lane_evidence, dict) and set(lane_evidence) == set(SLOTS), "A004 aggregation lane evidence is incomplete")
    require(sum(item.get("infrastructure_attempt_count", -1) for item in lane_evidence.values() if isinstance(item, dict)) == INFRA_COUNT, "A004 aggregation lane infrastructure count changed")
    infrastructure_rows = list(infrastructure_rows)
    validate_infrastructure(infrastructure_rows, cells_by_id=cells_by_id, assignment=assignment)
    roots_by_slot: dict[str, str] = {}
    for slot, item in lane_evidence.items():
        require(isinstance(item, dict) and isinstance(item.get("raw_root"), str) and Path(item["raw_root"]).is_absolute(), f"A004 aggregation raw root missing for {slot}")
        roots_by_slot[slot] = item["raw_root"]
        ledger = item.get("infrastructure_ledger")
        count = item.get("infrastructure_attempt_count")
        require(type(count) is int and count >= 0, f"A004 aggregation ledger count invalid for {slot}")
        if count:
            validate_file_binding(ledger, f"A004 aggregation infrastructure ledger {slot}")
        else:
            require(ledger is None, f"A004 aggregation empty ledger binding should be null for {slot}")
    by_slot = defaultdict(list)
    for row in infrastructure_rows:
        cell_id = str(row["cell_id"])
        slot = assignment[cells_by_id[cell_id].seed]
        by_slot[slot].append(row)
        require(str(row["attempt_root"]).startswith(roots_by_slot[slot]), "A004 infrastructure attempt root is outside its retained assigned lane root")
    for slot in SLOTS:
        require(len(by_slot[slot]) == lane_evidence[slot]["infrastructure_attempt_count"], f"A004 aggregation infrastructure lane count disagrees for {slot}")
    expected_counts = defaultdict(int)
    for route in routing.values():
        expected_counts[str(route["epoch"])] += 4
    require(receipt.get("epoch_episode_counts") == dict(expected_counts), "A004 aggregation epoch counts disagree with exact seed routing")
    return receipt


def identity_normalized_copy(episodes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return a deep, analysis-local identity normalization without mutation."""
    canonical: dict[str, dict[str, Any]] = {}
    result: list[dict[str, Any]] = []
    for episode in episodes:
        item = copy.deepcopy(dict(episode))
        lane_id = str(item["lane_id"])
        identity = {key: item[key] for key in IDENTITY_KEYS}
        canonical.setdefault(lane_id, identity)
        for key, value in canonical[lane_id].items():
            item[key] = value
        result.append(item)
    return result


def add_r001_diagnostics(results: dict[str, Any], pairs: Iterable[Mapping[str, Any]], assignment: Mapping[int, str]) -> None:
    """Copy the frozen R001 lane/leave-one-out diagnostic calculation exactly."""
    pair_rows = list(pairs)
    lane_diagnostics = {}
    for slot in sorted(SLOTS):
        slot_pairs = [pair for pair in pair_rows if assignment[int(pair["episode_seed"])] == slot]
        lane_diagnostics[slot] = {
            "seed_blocks": len({pair["episode_seed"] for pair in slot_pairs}),
            "left_depth_inverse_minus_canonical_mean_m": fmean(
                pair["depth_difference_inverse_minus_canonical_m"] for pair in slot_pairs if pair["physical_goal"] == "left"
            ),
            "right_depth_inverse_minus_canonical_mean_m": fmean(
                pair["depth_difference_inverse_minus_canonical_m"] for pair in slot_pairs if pair["physical_goal"] == "right"
            ),
        }
    leave_one_out = {}
    for omitted in sorted(SLOTS):
        subset = [pair for pair in pair_rows if assignment[int(pair["episode_seed"])] != omitted]
        leave_one_out[omitted] = {
            goal: fmean(pair["depth_difference_inverse_minus_canonical_m"] for pair in subset if pair["physical_goal"] == goal)
            for goal in ("left", "right")
        }
    results["repair_id"] = "V3-C002-R001"
    results["original_cross_lane_gate_remains_failed"] = True
    results["cross_lane_numerical_tolerance_used"] = False
    results["lane_diagnostics_descriptive_only"] = lane_diagnostics
    results["leave_one_lane_out_diagnostics_descriptive_only"] = leave_one_out


def compile_final(
    *,
    parent_registration: Path,
    queue: Path,
    raw_episodes: Path,
    infrastructure_attempts: Path,
    aggregation_receipt: Path,
    finalization_registration: Path,
    finalization_source_gate: Path,
    original_release: Path,
    a003_release: Path,
    continuation_gate: Path,
    v11_registration: Path,
    v11_source_gate: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Pure final compilation; this is the sole behavioral-result reader."""
    a003_gate, a003_lanes = _released_lanes(a003_release, expected_sha=A003_RELEASE_SHA, label="A0037b08")
    original_gate, original_lanes = _released_lanes(original_release, expected_sha=ORIGINAL_RELEASE_SHA, label="original28ee")
    # Retain paths only locally so routing can use portable hash-bound files.
    a003_gate = {**a003_gate, "_path": str(a003_release.resolve())}
    original_gate = {**original_gate, "_path": str(original_release.resolve())}
    repair_registration = _validate_checkout_binding(a003_gate.get("repair_registration"), "A003 repair registration")
    assignment_binding = _validate_checkout_binding(a003_gate.get("assignment_manifest"), "A003 assignment")
    admission = validate_finalization_admission(
        finalization_registration=finalization_registration,
        finalization_source_gate=finalization_source_gate,
        parent_registration=parent_registration,
        repair_registration=repair_registration,
        queue=queue,
        original_release=original_release,
        a003_release=a003_release,
        continuation_gate=continuation_gate,
        v11_registration=v11_registration,
        v11_source_gate=v11_source_gate,
    )
    registration, parent_cells = load_cells(registration_path=parent_registration, queue_path=queue)
    repair, repair_cells = load_repair(registration_path=Path(repair_registration["path"]), queue_path=queue)
    require([cell.cell_id for cell in repair_cells] == [cell.cell_id for cell in parent_cells], "A004 repair queue differs from frozen parent queue")
    assignment = {int(row["episode_seed"]): str(row["lane_slot"]) for row in validate_assignment(assignment_binding)}
    continuation, continuation_lanes, remaining = _continuation_lanes(
        continuation_gate,
        assignment=assignment,
        original_release=original_release,
        original=original_lanes,
        a003_release=a003_release,
        replacement=a003_lanes,
    )
    continuation = {**continuation, "_path": str(continuation_gate.resolve())}
    routing = build_exact_routing(
        assignment=assignment,
        original_gate=original_gate,
        original_lanes=original_lanes,
        a003_gate=a003_gate,
        a003_lanes=a003_lanes,
        continuation_gate=continuation,
        continuation_lanes=continuation_lanes,
        continuation_remaining=remaining,
    )
    cells_by_id = {cell.cell_id: cell for cell in parent_cells}
    infra = validate_infrastructure(
        _jsonl(infrastructure_attempts) if infrastructure_attempts.stat().st_size else [],
        cells_by_id=cells_by_id,
        assignment=assignment,
    )
    validate_aggregation_receipt(
        receipt_path=aggregation_receipt,
        raw_episodes=raw_episodes,
        infrastructure_attempts=infrastructure_attempts,
        repair_registration=repair_registration,
        queue=queue,
        assignment_binding=assignment_binding,
        original_release=original_release,
        a003_release=a003_release,
        continuation_gate=continuation_gate,
        routing=routing,
        infrastructure_rows=infra,
        cells_by_id=cells_by_id,
        assignment=assignment,
    )
    raw = _jsonl(raw_episodes)
    require(len(raw) == RAW_COUNT and len({row.get("cell_id") for row in raw}) == RAW_COUNT, "A004 raw cohort is not exactly 1,364 unique cells")
    require(set(row.get("cell_id") for row in raw) == set(cells_by_id), "A004 raw cell set differs from the frozen queue")
    sidecars, diagnostics = validate_mixed_provenance(
        raw,
        cells_by_id=cells_by_id,
        routing=routing,
        repair_registration=repair_registration,
        assignment_binding=assignment_binding,
    )
    actual = [
        parent.compile_episode(
            row,
            cell=cells_by_id[str(row["cell_id"])],
            registration_sha256=sha256_file(parent_registration),
            queue_sha256=sha256_file(queue),
            exact_runtime_contract=registration["exact_e004_pi05_runtime"],
        )
        for row in raw
    ]
    for episode in actual:
        route = routing[int(episode["episode_seed"])]
        episode["repair_id"] = "V3-C002-R001"
        episode["repair_lane_slot"] = route["lane_slot"]
        episode["authorization_epoch"] = route["epoch"]
        episode["authorization_gate"] = copy.deepcopy(route["authorization_gate"])
        episode["released_lane_manifest"] = copy.deepcopy(route["lane_manifest"])
    normalized = identity_normalized_copy(actual)
    actual_pairs = parent._pair_rows(actual)
    normalized_pairs = parent._pair_rows(normalized)
    require(actual_pairs == normalized_pairs, "identity normalization changed parent _pair_rows")
    pairs, results = parent.compile_results(
        normalized,
        registration_sha256=sha256_file(parent_registration),
        queue_sha256=sha256_file(queue),
    )
    require(pairs == actual_pairs, "identity-normalized compile_results changed parent pair rows")
    add_r001_diagnostics(results, pairs, assignment)
    diagnostics.update({
        "identity_normalization_analysis_local_only": True,
        "pair_rows_equal_before_after_normalization": True,
        "infrastructure_attempt_count_excluded": len(infra),
        "routing_seed_blocks": [
            {"episode_seed": seed, "lane_slot": route["lane_slot"], "epoch": route["epoch"], "authorization_gate_sha256": route["authorization_gate"]["sha256"], "lane_manifest_sha256": route["lane_manifest"]["sha256"]}
            for seed, route in sorted(routing.items())
        ],
        "repair_registration": repair_registration,
        "assignment_manifest": assignment_binding,
        "aggregation_receipt": file_binding(aggregation_receipt),
        **admission,
    })
    return actual, pairs, results, diagnostics


def decision_memo(results: Mapping[str, Any]) -> str:
    return parent.decision_memo(results) + R001_DISCLOSURE


def manuscript_insert(results: Mapping[str, Any]) -> str:
    return parent.manuscript_insert(results) + R001_DISCLOSURE


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), allow_nan=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--raw-episodes", type=Path, required=True)
    parser.add_argument("--infrastructure-attempts", type=Path, required=True)
    parser.add_argument("--aggregation-receipt", type=Path, required=True)
    parser.add_argument("--finalization-registration", type=Path, required=True)
    parser.add_argument("--finalization-source-gate", type=Path, required=True)
    parser.add_argument("--original-release", type=Path, required=True)
    parser.add_argument("--a003-release", type=Path, required=True)
    parser.add_argument("--continuation-gate", type=Path, required=True)
    parser.add_argument("--v11-registration", type=Path, required=True)
    parser.add_argument("--v11-source-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "A004 finalizer refuses to overwrite an output directory")
    episodes, pairs, results, diagnostics = compile_final(
        parent_registration=args.parent_registration,
        queue=args.queue,
        raw_episodes=args.raw_episodes,
        infrastructure_attempts=args.infrastructure_attempts,
        aggregation_receipt=args.aggregation_receipt,
        finalization_registration=args.finalization_registration,
        finalization_source_gate=args.finalization_source_gate,
        original_release=args.original_release,
        a003_release=args.a003_release,
        continuation_gate=args.continuation_gate,
        v11_registration=args.v11_registration,
        v11_source_gate=args.v11_source_gate,
    )
    args.output_dir.mkdir(parents=True)
    _write_jsonl(args.output_dir / "episodes.jsonl", episodes)
    _write_jsonl(args.output_dir / "pairs.jsonl", pairs)
    _write_json(args.output_dir / "results.json", results)
    _write_json(args.output_dir / "epoch_diagnostics.json", diagnostics)
    with (args.output_dir / "DECISION_MEMO.md").open("x", encoding="utf-8") as handle:
        handle.write(decision_memo(results))
    with (args.output_dir / "MANUSCRIPT_INSERT.md").open("x", encoding="utf-8") as handle:
        handle.write(manuscript_insert(results))
    _write_json(args.output_dir / "evidence_manifest.json", {
        "schema_version": FINALIZER_SCHEMA,
        "repair_id": "V3-C002-R001",
        "status": "complete_hash_bound_mixed_epoch_finalization",
        "parent_registration": file_binding(args.parent_registration),
        "queue": file_binding(args.queue),
        "raw_episodes": file_binding(args.raw_episodes),
        "infrastructure_attempts": file_binding(args.infrastructure_attempts),
        "aggregation_receipt": file_binding(args.aggregation_receipt),
        "finalization_registration": file_binding(args.finalization_registration),
        "finalization_source_gate": file_binding(args.finalization_source_gate),
        "original_release28ee": file_binding(args.original_release),
        "a003_release7b08": file_binding(args.a003_release),
        "continuation_gate": file_binding(args.continuation_gate),
        "v11_registration": file_binding(args.v11_registration),
        "v11_source_gate": file_binding(args.v11_source_gate),
        "compiled_outputs": {
            name: file_binding(args.output_dir / name)
            for name in ("episodes.jsonl", "pairs.jsonl", "results.json", "epoch_diagnostics.json", "DECISION_MEMO.md", "MANUSCRIPT_INSERT.md")
        },
        "actual_identity_preserved": True,
        "identity_normalized_copy_not_published_as_episode_data": True,
        "pair_rows_equal_before_after_normalization": True,
        "infrastructure_attempt_count_excluded": INFRA_COUNT,
        "repair_provenance_sidecars": diagnostics["provenance_sidecars"],
        "parent_raw_episodes": diagnostics["parent_raw_episodes"],
        "raw_source_artifact_count_rehashed": sum(
            len(episode["raw_artifacts"]) + len(episode["policy_camera_image_artifacts"])
            for episode in episodes
        ),
        "raw_source_bytes_rehashed": sum(
            binding["bytes"]
            for episode in episodes
            for binding in [*episode["raw_artifacts"].values(), *episode["policy_camera_image_artifacts"].values()]
        ),
        "raw_source_unique_sha256_count": len({
            binding["sha256"]
            for episode in episodes
            for binding in [*episode["raw_artifacts"].values(), *episode["policy_camera_image_artifacts"].values()]
        }),
        "provenance_sidecar_count_rehashed": len(diagnostics["provenance_sidecars"]),
        "parent_raw_episode_count_rehashed": len(diagnostics["parent_raw_episodes"]),
        "compiler": file_binding(Path(__file__)),
        "invocation": [sys.executable, *sys.argv],
    })


if __name__ == "__main__":
    try:
        main()
    except ContractError as exc:
        raise SystemExit(f"A004 finalizer failed: {exc}") from exc
