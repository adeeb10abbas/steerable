"""Reproduce source-implied reset exclusions, without recovering missing states."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess

from experiments.workshops.spatial_grounding_v1.historical_root_bounds import (
    RootPositionBounds, prove_bounded_root_nonmatch,
)
from experiments.workshops.spatial_grounding_v1.historical_root_nonmatch import RootPosition
from tools.audit_sgw_historical_lineage import require
from tools.prove_sgw_observed_root_nonmatches import (
    ACTORS, FRAME, FRAMES, INFRA, ROOT, compile_proof as compile_observed_proof,
)


EXPORT = INFRA / "historical-reset-bounds-20260923ca"
INVENTORY = INFRA / "historical-r005-attempt01-20260923by/inventory.json"
LEDGER = ROOT / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r005/infrastructure_attempts.jsonl"
REVISION = "98f0234cd124d24004fd0053bc8f61f71e8dd50b"
CONTRACT_SHA = "2476b28d2867c1b87f477fd5f89e545616be00d860d4144f8cbdb70af10f3c18"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_bound(path: Path, binding: dict) -> dict:
    raw = path.read_bytes()
    require(len(raw) == binding["bytes"] and sha(raw) == binding["sha256"],
            f"retained input byte/hash mismatch: {path.name}")
    return json.loads(raw)


def source_blob(binding: dict, checkout: str) -> tuple[bytes, dict]:
    relative = binding["path"].removeprefix(checkout + "/")
    require(not PurePosixPath(relative).is_absolute() and ".." not in PurePosixPath(relative).parts,
            "invalid source binding path")
    raw = subprocess.check_output(["git", "-C", str(ROOT), "cat-file", "blob", f"{REVISION}:{relative}"])
    require(len(raw) == binding["bytes"] and sha(raw) == binding["sha256"],
            f"historical source binding mismatch: {relative}")
    return raw, {"path": relative, "bytes": len(raw), "sha256": sha(raw)}


def function(tree: ast.AST, name: str) -> ast.FunctionDef:
    found = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(found) == 1, f"source function absent or ambiguous: {name}")
    return found[0]


def compile_proof() -> dict:
    ledger = read_bound(LEDGER, {
        "bytes": 3382, "sha256": "3efd158f2502d6495006e32cdb9c715e0d8f23286242045148185839051c349c",
    })
    inventory = read_bound(INVENTORY, {
        "bytes": 137174, "sha256": "1bfdfba81f1b177a080ccc73ad99366a8c3bc924f3343e3a74b28b443e4539de",
    })
    require(ledger["source"]["study_commit"] == inventory["construction_source"]["study_commit"] == REVISION,
            "historical source revision differs")
    require(inventory["failure_report"] == ledger["raw_bindings"]["failure_report"],
            "failed-attempt inventory belongs to another source")
    launch = read_bound(EXPORT / "launch.json", ledger["raw_bindings"]["launch"])
    validation = read_bound(EXPORT / "target-validation.json", ledger["raw_bindings"]["target_validation_receipt"])
    require(validation["passed"] is True
            and validation["candidate_evidence"]["child_report"] == inventory["failure_report"],
            "historical validation receipt differs; this is not a scientific gate pass")
    require(launch["study_commit"] == REVISION
            and launch["status"] == "launched_after_prospective_source_push_gate",
            "historical launch identity differs")
    checkout = ledger["source"]["study_checkout"]
    require(launch["environment"]["PYTHONPATH"].split(":")[0] == checkout,
            "historical helper import root differs")
    wrapper, wrapper_binding = source_blob(launch["harness_source"], checkout)
    require(sha(wrapper) == "b95166ddc0ca7e3e618d7315bbad438e32fea9d55a0e443e30a5c9cf4f0e61cd",
            "reviewed clean-source launch control flow differs")
    producer, producer_binding = source_blob(inventory["construction_source"], checkout)
    require(sha(producer) == "ac6ae458accaa1428e4f51cb44fb07115b5ad746bc746c33970221519477ea3b"
            and launch["input_bindings"]["repair_source"]["sha256"] == sha(producer),
            "reviewed fresh-reset marker control flow differs")
    schedule_raw, schedule_binding = source_blob(inventory["input_bindings"]["candidate_schedule"], checkout)
    schedule = json.loads(schedule_raw)
    gate_raw, gate_binding = source_blob(inventory["input_bindings"]["source_push_gate"], checkout)
    gate = json.loads(gate_raw)
    implementation = [source_blob(row, checkout)[1] for row in gate["implementation_files"]]
    require(wrapper_binding in implementation and producer_binding in implementation
            and schedule_binding in implementation and len(implementation) == 11,
            "source-push inventory differs")
    contract_raw, contract_binding = source_blob(schedule["unchanged_gate_bindings"]["state_contract"], checkout)
    require(sha(contract_raw) == CONTRACT_SHA, "reviewed full-reset comparison differs")
    reference_raw, reference_binding = source_blob(inventory["input_bindings"]["e004_full_reset_reference"], checkout)
    require(reference_binding == schedule["unchanged_gate_bindings"]["e004_full_reset_reference"],
            "frozen reset-reference bindings differ")
    require(launch["input_bindings"]["e004_reset_reference"]["sha256"] == sha(reference_raw),
            "launch used a different reset reference")
    reference = json.loads(reference_raw)
    contract_function = function(ast.parse(contract_raw), "compare_full_reset_to_e004")
    threshold_nodes = [
        node.value for node in contract_function.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "thresholds" for target in node.targets)
    ]
    require(len(threshold_nodes) == 1 and isinstance(threshold_nodes[0], ast.Dict), "reset thresholds differ")
    thresholds = {ast.literal_eval(key): value
                  for key, value in zip(threshold_nodes[0].keys, threshold_nodes[0].values, strict=True)}
    radius = ast.literal_eval(thresholds["max_object_position_delta_m_inclusive"])
    require(radius == 0.005, "historical object-position gate is not the reviewed inclusive 5 mm bound")

    controller = read_bound(EXPORT / "controller-verification.json", {
        "bytes": 1790, "sha256": "6a82d78d2c85221e0f23dcf73a63ee980c1c078770979a292770c364ab65465c",
    })
    require(controller["failure_report"] == inventory["failure_report"]
            and controller["json_pointer"] == "/controller_source_verification_before_AppLauncher",
            "historical recorder-verification provenance differs")
    expected_controller = {
        name: {key: row[key] for key in ("path", "bytes", "sha256")}
        for name, row in schedule["controller_source_bindings"].items()
    }
    require(controller["value"] == expected_controller, "recorded prelaunch controller hashes differ")

    lifecycles = inventory["environment_lifecycle"]
    require(len(lifecycles) == 20 and [row["environment_ordinal"] for row in lifecycles] == list(range(1, 21)),
            "completed-reset lifecycle inventory differs")
    require(len({row["label"] for row in lifecycles}) == 20
            and all(row.get("fresh_reset_completed_in_this_environment") is True for row in lifecycles),
            "cannot infer a reset bound from an absent/failed completion marker")
    producer_tree = ast.parse(producer)
    control_flow = {
        name: {"first_line": node.lineno, "last_line": node.end_lineno}
        for name in ("_capture_state", "_fresh_reset_and_gate", "_open_exact_stage_environment")
        for node in [function(producer_tree, name)]
    }

    # Widen, never tighten, the source gate to cover floating-point arithmetic.
    guard = 16 * math.ulp(1.0)
    bounds = {}
    centers = {}
    for actor in ACTORS:
        center = tuple(reference["rigid_objects"][actor]["root_position"]["values"])
        require(len(center) == 3 and all(math.isfinite(value) and abs(value) < 1 for value in center),
                "reference exceeds the reviewed finite unit-scale numeric domain")
        centers[actor] = center
        bounds[actor] = RootPositionBounds(
            tuple(math.nextafter(value - radius - guard, -math.inf) for value in center),
            tuple(math.nextafter(value + radius + guard, math.inf) for value in center),
            FRAME.frame_id,
        )
    for snapshot in inventory["partial_snapshots"]:
        if not snapshot["json_pointer"].endswith("/fresh_reset"):
            continue
        for actor, bound in bounds.items():
            position = snapshot["objects"][actor]["position_world_m"]
            require(all(lo <= value <= hi for lo, value, hi in zip(bound.lower_m, position, bound.upper_m, strict=True)),
                    "a retained fresh-reset observation contradicts its source-implied bound")

    previous = compile_observed_proof()
    require(previous == json.loads((FRAMES / "observed-root-nonmatches.json").read_text()),
            "prerequisite native-frame/observed-root proof no longer reproduces")
    frame_manifest = json.loads((FRAMES / "manifest.json").read_text())
    results = []
    for row in frame_manifest["records"]:
        capture = read_bound(FRAMES / row["export_path"], {
            "bytes": row["capture_bytes"], "sha256": row["capture_sha256"],
        })
        roots = {actor: RootPosition(
            tuple(capture["objects"][actor]["root_position_env_local_xyz_m"]), FRAME.frame_id,
        ) for actor in ACTORS}
        result = prove_bounded_root_nonmatch(bounds, roots, required_actors=ACTORS, frame=FRAME)
        results.append({
            "candidate_id": row["candidate_id"], "capture_sha256": row["capture_sha256"],
            "status": result.status, "reason": result.reason,
            "minimum_component_separations_m": dict(result.max_component_differences_m),
            "completed_fresh_reset_lifecycles_compared": len(lifecycles),
        })
    return {
        "schema_version": "sgw-01-r005-gate-bounded-reset-exclusions-v1",
        "producer_sha256": sha(Path(__file__).read_bytes()),
        "primitive_sha256": sha((ROOT / "experiments/workshops/spatial_grounding_v1/historical_root_bounds.py").read_bytes()),
        "point_primitive_sha256": previous["primitive_sha256"],
        "failure_report": inventory["failure_report"],
        "inventory_sha256": sha(INVENTORY.read_bytes()),
        "launch_sha256": sha((EXPORT / "launch.json").read_bytes()),
        "target_validation_sha256": sha((EXPORT / "target-validation.json").read_bytes()),
        "historical_controller_verification_sha256": sha((EXPORT / "controller-verification.json").read_bytes()),
        "historical_recorder_binding": controller["value"]["robolab_post_step_end_effector_pose_recorder"],
        "recorder_attestation_boundary": (
            "The recorded pre-AppLauncher check rehashed the entire basic_recorders.py, "
            "including InitialStateRecorder and PostStepStatesRecorder, not just the named "
            "EEF recorder. It does not attest historical IsaacLab get_state implementation "
            "bytes or recover missing materialization records."
        ),
        "source_commit": REVISION,
        "source_bindings": {
            "wrapper": wrapper_binding, "producer": producer_binding, "candidate_schedule": schedule_binding,
            "source_push_gate": gate_binding, "state_contract": contract_binding, "reset_reference": reference_binding,
        },
        "source_control_flow": control_flow,
        "state_contract_enforcement": (
            "The hash-bound outer launcher checks exact HEAD and clean git status before launch. "
            "The helper is tracked at that revision and imported through the recorded study-first "
            "PYTHONPATH. It is named in unchanged_gate_bindings, not individually rehashed in "
            "the 11-entry source-push implementation inventory. This is a clean pinned source "
            "contract, not an independent imported-module byte attestation."
        ),
        "reset_bound_implication": (
            "_capture_state reads every MOVABLE object's native root_pos_w. _fresh_reset_and_gate "
            "raises unless the full reset comparison passes. The lifecycle completion marker is "
            "set only after that function returns. The inclusive Euclidean 5 mm object bound "
            "therefore encloses each coordinate of that completed settled reset."
        ),
        "common_frame_id": FRAME.frame_id,
        "frame_basis": (
            "Historical bounds are on world-root values by the producer's comparison, without "
            "using HDF5. The prerequisite native proof checks the relative getter and zero "
            "origins for these four captures, placing their roots in the same world axes."
        ),
        "historical_euclidean_position_bound_m": radius,
        "outward_numeric_guard_m": guard,
        "sgw_componentwise_reset_tolerance_m": 0.003,
        "bounds": {actor: {
            "reference_root_position_m": centers[actor],
            "lower_m": bound.lower_m, "upper_m": bound.upper_m,
        } for actor, bound in bounds.items()},
        "completed_fresh_reset_lifecycles": [
            {key: row[key] for key in ("environment_ordinal", "label", "candidate_rank", "stage", "role")}
            for row in lifecycles
        ],
        "candidates": results,
        "comparison_count": len(results) * len(lifecycles),
        "nonmatch_count": sum(row["status"] == "nonmatch" for row in results) * len(lifecycles),
        "historical_population_coverage_complete": False, "release_permitted": False,
        "model_requests": 0, "behavioral_episodes": 0,
        "claim_boundary": (
            "These are source-implied intervals for 20 completed settled fresh resets, not "
            "20 recovered point snapshots. They overlap previously retained rank4 reset evidence "
            "and are not additive population counts. They say nothing about later constructed "
            "states, initialization transients, missing rank1-3 materialization payloads, or "
            "unaccounted historical cohorts. No historical result is relabeled or released."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compile_proof()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
