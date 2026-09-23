"""Audit source-bound Phase-A resets and their conditional DIST000 separation."""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from experiments.workshops.spatial_grounding_v1.historical_root_nonmatch import RootPosition, _validated_position
from experiments.workshops.spatial_grounding_v1.historical_root_separation import (
    MetricFrameContract, prove_root_separation_nonmatch,
)
from tools.audit_sgw_historical_lineage import require
from tools.prove_sgw_r005_reset_bounds import function, read_bound, sha
from tools import prove_sgw_pi05_root_separation as baseline
from tools.vla_wam_v3_episode_schema import derive_initial_state_sha256

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "artifacts/workshops/spatial_grounding_v1/infrastructure/historical-pi05-phase-a-20260923ci"
CATALOG = ROOT / "artifacts/vla_wam_shared_v3/results/pi05_current_stack_droid_phase_a_evidence_hash_manifest.json"
CATALOG_BINDING = {"bytes": 45987, "sha256": "dca633cfa74ff93926cb1d010d5a329908a02d81181f23cca4969c5d5f1f3cfd"}
RECOVERY_BINDING = {"bytes": 257095, "sha256": "a123a7ca525641b231c5d04311cefbcad8059f931c01952e657d2ce206c43bcb"}
REVISION = "0b24153a2ba1fff9173fc47937ad291d24959e72"
BRIDGE_SHA = "e65bede7466c6d9ddb49ab37f4b96f5e3945830f30005045352e320645759589"
FRAME = "pi05_phase_a_registered_native_robot_base_root_m"


def source_contract(runtime: dict) -> dict:
    require(runtime["study_repository_commit"] == REVISION, "registered source revision differs")
    sources = {**runtime["adapter_source_sha256"], **runtime["frozen_v2_source_sha256"]}
    require(len(sources) == 8, "registered source population differs")
    blobs = {}
    for path, expected in sources.items():
        raw = subprocess.check_output(["git", "-C", str(ROOT), "cat-file", "blob", f"{REVISION}:{path}"])
        require(sha(raw) == expected, f"historical source hash differs: {path}")
        blobs[path] = raw
    bridge = (EXPORT / "runtime_bridge.py").read_bytes()
    require(sha(bridge) == runtime["runtime_instrumentation"]["sha256"] == BRIDGE_SHA,
            "actual historical runtime instrumentation differs")
    original = ast.parse(blobs["experiments/v3/pi05_droid/robolab_bridge.py"])
    actual = ast.parse(bridge)
    names = ("_quat_inverse_rotate_wxyz", "_sample")
    for name in names:
        require(ast.dump(function(original, name), include_attributes=False)
                == ast.dump(function(actual, name), include_attributes=False),
                "runtime root getter or inverse rotation differs from the registered source")
    return {
        "historically_hash_bound": True, "registered_git_commit": REVISION,
        "registered_source_files": sources, "runtime_instrumentation": runtime["runtime_instrumentation"],
        "runtime_bridge_bytes": len(bridge), "root_getter_and_rotation_ast_match_registered_source": True,
        "runtime_method_lines": {
            name: [node.lineno, node.end_lineno] for name in (*names, "reset")
            for node in [function(actual, name)]
        },
        "coordinate_contract": "Named cube/bowl root_pos_w minus the same robot root, inverse-rotated by its normalized wxyz quaternion; not geometric centers.",
        "reset_contract": "Two real pre-action resets: preserve setup_reset0 separately, then the canonical behavioral stream. The attested runtime rejects a third reset or any reset after a post-action sample.",
    }


def compile_proof() -> dict:
    catalog = read_bound(CATALOG, CATALOG_BINDING)
    recovered = read_bound(EXPORT / "resets.json", RECOVERY_BINDING)
    require(recovered["schema_version"] == "sgw-01-pi05-phase-a-reset-export-v1"
            and recovered["catalog_sha256"] == CATALOG_BINDING["sha256"], "recovery/catalog differs")
    require(recovered["runtime_identity"] == catalog["runtime_identity"], "runtime binding differs")
    runtime = read_bound(EXPORT / "runtime_identity.json", catalog["runtime_identity"])
    require(recovered["runtime_instrumentation"] == runtime["runtime_instrumentation"],
            "runtime instrumentation binding differs")
    source = source_contract(runtime)
    queue_raw = (ROOT / "artifacts/vla_wam_shared_v3/phase_a_cells.jsonl").read_bytes()
    require(sha(queue_raw) == runtime["phase_a_queue_sha256"], "frozen Phase-A queue differs")
    queue = {
        row["cell_id"]: row for line in queue_raw.splitlines() for row in [json.loads(line)]
        if row["model_id"] == "pi05_current_stack_droid" and row["status"] == "authorized_new"
    }
    pairs = recovered["pairs"]
    require(len(pairs) == catalog["pair_count"] == 27
            and {p["seed"] for p in pairs} == set(range(8303, 8330)), "final pair population differs")
    catalog_pairs = {p["pair_id"]: p for p in catalog["pairs"]}
    seen, points = set(), []
    for pair in pairs:
        registered = catalog_pairs[pair["pair_id"]]
        require(pair["manifest_binding"] == registered["pair_evidence_manifest"]
                and pair["raw_pair"] == registered["behavioral_jsonl"]
                and pair["seed"] == registered["seed"], "original pair binding differs")
        manifest = read_bound(EXPORT / pair["manifest_export_path"], pair["manifest_binding"])
        bindings = {row["path"]: row for row in manifest["files"]}
        require(len(bindings) == len(manifest["files"])
                and manifest["runtime_identity_sha256"] == catalog["runtime_identity"]["sha256"]
                and bindings[runtime["runtime_instrumentation"]["path"]]["sha256"] == BRIDGE_SHA,
                "pair runtime/producer binding differs")
        require(pair["preflight"] == bindings[pair["preflight"]["path"]]
                and pair["guard_log"] == bindings[pair["guard_log"]["path"]], "launch binding differs")
        command = pair["launch_event"]["command"]
        index = command.index("/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python")
        require(command[index + 1] == runtime["runtime_instrumentation"]["path"]
                and command[command.index("--environment-seed") + 1] == str(pair["seed"]),
                "bound launch command differs")
        capture_dir = Path(command[command.index("--state-capture-dir") + 1])
        require(len(pair["entries"]) == 2 and {r["relation"] for r in pair["entries"]} == {"left", "right"},
                "matched direction population differs")
        for row in pair["entries"]:
            cell_id, seed, relation = row["registered_cell_id"], row["environment_seed"], row["relation"]
            require(cell_id in queue and cell_id not in seen, "unregistered or duplicate final cell")
            seen.add(cell_id)
            require(queue[cell_id]["environment_seed"] == seed == pair["seed"]
                    and queue[cell_id]["relation"] == relation
                    and row["runtime_identity"]["sha256"] == catalog["runtime_identity"]["sha256"]
                    and row["raw_behavioral_result_valid"] is row["raw_capture_and_partial_streams_equal"] is True,
                    "raw identity or reconciliation differs")
            for key, path in (
                ("capture", capture_dir / f"seed{seed}_{relation}.json"),
                ("canonical_partial_stream", capture_dir / f"seed{seed}_{relation}_states.partial.jsonl"),
                ("setup_partial_stream", capture_dir / f"seed{seed}_{relation}_setup_reset0.partial.jsonl"),
            ):
                require(row[key] == bindings[str(path)], "original reset stream binding differs")
            frame = row["measurement_frame"]
            require(frame == "robot_base_object_minus_reference_xyz_m"
                    and row["initial_state_sha256"] == derive_initial_state_sha256({
                        "measurement_frame": frame, "steps": [row["initial_sample"]],
                    }), "initial-state hash differs")
            setup_raw = (json.dumps(row["setup_sample"], sort_keys=True, separators=(",", ":")) + "\n").encode()
            require(sha(setup_raw) == row["setup_partial_stream"]["sha256"]
                    and len(setup_raw) == row["setup_partial_stream"]["bytes"], "original setup reset bytes differ")
            for kind, sample in (("setup_reset0", row["setup_sample"]), ("behavioral_initial", row["initial_sample"])):
                require(type(sample["action_step"]) is int and sample["action_step"] == 0,
                        "retained point is not a pre-action reset")
                roots = [_validated_position(sample[key]) for key in ("object_xyz", "reference_xyz")]
                require(all(value is not None for value in roots), "invalid recorded root")
                points.append({"registered_cell_id": cell_id, "state_kind": kind, "roots": roots})
    require(seen == set(queue) and len(seen) == catalog["episode_count"] == 54,
            "final selection does not equal the authorized-new registration")

    native_proof = baseline.compile_proof()
    require(native_proof == json.loads((baseline.NATIVE / "proof.json").read_bytes()),
            "existing native API/candidate proof no longer reproduces")
    native = read_bound(baseline.NATIVE / "native_contract.json", baseline.NATIVE_BINDING)
    require(runtime["simulator_version"] == "Isaac Sim 5.0.0.0 / Isaac Lab 2.2.0 / RoboLab 0.2.1"
            and runtime["robolab_commit"] == native["robolab_getter_commit"],
            "registered Phase-A and native API contracts differ")
    capture = read_bound(baseline.NATIVE / "candidate_capture.json", native["files"]["candidate_capture.json"])
    prospective = {
        actor: RootPosition(tuple(capture["objects"][actor]["root_position_env_local_xyz_m"]), baseline.PROSPECTIVE_FRAME)
        for actor in baseline.ACTORS
    }
    certificate = baseline.rotation_roundoff_bound([list(root) for row in points for root in row["roots"]])
    frames = MetricFrameContract(FRAME, baseline.PROSPECTIVE_FRAME,
                                 certificate["per_root_euclidean_error_upper_bound_m"], 0.0)
    comparisons = []
    for row in points:
        roots = {actor: RootPosition(root, FRAME) for actor, root in zip(baseline.ACTORS, row["roots"], strict=True)}
        result = prove_root_separation_nonmatch(roots, prospective, actor_pair=baseline.ACTORS, frames=frames)
        comparisons.append({key: row[key] for key in ("registered_cell_id", "state_kind")} | asdict(result))
    return {
        "schema_version": "sgw-01-pi05-phase-a-reset-separation-proof-v1",
        "producer_sha256": sha(Path(__file__).read_bytes()), "recovery": RECOVERY_BINDING,
        "catalog_sha256": CATALOG_BINDING["sha256"], "source_contract": source,
        "runtime_identity_sha256": catalog["runtime_identity"]["sha256"],
        "queue_sha256": runtime["phase_a_queue_sha256"],
        "native_reference_proof_sha256": sha((baseline.NATIVE / "proof.json").read_bytes()),
        "native_contract_sha256": baseline.NATIVE_BINDING["sha256"],
        "prospective_candidate_id": native_proof["prospective_candidate_id"],
        "final_cell_count": len(seen), "recorded_setup_reset_samples": len(seen),
        "recorded_behavioral_initial_samples": len(seen),
        "distinct_numerical_root_pairs": len({tuple(row["roots"]) for row in points}),
        "frames": asdict(frames), "numerical_certificate": certificate,
        "comparisons": comparisons, "comparison_count": len(comparisons),
        "nonmatches_under_registered_api_contract": sum(r["status"] == "nonmatch" for r in comparisons),
        "unresolved_count": sum(r["status"] == "unresolved" for r in comparisons),
        "historical_native_import_bytes_independently_attested": False,
        "historical_population_coverage_complete": False,
        "new_model_requests": 0, "new_behavioral_episodes": 0, "release_permitted": False,
        "claim_boundary": (
            "54 named final Phase-A cells and54 separately preserved setup-reset snapshots only. "
            "Their actual runtime instrumentation is historically hash-bound, not inferred from a "
            "current checkout. Conditional necessary nonmatches use the recorded native root/metre/"
            "precision API; historical imported native bytes are not independently attested. "
            "No independent-layout count, constructor/intermediate-settle/preflight/infrastructure "
            "population closure, transplantation of this producer identity to V3-D001, or SGW release."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compile_proof()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
