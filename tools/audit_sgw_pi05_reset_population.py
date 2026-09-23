"""Audit historically source-bound V3-B002 final-cell reset evidence only."""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import subprocess

from experiments.v3.pi05_phase_b.contract import canonical_json_bytes
from experiments.workshops.spatial_grounding_v1.historical_root_nonmatch import _validated_position
from tools.audit_sgw_historical_lineage import require
from tools.prove_sgw_r005_reset_bounds import function, read_bound, sha
from tools.vla_wam_v3_episode_schema import derive_initial_state_sha256

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002"
EXPORT = ROOT / "artifacts/workshops/spatial_grounding_v1/infrastructure/historical-pi05-resets-20260923cf"
RECOVERY_BINDING = {
    "bytes": 267956, "sha256": "9ad3e91a3e16db1d24a18ebce7231d181d158b0a5f48fcbd8a21d58891a13c94",
}
REVISION = "636a33eedb9e4a9920f4ff357388099fba78a108"
RUNTIME_ID = "6e5e500f5efee2d7576d090a8ec544f28e47af4db6e797c1bb514a16a43fc944"
SOURCE = "experiments/v3/pi05_phase_b/robolab_bridge.py"


def _git(path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), "cat-file", "blob", f"{REVISION}:{path}"])


def _bound_bytes(path: Path, binding: dict) -> bytes:
    raw = path.read_bytes()
    require(len(raw) == binding["bytes"] and sha(raw) == binding["sha256"], "bound input bytes differ")
    return raw


def _source_contract(runtime: dict) -> dict:
    source_runtime = _git("experiments/v3/pi05_phase_b/runtime.py")
    tree = ast.parse(source_runtime)
    definitions = [
        node.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ADAPTER_FILES" for target in node.targets)
    ]
    require(len(definitions) == 1, "historical adapter list is ambiguous")
    paths = ast.literal_eval(definitions[0])
    require(isinstance(paths, tuple) and len(paths) == 11 and all(isinstance(p, str) for p in paths),
            "historical adapter list shape differs")
    wrappers = subprocess.check_output([
        "git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", REVISION,
        "experiments/v3/pi05_phase_b/task_files",
    ], text=True).splitlines()
    wrappers = sorted(path for path in wrappers if path.endswith(".py"))
    require(len(wrappers) == 5, "historical task wrapper population differs")
    bindings = [{"path": path, "sha256": sha(_git(path))} for path in (*paths, *wrappers)]
    digest = sha(canonical_json_bytes(bindings))
    require(digest == runtime["phase_b_adapter_contract_sha256"],
            "historical runtime does not bind reconstructed adapter sources")
    bridge = _git(SOURCE)
    bridge_tree = ast.parse(bridge)
    methods = {
        name: [node.lineno, node.end_lineno]
        for name in ("_inverse_rotate", "_sample", "reset", "_fixture_match", "write_reset_attestation")
        for node in [function(bridge_tree, name)]
    }
    return {
        "git_commit": REVISION, "adapter_contract_sha256": digest,
        "source_files": bindings, "bridge_sha256": sha(bridge),
        "source_method_lines": methods,
        "historically_hash_bound": True,
        "coordinate_contract": (
            "Named cube/bowl root_pos_w minus the same robot root_pos_w, then inverse rotation "
            "by the normalized robot wxyz quaternion. Not AABB or geometric centers."
        ),
        "reset_contract": (
            "Two logical pre-action reset calls, one physical reset, one 60-step settle plus "
            "15-step stability window; second logical reset returns the cached result. "
            "Attestation requires the final settled roots to match the bound fixture within "
            "3mm per component and precedes the first model request."
        ),
    }


def compile_audit() -> dict:
    recovered = read_bound(EXPORT / "resets.json", RECOVERY_BINDING)
    require(recovered["schema_version"] == "sgw-01-pi05-v3b002-final-reset-export-v1",
            "recovery schema differs")
    output = json.loads((COHORT / "results/pi05_v3b002_output_manifest.json").read_bytes())
    require(recovered["source_episodes"] == output["files"]["episodes"], "source episode binding differs")
    episode_bytes = _bound_bytes(COHORT / "results/pi05_v3b002_episodes.jsonl", output["files"]["episodes"])
    episode_rows = [json.loads(line) for line in episode_bytes.splitlines()]
    episodes = {row["registered_cell_id"]: row for row in episode_rows}
    registration_path = COHORT / "pi05_mirror_v3b002_manifest.json"
    registration = json.loads(registration_path.read_bytes())
    require(sha(registration_path.read_bytes()) == output["release_manifest_sha256"],
            "registration differs from completed cohort release")
    queue_binding = registration["files"]["cells"]
    queue_bytes = _bound_bytes(COHORT / queue_binding["path"], queue_binding)
    queue_rows = [json.loads(line) for line in queue_bytes.splitlines()]
    queue = {row["cell_id"]: row for row in queue_rows}
    records = recovered["entries"]
    require(len(records) == len(episode_rows) == len(episodes) == len(queue_rows) == len(queue) == 108
            and len({row["registered_cell_id"] for row in records}) == 108
            and {row["registered_cell_id"] for row in records} == set(episodes) == set(queue),
            "final-cell population differs from frozen registration")

    runtime = json.loads((COHORT / "gates/runtime_identity.json").read_bytes())
    body = {key: value for key, value in runtime.items() if key != "runtime_identity_sha256"}
    require(runtime["runtime_identity_sha256"] == RUNTIME_ID == sha(canonical_json_bytes(body))
            and runtime["study_git_commit"] == REVISION
            and runtime["release_manifest_sha256"] == output["release_manifest_sha256"],
            "historical runtime identity differs")
    source = _source_contract(runtime)
    points, distances = [], []
    for row in records:
        cell_id = row["registered_cell_id"]
        episode, cell = episodes[cell_id], queue[cell_id]
        require(episode["behavioral_result_valid"] is True
                and episode["runtime_identity"]["sha256"] == RUNTIME_ID, "episode runtime or validity differs")
        require(row["arm"] == episode["phase_b_arm"] == cell["arm"]
                and row["environment_seed"] == episode["environment_seed"] == cell["environment_seed"]
                and row["relation"] == episode["requested_relation"] == cell["relation"],
                "registered reset identity differs")
        sample = row["sample"]
        require(sample == episode["steps"][0] and type(sample["action_step"]) is int
                and sample["action_step"] == 0, "retained sample is not the bound initial state")
        require(row["measurement_frame"] == episode["measurement_frame"]
                == "robot_base_object_minus_reference_xyz_m", "measurement frame differs")
        require(row["initial_state_sha256"] == episode["initial_state_sha256"]
                == derive_initial_state_sha256({"measurement_frame": row["measurement_frame"], "steps": [sample]}),
                "initial state digest differs")
        attestation = row["reset_attestation"]
        binding, value = attestation["binding"], attestation["value"]
        require(binding == episode["source_artifacts"]["reset_attestation"], "reset attestation binding differs")
        original = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        require(sha(original) == binding["sha256"] and len(original) == binding["bytes"],
                "original reset attestation bytes differ")
        fingerprint = {key: child for key, child in value.items() if key != "reset_fingerprint_sha256"}
        require(value["reset_fingerprint_sha256"] == episode["reset_fingerprint_sha256"]
                == sha(canonical_json_bytes(fingerprint)), "reset fingerprint differs")
        expected = {
            "schema_version": "vla-wam-shared-v3b-pi05-live-reset-v1",
            "amendment_id": "V3-B002", "model_id": "pi05_current_stack_droid",
            "registered_cell_id": cell_id, "runtime_identity_sha256": RUNTIME_ID,
            "arm": row["arm"], "environment_seed": row["environment_seed"], "relation": row["relation"],
            "initial_state_sha256": row["initial_state_sha256"], "fixture_sha256": cell["fixture_sha256"],
            "fixture_candidate_sha256": "e1799b815da41f9a08a4000a360c4958003269fed27e2abe75b273519e4d1c88",
            "release_fingerprint_sha256": episode["release_fingerprint_sha256"],
            "passed": True, "physical_reset_calls": 1, "runner_pre_action_reset_calls": 2,
            "settle_gate_runs": 1, "settle_steps": 60, "stable_window_steps": 15,
            "duplicate_second_reset_idempotent": True, "model_request_count_before_attestation": 0,
        }
        require(all(type(value.get(key)) is type(wanted) and value[key] == wanted
                    for key, wanted in expected.items()), "attested reset contract differs")
        cube, bowl = (_validated_position(sample[key]) for key in ("object_xyz", "reference_xyz"))
        require(cube is not None and bowl is not None, "root vectors are invalid")
        points.append((cube, bowl))
        distances.append(math.dist(cube, bowl))
    return {
        "schema_version": "sgw-01-pi05-v3b002-final-reset-audit-v1",
        "producer_sha256": sha(Path(__file__).read_bytes()),
        "recovery": RECOVERY_BINDING, "source_episodes": output["files"]["episodes"],
        "registration_sha256": output["release_manifest_sha256"],
        "queue_sha256": queue_binding["sha256"], "runtime_identity_sha256": RUNTIME_ID,
        "source_contract": source,
        "final_cell_count": 108, "original_reset_attestations_reconstructed": 108,
        "post_settle_initial_root_pairs": 108, "distinct_numerical_root_pairs": len(set(points)),
        "declared_frame_root_separation_range_m": [min(distances), max(distances)],
        "complete_final_manifest_cell_selection": True,
        "historical_population_coverage_complete": False,
        "cross_frame_comparison_qualified": False, "proved_historical_nonmatches_added": 0,
        "new_model_requests": 0, "new_behavioral_episodes": 0, "release_permitted": False,
        "claim_boundary": (
            "The final 108 V3-B002 cells and their post-settle initial roots are bound to original "
            "reset attestations and a historically anchored 16-file adapter contract. The second "
            "logical reset is idempotent, not another independent layout. Constructor/settle "
            "states, model-blind preflights and infrastructure attempts remain outside this "
            "selection. No global coverage, numerical cross-frame qualification or SGW release."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compile_audit()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
