"""Reproduce a bounded four-candidate/100-snapshot proof, not cohort coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.workshops.spatial_grounding_v1.historical_root_nonmatch import (
    CommonFrameContract, RootPosition, prove_root_position_nonmatch,
)
from tools.audit_sgw_historical_lineage import audit, require, selected


ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "artifacts/workshops/spatial_grounding_v1/infrastructure"
HISTORY = INFRA / "historical-lineage-20260923bv"
FRAMES = INFRA / "native-root-frames-20260923bw"
PREFIX = INFRA / "family-partition-20260923bt-prefix-bv"
FRAME = CommonFrameContract("robolab_environment_local_world_axes_m")
ACTORS = ("rubiks_cube", "bowl")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_proof() -> dict:
    audited = audit(
        ROOT, HISTORY, INFRA / "historical-state-fields-20260923bl",
        ROOT / "handoff/k8s/sgw01-ali-historical-lineage-inputs-20260923bv.json",
    )
    require(
        json.dumps(audited, sort_keys=True) == json.dumps(json.loads((HISTORY / "source_audit.json").read_text()), sort_keys=True),
        "historical source audit no longer reproduces",
    )
    require(sha(PREFIX / "manifest.json") == "a6d52dd182367ad1b91634b745a5f3e19cc1e9959bcbabdbc8bfe66b218ccb0b",
            "native prefix differs from its retained identity")
    history_manifest = json.loads((HISTORY / "manifest.json").read_text())
    observations = []
    for row in history_manifest["records"]:
        payload = selected(HISTORY, row)
        frames = {item["json_pointer"]: item["value"] for item in payload["frame_identity"]}
        snapshots = {}
        for group in ("fresh_reset_objects", "candidate_state_objects"):
            for item in payload[group]:
                prefix, separator, actor = item["json_pointer"].rpartition("/objects/")
                require(bool(separator), "invalid historical object pointer")
                require(actor not in snapshots.setdefault(prefix, {}), "duplicate historical actor")
                snapshots[prefix][actor] = item["value"]
        for pointer, objects in snapshots.items():
            frame = frames[pointer + "/base_link_to_eef_frame_identity"]
            require(frame["passed"] is True and frame["scene_env_origin_world_m"] == [0, 0, 0],
                    "snapshot is outside the audited zero-origin frame contract")
            observations.append({
                "cohort_id": row["cohort_id"], "snapshot_pointer": pointer,
                "roots": {name: RootPosition(tuple(objects[name]["position_world_m"]), FRAME.frame_id)
                          for name in ACTORS},
            })
    require(len(observations) == 100, "named snapshot population differs")
    frame_manifest = json.loads((FRAMES / "manifest.json").read_text())
    require(frame_manifest["common_frame_id"] == FRAME.frame_id, "native frame differs")
    getter = frame_manifest["native_getter"]
    require(getter["commit"] == "0aef241fb088ca21bb4ebd24448940ed56620d17"
            and getter["path"] == "robolab/core/world/world_state.py"
            and getter["sha256"] == "a4c12dc07673b0733c53990244e86577d16fa98888772a6c93934903f9699bbf",
            "native actor-root getter differs from the audited source")
    prefix_manifest = json.loads((PREFIX / "manifest.json").read_text())
    candidates = {}
    for slot in prefix_manifest["records"]:
        for item in slot["retained_files"]:
            path = PREFIX / item["export_path"]
            require(sha(path) == item["sha256"] and path.stat().st_size == item["bytes"],
                    "retained native prefix file mismatch")
            if path.name == "candidate.json":
                candidate = json.loads(path.read_text())
                candidates[candidate["candidate_id"]] = (candidate, item["sha256"])
    require(len(candidates) == len(frame_manifest["records"]) == 4
            and {item["candidate_id"] for item in frame_manifest["records"]} == set(candidates),
            "native frame candidate inventory differs")
    results = []
    for row in frame_manifest["records"]:
        capture_path = (FRAMES / row["export_path"]).resolve()
        require(capture_path.is_relative_to(FRAMES.resolve()), "capture path escapes export")
        require(sha(capture_path) == row["capture_sha256"]
                and capture_path.stat().st_size == row["capture_bytes"], "capture byte/hash mismatch")
        capture = json.loads(capture_path.read_text())
        candidate, candidate_sha = candidates[row["candidate_id"]]
        require(candidate_sha == row["candidate_sha256"]
                and candidate["metadata"]["candidate_capture_sha256"] == row["capture_sha256"],
                "candidate/capture binding differs")
        require(capture["environment_origin_world_xyz_m"] == [0, 0, 0]
                and capture["robolab_commit"] == getter["commit"]
                and capture["study_source_commit"] == "68b9e1004ef289d320bfbdbfdf272b2c3cd9440d",
                "native capture frame differs")
        roots = {}
        for name in ACTORS:
            position = capture["objects"][name]["root_position_env_local_xyz_m"]
            require(position == candidate["object_poses"][name]["position_m"], "candidate roots differ")
            roots[name] = RootPosition(tuple(position), FRAME.frame_id)
        comparisons = []
        for old in observations:
            result = prove_root_position_nonmatch(old["roots"], roots, required_actors=ACTORS, frame=FRAME)
            comparisons.append({
                "cohort_id": old["cohort_id"], "snapshot_pointer": old["snapshot_pointer"],
                "status": result.status, "reason": result.reason,
                "max_component_differences_m": dict(result.max_component_differences_m),
            })
        results.append({
            "candidate_id": row["candidate_id"], "candidate_sha256": candidate_sha,
            "comparison_count": len(comparisons),
            "nonmatch_count": sum(item["status"] == "nonmatch" for item in comparisons),
            "unresolved_count": sum(item["status"] == "unresolved" for item in comparisons),
            "comparisons": comparisons,
        })
    require(len(results) == 4, "native candidate inventory differs")
    return {
        "schema_version": "sgw-01-named-observed-root-nonmatches-v1",
        "historical_source_audit_sha256": sha(HISTORY / "source_audit.json"),
        "native_frame_manifest_sha256": sha(FRAMES / "manifest.json"),
        "native_prefix_manifest_sha256": sha(PREFIX / "manifest.json"),
        "producer_sha256": sha(Path(__file__)),
        "primitive_sha256": sha(ROOT / "experiments/workshops/spatial_grounding_v1/historical_root_nonmatch.py"),
        "common_frame_id": FRAME.frame_id, "required_actors": list(ACTORS),
        "historical_snapshot_count": len(observations), "candidates": results,
        "historical_population_coverage_complete": False, "release_permitted": False,
        "model_requests": 0, "behavioral_episodes": 0,
        "claim_boundary": (
            "A separated required actor root is sufficient to disprove a within-reset-tolerance "
            "duplicate of that named snapshot, independently of missing centroid offsets. "
            "These 100 snapshots are not asserted to exhaust their cohorts or all historical "
            "layouts. No family fixture or learned-policy branch is released."
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
