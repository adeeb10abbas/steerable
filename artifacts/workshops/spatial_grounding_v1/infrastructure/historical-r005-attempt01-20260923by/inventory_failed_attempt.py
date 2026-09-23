"""Read-only bounded inventory; no historical-coverage or coordinate-frame release."""

import hashlib
import io
import json
from pathlib import Path
import sys

import h5py
import numpy as np


def bound_bytes(path, expected, maximum):
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError(f"byte/hash mismatch: {path}")
    if len(raw) > maximum:
        raise ValueError(f"bounded input size exceeded: {path}")
    return raw


binding = {
    "path": "/data/users/ali/vla_wam/raw/v3e006_r005/state_repair/98f0234-a40r06-attempt01/raw/state_construction_failure.json",
    "bytes": 25614537,
    "sha256": "e858a4431bd0451a1e380d66c8c5d05c5486718f5b23388f874d24a7d271ae8e",
}
path = Path(binding["path"])
failure = json.loads(bound_bytes(path, binding, 32 * 1024 * 1024))
assert failure["model_request_count"] == failure["behavioral_episode_count"] == 0
assert failure["partial_stage_evidence"]["candidate_rank"] == 4
available = failure["available_raw_artifacts"]
native = path.parent / "native"
actual_hdf5 = sorted(native.glob("*/data.hdf5"))
expected_hdf5 = {
    key: value for key, value in available.items()
    if key.startswith("native/") and key.endswith("/data.hdf5")
}
assert {p.relative_to(path.parent).as_posix() for p in actual_hdf5} == set(expected_hdf5)
assert len(actual_hdf5) == 8
rows = []
for item in actual_hdf5:
    relative = item.relative_to(path.parent).as_posix()
    expected = expected_hdf5[relative]
    assert expected["path"] == str(item)
    raw = bound_bytes(item, expected, 1024 * 1024)
    with h5py.File(io.BytesIO(raw), "r") as data:
        states = data["data/demo_0/states/rigid_object"]
        initial = data["data/demo_0/initial_state/rigid_object"]
        objects = {}
        for actor in ("banana", "banana_right", "bowl", "rubiks_cube"):
            dataset = states[actor]["root_pose"]
            assert dataset.ndim == 2 and 0 < dataset.shape[0] <= 900 and dataset.shape[1] == 7
            assert dataset.dtype == np.dtype("float32")
            values = dataset[:]
            start = initial[actor]["root_pose"]
            assert start.shape == (2, 7) and start.dtype == np.dtype("float32")
            initial_values = start[:]
            assert np.isfinite(values).all() and np.isfinite(initial_values).all()
            objects[actor] = {
                "state_dataset": dataset.name,
                "state_shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "recorded_position_component_min": values[:, :3].min(axis=0).tolist(),
                "recorded_position_component_max": values[:, :3].max(axis=0).tolist(),
                "initial_dataset": start.name,
                "initial_root_pose_samples": initial_values.tolist(),
            }
        rows.append({"binding": expected, "relative_path": relative, "objects": objects})
partial = failure["partial_stage_evidence"]
snapshots = []
for stage in ("canonical_grasp", "canonical_carry"):
    value = partial[stage]
    for suffix, state in (
        ("ik_solve_environment/fresh_reset", value["ik_solve_environment"]["fresh_reset"]),
        ("materialization_environment/fresh_reset", value["materialization_environment"]["fresh_reset"]),
        ("candidate_state", value["candidate_state"]),
    ):
        snapshots.append({
            "json_pointer": f"/partial_stage_evidence/{stage}/{suffix}",
            "frame_identity": state["base_link_to_eef_frame_identity"],
            "objects": {
                actor: {key: obj[key] for key in (
                    "position_world_m", "quaternion_world_wxyz",
                    "linear_velocity_m_s", "angular_velocity_rad_s",
                )}
                for actor, obj in state["objects"].items()
            },
        })
result = {
    "schema_version": "sgw-01-r005-failed-attempt-native-inventory-v1",
    "failure_report": binding,
    "construction_source": failure["construction_source"],
    "input_bindings": failure["input_bindings"],
    "environment_lifecycle": failure["environment_lifecycle"],
    "retained_partial_rank": 4,
    "partial_snapshots": snapshots,
    "native_hdf5": rows,
    "materialization_hdf5_present": any("__materialization/" in row["relative_path"] for row in rows),
    "model_requests": 0,
    "behavioral_episodes": 0,
    "release_permitted": False,
    "historical_population_coverage_complete": False,
    "hdf5_coordinate_frame_qualified": False,
    "claim_boundary": (
        "The same hash-verified failure bytes bind eight IK-only HDF5 files. Their finite raw "
        "float32 root channels and two initial samples are retained without assigning a common "
        "frame. Six source-defined snapshots survive for rank4. These do not recover rank1-3 "
        "materialization states, diagnostic snapshots, or complete historical populations."
    ),
}
json.dump(result, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
sys.stdout.write("\n")
