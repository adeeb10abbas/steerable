from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.episode_compiler import (
    EXPORT_SCHEMA,
    build_episode_record,
    compile_pair,
    write_episode,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.layout_contract import load_candidate
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.live_snapshot_adapter import (
    ModelBlindLiveGateAdapter,
    bind_camera_row,
    extract_arm_reset_pose,
    extract_realised_object_poses,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.runtime_contract import (
    RuntimeContractError,
    load_runtime_bundle,
    sha256_file,
    shard_cells,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"


def _bundle():
    registration = ARTIFACT / "registration.json"
    queue = ARTIFACT / "queue.jsonl"
    candidate = ARTIFACT / "layout/candidate.json"
    return load_runtime_bundle(
        registration_path=registration,
        registration_sha256=sha256_file(registration),
        queue_path=queue,
        queue_sha256=sha256_file(queue),
        candidate_path=candidate,
        candidate_sha256=sha256_file(candidate),
    )


class _Tensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value

    def __getitem__(self, index):
        return _Tensor(self.value[index])


class _Object:
    def __init__(self, xyz, yaw=0.0):
        import math

        self.data = SimpleNamespace(
            root_pos_w=_Tensor([list(xyz)]),
            root_quat_w=_Tensor([[math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]]),
        )


class _Env:
    def __init__(self, poses):
        self.scene = {"robot": _Object([0.0, 0.0, 0.0])}
        for name, pose in poses.items():
            self.scene[name] = _Object([pose.x_m, pose.y_m, pose.z_m], pose.yaw_rad)


def _camera_rows(candidate, poses):
    target = poses[candidate.target_object]
    reference = poses[candidate.reference_object]
    rows = {}
    for name in candidate.expected_cameras:
        row = {
            "camera_center_world_m": [target.x_m, target.y_m, target.z_m - 1.0],
            "camera_quaternion_world_wxyz_ros": [1.0, 0.0, 0.0, 0.0],
            "target_center_world_m": [target.x_m, target.y_m, target.z_m],
            "intrinsic_matrix_3x3": [[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]],
            "image_size_wh": [640, 480],
            "reference_bounds_world": {
                "center_world_m": [reference.x_m, reference.y_m, reference.z_m],
                "half_extents_m": [0.01, 0.01, 0.01],
                "yaw_world_rad": reference.yaw_rad,
            },
            "target_instance_visible_pixels": 100,
            "segmentation_source_sha256": "a" * 64,
            "target_projected_pixel_uv": [320.0, 240.0],
            "rgb_source_sha256": (name.encode("utf-8").hex() + "0" * 64)[:64],
            "rgb_source_shape": [480, 640, 3],
            "rgb_source_dtype": "uint8",
        }
        rows[name] = bind_camera_row(row)
    return rows


def _gate(tmp_path: Path, cell):
    bundle = _bundle()
    candidate = load_candidate(bundle.candidate_path, bundle.candidate_sha256)
    poses = candidate.layout(cell.symmetry_level_s)
    env = _Env(poses)
    adapter = ModelBlindLiveGateAdapter(
        bundle=bundle,
        cell=cell,
        snapshot_path=tmp_path / "snapshot.json",
        gate_path=tmp_path / "gate.json",
        minimum_visible_target_pixels=32,
    )
    output = adapter.capture_and_compile(
        env=env,
        observation={"proprio_obs": {"arm_joint_pos": [[0.0] * 7], "gripper_pos": [[1.0]]}},
        scene_object_mapping={name: name for name in candidate.symmetric_poses},
        camera_rows=_camera_rows(candidate, poses),
        settle_stability={
            "settle_steps": 60,
            "stability_window_steps": 15,
            "maxima_by_object": {name: {"linear_speed_m_s": 0.0, "angular_speed_rad_s": 0.0} for name in poses},
        },
    )
    return bundle, output


def test_runtime_bundle_selects_only_new_droid_rows_and_whole_pair_shards():
    bundle = _bundle()
    cells = bundle.droid_new_cells("cosmos3_edge_policy_droid")
    assert len(cells) == 108
    shard = shard_cells(cells, shard_index=1, shard_count=4)
    grouped = {}
    for cell in shard:
        grouped.setdefault(cell.matched_pair_id, set()).add(cell.relation)
    assert grouped and set(map(frozenset, grouped.values())) == {frozenset({"left", "right"})}
    with pytest.raises(RuntimeContractError, match="shard_index"):
        shard_cells(cells, shard_index=4, shard_count=4)


def test_live_adapter_extracts_full_pose_and_blocks_request_until_gate(tmp_path: Path):
    bundle = _bundle()
    cell = next(cell for cell in bundle.droid_new_cells("cosmos3_edge_policy_droid") if cell.symmetry_level_s == 1 and cell.relation == "left")
    candidate = load_candidate(bundle.candidate_path, bundle.candidate_sha256)
    poses = candidate.layout(1.0)
    extracted = extract_realised_object_poses(
        _Env(poses), candidate=candidate, symmetry_level_s=1.0,
        scene_object_mapping={name: name for name in candidate.symmetric_poses},
    )
    assert extracted == poses
    arm = extract_arm_reset_pose({"proprio_obs": {"arm_joint_pos": [[0.0] * 7], "gripper_pos": [[1.0]]}})
    assert len(arm["arm_joint_positions_rad"]) == 7
    adapter = ModelBlindLiveGateAdapter(
        bundle=bundle, cell=cell, snapshot_path=tmp_path / "snapshot.json",
        gate_path=tmp_path / "gate.json", minimum_visible_target_pixels=32,
    )
    with pytest.raises(RuntimeContractError, match="before live gate"):
        adapter.authorize_model_request()
    result = adapter.capture_and_compile(
        env=_Env(poses),
        observation={"proprio_obs": {"arm_joint_pos": [[0.0] * 7], "gripper_pos": [[1.0]]}},
        scene_object_mapping={name: name for name in candidate.symmetric_poses},
        camera_rows=_camera_rows(candidate, poses),
        settle_stability={"settle_steps": 60, "stability_window_steps": 15, "maxima_by_object": {name: {"linear_speed_m_s": 0.0, "angular_speed_rad_s": 0.0} for name in poses}},
    )
    assert adapter.authorize_model_request() == result["gate_sha256"]
    adapter.authorize_behavioral_action()


def _export(tmp_path: Path, *, bundle, cell, gate, relation: str):
    actions = tmp_path / f"{relation}.npy"
    np.save(actions, np.zeros((3, 8), dtype=np.float32), allow_pickle=False)
    video = tmp_path / f"{relation}.mp4"
    video.write_bytes(b"not-empty-test-video")
    runtime = tmp_path / f"{relation}.runtime.json"
    runtime.write_text(json.dumps({"model_id": cell.model_id, "runtime_identity_requirement": cell.row["runtime_identity_requirement"]}))
    sign = 1.0 if relation == "left" else -1.0
    steps = [
        {"action_step": 0, "object_xyz": [0.30, 0.0, 0.08], "reference_xyz": [0.44, 0.0, 0.08], "grippers_open": True, "object_grabbed": False},
        {"action_step": 1, "object_xyz": [0.44, 0.20 * sign, 0.12], "reference_xyz": [0.44, 0.0, 0.08], "grippers_open": False, "object_grabbed": True},
        {"action_step": 2, "object_xyz": [0.44, 0.20 * sign, 0.12], "reference_xyz": [0.44, 0.0, 0.08], "grippers_open": False, "object_grabbed": True},
        {"action_step": 3, "object_xyz": [0.44, 0.20 * sign, 0.08], "reference_xyz": [0.44, 0.0, 0.08], "grippers_open": True, "object_grabbed": False},
    ]
    return {
        "schema_version": EXPORT_SCHEMA,
        "study_id": cell.row["study_id"], "amendment_id": cell.row["amendment_id"],
        "registered_cell_id": cell.cell_id, "registered_cell_sha256": cell.row_sha256,
        "registration_sha256": bundle.registration_sha256, "queue_sha256": bundle.queue_sha256,
        "candidate_sha256": bundle.candidate_sha256, "model_id": cell.model_id,
        "arena": cell.row["arena"], "environment_seed": cell.environment_seed,
        "sampling_seed": cell.sampling_seed, "matched_pair_id": cell.matched_pair_id,
        "requested_relation": relation, "prompt": cell.row["prompt"],
        "prompt_sha256": cell.row["prompt_sha256"], "symmetry_level_s": cell.symmetry_level_s,
        "success_predicate_id": cell.row["success_predicate_id"],
        "runtime_identity_requirement": cell.row["runtime_identity_requirement"],
        "instruction_controller": "static_episode_prompt", "live_scene_gate": {
            "path": gate["gate_path"], "sha256": gate["gate_sha256"],
            "bytes": Path(gate["gate_path"]).stat().st_size,
        },
        "steps": steps, "actions_executed": 3, "requested_success": True,
        "right_censored": False, "final_detached_release": True,
        "executed_action_trace": {"path": str(actions), "sha256": sha256_file(actions), "bytes": actions.stat().st_size},
        "viewport_video": {"path": str(video), "sha256": sha256_file(video), "bytes": video.stat().st_size},
        "runtime_identity": {"path": str(runtime), "sha256": sha256_file(runtime), "bytes": runtime.stat().st_size},
    }


def test_episode_and_pair_compiler_preserve_raw_metrics(tmp_path: Path):
    bundle = _bundle()
    pair_cells = [cell for cell in bundle.droid_new_cells("cosmos3_edge_policy_droid") if cell.environment_seed == 9400 and cell.symmetry_level_s == 1]
    by_relation = {cell.relation: cell for cell in pair_cells}
    episode_paths = {}
    for relation in ("left", "right"):
        gate_dir = tmp_path / relation / "gate"
        gate_dir.mkdir(parents=True)
        _, gate = _gate(gate_dir, by_relation[relation])
        row = build_episode_record(
            export=_export(tmp_path / relation, bundle=bundle, cell=by_relation[relation], gate=gate, relation=relation),
            bundle=bundle, cell=by_relation[relation], output_path=tmp_path / relation / "raw_episode.jsonl",
        )
        assert row["success"] is True and row["cone_entry_sustained"] is True
        assert row["endpoint_shift"] is None and row["action_distinct"] is None
        episode_paths[relation] = tmp_path / relation / "raw_episode.jsonl"
        write_episode(record=row, output=episode_paths[relation])
    pair = compile_pair(left_jsonl=episode_paths["left"], right_jsonl=episode_paths["right"], output=tmp_path / "pair.jsonl")
    assert pair["endpoint_redirection_left_minus_right_m"] == pytest.approx(0.4)
    assert pair["endpoint_shift_right_minus_left_m"] == pytest.approx(-0.4)
    assert pair["action_distinct"] is False
