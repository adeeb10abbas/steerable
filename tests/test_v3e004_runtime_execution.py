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
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.request0_replay import (
    RESET_CONTRACT_SCHEMA,
    canonical_json_sha256,
    capture_left_observation,
    evidence_envelope,
    replay_left_observation_for_right,
    write_capture_attestation,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.run_droid_queue import (
    _bridge_command,
    _existing_valid_episode,
    _left_first,
    _validate_existing_r001_artifacts,
    _validate_resumed_left_cache,
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
    left_first = _left_first(list(shard))
    assert all(
        left_first[index].relation == "left"
        and left_first[index + 1].relation == "right"
        and left_first[index].matched_pair_id == left_first[index + 1].matched_pair_id
        for index in range(0, len(left_first), 2)
    )
    with pytest.raises(RuntimeContractError, match="shard_index"):
        shard_cells(cells, shard_index=4, shard_count=4)


def test_right_bridge_command_carries_exact_left_cache_digests(tmp_path: Path) -> None:
    bundle = _bundle()
    cell = bundle.cell("v3e004:nano:seed9400:s100:right")
    pair_root = tmp_path / "pair"
    pair_root.mkdir()
    retained = {
        "--request0-observation-cache-sha256": pair_root / "left_request0_observation.npz",
        "--request0-observation-manifest-sha256": pair_root / "left_request0_observation.manifest.json",
        "--request0-reset-contract-sha256": pair_root / "left_request0_reset_contract.json",
    }
    for index, path in enumerate(retained.values()):
        path.write_bytes(f"retained-{index}".encode("utf-8"))
    args = SimpleNamespace(
        bridge_module="example.bridge",
        repo_root=ROOT,
        registration=ARTIFACT / "registration.json",
        queue=ARTIFACT / "queue.jsonl",
        candidate=ARTIFACT / "layout/candidate.json",
        lane_release=tmp_path / "release.json",
        lane_pod_uid="pod-uid",
        lane_gpu_uuid="GPU-uuid",
        model_endpoint_host="127.0.0.1",
        model_endpoint_port=18011,
        request0_replay_amendment=ARTIFACT / "request0_observation_replay_amendment.json",
        request0_replay_amendment_sha256=sha256_file(
            ARTIFACT / "request0_observation_replay_amendment.json"
        ),
        live_orientation_tolerance_amendment=(
            ARTIFACT / "live_orientation_realisation_tolerance_amendment.json"
        ),
        live_orientation_tolerance_amendment_sha256=sha256_file(
            ARTIFACT / "live_orientation_realisation_tolerance_amendment.json"
        ),
        bridge_arg=[],
    )
    command = _bridge_command(
        args=args,
        cell=cell,
        attempt=tmp_path / "attempt",
        registration_sha256=bundle.registration_sha256,
        queue_sha256=bundle.queue_sha256,
        candidate_sha256=bundle.candidate_sha256,
        lane_release_sha256="0" * 64,
        request0_pair_root=pair_root,
    )
    for flag, path in retained.items():
        index = command.index(flag)
        assert command[index + 1] == sha256_file(path)
    assert command[command.index("--request0-mode") + 1] == "replay_right"


def test_resume_never_reruns_existing_left_when_pair_cache_is_missing(tmp_path: Path) -> None:
    pair_root = tmp_path / "request0_pairs" / "pair"
    pair_root.mkdir(parents=True)
    named_paths = {
        "amendment": pair_root / "amendment.json",
        "cache_manifest": pair_root / "left_request0_observation.manifest.json",
        "observation_cache": pair_root / "left_request0_observation.npz",
        "reset_contract": pair_root / "left_request0_reset_contract.json",
        "native_reset_contract": pair_root / "left_request0_reset_contract.json",
        "attestation": pair_root / "left.attestation.json",
    }
    for index, path in enumerate(dict.fromkeys(named_paths.values())):
        path.write_bytes(f"artifact-{index}".encode("utf-8"))
    artifacts = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in named_paths.items()
    }
    cell_root = tmp_path / "cells" / "left"
    attempt = cell_root / "attempt001"
    attempt.mkdir(parents=True)
    episode = attempt / "raw_episode.jsonl"
    episode.write_text(
        json.dumps(
            {
                "requested_relation": "left",
                "symmetry_level_s": 1.0,
                "request0_pair_identity_sha256": "a" * 64,
                "request0_replay": {
                    "schema_version": "vla-wam-shared-v3e004-request0-evidence-envelope-v1",
                    "artifacts": artifacts,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = episode.with_name(episode.name + ".manifest.json")
    manifest.write_text(
        json.dumps({"row_count": 1, "jsonl_sha256": sha256_file(episode)}),
        encoding="utf-8",
    )
    assert _existing_valid_episode(
        cell_root,
        amendment_sha256=sha256_file(
            ARTIFACT / "live_orientation_realisation_tolerance_amendment.json"
        ),
    ) == episode.resolve()
    _validate_existing_r001_artifacts(episode)
    _validate_resumed_left_cache(episode, pair_root)
    named_paths["observation_cache"].unlink()
    assert _existing_valid_episode(
        cell_root,
        amendment_sha256=sha256_file(
            ARTIFACT / "live_orientation_realisation_tolerance_amendment.json"
        ),
    ) == episode.resolve()
    with pytest.raises(RuntimeContractError, match="missing or changed"):
        _validate_existing_r001_artifacts(episode)


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


def _request0_evidence(tmp_path: Path, *, cell, relation: str):
    amendment = ARTIFACT / "request0_observation_replay_amendment.json"
    amendment_sha = sha256_file(amendment)
    pair_root = tmp_path / "request0"
    cache = pair_root / "left.npz"
    manifest = pair_root / "left.manifest.json"
    reset_path = pair_root / "left.reset.json"
    observation = {
        "image_obs": {"head": np.arange(18, dtype=np.uint8).reshape(1, 2, 3, 3)},
        "proprio_obs": {"joint": np.asarray([[0.1, 0.2]], dtype=np.float32)},
    }
    reset = {
        "schema_version": RESET_CONTRACT_SCHEMA,
        "robot": {"joint_position": [0.1, 0.2]},
        "rigid_objects": {"cube": [0.3, 0.0, 0.08]},
        "cameras": {"head": {"shape": [2, 3, 3], "dtype": "uint8"}},
        "observation_contract": {"version": 1},
    }
    reset["reset_contract_sha256"] = canonical_json_sha256(reset)
    if relation == "left":
        captured = capture_left_observation(
            observation=observation,
            reset_contract=reset,
            amendment_path=amendment,
            amendment_sha256=amendment_sha,
            cell_id=cell.cell_id,
            matched_pair_id=cell.matched_pair_id,
            cache_path=cache,
            manifest_path=manifest,
            reset_contract_path=reset_path,
        )
        attestation = pair_root / "left.attestation.json"
        write_capture_attestation(
            amendment_path=amendment,
            amendment_sha256=amendment_sha,
            cell_id=cell.cell_id,
            matched_pair_id=cell.matched_pair_id,
            cache_path=cache,
            manifest_path=manifest,
            reset_contract_path=reset_path,
            observation_payload_sha256=captured["observation_payload_sha256"],
            reset_contract_payload_sha256=captured["reset_contract"]["payload_sha256"],
            attestation_path=attestation,
        )
        observation_sha = captured["observation_payload_sha256"]
        reset_sha = captured["reset_contract"]["payload_sha256"]
        native_reset = reset_path
        mode = "capture_left"
    else:
        attestation = pair_root / "right.attestation.json"
        native_reset = pair_root / "right.reset.json"
        _, replayed = replay_left_observation_for_right(
            native_observation={
                "image_obs": {"head": np.full((1, 2, 3, 3), 99, dtype=np.uint8)},
                "proprio_obs": {"joint": np.asarray([[0.1, 0.2]], dtype=np.float32)},
            },
            native_reset_contract=reset,
            amendment_path=amendment,
            amendment_sha256=amendment_sha,
            cell_id=cell.cell_id,
            matched_pair_id=cell.matched_pair_id,
            cache_path=cache,
            cache_sha256=sha256_file(cache),
            manifest_path=manifest,
            manifest_sha256=sha256_file(manifest),
            reset_contract_path=reset_path,
            reset_contract_file_sha256=sha256_file(reset_path),
            native_reset_contract_path=native_reset,
            attestation_path=attestation,
        )
        observation_sha = replayed["request0_observation_payload_sha256"]
        reset_sha = replayed["right_reset_contract_sha256"]
        mode = "replay_right"
    return evidence_envelope(
        mode=mode,
        amendment_path=amendment,
        cache_path=cache,
        manifest_path=manifest,
        reset_contract_path=reset_path,
        native_reset_contract_path=native_reset,
        attestation_path=attestation,
        observation_payload_sha256=observation_sha,
        reset_contract_payload_sha256=reset_sha,
    )


def _export(tmp_path: Path, *, bundle, cell, gate, relation: str, request0):
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
        "request0_replay": request0,
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
            export=_export(
                tmp_path / relation,
                bundle=bundle,
                cell=by_relation[relation],
                gate=gate,
                relation=relation,
                request0=_request0_evidence(tmp_path, cell=by_relation[relation], relation=relation),
            ),
            bundle=bundle, cell=by_relation[relation], output_path=tmp_path / relation / "raw_episode.jsonl",
        )
        assert row["success"] is True and row["cone_entry_sustained"] is True
        assert row["endpoint_shift"] is None and row["action_distinct"] is None
        if relation == "right":
            row["native_initial_rgb_views"] = {
                **row["native_initial_rgb_views"],
                "head_camera": {
                    **row["native_initial_rgb_views"]["head_camera"],
                    "rgb_source_sha256": "f" * 64,
                },
            }
            row["native_initial_state_sha256"] = "e" * 64
            row["initial_state_sha256"] = "e" * 64
        episode_paths[relation] = tmp_path / relation / "raw_episode.jsonl"
        write_episode(record=row, output=episode_paths[relation])
    pair = compile_pair(left_jsonl=episode_paths["left"], right_jsonl=episode_paths["right"], output=tmp_path / "pair.jsonl")
    assert pair["endpoint_redirection_left_minus_right_m"] == pytest.approx(0.4)
    assert pair["endpoint_shift_right_minus_left_m"] == pytest.approx(-0.4)
    assert pair["action_distinct"] is False
    assert pair["identical_policy_request0_non_language_bytes"] is True
    assert pair["request0_pair_identity_sha256"]
    assert pair["native_initial_rgb_bytes_identical"] is False
    assert pair["left_native_initial_state_sha256"] != pair["right_native_initial_state_sha256"]
