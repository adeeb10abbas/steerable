from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.v3.phase_e.cross_arena_geometry_v3e005.evidence import (
    build_provisional_episode,
    close_pair,
    infrastructure_record,
)
from experiments.v3.phase_e.cross_arena_geometry_v3e005.runtime_contract import (
    E005ContractError,
    canonical_json_bytes,
    canonical_sha256,
    load_object,
    load_registered_bundle,
    shard_seed_blocks,
    validate_candidate_binding,
    validate_model_blind_gate_binding,
)
from experiments.v3.phase_e.cross_arena_geometry_v3e005.run_lingbot_queue import (
    STUDY_ROOT_DEFAULT,
    _bound_gate_scene,
    _frozen_curobo_source,
    _guard_command,
    _normalise_runtime_snapshot,
    _runtime_task_with_snapshot,
    _scene_runtime_modules,
    _worker_command,
    run_execute,
)


ROOT = Path(__file__).resolve().parents[1]


def test_e005_runner_default_resolves_repository_root() -> None:
    assert STUDY_ROOT_DEFAULT == ROOT


def test_e005_worker_uses_hash_bound_runtime_curobo_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "curobo-frozen"
    source = repository / "src"
    (source / "curobo").mkdir(parents=True)
    (source / "curobo" / "__init__.py").write_text("\n")
    extension = source / "curobo" / "curobolib" / "kinematics.so"
    extension.parent.mkdir()
    extension.write_bytes(b"registered-sm120-extension")
    extension_sha = hashlib.sha256(extension.read_bytes()).hexdigest()
    lock = tmp_path / "environment_lock.json"
    lock.write_text(
        json.dumps(
            {
                "curobo_repository": {
                    "path": str(repository),
                    "commit": "d" * 40,
                    "device_gate": {"status": "passed_real_cuda_kernel_execution"},
                    "extensions": [
                        {
                            "path": str(extension),
                            "bytes": extension.stat().st_size,
                            "sha256": extension_sha,
                        }
                    ],
                }
            }
        )
    )
    monkeypatch.setattr(
        "experiments.v3.phase_e.cross_arena_geometry_v3e005.run_lingbot_queue.verify_git_identity",
        lambda *args, **kwargs: None,
    )
    runtime = {
        "environment": {
            "lock_artifact": {
                "path": str(lock),
                "sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            }
        }
    }
    assert _frozen_curobo_source(runtime) == source
    extension.write_bytes(b"drift")
    with pytest.raises(E005ContractError, match="extension byte drift"):
        _frozen_curobo_source(runtime)


def test_e005_lingbot_shards_preserve_all_27_whole_four_cell_seed_blocks() -> None:
    bundle = load_registered_bundle(ROOT)
    shards = [shard_seed_blocks(bundle, shard_index=index, shard_count=5) for index in range(5)]
    flat = [cell for shard in shards for block in shard for cell in block]
    assert len(flat) == 108
    assert len({cell.cell_id for cell in flat}) == 108
    assert {cell.environment_seed for cell in flat} == set(range(9400, 9427))
    for shard in shards:
        for block in shard:
            assert len({cell.environment_seed for cell in block}) == 1
            assert {(cell.symmetry_level, cell.relation) for cell in block} == {
                (0.0, "left"),
                (0.0, "right"),
                (1.0, "left"),
                (1.0, "right"),
            }


def test_e005_lingbot_queue_rejects_prompt_and_predicate_drift(tmp_path: Path) -> None:
    source = ROOT / "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005"
    registration = tmp_path / "registration.json"
    queue = tmp_path / "queue.jsonl"
    registration.write_bytes((source / "registration.json").read_bytes())
    rows = [json.loads(line) for line in (source / "queue.jsonl").read_text().splitlines()]
    rows[0]["success_predicate_id"] = "DROID_45_degree_cone"
    queue.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    # The frozen queue digest rejects mutation before any semantic fallback.
    with pytest.raises(E005ContractError, match="queue hash drift"):
        load_registered_bundle(ROOT, registration_path=registration, queue_path=queue)


def test_e005_candidate_and_zero_request_gate_are_hash_bound() -> None:
    bundle = load_registered_bundle(ROOT)
    candidate_sha = "a" * 64
    candidate = {
        "study_id": bundle.registration["study_id"],
        "amendment_id": "V3-E005",
        "arena": "robotwin",
        "model_id": "lingbot_va_robotwin",
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "candidate_sha256": candidate_sha,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }
    validate_candidate_binding(candidate, bundle=bundle, candidate_sha256=candidate_sha)
    scene_ids = [f"robotwin_pair_{number:02d}" for number in range(3, 10)]
    resets = [
        {
            "symmetry_level_s": level,
            "relation": relation,
            "repeat_index": repeat,
            "validation": {"passed": True},
        }
        for level in (0.0, 1.0)
        for repeat in range(2)
        for relation in ("left", "right")
    ]
    gate = {
        "schema_version": "vla-wam-shared-v3e005-seven-scene-model-blind-gate-v1",
        "study_id": bundle.registration["study_id"],
        "amendment_id": "V3-E005",
        "arena": "robotwin",
        "model_id": "lingbot_va_robotwin",
        "status": "passed_model_blind_before_behavior",
        "passed": True,
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "candidate_sha256": candidate_sha,
        "simulator_repository_commit": "0aeea2d669c0f8516f4d5785f0aa33ba812c14b4",
        "scene_count": 7,
        "reset_count": 56,
        "scenes": {
            scene_id: {
                "scene_id": scene_id,
                "resolved_layouts": {"0.00": {}, "1.00": {}},
                "reset_evidence": copy.deepcopy(resets),
            }
            for scene_id in scene_ids
        },
        "reset_gate": {"repeat_count_per_scene_level_relation": 2},
        "model_request_count": 0,
        "model_action_request_count": 0,
        "behavioral_episode_count": 0,
    }
    validate_model_blind_gate_binding(gate, bundle=bundle, candidate_sha256=candidate_sha)
    changed = dict(gate, behavioral_episode_count=1)
    with pytest.raises(E005ContractError, match="behavioral_episode_count"):
        validate_model_blind_gate_binding(changed, bundle=bundle, candidate_sha256=candidate_sha)


def test_e005_worker_command_is_one_registered_cell_and_uses_native_guard(tmp_path: Path) -> None:
    bundle = load_registered_bundle(ROOT)
    cell = bundle.seed_block(9400)[0]
    args = SimpleNamespace(
        study_root=ROOT,
        registration=ROOT
        / "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005/registration.json",
        queue=ROOT
        / "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005/queue.jsonl",
        layout_candidate=tmp_path / "candidate.json",
        candidate_sha256="a" * 64,
        model_blind_gate=tmp_path / "gate.json",
        model_blind_gate_sha256="b" * 64,
        runtime_manifest=tmp_path / "runtime.json",
        external_repository=tmp_path / "lerobot",
        simulator_repository=tmp_path / "robotwin",
        expected_study_commit="c" * 40,
        shard_index=0,
        shard_count=4,
        attempt_id="attempt01",
        gpu_index=0,
        checkpoint=tmp_path / "checkpoint",
        frozen_assets=tmp_path / "frozen",
        pod="v3e005-lingbot-ali",
        pod_uid="pod-uid-test",
        gpu_uuid="GPU-test",
    )
    destination = tmp_path / "raw" / cell.cell_id.replace(":", "__")
    worker = _worker_command(args, cell, destination)
    rendered = " ".join(worker)
    assert "run_lingbot_queue worker" in rendered
    assert f"--cell-id {cell.cell_id}" in rendered
    assert "--checkpoint" in worker and "--frozen-assets" in worker
    assert "DROID" not in rendered
    guard = _guard_command(args, cell, destination, worker)
    assert "--launch" in guard
    assert guard.count("--requested-relation") == 1
    assert cell.relation in guard
    assert cell.matched_layout_pair_id in guard
    # The guard opens its log before launching the worker.  Its files must be
    # siblings, not children, so the fail-closed worker can atomically create
    # the exact cell directory.
    output_path = Path(guard[guard.index("--output") + 1])
    assert output_path.parent == destination.parent
    assert output_path.name.startswith(destination.name + ".")


def test_e005_final_scene_api_and_runtime_snapshot_normalisation() -> None:
    scene, geometry = _scene_runtime_modules()
    assert callable(scene.load_candidate)
    assert callable(scene.validate_live_snapshot)
    assert callable(geometry.build_runtime_task_class)
    bundle = load_registered_bundle(ROOT)
    cell = bundle.seed_block(9400)[0]
    gate_scene = _bound_gate_scene(
        {"scenes": {cell.scene_id: {"scene_id": cell.scene_id}}}, cell
    )
    assert gate_scene["scene_id"] == cell.scene_id
    synthetic = {
        "realised_object_poses": {},
        "arm_reset_pose": {"status": "available"},
        "occlusion_check": {
            "head_camera": False,
            "left_camera": False,
            "right_camera": False,
        },
    }
    fake_scene = SimpleNamespace(
        validate_live_snapshot=lambda *args: {
            "asymmetry_metric_A": 2.5,
            "position_residual_m": 0.0,
            "orientation_residual_rad": 0.0,
            "midline_residual_m": 0.0,
        }
    )
    output = _normalise_runtime_snapshot(
        scene=fake_scene,
        candidate={},
        gate_scene=gate_scene,
        cell=cell,
        snapshot=synthetic,
    )
    assert output["asymmetry_metric_A"] == pytest.approx(2.5)
    assert output["occlusion_check"] is False
    assert output["all_camera_occlusion_checks"] == synthetic["occlusion_check"]
    assert output["mirrored_asset_identity_verified"] is True
    assert output["mirrored_yaw_verified"] is True


def test_e005_runtime_capture_keeps_snapshot_fields_at_one_top_level() -> None:
    bundle = load_registered_bundle(ROOT)
    cell = bundle.seed_block(9400)[0]

    class FakeBase:
        def __init__(self):
            self.object = object()
            self.target_object = object()

        def setup_demo(self, *arguments, **keywords):
            return "setup-complete"

        def get_obs(self):
            return {"unused": True}

    pose_rows = iter(
        (
            {
                "position_xyz_m": [0.0, -0.1, 0.74],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "asset_identity": ["target", 1],
            },
            {
                "position_xyz_m": [0.0, 0.1, 0.74],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "asset_identity": ["reference", 1],
            },
        )
    )

    def fake_pose(actor, identity):
        row = next(pose_rows)
        return SimpleNamespace(
            position_xyz_m=tuple(row["position_xyz_m"]),
            to_json=lambda row=row: row,
        )

    def validate_snapshot(candidate, scene_id, level, snapshot, gate_scene):
        assert set(snapshot["realised_object_poses"]) == {"target", "reference"}
        assert "asset_contract" in snapshot
        assert "occlusion_check" in snapshot
        assert "arm_reset_pose" in snapshot
        assert "views" in snapshot
        assert "asset_contract" not in snapshot["realised_object_poses"]
        return {
            "asymmetry_metric_A": 0.0,
            "position_residual_m": 0.0,
            "orientation_residual_rad": 0.0,
            "midline_residual_m": 0.0,
        }

    cameras = ("head_camera", "left_camera", "right_camera")
    scene = SimpleNamespace(CAMERAS=cameras, validate_live_snapshot=validate_snapshot)
    geometry = SimpleNamespace(
        build_runtime_task_class=lambda *args: FakeBase,
        actor_pose=fake_pose,
        rgb_views=lambda observation: {
            camera: np.zeros((2, 2, 3), dtype=np.uint8) for camera in cameras
        },
        target_visibility_pixels=lambda env, actor: {
            camera: 0 for camera in cameras
        },
        camera_centers=lambda env: {camera: (0.0, 0.0, 1.0) for camera in cameras},
        collision_bounding_radius=lambda actor: 0.05,
        reference_occludes_target=lambda *args: False,
        arm_reset_pose=lambda env: {"status": "available"},
    )
    assets = {
        "target": {"asset_identity": ["target", 1]},
        "reference": {"asset_identity": ["reference", 1]},
    }
    candidate = {"scenes": {cell.scene_id: {"assets": assets}}}
    task = _runtime_task_with_snapshot(
        scene=scene,
        geometry=geometry,
        candidate=candidate,
        gate_scene={"scene_id": cell.scene_id},
        cell=cell,
        simulator=ROOT,
    )
    instance = task()
    assert instance.setup_demo() == "setup-complete"
    snapshot = task.latest_reset_snapshot
    assert snapshot is not None
    assert set(snapshot["realised_object_poses"]) == {"target", "reference"}
    assert snapshot["asset_contract"] == assets
    assert all(
        row["target_visible_pixels"] == 0 for row in snapshot["views"].values()
    )


def _native_state(dx: float, z: float, relation: str, open_: bool) -> dict:
    region = 0.08 < abs(dx) < 0.2 and ((dx < 0) if relation == "left" else (dx > 0))
    return {
        "success": bool(region and open_),
        "relation_region": bool(region),
        "object_xyz": [dx, 0.0, z],
        "target_xyz": [0.0, 0.0, 0.74],
        "object_minus_target_x": dx,
        "object_minus_target_y": 0.0,
        "distance_xy": abs(dx),
        "grippers_open": open_,
    }


def _snapshot() -> dict:
    return {
        "realised_object_poses": {
            "target": {"position_xyz_m": [0.0, -0.1, 0.74], "asset_identity": ["target", 1]},
            "reference": {"position_xyz_m": [0.0, 0.1, 0.74], "asset_identity": ["reference", 1]},
        },
        "arm_reset_pose": {"status": "available", "joint_positions_rad": [0.0]},
        "asymmetry_metric_A": 0.0,
        "position_residual_m": 0.0,
        "orientation_residual_rad": 0.0,
        "midline_residual_m": 0.0,
        "occlusion_check": False,
        "all_camera_occlusion_checks": {
            "head_camera": False,
            "left_camera": False,
            "right_camera": False,
        },
        "mirrored_asset_identity_verified": True,
        "mirrored_yaw_verified": True,
    }


def _write_cell(root: Path, cell, *, action_value: float) -> tuple[Path, Path]:
    condition = root / cell.cell_id.replace(":", "__") / f"{cell.level_code}__{cell.relation}"
    condition.mkdir(parents=True)
    direction = -1.0 if cell.relation == "left" else 1.0
    final_dx = direction * 0.12
    trajectory = [
        {"action_step": 0, **_native_state(0.01, 0.74, cell.relation, True)},
        {"action_step": 1, **_native_state(0.02, 0.78, cell.relation, False)},
        {"action_step": 2, **_native_state(final_dx, 0.78, cell.relation, False)},
        {"action_step": 3, **_native_state(final_dx, 0.78, cell.relation, False)},
        {"action_step": 4, **_native_state(final_dx, 0.74, cell.relation, True)},
    ]
    trajectory_path = condition / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory))
    action_path = condition / "action_trace.npz"
    executed = np.full((4, 14), action_value, dtype=np.float32)
    np.savez_compressed(action_path, executed=executed)
    video_path = condition / "simulator.mp4"
    video_path.write_bytes(b"synthetic-video")
    latent_path = condition / "first_predicted_latent.pt"
    latent_path.write_bytes(b"latent")
    result = {
        "environment_seed": cell.environment_seed,
        "sampling_seed": cell.sampling_seed,
        "requested_relation": cell.relation,
        "prompt": cell.prompt,
        "prompt_family": "direct_command",
        "condition": f"{cell.level_code}__{cell.relation}",
        "requested_success": True,
        "actions_executed": 4,
        "trajectory_path": str(trajectory_path.resolve()),
        "simulator_video": str(video_path.resolve()),
        "first_predicted_latent_path": str(latent_path.resolve()),
        "action_trace": {
            "path": str(action_path.resolve()),
            "sha256": hashlib.sha256(action_path.read_bytes()).hexdigest(),
            "count": 4,
            "shape": [4, 14],
        },
    }
    result_path = condition / "result.json"
    result_path.write_text(json.dumps(result))
    snapshot_path = condition / "live_reset_snapshot.json"
    snapshot_path.write_text(json.dumps(_snapshot()))
    return result_path, snapshot_path


def _runtime() -> dict:
    return {
        "runtime_identity_sha256": "f" * 64,
        "checkpoint": {
            "revision": "exact-test-revision",
            "hash_manifest_artifact": {"sha256": "e" * 64},
        },
        "external_repository": {"commit": "d" * 40},
        "simulator_repository": {"commit": "c" * 40},
    }


def test_e005_lingbot_evidence_keeps_robotwin_scoring_and_latent_future(tmp_path: Path) -> None:
    bundle = load_registered_bundle(ROOT)
    pair = [
        cell
        for cell in bundle.seed_block(9400)
        if cell.symmetry_level == 1.0
    ]
    by_relation = {cell.relation: cell for cell in pair}
    provisional = {}
    for index, relation in enumerate(("left", "right")):
        result_path, snapshot_path = _write_cell(
            tmp_path, by_relation[relation], action_value=float(index)
        )
        provisional[relation] = build_provisional_episode(
            bundle=bundle,
            cell=by_relation[relation],
            result_path=result_path,
            snapshot_path=snapshot_path,
            runtime=_runtime(),
            candidate_sha256="a" * 64,
            model_blind_gate_sha256="b" * 64,
            expected_study_commit="c" * 40,
            attempt_id="attempt01",
            verify_video_decode=False,
        )
    left, right, closed = close_pair(provisional["left"], provisional["right"])
    assert left["success"] is True and right["success"] is True
    assert left["failure_category"] == right["failure_category"] == "correct"
    assert left["requested_side_depth"] == pytest.approx(0.12)
    assert right["requested_side_depth"] == pytest.approx(0.12)
    assert left["endpoint_shift"] == right["endpoint_shift"] == pytest.approx(0.24)
    assert closed["endpoint_shift"] == pytest.approx(0.24)
    assert left["action_distinct"] is True
    assert left["time_to_first_contact"] is None
    assert left["future_interface"] == "latent_only_future_not_decodable"
    assert left["future_evidence"][0]["kind"] == "latent_tensor_not_decoded"
    assert left["success_predicate_id"] == "frozen_v3_robotwin_relation_aware_success"
    assert "DROID" not in json.dumps(left)


def test_e005_pair_fails_closed_on_reset_mismatch_and_infra_has_no_behavioral_zero(
    tmp_path: Path,
) -> None:
    bundle = load_registered_bundle(ROOT)
    pair = [cell for cell in bundle.seed_block(9400) if cell.symmetry_level == 0.0]
    by_relation = {cell.relation: cell for cell in pair}
    rows = {}
    for relation in ("left", "right"):
        result_path, snapshot_path = _write_cell(tmp_path, by_relation[relation], action_value=0.0)
        rows[relation] = build_provisional_episode(
            bundle=bundle,
            cell=by_relation[relation],
            result_path=result_path,
            snapshot_path=snapshot_path,
            runtime=_runtime(),
            candidate_sha256="a" * 64,
            model_blind_gate_sha256="b" * 64,
            expected_study_commit="c" * 40,
            attempt_id="attempt01",
            verify_video_decode=False,
        )
    changed = copy.deepcopy(rows["right"])
    changed["initial_physical_fingerprint_sha256"] = "0" * 64
    with pytest.raises(E005ContractError, match="initial physical"):
        close_pair(rows["left"], changed)
    infra = infrastructure_record(
        cell=by_relation["left"],
        attempt_id="attempt01",
        error="renderer failed before behavior",
        stage="worker_preflight",
        retained_paths=[],
        bundle=bundle,
        candidate_sha256="a" * 64,
        model_blind_gate_sha256="b" * 64,
    )
    assert infra["behavioral_result_valid"] is False
    assert "success" not in infra
    assert "failure_category" not in infra
    assert "never encoded as zero" in infra["denominator_policy"]


def test_e005_queue_emits_hash_closed_four_cell_seed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = load_registered_bundle(ROOT)
    runtime = _runtime()
    rows = {}
    for index, cell in enumerate(bundle.seed_block(9400)):
        result_path, snapshot_path = _write_cell(
            tmp_path / "native-fixtures", cell, action_value=float(index)
        )
        rows[cell.cell_id] = build_provisional_episode(
            bundle=bundle,
            cell=cell,
            result_path=result_path,
            snapshot_path=snapshot_path,
            runtime=runtime,
            candidate_sha256="a" * 64,
            model_blind_gate_sha256="b" * 64,
            expected_study_commit="c" * 40,
            attempt_id="attempt01",
            verify_video_decode=False,
        )

    module_name = (
        "experiments.v3.phase_e.cross_arena_geometry_v3e005.run_lingbot_queue"
    )
    monkeypatch.setattr(
        f"{module_name}._load_bound_inputs",
        lambda args, verify_live: (bundle, {}, {}, runtime),
    )
    monkeypatch.setattr(
        f"{module_name}._load_cell_after_worker",
        lambda output, cell, bundle, args: rows[cell.cell_id],
    )
    output = tmp_path / "raw-shard"
    args = SimpleNamespace(
        study_root=ROOT,
        shard_index=0,
        shard_count=27,
        limit_seed_blocks=None,
        output_dir=output,
        resume=False,
        candidate_sha256="a" * 64,
        model_blind_gate_sha256="b" * 64,
        expected_study_commit="c" * 40,
        attempt_id="attempt01",
        pod="v3e005-lingbot-ali",
        pod_uid="pod-uid-test",
        gpu_uuid="GPU-test",
        gpu_index=0,
    )
    assert run_execute(args) == 0
    marker_path = output / "seeds/seed_9400/seed_9400_manifest.json"
    marker = load_object(marker_path)
    body = dict(marker)
    claimed = body.pop("marker_sha256")
    assert claimed == canonical_sha256(body)
    assert marker["behavioral_episode_count"] == 4
    assert len(marker["compact_episode_paths"]) == 4
    assert len(marker["episode_sha256"]) == 4
    assert len(marker["pair_paths"]) == 2
    assert len(marker["pair_sha256"]) == 2
    assert (output / "shard_manifest.json").is_file()
