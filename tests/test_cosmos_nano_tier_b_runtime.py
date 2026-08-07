from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.v3.cosmos_nano_tier_b.fixed_observation_gate import collect, evaluate
from experiments.v3.cosmos_nano_tier_b.runtime_contract import CONFIG, load_release
from experiments.v3.cosmos_nano_tier_b.queue import cell_plan, ordered_cells


ROOT = Path(__file__).resolve().parents[1]


def _release(amendment_id: str):
    directory = amendment_id.lower().replace("-", "")
    return load_release(
        ROOT,
        amendment_id,
        ROOT
        / "artifacts/vla_wam_shared_v3/prospective_tier_b/releases"
        / directory
        / "release_manifest.json",
    )


def test_released_queues_are_exact_and_ports_are_disjoint() -> None:
    b008 = _release("V3-B008")
    b009 = _release("V3-B009")
    assert len(b008.cells) == 162
    assert len(b009.cells) == 108
    assert CONFIG["V3-B008"]["port"] == 18018
    assert CONFIG["V3-B009"]["port"] == 18019
    assert CONFIG["V3-B008"]["port"] != CONFIG["V3-B009"]["port"]


def test_fixed_observation_gate_requires_repeatability_and_sensitivity() -> None:
    release = _release("V3-B009")
    runtime = {"runtime_identity_sha256": "a" * 64}
    observations = {
        arm: {
            "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
        }
        for arm in release.config["arms"]
    }

    def infer(request):
        index = request["probe_request_index"]
        condition = request["probe_condition"]
        value = 1 if condition == "right" else 0
        return {
            "action": np.full((32, 8), value, dtype=np.float32),
            "video": np.full((2, 2, 2, 3), value, dtype=np.uint8),
            "nano_tier_b_live_stack": "isolated_v3b008_v3b009_v1",
            "nano_tier_b_server_mode": "probe_only",
            "amendment_id": "V3-B009",
            "registered_cell_id": request["registered_cell_id"],
            "sampling_seed": request["sampling_seed"],
            "request_index": index,
            "probe_request_index": index,
            "probe_arm": request["probe_arm"],
            "probe_condition": condition,
            "release_fingerprint_sha256": request["release_fingerprint_sha256"],
            "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        }

    responses, records = collect(
        release=release,
        runtime=runtime,
        observations=observations,
        infer=infer,
    )
    report = evaluate(
        release=release,
        runtime=runtime,
        responses=responses,
        records=records,
    )
    assert report["status"] == "passed"
    assert report["behavioral_episode_count"] == 0
    assert report["model_request_count"] == 6


def test_role_swap_live_plan_preserves_whole_seed_order_and_dynamic_roles(tmp_path: Path) -> None:
    release = _release("V3-B009")
    cells = ordered_cells(release)
    first_seed = [cell for cell in cells if cell.seed == 9800]
    assert [cell.row["execution_order_index_within_seed"] for cell in first_seed] == [0, 1, 2, 3]
    assert {(cell.row["target_object"], cell.row["reference_object"]) for cell in first_seed} == {
        ("rubiks_cube", "bowl"),
        ("bowl", "rubiks_cube"),
    }
    cell = first_seed[0]
    plan = cell_plan(
        study_root=ROOT,
        study_root_string=str(ROOT),
        amendment_id="V3-B009",
        release_manifest=release.manifest_path,
        release_manifest_sha256=release.manifest_sha256,
        runtime_manifest=tmp_path / "runtime.json",
        release_gate=tmp_path / "gate.json",
        safe_fixture=tmp_path / "candidate.json",
        safe_fixture_sha256=cell.row["candidate_sha256"],
        raw_root=tmp_path / "raw",
        cell=cell,
        remote_host="10.0.0.1",
        remote_port=18019,
        lane_pod_uid="pod-uid",
        lane_gpu_uuid="gpu-uuid",
    )
    assert plan["arm"] == "cube_target_bowl_reference"
    assert plan["target_object"] == "rubiks_cube"
    assert plan["reference_object"] == "bowl"
    assert plan["matched_block_id"] == "v3b009:nano:seed9800"
    assert "--amendment-id" in plan["bridge_command"]
    assert "V3-B009" in plan["bridge_command"]
    assert "--amendment-id" in plan["compiler_command"]
