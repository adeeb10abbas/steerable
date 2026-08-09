from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.layout_contract import (
    LayoutContractError,
    PoseSE2,
    evaluate_layout,
    load_candidate,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.r002_orientation_tolerance import (
    AMENDMENT_SHA256,
    CORRECTED_TOLERANCE_RAD,
    build_runtime_attestation,
    load_amendment,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.run_droid_queue import (
    _existing_valid_episode,
    _pair_inputs,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.runtime_contract import (
    RuntimeContractError,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"


def test_registered_r002_amendment_is_hash_bound() -> None:
    amendment_path = BASE / "live_orientation_realisation_tolerance_amendment.json"
    assert sha256_file(amendment_path) == AMENDMENT_SHA256
    value = load_amendment(
        amendment_path,
        AMENDMENT_SHA256,
        registration_sha256=sha256_file(BASE / "registration.json"),
        queue_sha256=sha256_file(BASE / "queue.jsonl"),
        candidate_sha256=sha256_file(BASE / "layout/candidate.json"),
    )
    assert value["frozen_change"]["corrected_live_orientation_realisation_tolerance_rad"] == 0.04
    assert len(value["excluded_s0_behavioral_rows_with_substituted_control_asset"]) == 8


def test_r002_tolerance_admits_observed_zero_request_settle_only() -> None:
    candidate_path = BASE / "layout/candidate.json"
    candidate = load_candidate(candidate_path, sha256_file(candidate_path))
    realised = candidate.layout(0.0)
    cube = realised["rubiks_cube"]
    realised["rubiks_cube"] = PoseSE2(
        cube.x_m,
        cube.y_m,
        cube.z_m,
        cube.yaw_rad + 0.0353741686,
        cube.asset_identity,
    )
    kwargs = {
        "candidate": candidate,
        "symmetry_level_s": 0.0,
        "realised_object_poses": realised,
        "occlusion_check_by_camera": {name: False for name in candidate.expected_cameras},
        "target_visible_by_camera": {name: True for name in candidate.expected_cameras},
        "arm_reset_pose": {
            "arm_joint_positions_rad": [0.0] * 7,
            "gripper_position": [1.0],
            "measurement_source_sha256": "0" * 64,
        },
    }
    with pytest.raises(LayoutContractError, match="live pose orientation differs"):
        evaluate_layout(**kwargs)
    scene = evaluate_layout(
        **kwargs,
        realisation_orientation_tolerance_rad=CORRECTED_TOLERANCE_RAD,
    )
    assert scene["live_orientation_realisation_tolerance_rad"] == 0.04


def test_r002_rejects_paired_usda_substituted_as_control(tmp_path: Path) -> None:
    amendment_path = tmp_path / "amendment.json"
    amendment_path.write_text("{}\n", encoding="utf-8")
    control = tmp_path / "control.usda"
    paired = tmp_path / "paired.usda"
    control.write_bytes(b"registered-control")
    paired.write_bytes(b"paired-clutter")
    amendment = {
        "control_asset_binding": {
            "required_original_control_asset": {
                "sha256": hashlib.sha256(control.read_bytes()).hexdigest(),
                "bytes": control.stat().st_size,
            },
            "incorrect_substituted_asset": {
                "sha256": hashlib.sha256(paired.read_bytes()).hexdigest(),
                "bytes": paired.stat().st_size,
            },
        }
    }
    record = build_runtime_attestation(
        amendment=amendment,
        amendment_path=amendment_path,
        amendment_sha256=sha256_file(amendment_path),
        control_scene_asset=control,
        paired_scene_asset=paired,
        symmetry_level_s=0.0,
    )
    assert record["s0_original_control_asset_binding_passed"] is True
    with pytest.raises(RuntimeContractError, match="not the original control USDA"):
        build_runtime_attestation(
            amendment=amendment,
            amendment_path=amendment_path,
            amendment_sha256=sha256_file(amendment_path),
            control_scene_asset=paired,
            paired_scene_asset=paired,
            symmetry_level_s=0.0,
        )


def _write_episode(
    cell_root: Path,
    *,
    symmetry_level_s: float,
    pair_id: str = "pair",
    relation: str = "left",
) -> Path:
    attempt = cell_root / "attempt001"
    attempt.mkdir(parents=True)
    episode = attempt / "raw_episode.jsonl"
    episode.write_text(
        json.dumps(
            {
                "symmetry_level_s": symmetry_level_s,
                "matched_pair_id": pair_id,
                "requested_relation": relation,
                "request0_pair_identity_sha256": "a" * 64,
                "request0_replay": {
                    "schema_version": "vla-wam-shared-v3e004-request0-evidence-envelope-v1"
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    episode.with_name(episode.name + ".manifest.json").write_text(
        json.dumps({"row_count": 1, "jsonl_sha256": sha256_file(episode)}),
        encoding="utf-8",
    )
    return episode


def test_resume_keeps_nonzero_r001_but_excludes_pre_r002_s0(tmp_path: Path) -> None:
    nonzero = _write_episode(tmp_path / "nonzero", symmetry_level_s=0.75)
    pre_r002_s0 = _write_episode(tmp_path / "s0", symmetry_level_s=0.0)
    assert _existing_valid_episode(
        nonzero.parent.parent,
        amendment_sha256=AMENDMENT_SHA256,
    ) == nonzero.resolve()
    assert _existing_valid_episode(
        pre_r002_s0.parent.parent,
        amendment_sha256=AMENDMENT_SHA256,
    ) is None


def test_pair_discovery_excludes_pre_r002_s0_without_hiding_nonzero_r001(tmp_path: Path) -> None:
    model_id = "cosmos3_nano_policy_droid"
    nonzero_pair = "v3e004:nano:seed9400:s075"
    s0_pair = "v3e004:nano:seed9400:s000"
    for relation in ("left", "right"):
        _write_episode(
            tmp_path
            / model_id
            / "shard-0-of-8"
            / "cells"
            / f"nonzero-{relation}",
            symmetry_level_s=0.75,
            pair_id=nonzero_pair,
            relation=relation,
        )
        _write_episode(
            tmp_path
            / model_id
            / "shard-0-of-8"
            / "cells"
            / f"s0-{relation}",
            symmetry_level_s=0.0,
            pair_id=s0_pair,
            relation=relation,
        )
    pair = _pair_inputs(
        tmp_path,
        model_id,
        nonzero_pair,
        amendment_sha256=AMENDMENT_SHA256,
    )
    assert pair is not None
    assert [json.loads(path.read_text(encoding="utf-8"))["requested_relation"] for path in pair] == [
        "left",
        "right",
    ]
    assert _pair_inputs(
        tmp_path,
        model_id,
        s0_pair,
        amendment_sha256=AMENDMENT_SHA256,
    ) is None
