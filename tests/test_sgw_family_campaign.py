import json
from argparse import Namespace
import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1 import family_campaign
from experiments.workshops.spatial_grounding_v1.family_campaign import compile_campaign, verify_external_postprocess
from experiments.workshops.spatial_grounding_v1.family_campaign_verifier import (
    _verify_banana_clearance,
    verify_design,
)
from experiments.workshops.spatial_grounding_v1.prospective_family_designs import _digest
from experiments.workshops.spatial_grounding_v1.prospective_family_designs import author_candidate_overlay, build_design_plan
from experiments.workshops.spatial_grounding_v1.height_dist_proposals import materialize_campaign_candidate
from experiments.workshops.spatial_grounding_v1.model_blind_qualification import _preflight_output_root, qualify_candidate
from experiments.workshops.spatial_grounding_v1.recorder import atomic_json
from experiments.workshops.spatial_grounding_v1.simulator_bridge import SimulatorSnapshot
from experiments.workshops.spatial_grounding_v1.lat_candidate_generator import workspace_digest
from test_sgw_prospective_family_designs import WORKSPACE, _baseline_files
from test_sgw_qualification_batch_verifier import NativeReceiptEnvironment, file_record


def _plan(tmp_path, captures):
    value = {
        "schema_version": "sgw-01-prospective-family-design-plan-v1",
        "family": "HEIGHT",
        "design_slot_count": 2,
        "accepted_design_count": 1,
        "geometric_rejection_count": 1,
        "accepted_design_ids": ["D-000"],
        "baselines": {
            side: {"capture": {"path": str(path.resolve()), "sha256": family_campaign._sha256(path)}}
            for side, path in captures.items()
        },
        "designs": [
            {"design_id": "D-000", "side": "left", "status": "prospective_design_requires_zero_model_capture"},
            {"design_id": "D-001", "side": "right", "status": "prospective_design_rejected_geometrically"},
        ],
    }
    value["plan_sha256"] = _digest(value, "plan_sha256")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(value))
    return path


def _reviews(tmp_path, captures):
    path = tmp_path / "sealed-native-visual-review.json"
    path.write_text(json.dumps({
        "schema_version": "synthetic-sealed-native-visual-review-v1",
        "scenes": [
            {
                "scene": f"synthetic-height-{side}",
                "capture": {"sha256": family_campaign._sha256(capture)},
                "disposition": "ACCEPT_NATIVE_VISUAL_SETUP_ONLY",
            }
            for side, capture in captures.items()
        ],
    }))
    return {"left": path, "right": path}


def test_campaign_compiles_fixed_slots_only_after_verified_baselines(tmp_path, monkeypatch):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text("{}")
    right.write_text("{}")
    plan = _plan(tmp_path, {"left": left, "right": right})
    monkeypatch.setattr(
        family_campaign, "verify_capture_artifacts",
        lambda path: {"receipt": {"path": str(path), "sha256": family_campaign._sha256(path)}},
    )
    reviews = _reviews(tmp_path, {"left": left, "right": right})

    campaign = compile_campaign(
        plan_path=plan, baseline_captures={"left": left, "right": right}, baseline_reviews=reviews,
        output=tmp_path / "campaign.json",
    )

    assert campaign["status"] == "compiled_native_visual_setup_baselines_not_authorized_to_launch_or_release"
    assert [job["status"] for job in campaign["jobs"]] == [
        "blocked_pending_candidate_overlay_and_fresh_zero_model_capture",
        "geometrically_rejected_slot_no_refill",
    ]
    assert campaign["jobs"][0]["fixed_trial_contract"]["total_scripted_trials"] == 6
    assert campaign["jobs"][0]["fixed_trial_contract"]["retry_permitted"] is False


def test_artifact_valid_baselines_cannot_replace_independent_scene_review(tmp_path, monkeypatch):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text("{}")
    right.write_text("{}")
    plan = _plan(tmp_path, {"left": left, "right": right})
    monkeypatch.setattr(
        family_campaign, "verify_capture_artifacts",
        lambda path: {"receipt": {"path": str(path), "sha256": family_campaign._sha256(path)}},
    )
    with pytest.raises(ValueError, match="sealed scene judgments"):
        compile_campaign(
            plan_path=plan, baseline_captures={"left": left, "right": right},
            baseline_reviews={"left": left, "right": right}, output=tmp_path / "campaign.json",
        )


def test_postprocess_refuses_exit_zero_without_bound_verification(tmp_path, monkeypatch):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text("{}")
    right.write_text("{}")
    plan = _plan(tmp_path, {"left": left, "right": right})
    monkeypatch.setattr(
        family_campaign, "verify_capture_artifacts",
        lambda path: {"receipt": {"path": str(path), "sha256": family_campaign._sha256(path)}},
    )
    campaign_path = tmp_path / "campaign.json"
    reviews = _reviews(tmp_path, {"left": left, "right": right})
    campaign = compile_campaign(
        plan_path=plan, baseline_captures={"left": left, "right": right}, baseline_reviews=reviews, output=campaign_path,
    )

    with pytest.raises(RuntimeError, match="without its verification"):
        verify_external_postprocess(
            campaign_path=campaign_path, output=tmp_path / "postprocess.json", returncode=0,
            verification_path=tmp_path / "missing.json",
        )
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"campaign_sha256": "wrong"}))
    with pytest.raises(RuntimeError, match="not a bound campaign evidence"):
        verify_external_postprocess(
            campaign_path=campaign_path, output=tmp_path / "postprocess.json", returncode=0, verification_path=bad,
        )
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_verification(campaign["campaign_sha256"], tmp_path)))
    assert verify_external_postprocess(
        campaign_path=campaign_path, output=tmp_path / "postprocess.json", returncode=0, verification_path=good,
    )["release_permitted"] is False


def test_family_verifier_refuses_missing_candidate_capture_and_trials(tmp_path, monkeypatch):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text("{}")
    right.write_text("{}")
    plan = _plan(tmp_path, {"left": left, "right": right})
    monkeypatch.setattr(
        family_campaign, "verify_capture_artifacts",
        lambda path: {"receipt": {"path": str(path), "sha256": family_campaign._sha256(path)}},
    )
    campaign_path = tmp_path / "campaign.json"
    compile_campaign(
        plan_path=plan, baseline_captures={"left": left, "right": right},
        baseline_reviews=_reviews(tmp_path, {"left": left, "right": right}), output=campaign_path,
    )
    with pytest.raises(ValueError, match="candidate manifest, capture, materialization"):
        verify_design(
            campaign_path=campaign_path, design_id="D-000", root=tmp_path / "design",
            output=tmp_path / "verification.json",
        )


@pytest.mark.parametrize("mutation", ("truncated", "extra", "duplicate_id", "missing_id"))
def test_campaign_rejects_nonfinite_or_nonunique_design_lists(tmp_path, monkeypatch, mutation):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text("{}")
    right.write_text("{}")
    plan = _plan(tmp_path, {"left": left, "right": right})
    value = json.loads(plan.read_text())
    if mutation == "truncated":
        value["designs"].pop()
    elif mutation == "extra":
        value["designs"].append(dict(value["designs"][0]))
    elif mutation == "duplicate_id":
        value["designs"][1]["design_id"] = "D-000"
    else:
        value["designs"][1].pop("design_id")
    value["plan_sha256"] = _digest(value, "plan_sha256")
    plan.write_text(json.dumps(value))
    monkeypatch.setattr(family_campaign, "verify_capture_artifacts", lambda path: pytest.fail("must fail before capture"))
    with pytest.raises(ValueError, match="fixed|duplicate|missing"):
        compile_campaign(
            plan_path=plan, baseline_captures={"left": left, "right": right},
            baseline_reviews=_reviews(tmp_path, {"left": left, "right": right}), output=tmp_path / "campaign.json",
        )


def test_campaign_rejects_swapped_plan_capture_or_review(tmp_path, monkeypatch):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text('{"side": "left"}')
    right.write_text('{"side": "right"}')
    plan = _plan(tmp_path, {"left": left, "right": right})
    monkeypatch.setattr(
        family_campaign, "verify_capture_artifacts",
        lambda path: {"receipt": {"path": str(path), "sha256": family_campaign._sha256(path)}},
    )
    reviews = _reviews(tmp_path, {"left": left, "right": right})
    with pytest.raises(ValueError, match="supplied left baseline"):
        compile_campaign(
            plan_path=plan, baseline_captures={"left": right, "right": left}, baseline_reviews=reviews,
            output=tmp_path / "campaign.json",
        )
    left_review = json.loads(reviews["left"].read_text())
    left_review["scenes"][0]["capture"]["sha256"] = family_campaign._sha256(right)
    reviews["left"].write_text(json.dumps(left_review))
    with pytest.raises(ValueError, match="sealed review"):
        compile_campaign(
            plan_path=plan, baseline_captures={"left": left, "right": right}, baseline_reviews=reviews,
            output=tmp_path / "review-campaign.json",
        )


def test_campaign_rejects_unknown_design_status(tmp_path, monkeypatch):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text("{}")
    right.write_text("{}")
    plan = _plan(tmp_path, {"left": left, "right": right})
    value = json.loads(plan.read_text())
    value["designs"][1]["status"] = "unrecognized_status"
    value["plan_sha256"] = _digest(value, "plan_sha256")
    plan.write_text(json.dumps(value))
    monkeypatch.setattr(
        family_campaign, "verify_capture_artifacts",
        lambda path: {"receipt": {"path": str(path), "sha256": family_campaign._sha256(path)}},
    )
    with pytest.raises(ValueError, match="unknown design status"):
        compile_campaign(
            plan_path=plan, baseline_captures={"left": left, "right": right},
            baseline_reviews=_reviews(tmp_path, {"left": left, "right": right}), output=tmp_path / "campaign.json",
        )


def test_family_verifier_rejects_measured_banana_table_escape_and_support_gap():
    objects = {
        "table": {
            "bbox_env_local_min_xyz_m": [0, 0, 0],
            "bbox_env_local_max_xyz_m": [1, 1, 1],
        },
        "banana": {
            "bbox_env_local_min_xyz_m": [.79, .38, .05],
            "bbox_env_local_max_xyz_m": [.90, .50, .09],
        },
        "support": {
            "bbox_env_local_min_xyz_m": [.70, .30, .05],
            "bbox_env_local_max_xyz_m": [.78, .37, .12],
        },
    }
    with pytest.raises(ValueError, match="clearance"):
        _verify_banana_clearance(objects, ["support"])
    objects["banana"]["bbox_env_local_max_xyz_m"][0] = 1.01
    with pytest.raises(ValueError, match="leaves the measured table"):
        _verify_banana_clearance(objects, ["support"])


def _verification(campaign_sha256, root):
    candidate = root / "candidate_capture.json"
    qualification = root / "qualification.json"
    candidate.write_text("{}")
    qualification.write_text("{}")
    trials = []
    for sign in (1, -1):
        for reset in range(3):
            trial = root / "trials" / f"goal-{sign:+d}" / f"reset-{reset}" / "trial.json"
            trial.parent.mkdir(parents=True, exist_ok=True)
            trial.write_text("{}")
            trials.append({
                "goal_sign": sign, "reset_index": reset, "trial_sha256": family_campaign._sha256(trial),
                "path": str(trial), "sha256": family_campaign._sha256(trial), "bytes": trial.stat().st_size,
            })
    value = {
        "schema_version": "sgw-01-family-campaign-verification-v1",
        "campaign_sha256": campaign_sha256,
        "status": "verified_evidence_not_fixture_release",
        "release_permitted": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "candidate_capture_sha256": family_campaign._sha256(candidate),
        "qualification_sha256": family_campaign._sha256(qualification),
        "candidate_capture": {
            "path": str(candidate), "sha256": family_campaign._sha256(candidate), "bytes": candidate.stat().st_size,
        },
        "qualification": {
            "path": str(qualification), "sha256": family_campaign._sha256(qualification), "bytes": qualification.stat().st_size,
        },
        "trials": trials,
    }
    value["verification_sha256"] = _digest(value, "verification_sha256")
    return value


@pytest.mark.parametrize("family", ("HEIGHT", "DIST"))
def test_complete_synthetic_family_campaign_chain_and_adversarial_bindings(tmp_path, monkeypatch, family):
    """Exercise materialization -> real recorder -> canonical family verification."""

    captures, manifests = _baseline_files(tmp_path, family)
    plan = build_design_plan(
        family=family, seed=91, count=4, baseline_capture_paths=captures, baseline_manifest_paths=manifests,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    design = next(row for row in plan["designs"] if row["status"].endswith("capture"))
    root = tmp_path / "design"
    root.mkdir()
    candidate_manifest = root / "candidate_manifest.json"
    manifest = author_candidate_overlay(
        plan=plan, design_id=design["design_id"], output=root / "candidate.usda", manifest_output=candidate_manifest,
    )
    capture = _candidate_capture(captures[design["side"]], manifest, design, root / "candidate_capture.json")
    calibration = tmp_path / "controller-calibration.json"
    calibration.write_bytes(Path(
        "artifacts/workshops/spatial_grounding_v1/controller_calibrations/lat-closed-pad-20260923.json"
    ).read_bytes())
    monkeypatch.setattr(
        "experiments.workshops.spatial_grounding_v1.height_dist_proposals.verify_capture_artifacts",
        lambda path: {"receipt": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
    )
    candidate = materialize_campaign_candidate(
        plan_path=plan_path, design_id=design["design_id"], candidate_manifest_path=candidate_manifest,
        candidate_capture_path=capture, controller_calibration_path=calibration, output=root / "candidate.json",
    )
    cli_args = Namespace(
        candidate_file=root / "candidate.json", candidate_manifest=candidate_manifest,
        candidate_capture=capture, candidate_id=candidate["candidate_id"], output_root=root,
    )
    _preflight_output_root(cli_args)
    (root / "qualification.json").write_text("{}")
    with pytest.raises(FileExistsError, match="overwrite"):
        _preflight_output_root(cli_args)
    (root / "qualification.json").unlink()
    monkeypatch.setattr(
        family_campaign, "verify_capture_artifacts",
        lambda path: {"receipt": {"path": str(path), "sha256": family_campaign._sha256(path)}},
    )
    review = _native_review(tmp_path, family, {"left": captures["left"], "right": captures["right"]})
    campaign_path = tmp_path / "campaign.json"
    compile_campaign(
        plan_path=plan_path, baseline_captures=captures, baseline_reviews={"left": review, "right": review},
        output=campaign_path,
    )
    monkeypatch.setattr(
        "experiments.workshops.spatial_grounding_v1.family_campaign_verifier.verify_capture_artifacts",
        lambda path: {"receipt": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
    )
    _produce_family_qualification(root, candidate, calibration, reject_reset=False)
    verified = verify_design(
        campaign_path=campaign_path, design_id=design["design_id"], root=root, output=root / "family_verification.json",
    )
    assert verified["status"] == "verified_evidence_not_fixture_release"

    calibration_bytes = calibration.read_bytes()
    calibration.write_text("{}")
    with pytest.raises(ValueError, match="calibration binding"):
        verify_design(campaign_path=campaign_path, design_id=design["design_id"], root=root, output=root / "wrong-calibration.json")
    calibration.write_bytes(calibration_bytes)
    controller_path = root / "controller.json"
    controller = json.loads(controller_path.read_text())
    controller["calibration"]["robot_asset"]["sha256"] = "0" * 64
    controller["calibration"]["receipt_sha256"] = workspace_digest(controller["calibration"])
    controller_path.write_text(json.dumps(controller))
    with pytest.raises(ValueError, match="calibration binding"):
        verify_design(campaign_path=campaign_path, design_id=design["design_id"], root=root, output=root / "wrong-embedded-calibration.json")
    _produce_family_qualification(root, candidate, calibration, reject_reset=False)
    plan_bytes = plan_path.read_bytes()
    altered_plan = json.loads(plan_bytes)
    altered_plan["designs"][0]["translation_xy_m"][0] += .001
    altered_plan["plan_sha256"] = _digest(altered_plan, "plan_sha256")
    plan_path.write_text(json.dumps(altered_plan))
    with pytest.raises(ValueError, match="plan"):
        verify_design(campaign_path=campaign_path, design_id=design["design_id"], root=root, output=root / "wrong-plan.json")
    plan_path.write_bytes(plan_bytes)

    # A score-success/reset mismatch is retained as a verified physical rejection.
    _produce_family_qualification(root, candidate, calibration, reject_reset=True)
    rejected = verify_design(
        campaign_path=campaign_path, design_id=design["design_id"], root=root, output=root / "reset-rejection.json",
    )
    assert rejected["status"] == "verified_evidence_not_fixture_release"

    # Restore passing output, then exercise capture/raw-media/plan/calibration failures.
    _produce_family_qualification(root, candidate, calibration, reject_reset=False)
    geometry_rejection, controller_calls = _produce_family_qualification(
        root, candidate, calibration, reject_reset=False, reject_geometry=True,
    )
    assert geometry_rejection["physical_geometry_rejection_before_actions"]["rejection_scope"] == "reset"
    assert controller_calls == []
    guard = json.loads((root / "trials/goal-+1/reset-0/preaction-geometry-guard.json").read_text())
    assert guard["status"] == "physical_geometry_rejection_before_actions"
    assert guard["controller_actions_executed"] == 0
    verified_geometry_rejection = verify_design(
        campaign_path=campaign_path, design_id=design["design_id"], root=root,
        output=root / "geometry-rejection.json",
    )
    assert verified_geometry_rejection["physical_geometry_rejection"]["reason"] == (
        geometry_rejection["physical_geometry_rejection_before_actions"]["reason"]
    )
    guard["reason"] = "forged physical rejection"
    guard_path = root / "trials/goal-+1/reset-0/preaction-geometry-guard.json"
    guard_path.write_text(json.dumps(guard))
    trial_receipt_path = guard_path.parent / "trial.json"
    trial_receipt = json.loads(trial_receipt_path.read_text())
    trial_receipt["files"][guard_path.name]["bytes"] = guard_path.stat().st_size
    trial_receipt["files"][guard_path.name]["sha256"] = hashlib.sha256(guard_path.read_bytes()).hexdigest()
    trial_receipt_path.write_text(json.dumps(trial_receipt))
    with pytest.raises(ValueError, match="recomputed"):
        verify_design(
            campaign_path=campaign_path, design_id=design["design_id"], root=root,
            output=root / "forged-geometry-rejection.json",
        )
    geometry_rejection, _ = _produce_family_qualification(
        root, candidate, calibration, reject_reset=False, reject_geometry=True,
    )
    (root / "trials/goal-+1/reset-0/action-0001.npy").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="contains controller"):
        verify_design(
            campaign_path=campaign_path, design_id=design["design_id"], root=root,
            output=root / "late-action-geometry-rejection.json",
        )
    _produce_family_qualification(root, candidate, calibration, reject_reset=False)
    raw = root / "trials/goal-+1/reset-0/frame-0001.npy"
    saved_raw = raw.read_bytes()
    raw.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="raw trial integrity"):
        verify_design(campaign_path=campaign_path, design_id=design["design_id"], root=root, output=root / "corrupt-media.json")
    raw.write_bytes(saved_raw)
    capture.write_text(capture.read_text() + "\n")
    with pytest.raises(ValueError, match="materialized candidate"):
        verify_design(campaign_path=campaign_path, design_id=design["design_id"], root=root, output=root / "corrupt-capture.json")


def _native_review(tmp_path, family, captures):
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"scenes": [
        {"scene": f"synthetic-{family.lower()}-{side}", "capture": {"sha256": family_campaign._sha256(capture)},
         "disposition": "ACCEPT_NATIVE_VISUAL_SETUP_ONLY"}
        for side, capture in captures.items()
    ]}))
    return path


def _candidate_capture(baseline_path, manifest, design, output):
    value = json.loads(baseline_path.read_text())
    objects = value["objects"]
    # The prospective baseline manifest provides measured/captured support dimensions.
    baseline_manifest = json.loads(Path(manifest["source_baseline"]["overlay_manifest"]["path"]).read_text())
    specs = {row["name"]: row for row in baseline_manifest["prospective_design"]["dimensions_and_poses"]}
    for name, pose in design["authored_scored_object_roots"].items():
        objects[name]["root_position_env_local_xyz_m"] = pose["position_m"]
        objects[name]["root_quaternion_world_wxyz"] = pose["quaternion_wxyz"]
        if objects[name].get("geometric_center_offset_root_local_xyz_m") == [0, 0, 0]:
            objects[name]["geometric_center_env_local_xyz_m"] = list(pose["position_m"])
    for name, spec in specs.items():
        if name in objects:
            continue
        center, size = spec["center_m"], spec["size_m"]
        objects[name] = {
            "root_position_env_local_xyz_m": center, "root_quaternion_world_wxyz": [1, 0, 0, 0],
            "geometric_center_env_local_xyz_m": center,
            "geometric_center_offset_root_local_xyz_m": [0, 0, 0],
            "bbox_env_local_min_xyz_m": [center[i] - size[i] / 2 for i in range(3)],
            "bbox_env_local_max_xyz_m": [center[i] + size[i] / 2 for i in range(3)],
        }
    if "plate" in objects and "bbox_env_local_min_xyz_m" not in objects["plate"]:
        center = objects["plate"]["geometric_center_env_local_xyz_m"] = objects["plate"]["root_position_env_local_xyz_m"]
        objects["plate"]["bbox_env_local_min_xyz_m"] = [center[0] - .06, center[1] - .06, center[2] - .006]
        objects["plate"]["bbox_env_local_max_xyz_m"] = [center[0] + .06, center[1] + .06, center[2] + .006]
    if manifest["family"] == "DIST":
        bowl = objects["bowl"]["geometric_center_env_local_xyz_m"]
        plate = objects["plate"]["geometric_center_env_local_xyz_m"]
        cube = objects["rubiks_cube"]
        offset = cube["geometric_center_offset_root_local_xyz_m"]
        from experiments.workshops.spatial_grounding_v1.fixtures import _rotate
        rotated_offset = _rotate(tuple(cube["root_quaternion_world_wxyz"]), tuple(offset))
        midpoint = [(bowl[i] + plate[i]) / 2 for i in range(3)]
        cube["geometric_center_env_local_xyz_m"] = midpoint
        cube["root_position_env_local_xyz_m"] = [midpoint[i] - rotated_offset[i] for i in range(3)]
    # Synthetic measured capture rows retain valid released goal centers.
    cube_center = objects["rubiks_cube"]["geometric_center_env_local_xyz_m"]
    cube_bottom = objects["rubiks_cube"]["bbox_env_local_min_xyz_m"][2]
    center_above_bottom = cube_center[2] - cube_bottom
    if manifest["family"] == "HEIGHT":
        bowl_z = objects["bowl"]["geometric_center_env_local_xyz_m"][2]
        targets = (("height_upper_support", None, None, bowl_z + .04), ("height_lower_support", None, None, bowl_z - .04))
    else:
        targets = (
            ("dist_bowl_landing_support", *objects["bowl"]["geometric_center_env_local_xyz_m"][:2], cube_center[2]),
            ("dist_plate_landing_support", *objects["plate"]["geometric_center_env_local_xyz_m"][:2], cube_center[2]),
        )
    for name, target_x, target_y, target_z in targets:
        row = objects[name]
        height = row["bbox_env_local_max_xyz_m"][2] - row["bbox_env_local_min_xyz_m"][2]
        top = target_z - center_above_bottom
        row["bbox_env_local_max_xyz_m"][2] = top
        row["bbox_env_local_min_xyz_m"][2] = top - height
        row["geometric_center_env_local_xyz_m"][2] = top - height / 2
        row["root_position_env_local_xyz_m"][2] = row["geometric_center_env_local_xyz_m"][2]
        if target_x is not None:
            for key in ("root_position_env_local_xyz_m", "geometric_center_env_local_xyz_m"):
                row[key][0], row[key][1] = target_x, target_y
            width = row["bbox_env_local_max_xyz_m"][0] - row["bbox_env_local_min_xyz_m"][0]
            depth = row["bbox_env_local_max_xyz_m"][1] - row["bbox_env_local_min_xyz_m"][1]
            row["bbox_env_local_min_xyz_m"][0:2] = [target_x - width / 2, target_y - depth / 2]
            row["bbox_env_local_max_xyz_m"][0:2] = [target_x + width / 2, target_y + depth / 2]
    value["objects"] = objects
    value["overlay_manifest_sha256"] = manifest["manifest_sha256"]
    value["usd_dependency_inventory"].append({"real_path": manifest["overlay_usda"]["path"], "sha256": manifest["overlay_usda"]["sha256"]})
    value["support_contact_measurements"] = {
        name: {"sensor": f"rubiks_cube__{name}", "force_matrix_world_n": [[[[0, 0, 0]]]], "shape": [1, 1, 1, 3]}
        for name in manifest["native_import_contract"]["kinematic_or_static_bodies"]
    }
    value["receipt_sha256"] = workspace_digest(value)
    output.write_text(json.dumps(value))
    return output


def _produce_family_qualification(root, candidate_value, calibration_path, *, reject_reset, reject_geometry=False):
    from experiments.workshops.spatial_grounding_v1.fixtures import FixtureCandidate

    candidate = FixtureCandidate.from_json(candidate_value)
    class Environment(NativeReceiptEnvironment):
        def objects(self, *args, **kwargs):
            rows = super().objects(*args, **kwargs)
            if self.candidate.family == "DIST":
                from experiments.workshops.spatial_grounding_v1.simulator_bridge import ObjectState
                rows["plate"] = ObjectState(self.candidate.scoring_poses()["plate"], 0.0, 0.0, True, False)
            return rows
        def _context(self):
            objects = json.loads((root / "candidate_capture.json").read_text())["objects"]
            rows = {name: {
                key: row[key] for key in ("root_position_env_local_xyz_m", "root_quaternion_world_wxyz",
                                           "geometric_center_env_local_xyz_m", "bbox_env_local_min_xyz_m", "bbox_env_local_max_xyz_m")
            } for name, row in objects.items()
              if all(key in row for key in ("root_position_env_local_xyz_m", "root_quaternion_world_wxyz",
                                             "geometric_center_env_local_xyz_m", "bbox_env_local_min_xyz_m", "bbox_env_local_max_xyz_m"))}
            if reject_geometry:
                rows["banana"]["root_position_env_local_xyz_m"][0] += .01
            return rows
        def reset(self):
            result = super().reset()
            snapshot = replace(result.snapshot, context_measurements=self._context())
            if reject_reset and self.ordinal == 3:
                bad = dict(snapshot.reset_root_poses)
                pose = bad["rubiks_cube"]
                bad["rubiks_cube"] = type(pose)((pose.position_m[0] + .01, pose.position_m[1], pose.position_m[2]), pose.quaternion_wxyz)
                snapshot = replace(snapshot, reset_root_poses=bad)
            return replace(result, snapshot=snapshot)
        def step(self, action):
            self.steps += 1
            cube = self.candidate.scoring_poses()["rubiks_cube"].position_m
            bowl = self.candidate.scoring_poses()["bowl"].position_m
            if self.candidate.family == "DIST" and self.steps > 3:
                support = self.candidate.metadata["goal_supports"][
                    "near_bowl" if self.goal == 1 else "near_plate"
                ]["cube_center_env_local_xyz_m"]
                state = self.objects()
                from experiments.workshops.spatial_grounding_v1.simulator_bridge import ObjectState
                state["rubiks_cube"] = ObjectState(
                    type(state["rubiks_cube"].pose)(tuple(support), (1, 0, 0, 0)), 0.0, 0.0, True, False,
                )
            else:
                z = cube[2] + (.04 * self.goal if self.steps > 3 else .04)
                state = self.objects(lift=z - cube[2], supported=self.steps > 3, attached=self.steps <= 3)
            return SimulatorSnapshot(state, self.steps / 15, context_measurements=self._context())
    class Bridge:
        def create_environment(self, task, seed):
            return Environment(candidate, root / "reset-warmup")
    calibration = json.loads(calibration_path.read_text())
    identity = {"recipe": calibration["schema_version"], "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(), "calibration": calibration}
    controller_calls = []
    class Controller:
        def __init__(self): self.identity = identity
        def actions_for_goal(self, environment, _candidate, sign):
            guard_path = (
                root / "trials" / f"goal-{sign:+d}" / f"reset-{len(controller_calls) % 3}"
                / "preaction-geometry-guard.json"
            )
            assert json.loads(guard_path.read_text())["status"] == "measured_banana_geometry_valid_before_actions"
            controller_calls.append((sign, environment.steps))
            environment.goal = sign
            return [np.asarray([[0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.float32) for _ in range(450)]
    shutil.rmtree(root / "trials", ignore_errors=True)
    shutil.rmtree(root / "reset-warmup", ignore_errors=True)
    atomic_json(root / "controller.json", identity)
    result = qualify_candidate(candidate, Bridge(), Controller(), seed=candidate.seed, evidence_root=root / "trials")
    atomic_json(root / "qualification.json", result)
    return result, controller_calls
