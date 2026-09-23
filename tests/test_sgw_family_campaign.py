import json

import pytest

from experiments.workshops.spatial_grounding_v1 import family_campaign
from experiments.workshops.spatial_grounding_v1.family_campaign import compile_campaign, verify_external_postprocess
from experiments.workshops.spatial_grounding_v1.family_campaign_verifier import (
    _verify_banana_clearance,
    verify_design,
)
from experiments.workshops.spatial_grounding_v1.prospective_family_designs import _digest


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
