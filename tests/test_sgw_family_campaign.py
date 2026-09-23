import json

import pytest

from experiments.workshops.spatial_grounding_v1 import family_campaign
from experiments.workshops.spatial_grounding_v1.family_campaign import compile_campaign, verify_external_postprocess
from experiments.workshops.spatial_grounding_v1.family_campaign_verifier import verify_design
from experiments.workshops.spatial_grounding_v1.prospective_family_designs import _digest


def _plan(tmp_path):
    value = {
        "schema_version": "sgw-01-prospective-family-design-plan-v1",
        "family": "HEIGHT",
        "design_slot_count": 2,
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
    paths = {}
    for side, capture in captures.items():
        path = tmp_path / f"{side}.review.json"
        path.write_text(json.dumps({
            "status": "accepted_model_blind_scene_baseline_for_prospective_design",
            "baseline_capture": {"sha256": family_campaign._sha256(capture)},
            "model_request_count": 0, "behavioral_episode_count": 0,
        }))
        paths[side] = path
    return paths


def test_campaign_compiles_fixed_slots_only_after_verified_baselines(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text("{}")
    right.write_text("{}")
    monkeypatch.setattr(family_campaign, "verify_capture_artifacts", lambda path: {"receipt": {"path": str(path)}})
    reviews = _reviews(tmp_path, {"left": left, "right": right})

    campaign = compile_campaign(
        plan_path=plan, baseline_captures={"left": left, "right": right}, baseline_reviews=reviews,
        output=tmp_path / "campaign.json",
    )

    assert campaign["status"] == "compiled_reviewed_baselines_not_authorized_to_launch_or_release"
    assert [job["status"] for job in campaign["jobs"]] == [
        "blocked_pending_candidate_overlay_and_fresh_zero_model_capture",
        "geometrically_rejected_slot_no_refill",
    ]
    assert campaign["jobs"][0]["fixed_trial_contract"]["total_scripted_trials"] == 6
    assert campaign["jobs"][0]["fixed_trial_contract"]["retry_permitted"] is False


def test_artifact_valid_baselines_cannot_replace_independent_scene_review(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text("{}")
    right.write_text("{}")
    monkeypatch.setattr(family_campaign, "verify_capture_artifacts", lambda path: {"receipt": {"path": str(path)}})
    with pytest.raises(ValueError, match="independently accepted"):
        compile_campaign(
            plan_path=plan, baseline_captures={"left": left, "right": right},
            baseline_reviews={"left": left, "right": right}, output=tmp_path / "campaign.json",
        )


def test_postprocess_refuses_exit_zero_without_bound_verification(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text("{}")
    right.write_text("{}")
    monkeypatch.setattr(family_campaign, "verify_capture_artifacts", lambda path: {"receipt": {"path": str(path)}})
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
    with pytest.raises(RuntimeError, match="not bound"):
        verify_external_postprocess(
            campaign_path=campaign_path, output=tmp_path / "postprocess.json", returncode=0, verification_path=bad,
        )
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"campaign_sha256": campaign["campaign_sha256"]}))
    assert verify_external_postprocess(
        campaign_path=campaign_path, output=tmp_path / "postprocess.json", returncode=0, verification_path=good,
    )["release_permitted"] is False


def test_family_verifier_refuses_missing_candidate_capture_and_trials(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text("{}")
    right.write_text("{}")
    monkeypatch.setattr(family_campaign, "verify_capture_artifacts", lambda path: {"receipt": {"path": str(path)}})
    campaign_path = tmp_path / "campaign.json"
    compile_campaign(
        plan_path=plan, baseline_captures={"left": left, "right": right},
        baseline_reviews=_reviews(tmp_path, {"left": left, "right": right}), output=campaign_path,
    )
    with pytest.raises(ValueError, match="candidate capture and qualification"):
        verify_design(
            campaign_path=campaign_path, design_id="D-000", root=tmp_path / "design",
            output=tmp_path / "verification.json",
        )
