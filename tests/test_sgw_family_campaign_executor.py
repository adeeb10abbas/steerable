import hashlib
import json
from pathlib import Path
import sys

import pytest

from experiments.workshops.spatial_grounding_v1 import family_campaign_executor as executor
from experiments.workshops.spatial_grounding_v1.prospective_family_designs import _digest


def _campaign(tmp_path: Path, *, status: str = "blocked_pending_candidate_overlay_and_fresh_zero_model_capture") -> Path:
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    campaign = {
        "schema_version": "sgw-01-family-finite-campaign-v1", "family": "HEIGHT",
        "plan": {"path": str(plan), "sha256": hashlib.sha256(plan.read_bytes()).hexdigest(), "plan_sha256": "p"},
        "jobs": [{"design_id": "HEIGHT-001", "status": status}],
        "model_request_count": 0, "behavioral_episode_count": 0, "release_permitted": False,
    }
    campaign["campaign_sha256"] = _digest(campaign, "campaign_sha256")
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(campaign))
    return path


def _calibration(tmp_path: Path) -> Path:
    path = tmp_path / "calibration.json"
    path.write_text("{}")
    return path


def _patch_fixture(monkeypatch: pytest.MonkeyPatch, calibration: Path) -> None:
    original = executor._sha256
    monkeypatch.setattr(executor, "_sha256", lambda path: executor.CALIBRATION_SHA256 if Path(path) == calibration else original(Path(path)))
    def author(**kwargs):
        kwargs["output"].write_text("#usda")
        kwargs["manifest_output"].write_text("{}")
    monkeypatch.setattr(executor, "author_candidate_overlay", author)
    monkeypatch.setattr(
        "experiments.workshops.spatial_grounding_v1.prospective_family_capture.verify_capture_artifacts",
        lambda path: {"verified": str(path)},
    )


def _success_commands() -> tuple[list[str], list[str]]:
    capture = [
        sys.executable, "-c",
        "import sys;open(sys.argv[1],'w').write('{}')", "{capture}",
    ]
    qualification = [
        sys.executable, "-c",
        (
            "import hashlib,json,sys;"
            "open(sys.argv[1],'w').write('{}');"
            "json.dump({'design_id':'HEIGHT-001','candidate_sha256':hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest(),"
            "'status':'measured_banana_geometry_valid_before_actions','actions_started':False,"
            "'checks':[{'goal_sign':s,'reset_index':r,'status':'measured_banana_geometry_valid_before_actions',"
            "'actions_started':False,'raw_reset_sha256':'a'*64} for s in (1,-1) for r in range(3)]},open(sys.argv[3],'w'))"
        ),
        "{qualification}", "{candidate}", "{root}/preaction-geometry-guard.json",
    ]
    return capture, qualification


def test_slot_executes_separate_children_and_records_verified_physical_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign, calibration = _campaign(tmp_path), _calibration(tmp_path)
    _patch_fixture(monkeypatch, calibration)
    capture, qualification = _success_commands()
    calls = []
    def materialize(**kwargs):
        calls.append(kwargs["design_id"])
        kwargs["output"].write_text(json.dumps({"candidate": "measured"}))
        return {"candidate": "measured"}
    def verify(**kwargs):
        assert Path(kwargs["root"] / "candidate.json").is_file()
        return {"verification_sha256": "f" * 64, "physical_outcome": "valid_physical_rejection"}
    result = executor.run_slot(
        campaign_path=campaign, index=0, root=tmp_path / "slot", controller_calibration=calibration,
        capture_command=capture, qualification_command=qualification, materialize=materialize, verify=verify,
    )
    assert calls == ["HEIGHT-001"]
    assert result["status"] == "externally_verified_candidate_slot_not_fixture_or_behavioral_release"
    assert (tmp_path / "slot" / "capture-process.json").is_file()
    assert (tmp_path / "slot" / "qualification-process.json").is_file()
    assert result["model_request_count"] == result["behavioral_episode_count"] == 0


def test_exit_zero_without_capture_output_is_infrastructure_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign, calibration = _campaign(tmp_path), _calibration(tmp_path)
    _patch_fixture(monkeypatch, calibration)
    with pytest.raises(Exception):
        executor.run_slot(
            campaign_path=campaign, index=0, root=tmp_path / "slot", controller_calibration=calibration,
            capture_command=[sys.executable, "-c", "pass"], qualification_command=[sys.executable, "-c", "pass"],
            materialize=lambda **_: pytest.fail("must not materialize without capture"),
        )
    failure = json.loads((tmp_path / "slot" / "executor-failure.json").read_text())
    assert failure["model_request_count"] == 0 and "capture" in failure["error"].lower()


def test_missing_or_late_preaction_guard_stops_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign, calibration = _campaign(tmp_path), _calibration(tmp_path)
    _patch_fixture(monkeypatch, calibration)
    capture, _qualification = _success_commands()
    with pytest.raises(RuntimeError, match="pre-action geometry guard"):
        executor.run_slot(
            campaign_path=campaign, index=0, root=tmp_path / "slot", controller_calibration=calibration,
            capture_command=capture, qualification_command=[sys.executable, "-c", "pass"],
            materialize=lambda **kwargs: kwargs["output"].write_text("{}"),
        )


def test_geometric_rejection_is_accounted_without_child_or_refill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign, calibration = _campaign(tmp_path, status="geometrically_rejected_slot_no_refill"), _calibration(tmp_path)
    _patch_fixture(monkeypatch, calibration)
    result = executor.run_slot(
        campaign_path=campaign, index=0, root=tmp_path / "slot", controller_calibration=calibration,
        capture_command=[], qualification_command=[],
    )
    assert result["status"] == "geometric_rejection_accounted_slot_no_refill"
    assert not (tmp_path / "slot" / "capture-process.json").exists()
