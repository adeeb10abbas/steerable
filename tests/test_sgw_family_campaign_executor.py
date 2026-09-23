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
        assert kwargs["manifest_output"].name == "candidate_manifest.json"
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
            "import hashlib,json,pathlib,sys;"
            "q,c,r=map(pathlib.Path,sys.argv[1:]);q.write_text('{}');v=json.loads(c.read_text());"
            "[(lambda d,s:(d.mkdir(parents=True), (d/'state-0000.json').write_text('{}'),"
            "(d/'preaction-geometry-guard.json').write_text(json.dumps({'schema_version':'sgw-01-family-preaction-geometry-guard-v1',"
            "'design_id':'HEIGHT-001','candidate_sha256':hashlib.sha256(c.read_bytes()).hexdigest(),"
            "'candidate_capture_sha256':v['metadata']['candidate_capture_sha256'],'goal_sign':s,'reset_index':i,"
            "'raw_reset':{'path':'state-0000.json','sha256':hashlib.sha256((d/'state-0000.json').read_bytes()).hexdigest(),"
            "'bytes':(d/'state-0000.json').stat().st_size},'status':'measured_banana_geometry_valid_before_actions',"
            "'controller_actions_executed':0}))))(r/f'goal-{s:+d}'/f'reset-{i}',s) for s in (1,-1) for i in range(3)]"
        ),
        "{qualification}", "{candidate}", "{root}",
    ]
    return capture, qualification


def test_slot_executes_separate_children_and_records_verified_physical_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign, calibration = _campaign(tmp_path), _calibration(tmp_path)
    _patch_fixture(monkeypatch, calibration)
    capture, qualification = _success_commands()
    calls = []
    def materialize(**kwargs):
        calls.append(kwargs["design_id"])
        kwargs["output"].write_text(json.dumps({
            "candidate": "measured",
            "metadata": {"candidate_capture_sha256": hashlib.sha256(kwargs["capture"].read_bytes()).hexdigest()},
        }))
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
            materialize=lambda **kwargs: kwargs["output"].write_text(json.dumps({
                "metadata": {"candidate_capture_sha256": hashlib.sha256(kwargs["capture"].read_bytes()).hexdigest()},
            })),
        )


def test_typed_physical_rejection_accounts_slot_without_controller_actions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign, calibration = _campaign(tmp_path), _calibration(tmp_path)
    _patch_fixture(monkeypatch, calibration)
    capture, _qualification = _success_commands()
    rejection = [
        sys.executable, "-c",
        (
            "import hashlib,json,pathlib,sys;"
            "q,c,r=map(pathlib.Path,sys.argv[1:]);q.write_text('{}');r=r/'goal-+1'/'reset-0';r.mkdir(parents=True);s=r/'state-0000.json';s.write_text('{}');"
            "v=json.loads(c.read_text());json.dump({'schema_version':'sgw-01-family-preaction-geometry-guard-v1',"
            "'design_id':'HEIGHT-001','candidate_sha256':hashlib.sha256(c.read_bytes()).hexdigest(),"
            "'candidate_capture_sha256':v['metadata']['candidate_capture_sha256'],'goal_sign':1,'reset_index':0,"
            "'raw_reset':{'path':'state-0000.json','sha256':hashlib.sha256(s.read_bytes()).hexdigest(),'bytes':s.stat().st_size},"
            "'status':'physical_geometry_rejection_before_actions','controller_actions_executed':0,"
            "'rejection_scope':'candidate','reason':'measured banana intersects support'},open(r/'preaction-geometry-guard.json','w'))"
        ),
        "{qualification}", "{candidate}", "{root}",
    ]
    result = executor.run_slot(
        campaign_path=campaign, index=0, root=tmp_path / "slot", controller_calibration=calibration,
        capture_command=capture, qualification_command=rejection,
        materialize=lambda **kwargs: kwargs["output"].write_text(json.dumps({
            "metadata": {"candidate_capture_sha256": hashlib.sha256(kwargs["capture"].read_bytes()).hexdigest()},
        })),
        verify=lambda **_: pytest.fail("physical rejection must not invoke six-trial verifier"),
    )
    assert result["status"] == "physical_geometry_rejection_accounted_slot_no_refill"
    assert result["physical_rejection"]["controller_actions_executed"] == 0


def test_rejection_followed_by_later_controller_action_is_technical_invalid(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    trial = tmp_path / "goal-+1" / "reset-0"
    trial.mkdir(parents=True)
    state = trial / "state-0000.json"
    state.write_text("{}")
    trial_guard = {
        "schema_version": "sgw-01-family-preaction-geometry-guard-v1",
        "design_id": "HEIGHT-001", "candidate_sha256": candidate_sha, "candidate_capture_sha256": "c" * 64,
        "goal_sign": 1, "reset_index": 0,
        "raw_reset": {"path": "state-0000.json", "sha256": hashlib.sha256(state.read_bytes()).hexdigest(), "bytes": 2},
        "status": "physical_geometry_rejection_before_actions", "controller_actions_executed": 0,
        "rejection_scope": "reset", "reason": "measured banana collision",
    }
    (trial / "preaction-geometry-guard.json").write_text(json.dumps(trial_guard))
    future = tmp_path / "goal-+1" / "reset-1"
    future.mkdir(parents=True)
    (future / "action-0001.npy").write_bytes(b"must not exist")
    with pytest.raises(RuntimeError, match="followed by controller actions"):
        executor._trial_guards(
            tmp_path, design_id="HEIGHT-001", candidate_sha256=candidate_sha, candidate_capture_sha256="c" * 64,
        )


def test_child_timeout_retains_fsynced_logs_and_fails_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign, calibration = _campaign(tmp_path), _calibration(tmp_path)
    _patch_fixture(monkeypatch, calibration)
    with pytest.raises(TimeoutError, match="wall-time"):
        executor.run_slot(
            campaign_path=campaign, index=0, root=tmp_path / "slot", controller_calibration=calibration,
            capture_command=[sys.executable, "-c", "import time; print('native log', flush=True); time.sleep(5)"],
            qualification_command=[sys.executable, "-c", "pass"], child_timeout_seconds=1,
        )
    process = json.loads((tmp_path / "slot" / "capture-process.json").read_text())
    assert process["timed_out"] is True
    assert Path(process["stdout"]["path"]).read_bytes()


def test_geometric_rejection_is_accounted_without_child_or_refill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign, calibration = _campaign(tmp_path, status="geometrically_rejected_slot_no_refill"), _calibration(tmp_path)
    _patch_fixture(monkeypatch, calibration)
    result = executor.run_slot(
        campaign_path=campaign, index=0, root=tmp_path / "slot", controller_calibration=calibration,
        capture_command=[], qualification_command=[],
    )
    assert result["status"] == "geometric_rejection_accounted_slot_no_refill"
    assert not (tmp_path / "slot" / "capture-process.json").exists()
