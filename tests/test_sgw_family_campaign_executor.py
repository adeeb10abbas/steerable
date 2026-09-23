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
    monkeypatch.setattr(
        executor, "_candidate_identity",
        lambda path: (executor._sha256(path), json.loads(Path(path).read_text())["metadata"]["candidate_capture_sha256"]),
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
            "q,c,r=map(pathlib.Path,sys.argv[1:]);q.write_text('{}');r=r/'trials';v=json.loads(c.read_text());"
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
            "candidate_id": "mock-candidate",
            "metadata": {"candidate_capture_sha256": hashlib.sha256(kwargs["capture"].read_bytes()).hexdigest()},
        }))
        return {"candidate_id": "mock-candidate"}
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
    with pytest.raises(ValueError, match="candidate manifest"):
        executor.run_slot(
            campaign_path=campaign, index=0, root=tmp_path / "slot", controller_calibration=calibration,
            capture_command=capture, qualification_command=[sys.executable, "-c", "pass"],
            materialize=lambda **kwargs: (
                kwargs["output"].write_text(json.dumps({"candidate_id": "mock-candidate"})),
                {"candidate_id": "mock-candidate"},
            )[1],
        )


def test_typed_physical_rejection_accounts_only_verifier_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign, calibration = _campaign(tmp_path), _calibration(tmp_path)
    _patch_fixture(monkeypatch, calibration)
    capture, _qualification = _success_commands()
    result = executor.run_slot(
        campaign_path=campaign, index=0, root=tmp_path / "slot", controller_calibration=calibration,
        capture_command=capture, qualification_command=[sys.executable, "-c", "pass"],
        materialize=lambda **kwargs: (
            kwargs["output"].write_text(json.dumps({"candidate_id": "mock-candidate"})),
            {"candidate_id": "mock-candidate"},
        )[1],
        verify=lambda **_: {
            "verification_sha256": "f" * 64,
            "physical_geometry_rejection": {
                "goal_sign": 1, "reset_index": 0, "rejection_scope": "reset",
                "reason": "fresh measured collision", "controller_actions_executed": 0,
            },
        },
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


def test_guard_cannot_bind_another_hash_valid_reset_snapshot(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    trial = tmp_path / "goal-+1" / "reset-0"
    trial.mkdir(parents=True)
    (trial / "state-0000.json").write_text('{"actual":true}')
    other = tmp_path / "other-state.json"
    other.write_text("{}")
    (trial / "preaction-geometry-guard.json").write_text(json.dumps({
        "schema_version": "sgw-01-family-preaction-geometry-guard-v1",
        "design_id": "HEIGHT-001", "candidate_sha256": candidate_sha, "candidate_capture_sha256": "c" * 64,
        "goal_sign": 1, "reset_index": 0,
        "raw_reset": {"path": str(other), "sha256": hashlib.sha256(other.read_bytes()).hexdigest(), "bytes": 2},
        "status": "physical_geometry_rejection_before_actions", "controller_actions_executed": 0,
        "rejection_scope": "reset", "reason": "wrong retained state",
    }))
    with pytest.raises(RuntimeError, match="raw reset evidence"):
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


def test_default_executor_materializes_and_verifies_real_synthetic_family_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise default author/materialize/verify functions across child boundaries."""
    from test_sgw_family_campaign import _native_review
    from test_sgw_prospective_family_designs import _baseline_files
    from experiments.workshops.spatial_grounding_v1 import family_campaign, family_campaign_verifier, height_dist_proposals
    from experiments.workshops.spatial_grounding_v1.prospective_family_designs import build_design_plan

    captures, manifests = _baseline_files(tmp_path, "HEIGHT")
    plan = build_design_plan(
        family="HEIGHT", seed=91, count=4, baseline_capture_paths=captures, baseline_manifest_paths=manifests,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(
        family_campaign, "verify_capture_artifacts",
        lambda path: {"receipt": {"path": str(path), "sha256": family_campaign._sha256(path)}},
    )
    review = _native_review(tmp_path, "HEIGHT", captures)
    campaign_path = tmp_path / "campaign.json"
    family_campaign.compile_campaign(
        plan_path=plan_path, baseline_captures=captures, baseline_reviews={"left": review, "right": review},
        output=campaign_path,
    )
    verified_capture = lambda path: {"receipt": {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}}
    monkeypatch.setattr(height_dist_proposals, "verify_capture_artifacts", verified_capture)
    monkeypatch.setattr(
        "experiments.workshops.spatial_grounding_v1.prospective_family_capture.verify_capture_artifacts",
        verified_capture,
    )
    monkeypatch.setattr(family_campaign_verifier, "verify_capture_artifacts", verified_capture)

    capture_child = tmp_path / "capture_child.py"
    capture_child.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "sys.path[:0]=[sys.argv[1], str(Path(sys.argv[1])/'tests')]\n"
        "from test_sgw_family_campaign import _candidate_capture\n"
        "plan=json.loads(Path(sys.argv[2]).read_text()); design=next(x for x in plan['designs'] if x['design_id']==sys.argv[3])\n"
        "manifest=json.loads(Path(sys.argv[4]).read_text())\n"
        "_candidate_capture(Path(design['baseline']['capture']['path']), manifest, design, Path(sys.argv[5]))\n"
    )
    qualification_child = tmp_path / "qualification_child.py"
    qualification_child.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "sys.path[:0]=[sys.argv[1], str(Path(sys.argv[1])/'tests')]\n"
        "from test_sgw_family_campaign import _produce_family_qualification\n"
        "_produce_family_qualification(Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text()), Path(sys.argv[4]), reject_reset=False)\n"
    )
    calibration = Path("artifacts/workshops/spatial_grounding_v1/controller_calibrations/lat-closed-pad-20260923.json").resolve()
    result = executor.run_slot(
        campaign_path=campaign_path, index=0, root=tmp_path / "slot", controller_calibration=calibration,
        capture_command=[sys.executable, str(capture_child), "{study_root}", str(plan_path), "{design_id}",
                         "{overlay_manifest}", "{capture}"],
        qualification_command=[sys.executable, str(qualification_child), "{study_root}", "{root}",
                               "{candidate}", "{calibration}"],
        child_timeout_seconds=120,
    )
    assert result["status"] == "externally_verified_candidate_slot_not_fixture_or_behavioral_release"


def test_geometric_rejection_is_accounted_without_child_or_refill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign, calibration = _campaign(tmp_path, status="geometrically_rejected_slot_no_refill"), _calibration(tmp_path)
    _patch_fixture(monkeypatch, calibration)
    result = executor.run_slot(
        campaign_path=campaign, index=0, root=tmp_path / "slot", controller_calibration=calibration,
        capture_command=[], qualification_command=[],
    )
    assert result["status"] == "geometric_rejection_accounted_slot_no_refill"
    assert not (tmp_path / "slot" / "capture-process.json").exists()
