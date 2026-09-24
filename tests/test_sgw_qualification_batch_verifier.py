from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1 import qualification_batch_verifier as verifier
from experiments.workshops.spatial_grounding_v1.grasp_calibration import calibrated_actions
from experiments.workshops.spatial_grounding_v1.lat_workspace_capture import render_only_warmup
from experiments.workshops.spatial_grounding_v1.model_blind_qualification import qualify_candidate
from experiments.workshops.spatial_grounding_v1.recorder import atomic_json
from experiments.workshops.spatial_grounding_v1.scoring import GoalSpec, score_episode
from experiments.workshops.spatial_grounding_v1.simulator_bridge import ResetResult
from test_sgw_simulator import FakeEnvironment


STUDY = Path(__file__).parents[1] / "artifacts/workshops/spatial_grounding_v1"
PLAN = STUDY / "qualification_batches/lat-remaining-20260923.json"
PROPOSALS = STUDY / "proposals/lat-20260922.json"
CALIBRATION = STUDY / "controller_calibrations/lat-closed-pad-20260923.json"
PLAN_SHA = "f5007f84d26f27946b38f2053ca7d640b56e50e92ae3d09a0261feb45f88275e"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


class CameraTensor:
    def __init__(self, array):
        self.array = array

    def __getitem__(self, index):
        return CameraTensor(self.array[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class NativeReceiptEnvironment(FakeEnvironment):
    """Synthetic physics with the real producer, arrays, reset schema and codecs."""

    def __init__(self, candidate, root):
        super().__init__(candidate=candidate)
        self.root = root
        self.ordinal = 0

    def reset(self):
        reset = super().reset()
        self.ordinal += 1
        path = self.root / f"reset-{self.ordinal:02d}"
        cameras = ("over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam")
        observation = {"image_obs": {
            camera: CameraTensor(self.render_viewport()[None, ...]) for camera in cameras
        }}
        render_env = SimpleNamespace(
            sim=SimpleNamespace(current_time=1 / 60, render=lambda: None),
            scene={camera: SimpleNamespace(update=lambda *args, **kwargs: None) for camera in cameras},
            observation_manager=SimpleNamespace(compute=lambda: observation),
        )
        _, warmup = render_only_warmup(render_env, observation, 120, path)
        fingerprint = hashlib.sha256(json.dumps(
            {name: asdict(obj.pose) for name, obj in reset.snapshot.objects.items()},
            sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        return ResetResult(reset.snapshot, {
            "reset_id": f"{self.candidate.candidate_id}:{self.ordinal}",
            "camera_id": "over_shoulder_left_camera", "camera_name": "over_shoulder_left_camera",
            "fingerprint": fingerprint, "temporal_cache_reset": True, "control_step_dt_s": 1 / 15,
            "render_only_warmup": warmup,
        })

    def step(self, action):
        snapshot = super().step(action)
        return replace(snapshot, simulated_time_s=self.steps / 15)


class Controller:
    def __init__(self, identity):
        self.identity = identity

    def actions_for_goal(self, environment, candidate, sign):
        environment.goal = sign
        return calibrated_actions(
            candidate, sign, self.identity["calibration"],
            flange_quaternion_world_wxyz=np.array([1., 0, 0, 0]),
            environment_origin_world_xyz_m=np.zeros(3), robot_root_world_xyz_m=np.zeros(3),
        )


@pytest.fixture(scope="module")
def produced(tmp_path_factory):
    registration = verifier.load_registration(PLAN, PROPOSALS, CALIBRATION, expected_plan_sha256=PLAN_SHA)
    name = registration.plan["candidate_ids_in_frozen_hash_order"][0]
    root = tmp_path_factory.mktemp("synthetic-qualification") / f"0-{name}"
    candidate = registration.candidates[name]
    environment = NativeReceiptEnvironment(candidate, root / "result/reset_warmup")

    class Bridge:
        def create_environment(self, task, seed):
            assert task.candidate == candidate and seed == candidate.seed
            return environment

    atomic_json(root / "storage_preflight.json", {
        "status": "passed_conservative_batch_space_check", "index": 0, "candidate_id": name,
        "plan_sha256": PLAN_SHA, "model_requests": 0, "behavioral_episodes": 0,
    })
    atomic_json(root / "result/controller.json", registration.controller)
    result = qualify_candidate(candidate, Bridge(), Controller(registration.controller),
                               seed=candidate.seed, evidence_root=root / "result/trials")
    atomic_json(root / "result/qualification.json", result)
    assert result["status"] == "accepted_model_blind_fixture_candidate"
    return registration, root


@pytest.fixture
def evidence(produced):
    registration, root = produced
    # Mutations touch only these JSON files or the one named action, then restore
    # the original producer evidence. Codec/array generation is shared.
    paths = [
        root / "result/qualification.json", root / "result/controller.json",
        root / "result/trials/goal-+1/reset-0/trial.json",
        root / "result/trials/goal-+1/reset-0/state-0442.json",
        root / "result/trials/goal-+1/reset-0/action-0450.npy",
    ]
    originals = {path: path.read_bytes() for path in paths}
    yield registration, root
    for path, raw in originals.items():
        path.write_bytes(raw)


def test_real_producer_and_full_decoders_verify_all_six(produced):
    registration, root = produced
    report = verifier.verify_candidate(registration, index=0, root=root)
    assert report["status"] == "verified_all_six_pass"
    assert report["passed_checks"] == 6
    assert sum(row["verified_trial_files"] for row in report["checks"]) == 10824
    assert sum(row["decoded_trial_frames"] + row["decoded_warmup_frames"] for row in report["checks"]) == 3432
    assert sum(row["verified_warmup_files"] for row in report["checks"]) == 804


def test_shared_trial_verifier_accepts_actual_unwrapped_producer_layout(produced):
    registration, root = produced
    check = json.loads((root / "result/qualification.json").read_text())["checks"][0]
    report, _ = verifier.verify_trial_evidence(
        evidence_root=root,
        trial=root / "result/trials/goal-+1/reset-0",
        check=check,
        candidate=registration.candidates[registration.plan["candidate_ids_in_frozen_hash_order"][0]],
    )
    assert report["decoded_trial_frames"] == 451
    assert report["decoded_warmup_frames"] == 121


def test_explicit_guard_requirement_rejects_old_lat_trace_without_guard(produced):
    registration, root = produced
    check = json.loads((root / "result/qualification.json").read_text())["checks"][0]
    with pytest.raises(verifier.VerificationError, match="trial manifest omits/adds files"):
        verifier.verify_trial_evidence(
            evidence_root=root, trial=root / "result/trials/goal-+1/reset-0",
            check=check,
            candidate=registration.candidates[registration.plan["candidate_ids_in_frozen_hash_order"][0]],
            expect_geometry_guard=True,
        )


@pytest.mark.parametrize("index,remove", [(2, False), (10, True)])
def test_warmup_enforces_frozen_sampled_camera_inventory(produced, index, remove):
    _, root = produced
    result = json.loads((root / "result/qualification.json").read_text())
    warmup = result["checks"][0]["reset_receipt"]["render_only_warmup"]
    views = warmup["snapshots"][index]["views"]
    if remove:
        del views["wrist_cam"]
    else:
        views["wrist_cam"] = warmup["snapshots"][0]["views"]["wrist_cam"]
    shape = np.load(root / "result/trials/goal-+1/reset-0/frame-0000.npy", allow_pickle=False).shape
    with pytest.raises(verifier.VerificationError, match="warmup camera inventory"):
        verifier._warmup(root, warmup, shape)


@pytest.mark.parametrize("mutation,match", [
    ("manifest", "manifest"),
    ("controller", "controller identity"),
    ("raw_state", "raw/scored state"),
    ("action", "hash/size"),
    ("missing_trial", "six-trial identity"),
    ("candidate", "candidate/seed"),
    ("order", "six-trial identity"),
    ("fps", "video receipt timing"),
])
def test_complete_evidence_rejects_identity_and_integrity_changes(evidence, mutation, match):
    registration, root = evidence
    result_path = root / "result/qualification.json"
    result = json.loads(result_path.read_text())
    trial = root / "result/trials/goal-+1/reset-0"
    saved = json.loads((trial / "trial.json").read_text())
    if mutation == "manifest":
        del saved["files"]["frame-0001.npy"]
        result["checks"][0]["files"] = saved["files"]
    elif mutation == "controller":
        atomic_json(root / "result/controller.json", {"recipe": "unattested"})
    elif mutation == "raw_state":
        path = trial / "state-0442.json"
        state = json.loads(path.read_text())
        state["raw_snapshot"]["objects"]["rubiks_cube"]["angular_speed_rad_s"] = 9.0
        atomic_json(path, state)
        saved["files"][path.name] = {key: value for key, value in file_record(path).items() if key != "path"}
        result["checks"][0]["files"] = saved["files"]
    elif mutation == "action":
        (trial / "action-0450.npy").write_bytes(b"not an array")
    elif mutation == "missing_trial":
        result["checks"].pop()
    elif mutation == "candidate":
        result["candidate_sha256"] = "0" * 64
    elif mutation == "order":
        result["checks"].reverse()
    elif mutation == "fps":
        saved["viewport_video"]["fps"] = 30
        result["checks"][0]["viewport_video"] = saved["viewport_video"]
    atomic_json(trial / "trial.json", saved)
    atomic_json(result_path, result)
    with pytest.raises(verifier.VerificationError, match=match):
        verifier.verify_candidate(registration, index=0, root=root)


def test_boundary_failure_is_verified_physical_rejection_not_technical(evidence):
    registration, root = evidence
    trial = root / "result/trials/goal-+1/reset-0"
    path = trial / "state-0442.json"
    state = json.loads(path.read_text())
    state["raw_snapshot"]["objects"]["rubiks_cube"]["angular_speed_rad_s"] = 0.20133880839766183
    state["state"]["angular_speed_rad_s"] = 0.20133880839766183
    atomic_json(path, state)
    saved = json.loads((trial / "trial.json").read_text())
    saved["files"][path.name] = {key: value for key, value in file_record(path).items() if key != "path"}
    saved["per_step_states"][442] = state["state"]
    score = score_episode({"states": saved["per_step_states"], "termination_reason": "action_cap"}, GoalSpec("LAT", 1))
    assert score.requested_success is False
    saved.update(score=asdict(score), passed=False, status="rejected_scripted_goal")
    atomic_json(trial / "trial.json", saved)
    result_path = root / "result/qualification.json"
    result = json.loads(result_path.read_text())
    result["checks"][0] = {**saved, "reset_validation": result["checks"][0]["reset_validation"]}
    result["status"] = "rejected_model_blind_fixture_candidate"
    atomic_json(result_path, result)
    report = verifier.verify_candidate(registration, index=0, root=root)
    assert report["status"] == "verified_physical_rejection"
    assert report["passed_checks"] == 5


def test_production_batch_keeps_pending_separate_and_never_releases(produced, tmp_path):
    registration, _ = produced
    name = registration.plan["candidate_ids_in_frozen_hash_order"][0]
    (tmp_path / f"0-{name}").mkdir()
    report = verifier.verify_batch(registration, tmp_path)
    assert report["counts"]["pending_or_partial"] == 1
    assert report["counts"]["not_started_or_missing"] == 38
    assert report["counts"]["verified_physical_rejection"] == 0
    assert report["release_permitted"] is False
    assert report["model_requests"] == report["behavioral_episodes"] == 0


def test_registration_requires_exact_plan_and_proposal_hashes(tmp_path):
    with pytest.raises(verifier.VerificationError, match="plan hash"):
        verifier.load_registration(PLAN, PROPOSALS, CALIBRATION, expected_plan_sha256="0" * 64)
    altered = tmp_path / "proposals.json"
    shutil.copyfile(PROPOSALS, altered)
    altered.write_bytes(altered.read_bytes() + b"\n")
    with pytest.raises(verifier.VerificationError, match="proposal hash"):
        verifier.load_registration(PLAN, altered, CALIBRATION, expected_plan_sha256=PLAN_SHA)


def test_bounded_watcher_persists_completed_attempts_once_and_preserves_missing(produced, tmp_path, monkeypatch):
    registration, _ = produced
    names = registration.plan["candidate_ids_in_frozen_hash_order"]
    clock = [0.0]
    first = tmp_path / "raw" / f"0-{names[0]}" / "result/qualification.json"
    atomic_json(first, {"status": "infrastructure_invalid_qualification"})
    calls = []
    original = verifier._candidate_report

    def track(*args):
        calls.append(args[1])
        return original(*args)

    def advance(seconds):
        assert seconds == 1
        clock[0] = 1
        atomic_json(tmp_path / "raw" / f"1-{names[1]}" / "result/qualification.json",
                    {"status": "infrastructure_invalid_qualification"})

    monkeypatch.setattr(verifier, "_candidate_report", track)
    monkeypatch.setattr(verifier.time, "time", lambda: clock[0])
    monkeypatch.setattr(verifier.time, "sleep", advance)
    output = tmp_path / "verified.json"
    report = verifier.watch_batch(registration, tmp_path / "raw", output,
                                  wait_until_utc="1970-01-01T00:00:01Z")
    assert calls == [0, 1]
    assert report["counts"]["technical_invalid_evidence"] == 2
    assert report["counts"]["not_started_or_missing"] == 37
    assert report["counts"]["verified_physical_rejection"] == 0
    assert len(list(output.with_suffix(".candidates").glob("*.json"))) == 2
