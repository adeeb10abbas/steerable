import hashlib
import json
from pathlib import Path

import pytest

from experiments.workshops.spatial_grounding_v1 import family_partition_summary as summary
from experiments.workshops.spatial_grounding_v1 import family_partition_worker as worker


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path, dict]:
    remaining = [
        *[{"family": "HEIGHT", "slot_index": index, "design_id": f"HEIGHT-{index:03d}", "side": "left"}
          for index in range(3, 58)],
        *[{"family": "DIST", "slot_index": index, "design_id": f"DIST-{index:03d}", "side": "right"}
          for index in range(4, 66)],
    ]
    campaigns, families = {}, []
    for family, proposed, geometric, eligible in (("HEIGHT", 100, 43, 57), ("DIST", 100, 36, 64)):
        path = _write(tmp_path / f"{family}.campaign.json", {"family": family})
        campaigns[family] = path
        families.append({
            "family": family, "proposed_slots": proposed, "geometric_rejections": geometric,
            "capture_eligible_slots": eligible, "campaign": {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
            },
        })
    freeze = _write(tmp_path / "freeze.json", {
        "release_permitted": False, "model_requests": 0, "behavioral_episodes": 0, "families": families,
        "remaining_capture_eligible_slots": remaining,
        "native_smoke_slots": [
            {"family": "HEIGHT", "slot_index": 0, "design_id": "HEIGHT-000", "side": "left"},
            {"family": "HEIGHT", "slot_index": 1, "design_id": "HEIGHT-001", "side": "right"},
            {"family": "DIST", "slot_index": 0, "design_id": "DIST-000", "side": "left"},
            {"family": "DIST", "slot_index": 3, "design_id": "DIST-003", "side": "right"},
        ],
    })
    monkeypatch.setattr(worker, "FREEZE_SHA256", hashlib.sha256(freeze.read_bytes()).hexdigest())
    source = _write(tmp_path / "source.py", {"source": "pinned"})
    calibration = _write(tmp_path / "calibration.json", {"calibration": "pinned"})
    monkeypatch.setattr(worker, "CALIBRATION_SHA256", hashlib.sha256(calibration.read_bytes()).hexdigest())
    smoke = [
        {"family": family, "slot_index": index, "design_id": design,
         "root": str(tmp_path / "smoke" / f"{family}-{index}"),
         "verification": str(tmp_path / "smoke" / f"{family}-{index}" / "v.json")}
        for family, index, design in (
            ("HEIGHT", 0, "HEIGHT-000"), ("HEIGHT", 1, "HEIGHT-001"),
            ("DIST", 0, "DIST-000"), ("DIST", 3, "DIST-003"),
        )
    ]
    for row, qualification_status in zip(smoke, (
        "accepted_model_blind_fixture_candidate", "rejected_model_blind_fixture_candidate",
        "accepted_model_blind_fixture_candidate", "rejected_model_blind_fixture_candidate",
    ), strict=True):
        receipt = _write(Path(row["verification"]), {"verification_sha256": "a" * 64})
        _write(Path(row["root"]) / "qualification.json", {"status": qualification_status})
        row["verification_file_sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
        row["verification_sha256"] = "a" * 64
    config = _write(tmp_path / "config.json", {
        "freeze_receipt": str(freeze), "campaigns": {family: str(path) for family, path in campaigns.items()},
        "source_path": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "controller_calibration": str(calibration), "capture_command": ["capture"], "qualification_command": ["qualify"],
        "smoke_evidence": smoke, "workers": 4, "free_space_floor_bytes": worker.MIN_FREE_SPACE_FLOOR_BYTES,
        "declared_slot_bytes": 1, "child_timeout_seconds": 2400,
    })
    workers_root = tmp_path / "workers-root"
    bindings = worker._config_bindings(_json(config), freeze, campaigns)
    _write(workers_root / "partition-binding.json", {
        "schema_version": worker.SCHEMA, "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
        "bindings_sha256": worker._digest(bindings), "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "workers": 4,
    })
    return config, workers_root, {
        "freeze": _json(freeze), "bindings": bindings, "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }


def _rank(worker_root: Path, metadata: dict, rank: int) -> list[dict]:
    slots = worker.partition_slots(metadata["freeze"], rank=rank, workers=4)
    root = worker_root / str(rank)
    _write(worker_root / "rank-claims" / f"rank-{rank}.json", {
        "rank": rank, "root": str(root.resolve()), "config_sha256": metadata["config_sha256"],
        "bindings_sha256": worker._digest(metadata["bindings"]),
    })
    _write(root / "worker-receipt.json", {
        "schema_version": worker.SCHEMA, "rank": rank, "slot_order": slots, "bindings": metadata["bindings"],
        "smoke_verifications": [],
    })
    return slots


def _verifier(**kwargs):
    output = kwargs["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("{}")
    return {
        "status": "verified_evidence_not_fixture_release",
        "physical_geometry_rejection": None, "verification_sha256": "a" * 64,
    }


def test_incomplete_summary_accounts_all_200_without_failures(tmp_path, monkeypatch):
    config, workers_root, metadata = _fixture(tmp_path, monkeypatch)
    result = summary.compile_summary(config_path=config, workers_root=workers_root, output=tmp_path / "summary.json",
                                     verifier=_verifier)
    assert result["status"] == "incomplete"
    assert result["families"]["HEIGHT"]["proposed"] == 100
    assert result["families"]["HEIGHT"]["geometric_rejections"] == 43
    assert result["families"]["HEIGHT"]["unstarted"] == 55
    assert result["families"]["DIST"]["unstarted"] == 62
    assert result["families"]["HEIGHT"]["physical_accepted"] == result["families"]["HEIGHT"]["physical_rejected"] == 1
    assert result["families"]["DIST"]["physical_accepted"] == result["families"]["DIST"]["physical_rejected"] == 1
    assert result["release_permitted"] is False and result["raw_evidence_retained_on_pvc"] is True


def test_duplicate_or_forged_completion_is_rejected(tmp_path, monkeypatch):
    config, workers_root, metadata = _fixture(tmp_path, monkeypatch)
    slot = _rank(workers_root, metadata, 0)[0]
    _write(workers_root / "slot-claims" / f"{slot['family'].lower()}-{slot['slot_index']:03d}.json",
           {"rank": 1, "slot": slot, "bindings_sha256": worker._digest(metadata["bindings"])})
    with pytest.raises(ValueError, match="slot claim"):
        summary.compile_summary(config_path=config, workers_root=workers_root, output=tmp_path / "summary.json",
                                verifier=_verifier)


def test_duplicate_slot_claim_is_rejected(tmp_path, monkeypatch):
    config, workers_root, metadata = _fixture(tmp_path, monkeypatch)
    slot = _rank(workers_root, metadata, 0)[0]
    _write(workers_root / "slot-claims" / "one.json", {"rank": 0, "slot": slot, "bindings_sha256": worker._digest(metadata["bindings"])})
    _write(workers_root / "slot-claims" / "two.json", {"rank": 0, "slot": slot, "bindings_sha256": worker._digest(metadata["bindings"])})
    with pytest.raises(ValueError, match="duplicated"):
        summary.compile_summary(config_path=config, workers_root=workers_root, output=tmp_path / "summary.json",
                                verifier=_verifier)


def test_stop_with_partial_is_explicitly_incomplete(tmp_path, monkeypatch):
    config, workers_root, metadata = _fixture(tmp_path, monkeypatch)
    slot = _rank(workers_root, metadata, 0)[0]
    _write(workers_root / "slot-claims" / f"{slot['family'].lower()}-{slot['slot_index']:03d}.json",
           {"rank": 0, "slot": slot, "bindings_sha256": worker._digest(metadata["bindings"])})
    _write(workers_root / "infrastructure-stop.json", {"status": "infrastructure_stop"})
    result = summary.compile_summary(config_path=config, workers_root=workers_root, output=tmp_path / "summary.json",
                                     verifier=_verifier)
    assert result["status"] == "incomplete"
    assert result["families"]["HEIGHT"]["partial"] == 1


def test_terminal_infrastructure_receipt_is_not_partial(tmp_path, monkeypatch):
    config, workers_root, metadata = _fixture(tmp_path, monkeypatch)
    slot = _rank(workers_root, metadata, 0)[0]
    rank_root = workers_root / "0"
    _write(workers_root / "slot-claims" / f"{slot['family'].lower()}-{slot['slot_index']:03d}.json",
           {"rank": 0, "slot": slot, "bindings_sha256": worker._digest(metadata["bindings"])})
    _write(rank_root / "slots" / f"{slot['family'].lower()}-{slot['slot_index']:03d}" / "executor-failure.json",
           {"schema_version": summary.EXECUTOR_SCHEMA, "index": slot["slot_index"],
            "campaign_path": _json(config)["campaigns"][slot["family"]],
            "model_request_count": 0, "behavioral_episode_count": 0, "release_permitted": False,
            "error": "retained native infrastructure traceback"})
    result = summary.compile_summary(config_path=config, workers_root=workers_root, output=tmp_path / "summary.json",
                                     verifier=_verifier)
    assert result["families"]["HEIGHT"]["infrastructure_invalid"] == 1
    assert result["families"]["HEIGHT"]["partial"] == 0


def test_completed_result_requires_authoritative_not_status_label(tmp_path, monkeypatch):
    config, workers_root, metadata = _fixture(tmp_path, monkeypatch)
    slot = _rank(workers_root, metadata, 0)[0]
    rank_root = workers_root / "0"
    _write(workers_root / "slot-claims" / f"{slot['family'].lower()}-{slot['slot_index']:03d}.json",
           {"rank": 0, "slot": slot, "bindings_sha256": worker._digest(metadata["bindings"])})
    slot_root = tmp_path / "slot-root"
    _write(slot_root / "qualification.json", {"status": "accepted_model_blind_fixture_candidate"})
    _write(rank_root / "completed" / f"{slot['family'].lower()}-{slot['slot_index']:03d}.json", {
        "schema_version": worker.SCHEMA, "slot": slot, "slot_root": str(slot_root),
        "result": {"status": "externally_verified_candidate_slot_not_fixture_or_behavioral_release"},
    })
    with pytest.raises(ValueError, match="authoritative"):
        summary.compile_summary(
            config_path=config, workers_root=workers_root, output=tmp_path / "summary.json",
            verifier=lambda **_: {"status": "forged"},
        )


@pytest.mark.parametrize("tamper_executor", [False, True])
def test_completed_canonical_slot_retains_trial_warmup_and_video_bindings(tmp_path, monkeypatch, tamper_executor):
    config, workers_root, metadata = _fixture(tmp_path, monkeypatch)
    slot = _rank(workers_root, metadata, 0)[0]
    rank_root = workers_root / "0"
    slot_root = rank_root / "slots" / f"{slot['family'].lower()}-{slot['slot_index']:03d}"
    _write(workers_root / "slot-claims" / f"{slot['family'].lower()}-{slot['slot_index']:03d}.json",
           {"rank": 0, "slot": slot, "bindings_sha256": worker._digest(metadata["bindings"])})
    trial = slot_root / "trials" / "goal-+1" / "reset-0"
    warmup = trial / "warmup.mp4"
    warmup.parent.mkdir(parents=True, exist_ok=True)
    warmup.write_bytes(b"warmup")
    (trial / "viewport.mp4").write_bytes(b"video")
    _write(trial / "reset.json", {"render_only_warmup": {"path": str(warmup), "sha256": hashlib.sha256(warmup.read_bytes()).hexdigest()}})
    _write(trial / "trial.json", {"reset_receipt": {"render_only_warmup": {"path": str(warmup)}}})
    _write(slot_root / "qualification.json", {"status": "accepted_model_blind_fixture_candidate"})
    digest = "c" * 64
    result = {"status": "externally_verified_candidate_slot_not_fixture_or_behavioral_release",
              "verification_sha256": digest}
    _write(slot_root / "executor-receipt.json",
           {**result, "verification_sha256": "d" * 64} if tamper_executor else result)
    _write(rank_root / "completed" / f"{slot['family'].lower()}-{slot['slot_index']:03d}.json", {
        "schema_version": worker.SCHEMA, "slot": slot, "slot_root": str(slot_root), "result": result,
    })
    worker_receipt = _json(rank_root / "worker-receipt.json")
    worker_receipt["smoke_verifications"] = []
    _write(rank_root / "worker-receipt.json", worker_receipt)
    _write(rank_root / "worker-completion.json", {
        "schema_version": worker.SCHEMA, "rank": 0, "workers": 4, "completed_slots": [slot],
        "status": "stopped_before_next_slot", "stopped_by_peer": True,
    })

    def verifier(**kwargs):
        kwargs["output"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output"].write_text("{}")
        if kwargs["design_id"] == slot["design_id"]:
            return {"status": "verified_evidence_not_fixture_release", "verification_sha256": digest,
                    "physical_geometry_rejection": None,
                    "trials": [{"goal_sign": 1, "reset_index": 0, "path": str(trial / "trial.json")}]}
        return {"status": "verified_evidence_not_fixture_release", "verification_sha256": "a" * 64,
                "physical_geometry_rejection": None}

    if tamper_executor:
        with pytest.raises(ValueError, match="executor receipt"):
            summary.compile_summary(config_path=config, workers_root=workers_root,
                                    output=tmp_path / "summary.json", verifier=verifier)
        return
    result = summary.compile_summary(config_path=config, workers_root=workers_root, output=tmp_path / "summary.json",
                                     verifier=verifier)
    retained = next(row for row in result["results"] if row["slot_index"] == slot["slot_index"])
    assert retained["retained_trial_bindings"][0]["viewport_video"]["bytes"] == len(b"video")
    assert retained["retained_trial_bindings"][0]["render_only_warmup"]["path"] == str(warmup)


def test_missing_startup_receipt_is_incomplete_not_failed(tmp_path, monkeypatch):
    config, workers_root, metadata = _fixture(tmp_path, monkeypatch)
    _rank(workers_root, metadata, 0)
    (workers_root / "0/worker-receipt.json").unlink()
    result = summary.compile_summary(config_path=config, workers_root=workers_root,
                                     output=tmp_path / "summary.json", verifier=_verifier)
    assert result["status"] == "incomplete"
    assert sum(row["unstarted"] for row in result["families"].values()) == 117
    assert sum(row["infrastructure_invalid"] for row in result["families"].values()) == 0


def test_complete_worker_cannot_claim_an_incomplete_slice(tmp_path, monkeypatch):
    config, workers_root, metadata = _fixture(tmp_path, monkeypatch)
    slots = _rank(workers_root, metadata, 0)
    _write(workers_root / "0/worker-completion.json", {
        "schema_version": worker.SCHEMA, "rank": 0, "workers": 4,
        "completed_slots": slots[:1], "status": "complete", "stopped_by_peer": False,
    })
    with pytest.raises(ValueError, match="entire frozen slice"):
        summary.compile_summary(config_path=config, workers_root=workers_root,
                                output=tmp_path / "summary.json", verifier=_verifier)


def test_infrastructure_receipt_must_bind_the_actual_slot(tmp_path, monkeypatch):
    config, workers_root, metadata = _fixture(tmp_path, monkeypatch)
    slot = _rank(workers_root, metadata, 0)[0]
    _write(workers_root / "slot-claims" / f"{slot['family'].lower()}-{slot['slot_index']:03d}.json",
           {"rank": 0, "slot": slot, "bindings_sha256": worker._digest(metadata["bindings"])})
    _write(workers_root / "0/slots" / f"{slot['family'].lower()}-{slot['slot_index']:03d}" / "executor-failure.json",
           {"error": "unbound status-only failure"})
    with pytest.raises(ValueError, match="infrastructure receipt"):
        summary.compile_summary(config_path=config, workers_root=workers_root,
                                output=tmp_path / "summary.json", verifier=_verifier)
