import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from experiments.workshops.spatial_grounding_v1 import family_partition_worker as worker


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _freeze(tmp_path: Path) -> tuple[Path, dict]:
    remaining = [
        {"family": family, "slot_index": index, "design_id": f"{family}-{index:03d}"}
        for family, indices in (("HEIGHT", range(3, 61)), ("DIST", range(4, 63)))
        for index in indices
    ][:117]
    smoke = [
        {"family": "HEIGHT", "slot_index": 0, "design_id": "HEIGHT-000"},
        {"family": "HEIGHT", "slot_index": 1, "design_id": "HEIGHT-001"},
        {"family": "DIST", "slot_index": 0, "design_id": "DIST-000"},
        {"family": "DIST", "slot_index": 3, "design_id": "DIST-003"},
    ]
    campaigns = {}
    families = []
    for family in ("HEIGHT", "DIST"):
        path = _write(tmp_path / f"{family}.json", {"family": family})
        campaigns[family] = path
        families.append({"family": family, "campaign": {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
        }})
    value = {
        "release_permitted": False, "model_requests": 0, "behavioral_episodes": 0,
        "remaining_capture_eligible_slots": remaining, "native_smoke_slots": smoke, "families": families,
    }
    freeze = _write(tmp_path / "freeze.json", value)
    return freeze, campaigns


def _config(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    freeze, campaigns = _freeze(tmp_path)
    monkeypatch.setattr(worker, "FREEZE_SHA256", hashlib.sha256(freeze.read_bytes()).hexdigest())
    source = _write(tmp_path / "source.py", {"source": "pinned"})
    calibration = _write(tmp_path / "calibration.json", {"calibration": "pinned"})
    monkeypatch.setattr(worker, "CALIBRATION_SHA256", hashlib.sha256(calibration.read_bytes()).hexdigest())
    smoke = []
    for row in _json(freeze)["native_smoke_slots"]:
        root = tmp_path / "smoke" / row["family"] / str(row["slot_index"])
        root.mkdir(parents=True)
        verification = _write(root / "verification.json", {
            "status": "verified_evidence_not_fixture_release", "release_permitted": False,
            "model_request_count": 0, "behavioral_episode_count": 0,
            "family": row["family"], "design_id": row["design_id"],
            "verification_sha256": "a" * 64,
        })
        smoke.append({
            **row, "root": str(root), "verification": str(verification),
            "verification_file_sha256": hashlib.sha256(verification.read_bytes()).hexdigest(),
            "verification_sha256": "a" * 64,
        })
    config = _write(tmp_path / "config.json", {
        "freeze_receipt": str(freeze), "campaigns": {k: str(v) for k, v in campaigns.items()},
        "source_path": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "controller_calibration": str(calibration), "capture_command": ["capture"], "qualification_command": ["qualify"],
        "smoke_evidence": smoke, "workers": 4,
        "free_space_floor_bytes": worker.MIN_FREE_SPACE_FLOOR_BYTES, "declared_slot_bytes": 1,
    })
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda _: SimpleNamespace(
        free=worker.MIN_FREE_SPACE_FLOOR_BYTES + 117,
    ))
    return config, _json(freeze)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _verifier(**kwargs):
    kwargs["output"].parent.mkdir(parents=True, exist_ok=True)
    kwargs["output"].write_text("{}")
    return {
        "status": "verified_evidence_not_fixture_release", "release_permitted": False,
        "model_request_count": 0, "behavioral_episode_count": 0,
        "family": "HEIGHT" if "HEIGHT" in str(kwargs["design_id"]) else "DIST",
        "design_id": kwargs["design_id"], "verification_sha256": "a" * 64,
    }


def _pin_smoke_receipt(config: Path, index: int, value: dict) -> None:
    raw = _json(config)
    path = Path(raw["smoke_evidence"][index]["verification"])
    _write(path, value)
    raw["smoke_evidence"][index]["verification_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(config, raw)


def test_partition_is_exact_disjoint_and_excludes_smoke(tmp_path, monkeypatch):
    config, freeze = _config(tmp_path, monkeypatch)
    rows = [worker.partition_slots(freeze, rank=rank, workers=4) for rank in range(4)]
    assert sum(map(len, rows)) == 117
    assert [row for partition in rows for row in partition] != freeze["remaining_capture_eligible_slots"]
    assert sorted(
        (row["family"], row["slot_index"]) for partition in rows for row in partition
    ) == sorted((row["family"], row["slot_index"]) for row in freeze["remaining_capture_eligible_slots"])
    assert len({(row["family"], row["slot_index"]) for partition in rows for row in partition}) == 117
    assert not {(row["family"], row["slot_index"]) for partition in rows for row in partition} & worker.SMOKE_SLOTS


def test_missing_smoke_blocks_before_slot_runner(tmp_path, monkeypatch):
    config, _ = _config(tmp_path, monkeypatch)
    value = _json(config); value["smoke_evidence"].pop(); _write(config, value)
    with pytest.raises(ValueError, match="four independently"):
        worker.run_partition(config_path=config, rank=0, root=tmp_path / "run",
                             slot_runner=lambda **_: pytest.fail("run"), verifier=_verifier)


def test_smoke_slot_in_remaining_registry_is_rejected(tmp_path, monkeypatch):
    config, freeze = _config(tmp_path, monkeypatch)
    freeze["remaining_capture_eligible_slots"][0] = {
        "family": "HEIGHT", "slot_index": 0, "design_id": "HEIGHT-000",
    }
    freeze_path = Path(_json(config)["freeze_receipt"])
    _write(freeze_path, freeze)
    monkeypatch.setattr(worker, "FREEZE_SHA256", hashlib.sha256(freeze_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="malformed"):
        worker.run_partition(config_path=config, rank=0, root=tmp_path / "run",
                             slot_runner=lambda **_: pytest.fail("run"), verifier=_verifier)


def test_physical_rejection_continues_and_existing_root_refuses(tmp_path, monkeypatch):
    config, _ = _config(tmp_path, monkeypatch)
    calls = []
    def runner(**kwargs):
        calls.append(kwargs["index"])
        return {"status": "physical_geometry_rejection_accounted_slot_no_refill"}
    result = worker.run_partition(config_path=config, rank=0, root=tmp_path / "run", slot_runner=runner, verifier=_verifier)
    assert result["status"] == "complete" and calls
    with pytest.raises(FileExistsError, match="reuse"):
        worker.run_partition(config_path=config, rank=0, root=tmp_path / "run", slot_runner=runner, verifier=_verifier)


def test_infrastructure_failure_stops_peer_before_next_slot(tmp_path, monkeypatch):
    config, _ = _config(tmp_path, monkeypatch)
    root = tmp_path / "shared" / "rank-0"
    def failing_runner(**kwargs):
        kwargs["root"].mkdir(parents=True)
        (kwargs["root"] / "partial-native-evidence.txt").write_text("preserve")
        raise RuntimeError("failure")
    with pytest.raises(RuntimeError, match="failure"):
        worker.run_partition(
            config_path=config, rank=0, root=root,
            slot_runner=failing_runner,
            verifier=_verifier,
        )
    assert (root / "slots" / "height-003" / "partial-native-evidence.txt").read_text() == "preserve"
    if (root / "completed").exists():
        assert not list((root / "completed").glob("*.json"))
    peer_calls = []
    result = worker.run_partition(
        config_path=config, rank=1, root=tmp_path / "shared" / "rank-1",
        slot_runner=lambda **kwargs: peer_calls.append(kwargs) or {"status": "physical_geometry_rejection_accounted_slot_no_refill"},
        verifier=_verifier,
    )
    assert result["status"] == "stopped_before_next_slot" and not peer_calls


def test_mutated_smoke_receipt_blocks_before_slot_runner(tmp_path, monkeypatch):
    config, _ = _config(tmp_path, monkeypatch)
    value = _json(config)
    verification = Path(value["smoke_evidence"][0]["verification"])
    _write(verification, {"status": "bad"})
    with pytest.raises(ValueError, match="identity differs"):
        worker.run_partition(config_path=config, rank=0, root=tmp_path / "run",
                             slot_runner=lambda **_: pytest.fail("run"), verifier=_verifier)


@pytest.mark.parametrize("forged", [
    {"status": "verified_evidence_not_fixture_release"},
    {
        "status": "verified_evidence_not_fixture_release", "release_permitted": False,
        "model_request_count": 0, "behavioral_episode_count": 0,
        "family": "DIST", "design_id": "HEIGHT-000", "verification_sha256": "a" * 64,
    },
])
def test_rehashed_forged_smoke_receipt_still_blocks(tmp_path, monkeypatch, forged):
    config, _ = _config(tmp_path, monkeypatch)
    _pin_smoke_receipt(config, 0, forged)
    with pytest.raises(ValueError, match="not independently"):
        worker.run_partition(config_path=config, rank=0, root=tmp_path / "run",
                             slot_runner=lambda **_: pytest.fail("run"), verifier=_verifier)


def test_storage_reserves_all_117_slots(tmp_path, monkeypatch):
    config, _ = _config(tmp_path, monkeypatch)
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda _: SimpleNamespace(
        free=worker.MIN_FREE_SPACE_FLOOR_BYTES + 116,
    ))
    with pytest.raises(RuntimeError, match="storage allowance"):
        worker.run_partition(config_path=config, rank=0, root=tmp_path / "run",
                             slot_runner=lambda **_: pytest.fail("run"), verifier=_verifier)


def test_authoritative_smoke_recheck_rejects_wrong_campaign_digest(tmp_path, monkeypatch):
    config, _ = _config(tmp_path, monkeypatch)
    def wrong_campaign_verifier(**kwargs):
        result = _verifier(**kwargs)
        result["verification_sha256"] = "b" * 64
        return result
    with pytest.raises(ValueError, match="raw evidence differs"):
        worker.run_partition(config_path=config, rank=0, root=tmp_path / "run",
                             slot_runner=lambda **_: pytest.fail("run"), verifier=wrong_campaign_verifier)


def test_duplicate_rank_distinct_root_is_rejected(tmp_path, monkeypatch):
    config, _ = _config(tmp_path, monkeypatch)
    runner = lambda **_: {"status": "physical_geometry_rejection_accounted_slot_no_refill"}
    worker.run_partition(config_path=config, rank=0, root=tmp_path / "shared" / "first",
                         slot_runner=runner, verifier=_verifier)
    with pytest.raises(FileExistsError):
        worker.run_partition(config_path=config, rank=0, root=tmp_path / "shared" / "second",
                             slot_runner=runner, verifier=_verifier)


def test_concurrent_identity_and_stop_prevent_next_claim(tmp_path, monkeypatch):
    config, _ = _config(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    calls: list[int] = []
    peer_error: list[Exception] = []

    def peer_runner(**kwargs):
        calls.append(kwargs["index"])
        entered.set()
        assert release.wait(3)
        return {"status": "physical_geometry_rejection_accounted_slot_no_refill"}

    def peer():
        try:
            worker.run_partition(config_path=config, rank=1, root=tmp_path / "shared" / "rank-1",
                                 slot_runner=peer_runner, verifier=_verifier)
        except Exception as error:  # pragma: no cover - assertion after join
            peer_error.append(error)

    thread = threading.Thread(target=peer)
    thread.start()
    assert entered.wait(3)
    with pytest.raises(RuntimeError, match="origin failure"):
        worker.run_partition(
            config_path=config, rank=0, root=tmp_path / "shared" / "rank-0",
            slot_runner=lambda **_: (_ for _ in ()).throw(RuntimeError("origin failure")),
            verifier=_verifier,
        )
    release.set()
    thread.join(3)
    assert not thread.is_alive() and not peer_error and len(calls) == 1
    assert _json(tmp_path / "shared" / "partition-binding.json")["workers"] == 4


def test_startup_recheck_failure_stops_inflight_peer_before_next_claim(tmp_path, monkeypatch):
    config, _ = _config(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    calls: list[int] = []
    peer_error: list[Exception] = []

    def peer_runner(**kwargs):
        calls.append(kwargs["index"])
        entered.set()
        assert release.wait(3)
        return {"status": "physical_geometry_rejection_accounted_slot_no_refill"}

    def peer():
        try:
            worker.run_partition(
                config_path=config, rank=1, root=tmp_path / "shared" / "rank-1",
                slot_runner=peer_runner, verifier=_verifier,
            )
        except Exception as error:  # pragma: no cover - asserted after join
            peer_error.append(error)

    thread = threading.Thread(target=peer)
    thread.start()
    assert entered.wait(3)
    with pytest.raises(RuntimeError, match="recheck failure"):
        worker.run_partition(
            config_path=config, rank=0, root=tmp_path / "shared" / "rank-0",
            slot_runner=lambda **_: pytest.fail("slot runner must not start"),
            verifier=lambda **_: (_ for _ in ()).throw(RuntimeError("recheck failure")),
        )
    release.set()
    thread.join(3)
    sentinel = _json(tmp_path / "shared" / "infrastructure-stop.json")
    assert not thread.is_alive() and not peer_error and len(calls) == 1
    assert sentinel["origin_rank"] == 0 and "recheck failure" in sentinel["error"]
