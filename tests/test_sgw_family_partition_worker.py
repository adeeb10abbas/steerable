import hashlib
import json
from pathlib import Path

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
        })
        smoke.append({**row, "verification": str(verification)})
    config = _write(tmp_path / "config.json", {
        "freeze_receipt": str(freeze), "campaigns": {k: str(v) for k, v in campaigns.items()},
        "source_path": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "controller_calibration": str(calibration), "capture_command": ["capture"], "qualification_command": ["qualify"],
        "smoke_evidence": smoke, "workers": 4, "free_space_floor_bytes": 0, "declared_slot_bytes": 1,
    })
    return config, _json(freeze)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


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
        worker.run_partition(config_path=config, rank=0, root=tmp_path / "run", slot_runner=lambda **_: pytest.fail("run"))


def test_smoke_slot_in_remaining_registry_is_rejected(tmp_path, monkeypatch):
    config, freeze = _config(tmp_path, monkeypatch)
    freeze["remaining_capture_eligible_slots"][0] = {
        "family": "HEIGHT", "slot_index": 0, "design_id": "HEIGHT-000",
    }
    freeze_path = Path(_json(config)["freeze_receipt"])
    _write(freeze_path, freeze)
    monkeypatch.setattr(worker, "FREEZE_SHA256", hashlib.sha256(freeze_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="malformed"):
        worker.run_partition(config_path=config, rank=0, root=tmp_path / "run", slot_runner=lambda **_: pytest.fail("run"))


def test_physical_rejection_continues_and_existing_root_refuses(tmp_path, monkeypatch):
    config, _ = _config(tmp_path, monkeypatch)
    calls = []
    def runner(**kwargs):
        calls.append(kwargs["index"])
        return {"status": "physical_geometry_rejection_accounted_slot_no_refill"}
    result = worker.run_partition(config_path=config, rank=0, root=tmp_path / "run", slot_runner=runner)
    assert result["status"] == "complete" and calls
    with pytest.raises(FileExistsError, match="reuse"):
        worker.run_partition(config_path=config, rank=0, root=tmp_path / "run", slot_runner=runner)


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
        )
    assert (root / "slots" / "height-003" / "partial-native-evidence.txt").read_text() == "preserve"
    if (root / "completed").exists():
        assert not list((root / "completed").glob("*.json"))
    peer_calls = []
    result = worker.run_partition(
        config_path=config, rank=1, root=tmp_path / "shared" / "rank-1",
        slot_runner=lambda **kwargs: peer_calls.append(kwargs) or {"status": "physical_geometry_rejection_accounted_slot_no_refill"},
    )
    assert result["status"] == "stopped_before_next_slot" and not peer_calls


def test_smoke_verification_failure_blocks_before_slot_runner(tmp_path, monkeypatch):
    config, _ = _config(tmp_path, monkeypatch)
    value = _json(config)
    verification = Path(value["smoke_evidence"][0]["verification"])
    _write(verification, {"status": "bad"})
    with pytest.raises(ValueError, match="not independently"):
        worker.run_partition(config_path=config, rank=0, root=tmp_path / "run", slot_runner=lambda **_: pytest.fail("run"))
