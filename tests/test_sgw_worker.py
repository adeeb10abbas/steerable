import fcntl
import os
from pathlib import Path
import subprocess
import sys
import json
import math
import hashlib
from datetime import datetime, timedelta, timezone

import experiments.workshops.spatial_grounding_v1.worker as worker_module
import pytest
from experiments.workshops.spatial_grounding_v1.contract import load_release
from experiments.workshops.spatial_grounding_v1.worker import (
    EXIT_ATTEMPTS_EXHAUSTED, EXIT_STORAGE_BUDGET_BLOCKED, ResourceBlocked, run_partition,
)
from tests.test_sgw_contract import make_release


class FakeAdapter:
    def __init__(self, status="valid_model_failure"):
        self.status = status
        self.resets = 0

    def reset(self, cell, recorder):
        self.resets += 1
        return {"full_reset": True, "reset_number": self.resets}

    def run_episode(self, cell, recorder, reset):
        recorder.request({"request_id": f"r{self.resets}", "request_index": 0,
                          "started_at_utc": "2026-01-01T00:00:00Z",
                          "prompt_sha256": cell.row["prompt_sha256"], "reset_sha256": "reset"})
        for directory, filename in (("actions", "executed.json"), ("states", "states.json"),
                                    ("observations", "frames.json"), ("videos", "viewport.mp4")):
            path = recorder.path / directory / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("retained")
        return {"status": self.status, "failure_reason": "policy failure",
                "executed_action_count": 450, "safety_terminated": False,
                "episode_mapping": [{"action_step": step} for step in range(451)]}

    def close(self):
        pass


def test_worker_preserves_model_failures_and_skips_them_on_resume(tmp_path: Path) -> None:
    release = load_release(make_release(tmp_path))
    adapter = FakeAdapter()
    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6,
                         max_attempts=3, worker_id="test", adapter=adapter,
                         scorer=lambda _trace, _cell: {"status": "valid_model_failure"}) == 0
    assert adapter.resets == 6
    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6,
                         max_attempts=3, worker_id="resume", adapter=adapter,
                         scorer=lambda _trace, _cell: {"status": "valid_model_failure"}) == 0
    assert adapter.resets == 6


def test_attempt_limit_survives_restart(tmp_path: Path) -> None:
    release = load_release(make_release(tmp_path))

    class Invalid(FakeAdapter):
        def run_episode(self, cell, recorder, reset):
            return {"status": "technical_invalid", "technical_cause": "renderer lost"}

    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6,
                         max_attempts=3, worker_id="test", adapter=Invalid()) == EXIT_ATTEMPTS_EXHAUSTED
    assert len(list((release.root.parent / "attempts" / "cell-0").iterdir())) == 3


def test_flock_blocks_a_second_process(tmp_path: Path) -> None:
    lock = tmp_path / "model.lock"
    lock.write_text("")
    with lock.open() as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = (
            "import fcntl,sys; f=open(sys.argv[1]);\n"
            "try: fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
            "except BlockingIOError: raise SystemExit(0)\n"
            "raise SystemExit(1)"
        )
        assert subprocess.run([sys.executable, "-c", code, str(lock)], check=False).returncode == 0


def _rehash_binding(release):
    import hashlib
    hashes = json.loads((release.root / "hashes.json").read_text())
    hashes["runtime_binding.json"] = hashlib.sha256((release.root / "runtime_binding.json").read_bytes()).hexdigest()
    (release.root / "hashes.json").write_text(json.dumps(hashes))


def test_floor_override_blocks_before_adapter_start(tmp_path: Path) -> None:
    release = load_release(make_release(tmp_path))
    binding = dict(release.binding)
    binding["minimum_free_bytes"] = 1
    (release.root / "runtime_binding.json").write_text(json.dumps(binding))
    _rehash_binding(release)
    release = load_release(release.root)

    class MustNotStart:
        def reset(self, *_args):
            raise AssertionError("resource gate must run before model work")
        def close(self):
            pass

    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6, max_attempts=3,
                         worker_id="storage", adapter=MustNotStart(),
                         scorer=lambda _trace, _cell: {}) == EXIT_STORAGE_BUDGET_BLOCKED


def test_storage_uses_global_remaining_not_partition_size(tmp_path: Path, monkeypatch) -> None:
    release = load_release(make_release(tmp_path))
    storage = tmp_path / "storage.json"
    identity = json.loads(Path(release.binding["resource_budget_receipt"]["path"]).read_text())
    storage.write_text(json.dumps({
        "status": "approved", "source_queue_sha256": "q" * 64,
        "release_id": release.release_id, "source_queue_episode_count": 1044,
        "pvc_name": identity["pvc_name"], "pvc_mount_path": identity["pvc_mount_path"],
        "study_root": identity["study_root"], "runtime_identity_sha256": identity["runtime_identity_sha256"],
        "pilot_p95_episode_bytes": 300 * 1024**2, "global_remaining_episode_count": 1044,
        "expires_at_utc": "2030-01-01T00:00:00Z",
    }))
    binding = dict(release.binding)
    binding["storage_budget_receipt"] = {"path": str(storage), "sha256": __import__("hashlib").sha256(storage.read_bytes()).hexdigest()}
    (release.root / "runtime_binding.json").write_text(json.dumps(binding))
    _rehash_binding(release)
    release = load_release(release.root)
    monkeypatch.setattr("experiments.workshops.spatial_grounding_v1.worker.shutil.disk_usage",
                        lambda _path: type("Disk", (), {"free": 200 * 1024**3})())

    class MustNotStart:
        def reset(self, *_args):
            raise AssertionError("global storage accounting must block before model start")
        def close(self):
            pass

    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6, max_attempts=3,
                         worker_id="global", adapter=MustNotStart(),
                         scorer=lambda _trace, _cell: {}) == EXIT_STORAGE_BUDGET_BLOCKED


def test_missing_budget_blocks_before_adapter_start(tmp_path: Path) -> None:
    release = load_release(make_release(tmp_path))
    binding = dict(release.binding)
    binding.pop("resource_budget_receipt")
    (release.root / "runtime_binding.json").write_text(json.dumps(binding))
    _rehash_binding(release)
    release = load_release(release.root)

    class MustNotStart:
        def reset(self, *_args):
            raise AssertionError("missing budget must block before model start")
        def close(self):
            pass

    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6, max_attempts=3,
                         worker_id="budget", adapter=MustNotStart(),
                         scorer=lambda _trace, _cell: {}) == EXIT_STORAGE_BUDGET_BLOCKED


def test_expired_budget_blocks_before_adapter_start(tmp_path: Path) -> None:
    release = load_release(make_release(tmp_path))
    receipt = Path(release.binding["resource_budget_receipt"]["path"])
    budget = json.loads(receipt.read_text())
    budget["expires_at_utc"] = "2000-01-01T00:00:00Z"
    receipt.write_text(json.dumps(budget))
    binding = dict(release.binding)
    binding["resource_budget_receipt"] = {"path": str(receipt), "sha256": __import__("hashlib").sha256(receipt.read_bytes()).hexdigest()}
    (release.root / "runtime_binding.json").write_text(json.dumps(binding))
    _rehash_binding(release)
    release = load_release(release.root)

    class MustNotStart:
        def reset(self, *_args):
            raise AssertionError("expired budget must block before model start")
        def close(self):
            pass

    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6, max_attempts=3,
                         worker_id="expired", adapter=MustNotStart(),
                         scorer=lambda _trace, _cell: {}) == EXIT_STORAGE_BUDGET_BLOCKED


def test_storage_receipt_must_cover_full_source_queue(tmp_path: Path) -> None:
    release = load_release(make_release(tmp_path))
    budget = json.loads(Path(release.binding["resource_budget_receipt"]["path"]).read_text())
    storage = tmp_path / "storage.json"
    storage.write_text(json.dumps({
        **{key: budget[key] for key in ("release_id", "source_queue_sha256", "source_queue_episode_count",
                                        "pvc_name", "pvc_mount_path", "study_root", "runtime_identity_sha256")},
        "status": "approved", "expires_at_utc": "2030-01-01T00:00:00Z",
        "pilot_p95_episode_bytes": 1, "global_remaining_episode_count": 6,
    }))
    binding = dict(release.binding)
    binding["storage_budget_receipt"] = {"path": str(storage), "sha256": __import__("hashlib").sha256(storage.read_bytes()).hexdigest()}
    (release.root / "runtime_binding.json").write_text(json.dumps(binding))
    _rehash_binding(release)
    release = load_release(release.root)
    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6, max_attempts=3,
                         worker_id="global-1044", adapter=FakeAdapter(),
                         scorer=lambda _trace, _cell: {}) == EXIT_STORAGE_BUDGET_BLOCKED


@pytest.mark.parametrize("value", [True, math.nan, math.inf])
def test_budget_rejects_bool_nan_and_infinity_before_attempts(tmp_path: Path, value) -> None:
    release = load_release(make_release(tmp_path))
    receipt = Path(release.binding["resource_budget_receipt"]["path"])
    budget = json.loads(receipt.read_text())
    budget["estimated_remaining_gpu_hours"] = value
    receipt.write_text(json.dumps(budget))
    binding = dict(release.binding)
    binding["resource_budget_receipt"] = {"path": str(receipt), "sha256": __import__("hashlib").sha256(receipt.read_bytes()).hexdigest()}
    (release.root / "runtime_binding.json").write_text(json.dumps(binding))
    _rehash_binding(release)
    release = load_release(release.root)
    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6, max_attempts=3,
                         worker_id="bad-number", adapter=FakeAdapter(),
                         scorer=lambda _trace, _cell: {}) == EXIT_STORAGE_BUDGET_BLOCKED
    assert not (release.root.parent / "attempts").exists()


def test_allocation_is_rechecked_before_each_unfinished_cell(tmp_path: Path, monkeypatch) -> None:
    release = load_release(make_release(tmp_path))
    original = worker_module._allocation_check
    calls = 0

    def expire_after_first_cell(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 2:  # startup, cell 0, then cell 1
            raise ResourceBlocked("external allocation expired")
        return original(*args, **kwargs)

    monkeypatch.setattr(worker_module, "_allocation_check", expire_after_first_cell)
    adapter = FakeAdapter()
    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6, max_attempts=3,
                         worker_id="between-cells", adapter=adapter,
                         scorer=lambda _trace, _cell: {"status": "valid_model_failure"}) == EXIT_STORAGE_BUDGET_BLOCKED
    assert adapter.resets == 1
    assert not (release.root.parent / "attempts" / "cell-1").exists()


def test_completed_resume_never_loads_adapter_or_checks_expired_receipts(tmp_path: Path, monkeypatch) -> None:
    release = load_release(make_release(tmp_path))
    adapter = FakeAdapter()
    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6, max_attempts=3,
                         worker_id="complete", adapter=adapter,
                         scorer=lambda _trace, _cell: {"status": "valid_model_failure"}) == 0
    monkeypatch.setattr(worker_module, "load_adapter", lambda _model: (_ for _ in ()).throw(AssertionError("must not load")))
    monkeypatch.setattr(worker_module, "_space_check", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not check")))
    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6, max_attempts=3,
                         worker_id="resume-no-load", scorer=lambda _trace, _cell: {}) == 0


@pytest.mark.parametrize("field,value", [
    ("deadline_utc", "2099-01-01T00:00:00Z"),
    ("startTime", "2099-01-01T00:00:00Z"),
])
def test_allocation_rejects_forged_or_future_job_interval(tmp_path: Path, field: str, value: str) -> None:
    release = load_release(make_release(tmp_path))
    receipt = Path(release.binding["external_allocation_receipt"]["path"])
    allocation = json.loads(receipt.read_text())
    allocation[field] = value
    receipt.write_text(json.dumps(allocation))
    binding = dict(release.binding)
    binding["external_allocation_receipt"] = {"path": str(receipt), "sha256": __import__("hashlib").sha256(receipt.read_bytes()).hexdigest()}
    (release.root / "runtime_binding.json").write_text(json.dumps(binding))
    _rehash_binding(release)
    release = load_release(release.root)
    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6, max_attempts=3,
                         worker_id="interval", adapter=FakeAdapter(),
                         scorer=lambda _trace, _cell: {}) == EXIT_STORAGE_BUDGET_BLOCKED


def test_retry_rechecks_resource_gate_before_next_attempt(tmp_path: Path, monkeypatch) -> None:
    release = load_release(make_release(tmp_path))
    calls = 0
    original = worker_module._budget_check

    def reject_second_attempt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 2:  # startup, first attempt, then retry
            raise ResourceBlocked("budget expired between attempts")
        return original(*args, **kwargs)

    class Invalid(FakeAdapter):
        def run_episode(self, cell, recorder, reset):
            return {"status": "technical_invalid", "technical_cause": "temporary renderer failure"}

    monkeypatch.setattr(worker_module, "_budget_check", reject_second_attempt)
    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6, max_attempts=3,
                         worker_id="retry-gate", adapter=Invalid()) == EXIT_STORAGE_BUDGET_BLOCKED
    assert len(list((release.root.parent / "attempts" / "cell-0").iterdir())) == 1


def _replace_receipt(release, key, contents):
    binding = dict(release.binding)
    path = Path(binding[key]["path"])
    path.write_text(json.dumps(contents))
    binding[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (release.root / "runtime_binding.json").write_text(json.dumps(binding))
    _rehash_binding(release)
    return load_release(release.root)


def _expanded_authorization(release):
    authorization = json.loads(Path(release.binding["operational_authorization_receipt"]["path"]).read_text())
    authorization["budget_mode"] = "existing_idle_capacity_no_aggregate_hour_cap"
    authorization["constraints"].update({
        "allocation_scaling": "as_needed_verified_idle_capacity",
        "max_concurrent_model_workers": None,
        "max_total_allocated_gpus": None,
    })
    return authorization


def _replace_binding(release, **changes):
    binding = dict(release.binding)
    binding.update(changes)
    (release.root / "runtime_binding.json").write_text(json.dumps(binding))
    _rehash_binding(release)
    return load_release(release.root)


def test_explicit_as_needed_authorization_allows_finite_expanded_binding(tmp_path):
    release = load_release(make_release(tmp_path))
    release = _replace_receipt(release, "operational_authorization_receipt", _expanded_authorization(release))
    release = _replace_binding(
        release, max_concurrent_model_workers=5, max_total_allocated_gpus=8,
        model_gpu_counts={"N3": 5, "D1": 3},
    )
    assert worker_module._authorization_check(release, model="N3") == "existing_idle_capacity_no_aggregate_hour_cap"
    worker_module._budget_check(release, stage="P", model="N3", minimum_runtime_seconds=1200)


@pytest.mark.parametrize("mutate", [
    lambda authorization: authorization["constraints"].pop("max_total_allocated_gpus"),
    lambda authorization: authorization["constraints"].update({"max_concurrent_model_workers": 5, "max_total_allocated_gpus": None}),
    lambda authorization: authorization.update({"status": "unapproved"}),
    lambda authorization: authorization["constraints"].update({"allocation_scaling": "unbounded"}),
])
def test_as_needed_authorization_requires_explicit_approved_null_caps(tmp_path, mutate):
    release = load_release(make_release(tmp_path))
    authorization = _expanded_authorization(release)
    mutate(authorization)
    release = _replace_receipt(release, "operational_authorization_receipt", authorization)
    release = _replace_binding(release, max_concurrent_model_workers=5, max_total_allocated_gpus=8,
                               model_gpu_counts={"N3": 5, "D1": 3})
    with pytest.raises(ResourceBlocked):
        worker_module._authorization_check(release, model="N3")


@pytest.mark.parametrize("workers,gpus,counts", [(0, 8, {"N3": 5}), (5, 0, {"N3": 5}), (5, 8, {"N3": 9}), (5.0, 8, {"N3": 5})])
def test_expanded_authorization_still_requires_finite_positive_runtime_plan(tmp_path, workers, gpus, counts):
    release = load_release(make_release(tmp_path))
    release = _replace_receipt(release, "operational_authorization_receipt", _expanded_authorization(release))
    release = _replace_binding(release, max_concurrent_model_workers=workers, max_total_allocated_gpus=gpus,
                               model_gpu_counts=counts)
    with pytest.raises(ResourceBlocked):
        worker_module._budget_check(release, stage="P", model="N3", minimum_runtime_seconds=1200)


def test_legacy_authorization_retains_two_worker_four_gpu_ceiling(tmp_path):
    release = load_release(make_release(tmp_path))
    release = _replace_binding(release, max_concurrent_model_workers=3, max_total_allocated_gpus=5,
                               model_gpu_counts={"N3": 3, "D1": 2})
    with pytest.raises(ResourceBlocked):
        worker_module._authorization_check(release, model="N3")


@pytest.mark.parametrize("fault", [
    "before_job", "future", "stale", "nan", "busy_memory", "busy_utilization",
    "insufficient_memory", "wrong_count", "unhashable_uuid", "second_gpu_busy",
])
def test_idle_proof_rejects_bad_timing_or_any_busy_device_before_load(tmp_path, monkeypatch, fault):
    release = load_release(make_release(tmp_path))
    allocation = json.loads(Path(release.binding["external_allocation_receipt"]["path"]).read_text())
    idle_path = Path(allocation["gpu_idle_probe_receipt"]["path"])
    idle = json.loads(idle_path.read_text())
    now = datetime.now(timezone.utc)
    if fault == "before_job":
        idle["observed_at_unix"] = datetime.fromisoformat(allocation["startTime"]).timestamp() - 1
    elif fault == "future":
        idle["observed_at_unix"] = now.timestamp() + 60
    elif fault == "stale":
        start = now - timedelta(seconds=1000)
        allocation["startTime"] = start.isoformat()
        allocation["deadline_utc"] = (start + timedelta(seconds=allocation["activeDeadlineSeconds"])).isoformat()
        idle["observed_at_unix"] = now.timestamp() - 600
    elif fault == "nan":
        idle["observed_at_unix"] = math.nan
    elif fault in {"busy_memory", "busy_utilization", "insufficient_memory"}:
        field, value = {
            "busy_memory": ("memory_used_mib", 94000),
            "busy_utilization": ("utilization_percent", 1),
            "insufficient_memory": ("memory_free_mib", 16383),
        }[fault]
        idle["gpus"][0][field] = value
        idle["selected_gpu"] = dict(idle["gpus"][0])
    elif fault == "wrong_count":
        idle["expected_visible_gpu_count"] = 4
    elif fault == "unhashable_uuid":
        allocation["allocated_gpu_uuids"] = [[]]
    else:
        binding = dict(release.binding)
        binding["model_gpu_counts"] = {"N3": 2, "D1": 2}
        (release.root / "runtime_binding.json").write_text(json.dumps(binding))
        _rehash_binding(release)
        release = load_release(release.root)
        allocation["allocated_gpu_count"] = 2
        allocation["allocated_gpu_uuids"].append("GPU-busy")
        allocation["reservation_gpu_hours"] *= 2
        idle["expected_visible_gpu_count"] = 2
        idle["gpus"].append({**idle["gpus"][0], "index": 1, "uuid": "GPU-busy", "memory_used_mib": 94000})
    idle_path.write_text(json.dumps(idle))
    allocation["gpu_idle_probe_receipt"]["sha256"] = hashlib.sha256(idle_path.read_bytes()).hexdigest()
    release = _replace_receipt(release, "external_allocation_receipt", allocation)
    monkeypatch.setattr(worker_module, "load_adapter", lambda _: pytest.fail("resource refusal must precede model load"))
    assert run_partition(release, model="N3", family="LAT", stage="P", max_valid=6,
                         max_attempts=3, worker_id="bad-idle") == EXIT_STORAGE_BUDGET_BLOCKED
    assert not (release.root.parent / "attempts").exists()
    status = json.loads((release.root.parent / "status" / "bad-idle.json").read_text())
    assert status["state"] == "blocked"


def test_genuine_idle_probe_is_after_job_start_and_only_required_before_load(tmp_path):
    release = load_release(make_release(tmp_path))
    assert worker_module._allocation_check(release, model="N3", minimum_runtime_seconds=1200) > 1200
    allocation = json.loads(Path(release.binding["external_allocation_receipt"]["path"]).read_text())
    idle_path = Path(allocation["gpu_idle_probe_receipt"]["path"])
    idle = json.loads(idle_path.read_text())
    idle["observed_at_unix"] -= 600
    idle_path.write_text(json.dumps(idle))
    with pytest.raises(ResourceBlocked):
        worker_module._allocation_check(release, model="N3", minimum_runtime_seconds=1200)
    assert worker_module._allocation_check(
        release, model="N3", minimum_runtime_seconds=1200, verify_idle_probe=False,
    ) > 1200


@pytest.mark.parametrize("approved,reserved", [(1000.0, 5.0), (10.0, 1.0), (10.0, 11.0)])
def test_numeric_allocation_cannot_invent_cap_or_omit_own_reservation(tmp_path, approved, reserved):
    release = load_release(make_release(tmp_path))
    authorization = json.loads(Path(release.binding["operational_authorization_receipt"]["path"]).read_text())
    authorization.update(budget_mode="numeric_global_gpu_hour_cap", approved_global_gpu_hours=10.0)
    release = _replace_receipt(release, "operational_authorization_receipt", authorization)
    allocation = json.loads(Path(release.binding["external_allocation_receipt"]["path"]).read_text())
    allocation.update(budget_mode="numeric_global_gpu_hour_cap",
                      approved_global_gpu_hours=approved, globally_reserved_gpu_hours=reserved)
    release = _replace_receipt(release, "external_allocation_receipt", allocation)
    with pytest.raises(ResourceBlocked, match="capped global reservation"):
        worker_module._allocation_check(release, model="N3", minimum_runtime_seconds=1200)
