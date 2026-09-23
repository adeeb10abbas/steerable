from copy import deepcopy
import json

import pytest

from experiments.workshops.spatial_grounding_v1.capture_sequencer import (
    activation_patch, check_job, predecessor_released, run, spec_digest,
)


def job(name, *, complete=False, suspended=False):
    return {
        "metadata": {
            "name": name, "uid": f"{name}-uid", "resourceVersion": "1",
            "labels": {"owner": "ali", "app.kubernetes.io/name": "sgw-01"},
        },
        "spec": {
            "suspend": suspended, "completions": 4, "parallelism": 4,
            "template": {"spec": {"containers": [
                {"resources": {"limits": {"nvidia.com/gpu": "1"}}},
            ]}},
        },
        "status": {
            "active": 0 if complete or suspended else 4,
            "succeeded": 4 if complete else 0,
            "conditions": [{"type": "Complete", "status": "True"}] if complete else [],
        },
    }


def identity(value):
    return {"name": value["metadata"]["name"], "uid": value["metadata"]["uid"],
            "spec_sha256": spec_digest(value)}


def test_completion_waits_for_actual_allocation_release():
    value = job("producer", complete=True)
    assert predecessor_released(value)
    value["status"]["terminating"] = 1
    assert not predecessor_released(value)
    value["status"]["terminating"] = 0
    value["status"]["succeeded"] = 3
    assert not predecessor_released(value)
    value["status"]["conditions"] = [{"type": "Failed", "status": "True"}]
    with pytest.raises(ValueError, match="predecessor failed"):
        predecessor_released(value)


def test_frozen_spec_and_optimistic_activation():
    value = job("capture", suspended=True)
    expected = identity(value)
    patch = activation_patch(value)
    assert patch[0] == {"op": "test", "path": "/metadata/uid", "value": "capture-uid"}
    assert patch[1]["path"] == "/metadata/resourceVersion"
    changed = deepcopy(value)
    changed["spec"]["suspend"] = False
    check_job(changed, expected)
    changed["spec"]["parallelism"] = 5
    with pytest.raises(ValueError, match="specification changed"):
        check_job(changed, expected)


def test_durable_release_occurs_only_after_predecessor_then_records_completion(tmp_path):
    producer, capture = job("producer"), job("capture", suspended=True)
    events = []

    class Cluster:
        def request(self, name, patch=None):
            if patch is not None:
                assert predecessor_released(producer)
                assert name == "capture"
                events.append(patch)
                capture["spec"]["suspend"] = False
                capture["status"] = job("capture", complete=True)["status"]
            return deepcopy(producer if name == "producer" else capture)

    plan = {
        "scope": "zero_model_prospective_capture_only", "gpu_ceiling": 4,
        "predecessor": identity(producer), "target": identity(capture), "timeout_seconds": 100,
    }

    def wait(seconds):
        assert not events
        producer["status"] = job("producer", complete=True)["status"]

    run(plan, Cluster(), tmp_path / "receipt", wait=wait)
    assert len(events) == 1
    result = json.loads((tmp_path / "receipt/receipt.json").read_text())
    assert result["status"] == "zero_model_capture_job_completed_requires_evidence_review"
    assert result["release_permitted"] is False


def test_failed_predecessor_never_activates_target(tmp_path):
    producer, capture = job("producer"), job("capture", suspended=True)
    producer["status"]["conditions"] = [{"type": "Failed", "status": "True"}]

    class Cluster:
        def request(self, name, patch=None):
            assert patch is None
            return producer if name == "producer" else capture

    plan = {
        "scope": "zero_model_prospective_capture_only", "gpu_ceiling": 4,
        "predecessor": identity(producer), "target": identity(capture), "timeout_seconds": 100,
    }
    with pytest.raises(ValueError, match="predecessor failed"):
        run(plan, Cluster(), tmp_path / "receipt")
    result = json.loads((tmp_path / "receipt/receipt.json").read_text())
    assert result["status"] == "blocked"
