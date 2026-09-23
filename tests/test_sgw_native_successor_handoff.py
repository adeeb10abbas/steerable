import copy
import json
from urllib.error import URLError

import pytest

from experiments.workshops.spatial_grounding_v1 import native_successor_handoff as handoff


def _inputs():
    pods = [{
        "metadata": {
            "name": f"worker-{index}", "uid": f"worker-uid-{index}", "namespace": handoff.NAMESPACE,
            "ownerReferences": [{"kind": "Job", "uid": "source-job", "controller": True}],
            "annotations": {"batch.kubernetes.io/job-completion-index": str(index)},
        },
        "spec": {"nodeName": f"node-{index}", "restartPolicy": "Never",
                 "containers": [{"resources": {"requests": {"nvidia.com/gpu": "1"}}}]},
        "status": {"phase": "Running"},
    } for index in range(4)]
    job = {
        "metadata": {"name": "source", "uid": "source-job", "namespace": handoff.NAMESPACE},
        "spec": {"completions": 4, "parallelism": 4, "completionMode": "Indexed", "backoffLimit": 0},
        "status": {},
    }
    target = {
        "metadata": {"name": "successor-pod", "uid": "successor-pod-uid", "namespace": handoff.NAMESPACE,
                     "ownerReferences": [{"kind": "Job", "uid": "target-job", "controller": True}]},
        "spec": {"schedulerName": "sgw01-ali-example", "priority": 0,
                 "containers": [{"resources": {"requests": {"nvidia.com/gpu": "1"}}}]},
        "status": {"phase": "Pending"},
    }
    config = {
        "schema_version": handoff.SCHEMA, "namespace": handoff.NAMESPACE, "timeout_seconds": 3600,
        "predecessor": {"name": "source", "uid": "source-job", "pods": [
            {"name": pod["metadata"]["name"], "uid": pod["metadata"]["uid"],
             "node": pod["spec"]["nodeName"], "index": index}
            for index, pod in enumerate(pods)
        ]},
        "target": {"name": "successor-pod", "uid": "successor-pod-uid", "job_uid": "target-job",
                   "scheduler_name": "sgw01-ali-example", "spec_sha256": handoff.digest(target["spec"])},
    }
    return config, job, pods, target


def _succeed(job, pods, index=2):
    job["status"]["completedIndexes"] = str(index)
    pods[index]["status"] = {
        "phase": "Succeeded",
        "containerStatuses": [{"state": {"terminated": {"exitCode": 0}}, "restartCount": 0}],
    }


def test_no_binding_until_natural_success_is_acknowledged_by_job():
    config, job, pods, target = _inputs()
    assert handoff.select_node(config, job, pods, target) is None
    _succeed(job, pods)
    job["status"].clear()
    assert handoff.select_node(config, job, pods, target) is None
    job["status"]["completedIndexes"] = "2"
    assert handoff.select_node(config, job, pods, target) == "node-2"


@pytest.mark.parametrize("status", [
    {"phase": "Failed"},
    {"phase": "Succeeded"},
    {"phase": "Succeeded", "containerStatuses": [{"state": {"terminated": {"exitCode": 1}}, "restartCount": 0}]},
    {"phase": "Succeeded", "containerStatuses": [{"state": {"running": {}}, "restartCount": 0}]},
])
def test_failed_or_incomplete_source_is_not_capacity(status):
    config, job, pods, target = _inputs()
    job["status"]["completedIndexes"] = "2"
    pods[2]["status"] = status
    assert handoff.select_node(config, job, pods, target) is None


@pytest.mark.parametrize("mutate", [
    lambda job, pods, target: job["metadata"].update(uid="replacement-job"),
    lambda job, pods, target: job["spec"].update(backoffLimit=1),
    lambda job, pods, target: job["spec"].update(parallelism=5),
    lambda job, pods, target: pods[2]["metadata"].update(uid="replacement-pod"),
    lambda job, pods, target: pods[2]["metadata"].update(deletionTimestamp="now"),
    lambda job, pods, target: pods[2]["spec"].update(nodeName="unregistered-node"),
    lambda job, pods, target: target["metadata"].update(uid="replacement-target"),
    lambda job, pods, target: target["spec"].update(nodeName="node-0"),
    lambda job, pods, target: target["spec"].update(schedulerName="default-scheduler"),
    lambda job, pods, target: target["spec"].update(priority=2000000000),
    lambda job, pods, target: target["spec"].update(unregistered_mutation=True),
])
def test_identity_and_resource_changes_fail_closed(mutate):
    config, job, pods, target = _inputs()
    _succeed(job, pods)
    mutate(job, pods, target)
    with pytest.raises(ValueError):
        handoff.select_node(config, job, pods, target)


def test_completed_index_ranges_are_finite():
    assert handoff.completed_indexes("0-2,3", 4) == {0, 1, 2, 3}
    for value in ("-1", "2-1", "0-4", "0-9999999999", "x", "1,"):
        with pytest.raises(ValueError):
            handoff.completed_indexes(value, 4)


def test_per_index_zero_retry_is_supported_without_ignoring_failures():
    config, job, pods, target = _inputs()
    _succeed(job, pods)
    job["spec"].update(backoffLimit=2147483647, backoffLimitPerIndex=0)
    assert handoff.select_node(config, job, pods, target) == "node-2"
    job["spec"]["podFailurePolicy"] = {"rules": [{"action": "Ignore"}]}
    with pytest.raises(ValueError, match="no-retry"):
        handoff.select_node(config, job, pods, target)


class FakeKubernetes:
    def __init__(self, job, pods, target, *, lost_ack=False, refuse=False):
        self.resources = {
            handoff.api_path("jobs", job["metadata"]["name"]): copy.deepcopy(job),
            **{handoff.api_path("pods", pod["metadata"]["name"]): copy.deepcopy(pod) for pod in pods},
            handoff.api_path("pods", target["metadata"]["name"]): copy.deepcopy(target),
        }
        self.calls = []
        self.lost_ack = lost_ack
        self.refuse = refuse

    def request(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if method == "GET":
            return copy.deepcopy(self.resources[path])
        assert method == "POST" and path.endswith("/binding")
        if self.refuse:
            raise URLError("binding denied")
        self.resources[path.removesuffix("/binding")]["spec"]["nodeName"] = payload["target"]["name"]
        if self.lost_ack:
            raise URLError("acknowledgement lost")
        return {}


@pytest.mark.parametrize("lost_ack", [False, True])
def test_one_binding_only_and_durable_receipts(tmp_path, lost_ack):
    config, job, pods, target = _inputs()
    _succeed(job, pods)
    api = FakeKubernetes(job, pods, target, lost_ack=lost_ack)
    handoff.run(config, tmp_path, api)
    posts = [call for call in api.calls if call[0] == "POST"]
    assert len(posts) == 1
    assert posts[0][1] == handoff.api_path("pods", "successor-pod") + "/binding"
    assert posts[0][2]["target"]["name"] == "node-2"
    assert posts[0][2]["metadata"]["uid"] == "successor-pod-uid"
    assert (tmp_path / "binding-intent.json").is_file()
    assert json.loads((tmp_path / "result.json").read_text())["release_permitted"] is False
    assert all(call[0] in {"GET", "POST"} for call in api.calls)


def test_binding_denial_is_not_success(tmp_path):
    config, job, pods, target = _inputs()
    _succeed(job, pods)
    api = FakeKubernetes(job, pods, target, refuse=True)
    with pytest.raises(URLError):
        handoff.run(config, tmp_path, api)
    assert not (tmp_path / "result.json").exists()
