import hashlib
import json
from pathlib import Path

import pytest

from experiments.workshops.spatial_grounding_v1 import native_successor_job as successor


def _config(tmp_path: Path) -> Path:
    source = tmp_path / "source"; source.mkdir()
    receipt = tmp_path / "source-receipt.json"; receipt.write_text('{"source":"pinned"}')
    registration = tmp_path / "registration.json"; registration.write_text('{"registration":"pinned"}')
    python = "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python"
    config = {
        "schema_version": successor.SCHEMA, "job_name": "sgw01-ali-native-successor-example",
        "source_path": str(source), "source_commit": "e9ffc35635e96b5f66ee906f4c84062fb36002b1",
        "source_receipt": str(receipt), "source_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "registration": str(registration), "registration_sha256": hashlib.sha256(registration.read_bytes()).hexdigest(),
        "output_root": "/data/users/ali/sgw-01/qualification/native-successor-example",
        "robolab_root": "/data/users/ali/vla_wam/external/RoboLab-pi05-v3-0aef241",
        "python": python,
        "assets_manifest": "/data/users/ali/sgw-01/preflight/a40-20260922e/assets.json",
        "preflight_root": "/data/users/ali/sgw-01/preflight/a40-20260922e",
        "native_command": [python, "-m", "experiments.workshops.spatial_grounding_v1.paper_engineering",
                           "--registration", str(registration),
                           "--output-root", "/data/users/ali/sgw-01/qualification/native-successor-example/evidence"],
        "bt_node_names": ["bt-node-a", "bt-node-b"], "active_deadline_seconds": 7200,
        "native_child_timeout_seconds": 4800, "storage_reserve_bytes": successor.MIN_STORAGE_RESERVE_BYTES,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def _render(tmp_path: Path):
    config = _config(tmp_path)
    return config, successor.render(config)


def test_successor_requires_exact_nodes_and_bt_anti_affinity(tmp_path):
    config, job = _render(tmp_path)
    successor.validate(job, nodes=["bt-node-a", "bt-node-b"])
    pod = job["spec"]["template"]["spec"]
    assert job["spec"]["parallelism"] == job["spec"]["completions"] == 1
    assert job["spec"]["backoffLimit"] == 0 and pod["preemptionPolicy"] == "Never"
    assert len(job["metadata"]["annotations"]["sgw-01/config-sha256"]) == 64
    terms = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    assert [term["matchFields"] for term in terms] == [
        [{"key": "metadata.name", "operator": "In", "values": ["bt-node-a"]}],
        [{"key": "metadata.name", "operator": "In", "values": ["bt-node-b"]}],
    ]
    assert pod["affinity"]["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][0]["topologyKey"] == "kubernetes.io/hostname"
    assert pod["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert pod["nodeSelector"] == {"node-role.kubernetes.io/worker-gpu": ""}
    assert pod["containers"][0]["env"][-1]["name"] == "LD_LIBRARY_PATH"
    assert '"$PY" - "$OUT"' in pod["containers"][0]["args"][0]
    assert "python3 -" not in pod["containers"][0]["args"][0]


@pytest.mark.parametrize("mutate", [
    lambda job: job["spec"].update({"parallelism": 2}),
    lambda job: job["spec"]["template"]["spec"].update({"preemptionPolicy": "PreemptLowerPriority"}),
    lambda job: job["spec"]["template"]["spec"]["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchFields"][0].update({"values": []}),
    lambda job: job["spec"]["template"]["spec"]["affinity"]["podAntiAffinity"].pop("requiredDuringSchedulingIgnoredDuringExecution"),
    lambda job: job["spec"]["template"]["spec"]["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"].append({}),
    lambda job: job["spec"].update({"activeDeadlineSeconds": 1}),
])
def test_weakened_successor_constraints_are_rejected(tmp_path, mutate):
    _config, job = _render(tmp_path)
    mutate(job)
    with pytest.raises(ValueError, match="weakened|required scheduling"):
        successor.validate(job, nodes=["bt-node-a", "bt-node-b"])


def test_absent_nodes_or_unpinned_config_is_rejected(tmp_path):
    config = _config(tmp_path)
    value = json.loads(config.read_text())
    value["bt_node_names"] = []
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="allowlist"):
        successor.render(config)


def test_command_cannot_use_a_different_registration_than_the_hash_guard(tmp_path):
    path = _config(tmp_path)
    value = json.loads(path.read_text())
    value["native_command"][4] = "/unbound-registration.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="command differs"):
        successor.render(path)
