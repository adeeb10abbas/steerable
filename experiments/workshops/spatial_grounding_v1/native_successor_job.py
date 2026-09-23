"""Render and validate a bounded, non-policy one-GPU SGW native successor Job."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

NAMESPACE = "211247-prod"
BT_JOB = "sgw01-ali-family-partition-20260923bt"
IMAGE = "artifactory-ci.gm.com/docker-approved/devcontainers/base@sha256:03f5ce7d090fbd378070a8216d0aedfc6e473c52da99b40b0cf53918612a297c"
PVC = "211247-prod-pvc"
ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
SCHEMA = "sgw-01-native-successor-job-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is required")
    return value


def _config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ValueError("successor config schema differs")
    return value


def render(config_path: Path) -> dict[str, Any]:
    """Render a single finite Job from a hash-bound source/config/native command."""
    config = _config(config_path)
    name = _required_text(config.get("job_name"), "job_name")
    source = Path(_required_text(config.get("source_path"), "source_path"))
    source_receipt = Path(_required_text(config.get("source_receipt"), "source_receipt"))
    output = _required_text(config.get("output_root"), "output_root")
    command = config.get("native_command")
    nodes = config.get("bt_node_names")
    if (
        not source.is_dir() or not isinstance(command, list) or not command
        or not all(isinstance(item, str) and item for item in command)
        or not source_receipt.is_file() or not isinstance(config.get("source_receipt_sha256"), str)
        or _sha256(source_receipt) != config["source_receipt_sha256"]
        or not isinstance(nodes, list) or not nodes or not all(isinstance(node, str) and node for node in nodes)
        or len(set(nodes)) != len(nodes)
    ):
        raise ValueError("successor config lacks immutable source, command, or exact bt node allowlist")
    source_commit = _required_text(config.get("source_commit"), "source_commit")
    assets = _required_text(config.get("assets_manifest"), "assets_manifest")
    preflight = _required_text(config.get("preflight_root"), "preflight_root")
    deadline = config.get("active_deadline_seconds")
    if type(deadline) is not int or not 1 <= deadline <= 172800:
        raise ValueError("successor active deadline must be finite and at most 48 hours")
    script = """set -euo pipefail
OUT="${SGW_OUTPUT_ROOT}"
mkdir "$OUT"
exec > >(tee "$OUT/job.log") 2>&1
python3 - "$OUT" <<'PY'
from datetime import datetime, timezone
import json, os, shutil, sys
from pathlib import Path
out = Path(sys.argv[1]); free = shutil.disk_usage(out).free; required = 100 * 1024**3
receipt = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "free_bytes": free,
           "required_free_bytes": required, "model_requests": 0, "behavioral_episodes": 0,
           "status": "passed_conservative_space_check" if free >= required else "insufficient_space"}
with (out / "storage_preflight.json").open("x") as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True); stream.write("\\n"); stream.flush(); os.fsync(stream.fileno())
if free < required: raise RuntimeError("insufficient persistent storage")
PY
test "$(git -C "$SGW_SOURCE" rev-parse HEAD)" = "$SGW_SOURCE_COMMIT"
test -z "$(git -C "$SGW_SOURCE" status --porcelain)"
test "$(git -C "$ROBOLAB" rev-parse HEAD)" = "$ROBOLAB_COMMIT"
test -z "$(git -C "$ROBOLAB" status --porcelain --untracked-files=no)"
cd "$SGW_SOURCE"
"$PY" tools/validate_vla_wam_v3_protocol.py --quiet
"$PY" tools/validate_vla_wam_v2_protocol.py > "$OUT/v2_validation.json"
export PYTHONPATH="$SGW_SOURCE:$ROBOLAB"
CUDA_VISIBLE_DEVICES=$("$PY" -m experiments.workshops.spatial_grounding_v1.gpu_idle_probe --expected-count 1 --output "$OUT/gpu_selection.json")
export CUDA_VISIBLE_DEVICES
mkdir -p /data/users/ali/sgw-01/locks
exec 9>"/data/users/ali/sgw-01/locks/gpu-$CUDA_VISIBLE_DEVICES.lock"
flock --nonblock 9
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$OUT/gpu_identity.csv"
"$PY" - "$OUT" "$SGW_NATIVE_COMMAND_JSON" <<'PY'
from datetime import datetime, timezone
import json, os, subprocess, sys, traceback
from pathlib import Path
out, command = Path(sys.argv[1]), json.loads(sys.argv[2])
receipt = {"command": command, "model_requests": 0, "behavioral_episodes": 0, "release_permitted": False}
try:
    result = subprocess.run(command, cwd=os.environ["SGW_SOURCE"], check=False)
    receipt["native_exit_code"] = result.returncode
    if result.returncode: raise RuntimeError(f"native command exited {result.returncode}")
    receipt["status"] = "native_exit_zero_requires_external_evidence_verification"
except BaseException:
    receipt["status"] = "infrastructure_invalid_native_attempt"; receipt["traceback"] = traceback.format_exc(); raise
finally:
    receipt["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    with (out / "process_outcome.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True); stream.write("\\n"); stream.flush(); os.fsync(stream.fileno())
PY
sync"""
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {
            "name": name, "namespace": NAMESPACE, "labels": {"owner": "ali", "app.kubernetes.io/name": "sgw-01"},
            "annotations": {
                "sgw-01/config-sha256": _sha256(config_path),
                "sgw-01/source-receipt-sha256": config["source_receipt_sha256"],
                "sgw-01/robolab-commit": ROBOLAB_COMMIT,
            },
        },
        "spec": {
            "parallelism": 1, "completions": 1, "backoffLimit": 0, "activeDeadlineSeconds": deadline,
            "template": {
                "metadata": {"labels": {"owner": "ali", "app.kubernetes.io/name": "sgw-01",
                                         "purpose": "zero-model-native-engineering-successor"}},
                "spec": {
                    "restartPolicy": "Never", "preemptionPolicy": "Never", "automountServiceAccountToken": False,
                    "activeDeadlineSeconds": deadline, "terminationGracePeriodSeconds": 180,
                    "affinity": {
                        "nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [{
                            "matchFields": [{"key": "metadata.name", "operator": "In", "values": nodes}]
                        }]}},
                        "podAntiAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": [{
                            "labelSelector": {"matchExpressions": [{
                                "key": "batch.kubernetes.io/job-name", "operator": "In", "values": [BT_JOB],
                            }]},
                            "namespaces": [NAMESPACE], "topologyKey": "kubernetes.io/hostname",
                        }]},
                    },
                    "imagePullSecrets": [{"name": "artifactory-ci-pull-secret"}],
                    "containers": [{
                        "name": "native-engineering", "image": IMAGE, "command": ["/bin/bash", "-ec"], "args": [script],
                        "env": [
                            {"name": "SGW_OUTPUT_ROOT", "value": output}, {"name": "SGW_SOURCE", "value": str(source)},
                            {"name": "SGW_SOURCE_COMMIT", "value": source_commit}, {"name": "ROBOLAB", "value": _required_text(config.get("robolab_root"), "robolab_root")},
                            {"name": "ROBOLAB_COMMIT", "value": ROBOLAB_COMMIT}, {"name": "PY", "value": _required_text(config.get("python"), "python")},
                            {"name": "SGW_ASSETS_MANIFEST", "value": assets}, {"name": "SGW_PREFLIGHT_ROOT", "value": preflight},
                            {"name": "SGW_NATIVE_COMMAND_JSON", "value": json.dumps(command)}, {"name": "HOME", "value": "/home/ali"},
                            {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"}, {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility,graphics,display"},
                            {"name": "OMNI_KIT_ACCEPT_EULA", "value": "YES"}, {"name": "ACCEPT_EULA", "value": "Y"},
                        ],
                        "resources": {"requests": {"cpu": "16", "memory": "64Gi", "nvidia.com/gpu": "1"},
                                      "limits": {"cpu": "32", "memory": "128Gi", "nvidia.com/gpu": "1"}},
                        "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]},
                                            "runAsNonRoot": True, "runAsUser": 816149040, "runAsGroup": 2518800},
                        "volumeMounts": [{"name": "workspace", "mountPath": "/data"}, {"name": "dshm", "mountPath": "/dev/shm"}],
                    }],
                    "volumes": [{"name": "workspace", "persistentVolumeClaim": {"claimName": PVC}},
                                {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}}],
                },
            },
        },
    }


def validate(job: Mapping[str, Any], *, nodes: list[str]) -> None:
    """Reject every weakening of the successor scheduling/persistence contract."""
    try:
        spec, pod = job["spec"], job["spec"]["template"]["spec"]
        container = pod["containers"][0]
        node_expression = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchFields"][0]
        anti = pod["affinity"]["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("successor Job lacks required scheduling structure") from error
    if (
        job.get("kind") != "Job" or job.get("metadata", {}).get("namespace") != NAMESPACE
        or spec.get("parallelism") != 1 or spec.get("completions") != 1 or spec.get("backoffLimit") != 0
        or pod.get("restartPolicy") != "Never" or pod.get("preemptionPolicy") != "Never"
        or pod.get("automountServiceAccountToken") is not False
        or node_expression != {"key": "metadata.name", "operator": "In", "values": nodes}
        or anti.get("namespaces") != [NAMESPACE] or anti.get("topologyKey") != "kubernetes.io/hostname"
        or anti.get("labelSelector", {}).get("matchExpressions") != [{
            "key": "batch.kubernetes.io/job-name", "operator": "In", "values": [BT_JOB],
        }]
        or container.get("image") != IMAGE or container.get("resources", {}).get("limits", {}).get("nvidia.com/gpu") != "1"
        or container.get("resources", {}).get("requests", {}).get("nvidia.com/gpu") != "1"
        or "gpu_idle_probe" not in container.get("args", [""])[0] or "flock --nonblock" not in container.get("args", [""])[0]
        or not isinstance(job.get("metadata", {}).get("annotations", {}).get("sgw-01/config-sha256"), str)
    ):
        raise ValueError("successor Job constraints are weakened")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    job = render(args.config)
    validate(job, nodes=_config(args.config)["bt_node_names"])
    args.output.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
