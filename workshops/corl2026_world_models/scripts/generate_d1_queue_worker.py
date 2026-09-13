#!/usr/bin/env python3
"""Generate the task-owned two-B200 queue worker and its stable Service.

The parent bootstrap deliberately created one-GPU workers.  Core configuration
D1 uses the unchanged official two-rank DreamZero path, so its queue worker must
expose exactly two B200 devices.  This generator preserves the frozen queue
bootstrap and admission cutoff while changing only the owned worker identity,
role and resource request.  It generates objects; it never talks to Kubernetes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


K8S_NAMESPACE = "211247-prod"
STUDY = "wmf_ablation_001_20260912"
NAME = "wmf-forecast-0912-worker-d1-00"
ROLE = "d1"
SOURCE = f"/data/users/ali/vla_wam/src/{STUDY}"
STATE = f"/data/users/ali/vla_wam/raw/{STUDY}/control"
BOOTSTRAP = f"{STATE}/bootstrap/cluster_queue.py"
IMAGE = (
    "artifactory-ci.gm.com/docker-approved/devcontainers/base@"
    "sha256:03f5ce7d090fbd378070a8216d0aedfc6e473c52da99b40b0cf53918612a297c"
)
WORKER_WALL_SECONDS = 604800
HASH_CHECK = '''import hashlib, os, sys
path, expected = sys.argv[1:3]
with open(path, "rb") as stream:
    actual = hashlib.sha256(stream.read()).hexdigest()
if actual != expected:
    raise SystemExit("bootstrap digest mismatch; no queue code executed")
os.execv(sys.executable, [sys.executable, path, *sys.argv[3:]])
'''


def build_manifest(bootstrap_sha256: str, admission_deadline_unix: int) -> dict:
    if not isinstance(bootstrap_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", bootstrap_sha256):
        raise ValueError("a verified lowercase SHA-256 for the frozen bootstrap is required")
    if type(admission_deadline_unix) is not int or admission_deadline_unix <= 0:
        raise ValueError("admission_deadline_unix must be a positive integer")

    labels = {
        "app.kubernetes.io/name": "wmf-forecast-queue",
        "app.kubernetes.io/part-of": STUDY,
        "user": "ali",
        "wmf-role": ROLE,
    }
    environment = {
        "USER": "ali",
        "LOGNAME": "ali",
        "HOME": "/home/ali",
        "XDG_CACHE_HOME": "/home/ali/.cache",
        "TMPDIR": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics,display,video",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "VK_DRIVER_FILES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "LD_LIBRARY_PATH": "/data/users/jsalfity/glvnd/lib",
    }
    args = [
        BOOTSTRAP,
        bootstrap_sha256,
        "worker",
        "--repo",
        SOURCE,
        "--state-dir",
        STATE,
        "--max-wall-seconds",
        str(WORKER_WALL_SECONDS),
        "--poll-seconds",
        "30",
        "--admission-deadline-unix",
        str(admission_deadline_unix),
        "--worker-id",
        NAME,
        "--role",
        ROLE,
    ]
    pod = {
        "restartPolicy": "OnFailure",
        "terminationGracePeriodSeconds": 120,
        "automountServiceAccountToken": False,
        "securityContext": {
            "fsGroup": 2518800,
            "supplementalGroups": [2518800],
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "imagePullSecrets": [{"name": "artifactory-ci-pull-secret"}],
        "nodeSelector": {
            "node-role.kubernetes.io/worker-gpu": "",
            "nvidia.com/gpu.product": "NVIDIA-B200",
        },
        "tolerations": [
            {
                "effect": "NoSchedule",
                "key": "nvidia.com/gpu",
                "operator": "Equal",
                "value": "present",
            }
        ],
        "containers": [
            {
                "name": "worker",
                "image": IMAGE,
                "imagePullPolicy": "IfNotPresent",
                "command": ["/usr/bin/python3", "-c", HASH_CHECK],
                "args": args,
                "workingDir": "/home/ali",
                "ports": [{"name": "d1-policy", "containerPort": 18021, "protocol": "TCP"}],
                "resources": {
                    "requests": {"cpu": "48", "memory": "256Gi", "nvidia.com/gpu": 2},
                    "limits": {"cpu": "80", "memory": "512Gi", "nvidia.com/gpu": 2},
                },
                "env": [{"name": key, "value": value} for key, value in environment.items()],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "readOnlyRootFilesystem": False,
                    "runAsGroup": 2518800,
                    "runAsUser": 816149040,
                    "runAsNonRoot": True,
                },
                "volumeMounts": [
                    {"name": "workspace", "mountPath": "/data"},
                    {"name": "dshm", "mountPath": "/dev/shm"},
                    {"name": "tmp", "mountPath": "/tmp"},
                    {"name": "vartmp", "mountPath": "/var/tmp"},
                    {"name": "home", "mountPath": "/home/ali"},
                    {"name": "userdb", "mountPath": "/etc/passwd", "subPath": "passwd", "readOnly": True},
                    {"name": "userdb", "mountPath": "/etc/group", "subPath": "group", "readOnly": True},
                ],
            }
        ],
        "volumes": [
            {"name": "workspace", "persistentVolumeClaim": {"claimName": "211247-prod-pvc"}},
            {"name": "dshm", "emptyDir": {"medium": "Memory"}},
            {"name": "tmp", "emptyDir": {}},
            {"name": "vartmp", "emptyDir": {}},
            {"name": "home", "emptyDir": {}},
            {"name": "userdb", "configMap": {"name": "211247-ali-b200-1gpu-userdb"}},
        ],
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": NAME,
            "namespace": K8S_NAMESPACE,
            "labels": labels,
            "annotations": {"wmf-bootstrap-sha256": bootstrap_sha256},
        },
        "spec": {
            "completions": 1,
            "parallelism": 1,
            "backoffLimit": 3,
            "activeDeadlineSeconds": WORKER_WALL_SECONDS + 172800 + 300,
            "template": {
                "metadata": {
                    "labels": labels,
                    "annotations": {"wmf-bootstrap-sha256": bootstrap_sha256},
                },
                "spec": pod,
            },
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": "wmf-forecast-0912-d1", "namespace": K8S_NAMESPACE, "labels": labels},
        "spec": {
            "type": "ClusterIP",
            "selector": {"batch.kubernetes.io/job-name": NAME},
            "ports": [{"name": "d1-policy", "port": 18021, "targetPort": "d1-policy", "protocol": "TCP"}],
        },
    }
    return {"apiVersion": "v1", "kind": "List", "items": [job, service]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-sha256", required=True)
    parser.add_argument("--admission-deadline-unix", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = build_manifest(args.bootstrap_sha256, args.admission_deadline_unix)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text() != payload:
        raise FileExistsError("refusing to overwrite a different D1 worker manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        args.output.write_text(payload)
    print(json.dumps({"output": str(args.output), "workers": 1, "gpus_per_worker": 2, "applied": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
