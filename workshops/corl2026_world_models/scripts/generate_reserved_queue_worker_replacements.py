#!/usr/bin/env python3
"""Generate Pod-UID-bound replacements for the reserved worker 00 and D1 Jobs.

The reviewed source manifests are immutable deployment evidence.  This tool
verifies their exact byte identities, selects only the two explicitly named
Jobs, and adds the Kubernetes downward-API Pod UID binding used by the worker
diagnostic.  It does not include or alter the D1 Service and never contacts
Kubernetes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


WORKSHOP = Path(__file__).resolve().parents[1]
AUTONOMY = WORKSHOP / "execution/20260912/autonomy"
GENERIC_SOURCE = AUTONOMY / "science_queue_workers_v3.json"
D1_SOURCE = AUTONOMY / "d1_queue_worker_v2.json"
SOURCE_SHA256 = {
    GENERIC_SOURCE.name: "9a8e687b547a2fb97047110fe83cefa373dbb4d9c8660cb2b74ffb1545bac94f",
    D1_SOURCE.name: "23f1299f58761676742bf639f885051409209b9ca42575f758df641cb3c1d093",
}
TARGETS = (
    (GENERIC_SOURCE, "wmf-forecast-0912-worker-00"),
    (D1_SOURCE, "wmf-forecast-0912-worker-d1-00"),
)
POD_UID_ANNOTATION = "wmf-pod-uid-source"
POD_UID_ANNOTATION_VALUE = "downward-api-metadata.uid"
POD_UID_ENV = {
    "name": "POD_UID",
    "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
}


def _load_verified_source(path: Path) -> dict:
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    expected_sha256 = SOURCE_SHA256[path.name]
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"reviewed source manifest digest mismatch for {path.name}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    value = json.loads(payload)
    if value.get("apiVersion") != "v1" or value.get("kind") != "List":
        raise RuntimeError(f"reviewed source is not a Kubernetes List: {path.name}")
    return value


def _select_exact_job(source: dict, source_name: str, target_name: str) -> dict:
    matches = [
        item
        for item in source.get("items", [])
        if item.get("kind") == "Job"
        and item.get("metadata", {}).get("name") == target_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"reviewed source {source_name} must contain exactly one Job named "
            f"{target_name}; found {len(matches)}"
        )
    return matches[0]


def _bind_pod_uid(source_job: dict) -> dict:
    item = copy.deepcopy(source_job)
    name = item["metadata"]["name"]
    container = item["spec"]["template"]["spec"]["containers"][0]
    environment = container["env"]
    if any(row.get("name") == "POD_UID" for row in environment):
        raise RuntimeError(f"reviewed source Job already binds POD_UID: {name}")

    job_annotations = item["metadata"].setdefault("annotations", {})
    pod_annotations = item["spec"]["template"]["metadata"].setdefault(
        "annotations", {}
    )
    for annotations in (job_annotations, pod_annotations):
        if POD_UID_ANNOTATION in annotations:
            raise RuntimeError(
                f"reviewed source Job already declares {POD_UID_ANNOTATION}: {name}"
            )
        annotations[POD_UID_ANNOTATION] = POD_UID_ANNOTATION_VALUE
    environment.append(copy.deepcopy(POD_UID_ENV))
    return item


def build_manifest() -> dict:
    items = []
    for source_path, target_name in TARGETS:
        source = _load_verified_source(source_path)
        source_job = _select_exact_job(source, source_path.name, target_name)
        items.append(_bind_pod_uid(source_job))
    if [item["metadata"]["name"] for item in items] != [
        target_name for _, target_name in TARGETS
    ]:
        raise RuntimeError("replacement selection is not the exact reserved-worker set")
    return {"apiVersion": "v1", "kind": "List", "items": items}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = build_manifest()
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text() != payload:
        raise FileExistsError(
            "refusing to overwrite a different reserved-worker replacement manifest"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        args.output.write_text(payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "worker_names": [item["metadata"]["name"] for item in value["items"]],
                "pod_uid_source": POD_UID_ANNOTATION_VALUE,
                "applied": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
