#!/usr/bin/env python3
"""Generate corrected one-B200 science workers from the reviewed bootstrap.

The initial manifest explicitly set ``NVIDIA_VISIBLE_DEVICES=all``.  That was
benign on the first fully partitioned node but exposed every node GPU when later
workers landed on newly available nodes.  The infrastructure diagnostic caught
the mismatch and failed closed.  Kubernetes' device plugin already injects the
allocated UUID; these replacement workers therefore omit the override while
retaining the frozen queue controller, security context, PVC and cutoff.

Indices 02 and 03 are deliberately omitted: their two diagnosed one-GPU slots
were converted into the exact two-B200 D1 worker.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "wmf_parent_bootstrap", SCRIPTS / "generate_cluster_bootstrap.py"
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load reviewed cluster bootstrap generator")
_BOOTSTRAP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BOOTSTRAP)

REPLACEMENT_INDICES = tuple(index for index in range(32) if index not in {2, 3})


def build_manifest(bootstrap_sha256: str, admission_deadline_unix: int) -> dict:
    original = _BOOTSTRAP.build_manifest(
        bootstrap_sha256,
        worker_count=32,
        admission_deadline_unix=admission_deadline_unix,
    )
    wanted = {f"wmf-forecast-0912-worker-{index:02d}" for index in REPLACEMENT_INDICES}
    workers = []
    for item in original["items"]:
        if item.get("kind") != "Job" or item.get("metadata", {}).get("name") not in wanted:
            continue
        container = item["spec"]["template"]["spec"]["containers"][0]
        container["env"] = [row for row in container["env"] if row["name"] != "NVIDIA_VISIBLE_DEVICES"]
        annotations = item["metadata"].setdefault("annotations", {})
        annotations["wmf-device-visibility"] = "kubernetes-device-plugin-injected"
        template_annotations = item["spec"]["template"]["metadata"].setdefault("annotations", {})
        template_annotations["wmf-device-visibility"] = "kubernetes-device-plugin-injected"
        workers.append(item)
    workers.sort(key=lambda item: item["metadata"]["name"])
    if [item["metadata"]["name"] for item in workers] != sorted(wanted):
        raise RuntimeError("reviewed bootstrap did not yield the exact replacement pool")
    return {"apiVersion": "v1", "kind": "List", "items": workers}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-sha256", required=True)
    parser.add_argument("--admission-deadline-unix", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = build_manifest(args.bootstrap_sha256, args.admission_deadline_unix)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text() != payload:
        raise FileExistsError("refusing to overwrite a different corrected worker manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        args.output.write_text(payload)
    print(json.dumps({
        "output": str(args.output),
        "one_gpu_workers": len(value["items"]),
        "converted_indices": [2, 3],
        "applied": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
