#!/usr/bin/env python3
"""Generate explicit generic-worker replacements with Pod UID provenance.

The applied ``science_queue_workers_v3.json`` is immutable historical evidence
and remains reproducible from its original generator.  This generator derives
only explicitly selected, still-generic one-B200 Jobs from that reviewed
manifest builder, then adds the Kubernetes downward-API binding required by
the dynamic worker diagnostic.  It never contacts or mutates Kubernetes.

Workers 00 through 04 are reserved or superseded by established allocations
and cannot be selected here.  Every invocation must name each intended worker
index so a generated manifest cannot silently replace the wider pool.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Iterable


SCRIPTS = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "wmf_science_workers_v3", SCRIPTS / "generate_science_queue_workers.py"
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load reviewed generic worker generator")
_SCIENCE_WORKERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCIENCE_WORKERS)

FIRST_TARGETABLE_INDEX = 5
LAST_TARGETABLE_INDEX = 31
POD_UID_ANNOTATION = "wmf-pod-uid-source"
POD_UID_ANNOTATION_VALUE = "downward-api-metadata.uid"
POD_UID_ENV = {
    "name": "POD_UID",
    "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
}


def _validate_worker_indices(worker_indices: Iterable[int]) -> tuple[int, ...]:
    if isinstance(worker_indices, (str, bytes)):
        raise ValueError("worker_indices must be an iterable of integer indices")
    try:
        indices = tuple(worker_indices)
    except TypeError as error:
        raise ValueError("worker_indices must be an iterable of integer indices") from error
    if not indices:
        raise ValueError("at least one explicit generic worker index is required")
    if any(type(index) is not int for index in indices):
        raise ValueError("worker indices must be integers")
    if any(not FIRST_TARGETABLE_INDEX <= index <= LAST_TARGETABLE_INDEX for index in indices):
        raise ValueError(
            f"worker indices must be between {FIRST_TARGETABLE_INDEX:02d} and "
            f"{LAST_TARGETABLE_INDEX:02d}; workers 00-04 are out of scope"
        )
    if len(indices) != len(set(indices)):
        raise ValueError("worker indices must be unique")
    return tuple(sorted(indices))


def build_manifest(
    bootstrap_sha256: str,
    admission_deadline_unix: int,
    worker_indices: Iterable[int],
) -> dict:
    indices = _validate_worker_indices(worker_indices)
    original = _SCIENCE_WORKERS.build_manifest(
        bootstrap_sha256,
        admission_deadline_unix,
    )
    wanted = {f"wmf-forecast-0912-worker-{index:02d}" for index in indices}
    workers = []
    for source_item in original["items"]:
        name = source_item.get("metadata", {}).get("name")
        if source_item.get("kind") != "Job" or name not in wanted:
            continue
        item = copy.deepcopy(source_item)
        container = item["spec"]["template"]["spec"]["containers"][0]
        environment = container["env"]
        if any(row.get("name") == "POD_UID" for row in environment):
            raise RuntimeError(f"reviewed generic worker already binds POD_UID: {name}")
        environment.append(copy.deepcopy(POD_UID_ENV))
        item["metadata"].setdefault("annotations", {})[
            POD_UID_ANNOTATION
        ] = POD_UID_ANNOTATION_VALUE
        item["spec"]["template"]["metadata"].setdefault("annotations", {})[
            POD_UID_ANNOTATION
        ] = POD_UID_ANNOTATION_VALUE
        workers.append(item)
    workers.sort(key=lambda item: item["metadata"]["name"])
    if [item["metadata"]["name"] for item in workers] != sorted(wanted):
        raise RuntimeError("reviewed generic worker generator did not yield the exact selection")
    return {"apiVersion": "v1", "kind": "List", "items": workers}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-sha256", required=True)
    parser.add_argument("--admission-deadline-unix", required=True, type=int)
    parser.add_argument(
        "--worker-index",
        required=True,
        action="append",
        type=int,
        help="Explicit generic worker index to replace; repeat for each intended Job.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = build_manifest(
        args.bootstrap_sha256,
        args.admission_deadline_unix,
        args.worker_index,
    )
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text() != payload:
        raise FileExistsError("refusing to overwrite a different worker replacement manifest")
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
