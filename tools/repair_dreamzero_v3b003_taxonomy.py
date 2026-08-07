#!/usr/bin/env python3
"""Repair one proven V3-B003 packaging-only taxonomy mismatch.

The simulator bridge labeled the registered seed9411 mirrored-RIGHT failure
from *any* requested-cone entry.  The frozen V3 precedence instead uses the
final sustained requested/opposite state.  The retained samples prove that
this episode picked the object and entered the requested cone transiently, but
did not finish in either sustained cone; its frozen label is therefore
``transport_failed``.  This tool never mutates the original JSONL.  It writes a
hash-linked packaging copy plus an explicit before/after audit report, performs
zero model requests, and executes zero actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.vla_wam_v3_episode_schema import (
    derive_failure_taxonomy,
    parse_jsonl_record,
)


CELL_ID = "v3b003:dreamzero:seed9411:position_mirrored:right"
SOURCE_SHA256 = "172a474cfb545c71051529fcc64c4d6613b6da3616fe891336aa6c7ffbd094cc"
SOURCE_BYTES = 124855
CELL_MANIFEST_SCHEMA = "vla-wam-shared-v3b-dreamzero-cell-jsonl-manifest-v1"
REPORT_SCHEMA = "vla-wam-shared-v3b-dreamzero-taxonomy-packaging-repair-v1"


class RepairError(RuntimeError):
    """Raised if the retained evidence differs from the proven repair case."""


def _fail(message: str) -> None:
    raise RepairError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def repair(source: Path, output_dir: Path) -> dict[str, Any]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    payload = source.read_bytes()
    if len(payload) != SOURCE_BYTES or _sha256(payload) != SOURCE_SHA256:
        _fail("source JSONL differs from the exact retained repair case")
    lines = payload.splitlines()
    if len(lines) != 1:
        _fail("source JSONL must contain exactly one row")
    record = json.loads(lines[0], object_pairs_hook=_unique_object)
    if record.get("registered_cell_id") != CELL_ID:
        _fail("source JSONL names another cell")
    source_manifest_path = source.with_name(source.name + ".manifest.json")
    source_manifest_bytes = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_manifest_bytes, object_pairs_hook=_unique_object)
    expected_manifest = {
        "schema_version": CELL_MANIFEST_SCHEMA,
        "registered_cell_id": CELL_ID,
        "row_count": 1,
        "jsonl_sha256": SOURCE_SHA256,
        "jsonl_bytes": SOURCE_BYTES,
    }
    for key, value in expected_manifest.items():
        if source_manifest.get(key) != value:
            _fail(f"source manifest mismatch for {key}")

    measurements = record.get("measurements")
    if not isinstance(measurements, dict):
        _fail("source record lacks derived measurements")
    facts = {
        "requested_success": record.get("requested_success"),
        "failure_taxonomy": record.get("failure_taxonomy"),
        "verified_pickup": measurements.get("verified_pickup"),
        "first_sustained_requested_entry_step": measurements.get(
            "first_sustained_requested_entry_step"
        ),
        "final_requested_region_sustained": measurements.get(
            "final_requested_region_sustained"
        ),
        "final_opposite_region_sustained": measurements.get(
            "final_opposite_region_sustained"
        ),
        "final_detached_release": measurements.get("final_detached_release"),
    }
    if facts != {
        "requested_success": False,
        "failure_taxonomy": "release_failed",
        "verified_pickup": True,
        "first_sustained_requested_entry_step": 152,
        "final_requested_region_sustained": False,
        "final_opposite_region_sustained": False,
        "final_detached_release": False,
    }:
        _fail(f"retained repair facts changed: {facts}")
    expected_taxonomy = derive_failure_taxonomy(record, measurements)
    if expected_taxonomy != "transport_failed":
        _fail(f"frozen precedence no longer derives transport_failed: {expected_taxonomy}")

    repaired = dict(record)
    repaired["failure_taxonomy"] = expected_taxonomy
    repaired_payload = _canonical(repaired)
    # Complete schema validation proves that taxonomy is the only required
    # semantic packaging change.  It also re-derives every continuous measure
    # from the retained raw step stream.
    normalized = parse_jsonl_record(repaired_payload.decode("utf-8").strip())
    if normalized.get("failure_taxonomy") != expected_taxonomy:
        _fail("corrected record did not survive frozen schema validation")

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    output_jsonl = output_dir / "raw_episode.jsonl"
    output_manifest = output_dir / "raw_episode.jsonl.manifest.json"
    report_path = output_dir / "repair_report.json"
    output_jsonl.write_bytes(repaired_payload)
    repaired_manifest = {
        "schema_version": CELL_MANIFEST_SCHEMA,
        "registered_cell_id": CELL_ID,
        "row_count": 1,
        "jsonl_sha256": _sha256(repaired_payload),
        "jsonl_bytes": len(repaired_payload),
    }
    output_manifest.write_bytes(json.dumps(repaired_manifest, indent=2, sort_keys=True).encode() + b"\n")
    report = {
        "schema_version": REPORT_SCHEMA,
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-B003",
        "registered_cell_id": CELL_ID,
        "repair_kind": "deterministic_retained_record_packaging_only",
        "behavioral_episode_rerun": False,
        "model_inference_requests": 0,
        "actions_executed": 0,
        "original_preserved": True,
        "reason": (
            "The bridge used any requested-cone entry for release_failed; frozen V3 precedence "
            "requires final sustained requested-cone occupancy. The raw stream proves transient "
            "entry, verified pickup, and neither final sustained requested nor opposite occupancy."
        ),
        "before": {
            "failure_taxonomy": "release_failed",
            "jsonl": {"path": str(source), "sha256": SOURCE_SHA256, "bytes": SOURCE_BYTES},
            "manifest": {
                "path": str(source_manifest_path),
                "sha256": _sha256(source_manifest_bytes),
                "bytes": len(source_manifest_bytes),
            },
        },
        "after": {
            "failure_taxonomy": expected_taxonomy,
            "jsonl": {
                "path": str(output_jsonl),
                "sha256": repaired_manifest["jsonl_sha256"],
                "bytes": repaired_manifest["jsonl_bytes"],
            },
            "manifest": {
                "path": str(output_manifest),
                "sha256": _sha256(output_manifest.read_bytes()),
                "bytes": output_manifest.stat().st_size,
            },
        },
        "frozen_derivation_facts": facts,
    }
    report_path.write_bytes(json.dumps(report, indent=2, sort_keys=True).encode() + b"\n")
    report_manifest = {
        "schema_version": "vla-wam-shared-v3b-dreamzero-taxonomy-packaging-repair-manifest-v1",
        "path": str(report_path),
        "sha256": _sha256(report_path.read_bytes()),
        "bytes": report_path.stat().st_size,
    }
    (output_dir / "repair_report.manifest.json").write_bytes(
        json.dumps(report_manifest, indent=2, sort_keys=True).encode() + b"\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = repair(args.source, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
