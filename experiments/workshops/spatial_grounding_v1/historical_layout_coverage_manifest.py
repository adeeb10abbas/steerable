"""Compile the 45-source historical layout coverage/read-request manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
INVENTORY = (
    REPO_ROOT
    / "artifacts/workshops/spatial_grounding_v1/infrastructure/"
    "historical-droid-layout-source-inventory-v2.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status(cohort: dict[str, Any]) -> tuple[str, str]:
    cid = cohort["cohort_id"]
    paths = cohort.get("relevant_path_evidence", [])
    if cid.startswith("V3-E006"):
        return "requires_named_hash_anchored_pvc_payloads", "committed registration/result source exists, but physical reset/layout payload is external"
    if any(item.get("expected_sha256") for item in paths):
        return "requires_named_hash_anchored_pvc_payloads", "committed source names external raw evidence with a hash anchor"
    if paths:
        return "source_inspection_needed", "source names external paths, but those paths have no hash anchor"
    return "source_inspection_needed", "source is locally inspectable, but no exact source-to-registry geometry binding is established"


def _recover_explicit_root_only(cohort: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    """Recover measured root fields without promoting them to geometry."""
    if cohort["cohort_id"] != "V3-E006":
        return []
    path = repo_root / cohort["source_path"]
    if not path.is_file():
        return []
    document = json.loads(path.read_text())
    rows = []
    for object_name, value in document.get("rigid_objects", {}).items():
        position = value.get("root_position", {})
        quaternion = value.get("root_quaternion_wxyz", {})
        if not (isinstance(position.get("values"), list) and isinstance(quaternion.get("values"), list)):
            continue
        rows.append(
            {
                "cohort_id": cohort["cohort_id"],
                "object": object_name,
                "source_path": cohort["source_path"],
                "source_sha256": cohort["source_sha256"],
                "position_json_pointer": f"/rigid_objects/{object_name}/root_position",
                "quaternion_json_pointer": f"/rigid_objects/{object_name}/root_quaternion_wxyz",
                "position": position["values"],
                "quaternion_wxyz": quaternion["values"],
                "position_data_sha256": position.get("data_sha256"),
                "quaternion_data_sha256": quaternion.get("data_sha256"),
                "semantic_status": "measured_root_only",
                "blocker": "no source-defined centroid/AABB or configured-reset pair; not comparable geometry",
            }
        )
    return rows


def compile_manifest(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    inventory_path = repo_root / INVENTORY.relative_to(REPO_ROOT)
    inventory = json.loads(inventory_path.read_text())
    records = []
    recovered_root_only = []
    indispensable_requests = []
    for cohort in inventory["cohorts"]:
        source = repo_root / cohort["source_path"]
        status, reason = _status(cohort)
        actual_sha256 = _sha256(source) if source.is_file() else None
        recovered_root_only.extend(_recover_explicit_root_only(cohort, repo_root))
        hash_status = (
            "verified_against_inventory"
            if actual_sha256 == cohort["source_sha256"]
            else "missing_or_mismatched"
        )
        record = {
            "cohort_id": cohort["cohort_id"],
            "source_path": cohort["source_path"],
            "source_sha256": cohort["source_sha256"],
            "coverage_status": status,
            "reason": reason,
            "arena": cohort["arena"],
            "known_source_file": source.is_file(),
            "source_hash_status": hash_status,
            "actual_source_sha256": actual_sha256,
            "required_pvc_payloads": [
                {
                    "path": item["path"],
                    "source_json_pointer": item["source_json_pointer"],
                    "expected_sha256": item["expected_sha256"],
                    "hash_binding_status": (
                        "hash_anchored"
                        if item["expected_sha256"]
                        else "unanchored_path_only"
                    ),
                }
                for item in cohort.get("relevant_path_evidence", [])
            ],
            "explicit_root_pose_count": len(cohort.get("explicit_root_poses", [])),
            "comparable_geometry_eligible": False,
        }
        records.append(record)
        candidates = []
        for payload in record["required_pvc_payloads"]:
            if payload["hash_binding_status"] != "hash_anchored":
                continue
            lower = payload["path"].lower()
            if any(token in lower for token in ("reset", "pose", "layout", "geometry", "aabb", "scan", "state")):
                candidates.append(payload)
        if candidates:
            # One exact reset/state payload per source is sufficient to verify
            # producer semantics; full coverage still requires the source's
            # complete manifest and remains unresolved.
            indispensable_requests.append(
                {
                    "cohort_id": cohort["cohort_id"],
                    **candidates[0],
                    "request_scope": "exact_file_only",
                    "selection_reason": "minimal_semantics_probe; not a complete cohort export",
                }
            )

    return {
        "schema_version": "sgw-01-historical-layout-coverage-read-request-v1",
        "purpose": "exhaustive_accounting_of_the_45_record_inventory",
        "inventory_source": {
            "path": str(INVENTORY.relative_to(REPO_ROOT)),
            "sha256": _sha256(inventory_path),
            "record_count": len(records),
        },
        "integrated_registry_reference": {
            "commit": "6484f244",
            "claim": "existing 24-layout registry is preserved; this manifest does not replace or extend its comparable rows",
        },
        "records": records,
        "recovered_root_only_evidence": recovered_root_only,
        "recovered_geometry_rows": [],
        "indispensable_external_requests": indispensable_requests,
        "counts": {
            "total": len(records),
            "already_covered": sum(r["coverage_status"] == "already_covered" for r in records),
            "source_inspection_needed": sum(r["coverage_status"] == "source_inspection_needed" for r in records),
            "requires_named_hash_anchored_pvc_payloads": sum(
                r["coverage_status"] == "requires_named_hash_anchored_pvc_payloads" for r in records
            ),
            "arena_excluded": 0,
            "comparable_geometry_eligible": 0,
            "recovered_root_only_rows": len(recovered_root_only),
            "recovered_geometry_rows": 0,
            "indispensable_external_requests": len(indispensable_requests),
        },
        "prospective_neutral_centers": {
            "height_center_m": 0.16,
            "distance_center_m": 0.12,
            "conservative_exclusion_candidates": [],
            "note": "Configured tabletop coordinates are not treated as measured geometric centers or reset bounds; no >3 mm exclusion is asserted without source semantics.",
        },
        "unresolved_domains": [
            "external reset/layout payloads for the V3-E006 registration and repair lineage",
            "hash-anchored payloads named by V2/V3 evidence manifests",
            "raw reset geometry for locally committed sources lacking a payload locator",
            "exact source-to-6484f244 registry bindings for any proposed already-covered record",
        ],
        "coverage_status": "incomplete_unresolved_historical_layout_coverage",
        "release_authorization": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compile_manifest(), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
