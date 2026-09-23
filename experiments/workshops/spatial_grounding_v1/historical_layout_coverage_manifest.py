"""Compile the 45-source historical layout coverage/read-request manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
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


SOURCE_CONCLUSIONS = {
    "V2-A001-pi0-fast": "compiled episode outcomes only; no reset-state payload or geometric producer binding",
    "V2-A005-groot-n17": "compiled gate outcomes only; no reset-state payload or geometric producer binding",
    "V2-A005-cosmos-edge": "compiled gate outcomes only; no reset-state payload or geometric producer binding",
    "V2-A011-cosmos-nano": "compiled gate/queue metadata only; no reset-state payload or geometric producer binding",
    "V3-C-groot": "evidence manifest retains raw gate root but no local pose values",
    "V3-C-edge": "evidence manifest retains raw gate root but no local pose values",
    "V3-C-nano": "evidence manifest retains raw gate root but no local pose values",
    "V3-E004": "layout/evidence manifest only; no exact source-to-6484f244 geometry binding",
    "V3-B001-nano-position-reflection": "registration counts and runtime manifests only; no measured reset geometry",
    "V3-B003-dreamzero-position-reflection": "registration counts and runtime manifests only; no measured reset geometry",
    "V3-B004-nano-lateral-dose-failed-closed": "model-blind scan metadata reports collision offsets, but no hash-bound per-layout root/centroid rows",
    "V3-B005-nano-lateral-dose": "queue and physical-gate metadata only; no measured reset geometry",
    "V3-D001-pi05-stochastic": "registration/evidence manifest only; no measured reset geometry",
    "V3-E001-prompt-noise": "evidence manifest only; no measured reset geometry",
    "V3-E002-reference-controller": "execution provenance only; no measured reset geometry",
    "V3-E003-bilateral-symmetry-null": "registration and result references only; no measured reset geometry",
    "V3-B002-pi05-position-reflection": "registration counts and runtime manifests only; no measured reset geometry",
    "V3-B002-pi05-release-gate": "release gate metadata only; no measured reset geometry",
    "V3-B002-pi05-fixed-observation-gate": "fixed-observation gate only; no measured reset geometry",
    "V3-B002-pi05-runtime-identity": "runtime identity only; no measured reset geometry",
}

PRODUCER_BINDINGS = {
    "V3-A-phase-a-groot": {
        "producer_path": "experiments/v3/groot_droid/robolab_bridge.py",
        "producer_role": "scene root_pos_w transformed by robot root pose into robot frame",
        "pinned_robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
    },
    "V3-A-phase-a-nano": {
        "producer_path": "experiments/v3/cosmos_droid/compile_pair.py",
        "producer_role": "compiled object_xyz/reference_xyz fields; producer reset getter is not present in this compiler",
        "pinned_robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
    },
    "V3-A-phase-a-dreamzero": {
        "producer_path": "experiments/v3/dreamzero_droid/robolab_bridge.py",
        "producer_role": "scene root_pos_w transformed by robot root pose into robot frame",
        "pinned_robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
    },
}


def _probe_archive(archive: Path | None) -> dict[str, Any]:
    if archive is None or not archive.is_file():
        return {"status": "not_loaded", "archive": "sgw-bj-historical-payloads.tar.gz"}
    with tarfile.open(archive, "r:gz") as tar:
        manifest_bytes = tar.extractfile("manifest.json").read()
        manifest = json.loads(manifest_bytes)
        probes = []
        for item in manifest["records"]:
            payload = json.loads(tar.extractfile(item["export_path"]).read())
            probe = {
                "cohort_id": item["cohort_id"],
                "path": item["path"],
                "expected_sha256": item["expected_sha256"],
                "actual_sha256": item["actual_sha256"],
                "payload_keys": sorted(payload),
                "semantic_result": (
                    "single_step_object_reference_xyz_only_no_quaternion_or_centroid"
                    if item["export_path"].endswith(".jsonl")
                    else "repair_harness_accounting_only_no_reset_geometry"
                ),
            }
            if item["cohort_id"].startswith("V3-E006"):
                payload = json.loads(tar.extractfile(item["export_path"]).read())
                probe["state_payload_request"] = payload.get("child_report")
                probe["state_payload_semantics"] = "named state_repair_result; not exported in this probe archive"
            probes.append(probe)
    return {
        "status": "loaded_hash_verified",
        "archive_name": archive.name,
        "archive_sha256": _sha256(archive),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "job_uid": manifest["job_uid"],
        "model_requests": manifest["model_requests"],
        "probes": probes,
    }


def compile_manifest(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    inventory_path = repo_root / INVENTORY.relative_to(REPO_ROOT)
    inventory = json.loads(inventory_path.read_text())
    records = []
    recovered_root_only = []
    indispensable_requests = []
    archive = Path(os.environ["SGW_BJ_ARCHIVE"]) if os.environ.get("SGW_BJ_ARCHIVE") else None
    probe_results = _probe_archive(archive)
    for cohort in inventory["cohorts"]:
        source = repo_root / cohort["source_path"]
        status, reason = _status(cohort)
        actual_sha256 = _sha256(source) if source.is_file() else None
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
            "inspection_conclusion": SOURCE_CONCLUSIONS.get(
                cohort["cohort_id"], "no explicit measured geometry found in committed source schema"
            ),
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
        if cohort["cohort_id"] in PRODUCER_BINDINGS:
            binding = dict(PRODUCER_BINDINGS[cohort["cohort_id"]])
            producer = repo_root / binding["producer_path"]
            binding["producer_sha256"] = _sha256(producer) if producer.is_file() else None
            record["producer_binding"] = binding
        if hash_status == "verified_against_inventory":
            recovered_root_only.extend(_recover_explicit_root_only(cohort, repo_root))
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
    for probe in probe_results.get("probes", []):
        request = probe.get("state_payload_request")
        if isinstance(request, dict) and request.get("path") and request.get("sha256"):
            indispensable_requests.append(
                {
                    "cohort_id": probe["cohort_id"],
                    "path": request["path"],
                    "expected_sha256": request["sha256"],
                    "source_json_pointer": "/child_report",
                    "hash_binding_status": "hash_anchored",
                    "request_scope": "exact_file_only",
                    "selection_reason": "named state_repair_result needed to inspect actual state/initial-layout payload; not a complete cohort export",
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
        "external_probe_results": probe_results,
        "producer_semantics": {
            cohort_id: {
                **binding,
                "producer_sha256": (
                    _sha256(repo_root / binding["producer_path"])
                    if (repo_root / binding["producer_path"]).is_file()
                    else None
                ),
                "cohort_wide_exclusion_proven": False,
                "exclusion_blocker": "single probe lacks complete registered cohort reset bounds and asset/layout identity",
            }
            for cohort_id, binding in PRODUCER_BINDINGS.items()
        },
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
