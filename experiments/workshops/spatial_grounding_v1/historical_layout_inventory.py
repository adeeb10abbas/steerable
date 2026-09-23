"""Additive, source-bound historical DROID layout inventory.

This module is intentionally separate from the legacy comparator API. It is
diagnostic evidence only: it does not infer centers, assign layout IDs, or
authorize an SGW release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .historical_layout_dedup import REPO_ROOT, SOURCE_SPECS as LEGACY_SOURCE_SPECS


SOURCE_SPECS = LEGACY_SOURCE_SPECS + (
    (
        "V3-B002-pi05-position-reflection",
        "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/pi05_mirror_v3b002_manifest.json",
        "complete",
    ),
    (
        "V3-B002-pi05-release-gate",
        "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/gates/release_gate.json",
        "release_record",
    ),
    (
        "V3-B002-pi05-fixed-observation-gate",
        "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/gates/fixed_observation_gate.json",
        "gate_record",
    ),
    (
        "V3-B002-pi05-runtime-identity",
        "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/gates/runtime_identity.json",
        "runtime_record",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _escape(token: str) -> str:
    return str(token).replace("~", "~0").replace("/", "~1")


def _index(value: Any, pointer: str = "") -> dict[str, Any]:
    """Return one JSON-pointer lookup table, preserving RFC 6901 escaping."""
    result = {pointer: value}
    if isinstance(value, Mapping):
        for key, child in value.items():
            result.update(_index(child, f"{pointer}/{_escape(key)}"))
    elif isinstance(value, list):
        for number, child in enumerate(value):
            result.update(_index(child, f"{pointer}/{number}"))
    return result


def _path_evidence(index: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for pointer, value in index.items():
        if not isinstance(value, str) or not value.startswith(("/data/", "/mnt/", "/scratch/")):
            continue
        lowered = value.lower()
        if not any(token in lowered for token in ("reset", "layout", "trajectory", "fixture", "manifest", "result.json")):
            continue
        parent = pointer.rsplit("/", 1)[0]
        expected = next(
            (
                index.get(f"{parent}/{name}")
                for name in ("sha256", "file_sha256", "expected_sha256", "artifact_sha256")
                if isinstance(index.get(f"{parent}/{name}"), str)
                and len(index[f"{parent}/{name}"]) == 64
            ),
            None,
        )
        evidence[value] = {
            "path": value,
            "source_json_pointer": pointer,
            "expected_sha256": expected,
        }
    return [evidence[path] for path in sorted(evidence)]


def _explicit_poses(index: Mapping[str, Any]) -> list[dict[str, Any]]:
    poses = []
    for pointer, value in index.items():
        if not isinstance(value, Mapping):
            continue
        position = value.get("root_position")
        quaternion = value.get("root_quaternion_wxyz")
        if not isinstance(position, Mapping) or not isinstance(quaternion, Mapping):
            continue
        if "values" not in position or "values" not in quaternion:
            continue
        poses.append({
            "json_pointer": pointer,
            "root_position": position,
            "root_quaternion_wxyz": quaternion,
            "extraction": "explicit paired root_position/root_quaternion_wxyz only",
        })
    return poses


def build_inventory(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    cohorts = []
    missing_sources = []
    for cohort_id, relative, declared_status in SOURCE_SPECS:
        source = repo_root / relative
        if not source.is_file():
            missing_sources.append(relative)
            continue
        data = json.loads(source.read_text())
        lookup = _index(data)
        cohorts.append({
            "cohort_id": cohort_id,
            "arena": "droid_robolab",
            "source_path": relative,
            "source_sha256": _sha256(source),
            "declared_status": declared_status,
            "relevant_path_evidence": _path_evidence(lookup),
            "explicit_root_poses": _explicit_poses(lookup),
            "layout_ids": [],
            "layout_status": "unresolved_without_explicit_layout_or_reset_record",
            "deduplication_claim": "none",
        })
    unresolved = [
        {
            "cohort_id": cohort_id,
            "reason": "provenance is committed, but no unambiguous layout/reset identity with paired actor-root pose is established",
        }
        for cohort_id, _, _ in SOURCE_SPECS
    ]
    return {
        "schema_version": "sgw-01-historical-droid-layout-source-inventory-v2",
        "purpose": "diagnostic_only_historical_deduplication",
        "arena_inclusion": ["droid_robolab"],
        "arena_exclusion": {
            "robotwin": "excluded explicitly; no RoboTwin cohort is included",
            "reason": "SGW layout deduplication cannot pool or compare arenas",
        },
        "source_policy": {
            "only_committed_sources": True,
            "source_hashes_required": True,
            "json_pointer_standard": "RFC 6901",
            "pose_extraction": "only paired explicit root_position and root_quaternion_wxyz values",
            "no_scoring_center_inference": True,
            "no_coordinate_alias_inference": True,
        },
        "cohorts": cohorts,
        "missing_curated_sources": missing_sources,
        "unresolved_coverage": unresolved,
        "coverage_status": "incomplete_unresolved_historical_layout_coverage",
        "release_authorization": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_inventory(), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
