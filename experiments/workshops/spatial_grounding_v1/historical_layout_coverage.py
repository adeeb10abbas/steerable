"""Additive coverage audit for locally committed historical layout sources.

The audit supplements the hash-bound evidence registry without changing its
release semantics. It extracts configured root-only rows when a source has
explicit object coordinates, and records why those rows cannot be promoted to
comparable geometry without measured quaternions and offsets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

SOURCE_SPECS = (
    ("v3e004-builder-input", "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/layout/builder_input.json", "DROID/RoboLab configured source coordinates; no full quaternion or measured reset"),
    ("v3e004-static-layout-gate", "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/gates/static_layout_gate.json", "gate accounting only; no object pose payload"),
    ("v3e005-scene-candidate", "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005/layout/scene_candidate.json", "RoboTwin cross-arena candidate; excluded from DROID comparison"),
    ("v2-nano-raw-layout-compatibility", "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_raw_layout_compatibility.json", "DROID raw-path compatibility ledger; payloads remain PVC-bound"),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob(repo_root: Path, path: str) -> tuple[str | None, str | None]:
    try:
        commit = subprocess.check_output(["git", "-C", str(repo_root), "log", "-1", "--format=%H", "--", path], text=True).strip()
        blob = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", f"{commit}:{path}"], text=True).strip()
        return commit, blob
    except subprocess.CalledProcessError:
        return None, None


def _configured_rows(document: dict[str, Any], source_id: str) -> list[dict[str, Any]]:
    rows = []
    for variant_key in ("control_poses", "symmetric_poses"):
        for object_name, pose in document.get(variant_key, {}).items():
            if not all(key in pose for key in ("x_m", "y_m", "z_m")):
                continue
            rows.append({
                "source_id": source_id,
                "variant": variant_key,
                "object": object_name,
                "configured_position_robot_base_m": [pose["x_m"], pose["y_m"], pose["z_m"]],
                "yaw_rad": pose.get("yaw_rad"),
                "asset_identity": pose.get("asset_identity"),
                "status": "configured_root_only",
                "blocker": "no explicit measured root quaternion and no reset/local geometric offset",
            })
    return rows


def compile_coverage(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    sources = []
    rows = []
    blockers = []
    for source_id, relative, purpose in SOURCE_SPECS:
        path = repo_root / relative
        commit, blob = _git_blob(repo_root, relative)
        record = {
            "source_id": source_id,
            "path": relative,
            "purpose": purpose,
            "arena": "excluded_robotwin" if source_id == "v3e005-scene-candidate" else "droid_robolab",
            "commit": commit,
            "git_blob_sha1": blob,
        }
        if not path.is_file():
            record["status"] = "git_bound_not_checked_out"
            blockers.append({"source_id": source_id, "reason": "source path unavailable in current checkout"})
            sources.append(record)
            continue
        data = path.read_bytes()
        record.update({"status": "locally_verified", "bytes": len(data), "sha256": _sha256(data)})
        document = json.loads(data)
        extracted = _configured_rows(document, source_id)
        rows.extend(extracted)
        if not extracted:
            blockers.append({"source_id": source_id, "reason": "no explicit configured object coordinates in this source schema"})
        elif source_id == "v3e005-scene-candidate":
            blockers.append({"source_id": source_id, "reason": "RoboTwin source excluded; assets/tasks are not DROID cube/bowl layouts"})
        sources.append(record)
    return {
        "schema_version": "sgw-01-historical-layout-coverage-v1",
        "diagnostic_only": True,
        "sources": sources,
        "configured_root_only_rows": rows,
        "comparable_geometry_rows": [],
        "blockers": blockers,
        "unresolved_domains": [
            "full reset payloads and measured root-local offsets for configured V3-E004 coordinates",
            "PVC-bound V2 raw layout compatibility payloads",
            "RoboTwin E005 source excluded from DROID deduplication",
        ],
        "coverage_status": "incomplete_unresolved_historical_layout_coverage",
        "release_authorization": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compile_coverage(), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
