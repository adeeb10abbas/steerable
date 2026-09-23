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

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_SPECS = (
    ("V2-A001-pi0-fast", "artifacts/vla_wam_shared_v2/pilot/results/pi0_fast_direct_confirmation.json", "complete"),
    ("V2-A005-groot-n17", "artifacts/vla_wam_shared_v2/pilot/expansion/groot_n17_droid_v2_registry.json", "complete"),
    ("V2-A005-cosmos-edge", "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_droid_direct_gate.json", "complete"),
    ("V2-A007-dreamzero", "artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_droid_direct_gate.json", "complete"),
    ("V2-A010-pi05", "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_direct_gate.json", "complete"),
    ("V2-A011-cosmos-nano", "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_v2a011_registry.json", "complete"),
    ("V2-A015-cosmos-nano-g1", "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_v2a015_no_cfg_g1_result.json", "complete"),
    ("V2-A015-dreamzero-s2", "artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_v2a015_action_cfg_s2_result.json", "complete"),
    ("V3-A-phase-a-groot", "artifacts/vla_wam_shared_v3/results/groot_n17_droid_phase_a_evidence_hash_manifest.json", "complete"),
    ("V3-A-phase-a-edge", "artifacts/vla_wam_shared_v3/results/cosmos3_edge_policy_droid_phase_a_evidence_hash_manifest.json", "complete"),
    ("V3-A-phase-a-nano", "artifacts/vla_wam_shared_v3/results/cosmos3_nano_policy_droid_phase_a_evidence_hash_manifest.json", "complete"),
    ("V3-A-phase-a-dreamzero", "artifacts/vla_wam_shared_v3/results/dreamzero_droid_action_cfg_phase_a_evidence_hash_manifest.json", "complete"),
    ("V3-A-phase-a-pi05", "artifacts/vla_wam_shared_v3/results/pi05_current_stack_droid_phase_a_evidence_hash_manifest.json", "complete"),
    ("V3-C-groot", "artifacts/vla_wam_shared_v3/phase_c/four_phrasings_v3c001/results/groot_n17_droid_vla/groot_n17_droid_vla_phase_c_evidence_manifest.json", "complete"),
    ("V3-C-edge", "artifacts/vla_wam_shared_v3/phase_c/four_phrasings_v3c001/results/cosmos3_edge_policy_droid/cosmos3_edge_policy_droid_phase_c_evidence_manifest.json", "complete"),
    ("V3-C-nano", "artifacts/vla_wam_shared_v3/phase_c/four_phrasings_v3c001/results/cosmos3_nano_policy_droid/cosmos3_nano_policy_droid_phase_c_evidence_manifest.json", "complete"),
    ("V3-E004", "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/evidence_manifest.json", "complete"),
    ("V3-E006", "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/gates/e004_full_reset_reference.json", "reference_only"),
    ("V3-A002-public-old-name-config", "artifacts/vla_wam_shared_v3/results/pi0_fast_old_name_config_v3a002_evidence_hash_manifest.json", "complete"),
    ("V3-B001-nano-position-reflection", "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_manifest.json", "complete"),
    ("V3-B003-dreamzero-position-reflection", "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/dreamzero_mirror_v3b003_manifest.json", "complete"),
    ("V3-B004-nano-lateral-dose-failed-closed", "artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b004/model_blind_calibration_failure_report.json", "failed_closed"),
    ("V3-B005-nano-lateral-dose", "artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b005/nano_lateral_v3b005_manifest.json", "complete"),
    ("V3-D001-pi05-stochastic", "artifacts/vla_wam_shared_v3/prospective_tier_b/results/v3d001/evidence_manifest.json", "complete"),
    ("V3-E001-prompt-noise", "artifacts/vla_wam_shared_v3/phase_e/fixed_observation_prompt_noise_v3e001/evidence_manifest.json", "complete"),
    ("V3-E002-reference-controller", "artifacts/vla_wam_shared_v3/phase_e/reference_controller_symmetry_v3e002/evidence_manifest.json", "complete"),
    ("V3-E003-bilateral-symmetry-null", "artifacts/vla_wam_shared_v3/phase_e/bilateral_symmetry_null_control_v3e003/evidence_manifest.json", "complete"),
    ("V3-E006-canonical-results", "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/results/evidence_manifest.json", "complete"),
    ("V3-E006-source-lineage", "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/source_lineage.json", "lineage_only"),
) + tuple(
    (
        f"V3-E006-repair-r{index:03d}",
        f"artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r{index:03d}/results/"
        + ("results.json" if index in (10, 11) else "evidence_manifest.json"),
        "repair_record",
    )
    for index in range(1, 13)
) + (
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
        if "robotwin" in lowered:
            continue
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
