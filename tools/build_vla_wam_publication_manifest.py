#!/usr/bin/env python3
"""Build the bounded, hash-bearing publication asset whitelist."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/vla_wam_shared_v2/publication_manifest.json"

CORE = [
    "README.md",
    "docs/VLA_WAM_RESEARCH_BLOG.md",
    "docs/PUBLICATION_HANDOFF.md",
    "docs/VLA_WAM_STEERABILITY_VIDEO_GALLERY.html",
    "docs/VLA_WAM_STEERABILITY_VIDEO_GALLERY.md",
    "docs/VLA_WAM_STEERABILITY_V2_PROTOCOL.md",
    "artifacts/vla_wam_shared_v2/media/README.md",
    "artifacts/vla_wam_shared_v2/media/media_catalog.csv",
    "artifacts/vla_wam_shared_v2/media/media_catalog.json",
    "outputs/vla_wam_research_handoff/vla_wam_study_stats.xlsx",
    "tools/build_vla_wam_media_catalog.py",
    "tools/build_vla_wam_stats_workbook.mjs",
    "artifacts/vla_wam_shared_v2/continuation_state.json",
    "artifacts/vla_wam_shared_v2/pilot/post_result_current_stack_replication_amendment.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/pi0_fast_current_stack_v2a008_release_probe.json",
    "artifacts/vla_wam_shared_v2/pilot/post_result_pi05_current_stack_media_gate_amendment.json",
    "artifacts/vla_wam_shared_v2/pilot/post_result_cosmos3_nano_droid_amendment.json",
    "artifacts/vla_wam_shared_v2/pilot/post_result_cosmos3_super_droid_amendment.json",
    "artifacts/vla_wam_shared_v2/pilot/post_result_cosmos3_edge_base_amendment.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/pi0_fast_current_stack_v2a008_registry.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_checkpoint_manifest.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_registry.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_direct_gate.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_release_probe.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_fixed_observation.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_invalid_attempts.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_provenance.json",
    "artifacts/vla_wam_shared_v2/media/pi05_current_stack_v2a010/media_manifest.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_v2a011_registry.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_direct_gate.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_fixed_observation.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_invalid_attempts.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_runtime_interventions.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_raw_layout_compatibility.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_evidence_index.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_provenance.json",
    "artifacts/vla_wam_shared_v2/media/cosmos3_nano_policy_droid_v2a011/media_manifest.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_droid_v2a012_registry.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_v2a012_hf_snapshot.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_base_v2a013_registry.json",
    "artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison.json",
    "artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison.csv",
    "artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison.md",
    "artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison_manifest.json",
    "artifacts/vla_wam_shared_v2/media/video_first_gallery_manifest.json",
]

# Compact, reproducible provenance for the Cosmos3 Super / Edge base-model
# feasibility work.  Deliberately exclude PVC-resident outputs, checkpoints,
# environments, and any media asset that is not selected through the gallery.
COSMOS3_SUPER_EDGE_BASE = [
    "artifacts/vla_wam_shared_v2/pilot/post_result_cosmos3_super_droid_amendment.json",
    "artifacts/vla_wam_shared_v2/pilot/post_result_cosmos3_edge_base_amendment.json",
    "artifacts/vla_wam_shared_v2/pilot/post_result_cosmos3_super_image_only_v2a014_amendment.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_droid_v2a012_registry.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_v2a012_hf_snapshot.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_v2a012_runtime_gate.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_image_only_v2a014_registry_overlay.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_image_only_v2a014_result.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_image_only_v2a014_provenance.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_image_only_v2a014_invalid_attempts.json",
    "artifacts/vla_wam_shared_v2/media/cosmos3_super_base_v2a014/media_manifest.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_base_v2a013_registry.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_base_v2a013_fixed_observation.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_base_v2a013_invalid_attempts.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_base_v2a013_curobo_usd_audit.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_base_v2a013_provenance.json",
    "artifacts/vla_wam_shared_v2/media/cosmos3_edge_base_v2a013/media_manifest.json",
    "experiments/cosmos/COSMOS3_SUPER_V2A012.md",
    "experiments/cosmos/COSMOS3_EDGE_BASE_V2A013.md",
    "experiments/cosmos/run_cosmos3_super_v2a014_probe.py",
    "experiments/cosmos/run_edge_base_v2a013_fixed_observation.py",
    "tools/build_cosmos3_super_checkpoint_manifest.py",
    "tools/finalize_cosmos3_super_registry.py",
    "tools/build_cosmos3_super_v2a014_media.py",
    "tools/build_v2a013_cosmos3_edge_base_registry.py",
    "tools/build_cosmos3_edge_base_v2a013_media.py",
    "tools/audit_v2a013_curobo_usd.py",
    "handoff/k8s/cosmos3-super-b200-2gpu-256gi-ali.yaml",
    "handoff/k8s/cosmos3-super-b200-4gpu-256gi-ali.yaml",
    "handoff/k8s/cosmos3-super-a100-2gpu-256gi-ali.yaml",
]

FIGURES = [
    "artifacts/vla_wam_shared_v2/figures/direct_command_cross_model_comparison_1600x900.svg",
    "artifacts/vla_wam_shared_v2/figures/groot_n17_droid_endpoint_redirection.svg",
    "artifacts/vla_wam_shared_v2/figures/robotwin_wam_confirmation_pairs03_09_1600x900.png",
    "artifacts/vla_wam_shared_v2/figures/robotwin_wam_paired_endpoints_1600x900.png",
    "artifacts/vla_wam_shared_v2/figures/pi0_fast_paired_paths_1600x900.png",
    "artifacts/vla_wam_shared_v2/figures/v2a015_cfg_guidance_ablation.svg",
    "artifacts/vla_wam_shared_v2/figures/v2a015_cfg_guidance_ablation.png",
]

CFG_ABLATION_V2A015 = [
    "artifacts/vla_wam_shared_v2/pilot/post_result_cfg_ablation_v2a015_amendment.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cfg_ablation_v2a015_preflight.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_v2a015_no_cfg_g1_result.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_v2a015_action_cfg_s2_result.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cfg_ablation_v2a015_comparison.json",
    "tools/build_v2a015_cfg_media.py",
    "tools/render_v2a015_cfg_scientific_figure.py",
    "tools/render_vla_wam_video_first_gallery.py",
    "tools/validate_vla_wam_v2_protocol.py",
    "artifacts/vla_wam_shared_v2/protocol_validation.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in strings(child)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in strings(child)]
    return []


def media_assets() -> list[str]:
    queue = ["artifacts/vla_wam_shared_v2/media/video_first_gallery_manifest.json"]
    visited: set[str] = set()
    selected: set[str] = set()
    while queue:
        relative = queue.pop()
        if relative in visited:
            continue
        visited.add(relative)
        path = ROOT / relative
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        data = json.loads(path.read_text())
        for candidate in strings(data):
            if not candidate.startswith("artifacts/vla_wam_shared_v2/"):
                continue
            candidate_path = ROOT / candidate
            if not candidate_path.is_file():
                continue
            if "/media/" not in candidate:
                continue
            selected.add(candidate)
            if candidate_path.suffix.lower() == ".json":
                queue.append(candidate)
    return sorted(selected)


def record(relative: str, category: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    return {
        "path": relative,
        "category": category,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    media = media_assets()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_unique(paths: list[str], category: str) -> None:
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            records.append(record(path, category))

    add_unique(CORE, "reader_core")
    add_unique(COSMOS3_SUPER_EDGE_BASE, "cosmos3_super_edge_base_provenance")
    add_unique(CFG_ABLATION_V2A015, "cfg_ablation_v2a015")
    add_unique(FIGURES, "publication_figure")
    add_unique(media, "selected_media")
    records.sort(key=lambda item: (item["category"], item["path"]))
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    output = {
        "schema_version": "vla_wam_publication_manifest_v1",
        "status": "article_ready_asset_whitelist",
        "source_git_head": git_head,
        "claim_boundary": [
            "DROID and RoboTwin success rates remain separate",
            "invalid infrastructure attempts are outside behavioral denominators",
            "missing, latent-only, and action-only futures are not scored as zeros",
            "historical reference media is labeled and never substituted for current evidence",
            "Cosmos3 base-model probes remain non-behavioral; selected generated media is labeled prediction-only with actual rollout unavailable",
        ],
        "asset_count": len(records),
        "assets": records,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(records)} assets")


if __name__ == "__main__":
    main()
