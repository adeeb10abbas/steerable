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
    "artifacts/vla_wam_shared_v2/continuation_state.json",
    "artifacts/vla_wam_shared_v2/pilot/post_result_current_stack_replication_amendment.json",
    "artifacts/vla_wam_shared_v2/pilot/post_result_lawam_withdrawal_amendment.json",
    "artifacts/vla_wam_shared_v2/pilot/post_result_pi05_current_stack_media_gate_amendment.json",
    "artifacts/vla_wam_shared_v2/pilot/post_result_cosmos3_nano_droid_amendment.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/pi0_fast_current_stack_v2a008_registry.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_checkpoint_manifest.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_registry.json",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_v2a011_registry.json",
    "artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison.json",
    "artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison.md",
    "artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison_manifest.json",
    "artifacts/vla_wam_shared_v2/media/video_first_gallery_manifest.json",
]

FIGURES = [
    "artifacts/vla_wam_shared_v2/figures/direct_command_cross_model_comparison_1600x900.svg",
    "artifacts/vla_wam_shared_v2/figures/groot_n17_droid_endpoint_redirection.svg",
    "artifacts/vla_wam_shared_v2/figures/robotwin_wam_confirmation_pairs03_09_1600x900.png",
    "artifacts/vla_wam_shared_v2/figures/robotwin_wam_paired_endpoints_1600x900.png",
    "artifacts/vla_wam_shared_v2/figures/pi0_fast_paired_paths_1600x900.png",
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
    records = [record(path, "reader_core") for path in CORE]
    records += [record(path, "publication_figure") for path in FIGURES]
    records += [record(path, "selected_media") for path in media if path not in CORE]
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
        ],
        "asset_count": len(records),
        "assets": records,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(records)} assets")


if __name__ == "__main__":
    main()
