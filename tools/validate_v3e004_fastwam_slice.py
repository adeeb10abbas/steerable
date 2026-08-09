#!/usr/bin/env python3
"""Fail-closed validator for the complete descriptive FastWAM E004 slice."""
from __future__ import annotations

import json
from pathlib import Path

from build_v3e004_fastwam_slice_bundle import (
    BASE,
    MODEL_ID,
    RAW_MANIFEST_SHA256,
    faststart_contract,
    load_json,
    load_jsonl,
    require,
    sha256,
)


def validate() -> dict[str, object]:
    manifest = load_json(BASE / "evidence_manifest.json")
    require(manifest.get("status") == "complete_hash_closed_arena_slice_descriptive_only", "slice evidence status differs")
    require(manifest.get("behavioral_denominator") == {"registered": 108, "valid": 108}, "slice denominator differs")
    require(sha256(BASE / "raw_cohort_manifest.json") == RAW_MANIFEST_SHA256, "raw cohort digest differs")
    raw = load_json(BASE / "raw_cohort_manifest.json")
    require(raw.get("status") == "complete_hash_closed_108_registered_behavioral_cells", "raw cohort status differs")
    require(len(raw.get("episodes", [])) == 108, "raw cohort episode ledger differs")

    episodes = load_jsonl(BASE / "results/episodes.jsonl")
    pairs = load_jsonl(BASE / "results/pairs.jsonl")
    require(len(episodes) == 108 and len(pairs) == 54, "compact episode/pair counts differ")
    require(len({row["cell_id"] for row in episodes}) == 108, "compact cells are not unique")
    require(all(row["model_id"] == MODEL_ID and row["arena"] == "robotwin" for row in episodes), "arena/model boundary differs")
    require({row["cell_id"] for row in episodes} == {row["cell_id"] for row in raw["episodes"]}, "raw/compact cell multisets differ")

    report = load_json(BASE / "results/results.json")
    checkpoint = report["checkpoints"][MODEL_ID]
    require(checkpoint["valid_episodes"] == 108 and checkpoint["complete_pairs"] == 54, "compiled checkpoint coverage differs")
    require(checkpoint["core_s0_s1_complete"] is True, "paired core is incomplete")
    require(checkpoint["claim_gate"]["publication_claims_enabled"] is False, "partial report enabled publication claims")
    require(checkpoint["analysis"]["equivalence_at_s1"]["binary_gap"]["equivalent"] is False, "binary equivalence was improperly claimed")
    require(checkpoint["analysis"]["equivalence_at_s1"]["depth_gap_m"]["equivalent"] is False, "depth equivalence was improperly claimed")

    media = load_json(BASE / "media/media_manifest.json")
    require(len(media.get("videos", [])) == 4, "selected video count differs")
    for item in media["videos"]:
        path = Path(__file__).resolve().parents[1] / item["path"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"], f"selected video binding differs: {path}")
        require(item["sha256"] != item["source_video"]["sha256"], f"selected video was not converted to fast-start: {path}")
        require(item["publication_container"] == faststart_contract(path), f"selected video container contract differs: {path}")

    figure_manifest = load_json(BASE / "figures/figure_manifest.json")
    require(figure_manifest.get("claim_boundary", {}).get("positive_control_failed_closed") is True, "figure claim boundary differs")
    for item in figure_manifest["figures"]:
        path = Path(__file__).resolve().parents[1] / item["path"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"], f"figure binding differs: {path}")

    compact_records = manifest.get("compact_files", [])
    require(compact_records, "compact evidence ledger is empty")
    for item in compact_records:
        path = Path(__file__).resolve().parents[1] / item["path"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"], f"compact binding differs: {path}")
    return {
        "status": "valid_complete_fastwam_robotwin_slice",
        "behavioral_episodes": len(episodes),
        "matched_pairs": len(pairs),
        "selected_videos": len(media["videos"]),
        "publication_claims_enabled": False,
    }


def main() -> None:
    print(json.dumps(validate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
