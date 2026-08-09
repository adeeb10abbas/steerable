#!/usr/bin/env python3
"""Build compact, descriptive evidence for the complete FastWAM E004 slice."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
E004 = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"
BASE = E004 / "slices/fastwam_robotwin"
MODEL_ID = "fastwam_robotwin"
ARENA = "robotwin"
RAW_MANIFEST_SHA256 = "6fe5cc00c07aa88ce6b1828f52a51e5d762a75d922927fa08e5b7b92bf63f515"
PROMPTS = {
    "left": "Put the small woodenblock to the left of the red playingcards box.",
    "right": "Put the small woodenblock to the right of the red playingcards box.",
}
MEDIA = {
    (0.0, "left"): "seed9413_asymmetric_left.mp4",
    (0.0, "right"): "seed9413_asymmetric_right.mp4",
    (1.0, "left"): "seed9413_symmetric_left.mp4",
    (1.0, "right"): "seed9413_symmetric_right.mp4",
}


class BundleError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    require(path.is_file(), f"missing compact file: {path}")
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def faststart_contract(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    offsets = {name: payload.find(name.encode("ascii")) for name in ("ftyp", "moov", "mdat", "avc1")}
    require(all(value >= 0 for value in offsets.values()), f"selected MP4 lacks required H.264 atoms: {path}")
    require(offsets["ftyp"] < offsets["moov"] < offsets["mdat"], f"selected MP4 is not fast-start ordered: {path}")
    return {"codec": "H.264/avc1", "fast_start": True, "atom_offsets": offsets}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _episode_index() -> dict[tuple[int, float, str], Mapping[str, Any]]:
    rows = load_jsonl(BASE / "results/episodes.jsonl")
    require(len(rows) == 108, "FastWAM compact episode count differs")
    require(all(row.get("model_id") == MODEL_ID and row.get("arena") == ARENA for row in rows), "arena/model boundary differs")
    index = {(int(row["environment_seed"]), float(row["symmetry_level_s"]), str(row["relation"])): row for row in rows}
    require(len(index) == 108, "FastWAM compact cells are not unique")
    return index


def build_media_manifest(index: Mapping[tuple[int, float, str], Mapping[str, Any]]) -> dict[str, Any]:
    videos = []
    for (level, relation), filename in MEDIA.items():
        episode = index[(9413, level, relation)]
        path = BASE / "media" / filename
        item = record(path)
        raw = episode["raw_artifacts"]["simulator_video"]
        require(item["sha256"] != raw["sha256"], f"selected media was not converted to fast-start: {filename}")
        require(episode["prompt"] == PROMPTS[relation], f"selected media prompt differs: {filename}")
        videos.append(
            {
                **item,
                "cell_id": episode["cell_id"],
                "environment_seed": 9413,
                "layout": "asymmetric_registered_s0_fixture" if level == 0.0 else "symmetric_object_layout_s1_not_symmetric_robot",
                "symmetry_level_s": level,
                "relation": relation,
                "exact_prompt": PROMPTS[relation],
                "success": episode["success"],
                "failure_category": episode["failure_category"],
                "source_video": raw,
                "publication_container": faststart_contract(path),
                "derivation": {
                    "operation": "video-stream copy with MP4 moov atom moved before mdat",
                    "command_contract": "ffmpeg 7.0.2 -map 0:v:0 -c copy -movflags +faststart",
                    "decoded_frames_reencoded": False,
                },
            }
        )
    return {
        "schema_version": "vla-wam-shared-v3e004-fastwam-selected-video-manifest-v1",
        "amendment_id": "V3-E004",
        "model_id": MODEL_ID,
        "arena": ARENA,
        "status": "four_source_bound_faststart_selected_publication_clips",
        "selection_reason": "Seed 9413 provides a matched four-cell layout-by-prompt comparison: both s=0 cells failed, while s=1 LEFT succeeded and s=1 RIGHT failed. Selection is illustrative and is not the statistical denominator.",
        "videos": videos,
        "claim_boundary": {
            "raw_videos_are_actual_simulator_execution_not_imagination": True,
            "selection_is_not_a_success_rate": True,
            "robotwin_never_pooled_with_droid": True,
        },
    }


def memo(report: Mapping[str, Any], media_manifest: Mapping[str, Any]) -> str:
    checkpoint = report["checkpoints"][MODEL_ID]
    levels = checkpoint["analysis"]["levels"]
    interaction = checkpoint["analysis"]["interaction_s1_minus_s0_core"]
    s0_depth = levels["0.00"]["requested_depth_gap_R_minus_L_m"]
    s1_depth = levels["1.00"]["requested_depth_gap_R_minus_L_m"]
    depth_interaction = interaction["depth_gap_m"]

    def ci_cm(row: Mapping[str, Any]) -> str:
        ci = row["bootstrap_mean95"]
        return f"{row['mean'] * 100:+.1f} cm (95% CI {ci['low'] * 100:+.1f} to {ci['high'] * 100:+.1f})"

    lines = [
        "# V3-E004 FastWAM RoboTwin slice",
        "",
        "**Status:** complete arena-separated stretch slice; descriptive only until the full registered V3-E004 cohort closes. RoboTwin is never pooled with DROID.",
        "",
        "## Design",
        "",
        "FastWAM was evaluated on 27 matched environment/sampling seeds (9400–9426), two object layouts (registered asymmetric s=0 and symmetric-object s=1), and two exact static prompts, for 108 valid behavioral episodes. Infrastructure-invalid setup and acceleration attempts remain outside the denominator. The s=1 fixture is symmetric in its movable-object layout, not in the robot embodiment.",
        "",
        "- LEFT: `Put the small woodenblock to the left of the red playingcards box.`",
        "- RIGHT: `Put the small woodenblock to the right of the red playingcards box.`",
        "",
        "## Descriptive result",
        "",
        f"Binary task success was at floor under s=0 (LEFT 0/27; RIGHT 0/27) and remained very low under s=1 (LEFT 1/27; RIGHT 2/27). The continuous requested-depth contrast changed from {ci_cm(s0_depth)} under s=0 to {ci_cm(s1_depth)} under s=1. The paired s=1-minus-s=0 interaction was {ci_cm(depth_interaction)} (exact layout-label permutation p={depth_interaction['exact_layout_label_permutation']['exact_two_sided_p']:.3g}).",
        "",
        "That reversal is not sufficient for the registered equalisation interpretation. The endpoint-redirection positive control did not remain detectably positive: the paired LEFT-minus-RIGHT endpoint estimate was +1.3 cm (95% CI −2.9 to +6.2) at s=0 and +0.6 cm (95% CI −4.2 to +5.6) at s=1. Equivalence is also not claimed: both registered FastWAM estimands were classified as underpowered stretch analyses. The safe statement is therefore that the object-layout intervention strongly changed FastWAM's continuous depth contrast in this RoboTwin slice, while near-zero competence and a failed prompt-redirection positive control prevent attributing that change to reliable language steering.",
        "",
        "Failure decomposition supports the competence boundary: s=0 contained 53 pick failures and one transport failure; s=1 contained 47 pick failures, four wrong-side failures, and three correct episodes.",
        "",
        "## Figure",
        "",
        "The complete slice figure reports binary success, the requested-depth interaction, the failed endpoint-redirection positive control, and failure composition together: [PNG](figures/v3e004_fastwam_robotwin_slice.png) · [SVG](figures/v3e004_fastwam_robotwin_slice.svg).",
        "",
        "## Selected actual-rollout videos",
        "",
        "These four clips are the complete matched seed-9413 layout-by-prompt set. They are illustrative actual simulator executions, not imagined futures and not a replacement for the 108-episode denominator.",
        "",
        "| Layout | Exact prompt | Outcome | Clip |",
        "|---|---|---|---|",
    ]
    for item in media_manifest["videos"]:
        layout = "asymmetric s=0" if item["symmetry_level_s"] == 0.0 else "symmetric-object s=1"
        outcome = "success" if item["success"] else item["failure_category"]
        filename = Path(item["path"]).name
        lines.append(f"| {layout} | {item['exact_prompt']} | {outcome} | [{filename}](media/{filename}) |")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This slice supports no VLA-versus-WAM comparison, no DROID/RoboTwin pooled rate, no equivalence statement, and no claim that the symmetric-object layout is a symmetric robot. Cross-checkpoint manuscript language remains withheld until the complete registered V3-E004 evidence is hash-closed.",
            "",
        ]
    )
    return "\n".join(lines)


def build() -> dict[str, Any]:
    raw_manifest_path = BASE / "raw_cohort_manifest.json"
    require(sha256(raw_manifest_path) == RAW_MANIFEST_SHA256, "canonical raw cohort manifest digest differs")
    raw_manifest = load_json(raw_manifest_path)
    require(raw_manifest.get("status") == "complete_hash_closed_108_registered_behavioral_cells", "raw cohort is not hash-closed")
    require(raw_manifest.get("behavioral_denominator", {}).get("valid") == 108, "raw denominator differs")
    report_path = BASE / "results/results.json"
    report = load_json(report_path)
    checkpoint = report.get("checkpoints", {}).get(MODEL_ID, {})
    require(checkpoint.get("valid_episodes") == 108 and checkpoint.get("core_s0_s1_complete") is True, "compiled FastWAM slice is incomplete")
    require(checkpoint.get("claim_gate", {}).get("publication_claims_enabled") is False, "partial compiler improperly enabled publication claims")
    index = _episode_index()
    raw_ids = {item["cell_id"] for item in raw_manifest["episodes"]}
    require(raw_ids == {row["cell_id"] for row in index.values()}, "raw and compact episode multisets differ")

    media_manifest = build_media_manifest(index)
    media_path = BASE / "media/media_manifest.json"
    media_path.write_text(json.dumps(media_manifest, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    memo_path = BASE / "FASTWAM_SLICE_MEMO.md"
    memo_path.write_text(memo(report, media_manifest), encoding="utf-8")

    figure_manifest = load_json(BASE / "figures/figure_manifest.json")
    require(figure_manifest.get("status") == "complete_arena_slice_descriptive_only", "slice figure manifest status differs")
    compact_paths = [
        raw_manifest_path,
        report_path,
        BASE / "results/episodes.jsonl",
        BASE / "results/pairs.jsonl",
        BASE / "results/source_ledger.jsonl",
        BASE / "results/infrastructure_invalid.jsonl",
        BASE / "figures/figure_manifest.json",
        BASE / "figures/v3e004_fastwam_robotwin_slice.png",
        BASE / "figures/v3e004_fastwam_robotwin_slice.svg",
        media_path,
        memo_path,
        E004 / "registration.json",
        E004 / "queue.jsonl",
        E004 / "layout/fastwam_robotwin_candidate.json",
    ]
    compact_paths.extend(BASE / item["path"].split("slices/fastwam_robotwin/", 1)[1] for item in media_manifest["videos"])
    implementation_paths = [
        ROOT / "tools/build_v3e004_fastwam_cohort_manifest.py",
        ROOT / "tools/compile_v3e004_results.py",
        ROOT / "tools/render_v3e004_fastwam_slice.py",
        ROOT / "tools/build_v3e004_fastwam_slice_bundle.py",
        ROOT / "tools/validate_v3e004_fastwam_slice.py",
        ROOT / "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/fastwam_runtime.py",
        ROOT / "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/fastwam_robotwin.py",
    ]
    manifest = {
        "schema_version": "vla-wam-shared-v3e004-fastwam-slice-evidence-manifest-v1",
        "amendment_id": "V3-E004",
        "model_id": MODEL_ID,
        "arena": ARENA,
        "status": "complete_hash_closed_arena_slice_descriptive_only",
        "behavioral_denominator": {"registered": 108, "valid": 108},
        "canonical_raw_manifest": record(raw_manifest_path),
        "compact_files": [record(path) for path in compact_paths],
        "implementation_files": [record(path) for path in implementation_paths],
        "selected_media_manifest": record(media_path),
        "figure_manifest": record(BASE / "figures/figure_manifest.json"),
        "scientific_boundaries": {
            "full_v3e004_publication_claims_withheld": True,
            "equivalence_not_claimed": True,
            "endpoint_positive_control_failed_closed": True,
            "robotwin_never_pooled_with_droid": True,
            "symmetric_object_layout_not_symmetric_robot": True,
            "behavioral_failures_retained": True,
        },
        "raw_evidence_policy": "Full rollouts, action traces, and unselected videos remain on the ali-owned PVC. The committed raw cohort manifest binds every source file by SHA-256; four small source-bound actual-rollout clips are stream-copied into fast-start MP4 containers for browser playback.",
    }
    output = BASE / "evidence_manifest.json"
    output.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    value = build()
    print(json.dumps({"status": value["status"], "compact_files": len(value["compact_files"])}, indent=2))


if __name__ == "__main__":
    main()
