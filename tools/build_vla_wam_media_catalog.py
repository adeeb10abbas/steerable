#!/usr/bin/env python3
"""Build a complete, role-aware catalog of committed VLA/WAM MP4 evidence."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MEDIA_ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v2/media"
PILOT_MEDIA_ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v2/pilot/expansion/media"
GALLERY_MANIFEST = MEDIA_ROOT / "video_first_gallery_manifest.json"
CFG_MEDIA_ROOT = MEDIA_ROOT / "cfg_v2a015"


CFG_ARM_METADATA = {
    "dreamzero_action_cfg_s2": {
        "model_id": "dreamzero_droid_action_cfg",
        "model": "DreamZero DROID — V2-A015 s=2",
        "prediction_kind": "model prediction / imagination",
        "prediction_role": "CFG ablation complete imagination",
    },
    "cosmos3_nano_no_cfg_g1": {
        "model_id": "cosmos3_nano_policy_droid",
        "model": "Cosmos3 Nano Policy DROID — V2-A015 g=1",
        "prediction_kind": "model prediction / local horizon",
        "prediction_role": "CFG ablation complete local prediction",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / relative).read_text())


def add_reference(
    references: dict[str, dict[str, str]],
    video: dict[str, Any],
    *,
    model: str,
    arena: str,
    model_class: str,
    evidence_kind: str,
    publication_role: str,
    source_manifest: str,
) -> None:
    path = video["path"]
    references[path] = {
        "model": model,
        "arena": arena,
        "model_class": model_class,
        "evidence_kind": evidence_kind,
        "publication_role": publication_role,
        "source_manifest": source_manifest,
    }


def validate_cfg_output(
    record: dict[str, Any], *, label: str, suffixes: tuple[str, ...]
) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} is not a file record")
    path_value = record.get("path")
    expected_bytes = record.get("bytes")
    expected_sha256 = record.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} lacks a path")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise ValueError(f"{label} lacks a positive byte count")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"{label} lacks a SHA-256")
    path = (REPO_ROOT / path_value).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
        path.relative_to(MEDIA_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository media root: {path}") from exc
    if path.suffix.lower() not in suffixes:
        raise ValueError(f"{label} has the wrong media suffix: {path}")
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    if path.stat().st_size != expected_bytes or sha256(path) != expected_sha256:
        raise ValueError(f"{label} path/hash/bytes validation failed: {path}")
    return path


def cfg_ablation_references(
    gallery: dict[str, Any],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    contracts = gallery.get("cfg_ablation_media_contracts")
    if contracts is None:
        return {}, []
    if not isinstance(contracts, list):
        raise ValueError("cfg_ablation_media_contracts must be a list when present")

    references: dict[str, dict[str, str]] = {}
    manifests: list[str] = []
    observed_arms: set[str] = set()
    observed_outputs: set[str] = set()
    expected_outputs = {
        "actual_video",
        "actual_poster",
        "prediction_or_imagination_video",
        "prediction_or_imagination_poster",
    }
    for index, contract in enumerate(contracts):
        if not isinstance(contract, dict) or not isinstance(contract.get("path"), str):
            raise ValueError(f"CFG media contract {index} lacks a manifest path")
        manifest_path = contract["path"]
        manifest_file = (REPO_ROOT / manifest_path).resolve()
        try:
            manifest_file.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"CFG media manifest escapes the repository: {manifest_file}") from exc
        manifest = load_json(manifest_path)
        if (
            manifest.get("schema_version") != "vla-wam-shared-v2-v2a015-cfg-media-v1"
            or manifest.get("status")
            != "complete_all_six_cells_actual_and_prediction_media"
            or manifest.get("amendment_id") != "V2-A015"
        ):
            raise ValueError(f"CFG media manifest is not a complete V2-A015 arm: {manifest_path}")
        arm_id = manifest.get("arm_id")
        if arm_id not in CFG_ARM_METADATA or arm_id in observed_arms:
            raise ValueError(f"Unknown or duplicate CFG media arm: {arm_id!r}")
        observed_arms.add(arm_id)
        metadata = CFG_ARM_METADATA[arm_id]
        if manifest.get("model_id") != metadata["model_id"]:
            raise ValueError(f"CFG media model/arm mismatch: {manifest_path}")
        outputs = manifest.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != expected_outputs:
            raise ValueError(f"CFG media outputs are incomplete: {manifest_path}")

        specs = (
            (
                "actual_video",
                (".mp4",),
                "actual rollout",
                "CFG ablation complete actual execution",
                True,
            ),
            (
                "actual_poster",
                (".jpg", ".jpeg"),
                None,
                None,
                False,
            ),
            (
                "prediction_or_imagination_video",
                (".mp4",),
                metadata["prediction_kind"],
                metadata["prediction_role"],
                True,
            ),
            (
                "prediction_or_imagination_poster",
                (".jpg", ".jpeg"),
                None,
                None,
                False,
            ),
        )
        for output_key, suffixes, evidence_kind, publication_role, catalog_video in specs:
            path = validate_cfg_output(
                outputs[output_key],
                label=f"{arm_id} {output_key}",
                suffixes=suffixes,
            )
            relative = str(path.relative_to(REPO_ROOT.resolve()))
            if relative in observed_outputs:
                raise ValueError(f"Duplicate CFG media output path: {relative}")
            observed_outputs.add(relative)
            if not catalog_video:
                continue
            references[relative] = {
                "model": metadata["model"],
                "arena": "DROID / RoboLab",
                "model_class": "WAM",
                "evidence_kind": evidence_kind,
                "publication_role": publication_role,
                "source_manifest": manifest_path,
            }
        manifests.append(manifest_path)
    return references, manifests


def reference_map() -> tuple[dict[str, dict[str, str]], list[str]]:
    gallery = json.loads(GALLERY_MANIFEST.read_text())
    refs: dict[str, dict[str, str]] = {}
    source_manifests = [str(GALLERY_MANIFEST.relative_to(REPO_ROOT))]

    for entry in gallery["entries"]:
        add_reference(
            refs,
            entry["video"],
            model=entry["model_label"],
            arena=entry["arena_label"],
            model_class=entry["category"],
            evidence_kind="actual rollout",
            publication_role="canonical gallery",
            source_manifest=entry["source_manifest"],
        )

    behavioral_manifests = [
        gallery["dreamzero_manifest_contract"]["path"],
        *[item["path"] for item in gallery["additional_manifest_contracts"]],
    ]
    for manifest_path in behavioral_manifests:
        source_manifests.append(manifest_path)
        manifest = load_json(manifest_path)
        for entry in manifest["gallery_entries"]:
            add_reference(
                refs,
                entry["video"],
                model=entry["model_label"],
                arena=entry["arena_label"],
                model_class=entry["category"],
                evidence_kind="actual rollout",
                publication_role="canonical gallery",
                source_manifest=manifest_path,
            )
            if "comparison_media" in entry:
                add_reference(
                    refs,
                    entry["comparison_media"]["video"],
                    model=entry["model_label"],
                    arena=entry["arena_label"],
                    model_class=entry["category"],
                    evidence_kind="model prediction",
                    publication_role="canonical paired comparison",
                    source_manifest=manifest_path,
                )

    imagination_path = gallery["dreamzero_imagination_manifest_contract"]["path"]
    source_manifests.append(imagination_path)
    imagination = load_json(imagination_path)
    for entry in imagination["gallery_entries"]:
        add_reference(
            refs,
            entry["video"],
            model="DreamZero DROID",
            arena=entry["arena_label"],
            model_class="WAM",
            evidence_kind="model prediction / imagination",
            publication_role="canonical paired comparison",
            source_manifest=imagination_path,
        )
    for entry in imagination["official_decodes"]:
        add_reference(
            refs,
            entry["archived_video"],
            model="DreamZero DROID",
            arena="DROID / RoboLab",
            model_class="WAM",
            evidence_kind="model prediction / imagination",
            publication_role="complete decode archive",
            source_manifest=imagination_path,
        )

    for contract in gallery["prediction_only_manifest_contracts"]:
        manifest_path = contract["path"]
        source_manifests.append(manifest_path)
        manifest = load_json(manifest_path)
        for entry in manifest[contract["entries_key"]]:
            add_reference(
                refs,
                entry[contract["video_field"]],
                model=contract["model_label"],
                arena=contract["arena_label"],
                model_class=contract["category"],
                evidence_kind="prediction-only interface probe",
                publication_role="canonical gallery",
                source_manifest=manifest_path,
            )
    cfg_refs, cfg_manifests = cfg_ablation_references(gallery)
    for path, metadata in cfg_refs.items():
        if path in refs:
            raise ValueError(f"CFG media path duplicates an existing catalog reference: {path}")
        refs[path] = metadata
    source_manifests.extend(cfg_manifests)
    return refs, source_manifests


FOLDER_METADATA = {
    "cosmos3_edge_base_v2a013": ("Cosmos3 Edge base — DROID", "DROID / RoboLab", "WAM"),
    "cosmos3_edge_droid": ("Cosmos3 Edge DROID", "DROID / RoboLab", "WAM"),
    "cosmos3_nano_policy_droid_v2a011": ("Cosmos3 Nano Policy DROID — V2-A011", "DROID / RoboLab", "WAM"),
    "cosmos3_super_base_v2a014": ("Cosmos3-Super base", "DROID / RoboLab conditioning image only", "WAM"),
    "dreamzero_droid": ("DreamZero DROID", "DROID / RoboLab", "WAM"),
    "droid_pi0_fast_pairs": ("π0-FAST DROID — historical frozen-stack reference", "DROID / RoboLab", "VLA"),
    "groot_n17_droid": ("GR00T N1.7 DROID", "DROID / RoboLab", "VLA"),
    "light_wam_robotwin": ("Light-WAM", "RoboTwin place-A-relative-to-B", "WAM"),
    "pi05_current_stack_v2a010": ("π0.5 DROID — current-stack V2-A010", "DROID / RoboLab", "VLA"),
    "robotwin_wam_confirmation": ("RoboTwin WAM confirmation selections", "RoboTwin place-A-relative-to-B", "WAM"),
    "robotwin_wam_pairs": ("RoboTwin WAM pilot selections", "RoboTwin place-A-relative-to-B", "WAM"),
    "media": ("LingBot-VLA 4B", "RoboTwin place-A-relative-to-B", "VLA"),
}


def fallback_metadata(relative: str) -> dict[str, str]:
    parts = Path(relative).parts
    folder = parts[3] if relative.startswith("artifacts/vla_wam_shared_v2/media/") else parts[-2]
    filename = Path(relative).name
    model, arena, model_class = FOLDER_METADATA.get(folder, (folder, "not classified", "not classified"))
    if folder == "robotwin_wam_pairs":
        if filename.startswith("efficient_wam"):
            model = "Efficient-WAM-RT"
        elif filename.startswith("fastwam"):
            model = "FastWAM"
        elif filename.startswith("lingbot_va"):
            model = "LingBot-VA"
    if "reconstructed_head" in filename:
        role = "reconstruction component"
    elif "1200x1200" in filename:
        role = "alternate square encode"
    else:
        role = "supporting archive"
    return {
        "model": model,
        "arena": arena,
        "model_class": model_class,
        "evidence_kind": "actual rollout",
        "publication_role": role,
        "source_manifest": "",
    }


def build_rows() -> tuple[list[dict[str, Any]], list[str]]:
    refs, manifests = reference_map()
    discovered = [*MEDIA_ROOT.rglob("*.mp4"), *PILOT_MEDIA_ROOT.rglob("*.mp4")]
    paths = []
    for path in discovered:
        relative = str(path.relative_to(REPO_ROOT))
        if path.is_relative_to(CFG_MEDIA_ROOT) and relative not in refs:
            continue
        paths.append(path)
    paths = sorted(set(paths))
    rows = []
    for path in paths:
        relative = str(path.relative_to(REPO_ROOT))
        meta = refs.get(relative, fallback_metadata(relative))
        rows.append({
            "model_class": meta["model_class"],
            "model": meta["model"],
            "arena": meta["arena"],
            "evidence_kind": meta["evidence_kind"],
            "publication_role": meta["publication_role"],
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "source_manifest": meta["source_manifest"],
        })
    return rows, sorted(set(manifests))


def write_outputs(rows: list[dict[str, Any]], manifests: list[str]) -> None:
    cfg_rows = [row for row in rows if row["publication_role"].startswith("CFG ablation")]
    catalog_json = {
        "schema_version": "vla-wam-shared-v2-media-catalog-v1",
        "claim_boundary": (
            "Actual rollouts are behavioral evidence. Predictions, imagination, "
            "reconstruction components, and alternate encodes are not additional episodes."
        ),
        "video_count": len(rows),
        "role_counts": dict(sorted(Counter(row["publication_role"] for row in rows).items())),
        "evidence_kind_counts": dict(sorted(Counter(row["evidence_kind"] for row in rows).items())),
        "source_manifests": [
            {
                "path": path,
                "bytes": (REPO_ROOT / path).stat().st_size,
                "sha256": sha256(REPO_ROOT / path),
            }
            for path in manifests
        ],
        "videos": rows,
    }
    if cfg_rows:
        catalog_json.update(
            {
                "cfg_ablation_video_count": len(cfg_rows),
                "cfg_ablation_episode_count_contribution": 0,
            }
        )
    (MEDIA_ROOT / "media_catalog.json").write_text(
        json.dumps(catalog_json, indent=2, sort_keys=True) + "\n"
    )

    fields = [
        "model_class", "model", "arena", "evidence_kind", "publication_role",
        "path", "bytes", "sha256", "source_manifest",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (MEDIA_ROOT / "media_catalog.csv").write_text(output.getvalue())

    grouped: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (row["model_class"], row["model"], row["arena"])
        if row["evidence_kind"] == "actual rollout" and row["publication_role"] == "canonical gallery":
            grouped[key]["canonical execution"] += 1
        elif (
            row["evidence_kind"] != "actual rollout"
            and row["publication_role"] in {"canonical gallery", "canonical paired comparison"}
        ):
            grouped[key]["canonical prediction"] += 1
        else:
            grouped[key]["archive/support"] += 1
    lines = [
        "# VLA/WAM video map",
        "",
        "Start with the [interactive gallery](../../../docs/VLA_WAM_STEERABILITY_VIDEO_GALLERY.html). "
        "This directory is the complete inventory of committed MP4s; the gallery is the curated reading order.",
        "",
        "## How to read the media",
        "",
        "- **Actual rollout** means simulator execution and may support behavioral outcomes.",
        "- **Model prediction / imagination** is generated future evidence, never an executed episode.",
        "- **Prediction-only interface probe** has no controller rollout and no success score.",
        "- **Alternate square encodes and reconstruction components** are presentation/support files, not extra evidence.",
        "",
        "DROID/RoboLab and RoboTwin use different tasks and denominators. Never pool their success rates.",
        "",
        "## Model-level inventory",
        "",
        "| Class | Model | Arena | Canonical execution | Canonical prediction | Archive / support |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for (model_class, model, arena), counts in sorted(grouped.items()):
        lines.append(
            f"| {model_class} | {model} | {arena} | "
            f"{counts['canonical execution']} | {counts['canonical prediction']} | "
            f"{counts['archive/support']} |"
        )
    lines.extend([
        "",
        "## Machine-readable inventory",
        "",
        "- [Complete CSV](media_catalog.csv): one row per committed MP4.",
        "- [Hash-bearing JSON](media_catalog.json): roles, sizes, SHA-256 digests, and source manifests.",
        "- [Gallery manifest](video_first_gallery_manifest.json): canonical publication selections.",
        "",
        f"Catalog total: **{len(rows)} committed MP4s**. A file count is not an episode count.",
        "",
    ])
    (MEDIA_ROOT / "README.md").write_text("\n".join(lines))


def main() -> None:
    rows, manifests = build_rows()
    gallery = json.loads(GALLERY_MANIFEST.read_text())
    contracts = gallery.get("cfg_ablation_media_contracts")
    expected = 36 if contracts is None else 36 + 2 * len(contracts)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} committed catalog items, found {len(rows)}")
    write_outputs(rows, manifests)


if __name__ == "__main__":
    main()
