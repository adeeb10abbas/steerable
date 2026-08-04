#!/usr/bin/env python3
"""Render and validate the video-first VLA/WAM evidence gallery.

The committed gallery manifest is the metadata source. DreamZero execution and
imagination entries are optionally loaded from separate canonical manifests,
and supplementary completed-model manifests are ingested through hash-pinned
contracts. Model-predicted video is rendered in a dedicated section and is
never presented as simulator execution.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "artifacts/vla_wam_shared_v2/media/video_first_gallery_manifest.json"
DEFAULT_HTML = REPO_ROOT / "docs/VLA_WAM_STEERABILITY_VIDEO_GALLERY.html"
DEFAULT_MARKDOWN = REPO_ROOT / "docs/VLA_WAM_STEERABILITY_VIDEO_GALLERY.md"

CFG_MEDIA_SCHEMA = "vla-wam-shared-v2-v2a015-cfg-media-v1"
CFG_MEDIA_STATUS = "complete_all_six_cells_actual_and_prediction_media"
CFG_COMPARISON_SCHEMA = "vla-wam-shared-v2-cfg-ablation-v2a015-comparison-v1"
CFG_SELECTION_POLICY = (
    "No outcome-based or request-based selection. Exact supplied pair manifests "
    "for seeds 8300--8302 are matched against compiled-result provenance; all six "
    "complete viewport videos and all exposed prediction/imagination sources are retained."
)
CFG_COMPARISON_BOUNDARY = (
    "Each comparison is an exact paired, descriptive n=6 post-result pilot. "
    "Improved/regressed/unchanged cell transitions and paired effect sizes are reported without a powered or general performance-gain claim. "
    "Cosmos3 Nano and DreamZero denominators remain separate, and neither is pooled with RoboTwin."
)
CFG_COMPARISON_DISPLAY_BOUNDARY = (
    "Each comparison is exact paired post-result evidence with n=3 per direction "
    "in each setting (six cells per setting). Improved, regressed, and unchanged "
    "cell transitions and paired effect sizes are descriptive; they do not support "
    "a powered or general performance-gain claim. Cosmos3 Nano and DreamZero "
    "denominators remain separate, and neither is pooled with RoboTwin."
)
CFG_SEEDS = (8300, 8301, 8302)
CFG_RELATIONS = ("left", "right")
CFG_PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
CFG_ARM_ORDER = ("cosmos3_nano_no_cfg_g1", "dreamzero_action_cfg_s2")
CFG_ARM_SPECS: dict[str, dict[str, Any]] = {
    "cosmos3_nano_no_cfg_g1": {
        "model_id": "cosmos3_nano_policy_droid",
        "media_manifest_path": "artifacts/vla_wam_shared_v2/media/cfg_v2a015/cosmos3_nano_g1/media_manifest.json",
        "baseline_result_path": "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_direct_gate.json",
        "intervention_result_path": "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_v2a015_no_cfg_g1_result.json",
        "model_label": "Cosmos3 Nano Policy DROID",
        "card_id": "v2a015-cosmos3-nano-g1-complete-media",
        "setting_label": "joint action/video CFG g=1 (CFG blend removed); baseline g=3",
        "comparison_key": "cosmos3_nano",
        "comparison_model": "Cosmos3 Nano Policy DROID",
        "baseline_label": "g=3 baseline",
        "intervention_label": "g=1 intervention",
        "baseline_schema": "vla-wam-shared-v2-cosmos3-nano-policy-droid-result-v1",
        "baseline_model_id": "cosmos3_nano_policy_droid",
        "baseline_amendment_id": "V2-A011",
        "intervention_schema": "vla-wam-shared-v2-cosmos3-nano-v2a015-g1-result-v1",
        "intervention_model_id": "cosmos3_nano_policy_droid",
        "intervention_amendment_id": "V2-A015",
        "baseline_provenance_key": "cosmos3_nano_baseline",
        "intervention_provenance_key": "cosmos3_nano_intervention",
        "actual_filename": "cosmos3_nano_no_cfg_g1_all_seeds_actual.mp4",
        "actual_poster_filename": "cosmos3_nano_no_cfg_g1_all_seeds_actual_poster.jpg",
        "prediction_filename": "cosmos3_nano_no_cfg_g1_all_seeds_local_predictions.mp4",
        "prediction_poster_filename": "cosmos3_nano_no_cfg_g1_all_seeds_local_predictions_poster.jpg",
        "prediction_label": "ALL LOCAL MODEL-PREDICTION HORIZONS — NOT EXECUTION",
        "prediction_note": (
            "Every retained 33-frame request horizon is shown in request order. "
            "The joined review video is not a continuous imagined rollout."
        ),
        "result_summary": (
            "At g=1, total success was 4/6 versus 6/6 at g=3. "
            "The smaller RIGHT-minus-LEFT margin gap is not improved balance: "
            "both requested-side margins fell, and LEFT lost two successes."
        ),
        "future_interface": "Joint actions plus decoded 33-frame RGB local prediction horizons",
        "claim_boundary": (
            "The actual composite contains complete simulator viewport executions. "
            "The prediction composite contains every retained 33-frame RGB future in "
            "request order. Each request is a local model-prediction horizon; joining "
            "them for review does not make a continuous imagined rollout, simulator "
            "execution, task outcome, or additional behavioral episode."
        ),
    },
    "dreamzero_action_cfg_s2": {
        "model_id": "dreamzero_droid_action_cfg",
        "media_manifest_path": "artifacts/vla_wam_shared_v2/media/cfg_v2a015/dreamzero_action_cfg_s2/media_manifest.json",
        "baseline_result_path": "artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_droid_direct_gate.json",
        "intervention_result_path": "artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_v2a015_action_cfg_s2_result.json",
        "model_label": "DreamZero DROID",
        "card_id": "v2a015-dreamzero-action-guidance-s2-complete-media",
        "setting_label": "CFG-style negative-branch action guidance s=2; video CFG=5",
        "display_setting_label": (
            "conditional-action equivalent s=1 → derived negative-branch action "
            "guidance s=2; video CFG fixed at 5"
        ),
        "comparison_key": "dreamzero",
        "comparison_model": "DreamZero DROID",
        "baseline_label": "s=1 conditional-action equivalent",
        "intervention_label": "s=2 CFG-style negative-branch action guidance",
        "baseline_schema": "vla-wam-shared-v2-dreamzero-droid-direct-gate-v1",
        "baseline_model_id": "dreamzero_droid",
        "baseline_amendment_id": "V2-A007",
        "intervention_schema": "vla-wam-shared-v2-dreamzero-v2a015-s2-result-v1",
        "intervention_model_id": "dreamzero_droid_action_cfg",
        "intervention_amendment_id": "V2-A015",
        "baseline_provenance_key": "dreamzero_baseline",
        "intervention_provenance_key": "dreamzero_intervention",
        "actual_filename": "dreamzero_action_cfg_s2_all_seeds_actual.mp4",
        "actual_poster_filename": "dreamzero_action_cfg_s2_all_seeds_actual_poster.jpg",
        "prediction_filename": "dreamzero_action_cfg_s2_all_seeds_imagination.mp4",
        "prediction_poster_filename": "dreamzero_action_cfg_s2_all_seeds_imagination_poster.jpg",
        "prediction_label": (
            "ALL SIX COMPLETE OFFICIAL DECODER OUTPUTS (MODEL IMAGINATIONS) — "
            "NOT EXECUTION"
        ),
        "prediction_note": (
            "Every complete official reset decode exposed by the six behavioral cells is shown. "
            "These are model imaginations, not executed robot trajectories."
        ),
        "result_summary": (
            "The one-cell aggregate gain was not uniform: LEFT lost one success "
            "while RIGHT gained two. Guidance reversed the favored direction "
            "rather than establishing direction-independent robustness."
        ),
        "future_interface": "Joint actions plus latent-video predictions with official decoded imaginations",
        "claim_boundary": (
            "The actual composite contains complete simulator viewport executions. "
            "The imagination composite contains every complete official_reset_decode "
            "listed by each of the six DreamZero behavioral cells. These official "
            "model decodes are not simulator execution, task outcomes, or additional "
            "behavioral episodes. The s=2 arm is derived CFG-style negative-branch "
            "action guidance, not an official DreamZero action-CFG mode."
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_file(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    path.relative_to(REPO_ROOT.resolve())
    return path


def validate_file(record: dict[str, Any], label: str) -> None:
    path = repo_file(record["path"])
    if not path.is_file():
        raise SystemExit(f"missing {label}: {record['path']}")
    actual_bytes = path.stat().st_size
    if actual_bytes != record["bytes"]:
        raise SystemExit(
            f"byte mismatch for {label}: {record['path']} "
            f"expected={record['bytes']} actual={actual_bytes}"
        )
    actual_sha = sha256(path)
    if actual_sha != record["sha256"]:
        raise SystemExit(
            f"SHA-256 mismatch for {label}: {record['path']} "
            f"expected={record['sha256']} actual={actual_sha}"
        )


def validate_file_record_shape(record: Any, label: str) -> None:
    if not isinstance(record, dict):
        raise SystemExit(f"{label} must be a file record")
    if not isinstance(record.get("path"), str) or not record["path"]:
        raise SystemExit(f"{label} has no path")
    if not isinstance(record.get("bytes"), int) or record["bytes"] < 0:
        raise SystemExit(f"{label} has no valid byte count")
    digest = record.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise SystemExit(f"{label} has no valid lowercase SHA-256")


def validate_bound_file(record: Any, label: str) -> Path:
    validate_file_record_shape(record, label)
    validate_file(record, label)
    return repo_file(record["path"])


def records_bind_same_bytes(left: Any, right: Any, label: str) -> None:
    validate_file_record_shape(left, f"{label} left record")
    validate_file_record_shape(right, f"{label} right record")
    if left["bytes"] != right["bytes"] or left["sha256"] != right["sha256"]:
        raise SystemExit(f"{label} file records do not bind the same bytes")


def validate_local_media_output(
    record: Any,
    *,
    media_manifest_path: Path,
    expected_filename: str,
    label: str,
) -> dict[str, Any]:
    validate_file_record_shape(record, label)
    if Path(record["path"]).name != expected_filename:
        raise SystemExit(
            f"{label} filename mismatch: expected={expected_filename} "
            f"actual={Path(record['path']).name}"
        )
    local = (media_manifest_path.parent / expected_filename).resolve()
    local.relative_to(REPO_ROOT.resolve())
    if not local.is_file():
        raise SystemExit(f"missing {label}: {local.relative_to(REPO_ROOT)}")
    if local.stat().st_size != record["bytes"]:
        raise SystemExit(f"byte mismatch for {label}: {local.relative_to(REPO_ROOT)}")
    if sha256(local) != record["sha256"]:
        raise SystemExit(f"SHA-256 mismatch for {label}: {local.relative_to(REPO_ROOT)}")
    return {
        "path": str(local.relative_to(REPO_ROOT)),
        "bytes": record["bytes"],
        "sha256": record["sha256"],
    }


def cfg_requested_margin(row: dict[str, Any], relation: str, label: str) -> float:
    value = row.get("final_lateral_display_m")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SystemExit(f"{label} has no numeric final_lateral_display_m")
    final_lateral = float(value)
    if not math.isfinite(final_lateral):
        raise SystemExit(f"{label} final_lateral_display_m is not finite")
    derived = -final_lateral if relation == "left" else final_lateral
    for key in ("requested_signed_final_margin_m", "requested_signed_final_offset_m"):
        if row.get(key) is None:
            continue
        explicit = row[key]
        if (
            not isinstance(explicit, (int, float))
            or isinstance(explicit, bool)
            or not math.isfinite(float(explicit))
            or not math.isclose(float(explicit), derived, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise SystemExit(f"{label} requested margin disagrees with final endpoint")
    return derived


def load_cfg_result(
    record: Any,
    *,
    schema: str,
    model_id: str,
    amendment_id: str,
    label: str,
) -> tuple[Path, dict[tuple[int, str], dict[str, Any]]]:
    path = validate_bound_file(record, label)
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != schema
        or payload.get("status") != "complete"
        or payload.get("model_id") != model_id
        or payload.get("amendment_id") != amendment_id
    ):
        raise SystemExit(f"{label} identity contract mismatch")
    if "arena" in payload and payload["arena"] != "droid_robolab":
        raise SystemExit(f"{label} is not DROID/RoboLab evidence")
    if "exact_prompts" in payload and payload["exact_prompts"] != CFG_PROMPTS:
        raise SystemExit(f"{label} exact prompt registry changed")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 6:
        raise SystemExit(f"{label} must contain exactly six episodes")
    mapped: dict[tuple[int, str], dict[str, Any]] = {}
    for index, row in enumerate(episodes):
        if not isinstance(row, dict):
            raise SystemExit(f"{label} episode {index} is not an object")
        seed = row.get("environment_seed")
        relation = row.get("requested_relation")
        if type(seed) is not int or seed not in CFG_SEEDS or relation not in CFG_RELATIONS:
            raise SystemExit(f"{label} episode {index} has an invalid seed/relation")
        if row.get("sampling_seed") != seed:
            raise SystemExit(f"{label} episode {index} changed the paired sampling seed")
        if row.get("prompt") != CFG_PROMPTS[relation]:
            raise SystemExit(f"{label} episode {index} changed the exact prompt")
        if type(row.get("requested_success")) is not bool:
            raise SystemExit(f"{label} episode {index} has a non-boolean success")
        key = (seed, relation)
        if key in mapped:
            raise SystemExit(f"{label} duplicates seed/relation {key}")
        mapped[key] = {
            "seed": seed,
            "relation": relation,
            "prompt": CFG_PROMPTS[relation],
            "requested_success": row["requested_success"],
            "requested_margin_m": cfg_requested_margin(
                row, relation, f"{label} episode {index}"
            ),
        }
    expected = {(seed, relation) for seed in CFG_SEEDS for relation in CFG_RELATIONS}
    if set(mapped) != expected:
        raise SystemExit(f"{label} does not contain the exact six-cell grid")
    return path, mapped


def cfg_configuration_summary(
    cells: dict[tuple[int, str], dict[str, Any]]
) -> dict[str, Any]:
    by_direction = {}
    for relation in CFG_RELATIONS:
        rows = [cells[(seed, relation)] for seed in CFG_SEEDS]
        margins = [row["requested_margin_m"] for row in rows]
        by_direction[relation] = {
            "successes": sum(row["requested_success"] for row in rows),
            "trials": 3,
            "mean_requested_margin_m": sum(margins) / 3,
        }
    gap = (
        by_direction["right"]["mean_requested_margin_m"]
        - by_direction["left"]["mean_requested_margin_m"]
    )
    return {
        "by_direction": by_direction,
        "successes": sum(record["successes"] for record in by_direction.values()),
        "trials": 6,
        "signed_direction_gap_m": gap,
        "absolute_direction_imbalance_m": abs(gap),
    }


def validate_cfg_comparison(
    contract: dict[str, Any],
    spec: dict[str, Any],
    baseline_cells: dict[tuple[int, str], dict[str, Any]],
    intervention_cells: dict[tuple[int, str], dict[str, Any]],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    comparison_contract = contract.get("comparison")
    if not isinstance(comparison_contract, dict):
        raise SystemExit(f"CFG contract {spec['model_label']} is missing comparison")
    required = {
        "schema_version": CFG_COMPARISON_SCHEMA,
        "status": "complete",
        "comparison_key": spec["comparison_key"],
    }
    for key, expected in required.items():
        if comparison_contract.get(key) != expected:
            raise SystemExit(
                f"CFG comparison contract mismatch for {spec['model_label']}: {key}"
            )
    comparison_path = validate_bound_file(
        comparison_contract, f"{spec['model_label']} CFG comparison"
    )
    comparison = json.loads(comparison_path.read_text())
    if (
        comparison.get("schema_version") != CFG_COMPARISON_SCHEMA
        or comparison.get("status") != "complete"
        or comparison.get("amendment_id") != "V2-A015"
        or comparison.get("arena") != "droid_robolab"
        or comparison.get("exact_prompts") != CFG_PROMPTS
        or comparison.get("inference_boundary") != CFG_COMPARISON_BOUNDARY
    ):
        raise SystemExit(f"CFG comparison evidence mismatch for {spec['model_label']}")
    comparisons = comparison.get("comparisons")
    if not isinstance(comparisons, dict):
        raise SystemExit("CFG comparison has no comparisons object")
    model = comparisons.get(spec["comparison_key"])
    if (
        not isinstance(model, dict)
        or model.get("model") != spec["comparison_model"]
        or model.get("baseline_label") != spec["baseline_label"]
        or model.get("intervention_label") != spec["intervention_label"]
        or model.get("exact_prompts") != CFG_PROMPTS
    ):
        raise SystemExit(f"CFG comparison model mismatch for {spec['model_label']}")
    comparison_cells = model.get("cells")
    if not isinstance(comparison_cells, list) or len(comparison_cells) != 6:
        raise SystemExit(f"CFG comparison grid mismatch for {spec['model_label']}")
    observed = set()
    for index, row in enumerate(comparison_cells):
        if not isinstance(row, dict):
            raise SystemExit(f"CFG comparison cell {index} is not an object")
        seed = row.get("environment_seed")
        relation = row.get("requested_relation")
        key = (seed, relation)
        if (
            type(seed) is not int
            or seed not in CFG_SEEDS
            or relation not in CFG_RELATIONS
            or key in observed
            or row.get("cell_id") != f"seed{seed}_{relation}"
            or row.get("prompt") != CFG_PROMPTS[relation]
        ):
            raise SystemExit(f"CFG comparison cell grid changed for {spec['model_label']}")
        observed.add(key)
        success = row.get("success")
        margins = row.get("requested_signed_final_margin_m")
        if not isinstance(success, dict) or not isinstance(margins, dict):
            raise SystemExit(f"CFG comparison cell evidence missing for {spec['model_label']}")
        expected_values = (
            (
                spec["baseline_label"],
                baseline_cells[key]["requested_success"],
                baseline_cells[key]["requested_margin_m"],
            ),
            (
                spec["intervention_label"],
                intervention_cells[key]["requested_success"],
                intervention_cells[key]["requested_margin_m"],
            ),
        )
        for setting_label, expected_success, expected_margin in expected_values:
            observed_margin = margins.get(setting_label)
            if (
                success.get(setting_label) is not expected_success
                or not isinstance(observed_margin, (int, float))
                or isinstance(observed_margin, bool)
                or not math.isclose(
                    float(observed_margin), expected_margin, rel_tol=0.0, abs_tol=1e-9
                )
            ):
                raise SystemExit(
                    f"CFG comparison/source mismatch for {spec['model_label']} {seed}/{relation}"
                )
    expected_grid = {(seed, relation) for seed in CFG_SEEDS for relation in CFG_RELATIONS}
    if observed != expected_grid:
        raise SystemExit(f"CFG comparison grid incomplete for {spec['model_label']}")
    baseline_summary = cfg_configuration_summary(baseline_cells)
    intervention_summary = cfg_configuration_summary(intervention_cells)
    success = model.get("success")
    if (
        not isinstance(success, dict)
        or success.get("baseline_total") != baseline_summary["successes"]
        or success.get("intervention_total") != intervention_summary["successes"]
        or success.get("net_success_change")
        != intervention_summary["successes"] - baseline_summary["successes"]
    ):
        raise SystemExit(f"CFG comparison success summary changed for {spec['model_label']}")
    for key, cells, expected_summary in (
        ("baseline_configuration_summary", baseline_cells, baseline_summary),
        ("intervention_configuration_summary", intervention_cells, intervention_summary),
    ):
        summary = model.get(key)
        if not isinstance(summary, dict) or summary.get("valid_episode_count") != 6:
            raise SystemExit(f"CFG comparison {key} changed for {spec['model_label']}")
        by_direction = summary.get("by_direction")
        balance = summary.get("mean_margin_balance")
        if not isinstance(by_direction, dict) or not isinstance(balance, dict):
            raise SystemExit(f"CFG comparison {key} metrics missing for {spec['model_label']}")
        for relation in CFG_RELATIONS:
            record = by_direction.get(relation)
            expected = expected_summary["by_direction"][relation]
            requested_margin = record.get("requested_margin_m") if isinstance(record, dict) else None
            if (
                not isinstance(record, dict)
                or record.get("prompt") != CFG_PROMPTS[relation]
                or record.get("episodes") != 3
                or record.get("successes") != expected["successes"]
                or not isinstance(requested_margin, dict)
                or not isinstance(requested_margin.get("mean"), (int, float))
                or not math.isclose(
                    float(requested_margin["mean"]),
                    expected["mean_requested_margin_m"],
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise SystemExit(
                    f"CFG comparison {key}/{relation} disagrees for {spec['model_label']}"
                )
        if (
            not isinstance(balance.get("right_minus_left_m"), (int, float))
            or not isinstance(balance.get("absolute_direction_imbalance_m"), (int, float))
            or not math.isclose(
                float(balance["right_minus_left_m"]),
                expected_summary["signed_direction_gap_m"],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(balance["absolute_direction_imbalance_m"]),
                expected_summary["absolute_direction_imbalance_m"],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise SystemExit(f"CFG comparison {key} direction balance disagrees")
    return comparison_path, baseline_summary, intervention_summary


def validate_cfg_output_probe(record: Any, label: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise SystemExit(f"{label} has no output-validation record")
    if (
        record.get("codec_name") != "h264"
        or record.get("pixel_format") != "yuv420p"
        or record.get("width") != 1280
        or record.get("height") != 480
    ):
        raise SystemExit(f"{label} is not the required H.264/yuv420p 1280x480 media")
    for key in ("fps", "duration_s", "frame_count"):
        value = record.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise SystemExit(f"{label} has an invalid {key}")
    indices = record.get("decoded_frame_indices")
    samples = record.get("decoded_frame_samples")
    if (
        not isinstance(indices, list)
        or len(indices) != 3
        or not isinstance(samples, list)
        or len(samples) != 3
        or [sample.get("frame_index") for sample in samples if isinstance(sample, dict)]
        != indices
    ):
        raise SystemExit(f"{label} lacks first/middle/last decoded-frame validation")
    for index, sample in enumerate(samples):
        digest = sample.get("decoded_bgr_sha256") if isinstance(sample, dict) else None
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise SystemExit(f"{label} decoded-frame sample {index} has no valid SHA-256")
    return {
        "duration_s": float(record["duration_s"]),
        "frame_count": int(record["frame_count"]),
        "fps": float(record["fps"]),
    }


def load_cfg_ablation_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Load optional V2-A015 media with transitive, fail-closed provenance checks."""

    if "cfg_ablation_media_contracts" not in manifest:
        return []
    contracts = manifest["cfg_ablation_media_contracts"]
    if not isinstance(contracts, list) or len(contracts) != len(CFG_ARM_ORDER):
        raise SystemExit("cfg_ablation_media_contracts must contain exactly the two V2-A015 arms")
    by_arm: dict[str, dict[str, Any]] = {}
    for index, contract in enumerate(contracts):
        if not isinstance(contract, dict):
            raise SystemExit(f"CFG media contract {index} is not an object")
        arm_id = contract.get("arm_id")
        if arm_id not in CFG_ARM_SPECS or arm_id in by_arm:
            raise SystemExit(f"CFG media contract has an invalid or duplicate arm_id: {arm_id}")
        by_arm[arm_id] = contract
    if set(by_arm) != set(CFG_ARM_ORDER):
        raise SystemExit("CFG media contracts do not name the exact two V2-A015 arms")

    entries: list[dict[str, Any]] = []
    for arm_id in CFG_ARM_ORDER:
        contract = by_arm[arm_id]
        spec = CFG_ARM_SPECS[arm_id]
        expected_contract = {
            "path": spec["media_manifest_path"],
            "schema_version": CFG_MEDIA_SCHEMA,
            "status": CFG_MEDIA_STATUS,
            "amendment_id": "V2-A015",
            "arm_id": arm_id,
            "model_id": spec["model_id"],
        }
        for key, expected in expected_contract.items():
            if contract.get(key) != expected:
                raise SystemExit(f"CFG media contract {arm_id} changed {key}")
        media_manifest_path = validate_bound_file(contract, f"{arm_id} media manifest")
        media = json.loads(media_manifest_path.read_text())
        for key, expected in expected_contract.items():
            if key != "path" and media.get(key) != expected:
                raise SystemExit(f"CFG media manifest {arm_id} changed {key}")
        if (
            media.get("setting_label") != spec["setting_label"]
            or media.get("exact_prompts") != CFG_PROMPTS
            or media.get("selection_policy") != CFG_SELECTION_POLICY
            or media.get("claim_boundary") != spec["claim_boundary"]
        ):
            raise SystemExit(f"CFG media scientific contract changed for {arm_id}")

        baseline_contract = contract.get("baseline_result")
        intervention_contract = contract.get("intervention_result")
        if not isinstance(baseline_contract, dict) or not isinstance(intervention_contract, dict):
            raise SystemExit(f"CFG media contract {arm_id} lacks source-result records")
        if baseline_contract.get("path") != spec["baseline_result_path"]:
            raise SystemExit(f"CFG media contract {arm_id} changed the baseline result path")
        if intervention_contract.get("path") != spec["intervention_result_path"]:
            raise SystemExit(f"CFG media contract {arm_id} changed the intervention result path")
        baseline_path, baseline_cells = load_cfg_result(
            baseline_contract,
            schema=spec["baseline_schema"],
            model_id=spec["baseline_model_id"],
            amendment_id=spec["baseline_amendment_id"],
            label=f"{arm_id} baseline result",
        )
        intervention_path, intervention_cells = load_cfg_result(
            intervention_contract,
            schema=spec["intervention_schema"],
            model_id=spec["intervention_model_id"],
            amendment_id=spec["intervention_amendment_id"],
            label=f"{arm_id} intervention result",
        )
        comparison_path, baseline_summary, intervention_summary = validate_cfg_comparison(
            contract, spec, baseline_cells, intervention_cells
        )
        comparison = json.loads(comparison_path.read_text())
        provenance = comparison.get("provenance")
        if not isinstance(provenance, dict):
            raise SystemExit(f"CFG comparison provenance is missing for {arm_id}")
        records_bind_same_bytes(
            baseline_contract,
            provenance.get(spec["baseline_provenance_key"]),
            f"{arm_id} baseline/comparison provenance",
        )
        records_bind_same_bytes(
            intervention_contract,
            provenance.get(spec["intervention_provenance_key"]),
            f"{arm_id} intervention/comparison provenance",
        )
        records_bind_same_bytes(
            intervention_contract,
            media.get("source_result"),
            f"{arm_id} intervention/media provenance",
        )

        cells = media.get("input_cells")
        counts = media.get("request_or_decode_counts")
        if not isinstance(cells, list) or len(cells) != 6 or not isinstance(counts, dict):
            raise SystemExit(f"CFG media {arm_id} does not retain the exact six cells")
        observed: set[tuple[int, str]] = set()
        expected_count_keys = set()
        for index, row in enumerate(cells):
            if not isinstance(row, dict):
                raise SystemExit(f"CFG media {arm_id} cell {index} is not an object")
            seed = row.get("environment_seed")
            relation = row.get("relation")
            key = (seed, relation)
            count_key = f"seed{seed}_{relation}"
            if (
                type(seed) is not int
                or seed not in CFG_SEEDS
                or relation not in CFG_RELATIONS
                or key in observed
                or row.get("prompt") != CFG_PROMPTS[relation]
                or row.get("requested_success")
                is not intervention_cells[key]["requested_success"]
            ):
                raise SystemExit(f"CFG media {arm_id} cell grid/evidence changed")
            observed.add(key)
            expected_count_keys.add(count_key)
            validate_file_record_shape(
                row.get("complete_viewport_video"), f"{arm_id} {count_key} viewport source"
            )
            sources = row.get("prediction_sources_in_order")
            source_count = row.get("prediction_source_count")
            shapes = row.get("prediction_shapes")
            if (
                type(source_count) is not int
                or source_count <= 0
                or not isinstance(sources, list)
                or len(sources) != source_count
                or counts.get(count_key) != source_count
            ):
                raise SystemExit(f"CFG media {arm_id} {count_key} prediction count changed")
            for source_index, source in enumerate(sources):
                validate_file_record_shape(
                    source, f"{arm_id} {count_key} prediction source {source_index}"
                )
            if arm_id == "dreamzero_action_cfg_s2":
                if source_count != 1 or shapes != []:
                    raise SystemExit(f"DreamZero {count_key} must retain one full official decode")
            elif (
                not isinstance(shapes, list)
                or len(shapes) != source_count
                or any(shape != [33, 528, 640, 3] for shape in shapes)
            ):
                raise SystemExit(f"Cosmos3 Nano {count_key} local-horizon shapes changed")
        expected_grid = {(seed, relation) for seed in CFG_SEEDS for relation in CFG_RELATIONS}
        if observed != expected_grid or set(counts) != expected_count_keys:
            raise SystemExit(f"CFG media {arm_id} six-cell/count registry changed")

        outputs = media.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != {
            "actual_video",
            "actual_poster",
            "prediction_or_imagination_video",
            "prediction_or_imagination_poster",
        }:
            raise SystemExit(f"CFG media {arm_id} output registry changed")
        actual_video = validate_local_media_output(
            outputs["actual_video"], media_manifest_path=media_manifest_path,
            expected_filename=spec["actual_filename"], label=f"{arm_id} complete actual video",
        )
        actual_poster = validate_local_media_output(
            outputs["actual_poster"], media_manifest_path=media_manifest_path,
            expected_filename=spec["actual_poster_filename"], label=f"{arm_id} actual poster",
        )
        prediction_video = validate_local_media_output(
            outputs["prediction_or_imagination_video"], media_manifest_path=media_manifest_path,
            expected_filename=spec["prediction_filename"],
            label=f"{arm_id} complete prediction/imagination video",
        )
        prediction_poster = validate_local_media_output(
            outputs["prediction_or_imagination_poster"], media_manifest_path=media_manifest_path,
            expected_filename=spec["prediction_poster_filename"],
            label=f"{arm_id} prediction/imagination poster",
        )
        validation = media.get("output_validation")
        if not isinstance(validation, dict):
            raise SystemExit(f"CFG media {arm_id} lacks output validation")
        actual_probe = validate_cfg_output_probe(
            validation.get("actual"), f"{arm_id} complete actual video"
        )
        prediction_probe = validate_cfg_output_probe(
            validation.get("prediction_or_imagination"),
            f"{arm_id} complete prediction/imagination video",
        )
        entries.append({
            "id": spec["card_id"], "arm_id": arm_id,
            "model_label": spec["model_label"], "setting_label": spec["setting_label"],
            "display_setting_label": spec.get("display_setting_label", spec["setting_label"]),
            "future_interface": spec["future_interface"],
            "prediction_label": spec["prediction_label"],
            "prediction_note": spec["prediction_note"],
            "result_summary": spec["result_summary"],
            "claim_boundary": spec["claim_boundary"],
            "comparison_boundary": CFG_COMPARISON_DISPLAY_BOUNDARY,
            "prompts": dict(CFG_PROMPTS),
            "actual_video": actual_video, "actual_poster": actual_poster,
            "prediction_video": prediction_video, "prediction_poster": prediction_poster,
            "actual_probe": actual_probe, "prediction_probe": prediction_probe,
            "baseline": baseline_summary, "intervention": intervention_summary,
            "media_manifest": str(media_manifest_path.relative_to(REPO_ROOT)),
            "baseline_result": str(baseline_path.relative_to(REPO_ROOT)),
            "intervention_result": str(intervention_path.relative_to(REPO_ROOT)),
            "comparison": str(comparison_path.relative_to(REPO_ROOT)),
        })
    return entries


def validate_entry(entry: dict[str, Any], required_fields: list[str] | None = None) -> None:
    required = required_fields or [
        "id",
        "arena",
        "arena_label",
        "model_label",
        "category",
        "future_interface",
        "evidence_status",
        "pair_label",
        "seed",
        "video",
        "directions",
        "source_manifest",
    ]
    missing = [field for field in required if field not in entry]
    if missing:
        raise SystemExit(f"gallery entry {entry.get('id', '<unknown>')} missing: {missing}")
    if entry["arena"] not in {"droid", "robotwin"}:
        raise SystemExit(f"invalid arena on {entry['id']}: {entry['arena']}")
    if len(entry["directions"]) != 2:
        raise SystemExit(f"gallery entry {entry['id']} must have exactly two directions")
    for direction in entry["directions"]:
        missing_direction = [field for field in ("relation", "prompt", "outcome") if field not in direction]
        if missing_direction:
            raise SystemExit(
                f"gallery entry {entry['id']} has direction missing: {missing_direction}"
            )
    relations = {direction.get("relation") for direction in entry["directions"]}
    if relations != {"LEFT", "RIGHT"}:
        raise SystemExit(f"gallery entry {entry['id']} must contain LEFT and RIGHT")
    validate_file(entry["video"], f"{entry['id']} video")
    for optional in ("poster", "captions"):
        if optional in entry:
            validate_file(entry[optional], f"{entry['id']} {optional}")
    comparison = entry.get("comparison_media")
    if comparison is not None:
        if comparison.get("kind") != "model_prediction_not_execution":
            raise SystemExit(f"gallery entry {entry['id']} has an invalid comparison-media kind")
        if not comparison.get("label") or not comparison.get("note"):
            raise SystemExit(f"gallery entry {entry['id']} has incomplete comparison-media labels")
        validate_file(comparison["video"], f"{entry['id']} model prediction")
    source = repo_file(entry["source_manifest"])
    if not source.is_file():
        raise SystemExit(f"missing source manifest for {entry['id']}: {entry['source_manifest']}")


def load_prediction_only_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for contract in manifest.get("prediction_only_manifest_contracts", []):
        source_path = repo_file(contract["path"])
        if not source_path.is_file():
            raise SystemExit(f"missing prediction-only media manifest: {contract['path']}")
        if sha256(source_path) != contract["sha256"]:
            raise SystemExit(f"SHA-256 mismatch for prediction-only media manifest: {contract['path']}")
        source = json.loads(source_path.read_text())
        if (
            source.get("schema_version") != contract["schema_version"]
            or source.get("status") != contract["status"]
            or source.get("model_id") != contract["model_id"]
        ):
            raise SystemExit(f"prediction-only media manifest contract mismatch: {contract['path']}")
        records = source.get(contract["entries_key"])
        if not isinstance(records, list) or len(records) != contract["entry_count"]:
            raise SystemExit(f"prediction-only media entry count mismatch: {contract['path']}")
        for record in records:
            if (
                record.get("type") != contract["record_type"]
                or record.get("actual_rollout") is not None
                or record.get("actual_rollout_unavailable_reason")
                != contract["actual_rollout_unavailable_reason"]
            ):
                raise SystemExit(f"prediction-only execution boundary mismatch: {contract['path']}")
            video = record.get(contract["video_field"])
            if not isinstance(video, dict):
                raise SystemExit(f"prediction-only video field missing: {contract['path']}")
            if contract.get("requires_unexecuted_actions") and not isinstance(
                record.get("action_trajectories"), dict
            ):
                raise SystemExit(f"prediction-only unexecuted actions missing: {contract['path']}")
            supporting_evidence = contract.get("supporting_evidence", [])
            for evidence in supporting_evidence:
                evidence_path = repo_file(evidence["path"])
                if not evidence_path.is_file() or sha256(evidence_path) != evidence["sha256"]:
                    raise SystemExit(f"prediction-only supporting evidence mismatch: {evidence['path']}")
            entry = {
                "id": contract["entry_id"],
                "arena": contract["arena"],
                "arena_label": contract["arena_label"],
                "model_label": contract["model_label"],
                "category": contract["category"],
                "future_interface": contract["future_interface"],
                "evidence_status": contract["evidence_status"],
                "pair_label": contract["pair_label"],
                "seed": record["sampling_seed"],
                "video": video,
                "poster": record["poster"],
                "directions": contract["directions"],
                "source_manifest": contract["path"],
                "actual_rollout_unavailable_reason": contract["actual_rollout_unavailable_reason"],
                "actual_rollout_unavailable_detail": contract["actual_rollout_unavailable_detail"],
                "prediction_media_label": contract["prediction_media_label"],
                "supporting_evidence": supporting_evidence,
            }
            validate_entry(entry)
            entries.append(entry)
    return entries


def load_entries(
    manifest: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    bool,
    list[dict[str, Any]],
    list[dict[str, Any]],
    bool,
    list[dict[str, Any]],
]:
    entries = list(manifest["entries"])
    for entry in entries:
        validate_entry(entry)

    contract = manifest["dreamzero_manifest_contract"]
    dreamzero_path = repo_file(contract["path"])
    dreamzero_present = dreamzero_path.is_file()
    if dreamzero_present:
        dreamzero = json.loads(dreamzero_path.read_text())
        key = contract["gallery_entries_key"]
        dreamzero_entries = dreamzero.get(key)
        if not isinstance(dreamzero_entries, list) or not dreamzero_entries:
            raise SystemExit(
                f"DreamZero manifest exists but has no non-empty {key!r}: {contract['path']}"
            )
        for entry in dreamzero_entries:
            validate_entry(entry, contract["required_entry_fields"])
            if entry["arena"] != "droid" or "dreamzero" not in entry["id"].lower():
                raise SystemExit(f"non-DreamZero entry in canonical DreamZero manifest: {entry['id']}")
        entries = dreamzero_entries + entries

    for contract in manifest.get("additional_manifest_contracts", []):
        supplemental_path = repo_file(contract["path"])
        if not supplemental_path.is_file():
            raise SystemExit(f"missing supplementary media manifest: {contract['path']}")
        if contract.get("sha256") and sha256(supplemental_path) != contract["sha256"]:
            raise SystemExit(f"SHA-256 mismatch for supplementary media manifest: {contract['path']}")
        supplemental = json.loads(supplemental_path.read_text())
        if supplemental.get("status") != contract["status"]:
            raise SystemExit(f"supplementary media manifest status mismatch: {contract['path']}")
        supplemental_entries = supplemental.get(contract["gallery_entries_key"])
        if not isinstance(supplemental_entries, list) or len(supplemental_entries) != contract["entry_count"]:
            raise SystemExit(f"supplementary media manifest entry count mismatch: {contract['path']}")
        for entry in supplemental_entries:
            validate_entry(entry, contract.get("required_entry_fields"))
            if entry["id"] != contract["entry_id"] or entry["arena"] != contract["arena"]:
                raise SystemExit(f"supplementary media entry identity mismatch: {entry.get('id')}")
        entries = supplemental_entries + entries
    ids = [entry["id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate gallery entry id")

    imagination_entries: list[dict[str, Any]] = []
    official_decodes: list[dict[str, Any]] = []
    imagination_contract = manifest["dreamzero_imagination_manifest_contract"]
    imagination_path = repo_file(imagination_contract["path"])
    imagination_present = imagination_path.is_file()
    if imagination_present:
        imagination = json.loads(imagination_path.read_text())
        if (
            imagination.get("schema_version")
            != "vla-wam-shared-v2-dreamzero-imagination-media-v1"
            or imagination.get("status") != "complete_all_official_decodes_archived"
            or imagination.get("official_decode_count")
            != imagination_contract["required_official_decode_count"]
            or imagination.get("behavioral_decode_count")
            != imagination_contract["required_behavioral_decode_count"]
            or imagination.get("fixed_observation_probe_decode_count")
            != imagination_contract["required_fixed_observation_probe_decode_count"]
        ):
            raise SystemExit("DreamZero imagination manifest contract mismatch")
        imagination_entries = imagination.get(imagination_contract["gallery_entries_key"], [])
        official_decodes = imagination.get(imagination_contract["archive_entries_key"], [])
        if len(imagination_entries) != 3 or len(official_decodes) != 9:
            raise SystemExit("DreamZero imagination manifest has incomplete media lists")
        for entry in imagination_entries:
            validate_entry(entry)
            if entry.get("media_kind") != "model_prediction_not_execution":
                raise SystemExit(f"DreamZero imagination entry lacks prediction label: {entry['id']}")
        for record in official_decodes:
            if record.get("scope") not in {
                "valid_behavioral_episode",
                "fixed_observation_diagnostic",
            }:
                raise SystemExit(f"Invalid DreamZero decode scope: {record.get('id')}")
            validate_file(record["archived_video"], f"{record['id']} archived official decode")
        all_ids = ids + [entry["id"] for entry in imagination_entries]
        if len(all_ids) != len(set(all_ids)):
            raise SystemExit("duplicate execution/imagination gallery entry id")
    prediction_only_entries = load_prediction_only_entries(manifest)
    all_ids = ids + [entry["id"] for entry in prediction_only_entries]
    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("duplicate execution/prediction gallery entry id")
    return (
        entries,
        dreamzero_present,
        imagination_entries,
        official_decodes,
        imagination_present,
        prediction_only_entries,
    )


def rel(path: str) -> str:
    return "../" + path


def direction_card(direction: dict[str, Any]) -> str:
    relation = html.escape(direction["relation"])
    css = direction["relation"].lower()
    return (
        f'<div class="direction {css}"><div class="direction-top">'
        f'<strong>{relation}</strong><span>{html.escape(direction["outcome"])}</span></div>'
        f'<blockquote>“{html.escape(direction["prompt"])}”</blockquote></div>'
    )


def entry_card(entry: dict[str, Any]) -> str:
    video = rel(entry["video"]["path"])
    poster_path = rel(entry["poster"]["path"]) if "poster" in entry else ""
    poster = f' poster="{html.escape(poster_path)}"' if poster_path else ""
    captions = ""
    if "captions" in entry:
        captions_path = html.escape(rel(entry["captions"]["path"]))
        captions = (
            f'<track kind="captions" src="{captions_path}" '
            'srclang="en" label="English">'
        )
    notes = html.escape(entry.get("selection_note", ""))
    paired_control = entry.get("paired_control")
    paired_link = ""
    if paired_control:
        paired_link = (
            f' · <a href="{html.escape(paired_control["href"])}">'
            f'{html.escape(paired_control["label"])}</a>'
        )
    comparison = entry.get("comparison_media")
    if comparison:
        predicted = rel(comparison["video"]["path"])
        media_block = f'''<div class="comparison-media"><figure><figcaption>ACTUAL SIMULATOR EXECUTION</figcaption>
        <video controls preload="metadata"{poster}><source src="{html.escape(video)}" type="video/mp4"><a href="{html.escape(video)}">Open the actual MP4 directly</a>.</video></figure>
        <figure><figcaption>{html.escape(comparison["label"])}</figcaption><video controls preload="metadata"><source src="{html.escape(predicted)}" type="video/mp4"><a href="{html.escape(predicted)}">Open the model prediction MP4 directly</a>.</video>
        <p>{html.escape(comparison["note"])}</p></figure></div>'''
    else:
        media_block = f'''<video controls preload="metadata"{poster}>
          <source src="{html.escape(video)}" type="video/mp4">{captions}
          <a href="{html.escape(video)}">Open the MP4 directly</a>.
        </video>'''
    return f"""
      <article class="card" id="{html.escape(entry['id'])}">
        <header>
          <div><p class="overline">{html.escape(entry['category'])} · {html.escape(entry['pair_label'])}</p>
          <h3>{html.escape(entry['model_label'])}</h3></div>
          <span class="status">{html.escape(entry['evidence_status'])}</span>
        </header>
        {media_block}
        <div class="directions">{''.join(direction_card(d) for d in entry['directions'])}</div>
        <dl class="facts">
          <div><dt>Arena</dt><dd>{html.escape(entry['arena_label'])}</dd></div>
          <div><dt>Future interface</dt><dd>{html.escape(entry['future_interface'])}</dd></div>
          <div><dt>Video SHA-256</dt><dd><code>{html.escape(entry['video']['sha256'])}</code></dd></div>
        </dl>
        <p class="note">{notes} <a href="{html.escape(video)}">Open video</a> · <a href="{html.escape(rel(entry['source_manifest']))}">Evidence manifest</a>{paired_link}</p>
      </article>"""


def prediction_only_card(entry: dict[str, Any]) -> str:
    video = rel(entry["video"]["path"])
    poster = html.escape(rel(entry["poster"]["path"]))
    evidence_links = " · ".join(
        f'<a href="{html.escape(rel(evidence["path"]))}">{html.escape(evidence["label"])}</a>'
        for evidence in entry["supporting_evidence"]
    )
    return f"""
      <article class="card" id="{html.escape(entry['id'])}">
        <header>
          <div><p class="overline">{html.escape(entry['category'])} · PREDICTION ONLY · {html.escape(entry['pair_label'])}</p>
          <h3>{html.escape(entry['model_label'])}</h3></div>
          <span class="status">{html.escape(entry['evidence_status'])}</span>
        </header>
        <div class="comparison-media"><figure class="comparison-unavailable"><figcaption>ACTUAL SIMULATOR ROLLOUT — UNAVAILABLE</figcaption>
        <p><strong>{html.escape(entry['actual_rollout_unavailable_reason'])}</strong></p>
        <p>{html.escape(entry['actual_rollout_unavailable_detail'])} {evidence_links}</p></figure>
        <figure><figcaption>{html.escape(entry['prediction_media_label'])}</figcaption><video controls preload="metadata" poster="{poster}"><source src="{html.escape(video)}" type="video/mp4"><a href="{html.escape(video)}">Open the model-prediction MP4 directly</a>.</video></figure></div>
        <div class="directions">{''.join(direction_card(d) for d in entry['directions'])}</div>
        <dl class="facts">
          <div><dt>Arena</dt><dd>{html.escape(entry['arena_label'])}</dd></div>
          <div><dt>Future interface</dt><dd>{html.escape(entry['future_interface'])}</dd></div>
          <div><dt>Prediction SHA-256</dt><dd><code>{html.escape(entry['video']['sha256'])}</code></dd></div>
        </dl>
        <p class="note">This prediction-only evidence is not an executed episode. <a href="{html.escape(video)}">Open prediction</a> · <a href="{poster}">Open poster</a> · <a href="{html.escape(rel(entry['source_manifest']))}">Evidence manifest</a>{(' · ' + evidence_links) if evidence_links else ''}</p>
      </article>"""


def format_cfg_margin(value: float) -> str:
    return f"{value:+.3f} m"


def cfg_direction_card(entry: dict[str, Any], relation: str) -> str:
    baseline = entry["baseline"]["by_direction"][relation]
    intervention = entry["intervention"]["by_direction"][relation]
    return f"""
        <div class="cfg-direction {html.escape(relation)}">
          <p class="cfg-direction-name">Prompt asks {html.escape(relation.upper())}</p>
          <blockquote>“{html.escape(entry['prompts'][relation])}”</blockquote>
          <div class="cfg-direction-stats">
            <p><span>Requested-task success</span><strong>{baseline['successes']}/3 → {intervention['successes']}/3</strong></p>
            <p><span>Mean signed endpoint margin</span><strong>{format_cfg_margin(baseline['mean_requested_margin_m'])} → {format_cfg_margin(intervention['mean_requested_margin_m'])}</strong></p>
          </div>
        </div>"""


def cfg_ablation_card(entry: dict[str, Any]) -> str:
    actual_video = html.escape(rel(entry["actual_video"]["path"]))
    actual_poster = html.escape(rel(entry["actual_poster"]["path"]))
    prediction_video = html.escape(rel(entry["prediction_video"]["path"]))
    prediction_poster = html.escape(rel(entry["prediction_poster"]["path"]))
    baseline = entry["baseline"]
    intervention = entry["intervention"]
    display_setting_label = entry.get("display_setting_label", entry["setting_label"])
    is_imagination = entry["arm_id"] == "dreamzero_action_cfg_s2"
    prediction_fallback = (
        "Open all model imaginations" if is_imagination else "Open all model predictions"
    )
    prediction_link_label = (
        "Open complete imaginations" if is_imagination else "Open complete predictions"
    )
    return f"""
      <article class="cfg-card" id="{html.escape(entry['id'])}">
        <header>
          <div><p class="overline">V2-A015 · paired post-result guidance ablation · n=3 per direction per setting</p>
          <h3>{html.escape(entry['model_label'])}</h3>
          <p class="cfg-setting">{html.escape(display_setting_label)}</p></div>
          <span class="status">DROID/RoboLab · seeds 8300–8302</span>
        </header>
        <div class="cfg-media">
          <figure><figcaption>ALL SIX COMPLETE ACTUAL EXECUTIONS</figcaption>
            <video controls preload="metadata" poster="{actual_poster}"><source src="{actual_video}" type="video/mp4"><a href="{actual_video}">Open all actual executions</a>.</video>
            <p>{entry['actual_probe']['duration_s']:.1f} s · no outcome-based selection</p></figure>
          <figure><figcaption>{html.escape(entry['prediction_label'])}</figcaption>
            <video controls preload="metadata" poster="{prediction_poster}"><source src="{prediction_video}" type="video/mp4"><a href="{prediction_video}">{prediction_fallback}</a>.</video>
            <p>{entry['prediction_probe']['duration_s']:.1f} s · {html.escape(entry['prediction_note'])}</p></figure>
        </div>
        <div class="cfg-prompts">{''.join(cfg_direction_card(entry, relation) for relation in CFG_RELATIONS)}</div>
        <div class="cfg-metrics">
          <div><span>Total requested-task success</span><strong>{baseline['successes']}/6 → {intervention['successes']}/6</strong></div>
          <div><span>Mean-margin gap, RIGHT − LEFT</span><strong>{format_cfg_margin(baseline['signed_direction_gap_m'])} → {format_cfg_margin(intervention['signed_direction_gap_m'])}</strong></div>
          <div><span>Absolute mean-margin gap</span><strong>{baseline['absolute_direction_imbalance_m']:.3f} m → {intervention['absolute_direction_imbalance_m']:.3f} m</strong></div>
        </div>
        <p class="cfg-result"><strong>Interpretation.</strong> {html.escape(entry['result_summary'])}</p>
        <p class="cfg-definition"><strong>How to read the margin.</strong> Positive means the final cube endpoint lies on the requested side of the bowl; larger is farther into that side. The task-success checker is stricter than endpoint sign, so margin and success are reported separately.</p>
        <div class="cfg-boundaries"><p><strong>Media boundary.</strong> {html.escape(entry['claim_boundary'])}</p><p><strong>Inference boundary.</strong> {html.escape(entry['comparison_boundary'])}</p></div>
        <p class="note"><a href="{actual_video}">Open complete executions</a> · <a href="{prediction_video}">{prediction_link_label}</a> · <a href="{html.escape(rel(entry['media_manifest']))}">Media manifest</a> · <a href="{html.escape(rel(entry['comparison']))}">Paired comparison</a></p>
      </article>"""


def missing_card(item: dict[str, Any]) -> str:
    source = item.get("expected_manifest") or item.get("behavioral_manifest")
    source_link = (
        f' <a href="{html.escape(rel(source))}">Expected/source manifest</a>.'
        if source and repo_file(source).exists()
        else f" Expected manifest: <code>{html.escape(source or 'none')}</code>."
    )
    return (
        '<article class="missing"><p class="overline">No substituted media</p>'
        f'<h3>{html.escape(item["model_id"])}</h3>'
        f'<p><strong>{html.escape(item["status"])}</strong> — {html.escape(item["reason"])}'
        f'{source_link}</p></article>'
    )


def imagination_archive(official_decodes: list[dict[str, Any]]) -> str:
    items = []
    for record in official_decodes:
        video = html.escape(rel(record["archived_video"]["path"]))
        scope = (
            "behavioral session"
            if record["scope"] == "valid_behavioral_episode"
            else "fixed-observation probe"
        )
        label = record["id"].replace("_", " ")
        items.append(
            f'<li><a href="{video}">{html.escape(label)}</a> '
            f'<span>({scope}; SHA-256 <code>{html.escape(record["archived_video"]["sha256"])}</code>)</span></li>'
        )
    return "<ul class=\"archive\">" + "".join(items) + "</ul>"


def render_html(
    manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    dreamzero_present: bool,
    imagination_entries: list[dict[str, Any]],
    official_decodes: list[dict[str, Any]],
    imagination_present: bool,
    prediction_only_entries: list[dict[str, Any]],
    cfg_ablation_entries: list[dict[str, Any]] | None = None,
) -> str:
    cfg_ablation_entries = cfg_ablation_entries or []
    sections = []
    dreamzero_execution = [
        dict(
            entry,
            category="ACTUAL ROLLOUT / EXECUTED BEHAVIOR",
            paired_control={
                "href": f'#{entry["id"]}_imagined_futures',
                "label": "Open same-seed imagined-future control",
            },
        )
        for entry in entries
        if entry["id"].startswith("dreamzero_droid_seed")
    ]
    world_model_execution = dreamzero_execution + [
        entry
        for entry in entries
        if entry["category"] == "WAM"
        and not entry["id"].startswith("dreamzero_droid_seed")
    ]
    vla_execution = [entry for entry in entries if entry["category"] == "VLA"]
    section_specs = (
        (
            "world-model-execution",
            "WORLD MODELS — actual simulator execution",
            "Every card is an executed rollout. Its future interface states whether decoded, latent-only, or action-only future evidence was exposed.",
            world_model_execution,
        ),
        (
            "vla-execution",
            "VLAs — actual simulator execution",
            "Behavioral rollout evidence from action-producing policies; no imagined video is inferred when the interface exposes none.",
            vla_execution,
        ),
    )
    cfg_dreamzero_present = any(
        entry["arm_id"] == "dreamzero_action_cfg_s2" for entry in cfg_ablation_entries
    )
    for section_id, title, intro, section_entries in section_specs:
        cards = "".join(entry_card(entry) for entry in section_entries)
        cfg_cards = ""
        if section_id == "world-model-execution" and cfg_ablation_entries:
            cfg_cards = (
                '<div class="cfg-intro"><p class="overline">Guidance ablation · complete media</p>'
                '<h3>What changed when inference-time guidance changed?</h3>'
                '<p>Each wide card keeps the six matched intervention executions beside all decodable retained media covered by the released interface: 64 Cosmos local RGB horizons or six DreamZero official reset decodes. It then reports exact before→after success counts and signed endpoint margins. Each setting has n=3 per direction; the two models remain separate descriptive pilots.</p></div>'
                f'<div class="cfg-stack">{"".join(cfg_ablation_card(entry) for entry in cfg_ablation_entries)}</div>'
            )
        if (
            section_id == "world-model-execution"
            and not dreamzero_present
            and not cfg_dreamzero_present
        ):
            contract = manifest["dreamzero_manifest_contract"]
            cards = f"""
      <article class="pending" id="dreamzero-pending">
        <p class="overline">RTX rollout target · evidence pending</p>
        <h3>DreamZero DROID</h3>
        <p>No valid DreamZero behavioral clip is committed yet. This is intentionally not a zero and not a placeholder rollout.</p>
        <p>When the RTX lane produces valid videos, the renderer will ingest hash-validated <code>gallery_entries</code> from <code>{html.escape(contract['path'])}</code>.</p>
      </article>""" + cards
        sections.append(
            f'<section id="{html.escape(section_id)}"><div class="section-head"><h2>{title}</h2><p>{intro}</p></div>'
            f'{cfg_cards}<div class="grid">{cards}</div></section>'
        )

    prediction_only_section = ""
    if prediction_only_entries:
        prediction_only_section = (
            '<section id="world-model-prediction-only"><div class="section-head">'
            '<h2>WORLD MODELS — prediction-only</h2><p>Model futures are visible; simulator execution is unavailable.</p></div>'
            '<p class="boundary"><strong>Execution boundary.</strong> These are fixed-observation model predictions, not robot rollouts. '
            'The adjacent panel records why physical execution remains unavailable.</p>'
            f'<div class="grid">{"".join(prediction_only_card(entry) for entry in prediction_only_entries)}</div></section>'
        )

    if imagination_present:
        imagination_cards = "".join(
            entry_card(dict(
                entry,
                category="IMAGINED FUTURE / MODEL PREDICTION — NOT EXECUTION",
                paired_control={
                    "href": f'#{entry["id"].removesuffix("_imagined_futures")}',
                    "label": "Open same-seed actual rollout control",
                },
            ))
            for entry in imagination_entries
        )
        imagination_section = (
            '<section id="dreamzero-imagination"><div class="section-head">'
            '<h2>DreamZero imagined futures — not execution</h2><p>IMAGINED FUTURES / MODEL PREDICTIONS.</p>'
            f'</div><p class="boundary"><strong>Prediction boundary.</strong> These MP4s decode DreamZero’s retained latent video predictions. '
            'They do not show what the robot actually executed and do not add behavioral trials.</p>'
            f'<div class="grid">{imagination_cards}</div>'
            '<div class="archive-box"><h3>All nine original official decodes</h3>'
            '<p>Six valid behavioral-session decodes and three fixed-observation diagnostic decodes are archived byte-for-byte.</p>'
            f'{imagination_archive(official_decodes)}</div></section>'
        )
    else:
        imagination_contract = manifest["dreamzero_imagination_manifest_contract"]
        imagination_section = (
            '<section id="dreamzero-imagination"><div class="section-head">'
            '<h2>DreamZero imagined futures</h2><p>Official model predictions — not simulator execution.</p></div>'
            '<article class="pending"><p class="overline">Imagination archive pending</p>'
            '<h3>No committed official decodes yet</h3>'
            f'<p>The renderer will ingest the bounded archive from <code>{html.escape(imagination_contract["path"])}</code> '
            'only after all nine files validate.</p></article></section>'
        )

    missing_items = [
        item for item in manifest["missing_publication_media"]
        if not (
            (dreamzero_present or cfg_dreamzero_present)
            and item["model_id"] == "dreamzero_droid"
        )
    ]
    missing = "".join(missing_card(item) for item in missing_items)
    manifest_digest = sha256(DEFAULT_MANIFEST)
    cfg_css = ""
    if cfg_ablation_entries:
        cfg_css = """
.cfg-intro{max-width:920px;margin:4px 0 20px}.cfg-intro h3{font-size:clamp(25px,3vw,35px);margin:4px 0}.cfg-intro p:last-child{color:var(--muted);margin:.45em 0 0}.cfg-stack{display:grid;gap:24px;margin-bottom:30px}.cfg-card{border-color:#cfc8df;box-shadow:0 14px 35px rgba(40,27,72,.06)}.cfg-card header{align-items:start}.cfg-setting{color:var(--muted);margin:6px 0 0}.cfg-media{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}.cfg-media figure{margin:0;background:#fff;min-width:0}.cfg-media figcaption{padding:11px 16px 9px;font-weight:800;font-size:12px;letter-spacing:.04em;color:var(--accent)}.cfg-media figure>p{padding:9px 16px 13px;margin:0;color:var(--muted);font-size:13px}.cfg-prompts{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:18px}.cfg-direction{border-radius:12px;padding:15px 16px}.cfg-direction.left{background:var(--left)}.cfg-direction.right{background:var(--right)}.cfg-direction-name{margin:0;text-transform:uppercase;letter-spacing:.05em;font-size:11px;font-weight:850}.cfg-direction blockquote{font-size:15px}.cfg-direction-stats{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:13px}.cfg-direction-stats p{margin:0;padding-top:10px;border-top:1px solid rgba(23,32,42,.16)}.cfg-direction-stats span,.cfg-metrics span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.035em}.cfg-direction-stats strong,.cfg-metrics strong{display:block;margin-top:3px;font-size:17px}.cfg-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border-block:1px solid var(--line)}.cfg-metrics>div{background:#fff;padding:14px 18px}.cfg-result{margin:17px 18px 0;padding:13px 15px;background:#f2f5f8;border-radius:10px;color:var(--ink);font-size:15px}.cfg-definition{margin:13px 18px 0;color:var(--muted);font-size:14px}.cfg-boundaries{margin:13px 18px 0;padding:14px 16px;background:#f8f6fc;border-left:4px solid var(--accent);border-radius:0 10px 10px 0;color:var(--muted);font-size:13px}.cfg-boundaries p{margin:0}.cfg-boundaries p+p{margin-top:8px}
@media(max-width:840px){.cfg-media,.cfg-prompts,.cfg-metrics,.cfg-direction-stats{grid-template-columns:1fr}.cfg-card header{display:block}.cfg-card .status{margin-top:8px}.cfg-media video{max-height:none}.cfg-prompts{padding:12px}.cfg-metrics>div{padding:13px 15px}}
"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(manifest['title'])}</title>
<style>
:root{{--ink:#17202a;--muted:#596775;--paper:#f4f1ea;--card:#fff;--line:#d8d8d2;--left:#fff0d4;--right:#e4f1ff;--accent:#6941c6}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{width:min(1420px,calc(100% - 32px));margin:auto;padding:52px 0 80px}}h1{{max-width:980px;margin:.12em 0;font-size:clamp(42px,7vw,84px);line-height:.98;letter-spacing:-.05em}}h2{{font-size:clamp(31px,4vw,48px);margin:0}}h3{{font-size:25px;margin:2px 0 0}}.lede{{max-width:920px;color:var(--muted);font-size:20px}}.boundary{{padding:16px 20px;border-left:5px solid var(--accent);background:#fff;border-radius:0 12px 12px 0;max-width:1050px}}section{{margin-top:62px}}.section-head{{display:flex;align-items:end;justify-content:space-between;gap:24px;margin-bottom:20px;border-bottom:1px solid var(--line);padding-bottom:14px}}.section-head p{{color:var(--muted);margin:0}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}}article{{background:var(--card);border:1px solid var(--line);border-radius:16px;overflow:hidden}}article header{{padding:20px 22px 14px;display:flex;justify-content:space-between;gap:18px}}.overline{{margin:0;color:var(--accent);text-transform:uppercase;letter-spacing:.06em;font-size:12px;font-weight:800}}.status{{max-width:46%;color:var(--muted);font-size:12px;text-align:right}}video{{display:block;width:100%;max-height:560px;background:#111}}.comparison-media{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}}.comparison-media figure{{margin:0;background:#fff}}.comparison-media figcaption{{padding:10px 14px 8px;font-weight:800;font-size:12px;letter-spacing:.04em;color:var(--accent)}}.comparison-media p{{padding:0 14px 10px;margin:0;color:var(--muted);font-size:12px}}.comparison-unavailable{{display:flex;min-height:280px;flex-direction:column;justify-content:center;background:#faf7ff!important}}.comparison-unavailable p{{max-width:43ch}}.directions{{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:16px 18px 10px}}.direction{{padding:14px;border-radius:10px}}.direction.left{{background:var(--left)}}.direction.right{{background:var(--right)}}.direction-top{{display:flex;justify-content:space-between;gap:10px;font-size:13px}}blockquote{{margin:10px 0 0;font-weight:650}}.facts{{display:grid;grid-template-columns:1fr 1.4fr;gap:1px;background:var(--line);border-block:1px solid var(--line)}}.facts div{{background:#fff;padding:12px 18px}}.facts div:last-child{{grid-column:1/-1}}dt{{font-size:11px;text-transform:uppercase;color:var(--muted);font-weight:800}}dd{{margin:3px 0 0}}code{{overflow-wrap:anywhere;font-size:12px}}.note{{padding:0 18px 18px;color:var(--muted);font-size:14px}}a{{color:#4a2aa5}}.pending,.missing{{padding:24px;border-style:dashed}}.pending{{border-color:#8c6ddb;background:#faf7ff}}.missing h3{{margin-top:4px}}.missing-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.archive-box{{margin-top:22px;padding:22px;background:#fff;border:1px solid var(--line);border-radius:16px}}.archive{{columns:2;column-gap:32px;padding-left:22px}}.archive li{{break-inside:avoid;margin:8px 0}}.archive span{{color:var(--muted);font-size:12px}}footer{{margin-top:52px;color:var(--muted);font-size:13px;border-top:1px solid var(--line);padding-top:20px}}@media(max-width:840px){{.grid,.missing-grid,.comparison-media{{grid-template-columns:1fr}}.section-head{{display:block}}.directions,.facts{{grid-template-columns:1fr}}.facts div:last-child{{grid-column:auto}}article header{{display:block}}.status{{display:block;max-width:none;text-align:left;margin-top:7px}}.archive{{columns:1}}}}
{cfg_css}</style></head><body><main>
<p class="overline">Video-first evidence index · direct static LEFT/RIGHT commands</p><h1>{html.escape(manifest['title'])}</h1>
<p class="lede">Videos are embedded at full card width, separated into WORLD MODEL and VLA sections, and labeled with arena, prompt, direction, outcome, model interface, and evidence status. Missing media stays missing. {html.escape(manifest['display_policy'])}</p>
<p class="boundary"><strong>Claim boundary.</strong> {html.escape(manifest['claim_boundary'])}</p>
{sections[0]}
{prediction_only_section}
{imagination_section}
{''.join(sections[1:])}
<section><div class="section-head"><h2>Explicit media gaps</h2><p>No raw or diagnostic artifact is substituted for publication video.</p></div><div class="missing-grid">{missing}</div></section>
<footer>Generated by <code>tools/render_vla_wam_video_first_gallery.py</code> from the hash-bearing gallery manifest (SHA-256 <code>{manifest_digest}</code>). Re-run the generator after any conforming media-manifest update.</footer>
</main></body></html>
"""


def render_markdown(
    manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    dreamzero_present: bool,
    imagination_entries: list[dict[str, Any]],
    official_decodes: list[dict[str, Any]],
    imagination_present: bool,
    prediction_only_entries: list[dict[str, Any]],
    cfg_ablation_entries: list[dict[str, Any]] | None = None,
) -> str:
    cfg_ablation_entries = cfg_ablation_entries or []
    lines = [
        f"# {manifest['title']}",
        "",
        "This is the portable index for the embedded [HTML video gallery](VLA_WAM_STEERABILITY_VIDEO_GALLERY.html). "
        "DROID and RoboTwin are listed separately and their success rates are never pooled.",
        "",
        "## WORLD MODELS — actual simulator execution",
        "",
    ]
    cfg_dreamzero_present = any(
        entry["arm_id"] == "dreamzero_action_cfg_s2" for entry in cfg_ablation_entries
    )
    if cfg_ablation_entries:
        lines.extend([
            "### V2-A015 guidance ablation — complete six-cell media",
            "",
            "These are separate paired, descriptive post-result pilots with n=3 per direction in each setting. Each card keeps all six intervention executions beside all decodable retained media covered by the released interface: 64 Cosmos local RGB horizons or six DreamZero official reset decodes. No outcome-based selection is used.",
            "",
        ])
        for entry in cfg_ablation_entries:
            baseline = entry["baseline"]
            intervention = entry["intervention"]
            display_setting_label = entry.get("display_setting_label", entry["setting_label"])
            prediction_composite_label = (
                "Complete imagination composite"
                if entry["arm_id"] == "dreamzero_action_cfg_s2"
                else "Complete local-prediction composite"
            )
            lines.extend([
                f"#### {entry['model_label']} — {display_setting_label}",
                "",
                f"[▶ All six complete actual executions]({rel(entry['actual_video']['path'])}) · "
                f"[▶ {prediction_composite_label}]({rel(entry['prediction_video']['path'])}) · "
                f"[Media manifest]({rel(entry['media_manifest'])}) · [Paired comparison]({rel(entry['comparison'])})",
                "",
            ])
            for relation in CFG_RELATIONS:
                before = baseline["by_direction"][relation]
                after = intervention["by_direction"][relation]
                lines.extend([
                    f"> **Prompt asks {relation.upper()}:** “{entry['prompts'][relation]}”",
                    ">",
                    f"> Requested-task success: **{before['successes']}/3 → {after['successes']}/3**; "
                    f"mean signed endpoint margin: **{format_cfg_margin(before['mean_requested_margin_m'])} → {format_cfg_margin(after['mean_requested_margin_m'])}**.",
                    "",
                ])
            lines.extend([
                f"- Total requested-task success: **{baseline['successes']}/6 → {intervention['successes']}/6**.",
                f"- Mean-margin gap (RIGHT − LEFT): **{format_cfg_margin(baseline['signed_direction_gap_m'])} → {format_cfg_margin(intervention['signed_direction_gap_m'])}**.",
                f"- Absolute mean-margin gap: **{baseline['absolute_direction_imbalance_m']:.3f} m → {intervention['absolute_direction_imbalance_m']:.3f} m**.",
                f"- Interpretation: {entry['result_summary']}",
                f"- Prediction boundary: {entry['prediction_note']}",
                f"- Media boundary: {entry['claim_boundary']}",
                f"- Inference boundary: {entry['comparison_boundary']}",
                "",
            ])
    if dreamzero_present:
        lines.append(
            "DreamZero's three hash-validated actual simulator pairs are listed in this section and link to "
            "same-seed imagined-future controls later in this document. Every other WAM states whether its released "
            "interface exposes decoded, latent-only, or action-only future evidence."
        )
    elif not cfg_dreamzero_present:
        path = manifest["dreamzero_manifest_contract"]["path"]
        lines.extend([
            "**Pending — no behavioral video exists in the committed evidence.** This is not a zero. The generator is wired to "
            f"`{path}` and will ingest its `gallery_entries` only after every referenced clip validates.",
        ])
    lines.append("")
    imagined_by_seed = {entry["seed"]: entry for entry in imagination_entries}
    dreamzero_actual_by_seed = {
        entry["seed"]: entry
        for entry in entries
        if entry["id"].startswith("dreamzero_droid_seed")
    }

    def append_behavior_section(
        title: str,
        section_entries: list[dict[str, Any]],
        heading_already_emitted: bool = False,
    ) -> None:
        if not heading_already_emitted:
            lines.extend(["", f"## {title}", ""])
        for entry in section_entries:
            outcomes = "; ".join(
                f"{direction['relation']}: {direction['outcome']}" for direction in entry["directions"]
            )
            lines.extend([
                f"### {entry['model_label']} — {entry['pair_label']}",
                "",
                f"[▶ Open video]({rel(entry['video']['path'])}) · [Evidence manifest]({rel(entry['source_manifest'])})",
                "",
                f"- Outcome: {outcomes}",
                f"- Future interface: {entry['future_interface']}",
                f"- Evidence status: {entry['evidence_status']}",
                f"- Video SHA-256: `{entry['video']['sha256']}`",
                "",
            ])
            if entry["id"].startswith("dreamzero_droid_seed") and entry["seed"] in imagined_by_seed:
                imagined = imagined_by_seed[entry["seed"]]
                lines.append(f"[Open same-seed imagined-future control]({rel(imagined['video']['path'])})")
                lines.append("")
            comparison = entry.get("comparison_media")
            if comparison:
                lines.extend([
                    f"[Open adjacent {comparison['label']}]({rel(comparison['video']['path'])})",
                    "",
                    f"- Prediction boundary: {comparison['note']}",
                    "",
                ])
            for direction in entry["directions"]:
                lines.append(f"> {direction['relation']}: “{direction['prompt']}”")
            lines.append("")

    append_behavior_section(
        "WORLD MODELS — actual simulator execution",
        [entry for entry in entries if entry["category"] == "WAM"],
        heading_already_emitted=True,
    )
    if prediction_only_entries:
        lines.extend([
            "## WORLD MODELS — prediction-only",
            "",
            "These fixed-observation futures are model predictions, not simulator executions or behavioral episodes. "
            "The paired rollout panel is explicitly unavailable because the exact controller mapping remains blocked.",
            "",
        ])
        for entry in prediction_only_entries:
            evidence_links = " · ".join(
                f"[{evidence['label']}]({rel(evidence['path'])})"
                for evidence in entry["supporting_evidence"]
            )
            lines.extend([
                f"### {entry['model_label']} — {entry['pair_label']}",
                "",
                f"[▶ Open paired model prediction]({rel(entry['video']['path'])}) · "
                f"[Poster]({rel(entry['poster']['path'])}) · "
                f"[Evidence manifest]({rel(entry['source_manifest'])})"
                f"{(' · ' + evidence_links) if evidence_links else ''}",
                "",
                f"- Actual rollout: unavailable — `{entry['actual_rollout_unavailable_reason']}`.",
                f"- Reason: {entry['actual_rollout_unavailable_detail']}",
                f"- Future interface: {entry['future_interface']}",
                f"- Prediction SHA-256: `{entry['video']['sha256']}`",
                "",
            ])
            for direction in entry["directions"]:
                lines.append(f"> {direction['relation']}: “{direction['prompt']}” — {direction['outcome']}")
            lines.append("")
    lines.extend(["## DreamZero imagined futures — not execution", ""])
    if imagination_present:
        lines.append(
            "These are official model-predicted video decodes, not simulator executions, task outcomes, or additional episodes."
        )
        lines.append("")
        for entry in imagination_entries:
            lines.extend([
                f"### {entry['model_label']} — {entry['pair_label']}",
                "",
                f"[▶ Open paired imagination video]({rel(entry['video']['path'])}) · "
                f"[Imagination manifest]({rel(entry['source_manifest'])})",
                "",
                f"- Evidence status: {entry['evidence_status']}",
                f"- Video SHA-256: `{entry['video']['sha256']}`",
                "",
            ])
            if entry["seed"] in dreamzero_actual_by_seed:
                actual = dreamzero_actual_by_seed[entry["seed"]]
                lines.extend([
                    f"[Open same-seed actual rollout control]({rel(actual['video']['path'])})",
                    "",
                ])
        lines.extend(["### All nine original official decodes", ""])
        for record in official_decodes:
            lines.append(
                f"- [{record['id']}]({rel(record['archived_video']['path'])}) — "
                f"`{record['archived_video']['sha256']}`"
            )
        lines.append("")
    else:
        lines.extend([
            "**Pending — the bounded nine-file official decode archive is not committed yet.**",
            "",
        ])
    append_behavior_section(
        "VLAs — actual simulator execution",
        [entry for entry in entries if entry["category"] == "VLA"],
    )
    lines.extend(["## Missing publication media", ""])
    for item in manifest["missing_publication_media"]:
        if (dreamzero_present or cfg_dreamzero_present) and item["model_id"] == "dreamzero_droid":
            continue
        lines.append(f"- **{item['model_id']} — {item['status']}:** {item['reason']}")
    lines.extend([
        "",
        "Regenerate and validate with:",
        "",
        "```bash",
        "python3 tools/render_vla_wam_video_first_gallery.py",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    html_path = args.html if args.html.is_absolute() else REPO_ROOT / args.html
    markdown_path = args.markdown if args.markdown.is_absolute() else REPO_ROOT / args.markdown

    manifest = json.loads(args.manifest.read_text())
    (
        entries,
        dreamzero_present,
        imagination_entries,
        official_decodes,
        imagination_present,
        prediction_only_entries,
    ) = load_entries(manifest)
    cfg_ablation_entries = load_cfg_ablation_entries(manifest)
    html_path.write_text(
        render_html(
            manifest,
            entries,
            dreamzero_present,
            imagination_entries,
            official_decodes,
            imagination_present,
            prediction_only_entries,
            cfg_ablation_entries,
        )
    )
    markdown_path.write_text(
        render_markdown(
            manifest,
            entries,
            dreamzero_present,
            imagination_entries,
            official_decodes,
            imagination_present,
            prediction_only_entries,
            cfg_ablation_entries,
        )
    )
    result = {
                "status": "valid",
                "entry_count": len(entries),
                "dreamzero_media_present": dreamzero_present,
                "dreamzero_imagination_media_present": imagination_present,
                "dreamzero_imagination_entry_count": len(imagination_entries),
                "dreamzero_official_decode_count": len(official_decodes),
                "prediction_only_entry_count": len(prediction_only_entries),
                "html": str(html_path.relative_to(REPO_ROOT)),
                "markdown": str(markdown_path.relative_to(REPO_ROOT)),
            }
    if "cfg_ablation_media_contracts" in manifest:
        result["cfg_ablation_entry_count"] = len(cfg_ablation_entries)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
