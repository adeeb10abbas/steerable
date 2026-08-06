#!/usr/bin/env python3
"""Freeze the pre-inference V3-B003 DreamZero position-reflection cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "vla_wam_language_steerability_v3"
AMENDMENT_ID = "V3-B003"
MODEL_ID = "dreamzero_droid_action_cfg"
SEEDS = tuple(range(9400, 9427))
ARMS = ("control", "position_mirrored")
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
SOURCE_CELLS = Path("artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_cells.jsonl")
SOURCE_MANIFEST = Path("artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_manifest.json")
SOURCE_AMENDMENT = Path("artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/post_result_nano_mirror_v3b001_amendment.json")
DREAM_SUMMARY = Path("artifacts/vla_wam_shared_v3/results/dreamzero_droid_action_cfg_phase_a_summary.json")
DREAM_EVIDENCE = Path("artifacts/vla_wam_shared_v3/results/dreamzero_droid_action_cfg_phase_a_evidence_hash_manifest.json")
DREAM_CHECKPOINT = Path("artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_official_source_checkpoint_manifest.json")
V2A015_AMENDMENT = Path("artifacts/vla_wam_shared_v2/pilot/post_result_cfg_ablation_v2a015_amendment.json")
PI05_B002_RESULTS = Path("artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/results/pi05_v3b002_output_manifest.json")
FAILURE_REPORT = Path("artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/analysis/failure_mode_split_report.json")
OUTPUTS = {
    "amendment": "post_result_dreamzero_mirror_v3b003_amendment.json",
    "cells": "dreamzero_mirror_v3b003_cells.jsonl",
    "manifest": "dreamzero_mirror_v3b003_manifest.json",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def canonical_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(row, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode() for row in rows)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} contains a non-object row")
    return rows


def validate_sources(root: Path) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    paths = (SOURCE_CELLS, SOURCE_MANIFEST, SOURCE_AMENDMENT, DREAM_SUMMARY, DREAM_EVIDENCE, DREAM_CHECKPOINT, V2A015_AMENDMENT, PI05_B002_RESULTS, FAILURE_REPORT)
    for relative in paths:
        if not (root / relative).is_file():
            raise ValueError(f"missing source: {relative}")
    hashes = {str(relative): sha256_file(root / relative) for relative in paths}
    expected = {
        str(SOURCE_CELLS): "018b8b6ae76ac46f2f89eef83c4b16d7a4ff3d1ff15d91527b96fb56b5432c5a",
        str(SOURCE_MANIFEST): "5c82268739feb41281435a51dcd848b575218cd9fbe5839d9ad130d1a7888830",
        str(SOURCE_AMENDMENT): "9d88c29733fa3b24a154977bc25d04d2d77df5be59e3213f0c3a6cfbe3edc6a0",
        str(DREAM_SUMMARY): "50ec800009b1f128ce8fe877c186b08a47ec3021feaa7cf4f8baab984549b120",
        str(DREAM_EVIDENCE): "0150afb2145cc2e1d18bc16bd34b4af404c7fabb77486d197af78776fcd93427",
        str(DREAM_CHECKPOINT): "75fd6c6b7601f5706eb70140519ee8d57b18fe79e49cc2792c30b0d9be016eeb",
        str(V2A015_AMENDMENT): "7c3b98d5e578a5744a463201060ccdacf7462471418b08dc64a2ac29a9a9fece",
        str(PI05_B002_RESULTS): "523a6a2625dc0b67c1f425220006bdca6edea32e554f826f2b6f616c57393090",
        str(FAILURE_REPORT): "f6289333d77538ed9235a567cb98d70333dd50549f7e52d13bfe5d34a10bdf96",
    }
    if hashes != expected:
        raise ValueError("a hash-bound registration source changed")

    rows = read_jsonl(root / SOURCE_CELLS)
    if len(rows) != 108:
        raise ValueError("V3-B001 source must contain 108 cells")
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["environment_seed"]), []).append(row)
    if tuple(sorted(by_seed)) != SEEDS:
        raise ValueError("V3-B001 seed registry changed")
    expected_conditions = {(arm, relation) for arm in ARMS for relation in RELATIONS}
    for seed_rows in by_seed.values():
        if {(row["arm"], row["relation"]) for row in seed_rows} != expected_conditions:
            raise ValueError("V3-B001 condition grid changed")
        if sorted(row["execution_order_index_within_seed"] for row in seed_rows) != [1, 2, 3, 4]:
            raise ValueError("V3-B001 block order changed")

    summary = read_json(root / DREAM_SUMMARY)
    contract = summary["registered_execution_contract"]
    if summary.get("model_id") != MODEL_ID or contract.get("identity_binding") != "V2-A015:dreamzero_action_cfg_s2":
        raise ValueError("DreamZero s=2 identity changed")
    if contract.get("baseline_s1_used") is not False or contract.get("effective_official_model_noise_seed") != 1140:
        raise ValueError("DreamZero released seed/configuration semantics changed")
    return rows, hashes, contract


def make_amendment(recorded_at_utc: str, hashes: dict[str, str], contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "vla-wam-shared-v3b-dreamzero-mirror-amendment-v1",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "recorded_at_utc": recorded_at_utc,
        "status": "frozen_before_any_v3b003_model_request_or_behavioral_episode",
        "post_result_disclosure": {
            "dreamzero_phase_a": {"left": "3/27", "right": "17/27", "failure_shape_exact_p": 0.0017219568583829528},
            "nano_v3b001_known": True,
            "pi05_v3b002_known": True,
            "reason_selected": "Third checkpoint replication after A1 rejected a universal same-shape failure account for DreamZero and B1 found no monotonic competence-gap relation.",
        },
        "design": {
            "model_id": MODEL_ID,
            "identity_binding": "V2-A015:dreamzero_action_cfg_s2",
            "arena": "droid_robolab",
            "matched_seeds": list(SEEDS),
            "matched_seed_count": 27,
            "cells_per_seed": 4,
            "behavioral_episode_ceiling": 108,
            "arms": list(ARMS),
            "relations": list(RELATIONS),
            "exact_prompts": PROMPTS,
            "reflection": "For rubiks_cube, bowl, and banana center positions only: (x,y,z) -> (x,-y,z) about robot-base sagittal plane y=0.",
            "execution_order": "Reuse the exact V3-B001 within-seed randomized cell order.",
            "block_lane_rule": "All four cells for one seed execute on one runtime lane.",
            "registered_seed_semantics": "Environment and matched-block labels are 9400-9426; the released DreamZero model-noise seed remains constant at 1140.",
            "held_fixed": ["robot base", "cameras", "nonmovable geometry", "prompt bytes", "DreamZero s=2 identity", "video guidance 5", "16 inference steps", "8-action execution horizon", "450-action cap", "frozen success predicate"],
        },
        "registered_predictions": {
            "endpoint_redirection": "Two-sided test of the reflected-minus-control interaction; no post-result direction is imposed.",
            "requested_side_depth": "Reflection is predicted to produce a negative interaction, matching the known Nano and pi0.5 pattern.",
            "binary_success": "Reflection is predicted to attenuate or reverse DreamZero's Phase-A RIGHT advantage; the primary exact test remains two-sided.",
        },
        "analysis_plan": {
            "H1": {"per_layout": "D_layout = signed_offset_LEFT - signed_offset_RIGHT", "interaction": "J = D_position_mirrored - D_control"},
            "H2": {"per_layout": "B_layout = -signed_offset_RIGHT - signed_offset_LEFT", "interaction": "I = B_position_mirrored - B_control"},
            "H3": {"per_seed": "DiD = (success_RIGHT-success_LEFT)_position_mirrored - (success_RIGHT-success_LEFT)_control", "test": "exact two-sided within-seed control/reflected label permutation using absolute summed DiD"},
            "continuous_reporting": {"bootstrap_replicates": 20000, "bootstrap_master_seed": 3104159, "bootstrap_unit": "matched seed", "sign_test": "exact two-sided paired sign test; ties excluded and reported", "report_median": True},
            "requested_margin_secondary": "Only the named all-four-cells-correct complete-case subset; report realized n and never mix unmatched successes.",
            "missingness": "No imputation. Every valid failure remains in full-sample signed-offset analysis; infrastructure attempts remain separate.",
        },
        "runtime_contract_source": contract,
        "release_boundary": {"model_requests_before_registration": 0, "behavioral_episodes_before_registration": 0, "behavioral_release": False, "next_gate": "Fresh hash-bound DreamZero runtime identity, model-blind physical/reset/renderer/writer gates, exact bridge-reset attestation, and fixed-observation repeat/prompt-sensitivity/future-retention gate."},
        "source_sha256": hashes,
    }


def make_rows(source_rows: list[dict[str, Any]], amendment_sha: str, contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in source_rows:
        seed = int(source["environment_seed"])
        arm = str(source["arm"])
        relation = str(source["relation"])
        rows.append({
            "schema_version": "vla-wam-shared-v3b-dreamzero-mirror-cell-v1",
            "study_id": STUDY_ID,
            "amendment_id": AMENDMENT_ID,
            "amendment_sha256": amendment_sha,
            "phase": "B_confound_ablation",
            "arena": "droid_robolab",
            "model_id": MODEL_ID,
            "cell_id": f"v3b003:dreamzero:seed{seed}:{arm}:{relation}",
            "matched_block_id": f"v3b003:dreamzero:seed{seed}",
            "environment_seed": seed,
            "registered_sampling_seed_label": seed,
            "effective_model_noise_seed": 1140,
            "arm": arm,
            "relation": relation,
            "prompt_family": "direct_command",
            "prompt": PROMPTS[relation],
            "prompt_sha256": sha256_bytes(PROMPTS[relation].encode()),
            "fixture_id": f"v3b001_nano_{arm}",
            "fixture_sha256": source["fixture_sha256"],
            "source_v3b001_cell_id": source["cell_id"],
            "source_v3b001_queue_sha256": "018b8b6ae76ac46f2f89eef83c4b16d7a4ff3d1ff15d91527b96fb56b5432c5a",
            "execution_order_index_within_seed": source["execution_order_index_within_seed"],
            "source_v3b001_randomization_key_sha256": source["randomization_key_sha256"],
            "randomization_key_sha256": sha256_bytes(f"{STUDY_ID}:{AMENDMENT_ID}:dreamzero:seed{seed}:{arm}:{relation}:reuse_v3b001_order".encode()),
            "success_predicate_id": "v2_frozen_droid_robolab_release_inside_45deg_requested_relation",
            "runtime_identity_requirement": {
                "identity_binding": "V2-A015:dreamzero_action_cfg_s2",
                "checkpoint": contract["checkpoint"],
                "action_cfg_style_scale": 2,
                "video_cfg_scale": 5,
                "effective_official_model_noise_seed": 1140,
                "action_shape": [24, 8],
                "open_loop_horizon": 8,
                "action_cap": 450,
                "isolated_two_rank_server_required": True,
                "port_5000_prohibited": True,
            },
            "required_raw_outputs": ["viewport_video", "executed_action_trace", "raw_behavioral_episode_jsonl", "returned_action_chunks", "latent_future_per_request", "official_full_reset_decode"],
            "required_measurements": ["signed_final_lateral_offset_m", "final_requested_signed_margin_m", "first_cone_or_native_region_entry_step", "entry_kind", "episode_length_steps", "first_contact_step_or_explicit_unavailable", "object_path_length_m", "failure_taxonomy"],
            "future_policy": "Retain every exposed latent future and at least one official full-reset decode per valid episode; missing future is infrastructure-invalid and never zero.",
            "valid_failure_policy": "Retain every valid behavioral failure in all full-sample analyses.",
            "technical_invalidity_policy": "Separate ledger; repair only this identical registered cell.",
            "execution_status": "registered_pre_inference_runtime_release_gate_required",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recorded-at-utc", required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    source_rows, hashes, contract = validate_sources(root)
    amendment_value = make_amendment(args.recorded_at_utc, hashes, contract)
    amendment_bytes = canonical_json(amendment_value)
    amendment_sha = sha256_bytes(amendment_bytes)
    rows = make_rows(source_rows, amendment_sha, contract)
    cells_bytes = canonical_jsonl(rows)
    order_counts = {}
    for arm in ARMS:
        for relation in RELATIONS:
            counts = Counter(row["execution_order_index_within_seed"] for row in rows if row["arm"] == arm and row["relation"] == relation)
            order_counts[f"{arm}:{relation}"] = {str(index): counts[index] for index in range(1, 5)}
    manifest = {
        "schema_version": "vla-wam-shared-v3b-dreamzero-mirror-manifest-v1",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "recorded_at_utc": args.recorded_at_utc,
        "status": "hash_bound_registered_not_behaviorally_released",
        "counts": {"matched_seeds": 27, "registered_behavioral_cells": 108, "control_cells": 54, "position_mirrored_cells": 54, "left_cells": 54, "right_cells": 54},
        "execution_order_position_counts": order_counts,
        "files": {
            "amendment": {"path": OUTPUTS["amendment"], "bytes": len(amendment_bytes), "sha256": amendment_sha},
            "cells": {"path": OUTPUTS["cells"], "bytes": len(cells_bytes), "sha256": sha256_bytes(cells_bytes), "row_count": len(rows)},
        },
        "source_v3b001_cells_sha256": hashes[str(SOURCE_CELLS)],
        "release_rule": "No row may launch until a separately persisted DreamZero runtime/reset/output/repeat/prompt-sensitivity/future-retention release gate binds this manifest hash.",
    }
    payloads = {OUTPUTS["amendment"]: amendment_bytes, OUTPUTS["cells"]: cells_bytes, OUTPUTS["manifest"]: canonical_json(manifest)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        path = args.output_dir / name
        if path.exists() and path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite nonidentical registration: {path}")
        path.write_bytes(payload)
    print(json.dumps({"status": "registered_not_released", "rows": len(rows), "outputs": {name: {"bytes": len(payload), "sha256": sha256_bytes(payload)} for name, payload in payloads.items()}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
