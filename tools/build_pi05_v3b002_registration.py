#!/usr/bin/env python3
"""Build the pre-inference V3-B002 pi0.5 position-reflection registration.

The registration deliberately reuses the exact V3-B001 seed blocks, physical
layouts, prompt bytes, and within-seed execution order.  It releases no model
request or behavioral episode; a separately hash-bound runtime gate is still
required before launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "vla_wam_language_steerability_v3"
AMENDMENT_ID = "V3-B002"
MODEL_ID = "pi05_current_stack_droid"
PHASE = "B_confound_ablation"
SEEDS = tuple(range(9400, 9427))
ARMS = ("control", "position_mirrored")
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
SUCCESS_PREDICATE_ID = (
    "v2_frozen_droid_robolab_release_inside_45deg_requested_relation"
)
SOURCE_CELLS = Path(
    "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/"
    "nano_mirror_v3b001_cells.jsonl"
)
SOURCE_MANIFEST = Path(
    "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/"
    "nano_mirror_v3b001_manifest.json"
)
SOURCE_RESULTS = Path(
    "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/"
    "nano_v3b001_results_manifest.json"
)
PI05_PHASE_A = Path(
    "artifacts/vla_wam_shared_v3/results/"
    "pi05_current_stack_droid_phase_a_summary.json"
)
CHECKPOINT_MANIFEST = Path(
    "artifacts/vla_wam_shared_v2/pilot/expansion/"
    "pi05_current_stack_checkpoint_manifest.json"
)
OUTPUTS = {
    "amendment": "post_result_pi05_mirror_v3b002_amendment.json",
    "cells": "pi05_mirror_v3b002_cells.jsonl",
    "manifest": "pi05_mirror_v3b002_manifest.json",
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
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def canonical_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} contains a non-object row")
    return rows


def validate_sources(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    source_paths = [SOURCE_CELLS, SOURCE_MANIFEST, SOURCE_RESULTS, PI05_PHASE_A, CHECKPOINT_MANIFEST]
    for relative in source_paths:
        if not (repo_root / relative).is_file():
            raise ValueError(f"missing registration source: {relative}")
    manifest = load_json(repo_root / SOURCE_MANIFEST)
    cells_sha = sha256_file(repo_root / SOURCE_CELLS)
    if (
        manifest.get("amendment_id") != "V3-B001"
        or manifest.get("status") != "hash_bound_release_ready"
        or manifest.get("counts", {}).get("behavioral_cells") != 108
        or manifest.get("files", {}).get("cells", {}).get("sha256") != cells_sha
    ):
        raise ValueError("V3-B001 source registry is not the exact hash-bound release")
    rows = load_jsonl(repo_root / SOURCE_CELLS)
    if len(rows) != 108:
        raise ValueError("V3-B001 source registry must contain 108 cells")
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        seed = row.get("environment_seed")
        if type(seed) is not int:
            raise ValueError("V3-B001 cell lacks an integer environment seed")
        by_seed.setdefault(seed, []).append(row)
    if tuple(sorted(by_seed)) != SEEDS:
        raise ValueError("V3-B001 seed list changed")
    expected_conditions = {(arm, relation) for arm in ARMS for relation in RELATIONS}
    for seed, seed_rows in by_seed.items():
        conditions = {(row.get("arm"), row.get("relation")) for row in seed_rows}
        order = sorted(row.get("execution_order_index_within_seed") for row in seed_rows)
        if conditions != expected_conditions or order != [1, 2, 3, 4]:
            raise ValueError(f"V3-B001 seed {seed} is not a complete randomized block")
    source_hashes = {str(path): sha256_file(repo_root / path) for path in source_paths}
    return rows, source_hashes


def build_rows(source_rows: list[dict[str, Any]], amendment_sha256: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        seed = int(source["environment_seed"])
        arm = str(source["arm"])
        relation = str(source["relation"])
        cell_id = f"v3b002:pi05:seed{seed}:{arm}:{relation}"
        randomization_key = sha256_bytes(
            (
                "vla_wam_language_steerability_v3:V3-B002:pi05:"
                f"seed{seed}:{arm}:{relation}:reuse_v3b001_order"
            ).encode()
        )
        rows.append(
            {
                "schema_version": "vla-wam-shared-v3b-pi05-mirror-cell-v1",
                "study_id": STUDY_ID,
                "amendment_id": AMENDMENT_ID,
                "amendment_sha256": amendment_sha256,
                "phase": PHASE,
                "arena": "droid_robolab",
                "model_id": MODEL_ID,
                "cell_id": cell_id,
                "matched_block_id": f"v3b002:pi05:seed{seed}",
                "environment_seed": seed,
                "sampling_seed": seed,
                "arm": arm,
                "relation": relation,
                "prompt_family": "direct_command",
                "prompt": PROMPTS[relation],
                "prompt_sha256": sha256_bytes(PROMPTS[relation].encode()),
                "factor": "movable_object_center_position_reflection_about_robot_sagittal_plane",
                "fixture_id": f"v3b001_nano_{arm}",
                "fixture_sha256": source["fixture_sha256"],
                "source_v3b001_cell_id": source["cell_id"],
                "source_v3b001_queue_sha256": "018b8b6ae76ac46f2f89eef83c4b16d7a4ff3d1ff15d91527b96fb56b5432c5a",
                "source_v3b001_randomization_key_sha256": source["randomization_key_sha256"],
                "source_execution_order_index_within_seed": source["execution_order_index_within_seed"],
                "execution_order_index_within_seed": source["execution_order_index_within_seed"],
                "randomization_key_sha256": randomization_key,
                "success_predicate_id": SUCCESS_PREDICATE_ID,
                "execution_status": "registered_pre_inference_runtime_release_gate_required",
                "runtime_identity_requirement": {
                    "openpi_commit": "c23745b5ad24e98f66967ea795a07b2588ed6c79",
                    "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
                    "openpi_config": "pi05_droid_jointpos_polaris",
                    "checkpoint_manifest_sha256": "f5a56d9565f9381ccdeeaa165b0495dab6d17a81836cc7b01c5fbc6ab89e74ca",
                    "open_loop_horizon": 15,
                    "action_shape": [15, 8],
                    "clean_external_repositories_required": True,
                },
                "required_raw_outputs": [
                    "viewport_video",
                    "executed_action_trace",
                    "raw_behavioral_episode_jsonl",
                ],
                "required_episode_fields": {
                    "success": "boolean alias of frozen requested_success",
                    "failure_category": "correct|pick_failed|transport_failed|wrong_side|release_failed",
                    "signed_final_lateral_offset_m": "finite for every valid episode; +Y is robot LEFT",
                    "requested_side_depth_m": "finite alias of frozen final_requested_signed_margin_m",
                    "cone_entry_step": "integer or null from first requested-cone entry",
                    "cone_entry_sustained": "boolean from frozen three-sample rule",
                    "episode_length_steps": "integer executed simulator steps",
                    "time_to_first_contact_steps": "integer or null with explicit contact instrumentation status",
                    "grasp_step": "integer or null from a separately retained object_grabbed condition stream; never substituted by verified pickup",
                    "cumulative_lateral_path_m": "sum_t abs(signed_lateral_offset[t]-signed_lateral_offset[t-1]) in robot-base frame",
                    "peak_lateral_excursion_m": "max_t abs(signed_lateral_offset[t]-signed_lateral_offset[0]) in robot-base frame",
                    "endpoint_shift_m": "pair-derived LEFT offset minus RIGHT offset in a separate seed-layout pair JSONL",
                    "action_distinct": "pair-derived exact action-trace inequality on common executed prefix in the pair JSONL",
                },
                "required_pair_outputs": [
                    "endpoint_redirection_D_m",
                    "executed_actions_distinct",
                    "common_prefix_action_count",
                    "common_prefix_action_rms",
                ],
                "missing_future_policy": "action_only_interface_not_applicable_never_zero",
                "technical_invalidity_policy": "retain in a separate stream and repair only this identical registered cell",
                "valid_failure_policy": "retain every valid behavioral failure in all full-sample analyses",
            }
        )
    return rows


def amendment(recorded_at_utc: str, source_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": "vla-wam-shared-v3b-pi05-mirror-amendment-v1",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "recorded_at_utc": recorded_at_utc,
        "status": "frozen_before_any_v3b002_model_request_or_behavioral_episode",
        "known_results_disclosure": {
            "nano_v3b001": {
                "endpoint_redirection_interaction_mean_m": 0.004680250219448849,
                "endpoint_redirection_exact_sign_test_p": 0.7011080384254456,
                "requested_side_depth_interaction_mean_m": -0.24766719351626104,
                "requested_side_depth_exact_sign_test_p": 4.9233436584472656e-05,
                "successes": {
                    "control:left": "26/27",
                    "control:right": "26/27",
                    "position_mirrored:left": "27/27",
                    "position_mirrored:right": "23/27",
                },
            },
            "pi05_phase_a": {
                "left": "5/27",
                "right": "24/27",
                "right_minus_left_success_count_gap": 19,
            },
        },
        "design": {
            "model_id": MODEL_ID,
            "arena": "droid_robolab",
            "matched_seeds": list(SEEDS),
            "matched_seed_count": 27,
            "cells_per_seed": 4,
            "behavioral_episode_ceiling": 108,
            "arms": list(ARMS),
            "relations": list(RELATIONS),
            "exact_prompts": PROMPTS,
            "reflection": "For rubiks_cube, bowl, and banana center positions only: (x,y,z) -> (x,-y,z) about robot-base sagittal plane y=0.",
            "held_fixed": [
                "robot base",
                "cameras",
                "all non-movable scene geometry",
                "prompt bytes",
                "controller and 15-action horizon",
                "frozen 45-degree cone and detached-release success predicate",
                "environment and sampling seed within every four-cell block",
            ],
            "execution_order": "Reuse the exact V3-B001 within-seed randomized cell order.",
            "source_registry": str(SOURCE_CELLS),
        },
        "registered_predictions": {
            "H1_endpoint_redirection": "The reflected-minus-control interaction J is approximately zero, replicating Nano V3-B001.",
            "H2_requested_side_depth": "The reflected-minus-control interaction I is strongly negative, replicating Nano V3-B001.",
            "H3_binary_success": "Two-sided test of whether position reflection changes, attenuates, or reverses pi0.5's prior RIGHT-over-LEFT binary success gap; no outcome direction is added after registration.",
        },
        "analysis_plan": {
            "H1": {
                "per_layout": "D_layout = signed_offset_LEFT - signed_offset_RIGHT",
                "interaction": "J = D_position_mirrored - D_control",
            },
            "H2": {
                "per_layout": "B_layout = requested_depth_RIGHT - requested_depth_LEFT = -signed_offset_RIGHT - signed_offset_LEFT",
                "interaction": "I = B_position_mirrored - B_control",
            },
            "H3": {
                "per_seed": "DiD = (success_RIGHT-success_LEFT)_position_mirrored - (success_RIGHT-success_LEFT)_control",
                "support": [-2, -1, 0, 1, 2],
                "test": "exact two-sided within-seed control/reflected label permutation using the absolute summed DiD statistic",
                "outputs": ["per-seed DiD distribution", "2x2 layout-by-direction success table"],
            },
            "continuous_reporting": {
                "per_layout": "mean paired contrast with matched-seed percentile-bootstrap 95% CI",
                "interaction": "mean with matched-seed percentile-bootstrap 95% CI plus median",
                "bootstrap_replicates": 20000,
                "bootstrap_master_seed": 3104159,
                "bootstrap_unit": "matched seed",
                "sign_test": "exact two-sided paired sign test, ties excluded and sign counts reported",
            },
            "missingness": "No imputation. Every valid behavioral failure remains in full-sample offset/depth analyses; infrastructure failures are excluded and separately ledgered.",
        },
        "failure_mode_split": {
            "models": ["pi05_current_stack_droid", "dreamzero_droid_action_cfg", "cosmos3_edge_policy_droid"],
            "cohort": "existing V3 Phase-A 54-episode expanded cohort for each checkpoint",
            "table": "direction by correct/pick_failed/transport_failed/wrong_side/release_failed",
            "test": "Fisher exact test on the direction by four-class failure-only subtable",
            "report": "raw counts and row-normalized proportions",
        },
        "release_boundary": {
            "model_requests_before_registration": 0,
            "behavioral_episodes_before_registration": 0,
            "behavioral_release": False,
            "next_gate": "Hash-bound pi0.5 runtime identity, exact reflected-reset attestation, raw-output write proof, deterministic repeat, and LEFT/RIGHT prompt-sensitivity gate.",
        },
        "source_sha256": source_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recorded-at-utc", required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    source_rows, source_hashes = validate_sources(root)
    amendment_value = amendment(args.recorded_at_utc, source_hashes)
    amendment_bytes = canonical_json(amendment_value)
    amendment_sha = sha256_bytes(amendment_bytes)
    rows = build_rows(source_rows, amendment_sha)
    cells_bytes = canonical_jsonl(rows)
    order_counts: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        for relation in RELATIONS:
            counts = Counter(
                row["execution_order_index_within_seed"]
                for row in rows
                if row["arm"] == arm and row["relation"] == relation
            )
            order_counts[f"{arm}:{relation}"] = {str(i): counts[i] for i in range(1, 5)}
    manifest_value = {
        "schema_version": "vla-wam-shared-v3b-pi05-mirror-manifest-v1",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "recorded_at_utc": args.recorded_at_utc,
        "status": "hash_bound_registered_not_behaviorally_released",
        "counts": {
            "matched_seeds": 27,
            "registered_behavioral_cells": 108,
            "control_cells": 54,
            "position_mirrored_cells": 54,
            "left_cells": 54,
            "right_cells": 54,
        },
        "execution_order_position_counts": order_counts,
        "files": {
            "amendment": {
                "path": OUTPUTS["amendment"],
                "bytes": len(amendment_bytes),
                "sha256": amendment_sha,
            },
            "cells": {
                "path": OUTPUTS["cells"],
                "bytes": len(cells_bytes),
                "sha256": sha256_bytes(cells_bytes),
                "row_count": len(rows),
            },
        },
        "source_v3b001_cells_sha256": source_hashes[str(SOURCE_CELLS)],
        "release_rule": "This registration freezes design and predictions only. No row may launch until a separately persisted runtime/reset/output/sensitivity release gate binds this manifest hash.",
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        OUTPUTS["amendment"]: amendment_bytes,
        OUTPUTS["cells"]: cells_bytes,
        OUTPUTS["manifest"]: canonical_json(manifest_value),
    }
    for name, payload in payloads.items():
        path = output / name
        if path.exists() and path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite nonidentical registration: {path}")
        path.write_bytes(payload)
    print(json.dumps({
        "status": "registered_not_released",
        "rows": len(rows),
        "outputs": {
            name: {"sha256": sha256_bytes(payload), "bytes": len(payload)}
            for name, payload in payloads.items()
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
