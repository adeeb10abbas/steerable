#!/usr/bin/env python3
"""Materialize the prospectively frozen V3-C001 registry and randomization."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from .contract import (
    EXPERIMENT_ID,
    MODEL_CONTRACTS,
    PROMPT_FORMS,
    PROMPTS,
    RANDOMIZATION_NAMESPACE,
    RELATIONS,
    SCHEMA_PREFIX,
    SEEDS,
    canonical_json_bytes,
    prompt_sha256,
    randomized_conditions,
    sha256_file,
    validate_cells,
)


REGISTRY_RELATIVE = Path("artifacts/vla_wam_shared_v3/four_phrasings_registry.json")
ANALYSIS_RELATIVE = Path("artifacts/vla_wam_shared_v3/analysis_plan.json")
OUTPUT_RELATIVE = Path("artifacts/vla_wam_shared_v3/phase_c/four_phrasings_v3c001")
FROZEN_AT = "2026-08-07T00:58:37Z"
EXPECTED_SOURCE_HASHES = {
    str(REGISTRY_RELATIVE): "30cf5594eb74bc195e7ec6ae878523fb302771e6548c794127e987c0066513e6",
    str(ANALYSIS_RELATIVE): "f0f1c327ff1da0bfd74a3e604abbb3d8c243a0b0e7b41f6836ed689846c7426a",
    "artifacts/vla_wam_shared_v3/results/groot_n17_droid_phase_a_summary.json": "0db751e94a48caa5815aa6d656ff8cf062283eba47b3048c7c2ac37744581486",
    "artifacts/vla_wam_shared_v3/results/groot_n17_droid_phase_a_evidence_hash_manifest.json": "c7ebddea0413090b18cec6e2044251e929aeef77d24fa0102fdd4f9ac63c09bd",
    "artifacts/vla_wam_shared_v3/results/cosmos3_edge_policy_droid_phase_a_summary.json": "fc93b6518185641068ce51a4d81882dd1828ebeb9227b44c4ffc21a07b526c87",
    "artifacts/vla_wam_shared_v3/results/cosmos3_edge_policy_droid_phase_a_evidence_hash_manifest.json": "e37213a684ff07d379edc46754cb3167c476663921039f47e8f6ee3ff8c9556d",
    "artifacts/vla_wam_shared_v3/results/cosmos3_nano_policy_droid_phase_a_summary.json": "677c62845ff72bceb35b0c5e3e5b259a5c24da407648fe50f475656fe86cbcea",
    "artifacts/vla_wam_shared_v3/results/cosmos3_nano_policy_droid_phase_a_evidence_hash_manifest.json": "16c4d8afc40fa2f2417dbe128aff929642cf05d44d52c4687a6923de647aab64",
}


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_sources(root: Path) -> dict[str, str]:
    observed = {relative: sha256_file(root / relative) for relative in EXPECTED_SOURCE_HASHES}
    mismatched = {
        relative: {"expected": EXPECTED_SOURCE_HASHES[relative], "observed": digest}
        for relative, digest in observed.items()
        if digest != EXPECTED_SOURCE_HASHES[relative]
    }
    if mismatched:
        raise ValueError(f"source artifact hash mismatch: {mismatched}")
    registry = _load(root / REGISTRY_RELATIVE)
    analysis = _load(root / ANALYSIS_RELATIVE)
    if registry.get("status") != "frozen_separately_gated_before_any_phase_c_model_request_or_behavioral_inference":
        raise ValueError("source Phase-C registry is not frozen and unreleased")
    if registry.get("eligible_model_ids") != list(MODEL_CONTRACTS):
        raise ValueError("source Phase-C model order changed")
    if registry.get("prompt_forms") != PROMPTS:
        raise ValueError("source Phase-C prompt bytes changed")
    if analysis.get("claim_boundary") != (
        "This is a disclosed, checkpoint- and arena-specific prospective extension. Separate randomized ablation and wording blocks are exploratory unless a later amendment supplies their own power and inference plan."
    ):
        raise ValueError("exploratory Phase-C claim boundary changed")
    return observed


def build_cells() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id, model in MODEL_CONTRACTS.items():
        for seed in SEEDS:
            for order, (form, relation) in enumerate(randomized_conditions(model_id, seed), 1):
                prompt = PROMPTS[form][relation]
                rows.append(
                    {
                        "schema_version": f"{SCHEMA_PREFIX}-cell-v1",
                        "experiment_id": EXPERIMENT_ID,
                        "phase": "C_four_phrasings",
                        "arena": "droid_robolab",
                        "model_id": model_id,
                        "checkpoint": model["checkpoint"],
                        "checkpoint_revision": model["checkpoint_revision"],
                        "phase_a_runtime_identity_sha256": model["phase_a_runtime_identity_sha256"],
                        "registered_cell_id": f"v3c001:droid:{model_id}:seed{seed}:{form}:{relation}",
                        "seed_block_id": f"v3c001:droid:{model_id}:seed{seed}",
                        "pair_id": f"v3c001:droid:{model_id}:seed{seed}:{form}",
                        "seed": seed,
                        "environment_seed": seed,
                        "sampling_seed": seed,
                        "prompt_family": form,
                        "relation": relation,
                        "prompt": prompt,
                        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                        "prompt_sha256": prompt_sha256(prompt),
                        "within_seed_execution_order": order,
                        "randomization_namespace": RANDOMIZATION_NAMESPACE,
                        "static_episode_prompt": True,
                        "direct_command_within_seed_control": form == "direct_command",
                        "action_cap": model["action_cap"],
                        "action_horizon": model["action_horizon"],
                        "future_contract": model["future_contract"],
                        "behavioral_status": "prospectively_registered_not_released",
                    }
                )
    return validate_cells(rows)


def build_gate_requests() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id, model in MODEL_CONTRACTS.items():
        for form in PROMPT_FORMS:
            for request_order, (condition, relation) in enumerate(
                (("left", "left"), ("left_exact_repeat", "left"), ("right", "right")), 1
            ):
                prompt = PROMPTS[form][relation]
                rows.append(
                    {
                        "schema_version": f"{SCHEMA_PREFIX}-fixed-observation-request-v1",
                        "experiment_id": EXPERIMENT_ID,
                        "model_id": model_id,
                        "checkpoint": model["checkpoint"],
                        "checkpoint_revision": model["checkpoint_revision"],
                        "phase_a_runtime_identity_sha256": model["phase_a_runtime_identity_sha256"],
                        "probe_id": f"v3c001:fixed:{model_id}:{form}",
                        "observation_identity_requirement": "byte_identical_within_probe_id",
                        "prompt_family": form,
                        "condition": condition,
                        "relation": relation,
                        "request_order": request_order,
                        "sampling_seed_requirement": "identical_within_probe_id",
                        "prompt": prompt,
                        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                        "prompt_sha256": prompt_sha256(prompt),
                        "actions_required": True,
                        "decoded_future_required": model["fixed_observation_future_required"],
                        "behavioral_episode": False,
                    }
                )
    return rows


def _write_frozen_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite frozen V3-C001 artifact: {path}")
        return
    path.write_bytes(payload)


def _write_json(path: Path, value: Any) -> None:
    _write_frozen_bytes(path, canonical_json_bytes(value))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_frozen_bytes(path, b"".join(canonical_json_bytes(row) for row in rows))


def materialize(root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    source_hashes = validate_sources(root)
    output_dir = output_dir or root / OUTPUT_RELATIVE
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = build_cells()
    gate_requests = build_gate_requests()
    registration = {
        "schema_version": f"{SCHEMA_PREFIX}-registration-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "experiment_id": EXPERIMENT_ID,
        "frozen_at": FROZEN_AT,
        "status": "prospectively_frozen_not_released_no_phase_c_inference",
        "arena": "droid_robolab",
        "models": MODEL_CONTRACTS,
        "design": {
            "models": list(MODEL_CONTRACTS),
            "seeds": list(SEEDS),
            "matched_pairs_per_model": 20,
            "prompt_forms": list(PROMPT_FORMS),
            "relations": list(RELATIONS),
            "cells_per_seed_block": 8,
            "seed_blocks": 60,
            "episodes_per_model": 160,
            "behavioral_cells": 480,
            "same_seed_is_environment_and_sampling_seed": True,
            "direct_command_is_within_seed_control": True,
        },
        "randomization": {
            "method": "ascending_sha256_rank_over_all_eight_prompt_form_relation_conditions",
            "namespace": RANDOMIZATION_NAMESPACE,
            "unit": "model_id_and_seed",
            "seed_block_indivisible_across_execution_lanes": True,
            "generated_before_any_phase_c_model_request": True,
        },
        "exact_prompts": PROMPTS,
        "release_policy": {
            "status": "all_models_unreleased",
            "per_model_required_gates": [
                "exact_phase_a_direct_release_for_runtime_identity",
                "prompt_byte_hash",
                "fixed_observation_exact_repeat",
                "fixed_observation_prompt_only_sensitivity",
                "raw_video_action_jsonl_state_write",
            ],
            "no_behavior_before_release": True,
        },
        "raw_output_contract": {
            "behavioral_jsonl_per_episode": True,
            "executed_actions_per_episode": True,
            "simulator_viewport_video_per_episode": True,
            "state_trace_per_episode": True,
            "decoded_future_when_exposed": True,
            "infrastructure_failures_separate_from_behavioral_denominators": True,
            "valid_behavioral_failures_retained": True,
        },
        "analysis_status": {
            "wording_block": "exploratory",
            "confirmatory_inference": "not_registered",
            "constraint": "A later prospective amendment must add power and inference thresholds before any confirmatory wording claim.",
            "descriptive_outputs": [
                "raw success counts and Wilson intervals by checkpoint, prompt form, and direction",
                "within-seed direct-control contrasts for success and registered continuous measures",
                "exact matched discordance tables",
                "failure taxonomy by prompt form and direction",
            ],
        },
        "prohibited": [
            "dynamic_or_progress_conditioned_prompting",
            "cross_arena_pooling",
            "partial_seed_block_execution_as_a_complete_lane",
            "behavioral_inference_before_model_specific_release",
            "post_result_confirmatory_claims_without_a_later_prospective_amendment",
        ],
        "source_artifact_sha256": source_hashes,
    }
    registration_path = output_dir / "prospective_phase_c_v3c001_registration.json"
    cells_path = output_dir / "phase_c_v3c001_cells.jsonl"
    requests_path = output_dir / "phase_c_v3c001_fixed_observation_requests.jsonl"
    _write_json(registration_path, registration)
    _write_jsonl(cells_path, cells)
    _write_jsonl(requests_path, gate_requests)
    counts = Counter(row["model_id"] for row in cells)
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "experiment_id": EXPERIMENT_ID,
        "frozen_at": FROZEN_AT,
        "status": "hash_bound_registration_ready_models_unreleased",
        "counts": {
            "behavioral_cells": len(cells),
            "seed_blocks": len(MODEL_CONTRACTS) * len(SEEDS),
            "fixed_observation_requests": len(gate_requests),
            "cells_by_model": dict(counts),
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (registration_path, cells_path, requests_path)
        },
        "behavioral_inference_count_at_freeze": 0,
        "behavioral_release": False,
    }
    _write_json(output_dir / "phase_c_v3c001_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = materialize(args.repo_root.resolve(), args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
