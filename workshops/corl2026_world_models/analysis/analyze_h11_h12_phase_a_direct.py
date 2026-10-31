#!/usr/bin/env python3
"""Authenticate and package retained H11/H12 Phase-A direct-command evidence.

The shared V3 plan prospectively registered matched LEFT/RIGHT Phase-A direct
commands.  ABLATION_SPEC.md retains the completed 27-pair Cosmos3 Nano (H11)
and custom DreamZero-s2 (H12) additions as development/recovery sources.  The
six preserved V2 candidate cells per checkpoint failed the complete V3 runtime
identity requirement and are deliberately excluded from these denominators.

This adapter reads immutable compact Git objects only.  It authenticates the
frozen plan, complete 780-row Phase-A registry, model summaries, evidence
manifests and infrastructure ledgers; reconstructs all 108 selected episodes
and 54 matched pairs; and reconciles model-request, decoded-future, latent and
official-reset-decode accounting without pooling checkpoints.  It runs no
model, robot, remote retrieval, annotation or release workflow and advances no
forecast-ablation gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Mapping, Sequence


PIN = "ce561e66f82e95055e39d3d7711691982f6b2086"
PROSPECTIVE_FREEZE_COMMIT = "0b24153a2ba1fff9173fc47937ad291d24959e72"
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
SPEC_PATH = PACKAGE / "docs" / "ABLATION_SPEC.md"
DEFAULT_JSON = PACKAGE / "paper" / "data" / "h11_h12_phase_a_direct.json"
DEFAULT_CSV = PACKAGE / "paper" / "data" / "h11_h12_phase_a_direct.csv"

STUDY = "vla_wam_language_steerability_v3"
SEEDS = tuple(range(8303, 8330))
PRESERVED_SEEDS = (8300, 8301, 8302)
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
PROMPT_SHA256 = {
    relation: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    for relation, prompt in PROMPTS.items()
}

BASE = "artifacts/vla_wam_shared_v3"
ANALYSIS_PLAN = f"{BASE}/analysis_plan.json"
CELLS = f"{BASE}/phase_a_cells.jsonl"
CELLS_MANIFEST = f"{BASE}/phase_a_cells_manifest.json"
SYMMETRY_REGISTRATION = (
    f"{BASE}/phase_e/symmetric_layout_cohort_v3e004/registration.json"
)

MODELS: dict[str, dict[str, Any]] = {
    "cosmos3_nano_policy_droid": {
        "cohort": "H11",
        "label": "Cosmos3 Nano g3",
        "summary": f"{BASE}/results/cosmos3_nano_policy_droid_phase_a_summary.json",
        "evidence": f"{BASE}/results/cosmos3_nano_policy_droid_phase_a_evidence_hash_manifest.json",
        "infrastructure": f"{BASE}/results/cosmos3_nano_policy_droid_phase_a_infrastructure_ledger.json",
        "result_commit": "150a362701656a3992c9738f497f1fd09605dc54",
        "registry_rows": [67, 120],
        "preserved_rows": [61, 66],
        "expected_successes": {"left": 26, "right": 25},
        "expected_requests": {"left": 179, "right": 170},
        "expected_failures": 3,
    },
    "dreamzero_droid_action_cfg": {
        "cohort": "H12",
        "label": "DreamZero custom action guidance s2",
        "summary": f"{BASE}/results/dreamzero_droid_action_cfg_phase_a_summary.json",
        "evidence": f"{BASE}/results/dreamzero_droid_action_cfg_phase_a_evidence_hash_manifest.json",
        "infrastructure": f"{BASE}/results/dreamzero_droid_action_cfg_phase_a_infrastructure_ledger.json",
        "result_commit": "8651aef704f6369b5b514d17ba7fd4bd0913b8d3",
        "registry_rows": [127, 180],
        "preserved_rows": [121, 126],
        "expected_successes": {"left": 3, "right": 17},
        "expected_requests": {"left": 1_470, "right": 1_084},
        "expected_failures": 34,
    },
}

EXPECTED_SOURCES: dict[str, dict[str, Any]] = {
    ANALYSIS_PLAN: {
        "git_blob": "27fd9d2f601eae1fc6ba83a8d7bf5fb70270344e",
        "sha256": "f0f1c327ff1da0bfd74a3e604abbb3d8c243a0b0e7b41f6836ed689846c7426a",
        "bytes": 2_693,
        "lines": 24,
    },
    CELLS: {
        "git_blob": "5908613a95509dd1cfb68d8d31404c9708b2a087",
        "sha256": "8350b98f958424b56b66e67e8c70ec3951d27f4ae257476d6f08c0aaa873cb7c",
        "bytes": 1_252_888,
        "lines": 780,
    },
    CELLS_MANIFEST: {
        "git_blob": "1d08ffaaac30c61f0d554ae0d1b3dc88838e7b19",
        "sha256": "abbe1a50e092f153292cc87016a37859b153bd819e467fb396a275bed6528fef",
        "bytes": 2_974,
        "lines": 102,
    },
    SYMMETRY_REGISTRATION: {
        "git_blob": "758a77c1227b43c41c66e3ffea27882c78382f02",
        "sha256": "3e3786510ebba933d0c7a4606afa51627ea6bf706f9e64b34cfd74b29e318db2",
        "bytes": 16_100,
        "lines": 359,
    },
    MODELS["cosmos3_nano_policy_droid"]["summary"]: {
        "git_blob": "75c012124b45599c9963212f64e9735aaed7aabf",
        "sha256": "677c62845ff72bceb35b0c5e3e5b259a5c24da407648fe50f475656fe86cbcea",
        "bytes": 210_426,
        "lines": 4_804,
    },
    MODELS["cosmos3_nano_policy_droid"]["evidence"]: {
        "git_blob": "ebaa77bf5ec5d3c9ce4a4171904ef46583a5ad88",
        "sha256": "16c4d8afc40fa2f2417dbe128aff929642cf05d44d52c4687a6923de647aab64",
        "bytes": 663_149,
        "lines": 13_683,
    },
    MODELS["cosmos3_nano_policy_droid"]["infrastructure"]: {
        "git_blob": "98c0d4fc5ecdcd7dfda59af3f5383d89f5f78916",
        "sha256": "0ce81ac664c5187b69f96106c25a0b6f30852fa738aa6decfaef09fd8d51bd5c",
        "bytes": 19_867,
        "lines": 457,
    },
    MODELS["dreamzero_droid_action_cfg"]["summary"]: {
        "git_blob": "54637eab1ce615cf9842c59e15ff13d19287c7bb",
        "sha256": "50ec800009b1f128ce8fe877c186b08a47ec3021feaa7cf4f8baab984549b120",
        "bytes": 375_241,
        "lines": 6_381,
    },
    MODELS["dreamzero_droid_action_cfg"]["evidence"]: {
        "git_blob": "520970ad7accad89d42e6d742b20f9c49097089f",
        "sha256": "0150afb2145cc2e1d18bc16bd34b4af404c7fabb77486d197af78776fcd93427",
        "bytes": 3_088_542,
        "lines": 73_879,
    },
    MODELS["dreamzero_droid_action_cfg"]["infrastructure"]: {
        "git_blob": "3ccca43b57cb24d06d781383fc6956d2806f1c98",
        "sha256": "97305ed45b616306788f64be692be4fba77824ce3ef88816d6cd67f15f3b28da",
        "bytes": 23_087,
        "lines": 589,
    },
}
HELPER_SOURCES = {ANALYSIS_PLAN, SYMMETRY_REGISTRATION}
PAYLOAD_SOURCES = set(EXPECTED_SOURCES) - HELPER_SOURCES

PLAN_KEYS = {
    "claim_boundary", "hierarchical_16_rollout_analysis",
    "infrastructure_and_latency", "primary_estimands", "reporting",
    "schema_version", "status", "study_id",
}
MANIFEST_KEYS = {
    "launch_rule", "phase", "queue_file", "queue_sha256", "row_counts",
    "schema_version", "source_registry_sha256", "status_counts", "study_id",
    "target_counts",
}
DROID_CELL_KEYS = {
    "anchor_task", "arena", "cell_id", "environment_seed", "execution_status",
    "model_id", "pair_id", "phase", "prompt", "prompt_family",
    "prompt_fixture_id", "prompt_sha256", "relation", "replicate",
    "required_raw_outputs", "reset_identity", "runtime_identity_requirement",
    "sampling_seed", "schema_version", "status", "status_reason", "study_id",
    "success_predicate_id",
}
ROBOTWIN_EXTRA_KEYS = {"scene_pair", "seen_object_description", "seen_reference_description"}

NANO_SUMMARY_KEYS = {
    "action_rms_common_prefix", "all_continuous_field_summaries", "arena", "cells",
    "claim_boundary", "denominator_boundary", "directional", "endpoint_ordering_counts",
    "future_retention", "infrastructure", "initial_state_exact_match_pair_count",
    "matched_discordance", "model_id", "overall_failure_taxonomy_counts",
    "overall_frozen_failure_stage_counts", "pairs", "registered_execution_contract",
    "right_minus_left_raw_robot_y_shift_m", "schema_version", "status", "study_id",
}
NANO_CELL_KEYS = {
    "action_cap", "actions_executed", "failure_taxonomy", "frozen_failure_stage",
    "future_request_count", "initial_state_sha256", "measurements",
    "operational_wall_time_valid", "pair_id", "predicate_id", "prompt",
    "prompt_family", "prompt_sha256", "raw_behavioral_pair_jsonl_sha256",
    "registered_cell_id", "relation", "right_censored", "runtime_identity_sha256",
    "seed", "success",
}
NANO_MEASUREMENT_KEYS = {
    "actions_executed", "entry_kind", "episode_length_steps", "executed_action_count",
    "final_detached_release", "final_opposite_region_sustained",
    "final_opposite_signed_margin_m", "final_requested_region_sustained",
    "final_requested_signed_margin_m", "first_cone_or_native_region_entry_step",
    "first_contact_status", "first_contact_step", "first_contact_unavailable_reason",
    "first_opposite_entry_step", "first_requested_entry_step",
    "first_requested_region_step", "first_sustained_opposite_entry_step",
    "first_sustained_requested_entry_step", "first_verified_pickup_step",
    "max_object_lift_m", "maximum_pickup_height_m", "maximum_requested_signed_margin_m",
    "object_path_length_m", "operational_wall_time_valid", "opposite_entry_kind",
    "pickup_lift_threshold_m", "region_kind", "requested_entry_kind",
    "requested_final_lateral_margin_m", "right_censored", "scored_state_sample_count",
    "signed_final_lateral_offset_m", "sustained_steps", "verified_pickup", "wall_time_s",
}
NANO_PAIR_KEYS = {
    "action_rms_common_prefix", "common_prefix_actions", "endpoint_ordering",
    "executed_actions_distinct", "future_request_counts", "initial_state_exact_match",
    "left_raw_robot_y_m", "left_success", "left_taxonomy", "progression",
    "right_minus_left_raw_robot_y_shift_m", "right_raw_robot_y_m", "right_success",
    "right_taxonomy", "seed",
}
NANO_EVIDENCE_KEYS = {
    "assembler_instrumentation", "completion_manifest", "episode_count",
    "fixed_observation_gate", "infrastructure_ledger", "model_id", "pair_count", "pairs",
    "phase_a_registry", "phase_a_registry_manifest", "queue_launcher_instrumentation",
    "queue_progress_normalized", "queue_progress_raw", "runtime_identity",
    "runtime_sidecar_instrumentation", "schema_version", "source_study_commit", "study_id",
    "summary", "verification",
}
NANO_EVIDENCE_PAIR_KEYS = {
    "behavioral_jsonl", "behavioral_jsonl_manifest", "cells", "compiler_manifest_stdout",
    "pair_id", "seed", "thermal_events",
}
NANO_EVIDENCE_CELL_KEYS = {
    "action_trace_metadata", "executed_action_trace", "future_request_count",
    "future_requests", "registered_cell_id", "simulator_export", "simulator_log",
    "source_artifacts", "state_stream_partial", "viewport_video", "warmup_reset",
}
NANO_INFRA_KEYS = {
    "behavioral_denominator_excludes_all_entries", "completed_guarded_pairs",
    "excluded_attempt_count", "excluded_attempts", "finalizer_and_normalization", "model_id",
    "operational_event_count", "operational_events", "operational_wall_time_valid_episode_count",
    "post_queue_gpu_recycle", "queue_infrastructure_invalid_count", "runtime_intervention_count",
    "schema_version", "server_shutdown_after_queue_close", "study_id", "thermal_guard_pair_count",
}

DREAM_SUMMARY_KEYS = {
    "action_rms_common_prefix", "actual_rollout_video_count", "arena", "cells",
    "denominator_boundary", "directional", "directional_success_rate_difference_right_minus_left",
    "endpoint_ordering_counts", "endpoint_ordering_exact_two_sided_sign_test_p",
    "infrastructure", "initial_state_exact_match_pair_count", "matched_discordance", "model_id",
    "official_full_reset_decode_count", "overall_failure_taxonomy_counts",
    "overall_frozen_failure_stage_counts", "pairs", "registered_execution_contract",
    "retained_latent_future_count", "right_minus_left_endpoint_shift_m",
    "right_minus_left_mean_requested_margin_m", "schema_version", "status", "study_id",
}
DREAM_CELL_KEYS = {
    "action_cap", "action_quality", "actions_executed", "cone_or_native_region_entry_step",
    "episode_length_steps", "failure_taxonomy", "final_detached_release",
    "final_requested_region_sustained", "final_requested_signed_margin_m",
    "first_contact_status", "first_contact_unavailable_reason", "first_requested_entry_step",
    "first_sustained_requested_entry_step", "frozen_failure_stage", "hashes",
    "initial_state_sha256", "latent_future_count", "latent_shape_counts",
    "maximum_requested_signed_margin_m", "object_path_length_m", "official_reset_decode_count",
    "operational_wall_time_valid", "pair_id", "policy_request_count", "prompt", "prompt_sha256",
    "registered_cell_id", "relation", "requested_entry_kind", "requested_success",
    "right_censored", "seed", "signed_final_lateral_offset_m", "time_to_first_contact_step",
    "time_to_first_verified_pickup_step", "wall_time_s",
}
DREAM_PAIR_KEYS = {
    "action_rms_common_prefix", "common_prefix_actions", "endpoint_ordering",
    "initial_state_exact_match", "left_failure_taxonomy", "left_success",
    "right_failure_taxonomy", "right_minus_left_endpoint_shift_m",
    "right_minus_left_requested_margin_m", "right_success", "seed",
}
DREAM_EVIDENCE_KEYS = {
    "actual_rollout_video_count", "audit_compiler", "content_audit", "environment_lock",
    "episode_count", "model_id", "official_full_reset_decode_count", "pair_count",
    "pair_launcher", "pairs", "preserved_v2_seed_evidence_not_pooled", "queue_complete",
    "queue_launcher", "queue_progress", "release_gate", "retained_latent_future_count",
    "runtime_identity", "runtime_instrumentation_manifest", "runtime_tools_manifest",
    "schema_version", "server_shutdown", "study_commit", "study_id", "study_source_files",
}
DREAM_EVIDENCE_PAIR_KEYS = {
    "attempt", "cells", "episode_results", "pair_id", "pair_manifest", "runtime_identity",
    "seed", "thermal_events",
}
DREAM_EVIDENCE_CELL_KEYS = {
    "action_trace_manifest", "compiled_jsonl", "compiled_jsonl_manifest",
    "executed_action_trace", "exposed_future_manifest", "official_reset_decodes",
    "pair_manifest_relation_artifacts", "policy_requests", "registered_cell_id", "relation",
    "returned_executable_action_chunks", "returned_raw_action_chunks", "viewport_video",
}
DREAM_INFRA_KEYS = {
    "behavioral_denominator_excludes_all_entries", "completed_guarded_pairs",
    "excluded_attempt_count", "excluded_attempts", "model_id",
    "runtime_instrumentation_amendment", "runtime_intervention_count", "schema_version",
    "server_shutdown", "study_id", "thermal_guard_pair_count",
}


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "--no-replace-objects", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    result = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(f"prospective freeze ancestry check failed: {ancestor} -> {descendant}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON object key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _reject_nonfinite_tree(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number in {label}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite_tree(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_tree(item, f"{label}[{index}]")


def parse_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    _reject_nonfinite_tree(value, label)
    return value


def _parse_jsonl(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL row in {label}:{number}")
        row = parse_json(line, label=f"{label}:{number}")
        row["_source_row"] = number
        rows.append(row)
    return rows


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} schema changed: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _close(observed: Any, expected: float, label: str, *, tolerance: float = 1e-12) -> None:
    if not math.isclose(_finite(observed, label), expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} disagrees with compact-row reproduction")


def _valid_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def read_pinned_sources(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """Byte-authenticate and strictly parse the compact Git evidence."""

    root = Path(root)
    resolved = _git(root, "rev-parse", "--verify", PIN + "^{commit}").decode().strip()
    if resolved != PIN:
        raise ValueError("resolved historical evidence commit differs from fixed pin")
    _require_ancestor(root, PROSPECTIVE_FREEZE_COMMIT, PIN)
    for path in (ANALYSIS_PLAN, CELLS, CELLS_MANIFEST):
        frozen = _git(root, "rev-parse", "--verify", f"{PROSPECTIVE_FREEZE_COMMIT}:{path}").decode().strip()
        if frozen != EXPECTED_SOURCES[path]["git_blob"]:
            raise ValueError(f"prospective Phase-A blob changed after freeze: {path}")
    for model, info in MODELS.items():
        result_commit = info["result_commit"]
        _require_ancestor(root, PROSPECTIVE_FREEZE_COMMIT, result_commit)
        _require_ancestor(root, result_commit, PIN)
        for path_key in ("summary", "evidence", "infrastructure"):
            path = info[path_key]
            result_blob = _git(root, "rev-parse", "--verify", f"{result_commit}:{path}").decode().strip()
            if result_blob != EXPECTED_SOURCES[path]["git_blob"]:
                raise ValueError(f"{model} result blob changed after its result commit: {path}")

    raw_by_path: dict[str, bytes] = {}
    observed_files: dict[str, dict[str, Any]] = {}
    for path, expected in EXPECTED_SOURCES.items():
        object_id = _git(root, "rev-parse", "--verify", f"{PIN}:{path}").decode().strip()
        if object_id != expected["git_blob"]:
            raise ValueError(f"pinned Git blob mismatch for {path}")
        raw = _git(root, "cat-file", "blob", object_id)
        observed = {
            "git_blob": object_id,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "lines": len(raw.splitlines()),
        }
        if observed != expected:
            raise ValueError(f"pinned source bytes disagree with fixed identity for {path}")
        raw_by_path[path] = raw
        observed_files[path] = observed

    documents: dict[str, Any] = {
        "analysis_plan": parse_json(raw_by_path[ANALYSIS_PLAN], label=ANALYSIS_PLAN),
        "cells_manifest": parse_json(raw_by_path[CELLS_MANIFEST], label=CELLS_MANIFEST),
        "symmetry_registration": parse_json(
            raw_by_path[SYMMETRY_REGISTRATION], label=SYMMETRY_REGISTRATION
        ),
        "cells": _parse_jsonl(raw_by_path[CELLS], label=CELLS),
        "models": {},
        "ablation_spec": SPEC_PATH.read_text(encoding="utf-8"),
    }
    for model, info in MODELS.items():
        documents["models"][model] = {
            "summary": parse_json(raw_by_path[info["summary"]], label=info["summary"]),
            "evidence": parse_json(raw_by_path[info["evidence"]], label=info["evidence"]),
            "infrastructure": parse_json(
                raw_by_path[info["infrastructure"]], label=info["infrastructure"]
            ),
        }
    provenance = {
        "commit": PIN,
        "prospective_freeze_commit": PROSPECTIVE_FREEZE_COMMIT,
        "freeze_is_ancestor_of_pinned_evidence": True,
        "frozen_plan_registry_and_manifest_blobs_unchanged": True,
        "result_commits": {
            info["cohort"]: info["result_commit"] for info in MODELS.values()
        },
        "result_commits_are_ancestors_of_pinned_evidence": True,
        "result_blobs_unchanged_at_pin": True,
        "files": observed_files,
    }
    return documents, provenance


def _validate_ablation_scope(text: str) -> None:
    required = (
        "| H11 | Nano V3 Phase-A direct commands | 54 new episodes, excluding the six historical V2 baseline episodes. 51/54 successes. | Additional development/recovery source |",
        "| H12 | DreamZero-s2 V3 Phase-A direct commands | 54 new episodes; LEFT 3/27, RIGHT 17/27. 2,554 latent-future records and 54 official full-reset decodes recorded. | Recovery source; latent records are not per-request decoded RGB |",
        "H03/H04 include previously collected baselines; do not count those baselines again when totaling another cohort.",
        "No historical collection above is an unseen confirmation set for the new forecast metric.",
    )
    for sentence in required:
        if sentence not in text:
            raise ValueError("ABlation-spec H11/H12 scope binding changed")


def _validate_registration(documents: Mapping[str, Any]) -> dict[str, dict[tuple[int, str], Mapping[str, Any]]]:
    _validate_ablation_scope(documents["ablation_spec"])
    plan = documents["analysis_plan"]
    _require_keys(plan, PLAN_KEYS, "shared analysis plan")
    if plan.get("study_id") != STUDY or plan.get("status") != "frozen_before_any_v3_behavioral_inference":
        raise ValueError("shared Phase-A plan is not the frozen pre-inference record")
    droid_estimand = plan.get("primary_estimands", {}).get("droid", "")
    if not all(term in droid_estimand for term in ("requested-relation-and-release success", "exact matched pairs", "27-pair prospective addition")):
        raise ValueError("registered DROID Phase-A estimand changed")
    reporting = plan.get("reporting", {})
    if reporting.get("binary") != [
        "raw successes/valid denominator",
        "Wilson 95% interval",
        "exact matched discordance table",
    ]:
        raise ValueError("registered binary reporting plan changed")
    for metric in ("final requested signed margin", "paired RIGHT-minus-LEFT endpoint shift"):
        if metric not in reporting.get("paired_continuous", []):
            raise ValueError(f"registered continuous metric missing: {metric}")
    if "Never combine DROID and RoboTwin" not in plan.get("primary_estimands", {}).get("no_pooling", ""):
        raise ValueError("registered no-pooling boundary changed")

    manifest = documents["cells_manifest"]
    _require_keys(manifest, MANIFEST_KEYS, "Phase-A cells manifest")
    if manifest.get("study_id") != STUDY or manifest.get("phase") != "A_direct_command_matched_pairs_and_sampling_replication":
        raise ValueError("Phase-A manifest identity changed")
    if manifest.get("queue_file") != CELLS or manifest.get("queue_sha256") != EXPECTED_SOURCES[CELLS]["sha256"]:
        raise ValueError("Phase-A manifest queue identity changed")
    expected_counts = {
        "by_arena": {"droid_robolab": 360, "robotwin": 420},
        "by_model": {
            "cosmos3_edge_policy_droid": 60,
            "cosmos3_nano_policy_droid": 60,
            "dreamzero_droid_action_cfg": 60,
            "efficient_wam_rt_robotwin": 140,
            "fastwam_robotwin": 140,
            "groot_n17_droid_vla": 60,
            "lingbot_va_robotwin": 140,
            "pi05_current_stack_droid": 60,
            "pi0_fast_droid_vla": 60,
        },
        "by_status": {
            "authorized_new": 648,
            "blocked_pi0": 40,
            "preserved_candidate": 50,
            "preserved_r0": 42,
        },
        "total": 780,
    }
    if manifest.get("row_counts") != expected_counts:
        raise ValueError("Phase-A manifest population changed")

    symmetry = documents["symmetry_registration"]
    if set(symmetry) != {
        "amendment_id", "analysis_plan", "behavioral_episode_count_before_registration",
        "design", "estimands", "exact_prompts", "layout", "model_request_count_before_registration",
        "power_registration", "pre_inference_correction", "pre_inference_layout_gate_correction",
        "prior_evidence_policy", "queue", "registered_at_utc", "registered_predictions",
        "release_boundary", "schema_version", "sign_convention", "source_bindings", "status",
        "study_id", "success_predicates_frozen", "supersedes",
    }:
        raise ValueError("V3-E004 symmetry registration schema changed")
    symmetry_seeds = symmetry.get("design", {}).get("core_seeds")
    if symmetry.get("study_id") != STUDY or symmetry.get("amendment_id") != "V3-E004" or symmetry_seeds != list(range(9400, 9427)):
        raise ValueError("H06/H07/H08 canonical core population changed")
    if set(SEEDS) & set(symmetry_seeds):
        raise ValueError("H11/H12 population overlaps H06/H07/H08 canonical core")

    cells = documents["cells"]
    if len(cells) != 780:
        raise ValueError("Phase-A registry must contain exactly 780 cells")
    ids: set[str] = set()
    observed_arena: Counter[str] = Counter()
    observed_model: Counter[str] = Counter()
    observed_status: Counter[str] = Counter()
    for row in cells:
        source_row = row.get("_source_row")
        payload_keys = set(row) - {"_source_row"}
        expected_keys = DROID_CELL_KEYS if row.get("arena") == "droid_robolab" else DROID_CELL_KEYS | ROBOTWIN_EXTRA_KEYS
        if payload_keys != expected_keys:
            raise ValueError(f"Phase-A registry row schema changed at {source_row}")
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or cell_id in ids:
            raise ValueError(f"missing or duplicate Phase-A registry cell at {source_row}")
        ids.add(cell_id)
        observed_arena[str(row.get("arena"))] += 1
        observed_model[str(row.get("model_id"))] += 1
        observed_status[str(row.get("status"))] += 1
    if dict(sorted(observed_arena.items())) != expected_counts["by_arena"]:
        raise ValueError("Phase-A registry arena counts changed")
    if dict(sorted(observed_model.items())) != expected_counts["by_model"]:
        raise ValueError("Phase-A registry model counts changed")
    if dict(sorted(observed_status.items())) != expected_counts["by_status"]:
        raise ValueError("Phase-A registry status counts changed")

    selected_by_model: dict[str, dict[tuple[int, str], Mapping[str, Any]]] = {}
    for model, info in MODELS.items():
        model_rows = [row for row in cells if row.get("model_id") == model]
        if len(model_rows) != 60:
            raise ValueError(f"Phase-A model registry must contain 60 cells: {model}")
        preserved = [row for row in model_rows if row.get("status") == "preserved_candidate"]
        selected = [row for row in model_rows if row.get("status") == "authorized_new"]
        if len(preserved) != 6 or len(selected) != 54:
            raise ValueError(f"Phase-A preserved/new split changed for {model}")
        if [row["_source_row"] for row in preserved] != list(range(info["preserved_rows"][0], info["preserved_rows"][1] + 1)):
            raise ValueError(f"Phase-A preserved source rows changed for {model}")
        if [row["_source_row"] for row in selected] != list(range(info["registry_rows"][0], info["registry_rows"][1] + 1)):
            raise ValueError(f"Phase-A selected source rows changed for {model}")
        if {row.get("environment_seed") for row in preserved} != set(PRESERVED_SEEDS):
            raise ValueError(f"preserved V2 seed membership changed for {model}")
        if any(row.get("execution_status") != "preserved_evidence_only_runtime_identity_check" for row in preserved):
            raise ValueError(f"preserved V2 execution boundary changed for {model}")
        by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
        for row in selected:
            seed, relation = row.get("environment_seed"), row.get("relation")
            key = (seed, relation)
            if seed not in SEEDS or relation not in RELATIONS or key in by_key:
                raise ValueError(f"invalid or duplicate selected Phase-A cell: {model}:{key}")
            if row.get("sampling_seed") != seed or row.get("replicate") != 0:
                raise ValueError(f"Phase-A seed/replicate contract changed: {model}:{key}")
            if row.get("status") != "authorized_new" or row.get("execution_status") != "authorized_after_all_registered_release_gates":
                raise ValueError(f"selected Phase-A authorization changed: {model}:{key}")
            expected_pair = f"v3:droid:{model}:seed{seed}"
            expected_cell = f"{expected_pair}:{relation}"
            if row.get("pair_id") != expected_pair or row.get("cell_id") != expected_cell:
                raise ValueError(f"selected Phase-A identity changed: {model}:{key}")
            if row.get("arena") != "droid_robolab" or row.get("phase") != "A_direct_command_matched_pairs":
                raise ValueError(f"selected Phase-A arena/phase changed: {model}:{key}")
            if row.get("prompt_family") != "direct_command" or row.get("prompt") != PROMPTS[relation] or row.get("prompt_sha256") != PROMPT_SHA256[relation]:
                raise ValueError(f"selected Phase-A prompt changed: {model}:{key}")
            by_key[key] = row
        expected_grid = {(seed, relation) for seed in SEEDS for relation in RELATIONS}
        if set(by_key) != expected_grid:
            raise ValueError(f"selected Phase-A 27-pair grid incomplete for {model}")
        selected_by_model[model] = by_key
    return selected_by_model


def _wilson(successes: int, trials: int) -> list[float]:
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [center - radius, center + radius]


def _exact_two_sided_equal_binomial(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("numeric summary requires nonempty finite values")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _validate_common_summary(model: str, summary: Mapping[str, Any]) -> None:
    if summary.get("study_id") != STUDY or summary.get("model_id") != model or summary.get("arena") != "droid_robolab":
        raise ValueError(f"Phase-A summary identity changed for {model}")
    if summary.get("status") != "complete_27_new_matched_pairs":
        raise ValueError(f"Phase-A summary is not complete for {model}")
    boundary = summary.get("denominator_boundary", {})
    audit = boundary.get("preserved_identity_audit", {})
    if boundary.get("v3_behavioral_episode_count") != 54 or boundary.get("v3_pair_count") != 27:
        raise ValueError(f"V3 Phase-A denominator changed for {model}")
    if model == "cosmos3_nano_policy_droid":
        if boundary.get("preserved_seeds_8300_8302") != "v2_only_not_in_v3_denominator":
            raise ValueError(f"preserved V2 denominator boundary changed for {model}")
        if boundary.get("v3_new_pair_seeds") != [8303, 8329] or audit.get("candidate_cell_count") != 6:
            raise ValueError(f"new/preserved Phase-A seed range changed for {model}")
        if audit.get("candidate_seeds") != list(PRESERVED_SEEDS):
            raise ValueError(f"preserved V2 seed membership changed for {model}")
    else:
        if boundary.get("preserved_v2_seeds_8300_8302") != "excluded_from_v3_denominator_and_retained_as_separate_v2_a015_evidence":
            raise ValueError(f"preserved V2 denominator boundary changed for {model}")
        if boundary.get("v3_new_pair_seeds_inclusive") != [8303, 8329]:
            raise ValueError(f"new Phase-A seed range changed for {model}")
    if audit.get("decision") != "not_carried_forward":
        raise ValueError(f"preserved V2 candidates were carried into {model}")


def _normalize_nano(
    model_docs: Mapping[str, Any], registry: Mapping[tuple[int, str], Mapping[str, Any]]
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[int, Mapping[str, Any]]]:
    model = "cosmos3_nano_policy_droid"
    summary, evidence, infra = (
        model_docs["summary"], model_docs["evidence"], model_docs["infrastructure"]
    )
    _require_keys(summary, NANO_SUMMARY_KEYS, "H11 summary")
    _require_keys(evidence, NANO_EVIDENCE_KEYS, "H11 evidence manifest")
    _require_keys(infra, NANO_INFRA_KEYS, "H11 infrastructure ledger")
    _validate_common_summary(model, summary)
    if summary.get("schema_version") != "vla-wam-v3-cosmos3-nano-policy-droid-phase-a-summary-v1":
        raise ValueError("H11 summary schema version changed")
    contract = summary.get("registered_execution_contract", {})
    if contract.get("checkpoint") != {
        "id": "nvidia/Cosmos3-Nano-Policy-DROID",
        "revision": "6706d7680581c255ff61e0f3bb49d90eac55c79e",
    } or contract.get("open_loop_horizon") != 32:
        raise ValueError("H11 checkpoint/horizon contract changed")
    if contract.get("future_interface") != "decoded_rgb_uint8_33_frames_per_policy_request":
        raise ValueError("H11 decoded-future interface changed")

    cells = summary.get("cells")
    pairs = summary.get("pairs")
    if not isinstance(cells, list) or len(cells) != 54 or not isinstance(pairs, list) or len(pairs) != 27:
        raise ValueError("H11 summary population must be 54 episodes/27 pairs")
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for source_row, row in enumerate(cells, 1):
        _require_keys(row, NANO_CELL_KEYS, f"H11 summary cell {source_row}")
        measurements = row.get("measurements")
        if not isinstance(measurements, dict):
            raise ValueError(f"H11 cell measurements missing at {source_row}")
        _require_keys(measurements, NANO_MEASUREMENT_KEYS, f"H11 measurements {source_row}")
        seed, relation = row.get("seed"), row.get("relation")
        key = (seed, relation)
        if key not in registry or key in by_key:
            raise ValueError(f"H11 invalid or duplicate compact episode: {key}")
        registered = registry[key]
        for summary_field, registry_field in (
            ("registered_cell_id", "cell_id"), ("pair_id", "pair_id"),
            ("prompt", "prompt"), ("prompt_sha256", "prompt_sha256"),
        ):
            if row.get(summary_field) != registered.get(registry_field):
                raise ValueError(f"H11 compact episode/registry mismatch: {key}:{summary_field}")
        if row.get("prompt_family") != "direct_command" or type(row.get("success")) is not bool:
            raise ValueError(f"H11 prompt/success field changed: {key}")
        requests = _positive_int(row.get("future_request_count"), f"H11 {key} future_request_count")
        actions = _positive_int(row.get("actions_executed"), f"H11 {key} actions_executed")
        if actions > 450 or row.get("action_cap") != 450:
            raise ValueError(f"H11 action cap/count changed: {key}")
        if any(measurements.get(name) != actions for name in ("actions_executed", "executed_action_count", "episode_length_steps")):
            raise ValueError(f"H11 action accounting changed: {key}")
        margin = _finite(measurements.get("final_requested_signed_margin_m"), f"H11 {key} margin")
        endpoint = _finite(measurements.get("signed_final_lateral_offset_m"), f"H11 {key} endpoint")
        _close(margin, endpoint if relation == "left" else -endpoint, f"H11 {key} margin sign")
        _close(measurements.get("requested_final_lateral_margin_m"), margin, f"H11 {key} margin alias")
        if not _valid_sha(row.get("initial_state_sha256")) or not _valid_sha(row.get("runtime_identity_sha256")):
            raise ValueError(f"H11 compact hash identity changed: {key}")
        normalized = dict(row)
        normalized.update({"request_count": requests, "latent_count": 0, "decode_count": requests, "margin_m": margin, "endpoint_m": endpoint, "_source_row": source_row, "_registry_source_row": registered["_source_row"]})
        by_key[key] = normalized
    if set(by_key) != set(registry):
        raise ValueError("H11 compact 27-pair episode grid is incomplete")

    by_seed: dict[int, Mapping[str, Any]] = {}
    for source_row, pair in enumerate(pairs, 1):
        _require_keys(pair, NANO_PAIR_KEYS, f"H11 pair {source_row}")
        seed = pair.get("seed")
        if seed not in SEEDS or seed in by_seed:
            raise ValueError(f"H11 invalid or duplicate summary pair: {seed}")
        left, right = by_key[(seed, "left")], by_key[(seed, "right")]
        if pair.get("left_success") is not left["success"] or pair.get("right_success") is not right["success"]:
            raise ValueError(f"H11 pair success mismatch: {seed}")
        if pair.get("left_taxonomy") != left["failure_taxonomy"] or pair.get("right_taxonomy") != right["failure_taxonomy"]:
            raise ValueError(f"H11 pair taxonomy mismatch: {seed}")
        if pair.get("future_request_counts") != {"left": left["request_count"], "right": right["request_count"]}:
            raise ValueError(f"H11 pair request accounting mismatch: {seed}")
        if pair.get("initial_state_exact_match") is not True or left["initial_state_sha256"] != right["initial_state_sha256"]:
            raise ValueError(f"H11 pair initial-state identity changed: {seed}")
        shift = right["endpoint_m"] - left["endpoint_m"]
        _close(pair.get("left_raw_robot_y_m"), left["endpoint_m"], f"H11 {seed} left endpoint")
        _close(pair.get("right_raw_robot_y_m"), right["endpoint_m"], f"H11 {seed} right endpoint")
        _close(pair.get("right_minus_left_raw_robot_y_shift_m"), shift, f"H11 {seed} endpoint shift")
        expected_order = "aligned" if shift < 0 else "anti_aligned" if shift > 0 else "tie"
        if pair.get("endpoint_ordering") != expected_order:
            raise ValueError(f"H11 endpoint ordering mismatch: {seed}")
        by_seed[seed] = pair

    if evidence.get("study_id") != STUDY or evidence.get("model_id") != model or evidence.get("source_study_commit") != PROSPECTIVE_FREEZE_COMMIT:
        raise ValueError("H11 evidence identity/freeze binding changed")
    if evidence.get("episode_count") != 54 or evidence.get("pair_count") != 27:
        raise ValueError("H11 evidence population changed")
    if evidence.get("phase_a_registry", {}).get("sha256") != EXPECTED_SOURCES[CELLS]["sha256"] or evidence.get("phase_a_registry", {}).get("bytes") != EXPECTED_SOURCES[CELLS]["bytes"]:
        raise ValueError("H11 evidence registry binding changed")
    if evidence.get("phase_a_registry_manifest", {}).get("sha256") != EXPECTED_SOURCES[CELLS_MANIFEST]["sha256"] or evidence.get("summary", {}).get("sha256") != EXPECTED_SOURCES[MODELS[model]["summary"]]["sha256"]:
        raise ValueError("H11 evidence manifest/summary binding changed")
    if evidence.get("infrastructure_ledger", {}).get("sha256") != EXPECTED_SOURCES[MODELS[model]["infrastructure"]]["sha256"]:
        raise ValueError("H11 infrastructure ledger binding changed")
    evidence_pairs = evidence.get("pairs")
    if not isinstance(evidence_pairs, list) or len(evidence_pairs) != 27:
        raise ValueError("H11 evidence must contain 27 pairs")
    seen_evidence: set[int] = set()
    evidence_requests = 0
    for pair in evidence_pairs:
        _require_keys(pair, NANO_EVIDENCE_PAIR_KEYS, "H11 evidence pair")
        seed = pair.get("seed")
        if seed not in SEEDS or seed in seen_evidence or pair.get("pair_id") != f"v3:droid:{model}:seed{seed}":
            raise ValueError(f"H11 invalid or duplicate evidence pair: {seed}")
        seen_evidence.add(seed)
        evidence_cells = pair.get("cells")
        if not isinstance(evidence_cells, dict) or set(evidence_cells) != set(RELATIONS):
            raise ValueError(f"H11 evidence relation grid changed: {seed}")
        for relation in RELATIONS:
            cell = evidence_cells[relation]
            _require_keys(cell, NANO_EVIDENCE_CELL_KEYS, f"H11 evidence cell {seed}:{relation}")
            expected = by_key[(seed, relation)]
            if cell.get("registered_cell_id") != expected["registered_cell_id"]:
                raise ValueError(f"H11 evidence/summary cell join changed: {seed}:{relation}")
            count = _positive_int(cell.get("future_request_count"), f"H11 evidence {seed}:{relation} request count")
            requests = cell.get("future_requests")
            if count != expected["request_count"] or not isinstance(requests, list) or len(requests) != count:
                raise ValueError(f"H11 evidence request accounting changed: {seed}:{relation}")
            for index, request in enumerate(requests):
                if request.get("request_index") != index or request.get("decoded_future_shape") != [33, 528, 640, 3] or request.get("returned_action_shape") != [32, 8]:
                    raise ValueError(f"H11 decoded future/action contract changed: {seed}:{relation}:{index}")
            evidence_requests += count
    if seen_evidence != set(SEEDS):
        raise ValueError("H11 evidence pair grid is incomplete")
    verification = evidence.get("verification", {})
    if verification.get("all_referenced_payloads_rehashed") is not True or verification.get("raw_behavioral_row_count") != 54:
        raise ValueError("H11 evidence verification status changed")
    if verification.get("decoded_future_count") != evidence_requests or verification.get("returned_action_chunk_count") != evidence_requests:
        raise ValueError("H11 decoded-future/action-chunk totals changed")
    future = summary.get("future_retention", {})
    if future.get("decoded_future_count") != evidence_requests or future.get("returned_action_chunk_count") != evidence_requests or future.get("missing_or_latent_only_future_scored_as_zero") is not False:
        raise ValueError("H11 summary future accounting changed")

    if infra.get("study_id") != STUDY or infra.get("model_id") != model:
        raise ValueError("H11 infrastructure identity changed")
    if infra.get("excluded_attempt_count") != 2 or len(infra.get("excluded_attempts", [])) != 2:
        raise ValueError("H11 infrastructure-invalid attempt count changed")
    if infra.get("behavioral_denominator_excludes_all_entries") is not True or infra.get("runtime_intervention_count") != 0:
        raise ValueError("H11 infrastructure denominator/intervention policy changed")
    if infra.get("operational_wall_time_valid_episode_count") != 54 or infra.get("thermal_guard_pair_count") != 27:
        raise ValueError("H11 guarded completion accounting changed")
    for attempt in infra.get("excluded_attempts", []):
        if attempt.get("behavioral_result_valid") is not False or attempt.get("model_request_started") is not False:
            raise ValueError("H11 excluded infrastructure attempt entered behavior")
    return by_key, by_seed


def _normalize_dream(
    model_docs: Mapping[str, Any], registry: Mapping[tuple[int, str], Mapping[str, Any]]
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[int, Mapping[str, Any]]]:
    model = "dreamzero_droid_action_cfg"
    summary, evidence, infra = (
        model_docs["summary"], model_docs["evidence"], model_docs["infrastructure"]
    )
    _require_keys(summary, DREAM_SUMMARY_KEYS, "H12 summary")
    _require_keys(evidence, DREAM_EVIDENCE_KEYS, "H12 evidence manifest")
    _require_keys(infra, DREAM_INFRA_KEYS, "H12 infrastructure ledger")
    _validate_common_summary(model, summary)
    if summary.get("schema_version") != "vla-wam-v3-dreamzero-s2-phase-a-summary-v1":
        raise ValueError("H12 summary schema version changed")
    contract = summary.get("registered_execution_contract", {})
    checkpoint = contract.get("checkpoint", {})
    if checkpoint.get("revision") != "96ad344138c66e82536422432ad742f015784942" or contract.get("action_cfg_style_scale") != 2 or contract.get("baseline_s1_used") is not False:
        raise ValueError("H12 custom-s2 checkpoint/guidance contract changed")
    if contract.get("video_cfg_scale") != 5 or contract.get("open_loop_horizon") != 8 or contract.get("effective_official_model_noise_seed") != 1140:
        raise ValueError("H12 execution constants changed")
    if contract.get("future_interface") != "joint action and latent video with official full reset decode":
        raise ValueError("H12 latent/decode interface changed")

    cells = summary.get("cells")
    pairs = summary.get("pairs")
    if not isinstance(cells, list) or len(cells) != 54 or not isinstance(pairs, list) or len(pairs) != 27:
        raise ValueError("H12 summary population must be 54 episodes/27 pairs")
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for source_row, row in enumerate(cells, 1):
        _require_keys(row, DREAM_CELL_KEYS, f"H12 summary cell {source_row}")
        seed, relation = row.get("seed"), row.get("relation")
        key = (seed, relation)
        if key not in registry or key in by_key:
            raise ValueError(f"H12 invalid or duplicate compact episode: {key}")
        registered = registry[key]
        for summary_field, registry_field in (
            ("registered_cell_id", "cell_id"), ("pair_id", "pair_id"),
            ("prompt", "prompt"), ("prompt_sha256", "prompt_sha256"),
        ):
            if row.get(summary_field) != registered.get(registry_field):
                raise ValueError(f"H12 compact episode/registry mismatch: {key}:{summary_field}")
        if type(row.get("requested_success")) is not bool:
            raise ValueError(f"H12 requested_success changed: {key}")
        requests = _positive_int(row.get("policy_request_count"), f"H12 {key} policy_request_count")
        latent = _positive_int(row.get("latent_future_count"), f"H12 {key} latent_future_count")
        decodes = _positive_int(row.get("official_reset_decode_count"), f"H12 {key} official_reset_decode_count")
        if latent != requests or decodes != 1:
            raise ValueError(f"H12 request/latent/reset-decode accounting changed: {key}")
        shape_counts = row.get("latent_shape_counts")
        if not isinstance(shape_counts, dict) or set(shape_counts) - {"1x16x2x44x80", "1x16x3x44x80"}:
            raise ValueError(f"H12 latent shape schema changed: {key}")
        if sum(_nonnegative_int(value, f"H12 {key} latent shape count") for value in shape_counts.values()) != latent:
            raise ValueError(f"H12 latent shape count changed: {key}")
        actions = _positive_int(row.get("actions_executed"), f"H12 {key} actions_executed")
        if actions > 450 or row.get("action_cap") != 450 or row.get("episode_length_steps") != actions:
            raise ValueError(f"H12 action cap/count changed: {key}")
        margin = _finite(row.get("final_requested_signed_margin_m"), f"H12 {key} margin")
        endpoint = _finite(row.get("signed_final_lateral_offset_m"), f"H12 {key} endpoint")
        _close(margin, endpoint if relation == "left" else -endpoint, f"H12 {key} margin sign")
        hashes = row.get("hashes")
        expected_hash_keys = {
            "compiled_jsonl_sha256", "executed_action_trace_sha256", "future_manifest_sha256",
            "latent_future_sha256", "official_reset_decode_sha256", "state_capture_sha256",
            "viewport_video_sha256",
        }
        if not isinstance(hashes, dict) or set(hashes) != expected_hash_keys:
            raise ValueError(f"H12 compact hash identity changed: {key}")
        for name, value in hashes.items():
            if name == "latent_future_sha256":
                valid = isinstance(value, list) and len(value) == latent and all(_valid_sha(item) for item in value)
            elif name == "official_reset_decode_sha256":
                valid = isinstance(value, list) and len(value) == decodes and all(_valid_sha(item) for item in value)
            else:
                valid = _valid_sha(value)
            if not valid:
                raise ValueError(f"H12 compact hash identity changed: {key}:{name}")
        if not _valid_sha(row.get("initial_state_sha256")):
            raise ValueError(f"H12 initial-state identity changed: {key}")
        normalized = dict(row)
        normalized.update({"success": row["requested_success"], "request_count": requests, "latent_count": latent, "decode_count": decodes, "margin_m": margin, "endpoint_m": endpoint, "_source_row": source_row, "_registry_source_row": registered["_source_row"]})
        by_key[key] = normalized
    if set(by_key) != set(registry):
        raise ValueError("H12 compact 27-pair episode grid is incomplete")

    by_seed: dict[int, Mapping[str, Any]] = {}
    for source_row, pair in enumerate(pairs, 1):
        _require_keys(pair, DREAM_PAIR_KEYS, f"H12 pair {source_row}")
        seed = pair.get("seed")
        if seed not in SEEDS or seed in by_seed:
            raise ValueError(f"H12 invalid or duplicate summary pair: {seed}")
        left, right = by_key[(seed, "left")], by_key[(seed, "right")]
        if pair.get("left_success") is not left["success"] or pair.get("right_success") is not right["success"]:
            raise ValueError(f"H12 pair success mismatch: {seed}")
        if pair.get("left_failure_taxonomy") != left["failure_taxonomy"] or pair.get("right_failure_taxonomy") != right["failure_taxonomy"]:
            raise ValueError(f"H12 pair taxonomy mismatch: {seed}")
        if pair.get("initial_state_exact_match") is not True or left["initial_state_sha256"] != right["initial_state_sha256"]:
            raise ValueError(f"H12 pair initial-state identity changed: {seed}")
        endpoint_shift = right["endpoint_m"] - left["endpoint_m"]
        margin_shift = right["margin_m"] - left["margin_m"]
        _close(pair.get("right_minus_left_endpoint_shift_m"), endpoint_shift, f"H12 {seed} endpoint shift")
        _close(pair.get("right_minus_left_requested_margin_m"), margin_shift, f"H12 {seed} margin shift")
        expected_order = "aligned" if endpoint_shift < 0 else "anti_aligned" if endpoint_shift > 0 else "tie"
        if pair.get("endpoint_ordering") != expected_order:
            raise ValueError(f"H12 endpoint ordering mismatch: {seed}")
        by_seed[seed] = pair

    if evidence.get("study_id") != STUDY or evidence.get("model_id") != model or evidence.get("study_commit") != PROSPECTIVE_FREEZE_COMMIT:
        raise ValueError("H12 evidence identity/freeze binding changed")
    if evidence.get("episode_count") != 54 or evidence.get("pair_count") != 27:
        raise ValueError("H12 evidence population changed")
    evidence_pairs = evidence.get("pairs")
    if not isinstance(evidence_pairs, list) or len(evidence_pairs) != 27:
        raise ValueError("H12 evidence must contain 27 pairs")
    seen_evidence: set[int] = set()
    evidence_requests = evidence_latents = evidence_decodes = 0
    for pair in evidence_pairs:
        _require_keys(pair, DREAM_EVIDENCE_PAIR_KEYS, "H12 evidence pair")
        seed = pair.get("seed")
        if seed not in SEEDS or seed in seen_evidence or pair.get("pair_id") != f"v3:droid:{model}:seed{seed}":
            raise ValueError(f"H12 invalid or duplicate evidence pair: {seed}")
        seen_evidence.add(seed)
        evidence_cells = pair.get("cells")
        if not isinstance(evidence_cells, list) or len(evidence_cells) != 2:
            raise ValueError(f"H12 evidence relation grid changed: {seed}")
        relation_map: dict[str, Mapping[str, Any]] = {}
        for cell in evidence_cells:
            _require_keys(cell, DREAM_EVIDENCE_CELL_KEYS, f"H12 evidence cell {seed}")
            relation = cell.get("relation")
            if relation not in RELATIONS or relation in relation_map:
                raise ValueError(f"H12 duplicate evidence relation: {seed}:{relation}")
            relation_map[relation] = cell
        if set(relation_map) != set(RELATIONS):
            raise ValueError(f"H12 evidence relation grid incomplete: {seed}")
        for relation, cell in relation_map.items():
            expected = by_key[(seed, relation)]
            if cell.get("registered_cell_id") != expected["registered_cell_id"]:
                raise ValueError(f"H12 evidence/summary cell join changed: {seed}:{relation}")
            requests = cell.get("policy_requests")
            decodes = cell.get("official_reset_decodes")
            if not isinstance(requests, list) or len(requests) != expected["request_count"]:
                raise ValueError(f"H12 evidence request accounting changed: {seed}:{relation}")
            if not isinstance(decodes, list) or len(decodes) != expected["decode_count"]:
                raise ValueError(f"H12 evidence reset-decode accounting changed: {seed}:{relation}")
            for index, request in enumerate(requests):
                if request.get("request_index") != index or request.get("future_evidence_status") != "exposed_and_retained" or request.get("returned_action_shape") != [24, 8]:
                    raise ValueError(f"H12 latent/action contract changed: {seed}:{relation}:{index}")
                if request.get("latent_video_shape") not in ([1, 16, 2, 44, 80], [1, 16, 3, 44, 80]):
                    raise ValueError(f"H12 latent shape changed: {seed}:{relation}:{index}")
            evidence_requests += len(requests)
            evidence_latents += len(requests)
            evidence_decodes += len(decodes)
    if seen_evidence != set(SEEDS):
        raise ValueError("H12 evidence pair grid is incomplete")
    content = evidence.get("content_audit", {})
    expected_content_counts = {
        "episode_count": 54,
        "pair_count": 27,
        "retained_latent_future_file_count": evidence_latents,
        "official_reset_decode_file_count": evidence_decodes,
        "actual_rollout_video_count": 54,
        "status": "passed_after_queue_seal",
    }
    for key, expected in expected_content_counts.items():
        if content.get(key) != expected:
            raise ValueError(f"H12 content audit changed: {key}")
    if evidence.get("retained_latent_future_count") != evidence_latents or evidence.get("official_full_reset_decode_count") != evidence_decodes or evidence.get("actual_rollout_video_count") != 54:
        raise ValueError("H12 evidence latent/decode/video totals changed")
    if summary.get("retained_latent_future_count") != evidence_latents or summary.get("official_full_reset_decode_count") != evidence_decodes or summary.get("actual_rollout_video_count") != 54:
        raise ValueError("H12 summary latent/decode/video totals changed")
    preserved = evidence.get("preserved_v2_seed_evidence_not_pooled")
    if not isinstance(preserved, list) or len(preserved) != 2:
        raise ValueError("H12 preserved V2 evidence boundary changed")

    if infra.get("study_id") != STUDY or infra.get("model_id") != model:
        raise ValueError("H12 infrastructure identity changed")
    if infra.get("excluded_attempt_count") != 4 or len(infra.get("excluded_attempts", [])) != 4:
        raise ValueError("H12 infrastructure-invalid attempt count changed")
    if infra.get("behavioral_denominator_excludes_all_entries") is not True or infra.get("runtime_intervention_count") != 0 or infra.get("thermal_guard_pair_count") != 27:
        raise ValueError("H12 infrastructure denominator/intervention policy changed")
    for attempt in infra.get("excluded_attempts", []):
        if attempt.get("behavioral_result_valid") is not False or attempt.get("actions_executed") != 0 or attempt.get("model_request_count") != 0:
            raise ValueError("H12 excluded infrastructure attempt entered behavior")
    return by_key, by_seed


def _cohort_report(
    cohort: str,
    model: str,
    info: Mapping[str, Any],
    rows: Mapping[tuple[int, str], Mapping[str, Any]],
    source_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    direction_outputs: dict[str, Any] = {}
    for relation in RELATIONS:
        selected = [rows[(seed, relation)] for seed in SEEDS]
        successes = sum(row["success"] for row in selected)
        requests = sum(row["request_count"] for row in selected)
        if successes != info["expected_successes"][relation] or requests != info["expected_requests"][relation]:
            raise ValueError(f"{cohort} directional success/request count changed: {relation}")
        source_direction = source_summary.get("directional", {}).get(relation, {})
        source_trials_key = "trials" if cohort == "H11" else "valid_denominator"
        source_request_key = "future_request_count" if cohort == "H11" else None
        if source_direction.get("successes") != successes or source_direction.get(source_trials_key) != 27:
            raise ValueError(f"{cohort} source directional success denominator changed: {relation}")
        if source_request_key and source_direction.get(source_request_key) != requests:
            raise ValueError(f"{cohort} source directional request count changed: {relation}")
        interval = _wilson(successes, 27)
        source_interval = source_direction.get("wilson_95_interval")
        if not isinstance(source_interval, list) or len(source_interval) != 2:
            raise ValueError(f"{cohort} source Wilson interval missing: {relation}")
        _close(source_interval[0], interval[0], f"{cohort} {relation} Wilson low")
        _close(source_interval[1], interval[1], f"{cohort} {relation} Wilson high")
        direction_outputs[relation] = {
            "behavioral_episodes": 27,
            "successes": successes,
            "valid_behavioral_failures": 27 - successes,
            "success_rate": successes / 27.0,
            "wilson_95_interval": interval,
            "model_requests": requests,
            "latent_future_records": sum(row["latent_count"] for row in selected),
            "decoded_rgb_future_records": (
                sum(row["decode_count"] for row in selected) if cohort == "H11" else 0
            ),
            "official_full_reset_decodes": (
                sum(row["decode_count"] for row in selected) if cohort == "H12" else 0
            ),
            "failure_taxonomy": dict(sorted(Counter(row["failure_taxonomy"] for row in selected).items())),
            "frozen_failure_stage": dict(sorted(Counter(row["frozen_failure_stage"] for row in selected).items())),
            "mean_final_requested_margin_m": statistics.fmean(row["margin_m"] for row in selected),
            "mean_signed_final_lateral_endpoint_m": statistics.fmean(row["endpoint_m"] for row in selected),
        }

    pair_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        left, right = rows[(seed, "left")], rows[(seed, "right")]
        pair_rows.append({
            "cohort": cohort,
            "model_id": model,
            "seed": seed,
            "left_registry_source_row": left["_registry_source_row"],
            "right_registry_source_row": right["_registry_source_row"],
            "left_summary_source_row": left["_source_row"],
            "right_summary_source_row": right["_source_row"],
            "left_success": left["success"],
            "right_success": right["success"],
            "success_right_minus_left": int(right["success"]) - int(left["success"]),
            "left_model_requests": left["request_count"],
            "right_model_requests": right["request_count"],
            "pair_model_requests": left["request_count"] + right["request_count"],
            "left_latent_future_records": left["latent_count"],
            "right_latent_future_records": right["latent_count"],
            "left_decoded_rgb_future_records": left["decode_count"] if cohort == "H11" else 0,
            "right_decoded_rgb_future_records": right["decode_count"] if cohort == "H11" else 0,
            "left_official_full_reset_decodes": left["decode_count"] if cohort == "H12" else 0,
            "right_official_full_reset_decodes": right["decode_count"] if cohort == "H12" else 0,
            "left_final_requested_margin_m": left["margin_m"],
            "right_final_requested_margin_m": right["margin_m"],
            "requested_margin_right_minus_left_m": right["margin_m"] - left["margin_m"],
            "left_signed_endpoint_m": left["endpoint_m"],
            "right_signed_endpoint_m": right["endpoint_m"],
            "endpoint_right_minus_left_m": right["endpoint_m"] - left["endpoint_m"],
            "left_failure_taxonomy": left["failure_taxonomy"],
            "right_failure_taxonomy": right["failure_taxonomy"],
            "left_frozen_failure_stage": left["frozen_failure_stage"],
            "right_frozen_failure_stage": right["frozen_failure_stage"],
            "initial_state_exact_match": left["initial_state_sha256"] == right["initial_state_sha256"],
        })

    both = sum(row["left_success"] and row["right_success"] for row in pair_rows)
    left_only = sum(row["left_success"] and not row["right_success"] for row in pair_rows)
    right_only = sum(not row["left_success"] and row["right_success"] for row in pair_rows)
    neither = 27 - both - left_only - right_only
    source_discordance = source_summary.get("matched_discordance", {})
    expected_source_discordance = {
        "both_success": both,
        "left_only_success": left_only,
        "neither_success": neither,
        "right_only_success": right_only,
    }
    if cohort == "H11":
        if source_discordance != expected_source_discordance:
            raise ValueError(f"{cohort} source matched discordance changed")
    else:
        for key, expected in expected_source_discordance.items():
            if source_discordance.get(key) != expected:
                raise ValueError(f"{cohort} source matched discordance changed: {key}")
        if source_discordance.get("discordant_pair_count") != left_only + right_only:
            raise ValueError(f"{cohort} source discordant-pair count changed")
        _close(source_discordance.get("exact_two_sided_mcnemar_p"), _exact_two_sided_equal_binomial(left_only, right_only), f"{cohort} source McNemar p")
        if source_discordance.get("right_only_minus_left_only") != right_only - left_only:
            raise ValueError(f"{cohort} source discordance direction changed")
        _close(source_discordance.get("right_only_fraction_among_discordant"), right_only / (left_only + right_only), f"{cohort} source right-only fraction")
    success_values = [float(row["success_right_minus_left"]) for row in pair_rows]
    endpoint_values = [float(row["endpoint_right_minus_left_m"]) for row in pair_rows]
    margin_values = [float(row["requested_margin_right_minus_left_m"]) for row in pair_rows]
    endpoint_aligned = sum(value < 0 for value in endpoint_values)
    endpoint_anti = sum(value > 0 for value in endpoint_values)
    endpoint_ties = 27 - endpoint_aligned - endpoint_anti
    source_ordering = source_summary.get("endpoint_ordering_counts", {})
    expected_ordering = {"aligned": endpoint_aligned}
    if endpoint_anti:
        expected_ordering["anti_aligned"] = endpoint_anti
    if endpoint_ties:
        expected_ordering["tie"] = endpoint_ties
    if source_ordering != expected_ordering:
        raise ValueError(f"{cohort} endpoint ordering counts changed")
    endpoint_summary = _numeric_summary(endpoint_values)
    source_endpoint = source_summary.get(
        "right_minus_left_raw_robot_y_shift_m" if cohort == "H11" else "right_minus_left_endpoint_shift_m",
        {},
    )
    for key in ("mean", "median", "minimum", "maximum"):
        _close(source_endpoint.get(key), endpoint_summary[key], f"{cohort} endpoint {key}")
    if source_endpoint.get("observed_count", source_endpoint.get("count")) != 27:
        raise ValueError(f"{cohort} endpoint observed count changed")
    if cohort == "H12":
        _close(source_summary.get("right_minus_left_mean_requested_margin_m"), statistics.fmean(margin_values), "H12 requested-margin shift mean")
        _close(source_summary.get("directional_success_rate_difference_right_minus_left"), statistics.fmean(success_values), "H12 success-rate difference")
    endpoint_sign_p = _exact_two_sided_equal_binomial(endpoint_aligned, endpoint_anti)
    if cohort == "H12":
        _close(source_summary.get("endpoint_ordering_exact_two_sided_sign_test_p"), endpoint_sign_p, "H12 endpoint sign p")

    all_rows = list(rows.values())
    failures = sum(not row["success"] for row in all_rows)
    if failures != info["expected_failures"]:
        raise ValueError(f"{cohort} valid behavioral failure count changed")
    taxonomy = dict(sorted(Counter(row["failure_taxonomy"] for row in all_rows).items()))
    stages = dict(sorted(Counter(row["frozen_failure_stage"] for row in all_rows).items()))
    if taxonomy != source_summary.get("overall_failure_taxonomy_counts") or stages != source_summary.get("overall_frozen_failure_stage_counts"):
        raise ValueError(f"{cohort} failure taxonomy/stage totals changed")
    if source_summary.get("initial_state_exact_match_pair_count") != 27 or not all(row["initial_state_exact_match"] for row in pair_rows):
        raise ValueError(f"{cohort} initial-state pair matching changed")

    total_requests = sum(row["request_count"] for row in all_rows)
    total_latents = sum(row["latent_count"] for row in all_rows)
    total_decodes = sum(row["decode_count"] for row in all_rows)
    future_semantics = (
        {
            "model_requests": total_requests,
            "decoded_rgb_future_records": total_decodes,
            "returned_action_chunks": total_requests,
            "retained_latent_future_records": 0,
            "official_full_reset_decodes": 0,
            "interface": "one retained decoded 33-frame RGB future per policy request",
            "forecast_accuracy_scored": False,
        }
        if cohort == "H11"
        else {
            "model_requests": total_requests,
            "decoded_rgb_future_records": 0,
            "retained_latent_future_records": total_latents,
            "official_full_reset_decodes": total_decodes,
            "actual_rollout_videos": 54,
            "interface": "one retained latent future per request plus one official full-reset decode per episode",
            "latent_records_are_per_request_decoded_rgb": False,
            "forecast_accuracy_scored": False,
        }
    )
    return {
        "model_id": model,
        "configuration_label": info["label"],
        "authenticated_population": {
            "behavioral_episodes": 54,
            "matched_pairs": 27,
            "matched_seed_labels": 27,
            "model_requests": total_requests,
            "valid_behavioral_failures": failures,
            "preserved_v2_candidate_episodes_excluded": 6,
            "historical_v2_baseline_episodes_counted": 0,
            "infrastructure_invalid_attempts_excluded": 2 if cohort == "H11" else 4,
            "valid_behavioral_failures_included": True,
            "missing_value_imputation": "none",
        },
        "source_rows": {
            "registered_cells_file": CELLS,
            "authorized_new_rows_inclusive": info["registry_rows"],
            "preserved_candidate_rows_excluded_inclusive": info["preserved_rows"],
            "summary_file": info["summary"],
            "summary_cell_rows_inclusive": [1, 54],
            "summary_pair_rows_inclusive": [1, 27],
        },
        "directional": direction_outputs,
        "paired_estimands": {
            "success_right_minus_left": {
                "estimand": "mean over matched seed/reset pairs of requested-success RIGHT minus LEFT",
                "mean": statistics.fmean(success_values),
                "matched_discordance": {
                    "both_success": both,
                    "left_only_success": left_only,
                    "right_only_success": right_only,
                    "neither_success": neither,
                    "discordant_pairs": left_only + right_only,
                },
                "exact_two_sided_mcnemar_p": _exact_two_sided_equal_binomial(left_only, right_only),
                "inference_status": "source-registered descriptive matched reporting; retained recovery evidence only",
            },
            "endpoint_right_minus_left_m": {
                "estimand": "mean over matched seed/reset pairs of signed robot-base lateral endpoint RIGHT minus LEFT",
                **endpoint_summary,
                "aligned_negative": endpoint_aligned,
                "anti_aligned_positive": endpoint_anti,
                "ties": endpoint_ties,
                "exact_two_sided_sign_p": endpoint_sign_p,
                "sign_convention": "negative is instruction-aligned RIGHT-versus-LEFT endpoint ordering",
            },
            "requested_margin_right_minus_left_m": {
                "estimand": "mean over matched seed/reset pairs of final requested signed margin RIGHT minus LEFT",
                **_numeric_summary(margin_values),
                "algebraically_distinct_from_endpoint_shift": True,
                "reason": "LEFT margin equals +y_left while RIGHT margin equals -y_right; their difference is -(y_right+y_left), not y_right-y_left",
            },
        },
        "overall_failure_taxonomy": taxonomy,
        "overall_frozen_failure_stage": stages,
        "future_accounting": future_semantics,
        "configuration_boundary": (
            "Cosmos3 Nano direct-command g3 retained Phase-A behavior"
            if cohort == "H11"
            else "custom DreamZero action-guidance s2 with fixed effective noise 1140; not default DreamZero and not the conditional-action D1 core configuration"
        ),
    }, pair_rows


def analyze_documents(documents: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    registered = _validate_registration(documents)
    normalized: dict[str, tuple[dict[tuple[int, str], dict[str, Any]], dict[int, Mapping[str, Any]]]] = {
        "cosmos3_nano_policy_droid": _normalize_nano(
            documents["models"]["cosmos3_nano_policy_droid"],
            registered["cosmos3_nano_policy_droid"],
        ),
        "dreamzero_droid_action_cfg": _normalize_dream(
            documents["models"]["dreamzero_droid_action_cfg"],
            registered["dreamzero_droid_action_cfg"],
        ),
    }
    cohorts: dict[str, Any] = {}
    matched_pairs: list[dict[str, Any]] = []
    for model, info in MODELS.items():
        rows, _ = normalized[model]
        cohort, pair_rows = _cohort_report(
            info["cohort"], model, info, rows, documents["models"][model]["summary"]
        )
        cohorts[info["cohort"]] = cohort
        matched_pairs.extend(pair_rows)

    activity = {
        "annotation_media": 0,
        "behavioral_actions": 0,
        "behavioral_cells": 0,
        "confirmation_inference_cells": 0,
        "forecast_accuracy_estimates": 0,
        "human_responses": 0,
        "labels": 0,
        "model_requests": 0,
        "queue_descriptors": 0,
        "request_rows": 0,
        "request_selections": 0,
        "retrieval_jobs": 0,
        "runnable_manifests": 0,
    }
    gates = {
        "annotation_media_ready": False,
        "camera_alignment_ready": False,
        "confirmation_release_ready": False,
        "current_GM_PVC_worker_state_reconciled": False,
        "development_annotation_gate_ready": False,
        "human_annotation_ready": False,
        "live_quiescence_inferred": False,
        "resource_freeze_ready": False,
        "safe_to_release_new_work": False,
    }
    return {
        "schema_version": "wmf-retained-h11-h12-phase-a-direct-v1",
        "provenance": dict(provenance),
        "design": {
            "study_id": STUDY,
            "phase": "A_direct_command_matched_pairs",
            "cohorts": ["H11", "H12"],
            "prospective_plan_status": "frozen_before_any_v3_behavioral_inference",
            "matched_environment_seed_labels": list(SEEDS),
            "preserved_v2_seed_labels_excluded": list(PRESERVED_SEEDS),
            "h06_h07_h08_canonical_core_seed_labels": list(range(9400, 9427)),
            "h11_h12_overlap_with_h06_h07_h08_canonical_core": [],
            "population_identity_separation_verified_against": SYMMETRY_REGISTRATION,
            "relations": list(RELATIONS),
            "exact_prompts": PROMPTS,
            "inferential_unit": "matched environment/reset seed pair within checkpoint",
            "registered_droid_estimand": documents["analysis_plan"]["primary_estimands"]["droid"],
            "checkpoint_separation": "all summaries and paired estimands remain model-specific",
        },
        "cohorts": cohorts,
        "matched_seed_pairs": matched_pairs,
        "source_manifest": {
            "authenticated_file_count": len(EXPECTED_SOURCES),
            "authenticated_bytes": sum(source["bytes"] for source in EXPECTED_SOURCES.values()),
            "authenticated_lines": sum(source["lines"] for source in EXPECTED_SOURCES.values()),
            "h11_h12_payload_file_count": len(PAYLOAD_SOURCES),
            "h11_h12_payload_bytes": sum(EXPECTED_SOURCES[path]["bytes"] for path in PAYLOAD_SOURCES),
            "h11_h12_payload_lines": sum(EXPECTED_SOURCES[path]["lines"] for path in PAYLOAD_SOURCES),
            "design_and_overlap_helper_file_count": len(HELPER_SOURCES),
            "design_and_overlap_helper_bytes": sum(EXPECTED_SOURCES[path]["bytes"] for path in HELPER_SOURCES),
            "design_and_overlap_helper_lines": sum(EXPECTED_SOURCES[path]["lines"] for path in HELPER_SOURCES),
            "compact_git_files_byte_rehashed_now": True,
            "prospective_freeze_ancestry_verified": True,
            "raw_pvc_files_revalidated_now": False,
            "raw_source_paths_exported": False,
            "exported_source_paths_are_git_object_paths_only": True,
        },
        "reproduction": {
            "full_phase_a_registry_rows_strictly_parsed": 780,
            "h11_h12_registered_cells_rejoined": 108,
            "h11_h12_summary_episode_rows_strictly_parsed": 108,
            "h11_h12_matched_pairs_reconstructed": 54,
            "preserved_v2_candidate_cells_excluded": 12,
            "historical_v2_baseline_cells_double_counted": 0,
            "h11_h12_identity_overlap_with_h06_h07_h08_canonical_core": 0,
            "model_request_counts_exactly_reconciled": True,
            "nano_decoded_future_and_action_chunk_counts_exactly_reconciled": True,
            "dream_request_latent_and_official_decode_counts_exactly_reconciled": True,
            "directional_successes_wilson_intervals_and_failure_taxonomies_exactly_reproduced": True,
            "matched_discordance_and_endpoint_ordering_exactly_reproduced": True,
            "infrastructure_invalid_attempts_verified_prebehavior_and_excluded": True,
            "models_pooled": False,
        },
        "new_activity": activity,
        "gate_state": gates,
        "claim_boundary": {
            "status": "retrospective compact-Git authentication and descriptive reproduction of prospectively registered retained Phase-A direct-command behavior",
            "historical_behavior_and_recovery_source_only": True,
            "new_model_inference": False,
            "new_robot_episodes": False,
            "forecast_accuracy_evidence": False,
            "independent_scene_claim": False,
            "annotation_evidence": False,
            "equivalence_claim": False,
            "confirmation_test_or_stage_gate": False,
            "models_pooled": False,
            "historical_v2_baselines_double_counted": False,
            "changes_232_cell_core_bound": False,
            "maximum_core_behavioral_cells": 232,
            "boundaries": [
                "Each checkpoint contributes 54 retained episodes nested in 27 matched LEFT/RIGHT neutral-reset seed labels, not 54 independent scenes.",
                "The six preserved V2 candidate cells at seeds 8300-8302 per checkpoint are excluded because their complete V3 runtime identity was unavailable; they are not copied, rerun, or counted again.",
                "Nano and custom DreamZero-s2 remain separate checkpoints; no combined success rate, effect, ranking, or cross-model test is reported.",
                "DreamZero-s2 is the historical custom negative-branch action-guidance configuration with fixed effective model noise 1140, not default DreamZero and not the conditional-action D1 core configuration; environment seed labels are not independent policy-noise draws.",
                "Nano's 349 records are decoded RGB futures, whereas DreamZero-s2's 2,554 per-request records are latent futures with only 54 official full-reset decodes; neither is scored for forecast accuracy here.",
                "Final margins and endpoints are goal-stopped outcome measurements, not common-time trajectory or physical-time forecast/execution comparisons.",
                "The compact Git sources are byte-authenticated; external raw simulator, action-array, video, latent, and PVC paths are neither exported nor revalidated.",
                "These retained sources support development and recovery only, not unseen confirmation, annotation, equivalence, new inference, or a change to the 232-cell core ceiling.",
            ],
        },
    }


def render_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_csv(report: Mapping[str, Any]) -> str:
    rows = report["matched_seed_pairs"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("matched_seed_pairs must be a nonempty list")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--check", action="store_true", help="require checked-in outputs to match exactly")
    args = parser.parse_args(argv)
    documents, provenance = read_pinned_sources(ROOT)
    report = analyze_documents(documents, provenance)
    json_text = render_json(report)
    csv_text = render_csv(report)
    if args.check:
        if args.json_output.read_text(encoding="utf-8") != json_text:
            raise SystemExit(f"stale generated output: {args.json_output}")
        if args.csv_output.read_text(encoding="utf-8") != csv_text:
            raise SystemExit(f"stale generated output: {args.csv_output}")
    else:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json_text, encoding="utf-8")
        args.csv_output.write_text(csv_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
