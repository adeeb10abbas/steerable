#!/usr/bin/env python3
"""Authenticate and package retained H09/H10 wording-control evidence.

V3-C001 prospectively froze a 480-cell, three-checkpoint exploratory wording
block.  ABLATION_SPEC.md selects only the completed 160-row Cosmos3 Nano
(H09) and 160-row Cosmos3 Edge (H10) cohorts for this workshop.  This adapter
authenticates the immutable registration and compact Git evidence, validates
the complete registered grid, and reports model-specific same-seed contrasts
between each alternative wording and the registered direct-command control.

The wording block explicitly has no registered confirmatory inference.  This
script therefore reports descriptive contrasts and exact matched discordance
counts only.  It opens no raw/PVC artifact path, runs no model or robot, requests no
annotation, pools no checkpoints, and advances no forecast-ablation gate.
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
REGISTRATION_COMMIT = "05f230000010119ef5e8303c968cbea53ceb4260"
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
SPEC_PATH = PACKAGE / "docs" / "ABLATION_SPEC.md"
DEFAULT_JSON = PACKAGE / "paper" / "data" / "h09_h10_wording_control.json"
DEFAULT_CSV = PACKAGE / "paper" / "data" / "h09_h10_wording_control.csv"

STUDY = "vla_wam_language_steerability_v3"
EXPERIMENT = "V3-C001"
SEEDS = tuple(range(8500, 8520))
PROMPT_FORMS = (
    "direct_command",
    "short_command",
    "goal_as_outcome",
    "desired_plus_negated_opposite",
)
ALTERNATIVES = PROMPT_FORMS[1:]
RELATIONS = ("left", "right")
ALL_REGISTERED_MODELS = (
    "groot_n17_droid_vla",
    "cosmos3_edge_policy_droid",
    "cosmos3_nano_policy_droid",
)

BASE = "artifacts/vla_wam_shared_v3/phase_c/four_phrasings_v3c001"
ANALYSIS_PLAN = "artifacts/vla_wam_shared_v3/analysis_plan.json"
WORDING_REGISTRY = "artifacts/vla_wam_shared_v3/four_phrasings_registry.json"
REGISTRATION = f"{BASE}/prospective_phase_c_v3c001_registration.json"
MANIFEST = f"{BASE}/phase_c_v3c001_manifest.json"
CELLS = f"{BASE}/phase_c_v3c001_cells.jsonl"

MODELS: dict[str, dict[str, Any]] = {
    "cosmos3_nano_policy_droid": {
        "cohort": "H09",
        "label": "Cosmos3 Nano",
        "result_commit": "871d4322c786ff81b75da9215a53da87d0be9961",
        "cell_rows": [321, 480],
        "episodes": f"{BASE}/results/cosmos3_nano_policy_droid/cosmos3_nano_policy_droid_phase_c_episodes.jsonl",
        "summary": f"{BASE}/results/cosmos3_nano_policy_droid/cosmos3_nano_policy_droid_phase_c_summary.json",
        "evidence": f"{BASE}/results/cosmos3_nano_policy_droid/cosmos3_nano_policy_droid_phase_c_evidence_manifest.json",
    },
    "cosmos3_edge_policy_droid": {
        "cohort": "H10",
        "label": "Cosmos3 Edge",
        "result_commit": "ed7e00e9de0a90af227180cf9e4e13ff1ddb5801",
        "cell_rows": [161, 320],
        "episodes": f"{BASE}/results/cosmos3_edge_policy_droid/cosmos3_edge_policy_droid_phase_c_episodes.jsonl",
        "summary": f"{BASE}/results/cosmos3_edge_policy_droid/cosmos3_edge_policy_droid_phase_c_summary.json",
        "evidence": f"{BASE}/results/cosmos3_edge_policy_droid/cosmos3_edge_policy_droid_phase_c_evidence_manifest.json",
    },
}

EXPECTED_SOURCES: dict[str, dict[str, Any]] = {
    ANALYSIS_PLAN: {
        "git_blob": "27fd9d2f601eae1fc6ba83a8d7bf5fb70270344e",
        "sha256": "f0f1c327ff1da0bfd74a3e604abbb3d8c243a0b0e7b41f6836ed689846c7426a",
        "bytes": 2_693,
        "lines": 24,
    },
    WORDING_REGISTRY: {
        "git_blob": "9fd854311a889905f8eae2e13aeef1f654f2e317",
        "sha256": "30cf5594eb74bc195e7ec6ae878523fb302771e6548c794127e987c0066513e6",
        "bytes": 2_730,
        "lines": 38,
    },
    REGISTRATION: {
        "git_blob": "8d1108308c37cf8b5fb935a83f0348fc8cb6feae",
        "sha256": "55ee72fae657e64f9099ac5a91c33cc9e5db78f4da511d420624cdadd898ebc2",
        "bytes": 5_406,
        "lines": 1,
    },
    MANIFEST: {
        "git_blob": "ce4cf4406c51065e856956c28200439b9b0b9b27",
        "sha256": "fe84a7beb072eb9299deab4fa945401c913f430f75e5a972941307c943d464fe",
        "bytes": 859,
        "lines": 1,
    },
    CELLS: {
        "git_blob": "e89344d00189a9b57e6ff0253483ce8dc1396202",
        "sha256": "bb8c28773a6e8e4b1de679a54af5e3e53ecfda7924abda13565d7615dde6e686",
        "bytes": 589_100,
        "lines": 480,
    },
    MODELS["cosmos3_nano_policy_droid"]["episodes"]: {
        "git_blob": "95c61797d01da2890271980d117c15f6c07b2c1b",
        "sha256": "d7016c592e492c028d072eab2d191f139f4966cfc08f2222ed954cfdfb02c0f5",
        "bytes": 592_172,
        "lines": 160,
    },
    MODELS["cosmos3_nano_policy_droid"]["summary"]: {
        "git_blob": "7d03486032ffe97f9d5a705b5e152d3cad6a882c",
        "sha256": "0f134044065dc0f1501a91a69d8e9b10d0a13e9a4065a7c4d33f7854bab73ee9",
        "bytes": 6_478,
        "lines": 1,
    },
    MODELS["cosmos3_nano_policy_droid"]["evidence"]: {
        "git_blob": "5df772987fbe83d47d6883525ef1a50cdf1e9bbc",
        "sha256": "16f2bcb1e7de10d9a5adb692a433809845517bee1466fb34589200f6bb9be530",
        "bytes": 8_244,
        "lines": 1,
    },
    MODELS["cosmos3_edge_policy_droid"]["episodes"]: {
        "git_blob": "dbe442c7ec0972c05c880fbf152bb814d8d3cfe3",
        "sha256": "317aa9e7eb18c5ef77b9445483128a7621df5b777835d8eb9e383939acf0a1f5",
        "bytes": 592_184,
        "lines": 160,
    },
    MODELS["cosmos3_edge_policy_droid"]["summary"]: {
        "git_blob": "056be8c99f5e4a7a482725373ea49c0f1d8b1a6c",
        "sha256": "06a8fd1ff425bce10cbdbeb4b73e9a8ca5ff31da0c5e389ae85f5b64bc451861",
        "bytes": 6_531,
        "lines": 1,
    },
    MODELS["cosmos3_edge_policy_droid"]["evidence"]: {
        "git_blob": "3c4ae95f168a1a5d941069e0d5dcd699972dd9fa",
        "sha256": "2c72c90b71c5ea0001d917ce4464a3207faa1c6ae62b1e0200926d435c90c97a",
        "bytes": 8_250,
        "lines": 1,
    },
}

CELL_KEYS = {
    "action_cap", "action_horizon", "arena", "behavioral_status", "checkpoint",
    "checkpoint_revision", "direct_command_within_seed_control", "environment_seed",
    "experiment_id", "future_contract", "model_id", "pair_id", "phase",
    "phase_a_runtime_identity_sha256", "prompt", "prompt_family", "prompt_sha256",
    "prompt_utf8_bytes", "randomization_namespace", "registered_cell_id", "relation",
    "sampling_seed", "schema_version", "seed", "seed_block_id", "static_episode_prompt",
    "within_seed_execution_order",
}
EPISODE_KEYS = {
    "actions_executed", "artifacts", "decoded_future_shapes", "experiment_id",
    "failure_taxonomy", "initial_state_sha256", "measurements", "model_id",
    "model_request_count", "prompt", "prompt_family", "prompt_sha256",
    "registered_cell_id", "relation", "requested_success", "schema_version", "seed",
    "source_whole_seed_report", "video", "within_seed_execution_order",
}
MEASUREMENT_KEYS = {
    "actions_executed", "entry_kind", "episode_length_steps", "executed_action_count",
    "final_detached_release", "final_opposite_region_sustained",
    "final_opposite_signed_margin_m", "final_requested_region_sustained",
    "final_requested_signed_margin_m", "first_cone_or_native_region_entry_step",
    "first_contact_status", "first_contact_step", "first_contact_unavailable_reason",
    "first_opposite_entry_step", "first_requested_entry_step", "first_requested_region_step",
    "first_sustained_opposite_entry_step", "first_sustained_requested_entry_step",
    "first_verified_pickup_step", "max_object_lift_m", "maximum_pickup_height_m",
    "maximum_requested_signed_margin_m", "object_path_length_m",
    "operational_wall_time_valid", "opposite_entry_kind", "pickup_lift_threshold_m",
    "region_kind", "requested_entry_kind", "requested_final_lateral_margin_m",
    "right_censored", "scored_state_sample_count", "signed_final_lateral_offset_m",
    "sustained_steps", "verified_pickup", "wall_time_s",
}
FAILURE_CLASSES = {"correct", "pick_failed", "release_failed", "transport_failed", "wrong_side"}


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
        raise ValueError(f"prospective registration ancestry check failed: {ancestor} -> {descendant}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


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
        raise ValueError(f"{label} schema changed: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _close(observed: Any, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(_finite(observed, label), expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} disagrees with compact-row reproduction")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_pinned_sources(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate immutable Git blobs and strictly parse compact evidence."""

    root = Path(root)
    resolved = _git(root, "rev-parse", "--verify", PIN + "^{commit}").decode().strip()
    if resolved != PIN:
        raise ValueError("resolved historical evidence commit differs from fixed pin")

    _require_ancestor(root, REGISTRATION_COMMIT, PIN)
    for info in MODELS.values():
        _require_ancestor(root, REGISTRATION_COMMIT, info["result_commit"])
        _require_ancestor(root, info["result_commit"], PIN)
    for path in (REGISTRATION, CELLS):
        frozen = _git(root, "rev-parse", "--verify", f"{REGISTRATION_COMMIT}:{path}").decode().strip()
        if frozen != EXPECTED_SOURCES[path]["git_blob"]:
            raise ValueError(f"prospective registration blob changed after freeze: {path}")

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
        "wording_registry": parse_json(raw_by_path[WORDING_REGISTRY], label=WORDING_REGISTRY),
        "registration": parse_json(raw_by_path[REGISTRATION], label=REGISTRATION),
        "manifest": parse_json(raw_by_path[MANIFEST], label=MANIFEST),
        "cells": _parse_jsonl(raw_by_path[CELLS], label=CELLS),
        "models": {},
        "ablation_spec": SPEC_PATH.read_text(encoding="utf-8"),
    }
    for model, info in MODELS.items():
        documents["models"][model] = {
            "episodes": _parse_jsonl(raw_by_path[info["episodes"]], label=info["episodes"]),
            "summary": parse_json(raw_by_path[info["summary"]], label=info["summary"]),
            "evidence": parse_json(raw_by_path[info["evidence"]], label=info["evidence"]),
        }
    provenance = {
        "commit": PIN,
        "prospective_registration_commit": REGISTRATION_COMMIT,
        "registration_is_ancestor_of_result_commits": True,
        "files": observed_files,
    }
    return documents, provenance


def _validate_ablation_scope(text: str) -> None:
    required = (
        "| H09 | Nano four phrasings, V3-C001 | 160 episodes: 20 blocks × four phrasings × two commands; 134/160 successes. | Existing wording control; do not rerun a wording sweep |",
        "| H10 | Cosmos Edge four phrasings, V3-C001 | 160 episodes with the same factorial; 123/160 successes. | Existing wording control, separate checkpoint |",
        "H09/H10 direct/short/outcome/desired-plus-negated-opposite phrasings are already observed. New confirmation uses only the two literal direct commands.",
    )
    for sentence in required:
        if sentence not in text:
            raise ValueError("ABlation-spec H09/H10 scope binding changed")


def _validate_registration(documents: Mapping[str, Any]) -> None:
    _validate_ablation_scope(documents["ablation_spec"])
    registration = documents["registration"]
    _require_keys(registration, {
        "analysis_status", "arena", "design", "exact_prompts", "experiment_id", "frozen_at",
        "models", "prohibited", "randomization", "raw_output_contract", "release_policy",
        "schema_version", "source_artifact_sha256", "status", "study_id",
    }, "V3-C001 registration")
    if registration.get("study_id") != STUDY or registration.get("experiment_id") != EXPERIMENT:
        raise ValueError("V3-C001 registration identity changed")
    if registration.get("status") != "prospectively_frozen_not_released_no_phase_c_inference":
        raise ValueError("V3-C001 is not the frozen pre-inference registration")
    if registration.get("schema_version") != "vla-wam-shared-v3c-four-phrasings-registration-v1":
        raise ValueError("V3-C001 registration schema changed")
    analysis_status = registration.get("analysis_status", {})
    if analysis_status != {
        "confirmatory_inference": "not_registered",
        "constraint": "A later prospective amendment must add power and inference thresholds before any confirmatory wording claim.",
        "descriptive_outputs": [
            "raw success counts and Wilson intervals by checkpoint, prompt form, and direction",
            "within-seed direct-control contrasts for success and registered continuous measures",
            "exact matched discordance tables",
            "failure taxonomy by prompt form and direction",
        ],
        "wording_block": "exploratory",
    }:
        raise ValueError("V3-C001 exploratory analysis status changed")
    design = registration.get("design", {})
    expected_design = {
        "behavioral_cells": 480,
        "cells_per_seed_block": 8,
        "direct_command_is_within_seed_control": True,
        "episodes_per_model": 160,
        "matched_pairs_per_model": 20,
        "models": list(ALL_REGISTERED_MODELS),
        "prompt_forms": list(PROMPT_FORMS),
        "relations": list(RELATIONS),
        "same_seed_is_environment_and_sampling_seed": True,
        "seed_blocks": 60,
        "seeds": list(SEEDS),
    }
    if design != expected_design:
        raise ValueError("V3-C001 registered design changed")
    if set(registration.get("models", {})) != set(ALL_REGISTERED_MODELS):
        raise ValueError("V3-C001 registered model population changed")
    if registration.get("exact_prompts") != documents["wording_registry"].get("prompt_forms"):
        raise ValueError("V3-C001 prompt registry disagrees with registration")
    release = registration.get("release_policy", {})
    if release.get("no_behavior_before_release") is not True or release.get("status") != "all_models_unreleased":
        raise ValueError("V3-C001 registration was not frozen before behavioral release")
    source_hashes = registration.get("source_artifact_sha256", {})
    for path in (ANALYSIS_PLAN, WORDING_REGISTRY):
        if source_hashes.get(path) != EXPECTED_SOURCES[path]["sha256"]:
            raise ValueError(f"V3-C001 registration source hash changed: {path}")

    plan = documents["analysis_plan"]
    if plan.get("study_id") != STUDY or plan.get("status") != "frozen_before_any_v3_behavioral_inference":
        raise ValueError("shared analysis plan is not the frozen pre-inference record")
    paired = plan.get("reporting", {}).get("paired_continuous", [])
    for measure in ("final requested signed margin", "paired RIGHT-minus-LEFT endpoint shift"):
        if measure not in paired:
            raise ValueError(f"registered continuous measure missing: {measure}")
    if "exploratory" not in plan.get("claim_boundary", ""):
        raise ValueError("shared analysis plan lost exploratory wording boundary")

    registry = documents["wording_registry"]
    if registry.get("study_id") != STUDY or registry.get("status") != "frozen_separately_gated_before_any_phase_c_model_request_or_behavioral_inference":
        raise ValueError("four-phrasing registry is not prospectively frozen")
    if registry.get("eligible_model_ids") != list(ALL_REGISTERED_MODELS):
        raise ValueError("four-phrasing eligible model list changed")
    accounting = registry.get("episode_accounting", {})
    if accounting.get("frozen_total_episodes") != 480 or accounting.get("episodes_per_checkpoint") != 160:
        raise ValueError("four-phrasing frozen episode accounting changed")
    if "direct command is the within-seed control" not in registry.get("randomization", ""):
        raise ValueError("direct-command control registration changed")

    manifest = documents["manifest"]
    if manifest.get("behavioral_inference_count_at_freeze") != 0 or manifest.get("behavioral_release") is not False:
        raise ValueError("V3-C001 manifest no longer records a zero-inference freeze")
    if manifest.get("status") != "hash_bound_registration_ready_models_unreleased":
        raise ValueError("V3-C001 registration manifest status changed")
    if manifest.get("counts") != {
        "behavioral_cells": 480,
        "cells_by_model": {model: 160 for model in ALL_REGISTERED_MODELS},
        "fixed_observation_requests": 36,
        "seed_blocks": 60,
    }:
        raise ValueError("V3-C001 registration manifest counts changed")
    for name, path in (("phase_c_v3c001_cells.jsonl", CELLS), ("prospective_phase_c_v3c001_registration.json", REGISTRATION)):
        item = manifest.get("files", {}).get(name, {})
        if item.get("bytes") != EXPECTED_SOURCES[path]["bytes"] or item.get("sha256") != EXPECTED_SOURCES[path]["sha256"]:
            raise ValueError(f"V3-C001 registration manifest file identity changed: {name}")


def _normalize_cells(documents: Mapping[str, Any]) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    rows = documents["cells"]
    if len(rows) != 480:
        raise ValueError("full V3-C001 registration must contain exactly 480 cells")
    prompts = documents["registration"]["exact_prompts"]
    registered_models = documents["registration"]["models"]
    by_key: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    order_by_block: dict[tuple[str, int], list[int]] = {}
    rows_by_model: dict[str, list[int]] = {model: [] for model in ALL_REGISTERED_MODELS}
    for row in rows:
        number = row.get("_source_row")
        payload = {key: value for key, value in row.items() if key != "_source_row"}
        _require_keys(payload, CELL_KEYS, f"registered cell row {number}")
        model, seed, form, relation = (
            row.get("model_id"), row.get("seed"), row.get("prompt_family"), row.get("relation")
        )
        key = (model, seed, form, relation)
        if model not in ALL_REGISTERED_MODELS or seed not in SEEDS or form not in PROMPT_FORMS or relation not in RELATIONS:
            raise ValueError(f"registered cell outside frozen grid: {key}")
        if key in by_key:
            raise ValueError(f"duplicate registered V3-C001 cell: {key}")
        if row.get("schema_version") != "vla-wam-shared-v3c-four-phrasings-cell-v1" or row.get("experiment_id") != EXPERIMENT:
            raise ValueError(f"registered cell identity changed: {key}")
        if row.get("arena") != "droid_robolab" or row.get("phase") != "C_four_phrasings":
            raise ValueError(f"registered cell arena/phase changed: {key}")
        if row.get("environment_seed") != seed or row.get("sampling_seed") != seed:
            raise ValueError(f"registered environment/sampling seed mismatch: {key}")
        if row.get("prompt") != prompts[form][relation] or row.get("prompt_sha256") != _sha256_text(row["prompt"]):
            raise ValueError(f"registered prompt bytes changed: {key}")
        if row.get("prompt_utf8_bytes") != len(row["prompt"].encode("utf-8")):
            raise ValueError(f"registered prompt byte count changed: {key}")
        expected_cell = f"v3c001:droid:{model}:seed{seed}:{form}:{relation}"
        expected_pair = f"v3c001:droid:{model}:seed{seed}:{form}"
        if row.get("registered_cell_id") != expected_cell or row.get("pair_id") != expected_pair:
            raise ValueError(f"registered cell/pair identity changed: {key}")
        if row.get("seed_block_id") != f"v3c001:droid:{model}:seed{seed}":
            raise ValueError(f"registered seed-block identity changed: {key}")
        if row.get("direct_command_within_seed_control") is not (form == "direct_command"):
            raise ValueError(f"direct-command control flag changed: {key}")
        if row.get("static_episode_prompt") is not True or row.get("behavioral_status") != "prospectively_registered_not_released":
            raise ValueError(f"registered behavioral release state changed: {key}")
        model_contract = registered_models[model]
        for field in ("action_cap", "action_horizon", "checkpoint", "checkpoint_revision", "future_contract", "phase_a_runtime_identity_sha256"):
            if row.get(field) != model_contract.get(field):
                raise ValueError(f"registered model contract changed for {key}:{field}")
        order = row.get("within_seed_execution_order")
        if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= 8:
            raise ValueError(f"invalid execution order: {key}")
        by_key[key] = row
        order_by_block.setdefault((model, seed), []).append(order)
        rows_by_model[model].append(number)
    expected = {
        (model, seed, form, relation)
        for model in ALL_REGISTERED_MODELS for seed in SEEDS for form in PROMPT_FORMS for relation in RELATIONS
    }
    if set(by_key) != expected:
        raise ValueError("full 480-cell V3-C001 registered grid is incomplete")
    if any(sorted(orders) != list(range(1, 9)) for orders in order_by_block.values()):
        raise ValueError("registered within-seed execution order is incomplete or duplicated")
    for model, info in MODELS.items():
        observed = rows_by_model[model]
        if observed != list(range(info["cell_rows"][0], info["cell_rows"][1] + 1)):
            raise ValueError(f"registered source-row range changed for {model}")
    return by_key


def _wilson(successes: int, trials: int) -> list[float]:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid Wilson count")
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _exact_discordance(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("matched discordance requires equally sized nonempty vectors")
    both = sum(a and b for a, b in zip(left, right))
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(not a and b for a, b in zip(left, right))
    neither = len(left) - both - left_only - right_only
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)) / 2**discordant
        p_value = min(1.0, 2.0 * tail)
    return {
        "both_success": both,
        "left_only": left_only,
        "right_only": right_only,
        "neither": neither,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
    }


def _validate_summary(model: str, rows: Mapping[tuple[int, str, str], Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    _require_keys(summary, {
        "counts", "direction_totals_descriptive_only", "experiment_id", "failure_taxonomy_overall",
        "interpretation_boundary", "model_id", "paired_diagnostics_by_prompt_family", "schema_version",
        "status", "success_by_condition",
    }, f"{model} summary")
    successes = sum(bool(row["requested_success"]) for row in rows.values())
    model_requests = sum(row["model_request_count"] for row in rows.values())
    counts = summary.get("counts", {})
    required_counts = {
        "directions": 2, "infrastructure_episodes_in_denominator": 0, "matched_seeds": 20,
        "model_requests": model_requests, "prompt_families": 4, "valid_behavioral_episodes": 160,
        "valid_behavioral_failures": 160 - successes, "viewport_videos": 160,
    }
    for field, expected in required_counts.items():
        if counts.get(field) != expected:
            raise ValueError(f"{model} source summary count changed: {field}")
    if summary.get("model_id") != model or summary.get("experiment_id") != EXPERIMENT:
        raise ValueError(f"{model} summary identity changed")
    if summary.get("status") != "complete_20_seed_160_behavioral_episode_result":
        raise ValueError(f"{model} source summary is incomplete")
    if summary.get("schema_version") != "vla-wam-shared-v3c-model-summary-v1":
        raise ValueError(f"{model} source summary schema changed")
    taxonomy = Counter(row["failure_taxonomy"] for row in rows.values())
    expected_taxonomy = {name: taxonomy.get(name, 0) for name in FAILURE_CLASSES}
    source_taxonomy = {name: summary.get("failure_taxonomy_overall", {}).get(name, 0) for name in FAILURE_CLASSES}
    if source_taxonomy != expected_taxonomy:
        raise ValueError(f"{model} failure taxonomy changed")

    condition_keys = {f"{form}:{relation}" for form in PROMPT_FORMS for relation in RELATIONS}
    if set(summary.get("success_by_condition", {})) != condition_keys:
        raise ValueError(f"{model} source condition grid changed")
    for form in PROMPT_FORMS:
        left_success: list[bool] = []
        right_success: list[bool] = []
        shifts: list[float] = []
        for relation in RELATIONS:
            condition_rows = [rows[(seed, form, relation)] for seed in SEEDS]
            success_count = sum(bool(row["requested_success"]) for row in condition_rows)
            source = summary["success_by_condition"][f"{form}:{relation}"]
            if source.get("trials") != 20 or source.get("successes") != success_count:
                raise ValueError(f"{model}:{form}:{relation} success count changed")
            _close(source.get("success_rate"), success_count / 20.0, f"{model}:{form}:{relation}.rate")
            for observed, expected in zip(source.get("wilson_95", []), _wilson(success_count, 20)):
                _close(observed, expected, f"{model}:{form}:{relation}.wilson")
            condition_taxonomy = Counter(row["failure_taxonomy"] for row in condition_rows)
            if source.get("failure_taxonomy") != dict(condition_taxonomy):
                raise ValueError(f"{model}:{form}:{relation} failure taxonomy changed")
        for seed in SEEDS:
            left = rows[(seed, form, "left")]
            right = rows[(seed, form, "right")]
            left_success.append(left["requested_success"])
            right_success.append(right["requested_success"])
            shifts.append(right["endpoint_m"] - left["endpoint_m"])
        paired = summary.get("paired_diagnostics_by_prompt_family", {}).get(form, {})
        if paired.get("matched_seed_count") != 20:
            raise ValueError(f"{model}:{form} matched seed count changed")
        source_shifts = paired.get("right_minus_left_endpoint_shift_m")
        if not isinstance(source_shifts, list) or len(source_shifts) != 20:
            raise ValueError(f"{model}:{form} endpoint shift vector changed")
        for observed, expected in zip(source_shifts, shifts):
            _close(observed, expected, f"{model}:{form}.endpoint shift")
        _close(paired.get("median_right_minus_left_endpoint_shift_m"), statistics.median(shifts), f"{model}:{form}.median shift")
        aligned = sum(value < 0.0 for value in shifts)
        anti = sum(value > 0.0 for value in shifts)
        ties = len(shifts) - aligned - anti
        if (paired.get("endpoint_ordering_aligned"), paired.get("endpoint_ordering_anti_aligned"), paired.get("endpoint_ordering_ties")) != (aligned, anti, ties):
            raise ValueError(f"{model}:{form} endpoint ordering count changed")
        if paired.get("success_discordance") != _exact_discordance(left_success, right_success):
            raise ValueError(f"{model}:{form} LEFT/RIGHT discordance changed")

    for relation in RELATIONS:
        relation_rows = [row for key, row in rows.items() if key[2] == relation]
        success_count = sum(bool(row["requested_success"]) for row in relation_rows)
        source = summary.get("direction_totals_descriptive_only", {}).get(relation, {})
        if source.get("trials") != 80 or source.get("successes") != success_count:
            raise ValueError(f"{model}:{relation} descriptive direction total changed")
        _close(source.get("descriptive_rate"), success_count / 80.0, f"{model}:{relation}.descriptive rate")


def _normalize_model(
    model: str,
    model_docs: Mapping[str, Any],
    cells: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
) -> dict[tuple[int, str, str], dict[str, Any]]:
    episodes = model_docs["episodes"]
    if len(episodes) != 160:
        raise ValueError(f"{model} must contain exactly 160 compact episodes")
    evidence = model_docs["evidence"]
    _require_keys(evidence, {
        "experiment_id", "matched_initial_state_sha256", "model_id", "outputs", "raw_retention",
        "registration", "release_manifest_sha256", "runtime_identity_sha256", "schema_version",
        "source_whole_seed_reports", "status",
    }, f"{model} evidence manifest")
    if evidence.get("model_id") != model or evidence.get("experiment_id") != EXPERIMENT:
        raise ValueError(f"{model} evidence manifest identity changed")
    if evidence.get("status") != "complete_20_seed_160_behavioral_episode_result":
        raise ValueError(f"{model} evidence manifest is incomplete")
    info = MODELS[model]
    outputs = evidence.get("outputs", {})
    episode_name = Path(info["episodes"]).name
    summary_name = Path(info["summary"]).name
    expected_episode_output = {
        "bytes": EXPECTED_SOURCES[info["episodes"]]["bytes"],
        "rows": 160,
        "sha256": EXPECTED_SOURCES[info["episodes"]]["sha256"],
    }
    expected_summary_output = {
        "bytes": EXPECTED_SOURCES[info["summary"]]["bytes"],
        "sha256": EXPECTED_SOURCES[info["summary"]]["sha256"],
    }
    if outputs.get(episode_name) != expected_episode_output or outputs.get(summary_name) != expected_summary_output:
        raise ValueError(f"{model} evidence output identity changed")
    registration = evidence.get("registration", {})
    if registration.get("bytes") != EXPECTED_SOURCES[CELLS]["bytes"] or registration.get("sha256") != EXPECTED_SOURCES[CELLS]["sha256"]:
        raise ValueError(f"{model} evidence registration binding changed")
    reports = evidence.get("source_whole_seed_reports")
    if not isinstance(reports, list) or len(reports) != 20:
        raise ValueError(f"{model} whole-seed evidence report count changed")
    reports_by_seed: dict[int, tuple[str, str]] = {}
    for report in reports:
        seed = report.get("seed")
        path, digest = report.get("path"), report.get("sha256")
        if seed not in SEEDS or seed in reports_by_seed or not isinstance(path, str) or not _valid_sha(digest):
            raise ValueError(f"{model} invalid or duplicate whole-seed evidence report")
        reports_by_seed[seed] = (path, digest)
    if set(reports_by_seed) != set(SEEDS):
        raise ValueError(f"{model} whole-seed evidence grid changed")

    by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    orders: dict[int, list[int]] = {}
    for row in episodes:
        number = row.get("_source_row")
        payload = {key: value for key, value in row.items() if key != "_source_row"}
        _require_keys(payload, EPISODE_KEYS, f"{model} episode row {number}")
        seed, form, relation = row.get("seed"), row.get("prompt_family"), row.get("relation")
        key = (seed, form, relation)
        if seed not in SEEDS or form not in PROMPT_FORMS or relation not in RELATIONS or key in by_key:
            raise ValueError(f"{model} invalid or duplicate compact episode: {key}")
        cell = cells[(model, seed, form, relation)]
        for field in ("model_id", "experiment_id", "registered_cell_id", "seed", "prompt_family", "relation", "prompt", "prompt_sha256", "within_seed_execution_order"):
            if row.get(field) != cell.get(field):
                raise ValueError(f"{model} compact episode/cell mismatch: {key}:{field}")
        if row.get("schema_version") != "vla-wam-shared-v3c-compiled-episode-v1":
            raise ValueError(f"{model} compact episode schema changed: {key}")
        model_requests = row.get("model_request_count")
        if isinstance(model_requests, bool) or not isinstance(model_requests, int) or model_requests <= 0:
            raise ValueError(f"{model} model_request_count must be a positive integer: {key}")
        if type(row.get("requested_success")) is not bool or row.get("failure_taxonomy") not in FAILURE_CLASSES:
            raise ValueError(f"{model} invalid success/taxonomy: {key}")
        if row["requested_success"] != (row["failure_taxonomy"] == "correct"):
            raise ValueError(f"{model} success/taxonomy disagreement: {key}")
        measurements = row.get("measurements")
        if not isinstance(measurements, dict):
            raise ValueError(f"{model} missing measurements: {key}")
        _require_keys(measurements, MEASUREMENT_KEYS, f"{model} measurements {key}")
        actions = row.get("actions_executed")
        if isinstance(actions, bool) or not isinstance(actions, int) or not 0 < actions <= 450:
            raise ValueError(f"{model} invalid action count: {key}")
        for field in ("actions_executed", "episode_length_steps", "executed_action_count"):
            if measurements.get(field) != actions:
                raise ValueError(f"{model} action-count field mismatch: {key}:{field}")
        margin = _finite(measurements.get("final_requested_signed_margin_m"), f"{model}:{key}.margin")
        endpoint = _finite(measurements.get("signed_final_lateral_offset_m"), f"{model}:{key}.endpoint")
        expected_margin = endpoint if relation == "left" else -endpoint
        _close(margin, expected_margin, f"{model}:{key}.requested margin sign")
        _close(measurements.get("requested_final_lateral_margin_m"), margin, f"{model}:{key}.requested margin alias")
        if row.get("initial_state_sha256") != evidence.get("matched_initial_state_sha256") or not _valid_sha(row.get("initial_state_sha256")):
            raise ValueError(f"{model} initial-state identity changed: {key}")
        source_report = row.get("source_whole_seed_report")
        if not isinstance(source_report, dict) or set(source_report) != {"path", "sha256"}:
            raise ValueError(f"{model} source report schema changed: {key}")
        if (source_report.get("path"), source_report.get("sha256")) != reports_by_seed[seed]:
            raise ValueError(f"{model} source report/evidence mismatch: {key}")
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != {"behavioral_jsonl", "decoded_future_trace", "executed_actions", "state_trace", "viewport_video"}:
            raise ValueError(f"{model} compact artifact schema changed: {key}")
        for descriptor in artifacts.values():
            if not isinstance(descriptor, dict) or set(descriptor) != {"bytes", "path", "sha256"}:
                raise ValueError(f"{model} compact artifact descriptor changed: {key}")
            if not isinstance(descriptor["bytes"], int) or descriptor["bytes"] <= 0 or not isinstance(descriptor["path"], str) or not _valid_sha(descriptor["sha256"]):
                raise ValueError(f"{model} invalid compact artifact identity: {key}")
        normalized = dict(row)
        normalized["margin_m"] = margin
        normalized["endpoint_m"] = endpoint
        by_key[key] = normalized
        orders.setdefault(seed, []).append(row["within_seed_execution_order"])
    expected = {(seed, form, relation) for seed in SEEDS for form in PROMPT_FORMS for relation in RELATIONS}
    if set(by_key) != expected:
        raise ValueError(f"{model} 160-cell compact episode grid is incomplete")
    if any(sorted(values) != list(range(1, 9)) for values in orders.values()):
        raise ValueError(f"{model} compact execution order is incomplete or duplicated")
    if [row["_source_row"] for row in episodes] != list(range(1, 161)):
        raise ValueError(f"{model} compact episode source-row sequence changed")
    _validate_summary(model, by_key, model_docs["summary"])
    return by_key


def _condition_summaries(rows: Mapping[tuple[int, str, str], Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for form in PROMPT_FORMS:
        conditions: dict[str, Any] = {}
        all_rows: list[Mapping[str, Any]] = []
        for relation in RELATIONS:
            selected = [rows[(seed, form, relation)] for seed in SEEDS]
            all_rows.extend(selected)
            successes = sum(row["requested_success"] for row in selected)
            conditions[relation] = {
                "successes": successes,
                "trials": 20,
                "success_rate": successes / 20.0,
                "model_requests": sum(row["model_request_count"] for row in selected),
                "wilson_95": _wilson(successes, 20),
                "failure_taxonomy": dict(sorted(Counter(row["failure_taxonomy"] for row in selected).items())),
            }
        shifts = [rows[(seed, form, "right")]["endpoint_m"] - rows[(seed, form, "left")]["endpoint_m"] for seed in SEEDS]
        conditions["both_directions"] = {
            "successes": sum(row["requested_success"] for row in all_rows),
            "trials": 40,
            "descriptive_rate": sum(row["requested_success"] for row in all_rows) / 40.0,
            "model_requests": sum(row["model_request_count"] for row in all_rows),
            "mean_requested_margin_m": statistics.fmean(row["margin_m"] for row in all_rows),
            "mean_right_minus_left_endpoint_shift_m": statistics.fmean(shifts),
            "median_right_minus_left_endpoint_shift_m": statistics.median(shifts),
            "endpoint_ordering_aligned": sum(value < 0.0 for value in shifts),
            "endpoint_ordering_anti_aligned": sum(value > 0.0 for value in shifts),
            "endpoint_ordering_ties": sum(value == 0.0 for value in shifts),
        }
        output[form] = conditions
    return output


def _direct_contrasts(
    cohort: str,
    model: str,
    rows: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summaries: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    for alternative in ALTERNATIVES:
        success_values: list[float] = []
        margin_values: list[float] = []
        endpoint_values: list[float] = []
        direct_successes: list[bool] = []
        alternative_successes: list[bool] = []
        for seed in SEEDS:
            direct = {relation: rows[(seed, "direct_command", relation)] for relation in RELATIONS}
            alt = {relation: rows[(seed, alternative, relation)] for relation in RELATIONS}
            direct_success = statistics.fmean(float(direct[r]["requested_success"]) for r in RELATIONS)
            alt_success = statistics.fmean(float(alt[r]["requested_success"]) for r in RELATIONS)
            direct_margin = statistics.fmean(direct[r]["margin_m"] for r in RELATIONS)
            alt_margin = statistics.fmean(alt[r]["margin_m"] for r in RELATIONS)
            direct_endpoint = direct["right"]["endpoint_m"] - direct["left"]["endpoint_m"]
            alt_endpoint = alt["right"]["endpoint_m"] - alt["left"]["endpoint_m"]
            success_delta = alt_success - direct_success
            margin_delta = alt_margin - direct_margin
            endpoint_delta = alt_endpoint - direct_endpoint
            _close(
                direct_margin,
                -0.5 * direct_endpoint,
                f"{model}:{seed}:direct requested-margin/endpoint identity",
            )
            _close(
                alt_margin,
                -0.5 * alt_endpoint,
                f"{model}:{seed}:{alternative} requested-margin/endpoint identity",
            )
            _close(
                margin_delta,
                -0.5 * endpoint_delta,
                f"{model}:{seed}:{alternative} contrast algebraic dependence",
            )
            success_values.append(success_delta)
            margin_values.append(margin_delta)
            endpoint_values.append(endpoint_delta)
            for relation in RELATIONS:
                direct_successes.append(direct[relation]["requested_success"])
                alternative_successes.append(alt[relation]["requested_success"])
            flat_rows.append({
                "cohort": cohort,
                "model_id": model,
                "seed": seed,
                "alternative_prompt_family": alternative,
                "direct_left_success": direct["left"]["requested_success"],
                "direct_right_success": direct["right"]["requested_success"],
                "alternative_left_success": alt["left"]["requested_success"],
                "alternative_right_success": alt["right"]["requested_success"],
                "direct_block_success_rate": direct_success,
                "alternative_block_success_rate": alt_success,
                "success_alternative_minus_direct": success_delta,
                "direct_mean_requested_margin_m": direct_margin,
                "alternative_mean_requested_margin_m": alt_margin,
                "requested_margin_alternative_minus_direct_m": margin_delta,
                "direct_right_minus_left_endpoint_shift_m": direct_endpoint,
                "alternative_right_minus_left_endpoint_shift_m": alt_endpoint,
                "endpoint_shift_alternative_minus_direct_m": endpoint_delta,
                "direct_left_source_row": direct["left"]["_source_row"],
                "direct_right_source_row": direct["right"]["_source_row"],
                "alternative_left_source_row": alt["left"]["_source_row"],
                "alternative_right_source_row": alt["right"]["_source_row"],
            })
        mean_margin = statistics.fmean(margin_values)
        mean_endpoint = statistics.fmean(endpoint_values)
        _close(
            mean_margin,
            -0.5 * mean_endpoint,
            f"{model}:{alternative} mean contrast algebraic dependence",
        )
        summaries[alternative] = {
            "matched_seed_blocks": 20,
            "direction_matched_episode_pairs": 40,
            "success": {
                "estimand": "mean over seed blocks of the balanced LEFT/RIGHT alternative-minus-direct requested-success difference",
                "mean": statistics.fmean(success_values),
                "median": statistics.median(success_values),
                "exact_matched_discordance": {
                    "both_success": sum(a and b for a, b in zip(direct_successes, alternative_successes)),
                    "direct_only": sum(a and not b for a, b in zip(direct_successes, alternative_successes)),
                    "alternative_only": sum(not a and b for a, b in zip(direct_successes, alternative_successes)),
                    "neither": sum(not a and not b for a, b in zip(direct_successes, alternative_successes)),
                    "discordant_pairs": sum(a != b for a, b in zip(direct_successes, alternative_successes)),
                },
            },
            "requested_margin_m": {
                "estimand": "mean over seed blocks of the balanced LEFT/RIGHT alternative-minus-direct final requested signed margin",
                "mean": mean_margin,
                "median": statistics.median(margin_values),
            },
            "right_minus_left_endpoint_shift_m": {
                "estimand": "mean over seed blocks of [(RIGHT-minus-LEFT endpoint shift under alternative) minus (RIGHT-minus-LEFT endpoint shift under direct)]",
                "mean": mean_endpoint,
                "median": statistics.median(endpoint_values),
                "sign_note": "Positive values attenuate the registered negative aligned RIGHT-minus-LEFT shift relative to direct command.",
            },
            "algebraic_dependency": {
                "identity": "requested_margin_alternative_minus_direct_m = -0.5 * endpoint_shift_alternative_minus_direct_m",
                "reason": "requested margin is +LEFT endpoint for LEFT and -RIGHT endpoint for RIGHT, balanced equally within each seed block",
                "validated_for_all_20_seed_blocks": True,
                "validated_for_reported_means": True,
                "metrics_are_not_independent_estimands": True,
            },
            "inference": "descriptive_only_not_registered",
        }
    return summaries, flat_rows


def analyze_documents(documents: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    _validate_registration(documents)
    cells = _normalize_cells(documents)
    cohorts: dict[str, Any] = {}
    matched_rows: list[dict[str, Any]] = []
    for model, info in MODELS.items():
        rows = _normalize_model(model, documents["models"][model], cells)
        conditions = _condition_summaries(rows)
        contrasts, flat = _direct_contrasts(info["cohort"], model, rows)
        matched_rows.extend(flat)
        failure_taxonomy = {
            name: sum(row["failure_taxonomy"] == name for row in rows.values())
            for name in sorted(FAILURE_CLASSES)
        }
        model_requests = sum(row["model_request_count"] for row in rows.values())
        valid_failures = sum(not row["requested_success"] for row in rows.values())
        cohorts[info["cohort"]] = {
            "model_id": model,
            "configuration_label": info["label"],
            "authenticated_population": {
                "behavioral_episodes": 160,
                "model_requests": model_requests,
                "valid_behavioral_failures": valid_failures,
                "matched_seed_blocks": 20,
                "prompt_forms": 4,
                "directions": 2,
                "valid_behavioral_failures_included": True,
                "infrastructure_attempts_included": False,
                "missing_value_imputation": "none",
            },
            "overall_failure_taxonomy": failure_taxonomy,
            "source_rows": {
                "registered_cells_file": CELLS,
                "registered_cell_rows_inclusive": info["cell_rows"],
                "episodes_file": info["episodes"],
                "episode_rows_inclusive": [1, 160],
            },
            "condition_summaries": conditions,
            "same_seed_direct_control_contrasts": contrasts,
        }

    if len(matched_rows) != 120:
        raise ValueError("H09/H10 direct-control output must contain exactly 120 matched seed contrasts")
    activity = {
        "annotation_media": 0, "behavioral_actions": 0, "behavioral_cells": 0,
        "confirmation_inference_cells": 0, "forecast_accuracy_estimates": 0,
        "human_responses": 0, "labels": 0, "model_requests": 0,
        "queue_descriptors": 0, "request_rows": 0, "request_selections": 0,
        "retrieval_jobs": 0, "runnable_manifests": 0,
    }
    gates = {
        "annotation_media_ready": False, "camera_alignment_ready": False,
        "confirmation_release_ready": False, "current_GM_PVC_worker_state_reconciled": False,
        "development_annotation_gate_ready": False, "human_annotation_ready": False,
        "live_quiescence_inferred": False, "resource_freeze_ready": False,
        "safe_to_release_new_work": False,
    }
    return {
        "schema_version": "wmf-retained-h09-h10-wording-control-v1",
        "provenance": dict(provenance),
        "design": {
            "study_id": STUDY,
            "experiment_id": EXPERIMENT,
            "full_prospective_registration": {
                "behavioral_cells": 480,
                "models": list(ALL_REGISTERED_MODELS),
                "includes_out_of_scope_groot": True,
            },
            "ablation_spec_h09_h10_slice": {
                "behavioral_episodes": 320,
                "cohorts": ["H09", "H10"],
                "models": list(MODELS),
                "excluded_registered_model": "groot_n17_droid_vla",
            },
            "matched_seeds": list(SEEDS),
            "prompt_forms": list(PROMPT_FORMS),
            "relations": list(RELATIONS),
            "direct_control": "direct_command",
            "descriptive_unit": "matched seed block with LEFT and RIGHT balanced within block",
            "continuous_estimand_dependency": "Within every seed and wording, balanced requested margin = -0.5 * registered RIGHT-minus-LEFT endpoint shift; their alternative-minus-direct contrasts are algebraically dependent.",
            "wording_block": "exploratory",
            "confirmatory_inference": "not_registered",
            "models_are_reported_separately": True,
        },
        "cohorts": cohorts,
        "matched_seed_contrasts": matched_rows,
        "reproduction": {
            "full_registered_cells_strictly_parsed": 480,
            "h09_h10_registered_cells_rejoined": 320,
            "h09_h10_compact_episode_rows_strictly_parsed": 320,
            "model_episode_rows": {info["cohort"]: 160 for info in MODELS.values()},
            "complete_seed_prompt_direction_grids": True,
            "registered_success_counts_and_wilson_intervals_exactly_reproduced": True,
            "registered_failure_taxonomies_exactly_reproduced": True,
            "registered_relation_endpoint_diagnostics_exactly_reproduced": True,
            "model_request_counts_exactly_reconciled": True,
            "failure_counts_and_taxonomies_exactly_reconciled": True,
            "requested_margin_endpoint_algebraic_dependency_validated": True,
            "same_seed_direct_control_contrasts_recomputed": 6,
            "matched_seed_contrast_rows": 120,
        },
        "source_manifest": {
            "compact_git_files_byte_rehashed_now": True,
            "authenticated_file_count": len(EXPECTED_SOURCES),
            "authenticated_bytes": sum(item["bytes"] for item in EXPECTED_SOURCES.values()),
            "prospective_registration_ancestry_verified": True,
            "raw_pvc_files_revalidated_now": False,
            "raw_source_paths_exported": False,
            "exported_source_paths_are_git_object_paths_only": True,
        },
        "claim_boundary": {
            "status": "retrospective compact-Git authentication and descriptive reproduction of a prospectively registered exploratory wording-control family",
            "wording_inference": "not_registered",
            "exploratory_only": True,
            "confirmatory_wording_claim": False,
            "equivalence_claim": False,
            "models_pooled": False,
            "new_model_inference": False,
            "new_robot_episodes": False,
            "new_human_labels": False,
            "forecast_accuracy_evidence": False,
            "confirmation_test_or_stage_gate": False,
            "changes_232_cell_core_bound": False,
            "maximum_core_behavioral_cells": 232,
            "rows_are_independent_scenes": False,
            "boundaries": [
                "The 160 rows per checkpoint are repeated prompt and direction conditions within 20 matched seed blocks on one neutral scene, not 160 independent scenes.",
                "All three alternative wordings are reported against the direct-command control; no favorable wording is selected alone.",
                "The registration fixes an exploratory descriptive contrast family but no power threshold, confirmatory test, or equivalence procedure.",
                "Nano and Edge are separate checkpoints and are never pooled or ranked by a combined effect.",
                "Final margins and endpoints are goal-stopped, outcome-dependent terminal measurements, not common-time trajectory contrasts.",
                "The compact Git evidence is byte-authenticated; external raw simulator, action-array, video, and PVC paths are neither exported nor revalidated.",
                "This is retained executed behavior, not language comprehension, new-scene generalization, generated-future accuracy, annotation evidence, or confirmation.",
            ],
        },
        "new_activity": activity,
        "gate_state": gates,
    }


def render_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"


def render_csv(report: Mapping[str, Any]) -> str:
    fields = [
        "cohort", "model_id", "seed", "alternative_prompt_family",
        "direct_left_success", "direct_right_success", "alternative_left_success",
        "alternative_right_success", "direct_block_success_rate",
        "alternative_block_success_rate", "success_alternative_minus_direct",
        "direct_mean_requested_margin_m", "alternative_mean_requested_margin_m",
        "requested_margin_alternative_minus_direct_m",
        "direct_right_minus_left_endpoint_shift_m",
        "alternative_right_minus_left_endpoint_shift_m",
        "endpoint_shift_alternative_minus_direct_m", "direct_left_source_row",
        "direct_right_source_row", "alternative_left_source_row", "alternative_right_source_row",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in report["matched_seed_contrasts"]:
        writer.writerow(row)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    documents, provenance = read_pinned_sources(ROOT)
    report = analyze_documents(documents, provenance)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(render_json(report), encoding="utf-8")
    args.csv.write_text(render_csv(report), encoding="utf-8")
    print(render_json(report), end="")


if __name__ == "__main__":
    main()
