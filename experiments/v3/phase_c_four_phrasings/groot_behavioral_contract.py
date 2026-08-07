#!/usr/bin/env python3
"""Prompt-aware, whole-seed contract for the GR00T V3-C001 bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contract import (
    EXPERIMENT_ID,
    PROMPTS,
    SEEDS,
    ContractError,
    prompt_sha256,
    randomized_conditions,
    sha256_file,
    validate_release_manifest,
)


MODEL_ID = "groot_n17_droid_vla"
TASK_ROOT = Path("experiments/v3/phase_c_four_phrasings/groot_task_files")
TASK_SPECS = {
    ("direct_command", "left"): ("direct_command_left.py", "V3C001GrootDirectCommandLeftTask"),
    ("direct_command", "right"): ("direct_command_right.py", "V3C001GrootDirectCommandRightTask"),
    ("short_command", "left"): ("short_command_left.py", "V3C001GrootShortCommandLeftTask"),
    ("short_command", "right"): ("short_command_right.py", "V3C001GrootShortCommandRightTask"),
    ("goal_as_outcome", "left"): ("goal_as_outcome_left.py", "V3C001GrootGoalAsOutcomeLeftTask"),
    ("goal_as_outcome", "right"): ("goal_as_outcome_right.py", "V3C001GrootGoalAsOutcomeRightTask"),
    ("desired_plus_negated_opposite", "left"): (
        "desired_plus_negated_opposite_left.py", "V3C001GrootContrastiveLeftTask"
    ),
    ("desired_plus_negated_opposite", "right"): (
        "desired_plus_negated_opposite_right.py", "V3C001GrootContrastiveRightTask"
    ),
}


def prompt_condition(prompt: str) -> tuple[str, str]:
    """Resolve only exact registered bytes; substring parsing is prohibited."""

    matches = [
        (form, relation)
        for form, relations in PROMPTS.items()
        for relation, registered in relations.items()
        if prompt == registered
    ]
    if len(matches) != 1:
        raise ContractError("GR00T Phase-C prompt is not one exact registered string")
    return matches[0]


def validate_task_sources(study_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for condition, (filename, _) in TASK_SPECS.items():
        path = study_root / TASK_ROOT / filename
        if not path.is_file():
            raise ContractError(f"missing GR00T Phase-C task file: {path}")
        text = path.read_text()
        exact_prompt = PROMPTS[condition[0]][condition[1]]
        if exact_prompt not in text:
            raise ContractError(f"task source does not retain exact prompt bytes: {path}")
        hashes[str(TASK_ROOT / filename)] = sha256_file(path)
    return hashes


def validate_seed_block(
    *,
    study_root: Path,
    execution_plan: Path,
    release_manifest: Path,
    registration_manifest: Path,
    seed: int,
    require_fresh_outputs: bool = True,
) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ContractError("Phase-C GR00T seeds are exactly 8500-8519")
    release = validate_release_manifest(
        release_manifest,
        model_id=MODEL_ID,
        registration_manifest_sha256=sha256_file(registration_manifest),
    )
    plan = json.loads(execution_plan.read_text())
    expected = {
        "schema_version": "vla-wam-shared-v3c-four-phrasings-execution-plan-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "release_manifest_sha256": release.release_manifest_sha256,
        "registration_manifest_sha256": sha256_file(registration_manifest),
        "runtime_identity_sha256": release.runtime_identity_sha256,
        "inference_launched": False,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ContractError(f"GR00T Phase-C execution plan mismatch for {key}")
    cells = [cell for cell in plan.get("cells", []) if cell.get("environment_seed") == seed]
    cells.sort(key=lambda cell: cell.get("within_seed_execution_order", -1))
    if len(cells) != 8 or [cell.get("within_seed_execution_order") for cell in cells] != list(range(1, 9)):
        raise ContractError("GR00T Phase-C seed block must contain exactly orders 1..8")
    observed = []
    task_rows = []
    raw_dirs: set[str] = set()
    for cell in cells:
        form, relation = prompt_condition(cell.get("prompt", ""))
        observed.append((form, relation))
        if cell.get("prompt_sha256") != prompt_sha256(cell["prompt"]):
            raise ContractError("GR00T Phase-C plan prompt hash mismatch")
        expected_id = f"v3c001:droid:{MODEL_ID}:seed{seed}:{form}:{relation}"
        if cell.get("registered_cell_id") != expected_id or cell.get("sampling_seed") != seed:
            raise ContractError("GR00T Phase-C plan cell identity mismatch")
        raw_dir = cell.get("raw_cell_directory")
        if not isinstance(raw_dir, str) or raw_dir in raw_dirs:
            raise ContractError("GR00T Phase-C raw cell directories must be unique")
        raw_dirs.add(raw_dir)
        if require_fresh_outputs and Path(raw_dir).exists():
            raise ContractError(f"refusing to overwrite retained Phase-C cell: {raw_dir}")
        filename, task_name = TASK_SPECS[(form, relation)]
        task_rows.append({
            **cell,
            "prompt_family": form,
            "relation": relation,
            "task_file": str(study_root / TASK_ROOT / filename),
            "task_name": task_name,
        })
    if observed != randomized_conditions(MODEL_ID, seed):
        raise ContractError("GR00T Phase-C within-seed condition order changed")
    return {
        "schema_version": "vla-wam-shared-v3c-groot-seed-block-preflight-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "seed": seed,
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "release_manifest_sha256": release.release_manifest_sha256,
        "execution_plan_sha256": sha256_file(execution_plan),
        "task_source_sha256": validate_task_sources(study_root),
        "cells": task_rows,
        "execution_status": "prompt_aware_bridge_contract_ready_live_zero_action_registration_pending",
    }


def validate_live_task_registration(
    *, bridge_preflight_path: Path, task_registration_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the real-Isaac, zero-action task-registration gate."""

    bridge = json.loads(bridge_preflight_path.read_text())
    registration = json.loads(task_registration_path.read_text())
    required_registration = {
        "schema_version": "vla-wam-shared-v3c-groot-live-task-registration-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "seed": bridge.get("seed"),
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "executed_action_count": 0,
        "renderer_initialized": True,
    }
    for key, expected in required_registration.items():
        if registration.get(key) != expected:
            raise ContractError(f"GR00T live task-registration mismatch for {key}")
    proof = registration.get("bridge_preflight", {})
    if (
        proof.get("path") != str(bridge_preflight_path)
        or proof.get("sha256") != sha256_file(bridge_preflight_path)
    ):
        raise ContractError("GR00T live task-registration bridge proof changed")
    if registration.get("max_cube_position_spread_m", 1.0) > registration.get(
        "matched_reset_tolerance_m", 0.0
    ) or registration.get("max_bowl_position_spread_m", 1.0) > registration.get(
        "matched_reset_tolerance_m", 0.0
    ):
        raise ContractError("GR00T live task-registration resets are not matched")
    bridge_cells = bridge.get("cells", [])
    registration_cells = registration.get("cells", [])
    if len(bridge_cells) != 8 or len(registration_cells) != 8:
        raise ContractError("GR00T live task-registration requires all eight cells")
    registered_by_id = {
        row.get("registered_cell_id"): row for row in registration_cells
    }
    if len(registered_by_id) != 8:
        raise ContractError("GR00T live task-registration contains duplicate cells")
    for cell in bridge_cells:
        observed = registered_by_id.get(cell.get("registered_cell_id"))
        if observed is None:
            raise ContractError("GR00T live task-registration is missing a planned cell")
        expected = {
            "within_seed_execution_order": cell["within_seed_execution_order"],
            "task_name": cell["task_name"],
            "prompt": cell["prompt"],
            "left_predicate_at_reset": False,
            "right_predicate_at_reset": False,
            "model_requests": 0,
            "actions_executed": 0,
        }
        for key, value in expected.items():
            if observed.get(key) != value:
                raise ContractError(
                    f"GR00T live task-registration cell mismatch for "
                    f"{cell['registered_cell_id']}.{key}"
                )
    return bridge, registration


def validate_live_output_contract(cell: dict[str, Any], *, fresh: bool = True) -> dict[str, Path]:
    """Resolve the four bounded raw outputs and reject path substitution."""

    raw_dir = Path(cell["raw_cell_directory"])
    required = cell.get("required_outputs", {})
    expected = {
        "behavioral_jsonl": raw_dir / "episode.jsonl",
        "executed_actions": raw_dir / "executed_actions.npy",
        "simulator_viewport_video": raw_dir / "viewport.mp4",
        "state_trace": raw_dir / "state_trace.jsonl",
    }
    for key, path in expected.items():
        if required.get(key) != str(path):
            raise ContractError(
                f"GR00T Phase-C output path changed for {cell['registered_cell_id']}.{key}"
            )
    if required.get("decoded_future") != "required_when_exposed_by_runtime":
        raise ContractError("GR00T Phase-C future-retention contract changed")
    if fresh and (raw_dir.exists() or any(path.exists() for path in expected.values())):
        raise ContractError(f"refusing to overwrite retained Phase-C cell: {raw_dir}")
    return expected
