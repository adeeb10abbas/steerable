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
