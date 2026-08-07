#!/usr/bin/env python3
"""Fail-closed whole-seed contract for Cosmos V3-C001 behavior."""

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
from .groot_behavioral_contract import (
    TASK_SPECS,
    prompt_condition,
    validate_task_sources,
)


COSMOS_MODELS = frozenset({
    "cosmos3_edge_policy_droid",
    "cosmos3_nano_policy_droid",
})


def validate_seed_block(
    *,
    study_root: Path,
    execution_plan: Path,
    release_manifest: Path,
    registration_manifest: Path,
    model_id: str,
    seed: int,
    require_fresh_outputs: bool = True,
) -> dict[str, Any]:
    if model_id not in COSMOS_MODELS:
        raise ContractError("Cosmos Phase-C model must be Edge Policy or Nano Policy")
    if seed not in SEEDS:
        raise ContractError("Phase-C Cosmos seeds are exactly 8500-8519")
    release = validate_release_manifest(
        release_manifest,
        model_id=model_id,
        registration_manifest_sha256=sha256_file(registration_manifest),
    )
    plan = json.loads(execution_plan.read_text())
    expected = {
        "schema_version": "vla-wam-shared-v3c-four-phrasings-execution-plan-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": model_id,
        "release_manifest_sha256": release.release_manifest_sha256,
        "registration_manifest_sha256": sha256_file(registration_manifest),
        "runtime_identity_sha256": release.runtime_identity_sha256,
        "inference_launched": False,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ContractError(f"Cosmos Phase-C execution plan mismatch for {key}")
    cells = [cell for cell in plan.get("cells", []) if cell.get("environment_seed") == seed]
    cells.sort(key=lambda cell: cell.get("within_seed_execution_order", -1))
    if len(cells) != 8 or [cell.get("within_seed_execution_order") for cell in cells] != list(range(1, 9)):
        raise ContractError("Cosmos Phase-C seed block must contain exactly orders 1..8")

    observed: list[tuple[str, str]] = []
    task_rows: list[dict[str, Any]] = []
    raw_dirs: set[str] = set()
    for cell in cells:
        form, relation = prompt_condition(cell.get("prompt", ""))
        observed.append((form, relation))
        if cell.get("prompt") != PROMPTS[form][relation]:
            raise ContractError("Cosmos Phase-C prompt bytes changed")
        if cell.get("prompt_sha256") != prompt_sha256(cell["prompt"]):
            raise ContractError("Cosmos Phase-C plan prompt hash mismatch")
        expected_id = f"v3c001:droid:{model_id}:seed{seed}:{form}:{relation}"
        if cell.get("registered_cell_id") != expected_id or cell.get("sampling_seed") != seed:
            raise ContractError("Cosmos Phase-C plan cell identity mismatch")
        raw_dir = cell.get("raw_cell_directory")
        if not isinstance(raw_dir, str) or raw_dir in raw_dirs:
            raise ContractError("Cosmos Phase-C raw cell directories must be unique")
        raw_dirs.add(raw_dir)
        if require_fresh_outputs and Path(raw_dir).exists():
            raise ContractError(f"refusing to overwrite retained Phase-C cell: {raw_dir}")
        filename, task_name = TASK_SPECS[(form, relation)]
        task_rows.append({
            **cell,
            "prompt_family": form,
            "relation": relation,
            "task_file": str(
                study_root
                / "experiments/v3/phase_c_four_phrasings/groot_task_files"
                / filename
            ),
            "task_name": task_name,
        })
    if observed != randomized_conditions(model_id, seed):
        raise ContractError("Cosmos Phase-C within-seed condition order changed")
    return {
        "schema_version": "vla-wam-shared-v3c-cosmos-seed-block-preflight-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": model_id,
        "seed": seed,
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "release_manifest_sha256": release.release_manifest_sha256,
        "execution_plan_sha256": sha256_file(execution_plan),
        "task_source_sha256": validate_task_sources(study_root),
        "cells": task_rows,
        "execution_status": "prompt_aware_cosmos_bridge_contract_ready_live_zero_action_registration_pending",
    }


def validate_live_task_registration(
    *, bridge_preflight_path: Path, task_registration_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    bridge = json.loads(bridge_preflight_path.read_text())
    registration = json.loads(task_registration_path.read_text())
    if bridge.get("model_id") not in COSMOS_MODELS:
        raise ContractError("Cosmos bridge preflight has an unsupported model")
    required_registration = {
        "schema_version": "vla-wam-shared-v3c-cosmos-live-task-registration-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": bridge["model_id"],
        "seed": bridge.get("seed"),
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "executed_action_count": 0,
        "renderer_initialized": True,
    }
    for key, expected in required_registration.items():
        if registration.get(key) != expected:
            raise ContractError(f"Cosmos live task-registration mismatch for {key}")
    proof = registration.get("bridge_preflight", {})
    if (
        proof.get("path") != str(bridge_preflight_path)
        or proof.get("sha256") != sha256_file(bridge_preflight_path)
    ):
        raise ContractError("Cosmos live task-registration bridge proof changed")
    if registration.get("max_cube_position_spread_m", 1.0) > registration.get(
        "matched_reset_tolerance_m", 0.0
    ) or registration.get("max_bowl_position_spread_m", 1.0) > registration.get(
        "matched_reset_tolerance_m", 0.0
    ):
        raise ContractError("Cosmos live task-registration resets are not matched")
    bridge_cells = bridge.get("cells", [])
    registered_by_id = {
        row.get("registered_cell_id"): row for row in registration.get("cells", [])
    }
    if len(bridge_cells) != 8 or len(registered_by_id) != 8:
        raise ContractError("Cosmos live task-registration requires eight unique cells")
    for cell in bridge_cells:
        observed = registered_by_id.get(cell.get("registered_cell_id"))
        if observed is None:
            raise ContractError("Cosmos live task-registration is missing a planned cell")
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
                    f"Cosmos live task-registration mismatch for "
                    f"{cell['registered_cell_id']}.{key}"
                )
    return bridge, registration


def validate_live_output_contract(cell: dict[str, Any], *, fresh: bool = True) -> dict[str, Path]:
    raw_dir = Path(cell["raw_cell_directory"])
    required = cell.get("required_outputs", {})
    expected = {
        "behavioral_jsonl": raw_dir / "episode.jsonl",
        "executed_actions": raw_dir / "executed_actions.npy",
        "simulator_viewport_video": raw_dir / "viewport.mp4",
        "state_trace": raw_dir / "state_trace.jsonl",
        "decoded_future_directory": raw_dir / "decoded_futures",
        "action_future_metadata": raw_dir / "action_future_trace.json",
    }
    for key in (
        "behavioral_jsonl", "executed_actions", "simulator_viewport_video", "state_trace"
    ):
        if required.get(key) != str(expected[key]):
            raise ContractError(
                f"Cosmos Phase-C output path changed for {cell['registered_cell_id']}.{key}"
            )
    if required.get("decoded_future") != "required_when_exposed_by_runtime":
        raise ContractError("Cosmos Phase-C future-retention contract changed")
    if fresh and (
        raw_dir.exists()
        or any(path.exists() for path in expected.values())
    ):
        raise ContractError(f"refusing to overwrite retained Phase-C cell: {raw_dir}")
    return expected
