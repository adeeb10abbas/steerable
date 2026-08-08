"""Pure runtime helpers for the V3-E004 DROID behavioral bridge.

This module deliberately imports no simulator or policy package.  It keeps the
model dispatch table and the canonical simulator-export envelope testable on a
workstation before an Isaac pod is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .episode_compiler import EXPORT_SCHEMA
from .runtime_contract import E004Cell, E004RuntimeBundle, RuntimeContractError, sha256_file


@dataclass(frozen=True)
class DROIDModelSpec:
    model_id: str
    policy_id: str
    action_horizon: int
    action_dim: int
    action_cap: int
    future_interface: str
    endpoint_port: int | None = None


MODEL_SPECS: dict[str, DROIDModelSpec] = {
    "pi05_current_stack_droid": DROIDModelSpec(
        model_id="pi05_current_stack_droid",
        policy_id="pi05_v2a010_current",
        action_horizon=15,
        action_dim=8,
        action_cap=450,
        future_interface="actions_only",
    ),
    "cosmos3_nano_policy_droid": DROIDModelSpec(
        model_id="cosmos3_nano_policy_droid",
        policy_id="cosmos3_nano_v2",
        action_horizon=32,
        action_dim=8,
        action_cap=450,
        future_interface="decoded_rgb_future",
        endpoint_port=18011,
    ),
    "dreamzero_droid_action_cfg": DROIDModelSpec(
        model_id="dreamzero_droid_action_cfg",
        policy_id="dreamzero_v2",
        action_horizon=8,
        action_dim=8,
        action_cap=450,
        future_interface="native_latent_and_official_decoded_future",
    ),
    "cosmos3_edge_policy_droid": DROIDModelSpec(
        model_id="cosmos3_edge_policy_droid",
        policy_id="cosmos3_v2",
        action_horizon=32,
        action_dim=8,
        action_cap=450,
        future_interface="decoded_rgb_future",
        endpoint_port=18010,
    ),
}


def model_spec(cell: E004Cell, *, endpoint_port: int) -> DROIDModelSpec:
    """Return and cross-check the frozen model contract for one queue cell."""

    try:
        spec = MODEL_SPECS[cell.model_id]
    except KeyError as exc:
        raise RuntimeContractError(f"unsupported E004 DROID model: {cell.model_id}") from exc
    requirement = cell.row.get("runtime_identity_requirement")
    if not isinstance(requirement, Mapping):
        raise RuntimeContractError("cell runtime identity requirement is missing")
    expected = {
        "action_cap": spec.action_cap,
        "action_horizon": 24 if cell.model_id == "dreamzero_droid_action_cfg" else spec.action_horizon,
    }
    if cell.model_id == "pi05_current_stack_droid":
        expected["action_dim"] = spec.action_dim
    for key, wanted in expected.items():
        if requirement.get(key) != wanted:
            raise RuntimeContractError(f"{cell.cell_id} runtime differs for {key}")
    if spec.endpoint_port is not None and endpoint_port != spec.endpoint_port:
        raise RuntimeContractError(
            f"{cell.model_id} endpoint port must be {spec.endpoint_port}, got {endpoint_port}"
        )
    if cell.model_id == "dreamzero_droid_action_cfg" and endpoint_port == 5000:
        raise RuntimeContractError("the protected DreamZero port 5000 is prohibited")
    return spec


def bind_runtime_identity(
    *,
    cell: E004Cell,
    bundle: E004RuntimeBundle,
    source_path: Path,
    source_expected_sha256: str,
    lane_release_path: Path,
    lane_release_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Write the exact E004 wrapper around a model-specific runtime proof."""

    source_path = Path(source_path).resolve()
    lane_release_path = Path(lane_release_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise RuntimeContractError(f"refusing to overwrite runtime identity: {output_path}")
    if not source_path.is_file() or sha256_file(source_path) != source_expected_sha256:
        raise RuntimeContractError("model runtime manifest does not match the lane release")
    if not lane_release_path.is_file() or sha256_file(lane_release_path) != lane_release_sha256:
        raise RuntimeContractError("lane release changed before runtime binding")
    try:
        source_value = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(f"model runtime manifest is invalid: {exc}") from exc
    if not isinstance(source_value, dict):
        raise RuntimeContractError("model runtime manifest must be an object")
    claimed_model = source_value.get("model_id")
    if claimed_model not in (None, cell.model_id):
        raise RuntimeContractError("model runtime manifest names a different checkpoint")
    value = {
        "schema_version": "vla-wam-shared-v3e004-bound-runtime-identity-v1",
        "study_id": cell.row["study_id"],
        "amendment_id": cell.row["amendment_id"],
        "registered_cell_id": cell.cell_id,
        "registered_cell_sha256": cell.row_sha256,
        "model_id": cell.model_id,
        "runtime_identity_requirement": cell.row["runtime_identity_requirement"],
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "candidate_sha256": bundle.candidate_sha256,
        "source_runtime_manifest": {
            "path": str(source_path),
            "sha256": source_expected_sha256,
            "bytes": source_path.stat().st_size,
        },
        "lane_release": {
            "path": str(lane_release_path),
            "sha256": lane_release_sha256,
            "bytes": lane_release_path.stat().st_size,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return value


def simulator_export_envelope(
    *,
    cell: E004Cell,
    bundle: E004RuntimeBundle,
    steps: list[dict[str, Any]],
    requested_success: bool,
    right_censored: bool,
    final_detached_release: bool,
    live_gate: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    executed_action_trace: Mapping[str, Any],
    viewport_video: Mapping[str, Any],
    future_evidence: Any,
    future_evidence_status: str,
) -> dict[str, Any]:
    """Build the identity-complete export consumed by ``episode_compiler``."""

    return {
        "schema_version": EXPORT_SCHEMA,
        "study_id": cell.row["study_id"],
        "amendment_id": cell.row["amendment_id"],
        "registered_cell_id": cell.cell_id,
        "registered_cell_sha256": cell.row_sha256,
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "candidate_sha256": bundle.candidate_sha256,
        "model_id": cell.model_id,
        "arena": cell.row["arena"],
        "environment_seed": cell.environment_seed,
        "sampling_seed": cell.sampling_seed,
        "matched_pair_id": cell.matched_pair_id,
        "requested_relation": cell.relation,
        "prompt": cell.row["prompt"],
        "prompt_sha256": cell.row["prompt_sha256"],
        "symmetry_level_s": cell.symmetry_level_s,
        "success_predicate_id": cell.row["success_predicate_id"],
        "runtime_identity_requirement": cell.row["runtime_identity_requirement"],
        "instruction_controller": "static_episode_prompt",
        "steps": steps,
        "actions_executed": len(steps) - 1,
        "requested_success": requested_success,
        "right_censored": right_censored,
        "final_detached_release": final_detached_release,
        "live_scene_gate": dict(live_gate),
        "runtime_identity": dict(runtime_identity),
        "executed_action_trace": dict(executed_action_trace),
        "viewport_video": dict(viewport_video),
        "future_evidence": future_evidence,
        "future_evidence_status": future_evidence_status,
    }
