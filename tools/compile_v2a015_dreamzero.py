#!/usr/bin/env python3
"""Compile three explicit V2-A015 DreamZero s=2 pair manifests.

The three runner-emitted pair manifests are the only source of candidate
behavioral cells.  Invalid/partial attempts and runtime interventions are
supplied as separate ledgers and never inferred from directory names.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import v2a015_compilation as shared


SCHEMA = "vla-wam-shared-v2-dreamzero-v2a015-s2-result-v1"
PAIR_SCHEMA = "vla-wam-shared-v2-dreamzero-v2a015-pair-collection-v1"
TRACE_SCHEMA = "vla-wam-shared-v2-dreamzero-v2a015-action-trace-v1"
FUTURE_SCHEMA = "vla-wam-shared-v2-dreamzero-v2a015-future-retention-v1"
FIXED_GATE_SCHEMA = "vla-wam-shared-v2-dreamzero-v2a015-fixed-observation-probe-v1"
SERVER_SCHEMA = "vla-wam-shared-v2-dreamzero-v2a015-server-contract-v1"
MODEL_ID = "dreamzero_droid_action_cfg"
ARM_ID = "dreamzero_action_cfg_s2"
CHECKPOINT = "GEAR-Dreams/DreamZero-DROID"
CHECKPOINT_REVISION = "96ad344138c66e82536422432ad742f015784942"
SOURCE_COMMIT = "ab790c198fbce33503358efbbd4187ce9a89adf3"
OFFICIAL_NOISE_SEED = 1140
ACTION_HORIZON = 8
RETURNED_HORIZON = 24
BASELINE_PATH = (
    shared.REPO_ROOT
    / "artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_droid_direct_gate.json"
)
BASELINE_SHA256 = "4c76cdc3ca9eaf227d21d160199408f22e1b3dd7a71176a5a5dbe22223714461"
AMENDMENT_PATH = (
    shared.REPO_ROOT
    / "artifacts/vla_wam_shared_v2/pilot/"
    "post_result_cfg_ablation_v2a015_amendment.json"
)


def _validate_amendment(path: Path) -> dict[str, Any]:
    amendment = shared.load_json(path)
    if amendment.get("amendment_id") != shared.AMENDMENT_ID:
        raise RuntimeError("Supplied amendment is not V2-A015")
    arm = next(
        (value for value in amendment.get("arms", []) if value.get("arm_id") == ARM_ID),
        None,
    )
    expected = {
        "model_id": MODEL_ID,
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_commit": SOURCE_COMMIT,
        "action_guidance": 2.0,
        "baseline_action_guidance_equivalent": 1.0,
        "video_guidance": 5.0,
        "runtime_num_inference_steps": 16,
        "dit_cache": True,
        "evaluated_dit_steps": 8,
        "action_chunk_shape": [24, 8],
        "executed_open_loop_horizon": ACTION_HORIZON,
        "behavioral_episode_count": 6,
    }
    if not isinstance(arm, dict):
        raise RuntimeError(f"Amendment lacks arm {ARM_ID}")
    for key, value in expected.items():
        if arm.get(key) != value:
            raise RuntimeError(
                f"DreamZero arm mismatch for {key}: expected={value!r}, observed={arm.get(key)!r}"
            )
    grid = amendment.get("behavioral_grid", {})
    if (
        grid.get("prompts") != shared.PROMPTS
        or grid.get("environment_seeds") != list(shared.SEEDS)
        or grid.get("sampling_seed_labels") != list(shared.SEEDS)
        or grid.get("prompt_controller") != "episode_static"
        or grid.get("oracle_actions") != 0
        or grid.get("subtask_coach") is not False
        or grid.get("prompt_switching") is not False
        or grid.get("progress_conditioned_language") is not False
        or grid.get("simulator_video_required") is not True
        or grid.get("executed_action_trace_required") is not True
        or grid.get("all_exposed_futures_retained") is not True
    ):
        raise RuntimeError("V2-A015 behavioral grid contract changed")
    return arm


def _validate_fixed_gate(path: Path) -> dict[str, Any]:
    gate = shared.load_json(path)
    expected = {
        "schema_version": FIXED_GATE_SCHEMA,
        "status": "passed",
        "amendment_id": shared.AMENDMENT_ID,
        "action_cfg_style_scale": 2.0,
        "video_cfg_scale": 5.0,
        "sampling_seed_label": 8300,
        "internal_gate_passed": True,
        "comparison_gate_passed": True,
        "release_gate_passed": True,
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            raise RuntimeError(
                f"DreamZero fixed gate mismatch for {key}: expected={value!r}, observed={gate.get(key)!r}"
            )
    metrics = gate.get("metrics", {})
    for key in (
        "all_actions_finite_shape_24x8",
        "all_latents_finite",
        "left_exact_repeat_action_array_equal",
        "left_exact_repeat_latent_tensor_equal",
    ):
        if metrics.get(key) is not True:
            raise RuntimeError(f"DreamZero fixed gate did not pass {key}")
    if not float(metrics.get("left_vs_right_action_rms", 0.0)) > 0.0:
        raise RuntimeError("DreamZero fixed gate has no LEFT/RIGHT action response")
    comparison = gate.get("comparison", {})
    if comparison.get("status") != "passed" or comparison.get(
        "reference_action_cfg_style_scale"
    ) != 1.0:
        raise RuntimeError("DreamZero s=2 gate is not bound to passed scale-1 equivalence")
    return gate


def _validate_pair_manifest(
    path: Path, *, amendment_path: Path, fixed_gate_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    pair = shared.load_json(path)
    seed = int(pair.get("environment_seed", -1))
    expected = {
        "schema_version": PAIR_SCHEMA,
        "status": "complete_behavioral_pair_candidate",
        "amendment_id": shared.AMENDMENT_ID,
        "arm_id": ARM_ID,
        "model_id": MODEL_ID,
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "official_repository_commit": SOURCE_COMMIT,
        "condition": "both",
        "environment_seed": seed,
        "sampling_seed": seed,
        "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
        "action_cfg_style_scale": 2.0,
        "baseline_action_cfg_equivalent": 1.0,
        "video_cfg_scale": 5.0,
        "simulator_gpu_lane": "raytrace-rtxpro6000-ali",
    }
    if seed not in shared.SEEDS:
        raise RuntimeError(f"Unauthorized DreamZero pair seed: {seed}")
    for key, value in expected.items():
        if pair.get(key) != value:
            raise RuntimeError(
                f"DreamZero pair {seed} mismatch for {key}: expected={value!r}, observed={pair.get(key)!r}"
            )
    base = path.parent
    recorded_amendment = shared.validate_file_record(
        pair.get("amendment", {}), base, f"DreamZero pair {seed} amendment", require_bytes=True
    )
    if recorded_amendment != amendment_path.resolve():
        raise RuntimeError(f"DreamZero pair {seed} records a different amendment")
    recorded_gate = shared.validate_file_record(
        pair.get("fixed_observation_release_gate", {}),
        base,
        f"DreamZero pair {seed} fixed gate",
        require_bytes=True,
    )
    if recorded_gate != fixed_gate_path.resolve():
        raise RuntimeError(f"DreamZero pair {seed} records a different fixed gate")
    contract_path = shared.validate_file_record(
        pair.get("server_contract", {}),
        base,
        f"DreamZero pair {seed} server contract",
        require_bytes=True,
    )
    contract = shared.load_json(contract_path)
    contract_expected = {
        "schema_version": SERVER_SCHEMA,
        "amendment_id": shared.AMENDMENT_ID,
        "official_repository_commit": SOURCE_COMMIT,
        "official_noise_seed": OFFICIAL_NOISE_SEED,
        "world_size": 2,
        "enable_dit_cache": True,
        "runtime_num_inference_steps": 16,
        "evaluated_dit_steps_with_cache": 8,
        "action_cfg_style_scale": 2.0,
        "video_cfg_scale": 5.0,
    }
    for key, value in contract_expected.items():
        if contract.get(key) != value:
            raise RuntimeError(f"DreamZero pair {seed} server contract mismatch for {key}")
    future_root = shared.resolve_path(pair["future_root"], base)
    if str(contract.get("future_root")) != str(future_root) or not future_root.is_dir():
        raise RuntimeError(f"DreamZero pair {seed} future-root contract mismatch")
    cells = pair.get("cells")
    if not isinstance(cells, list) or len(cells) != 2:
        raise RuntimeError(f"DreamZero pair {seed} must contain exactly two cells")
    observed = [shared.validate_cell_protocol(cell) for cell in cells]
    if set(observed) != {(seed, "left"), (seed, "right")}:
        raise RuntimeError(f"DreamZero pair {seed} is not an exact LEFT/RIGHT pair")
    return pair, cells, contract_path


def _load_policy_evidence(
    cell: dict[str, Any], base: Path, simulator_actions: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    seed, relation = shared.validate_cell_protocol(cell)
    cell_expected = {
        "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
        "open_loop_execution_horizon": ACTION_HORIZON,
        "action_cfg_style_scale": 2.0,
        "video_cfg_scale": 5.0,
        "simulator_gpu_lane": "raytrace-rtxpro6000-ali",
    }
    for key, value in cell_expected.items():
        if cell.get(key) != value:
            raise RuntimeError(f"DreamZero cell {seed}/{relation} mismatch for {key}")

    trace_path = shared.validate_file_record(
        cell.get("action_trace_metadata", {}),
        base,
        f"DreamZero action trace {seed}/{relation}",
        require_bytes=True,
    )
    trace = shared.load_json(trace_path)
    trace_expected = {
        "schema_version": TRACE_SCHEMA,
        "amendment_id": shared.AMENDMENT_ID,
        "arm_id": ARM_ID,
        "model_id": MODEL_ID,
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "official_repository_commit": SOURCE_COMMIT,
        "environment_seed": seed,
        "sampling_seed_label": seed,
        "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
        "prompt": shared.PROMPTS[relation],
        "requested_relation": relation,
        "open_loop_execution_horizon": ACTION_HORIZON,
        "returned_action_horizon": RETURNED_HORIZON,
        "action_cfg_style_scale": 2.0,
        "baseline_action_cfg_equivalent": 1.0,
        "video_cfg_scale": 5.0,
        "baseline_result_artifact": str(BASELINE_PATH.relative_to(shared.REPO_ROOT)),
        "baseline_result_sha256": BASELINE_SHA256,
    }
    for key, value in trace_expected.items():
        if trace.get(key) != value:
            raise RuntimeError(
                f"DreamZero trace {seed}/{relation} mismatch for {key}: expected={value!r}, observed={trace.get(key)!r}"
            )
    evidence_paths: dict[str, Path] = {}
    for key in ("executed_actions", "returned_raw_chunks", "returned_executable_chunks"):
        manifest_path = shared.validate_file_record(
            cell.get(key, {}), base, f"DreamZero pair-manifest {key} {seed}/{relation}"
        )
        trace_record_path = shared.validate_file_record(
            trace.get(key, {}), trace_path.parent, f"DreamZero trace {key} {seed}/{relation}"
        )
        if manifest_path != trace_record_path or cell[key].get("sha256") != trace[key].get("sha256"):
            raise RuntimeError(f"DreamZero pair manifest and trace disagree for {key}: {seed}/{relation}")
        evidence_paths[key] = trace_record_path
    executed = np.load(evidence_paths["executed_actions"], allow_pickle=False)
    raw_chunks = np.load(evidence_paths["returned_raw_chunks"], allow_pickle=False)
    executable_chunks = np.load(evidence_paths["returned_executable_chunks"], allow_pickle=False)
    request_count = int(trace["request_count"])
    if executed.shape != simulator_actions.shape or not np.array_equal(executed, simulator_actions):
        raise RuntimeError(f"DreamZero trace differs from simulator HDF5: {seed}/{relation}")
    if raw_chunks.shape != (request_count, RETURNED_HORIZON, 8):
        raise RuntimeError(f"DreamZero raw returned-chunk contract mismatch: {seed}/{relation}")
    if executable_chunks.shape != raw_chunks.shape:
        raise RuntimeError(f"DreamZero executable returned-chunk contract mismatch: {seed}/{relation}")
    reconstructed = executable_chunks[:, :ACTION_HORIZON].reshape(-1, 8)[: len(executed)]
    if request_count != math.ceil(len(executed) / ACTION_HORIZON) or not np.array_equal(
        reconstructed, executed
    ):
        raise RuntimeError(f"DreamZero executed/open-loop reconstruction mismatch: {seed}/{relation}")

    manifest_record = cell.get("future_manifest", {})
    future_path = shared.validate_file_record(
        manifest_record, base, f"DreamZero future manifest {seed}/{relation}"
    )
    trace_future = trace.get("future_manifest", {})
    if (
        shared.resolve_path(trace_future.get("path", ""), trace_path.parent) != future_path
        or trace_future.get("sha256") != manifest_record.get("sha256")
    ):
        raise RuntimeError(f"DreamZero pair manifest and trace bind different futures: {seed}/{relation}")
    future = shared.load_json(future_path)
    future_expected = {
        "schema_version": FUTURE_SCHEMA,
        "amendment_id": shared.AMENDMENT_ID,
        "official_repository_commit": SOURCE_COMMIT,
        "action_cfg_style_scale": 2.0,
        "video_cfg_scale": 5.0,
        "request_count": request_count,
    }
    for key, value in future_expected.items():
        if future.get(key) != value:
            raise RuntimeError(f"DreamZero future manifest mismatch for {seed}/{relation}/{key}")
    if manifest_record.get("request_count") != request_count:
        raise RuntimeError(f"DreamZero pair future request count mismatch: {seed}/{relation}")
    future_requests = future.get("requests")
    if not isinstance(future_requests, list) or len(future_requests) != request_count:
        raise RuntimeError(f"DreamZero future retention is incomplete: {seed}/{relation}")
    retained = []
    for index, (request, raw_chunk) in enumerate(zip(future_requests, raw_chunks, strict=True)):
        if (
            request.get("request_index") != index
            or request.get("prompt") != shared.PROMPTS[relation]
            or request.get("action_cfg_style_scale") != 2.0
        ):
            raise RuntimeError(f"DreamZero future request mismatch: {seed}/{relation}/{index}")
        action_path = shared.validate_file_record(
            request.get("returned_action", {}), future_path.parent, "DreamZero returned server action"
        )
        latent_path = shared.validate_file_record(
            request.get("latent_video", {}), future_path.parent, "DreamZero latent future"
        )
        returned = np.load(action_path, allow_pickle=False)
        if returned.shape != (RETURNED_HORIZON, 8) or not np.array_equal(returned, raw_chunk):
            raise RuntimeError(f"DreamZero retained server action mismatch: {seed}/{relation}/{index}")
        retained.append(
            {
                "request_index": index,
                "prompt": shared.PROMPTS[relation],
                "returned_action": shared.file_record(action_path),
                "latent_video": shared.file_record(latent_path),
                "latent_video_shape": request.get("latent_video", {}).get("shape"),
            }
        )
    decoded = []
    for record in future.get("official_reset_decode", []):
        decoded_path = shared.validate_file_record(
            record, future_path.parent, "DreamZero official decoded future"
        )
        decoded.append(shared.file_record(decoded_path))
    if not decoded or manifest_record.get("official_decode_count") != len(decoded):
        raise RuntimeError(f"DreamZero official decoded future is absent: {seed}/{relation}")
    return (
        {
            "action_trace_metadata": shared.file_record(trace_path),
            "executed_action_trace": shared.file_record(evidence_paths["executed_actions"]),
            "returned_raw_chunks": shared.file_record(evidence_paths["returned_raw_chunks"]),
            "returned_executable_chunks": shared.file_record(
                evidence_paths["returned_executable_chunks"]
            ),
            "policy_request_count": request_count,
            "future_interface": "joint_action_and_latent_video_prediction_with_official_decode_path",
            "future_manifest": shared.file_record(future_path),
            "latent_future_request_count": len(retained),
            "latent_future_requests": retained,
            "official_decoded_future_count": len(decoded),
            "official_decoded_futures": decoded,
            "missing_or_unexposed_future_evidence_scored_as_zero": False,
        },
        executed,
    )


def compile_result(args: argparse.Namespace) -> dict[str, Any]:
    baseline_record = shared.validate_exact_file(
        args.baseline_result,
        expected_sha256=BASELINE_SHA256,
        label="preserved DreamZero s=1-equivalent baseline",
    )
    baseline = shared.load_json(args.baseline_result)
    if (
        baseline.get("model_id") != "dreamzero_droid"
        or baseline.get("amendment_id") != "V2-A007"
        or baseline.get("valid_episode_count") != 6
    ):
        raise RuntimeError("Preserved DreamZero baseline identity or denominator changed")
    arm = _validate_amendment(args.amendment)
    fixed_gate = _validate_fixed_gate(args.fixed_observation_gate)
    invalid = shared.ledger_summary(
        args.invalid_attempt_ledger,
        invalid=True,
        expected_model_id=MODEL_ID,
        expected_arm_id=ARM_ID,
    )
    interventions = shared.ledger_summary(
        args.runtime_intervention_ledger,
        invalid=False,
        expected_model_id=MODEL_ID,
        expected_arm_id=ARM_ID,
    )

    pair_manifests, cells, contract_paths = [], [], []
    for pair_path in args.pair_manifest:
        pair, pair_cells, contract_path = _validate_pair_manifest(
            pair_path,
            amendment_path=args.amendment,
            fixed_gate_path=args.fixed_observation_gate,
        )
        pair_manifests.append((pair_path, pair))
        cells.extend((pair_path.parent, cell) for cell in pair_cells)
        contract_paths.append(contract_path)
    if len(pair_manifests) != 3 or {pair["environment_seed"] for _, pair in pair_manifests} != set(shared.SEEDS):
        raise RuntimeError("DreamZero compilation requires exactly pair manifests for seeds 8300, 8301, and 8302")
    if len(set(contract_paths)) != 1:
        raise RuntimeError("All DreamZero behavioral pairs must bind the same live s=2 server contract")
    shared.validate_complete_grid([cell for _, cell in cells])

    episodes, actions = [], {}
    for base, cell in sorted(
        cells,
        key=lambda item: (int(item[1]["environment_seed"]), item[1]["requested_relation"]),
    ):
        seed, relation = shared.validate_cell_protocol(cell)
        simulation, simulator_actions = shared.load_simulator_cell(cell, base)
        policy, executed = _load_policy_evidence(cell, base, simulator_actions)
        episode = {
            "schema_version": "vla-wam-shared-v2-dreamzero-v2a015-s2-episode-v1",
            "model_id": MODEL_ID,
            "amendment_id": shared.AMENDMENT_ID,
            "arm_id": ARM_ID,
            "checkpoint": CHECKPOINT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "official_repository_commit": SOURCE_COMMIT,
            "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
            "action_cfg_style_scale": 2.0,
            "baseline_action_cfg_equivalent": 1.0,
            "video_cfg_scale": 5.0,
            "negative_branch_caveat": arm["negative_branch_caveat"],
            **simulation,
            **policy,
        }
        episodes.append(episode)
        actions[(seed, relation)] = executed
    pairs = shared.build_pairs(episodes, actions, horizon=ACTION_HORIZON)
    summary = shared.configuration_summary(episodes, pairs)
    behavioral_latents = sum(row["latent_future_request_count"] for row in episodes)
    behavioral_decodes = sum(row["official_decoded_future_count"] for row in episodes)
    return {
        "schema_version": SCHEMA,
        "status": "complete",
        "compiled_at_git_head": args.compiled_at_git_head,
        "model_id": MODEL_ID,
        "arena": "droid_robolab",
        "amendment_id": shared.AMENDMENT_ID,
        "arm_id": ARM_ID,
        "configuration": {
            "checkpoint": CHECKPOINT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "source_commit": SOURCE_COMMIT,
            "action_cfg_style_scale": 2.0,
            "baseline_action_cfg_equivalent": 1.0,
            "video_cfg_scale": 5.0,
            "negative_branch": "released fixed visual-quality negative prompt",
            "negative_branch_caveat": arm["negative_branch_caveat"],
            "runtime_num_inference_steps": 16,
            "dit_cache": True,
            "evaluated_dit_steps": 8,
            "executed_open_loop_horizon": ACTION_HORIZON,
        },
        "exact_prompts": shared.PROMPTS,
        "metric_definitions": shared.metric_definitions(),
        "summary": summary,
        "future_retention_audit": {
            "behavioral_episode_count": len(episodes),
            "behavioral_latent_future_count": behavioral_latents,
            "behavioral_official_decoded_future_count": behavioral_decodes,
            "missing_or_unexposed_future_evidence_scored_as_zero": False,
        },
        "pairs": pairs,
        "episodes": episodes,
        "provenance": {
            "amendment": shared.file_record(args.amendment),
            "pair_manifests": [
                shared.file_record(path) for path, _ in sorted(pair_manifests)
            ],
            "fixed_observation_release_gate": shared.file_record(
                args.fixed_observation_gate
            ),
            "fixed_observation_release_metrics": fixed_gate["metrics"],
            "server_contract": shared.file_record(contract_paths[0]),
            "preserved_baseline_result": baseline_record,
            "invalid_attempts": invalid,
            "runtime_interventions": interventions,
        },
        "denominator_policy": {
            "valid_behavioral_episodes": 6,
            "valid_failures_retained": summary["valid_failure_count"],
            "fixed_observation_requests_excluded": 6,
            "technical_invalid_or_partial_attempts_excluded": invalid["row_count"],
            "runtime_interventions_do_not_remove_valid_behavior": True,
        },
        "claim_boundary": (
            "Descriptive post-result n=6 DreamZero CFG-style negative-branch action-guidance s=2 ablation compared only with the hash-pinned s=1-equivalent V2-A007 baseline. "
            "This is not an official DreamZero action-CFG mode, a powered/general performance-gain claim, or evidence pooled with Cosmos3 or RoboTwin."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, action="append", required=True)
    parser.add_argument("--fixed-observation-gate", type=Path, required=True)
    parser.add_argument(
        "--invalid-attempt-ledger", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--runtime-intervention-ledger", type=Path, action="append", required=True
    )
    parser.add_argument("--amendment", type=Path, default=AMENDMENT_PATH)
    parser.add_argument("--baseline-result", type=Path, default=BASELINE_PATH)
    parser.add_argument("--result-output", type=Path, required=True)
    parser.add_argument("--compiled-at-git-head", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.pair_manifest = [path.resolve() for path in args.pair_manifest]
    args.fixed_observation_gate = args.fixed_observation_gate.resolve()
    args.invalid_attempt_ledger = [path.resolve() for path in args.invalid_attempt_ledger]
    args.runtime_intervention_ledger = [
        path.resolve() for path in args.runtime_intervention_ledger
    ]
    args.amendment = args.amendment.resolve()
    args.baseline_result = args.baseline_result.resolve()
    args.result_output = args.result_output.resolve()
    result = compile_result(args)
    shared.dump_json(args.result_output, result, overwrite=args.overwrite)
    print(json.dumps({"status": "complete", **result["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
