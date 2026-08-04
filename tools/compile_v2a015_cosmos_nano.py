#!/usr/bin/env python3
"""Compile the six V2-A015 Cosmos3 Nano g=1 behavioral cells.

Input is exactly three explicit, hash-bearing pair manifests.  The compiler
does not search simulator or policy output directories and therefore cannot
silently select a retry or omit a valid failure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import v2a015_compilation as shared


SCHEMA = "vla-wam-shared-v2-cosmos3-nano-v2a015-g1-result-v1"
PAIR_SCHEMA = (
    "vla-wam-shared-v2-cosmos3-nano-v2a015-g1-pair-collection-v1"
)
TRACE_SCHEMA = (
    "vla-wam-shared-v2-cosmos3-nano-v2a015-g1-action-future-trace-v1"
)
FIXED_GATE_SCHEMA = (
    "vla-wam-shared-v2-cosmos3-nano-policy-droid-v2a015-g1-"
    "fixed-observation-v1"
)
MODEL_ID = "cosmos3_nano_policy_droid"
ARM_ID = "cosmos3_nano_no_cfg_g1"
CHECKPOINT = "nvidia/Cosmos3-Nano-Policy-DROID"
CHECKPOINT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
SOURCE_COMMIT = "411d25b2e35bc441126f48c44a4b93e1c0564274"
SIMULATOR_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
BASELINE_PATH = (
    shared.REPO_ROOT
    / "artifacts/vla_wam_shared_v2/pilot/expansion/"
    "cosmos3_nano_policy_droid_direct_gate.json"
)
BASELINE_SHA256 = "4a6cc1d61593c7ba5272e1707f6bbe51261f7d23438070992bd75fd9e95fdb93"
AMENDMENT_PATH = (
    shared.REPO_ROOT
    / "artifacts/vla_wam_shared_v2/pilot/"
    "post_result_cfg_ablation_v2a015_amendment.json"
)
ACTION_HORIZON = 32


def _validate_amendment(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
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
        "guidance": 1.0,
        "baseline_guidance": 3.0,
        "num_steps": 4,
        "shift": 5.0,
        "action_chunk_shape": [32, 8],
        "behavioral_episode_count": 6,
    }
    if not isinstance(arm, dict):
        raise RuntimeError(f"Amendment lacks arm {ARM_ID}")
    for key, value in expected.items():
        if arm.get(key) != value:
            raise RuntimeError(
                f"V2-A015 Cosmos arm mismatch for {key}: expected={value!r}, observed={arm.get(key)!r}"
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
    return amendment, arm


def _validate_fixed_gate(path: Path) -> dict[str, Any]:
    gate = shared.load_json(path)
    expected = {
        "schema_version": FIXED_GATE_SCHEMA,
        "status": "passed",
        "model_id": MODEL_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "amendment_id": shared.AMENDMENT_ID,
        "arm_id": ARM_ID,
        "guidance": 1.0,
        "baseline_guidance": 3.0,
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            raise RuntimeError(
                f"Cosmos fixed gate mismatch for {key}: expected={value!r}, observed={gate.get(key)!r}"
            )
    for key in (
        "left_repeat_action_bit_identical",
        "left_repeat_future_bit_identical",
        "left_right_action_distinct",
        "left_right_future_distinct",
    ):
        if gate.get("metrics", {}).get(key) is not True:
            raise RuntimeError(f"Cosmos fixed gate did not pass {key}")
    if [row.get("condition") for row in gate.get("records", [])] != [
        "left",
        "left_exact_repeat",
        "right",
    ]:
        raise RuntimeError("Cosmos fixed gate lacks the exact three-request release probe")
    return gate


def _validate_pair_manifest(
    path: Path, *, amendment_path: Path, fixed_gate_path: Path, baseline_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        "simulator_repository_commit": SIMULATOR_COMMIT,
        "pair_id": f"seed{seed}",
        "environment_seed": seed,
        "sampling_seed": seed,
        "guidance": 1.0,
        "baseline_guidance": 3.0,
        "num_steps": 4,
        "shift": 5.0,
        "action_chunk_shape": [32, 8],
        "future_contract": "decoded 33-frame RGB future for every policy request",
        "output_folder_name": f"v2a015_cosmos3_nano_g1_seed{seed}",
    }
    if seed not in shared.SEEDS:
        raise RuntimeError(f"Unauthorized Cosmos pair seed: {seed}")
    for key, value in expected.items():
        if pair.get(key) != value:
            raise RuntimeError(
                f"Cosmos pair {seed} mismatch for {key}: expected={value!r}, observed={pair.get(key)!r}"
            )
    base = path.parent
    recorded_amendment = shared.validate_file_record(
        pair.get("amendment", {}),
        base,
        f"Cosmos pair {seed} amendment",
        require_bytes=True,
    )
    if recorded_amendment != amendment_path.resolve():
        raise RuntimeError(f"Cosmos pair {seed} records a different amendment")
    recorded_gate = shared.validate_file_record(
        pair.get("fixed_observation_release_gate", {}),
        base,
        f"Cosmos pair {seed} fixed-observation gate",
        require_bytes=True,
    )
    if recorded_gate != fixed_gate_path.resolve():
        raise RuntimeError(f"Cosmos pair {seed} and CLI fixed-observation gates differ")
    recorded_baseline = shared.validate_file_record(
        pair.get("baseline_result", {}),
        base,
        f"Cosmos pair {seed} baseline result",
        require_bytes=True,
    )
    if recorded_baseline != baseline_path.resolve():
        raise RuntimeError(f"Cosmos pair {seed} and CLI baseline results differ")
    simulator_root = shared.resolve_path(pair["simulator_output_root"], base)
    if simulator_root.name != f"v2a015_cosmos3_nano_g1_seed{seed}":
        raise RuntimeError(f"Cosmos pair {seed} simulator output root changed")
    adapters = pair.get("adapter_files", {})
    if set(adapters) != {"runner", "client"}:
        raise RuntimeError(f"Cosmos pair {seed} lacks exact adapter file records")
    for label, record in adapters.items():
        shared.validate_file_record(
            record, base, f"Cosmos pair {seed} {label} adapter", require_bytes=True
        )
    cells = pair.get("cells")
    if not isinstance(cells, list) or len(cells) != 2:
        raise RuntimeError(f"Cosmos pair {seed} must contain exactly two cells")
    observed = [shared.validate_cell_protocol(cell) for cell in cells]
    if set(observed) != {(seed, "left"), (seed, "right")}:
        raise RuntimeError(f"Cosmos pair {seed} is not an exact LEFT/RIGHT pair")
    for cell in cells:
        if (
            cell.get("amendment_sha256") != pair["amendment"]["sha256"]
            or cell.get("fixed_observation_gate_sha256")
            != pair["fixed_observation_release_gate"]["sha256"]
        ):
            raise RuntimeError(f"Cosmos pair {seed} cell provenance hashes differ")
        if shared.resolve_path(cell["simulator_task_dir"], base).parent != simulator_root:
            raise RuntimeError(f"Cosmos pair {seed} cell escapes its simulator output root")
    checks = pair.get("pair_checks", {})
    if (
        checks.get("cell_count") != 2
        or checks.get("relations") != ["left", "right"]
        or checks.get("prompts")
        != [shared.PROMPTS["left"], shared.PROMPTS["right"]]
        or checks.get("all_prompts_episode_static") is not True
        or checks.get("all_exposed_futures_retained") is not True
    ):
        raise RuntimeError(f"Cosmos pair {seed} checks do not certify the exact pair")
    return pair, cells


def _load_policy_evidence(
    cell: dict[str, Any], base: Path, simulator_actions: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    seed, relation = shared.validate_cell_protocol(cell)
    cell_expected = {
        "amendment_id": shared.AMENDMENT_ID,
        "arm_id": ARM_ID,
        "model_id": MODEL_ID,
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "official_repository_commit": SOURCE_COMMIT,
        "open_loop_execution_horizon": ACTION_HORIZON,
        "guidance": 1.0,
        "baseline_guidance": 3.0,
        "num_steps": 4,
        "shift": 5.0,
        "action_chunk_shape": [32, 8],
        "future_contract": "decoded 33-frame RGB future for every policy request",
    }
    for key, value in cell_expected.items():
        if cell.get(key) != value:
            raise RuntimeError(f"Cosmos cell {seed}/{relation} mismatch for {key}")
    metadata_path = shared.validate_file_record(
        cell.get("action_future_trace_metadata", {}),
        base,
        f"Cosmos action trace {seed}/{relation}",
        require_bytes=True,
    )
    metadata = shared.load_json(metadata_path)
    expected = {
        "schema_version": TRACE_SCHEMA,
        "prompt": shared.PROMPTS[relation],
        "sampling_seed_base": seed,
        "model_id": MODEL_ID,
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "official_source_commit": SOURCE_COMMIT,
        "environment_seed": seed,
        "requested_relation": relation,
        "amendment_id": shared.AMENDMENT_ID,
        "arm_id": ARM_ID,
        "guidance": 1.0,
        "baseline_guidance": 3.0,
        "baseline_result_artifact": str(BASELINE_PATH.relative_to(shared.REPO_ROOT)),
        "baseline_result_sha256": BASELINE_SHA256,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"Cosmos trace mismatch {seed}/{relation} for {key}: expected={value!r}, observed={metadata.get(key)!r}"
            )
    executed_path = shared.validate_file_record(
        metadata.get("executed_actions", {}), metadata_path.parent, "Cosmos executed actions"
    )
    cell_executed_path = shared.validate_file_record(
        cell.get("executed_actions", {}), base, f"Cosmos pair executed actions {seed}/{relation}"
    )
    if (
        cell_executed_path != executed_path
        or cell.get("executed_actions", {}).get("sha256")
        != metadata.get("executed_actions", {}).get("sha256")
    ):
        raise RuntimeError(f"Cosmos pair manifest and trace disagree on executed actions: {seed}/{relation}")
    executed = np.load(executed_path, allow_pickle=False)
    if executed.shape != simulator_actions.shape or not np.array_equal(executed, simulator_actions):
        raise RuntimeError(f"Cosmos trace differs from simulator HDF5: {seed}/{relation}")
    requests = metadata.get("requests")
    expected_count = math.ceil(len(executed) / ACTION_HORIZON)
    if not isinstance(requests, list) or len(requests) != expected_count:
        raise RuntimeError(f"Cosmos request count mismatch: {seed}/{relation}")
    model_requests = cell.get("model_requests")
    if not isinstance(model_requests, list) or len(model_requests) != expected_count:
        raise RuntimeError(f"Cosmos pair manifest request count mismatch: {seed}/{relation}")
    retained = []
    for index, (request, pair_request) in enumerate(zip(requests, model_requests, strict=True)):
        expected_request = {
            "request_index": index,
            "requested_sampling_seed": seed,
            "server_sampling_seed": seed,
            "environment_seed": seed,
            "prompt": shared.PROMPTS[relation],
            "requested_relation": relation,
            "model_id": MODEL_ID,
            "checkpoint": CHECKPOINT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "official_source_commit": SOURCE_COMMIT,
            "amendment_id": shared.AMENDMENT_ID,
            "arm_id": ARM_ID,
            "guidance": 1.0,
            "baseline_guidance": 3.0,
            "baseline_result_artifact": str(BASELINE_PATH.relative_to(shared.REPO_ROOT)),
            "baseline_result_sha256": BASELINE_SHA256,
            "amendment_sha256": cell["amendment_sha256"],
            "fixed_observation_gate_sha256": cell[
                "fixed_observation_gate_sha256"
            ],
            "action_shape": [32, 8],
        }
        for key, value in expected_request.items():
            if request.get(key) != value:
                raise RuntimeError(
                    f"Cosmos request mismatch {seed}/{relation}/{index} for {key}"
                )
        action_path = shared.resolve_path(request["action_path"], metadata_path.parent)
        future_path = shared.resolve_path(request["future_path"], metadata_path.parent)
        for candidate, hash_key, label in (
            (action_path, "action_sha256", "returned action"),
            (future_path, "future_sha256", "decoded future"),
        ):
            if not candidate.is_file() or shared.sha256(candidate) != request.get(hash_key):
                raise RuntimeError(
                    f"Cosmos {label} hash mismatch: {seed}/{relation}/{index}/{candidate}"
                )
        pair_action_path = shared.validate_file_record(
            pair_request.get("returned_action", {}),
            base,
            f"Cosmos pair returned action {seed}/{relation}/{index}",
            require_bytes=True,
        )
        pair_future_path = shared.validate_file_record(
            pair_request.get("decoded_future", {}),
            base,
            f"Cosmos pair decoded future {seed}/{relation}/{index}",
            require_bytes=True,
        )
        if (
            pair_request.get("request_index") != index
            or pair_request.get("sampling_seed") != seed
            or pair_action_path != action_path
            or pair_future_path != future_path
            or pair_request["returned_action"].get("sha256")
            != request.get("action_sha256")
            or pair_request["decoded_future"].get("sha256")
            != request.get("future_sha256")
        ):
            raise RuntimeError(
                f"Cosmos pair manifest and trace disagree on request {seed}/{relation}/{index}"
            )
        action = np.load(action_path, allow_pickle=False)
        future = np.load(future_path, allow_pickle=False)
        if action.shape != (32, 8) or not np.isfinite(action).all():
            raise RuntimeError(f"Cosmos returned-action contract mismatch: {seed}/{relation}/{index}")
        if future.ndim != 4 or future.shape[0] != 33 or future.shape[-1] != 3:
            raise RuntimeError(f"Cosmos decoded-future contract mismatch: {seed}/{relation}/{index}")
        retained.append(
            {
                "request_index": index,
                "prompt": shared.PROMPTS[relation],
                "returned_action": shared.file_record(action_path),
                "returned_action_shape": list(action.shape),
                "decoded_future": shared.file_record(future_path),
                "decoded_future_shape": list(future.shape),
            }
        )
    if cell.get("decoded_future_count") != len(retained):
        raise RuntimeError(f"Cosmos pair decoded-future count mismatch: {seed}/{relation}")
    return (
        {
            "action_trace_metadata": shared.file_record(metadata_path),
            "executed_action_trace": shared.file_record(executed_path),
            "policy_request_count": len(retained),
            "future_interface": "decoded_rgb_uint8_33_frames_per_policy_request",
            "decoded_future_count": len(retained),
            "imagined_future_requests": retained,
            "missing_or_unexposed_future_evidence_scored_as_zero": False,
        },
        executed,
    )


def compile_result(args: argparse.Namespace) -> dict[str, Any]:
    baseline_record = shared.validate_exact_file(
        args.baseline_result,
        expected_sha256=BASELINE_SHA256,
        label="preserved Cosmos3 Nano g=3 baseline",
    )
    baseline = shared.load_json(args.baseline_result)
    if (
        baseline.get("model_id") != MODEL_ID
        or baseline.get("amendment_id") != "V2-A011"
        or baseline.get("summary", {}).get("successes") != 6
    ):
        raise RuntimeError("Preserved Cosmos baseline identity or summary changed")
    amendment, arm = _validate_amendment(args.amendment)
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

    pair_manifests, cells = [], []
    for pair_path in args.pair_manifest:
        pair, pair_cells = _validate_pair_manifest(
            pair_path,
            amendment_path=args.amendment,
            fixed_gate_path=args.fixed_observation_gate,
            baseline_path=args.baseline_result,
        )
        pair_manifests.append((pair_path, pair))
        cells.extend((pair_path.parent, cell) for cell in pair_cells)
    if (
        len(pair_manifests) != 3
        or {pair["environment_seed"] for _, pair in pair_manifests}
        != set(shared.SEEDS)
    ):
        raise RuntimeError(
            "Cosmos compilation requires exactly pair manifests for seeds 8300, 8301, and 8302"
        )
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
            "schema_version": "vla-wam-shared-v2-cosmos3-nano-v2a015-g1-episode-v1",
            "model_id": MODEL_ID,
            "amendment_id": shared.AMENDMENT_ID,
            "arm_id": ARM_ID,
            "checkpoint": CHECKPOINT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "source_commit": SOURCE_COMMIT,
            "guidance": 1.0,
            "baseline_guidance": 3.0,
            **simulation,
            **policy,
        }
        episodes.append(episode)
        actions[(seed, relation)] = executed
    pairs = shared.build_pairs(episodes, actions, horizon=ACTION_HORIZON)
    summary = shared.configuration_summary(episodes, pairs)
    if summary["valid_episode_count"] != 6:
        raise RuntimeError("Cosmos V2-A015 result is not the complete six-cell denominator")
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
            "guidance": 1.0,
            "baseline_guidance": 3.0,
            "num_steps": arm["num_steps"],
            "shift": arm["shift"],
            "executed_open_loop_horizon": ACTION_HORIZON,
        },
        "exact_prompts": shared.PROMPTS,
        "metric_definitions": shared.metric_definitions(),
        "summary": summary,
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
            "preserved_baseline_result": baseline_record,
            "invalid_attempts": invalid,
            "runtime_interventions": interventions,
        },
        "denominator_policy": {
            "valid_behavioral_episodes": 6,
            "valid_failures_retained": summary["valid_failure_count"],
            "fixed_observation_requests_excluded": 3,
            "technical_invalid_or_partial_attempts_excluded": invalid["row_count"],
            "runtime_interventions_do_not_remove_valid_behavior": True,
        },
        "claim_boundary": (
            "Descriptive post-result n=6 Cosmos3 Nano g=1 ablation compared only with the hash-pinned g=3 V2-A011 baseline. "
            "No powered/general performance-gain claim and no pooling with DreamZero or RoboTwin."
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
    for name in ("fixed_observation_gate", "amendment", "baseline_result", "result_output"):
        setattr(args, name, getattr(args, name).resolve())
    args.invalid_attempt_ledger = [path.resolve() for path in args.invalid_attempt_ledger]
    args.runtime_intervention_ledger = [
        path.resolve() for path in args.runtime_intervention_ledger
    ]
    result = compile_result(args)
    shared.dump_json(args.result_output, result, overwrite=args.overwrite)
    print(json.dumps({"status": "complete", **result["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
