"""Compile V3-C002 raw episodes without parsing language to recover a goal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import NormalDist, fmean, stdev
from typing import Any, Iterable, Mapping

from .contract import (
    AMENDMENT_ID,
    MODEL_ID,
    STUDY_ID,
    C002Cell,
    ContractError,
    canonical_json_sha256,
    finite_number,
    load_cells,
    require,
    sha256_file,
    validate_exact_runtime_contract,
)


RAW_SCHEMA = "vla-wam-shared-v3c002-raw-episode-v1"
EPISODE_SCHEMA = "vla-wam-shared-v3c002-compact-episode-v1"
PAIR_SCHEMA = "vla-wam-shared-v3c002-prompt-form-pair-v1"
RESULT_SCHEMA = "vla-wam-shared-v3c002-results-v1"
BOOTSTRAP_RESAMPLES = 20_000
DEPTH_MARGIN_M = 0.0415
BINARY_MARGIN = 0.1556


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        require(line.strip(), f"blank JSONL source line {line_number}")
        try:
            row = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ContractError(f"invalid JSONL source line {line_number}: {exc}") from exc
        require(isinstance(row, dict), f"JSONL source line {line_number} is not an object")
        rows.append(row)
    return rows


def _artifact(record: Any, label: str) -> dict[str, Any]:
    require(isinstance(record, dict), f"{label} artifact is missing")
    path = Path(str(record.get("path", "")))
    require(path.is_file(), f"{label} artifact does not exist: {path}")
    require(record.get("bytes") == path.stat().st_size, f"{label} bytes changed")
    require(record.get("sha256") == sha256_file(path), f"{label} digest changed")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": record["sha256"]}


def _cone(step: Mapping[str, Any], relation: str) -> bool:
    """Literal pure-Python transcription of E004's frozen 45-degree scorer."""

    object_xyz, reference_xyz = step.get("object_xyz"), step.get("reference_xyz")
    require(isinstance(object_xyz, list) and isinstance(reference_xyz, list) and len(object_xyz) == len(reference_xyz) == 3, "frozen scorer needs two 3-vectors")
    forward = float(object_xyz[0]) - float(reference_xyz[0])
    lateral = float(object_xyz[1]) - float(reference_xyz[1])
    radius = math.hypot(forward, lateral)
    margin = lateral if relation == "left" else -lateral
    return radius > 1e-8 and margin / radius >= math.cos(math.radians(45.0))


def _first_sustained(mask: list[bool], width: int = 3) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == width:
            return index - width + 1
    return None


def _frozen_success(steps: Any, physical_goal: str, detached_release: bool) -> bool:
    """Exact E004/B001 binary predicate, kept dependency-free for lane tests."""

    require(isinstance(steps, list) and steps, "raw state trace is missing")
    return bool(_cone(steps[-1], physical_goal) and detached_release)


def _failure_category(steps: Any, physical_goal: str, detached_release: bool, success: bool) -> str:
    """Exact E004 taxonomy precedence, using physical-goal metadata only."""

    require(isinstance(steps, list), "raw state trace is missing")
    if success:
        return "correct"
    if not any(bool(step.get("object_grabbed")) for step in steps):
        return "pick_failed"
    opposite = "right" if physical_goal == "left" else "left"
    if _first_sustained([_cone(step, opposite) for step in steps]) is not None and all(_cone(step, opposite) for step in steps[-3:]):
        return "wrong_side"
    if _first_sustained([_cone(step, physical_goal) for step in steps]) is not None and all(_cone(step, physical_goal) for step in steps[-3:]) and not detached_release:
        return "release_failed"
    return "transport_failed"


def _validate_request_seeds(raw: Mapping[str, Any], cell: C002Cell) -> list[int]:
    events = raw.get("request_events")
    require(isinstance(events, list) and events, "raw request events are missing")
    result = []
    for index, event in enumerate(events):
        require(isinstance(event, dict) and event.get("replan_index") == index, "request replan index is not contiguous")
        expected = cell.seed * 1000 + index
        require(event.get("request_seed") == expected, "request seed violates registered formula")
        result.append(expected)
    return result


def compile_episode(raw: Mapping[str, Any], *, cell: C002Cell, registration_sha256: str, queue_sha256: str, exact_runtime_contract: Mapping[str, Any]) -> dict[str, Any]:
    exact_sha = validate_exact_runtime_contract(exact_runtime_contract)
    require(raw.get("schema_version") == RAW_SCHEMA, "raw episode schema changed")
    for key, expected in (
        ("study_id", STUDY_ID),
        ("amendment_id", AMENDMENT_ID),
        ("cell_id", cell.cell_id),
        ("cell_sha256", cell.row_sha256),
        ("registration_sha256", registration_sha256),
        ("queue_sha256", queue_sha256),
        ("model_id", MODEL_ID),
        ("physical_goal", cell.physical_goal),
        ("surface_direction_word", cell.row["surface_direction_word"]),
        ("prompt", cell.row["prompt"]),
        ("prompt_utf8_hex", cell.row["prompt_utf8_hex"]),
        ("prompt_sha256", cell.row["prompt_sha256"]),
    ):
        require(raw.get(key) == expected, f"raw episode differs from registered cell for {key}")
    detached = raw.get("final_detached_release")
    require(type(detached) is bool, "detached-release flag is invalid")
    success = _frozen_success(raw.get("state_trace"), cell.physical_goal, detached)
    require(raw.get("reported_frozen_task_success") is success, "raw frozen success differs from frozen scorer")
    signed = finite_number(raw.get("signed_final_lateral_offset"), "signed final lateral offset")
    depth = finite_number(raw.get("requested_side_depth"), "requested-side depth")
    expected_depth = signed if cell.physical_goal == "left" else -signed
    require(math.isclose(depth, expected_depth, rel_tol=0.0, abs_tol=1e-12), "requested-side depth is not scored by registered physical goal")
    request_seeds = _validate_request_seeds(raw, cell)
    state_hash = raw.get("initial_state_sha256")
    require(isinstance(state_hash, str) and len(state_hash) == 64, "initial state hash is missing")
    runtime = raw.get("runtime_identity")
    require(isinstance(runtime, dict), "runtime identity is missing")
    expected_identity = exact_runtime_contract["identity_values"]
    for key in ("checkpoint_digest", "source_commit", "simulator_identity", "renderer_backend", "policy_cameras"):
        require(runtime.get(key) == expected_identity[key], f"runtime identity differs for {key}")
    require(runtime.get("exact_runtime_contract_sha256") == exact_sha, "runtime exact E004 contract changed")
    require(runtime.get("component_digests") == exact_runtime_contract["component_digests"], "runtime component digests changed")
    require(runtime.get("dependency_bindings") == exact_runtime_contract["dependency_bindings"], "runtime source path/hash bindings changed")
    for key in ("lane_id", "simulator_pod_uid", "simulator_gpu_uuid", "policy_server_pod_uid", "policy_server_gpu_uuid", "raw_root", "container_identity", "runtime_identity", "server_process_identity", "server_lock_identity"):
        require(isinstance(runtime.get(key), str) and runtime.get(key), f"runtime identity lacks {key}")
    require(type(runtime.get("server_port")) is int and 1024 <= runtime["server_port"] <= 65535, "runtime server port is invalid")
    require(runtime.get("full_reset") is True and runtime.get("stage_identifier") == "full_reset", "episode reset/stage identifier changed")
    raw_artifacts = raw.get("raw_artifacts")
    require(isinstance(raw_artifacts, dict), "raw artifact map is missing")
    artifacts = {
        name: _artifact(raw_artifacts.get(name), name.replace("_", " "))
        for name in ("simulator_video", "executed_action_trace", "raw_episode_jsonl", "final_state", "state_trace")
    }
    camera_artifacts = raw_artifacts.get("policy_camera_images")
    require(isinstance(camera_artifacts, dict) and set(camera_artifacts) == set(expected_identity["policy_cameras"]), "policy camera image artifacts are incomplete")
    bound_cameras = {name: _artifact(record, f"policy camera image {name}") for name, record in camera_artifacts.items()}
    require(runtime.get("policy_camera_image_artifact_hashes") == {name: record["sha256"] for name, record in bound_cameras.items()}, "runtime camera/image hashes differ from retained artifacts")
    outcome = _failure_category(raw.get("state_trace"), cell.physical_goal, detached, success)
    reported = raw.get("reported_failure_category")
    require(reported == outcome, "raw failure category differs from frozen taxonomy")
    return {
        "schema_version": EPISODE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "cell_id": cell.cell_id,
        "registered_cell_sha256": cell.row_sha256,
        "registration_sha256": registration_sha256,
        "queue_sha256": queue_sha256,
        "model_id": MODEL_ID,
        "arena": cell.row["arena"],
        "seed_block_id": cell.block_id,
        "episode_seed": cell.seed,
        "sampling_seed": cell.seed,
        "prompt_condition": cell.condition,
        "physical_goal": cell.physical_goal,
        "surface_direction_word": cell.row["surface_direction_word"],
        "prompt": cell.row["prompt"],
        "prompt_utf8_hex": cell.row["prompt_utf8_hex"],
        "prompt_sha256": cell.row["prompt_sha256"],
        "initial_state_sha256": state_hash,
        "request_seeds": request_seeds,
        "success": success,
        "failure_category": outcome,
        "signed_final_lateral_offset": signed,
        "requested_side_depth": depth,
        "final_detached_release": detached,
        "endpoint_value": signed,
        "action_trace_sha256": artifacts["executed_action_trace"]["sha256"],
        "runtime_identity": runtime,
        "lane_id": runtime["lane_id"],
        "server_port": runtime["server_port"],
        "raw_root": runtime["raw_root"],
        "simulator_pod_uid": runtime["simulator_pod_uid"],
        "simulator_gpu_uuid": runtime["simulator_gpu_uuid"],
        "policy_server_pod_uid": runtime["policy_server_pod_uid"],
        "policy_server_gpu_uuid": runtime["policy_server_gpu_uuid"],
        "container_identity": runtime["container_identity"],
        "runtime_identity_label": runtime["runtime_identity"],
        "source_commit": runtime["source_commit"],
        "checkpoint_digest": runtime["checkpoint_digest"],
        "full_reset": runtime["full_reset"],
        "stage_identifier": runtime["stage_identifier"],
        "exact_runtime_contract_sha256": exact_sha,
        "policy_camera_image_artifacts": bound_cameras,
        "raw_artifacts": artifacts,
        "infrastructure_status": "valid_behavioral_episode",
    }


def _percentile(values: list[float], q: float) -> float:
    require(values, "cannot calculate a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_mean_ci(values: list[float], *, level: float, seed: int) -> list[float]:
    require(values and 0 < level < 1, "invalid bootstrap inputs")
    rng = random.Random(seed)
    size = len(values)
    means = [fmean(values[rng.randrange(size)] for _ in range(size)) for _ in range(BOOTSTRAP_RESAMPLES)]
    tail = (1 - level) / 2
    return [_percentile(means, tail), _percentile(means, 1 - tail)]


def paired_tost(values: list[float], *, margin: float) -> dict[str, Any]:
    require(values and margin > 0, "invalid TOST input")
    mean = fmean(values)
    if len(values) == 1 or all(math.isclose(value, mean, abs_tol=0.0) for value in values):
        lower_p = 0.0 if mean > -margin else 1.0
        upper_p = 0.0 if mean < margin else 1.0
    else:
        se = stdev(values) / math.sqrt(len(values))
        normal = NormalDist()
        lower_p = 1 - normal.cdf((mean + margin) / se)
        upper_p = normal.cdf((mean - margin) / se)
    return {
        "method": "paired_two_one_sided_normal_approximation",
        "margin": margin,
        "lower_test_p": lower_p,
        "upper_test_p": upper_p,
        "pass_alpha_0_05": bool(lower_p < 0.05 and upper_p < 0.05),
    }


def _pair_rows(episodes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in episodes:
        by_seed.setdefault(int(row["episode_seed"]), {})[str(row["prompt_condition"])] = row
    pairs = []
    for seed, forms in sorted(by_seed.items()):
        require(set(forms) == {"canonical_left", "inverse_reference_left", "canonical_right", "inverse_reference_right"}, f"seed {seed} lacks a complete block")
        state_hashes = {str(row["initial_state_sha256"]) for row in forms.values()}
        require(len(state_hashes) == 1, f"seed {seed} is not state matched")
        for physical_goal, canonical, inverse in (
            ("left", forms["canonical_left"], forms["inverse_reference_left"]),
            ("right", forms["canonical_right"], forms["inverse_reference_right"]),
        ):
            pairs.append(
                {
                    "schema_version": PAIR_SCHEMA,
                    "study_id": STUDY_ID,
                    "amendment_id": AMENDMENT_ID,
                    "seed_block_id": f"v3c002:seed{seed}",
                    "episode_seed": seed,
                    "physical_goal": physical_goal,
                    "canonical_cell_id": canonical["cell_id"],
                    "inverse_cell_id": inverse["cell_id"],
                    "initial_state_sha256": next(iter(state_hashes)),
                    "depth_difference_inverse_minus_canonical_m": float(inverse["requested_side_depth"]) - float(canonical["requested_side_depth"]),
                    "success_risk_difference_inverse_minus_canonical": int(bool(inverse["success"])) - int(bool(canonical["success"])),
                    "action_traces_equal_exactly": inverse["action_trace_sha256"] == canonical["action_trace_sha256"],
                    "canonical_action_trace_sha256": canonical["action_trace_sha256"],
                    "inverse_action_trace_sha256": inverse["action_trace_sha256"],
                }
            )
    require(len(pairs) == 682, "C002 must contain 682 matched prompt-form pairs")
    return pairs


def compile_results(episodes: list[dict[str, Any]], *, registration_sha256: str, queue_sha256: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lanes: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        lane_id = str(episode["lane_id"])
        identity = {key: episode[key] for key in ("server_port", "raw_root", "simulator_pod_uid", "simulator_gpu_uuid", "policy_server_pod_uid", "policy_server_gpu_uuid", "container_identity", "runtime_identity_label", "source_commit", "checkpoint_digest")}
        if lane_id in lanes:
            require(lanes[lane_id] == identity, f"lane {lane_id} infrastructure identity changed between episodes")
        lanes[lane_id] = identity
    require(len(lanes) >= 2, "compiled cohort lacks two retained execution lanes")
    for key in ("server_port", "raw_root", "simulator_pod_uid", "simulator_gpu_uuid", "policy_server_pod_uid", "policy_server_gpu_uuid"):
        require(len({identity[key] for identity in lanes.values()}) == len(lanes), f"lane {key} allocations are not unique")
    pairs = _pair_rows(episodes)
    analyses = {}
    for goal in ("left", "right"):
        rows = [pair for pair in pairs if pair["physical_goal"] == goal]
        depth = [float(pair["depth_difference_inverse_minus_canonical_m"]) for pair in rows]
        binary = [float(pair["success_risk_difference_inverse_minus_canonical"]) for pair in rows]
        depth_ci = bootstrap_mean_ci(depth, level=0.90, seed=20260812 + (0 if goal == "left" else 1))
        binary_ci = bootstrap_mean_ci(binary, level=0.90, seed=20260814 + (0 if goal == "left" else 1))
        tost = paired_tost(depth, margin=DEPTH_MARGIN_M)
        depth_equivalent = bool(tost["pass_alpha_0_05"] and depth_ci[0] > -DEPTH_MARGIN_M and depth_ci[1] < DEPTH_MARGIN_M)
        binary_equivalent = bool(binary_ci[0] > -BINARY_MARGIN and binary_ci[1] < BINARY_MARGIN)
        analyses[goal] = {
            "pair_count": len(rows),
            "depth_inverse_minus_canonical_m": {"mean": fmean(depth), "bootstrap_90_ci": depth_ci, "margin_m": DEPTH_MARGIN_M, "tost": tost, "equivalent": depth_equivalent},
            "success_inverse_minus_canonical": {"mean_risk_difference": fmean(binary), "bootstrap_90_ci": binary_ci, "margin_probability": BINARY_MARGIN, "equivalent_by_interval": binary_equivalent},
            "action_equality_descriptive": {
                "exactly_equal_pairs": sum(bool(pair["action_traces_equal_exactly"]) for pair in rows),
                "distinct_pairs": sum(not bool(pair["action_traces_equal_exactly"]) for pair in rows),
                "not_a_primary_grounding_test": True,
            },
        }
    positive_controls = {}
    for label, left_condition, right_condition in (
        ("canonical", "canonical_left", "canonical_right"),
        ("inverse_reference", "inverse_reference_left", "inverse_reference_right"),
    ):
        ordered = []
        for seed in range(12000, 12341):
            left = next(row for row in episodes if row["episode_seed"] == seed and row["prompt_condition"] == left_condition)
            right = next(row for row in episodes if row["episode_seed"] == seed and row["prompt_condition"] == right_condition)
            ordered.append(float(left["endpoint_value"]) - float(right["endpoint_value"]))
        ci = bootstrap_mean_ci(ordered, level=0.95, seed=20260816 + len(positive_controls))
        positive_controls[label] = {
            "definition": "mean(signed endpoint LEFT physical-goal condition minus RIGHT physical-goal condition)",
            "mean_m": fmean(ordered),
            "bootstrap_95_ci": ci,
            "positive_with_ci_excluding_zero": bool(fmean(ordered) > 0 and ci[0] > 0),
        }
    depth_claim = all(analyses[goal]["depth_inverse_minus_canonical_m"]["equivalent"] for goal in ("left", "right"))
    inverse_control = positive_controls["inverse_reference"]["positive_with_ci_excluding_zero"]
    semantic_claim = bool(depth_claim and inverse_control)
    return pairs, {
        "schema_version": RESULT_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "registration_sha256": registration_sha256,
        "queue_sha256": queue_sha256,
        "valid_behavioral_episodes": len(episodes),
        "complete_seed_blocks": 341,
        "prompt_form_pairs": len(pairs),
        "primary_requested_side_depth_equivalence": analyses,
        "positive_controls": positive_controls,
        "semantic_redirection_supported": inverse_control,
        "descriptive_directional_depth_form_equivalence": depth_claim,
        "model_level_semantic_depth_equivalence_claim_authorized": semantic_claim,
        "model_level_semantic_depth_equivalence_claim_withheld": not semantic_claim,
        "claim_gate_components": {"directional_depth_tost_conjunction": depth_claim, "inverse_reference_endpoint_positive_control": inverse_control},
        "claim_rule": "Model-level semantic depth equivalence requires both directional depth TOSTs AND the inverse-reference endpoint positive control. Otherwise the semantic claim is explicitly withheld; descriptive form equivalence remains reportable.",
    }


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise ContractError(f"refusing to overwrite retained output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise ContractError(f"refusing to overwrite retained output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--raw-episodes", type=Path, required=True)
    parser.add_argument("--episodes-output", type=Path, required=True)
    parser.add_argument("--pairs-output", type=Path, required=True)
    parser.add_argument("--results-output", type=Path, required=True)
    args = parser.parse_args()
    registration, cells = load_cells(registration_path=args.registration, queue_path=args.queue)
    require(registration.get("registration_status") == "registered_after_two_human_wording_agreements", "C002 is not behaviorally registered")
    registration_sha = sha256_file(args.registration)
    queue_sha = sha256_file(args.queue)
    cell_map = {cell.cell_id: cell for cell in cells}
    raw_rows = _read_jsonl(args.raw_episodes)
    require(len(raw_rows) == 1364, "raw source must contain exactly 1,364 behavioral episodes")
    require(len({str(row.get("cell_id")) for row in raw_rows}) == 1364, "raw source has duplicate cells")
    require({str(row.get("cell_id")) for row in raw_rows} == set(cell_map), "raw source does not exactly cover the queue")
    episodes = [compile_episode(row, cell=cell_map[str(row["cell_id"])], registration_sha256=registration_sha, queue_sha256=queue_sha, exact_runtime_contract=registration["exact_e004_pi05_runtime"]) for row in raw_rows]
    pairs, results = compile_results(episodes, registration_sha256=registration_sha, queue_sha256=queue_sha)
    _write_jsonl(args.episodes_output, episodes)
    _write_jsonl(args.pairs_output, pairs)
    _write_json(args.results_output, results)


if __name__ == "__main__":
    main()
