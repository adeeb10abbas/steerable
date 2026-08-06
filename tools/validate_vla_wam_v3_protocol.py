#!/usr/bin/env python3
"""Fail-closed validation for the frozen VLA/WAM steerability v3 registry."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


class ValidationError(ValueError):
    """Raised when a v3 invariant is absent or altered."""


V3 = "artifacts/vla_wam_shared_v3"
REQUIRED = {
    "protocol": f"{V3}/protocol.json",
    "amendment": f"{V3}/post_result_power_failure_ablation_amendment.json",
    "droid": f"{V3}/droid_direct_registry.json",
    "robotwin": f"{V3}/robotwin_direct_registry.json",
    "wording": f"{V3}/four_phrasings_registry.json",
    "stochastic": f"{V3}/stochastic_rollout_registry.json",
    "confound_calibration": f"{V3}/confound_fixture_calibration_registry.json",
    "phase_a_queue": f"{V3}/phase_a_cells.jsonl",
    "phase_a_manifest": f"{V3}/phase_a_cells_manifest.json",
    "taxonomy": f"{V3}/failure_taxonomy.json",
    "analysis": f"{V3}/analysis_plan.json",
    "measurement_audit": f"{V3}/measurement_coverage_audit.json",
    "document": "docs/VLA_WAM_STEERABILITY_V3_PROTOCOL.md",
    "continuation": f"{V3}/continuation_state.json",
    "continuation_document": "docs/VLA_WAM_V3_CONTINUATION.md",
    "work_laptop_handoff": "docs/WORK_LAPTOP_B200_HANDOFF.md",
    "pi0_bridge_amendment": f"{V3}/post_result_pi0_fast_old_name_config_amendment.json",
    "pi0_trace_amendment": f"{V3}/post_result_pi0_fast_token_trace_validation_amendment.json",
    "pi0_bridge_release_gate": f"{V3}/results/pi0_fast_old_name_config_v3a002_release_gate.json",
    "pi0_bridge_summary": f"{V3}/results/pi0_fast_old_name_config_v3a002_summary.json",
    "pi0_bridge_manifest": f"{V3}/results/pi0_fast_old_name_config_v3a002_evidence_hash_manifest.json",
    "pi0_bridge_ledger": f"{V3}/results/pi0_fast_old_name_config_v3a002_infrastructure_ledger.json",
    "pi0_bridge_media_manifest": f"{V3}/media/pi0_fast_old_name_config_v3a002/media_manifest.json",
    "pi0_bridge_media_video": f"{V3}/media/pi0_fast_old_name_config_v3a002/pi0_fast_v3a002_seed8311_paired_actual.mp4",
    "pi0_bridge_media_poster": f"{V3}/media/pi0_fast_old_name_config_v3a002/pi0_fast_v3a002_seed8311_paired_actual_poster.jpg",
    "pi0_bridge_media_renderer": "tools/build_v3a002_pi0_fast_media.py",
    "pi0_bridge_server": "experiments/v3/pi0_fast_old_name_config_bridge/serve_policy.py",
    "pi0_bridge_gate": "experiments/v3/pi0_fast_old_name_config_bridge/fixed_observation_gate.py",
    "pi0_bridge_adapter": "experiments/v3/pi0_fast_old_name_config_bridge/adapter.py",
    "pi0_bridge_compiler": "experiments/v3/pi0_fast_old_name_config_bridge/compile_shards.py",
    "nano_mirror_calibration": f"{V3}/phase_b/nano_mirror_v3b001/model_blind_calibration_report.json",
    "nano_mirror_amendment": f"{V3}/phase_b/nano_mirror_v3b001/post_result_nano_mirror_v3b001_amendment.json",
    "nano_mirror_cells": f"{V3}/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_cells.jsonl",
    "nano_mirror_manifest": f"{V3}/phase_b/nano_mirror_v3b001/nano_mirror_v3b001_manifest.json",
    "nano_mirror_live_ledger": f"{V3}/phase_b/nano_mirror_v3b001/live_infrastructure_ledger.json",
    "nano_mirror_reset_correction": f"{V3}/phase_b/nano_mirror_v3b001/live_reset_semantics_correction.json",
    "nano_mirror_reset_preflight": f"{V3}/phase_b/nano_mirror_v3b001/live_reset_preflight_report.json",
    "nano_mirror_live_snapshot": f"{V3}/phase_b/nano_mirror_v3b001/live_queue_snapshot.json",
    "nano_mirror_results_manifest": f"{V3}/phase_b/nano_mirror_v3b001/results/nano_v3b001_results_manifest.json",
    "nano_mirror_summary": f"{V3}/phase_b/nano_mirror_v3b001/results/nano_v3b001_summary.json",
    "nano_mirror_episodes": f"{V3}/phase_b/nano_mirror_v3b001/results/nano_v3b001_episodes.jsonl",
    "nano_mirror_episodes_manifest": f"{V3}/phase_b/nano_mirror_v3b001/results/nano_v3b001_episodes.jsonl.manifest.json",
    "nano_mirror_final_evidence": f"{V3}/phase_b/nano_mirror_v3b001/results/nano_v3b001_final_evidence_manifest.json",
    "nano_mirror_figure_media_manifest": f"{V3}/phase_b/nano_mirror_v3b001/results/figures/nano_v3b001_media_manifest.json",
    "nano_mirror_publication_media_manifest": f"{V3}/phase_b/nano_mirror_v3b001/results/publication_media/nano_v3b001_publication_media_manifest.json",
    "nano_mirror_runtime_adapter": "experiments/v3/cosmos_nano_phase_b/runtime_adapter.py",
    "nano_mirror_compiler": "experiments/v3/cosmos_nano_phase_b/compile_cell.py",
    "nano_mirror_live_support": "experiments/v3/cosmos_nano_phase_b/live_support.py",
    "nano_mirror_live_client": "experiments/v3/cosmos_nano_phase_b/live_client.py",
    "nano_mirror_live_server": "experiments/v3/cosmos_nano_phase_b/serve_nano.py",
    "nano_mirror_live_bridge": "experiments/v3/cosmos_nano_phase_b/robolab_bridge.py",
    "nano_mirror_queue_launcher": "experiments/v3/cosmos_nano_phase_b/queue_launcher.py",
    "nano_mirror_results_compiler": "tools/compile_nano_v3b001_results.py",
    "nano_mirror_evidence_finalizer": "tools/finalize_nano_v3b001_evidence.py",
    "nano_mirror_results_renderer": "tools/render_nano_v3b001_results.py",
    "nano_mirror_publication_media_builder": "tools/build_nano_v3b001_publication_media.py",
    "nano_mirror_runtime_test": "tests/test_v3b_nano_runtime_adapter.py",
    "nano_mirror_live_queue_test": "tests/test_v3b_nano_live_queue.py",
    "nano_mirror_results_compiler_test": "tests/test_compile_nano_v3b001_results.py",
    "nano_mirror_evidence_finalizer_test": "tests/test_finalize_nano_v3b001_evidence.py",
    "nano_mirror_results_renderer_test": "tests/test_render_nano_v3b001_results.py",
    "nano_mirror_publication_media_test": "tests/test_build_nano_v3b001_publication_media.py",
    "pi05_mirror_amendment": f"{V3}/phase_b/pi05_mirror_v3b002/post_result_pi05_mirror_v3b002_amendment.json",
    "pi05_mirror_cells": f"{V3}/phase_b/pi05_mirror_v3b002/pi05_mirror_v3b002_cells.jsonl",
    "pi05_mirror_manifest": f"{V3}/phase_b/pi05_mirror_v3b002/pi05_mirror_v3b002_manifest.json",
    "pi05_failure_split_episodes": f"{V3}/phase_b/pi05_mirror_v3b002/analysis/failure_mode_split_episodes.jsonl",
    "pi05_failure_split_report": f"{V3}/phase_b/pi05_mirror_v3b002/analysis/failure_mode_split_report.json",
    "pi05_failure_split_manifest": f"{V3}/phase_b/pi05_mirror_v3b002/analysis/failure_mode_split_manifest.json",
    "pi05_mirror_registration_builder": "tools/build_pi05_v3b002_registration.py",
    "pi05_failure_split_analyzer": "tools/analyze_v3_failure_mode_split.py",
    "pi05_mirror_registration_test": "tests/test_build_pi05_v3b002_registration.py",
    "pi05_mirror_runtime_readme": "experiments/v3/pi05_phase_b/README.md",
    "pi05_mirror_contract": "experiments/v3/pi05_phase_b/contract.py",
    "pi05_mirror_runtime": "experiments/v3/pi05_phase_b/runtime.py",
    "pi05_mirror_preflight": "experiments/v3/pi05_phase_b/model_blind_preflight.py",
    "pi05_mirror_probe": "experiments/v3/pi05_phase_b/fixed_observation_probe.py",
    "pi05_mirror_fixture_tasks": "experiments/v3/pi05_phase_b/fixture_tasks.py",
    "pi05_mirror_client": "experiments/v3/pi05_phase_b/client.py",
    "pi05_mirror_bridge": "experiments/v3/pi05_phase_b/robolab_bridge.py",
    "pi05_mirror_queue": "experiments/v3/pi05_phase_b/queue.py",
    "pi05_mirror_cell_compiler": "experiments/v3/pi05_phase_b/compile_cell.py",
    "pi05_mirror_diagnostics": "experiments/v3/pi05_phase_b/diagnostics.py",
    "pi05_mirror_results_compiler": "experiments/v3/pi05_phase_b/compiler.py",
    "pi05_mirror_runtime_test": "tests/test_pi05_v3b002_runtime.py",
    "pi05_mirror_compiler_test": "tests/test_pi05_v3b002_compiler.py",
    "pi05_mirror_result_episodes": f"{V3}/phase_b/pi05_mirror_v3b002/results/pi05_v3b002_episodes.jsonl",
    "pi05_mirror_result_episodes_manifest": f"{V3}/phase_b/pi05_mirror_v3b002/results/pi05_v3b002_episodes.jsonl.manifest.json",
    "pi05_mirror_result_pairs": f"{V3}/phase_b/pi05_mirror_v3b002/results/pi05_v3b002_pairs.jsonl",
    "pi05_mirror_result_pairs_manifest": f"{V3}/phase_b/pi05_mirror_v3b002/results/pi05_v3b002_pairs.jsonl.manifest.json",
    "pi05_mirror_result_infrastructure": f"{V3}/phase_b/pi05_mirror_v3b002/results/pi05_v3b002_infrastructure_attempts.jsonl",
    "pi05_mirror_result_infrastructure_manifest": f"{V3}/phase_b/pi05_mirror_v3b002/results/pi05_v3b002_infrastructure_attempts.jsonl.manifest.json",
    "pi05_mirror_result_report": f"{V3}/phase_b/pi05_mirror_v3b002/results/pi05_v3b002_report.json",
    "pi05_mirror_result_output_manifest": f"{V3}/phase_b/pi05_mirror_v3b002/results/pi05_v3b002_output_manifest.json",
    "pi05_mirror_gate_fixed_observation": f"{V3}/phase_b/pi05_mirror_v3b002/gates/fixed_observation_gate.json",
    "pi05_mirror_gate_release": f"{V3}/phase_b/pi05_mirror_v3b002/gates/release_gate.json",
    "pi05_mirror_gate_runtime_identity": f"{V3}/phase_b/pi05_mirror_v3b002/gates/runtime_identity.json",
    "pi05_mirror_gate_manifest": f"{V3}/phase_b/pi05_mirror_v3b002/gates/gate_manifest.json",
    "pi05_mirror_gate_lane0": f"{V3}/phase_b/pi05_mirror_v3b002/gates/preflight/lane0_lingbot.json",
    "pi05_mirror_gate_lane1": f"{V3}/phase_b/pi05_mirror_v3b002/gates/preflight/lane1_raytrace.json",
    "pi05_mirror_gate_lane2": f"{V3}/phase_b/pi05_mirror_v3b002/gates/preflight/lane2_fastwam.json",
    "pi05_mirror_gate_lane3": f"{V3}/phase_b/pi05_mirror_v3b002/gates/preflight/lane3_robotwin.json",
    "pi05_mirror_gate_lane4": f"{V3}/phase_b/pi05_mirror_v3b002/gates/preflight/lane4_cosmos.json",
    "pi05_mirror_gate_lane5": f"{V3}/phase_b/pi05_mirror_v3b002/gates/preflight/lane5_nano.json",
    "dreamzero_mirror_amendment": f"{V3}/phase_b/dreamzero_mirror_v3b003/post_result_dreamzero_mirror_v3b003_amendment.json",
    "dreamzero_mirror_cells": f"{V3}/phase_b/dreamzero_mirror_v3b003/dreamzero_mirror_v3b003_cells.jsonl",
    "dreamzero_mirror_manifest": f"{V3}/phase_b/dreamzero_mirror_v3b003/dreamzero_mirror_v3b003_manifest.json",
    "dreamzero_mirror_registration_builder": "tools/build_dreamzero_v3b003_registration.py",
    "dreamzero_mirror_registration_test": "tests/test_build_dreamzero_v3b003_registration.py",
    "nano_lateral_sweep_amendment": f"{V3}/phase_b/nano_lateral_sweep_v3b004/post_result_nano_lateral_sweep_v3b004_amendment.json",
}
DROID_MODELS = [
    "pi0_fast_droid_vla",
    "groot_n17_droid_vla",
    "cosmos3_edge_policy_droid",
    "cosmos3_nano_policy_droid",
    "pi05_current_stack_droid",
    "dreamzero_droid_action_cfg",
]
ROBOTWIN_MODELS = [
    "efficient_wam_rt_robotwin",
    "fastwam_robotwin",
    "lingbot_va_robotwin",
]
WORDING_MODELS = [
    "groot_n17_droid_vla",
    "cosmos3_edge_policy_droid",
    "cosmos3_nano_policy_droid",
]
EXACT_V2_WORDINGS = {
    "direct_command": {
        "left": "Put the Rubik's cube to the left of the bowl.",
        "right": "Put the Rubik's cube to the right of the bowl.",
    },
    "short_command": {
        "left": "Put the cube left of the bowl.",
        "right": "Put the cube right of the bowl.",
    },
    "goal_as_outcome": {
        "left": "The Rubik's cube should end up to the left of the bowl.",
        "right": "The Rubik's cube should end up to the right of the bowl.",
    },
    "desired_plus_negated_opposite": {
        "left": "Put the Rubik's cube to the left of the bowl, not to the right of the bowl.",
        "right": "Put the Rubik's cube to the right of the bowl, not to the left of the bowl.",
    },
}


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise ValidationError(message)
    checks.append(message)


def load(path: Path) -> dict[str, Any]:
    try:
        with path.open() as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValidationError(f"{path}:{line_number} must contain a JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read {path}: {exc}") from exc
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_range(start: int, end: int) -> list[int]:
    return list(range(start, end + 1))


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    )


def wilson_95(successes: int, episodes: int) -> list[float]:
    if episodes <= 0 or not 0 <= successes <= episodes:
        raise ValidationError("Wilson interval inputs must be a valid nonempty binomial count")
    z = 1.959963984540054
    proportion = successes / episodes
    denominator = 1 + z * z / episodes
    center = proportion + z * z / (2 * episodes)
    radius = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * episodes)) / episodes)
    return [(center - radius) / denominator, (center + radius) / denominator]


def exact_two_sided_binomial_p(first: int, second: int) -> float:
    total = first + second
    if total == 0:
        return 1.0
    tail = min(first, second)
    return min(1.0, 2 * sum(math.comb(total, index) for index in range(tail + 1)) / (2**total))


def require_numeric_summary(
    record: dict[str, Any],
    values: list[float | int],
    label: str,
    checks: list[str],
) -> None:
    require(bool(values), f"{label} has observations", checks)
    expected = {
        "observed_count": len(values),
        "null_count": 0,
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }
    require(
        record.get("observed_count") == expected["observed_count"]
        and record.get("null_count") == expected["null_count"]
        and all(close(record.get(key), expected[key]) for key in ("minimum", "maximum", "mean", "median")),
        f"{label} statistics recompute exactly from all matched pairs",
        checks,
    )


def valid_file_record(record: Any, *, require_relative_path: bool = False) -> bool:
    if not isinstance(record, dict):
        return False
    path = record.get("path")
    byte_count = record.get("bytes")
    if not isinstance(path, str) or not path.startswith("/data/users/ali/vla_wam/"):
        return False
    if type(byte_count) is not int or byte_count <= 0 or not is_sha256(record.get("sha256")):
        return False
    if require_relative_path:
        relative = record.get("relative_path")
        if not isinstance(relative, str) or not relative or relative.startswith("/") or ".." in Path(relative).parts:
            return False
    return True


def same_local_file_reference(record: Any, path: Path) -> bool:
    return (
        isinstance(record, dict)
        and record.get("sha256") == sha256(path)
        and record.get("bytes") == path.stat().st_size
    )


def same_local_file_by_basename(record: Any, path: Path) -> bool:
    """Bind a copied compact artifact while ignoring its raw absolute parent."""

    return (
        same_local_file_reference(record, path)
        and isinstance(record.get("path"), str)
        and Path(record["path"]).name == path.name
    )


def validate_pi05_v3b002_evidence(
    paths: dict[str, Path],
    registered_cells: list[dict[str, Any]],
    checks: list[str],
) -> dict[str, Any]:
    """Validate the completed V3-B002 evidence without trusting raw path prefixes."""

    episodes = load_jsonl(paths["pi05_mirror_result_episodes"])
    episodes_manifest = load(paths["pi05_mirror_result_episodes_manifest"])
    pairs = load_jsonl(paths["pi05_mirror_result_pairs"])
    pairs_manifest = load(paths["pi05_mirror_result_pairs_manifest"])
    infrastructure = load_jsonl(paths["pi05_mirror_result_infrastructure"])
    infrastructure_manifest = load(paths["pi05_mirror_result_infrastructure_manifest"])
    report = load(paths["pi05_mirror_result_report"])
    output_manifest = load(paths["pi05_mirror_result_output_manifest"])
    fixed_gate = load(paths["pi05_mirror_gate_fixed_observation"])
    release_gate = load(paths["pi05_mirror_gate_release"])
    runtime_identity = load(paths["pi05_mirror_gate_runtime_identity"])
    gate_manifest = load(paths["pi05_mirror_gate_manifest"])
    preflights = [load(paths[f"pi05_mirror_gate_lane{lane}"]) for lane in range(6)]

    release_sha = sha256(paths["pi05_mirror_manifest"])
    expected_output_files = {
        "episodes": "pi05_mirror_result_episodes",
        "episodes_manifest": "pi05_mirror_result_episodes_manifest",
        "infrastructure": "pi05_mirror_result_infrastructure",
        "infrastructure_manifest": "pi05_mirror_result_infrastructure_manifest",
        "pairs": "pi05_mirror_result_pairs",
        "pairs_manifest": "pi05_mirror_result_pairs_manifest",
        "report": "pi05_mirror_result_report",
    }
    output_files = output_manifest.get("files", {})
    require(
        output_manifest.get("schema_version")
        == "vla-wam-shared-v3b-pi05-reflection-output-manifest-v1"
        and output_manifest.get("study_id") == "vla_wam_language_steerability_v3"
        and output_manifest.get("amendment_id") == "V3-B002"
        and output_manifest.get("release_manifest_sha256") == release_sha
        and set(output_files) == set(expected_output_files)
        and all(
            same_local_file_by_basename(output_files.get(label), paths[path_key])
            for label, path_key in expected_output_files.items()
        ),
        "V3-B002 output manifest byte/hash-binds every copied result by basename",
        checks,
    )

    manifest_specs = (
        (
            episodes_manifest,
            paths["pi05_mirror_result_episodes"],
            108,
            ["vla-wam-shared-v3-raw-episode-v1"],
        ),
        (
            pairs_manifest,
            paths["pi05_mirror_result_pairs"],
            54,
            ["vla-wam-shared-v3b-pi05-reflection-pair-v1"],
        ),
        (
            infrastructure_manifest,
            paths["pi05_mirror_result_infrastructure"],
            18,
            ["vla-wam-shared-v3-infrastructure-attempt-v1"],
        ),
    )
    require(
        all(
            manifest.get("schema_version")
            == "vla-wam-shared-v3b-pi05-reflection-jsonl-manifest-v1"
            and manifest.get("study_id") == "vla_wam_language_steerability_v3"
            and manifest.get("amendment_id") == "V3-B002"
            and manifest.get("release_manifest_sha256") == release_sha
            and manifest.get("row_count") == row_count
            and manifest.get("record_schema_versions") == schemas
            and manifest.get("jsonl_path") == jsonl_path.name
            and manifest.get("jsonl_bytes") == jsonl_path.stat().st_size
            and manifest.get("jsonl_sha256") == sha256(jsonl_path)
            for manifest, jsonl_path, row_count, schemas in manifest_specs
        ),
        "V3-B002 JSONL sidecars bind exact row counts, schemas, bytes, and hashes",
        checks,
    )

    registered_by_id = {row.get("cell_id"): row for row in registered_cells}
    episode_by_id = {row.get("registered_cell_id"): row for row in episodes}
    runtime_sha = runtime_identity.get("runtime_identity_sha256")
    allowed_failures = {
        "correct",
        "pick_failed",
        "transport_failed",
        "wrong_side",
        "release_failed",
    }
    require(
        is_sha256(runtime_sha)
        and runtime_identity.get("schema_version")
        == "vla-wam-shared-v3b-pi05-runtime-identity-v1"
        and runtime_identity.get("study_id") == "vla_wam_language_steerability_v3"
        and runtime_identity.get("amendment_id") == "V3-B002"
        and runtime_identity.get("model_id") == "pi05_current_stack_droid"
        and runtime_identity.get("release_manifest_sha256") == release_sha
        and runtime_identity.get("instruction_controller") == "static_episode_prompt"
        and runtime_identity.get("openpi_commit")
        == "c23745b5ad24e98f66967ea795a07b2588ed6c79"
        and runtime_identity.get("robolab_commit")
        == "0aef241fb088ca21bb4ebd24448940ed56620d17"
        and len(episodes) == len(episode_by_id) == len(registered_by_id) == 108
        and set(episode_by_id) == set(registered_by_id)
        and len({row.get("attempt_id") for row in episodes}) == 108
        and {row.get("runtime_identity", {}).get("sha256") for row in episodes}
        == {runtime_sha}
        and all(
            row.get("schema_version") == "vla-wam-shared-v3-raw-episode-v1"
            and row.get("record_type") == "behavioral_episode"
            and row.get("study_id") == "vla_wam_language_steerability_v3"
            and row.get("amendment_id") == "V3-B002"
            and row.get("model_id") == "pi05_current_stack_droid"
            and row.get("arena") == "droid_robolab"
            and row.get("behavioral_result_valid") is True
            and row.get("release_manifest_sha256") == release_sha
            and row.get("environment_seed")
            == registered_by_id[row["registered_cell_id"]].get("environment_seed")
            and row.get("phase_b_arm")
            == registered_by_id[row["registered_cell_id"]].get("arm")
            and row.get("requested_relation")
            == registered_by_id[row["registered_cell_id"]].get("relation")
            and row.get("prompt")
            == registered_by_id[row["registered_cell_id"]].get("prompt")
            and row.get("success") is row.get("requested_success")
            and row.get("failure_category") in allowed_failures
            and (row.get("failure_category") == "correct") is row.get("success")
            and isinstance(row.get("signed_final_lateral_offset_m"), (int, float))
            and isinstance(row.get("requested_side_depth_m"), (int, float))
            and isinstance(row.get("episode_length_steps"), int)
            and isinstance(row.get("cumulative_lateral_path_m"), (int, float))
            and isinstance(row.get("peak_lateral_excursion_m"), (int, float))
            for row in episodes
        ),
        "V3-B002 has exactly 108 registered, unique, single-runtime behavioral episodes",
        checks,
    )
    # Raw media are intentionally PVC-only, so validate their retained references rather
    # than attempting to resolve them from the compact Git checkout.
    require(
        all(
            valid_file_record(row.get("artifacts", {}).get("executed_action_trace"))
            and valid_file_record(row.get("artifacts", {}).get("viewport_video"))
            and row.get("artifacts", {}).get("executed_action_trace", {}).get("shape", [0])[0]
            == row.get("actions_executed")
            and row.get("episode_length_steps") == row.get("actions_executed")
            and row.get("measurements", {}).get("signed_final_lateral_offset_m")
            == row.get("signed_final_lateral_offset_m")
            and row.get("measurements", {}).get("final_requested_signed_margin_m")
            == row.get("requested_side_depth_m")
            for row in episodes
        ),
        "V3-B002 retains video/action references and all required per-episode measurements",
        checks,
    )
    failure_counts = Counter(row.get("failure_category") for row in episodes)
    require(
        failure_counts
        == Counter(
            {
                "correct": 63,
                "pick_failed": 14,
                "transport_failed": 27,
                "wrong_side": 3,
                "release_failed": 1,
            }
        )
        and report.get("behavioral_evidence", {}).get("failure_category_counts")
        == dict(failure_counts)
        and report.get("behavioral_evidence", {}).get("valid_failures_retained") is True,
        "V3-B002 retains every behavioral failure and reports the exact failure taxonomy",
        checks,
    )

    raw_episode_sources = {
        entry.get("registered_cell_id"): entry.get("jsonl", {}).get("sha256")
        for entry in episodes_manifest.get("source_batches", [])
    }
    action_sources = {
        entry.get("registered_cell_id"): entry.get("sha256")
        for entry in pairs_manifest.get("source_batches", [])
    }
    pairs_by_key = {(row.get("seed"), row.get("arm")): row for row in pairs}
    expected_pair_keys = {
        (seed, arm)
        for seed in exact_range(9400, 9426)
        for arm in ("control", "position_mirrored")
    }
    pair_rows_valid = len(pairs) == len(pairs_by_key) == 54 and set(pairs_by_key) == expected_pair_keys
    for (seed, arm), pair in pairs_by_key.items():
        left_id = f"v3b002:pi05:seed{seed}:{arm}:left"
        right_id = f"v3b002:pi05:seed{seed}:{arm}:right"
        left = episode_by_id.get(left_id, {})
        right = episode_by_id.get(right_id, {})
        pair_rows_valid = pair_rows_valid and (
            pair.get("schema_version") == "vla-wam-shared-v3b-pi05-reflection-pair-v1"
            and pair.get("left_registered_cell_id") == left_id
            and pair.get("right_registered_cell_id") == right_id
            and pair.get("initial_state_sha256") == left.get("initial_state_sha256")
            == right.get("initial_state_sha256")
            and pair.get("left_success") is left.get("success")
            and pair.get("right_success") is right.get("success")
            and pair.get("right_minus_left_success")
            == int(bool(right.get("success"))) - int(bool(left.get("success")))
            and close(
                pair.get("endpoint_redirection_D_m"),
                left.get("signed_final_lateral_offset_m", 0)
                - right.get("signed_final_lateral_offset_m", 0),
            )
            and close(pair.get("endpoint_shift_m"), pair.get("endpoint_redirection_D_m"))
            and close(
                pair.get("requested_side_depth_contrast_B_m"),
                right.get("requested_side_depth_m", 0)
                - left.get("requested_side_depth_m", 0),
            )
            and pair.get("action_distinct") is True
            and pair.get("executed_actions_distinct") is True
            and pair.get("common_prefix_action_count", 0) > 0
            and pair.get("common_prefix_action_rms", 0) > 0
            and pair.get("left_executed_action_trace_sha256")
            == left.get("artifacts", {}).get("executed_action_trace", {}).get("sha256")
            == action_sources.get(left_id)
            and pair.get("right_executed_action_trace_sha256")
            == right.get("artifacts", {}).get("executed_action_trace", {}).get("sha256")
            == action_sources.get(right_id)
            and pair.get("left_raw_episode_jsonl_sha256") == raw_episode_sources.get(left_id)
            and pair.get("right_raw_episode_jsonl_sha256") == raw_episode_sources.get(right_id)
        )
    require(
        pair_rows_valid,
        "V3-B002 has exactly 54 seed-by-layout pairs with recomputed endpoint, depth, success, and action diagnostics",
        checks,
    )

    analysis = report.get("analysis", {})
    seed_level = analysis.get("seed_level", [])
    seed_level_by_seed = {row.get("seed"): row for row in seed_level}
    h1_values: dict[str, list[float]] = {"control": [], "position_mirrored": []}
    h2_values: dict[str, list[float]] = {"control": [], "position_mirrored": []}
    h3_values: list[int] = []
    seed_rows_valid = len(seed_level) == len(seed_level_by_seed) == 27
    for seed in exact_range(9400, 9426):
        control = pairs_by_key[(seed, "control")]
        reflected = pairs_by_key[(seed, "position_mirrored")]
        d_control = control["endpoint_redirection_D_m"]
        d_reflected = reflected["endpoint_redirection_D_m"]
        b_control = control["requested_side_depth_contrast_B_m"]
        b_reflected = reflected["requested_side_depth_contrast_B_m"]
        did = reflected["right_minus_left_success"] - control["right_minus_left_success"]
        h1_values["control"].append(d_control)
        h1_values["position_mirrored"].append(d_reflected)
        h2_values["control"].append(b_control)
        h2_values["position_mirrored"].append(b_reflected)
        h3_values.append(did)
        row = seed_level_by_seed.get(seed, {})
        seed_rows_valid = seed_rows_valid and all(
            close(row.get(key), expected)
            for key, expected in {
                "D_control_m": d_control,
                "D_position_mirrored_m": d_reflected,
                "J_redirection_interaction_m": d_reflected - d_control,
                "B_control_m": b_control,
                "B_position_mirrored_m": b_reflected,
                "I_requested_side_depth_interaction_m": b_reflected - b_control,
                "control_right_minus_left_success": control["right_minus_left_success"],
                "position_mirrored_right_minus_left_success": reflected[
                    "right_minus_left_success"
                ],
                "binary_success_DiD": did,
            }.items()
        )
    require(seed_rows_valid, "V3-B002 seed-level H1/H2/H3 contrasts recompute from pairs", checks)

    def validate_continuous_summary(summary: Any, values: list[float]) -> bool:
        if not isinstance(summary, dict) or summary.get("n") != 27:
            return False
        positive = sum(value > 0 for value in values)
        negative = sum(value < 0 for value in values)
        ties = len(values) - positive - negative
        sign = summary.get("paired_sign_test", {})
        mean_ci = summary.get("mean_bootstrap_95", {})
        return (
            close(summary.get("mean_m"), statistics.fmean(values))
            and close(summary.get("median_m"), statistics.median(values))
            and sign.get("method") == "exact_two_sided_paired_sign_test"
            and sign.get("positive") == positive
            and sign.get("negative") == negative
            and sign.get("ties") == ties
            and sign.get("effective_n") == positive + negative
            and close(sign.get("p_value"), exact_two_sided_binomial_p(positive, negative))
            and mean_ci.get("method") == "matched_seed_nonparametric_percentile_bootstrap"
            and mean_ci.get("unit_of_resampling") == "matched_seed"
            and mean_ci.get("replicates") == 20000
            and close(mean_ci.get("confidence"), 0.95)
            and mean_ci.get("lower") <= summary.get("mean_m") <= mean_ci.get("upper")
        )

    h1 = analysis.get("H1_endpoint_redirection", {})
    h2 = analysis.get("H2_requested_side_depth", {})
    h1_interaction = [
        reflected - control
        for reflected, control in zip(h1_values["position_mirrored"], h1_values["control"])
    ]
    h2_interaction = [
        reflected - control
        for reflected, control in zip(h2_values["position_mirrored"], h2_values["control"])
    ]
    require(
        all(
            validate_continuous_summary(
                hypothesis.get("paired_contrast_by_layout", {}).get(layout), values
            )
            for hypothesis, values_by_layout in ((h1, h1_values), (h2, h2_values))
            for layout, values in values_by_layout.items()
        )
        and validate_continuous_summary(
            h1.get("reflected_minus_control_interaction"), h1_interaction
        )
        and validate_continuous_summary(
            h2.get("reflected_minus_control_interaction"), h2_interaction
        ),
        "V3-B002 H1/H2 use n=27, 20k matched-seed bootstrap CIs, and exact paired sign tests",
        checks,
    )
    require(
        close(statistics.fmean(h1_interaction), -0.011182827150656117)
        and close(statistics.median(h1_interaction), -0.03668234497308731)
        and Counter(value > 0 for value in h1_interaction) == Counter({False: 15, True: 12})
        and close(exact_two_sided_binomial_p(12, 15), 0.7011080384254456)
        and close(statistics.fmean(h2_interaction), -0.3459187115163163)
        and close(statistics.median(h2_interaction), -0.3503093309700489)
        and all(value < 0 for value in h2_interaction)
        and close(exact_two_sided_binomial_p(0, 27), 1.4901161193847656e-08),
        "V3-B002 reports the immutable H1 and H2 interaction estimates and exact sign results",
        checks,
    )

    h3 = analysis.get("H3_binary_success", {})
    expected_cell_table: dict[str, dict[str, dict[str, int]]] = {}
    for arm in ("control", "position_mirrored"):
        expected_cell_table[arm] = {}
        for relation in ("left", "right"):
            cell_rows = [
                row
                for row in episodes
                if row.get("phase_b_arm") == arm
                and row.get("requested_relation") == relation
            ]
            successes = sum(row.get("success") is True for row in cell_rows)
            expected_cell_table[arm][relation] = {
                "episodes": len(cell_rows),
                "failures": len(cell_rows) - successes,
                "successes": successes,
            }
    did_distribution = {str(value): h3_values.count(value) for value in range(-2, 3)}
    observed_sum = sum(h3_values)
    permutation_sums = Counter({0: 1})
    for value in h3_values:
        next_sums: Counter[int] = Counter()
        for partial, count in permutation_sums.items():
            next_sums[partial + value] += count
            next_sums[partial - value] += count
        permutation_sums = next_sums
    extreme = sum(
        count for value, count in permutation_sums.items() if abs(value) >= abs(observed_sum)
    )
    permutation = h3.get("exact_permutation_test", {})
    require(
        expected_cell_table
        == {
            "control": {
                "left": {"episodes": 27, "failures": 23, "successes": 4},
                "right": {"episodes": 27, "failures": 2, "successes": 25},
            },
            "position_mirrored": {
                "left": {"episodes": 27, "failures": 2, "successes": 25},
                "right": {"episodes": 27, "failures": 18, "successes": 9},
            },
        }
        == h3.get("cell_success_table_2x2")
        and did_distribution == {"-2": 12, "-1": 13, "0": 2, "1": 0, "2": 0}
        == h3.get("per_seed_DiD_distribution")
        and sum(did_distribution.values()) == 27
        and close(h3.get("mean_DiD"), statistics.fmean(h3_values))
        and close(h3.get("median_DiD"), statistics.median(h3_values))
        and permutation.get("method")
        == "exact_two_sided_within_seed_control_reflected_label_permutation"
        and permutation.get("registered_seed_count") == 27
        and permutation.get("observed_signed_sum") == observed_sum == -37
        and permutation.get("observed_absolute_sum") == abs(observed_sum) == 37
        and permutation.get("total_permutations") == sum(permutation_sums.values()) == 2**27
        and permutation.get("extreme_permutations") == extreme == 8
        and close(permutation.get("p_value"), extreme / 2**27)
        and close(permutation.get("p_value"), 5.960464477539063e-08),
        "V3-B002 H3 binds four n=27 cells, 27 DiDs, and the exact 2^27 permutation result",
        checks,
    )
    require(
        report.get("behavioral_evidence", {}).get("episode_count") == 108
        and report.get("behavioral_evidence", {}).get("pair_count") == 54
        and report.get("behavioral_evidence", {}).get("matched_seed_count") == 27
        and report.get("runtime_identity_sha256") == runtime_sha
        and report.get("uncertainty_contract", {}).get("bootstrap_master_seed")
        == 3104159
        and report.get("uncertainty_contract", {}).get("bootstrap_replicates")
        == 20000
        and report.get("analysis", {}).get("population")
        == {
            "behavioral_episode_count": 108,
            "infrastructure_attempts_included": False,
            "matched_left_right_pair_count": 54,
            "matched_seed_count": 27,
            "missing_value_imputation": "none",
            "valid_behavioral_failures_included": True,
        },
        "V3-B002 report keeps one runtime, all valid failures, and no infrastructure in denominators",
        checks,
    )

    require(
        len(infrastructure) == 18
        and len({row.get("attempt_id") for row in infrastructure}) == 18
        and {row.get("schema_version") for row in infrastructure}
        == {"vla-wam-shared-v3-infrastructure-attempt-v1"}
        and all(
            row.get("record_type") == "infrastructure_attempt"
            and row.get("classification") == "technical_invalid"
            and row.get("behavioral_result_valid") is False
            and row.get("denominator_policy") == "excluded_from_behavioral_denominator"
            and row.get("release_manifest_sha256") == release_sha
            for row in infrastructure
        )
        and sum(
            entry.get("attempt_count", 0)
            for entry in infrastructure_manifest.get("source_batches", [])
        )
        == 18
        and report.get("infrastructure_evidence")
        == {"attempt_count": 18, "included_in_behavioral_denominators": False},
        "V3-B002 preserves 18 unique common-schema nonbehavioral attempts outside denominators",
        checks,
    )

    expected_gate_files = {
        "pi05_mirror_gate_fixed_observation",
        "pi05_mirror_gate_lane0",
        "pi05_mirror_gate_lane1",
        "pi05_mirror_gate_lane2",
        "pi05_mirror_gate_lane3",
        "pi05_mirror_gate_lane4",
        "pi05_mirror_gate_lane5",
        "pi05_mirror_gate_release",
        "pi05_mirror_gate_runtime_identity",
    }
    gate_files_by_basename = {
        Path(record.get("path", "")).name: record for record in gate_manifest.get("files", [])
    }
    require(
        gate_manifest.get("schema_version")
        == "vla-wam-shared-v3b-pi05-gate-evidence-manifest-v1"
        and gate_manifest.get("study_id") == "vla_wam_language_steerability_v3"
        and gate_manifest.get("amendment_id") == "V3-B002"
        and gate_manifest.get("model_id") == "pi05_current_stack_droid"
        and gate_manifest.get("release_manifest_sha256") == release_sha
        and gate_manifest.get("runtime_identity_sha256") == runtime_sha
        and len(gate_manifest.get("files", [])) == len(gate_files_by_basename) == 9
        and {paths[key].name for key in expected_gate_files} == set(gate_files_by_basename)
        and all(
            same_local_file_by_basename(gate_files_by_basename.get(paths[key].name), paths[key])
            for key in expected_gate_files
        ),
        "V3-B002 gate manifest byte/hash-binds all nine committed gate files",
        checks,
    )
    lanes = runtime_identity.get("live_topology", {}).get("simulator_lanes", [])
    lane_keys = {(row.get("pod_uid"), row.get("gpu_uuid")) for row in lanes}
    preflight_keys = {(row.get("pod_uid"), row.get("gpu_uuid")) for row in preflights}
    release_lane_keys = {
        (row.get("pod_uid"), row.get("gpu_uuid"))
        for row in release_gate.get("model_blind_lane_bindings", [])
    }
    preflights_pass = all(
        preflight.get("schema_version")
        == "vla-wam-shared-v3b-pi05-model-blind-preflight-v1"
        and preflight.get("passed") is True
        and preflight.get("model_request_count") == 0
        and preflight.get("behavioral_episode_count") == 0
        and preflight.get("renderer_backend") == "realtime RTX Vulkan"
        and preflight.get("all_required_rgb_views_nonblank") is True
        and preflight.get("viewport_writer_passed") is True
        and preflight.get("action_trace_writer_passed") is True
        and preflight.get("raw_jsonl_writer_passed") is True
        and preflight.get("fixture_positions_match") is True
        and preflight.get("neutral_reset_passed") is True
        and len(preflight.get("tasks", [])) == 4
        and all(
            task.get("passed") is True and len(task.get("repeat_resets", [])) == 3
            for task in preflight.get("tasks", [])
        )
        and len(preflight.get("viewport_evidence", {})) == 4
        and all(
            valid_file_record(record) and record.get("decoded_frame_count") == 3
            for record in preflight.get("viewport_evidence", {}).values()
        )
        for preflight in preflights
    )
    require(
        len(lanes) == len(lane_keys) == len(preflight_keys) == len(release_lane_keys) == 6
        and lane_keys == preflight_keys == release_lane_keys
        and all(row.get("owner") == "ali" and row.get("pod", "").endswith("ali") for row in lanes)
        and preflights_pass,
        "V3-B002 model-blind gate passed renderer, writers, fixtures, and three resets on six unique ali lanes with zero behavior or requests",
        checks,
    )
    require(
        fixed_gate.get("schema_version")
        == "vla-wam-shared-v3b-pi05-fixed-observation-gate-v1"
        and fixed_gate.get("passed") is True
        and fixed_gate.get("model_request_count") == 3
        and fixed_gate.get("left_exact_repeat_bit_identical") is True
        and fixed_gate.get("fixed_observation_exact_repeat_passed") is True
        and fixed_gate.get("fixed_observation_left_right_prompt_sensitivity_passed") is True
        and fixed_gate.get("left_action_sha256") == fixed_gate.get("records", {}).get("left_a", {}).get("sha256")
        == fixed_gate.get("records", {}).get("left_b", {}).get("sha256")
        and fixed_gate.get("right_action_sha256")
        == fixed_gate.get("records", {}).get("right", {}).get("sha256")
        and fixed_gate.get("left_action_sha256") != fixed_gate.get("right_action_sha256")
        and fixed_gate.get("left_right_action_rms", 0) > 0,
        "V3-B002 fixed-observation gate has exact LEFT repeat and positive LEFT/RIGHT action RMS",
        checks,
    )
    require(
        release_gate.get("schema_version") == "vla-wam-shared-v3b-pi05-release-gate-v1"
        and release_gate.get("behavioral_release") is True
        and release_gate.get("release_manifest_sha256") == release_sha
        and release_gate.get("runtime_identity_sha256") == runtime_sha
        and release_gate.get("model_blind_lane_count") == 6
        and release_gate.get("model_blind_model_request_count") == 0
        and release_gate.get("model_blind_behavioral_episode_count") == 0
        and release_gate.get("model_blind_fixture_reset_renderer_writer_passed") is True
        and release_gate.get("left_exact_repeat_bit_identical") is True
        and release_gate.get("fixed_observation_exact_repeat_passed") is True
        and release_gate.get("fixed_observation_left_right_prompt_sensitivity_passed") is True
        and close(release_gate.get("left_right_action_rms"), fixed_gate.get("left_right_action_rms"))
        and gate_manifest.get("release_manifest_sha256") == release_sha
        and gate_manifest.get("runtime_identity_sha256") == runtime_sha
        and gate_manifest.get("gates", {}).get("behavioral_release") is True,
        "V3-B002 release gate is true and cross-binds the runtime, release, and physical gates",
        checks,
    )
    return {
        "episode_count": 108,
        "pair_count": 54,
        "infrastructure_attempt_count": 18,
        "runtime_identity_sha256": runtime_sha,
        "output_manifest_sha256": sha256(paths["pi05_mirror_result_output_manifest"]),
        "gate_manifest_sha256": sha256(paths["pi05_mirror_gate_manifest"]),
    }


def validate_pi0_fast_bridge(paths: dict[str, Path], checks: list[str]) -> dict[str, Any]:
    bridge_amendment = load(paths["pi0_bridge_amendment"])
    trace_amendment = load(paths["pi0_trace_amendment"])
    release_gate = load(paths["pi0_bridge_release_gate"])
    summary = load(paths["pi0_bridge_summary"])
    manifest = load(paths["pi0_bridge_manifest"])
    ledger = load(paths["pi0_bridge_ledger"])
    media = load(paths["pi0_bridge_media_manifest"])

    model_id = "pi0_fast_old_name_config_v3a002"
    cohort = "V3-A002_public_old_name_config_bridge"
    seeds = exact_range(8310, 8329)
    left_prompt = EXACT_V2_WORDINGS["direct_command"]["left"]
    right_prompt = EXACT_V2_WORDINGS["direct_command"]["right"]
    openpi_commit = "235044ed8a1502c0a18338eedc5d7adfe705af05"
    openpi_tree = "03a4387bedbc0fa1467c367c60fc24e28b61ec6c"
    robolab_commit = "0aef241fb088ca21bb4ebd24448940ed56620d17"

    require(
        bridge_amendment.get("schema_version") == "vla-wam-shared-v3-post-result-pi0-fast-old-name-config-amendment-v1"
        and bridge_amendment.get("study_id") == "vla_wam_language_steerability_v3"
        and bridge_amendment.get("amendment_id") == "V3-A002"
        and bridge_amendment.get("status") == "frozen_before_v3a002_model_load_or_request",
        "V3-A002 is the pre-request frozen public old-name-config bridge amendment",
        checks,
    )
    bridge_identity = bridge_amendment.get("bridge_identity", {})
    openpi = bridge_identity.get("openpi", {})
    robolab = bridge_identity.get("robolab", {})
    require(
        bridge_identity.get("model_id") == model_id
        and openpi.get("commit") == openpi_commit
        and openpi.get("tree") == openpi_tree
        and openpi.get("config") == "pi0_fast_droid_jointpos"
        and openpi.get("action_shape") == [10, 8]
        and openpi.get("max_token_len") == 250
        and openpi.get("data_config") == "SimpleDataConfig"
        and robolab.get("commit") == robolab_commit,
        "V3-A002 pins the exact public OpenPI old-name config and RoboLab bridge identity",
        checks,
    )
    implementation = bridge_amendment.get("implementation", {})
    require(
        implementation.get("server_path") == REQUIRED["pi0_bridge_server"]
        and implementation.get("server_sha256") == sha256(paths["pi0_bridge_server"])
        and implementation.get("gate_path") == REQUIRED["pi0_bridge_gate"]
        and implementation.get("gate_sha256") == sha256(paths["pi0_bridge_gate"]),
        "V3-A002 binds the checked-in bridge server and release-gate implementations",
        checks,
    )
    three_request_gate = bridge_amendment.get("three_request_gate", {})
    require(
        three_request_gate.get("sampling_seed") == 8310000
        and three_request_gate.get("order") == ["left", "left_exact_repeat", "right"]
        and three_request_gate.get("prompts") == {"left": left_prompt, "right": right_prompt},
        "V3-A002 freezes the exact three-request prompt-sensitivity gate",
        checks,
    )
    release = bridge_amendment.get("behavioral_release_if_gate_passes", {})
    source_filter = release.get("source_filter", {})
    require(
        release.get("source_queue") == REQUIRED["phase_a_queue"]
        and release.get("source_queue_sha256") == sha256(paths["phase_a_queue"])
        and source_filter == {
            "model_id": "pi0_fast_droid_vla",
            "status": "blocked_pi0",
            "execution_status": "blocked_pending_exact_historical_openpi_and_robolab_recovery",
            "phase": "A_direct_command_matched_pairs",
            "prompt_family": "direct_command",
            "environment_seeds_inclusive": [8310, 8329],
        }
        and release.get("new_model_id") == model_id
        and release.get("matched_pairs") == 20
        and release.get("behavioral_cells") == 40
        and release.get("action_cap") == 450
        and release.get("open_loop_horizon") == 10,
        "V3-A002 releases exactly 20 separate bridge pairs without rewriting the 40 blocked queue rows",
        checks,
    )
    prohibited = bridge_amendment.get("reporting_boundary", {}).get("prohibited", [])
    require(
        "pooled pi0-FAST x/30 rates" in prohibited
        and "pooled confidence intervals or tests across runtime identities" in prohibited
        and "pooling DROID and RoboTwin" in prohibited,
        "V3-A002 prohibits historical x/30 pooling and cross-arena pooling",
        checks,
    )

    require(
        trace_amendment.get("schema_version") == "vla-wam-shared-v3-post-result-pi0-fast-token-trace-validation-amendment-v1"
        and trace_amendment.get("study_id") == "vla_wam_language_steerability_v3"
        and trace_amendment.get("amendment_id") == "V3-A003"
        and trace_amendment.get("status") == "frozen_after_structural_compiler_failure_and_before_any_v3a002_cell_was_accepted",
        "V3-A003 is frozen after structural failure and before accepting a bridge cell",
        checks,
    )
    discovery = trace_amendment.get("discovery", {})
    require(
        discovery.get("behavioral_cells_inspected_for_contract_diagnosis") == 12
        and discovery.get("accepted_behavioral_cells_before_amendment") == 0
        and discovery.get("classification") == "validator_contract_bug_not_model_or_infrastructure_failure",
        "V3-A003 discloses the pre-acceptance token-trace validator diagnosis",
        checks,
    )
    invalid_assumption = trace_amendment.get("invalid_original_assumption", {})
    require(
        invalid_assumption.get("source_path") == REQUIRED["pi0_bridge_adapter"]
        and invalid_assumption.get("source_sha256") == sha256(paths["pi0_bridge_adapter"])
        and "tokenized_prompt_sha256" in invalid_assumption.get("rule", ""),
        "V3-A003 binds the original launch-time adapter and rejected token-equality rule",
        checks,
    )
    frozen_source = trace_amendment.get("frozen_source_evidence", {})
    require(
        frozen_source.get("openpi_commit") == openpi_commit
        and frozen_source.get("openpi_tree") == openpi_tree
        and frozen_source.get("path") == "src/openpi/transforms.py"
        and frozen_source.get("sha256") == "a1b94e9e72849a18834778f229c6bb389a495eb7fbe0aa800edea728b9424ff4"
        and frozen_source.get("transform") == "TokenizeFASTInputs",
        "V3-A003 pins the public TokenizeFASTInputs source evidence",
        checks,
    )
    replacement = trace_amendment.get("replacement_validation", {})
    require(
        replacement.get("only_removed_requirement")
        == "Within-episode equality of tokenized_prompt_sha256 across changing robot states.",
        "V3-A003 removes only within-episode state-conditioned token-hash equality",
        checks,
    )
    unchanged_requirements = replacement.get("unchanged_requirements", [])
    require(
        len(unchanged_requirements) == 8
        and any("Exact top-level frozen prompt string" in item for item in unchanged_requirements)
        and any("Contiguous deterministic per-request sampling seeds" in item for item in unchanged_requirements)
        and any("syntactically valid lowercase 64-hex" in item for item in unchanged_requirements)
        and any("prebehavior fixed-observation gate" in item for item in unchanged_requirements),
        "V3-A003 retains prompt, sampling, payload, token syntax, and prebehavior gate checks",
        checks,
    )
    token_boundary = replacement.get("matched_pair_token_boundary", {})
    trace_scope = trace_amendment.get("scope", {})
    require(
        trace_scope.get("model_id") == model_id
        and trace_scope.get("environment_seeds_inclusive") == [8310, 8329]
        and trace_scope.get("matched_pairs") == 20
        and trace_scope.get("behavioral_cells") == 40
        and trace_scope.get("runtime_identity_hashes") == [
            "591adbfed96fbe3a1f1016d08f0853b8f95e0b84895668e680610831270e96c2",
            "3f30ae51c44182636afc7bff6448241a9ee9579a73282cfb3a5abe40c5440edb",
            "49797dfb86f5b7f0ab9e4c06b14d8301e485b8e8cf06fabd909bcec0e0a730f0",
        ]
        and trace_scope.get("original_adapter_contract_sha256") == "0c7937482824090e3033fa3f6822c8277aff7b0b7f1403b565bc13140b5461db",
        "V3-A003 is scoped only to the 20-pair launch-time bridge identity",
        checks,
    )
    require(
        "original launch-time adapter" in trace_amendment.get("compiler_rule", "")
        and "without mutating any raw artifact" in trace_amendment.get("compiler_rule", ""),
        "V3-A003 requires original identity validation and read-only raw evidence",
        checks,
    )

    require(
        release_gate.get("schema_version") == "vla-wam-shared-v3-pi0-fast-old-name-config-gate-v1"
        and release_gate.get("study_id") == "vla_wam_language_steerability_v3"
        and release_gate.get("model_id") == model_id
        and release_gate.get("status") == "passed"
        and release_gate.get("behavioral_release") is True,
        "V3-A002 compact release gate passed and released behavior",
        checks,
    )
    require(
        release_gate.get("fixture", {}).get("sha256") == three_request_gate.get("fixture_sha256")
        and release_gate.get("raw_gate", {}).get("sha256") == trace_scope.get("release_gate_sha256"),
        "V3-A002 release gate binds the frozen fixture and raw gate hashes",
        checks,
    )
    server_metadata = release_gate.get("server_metadata", {})
    require(
        server_metadata.get("pi0_fast_old_name_config_bridge") == "v3a002"
        and server_metadata.get("openpi_commit") == openpi_commit
        and server_metadata.get("openpi_tree") == openpi_tree
        and server_metadata.get("openpi_config") == "pi0_fast_droid_jointpos"
        and server_metadata.get("max_token_len") == 250
        and server_metadata.get("checkpoint_assets_rule") == "checkpoint_local_assets_only"
        and server_metadata.get("sampling_contract") == "required_request_field:sampling_seed",
        "V3-A002 release gate attests the exact public runtime identity",
        checks,
    )
    gate_metrics = release_gate.get("metrics", {})
    require(
        gate_metrics.get("left_exact_repeat_bit_identical") is True
        and gate_metrics.get("left_exact_repeat_token_bytes_identical") is True
        and gate_metrics.get("left_right_token_bytes_differ") is True
        and gate_metrics.get("left_right_actions_bit_identical") is False
        and close(gate_metrics.get("left_right_action_rms"), 0.014522109180688858)
        and gate_metrics.get("left_right_action_rms", 0) > 0,
        "V3-A002 release gate repeats LEFT exactly and distinguishes LEFT from RIGHT actions",
        checks,
    )
    records = release_gate.get("records", {})
    left_a, left_b, right = records.get("left_a", {}), records.get("left_b", {}), records.get("right", {})
    require(
        left_a.get("prompt") == left_prompt
        and left_b.get("prompt") == left_prompt
        and right.get("prompt") == right_prompt
        and left_a.get("prompt_sha256") == left_b.get("prompt_sha256")
        and left_a.get("tokenized_prompt_sha256") == left_b.get("tokenized_prompt_sha256") == token_boundary.get("left_first_request_sha256")
        and right.get("tokenized_prompt_sha256") == token_boundary.get("right_first_request_sha256")
        and left_a.get("tokenized_prompt_sha256") != right.get("tokenized_prompt_sha256")
        and left_a.get("action_sha256") == left_b.get("action_sha256") != right.get("action_sha256")
        and all(record.get("sampling_seed") == 8310000 and record.get("shape") == [10, 8] and record.get("dtype") == "float32" for record in (left_a, left_b, right)),
        "V3-A002 release records preserve exact prompts, repeat identity, and LEFT/RIGHT token-action separation",
        checks,
    )
    release_scope = release_gate.get("behavioral_release_scope", {})
    require(
        release_scope == {
            "matched_pairs": 20,
            "cells": 40,
            "environment_seeds_inclusive": [8310, 8329],
            "cohort_identity": model_id,
            "historical_pooling": False,
        },
        "V3-A002 release gate fixes a separate 20-pair nonhistorical cohort",
        checks,
    )

    require(
        summary.get("schema_version") == "vla-wam-shared-v3-pi0-fast-old-name-config-summary-v3"
        and summary.get("study_id") == "vla_wam_language_steerability_v3"
        and summary.get("model_id") == model_id
        and summary.get("cohort") == cohort
        and summary.get("status") == "compiled_terminal_frozen_shards",
        "V3-A002 summary schema, study, model, cohort, and terminal status are exact",
        checks,
    )
    pair_rows = summary.get("pairs")
    require(
        isinstance(pair_rows, list)
        and len(pair_rows) == 20
        and summary.get("planned_matched_pairs") == 20
        and summary.get("behavioral_matched_pairs") == 20
        and summary.get("behavioral_episodes") == 40,
        "V3-A002 summary has exactly 20 matched pairs and 40 behavioral episodes",
        checks,
    )
    require(
        [row.get("seed") for row in pair_rows] == seeds and len({row.get("seed") for row in pair_rows}) == 20,
        "V3-A002 summary contains each exact seed 8310-8329 once",
        checks,
    )
    require(
        summary.get("historical_pi0_fast_denominator_included") is False
        and summary.get("historical_pooling_prohibited") is True
        and "not the missing historical runtime" in summary.get("claim_boundary", "")
        and "separate 20-pair denominator" in summary.get("claim_boundary", ""),
        "V3-A002 summary preserves a separate nonhistorical denominator",
        checks,
    )

    left_successes = sum(row.get("left_success") is True for row in pair_rows)
    right_successes = sum(row.get("right_success") is True for row in pair_rows)
    success_by_direction = summary.get("success_by_direction", {})
    require(
        left_successes == 0
        and right_successes == 12
        and success_by_direction.get("left", {}).get("successes") == left_successes
        and success_by_direction.get("left", {}).get("episodes") == 20
        and success_by_direction.get("right", {}).get("successes") == right_successes
        and success_by_direction.get("right", {}).get("episodes") == 20,
        "V3-A002 success denominators are exactly LEFT 0/20 and RIGHT 12/20",
        checks,
    )
    for relation, successes in (("left", left_successes), ("right", right_successes)):
        observed_interval = success_by_direction[relation].get("wilson_95", [])
        expected_interval = wilson_95(successes, 20)
        require(
            isinstance(observed_interval, list)
            and len(observed_interval) == 2
            and all(close(observed, expected) for observed, expected in zip(observed_interval, expected_interval)),
            f"V3-A002 {relation.upper()} Wilson interval recomputes from its separate denominator",
            checks,
        )

    discordance = Counter()
    for row in pair_rows:
        left_success = row.get("left_success") is True
        right_success = row.get("right_success") is True
        discordance[
            "both" if left_success and right_success
            else "left_only" if left_success
            else "right_only" if right_success
            else "neither"
        ] += 1
    observed_discordance = summary.get("success_discordance", {})
    expected_discordance = {"both": 0, "left_only": 0, "right_only": 12, "neither": 8}
    require(
        {key: discordance[key] for key in expected_discordance} == expected_discordance
        and {key: observed_discordance.get(key) for key in expected_discordance} == expected_discordance
        and sum(expected_discordance.values()) == 20,
        "V3-A002 paired success discordance recomputes to 0/0/12/8",
        checks,
    )
    expected_mcnemar = exact_two_sided_binomial_p(discordance["left_only"], discordance["right_only"])
    require(
        close(observed_discordance.get("exact_two_sided_mcnemar_p"), expected_mcnemar)
        and close(expected_mcnemar, 0.00048828125),
        "V3-A002 exact two-sided McNemar p-value recomputes from discordant pairs",
        checks,
    )

    ordering = Counter()
    endpoint_shifts: list[float] = []
    raw_y_shifts: list[float] = []
    action_rms_values: list[float] = []
    common_prefix_counts: list[int] = []
    for row in pair_rows:
        endpoint_shift = row.get("right_minus_left_endpoint_shift_m")
        raw_y_shift = row.get("right_minus_left_raw_object_robot_y_shift_m")
        action_rms = row.get("action_rms_common_prefix")
        common_prefix = row.get("common_prefix_actions")
        require(
            close(endpoint_shift, row.get("right_signed_final_lateral_offset_m") - row.get("left_signed_final_lateral_offset_m"))
            and close(raw_y_shift, row.get("right_raw_robot_y_m") - row.get("left_raw_robot_y_m")),
            "V3-A002 per-pair RIGHT-minus-LEFT endpoint and raw-y differences recompute",
            checks,
        )
        expected_order = "aligned" if endpoint_shift < 0 else "anti_aligned" if endpoint_shift > 0 else "tie"
        require(
            row.get("endpoint_ordering") == expected_order,
            "V3-A002 endpoint labels follow the strict RIGHT-minus-LEFT sign rule",
            checks,
        )
        require(
            isinstance(action_rms, (int, float))
            and not isinstance(action_rms, bool)
            and math.isfinite(action_rms)
            and action_rms > 0
            and row.get("executed_actions_distinct") is True
            and type(common_prefix) is int
            and 0 < common_prefix <= 450,
            "V3-A002 common-prefix RMS is finite, nonzero, and paired to a valid executed prefix",
            checks,
        )
        ordering[expected_order] += 1
        endpoint_shifts.append(endpoint_shift)
        raw_y_shifts.append(raw_y_shift)
        action_rms_values.append(action_rms)
        common_prefix_counts.append(common_prefix)
    observed_ordering = summary.get("endpoint_ordering", {})
    require(
        ordering == Counter({"aligned": 16, "anti_aligned": 3, "tie": 1})
        and observed_ordering.get("aligned") == 16
        and observed_ordering.get("anti_aligned") == 3
        and observed_ordering.get("ties") == 1
        and "strictly negative" in observed_ordering.get("definition", ""),
        "V3-A002 endpoint ordering recomputes to 16 aligned, 3 anti-aligned, and 1 tie",
        checks,
    )
    expected_sign_p = exact_two_sided_binomial_p(ordering["aligned"], ordering["anti_aligned"])
    require(
        close(observed_ordering.get("exact_two_sided_sign_test_p_excluding_ties"), expected_sign_p)
        and close(expected_sign_p, 0.004425048828125),
        "V3-A002 exact two-sided endpoint sign-test p-value recomputes excluding the tie",
        checks,
    )
    require_numeric_summary(summary.get("right_minus_left_endpoint_shift_m", {}), endpoint_shifts, "V3-A002 endpoint-shift", checks)
    require_numeric_summary(
        summary.get("right_minus_left_raw_object_robot_y_shift_m_geometry_diagnostic", {}),
        raw_y_shifts,
        "V3-A002 raw-y geometry-diagnostic shift",
        checks,
    )
    require_numeric_summary(summary.get("action_rms_common_prefix", {}), action_rms_values, "V3-A002 common-prefix action RMS", checks)
    require_numeric_summary(summary.get("common_prefix_action_count", {}), common_prefix_counts, "V3-A002 common-prefix action count", checks)
    require(
        summary.get("action_rms_common_prefix", {}).get("not_meters_or_path_distance") is True
        and summary.get("action_rms_common_prefix", {}).get("unit") == "descriptive_mixed_native_action_coordinates"
        and summary.get("distinct_executed_action_pairs", {}).get("count") == 20
        and summary.get("distinct_executed_action_pairs", {}).get("pairs") == 20,
        "V3-A002 reports 20/20 distinct traces and does not mislabel action RMS as distance",
        checks,
    )
    require(
        summary.get("whole_file_executed_action_hash_differences_integrity_only")
        == {"count": 20, "not_a_behavioral_metric": True, "pairs": 20}
        and all(row.get("whole_file_hashes_differ_integrity_only") is True for row in pair_rows),
        "V3-A002 keeps whole-file hash differences as integrity metadata only",
        checks,
    )
    require(
        summary.get("failure_taxonomy") == {
            "correct": 12,
            "pick_failed": 22,
            "release_failed": 2,
            "transport_failed": 4,
        }
        and sum(summary.get("failure_taxonomy", {}).values()) == 40
        and summary.get("failure_taxonomy", {}).get("correct") == left_successes + right_successes,
        "V3-A002 failure taxonomy accounts for all 40 episodes and all 12 successes",
        checks,
    )
    require(
        summary.get("total_actions_executed") == 16175
        and summary.get("infrastructure_attempt_pairs") == 0
        and summary.get("infrastructure_attempt_cell_records") == 0
        and summary.get("thermal_guard") == {
            "denominator_effect": "none",
            "intervention_pairs": 0,
            "retained_pair_ledgers": 20,
        },
        "V3-A002 summary accounts for actions and keeps zero infrastructure attempts outside behavior",
        checks,
    )
    require(
        same_local_file_reference(summary.get("post_result_trace_validation_amendment"), paths["pi0_trace_amendment"])
        and summary.get("post_result_trace_validation_amendment", {}).get("amendment_id") == "V3-A003"
        and summary.get("post_result_trace_validation_amendment", {}).get("applied_after_original_runtime_validation") is True
        and summary.get("post_result_trace_validation_amendment", {}).get("raw_artifacts_mutated") is False,
        "V3-A002 summary binds the read-only V3-A003 trace-validation amendment",
        checks,
    )
    require(
        same_local_file_reference(summary.get("infrastructure_intervention_ledger"), paths["pi0_bridge_ledger"]),
        "V3-A002 summary binds its compact infrastructure ledger",
        checks,
    )

    require(
        manifest.get("schema_version") == "vla-wam-shared-v3-pi0-fast-old-name-config-hash-manifest-v3"
        and manifest.get("study_id") == "vla_wam_language_steerability_v3"
        and manifest.get("model_id") == model_id
        and manifest.get("raw_inputs_read_only") is True
        and manifest.get("historical_pooling_prohibited") is True,
        "V3-A002 evidence manifest is read-only and prohibits historical pooling",
        checks,
    )
    require(
        same_local_file_reference(manifest.get("summary"), paths["pi0_bridge_summary"]),
        "V3-A002 evidence-manifest summary digest matches the committed summary",
        checks,
    )
    require(
        same_local_file_reference(manifest.get("post_result_trace_validation_amendment"), paths["pi0_trace_amendment"])
        and manifest.get("post_result_trace_validation_amendment", {}).get("raw_artifacts_mutated") is False,
        "V3-A002 evidence manifest binds the read-only V3-A003 amendment digest",
        checks,
    )
    require(
        same_local_file_reference(manifest.get("infrastructure_intervention_ledger"), paths["pi0_bridge_ledger"]),
        "V3-A002 evidence manifest binds the committed infrastructure ledger digest",
        checks,
    )
    compiled_pair_manifests = manifest.get("compiled_pair_manifests")
    require(
        isinstance(compiled_pair_manifests, list)
        and len(compiled_pair_manifests) == 20
        and all(valid_file_record(record) for record in compiled_pair_manifests),
        "V3-A002 evidence manifest has 20 syntactically valid compiled pair-manifest records",
        checks,
    )
    manifest_pair_seeds: list[int] = []
    for record in compiled_pair_manifests:
        match = re.search(r"/pairs/seed(\d{4})/pair_manifest\.json$", record["path"])
        require(match is not None, "V3-A002 compiled pair-manifest paths encode their seed", checks)
        manifest_pair_seeds.append(int(match.group(1)))
    require(
        manifest_pair_seeds == seeds and len(set(manifest_pair_seeds)) == 20,
        "V3-A002 compiled pair manifests cover each seed 8310-8329 once",
        checks,
    )
    derived_artifacts = manifest.get("derived_artifacts")
    raw_artifacts = manifest.get("raw_source_artifacts")
    require(
        isinstance(derived_artifacts, list)
        and len(derived_artifacts) == 102
        and all(valid_file_record(record, require_relative_path=True) for record in derived_artifacts)
        and len({record["path"] for record in derived_artifacts}) == 102
        and len({record["relative_path"] for record in derived_artifacts}) == 102,
        "V3-A002 evidence manifest has 102 unique valid derived-file records",
        checks,
    )
    require(
        isinstance(raw_artifacts, list)
        and len(raw_artifacts) == 289
        and all(valid_file_record(record) for record in raw_artifacts)
        and len({record["path"] for record in raw_artifacts}) == 289,
        "V3-A002 evidence manifest has 289 unique valid read-only raw-source records",
        checks,
    )
    derived_index = {(record["path"], record["sha256"], record["bytes"]) for record in derived_artifacts}
    require(
        all((record["path"], record["sha256"], record["bytes"]) in derived_index for record in compiled_pair_manifests),
        "V3-A002 compiled pair-manifest records are hash-identical members of derived evidence",
        checks,
    )
    trace_records = [
        record for record in raw_artifacts
        if record["path"].endswith("/" + REQUIRED["pi0_trace_amendment"])
    ]
    compiler_records = [
        record for record in raw_artifacts
        if record["path"].endswith("/" + REQUIRED["pi0_bridge_compiler"])
    ]
    require(
        len(trace_records) == 1
        and same_local_file_reference(trace_records[0], paths["pi0_trace_amendment"])
        and len(compiler_records) == 1
        and same_local_file_reference(compiler_records[0], paths["pi0_bridge_compiler"]),
        "V3-A002 raw-source manifest binds the amendment and reproducible compiler digests",
        checks,
    )

    require(
        ledger.get("schema_version") == "vla-wam-shared-v3-pi0-fast-old-name-config-infrastructure-ledger-v1"
        and ledger.get("study_id") == "vla_wam_language_steerability_v3"
        and ledger.get("model_id") == model_id
        and ledger.get("cohort") == cohort
        and ledger.get("status") == "complete_terminal_guarded_cohort",
        "V3-A002 infrastructure ledger schema, model, cohort, and terminal status are exact",
        checks,
    )
    require(
        ledger.get("behavioral_episode_count") == 40
        and ledger.get("completed_guarded_pairs") == 20
        and ledger.get("excluded_attempt_count") == 0
        and ledger.get("attempts") == []
        and ledger.get("technical_or_partial_pair_count") == 0
        and ledger.get("technical_or_partial_cell_count") == 0
        and ledger.get("behavioral_denominator_effect") == "none"
        and ledger.get("historical_pooling_prohibited") is True,
        "V3-A002 ledger keeps 20 complete pairs and zero infrastructure records outside its denominator",
        checks,
    )
    require(
        same_local_file_reference(ledger.get("post_result_trace_validation_amendment"), paths["pi0_trace_amendment"])
        and ledger.get("post_result_trace_validation_amendment", {}).get("applied_after_original_runtime_validation") is True
        and ledger.get("post_result_trace_validation_amendment", {}).get("raw_artifacts_mutated") is False,
        "V3-A002 infrastructure ledger binds the read-only V3-A003 amendment",
        checks,
    )
    terminal_shards = ledger.get("terminal_shard_ledgers")
    require(
        isinstance(terminal_shards, list)
        and len(terminal_shards) == 3
        and all(valid_file_record(record) and isinstance(record.get("pod"), str) for record in terminal_shards)
        and [record.get("shard_id") for record in terminal_shards] == summary.get("frozen_shard_ids")
        and len({record["path"] for record in terminal_shards}) == 3,
        "V3-A002 infrastructure ledger binds the three exact terminal shard ledgers",
        checks,
    )
    thermal_guard = ledger.get("thermal_guard", {})
    thermal_pairs = thermal_guard.get("pairs")
    require(
        thermal_guard.get("intervention_pair_count") == 0
        and thermal_guard.get("retained_pair_ledger_count") == 20
        and isinstance(thermal_pairs, list)
        and len(thermal_pairs) == 20
        and [record.get("seed") for record in thermal_pairs] == seeds
        and all(
            record.get("pair_id") == f"v3:droid:{model_id}:seed{record.get('seed')}"
            and record.get("intervention") is False
            and isinstance(record.get("pod"), str)
            and valid_file_record(record.get("ledger"))
            for record in thermal_pairs
        )
        and len({record["ledger"]["path"] for record in thermal_pairs}) == 20,
        "V3-A002 ledger retains 20 exact guarded pair records with no intervention",
        checks,
    )
    raw_index = {(record["path"], record["sha256"], record["bytes"]) for record in raw_artifacts}
    require(
        all((record["path"], record["sha256"], record["bytes"]) in raw_index for record in terminal_shards)
        and all(
            (record["ledger"]["path"], record["ledger"]["sha256"], record["ledger"]["bytes"]) in raw_index
            for record in thermal_pairs
        ),
        "V3-A002 terminal shard and thermal ledger records are hash-identical raw evidence",
        checks,
    )

    require(
        media.get("schema_version") == "vla-wam-shared-v3-pi0-fast-old-name-config-media-v1"
        and media.get("study_id") == "vla_wam_language_steerability_v3"
        and media.get("amendment_id") == "V3-A002"
        and media.get("model_id") == model_id
        and media.get("status") == "complete_selected_matched_actual_rollout"
        and media.get("seed") == 8311
        and media.get("browser_encoding") == "H.264 / yuv420p / faststart / no audio"
        and "no imagined-future counterpart" in media.get("claim_boundary", ""),
        "V3-A002 selected media is exact action-only seed-8311 H.264 execution evidence",
        checks,
    )
    media_pair = next(row for row in pair_rows if row["seed"] == 8311)
    directions = media.get("directions", {})
    require(
        directions.get("left", {}).get("prompt") == left_prompt
        and directions.get("left", {}).get("success") is False
        and directions.get("right", {}).get("prompt") == right_prompt
        and directions.get("right", {}).get("success") is True
        and all(
            (
                directions.get(relation, {}).get("source_video", {}).get("path"),
                directions.get(relation, {}).get("source_video", {}).get("sha256"),
                directions.get(relation, {}).get("source_video", {}).get("bytes"),
            )
            in raw_index
            for relation in ("left", "right")
        ),
        "V3-A002 selected media binds both exact prompts, outcomes, and raw viewport hashes",
        checks,
    )
    diagnostics = media.get("matched_pair_diagnostics", {})
    require(
        diagnostics.get("endpoint_ordering") == media_pair.get("endpoint_ordering")
        and close(diagnostics.get("right_minus_left_endpoint_shift_m"), media_pair.get("right_minus_left_endpoint_shift_m"))
        and close(diagnostics.get("action_rms_common_prefix"), media_pair.get("action_rms_common_prefix"))
        and diagnostics.get("common_prefix_actions") == media_pair.get("common_prefix_actions")
        and diagnostics.get("action_rms_unit") == summary.get("action_rms_common_prefix", {}).get("unit")
        and media.get("frame_count") == 450
        and close(media.get("fps"), 15.0)
        and close(media.get("duration_seconds"), 30.0),
        "V3-A002 selected media reproduces the seed-8311 matched-pair diagnostics and full rollout duration",
        checks,
    )
    require(
        media.get("source_summary", {}).get("path") == REQUIRED["pi0_bridge_summary"]
        and same_local_file_reference(media.get("source_summary"), paths["pi0_bridge_summary"])
        and media.get("source_evidence_manifest", {}).get("path") == REQUIRED["pi0_bridge_manifest"]
        and same_local_file_reference(media.get("source_evidence_manifest"), paths["pi0_bridge_manifest"])
        and media.get("source_amendment", {}).get("path") == REQUIRED["pi0_bridge_amendment"]
        and same_local_file_reference(media.get("source_amendment"), paths["pi0_bridge_amendment"])
        and media.get("renderer", {}).get("path") == REQUIRED["pi0_bridge_media_renderer"]
        and same_local_file_reference(media.get("renderer"), paths["pi0_bridge_media_renderer"])
        and media.get("publication_video", {}).get("path") == REQUIRED["pi0_bridge_media_video"]
        and same_local_file_reference(media.get("publication_video"), paths["pi0_bridge_media_video"])
        and media.get("poster", {}).get("path") == REQUIRED["pi0_bridge_media_poster"]
        and same_local_file_reference(media.get("poster"), paths["pi0_bridge_media_poster"]),
        "V3-A002 selected media hash-binds its sources, renderer, H.264 video, and poster",
        checks,
    )

    return {
        "summary": summary,
        "manifest": manifest,
        "ledger": ledger,
        "release_gate": release_gate,
        "media": media,
    }


def validate_pi0_fast_bridge_continuation(
    continuation: dict[str, Any],
    bridge_result: dict[str, Any],
    paths: dict[str, Path],
    checks: list[str],
) -> None:
    summary = bridge_result["summary"]
    require(
        continuation.get("schema_version") == "vla-wam-shared-v3-continuation-v1"
        and continuation.get("study_id") == "vla_wam_language_steerability_v3",
        "V3 continuation schema and study identity remain exact",
        checks,
    )
    authoritative = continuation.get("authoritative_files", [])
    required_authoritative = {
        REQUIRED["pi0_bridge_amendment"],
        REQUIRED["pi0_trace_amendment"],
        REQUIRED["pi0_bridge_release_gate"],
        REQUIRED["pi0_bridge_summary"],
        REQUIRED["pi0_bridge_manifest"],
        REQUIRED["pi0_bridge_ledger"],
        REQUIRED["pi0_bridge_media_manifest"],
    }
    require(
        required_authoritative.issubset(set(authoritative)),
        "V3 continuation names both bridge amendments, all four compact results, and selected media as authoritative",
        checks,
    )
    droid_results = continuation.get("phase_a_results", {}).get("droid_robolab", {})
    bridge = droid_results.get("post_result_bridge_cohorts", {}).get("pi0_fast_old_name_config_v3a002")
    require(isinstance(bridge, dict), "V3 continuation records the separate V3-A002 bridge cohort", checks)
    require(
        bridge.get("status") == "complete_20_matched_pairs_40_behavioral_episodes"
        and bridge.get("cohort") == summary.get("cohort")
        and bridge.get("historical_pooling_prohibited") is True
        and bridge.get("pooled_with_historical_pi0_fast") is False
        and bridge.get("environment_seeds_inclusive") == [8310, 8329]
        and bridge.get("matched_pairs") == 20
        and bridge.get("behavioral_episodes") == 40,
        "V3 continuation preserves exact separate bridge accounting and non-pooling",
        checks,
    )
    require(
        bridge.get("runtime_identity") == {
            "openpi_commit": "235044ed8a1502c0a18338eedc5d7adfe705af05",
            "openpi_tree": "03a4387bedbc0fa1467c367c60fc24e28b61ec6c",
            "openpi_config": "pi0_fast_droid_jointpos",
            "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
        },
        "V3 continuation binds the bridge runtime identity",
        checks,
    )
    require(
        bridge.get("success_by_direction") == {"left": "0/20", "right": "12/20"}
        and bridge.get("success_discordance") == summary.get("success_discordance")
        and all(
            bridge.get("endpoint_ordering", {}).get(key) == summary.get("endpoint_ordering", {}).get(key)
            for key in ("aligned", "anti_aligned", "ties", "exact_two_sided_sign_test_p_excluding_ties")
        )
        and close(bridge.get("right_minus_left_endpoint_shift_m", {}).get("mean"), summary.get("right_minus_left_endpoint_shift_m", {}).get("mean"))
        and close(bridge.get("right_minus_left_endpoint_shift_m", {}).get("median"), summary.get("right_minus_left_endpoint_shift_m", {}).get("median")),
        "V3 continuation mirrors bridge success, discordance, and endpoint statistics",
        checks,
    )
    state_taxonomy = dict(bridge.get("failure_taxonomy", {}))
    state_taxonomy.pop("wrong_side", None)
    require(
        state_taxonomy == summary.get("failure_taxonomy")
        and bridge.get("failure_taxonomy", {}).get("wrong_side") == 0
        and bridge.get("common_prefix_action_rms", {}).get("nonzero_pairs") == "20/20"
        and close(bridge.get("common_prefix_action_rms", {}).get("mean"), summary.get("action_rms_common_prefix", {}).get("mean"))
        and close(bridge.get("common_prefix_action_rms", {}).get("median"), summary.get("action_rms_common_prefix", {}).get("median")),
        "V3 continuation mirrors bridge taxonomy and common-prefix action sensitivity",
        checks,
    )
    require(
        bridge.get("contact_measurement") == "instrumentation_unavailable_in_all_40_episodes_not_encoded_as_zero"
        and bridge.get("infrastructure") == {
            "excluded_attempt_pairs": 0,
            "runtime_interventions": 0,
            "retained_thermal_ledgers": 20,
        },
        "V3 continuation keeps unavailable contact distinct from zero and infrastructure outside behavior",
        checks,
    )
    result_keys = {
        "release_gate": "pi0_bridge_release_gate",
        "summary": "pi0_bridge_summary",
        "evidence_manifest": "pi0_bridge_manifest",
        "infrastructure_ledger": "pi0_bridge_ledger",
    }
    require(
        all(
            bridge.get(field) == {"path": REQUIRED[path_key], "sha256": sha256(paths[path_key])}
            for field, path_key in result_keys.items()
        ),
        "V3 continuation binds bridge result paths to their exact committed SHA-256 digests",
        checks,
    )
    require(
        bridge.get("publication_media")
        == {
            "manifest_path": REQUIRED["pi0_bridge_media_manifest"],
            "manifest_sha256": sha256(paths["pi0_bridge_media_manifest"]),
            "selected_seed": 8311,
            "kind": "actual_simulator_execution_action_only",
        },
        "V3 continuation hash-binds the selected actual-only bridge media",
        checks,
    )
    blocked = continuation.get("blocked_and_unreleased", {}).get("pi0_fast_phase_a", {})
    require(
        blocked.get("status") == "historical_identity_still_blocked_separate_v3a002_bridge_complete"
        and blocked.get("preserved_v2_pairs") == 10
        and blocked.get("preserved_v2_cells") == 20
        and blocked.get("blocked_new_pairs") == 20
        and blocked.get("blocked_new_cells") == 40
        and blocked.get("blocked_seed_range") == [8310, 8329]
        and blocked.get("required_openpi_commit") == "9e46d3aea26417bfb564227734b95d010aa827e5"
        and blocked.get("required_robolab_commit") == "11142d4319e44401e0464866bb5fedf7ec8a8927"
        and blocked.get("historical_identity_v3_behavioral_episode_count") == 0
        and blocked.get("separate_bridge") == {
            "model_id": "pi0_fast_old_name_config_v3a002",
            "matched_pairs": 20,
            "behavioral_episodes": 40,
            "pooled_with_historical": False,
        },
        "V3 continuation leaves the historical 40-cell identity blocked while linking the separate bridge",
        checks,
    )
    accounting = continuation.get("phase_a_accounting", {})
    require(
        accounting.get("frozen_queue_rows") == 780
        and accounting.get("launch_authorized_new_cells") == 648
        and accounting.get("completed_valid_new_cells") == 648
        and accounting.get("blocked_pi0_fast_cells") == 40,
        "V3 continuation does not rewrite frozen 780/648/40 Phase-A queue accounting",
        checks,
    )


def validate(root: Path) -> list[str]:
    checks: list[str] = []
    paths = {name: root / relative for name, relative in REQUIRED.items()}
    for name, path in paths.items():
        require(path.is_file(), f"required v3 {name} artifact exists", checks)

    protocol = load(paths["protocol"])
    amendment = load(paths["amendment"])
    droid = load(paths["droid"])
    robotwin = load(paths["robotwin"])
    wording_registry = load(paths["wording"])
    stochastic_registry = load(paths["stochastic"])
    calibration_registry = load(paths["confound_calibration"])
    phase_a_manifest = load(paths["phase_a_manifest"])
    phase_a_rows = load_jsonl(paths["phase_a_queue"])
    taxonomy = load(paths["taxonomy"])
    analysis = load(paths["analysis"])
    measurement_audit = load(paths["measurement_audit"])
    continuation = load(paths["continuation"])

    nano_calibration = load(paths["nano_mirror_calibration"])
    nano_amendment = load(paths["nano_mirror_amendment"])
    nano_manifest = load(paths["nano_mirror_manifest"])
    nano_live_ledger = load(paths["nano_mirror_live_ledger"])
    nano_reset_correction = load(paths["nano_mirror_reset_correction"])
    nano_reset_preflight = load(paths["nano_mirror_reset_preflight"])
    nano_live_snapshot = load(paths["nano_mirror_live_snapshot"])
    nano_cells = load_jsonl(paths["nano_mirror_cells"])
    nano_results_manifest = load(paths["nano_mirror_results_manifest"])
    nano_summary = load(paths["nano_mirror_summary"])
    nano_episodes = load_jsonl(paths["nano_mirror_episodes"])
    nano_episodes_manifest = load(paths["nano_mirror_episodes_manifest"])
    nano_final_evidence = load(paths["nano_mirror_final_evidence"])
    nano_figure_media_manifest = load(paths["nano_mirror_figure_media_manifest"])
    nano_publication_media_manifest = load(paths["nano_mirror_publication_media_manifest"])
    pi05_mirror_amendment = load(paths["pi05_mirror_amendment"])
    pi05_mirror_cells = load_jsonl(paths["pi05_mirror_cells"])
    pi05_mirror_manifest = load(paths["pi05_mirror_manifest"])
    pi05_failure_split_episodes = load_jsonl(paths["pi05_failure_split_episodes"])
    pi05_failure_split_report = load(paths["pi05_failure_split_report"])
    pi05_failure_split_manifest = load(paths["pi05_failure_split_manifest"])
    dreamzero_mirror_amendment = load(paths["dreamzero_mirror_amendment"])
    dreamzero_mirror_cells = load_jsonl(paths["dreamzero_mirror_cells"])
    dreamzero_mirror_manifest = load(paths["dreamzero_mirror_manifest"])
    nano_lateral_sweep_amendment = load(paths["nano_lateral_sweep_amendment"])

    require(
        nano_calibration.get("schema_version")
        == "vla-wam-shared-v3b-nano-position-mirror-model-blind-calibration-v1"
        and nano_calibration.get("passed") is True
        and nano_calibration.get("model_request_count") == 0
        and nano_calibration.get("behavioral_episode_count") == 0,
        "Nano V3-B001 calibration passed model-blind with zero model requests and behavioral episodes",
        checks,
    )
    require(
        nano_calibration.get("model_id") == "cosmos3_nano_policy_droid"
        and nano_calibration.get("renderer", {}).get("backend") == "realtime RTX Vulkan"
        and nano_calibration.get("renderer", {}).get("all_required_rgb_views_nonblank") is True
        and nano_calibration.get("reset_gate", {}).get("neither_predicate_true_at_every_reset") is True
        and nano_calibration.get("reset_gate", {}).get("live_position_reflection_passed_at_every_repeat") is True
        and nano_calibration.get("reset_gate", {}).get("post_settle_quaternion_differences_recorded_not_gated") is True,
        "Nano V3-B001 calibration binds live RTX views, neutral resets, reflection, and orientation mediators",
        checks,
    )
    require(
        nano_amendment.get("schema_version")
        == "vla-wam-shared-v3b-nano-mirror-amendment-v1"
        and nano_amendment.get("amendment_id") == "V3-B001"
        and nano_amendment.get("status")
        == "released_after_model_blind_calibration_before_any_phase_b_model_request"
        and nano_amendment.get("exact_prompts")
        == {
            "left": EXACT_V2_WORDINGS["direct_command"]["left"],
            "right": EXACT_V2_WORDINGS["direct_command"]["right"],
        },
        "Nano V3-B001 amendment is released before inference with both exact static prompts",
        checks,
    )
    nano_design = nano_amendment.get("design", {})
    require(
        nano_design.get("arms") == ["control", "position_mirrored"]
        and nano_design.get("directions") == ["left", "right"]
        and nano_design.get("matched_seed_count") == 27
        and nano_design.get("behavioral_cell_count") == 108
        and nano_design.get("seeds") == exact_range(9400, 9426)
        and nano_design.get("factor")
        == "movable_object_center_position_reflection_about_robot_sagittal_plane",
        "Nano V3-B001 freezes 27 seeds and 108 position-reflection cells without calling it a full-scene mirror",
        checks,
    )
    require(
        nano_amendment.get("analysis_plan", {}).get("full_sample_primary", {}).get(
            "position_reflection_interaction"
        )
        == "I[i] = B[position_mirrored,i] - B[control,i]"
        and nano_amendment.get("analysis_plan", {}).get("success_conditional_secondary", {}).get(
            "complete_case_subset_id"
        )
        == "nano_v3b001_all_four_cells_correct",
        "Nano V3-B001 keeps signed offset primary and names the four-success margin subset",
        checks,
    )
    require(
        nano_manifest.get("schema_version")
        == "vla-wam-shared-v3b-nano-mirror-manifest-v1"
        and nano_manifest.get("status") == "hash_bound_release_ready"
        and nano_manifest.get("counts")
        == {
            "behavioral_cells": 108,
            "control_cells": 54,
            "left_cells": 54,
            "matched_seeds": 27,
            "position_mirrored_cells": 54,
            "right_cells": 54,
        },
        "Nano V3-B001 manifest accounts for all 108 released cells",
        checks,
    )
    require(
        nano_manifest.get("calibration_report")
        == {
            "path": "model_blind_calibration_report.json",
            "bytes": paths["nano_mirror_calibration"].stat().st_size,
            "sha256": sha256(paths["nano_mirror_calibration"]),
        }
        and nano_manifest.get("files", {}).get("amendment")
        == {
            "path": "post_result_nano_mirror_v3b001_amendment.json",
            "bytes": paths["nano_mirror_amendment"].stat().st_size,
            "sha256": sha256(paths["nano_mirror_amendment"]),
        }
        and nano_manifest.get("files", {}).get("cells")
        == {
            "path": "nano_mirror_v3b001_cells.jsonl",
            "bytes": paths["nano_mirror_cells"].stat().st_size,
            "row_count": 108,
            "sha256": sha256(paths["nano_mirror_cells"]),
        },
        "Nano V3-B001 manifest hash-binds calibration, amendment, and exact cell queue",
        checks,
    )
    require(
        nano_amendment.get("calibration_report") == nano_manifest.get("calibration_report"),
        "Nano V3-B001 amendment and manifest bind the same calibration report",
        checks,
    )
    require(
        all(
            isinstance(relative, str)
            and isinstance(digest, str)
            and (root / relative).is_file()
            and sha256(root / relative) == digest
            for relative, digest in nano_amendment.get("source_bindings", {}).items()
        )
        and len(nano_amendment.get("source_bindings", {})) == 12,
        "Nano V3-B001 amendment hash-binds all twelve committed source inputs",
        checks,
    )
    require(
        len(nano_cells) == 108
        and len({row.get("cell_id") for row in nano_cells}) == 108
        and Counter(row.get("environment_seed") for row in nano_cells)
        == Counter({seed: 4 for seed in exact_range(9400, 9426)})
        and Counter((row.get("arm"), row.get("relation")) for row in nano_cells)
        == Counter(
            {
                ("control", "left"): 27,
                ("control", "right"): 27,
                ("position_mirrored", "left"): 27,
                ("position_mirrored", "right"): 27,
            }
        ),
        "Nano V3-B001 queue contains each seed and arm-direction cell exactly once",
        checks,
    )
    require(
        all(
            row.get("sampling_seed") == row.get("environment_seed")
            and row.get("prompt")
            == EXACT_V2_WORDINGS["direct_command"].get(row.get("relation"))
            and row.get("amendment_sha256") == sha256(paths["nano_mirror_amendment"])
            and row.get("execution_status")
            == "authorized_after_v3b001_calibration_with_live_identity_and_output_gate_recheck"
            for row in nano_cells
        ),
        "Nano V3-B001 cells preserve matched seeds, exact prompts, release hash, and live recheck gate",
        checks,
    )
    nano_cells_by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in nano_cells:
        nano_cells_by_seed.setdefault(row["environment_seed"], []).append(row)
    require(
        all(
            {row.get("execution_order_index_within_seed") for row in rows} == {1, 2, 3, 4}
            for rows in nano_cells_by_seed.values()
        ),
        "Nano V3-B001 block randomization assigns each seed all four execution positions",
        checks,
    )
    nano_state = continuation.get("phase_b_releases", {}).get(
        "nano_position_reflection_v3b001", {}
    )
    require(
        nano_state.get("status")
        == "complete_hash_closed_27_seed_108_episode_result"
        and nano_state.get("released_behavioral_cell_count") == 108
        and nano_state.get("completed_behavioral_cell_count") == 108
        and nano_state.get("completed_behavioral_cell_count_scope")
        == "complete_prespecified_cohort"
        and nano_state.get("committed_complete_seed_blocks") == exact_range(9400, 9426)
        and nano_state.get("live_queue_may_be_ahead_of_committed_snapshot") is False
        and nano_state.get("pre_release_counts")
        == {"model_requests": 0, "behavioral_episodes": 0}
        and nano_state.get("seed_range_inclusive") == [9400, 9426]
        and nano_state.get("prespecified_matched_seed_count") == 27,
        "V3 continuation records Nano V3-B001 as a complete 108-cell cohort",
        checks,
    )
    nano_state_artifact_keys = {
        "calibration": "nano_mirror_calibration",
        "cells": "nano_mirror_cells",
        "manifest": "nano_mirror_manifest",
        "amendment": "nano_mirror_amendment",
        "live_infrastructure_ledger": "nano_mirror_live_ledger",
        "live_reset_semantics_correction": "nano_mirror_reset_correction",
        "live_reset_preflight_report": "nano_mirror_reset_preflight",
        "live_queue_snapshot": "nano_mirror_live_snapshot",
        "final_results_manifest": "nano_mirror_results_manifest",
        "final_summary": "nano_mirror_summary",
        "final_evidence_manifest": "nano_mirror_final_evidence",
        "publication_media_manifest": "nano_mirror_publication_media_manifest",
    }
    require(
        all(
            nano_state.get("artifacts", {}).get(label)
            == {"path": REQUIRED[path_key], "sha256": sha256(paths[path_key])}
            and REQUIRED[path_key] in continuation.get("authoritative_files", [])
            for label, path_key in nano_state_artifact_keys.items()
        ),
        "V3 continuation hash-binds Nano V3-B001 release, historical live, final-result, and publication artifacts",
        checks,
    )
    nano_results_root = paths["nano_mirror_results_manifest"].parent
    nano_result_files = nano_results_manifest.get("files", [])
    require(
        nano_results_manifest.get("schema_version")
        == "vla-wam-shared-v3b-nano-v3b001-results-manifest-v1"
        and nano_results_manifest.get("status")
        == "complete_hash_closed_27_seed_108_episode_result"
        and nano_results_manifest.get("runtime_identity", {}).get(
            "runtime_identity_sha256"
        )
        == "2c5e314a62926a3d3c9b84fa73cff634c378b6140d390cbef6407ac9c633e64e"
        and nano_results_manifest.get("runtime_identity", {}).get(
            "study_source_commit"
        )
        == "39f19b57d5f5e68d0f03e4e4c7569350cb9a467f"
        and nano_results_manifest.get("runtime_identity", {}).get(
            "release_manifest_sha256"
        )
        == sha256(paths["nano_mirror_manifest"])
        and nano_results_manifest.get("design", {}).get("matched_seed_count") == 27
        and nano_results_manifest.get("design", {}).get(
            "valid_behavioral_episode_count"
        )
        == 108
        and nano_results_manifest.get("design", {}).get("exact_prompts")
        == {
            "left": EXACT_V2_WORDINGS["direct_command"]["left"],
            "right": EXACT_V2_WORDINGS["direct_command"]["right"],
        }
        and len(nano_result_files) == 18
        and len({item.get("path") for item in nano_result_files}) == 18
        and all(
            isinstance(item.get("path"), str)
            and (nano_results_root / item["path"]).is_file()
            and (nano_results_root / item["path"]).stat().st_size == item.get("bytes")
            and sha256(nano_results_root / item["path"]) == item.get("sha256")
            for item in nano_result_files
        ),
        "Nano V3-B001 final results manifest hash-binds all compact evidence, figures, and publication media",
        checks,
    )
    expected_nano_condition_outcomes = {
        "control:left": {
            "episodes": 27,
            "failure_taxonomy_counts": {"correct": 26, "transport_failed": 1},
            "successes": 26,
        },
        "control:right": {
            "episodes": 27,
            "failure_taxonomy_counts": {"correct": 26, "release_failed": 1},
            "successes": 26,
        },
        "position_mirrored:left": {
            "episodes": 27,
            "failure_taxonomy_counts": {"correct": 27},
            "successes": 27,
        },
        "position_mirrored:right": {
            "episodes": 27,
            "failure_taxonomy_counts": {"correct": 23, "transport_failed": 4},
            "successes": 23,
        },
    }
    require(
        nano_summary.get("schema_version")
        == "vla-wam-shared-v3b-nano-v3b001-results-v1"
        and nano_summary.get("study_id") == "vla_wam_language_steerability_v3"
        and nano_summary.get("amendment_id") == "V3-B001"
        and nano_summary.get("model_id") == "cosmos3_nano_policy_droid"
        and nano_summary.get("runtime_identity_sha256")
        == "2c5e314a62926a3d3c9b84fa73cff634c378b6140d390cbef6407ac9c633e64e"
        and nano_summary.get("behavioral_evidence", {}).get("matched_seed_count") == 27
        and nano_summary.get("behavioral_evidence", {}).get("valid_episode_count") == 108
        and nano_summary.get("condition_outcomes") == expected_nano_condition_outcomes
        and nano_summary.get("failure_taxonomy_counts")
        == {"correct": 102, "release_failed": 1, "transport_failed": 5}
        and nano_summary.get("exact_prompts")
        == {
            "left": EXACT_V2_WORDINGS["direct_command"]["left"],
            "right": EXACT_V2_WORDINGS["direct_command"]["right"],
        }
        and nano_summary.get("uncertainty_contract", {}).get("bootstrap_replicates")
        == 20000
        and nano_summary.get("uncertainty_contract", {}).get("bootstrap_master_seed")
        == 3104159,
        "Nano V3-B001 summary retains the exact cohort, prompts, outcomes, and prespecified uncertainty contract",
        checks,
    )
    nano_primary = nano_summary.get("full_sample_primary", {})
    nano_D = nano_primary.get("D_by_arm", {})
    nano_B = nano_primary.get("B_by_arm", {})
    nano_I = nano_primary.get("I_position_reflection_interaction", {})
    nano_J = nano_primary.get("J_redirection_interaction", {})
    require(
        nano_D.get("control", {}).get("n") == 27
        and close(nano_D.get("control", {}).get("median_m"), 0.4667481780052185)
        and nano_D.get("control", {}).get("paired_sign_test", {}).get("positive") == 27
        and close(
            nano_D.get("position_mirrored", {}).get("median_m"),
            0.4373619556427002,
        )
        and nano_D.get("position_mirrored", {}).get("paired_sign_test", {}).get(
            "positive"
        )
        == 27
        and close(nano_J.get("median_m"), 0.007228171452879906)
        and close(nano_J.get("paired_sign_test", {}).get("p_value"), 0.7011080384254456)
        and close(nano_B.get("control", {}).get("median_m"), 0.14758701622486115)
        and close(
            nano_B.get("position_mirrored", {}).get("median_m"),
            -0.08814282715320587,
        )
        and close(nano_I.get("median_m"), -0.2463150918483734)
        and nano_I.get("paired_sign_test", {}).get("negative") == 24
        and close(
            nano_I.get("paired_sign_test", {}).get("p_value"),
            4.9233436584472656e-05,
        ),
        "Nano V3-B001 primary result separates persistent endpoint redirection from the reflected side-depth interaction",
        checks,
    )
    nano_secondary = nano_summary.get("success_conditional_secondary", {})
    require(
        nano_secondary.get("subset_id") == "nano_v3b001_all_four_cells_correct"
        and nano_secondary.get("realized_matched_seed_count") == 21
        and nano_secondary.get("failures_as_zero") is False
        and nano_secondary.get("unmatched_successful_cells_used") is False
        and close(
            nano_secondary.get("G_position_reflection_interaction", {}).get("median_m"),
            -0.2282327301800251,
        )
        and nano_secondary.get("G_position_reflection_interaction", {}).get(
            "paired_sign_test", {}
        ).get("negative")
        == 18
        and close(
            nano_secondary.get("G_position_reflection_interaction", {})
            .get("paired_sign_test", {})
            .get("p_value"),
            0.0014896392822265625,
        ),
        "Nano V3-B001 secondary margin result uses only the named 21-seed all-four-correct subset",
        checks,
    )
    nano_episode_ids = {row.get("registered_cell_id") for row in nano_episodes}
    require(
        len(nano_episodes) == 108
        and len(nano_episode_ids) == 108
        and nano_episode_ids == {row.get("cell_id") for row in nano_cells}
        and sum(row.get("actions_executed", 0) for row in nano_episodes) == 21972
        and sum(len(row.get("future_requests", [])) for row in nano_episodes) == 738
        and all(
            row.get("behavioral_result_valid") is True
            and row.get("runtime_identity", {}).get("sha256")
            == "2c5e314a62926a3d3c9b84fa73cff634c378b6140d390cbef6407ac9c633e64e"
            and row.get("prompt")
            == EXACT_V2_WORDINGS["direct_command"][row.get("requested_relation")]
            and row.get("failure_taxonomy")
            in {"correct", "pick_failed", "transport_failed", "wrong_side", "release_failed"}
            and isinstance(row.get("measurements", {}).get("signed_final_lateral_offset_m"), (int, float))
            for row in nano_episodes
        ),
        "Nano V3-B001 aggregate contains all 108 exact cells, 21,972 actions, 738 retained futures, and full-sample offsets",
        checks,
    )
    aggregate_evidence = nano_summary.get("behavioral_evidence", {}).get(
        "aggregate_jsonl", {}
    )
    require(
        aggregate_evidence
        == {
            "bytes": paths["nano_mirror_episodes"].stat().st_size,
            "manifest_path": paths["nano_mirror_episodes_manifest"].name,
            "manifest_sha256": sha256(paths["nano_mirror_episodes_manifest"]),
            "path": paths["nano_mirror_episodes"].name,
            "sha256": sha256(paths["nano_mirror_episodes"]),
        }
        and nano_episodes_manifest.get("jsonl_sha256")
        == sha256(paths["nano_mirror_episodes"])
        and nano_episodes_manifest.get("jsonl_bytes")
        == paths["nano_mirror_episodes"].stat().st_size
        and nano_episodes_manifest.get("row_count") == 108,
        "Nano V3-B001 summary and aggregate manifest bind the exact 108-row JSONL",
        checks,
    )
    require(
        nano_final_evidence.get("schema_version")
        == "vla-wam-shared-v3b-nano-v3b001-final-evidence-v1"
        and nano_final_evidence.get("status")
        == "complete_hash_closed_108_cell_evidence"
        and nano_final_evidence.get("runtime_identity_sha256")
        == "2c5e314a62926a3d3c9b84fa73cff634c378b6140d390cbef6407ac9c633e64e"
        and nano_final_evidence.get("counts", {}).get("behavioral_episode_count") == 108
        and nano_final_evidence.get("counts", {}).get("matched_seed_count") == 27
        and nano_final_evidence.get("counts", {}).get("raw_batch_count") == 108
        and nano_final_evidence.get("counts", {}).get("unique_referenced_artifact_count")
        == 2233
        and nano_final_evidence.get("compiled_outputs", {}).get(
            "aggregate_behavioral_jsonl", {}
        ).get("sha256")
        == sha256(paths["nano_mirror_episodes"])
        and nano_final_evidence.get("compiled_outputs", {}).get("summary", {}).get(
            "sha256"
        )
        == sha256(paths["nano_mirror_summary"])
        and nano_final_evidence.get("release_manifest", {}).get("sha256")
        == sha256(paths["nano_mirror_manifest"])
        and nano_final_evidence.get("statistics_contract", {}).get(
            "valid_behavioral_failures_included"
        )
        is True,
        "Nano V3-B001 evidence manifest hash-closes all batches, referenced assets, and compiled outputs",
        checks,
    )
    figure_root = paths["nano_mirror_figure_media_manifest"].parent
    require(
        nano_figure_media_manifest.get("schema_version")
        == "vla-wam-shared-v3b-nano-v3b001-results-media-manifest-v1"
        and nano_figure_media_manifest.get("status") == "complete"
        and nano_figure_media_manifest.get("selection", {}).get("selected_seed") == 9400
        and nano_figure_media_manifest.get("selection", {}).get(
            "selected_decoded_prediction_count"
        )
        == 30
        and nano_figure_media_manifest.get("selection", {}).get("outcome_used_for_selection")
        is False
        and all(
            (figure_root / item.get("path", "")).is_file()
            and (figure_root / item["path"]).stat().st_size == item.get("bytes")
            and sha256(figure_root / item["path"]) == item.get("sha256")
            for item in nano_figure_media_manifest.get("generated_files", {}).values()
        ),
        "Nano V3-B001 renderer retains all seeds in readable hash-bound figures and selects media without outcomes",
        checks,
    )
    publication_root = paths["nano_mirror_publication_media_manifest"].parent
    selected_publication = nano_publication_media_manifest.get("selected_media", [])
    require(
        nano_publication_media_manifest.get("schema_version")
        == "vla-wam-shared-v3b-nano-v3b001-publication-media-v1"
        and nano_publication_media_manifest.get("status")
        == "complete_minimum_seed_all_four_cells_actual_and_local_predictions"
        and nano_publication_media_manifest.get("selection", {}).get("selected_seed") == 9400
        and nano_publication_media_manifest.get("selection", {}).get("selected_cell_count") == 4
        and nano_publication_media_manifest.get("selection", {}).get("outcome_used_for_selection")
        is False
        and len(selected_publication) == 4
        and sum(
            len(item.get("source_local_prediction_horizons_in_order", []))
            for item in selected_publication
        )
        == 30
        and all(
            item.get("exact_prompt")
            == EXACT_V2_WORDINGS["direct_command"][item.get("requested_relation")]
            and all(
                (publication_root / asset.get("path", "")).is_file()
                and (publication_root / asset["path"]).stat().st_size == asset.get("bytes")
                and sha256(publication_root / asset["path"]) == asset.get("sha256")
                for asset in (item.get("publication_video", {}), item.get("poster", {}))
            )
            and item.get("output_validation", {}).get("codec_name") == "h264"
            and item.get("output_validation", {}).get("pixel_format") == "yuv420p"
            and item.get("output_validation", {}).get("has_audio") is False
            for item in selected_publication
        ),
        "Nano V3-B001 publication media shows all four selected cells with complete actual rollouts and every local future",
        checks,
    )
    nano_final_state = nano_state.get("final_result", {})
    require(
        nano_final_state.get("valid_behavioral_cells") == 108
        and nano_final_state.get("matched_seed_count") == 27
        and nano_final_state.get("valid_model_sampler_requests") == 738
        and nano_final_state.get("executed_actions") == 21972
        and nano_final_state.get("failure_taxonomy")
        == {
            "correct": 102,
            "pick_failed": 0,
            "transport_failed": 5,
            "wrong_side": 0,
            "release_failed": 1,
        }
        and close(
            nano_final_state.get("requested_depth", {}).get(
                "reflection_interaction_I_median_m"
            ),
            nano_I.get("median_m"),
        )
        and close(
            nano_final_state.get("endpoint_redirection", {}).get(
                "reflection_interaction_J_median_m"
            ),
            nano_J.get("median_m"),
        ),
        "V3 continuation mirrors the completed Nano counts and prespecified interaction estimates",
        checks,
    )
    nano_final_source_keys = {
        "nano_mirror_results_compiler",
        "nano_mirror_evidence_finalizer",
        "nano_mirror_results_renderer",
        "nano_mirror_publication_media_builder",
        "nano_mirror_results_compiler_test",
        "nano_mirror_evidence_finalizer_test",
        "nano_mirror_results_renderer_test",
        "nano_mirror_publication_media_test",
    }
    require(
        all(
            REQUIRED[key] in continuation.get("authoritative_files", [])
            for key in nano_final_source_keys
        ),
        "V3 continuation lists the Nano compiler, finalizer, renderer, media builder, and regression tests",
        checks,
    )
    nano_live = nano_state.get("live_runtime", {})
    live_identity = nano_live.get("live_bound_runtime_identity", {})
    require(
        nano_live.get("status")
        == "queue_complete_historical_live_prefix_details_retained_below"
        and nano_live.get("model_request_count") == 39
        and nano_live.get("model_request_count_semantics")
        == "prior_model_sampler_requests_retained_as_infrastructure_evidence; zero-sampling transport-handshake packets excluded"
        and nano_live.get("valid_model_sampler_request_count_in_committed_snapshot") == 90
        and nano_live.get("behavioral_episode_count") == 12
        and nano_live.get("completed_behavioral_cell_count") == 12
        and nano_live.get("infrastructure_invalid_complete_behavior_attempt_count") == 4
        and nano_live.get("behavioral_denominator") == 12
        and live_identity
        == {
            "runtime_identity_sha256": "2c5e314a62926a3d3c9b84fa73cff634c378b6140d390cbef6407ac9c633e64e",
            "manifest_path": "/data/users/ali/vla_wam/raw/v3b/cosmos3_nano_policy_droid/mirror/runtime_attempt08/runtime_identity.json",
            "manifest_bytes": 2910,
            "manifest_file_sha256": "a609300c4cbb826115692db8fecac7e14d7578e0ded00f63c06f5153619c2136",
            "study_head": "39f19b57d5f5e68d0f03e4e4c7569350cb9a467f",
            "source_state": "pushed_synced_clean",
            "preflight_status": "passed_both_physical_layouts_zero_sampling",
            "behavior_resume_policy": "same_runtime_identity_only",
        }
        and nano_live.get("invalidated_runtime_attempt06", {}).get(
            "runtime_identity_sha256"
        )
        == "fa962d1783a506abb2e0c11c5129087be62a9f698bd2a3bec2a97dddba9b6a71"
        and nano_live.get("registration_context_diagnostic", {}).get("model_requests")
        == 0,
        "V3 continuation preserves the historical Nano live prefix under the completed queue identity",
        checks,
    )
    require(
        nano_reset_correction.get("schema_version")
        == "vla-wam-shared-v3b-nano-live-reset-semantics-correction-v1"
        and nano_reset_correction.get("status")
        == "prospective_operational_correction_before_any_valid_phase_b_behavior"
        and nano_reset_correction.get("diagnosis", {}).get("registration_context_hypothesis")
        == "falsified_as_the_sufficient_root_cause"
        and nano_reset_correction.get("diagnosis", {}).get("pinned_runner", {}).get(
            "consecutive_pre_action_reset_calls"
        ) == 2
        and nano_reset_correction.get("diagnosis", {}).get("pinned_environment_runtime", {}).get(
            "sha256"
        ) == "c35a1aced26a30de1fde21f8db28f87910757d090b677172f9915432f3b45f12"
        and nano_reset_correction.get("diagnosis", {}).get("direct_evidence", {}).get(
            "committed_calibration_report", {}
        ).get("sha256") == sha256(paths["nano_mirror_calibration"])
        and nano_reset_correction.get("validity_and_resume", {}).get("behavioral_denominator") == 0
        and nano_reset_correction.get("validity_and_resume", {}).get(
            "prior_complete_attempts_invalidated_before_analysis"
        ) == 4
        and nano_reset_correction.get("validity_and_resume", {}).get(
            "total_model_requests_retained_as_infrastructure_evidence"
        ) == 39,
        "Nano reset correction is prospective, evidence-bound, and preserves a zero denominator",
        checks,
    )
    require(
        nano_reset_correction.get("prospective_live_contract")
        == {
            "runner_pre_action_reset_calls": 2,
            "physical_reset_calls": 1,
            "settle_gate_runs": 1,
            "duplicate_second_reset_idempotent": True,
            "settle_steps": 60,
            "stable_window_steps": 15,
            "episode_length_buf_before_reset": [75],
            "episode_length_buf_after_reset": [0],
            "model_request_count_before_attestation": 0,
            "position_tolerance_m": 0.003,
            "logical_second_reset_behavior": "Return the cached settled observation and info without another Isaac reset or settle gate.",
            "third_pre_action_reset_behavior": "Reject before inference.",
        }
        and all(
            (root / relative).is_file() and sha256(root / relative) == digest
            for relative, digest in nano_reset_correction.get("source_bindings", {}).items()
        )
        and len(nano_reset_correction.get("source_bindings", {})) == 4,
        "Nano reset correction hash-binds the two-call one-physical-reset implementation and tests",
        checks,
    )
    require(
        nano_reset_preflight.get("schema_version")
        == "vla-wam-shared-v3b-nano-live-reset-preflight-report-v1"
        and nano_reset_preflight.get("study_id")
        == "vla_wam_language_steerability_v3"
        and nano_reset_preflight.get("amendment_id") == "V3-B001"
        and nano_reset_preflight.get("status")
        == "passed_behavior_ready_under_same_runtime_identity"
        and nano_reset_preflight.get("source_bound_runtime")
        == {
            "branch": "codex/wam-language-steerability",
            "source_commit": "39f19b57d5f5e68d0f03e4e4c7569350cb9a467f",
            "source_state": "pushed_synced_clean",
            "manifest_path": "/data/users/ali/vla_wam/raw/v3b/cosmos3_nano_policy_droid/mirror/runtime_attempt08/runtime_identity.json",
            "manifest_bytes": 2910,
            "manifest_file_sha256": "a609300c4cbb826115692db8fecac7e14d7578e0ded00f63c06f5153619c2136",
            "runtime_identity_sha256": "2c5e314a62926a3d3c9b84fa73cff634c378b6140d390cbef6407ac9c633e64e",
        },
        "Nano reset preflight binds the clean pushed runtime-attempt-08 identity",
        checks,
    )
    require(
        nano_reset_preflight.get("execution_boundary")
        == {
            "owner": "ali",
            "omniverse_eula_acceptance": "OMNI_KIT_ACCEPT_EULA=YES",
            "model_server_loaded": False,
            "checkpoint_loaded": False,
            "request_packet_accepted_per_layout": True,
            "transport_termination": "explicit_no_inference_response",
            "model_sampler_request_count": 0,
            "model_sample_count": 0,
            "behavioral_action_count": 0,
            "behavioral_episode_count": 0,
            "denominator_eligible": False,
        }
        and nano_reset_preflight.get("common_reset_attestation")
        == {
            "runner_pre_action_reset_calls": 2,
            "physical_reset_calls": 1,
            "settle_gate_runs": 1,
            "duplicate_second_reset_idempotent": True,
            "episode_length_buf_before_reset": [75],
            "episode_length_buf_after_reset": [0],
            "model_request_count_during_gate": 0,
            "model_sample_count": 0,
        },
        "Nano exact-bridge preflight is zero-sampling, model-blind, and denominator-excluded",
        checks,
    )
    preflight_layouts = {
        item.get("attempt_id"): item
        for item in nano_reset_preflight.get("layout_preflights", [])
    }
    mirrored_preflight = preflight_layouts.get("reset_preflight_attempt01", {})
    control_preflight = preflight_layouts.get("reset_preflight_attempt02", {})
    require(
        set(preflight_layouts)
        == {"reset_preflight_attempt01", "reset_preflight_attempt02"}
        and mirrored_preflight.get("cell_id")
        == "v3b001:nano:seed9400:position_mirrored:right"
        and mirrored_preflight.get("prompt") == EXACT_V2_WORDINGS["direct_command"]["right"]
        and mirrored_preflight.get("evidence_sha256")
        == {
            "reset_attestation": "2846c4b61eba7f6036ca64cea4165ea9814e57c1c3abfedc1352d46222f17ea0",
            "settle_stability": "598fc91ad54f518a98cd1ccec39b2c7983aee41887dd4ef95736b0846689cc0f",
            "fixture_match": "55964aa7e2e3c71c13c30d7709df0d1bf97a4e6159d4661ba44227d0a48b1c75",
            "explicit_no_inference_termination": "7db0aed7731b81a64d4ff22352cbc3f8372576c2ca5f47a193701e71d5cf6bfb",
        }
        and close(mirrored_preflight.get("max_position_error_m", {}).get("banana"), 0.0002442300319671631)
        and close(mirrored_preflight.get("max_position_error_m", {}).get("bowl"), 0.00041985511779785156)
        and close(mirrored_preflight.get("max_position_error_m", {}).get("rubiks_cube"), 0.0006091296672821045)
        and control_preflight.get("cell_id") == "v3b001:nano:seed9400:control:left"
        and control_preflight.get("prompt") == EXACT_V2_WORDINGS["direct_command"]["left"]
        and control_preflight.get("invocation")
        == "exact_cell_plan_plus_run_cell_without_advancing_frozen_queue"
        and control_preflight.get("evidence_sha256")
        == {
            "reset_attestation": "162c5df2358a85de72de97d5c6b88ecbc18ed4ac69ccda30467c397513e493c6",
            "settle_stability": "7de76494434d2cc5326b9424ef2b83436e79143032e79730503118eddaf1e830",
            "fixture_match": "5f3b39d6d76921fa8bda8601dc232afe98128d4da2b792376fd21795ff381545",
            "explicit_no_inference_termination": "6c8c0e2b7690a7ac07f329f1fa883b6ed4caa97cbf5e1653f6a09f0407a5ccb1",
        }
        and close(control_preflight.get("max_position_error_m", {}).get("banana"), 0.00024374574422836304)
        and close(control_preflight.get("max_position_error_m", {}).get("bowl"), 0.000021666288375854492)
        and close(control_preflight.get("max_position_error_m", {}).get("rubiks_cube"), 0.0027577579021453857)
        and all(
            item.get("positions_match") is True
            and item.get("neutral_relation_at_reset") is True
            and item.get("passed") is True
            for item in preflight_layouts.values()
        ),
        "Nano reset preflight passes both exact physical layouts with hash-bound 3 mm evidence",
        checks,
    )
    require(
        nano_reset_preflight.get("accounting_after_preflight")
        == {
            "total_prior_model_sampler_requests_retained_as_infrastructure_evidence": 39,
            "completed_valid_behavioral_cells": 0,
            "behavioral_denominator": 0,
        }
        and nano_reset_preflight.get("resume_authority")
        == {
            "status": "behavior_ready_under_same_runtime_identity",
            "required_runtime_identity_sha256": "2c5e314a62926a3d3c9b84fa73cff634c378b6140d390cbef6407ac9c633e64e",
            "exact_next_behavioral_cell": "v3b001:nano:seed9400:position_mirrored:right",
            "prompt": EXACT_V2_WORDINGS["direct_command"]["right"],
        },
        "Nano reset preflight preserves zero behavioral evidence and authorizes only the exact next cell",
        checks,
    )
    require(
        nano_live.get("passed_reset_preflight")
        == {
            "artifact_path": REQUIRED["nano_mirror_reset_preflight"],
            "artifact_sha256": sha256(paths["nano_mirror_reset_preflight"]),
            "status": "passed_behavior_ready_under_same_runtime_identity",
            "layouts_passed": [
                "v3b001:nano:seed9400:position_mirrored:right",
                "v3b001:nano:seed9400:control:left",
            ],
            "runner_pre_action_reset_calls": 2,
            "physical_reset_calls": 1,
            "settle_gate_runs": 1,
            "duplicate_second_reset_idempotent": True,
            "episode_length_buf_transition": "[75] -> [0]",
            "model_request_count_during_gate": 0,
            "model_sample_count": 0,
            "behavioral_action_count": 0,
            "behavioral_denominator": 0,
        },
        "V3 continuation records the passed preflight without adding behavioral evidence",
        checks,
    )
    require(
        sha256(paths["nano_mirror_live_snapshot"])
        == "bfd8f0e095fc83dd2d09ba254c0860d8c2d8bbdf02205173369c50e56013e52c"
        and nano_live_snapshot.get("schema_version")
        == "vla-wam-shared-v3b-nano-live-queue-snapshot-v1"
        and nano_live_snapshot.get("status")
        == "coherent_three_complete_seed_block_snapshot_queue_may_be_ahead"
        and nano_live_snapshot.get("raw_root")
        == "/data/users/ali/vla_wam/raw/v3b/cosmos3_nano_policy_droid/mirror/behavioral_attempt11"
        and nano_live_snapshot.get("runtime_identity")
        == {
            "runtime_identity_sha256": "2c5e314a62926a3d3c9b84fa73cff634c378b6140d390cbef6407ac9c633e64e",
            "manifest_path": "/data/users/ali/vla_wam/raw/v3b/cosmos3_nano_policy_droid/mirror/runtime_attempt08/runtime_identity.json",
            "manifest_file_sha256": "a609300c4cbb826115692db8fecac7e14d7578e0ded00f63c06f5153619c2136",
            "source_commit": "39f19b57d5f5e68d0f03e4e4c7569350cb9a467f",
        },
        "Nano live queue snapshot is byte-frozen under runtime-attempt-08",
        checks,
    )
    require(
        nano_live_snapshot.get("snapshot_boundary")
        == {
            "complete_seed_blocks": [9400, 9401, 9402],
            "last_fully_compiled_seed_block": 9402,
            "committed_behavioral_cells": 12,
            "block_completeness_rule": "Each included seed has control LEFT, control RIGHT, position-mirrored LEFT, and position-mirrored RIGHT compiled as valid raw_episode.jsonl rows.",
            "queue_may_be_ahead_of_committed_snapshot": True,
            "later_cells_policy": "Cells from seed 9403 onward are intentionally absent even if the live queue completed them before this artifact was committed. They require a later coherent snapshot.",
            "global_queue_continues_under_same_runtime_identity": True,
        }
        and nano_live_snapshot.get("accounting")
        == {
            "valid_behavioral_cells": 12,
            "behavioral_denominator": 12,
            "correct": 10,
            "release_failed": 1,
            "transport_failed": 1,
            "other_failure_taxonomy_count": 0,
            "valid_model_sampler_requests": 90,
            "prior_infrastructure_model_sampler_requests": 39,
            "valid_failures_retained_in_denominator": True,
        },
        "Nano live snapshot freezes exactly three complete blocks and retains both valid failures",
        checks,
    )
    require(
        nano_live_snapshot.get("condition_summary")
        == {
            "control_left": {
                "correct": 2,
                "episodes": 3,
                "failure_taxonomy": {"correct": 2, "transport_failed": 1},
            },
            "control_right": {
                "correct": 2,
                "episodes": 3,
                "failure_taxonomy": {"correct": 2, "release_failed": 1},
            },
            "position_mirrored_left": {
                "correct": 3,
                "episodes": 3,
                "failure_taxonomy": {"correct": 3},
            },
            "position_mirrored_right": {
                "correct": 3,
                "episodes": 3,
                "failure_taxonomy": {"correct": 3},
            },
        }
        and nano_live_snapshot.get("analysis_boundary", {}).get(
            "primary_full_sample_offset_seed_blocks"
        )
        == [9400, 9401, 9402]
        and nano_live_snapshot.get("analysis_boundary", {}).get(
            "all_four_cells_correct_margin_subset_seed_blocks"
        )
        == [9400]
        and nano_live_snapshot.get("analysis_boundary", {}).get(
            "all_four_cells_correct_margin_subset_n"
        )
        == 1,
        "Nano snapshot condition counts and complete-case margin boundary are descriptive and exact",
        checks,
    )
    snapshot_cells = nano_live_snapshot.get("cells", [])
    expected_snapshot_ids = {
        f"v3b001:nano:seed{seed}:{arm}:{relation}"
        for seed in (9400, 9401, 9402)
        for arm in ("control", "position_mirrored")
        for relation in ("left", "right")
    }
    snapshot_by_id = {cell.get("cell_id"): cell for cell in snapshot_cells}
    snapshot_failures = {
        cell_id: cell.get("failure_taxonomy")
        for cell_id, cell in snapshot_by_id.items()
        if not cell.get("requested_success")
    }
    require(
        len(snapshot_cells) == 12
        and set(snapshot_by_id) == expected_snapshot_ids
        and snapshot_failures
        == {
            "v3b001:nano:seed9401:control:right": "release_failed",
            "v3b001:nano:seed9402:control:left": "transport_failed",
        }
        and sum(bool(cell.get("requested_success")) for cell in snapshot_cells) == 10
        and sum(cell.get("model_sampler_requests", -1000) for cell in snapshot_cells) == 90
        and all(
            cell.get("simulator_launch_environment") == "OMNI_KIT_ACCEPT_EULA=YES"
            and cell.get("prompt")
            == EXACT_V2_WORDINGS["direct_command"][cell.get("relation")]
            and cell.get("environment_seed") in {9400, 9401, 9402}
            and cell.get("arm") in {"control", "position_mirrored"}
            and close(
                abs(cell.get("signed_final_lateral_offset_m")),
                cell.get("final_requested_signed_margin_m"),
            )
            and (cell.get("signed_final_lateral_offset_m") > 0)
            == (cell.get("relation") == "left")
            and all(
                is_sha256(cell.get(key, {}).get("sha256"))
                and cell.get(key, {}).get("bytes", 0) > 0
                for key in ("raw_episode", "batch_manifest", "viewport_video")
            )
            and cell.get("viewport_video", {}).get("relative_path", "").endswith(".mp4")
            for cell in snapshot_cells
        ),
        "Nano snapshot retains exact prompts, EULA launch state, outcomes, offsets, margins, manifests, and videos",
        checks,
    )
    require(
        nano_live.get("committed_live_queue_snapshot")
        == {
            "artifact_path": REQUIRED["nano_mirror_live_snapshot"],
            "artifact_sha256": sha256(paths["nano_mirror_live_snapshot"]),
            "snapshot_at_utc": "2026-08-06T07:15:17Z",
            "complete_seed_blocks": [9400, 9401, 9402],
            "last_fully_compiled_seed_block": 9402,
            "valid_behavioral_cells": 12,
            "behavioral_denominator": 12,
            "correct": 10,
            "failure_taxonomy": {
                "correct": 10,
                "release_failed": 1,
                "transport_failed": 1,
            },
            "queue_may_be_ahead": True,
        },
        "V3 continuation binds the coherent Nano queue snapshot and warns live progress may be ahead",
        checks,
    )
    ledger_entries = nano_live_ledger.get("entries", [])
    invalid_behavior = next(
        (entry for entry in ledger_entries if entry.get("attempt_id") == "behavioral_attempt02"),
        {},
    )
    invalidated_compiled = next(
        (
            entry
            for entry in ledger_entries
            if entry.get("attempt_id")
            == "behavioral_attempt03_position_mirrored_right"
        ),
        {},
    )
    registration_invalidated_compiled = next(
        (
            entry
            for entry in ledger_entries
            if entry.get("attempt_id")
            == "behavioral_attempt05_position_mirrored_right"
        ),
        {},
    )
    cache_failures = {
        entry.get("attempt_id"): entry
        for entry in ledger_entries
        if entry.get("attempt_id") in {
            "behavioral_attempt08_position_mirrored_right",
            "behavioral_attempt09_position_mirrored_right",
        }
    }
    ordinal_invalidated_compiled = next(
        (entry for entry in ledger_entries if entry.get("attempt_id") == "behavioral_attempt10_position_mirrored_right"),
        {},
    )
    ordinal_gate_failure = next(
        (entry for entry in ledger_entries if entry.get("attempt_id") == "behavioral_attempt10_control_left"),
        {},
    )
    pre_request_failures = {
        entry.get("attempt_id"): entry
        for entry in ledger_entries
        if entry.get("attempt_id")
        in {
            "behavioral_attempt03_control_left",
            "behavioral_attempt04_control_left",
            "behavioral_attempt05_control_left",
            "behavioral_attempt06_control_left",
            "behavioral_attempt07_control_left_split_rtx",
        }
    }
    registration_diagnostic = next(
        (
            entry
            for entry in ledger_entries
            if entry.get("attempt_id") == "diagnostic_server_unloaded_attempt01"
        ),
        {},
    )
    passed_preflight_entry = next(
        (
            entry
            for entry in ledger_entries
            if entry.get("attempt_id") == "reset_preflight_runtime_attempt08"
        ),
        {},
    )
    require(
        nano_live_ledger.get("behavioral_denominator_excludes_all_entries") is True
        and nano_live_ledger.get("behavioral_denominator") == 12
        and nano_live_ledger.get("behavioral_denominator_source")
        == REQUIRED["nano_mirror_live_snapshot"]
        and nano_live_ledger.get("completed_valid_behavioral_cells") == 12
        and nano_live_ledger.get("complete_behavioral_attempts_invalidated_before_analysis") == 4
        and nano_live_ledger.get("total_model_requests_retained_as_infrastructure_evidence") == 39
        and nano_live_ledger.get("valid_model_sampler_requests_in_committed_snapshot") == 90
        and nano_live_ledger.get("request_count_semantics")
        == "Counts model-sampler requests only; zero-sampling transport-handshake packets are not model requests."
        and nano_live_ledger.get("prospective_reset_semantics_correction") == {
            "path": REQUIRED["nano_mirror_reset_correction"],
            "sha256": sha256(paths["nano_mirror_reset_correction"]),
            "status": "prospective_operational_correction_before_any_valid_phase_b_behavior",
        }
        and nano_live_ledger.get("passed_reset_preflight")
        == {
            "path": REQUIRED["nano_mirror_reset_preflight"],
            "sha256": sha256(paths["nano_mirror_reset_preflight"]),
            "status": "passed_behavior_ready_under_same_runtime_identity",
        }
        and nano_live_ledger.get("committed_live_queue_snapshot")
        == {
            "path": REQUIRED["nano_mirror_live_snapshot"],
            "sha256": sha256(paths["nano_mirror_live_snapshot"]),
            "snapshot_at_utc": "2026-08-06T07:15:17Z",
            "complete_seed_blocks": [9400, 9401, 9402],
            "completed_valid_behavioral_cells": 12,
            "queue_may_be_ahead": True,
        }
        and nano_live_ledger.get("eula_acceptance", {}).get("user_authorized") is True
        and nano_live_ledger.get("eula_acceptance", {}).get("environment")
        == "OMNI_KIT_ACCEPT_EULA=YES"
        and invalid_behavior.get("model_requests") == 15
        and invalid_behavior.get("behavioral_actions_executed") == 450
        and invalid_behavior.get("retained_state_count") == 451
        and invalid_behavior.get("retained_action_chunks") == 15
        and invalid_behavior.get("retained_decoded_futures") == 15
        and invalid_behavior.get("denominator_eligible") is False
        and invalid_behavior.get("disposition")
        == "preserved_complete_behavior_infrastructure_invalid"
        and invalidated_compiled.get("model_requests") == 8
        and invalidated_compiled.get("behavioral_actions_executed") == 233
        and invalidated_compiled.get("denominator_eligible") is False
        and invalidated_compiled.get("disposition")
        == "compiled_record_invalidated_before_analysis"
        and registration_invalidated_compiled.get("model_requests") == 8
        and registration_invalidated_compiled.get("behavioral_actions_executed")
        == 230
        and registration_invalidated_compiled.get("denominator_eligible") is False
        and registration_invalidated_compiled.get("disposition")
        == "compiled_record_invalidated_before_analysis"
        and set(cache_failures) == {
            "behavioral_attempt08_position_mirrored_right",
            "behavioral_attempt09_position_mirrored_right",
        }
        and sum(entry.get("model_requests", 0) for entry in cache_failures.values()) == 2
        and cache_failures["behavioral_attempt08_position_mirrored_right"].get("bridge_failure", {}).get("sha256")
        == "247d772c6895aa8fc69c7908bea11514b78c0643e29e7ed5ab020beb566dbc86"
        and cache_failures["behavioral_attempt09_position_mirrored_right"].get("bridge_failure", {}).get("sha256")
        == "ff2159bf3b5bec3f12d9a73eb6697ae2e5d1d52dfb7a28ea31b6251cc1a609ef"
        and ordinal_invalidated_compiled.get("model_requests") == 6
        and ordinal_invalidated_compiled.get("behavioral_actions_executed") == 169
        and ordinal_invalidated_compiled.get("denominator_eligible") is False
        and ordinal_invalidated_compiled.get("raw_episode_jsonl", {}).get("sha256")
        == "a51380c97cc6426332e3c8915d2950fb7785adabd13e5cab44ec34311141ce27"
        and ordinal_invalidated_compiled.get("viewport_video", {}).get("sha256")
        == "92e26891c9ec1871d1c167d3914f22aa3fe90fc6fecc6af95a55684b8de2e1db"
        and ordinal_invalidated_compiled.get("batch_manifest", {}).get("sha256")
        == "bca5342cbe6ccbf8a88d8c68b9900e7a1b6a79f1d39ececd6100ba667e244962"
        and ordinal_gate_failure.get("model_requests") == 0
        and ordinal_gate_failure.get("fixture_match", {}).get("sha256")
        == "573f1df5892d1137147945a2d8f531af9d59d1d60c744c717f396f0b17e4e276"
        and ordinal_gate_failure.get("settle_stability", {}).get("sha256")
        == "588297f9d058bb9997346f0c0c71e6420381ad5d9fae70687854adc4f76d4ef9"
        and ordinal_gate_failure.get("bridge_failure", {}).get("sha256")
        == "1f921e18c520dc6daa2d415cfe39f0f70903adc5a2b8c90e7c4e9f72db998519"
        and set(pre_request_failures)
        == {
            "behavioral_attempt03_control_left",
            "behavioral_attempt04_control_left",
            "behavioral_attempt05_control_left",
            "behavioral_attempt06_control_left",
            "behavioral_attempt07_control_left_split_rtx",
        }
        and all(
            entry.get("model_requests") == 0
            and entry.get("behavioral_actions_executed") == 0
            and entry.get("denominator_eligible") is False
            for entry in pre_request_failures.values()
        )
        and all(
            pre_request_failures[attempt].get("rubiks_cube_max_position_error_m")
            == 0.0034789443
            for attempt in {
                "behavioral_attempt05_control_left",
                "behavioral_attempt06_control_left",
                "behavioral_attempt07_control_left_split_rtx",
            }
        )
        and registration_diagnostic.get("model_requests") == 0
        and registration_diagnostic.get("behavioral_episodes") == 0
        and registration_diagnostic.get("registered_layouts") == 4
        and registration_diagnostic.get("resets_per_layout") == 2
        and registration_diagnostic.get("passed_all_released_layouts") is True
        and registration_diagnostic.get("report", {}).get("sha256")
        == "b15b3249647dbf468e745727e2d7677df9c74d919b399a3148be17312951ca04"
        and passed_preflight_entry.get("stage")
        == "zero_sampling_exact_bridge_preflight"
        and passed_preflight_entry.get("model_requests") == 0
        and passed_preflight_entry.get("model_samples") == 0
        and passed_preflight_entry.get("behavioral_actions_executed") == 0
        and passed_preflight_entry.get("behavioral_episodes") == 0
        and passed_preflight_entry.get("denominator_eligible") is False
        and passed_preflight_entry.get("disposition")
        == "passed_model_blind_preflight_outside_behavioral_denominator"
        and passed_preflight_entry.get("runtime_identity", {}).get(
            "runtime_identity_sha256"
        )
        == "2c5e314a62926a3d3c9b84fa73cff634c378b6140d390cbef6407ac9c633e64e"
        and passed_preflight_entry.get("runtime_identity", {}).get("sha256")
        == "a609300c4cbb826115692db8fecac7e14d7578e0ded00f63c06f5153619c2136"
        and passed_preflight_entry.get("reset_contract")
        == {
            "runner_pre_action_reset_calls": 2,
            "physical_reset_calls": 1,
            "settle_gate_runs": 1,
            "duplicate_second_reset_idempotent": True,
            "episode_length_buf_before_reset": [75],
            "episode_length_buf_after_reset": [0],
            "model_request_count_during_gate": 0,
        }
        and nano_live_ledger.get("repair", {}).get("export_before_isaac_close") is True
        and nano_live_ledger.get("repair", {}).get(
            "absolute_attempt_local_robolab_output_folder"
        )
        is True
        and nano_live_ledger.get("repair", {}).get("released_position_frame")
        == "robot"
        and nano_live_ledger.get("repair", {}).get("live_position_comparison_frame")
        == "robot"
        and nano_live_ledger.get("repair", {}).get("pre_fix_compiled_cell_invalidated")
        is True
        and nano_live_ledger.get("repair", {}).get(
            "durable_bridge_failure_before_isaac_close"
        )
        is True
        and nano_live_ledger.get("repair", {}).get(
            "queue_requires_export_before_compilation"
        )
        is True
        and nano_live_ledger.get("repair", {}).get(
            "full_four_wrapper_registration_context_retained"
        )
        is True
        and nano_live_ledger.get("repair", {}).get("registration_context_hypothesis")
        == "falsified_as_the_sufficient_root_cause"
        and nano_live_ledger.get("repair", {}).get("runner_pre_action_reset_calls") == 2
        and nano_live_ledger.get("repair", {}).get("physical_reset_calls") == 1
        and nano_live_ledger.get("repair", {}).get("settle_gate_runs") == 1
        and nano_live_ledger.get("repair", {}).get("duplicate_second_reset_idempotent") is True
        and nano_live_ledger.get("repair", {}).get("fresh_runtime_identity_required") is False
        and nano_live_ledger.get("repair", {}).get("live_bound_runtime_identity_sha256")
        == "2c5e314a62926a3d3c9b84fa73cff634c378b6140d390cbef6407ac9c633e64e"
        and nano_live_ledger.get("repair", {}).get(
            "zero_sampling_exact_bridge_preflight_passed"
        )
        is True
        and nano_live_ledger.get("repair", {}).get(
            "behavior_ready_under_same_runtime_identity"
        )
        is True
        and nano_live_ledger.get("repair", {}).get(
            "preflight_exact_next_cell_completed_valid_in_snapshot"
        )
        == "v3b001:nano:seed9400:position_mirrored:right"
        and nano_live_ledger.get("repair", {}).get(
            "committed_snapshot_last_fully_compiled_seed_block"
        )
        == 9402
        and nano_live_ledger.get("repair", {}).get(
            "live_queue_may_be_ahead_of_committed_snapshot"
        )
        is True
        and nano_live_ledger.get("repair", {}).get(
            "fixture_mismatch_evidence_persisted_before_raise"
        )
        is True
        and nano_live_ledger.get("execution_topology", {})
        .get("simulator", {})
        .get("pod_uid")
        == "30b51054-f277-480d-831d-337581e1cc49"
        and nano_live_ledger.get("execution_topology", {})
        .get("split_policy_server", {})
        .get("pod_uid")
        == "d5b0405a-a9b1-4baa-a802-d5171e03c228"
        and nano_live_ledger.get("execution_topology", {})
        .get("simulator", {})
        .get("owner_label")
        == "ali"
        and nano_live_ledger.get("execution_topology", {})
        .get("split_policy_server", {})
        .get("owner_label")
        == "ali",
        "Nano live ledger preserves the excluded smoke and EULA-authorized repair boundary",
        checks,
    )
    require(
        nano_live.get("preflight_exact_next_cell")
        == {
            "cell_id": nano_cells[0].get("cell_id"),
            "environment_seed": nano_cells[0].get("environment_seed"),
            "arm": nano_cells[0].get("arm"),
            "relation": nano_cells[0].get("relation"),
            "prompt": nano_cells[0].get("prompt"),
            "completion_status": "completed_valid_in_committed_live_queue_snapshot",
        }
        == {
            "cell_id": "v3b001:nano:seed9400:position_mirrored:right",
            "environment_seed": 9400,
            "arm": "position_mirrored",
            "relation": "right",
            "prompt": EXACT_V2_WORDINGS["direct_command"]["right"],
            "completion_status": "completed_valid_in_committed_live_queue_snapshot",
        },
        "V3 continuation preserves the exact first released Nano cell as completed snapshot provenance",
        checks,
    )
    nano_live_source_keys = {
        "runtime_adapter": "nano_mirror_runtime_adapter",
        "compiler": "nano_mirror_compiler",
        "live_support": "nano_mirror_live_support",
        "live_client": "nano_mirror_live_client",
        "live_server": "nano_mirror_live_server",
        "live_bridge": "nano_mirror_live_bridge",
        "queue_launcher": "nano_mirror_queue_launcher",
        "runtime_test": "nano_mirror_runtime_test",
        "live_queue_test": "nano_mirror_live_queue_test",
    }
    require(
        all(
            nano_live.get("source_bindings", {}).get(label)
            == {"path": REQUIRED[path_key], "sha256": sha256(paths[path_key])}
            and REQUIRED[path_key] in continuation.get("authoritative_files", [])
            for label, path_key in nano_live_source_keys.items()
        ),
        "V3 continuation hash-binds the complete Nano live stack and both regression suites",
        checks,
    )
    require(
        nano_live.get("model_blind_reset_gate")
        == {
            "runner_pre_action_reset_calls": 2,
            "physical_reset_calls": 1,
            "settle_gate_runs": 1,
            "duplicate_second_reset_idempotent": True,
            "call1_evidence": "provisional_not_persisted",
            "call2_evidence": "final_persisted_attestation",
            "settle_steps": 60,
            "stability_window_steps": 15,
            "episode_length_buf_before_reset": [75],
            "episode_length_buf_after_reset": [0],
            "model_request_count_before_attestation": 0,
            "frozen_flags_after_duplicate_reset": False,
            "linear_speed_threshold_m_s": 0.02,
            "angular_speed_threshold_rad_s": 0.2,
            "counter_reset_required_before_model_request": True,
        }
        and nano_live.get("execution_contract")
        == {
            "global_released_order_required": True,
            "retained_output_overwrite_prohibited": True,
            "partial_attempts_outside_behavioral_denominator": True,
            "static_prompt_only": True,
            "omniverse_eula_acceptance": "OMNI_KIT_ACCEPT_EULA=YES",
            "thermal_guard": "not_used",
            "full_calibration_registration_context_retained": True,
            "split_policy_server_preferred_for_simulator_load_isolation": True,
            "persistent_caches": "ali_owned_pvc",
            "temporary_directory": "pod_local_tmpdir",
        },
        "V3 continuation freezes the Nano settle, ordering, EULA, and evidence-retention boundary",
        checks,
    )
    nano_queue_source = paths["nano_mirror_queue_launcher"].read_text(encoding="utf-8")
    nano_bridge_source = paths["nano_mirror_live_bridge"].read_text(encoding="utf-8")
    nano_server_source = paths["nano_mirror_live_server"].read_text(encoding="utf-8")
    nano_compiler_source = paths["nano_mirror_compiler"].read_text(encoding="utf-8")
    require(
        '"OMNI_KIT_ACCEPT_EULA": "YES"' in nano_queue_source
        and "run-cell preserves released global order" in nano_queue_source
        and "retained partial attempt is preserved outside the denominator" in nano_queue_source
        and "episode_length_buf_before_reset" in nano_bridge_source
        and '"position_robot_xyz_m"' in nano_bridge_source
        and '"position_frame": "robot"' in nano_bridge_source
        and "TASK_REGISTRATION_ORDER" in nano_bridge_source
        and "for key in TASK_REGISTRATION_ORDER" in nano_bridge_source
        and "if self._runner_pre_action_reset_calls == 2:" in nano_bridge_source
        and "self._runner_pre_action_reset_calls != 2" in nano_bridge_source
        and "self._physical_reset_calls != 1" in nano_bridge_source
        and "self._settle_gate_runs != 1" in nano_bridge_source
        and "released fixture is not expressed in the robot frame" in nano_bridge_source
        and 'bridge_failure_path.write_text' in nano_bridge_source
        and 'bridge_failure = attempt / "bridge_failure.json"' in nano_queue_source
        and "verify_live_runtime_identity" in nano_server_source
        and "validate_pinned_server_cli" in nano_server_source
        and "verify_live_runtime_identity" in nano_compiler_source,
        "Nano live implementation fails closed on queue order, reset attestation, runtime identity, and retained attempts",
        checks,
    )
    require(
        pi05_mirror_amendment.get("schema_version")
        == "vla-wam-shared-v3b-pi05-mirror-amendment-v1"
        and pi05_mirror_amendment.get("amendment_id") == "V3-B002"
        and pi05_mirror_amendment.get("status")
        == "frozen_before_any_v3b002_model_request_or_behavioral_episode"
        and pi05_mirror_amendment.get("release_boundary", {}).get(
            "model_requests_before_registration"
        )
        == 0
        and pi05_mirror_amendment.get("release_boundary", {}).get(
            "behavioral_episodes_before_registration"
        )
        == 0
        and pi05_mirror_amendment.get("release_boundary", {}).get(
            "behavioral_release"
        )
        is False,
        "pi0.5 V3-B002 predictions were frozen before inference and do not release behavior",
        checks,
    )
    require(
        pi05_mirror_amendment.get("design", {}).get("matched_seeds")
        == exact_range(9400, 9426)
        and pi05_mirror_amendment.get("design", {}).get("cells_per_seed") == 4
        and pi05_mirror_amendment.get("design", {}).get(
            "behavioral_episode_ceiling"
        )
        == 108
        and pi05_mirror_amendment.get("analysis_plan", {}).get(
            "continuous_reporting", {}
        ).get("bootstrap_replicates")
        == 20000
        and pi05_mirror_amendment.get("analysis_plan", {}).get(
            "continuous_reporting", {}
        ).get("bootstrap_master_seed")
        == 3104159,
        "pi0.5 V3-B002 freezes the 27-seed four-cell design and exact uncertainty plan",
        checks,
    )
    require(
        len(pi05_mirror_cells) == 108
        and pi05_mirror_manifest.get("status")
        == "hash_bound_registered_not_behaviorally_released"
        and pi05_mirror_manifest.get("counts", {}).get(
            "registered_behavioral_cells"
        )
        == 108
        and pi05_mirror_manifest.get("files", {}).get("amendment", {}).get(
            "sha256"
        )
        == sha256(paths["pi05_mirror_amendment"])
        and pi05_mirror_manifest.get("files", {}).get("cells", {}).get("sha256")
        == sha256(paths["pi05_mirror_cells"]),
        "pi0.5 V3-B002 registration manifest binds all 108 unreleased cells",
        checks,
    )
    nano_cells_by_id = {row.get("cell_id"): row for row in nano_cells}
    pi05_conditions_by_seed: dict[int, set[tuple[str, str]]] = {}
    exact_pi05_source_reuse = True
    for row in pi05_mirror_cells:
        seed = row.get("environment_seed")
        source = nano_cells_by_id.get(row.get("source_v3b001_cell_id"))
        if type(seed) is not int or not source:
            exact_pi05_source_reuse = False
            continue
        pi05_conditions_by_seed.setdefault(seed, set()).add(
            (row.get("arm"), row.get("relation"))
        )
        exact_pi05_source_reuse = exact_pi05_source_reuse and all(
            row.get(key) == source.get(key)
            for key in (
                "environment_seed",
                "arm",
                "relation",
                "fixture_sha256",
                "execution_order_index_within_seed",
            )
        )
    expected_pi05_conditions = {
        ("control", "left"),
        ("control", "right"),
        ("position_mirrored", "left"),
        ("position_mirrored", "right"),
    }
    require(
        exact_pi05_source_reuse
        and sorted(pi05_conditions_by_seed) == exact_range(9400, 9426)
        and all(
            conditions == expected_pi05_conditions
            for conditions in pi05_conditions_by_seed.values()
        ),
        "pi0.5 V3-B002 reuses Nano seeds, fixtures, and within-seed order exactly",
        checks,
    )
    expected_failure_tables = {
        "pi05_current_stack_droid": ([5, 6, 11, 5, 0], [24, 0, 3, 0, 0]),
        "dreamzero_droid_action_cfg": ([3, 10, 14, 0, 0], [17, 10, 0, 0, 0]),
        "cosmos3_edge_policy_droid": ([18, 1, 6, 2, 0], [25, 1, 1, 0, 0]),
    }
    observed_failure_tables = {
        row.get("model_id"): (
            row.get("table_2x5", {}).get("rows", {}).get("left"),
            row.get("table_2x5", {}).get("rows", {}).get("right"),
        )
        for row in pi05_failure_split_report.get("results", [])
    }
    failure_outputs = {
        entry.get("path"): entry.get("sha256")
        for entry in pi05_failure_split_manifest.get("outputs", [])
    }
    require(
        len(pi05_failure_split_episodes) == 162
        and pi05_failure_split_report.get("status")
        == "complete_retrospective_analysis_no_new_inference"
        and observed_failure_tables == expected_failure_tables
        and failure_outputs.get(
            "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/analysis/failure_mode_split_episodes.jsonl"
        )
        == sha256(paths["pi05_failure_split_episodes"])
        and failure_outputs.get(
            "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/analysis/failure_mode_split_report.json"
        )
        == sha256(paths["pi05_failure_split_report"]),
        "V3-B002 failure-mode split binds 162 existing episodes and the three exact 2x5 tables",
        checks,
    )
    dream_boundary = dreamzero_mirror_amendment.get("release_boundary", {})
    require(
        dreamzero_mirror_amendment.get("schema_version")
        == "vla-wam-shared-v3b-dreamzero-mirror-amendment-v1"
        and dreamzero_mirror_amendment.get("amendment_id") == "V3-B003"
        and dreamzero_mirror_amendment.get("status")
        == "frozen_before_any_v3b003_model_request_or_behavioral_episode"
        and dream_boundary.get("model_requests_before_registration") == 0
        and dream_boundary.get("behavioral_episodes_before_registration") == 0
        and dream_boundary.get("behavioral_release") is False
        and dreamzero_mirror_amendment.get("design", {}).get("identity_binding")
        == "V2-A015:dreamzero_action_cfg_s2"
        and dreamzero_mirror_amendment.get("design", {}).get("matched_seeds")
        == exact_range(9400, 9426)
        and dreamzero_mirror_amendment.get("design", {}).get("exact_prompts")
        == EXACT_V2_WORDINGS["direct_command"],
        "DreamZero V3-B003 freezes the disclosed s=2 third-mirror design before any model request",
        checks,
    )
    dream_manifest_files = dreamzero_mirror_manifest.get("files", {})
    require(
        dreamzero_mirror_manifest.get("schema_version")
        == "vla-wam-shared-v3b-dreamzero-mirror-manifest-v1"
        and dreamzero_mirror_manifest.get("status")
        == "hash_bound_registered_not_behaviorally_released"
        and dreamzero_mirror_manifest.get("counts", {}).get("registered_behavioral_cells")
        == 108
        and dream_manifest_files.get("amendment", {}).get("sha256")
        == sha256(paths["dreamzero_mirror_amendment"])
        and dream_manifest_files.get("cells", {}).get("sha256")
        == sha256(paths["dreamzero_mirror_cells"])
        and dream_manifest_files.get("cells", {}).get("row_count") == 108,
        "DreamZero V3-B003 manifest byte/hash-binds all 108 registered unreleased cells",
        checks,
    )
    dream_conditions_by_seed: dict[int, set[tuple[str, str]]] = {}
    exact_dream_source_reuse = len(dreamzero_mirror_cells) == 108
    for row in dreamzero_mirror_cells:
        seed = row.get("environment_seed")
        source = nano_cells_by_id.get(row.get("source_v3b001_cell_id"))
        if type(seed) is not int or not source:
            exact_dream_source_reuse = False
            continue
        dream_conditions_by_seed.setdefault(seed, set()).add(
            (row.get("arm"), row.get("relation"))
        )
        exact_dream_source_reuse = exact_dream_source_reuse and all(
            row.get(key) == source.get(key)
            for key in (
                "environment_seed",
                "arm",
                "relation",
                "fixture_sha256",
                "execution_order_index_within_seed",
            )
        )
        exact_dream_source_reuse = (
            exact_dream_source_reuse
            and row.get("model_id") == "dreamzero_droid_action_cfg"
            and row.get("effective_model_noise_seed") == 1140
            and row.get("prompt")
            == EXACT_V2_WORDINGS["direct_command"][row.get("relation")]
            and row.get("execution_status")
            == "registered_pre_inference_runtime_release_gate_required"
        )
    require(
        exact_dream_source_reuse
        and sorted(dream_conditions_by_seed) == exact_range(9400, 9426)
        and all(
            conditions == expected_pi05_conditions
            for conditions in dream_conditions_by_seed.values()
        ),
        "DreamZero V3-B003 reuses Nano seeds, fixtures, prompts, and order while disclosing constant model-noise seed 1140",
        checks,
    )
    lateral_design = nano_lateral_sweep_amendment.get("design", {})
    lateral_calibration = nano_lateral_sweep_amendment.get(
        "model_blind_numeric_calibration", {}
    )
    lateral_release = nano_lateral_sweep_amendment.get("release_boundary", {})
    require(
        nano_lateral_sweep_amendment.get("schema_version")
        == "vla-wam-shared-v3b-nano-lateral-sweep-amendment-v1"
        and nano_lateral_sweep_amendment.get("amendment_id") == "V3-B004"
        and nano_lateral_sweep_amendment.get("status")
        == "design_frozen_before_model_blind_numeric_calibration_and_before_any_v3b004_model_request"
        and lateral_design.get("factor")
        == "reference_object_initial_lateral_position_y_m"
        and lateral_design.get("matched_seeds") == exact_range(9500, 9514)
        and lateral_design.get("lateral_level_count") == 7
        and lateral_design.get("cells_per_seed") == 14
        and lateral_design.get("registered_behavioral_episode_ceiling_after_calibration")
        == 210
        and lateral_design.get("exact_prompts")
        == EXACT_V2_WORDINGS["direct_command"]
        and lateral_design.get("numeric_lateral_levels_y_m") is None
        and lateral_calibration.get("model_request_count") == 0
        and lateral_calibration.get("behavioral_episode_count") == 0
        and lateral_calibration.get("seven_level_selection_rule", {}).get(
            "minimum_half_range_m"
        )
        == 0.09
        and lateral_release
        == {
            "model_requests_before_registration": 0,
            "behavioral_episodes_before_registration": 0,
            "behavioral_release": False,
            "calibration_release": False,
            "next_command_boundary": "Run only the zero-model-request dense physical calibration. Do not start the Nano server or generate behavioral queue rows until the seven numeric levels and their calibration report are committed and hash-bound.",
        },
        "Nano V3-B004 freezes the seven-level 15-seed dose-response design before numeric calibration or model requests",
        checks,
    )
    lateral_analysis = nano_lateral_sweep_amendment.get("analysis_plan", {})
    require(
        lateral_analysis.get("inferential_unit")
        == "matched seed; the 105 seed-by-level pairs are repeated observations, not independent seeds"
        and "B[i,j] = (-s[i,j,RIGHT]) - s[i,j,LEFT]"
        in lateral_analysis.get("full_sample_primary", {}).get(
            "requested_depth_contrast", ""
        )
        and lateral_analysis.get("full_sample_primary", {}).get(
            "bootstrap_master_seed"
        )
        == 3104161
        and "never mix unmatched successes"
        in lateral_analysis.get("success_conditional_margin", ""),
        "Nano V3-B004 freezes full-sample depth slopes, seed-level inference, and complete-pair margin reporting",
        checks,
    )
    pi05_runtime_source = paths["pi05_mirror_runtime"].read_text(encoding="utf-8")
    pi05_preflight_source = paths["pi05_mirror_preflight"].read_text(encoding="utf-8")
    pi05_bridge_source = paths["pi05_mirror_bridge"].read_text(encoding="utf-8")
    pi05_queue_source = paths["pi05_mirror_queue"].read_text(encoding="utf-8")
    pi05_diagnostics_source = paths["pi05_mirror_diagnostics"].read_text(encoding="utf-8")
    pi05_compiler_source = paths["pi05_mirror_results_compiler"].read_text(encoding="utf-8")
    require(
        "cover every runtime simulator lane exactly once" in pi05_runtime_source
        and '"model_request_count": 0' in pi05_preflight_source
        and "for repeat in range(3)" in pi05_preflight_source
        and "object_grabbed" in pi05_bridge_source
        and "no verified physical contact stream" in pi05_bridge_source
        and "native_process_group_thermal_guard.py" in pi05_queue_source
        and "cells_for_lane" in pi05_queue_source
        and "retained_object_grabbed_boolean_stream" in pi05_diagnostics_source
        and "exact_two_sided_within_seed_control_reflected_label_permutation"
        in pi05_compiler_source
        and "total_permutations = 2 ** len(values)" in pi05_compiler_source,
        "V3-B002 runtime enforces six-lane physical gates, guarded whole-seed queues, retained grasp/contact semantics, and the registered exact statistics",
        checks,
    )
    pi05_result = validate_pi05_v3b002_evidence(paths, pi05_mirror_cells, checks)
    pi05_release = continuation.get("phase_b_releases", {}).get(
        "pi05_position_reflection_v3b002", {}
    )
    if pi05_release:
        require(
            pi05_release.get("registered_behavioral_cell_count") == 108
            and pi05_release.get("completed_behavioral_cell_count") in {0, 108}
            and pi05_release.get("pre_registration_counts")
            == {"model_requests": 0, "behavioral_episodes": 0}
            and pi05_release.get("artifacts", {}).get("manifest", {}).get("sha256")
            == sha256(paths["pi05_mirror_manifest"]),
            "V3 continuation preserves the registered V3-B002 boundary and any recorded completion count",
            checks,
        )
        if pi05_release.get("completed_behavioral_cell_count") == 108:
            final_result = pi05_release.get("final_result", {})
            artifact_refs = pi05_release.get("artifacts", {})
            require(
                pi05_release.get("status")
                == "complete_hash_closed_27_seed_108_episode_result_no_rerun"
                and pi05_release.get("rerun_policy") == "do_not_rerun_any_valid_v3b002_cell"
                and final_result.get("valid_behavioral_cells") == 108
                and final_result.get("matched_seed_count") == 27
                and final_result.get("matched_left_right_pairs") == 54
                and final_result.get("condition_successes")
                == {
                    "control:left": "4/27",
                    "control:right": "25/27",
                    "position_mirrored:left": "25/27",
                    "position_mirrored:right": "9/27",
                }
                and final_result.get("infrastructure")
                == {
                    "excluded_attempt_count": 18,
                    "included_in_behavioral_denominator": False,
                }
                and artifact_refs.get("output_manifest", {}).get("sha256")
                == pi05_result["output_manifest_sha256"]
                and artifact_refs.get("gate_manifest", {}).get("sha256")
                == pi05_result["gate_manifest_sha256"],
                "V3 continuation records the complete no-rerun V3-B002 result and hash roots",
                checks,
            )
            continuation_result_refs = pi05_release.get("results", {})
            if continuation_result_refs:
                require(
                    continuation_result_refs.get("output_manifest_sha256")
                    == pi05_result["output_manifest_sha256"]
                    and continuation_result_refs.get("gate_manifest_sha256")
                    == pi05_result["gate_manifest_sha256"],
                    "V3 continuation hash-binds the completed V3-B002 result and gate manifests",
                    checks,
                )
    phase_b_state = continuation.get("blocked_and_unreleased", {}).get(
        "phase_b_confounds", {}
    )
    released_phase_b = phase_b_state.get("released", {})
    nano_phase_b = released_phase_b.get("nano_v3b001", released_phase_b)
    require(
        nano_phase_b.get("amendment_id") == "V3-B001"
        and nano_phase_b.get("released_behavioral_cells") == 108
        and nano_phase_b.get("completed_behavioral_cells") == 108
        and "27-seed"
        in nano_phase_b.get("release_condition", "")
        and "hash-closed"
        in nano_phase_b.get("release_condition", ""),
        "V3 continuation retains the completed Nano Phase-B evidence boundary",
        checks,
    )
    completed_pi05_phase_b = released_phase_b.get("pi05_v3b002")
    if completed_pi05_phase_b:
        require(
            completed_pi05_phase_b.get("amendment_id") == "V3-B002"
            and completed_pi05_phase_b.get("released_behavioral_cells") == 108
            and completed_pi05_phase_b.get("completed_behavioral_cells") == 108
            and "27-seed" in completed_pi05_phase_b.get("release_condition", "")
            and "hash-closed" in completed_pi05_phase_b.get("release_condition", "")
            and "DreamZero V3-B003 is hash-bound but not behaviorally released"
            in phase_b_state.get("unreleased", ""),
            "V3 continuation marks V3-B002 complete while keeping V3-B003 unreleased",
            checks,
        )
    registered_not_released = phase_b_state.get("registered_not_released", {}).get(
        "dreamzero_v3b003"
    )
    if registered_not_released:
        require(
            registered_not_released.get("amendment_id") == "V3-B003"
            and registered_not_released.get("registered_behavioral_cells") == 108
            and registered_not_released.get("model_requests_after_registration") == 0
            and registered_not_released.get("behavioral_episodes_after_registration") == 0
            and registered_not_released.get("amendment", {}).get("sha256")
            == sha256(paths["dreamzero_mirror_amendment"])
            and registered_not_released.get("cells", {}).get("sha256")
            == sha256(paths["dreamzero_mirror_cells"])
            and registered_not_released.get("manifest", {}).get("sha256")
            == sha256(paths["dreamzero_mirror_manifest"]),
            "V3 continuation preserves the unreleased zero-request DreamZero V3-B003 registration",
            checks,
        )
    nano_lateral_registered = phase_b_state.get("registered_not_released", {}).get(
        "nano_v3b004"
    )
    require(
        nano_lateral_registered.get("amendment_id") == "V3-B004"
        and nano_lateral_registered.get("model_id")
        == "cosmos3_nano_policy_droid"
        and nano_lateral_registered.get("matched_seeds") == 15
        and nano_lateral_registered.get("planned_lateral_levels") == 7
        and nano_lateral_registered.get("planned_behavioral_cells_after_calibration")
        == 210
        and nano_lateral_registered.get("numeric_levels_frozen") is False
        and nano_lateral_registered.get("model_requests_after_registration") == 0
        and nano_lateral_registered.get("behavioral_episodes_after_registration")
        == 0
        and nano_lateral_registered.get("amendment", {}).get("sha256")
        == sha256(paths["nano_lateral_sweep_amendment"]),
        "V3 continuation preserves the zero-request Nano V3-B004 design-only calibration boundary",
        checks,
    )
    inference_authority = continuation.get("next_agent", {}).get(
        "inference_authority", ""
    )
    if "V3-B002 is the exact next experiment" in inference_authority:
        require(
            "Do not rerun any valid Phase-A, V3-A002, or Nano V3-B001 cell"
            in inference_authority
            and "108 registered pi0.5 cells" in inference_authority
            and "only after its new runtime identity" in inference_authority
            and "No other Phase-B, Phase-C, or Phase-D cell is released"
            in inference_authority,
            "V3 continuation preserves the pre-run V3-B002 inference authority if still present",
            checks,
        )

    require(protocol["schema_version"] == "vla-wam-shared-v3-protocol-v1", "protocol schema is frozen", checks)
    require(protocol["study_id"] == "vla_wam_language_steerability_v3", "protocol study identifier is frozen", checks)
    require(protocol["status"] == "frozen_before_any_v3_model_request_or_behavioral_inference", "protocol prohibits v3 inference before freeze", checks)
    relation = protocol["relationship_to_v2"]
    require("after all v2 and V2-A015 results" in relation["disclosure"], "post-result disclosure covers all v2 and V2-A015 outcomes", checks)
    require("never rewritten" in relation["immutability"] and "rerun" in relation["immutability"], "valid v2 evidence is immutable and non-rerunnable", checks)
    require("never pooled" in relation["arena_boundary"], "protocol forbids cross-arena pooling", checks)
    require(protocol["required_artifacts"] == [
        f"{V3}/post_result_power_failure_ablation_amendment.json",
        f"{V3}/droid_direct_registry.json",
        f"{V3}/robotwin_direct_registry.json",
        f"{V3}/four_phrasings_registry.json",
        f"{V3}/stochastic_rollout_registry.json",
        f"{V3}/confound_fixture_calibration_registry.json",
        f"{V3}/phase_a_cells.jsonl",
        f"{V3}/phase_a_cells_manifest.json",
        f"{V3}/failure_taxonomy.json",
        f"{V3}/analysis_plan.json",
    ], "protocol references the complete immutable v3 artifact set", checks)
    common = protocol["common_execution_contract"]
    require(common["instruction_controller"] == "static_episode_prompt" and common["oracle_actions"] == 0 and not common["subtask_coach"] and not common["progress_conditioned_instruction"], "common contract fixes static prompts and prohibits oracle/coach/progress control", checks)
    require(common["required_raw_outputs"] == ["viewport_video", "executed_action_trace", "raw_result_jsonl"], "common contract requires video, actions, and raw JSONL", checks)

    require(droid["schema_version"] == "vla-wam-shared-v3-droid-direct-registry-v1", "DROID registry schema is frozen", checks)
    require(droid["priority"] == DROID_MODELS, "DROID priority order is exact", checks)
    target = droid["target"]
    require(target["exact_matched_pair_count_per_checkpoint"] == 30 and target["seed_range"] == [8300, 8329] and target["directions_per_pair"] == ["left", "right"], "DROID target is thirty exact LEFT/RIGHT pairs on seeds 8300-8329", checks)
    prompts = droid["direct_prompts"]
    require(prompts == {
        "left": "Put the Rubik's cube to the left of the bowl.",
        "right": "Put the Rubik's cube to the right of the bowl.",
        "predicate": "official release-inside-the-45-degree-cone requested relation termination",
    }, "DROID static direct prompts and predicate are unchanged", checks)
    require(droid["required_evidence"][:3] == ["viewport_video", "executed_action_trace", "raw_result_jsonl"], "DROID registry requires video/actions/JSONL", checks)
    pi0 = droid["checkpoint_rules"]["pi0_fast_droid_vla"]
    require(pi0["preserved_historical_seeds"] == exact_range(8300, 8309), "pi0-FAST preserves historical seeds 8300-8309", checks)
    require(pi0["blocked_new_seeds"] == exact_range(8310, 8329), "pi0-FAST blocks seeds 8310-8329", checks)
    require("9e46d3aea26417bfb564227734b95d010aa827e5" in pi0["blocker"] and "11142d4319e44401e0464866bb5fedf7ec8a8927" in pi0["blocker"] and "V2-A008" in pi0["blocker"], "pi0-FAST blocker pins missing commits and failed V2-A008 sensitivity", checks)
    other = droid["checkpoint_rules"]["other_checkpoints"]
    require(other["models"] == DROID_MODELS[1:], "other DROID checkpoint set is exact", checks)
    require(other["preserved_candidate_seeds"] == [8300, 8301, 8302] and other["new_addition_seeds"] == exact_range(8303, 8329), "other DROID additions use 8303-8329 with conditional preserved seeds", checks)
    require("exactly" in droid["runtime_reuse_rule"] and "never rerun" in droid["runtime_reuse_rule"], "DROID runtime reuse is identity-pinned and non-rerunnable", checks)

    require(robotwin["schema_version"] == "vla-wam-shared-v3-robotwin-direct-registry-v1", "RoboTwin registry schema is frozen", checks)
    require(robotwin["models"] == ROBOTWIN_MODELS, "RoboTwin core model set is exact", checks)
    require(robotwin["scene_pairs"] == [3, 4, 5, 6, 7, 8, 9], "RoboTwin core scenes are pairs03-09", checks)
    sampling = robotwin["sampling"]
    require(sampling["replicates"] == list(range(10)) and "8400 + pair_number" in sampling["seed_rule"] and "+ 100*r" in sampling["seed_rule"], "RoboTwin replicate seeds use the required r=0 and r=1..9 formula", checks)
    require("preserved" in sampling["r0_policy"] and "never rerun" in sampling["r0_policy"], "RoboTwin r=0 evidence is preserved and non-rerunnable", checks)
    accounting = robotwin["episode_accounting"]
    require(accounting == {"existing_r0_episodes": 42, "new_episodes_per_model": 126, "new_episodes_total": 378, "formula": "3 models * 7 pairs * 9 new replicates * 2 directions"}, "RoboTwin accounting fixes 378 new episodes", checks)
    require("same anchor reset" in robotwin["pairing"] and "Only requested direction" in robotwin["pairing"], "RoboTwin pairing keeps both directions in identical anchors", checks)
    require(robotwin["required_evidence"][:3] == ["simulator_viewport_video", "executed_action_trace", "raw_result_jsonl"], "RoboTwin registry requires video/actions/JSONL", checks)

    require(taxonomy["schema_version"] == "vla-wam-shared-v3-failure-taxonomy-v1", "taxonomy schema is frozen", checks)
    require(taxonomy["required_legacy_field"] == "frozen_failure_stage", "taxonomy preserves the frozen failure stage", checks)
    require(taxonomy["primary_precedence"] == ["correct", "pick_failed", "wrong_side", "release_failed", "transport_failed"], "taxonomy precedence is exact", checks)
    classes = taxonomy["classes"]
    require("frozen requested relation-and-detached-release success" in classes["correct"], "correct class defers to frozen success", checks)
    require("picked object" in classes["wrong_side"] and "three consecutive" in classes["wrong_side"] and "opposite" in classes["wrong_side"], "wrong_side requires pickup and sustained opposite region", checks)
    require("picked object" in classes["release_failed"] and "three consecutive" in classes["release_failed"] and "final_detached_release is false" in classes["release_failed"] and "not an inference" in classes["release_failed"], "release_failed requires pickup, sustained requested region, and independent failed release", checks)
    require("+3 cm" in classes["pick_failed"] and "three consecutive" in classes["pick_failed"], "pick_failed requires the frozen pickup threshold", checks)
    require(taxonomy["state_and_scoring_contract"]["sustained_samples"] == 3 and taxonomy["state_and_scoring_contract"]["pickup_threshold_m"] == 0.03, "taxonomy freezes three-sample and +3cm thresholds", checks)
    require(set(taxonomy["continuous_fields"]) == {"signed_final_lateral_offset_m", "final_requested_signed_margin_m", "final_opposite_signed_margin_m", "maximum_requested_signed_margin_m", "maximum_pickup_height_m", "first_verified_pickup_step", "first_cone_or_native_region_entry_step", "entry_kind", "first_requested_region_step", "final_detached_release", "executed_action_count", "episode_length_steps", "first_contact_status", "first_contact_step", "first_contact_unavailable_reason", "object_path_length_m", "wall_time_s"}, "taxonomy continuous fields are complete", checks)
    fields = taxonomy["continuous_fields"]
    require("sustained, transient, or none" in fields["entry_kind"] and "three consecutive" in fields["entry_kind"], "taxonomy records sustained/transient/none entry kind", checks)
    require("delta_y" in fields["signed_final_lateral_offset_m"] and "positive robot LEFT" in fields["signed_final_lateral_offset_m"] and "not the negated reader-display coordinate" in fields["signed_final_lateral_offset_m"], "taxonomy freezes the raw v2 robot-frame lateral offset", checks)
    require("+delta_y for LEFT" in fields["final_requested_signed_margin_m"] and "-delta_y for RIGHT" in fields["final_requested_signed_margin_m"], "taxonomy derives requested margin from raw lateral offset", checks)
    contact = taxonomy["contact_conditional_rules"]
    require(fields["first_contact_status"] == "one of observed, not_observed, or instrumentation_unavailable" and contact["allowed_statuses"] == ["observed", "not_observed", "instrumentation_unavailable"], "taxonomy freezes observed/not-observed/instrumentation-unavailable contact status", checks)
    require(contact["first_contact_step"] == {"observed": "required integer", "not_observed": "required null", "instrumentation_unavailable": "required null"}, "contact step is integer only for observed contact", checks)
    require(contact["first_contact_unavailable_reason"] == {"observed": "must be absent or null", "not_observed": "must be absent or null", "instrumentation_unavailable": "required nonempty string"}, "contact unavailable reason is required only for instrumentation failure", checks)
    require(contact["all_false_retained_contact_stream"] == "A retained all-false contact stream is not_observed, not instrumentation_unavailable.", "retained all-false contact streams are not mislabeled unavailable", checks)
    scorer_rules = taxonomy["scorer_consistency_rules"]
    require(scorer_rules["release_failed"] == "requires verified pickup, sustained final requested region, and final_detached_release=false" and scorer_rules["requested_success_false_with_sustained_requested_and_detached_true"] == "technical_invalid_scorer_inconsistency; preserve the record and invalidate it from behavioral classification until the frozen scorer/input provenance is repaired" and scorer_rules["prohibited_inference"] == "Never infer final_detached_release from requested_success, gripper command, or action trace.", "taxonomy treats detached release as independent and scorer inconsistency as technical invalid", checks)
    require("outside behavioral denominators" in taxonomy["infrastructure_separation"]["rule"], "taxonomy separates infrastructure attempts from behavior", checks)

    require(amendment["schema_version"] == "vla-wam-shared-v3-post-result-power-failure-ablation-amendment-v1", "amendment schema is frozen", checks)
    require(amendment["status"] == "frozen_after_all_v2_and_v2_a015_results_and_before_any_v3_model_request_or_behavioral_inference", "amendment discloses timing before v3 inference", checks)
    phases = amendment["phases"]
    require(set(phases) == {"A_direct_replication", "B_confound_ablation", "C_four_phrasings", "D_16_rollout_stochastic_block"}, "all four v3 phases are registered", checks)
    confound = phases["B_confound_ablation"]
    require(confound["status"] == "separately_gated_not_released_by_phase_a" and confound["calibration_registry"] == f"{V3}/confound_fixture_calibration_registry.json" and "exactly one named factor" in confound["one_factor_rule"] and "randomized" in confound["randomization"], "confound ablations are one-factor, randomized, and separately gated", checks)
    require(calibration_registry["schema_version"] == "vla-wam-shared-v3-confound-fixture-calibration-registry-v1" and calibration_registry["status"] == "model_blind_fixture_calibration_required_before_any_confound_level_is_released", "confound levels remain unreleased pending model-blind calibration", checks)
    require(calibration_registry["unreleased_fields"] == ["numeric_fixture_coordinates", "numeric_requested_region_margin_m", "numeric_opposite_region_margin_m", "confound_factor_levels", "control_intervention_allocation"] and "new hash-pinned amendment" in calibration_registry["freeze_requirement"], "confound calibration freezes no invented numeric level", checks)
    wording = phases["C_four_phrasings"]
    require(wording["status"] == "separately_gated_not_released_by_phase_a" and wording["registry"] == f"{V3}/four_phrasings_registry.json" and "V2-A008 remains failed" in wording["pi0_fast_boundary"], "four phrasing block is independently randomized/gated and cannot bypass V2-A008", checks)
    require(wording_registry["schema_version"] == "vla-wam-shared-v3-four-phrasings-registry-v1" and wording_registry["eligible_model_ids"] == WORDING_MODELS, "four phrasing registry fixes the first three non-pi0 checkpoints", checks)
    shared = wording_registry["shared_matched_pairs"]
    require(shared["seed_range"] == [8500, 8519] and shared["environment_seed_equals_sampling_seed"] is True and shared["pair_count_per_checkpoint"] == 20 and shared["directions_per_pair"] == ["left", "right"], "four phrasing registry fixes twenty shared matched seeds", checks)
    require(wording_registry["prompt_forms"] == EXACT_V2_WORDINGS, "four phrasing registry preserves exact v2 prompt bytes", checks)
    wording_accounting = wording_registry["episode_accounting"]
    require(wording_accounting == {"checkpoints": 3, "matched_pairs_per_checkpoint": 20, "prompt_forms": 4, "directions_per_pair": 2, "episodes_per_checkpoint": 160, "frozen_total_episodes": 480, "formula": "3 checkpoints * 20 pairs * 4 prompt forms * 2 directions"}, "four phrasing accounting fixes 480 episodes", checks)
    optional_pi0 = wording_registry["optional_model_after_independent_release"]
    optional_pi0_conditions = " ".join(optional_pi0["conditions"])
    require(optional_pi0["model_id"] == "pi0_fast_droid_vla" and "exact historical OpenPI/RoboLab revisions" in optional_pi0_conditions and "sensitivity release" in optional_pi0_conditions, "pi0-FAST wording participation requires exact recovery and new sensitivity release", checks)
    block = phases["D_16_rollout_stochastic_block"]
    require(block["status"] == "separately_gated_not_released_by_phase_a" and block["registry"] == f"{V3}/stochastic_rollout_registry.json", "Phase D is separately gated through its registry", checks)
    require(stochastic_registry["schema_version"] == "vla-wam-shared-v3-stochastic-rollout-registry-v1" and stochastic_registry["shared_sampling_seed_indices"] == list(range(16)), "Phase D freezes sixteen shared sampling-seed indices", checks)
    require("Every released Phase-A registered" in stochastic_registry["scope"] and "exactly one rollout" in stochastic_registry["rollout_contract"], "Phase D applies sixteen rollouts to every eligible released Phase-A condition", checks)
    require("deterministic runtime" in stochastic_registry["eligibility_gate"] and "no fake repeated rollouts" in stochastic_registry["eligibility_gate"], "Phase D makes deterministic or ineffective-seed runtimes ineligible", checks)
    require("not sixteen independent scenes" in stochastic_registry["analysis_unit"] and "not independent scenes" in block["analysis_boundary"], "Phase D rejects pseudoreplication", checks)
    require(amendment["shared_required_evidence"][:3] == ["viewport_video", "executed_action_trace", "raw_result_jsonl"], "amendment requires raw video/actions/JSONL", checks)
    require("never pooled" in amendment["arena_boundary"], "amendment maintains arena separation", checks)

    require(phase_a_manifest["schema_version"] == "vla-wam-shared-v3-phase-a-cells-manifest-v1" and phase_a_manifest["study_id"] == "vla_wam_language_steerability_v3", "Phase-A queue manifest schema and study identity are frozen", checks)
    require(phase_a_manifest["queue_file"] == f"{V3}/phase_a_cells.jsonl" and phase_a_manifest["queue_sha256"] == sha256(paths["phase_a_queue"]), "Phase-A queue manifest hash matches the generated JSONL", checks)
    expected_row_counts = {
        "by_arena": {"droid_robolab": 360, "robotwin": 420},
        "by_model": {
            "cosmos3_edge_policy_droid": 60, "cosmos3_nano_policy_droid": 60,
            "dreamzero_droid_action_cfg": 60, "efficient_wam_rt_robotwin": 140,
            "fastwam_robotwin": 140, "groot_n17_droid_vla": 60,
            "lingbot_va_robotwin": 140, "pi05_current_stack_droid": 60,
            "pi0_fast_droid_vla": 60,
        },
        "by_status": {"authorized_new": 648, "blocked_pi0": 40, "preserved_candidate": 50, "preserved_r0": 42},
        "total": 780,
    }
    require(phase_a_manifest["row_counts"] == expected_row_counts, "Phase-A manifest fixes 780 total rows, 360/420 arena rows, and 648/50/42/40 statuses", checks)
    require(len(phase_a_rows) == 780 and all(row.get("schema_version") == "vla-wam-shared-v3-phase-a-cells-v1" for row in phase_a_rows), "Phase-A JSONL has exactly 780 schema-valid rows", checks)
    queue_arena_counts = Counter(row.get("arena") for row in phase_a_rows)
    queue_status_counts = Counter(row.get("status") for row in phase_a_rows)
    require(dict(queue_arena_counts) == {"droid_robolab": 360, "robotwin": 420} and dict(queue_status_counts) == {"authorized_new": 648, "blocked_pi0": 40, "preserved_candidate": 50, "preserved_r0": 42}, "Phase-A JSONL row counts agree with the manifest", checks)

    require(analysis["schema_version"] == "vla-wam-shared-v3-analysis-plan-v1", "analysis schema is frozen", checks)
    require("Never combine" in analysis["primary_estimands"]["no_pooling"], "analysis forbids pooled arena inference", checks)
    hierarchical = analysis["hierarchical_16_rollout_analysis"]
    require("every eligible released Phase-A" in hierarchical["unit"] and "Deterministic" in hierarchical["eligibility"] and "independent scenes" in hierarchical["prohibition"] and "Do not" in hierarchical["prohibition"], "analysis nests Phase D rollouts within every eligible condition", checks)
    require("outside behavioral denominators" in analysis["infrastructure_and_latency"], "analysis separates infrastructure invalidity", checks)

    require(
        measurement_audit.get("schema_version") == "vla-wam-measurement-coverage-audit-v1"
        and measurement_audit.get("study_id") == "vla_wam_language_steerability_v3"
        and measurement_audit.get("status") == "complete_no_measurement_coverage_rerun_required",
        "measurement audit schema, study identity, and no-rerun conclusion are exact",
        checks,
    )
    require(
        measurement_audit.get("scope") == {
            "unique_behavioral_episode_count": 982,
            "droid_robolab_episode_count": 532,
            "robotwin_episode_count": 450,
            "nonbehavioral_interface_probes": "not applicable: Cosmos-Reason2 and Cosmos3 base probes have no robot episode endpoint",
            "withdrawn_or_unreleased_models": "not applicable: LaWAM has zero behavioral episodes",
        },
        "measurement audit accounts for all 982 unique behavioral episodes without nonbehavioral probes",
        checks,
    )
    require(
        measurement_audit.get("coverage") == {
            "requested_side_margin_available": "982/982",
            "signed_final_lateral_offset_available": "982/982",
            "values_imputed_from_success_labels": 0,
            "measurement_coverage_rerun_required": False,
        },
        "measurement audit proves complete margin and signed-offset coverage without success-label imputation",
        checks,
    )
    nano_margin = measurement_audit.get("nano_phase_a_margin_sensitivity_reproduction", {})
    require(
        nano_margin.get("matched_pair_count") == 27
        and nano_margin.get("positive_zero_negative_pair_counts") == [23, 0, 4]
        and close(nano_margin.get("right_minus_left_mean_margin_gap_m"), 0.12360139639565239)
        and close(nano_margin.get("exact_two_sided_sign_test_p_excluding_ties"), 0.000310748815536499),
        "measurement audit exactly reproduces Nano Phase-A margin sensitivity",
        checks,
    )
    require(
        measurement_audit.get("groot_phase_a_reconciliation") == {
            "matched_pairs": 27,
            "behavioral_episodes": 54,
            "status": "already_complete_do_not_rerun",
            "source": f"{V3}/results/groot_n17_droid_phase_a_summary.json",
        },
        "measurement audit records GR00T n=27 as complete and non-rerunnable",
        checks,
    )
    audit_cohorts = measurement_audit.get("cohorts", [])
    require(
        isinstance(audit_cohorts, list)
        and len(audit_cohorts) == 27
        and len({row.get("cohort_id") for row in audit_cohorts if isinstance(row, dict)}) == 27
        and sum(row.get("behavioral_episode_count", 0) for row in audit_cohorts if isinstance(row, dict)) == 982
        and all(
            row.get("rerun_required_for_these_two_measurements") is False
            for row in audit_cohorts
            if isinstance(row, dict)
        ),
        "measurement audit has 27 unique all-covered cohorts totaling 982 episodes",
        checks,
    )
    audit_state = continuation.get("measurement_coverage_audit", {})
    require(
        REQUIRED["measurement_audit"] in continuation.get("authoritative_files", [])
        and audit_state == {
            "path": REQUIRED["measurement_audit"],
            "sha256": sha256(paths["measurement_audit"]),
            "bytes": paths["measurement_audit"].stat().st_size,
            "unique_behavioral_episodes": 982,
            "requested_side_margin_available": "982/982",
            "signed_final_lateral_offset_available": "982/982",
            "legacy_measurement_rerun_required": False,
            "groot_phase_a": "already_complete_27_matched_pairs_do_not_rerun",
        },
        "V3 continuation hash-binds the complete measurement audit and no-rerun decision",
        checks,
    )

    bridge_result = validate_pi0_fast_bridge(paths, checks)
    validate_pi0_fast_bridge_continuation(continuation, bridge_result, paths, checks)
    require(
        all(
            isinstance(source, dict)
            and (root / source.get("path", "")).is_file()
            and source.get("bytes") == (root / source["path"]).stat().st_size
            and source.get("sha256") == sha256(root / source["path"])
            for row in audit_cohorts
            for source in row.get("sources", [])
        ),
        "measurement audit hash-binds every source artifact",
        checks,
    )

    doc = paths["document"].read_text()
    for phrase in ("30 exact matched", "378 new episodes", "480", "16 shared", "never pooled"):
        require(phrase in doc, f"human-readable protocol states {phrase!r}", checks)
    continuation_doc = " ".join(paths["continuation_document"].read_text().split())
    handoff_doc = " ".join(paths["work_laptop_handoff"].read_text().split())
    require(
        "Post-result π0-FAST compatibility cohort (V3-A002)" in continuation_doc
        and "public old-name OpenPI configuration" in continuation_doc
        and "20 matched pairs / 40 episodes" in continuation_doc,
        "V3 continuation documents the complete public old-name-config compatibility cohort",
        checks,
    )
    require(
        EXACT_V2_WORDINGS["direct_command"]["left"] in continuation_doc
        and EXACT_V2_WORDINGS["direct_command"]["right"] in continuation_doc,
        "V3 continuation states both exact bridge prompts rather than LEFT/RIGHT shorthand alone",
        checks,
    )
    require(
        "0/20 | 12/20" in continuation_doc
        and "20/20 action responses" in continuation_doc
        and "16/20 endpoint redirections" in continuation_doc,
        "V3 continuation states bridge success, action-sensitivity, and endpoint counts",
        checks,
    )
    require(
        "not historical recovery" in continuation_doc
        and "must remain separate" in continuation_doc
        and "must not be rerun" in continuation_doc
        and "Phase C: 480 registered episodes, not released" in continuation_doc,
        "V3 continuation preserves non-pooling, no-rerun, historical-blocker, and Phase-C boundaries",
        checks,
    )
    require(
        "982/982" in continuation_doc
        and "signed final lateral offset" in continuation_doc
        and "GR00T is already complete at 27 matched pairs" in continuation_doc,
        "V3 continuation documents complete measurement coverage and the GR00T no-rerun decision",
        checks,
    )
    require(
        "Nano V3-B001 completed result" in continuation_doc
        and "108/108 prespecified cells" in continuation_doc
        and "27/27 control seeds" in continuation_doc
        and "27/27" in continuation_doc
        and "Historical live-boundary and reset audit" in continuation_doc
        and "first three complete four-cell seed" in continuation_doc
        and "12 valid behavioral cells" in continuation_doc
        and "10 `correct`" in continuation_doc
        and "one `release_failed`" in continuation_doc
        and "`transport_failed`" in continuation_doc
        and "prefix, not current study state" in continuation_doc
        and "all 39 prior model-sampler requests" in continuation_doc
        and "Four earlier" in continuation_doc
        and "complete behavioral attempts" in continuation_doc
        and "invalidated before analysis" in continuation_doc
        and "falsified as the sufficient explanation" in continuation_doc
        and "episode-length mutation order" in continuation_doc
        and "two runner reset calls" in continuation_doc
        and "zero-sampling exact-bridge preflight has now passed" in continuation_doc
        and "No model server or" in continuation_doc
        and "no model sample was drawn" in continuation_doc
        and "completed queue used only that verified runtime identity" in continuation_doc
        and "descriptive historical-prefix counts" in continuation_doc
        and "no inference is made from that prefix" in continuation_doc
        and "v3b001:nano:seed9400:position_mirrored:right" in continuation_doc
        and EXACT_V2_WORDINGS["direct_command"]["right"] in continuation_doc,
        "V3 continuation distinguishes the bounded Nano live snapshot from completed-ablation inference",
        checks,
    )
    require(
        "π0-FAST V3-A002 public old-name-config compatibility cohort" in handoff_doc
        and "complete: 40/40 valid; LEFT 0/20, RIGHT 12/20" in handoff_doc
        and "16/20 endpoint pairs aligned" in handoff_doc
        and "20/20 common action prefixes differed" in handoff_doc,
        "work-laptop handoff carries the complete bridge result and diagnostics",
        checks,
    )
    require(
        "distinct compatibility cohort, not historical recovery" in handoff_doc
        and "must not be rerun or pooled" in handoff_doc
        and "direct-command completion as a Phase-C release" in handoff_doc,
        "work-laptop handoff prohibits bridge pooling, reruns, and implicit Phase-C release",
        checks,
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        checks = validate(args.root.resolve())
    except ValidationError as exc:
        print(f"V3 protocol validation failed: {exc}")
        return 1
    if not args.quiet:
        print(f"V3 protocol validation passed: {len(checks)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
