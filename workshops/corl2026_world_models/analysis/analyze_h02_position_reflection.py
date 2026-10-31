#!/usr/bin/env python3
"""Authenticate and reproduce retained H02 position-reflection evidence.

H02 is the completed V3-B003 DreamZero custom negative-branch action-guidance
s2 cohort.  It is not the official conditional-action D1 configuration.  This
adapter reads only immutable compact Git objects, authenticates the complete
27-seed x 2-layout x 2-command registration/results population, and
independently rebuilds the endpoint, requested-depth and binary-success
contrasts from the 108 compact episodes and 54 compact matched pairs.

The 540 raw-artifact descriptors are inventoried but their external PVC bytes
are not opened or rehashed.  The script runs no model, robot, remote retrieval,
annotation, publication or release workflow and advances no study gate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from typing import Any, Mapping, Sequence


PIN = "ce561e66f82e95055e39d3d7711691982f6b2086"
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
DEFAULT_JSON = PACKAGE / "paper" / "data" / "h02_position_reflection.json"
DEFAULT_CSV = PACKAGE / "paper" / "data" / "h02_position_reflection.csv"

BASE = "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003"
MANIFEST = f"{BASE}/dreamzero_mirror_v3b003_manifest.json"
CELLS = f"{BASE}/dreamzero_mirror_v3b003_cells.jsonl"
AMENDMENT = f"{BASE}/post_result_dreamzero_mirror_v3b003_amendment.json"
EPISODES = f"{BASE}/results/dreamzero_v3b003_episodes.jsonl"
PAIRS = f"{BASE}/results/dreamzero_v3b003_matched_pairs.jsonl"
SUMMARY = f"{BASE}/results/dreamzero_v3b003_summary.json"
EVIDENCE = f"{BASE}/results/evidence_manifest.json"
UPSTREAM_ANALYSIS = "experiments/v3/dreamzero_phase_b/analyze_results.py"
ROBOT_BRIDGE = "experiments/v3/dreamzero_phase_b/robolab_bridge.py"

EXPECTED_SOURCES: dict[str, dict[str, Any]] = {
    MANIFEST: {
        "git_blob": "a072b2ae0abe1bc757157656a7a1a7375f6482fd",
        "sha256": "efe50df701193e48b981c025ea3b4d27a80e3bdf83216e38a98a63e27061cb23",
        "bytes": 1_551,
        "lines": 56,
    },
    CELLS: {
        "git_blob": "e4a98f47422b098f23c28003f935fa7f4f233995",
        "sha256": "a6d0f0a5d4c7cdfa5d3de95d44d7b11f42750a76a603ff8c2e44848e34b8f70d",
        "bytes": 291_384,
        "lines": 108,
    },
    AMENDMENT: {
        "git_blob": "5501346ae2629213ea479048c820362681b36a2b",
        "sha256": "ba22681ae4d7f748e375617617d9e130e6f1bd5bc0af1e7a995365b145a470fc",
        "bytes": 7_807,
        "lines": 170,
    },
    EPISODES: {
        "git_blob": "cacce0095ccafa667a0dd4d93f448f3c31e3c55a",
        "sha256": "31f0715c3cbc682f90e5cc6420b9f45f49622f876734d43ba7c8cb6545b80f34",
        "bytes": 281_813,
        "lines": 108,
    },
    PAIRS: {
        "git_blob": "dae25a9e2d7bc18f5dd0f41111eb17787c64d1f8",
        "sha256": "bb88b62e9137f292a2a1739bd4976daf0d641d242876e921ed4f21d605d9e2a7",
        "bytes": 53_597,
        "lines": 54,
    },
    SUMMARY: {
        "git_blob": "879220bdd189ed874d9fc24fef83a405082d46e2",
        "sha256": "f45ab7d11281b994592948cdc11ea5d0fe24837d75d21a8a7dd42ff63b9c5817",
        "bytes": 124_097,
        "lines": 1,
    },
    EVIDENCE: {
        "git_blob": "e66833077cbea3c4de5f5e05ed66cd51bca8981c",
        "sha256": "a24278ee01624971877afd7c3db1b7f4377957c3f7d58b9fadc9bd6a70b6fd35",
        "bytes": 29_355,
        "lines": 1,
    },
    UPSTREAM_ANALYSIS: {
        "git_blob": "77a1849eb0722209fa229431d21f1d4cce676db8",
        "sha256": "4128c27a744d6d1f9cdfebdc19013291bf692de4d6d1f2874ba0ac36521805fe",
        "bytes": 18_232,
        "lines": 422,
    },
    ROBOT_BRIDGE: {
        "git_blob": "7e610c65d785a3c47674ec41290d91a7afe75d16",
        "sha256": "ba43a534a2a156093628e94085bf442ae8063e7a21942710cc888286fd1b05d2",
        "bytes": 23_423,
        "lines": 509,
    },
}

STUDY = "vla_wam_language_steerability_v3"
AMENDMENT_ID = "V3-B003"
MODEL = "dreamzero_droid_action_cfg"
SEEDS = tuple(range(9400, 9427))
ARMS = ("control", "position_mirrored")
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
FAILURE_CATEGORIES = (
    "correct", "pick_failed", "release_failed", "transport_failed", "wrong_side"
)
ARTIFACT_FIELDS = (
    "source_behavioral_jsonl", "source_batch_manifest", "executed_action_trace",
    "viewport_video", "exposed_future_manifest",
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "--no-replace-objects", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def parse_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _jsonl(raw: bytes, *, label: str, expected: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise ValueError(f"blank row in {label}:{number}")
        rows.append(parse_json(line, label=f"{label}:{number}"))
    if len(rows) != expected:
        raise ValueError(f"{label} must contain exactly {expected} rows")
    return rows


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _close(observed: Any, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(_finite(observed, label), expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} disagrees with independent reproduction")


def read_pinned_sources(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root)
    resolved = _git(root, "rev-parse", "--verify", PIN + "^{commit}").decode().strip()
    if resolved != PIN:
        raise ValueError("resolved historical evidence commit differs from fixed pin")
    raw_by_path: dict[str, bytes] = {}
    observed: dict[str, dict[str, Any]] = {}
    for path, expected in EXPECTED_SOURCES.items():
        object_id = _git(root, "rev-parse", "--verify", PIN + ":" + path).decode().strip()
        if object_id != expected["git_blob"]:
            raise ValueError(f"pinned Git blob mismatch for {path}")
        raw = _git(root, "cat-file", "blob", object_id)
        actual = {
            "git_blob": object_id,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "lines": len(raw.splitlines()),
        }
        if actual != expected:
            raise ValueError(f"pinned source bytes disagree with fixed identity for {path}")
        raw_by_path[path] = raw
        observed[path] = actual
    documents = {
        "manifest": parse_json(raw_by_path[MANIFEST], label=MANIFEST),
        "cells": _jsonl(raw_by_path[CELLS], label=CELLS, expected=108),
        "amendment": parse_json(raw_by_path[AMENDMENT], label=AMENDMENT),
        "episodes": _jsonl(raw_by_path[EPISODES], label=EPISODES, expected=108),
        "pairs": _jsonl(raw_by_path[PAIRS], label=PAIRS, expected=54),
        "summary": parse_json(raw_by_path[SUMMARY], label=SUMMARY),
        "evidence": parse_json(raw_by_path[EVIDENCE], label=EVIDENCE),
    }
    provenance = {
        "commit": PIN,
        "files": observed,
        "authenticated_file_count": len(observed),
        "authenticated_bytes": sum(row["bytes"] for row in observed.values()),
        "authenticated_lines": sum(row["lines"] for row in observed.values()),
    }
    return documents, provenance


def _descriptor(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "bytes"}:
        raise ValueError(f"{label} must be an exact path/SHA/byte descriptor")
    path = value["path"]
    sha = value["sha256"]
    size = value["bytes"]
    if not isinstance(path, str) or not path.startswith("/data/users/ali/vla_wam/"):
        raise ValueError(f"{label} has an invalid retained path")
    if not isinstance(sha, str) or not SHA_RE.fullmatch(sha):
        raise ValueError(f"{label} has an invalid SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{label} has an invalid byte count")
    return {"path": path, "sha256": sha, "bytes": size}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sign_test(values: Sequence[float]) -> dict[str, Any]:
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    ties = len(values) - positive - negative
    n = positive + negative
    low = min(positive, negative)
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(low + 1)) / (2**n)
        p_value = min(1.0, 2.0 * tail)
    return {
        "positive": positive,
        "negative": negative,
        "ties": ties,
        "effective_n": n,
        "exact_two_sided_p": p_value,
    }


def _summary(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": _mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "sample_standard_deviation": statistics.stdev(values),
        "exact_sign_test": _sign_test(values),
    }


def _exact_layout_permutation(values: Sequence[int]) -> dict[str, Any]:
    """Count all layout-label swaps by dynamic programming, not 2^27 enumeration."""
    observed = sum(values)
    counts: dict[int, int] = {0: 1}
    for value in values:
        next_counts: defaultdict[int, int] = defaultdict(int)
        for total, count in counts.items():
            next_counts[total + value] += count
            next_counts[total - value] += count
        counts = dict(next_counts)
    extreme = sum(count for total, count in counts.items() if abs(total) >= abs(observed))
    total = 2 ** len(values)
    return {
        "registered_seed_count": len(values),
        "observed_signed_sum": observed,
        "observed_absolute_sum": abs(observed),
        "total_permutations": total,
        "extreme_permutations": extreme,
        "exact_two_sided_p": extreme / total,
    }


def analyze_documents(documents: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    manifest = documents["manifest"]
    amendment = documents["amendment"]
    source_summary = documents["summary"]
    if manifest.get("amendment_id") != AMENDMENT_ID:
        raise ValueError("registration manifest amendment differs")
    design = amendment.get("design")
    release = amendment.get("release_boundary")
    runtime = amendment.get("runtime_contract_source")
    if not isinstance(design, Mapping) or design.get("behavioral_episode_ceiling") != 108:
        raise ValueError("amendment does not preserve the 108-episode design")
    if design.get("matched_seeds") != list(SEEDS) or design.get("model_id") != MODEL:
        raise ValueError("amendment seed/model design differs")
    if not isinstance(release, Mapping) or release.get("behavioral_episodes_before_registration") != 0:
        raise ValueError("amendment was not frozen before behavioral activity")
    if not isinstance(runtime, Mapping) or runtime.get("effective_official_model_noise_seed") != 1140:
        raise ValueError("fixed effective DreamZero noise identity differs")
    if runtime.get("identity_binding") != "V2-A015:dreamzero_action_cfg_s2":
        raise ValueError("custom s2 identity binding differs")

    expected_keys = {(seed, arm, relation) for seed in SEEDS for arm in ARMS for relation in RELATIONS}
    cells: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in documents["cells"]:
        key = (row.get("environment_seed"), row.get("arm"), row.get("relation"))
        if key in cells:
            raise ValueError(f"duplicate registered cell: {key}")
        if key not in expected_keys:
            raise ValueError(f"unexpected registered cell: {key}")
        relation = str(row["relation"])
        if row.get("cell_id") != f"v3b003:dreamzero:seed{row['environment_seed']}:{row['arm']}:{relation}":
            raise ValueError("registered cell identity does not match design")
        if row.get("model_id") != MODEL or row.get("effective_model_noise_seed") != 1140:
            raise ValueError("registered cell model/noise identity differs")
        if row.get("prompt") != PROMPTS[relation]:
            raise ValueError("registered cell prompt differs")
        cells[key] = row
    if set(cells) != expected_keys:
        raise ValueError("registration is not the exact 108-cell population")

    episodes: dict[tuple[int, str, str], dict[str, Any]] = {}
    artifact_rows: list[dict[str, Any]] = []
    for row in documents["episodes"]:
        key = (row.get("seed"), row.get("arm"), row.get("requested_relation"))
        if key in episodes or key not in expected_keys:
            raise ValueError(f"missing, duplicate or unexpected compact episode: {key}")
        seed, arm, relation = key
        cell_id = f"v3b003:dreamzero:seed{seed}:{arm}:{relation}"
        if row.get("registered_cell_id") != cell_id or cells[key].get("cell_id") != cell_id:
            raise ValueError("compact episode does not join its registered cell")
        if row.get("study_id") != STUDY or row.get("amendment_id") != AMENDMENT_ID:
            raise ValueError("compact episode study identity differs")
        if row.get("schema_version") != "vla-wam-shared-v3b-dreamzero-mirror-compact-episode-v1":
            raise ValueError("compact episode schema version differs")
        if row.get("model_id") != MODEL or row.get("prompt") != PROMPTS[str(relation)]:
            raise ValueError("compact episode model/prompt differs")
        if type(row.get("requested_success")) is not bool:
            raise ValueError("compact episode success must be boolean")
        if row.get("failure_category") not in FAILURE_CATEGORIES:
            raise ValueError("compact episode failure category differs")
        if (row.get("failure_category") == "correct") != row.get("requested_success"):
            raise ValueError("success/failure taxonomy contradiction")
        _finite(row.get("signed_final_lateral_offset_m"), f"{cell_id}.offset")
        _finite(row.get("requested_side_depth_m"), f"{cell_id}.depth")
        actions = row.get("actions_executed")
        if isinstance(actions, bool) or not isinstance(actions, int) or not 0 < actions <= 450:
            raise ValueError("compact episode action count is invalid")
        if row.get("episode_length_steps") != actions:
            raise ValueError("compact episode length must equal its executed action count")
        if type(row.get("right_censored")) is not bool:
            raise ValueError("compact episode censoring flag must be boolean")
        expected_censoring = not row["requested_success"] and actions == 450
        if row["right_censored"] != expected_censoring:
            raise ValueError("compact episode right-censoring contract differs")
        entry_step = row.get("cone_entry_step")
        if entry_step is not None and (
            isinstance(entry_step, bool)
            or not isinstance(entry_step, int)
            or not 0 <= entry_step <= actions
        ):
            raise ValueError("compact episode cone-entry action index is invalid")
        if type(row.get("cone_entry_sustained")) is not bool:
            raise ValueError("compact episode sustained-entry flag must be boolean")
        if row["cone_entry_sustained"] and entry_step is None:
            raise ValueError("sustained entry requires a recorded entry action index")
        initial = row.get("initial_state_sha256")
        if not isinstance(initial, str) or not SHA_RE.fullmatch(initial):
            raise ValueError("compact episode reset identity is invalid")
        for field in ARTIFACT_FIELDS:
            artifact_rows.append({"cell_id": cell_id, "role": field, **_descriptor(row.get(field), f"{cell_id}.{field}")})
        episodes[key] = row
    if set(episodes) != expected_keys:
        raise ValueError("compact episodes are not the exact 108-cell population")
    artifact_identities = {(row["path"], row["sha256"]) for row in artifact_rows}
    if len(artifact_rows) != 540 or len(artifact_identities) != 540:
        raise ValueError("raw descriptor inventory must contain 540 distinct path/SHA identities")

    pairs: dict[tuple[int, str], dict[str, Any]] = {}
    expected_pair_keys = {(seed, arm) for seed in SEEDS for arm in ARMS}
    for row in documents["pairs"]:
        key = (row.get("seed"), row.get("arm"))
        if key in pairs or key not in expected_pair_keys:
            raise ValueError(f"missing, duplicate or unexpected compact pair: {key}")
        seed, arm = key
        if row.get("schema_version") != "vla-wam-shared-v3b-dreamzero-mirror-pair-v1":
            raise ValueError("compact pair schema version differs")
        left = episodes[(seed, arm, "left")]
        right = episodes[(seed, arm, "right")]
        if left["initial_state_sha256"] != right["initial_state_sha256"]:
            raise ValueError("LEFT/RIGHT compact episodes do not share the reset identity")
        expected_ids = (left["registered_cell_id"], right["registered_cell_id"])
        if (row.get("left_registered_cell_id"), row.get("right_registered_cell_id")) != expected_ids:
            raise ValueError("compact pair does not join its episodes")
        if row.get("initial_state_sha256") != left["initial_state_sha256"]:
            raise ValueError("compact pair reset identity differs")
        d_value = float(left["signed_final_lateral_offset_m"]) - float(right["signed_final_lateral_offset_m"])
        b_value = float(right["requested_side_depth_m"]) - float(left["requested_side_depth_m"])
        _close(row.get("endpoint_redirection_D_m"), d_value, "pair endpoint")
        _close(row.get("requested_side_depth_contrast_B_m"), b_value, "pair depth")
        if row.get("left_success") != left["requested_success"] or row.get("right_success") != right["requested_success"]:
            raise ValueError("compact pair success flags differ from episodes")
        did = int(right["requested_success"]) - int(left["requested_success"])
        if row.get("right_minus_left_success") != did:
            raise ValueError("compact pair binary contrast differs")
        if row.get("left_executed_action_count") != left["actions_executed"] or row.get("right_executed_action_count") != right["actions_executed"]:
            raise ValueError("compact pair action counts differ")
        pairs[key] = row
    if set(pairs) != expected_pair_keys:
        raise ValueError("compact pairs are not the exact 54-pair population")

    condition_outcomes: dict[str, Any] = {}
    overall_taxonomy: Counter[str] = Counter()
    for arm in ARMS:
        for relation in RELATIONS:
            rows = [episodes[(seed, arm, relation)] for seed in SEEDS]
            taxonomy = Counter(str(row["failure_category"]) for row in rows)
            overall_taxonomy.update(taxonomy)
            condition_outcomes[f"{arm}:{relation}"] = {
                "episodes": 27,
                "successes": sum(bool(row["requested_success"]) for row in rows),
                "failures": sum(not bool(row["requested_success"]) for row in rows),
                "failure_taxonomy_counts": dict(sorted(taxonomy.items())),
            }

    d_by_arm = {arm: [float(pairs[(seed, arm)]["endpoint_redirection_D_m"]) for seed in SEEDS] for arm in ARMS}
    b_by_arm = {arm: [float(pairs[(seed, arm)]["requested_side_depth_contrast_B_m"]) for seed in SEEDS] for arm in ARMS}
    j_values = [d_by_arm["position_mirrored"][i] - d_by_arm["control"][i] for i in range(27)]
    i_values = [b_by_arm["position_mirrored"][i] - b_by_arm["control"][i] for i in range(27)]
    binary_by_arm = {
        arm: [int(pairs[(seed, arm)]["right_minus_left_success"]) for seed in SEEDS]
        for arm in ARMS
    }
    did_values = [binary_by_arm["position_mirrored"][i] - binary_by_arm["control"][i] for i in range(27)]
    permutation = _exact_layout_permutation(did_values)

    source_conditions = source_summary.get("condition_outcomes")
    if not isinstance(source_conditions, Mapping):
        raise ValueError("source summary lacks condition outcomes")
    for key, computed in condition_outcomes.items():
        reported = source_conditions.get(key)
        if not isinstance(reported, Mapping) or reported.get("episodes") != 27 or reported.get("successes") != computed["successes"]:
            raise ValueError(f"source summary condition outcome differs for {key}")
        if reported.get("failure_taxonomy_counts") != computed["failure_taxonomy_counts"]:
            raise ValueError(f"source summary taxonomy differs for {key}")
    reported_primary = source_summary.get("full_sample_primary")
    if not isinstance(reported_primary, Mapping):
        raise ValueError("source summary lacks registered primary analysis")
    for arm in ARMS:
        _close(reported_primary["D_by_arm"][arm]["mean_m"], _mean(d_by_arm[arm]), f"source D mean {arm}")
        _close(reported_primary["B_by_arm"][arm]["mean_m"], _mean(b_by_arm[arm]), f"source B mean {arm}")
    _close(reported_primary["J_redirection_interaction"]["mean_m"], _mean(j_values), "source J mean")
    _close(reported_primary["I_position_reflection_interaction"]["mean_m"], _mean(i_values), "source I mean")
    reported_binary = reported_primary["binary_success_DiD"]
    if reported_binary.get("per_seed_DiD_distribution") != {str(value): did_values.count(value) for value in (-1, -2, 0, 1, 2)}:
        raise ValueError("source binary DiD distribution differs")
    reported_permutation = reported_binary.get("exact_permutation_test")
    for key, value in {
        "registered_seed_count": permutation["registered_seed_count"],
        "observed_signed_sum": permutation["observed_signed_sum"],
        "observed_absolute_sum": permutation["observed_absolute_sum"],
        "total_permutations": permutation["total_permutations"],
        "extreme_permutations": permutation["extreme_permutations"],
    }.items():
        if reported_permutation.get(key) != value:
            raise ValueError(f"source binary permutation {key} differs")
    _close(reported_permutation.get("p_value"), permutation["exact_two_sided_p"], "source permutation p")

    evidence = documents["evidence"]
    evidence_files = evidence.get("files")
    if not isinstance(evidence_files, list) or len(evidence_files) != 3:
        raise ValueError("evidence manifest must bind the three compact result files")
    expected_result_files = {
        "dreamzero_v3b003_summary.json": EXPECTED_SOURCES[SUMMARY],
        "dreamzero_v3b003_episodes.jsonl": EXPECTED_SOURCES[EPISODES],
        "dreamzero_v3b003_matched_pairs.jsonl": EXPECTED_SOURCES[PAIRS],
    }
    for item in evidence_files:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise ValueError("invalid evidence-manifest compact result descriptor")
        name = Path(item["path"]).name
        expected = expected_result_files.get(name)
        if expected is None or item.get("sha256") != expected["sha256"] or item.get("bytes") != expected["bytes"]:
            raise ValueError("evidence manifest result identity differs")

    artifact_by_role: dict[str, Any] = {}
    for role in ARTIFACT_FIELDS:
        rows = [row for row in artifact_rows if row["role"] == role]
        artifact_by_role[role] = {
            "descriptors": len(rows),
            "distinct_path_sha_identities": len({(row["path"], row["sha256"]) for row in rows}),
            "declared_bytes": sum(row["bytes"] for row in rows),
        }
    action_counts = [int(row["actions_executed"]) for row in episodes.values()]
    entry_by_condition: dict[str, Any] = {}
    for arm in ARMS:
        for relation in RELATIONS:
            rows = [episodes[(seed, arm, relation)] for seed in SEEDS]
            observed = [row for row in rows if row["cone_entry_step"] is not None]
            missing = [row for row in rows if row["cone_entry_step"] is None]
            if any(not row["right_censored"] or row["actions_executed"] != 450 for row in missing):
                raise ValueError("no-entry H02 records must remain cap-censored at action 450")
            goal_terminated_without_observed_sustained = [
                row for row in observed if not row["cone_entry_sustained"]
            ]
            if any(
                not row["requested_success"]
                or row["cone_entry_step"] != row["episode_length_steps"]
                for row in goal_terminated_without_observed_sustained
            ):
                raise ValueError(
                    "non-sustained recorded entries must be successful goal-terminated endpoints"
                )
            steps = [int(row["cone_entry_step"]) for row in observed]
            restricted_waiting = [
                int(row["cone_entry_step"]) if row["cone_entry_step"] is not None else 450
                for row in rows
            ]
            entry_by_condition[f"{arm}:{relation}"] = {
                "episodes": 27,
                "recorded_entry_events": len(observed),
                "no_entry_cap_censored": len(missing),
                "observed_sustained_entries": sum(
                    bool(row["cone_entry_sustained"]) for row in observed
                ),
                "goal_terminated_entry_without_observed_three_sample_sustained_run": len(
                    goal_terminated_without_observed_sustained
                ),
                "successes_with_entry": sum(bool(row["requested_success"]) for row in observed),
                "valid_failures_with_entry": sum(not bool(row["requested_success"]) for row in observed),
                "recorded_action_index_summary": {
                    "n": len(steps),
                    "mean": _mean(steps) if steps else None,
                    "median": statistics.median(steps) if steps else None,
                    "minimum": min(steps) if steps else None,
                    "maximum": max(steps) if steps else None,
                },
                "restricted_waiting_action_index": {
                    "definition": "recorded entry action, or administrative cap 450 when no entry was observed",
                    "n": 27,
                    "mean": _mean(restricted_waiting),
                    "median": statistics.median(restricted_waiting),
                    "minimum": min(restricted_waiting),
                    "maximum": max(restricted_waiting),
                },
            }
    entry_rows = list(entry_by_condition.values())
    entry_by_layout = {
        arm: {
            "episodes": 54,
            "recorded_entry_events": sum(
                entry_by_condition[f"{arm}:{relation}"]["recorded_entry_events"]
                for relation in RELATIONS
            ),
            "no_entry_cap_censored": sum(
                entry_by_condition[f"{arm}:{relation}"]["no_entry_cap_censored"]
                for relation in RELATIONS
            ),
        }
        for arm in ARMS
    }
    complete_case_seeds = [
        seed for seed in SEEDS
        if all(bool(episodes[(seed, arm, relation)]["requested_success"]) for arm in ARMS for relation in RELATIONS)
    ]
    matched_blocks = []
    for seed in SEEDS:
        block: dict[str, Any] = {"seed": seed}
        for arm in ARMS:
            for relation in RELATIONS:
                row = episodes[(seed, arm, relation)]
                prefix = f"{arm}_{relation}"
                entry_observed = row["cone_entry_step"] is not None
                block.update({
                    f"{prefix}_cell_id": row["registered_cell_id"],
                    f"{prefix}_actions_executed": row["actions_executed"],
                    f"{prefix}_entry_observed": entry_observed,
                    f"{prefix}_entry_action_index": row["cone_entry_step"],
                    f"{prefix}_restricted_waiting_action_index": (
                        row["cone_entry_step"] if entry_observed else 450
                    ),
                    f"{prefix}_observed_sustained_entry": row["cone_entry_sustained"],
                    f"{prefix}_goal_terminated_entry_without_observed_three_sample_sustained_run": (
                        entry_observed
                        and not row["cone_entry_sustained"]
                        and row["requested_success"]
                        and row["cone_entry_step"] == row["episode_length_steps"]
                    ),
                    f"{prefix}_requested_success": row["requested_success"],
                    f"{prefix}_failure_category": row["failure_category"],
                    f"{prefix}_right_censored": row["right_censored"],
                })
        entry = {
            (arm, relation): int(
                episodes[(seed, arm, relation)]["cone_entry_step"] is not None
            )
            for arm in ARMS for relation in RELATIONS
        }
        waiting = {
            (arm, relation): (
                episodes[(seed, arm, relation)]["cone_entry_step"]
                if episodes[(seed, arm, relation)]["cone_entry_step"] is not None
                else 450
            )
            for arm in ARMS for relation in RELATIONS
        }
        for label, values in (("entry_observed", entry), ("restricted_waiting_action_index", waiting)):
            control_rl = values[("control", "right")] - values[("control", "left")]
            mirrored_rl = (
                values[("position_mirrored", "right")]
                - values[("position_mirrored", "left")]
            )
            block.update({
                f"control_right_minus_left_{label}": control_rl,
                f"position_mirrored_right_minus_left_{label}": mirrored_rl,
                f"position_mirrored_minus_control_left_{label}": (
                    values[("position_mirrored", "left")] - values[("control", "left")]
                ),
                f"position_mirrored_minus_control_right_{label}": (
                    values[("position_mirrored", "right")] - values[("control", "right")]
                ),
                f"{label}_difference_in_differences": mirrored_rl - control_rl,
            })
        matched_blocks.append(block)

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
        "schema_version": "wmf-retained-h02-position-reflection-v1",
        "status": "retained_compact_behavioral_evidence_authenticated_no_stage_advance",
        "source_provenance": dict(provenance),
        "configuration": {
            "cohort": "H02",
            "amendment": AMENDMENT_ID,
            "label": "DreamZero custom negative-branch action guidance s2",
            "model_id": MODEL,
            "checkpoint_revision": runtime["checkpoint"]["revision"],
            "action_guidance_scale": runtime["action_cfg_style_scale"],
            "effective_model_noise_seed": 1140,
            "noise_seed_semantics": "fixed effective model noise; the 27 environment labels are not independent model-noise draws",
            "official_conditional_D1": False,
        },
        "authenticated_population": {
            "registered_cells": 108,
            "behavioral_episodes": 108,
            "matched_left_right_pairs": 54,
            "matched_layout_blocks": 27,
            "seeds": list(SEEDS),
            "layouts": list(ARMS),
            "relations": list(RELATIONS),
            "valid_behavioral_failures_included": True,
            "infrastructure_attempts_included": False,
            "missing_value_imputation": "none",
        },
        "condition_outcomes": condition_outcomes,
        "layout_totals": {
            arm: {
                "episodes": 54,
                "successes": sum(condition_outcomes[f"{arm}:{relation}"]["successes"] for relation in RELATIONS),
                "failures": sum(condition_outcomes[f"{arm}:{relation}"]["failures"] for relation in RELATIONS),
            }
            for arm in ARMS
        },
        "failure_taxonomy_counts": {
            category: overall_taxonomy.get(category, 0) for category in FAILURE_CATEGORIES
        },
        "recorded_requested_region_entry": {
            "field": "cone_entry_step",
            "unit": "recorded simulator action index, not physical time",
            "provenance": (
                "post-result descriptive upstream compact episode field; external raw XYZ traces "
                "were not reopened or rederived"
            ),
            "event_definition": {
                "authenticated_source": ROBOT_BRIDGE,
                "authenticated_git_blob": EXPECTED_SOURCES[ROBOT_BRIDGE]["git_blob"],
                "requested_cone_half_angle_degrees": 45,
                "recorded_entry": "first observed sample inside the requested cone",
                "observed_sustained_entry": "first run of three consecutive in-cone samples",
                "goal_termination_boundary": (
                    "an entry on the final successful sample has no observed three-sample follow-up; "
                    "it is not classified as a transient exit"
                ),
            },
            "conditions": entry_by_condition,
            "layouts": entry_by_layout,
            "totals": {
                "episodes": 108,
                "recorded_entry_events": sum(row["recorded_entry_events"] for row in entry_rows),
                "no_entry_cap_censored": sum(row["no_entry_cap_censored"] for row in entry_rows),
                "observed_sustained_entries": sum(
                    row["observed_sustained_entries"] for row in entry_rows
                ),
                "goal_terminated_entry_without_observed_three_sample_sustained_run": sum(
                    row["goal_terminated_entry_without_observed_three_sample_sustained_run"]
                    for row in entry_rows
                ),
                "successes_with_entry": sum(row["successes_with_entry"] for row in entry_rows),
                "valid_failures_with_entry": sum(row["valid_failures_with_entry"] for row in entry_rows),
            },
            "selection_boundary": (
                "Entry action summaries condition on observed entries. No common-horizon value is invented for "
                "the 42 no-entry traces, and only one control seed has both LEFT and RIGHT entries."
            ),
        },
        "registered_point_estimands": {
            "endpoint_redirection_D_left_minus_right_m": {arm: _summary(d_by_arm[arm]) for arm in ARMS},
            "requested_side_depth_B_right_minus_left_m": {arm: _summary(b_by_arm[arm]) for arm in ARMS},
            "J_reflected_minus_control_endpoint_interaction_m": _summary(j_values),
            "I_reflected_minus_control_requested_depth_interaction_m": _summary(i_values),
            "binary_success_DiD": {
                "values": did_values,
                "distribution": {str(value): did_values.count(value) for value in (-2, -1, 0, 1, 2)},
                "mean": _mean(did_values),
                "median": statistics.median(did_values),
                "exact_layout_label_permutation": permutation,
            },
        },
        "descriptive_execution": {
            "actions_executed_total": sum(action_counts),
            "minimum_actions": min(action_counts),
            "maximum_actions": max(action_counts),
            "right_censored_episodes": sum(bool(row["right_censored"]) for row in episodes.values()),
            "all_four_cells_successful_seed_count": len(complete_case_seeds),
            "all_four_cells_successful_seeds": complete_case_seeds,
        },
        "declared_raw_artifact_inventory": {
            "descriptors": len(artifact_rows),
            "distinct_path_sha_identities": len(artifact_identities),
            "declared_bytes": sum(row["bytes"] for row in artifact_rows),
            "by_role": artifact_by_role,
            "availability_boundary": "immutable compact descriptors only; no current path existence, bytes or SHA-256 were checked",
        },
        "matched_entry_censor_blocks": matched_blocks,
        "audit_operations": {
            "compact_git_objects_byte_authenticated": len(EXPECTED_SOURCES),
            "remote_fetches": 0,
            "raw_pvc_files_opened": 0,
            "raw_pvc_bytes_rehashed": 0,
        },
        "new_activity": activity,
        "gate_state": gates,
        "claim_boundary": {
            "retained_behavior_only": True,
            "historical_post_result_packaging": True,
            "requested_region_entry_is_registered_estimand": False,
            "forecast_accuracy_evidence": False,
            "physical_time_evidence": False,
            "raw_trace_recovery": False,
            "entry_events_raw_trace_rederived": False,
            "entry_action_index_is_physical_time": False,
            "official_D1_evidence": False,
            "default_DreamZero_claim": False,
            "independent_model_noise_replicates": False,
            "new_inference_or_episodes": False,
            "changes_232_cell_core_bound": False,
            "advances_any_scientific_gate": False,
            "statement": (
                "H02 is a retained one-scene-plus-position-reflection behavior cohort for the custom s2 configuration. "
                "Its endpoint/success effects do not establish forecast fidelity, language understanding, official D1 behavior, "
                "independent model-noise replication, current raw availability or current GM/PVC quiescence."
            ),
        },
    }


def render_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


CONDITION_CSV_SUFFIXES = (
    "cell_id",
    "actions_executed",
    "entry_observed",
    "entry_action_index",
    "restricted_waiting_action_index",
    "observed_sustained_entry",
    "goal_terminated_entry_without_observed_three_sample_sustained_run",
    "requested_success",
    "failure_category",
    "right_censored",
)
CONTRAST_CSV_LABELS = (
    "entry_observed",
    "restricted_waiting_action_index",
)
CSV_FIELDS = (
    "seed",
    *(
        f"{arm}_{relation}_{suffix}"
        for arm in ARMS
        for relation in RELATIONS
        for suffix in CONDITION_CSV_SUFFIXES
    ),
    *(
        field
        for label in CONTRAST_CSV_LABELS
        for field in (
            f"control_right_minus_left_{label}",
            f"position_mirrored_right_minus_left_{label}",
            f"position_mirrored_minus_control_left_{label}",
            f"position_mirrored_minus_control_right_{label}",
            f"{label}_difference_in_differences",
        )
    ),
)


def render_csv(report: Mapping[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["matched_entry_censor_blocks"])
    return output.getvalue()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args(argv)
    documents, provenance = read_pinned_sources(ROOT)
    report = analyze_documents(documents, provenance)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(render_json(report), encoding="utf-8")
    args.csv_output.write_text(render_csv(report), encoding="utf-8")
    print(json.dumps({
        "json": str(args.json_output),
        "csv": str(args.csv_output),
        "episodes": report["authenticated_population"]["behavioral_episodes"],
        "pairs": report["authenticated_population"]["matched_left_right_pairs"],
        "new_activity": report["new_activity"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
