#!/usr/bin/env python3
"""Validate V3-E004 registration now and completed evidence when requested."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.layout_contract import (  # noqa: E402
    load_candidate,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.fastwam_robotwin import (  # noqa: E402
    load_candidate as load_fastwam_candidate,
)

BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"
CORE_SEEDS = set(range(9400, 9427))
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
EXPECTED = {
    "pi05_current_stack_droid": {0.0: 341, 0.25: 27, 0.5: 27, 0.75: 27, 1.0: 341},
    "cosmos3_nano_policy_droid": {0.0: 521, 0.25: 27, 0.5: 27, 0.75: 27, 1.0: 521},
    "dreamzero_droid_action_cfg": {0.0: 27, 1.0: 27},
    "cosmos3_edge_policy_droid": {0.0: 27, 1.0: 27},
    "fastwam_robotwin": {0.0: 27, 1.0: 27},
}


class Invalid(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Invalid(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_registration() -> dict[str, Any]:
    registration_path = BASE / "registration.json"
    queue_path = BASE / "queue.jsonl"
    candidate_path = BASE / "layout/candidate.json"
    fastwam_candidate_path = BASE / "layout/fastwam_robotwin_candidate.json"
    static_gate_path = BASE / "gates/static_layout_gate.json"
    request0_amendment_path = BASE / "request0_observation_replay_amendment.json"
    for path in (
        registration_path,
        queue_path,
        candidate_path,
        fastwam_candidate_path,
        static_gate_path,
        request0_amendment_path,
    ):
        require(path.is_file(), f"missing E004 preregistration file: {path}")

    registration = finite_json(registration_path)
    require(registration.get("amendment_id") == "V3-E004", "wrong amendment")
    require(registration.get("supersedes") == "V3-E003", "E003 supersession not declared")
    require(registration.get("model_request_count_before_registration") == 0, "registration follows model request")
    require(registration.get("behavioral_episode_count_before_registration") == 0, "registration follows episode")
    require(registration.get("success_predicates_frozen") is True, "predicate is not frozen")
    require(registration.get("design", {}).get("droid_and_robotwin_never_pooled") is True, "arena pooling boundary missing")
    request0_amendment = finite_json(request0_amendment_path)
    require(
        request0_amendment.get("schema_version")
        == "vla-wam-shared-v3e004-request0-observation-replay-amendment-v1",
        "R001 request-zero amendment schema changed",
    )
    require(request0_amendment.get("registered_before_new_request") is True, "R001 was not registered prospectively")
    require(request0_amendment.get("registration_sha256") == sha256_file(registration_path), "R001 registration binding changed")
    require(request0_amendment.get("queue_sha256") == sha256_file(queue_path), "R001 queue binding changed")
    require(request0_amendment.get("candidate_sha256") == sha256_file(candidate_path), "R001 candidate binding changed")
    invalid_discovery = request0_amendment.get("invalid_discovery_attempts", [])
    require(len(invalid_discovery) == 4, "R001 pre-amendment exclusion inventory changed")
    require(
        {row.get("cell_id") for row in invalid_discovery}
        == {
            "v3e004:pi05:seed9400:s100:left",
            "v3e004:dreamzero:seed9400:s100:left",
            "v3e004:nano:seed9400:s100:left",
            "v3e004:nano:seed9400:s100:right",
        },
        "R001 excluded cell inventory changed",
    )
    require(
        all("excluded from all matched-pair denominators" in row.get("behavioral_use", "") for row in invalid_discovery),
        "R001 pre-amendment rows are not excluded from matched analyses",
    )

    queue_record = registration.get("queue", {})
    require(queue_record.get("sha256") == sha256_file(queue_path), "queue hash mismatch")
    require(queue_record.get("bytes") == queue_path.stat().st_size, "queue byte count mismatch")
    candidate_record = registration.get("layout", {})
    require(candidate_record.get("candidate_sha256") == sha256_file(candidate_path), "candidate hash mismatch")
    candidate = load_candidate(candidate_path, candidate_record["candidate_sha256"])
    fastwam_sha256 = candidate_record.get("robotwin_stretch_candidate_sha256")
    require(
        fastwam_sha256 == sha256_file(fastwam_candidate_path),
        "FastWAM stretch candidate hash mismatch",
    )
    load_fastwam_candidate(fastwam_candidate_path, fastwam_sha256)

    rows = jsonl(queue_path)
    require(len(rows) == 4096, "queue is not the registered 4,096 cells")
    require(len({row["cell_id"] for row in rows}) == len(rows), "duplicate cell id")
    require(sum("historical_control_comparator_cell_id" in row for row in rows) == 162, "historical comparator links changed")
    require(sum(row["execution_mode"] == "new_behavioral_episode" for row in rows) == 4096, "new episode count changed")

    counts: Counter[tuple[str, float, str]] = Counter()
    seed_sets: defaultdict[tuple[str, float, str], set[int]] = defaultdict(set)
    for row in rows:
        model = row["model_id"]
        level = float(row["symmetry_level_s"])
        relation = row["relation"]
        require(model in EXPECTED and level in EXPECTED[model], f"unregistered cell: {model}/{level}")
        require(relation in ("left", "right"), "unknown relation")
        if row["arena"] == "droid_robolab":
            require(row["prompt"] == PROMPTS[relation], "DROID prompt bytes changed")
            require(row["layout_candidate_sha256"] == candidate_record["candidate_sha256"], "DROID candidate binding changed")
            expected_A = candidate.to_json()["levels"][f"{level:.2f}"]["asymmetry_metric_A"]
            require(math.isclose(row["registered_expected_asymmetry_A"], expected_A, abs_tol=1e-15), "registered A changed")
        else:
            require(
                row["layout_candidate_sha256"] == fastwam_sha256,
                "FastWAM candidate binding changed",
            )
        counts[(model, level, relation)] += 1
        seed_sets[(model, level, relation)].add(int(row["environment_seed"]))
        require(row["environment_seed"] == row["sampling_seed"], "environment/sampling seed mismatch")
        if "historical_control_comparator_cell_id" in row:
            require(level == 0.0 and int(row["environment_seed"]) in CORE_SEEDS, "invalid historical comparator link")
            require(row.get("historical_control_comparator_not_an_e004_cell") is True, "comparator is mislabeled as E004 evidence")
        require(row["execution_mode"] == "new_behavioral_episode", "E004 queue contains a non-new episode")

    for model, levels in EXPECTED.items():
        for level, pairs in levels.items():
            expected_seeds = set(range(9400, 9400 + pairs))
            for relation in ("left", "right"):
                require(counts[(model, level, relation)] == pairs, f"wrong count: {model}/{level}/{relation}")
                require(seed_sets[(model, level, relation)] == expected_seeds, f"wrong seeds: {model}/{level}/{relation}")

    full = candidate.layout(1.0)
    residuals = candidate.residuals(full)
    require(residuals["position_residual_m"] < 0.001, "s=1 position residual fails")
    require(residuals["orientation_residual_rad"] < math.radians(0.5), "s=1 yaw residual fails")
    require(residuals["midline_residual_m"] < 0.001, "s=1 midline residual fails")
    require(full["banana"].asset_identity == full["banana_right"].asset_identity, "clutter pair asset differs")
    require(math.isclose(full["banana"].yaw_rad, -full["banana_right"].yaw_rad, abs_tol=1e-15), "clutter yaw is not reflected")
    require(set(candidate.layout(0.0)) == {"banana", "bowl", "rubiks_cube"}, "s0 is not exact B001 inventory")
    require(set(candidate.layout(0.25)) == {"banana", "banana_right", "bowl", "rubiks_cube"}, "companion policy changed")

    static_gate = finite_json(static_gate_path)
    require(static_gate.get("passed") is True and static_gate.get("model_request_count") == 0, "static model-blind gate failed")
    require(static_gate.get("candidate_sha256") == candidate_record["candidate_sha256"], "static gate candidate changed")

    power_rows = registration.get("power_registration", {}).get("rows", [])
    require(len(power_rows) == 10, "power table is incomplete")
    for row in power_rows:
        require(math.isclose(row["margin"], 0.20 * abs(row["control_effect"]), abs_tol=1e-10), "margin is not 0.20 of control effect")
        if row["status"] == "strictly_powered_at_endpoints":
            require(row["target_n"] >= row["strict_n"], "powered endpoint target below strict n")
        if row["status"].startswith("underpowered"):
            require(row["target_n"] < row["strict_n"], "underpowered label inconsistent")
    nano_binary = [row for row in power_rows if row["model_id"] == "cosmos3_nano_policy_droid" and row["estimand"] == "binary_R_minus_L"][0]
    require(nano_binary["margin"] == 0.0 and nano_binary["strict_n"] is None, "Nano zero-margin boundary changed")

    for relative, record in registration.get("source_bindings", {}).items():
        source = ROOT / relative
        require(source.is_file(), f"missing source: {relative}")
        require(record["sha256"] == sha256_file(source), f"source hash changed: {relative}")
        require(record["bytes"] == source.stat().st_size, f"source bytes changed: {relative}")

    return {
        "registration": str(registration_path),
        "queue_rows": len(rows),
        "historical_comparator_links": 162,
        "new_behavioral_cells": 4096,
        "candidate_sha256": candidate_record["candidate_sha256"],
        "static_gate_sha256": sha256_file(static_gate_path),
        "request0_amendment_sha256": sha256_file(request0_amendment_path),
    }


def validate_results() -> dict[str, Any]:
    results_path = BASE / "results/results.json"
    episodes_path = BASE / "results/episodes.jsonl"
    invalid_path = BASE / "results/infrastructure_invalid.jsonl"
    manifest_path = BASE / "evidence_manifest.json"
    memo_path = BASE / "DECISION_MEMO.md"
    for path in (results_path, episodes_path, invalid_path, manifest_path, memo_path):
        require(path.is_file(), f"missing completed result: {path}")
    results = finite_json(results_path)
    episodes = jsonl(episodes_path)
    require(results.get("status") == "complete_hash_closed", "results are not complete")
    require(len(episodes) == 4096, "completed evidence does not contain 4,096 behavioral cells")
    require(len({row["cell_id"] for row in episodes}) == 4096, "completed episodes duplicate a cell")
    required = {
        "success", "failure_category", "signed_final_lateral_offset", "requested_side_depth",
        "cone_entry_step", "cone_entry_sustained", "episode_length", "symmetry_level_s",
        "asymmetry_metric_A", "position_residual", "orientation_residual", "midline_residual",
        "occlusion_check", "realised_object_poses", "arm_reset_pose",
    }
    for row in episodes:
        require(required <= set(row), f"episode fields missing: {row.get('cell_id')}")
        if row.get("arena") == "droid_robolab":
            require(
                row.get("request0_replay", {}).get("schema_version")
                == "vla-wam-shared-v3e004-request0-evidence-envelope-v1",
                f"DROID episode lacks R001 evidence: {row.get('cell_id')}",
            )
            require(
                isinstance(row.get("request0_pair_identity_sha256"), str),
                f"DROID episode lacks request-zero pair identity: {row.get('cell_id')}",
            )
        if float(row["symmetry_level_s"]) == 1.0:
            require(row["position_residual"] < 0.001, "completed s=1 position residual fails")
            require(row["orientation_residual"] < math.radians(0.5), "completed s=1 yaw residual fails")
            require(row["midline_residual"] < 0.001, "completed s=1 midline residual fails")
            check = row["occlusion_check"]
            require(isinstance(check, dict) and check and all(value is False for value in check.values()), "completed s=1 occlusion gate fails")
    manifest = finite_json(manifest_path)
    require(manifest.get("registration_sha256") == sha256_file(BASE / "registration.json"), "manifest registration hash mismatch")
    require(manifest.get("results_sha256") == sha256_file(results_path), "manifest results hash mismatch")
    return {"results": str(results_path), "episodes": len(episodes), "invalid_attempts": len(jsonl(invalid_path))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-results", action="store_true")
    args = parser.parse_args()
    try:
        report = {"registration": validate_registration()}
        if args.require_results:
            report["results"] = validate_results()
        report["status"] = "valid_complete" if args.require_results else "valid_registered_not_yet_complete"
    except (Invalid, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2))
        raise SystemExit(1)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
