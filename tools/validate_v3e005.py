#!/usr/bin/env python3
"""Validate the prospective or completed V3-E005 evidence bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from tools.validate_v3e005_evidence import validate as validate_compact_evidence
except ModuleNotFoundError:  # Direct ``python tools/validate_v3e005.py`` invocation.
    from validate_v3e005_evidence import validate as validate_compact_evidence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def validate_registration() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registration_path = BASE / "registration.json"
    queue_path = BASE / "queue.jsonl"
    require(registration_path.is_file(), f"missing {registration_path}")
    require(queue_path.is_file(), f"missing {queue_path}")
    registration = load_json(registration_path)
    queue = load_jsonl(queue_path)
    require(registration["amendment_id"] == "V3-E005", "wrong amendment id")
    require(registration["separate_amendment_not_e004_extension"] is True, "E005 is not marked separate")
    require(registration["model_request_count_before_registration"] == 0, "registration followed a model request")
    require(registration["behavioral_episode_count_before_registration"] == 0, "registration followed behavior")
    require(registration["design"]["seeds"] == list(range(9400, 9427)), "seed list drift")
    require(len(queue) == 108 and len({r["cell_id"] for r in queue}) == 108, "queue must contain 108 unique cells")
    require(digest(queue_path) == registration["queue"]["sha256"], "queue hash drift")
    require({r["model_id"] for r in queue} == {"lingbot_va_robotwin"}, "wrong checkpoint in queue")
    require({r["arena"] for r in queue} == {"robotwin"}, "wrong arena in queue")
    require({r["symmetry_level_s"] for r in queue} == {0.0, 1.0}, "wrong levels")
    require({r["relation"] for r in queue} == {"left", "right"}, "wrong directions")
    counts = Counter((r["symmetry_level_s"], r["relation"]) for r in queue)
    require(set(counts.values()) == {27} and len(counts) == 4, "not 27 cells per level/direction")
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in queue:
        by_seed[int(row["environment_seed"])].append(row)
        require(row["environment_seed"] == row["sampling_seed"], "environment/sampling seed mismatch")
        require(row["static_episode_prompt"] is True, "non-static prompt")
        require(row["success_predicate_id"] == "frozen_v3_robotwin_relation_aware_success", "predicate drift")
    require(set(by_seed) == set(range(9400, 9427)), "queue seed set drift")
    for seed, rows in by_seed.items():
        require(len(rows) == 4, f"seed {seed} is not a four-cell atomic block")
        require(len({r["scene_id"] for r in rows}) == 1, f"seed {seed} crosses scenes")
        require({(r["symmetry_level_s"], r["relation"]) for r in rows} == {(0.0, "left"), (0.0, "right"), (1.0, "left"), (1.0, "right")}, f"seed {seed} grid incomplete")
    h4 = registration["predictions"]["H4"]
    require(h4["threshold_m"] == 0.05 and "CI excludes zero" in h4["pass_at_each_level"], "H4 threshold drift")
    h2 = registration["predictions"]["H2"]
    require(h2["binary"]["margin"] == 0.0, "binary margin drift")
    require(h2["requested_depth_m"]["margin"] == 0.0051649959743141185, "depth margin drift")
    require(h2["binary"]["mde80_n27"] == 0.3646343647921233, "binary MDE drift")
    require(h2["requested_depth_m"]["mde80_n27"] == 0.47591192743266003, "depth MDE drift")
    for relative, expected in registration["source_bindings"].items():
        require(digest(ROOT / relative) == expected, f"source hash drift: {relative}")
    return registration, queue


def validate_results(registration: dict[str, Any], queue: list[dict[str, Any]]) -> None:
    results_path = BASE / "results/results.json"
    episodes_path = BASE / "results/episodes.jsonl"
    require(results_path.is_file() and episodes_path.is_file(), "completed results required but absent")
    results = load_json(results_path)
    episodes = load_jsonl(episodes_path)
    require(len(episodes) == 108 and len({r["cell_id"] for r in episodes}) == 108, "results lack 108 unique behavioral cells")
    require({r["cell_id"] for r in episodes} == {r["cell_id"] for r in queue}, "results/queue cell mismatch")
    counts = Counter((r["symmetry_level_s"], r["relation"]) for r in episodes)
    require(set(counts.values()) == {27}, "results not 27 per level/direction")
    for row in episodes:
        if row["symmetry_level_s"] == 1.0:
            require(row["position_residual"] < 0.001, "position tolerance failure")
            require(row["orientation_residual"] < 0.008726646259971648, "orientation tolerance failure")
            require(row["midline_residual"] < 0.001, "midline tolerance failure")
            require(row["occlusion_check"] is False, "target occluded")
            require(row.get("mirrored_asset_identity_verified") is True, "mirrored asset identity failure")
            require(row.get("mirrored_yaw_verified") is True, "mirrored yaw failure")
    require(results["registration_sha256"] == digest(BASE / "registration.json"), "results registration binding mismatch")
    require(results["queue_sha256"] == digest(BASE / "queue.jsonl"), "results queue binding mismatch")
    require(results["margins"] == registration["predictions"]["H2"], "compiled margins differ from registration")
    require(results["analysis_order"][0] == "H4", "H4 was not compiled first")
    require(results["h4_gate"]["recorded_before_h1_h3"] is True, "H4 outcome order not attested")
    memo = (BASE / "DECISION_MEMO.md").read_text()
    require(str(results["h4_gate"]["outcome"]) in memo, "memo inconsistent with H4 outcome")
    manifest = load_json(BASE / "evidence_manifest.json")
    require(manifest["registration_sha256"] == digest(BASE / "registration.json"), "manifest registration mismatch")
    require(manifest["results_sha256"] == digest(results_path), "manifest results mismatch")
    compact = validate_compact_evidence(BASE, require_complete=True, verify_raw_sources=False)
    require(compact["status"] == "valid_complete", "compact evidence validator did not close E005")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-results", action="store_true")
    args = parser.parse_args()
    registration, queue = validate_registration()
    if args.require_results:
        validate_results(registration, queue)
        print("valid_complete: 108 behavioral cells")
    else:
        print("valid_registered: 108 cells; no E005 behavior required")


if __name__ == "__main__":
    main()
