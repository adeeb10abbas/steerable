#!/usr/bin/env python3
"""Validate V3-E005 compact evidence, H4 gating, and arena boundaries."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005"
FAILURES = {"correct", "pick_failed", "transport_failed", "wrong_side", "release_failed"}


class Invalid(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Invalid(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve(path: str, *, base: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    repository = ROOT / value
    return repository if repository.exists() else base / value


def canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_interval(item: dict[str, Any], *, confidence: float) -> None:
    require(item.get("resamples") == 20_000, "registered bootstrap count differs")
    require(item.get("clusters") == 7 and item.get("cluster_unit") == "scene_cluster_id", "scene clustering differs")
    require(math.isclose(float(item.get("confidence")), confidence, abs_tol=1e-15), "confidence level differs")
    require(float(item["low"]) <= float(item["point"]) <= float(item["high"]), "interval does not contain point")


def validate(base: Path, *, require_complete: bool, verify_raw_sources: bool) -> dict[str, Any]:
    base = Path(base).resolve()
    results_dir = base / "results"
    paths = {
        "registration": base / "registration.json",
        "queue": base / "queue.jsonl",
        "results": results_dir / "results.json",
        "episodes": results_dir / "episodes.jsonl",
        "pairs": results_dir / "pairs.jsonl",
        "invalid": results_dir / "infrastructure_invalid.jsonl",
        "memo": base / "DECISION_MEMO.md",
        "decision": base / "V3E005_PUBLICATION_DECISION.md",
        "figures": results_dir / "figures/figure_manifest.json",
        "manifest": base / "evidence_manifest.json",
    }
    for path in paths.values():
        require(path.is_file(), f"missing compact E005 evidence: {path}")

    registration = load_json(paths["registration"])
    queue = load_jsonl(paths["queue"])
    results = load_json(paths["results"])
    episodes = load_jsonl(paths["episodes"])
    pairs = load_jsonl(paths["pairs"])
    invalid = load_jsonl(paths["invalid"])
    figures = load_json(paths["figures"])
    manifest = load_json(paths["manifest"])
    memo = paths["memo"].read_text(encoding="utf-8")
    decision = paths["decision"].read_text(encoding="utf-8")

    require(registration.get("amendment_id") == results.get("amendment_id") == "V3-E005", "wrong amendment")
    require(results.get("arena") == manifest.get("arena") == "robotwin", "non-RoboTwin evidence entered E005")
    require(results.get("model_id") == manifest.get("model_id") == "lingbot_va_robotwin", "wrong E005 checkpoint")
    require(results.get("registration_sha256") == sha256(paths["registration"]), "result registration binding differs")
    require(results.get("queue_sha256") == sha256(paths["queue"]), "result queue binding differs")
    require(results.get("bootstrap_resamples") == 20_000, "registered 20,000 resamples not used")
    require(results.get("scene_cluster_count") == 7, "registered scene cluster count differs")
    require(results.get("analysis_order", [None])[0] == "H4", "H4 was not evaluated first")
    require(results.get("h4_gate", {}).get("recorded_before_h1_h3") is True, "H4 order attestation missing")
    require(results.get("margins") == registration["predictions"]["H2"], "registered margins changed")

    queue_ids = {row["cell_id"] for row in queue}
    episode_ids = {row.get("cell_id") for row in episodes}
    require(len(queue) == len(queue_ids) == 108, "registered queue is not 108 unique cells")
    require(len(episodes) == len(episode_ids) == results.get("valid_behavioral_episodes"), "compact episode count differs")
    require(episode_ids <= queue_ids, "unregistered behavioral cell entered E005")
    require(len(pairs) == len({row.get("matched_pair_id") for row in pairs}) == results.get("complete_matched_pairs"), "pair count differs")
    require(len(invalid) == results.get("infrastructure_invalid_attempts"), "infrastructure-invalid count differs")
    require({row.get("arena") for row in episodes} <= {"robotwin"}, "DROID episode entered E005")
    require({row.get("model_id") for row in episodes} <= {"lingbot_va_robotwin"}, "wrong model entered E005")
    require({row.get("arena") for row in pairs} <= {"robotwin"}, "DROID pair entered E005")

    by_pair: dict[tuple[int, float], set[str]] = defaultdict(set)
    for row in episodes:
        require(row.get("missing_measurement_policy") == "NR remains null and is never converted to zero", "NR policy missing")
        require(row.get("failure_category") in FAILURES, "failure taxonomy differs")
        require(type(row.get("success")) is bool, "success is not boolean")
        require((row["failure_category"] == "correct") is row["success"], "success/taxonomy mismatch")
        require(row.get("scene_id") == row.get("scene_cluster_id"), "scene cluster identity drift")
        for field in ("signed_final_lateral_offset", "requested_side_depth", "asymmetry_metric_A"):
            value = row.get(field)
            require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)), f"invalid {field}")
        source = row.get("source_raw_episode", {})
        require(isinstance(source.get("sha256"), str) and len(source["sha256"]) == 64, "raw source digest missing")
        require(isinstance(source.get("bytes"), int) and source["bytes"] >= 0, "raw source byte count missing")
        require(isinstance(source.get("line"), int) and source["line"] >= 1, "raw source line missing")
        if verify_raw_sources:
            source_path = Path(source["path"])
            require(source_path.is_file(), f"raw source unavailable: {source_path}")
            require(source_path.stat().st_size == source["bytes"] and sha256(source_path) == source["sha256"], f"raw source changed: {source_path}")
        artifacts = row.get("raw_artifacts")
        required_artifacts = {
            "result",
            "trajectory",
            "simulator_viewport_video",
            "executed_action_trace",
            "live_reset_snapshot",
        }
        require(isinstance(artifacts, dict) and required_artifacts <= set(artifacts), "raw media/action inventory missing")
        for name in required_artifacts:
            record = artifacts[name]
            require(isinstance(record, dict), f"artifact binding malformed: {name}")
            require(isinstance(record.get("sha256"), str) and len(record["sha256"]) == 64, f"artifact digest missing: {name}")
            require(isinstance(record.get("bytes"), int) and record["bytes"] > 0, f"artifact bytes missing: {name}")
            if verify_raw_sources:
                artifact_path = Path(record["path"])
                require(artifact_path.is_file(), f"raw artifact unavailable: {artifact_path}")
                require(
                    artifact_path.stat().st_size == record["bytes"] and sha256(artifact_path) == record["sha256"],
                    f"raw artifact changed: {artifact_path}",
                )
        level = float(row["symmetry_level_s"])
        if level == 1.0:
            require(float(row["position_residual"]) < 0.001, "s1 position gate failed")
            require(float(row["orientation_residual"]) < math.radians(0.5), "s1 orientation gate failed")
            require(float(row["midline_residual"]) < 0.001, "s1 midline gate failed")
            occlusion = row.get("occlusion_check")
            require((occlusion is False) or (isinstance(occlusion, dict) and occlusion and not any(occlusion.values())), "s1 occlusion gate failed")
            require(row.get("mirrored_asset_identity_verified") is True, "s1 mirrored asset gate failed")
            require(row.get("mirrored_yaw_verified") is True, "s1 mirrored yaw gate failed")
        by_pair[(int(row["environment_seed"]), level)].add(str(row["relation"]))

    complete = results.get("coverage", {}).get("complete") is True
    require(complete is (len(episodes) == 108 and len(pairs) == 54), "coverage flag differs from compact evidence")
    if require_complete:
        require(complete, "registered E005 cohort is incomplete")
    if complete:
        require(episode_ids == queue_ids, "complete evidence does not match queue")
        require(len(by_pair) == 54 and all(value == {"left", "right"} for value in by_pair.values()), "complete paired grid differs")
        require({row["scene_cluster_id"] for row in episodes} == {f"robotwin_pair_{index:02d}" for index in range(3, 10)}, "seven-scene coverage differs")
        gate = results["h4_gate"]
        require(gate.get("outcome") in {"pass", "fail"}, "complete H4 outcome missing")
        for level in ("0.00", "1.00"):
            item = gate["levels"][level]
            require(item.get("pairs") == 27 and len(item.get("seed_level_effects", [])) == 27, "H4 level count differs")
            validate_interval(item["scene_clustered_bootstrap_mean95"], confidence=0.95)
            expected_pass = float(item["mean_m"]) > 0.05 and float(item["scene_clustered_bootstrap_mean95"]["low"]) > 0.0
            require(item.get("pass") is expected_pass, "H4 pass decision differs")
        expected_gate = all(gate["levels"][level]["pass"] for level in ("0.00", "1.00"))
        require(gate.get("hard_gate_passed") is expected_gate, "H4 aggregate gate differs")
        hypotheses = results["hypotheses"]
        if expected_gate:
            require(results["analysis_order"] == ["H4", "H1", "H2", "H3"], "post-H4 analysis order differs")
            require(all(hypotheses[name].get("status", "").startswith("reported_after_h4_pass") for name in ("H1", "H2", "H3")), "H1-H3 not reported after pass")
            require(figures.get("h1_h3_rendered") is True, "H1-H3 figures missing after H4 pass")
            h1 = hypotheses["H1"]
            for key in ("binary", "requested_depth_m"):
                validate_interval(h1["interaction_s1_minus_s0"][key]["scene_clustered_bootstrap_mean95"], confidence=0.95)
                require("exact_within_seed_layout_label_permutation" in h1["interaction_s1_minus_s0"][key], "exact H1 test missing")
            require(hypotheses["H2"]["binary"]["publication_equivalence_claim_allowed"] is False, "binary equivalence overclaim")
            require(hypotheses["H2"]["requested_depth_m"]["publication_equivalence_claim_allowed"] is False, "depth equivalence overclaim")
        else:
            require(results["analysis_order"] == ["H4"], "H1-H3 were computed after H4 failure")
            for name in ("H1", "H2", "H3"):
                require(hypotheses[name] == {"status": "withheld_due_h4_failure", "estimands_reported": False}, f"{name} leaked after H4 failure")
            require(figures.get("h1_h3_rendered") is False, "H1-H3 figure leaked after H4 failure")
            require(not any("v3e005_h1" in item["path"] or "v3e005_h3" in item["path"] for item in figures.get("figures", [])), "withheld figure present")
            combined_text = memo + "\n" + decision
            require("withheld" in combined_text.lower(), "H4-fail decision does not state withholding")
    else:
        require(results.get("h4_gate", {}).get("outcome") == "not_evaluable_incomplete", "partial H4 was evaluated")
        require(results.get("analysis_order") == ["H4"], "partial H1-H3 analysis leaked")
        require(figures.get("h1_h3_rendered") is False, "partial H1-H3 figure leaked")

    require(figures.get("arena") == "robotwin", "figure manifest crossed arena boundary")
    require(figures.get("scientific_boundaries", {}).get("droid_imported_or_pooled") is False, "figure manifest pools DROID")
    require(manifest.get("registration_sha256") == sha256(paths["registration"]), "manifest registration hash differs")
    require(manifest.get("queue_sha256") == sha256(paths["queue"]), "manifest queue hash differs")
    require(manifest.get("results_sha256") == sha256(paths["results"]), "manifest result hash differs")
    require(manifest.get("episodes_sha256") == sha256(paths["episodes"]), "manifest episode hash differs")
    require(manifest.get("pairs_sha256") == sha256(paths["pairs"]), "manifest pair hash differs")
    require(manifest.get("status") == ("hash_closed_compact_evidence" if complete else "partial_progress_not_publication_evidence"), "manifest status differs")
    require(manifest.get("scientific_boundaries", {}).get("droid_imported_or_pooled") is False, "evidence manifest pools DROID")
    source_identities = {
        (row["source_raw_episode"]["sha256"], row["source_raw_episode"]["bytes"], row["source_raw_episode"]["path"])
        for row in episodes
    }
    require(manifest.get("raw_source_count") == len(source_identities), "raw source inventory count differs")
    seed_records = manifest.get("whole_seed_manifests", [])
    require(manifest.get("whole_seed_manifest_count") == len(seed_records), "whole-seed count differs")
    if complete:
        require(len(seed_records) == 27 and {item.get("seed") for item in seed_records} == set(range(9400, 9427)), "complete whole-seed coverage differs")
    for item in seed_records:
        path = resolve(item["path"], base=base)
        require(path.is_file(), f"whole-seed manifest missing: {path}")
        require(path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"], f"whole-seed manifest changed: {path}")
        marker = load_json(path)
        require(marker.get("schema_version") == "vla-wam-shared-v3e005-lingbot-whole-seed-completion-v1", "whole-seed schema differs")
        require(marker.get("status") == "complete_four_valid_behavioral_cells", "whole-seed status differs")
        require(marker.get("behavioral_episode_count") == 4 and marker.get("matched_pair_count") == 2, "whole-seed cardinality differs")
        expected_ids = {row["cell_id"] for row in episodes if row["environment_seed"] == marker.get("seed")}
        require(set(marker.get("cell_ids", [])) == expected_ids, "whole-seed cell binding differs")
        supplied_marker_sha = marker.pop("marker_sha256", None)
        require(supplied_marker_sha == canonical_sha256(marker), "whole-seed self-hash differs")
    pair_file_records = manifest.get("whole_seed_pair_files", [])
    require(manifest.get("whole_seed_pair_file_count") == len(pair_file_records), "whole-seed pair-file count differs")
    if complete:
        require(len(pair_file_records) == 54, "complete whole-seed pair-file coverage differs")
    for item in pair_file_records:
        path = resolve(item["path"], base=base)
        require(path.is_file(), f"whole-seed pair file missing: {path}")
        require(path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"], f"whole-seed pair file changed: {path}")
        pair = load_json(path)
        require(pair.get("schema_version") == "vla-wam-shared-v3e005-lingbot-robotwin-pair-v1", "whole-seed pair schema differs")
        pair_sha = pair.pop("pair_sha256", None)
        require(pair_sha == canonical_sha256(pair), "whole-seed pair self-hash differs")
    for item in manifest.get("compact_files", []) + manifest.get("implementation_files", []):
        path = resolve(item["path"], base=base)
        require(path.is_file(), f"manifested file missing: {path}")
        require(path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"], f"manifested file changed: {path}")

    return {
        "status": "valid_complete" if complete else "valid_partial_no_publication_claims",
        "valid_behavioral_episodes": len(episodes),
        "complete_matched_pairs": len(pairs),
        "infrastructure_invalid_attempts": len(invalid),
        "h4_outcome": results["h4_gate"]["outcome"],
        "raw_sources_verified": verify_raw_sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--verify-raw-sources", action="store_true")
    args = parser.parse_args()
    try:
        report = validate(args.base, require_complete=args.require_complete, verify_raw_sources=args.verify_raw_sources)
    except (Invalid, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2))
        raise SystemExit(1)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
