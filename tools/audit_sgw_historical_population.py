"""Audit expanded reset/lifecycle records, not exhaustive historical coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.workshops.spatial_grounding_v1.historical_layout_streaming import OBJECT_FIELDS
from tools.audit_sgw_historical_lineage import audit as audit_lineage
from tools.audit_sgw_historical_lineage import digest, require, selected


def pointer_map(rows: list[dict]) -> dict:
    result = {row["json_pointer"]: row["value"] for row in rows}
    require(len(result) == len(rows), "duplicate selected pointer")
    return result


def supplement_preflight(repo: Path, infra: Path, export: Path, row: dict, current: dict) -> dict:
    inventory = json.loads((infra / "historical-droid-layout-source-inventory-v2.json").read_text())
    cohort = next(item for item in inventory["cohorts"] if item["cohort_id"] == row["cohort_id"])
    closure_raw = (repo / cohort["source_path"]).read_bytes()
    require(digest(closure_raw) == cohort["source_sha256"], "R012 closure manifest differs")
    evidence = json.loads(closure_raw)["raw_evidence"]
    require(evidence["child_result"]["sha256"] == row["expected_sha256"]
            and evidence["child_result"]["bytes"] == row["expected_bytes"],
            "R012 preflight and streamed execution differ")
    binding = evidence["geometry_attachment_preflight"]
    path = export / "r012-geometry-preflight.json"
    raw = path.read_bytes()
    require(len(raw) == binding["bytes"] and digest(raw) == binding["sha256"],
            "supplementary preflight byte/hash mismatch")
    value = json.loads(raw)
    prefix = "/geometry_attachment_preflight"
    fresh = value["fresh_reset"]
    current["fresh_reset_objects"].extend({
        "json_pointer": prefix + "/fresh_reset/objects/" + actor.replace("~", "~0").replace("/", "~1"),
        "value": {key: entry[key] for key in OBJECT_FIELDS if key in entry},
    } for actor, entry in fresh["objects"].items())
    for group, suffix, entry in (
        ("frame_identity", "/fresh_reset/base_link_to_eef_frame_identity",
         fresh["base_link_to_eef_frame_identity"]),
        ("snapshot_environment_bindings", "/environment_lifecycle", value["environment_lifecycle"]),
    ):
        current[group].append({"json_pointer": prefix + suffix, "value": entry})
    return {
        "export_path": path.name, "binding": binding,
        "closure_manifest": cohort["source_path"], "closure_manifest_sha256": digest(closure_raw),
        "scope": "Separately hash-bound preflight receipt from the same named execution, not another run.",
    }


def audit(repo: Path, export: Path) -> dict:
    infra = repo / "artifacts/workshops/spatial_grounding_v1/infrastructure"
    prior = infra / "historical-lineage-20260923bv"
    inputs = repo / "handoff/k8s/sgw01-ali-historical-population-inputs-20260923bx.json"
    config = json.loads(inputs.read_text())["data"]
    request = json.loads(config["requests.json"])
    prior_audit = audit_lineage(
        repo, prior, infra / "historical-state-fields-20260923bl",
        repo / "handoff/k8s/sgw01-ali-historical-lineage-inputs-20260923bv.json",
    )
    prior_raw = (prior / "manifest.json").read_bytes()
    require(digest(prior_raw) == request["prior_lineage_manifest_sha256"], "prior lineage manifest mismatch")
    manifest_raw = (export / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    for key, filename in (
        ("request_sha256", "requests.json"), ("extractor_sha256", "historical_layout_streaming.py"),
        ("launcher_sha256", "run.py"),
    ):
        require(manifest[key] == digest(config[filename].encode()), f"{key} mismatch")
    require(request["release_authorization"] is False
            and request["retain_source_lineage"] is True
            and request["retain_population_metadata"] is True, "unexpected extraction authority")
    require(manifest["model_requests"] == manifest["behavioral_episodes"] == manifest["allocated_gpus"] == 0
            and manifest["release_permitted"] is False, "unexpected export authority")
    old_records = json.loads(prior_raw)["records"]
    require(len(manifest["records"]) == len(old_records) == len(request["records"]), "source inventory differs")
    records = []
    for row, old, expected in zip(manifest["records"], old_records, request["records"], strict=True):
        for key in ("cohort_id", "path", "expected_sha256", "expected_bytes"):
            require(row[key] == old[key] == expected[key], f"source inventory {key} mismatch")
        current, previous = selected(export, row), selected(prior, old)
        require(current["selection_contract"] == request["selection_contract"], "selection contract mismatch")
        for group in (
            "fresh_reset_objects", "candidate_state_objects", "full_reset_comparisons", "reference_bounds",
            "geometry_preflight_identity", "source_lineage", "frame_identity",
        ):
            new_map, old_map = pointer_map(current[group]), pointer_map(previous[group])
            require(all(key in new_map and new_map[key] == value for key, value in old_map.items()),
                    f"prior {group} changed or missing")
        require(current["source_lineage"] == previous["source_lineage"], "source lineage differs")
        supplementary = None
        if row["cohort_id"] == "V3-E006-repair-r012":
            supplementary = supplement_preflight(repo, infra, export, row, current)
        lifecycles = pointer_map(current["environment_lifecycle"])
        require(set(lifecycles) == {"/execution_evidence/environment_lifecycle"}, "lifecycle inventory differs")
        lifecycle_rows = lifecycles["/execution_evidence/environment_lifecycle"]
        require(all(type(item["environment_ordinal"]) is int for item in lifecycle_rows),
                "invalid environment ordinal")
        by_ordinal = {item["environment_ordinal"]: item for item in lifecycle_rows}
        require(len(by_ordinal) == len(lifecycle_rows)
                and set(by_ordinal) == set(range(1, len(lifecycle_rows) + 1)), "lifecycle ordinal gap/duplicate")
        bindings = pointer_map(current["snapshot_environment_bindings"])
        fresh = {item["json_pointer"].rsplit("/objects/", 1)[0]
                 for item in current["fresh_reset_objects"]}
        candidates = {item["json_pointer"].rsplit("/objects/", 1)[0]
                      for item in current["candidate_state_objects"]}
        frames = pointer_map(current["frame_identity"])
        require(set(frames) == {
            prefix + "/base_link_to_eef_frame_identity" for prefix in fresh | candidates
        }, "snapshot/frame inventory differs")
        require(set(bindings) == {
            prefix.removesuffix("/fresh_reset") + "/environment_lifecycle" for prefix in fresh
        }, "reset/lifecycle binding inventory differs")
        bound_ordinals = []
        for binding in bindings.values():
            ordinal = binding["environment_ordinal"]
            require(type(ordinal) is int and ordinal in by_ordinal and binding == by_ordinal[ordinal],
                    "snapshot lifecycle differs from recorded global lifecycle")
            bound_ordinals.append(ordinal)
        require(len(bound_ordinals) == len(set(bound_ordinals))
                and set(bound_ordinals) == set(by_ordinal), "recorded lifecycle lacks a unique reset snapshot")
        for lifecycle in lifecycle_rows:
            require(all(lifecycle[key] is True for key in (
                "created", "fresh_reset_completed_in_this_environment", "closed_before_next_environment",
            )), "recorded lifecycle did not complete")
        outcomes = pointer_map(current["stage_outcomes"])
        accounting = pointer_map(current["population_accounting"])
        old_fresh = {item["json_pointer"].rsplit("/objects/", 1)[0]
                     for item in previous["fresh_reset_objects"]}
        records.append({
            "cohort_id": row["cohort_id"], "raw_source": row["path"],
            "raw_source_sha256": row["expected_sha256"], "selected_export_sha256": row["export_sha256"],
            "prior_selections_preserved": True,
            "recorded_environment_count": len(by_ordinal),
            "recorded_lifecycles_each_have_unique_reset_snapshot": True,
            "fresh_reset_snapshot_count": len(fresh), "candidate_state_snapshot_count": len(candidates),
            "additional_reset_count": len(fresh - old_fresh),
            "new_materialization_reset_count": sum(
                prefix.endswith("/materialization_environment/fresh_reset") for prefix in fresh - old_fresh
            ),
            "new_reset_pointers": sorted(fresh - old_fresh),
            "supplementary_preflight": supplementary,
            "frame_identity_count": len(frames),
            "all_frame_identity_checks_passed": all(value["passed"] is True for value in frames.values()),
            "observed_scene_environment_origins_world_m": sorted({
                tuple(value["scene_env_origin_world_m"]) for value in frames.values()
            }),
            "population_accounting": accounting, "stage_outcomes": outcomes,
        })
    ledger_path = repo / (
        "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r005/"
        "infrastructure_attempts.jsonl"
    )
    gate = json.loads((ledger_path.parent / "source_push_gate_v2.json").read_text())
    ledger_raw = ledger_path.read_bytes()
    require(digest(ledger_raw) == gate["infrastructure_attempts"]["sha256"]
            and len(ledger_raw) == gate["infrastructure_attempts"]["bytes"], "prior-attempt ledger binding differs")
    attempts = [json.loads(line) for line in ledger_raw.splitlines() if line]
    blockers = [{
        "attempt_id": attempt["attempt_id"], "source": attempt["source"],
        "raw_failure_report": attempt["raw_bindings"]["failure_report"],
        "recorded_completeness": attempt["completeness"],
    } for attempt in attempts]
    require(bool(blockers), "known earlier infrastructure attempt disappeared")
    return {
        "schema_version": "sgw-01-historical-reset-population-audit-v1",
        "export_manifest_sha256": digest(manifest_raw),
        "prior_lineage_manifest_sha256": digest(prior_raw),
        "prior_lineage_git_bindings_verified": sum(
            len(item["verified_git_bindings"]) for item in prior_audit["records"]
        ),
        "inputs_manifest_sha256": digest(inputs.read_bytes()),
        "auditor_sha256": digest(Path(__file__).read_bytes()),
        "records": records,
        "known_uncovered_infrastructure_attempts": blockers,
        "prior_attempt_ledger_sha256": digest(ledger_raw),
        "model_requests": 0, "behavioral_episodes": 0,
        "historical_population_coverage_complete": False, "release_permitted": False,
        "claim_boundary": (
            "Every retained lifecycle in these seven named executions has a matching retained "
            "fresh reset; prior selections are preserved. This is not a proof of every historical "
            "execution, intermediate constructed layout, asset dependency or geometry. R005 "
            "attempt01 is separate: its ledger records 20 completed environments and missing "
            "rank1-3 scientific state payloads. Zero accepted states or zero model requests do "
            "not exclude those layouts. No fixtures or model requests are released."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[1], args.export_root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
