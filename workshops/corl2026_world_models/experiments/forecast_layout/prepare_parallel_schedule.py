#!/usr/bin/env python3
"""Prepare immutable, unreleased four-condition jobs; never execute a policy.

Only standard-library file and read-only Git operations are used. A job is a
scientific block, not a worker allocation or an executable launch command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import subprocess


NAMESPACE = "wmf_ablation_001_20260912"
UPSTREAM = "ce561e66f82e95055e39d3d7711691982f6b2086"
CONDITIONS = ("original-left", "original-right", "reflected-left", "reflected-right")
PHASE_BLOCKS = {
    "pilot": ["P00"],
    "development": [f"D{i:02d}" for i in range(1, 5)],
    "confirmation": [f"C{i:02d}" for i in range(1, 25)],
}
MODELS = ("N3", "D1")
SEED_KEY = re.compile(r"(?:^|_)(?:seed|seeds)(?:_|$)", re.IGNORECASE)


def canonical_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def candidate_seed(phase, block):
    if phase == "pilot":
        return 2026091000
    return (2026091100 if phase == "development" else 2026091200) + int(block[1:])


def validate_inputs(spec, rows):
    """Reject changes to the fixed core matrix rather than silently adapting it."""
    if spec.get("spec_id") != "WMF-ABLATION-001" or spec.get("version") != "1.0":
        raise ValueError("this preparer requires WMF-ABLATION-001 version 1.0")
    if spec.get("upstream_pin") != UPSTREAM or spec.get("core_episode_ceiling") != 232:
        raise ValueError("fixed source pin or 232-cell ceiling changed")
    if spec.get("launch_ready") is not False or spec.get("status") != "DRAFT_NOT_RELEASED":
        raise ValueError("preparation requires the unreleased draft specification")
    if spec.get("models", {}).get("D2", {}).get("selected") is not False:
        raise ValueError("optional D2 is excluded from the core schedule")
    episode = spec.get("episode", {})
    if (episode.get("action_cap"), episode.get("stop_on_success"), episode.get("record_first_success")) != (450, False, True):
        raise ValueError("goal-independent 450-action stopping contract changed")
    for phase, blocks in PHASE_BLOCKS.items():
        stage = spec.get("stages", {}).get(phase, {})
        if (stage.get("layout_pairs") != len(blocks)
                or stage.get("episodes") != len(blocks) * 8
                or set(stage.get("models", [])) != set(MODELS)):
            raise ValueError(f"fixed stage factorial changed: {phase}")
    ids = [row.get("cell_id") for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate cell IDs")
    if len(rows) != 232:
        raise ValueError("exactly 232 cells are required; missing or extra cells")
    indexed = {}
    for row in rows:
        phase, block, model = (row.get(key) for key in ("phase", "layout_pair_id", "model_config"))
        condition = f"{row.get('layout_arm')}-{row.get('command')}"
        if phase not in PHASE_BLOCKS or block not in PHASE_BLOCKS[phase]:
            raise ValueError("cell has an incorrect phase or layout block")
        if model not in MODELS or condition not in CONDITIONS:
            raise ValueError("cell is outside the core model/condition factorial")
        expected_id = f"wmf1__{phase}__{block}__{model}__{row['layout_arm']}__{row['command']}"
        if row["cell_id"] != expected_id:
            raise ValueError("cell ID does not match its factorial coordinates")
        expected = {
            "action_cap": "450", "stop_on_success": "false", "replicate_id": "0",
            "selected": "true", "status": "NOT_RELEASED",
            "candidate_effective_policy_seed": str(candidate_seed(phase, block) if model == "N3" else 1140),
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"cell contract or seed changed: {row['cell_id']}")
        key = (phase, block, model, condition)
        if key in indexed:
            raise ValueError("duplicate factorial condition")
        indexed[key] = row
    wanted = {(phase, block, model, condition)
              for phase, blocks in PHASE_BLOCKS.items() for block in blocks
              for model in MODELS for condition in CONDITIONS}
    if set(indexed) != wanted:
        raise ValueError("incomplete per-stage four-condition factorial")
    return indexed


def build_schedule(spec, rows, namespace=NAMESPACE):
    indexed = validate_inputs(spec, rows)
    if not namespace or not namespace.isascii() or ":" in namespace:
        raise ValueError("namespace must be a nonempty ASCII identifier without ':'")
    permutations = list(itertools.permutations(CONDITIONS))
    block_hashes = {block: sha256(f"{namespace}:{block}".encode("ascii"))
                   for blocks in PHASE_BLOCKS.values() for block in blocks}
    confirmation_order = sorted(PHASE_BLOCKS["confirmation"], key=lambda block: (block_hashes[block], block))
    permutation_indices = {block: index for index, block in enumerate(confirmation_order)}
    for phase in ("pilot", "development"):
        for block in PHASE_BLOCKS[phase]:
            permutation_indices[block] = int(block_hashes[block], 16) % 24
    job_id = lambda phase, block, model: f"{namespace}__{phase}__{block}__{model}"
    development_ids = sorted(job_id("development", block, model)
                             for block in PHASE_BLOCKS["development"] for model in MODELS)
    jobs = []
    for phase, blocks in PHASE_BLOCKS.items():
        for block in blocks:
            condition_order = permutations[permutation_indices[block]]
            for model in MODELS:
                if phase == "pilot":
                    dependencies = []
                    gates = ["qualified_runtime_and_seed_behavior", "qualified_pilot_fixture",
                             "nonbehavioral_decode_and_repeat_checks", "bounded_resource_authorization"]
                elif phase == "development":
                    dependencies = [job_id("pilot", "P00", model)]
                    gates = ["all_four_pilot_recordings_qualified_or_documented_exact_runtime_waiver",
                             "qualified_development_fixtures", "bounded_resource_authorization"]
                else:
                    dependencies = development_ids
                    gates = list(spec["release_requirements"]) + ["post_development_confirmation_freeze"]
                jobs.append({
                    "job_id": job_id(phase, block, model), "phase": phase,
                    "layout_pair_id": block, "model_config": model,
                    "indivisible": True, "released": False, "status": "NOT_RELEASED",
                    "condition_order": list(condition_order),
                    "ordered_cell_ids": [indexed[(phase, block, model, condition)]["cell_id"] for condition in condition_order],
                    "candidate_effective_policy_seed": candidate_seed(phase, block) if model == "N3" else 1140,
                    "depends_on_jobs": dependencies, "required_gates": gates,
                    "execution_contract": {
                        "sequential_conditions": True, "same_isolated_worker_type_within_job": True,
                        "full_model_and_simulator_reset_before_each_condition": True,
                        "no_global_DreamZero_context_interleaving": True,
                    },
                })
    return {
        "schema_version": "wmf-unreleased-block-schedule-v1", "namespace": namespace,
        "spec_id": spec["spec_id"], "upstream_pin": UPSTREAM,
        "status": "ORDER_PREPARED_NOT_RELEASED", "launch_ready": False,
        "cell_count": 232, "job_count": 58, "models": list(MODELS), "D2_selected": False,
        "order_assignment": {
            "input": "ASCII namespace + ':' + layout_pair_id; SHA-256 hexadecimal digest",
            "condition_alphabet": list(CONDITIONS),
            "confirmation_algorithm": "Sort C01-C24 by digest (block ID tie-break); assign the 24 lexicographic permutations in that order.",
            "pilot_development_algorithm": "Interpret complete digest as unsigned hexadecimal integer; index modulo 24 into the same permutation list.",
            "confirmation_blocks_in_hash_order": confirmation_order,
            "block_sha256": block_hashes, "permutation_index_by_block": permutation_indices,
            "same_order_for_both_models": True, "assignment_uses_outcomes": False,
        },
        "interpretation": "Dependencies are readiness requirements, not a released queue. No worker count, GPU capacity, physical coordinates or physical-time mapping is assigned.",
        "pilot_waiver_policy": "A waiver requires a separate verified four-condition exact-runtime record; no waived cell becomes an executed episode. This prepared schedule contains no waiver.",
        "jobs": jobs,
    }


def git(repo, *args, input_text=None, accepted=(0,)):
    result = subprocess.run(["git", *args], cwd=repo, input=input_text,
                            text=True, capture_output=True, check=False)
    if result.returncode not in accepted:
        raise RuntimeError(f"read-only Git operation failed: {result.stderr.strip()}")
    return result.stdout


def audit_candidate_seeds(repo, revision, seeds):
    """Bounded audit of exact candidate values in pinned structured artifacts.

    Git grep selects matching files, then JSON/JSONL/CSV parsing identifies seed
    fields. Excludes raw external files, code, prose, other refs and host RNGs.
    """
    candidates = set(seeds)
    revision = git(repo, "rev-parse", f"{revision}^{{commit}}").strip()
    entries = []
    for record in git(repo, "ls-tree", "-r", "-z", revision, "--", "artifacts").split("\0"):
        if not record:
            continue
        metadata, path = record.split("\t", 1)
        mode, kind, oid = metadata.split()
        if kind == "blob" and Path(path).suffix.lower() in {".json", ".jsonl", ".csv"}:
            entries.append({"path": path, "git_blob": oid, "mode": mode})
    pathspecs = [f":(glob)artifacts/**/*{suffix}" for suffix in (".json", ".jsonl", ".csv")]
    matches = git(repo, "grep", "-l", "-z", "-I", "-w", "-F", "-f", "-", revision,
                  "--", *pathspecs, input_text="".join(f"{seed}\n" for seed in sorted(candidates)), accepted=(0, 1))
    matched_paths = sorted(item.split(":", 1)[1] for item in matches.split("\0") if item)
    collisions, other, parse_failures, unclassified = [], [], [], []

    def visit(value, path, pointer, seed_context=False):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, path, pointer + "/" + str(key), seed_context or bool(SEED_KEY.search(str(key))))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path, pointer + "/" + str(index), seed_context)
        elif not isinstance(value, bool) and str(value) in {str(seed) for seed in candidates}:
            hit = {"seed": int(value), "path": path, "field_pointer": pointer}
            (collisions if seed_context else other).append(hit)

    for path in matched_paths:
        content = git(repo, "show", f"{revision}:{path}")
        before = len(collisions) + len(other)
        try:
            if path.endswith(".jsonl"):
                for number, line in enumerate(content.splitlines(), 1):
                    if line.strip():
                        visit(json.loads(line), path, f"line:{number}")
            elif path.endswith(".csv"):
                for number, row in enumerate(csv.DictReader(io.StringIO(content)), 2):
                    visit(row, path, f"row:{number}")
            else:
                visit(json.loads(content), path, "")
        except (ValueError, TypeError) as error:
            parse_failures.append({"path": path, "error": str(error)})
        if before == len(collisions) + len(other):
            unclassified.append(path)
    hit_seeds = {hit["seed"] for hit in collisions}
    return {
        "schema_version": "wmf-bounded-seed-audit-v1", "upstream_pin": revision,
        "candidate_seeds": sorted(candidates), "artifact_file_count": len(entries),
        "artifact_inventory_sha256": sha256(canonical_bytes(entries)),
        "artifact_tree_git_object": git(repo, "rev-parse", f"{revision}:artifacts").strip(),
        "pathspecs": pathspecs, "seed_field_pattern": SEED_KEY.pattern,
        "known_field_examples": ["seed", "seeds", "environment_seed", "sampling_seed", "policy_seed", "effective_official_model_noise_seed"],
        "matched_files": matched_paths, "seed_field_collisions": collisions,
        "other_numeric_occurrences": other, "unclassified_matched_files": unclassified,
        "parse_failures": parse_failures,
        "candidate_results": {str(seed): "collision_in_scanned_seed_fields" if seed in hit_seeds
                              else "not_found_in_scanned_seed_fields" for seed in sorted(candidates)},
        "bounded_audit_clear": not (collisions or parse_failures or unclassified),
        "full_collision_proof": False, "proves_runtime_seed_compatibility": False,
        "scope_limit": "Pinned artifacts/**/*.json, .jsonl and .csv only; exact decimal candidate matches classified by seed-named fields. No raw external files, source code, prose, other refs, runtime RNG acceptance or host state was audited.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    script = Path(__file__).resolve()
    workshop = script.parents[2]
    repo = script.parents[4]
    spec_path = script.parent / "ablation_spec.json"
    cells_path = script.parent / "planned_cells.csv"
    spec = json.loads(spec_path.read_text())
    with cells_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    schedule = build_schedule(spec, rows, args.namespace)
    audit = audit_candidate_seeds(repo, UPSTREAM,
                                  sorted({job["candidate_effective_policy_seed"] for job in schedule["jobs"] if job["model_config"] == "N3"}))
    schedule["source_sha256"] = {str(path.relative_to(repo)): sha256(path.read_bytes())
                                  for path in (spec_path, cells_path, script)}
    schedule["nano_seed_audit"] = {"path": "nano_seed_audit.json", "sha256": sha256(canonical_bytes(audit)),
                                   "bounded_audit_clear": audit["bounded_audit_clear"], "runtime_compatibility_qualified": False}
    output_dir = args.output_dir or workshop / "execution/20260912"
    payloads = {output_dir / "nano_seed_audit.json": canonical_bytes(audit),
                output_dir / "parallel_schedule.json": canonical_bytes(schedule)}
    for path, data in payloads.items():
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f"refusing to replace different prepared evidence: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, data in payloads.items():
        if not path.exists():
            path.write_bytes(data)
    print(json.dumps({"jobs": 58, "cells": 232, "launch_ready": False,
                      "bounded_seed_audit_clear": audit["bounded_audit_clear"], "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
