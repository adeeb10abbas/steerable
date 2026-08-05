#!/usr/bin/env python3
"""Build the immutable Phase-A v3 matched-cell queue.

The queue is deliberately a *registry*, not a launcher.  A launcher must refuse
any row whose ``execution_status`` is not
``authorized_after_all_registered_release_gates``.
This keeps historical reuse candidates and the blocked pi0-FAST expansion visible
without accidentally treating either as runnable work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "vla-wam-shared-v3-phase-a-cells-v1"
MANIFEST_SCHEMA_VERSION = "vla-wam-shared-v3-phase-a-cells-manifest-v1"
STUDY_ID = "vla_wam_language_steerability_v3"
RELATIONS = ("left", "right")

# These are the committed pair03--09 first-seen descriptions.  They make the
# v3 prompts byte-stable rather than asking a live detector to name the objects
# at execution time.  The builder verifies every rendered prompt matches the
# v3 registry template before emitting a row.
ROBOTWIN_PAIR_OBJECTS: dict[int, tuple[str, str]] = {
    3: ("small woodenblock", "red playingcards box"),
    4: ("plastic mouse", "blue stapler"),
    5: ("box of playingcards", "rubikscube"),
    6: ("coffee box", "red playingcards box"),
    7: ("golden bread", "blue stapler"),
    8: ("box with cards inside", "black phone"),
    9: ("rubikscube", "brown woodenblock"),
}
ROBOTWIN_PROMPT_FIXTURE_ID = "v2_pairs03_09_first_seen_object_description_freeze"


class RegistryError(ValueError):
    """Raised when a proposed queue is not exactly the frozen design."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read JSON registry {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistryError(f"registry {path} must contain a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


def registry_paths(repo_root: Path) -> dict[str, Path]:
    base = repo_root / "artifacts" / "vla_wam_shared_v3"
    return {
        "protocol": base / "protocol.json",
        "droid": base / "droid_direct_registry.json",
        "robotwin": base / "robotwin_direct_registry.json",
    }


def load_registries(repo_root: Path) -> dict[str, dict[str, Any]]:
    paths = registry_paths(repo_root)
    registries = {name: read_json(path) for name, path in paths.items()}
    for name, registry in registries.items():
        require(registry.get("study_id") == STUDY_ID, f"{name} study_id is not {STUDY_ID}")
        require(
            str(registry.get("status", "")).startswith("frozen_before_any_v3"),
            f"{name} registry is not frozen before v3 inference",
        )
    return registries


def runtime_requirement(arena: str, model_id: str, status: str) -> dict[str, Any]:
    requirement = {
        "requirement_id": f"v3:{arena}:runtime_identity_and_checkpoint_hash",
        "model_id": model_id,
        "must_record": [
            "external_repository_commit",
            "external_repository_diff_hash",
            "checkpoint_identifier",
            "checkpoint_sha256",
            "environment_lock_hash",
            "adapter_contract_hash",
            "simulator_version",
            "renderer_backend",
        ],
        "left_right_must_match": True,
    }
    if status in {"preserved_candidate", "preserved_r0"}:
        requirement["preserved_evidence_only"] = True
        requirement["exact_match_required_before_carry_forward"] = True
    return requirement


def _droid_status(model_id: str, seed: int, droid: dict[str, Any]) -> tuple[str, str]:
    rules = droid["checkpoint_rules"]
    if model_id == "pi0_fast_droid_vla":
        if seed in rules[model_id]["preserved_historical_seeds"]:
            return "preserved_candidate", "preserved_evidence_only_runtime_identity_check"
        if seed in rules[model_id]["blocked_new_seeds"]:
            return "blocked_pi0", "blocked_pending_exact_historical_openpi_and_robolab_recovery"
        raise RegistryError(f"pi0-FAST seed {seed} is not classified")
    other = rules["other_checkpoints"]
    if seed in other["preserved_candidate_seeds"]:
        return "preserved_candidate", "preserved_evidence_only_runtime_identity_check"
    if seed in other["new_addition_seeds"]:
        return "authorized_new", "authorized_after_all_registered_release_gates"
    raise RegistryError(f"{model_id} seed {seed} is not classified")


def build_droid_rows(droid: dict[str, Any]) -> list[dict[str, Any]]:
    require(droid.get("arena") == "droid_robolab", "DROID arena changed")
    target = droid.get("target", {})
    seed_start, seed_end = target["seed_range"]
    seeds = list(range(seed_start, seed_end + 1))
    require(len(seeds) == target.get("exact_matched_pair_count_per_checkpoint") == 30, "DROID must contain exactly 30 seeds")
    require(tuple(target.get("directions_per_pair", [])) == RELATIONS, "DROID directions must be left/right")
    models = list(droid["priority"])
    require(len(models) == 6 and len(set(models)) == 6, "DROID priority must list six unique models")

    prompts = droid["direct_prompts"]
    required_outputs = list(droid["required_evidence"])
    rows: list[dict[str, Any]] = []
    for model_id in models:
        for seed in seeds:
            status, execution_status = _droid_status(model_id, seed, droid)
            pair_id = f"v3:droid:{model_id}:seed{seed}"
            reset_identity = f"v3:droid_robolab:neutral_reset:environment_seed_{seed}"
            for relation in RELATIONS:
                prompt = prompts[relation]
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "study_id": STUDY_ID,
                        "phase": "A_direct_command_matched_pairs",
                        "arena": "droid_robolab",
                        "model_id": model_id,
                        "cell_id": f"{pair_id}:{relation}",
                        "pair_id": pair_id,
                        "relation": relation,
                        "replicate": 0,
                        "environment_seed": seed,
                        "sampling_seed": seed,
                        "reset_identity": reset_identity,
                        "anchor_task": "droid_robolab_cube_to_bowl_neutral_reset",
                        "prompt_family": "direct_command",
                        "prompt": prompt,
                        "prompt_sha256": sha256_text(prompt),
                        "prompt_fixture_id": "droid_robolab_cube_bowl_direct_prompt_freeze",
                        "success_predicate_id": "v2_frozen_droid_robolab_release_inside_45deg_requested_relation",
                        "runtime_identity_requirement": runtime_requirement("droid_robolab", model_id, status),
                        "required_raw_outputs": required_outputs,
                        "status": status,
                        "execution_status": execution_status,
                        "status_reason": "" if status != "blocked_pi0" else droid["checkpoint_rules"][model_id]["blocker"],
                    }
                )
    return rows


def render_robotwin_prompt(robotwin: dict[str, Any], pair_number: int, relation: str) -> str:
    try:
        seen_object, seen_reference = ROBOTWIN_PAIR_OBJECTS[pair_number]
    except KeyError as exc:
        raise RegistryError(f"No exact frozen object descriptions for RoboTwin pair {pair_number}") from exc
    template = robotwin["direct_prompts"][relation]
    require("{seen_object}" in template and "{seen_reference}" in template, "RoboTwin prompt template lost frozen placeholders")
    return template.format(seen_object=seen_object, seen_reference=seen_reference)


def build_robotwin_rows(robotwin: dict[str, Any]) -> list[dict[str, Any]]:
    require(robotwin.get("arena") == "robotwin", "RoboTwin arena changed")
    models = list(robotwin["models"])
    pairs = list(robotwin["scene_pairs"])
    replicates = list(robotwin["sampling"]["replicates"])
    require(len(models) == 3 and len(set(models)) == 3, "RoboTwin must list three unique models")
    require(pairs == list(range(3, 10)), "RoboTwin scene pairs must be 03--09")
    require(replicates == list(range(10)), "RoboTwin replicates must be 0--9")
    require(set(pairs) == set(ROBOTWIN_PAIR_OBJECTS), "RoboTwin exact prompt map does not cover every pair")
    required_outputs = list(robotwin["required_evidence"])
    rows: list[dict[str, Any]] = []
    for model_id in models:
        for pair_number in pairs:
            environment_seed = 4_300_000 + pair_number
            anchor_task = "place_a2b_right" if pair_number % 2 else "place_a2b_left"
            for replicate in replicates:
                sampling_seed = 8_400 + pair_number + 100 * replicate
                status = "preserved_r0" if replicate == 0 else "authorized_new"
                execution_status = (
                    "preserved_evidence_only_runtime_identity_check"
                    if replicate == 0
                    else "authorized_after_all_registered_release_gates"
                )
                pair_id = f"v3:robotwin:{model_id}:pair{pair_number:02d}:replicate{replicate:02d}"
                reset_identity = (
                    f"v3:robotwin:pair{pair_number:02d}:anchor_{anchor_task}:environment_seed_{environment_seed}"
                )
                for relation in RELATIONS:
                    prompt = render_robotwin_prompt(robotwin, pair_number, relation)
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "study_id": STUDY_ID,
                            "phase": "A_direct_command_sampling_replication",
                            "arena": "robotwin",
                            "model_id": model_id,
                            "cell_id": f"{pair_id}:{relation}",
                            "pair_id": pair_id,
                            "scene_pair": pair_number,
                            "relation": relation,
                            "replicate": replicate,
                            "environment_seed": environment_seed,
                            "sampling_seed": sampling_seed,
                            "reset_identity": reset_identity,
                            "anchor_task": anchor_task,
                            "prompt_family": "direct_command",
                            "prompt": prompt,
                            "prompt_sha256": sha256_text(prompt),
                            "prompt_fixture_id": ROBOTWIN_PROMPT_FIXTURE_ID,
                            "seen_object_description": ROBOTWIN_PAIR_OBJECTS[pair_number][0],
                            "seen_reference_description": ROBOTWIN_PAIR_OBJECTS[pair_number][1],
                            "success_predicate_id": "v2_frozen_robotwin_relation_aware_detached_release_requested_relation",
                            "runtime_identity_requirement": runtime_requirement("robotwin", model_id, status),
                            "required_raw_outputs": required_outputs,
                            "status": status,
                            "execution_status": execution_status,
                            "status_reason": "",
                        }
                    )
    return rows


def sort_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            0 if row["arena"] == "droid_robolab" else 1,
            row["model_id"],
            row.get("scene_pair", -1),
            row["environment_seed"],
            row["replicate"],
            row["relation"],
        ),
    )


def _pairs(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["pair_id"]].append(row)
    return grouped


def validate_rows(rows: list[dict[str, Any]]) -> None:
    require(len(rows) == 780, f"Phase A must contain 780 cells, got {len(rows)}")
    ids = [row["cell_id"] for row in rows]
    require(len(ids) == len(set(ids)), "duplicate Phase-A cell_id")
    arena_counts = Counter(row["arena"] for row in rows)
    require(arena_counts == {"droid_robolab": 360, "robotwin": 420}, f"incorrect arena counts: {arena_counts}")
    droid_models = {row["model_id"] for row in rows if row["arena"] == "droid_robolab"}
    require(len(droid_models) == 6 and all(sum(row["model_id"] == model for row in rows) == 60 for model in droid_models), "each DROID model needs 30 pairs")
    robotwin_models = {row["model_id"] for row in rows if row["arena"] == "robotwin"}
    require(len(robotwin_models) == 3 and all(sum(row["model_id"] == model for row in rows) == 140 for model in robotwin_models), "each RoboTwin model needs 70 pairs")
    status_counts = Counter(row["status"] for row in rows)
    require(
        status_counts == {"authorized_new": 648, "preserved_candidate": 50, "blocked_pi0": 40, "preserved_r0": 42},
        f"unexpected status counts: {status_counts}",
    )

    for pair_id, pair_rows in _pairs(rows).items():
        require(len(pair_rows) == 2, f"{pair_id} is not a LEFT/RIGHT pair")
        require({row["relation"] for row in pair_rows} == set(RELATIONS), f"{pair_id} lacks a direction")
        shared = (
            "schema_version", "study_id", "phase", "arena", "model_id", "pair_id", "replicate",
            "environment_seed", "sampling_seed", "reset_identity", "anchor_task", "prompt_family",
            "runtime_identity_requirement", "required_raw_outputs", "status", "execution_status", "status_reason",
            "prompt_fixture_id",
        )
        for field in shared:
            require(pair_rows[0][field] == pair_rows[1][field], f"{pair_id} mismatched {field}")
        left = next(row for row in pair_rows if row["relation"] == "left")
        right = next(row for row in pair_rows if row["relation"] == "right")
        require(left["prompt"].replace("left", "{direction}") == right["prompt"].replace("right", "{direction}"), f"{pair_id} prompts change more than the direction word")
        require(left["success_predicate_id"] == right["success_predicate_id"], f"{pair_id} predicate changed")
        require(left["prompt_sha256"] != right["prompt_sha256"], f"{pair_id} prompt hashes unexpectedly match")

    for model_id in droid_models:
        seeds = {row["environment_seed"] for row in rows if row["model_id"] == model_id}
        require(seeds == set(range(8300, 8330)), f"{model_id} DROID schedule changed")
    robotwin_schedules = {
        model_id: {(row["scene_pair"], row["replicate"], row["environment_seed"], row["sampling_seed"])
                   for row in rows if row["model_id"] == model_id}
        for model_id in robotwin_models
    }
    require(len({frozenset(schedule) for schedule in robotwin_schedules.values()}) == 1, "RoboTwin model seed schedules differ")


def nested_counts(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        cursor = result
        for field in fields[:-1]:
            cursor = cursor.setdefault(str(row[field]), {})
        leaf = str(row[fields[-1]])
        cursor[leaf] = cursor.get(leaf, 0) + 1
    return result


def build_manifest(repo_root: Path, rows: list[dict[str, Any]], jsonl_sha256: str) -> dict[str, Any]:
    inputs = registry_paths(repo_root)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "phase": "A_direct_command_matched_pairs_and_sampling_replication",
        "queue_file": "artifacts/vla_wam_shared_v3/phase_a_cells.jsonl",
        "queue_sha256": jsonl_sha256,
        "row_counts": {
            "total": len(rows),
            "by_arena": dict(sorted(Counter(row["arena"] for row in rows).items())),
            "by_model": dict(sorted(Counter(row["model_id"] for row in rows).items())),
            "by_status": dict(sorted(Counter(row["status"] for row in rows).items())),
        },
        "status_counts": {
            "by_arena": nested_counts(rows, ("arena", "status")),
            "by_model": nested_counts(rows, ("model_id", "status")),
        },
        "target_counts": {
            "droid": {"checkpoints": 6, "matched_pairs_per_checkpoint": 30, "cells": 360},
            "robotwin": {"models": 3, "scene_pairs_per_model": 7, "sampling_replicates_per_pair": 10, "cells": 420},
            "total_cells": 780,
        },
        "launch_rule": "Only status=authorized_new and execution_status=authorized_after_all_registered_release_gates may be launched. Preserved rows are evidence-only runtime identity checks; blocked_pi0 rows are never launchable.",
        "source_registry_sha256": {
            name: sha256_bytes(path.read_bytes()) for name, path in sorted(inputs.items())
        },
    }


def render_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return ("\n".join(canonical_json(row) for row in rows) + "\n").encode("utf-8")


def build_phase_a(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], bytes]:
    registries = load_registries(repo_root)
    rows = sort_rows(build_droid_rows(registries["droid"]) + build_robotwin_rows(registries["robotwin"]))
    validate_rows(rows)
    payload = render_jsonl(rows)
    manifest = build_manifest(repo_root, rows, sha256_bytes(payload))
    return rows, manifest, payload


def write_outputs(jsonl_path: Path, manifest_path: Path, payload: bytes, manifest: dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_bytes(payload)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--check", action="store_true", help="fail if committed outputs differ from deterministic rebuild")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    default_base = repo_root / "artifacts" / "vla_wam_shared_v3"
    jsonl_path = (args.output_jsonl or default_base / "phase_a_cells.jsonl").resolve()
    manifest_path = (args.output_manifest or default_base / "phase_a_cells_manifest.json").resolve()
    try:
        _, manifest, payload = build_phase_a(repo_root)
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if args.check:
            require(jsonl_path.exists(), f"missing generated queue: {jsonl_path}")
            require(manifest_path.exists(), f"missing generated manifest: {manifest_path}")
            require(jsonl_path.read_bytes() == payload, "generated queue differs from deterministic rebuild")
            require(manifest_path.read_text(encoding="utf-8") == manifest_text, "generated manifest differs from deterministic rebuild")
        else:
            write_outputs(jsonl_path, manifest_path, payload, manifest)
    except RegistryError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
