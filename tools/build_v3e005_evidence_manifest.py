#!/usr/bin/env python3
"""Build the portable, hash-closed V3-E005 evidence ledger.

Raw RoboTwin media and runner products may remain on the execution volume.  The
portable bundle binds every compact row back to its immutable ``raw_episode``
source while keeping the preregistered H4 decision gate explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": display_path(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
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


def canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _seed_manifests(raw_roots: Sequence[Path]) -> list[Path]:
    return sorted(
        {
            path.resolve()
            for root in raw_roots
            for path in Path(root).rglob("seed_*_manifest.json")
            if path.is_file()
        }
    )


def _shard_manifests(raw_roots: Sequence[Path]) -> list[Path]:
    return sorted(
        {
            path.resolve()
            for root in raw_roots
            for path in [Path(root) / "shard_manifest.json"]
            if path.is_file()
        }
    )


def build(base: Path, output: Path, *, raw_roots: Sequence[Path] = ()) -> dict[str, Any]:
    base = Path(base).resolve()
    results_dir = base / "results"
    required = [
        base / "registration.json",
        base / "queue.jsonl",
        results_dir / "results.json",
        results_dir / "episodes.jsonl",
        results_dir / "pairs.jsonl",
        results_dir / "infrastructure_invalid.jsonl",
        base / "DECISION_MEMO.md",
        base / "V3E005_PUBLICATION_DECISION.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing E005 compact evidence: " + ", ".join(missing))

    results = load_json(results_dir / "results.json")
    episodes = load_jsonl(results_dir / "episodes.jsonl")
    complete = results.get("coverage", {}).get("complete") is True
    figure_manifest_path = results_dir / "figures/figure_manifest.json"
    if not figure_manifest_path.is_file():
        raise FileNotFoundError("E005 evidence requires a rendered figure manifest")
    figure_manifest = load_json(figure_manifest_path)

    compact_paths = list(required) + [figure_manifest_path]
    for item in figure_manifest.get("figures", []):
        path = Path(item["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"missing rendered E005 figure: {path}")
        compact_paths.append(path)

    implementation_paths = [
        ROOT / "tools/build_v3e005_registration.py",
        ROOT / "tools/validate_v3e005.py",
        ROOT / "tools/compile_v3e005_results.py",
        ROOT / "tools/render_v3e005_results.py",
        ROOT / "tools/build_v3e005_publication_decision.py",
        ROOT / "tools/build_v3e005_evidence_manifest.py",
        ROOT / "tools/validate_v3e005_evidence.py",
    ]
    implementation_paths = [path for path in implementation_paths if path.is_file()]

    source_inventory: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in episodes:
        source = row.get("source_raw_episode", {})
        key = (str(source.get("sha256")), int(source.get("bytes", -1)), str(source.get("path")))
        source_inventory[key] = {
            "path": key[2],
            "bytes": key[1],
            "sha256": key[0],
        }

    seed_paths = _seed_manifests(raw_roots)
    seeds: dict[int, dict[str, Any]] = {}
    seed_pair_files: dict[str, dict[str, Any]] = {}
    episode_by_id = {row["cell_id"]: row for row in episodes}
    for path in seed_paths:
        marker = load_json(path)
        if marker.get("amendment_id") != "V3-E005":
            continue
        if marker.get("schema_version") != "vla-wam-shared-v3e005-lingbot-whole-seed-completion-v1":
            raise ValueError(f"wrong E005 seed-manifest schema: {path}")
        seed = int(marker["seed"])
        if seed in seeds:
            raise ValueError(f"conflicting whole-seed manifests: {seed}")
        cell_ids = marker.get("cell_ids")
        if (
            marker.get("status") != "complete_four_valid_behavioral_cells"
            or marker.get("behavioral_episode_count") != 4
            or marker.get("matched_pair_count") != 2
            or not isinstance(cell_ids, list)
            or len(cell_ids) != 4
            or len(set(cell_ids)) != 4
        ):
            raise ValueError(f"incomplete whole-seed manifest: {path}")
        expected_ids = {
            row["cell_id"] for row in episodes if int(row["environment_seed"]) == seed
        }
        if set(cell_ids) != expected_ids:
            raise ValueError(f"whole-seed manifest/compact episode mismatch: {path}")
        episode_hashes = marker.get("episode_sha256")
        if not isinstance(episode_hashes, dict):
            raise ValueError(f"whole-seed manifest lacks episode hashes: {path}")
        compact_episode_paths = marker.get("compact_episode_paths")
        expected_paths = {
            str(Path(episode_by_id[cell_id]["source_raw_episode"]["path"]).resolve())
            for cell_id in cell_ids
        }
        if not isinstance(compact_episode_paths, list) or {
            str(Path(value).resolve()) for value in compact_episode_paths
        } != expected_paths:
            raise ValueError(f"whole-seed manifest episode-path binding differs: {path}")
        for cell_id in cell_ids:
            source = episode_by_id[cell_id]["source_raw_episode"]
            if episode_hashes.get(cell_id) != source["sha256"]:
                raise ValueError(f"whole-seed manifest episode hash mismatch: {cell_id}")
        if marker.get("registration_sha256") != sha256(base / "registration.json"):
            raise ValueError(f"whole-seed registration binding differs: {path}")
        if marker.get("queue_sha256") != sha256(base / "queue.jsonl"):
            raise ValueError(f"whole-seed queue binding differs: {path}")
        marker_payload = dict(marker)
        marker_digest = marker_payload.pop("marker_sha256", None)
        if marker_digest != canonical_sha256(marker_payload):
            raise ValueError(f"whole-seed self-hash differs: {path}")
        pair_paths = marker.get("pair_paths")
        if not isinstance(pair_paths, list) or len(pair_paths) != 2:
            raise ValueError(f"whole-seed manifest does not bind two pair files: {path}")
        for pair_value in pair_paths:
            pair_path = Path(str(pair_value)).expanduser().resolve()
            if not pair_path.is_file():
                raise FileNotFoundError(f"whole-seed pair file missing: {pair_path}")
            pair = load_json(pair_path)
            if pair.get("amendment_id") != "V3-E005" or int(pair.get("environment_seed", -1)) != seed:
                raise ValueError(f"whole-seed pair identity differs: {pair_path}")
            pair_payload = dict(pair)
            pair_digest = pair_payload.pop("pair_sha256", None)
            if pair_digest != canonical_sha256(pair_payload):
                raise ValueError(f"whole-seed pair self-hash differs: {pair_path}")
            seed_pair_files[str(pair_path)] = record(pair_path)
        seeds[seed] = {"seed": seed, **record(path)}
    if complete and set(seeds) != set(range(9400, 9427)):
        missing_seeds = sorted(set(range(9400, 9427)) - set(seeds))
        raise FileNotFoundError(f"complete E005 evidence requires 27 whole-seed manifests; missing {missing_seeds}")

    shard_manifests: dict[int, dict[str, Any]] = {}
    for path in _shard_manifests(raw_roots):
        marker = load_json(path)
        if marker.get("amendment_id") != "V3-E005":
            continue
        if marker.get("schema_version") != "vla-wam-shared-v3e005-lingbot-shard-manifest-v1":
            raise ValueError(f"wrong E005 shard-manifest schema: {path}")
        shard_index = int(marker["shard_index"])
        if shard_index in shard_manifests:
            raise ValueError(f"conflicting E005 shard manifests: {shard_index}")
        expected_cells = 20 if shard_index < 3 else 16
        if (
            marker.get("status") != "requested_shard_complete"
            or marker.get("shard_count") != 6
            or marker.get("behavioral_episode_count") != expected_cells
            or marker.get("matched_pair_count") != expected_cells // 2
            or marker.get("infrastructure_failure_count") != 0
            or marker.get("whole_seed_atomic") is not True
        ):
            raise ValueError(f"incomplete E005 shard manifest: {path}")
        if marker.get("registration_sha256") != sha256(base / "registration.json"):
            raise ValueError(f"shard-manifest registration binding differs: {path}")
        if marker.get("queue_sha256") != sha256(base / "queue.jsonl"):
            raise ValueError(f"shard-manifest queue binding differs: {path}")
        marker_payload = dict(marker)
        marker_digest = marker_payload.pop("manifest_sha256", None)
        if marker_digest != canonical_sha256(marker_payload):
            raise ValueError(f"shard-manifest self-hash differs: {path}")
        shard_manifests[shard_index] = {"shard_index": shard_index, **record(path)}
    if complete and shard_manifests and set(shard_manifests) != set(range(6)):
        missing_shards = sorted(set(range(6)) - set(shard_manifests))
        raise FileNotFoundError(f"complete E005 evidence requires six shard manifests; missing {missing_shards}")

    setup_failure_paths = sorted((base / "setup_failures").glob("v3e005_gate_failure_ledger_v*.json"))
    setup_failure_ledgers = [record(path) for path in setup_failure_paths if path.is_file()]
    compact_paths.extend(setup_failure_paths)

    manifest = {
        "schema_version": "vla-wam-shared-v3e005-evidence-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E005",
        "status": "hash_closed_compact_evidence" if complete else "partial_progress_not_publication_evidence",
        "arena": "robotwin",
        "model_id": "lingbot_va_robotwin",
        "registration_sha256": sha256(base / "registration.json"),
        "queue_sha256": sha256(base / "queue.jsonl"),
        "results_sha256": sha256(results_dir / "results.json"),
        "episodes_sha256": sha256(results_dir / "episodes.jsonl"),
        "pairs_sha256": sha256(results_dir / "pairs.jsonl"),
        "infrastructure_invalid_sha256": sha256(results_dir / "infrastructure_invalid.jsonl"),
        "valid_behavioral_episodes": results["valid_behavioral_episodes"],
        "complete_matched_pairs": results["complete_matched_pairs"],
        "registered_behavioral_cells": results["registered_behavioral_cells"],
        "infrastructure_invalid_attempts": results["infrastructure_invalid_attempts"],
        "h4_outcome": results["h4_gate"]["outcome"],
        "h4_evaluated_first": results["analysis_order"][0] == "H4",
        "h1_h3_disposition": results["h4_gate"]["h1_h3_disposition"],
        "publication_claim_status": results["publication_claim_status"],
        "bootstrap_resamples": results["bootstrap_resamples"],
        "scene_cluster_count": results["scene_cluster_count"],
        "compact_files": [record(path) for path in sorted(set(compact_paths))],
        "implementation_files": [record(path) for path in sorted(set(implementation_paths))],
        "raw_source_inventory": sorted(source_inventory.values(), key=lambda item: (item["path"], item["sha256"])),
        "raw_source_count": len(source_inventory),
        "whole_seed_manifest_count": len(seeds),
        "whole_seed_manifests": [seeds[seed] for seed in sorted(seeds)],
        "whole_seed_pair_file_count": len(seed_pair_files),
        "whole_seed_pair_files": [seed_pair_files[path] for path in sorted(seed_pair_files)],
        "shard_manifest_count": len(shard_manifests),
        "shard_manifests": [shard_manifests[index] for index in sorted(shard_manifests)],
        "zero_request_setup_failure_ledger_count": len(setup_failure_ledgers),
        "zero_request_setup_failure_ledgers": setup_failure_ledgers,
        "raw_evidence_policy": (
            "Full RoboTwin videos, action traces, trajectories, runner results, and whole-seed manifests remain "
            "on the execution volume. Each compact behavioral row retains the immutable raw_episode path, byte "
            "count, SHA-256 digest, and line number."
        ),
        "scientific_boundaries": {
            "droid_imported_or_pooled": False,
            "arena_specific_predicate_and_coordinate_preserved": True,
            "seed_replicates_treated_as_independent_scenes": False,
            "scene_clustered_intervals": True,
            "h4_gate_evaluated_before_h1_h3": True,
            "h1_h3_withheld_when_h4_fails": True,
            "symmetric_object_layout_not_symmetric_robot_or_embodiment": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=BASE / "evidence_manifest.json")
    parser.add_argument("--raw-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    value = build(args.base, args.output, raw_roots=args.raw_root)
    print(json.dumps({"status": value["status"], "compact_files": len(value["compact_files"])}, indent=2))


if __name__ == "__main__":
    main()
