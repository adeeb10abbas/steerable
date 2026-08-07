#!/usr/bin/env python3
"""Release the exact 864-cell pi0.5 Phase-D queue after V3-D001 passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STUDY_ID = "vla_wam_language_steerability_v3"
MODEL_ID = "pi05_current_stack_droid"
REGISTRATION_SHA256 = "899a52c79355919210d56fa8f31d944f8a373e1e184650ee8974d62acfd6c788"
PHASE_D_SHA256 = "e319f8dcaefa6803ca46989313ba737834eef1dd531c1898aeee5fa816a28ad9"
PHASE_A_SUMMARY_SHA256 = "5c6d07fca7a0d20ab8b757f028d469f864c78a1c43ffedc3d257a16caef2a02b"
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
SEEDS = tuple(range(8303, 8330))
SAMPLING_INDICES = tuple(range(16))


class ReleaseError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReleaseError(f"expected JSON object: {path}")
    return value


def validate_eligibility(report_path: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = load(report_path)
    manifest = load(manifest_path)
    expected = {
        "schema_version": "vla-wam-shared-v3d001-pi05-stochastic-eligibility-result-v1",
        "study_id": STUDY_ID,
        "registration_id": "V3-D001",
        "model_id": MODEL_ID,
        "status": "eligible_effective_sampling_seed",
        "passed": True,
        "model_request_count": 32,
        "behavioral_episode_count": 0,
        "registration_sha256": REGISTRATION_SHA256,
        "phase_d_registry_sha256": PHASE_D_SHA256,
        "sampling_seed_indices": list(range(8)),
        "exact_prompts": PROMPTS,
    }
    for key, wanted in expected.items():
        if report.get(key) != wanted:
            raise ReleaseError(f"V3-D001 eligibility mismatch for {key}")
    if set(report.get("direction_metrics", {})) != {"left", "right"}:
        raise ReleaseError("V3-D001 eligibility direction set mismatch")
    if not all(value.get("passed") is True for value in report["direction_metrics"].values()):
        raise ReleaseError("V3-D001 did not pass in both directions")
    if (
        manifest.get("schema_version") != "vla-wam-shared-v3d001-pi05-stochastic-eligibility-manifest-v1"
        or manifest.get("status") != "eligible_effective_sampling_seed"
        or manifest.get("model_request_count") != 32
        or manifest.get("behavioral_episode_count") != 0
    ):
        raise ReleaseError("V3-D001 eligibility manifest mismatch")
    files = {Path(row["path"]).resolve(): row for row in manifest.get("files", [])}
    record = files.get(report_path.resolve())
    if not record or record.get("sha256") != sha256_file(report_path):
        raise ReleaseError("eligibility manifest does not bind its report")
    return report, manifest


def phase_a_cells(summary_path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    if sha256_file(summary_path) != PHASE_A_SUMMARY_SHA256:
        raise ReleaseError("pi0.5 Phase-A summary hash changed")
    summary = load(summary_path)
    if summary.get("model_id") != MODEL_ID or summary.get("arena") != "droid_robolab":
        raise ReleaseError("unexpected pi0.5 Phase-A summary identity")
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for row in summary.get("cells", []):
        seed, relation = row.get("seed"), row.get("relation")
        if seed not in SEEDS or relation not in PROMPTS:
            raise ReleaseError("unexpected pi0.5 Phase-A cell")
        if row.get("prompt") != PROMPTS[relation]:
            raise ReleaseError("pi0.5 Phase-A prompt bytes changed")
        key = (seed, relation)
        if key in result:
            raise ReleaseError("duplicate pi0.5 Phase-A condition")
        result[key] = row
    expected = {(seed, relation) for seed in SEEDS for relation in PROMPTS}
    if set(result) != expected:
        raise ReleaseError("pi0.5 Phase-A summary is not the exact 27-pair released cohort")
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = args.study_root.resolve()
    registration = root / "artifacts/vla_wam_shared_v3/prospective_tier_b/pi05_stochastic_eligibility_v3d001.json"
    phase_d = root / "artifacts/vla_wam_shared_v3/stochastic_rollout_registry.json"
    summary_path = root / "artifacts/vla_wam_shared_v3/results/pi05_current_stack_droid_phase_a_summary.json"
    if sha256_file(registration) != REGISTRATION_SHA256 or sha256_file(phase_d) != PHASE_D_SHA256:
        raise ReleaseError("prospective V3-D001/Phase-D binding changed")
    report, manifest = validate_eligibility(args.eligibility_report.resolve(), args.eligibility_manifest.resolve())
    sources = phase_a_cells(summary_path)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Phase-D release: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for sampling_index in SAMPLING_INDICES:
            order = ("left", "right") if (seed + sampling_index) % 2 == 0 else ("right", "left")
            policy_seed_base = seed * 1_000_000 + sampling_index * 1_000
            for order_index, relation in enumerate(order):
                source = sources[(seed, relation)]
                row = {
                    "schema_version": "vla-wam-shared-v3d001-pi05-stochastic-cell-v1",
                    "study_id": STUDY_ID,
                    "phase": "D_16_rollout_stochastic_block",
                    "registration_id": "V3-D001",
                    "model_id": MODEL_ID,
                    "arena": "droid_robolab",
                    "cell_id": f"v3d001:pi05:env{seed}:{relation}:sample{sampling_index}",
                    "nested_condition_id": f"v3:droid:pi05_current_stack_droid:seed{seed}:{relation}",
                    "matched_stochastic_block_id": f"v3d001:pi05:env{seed}:sample{sampling_index}",
                    "environment_seed": seed,
                    "requested_relation": relation,
                    "prompt": PROMPTS[relation],
                    "prompt_mode": "static_episode_prompt",
                    "shared_policy_sampling_seed_index": sampling_index,
                    "policy_sampling_seed_base": policy_seed_base,
                    "per_request_sampling_seed_rule": "policy_sampling_seed_base + zero_based_request_index",
                    "execution_order_index_within_matched_stochastic_block": order_index,
                    "source_phase_a_runtime_identity_sha256": source["runtime_identity_sha256"],
                    "source_phase_a_initial_state_sha256": source["initial_state_sha256"],
                    "source_phase_a_raw_behavioral_pair_jsonl_sha256": source["raw_behavioral_pair_jsonl_sha256"],
                    "eligibility_report_sha256": sha256_file(args.eligibility_report),
                    "eligibility_manifest_sha256": sha256_file(args.eligibility_manifest),
                    "registration_sha256": REGISTRATION_SHA256,
                    "phase_d_registry_sha256": PHASE_D_SHA256,
                    "phase_a_summary_sha256": PHASE_A_SUMMARY_SHA256,
                    "behavioral_status": "authorized_not_launched",
                    "analysis_unit": "policy-sampling rollout nested within condition; not an independent scene",
                }
                row["cell_sha256"] = sha256_bytes(canonical_json_bytes(row))
                rows.append(row)
    if len(rows) != 864 or len({row["cell_id"] for row in rows}) != 864:
        raise ReleaseError("Phase-D release must contain exactly 864 unique cells")
    queue_path = args.output_dir / "pi05_v3d001_stochastic_cells.jsonl"
    queue_path.write_text("".join(json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    amendment = {
        "schema_version": "vla-wam-shared-v3d001-pi05-stochastic-release-v1",
        "study_id": STUDY_ID,
        "registration_id": "V3-D001",
        "model_id": MODEL_ID,
        "status": "released_after_effective_seed_probe_zero_behavior_launched",
        "behavioral_release": True,
        "authorized_behavioral_cells": 864,
        "launched_behavioral_cells_at_release": 0,
        "completed_behavioral_cells_at_release": 0,
        "conditions": 54,
        "rollouts_per_condition": 16,
        "matched_scene_pairs": 27,
        "directions": 2,
        "eligibility_report": {"path": str(args.eligibility_report.resolve()), "sha256": sha256_file(args.eligibility_report)},
        "eligibility_manifest": {"path": str(args.eligibility_manifest.resolve()), "sha256": sha256_file(args.eligibility_manifest)},
        "queue": {"path": str(queue_path.resolve()), "sha256": sha256_file(queue_path), "bytes": queue_path.stat().st_size, "rows": 864},
        "source_phase_a_summary_sha256": PHASE_A_SUMMARY_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "phase_d_registry_sha256": PHASE_D_SHA256,
        "invariants": [
            "same environment seed and reset within every original condition",
            "only shared effective policy-sampling seed index changes across repeats",
            "LEFT and RIGHT share every sampling-seed index",
            "all valid behavioral failures remain in denominators",
            "infrastructure failures and partial attempts remain separate",
            "stochastic rollouts are nested observations, not independent scenes",
        ],
    }
    amendment_path = args.output_dir / "release_amendment.json"
    amendment_path.write_text(json.dumps(amendment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    release_manifest = {
        "schema_version": "vla-wam-shared-v3d001-pi05-stochastic-release-manifest-v1",
        "study_id": STUDY_ID,
        "registration_id": "V3-D001",
        "status": "exact_864_cell_queue_released_zero_behavior_launched",
        "counts": {"cells": 864, "conditions": 54, "environment_seeds": 27, "directions": 2, "sampling_seed_indices": 16, "launched": 0},
        "files": [
            {"path": str(queue_path.resolve()), "sha256": sha256_file(queue_path), "bytes": queue_path.stat().st_size},
            {"path": str(amendment_path.resolve()), "sha256": sha256_file(amendment_path), "bytes": amendment_path.stat().st_size},
            {"path": str(args.eligibility_report.resolve()), "sha256": sha256_file(args.eligibility_report), "bytes": args.eligibility_report.stat().st_size},
            {"path": str(args.eligibility_manifest.resolve()), "sha256": sha256_file(args.eligibility_manifest), "bytes": args.eligibility_manifest.stat().st_size},
        ],
    }
    manifest_path = args.output_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(release_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"queue": str(queue_path), "queue_sha256": sha256_file(queue_path), "release_amendment": str(amendment_path), "release_amendment_sha256": sha256_file(amendment_path), "release_manifest": str(manifest_path), "release_manifest_sha256": sha256_file(manifest_path), "authorized_behavioral_cells": 864, "launched": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--eligibility-report", type=Path, required=True)
    parser.add_argument("--eligibility-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

