#!/usr/bin/env python3
"""Build the frozen C7 engineering-pilot manifest and pilot-only runtime lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.geometry import build_prompt  # noqa: E402


FIXTURE_ID = "object_pair"
POLICY_ID = "cosmos3_nano_droid"
CHECKPOINT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
COSMOS_COMMIT = "411d25b2e35bc441126f48c44a4b93e1c0564274"
IMAGE_DIGEST = (
    "sha256:03f5ce7d090fbd378070a8216d0aedfc6e473c52da99b40b0cf53918612a297c"
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def artifact(path: Path, *, runtime_path: str | None = None) -> dict[str, Any]:
    result = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if runtime_path is not None:
        result["runtime_path"] = runtime_path
    return result


def write_exclusive(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def require_passing(
    payload: dict[str, Any],
    *,
    fixture_id: str = FIXTURE_ID,
) -> None:
    if payload.get("fixture_id") != fixture_id:
        raise ValueError("qualification fixture mismatch")
    if payload.get("passed") is not True or payload.get("status") != "passed":
        raise ValueError("qualification receipt is not passing")


def pilot_rows(
    *,
    pilot_seed_registry: dict[str, Any],
    config_sha256: str,
) -> list[dict[str, Any]]:
    source_rows = pilot_seed_registry.get("rows")
    if not isinstance(source_rows, list) or len(source_rows) != 24:
        raise ValueError("pilot seed registry must contain 24 rows")
    goals = ("left", "right", "front", "behind")
    rows: list[dict[str, Any]] = []
    for offset, source in enumerate(source_rows):
        if not isinstance(source, dict) or source.get("pilot_offset") != offset:
            raise ValueError("pilot seed registry row order differs")
        if offset < 8:
            pilot_kind = "stationary"
            scenario = "original_sham"
        elif offset < 16:
            pilot_kind = "stationary"
            scenario = "destination_static"
        else:
            pilot_kind = "motion"
            scenario = "move_stop"
        goal = goals[offset % len(goals)]
        env_seed = int(source["env_seed"])
        policy_seed = int(source["policy_seed"])
        factors = {
            "policy": POLICY_ID,
            "goal": goal,
            "wording": "direct",
            "scenario": scenario,
            "schedule": "standard",
            "named_reference": "single",
        }
        identity = hashlib.sha256(
            canonical_json_bytes([FIXTURE_ID, "engineering_pilot", offset, factors])
        ).hexdigest()[:16]
        rows.append(
            {
                "schema_version": 1,
                "manifest_type": "planning_manifest",
                "runtime_bound": True,
                "episode_id": f"online_correction_v4-C7-pilot-{offset:02d}-{identity}",
                "campaign": "online_correction_v4",
                "family": "C7",
                "fixture": FIXTURE_ID,
                "block_id": offset,
                "block_key": f"{FIXTURE_ID}:engineering_pilot:{offset}",
                "env_seed": env_seed,
                "policy_seed": policy_seed,
                "cohort": "engineering_pilot",
                "priority": "engineering",
                "factors": factors,
                "prefix_group_id": f"C7-pilot-independent-{offset:02d}",
                "execution_group": (
                    f"{POLICY_ID}:{FIXTURE_ID}-pilot-shard-{offset % 8:02d}"
                ),
                "execution_order_key": f"pilot-{offset:02d}",
                "execution_order": offset,
                "config_sha256": config_sha256,
                "reuse_episode_ids": [],
                "counterbalance": {
                    "event_phase_fraction": (offset % 4) / 4.0,
                    "state_index": (offset // 4) % 4,
                    "physical_translation_sign": (
                        1 if ((offset // 4) % 2 == 0) else -1
                    ),
                    "pilot_kind": pilot_kind,
                },
                "prompt_recipe": {
                    "template": build_prompt(
                        "sponge",
                        "tray",
                        goal,
                        "direct",
                        horizontal=True,
                    ),
                    "object_role": "sponge",
                    "reference_role": "tray",
                },
            }
        )
    if len({row["env_seed"] for row in rows}) != 24:
        raise ValueError("pilot environment seeds are not unique")
    if len({row["policy_seed"] for row in rows}) != 24:
        raise ValueError("pilot policy seeds are not unique")
    if sum(row["counterbalance"]["pilot_kind"] == "stationary" for row in rows) != 16:
        raise ValueError("pilot manifest does not contain 16 stationary episodes")
    if sum(row["counterbalance"]["pilot_kind"] == "motion" for row in rows) != 8:
        raise ValueError("pilot manifest does not contain 8 motion episodes")
    return rows


def build_nano_seed_registry(
    *,
    rows: list[dict[str, Any]],
    source_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "v4-nano-policy-seed-registry-v1",
        "campaign_id": "online_correction_v4",
        "fixture_id": FIXTURE_ID,
        "policy_id": POLICY_ID,
        "scope": "g7_engineering_pilot",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "allowed_sampling_seeds": [row["policy_seed"] for row in rows],
        "source_pilot_seed_registry": artifact(source_path),
        "behavioral_episode_count": 24,
        "release_boundary": (
            "Allows only the 24 prospectively allocated, excluded C7 engineering-"
            "pilot sampling seeds. It does not authorize confirmatory policy seeds."
        ),
    }


def release_pilot_resets(
    *,
    candidate: dict[str, Any],
    candidate_path: Path,
    pilot_g2: dict[str, Any],
    pilot_g2_path: Path,
    pilot_g3: dict[str, Any],
    pilot_g3_path: Path,
) -> dict[str, Any]:
    if candidate.get("fixture_id") != FIXTURE_ID:
        raise ValueError("pilot reset fixture mismatch")
    if candidate.get("qualification_scope") != "engineering_pilot":
        raise ValueError("reset registry is not the engineering-pilot allocation")
    if candidate.get("registered_env_seed_count") != 24:
        raise ValueError("pilot reset registry must contain 24 seeds")
    if candidate.get("status") != "model_blind_candidate_not_released_for_inference":
        raise ValueError("pilot reset registry is not a model-blind candidate")
    require_passing(pilot_g2)
    require_passing(pilot_g3)
    if pilot_g2.get("expected_seed_count") != 24:
        raise ValueError("pilot G2 coverage differs")
    if (
        pilot_g3.get("qualification_scope") != "engineering_pilot"
        or pilot_g3.get("expected_scripted_check_count") != 112
        or pilot_g3.get("observed_scripted_check_count") != 112
    ):
        raise ValueError("pilot G3 coverage differs")
    return {
        **candidate,
        "status": "released_for_policy_inference",
        "qualification_release_basis": {
            "candidate": artifact(candidate_path),
            "pilot_g2": artifact(pilot_g2_path),
            "pilot_g3": artifact(pilot_g3_path),
        },
        "release_boundary": (
            "Released only for the 24 excluded C7 engineering-pilot episodes. "
            "These resets are disjoint from confirmatory C7 resets."
        ),
    }


def receipt_binding(
    *,
    path: Path,
    runtime_root: str,
    family_ids: list[str] | None = None,
    passed: bool = True,
) -> dict[str, Any]:
    return {
        "passed": passed,
        "family_ids": family_ids if family_ids is not None else ["C7"],
        "uri": f"{runtime_root}/{path.relative_to(ROOT)}",
        "sha256": sha256_file(path),
    }


def build_runtime_lock(
    *,
    manifest_path: Path,
    source_commit: str,
    runtime_root: str,
    released_reset_path: Path,
    nano_seed_registry_path: Path,
    geometry_path: Path,
    checkpoint_registry_path: Path,
    g2_path: Path,
    g3_path: Path,
    g4_path: Path,
    g5_path: Path,
    g6_path: Path,
) -> dict[str, Any]:
    if not HEX40.fullmatch(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    runner_path = ROOT / "tools/run_online_correction_v4.py"
    scorer_path = ROOT / "experiments/online_correction_v4/droid_scorer.py"
    campaign_path = ROOT / "docs/online_correction_v4/campaign.json"
    geometry = load_json(geometry_path)
    g5 = load_json(g5_path)
    require_passing(g5)
    d_cap_m = float(geometry["d_cap_m"])
    native_dt = 1.0 / 15.0
    timing = load_json(campaign_path)["timing"]

    def quantized(key: str) -> float:
        return math.ceil(float(timing[key]) / native_dt - 1e-10) * native_dt

    blocked = {
        family: (
            "Not included in the fixture-scoped C7 engineering-pilot release"
        )
        for family in ("C1", "C2", "C3", "C4", "C5", "C6", "C8")
    }
    receipts = {
        "source_and_checkpoint_identity": receipt_binding(path=g4_path, runtime_root=runtime_root),
        "historical_seed_collision_audit": receipt_binding(
            path=nano_seed_registry_path,
            runtime_root=runtime_root,
        ),
        "geometry_and_scripted_feasibility": receipt_binding(path=g3_path, runtime_root=runtime_root),
        "prompt_and_frame_review": receipt_binding(path=g2_path, runtime_root=runtime_root),
        "natural_trigger_and_first_release": receipt_binding(path=g5_path, runtime_root=runtime_root),
        "controlled_clock_and_queue": receipt_binding(path=g6_path, runtime_root=runtime_root),
        "prefix_replay_or_declared_fallback": receipt_binding(path=g5_path, runtime_root=runtime_root),
        "trace_video_and_atomic_storage": receipt_binding(path=g6_path, runtime_root=runtime_root),
        "scorer_independent_cases": receipt_binding(path=g6_path, runtime_root=runtime_root),
        "cluster_lane_qualification": receipt_binding(path=g4_path, runtime_root=runtime_root),
        "engineering_pilots_complete": receipt_binding(
            path=nano_seed_registry_path,
            runtime_root=runtime_root,
            passed=False,
        ),
        "frozen_analysis_and_inventory": receipt_binding(
            path=manifest_path,
            runtime_root=runtime_root,
            passed=False,
        ),
    }
    return {
        "schema_version": 1,
        "campaign_id": "online_correction_v4",
        "config_sha256": sha256_file(campaign_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_commit": source_commit,
        "release_status": "PILOT_RELEASED",
        "released_families": ["C7"],
        "blocked_families": blocked,
        "prefix_mode": "independent_natural_rollout_fallback",
        "prefix_mode_receipt_sha256": sha256_file(g5_path),
        "writer_contract": {
            "schema_version": "v4-droid-writer-contract-v1",
            "output_parent_uri": "/data/users/ali/vla_wam/raw/v4/g7-object-pair",
            "viewport_video_required": True,
            "write_once_attempt_directories": True,
            "incremental_fsync_required": True,
            "required_streams": [
                "viewport_video",
                "trajectory",
                "requests",
                "observations",
                "events",
            ],
        },
        "runner": {
            "commit": source_commit,
            "entrypoint": f"{runtime_root}/tools/run_online_correction_v4.py",
            "sha256": sha256_file(runner_path),
        },
        "policies": {
            POLICY_ID: {
                "checkpoint_sha256": sha256_file(checkpoint_registry_path),
                "runtime_image_digest": IMAGE_DIGEST,
                "native_control_dt_s": native_dt,
                "checkpoint_uri": "/data/users/ali/vla_wam/checkpoints/cosmos3_nano_policy_droid",
                "integration_commit": COSMOS_COMMIT,
                "achieved_delay_s": quantized("emulated_observation_action_delay_s"),
                "achieved_standard_query_period_s": quantized("standard_query_period_s"),
                "achieved_fast_query_period_s": quantized("fast_query_period_s"),
                "prediction_horizon_actions": 32,
                "policy_reset_and_history_contract_uri": f"{runtime_root}/{g4_path.relative_to(ROOT)}",
                "allowed_seed_registry_uri": f"{runtime_root}/{nano_seed_registry_path.relative_to(ROOT)}",
                "allowed_seed_registry_sha256": sha256_file(nano_seed_registry_path),
                "checkpoint_revision": CHECKPOINT_REVISION,
            }
        },
        "fixtures": {
            FIXTURE_ID: {
                "geometry_sha256": sha256_file(geometry_path),
                "geometry_uri": f"{runtime_root}/{geometry_path.relative_to(ROOT)}",
                "scorer_sha256": sha256_file(scorer_path),
                "scorer_uri": f"{runtime_root}/{scorer_path.relative_to(ROOT)}",
                "reset_registry_sha256": sha256_file(released_reset_path),
                "reset_registry_uri": f"{runtime_root}/{released_reset_path.relative_to(ROOT)}",
                "calibration_scale": 0.5,
                "D_cap_m": d_cap_m,
                "frame_transform_uri": f"{runtime_root}/{geometry_path.relative_to(ROOT)}",
                "goal_geometry_and_tolerances_uri": f"{runtime_root}/{geometry_path.relative_to(ROOT)}",
                "trigger_release_detector_uri": f"{runtime_root}/{g5_path.relative_to(ROOT)}",
                "intervention_trajectory_registry_uri": f"{runtime_root}/{g3_path.relative_to(ROOT)}",
                "scoring_and_visibility_thresholds_uri": f"{runtime_root}/{g6_path.relative_to(ROOT)}",
            }
        },
        "receipts": receipts,
        "pilot_release_boundary": (
            "PILOT_RELEASED authorizes only engineering_pilot manifest rows. "
            "Confirmatory dispatch remains fail-closed until G7 and G8 pass and "
            "a separate RELEASED lock is generated."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--runtime-root",
        default="/data/users/ali/vla_wam/src/steerable-v4-c7-pilot-g2",
    )
    parser.add_argument("--pilot-seed-registry", type=Path, required=True)
    parser.add_argument("--candidate-reset-registry", type=Path, required=True)
    parser.add_argument("--pilot-g2", type=Path, required=True)
    parser.add_argument("--pilot-g3", type=Path, required=True)
    parser.add_argument("--g4", type=Path, required=True)
    parser.add_argument("--g5", type=Path, required=True)
    parser.add_argument("--g6", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--checkpoint-registry", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--queue-manifest-out", type=Path, required=True)
    parser.add_argument("--nano-seed-registry-out", type=Path, required=True)
    parser.add_argument("--released-reset-out", type=Path, required=True)
    parser.add_argument("--runtime-lock-out", type=Path, required=True)
    args = parser.parse_args()
    outputs = (
        args.manifest_out,
        args.queue_manifest_out,
        args.nano_seed_registry_out,
        args.released_reset_out,
        args.runtime_lock_out,
    )
    for output in outputs:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite pilot release: {output}")

    campaign_path = ROOT / "docs/online_correction_v4/campaign.json"
    rows = pilot_rows(
        pilot_seed_registry=load_json(args.pilot_seed_registry),
        config_sha256=sha256_file(campaign_path),
    )
    manifest_body = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    write_exclusive(args.manifest_out.resolve(), manifest_body)
    manifest_sha256 = sha256_file(args.manifest_out.resolve())
    write_exclusive(
        args.queue_manifest_out.resolve(),
        canonical_json_bytes(
            {
                "schema_version": 1,
                "campaign_id": "online_correction_v4",
                "release_status": "PILOT_RELEASED",
                "queue_path": str(args.manifest_out),
                "row_count": 24,
                "excluded_engineering_policy_pilots": 24,
                "expected_confirmatory_episodes": 0,
                "queue_sha256": manifest_sha256,
                "frozen_queue_sha256": manifest_sha256,
                "planning_manifest_sha256": manifest_sha256,
                "rows_by_execution_group": {
                    group_id: 3
                    for group_id in sorted(
                        {row["execution_group"] for row in rows}
                    )
                },
                "release_boundary": (
                    "Frozen C7 engineering-pilot queue only; these rows are excluded "
                    "from the 17,664 confirmatory episodes."
                ),
            }
        ),
    )
    nano_registry = build_nano_seed_registry(
        rows=rows,
        source_path=args.pilot_seed_registry.resolve(),
    )
    write_exclusive(
        args.nano_seed_registry_out.resolve(),
        canonical_json_bytes(nano_registry),
    )
    released_resets = release_pilot_resets(
        candidate=load_json(args.candidate_reset_registry),
        candidate_path=args.candidate_reset_registry.resolve(),
        pilot_g2=load_json(args.pilot_g2),
        pilot_g2_path=args.pilot_g2.resolve(),
        pilot_g3=load_json(args.pilot_g3),
        pilot_g3_path=args.pilot_g3.resolve(),
    )
    write_exclusive(
        args.released_reset_out.resolve(),
        canonical_json_bytes(released_resets),
    )
    runtime_lock = build_runtime_lock(
        manifest_path=args.manifest_out.resolve(),
        source_commit=args.source_commit,
        runtime_root=args.runtime_root.rstrip("/"),
        released_reset_path=args.released_reset_out.resolve(),
        nano_seed_registry_path=args.nano_seed_registry_out.resolve(),
        geometry_path=args.geometry.resolve(),
        checkpoint_registry_path=args.checkpoint_registry.resolve(),
        g2_path=args.pilot_g2.resolve(),
        g3_path=args.pilot_g3.resolve(),
        g4_path=args.g4.resolve(),
        g5_path=args.g5.resolve(),
        g6_path=args.g6.resolve(),
    )
    write_exclusive(
        args.runtime_lock_out.resolve(),
        canonical_json_bytes(runtime_lock),
    )
    print(
        json.dumps(
            {
                "pilot_episode_count": len(rows),
                "stationary_episode_count": 16,
                "motion_episode_count": 8,
                "manifest": artifact(args.manifest_out.resolve()),
                "queue_manifest": artifact(args.queue_manifest_out.resolve()),
                "nano_seed_registry": artifact(args.nano_seed_registry_out.resolve()),
                "released_reset_registry": artifact(args.released_reset_out.resolve()),
                "runtime_lock": artifact(args.runtime_lock_out.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
