#!/usr/bin/env python3
"""Build fixture-parameterized engineering-pilot reset and policy-seed registries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.fixture_qualification import (  # noqa: E402
    qualification_profile,
)
from tools.build_v4_object_pair_reset_registry import (  # noqa: E402
    canonical_json_bytes,
    deterministic_axis_jitter,
    sha256_file,
)

DEFAULT_SEED_MANIFEST = ROOT / "artifacts/online_correction_v4/seed_manifest.json"
EXPECTED_PILOT_COUNT = 24


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def pilot_env_seeds(
    *,
    seed_manifest: dict[str, Any],
    fixture_id: str,
    policy_id: str,
    env_seed_start: int,
    env_seed_end: int,
) -> tuple[int, ...]:
    expected = tuple(range(env_seed_start, env_seed_end + 1))
    rows = seed_manifest.get("engineering_pilot_seeds")
    if not isinstance(rows, list):
        raise ValueError("seed manifest lacks engineering-pilot rows")
    seeds = tuple(
        int(row["env_seed"])
        for row in rows
        if isinstance(row, dict)
        and row.get("policy") == policy_id
        and row.get("fixture") == fixture_id
        and row.get("cohort") == "engineering_pilot"
    )
    if len(seeds) != EXPECTED_PILOT_COUNT or seeds != expected:
        raise ValueError(
            f"seed manifest does not bind the expected 24 pilot resets for {fixture_id}"
        )
    return seeds


def derive_policy_seed(
    *,
    namespace: str,
    policy_id: str,
    fixture_id: str,
    offset: int,
) -> int:
    digest = hashlib.sha256(
        json.dumps(
            [namespace, "engineering-pilot", policy_id, fixture_id, offset],
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16) % (2**31)


def object_base_positions(base_reset: dict[str, Any]) -> dict[str, list[float]]:
    scene = base_reset.get("scene_receipt") or {}
    specs = scene.get("object_specs")
    if not isinstance(specs, dict):
        raise ValueError("base reset registry lacks scene object specs")
    positions: dict[str, list[float]] = {}
    for name, spec in specs.items():
        if not isinstance(spec, dict):
            raise ValueError("scene object spec must be an object")
        base = spec.get("base_position_robot_m")
        if not isinstance(base, list) or len(base) != 3:
            raise ValueError(f"{name} lacks base_position_robot_m")
        positions[str(name)] = [float(base[0]), float(base[1]), float(base[2])]
    return positions


def build_payloads(
    *,
    fixture_id: str,
    seed_manifest_path: Path,
    base_reset_registry_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = qualification_profile(fixture_id)
    if profile.pilot_allocation is None:
        raise ValueError(f"{fixture_id} has no pilot allocation profile")
    allocation = profile.pilot_allocation
    seed_manifest = load_json(seed_manifest_path)
    base_reset = load_json(base_reset_registry_path)
    if base_reset.get("fixture_id") != fixture_id:
        raise ValueError("base reset registry fixture mismatch")
    if base_reset.get("status") != "model_blind_candidate_not_released_for_inference":
        raise ValueError("base reset registry is not a model-blind candidate")
    namespace = seed_manifest.get("reservation", {}).get("policy_seed_namespace")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("seed manifest lacks the frozen policy seed namespace")
    env_seeds = pilot_env_seeds(
        seed_manifest=seed_manifest,
        fixture_id=fixture_id,
        policy_id=profile.policy_id,
        env_seed_start=allocation.env_seed_start,
        env_seed_end=allocation.env_seed_end,
    )
    policy_seeds = tuple(
        derive_policy_seed(
            namespace=namespace,
            policy_id=profile.policy_id,
            fixture_id=fixture_id,
            offset=offset,
        )
        for offset in range(EXPECTED_PILOT_COUNT)
    )
    if len(set(policy_seeds)) != EXPECTED_PILOT_COUNT:
        raise ValueError("derived pilot policy seeds collide")
    confirmatory_policy = set(seed_manifest.get("confirmatory_unique_policy_seeds", []))
    if confirmatory_policy.intersection(policy_seeds):
        raise ValueError("derived pilot policy seeds collide with confirmatory seeds")
    confirmatory_env = {
        int(row["env_seed"])
        for row in seed_manifest.get("confirmatory_rows", [])
        if isinstance(row, dict) and type(row.get("env_seed")) is int
    }
    if confirmatory_env.intersection(env_seeds):
        raise ValueError("pilot reset seeds collide with confirmatory resets")

    base_positions = object_base_positions(base_reset)
    resets: dict[str, Any] = {}
    for offset, env_seed in enumerate(env_seeds):
        dx = deterministic_axis_jitter(env_seed=env_seed, axis="x")
        dy = deterministic_axis_jitter(env_seed=env_seed, axis="y")
        resets[str(env_seed)] = {
            "block_index": offset,
            "jitter_robot_base_xy_m": [dx, dy],
            "positions_robot_base_m": {
                name: [
                    float(position[0]) + dx,
                    float(position[1]) + dy,
                    float(position[2]),
                ]
                for name, position in base_positions.items()
            },
        }
    source = {
        "seed_manifest": {
            "path": str(seed_manifest_path),
            "bytes": seed_manifest_path.stat().st_size,
            "sha256": sha256_file(seed_manifest_path),
        },
        "base_reset_registry": {
            "path": str(base_reset_registry_path),
            "bytes": base_reset_registry_path.stat().st_size,
            "sha256": sha256_file(base_reset_registry_path),
        },
    }
    reset_registry = {
        **base_reset,
        "qualification_scope": "engineering_pilot",
        "registered_env_seed_count": len(env_seeds),
        "registered_env_seed_min": min(env_seeds),
        "registered_env_seed_max": max(env_seeds),
        "resets_by_env_seed": resets,
        "pilot_seed_source": source,
        "release_boundary": (
            f"Model-blind {profile.family_id} engineering-pilot reset candidate only. "
            "These 24 resets are disjoint from confirmatory resets and require "
            "G2/G3 qualification before any policy request."
        ),
    }
    rows = [
        {
            "pilot_offset": offset,
            "env_seed": env_seed,
            "policy_seed": policy_seed,
        }
        for offset, (env_seed, policy_seed) in enumerate(
            zip(env_seeds, policy_seeds, strict=True)
        )
    ]
    policy_seed_registry = {
        "schema_version": profile.pilot_seed_registry_schema,
        "campaign_id": "online_correction_v4",
        "fixture_id": fixture_id,
        "policy_id": profile.policy_id,
        "cohort": "engineering_pilot",
        "status": "candidate_not_released_for_policy_requests",
        "behavioral_episode_count": 0,
        "model_request_count": 0,
        "rows": rows,
        "policy_seed_derivation": (
            "sha256([policy_seed_namespace,engineering-pilot,policy,fixture,"
            "pilot_offset]) first_64_bits mod 2^31"
        ),
        "confirmatory_env_seed_disjoint": True,
        "confirmatory_policy_seed_disjoint": True,
        "source": source,
        "release_boundary": (
            f"Prospective {profile.family_id} engineering-pilot allocation only. "
            "G5-G8 control which rows, if any, may be exposed to the policy."
        ),
    }
    return reset_registry, policy_seed_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEED_MANIFEST)
    parser.add_argument("--base-reset-registry", type=Path, required=True)
    parser.add_argument("--reset-output", type=Path, required=True)
    parser.add_argument("--seed-output", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.reset_output, args.seed_output):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite pilot registry: {output}")
    reset, seeds = build_payloads(
        fixture_id=args.fixture_id,
        seed_manifest_path=args.seed_manifest.resolve(),
        base_reset_registry_path=args.base_reset_registry.resolve(),
    )
    args.reset_output.parent.mkdir(parents=True, exist_ok=True)
    args.seed_output.parent.mkdir(parents=True, exist_ok=True)
    args.reset_output.write_bytes(canonical_json_bytes(reset))
    args.seed_output.write_bytes(canonical_json_bytes(seeds))
    print(
        json.dumps(
            {
                "fixture_id": args.fixture_id,
                "reset_registry": {
                    "path": str(args.reset_output),
                    "sha256": sha256_file(args.reset_output),
                },
                "policy_seed_registry": {
                    "path": str(args.seed_output),
                    "sha256": sha256_file(args.seed_output),
                },
                "pilot_count": len(seeds["rows"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
