#!/usr/bin/env python3
"""Build disjoint C8 engineering-pilot reset and policy-seed registries."""

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

from tools.build_v4_second_stack_reset_registry import (  # noqa: E402
    BASE_POSITIONS_SCENE_XY_M,
    FIXTURE_ID,
    JITTER_ALGORITHM,
    SOURCE_OBJECT,
    REFERENCE_OBJECT,
    canonical_json_bytes,
    sha256_file,
)

DEFAULT_SEED_MANIFEST = ROOT / "artifacts/online_correction_v4/seed_manifest.json"
DEFAULT_BASE_RESET = (
    ROOT
    / "artifacts/online_correction_v4/setup/second_stack_reset_registry.candidate.json"
)
DEFAULT_RESET_OUTPUT = (
    ROOT
    / "artifacts/online_correction_v4/setup/second_stack_pilot_reset_registry.candidate.json"
)
DEFAULT_SEED_OUTPUT = (
    ROOT
    / "artifacts/online_correction_v4/setup/second_stack_pilot_seed_registry.candidate.json"
)
POLICY_ID = "groot_bridge_widowx"
EXPECTED_PILOT_COUNT = 24


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _axis_jitter(env_seed: int, axis: str) -> float:
    digest = hashlib.sha256(
        f"online-correction-v4:{FIXTURE_ID}:{env_seed}:{axis}:{JITTER_ALGORITHM}".encode(
            "utf-8"
        )
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)
    return (2.0 * unit - 1.0) * 0.01


def pilot_env_seeds(seed_manifest: dict[str, Any]) -> tuple[int, ...]:
    rows = seed_manifest.get("engineering_pilot_seeds")
    if not isinstance(rows, list):
        raise ValueError("seed manifest lacks engineering-pilot rows")
    seeds = tuple(
        int(row["env_seed"])
        for row in rows
        if isinstance(row, dict)
        and row.get("policy") == POLICY_ID
        and row.get("fixture") == FIXTURE_ID
        and row.get("cohort") == "engineering_pilot"
    )
    expected = tuple(range(2110000900, 2110000924))
    if seeds != expected:
        raise ValueError("seed manifest does not bind the expected 24 C8 pilot resets")
    return seeds


def derive_policy_seed(namespace: str, offset: int) -> int:
    digest = hashlib.sha256(
        json.dumps(
            [namespace, "engineering-pilot", POLICY_ID, FIXTURE_ID, offset],
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16) % (2**31)


def build_payloads(
    *,
    seed_manifest_path: Path,
    base_reset_registry_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_manifest = load_json(seed_manifest_path)
    base_reset = load_json(base_reset_registry_path)
    if base_reset.get("fixture_id") != FIXTURE_ID:
        raise ValueError("base reset registry fixture mismatch")
    if base_reset.get("status") != "model_blind_candidate_not_released_for_inference":
        raise ValueError("base reset registry is not a model-blind candidate")
    namespace = seed_manifest.get("reservation", {}).get("policy_seed_namespace")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("seed manifest lacks the frozen policy seed namespace")
    env_seeds = pilot_env_seeds(seed_manifest)
    policy_seeds = tuple(
        derive_policy_seed(namespace, offset)
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

    resets: dict[str, Any] = {}
    for offset, env_seed in enumerate(env_seeds):
        dx = _axis_jitter(env_seed=env_seed, axis="x")
        dy = _axis_jitter(env_seed=env_seed, axis="y")
        resets[str(env_seed)] = {
            "block_index": offset,
            "jitter_scene_xy_m": [dx, dy],
            "positions_scene_xy_m": {
                SOURCE_OBJECT: _translated_xy(
                    BASE_POSITIONS_SCENE_XY_M[SOURCE_OBJECT], dx, dy
                ),
                REFERENCE_OBJECT: _translated_xy(
                    BASE_POSITIONS_SCENE_XY_M[REFERENCE_OBJECT], dx, dy
                ),
            },
        }
    source = {
        "seed_manifest": {
            "path": str(seed_manifest_path.relative_to(ROOT)),
            "bytes": seed_manifest_path.stat().st_size,
            "sha256": sha256_file(seed_manifest_path),
        },
        "base_reset_registry": {
            "path": str(base_reset_registry_path.relative_to(ROOT)),
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
            "Model-blind C8 engineering-pilot reset candidate only. These 24 "
            "resets are disjoint from confirmatory C8 resets and require G2/G3 "
            "qualification before any policy request."
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
        "schema_version": "v4-second-stack-engineering-pilot-seed-registry-v1",
        "campaign_id": "online_correction_v4",
        "fixture_id": FIXTURE_ID,
        "policy_id": POLICY_ID,
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
            "Prospective C8 engineering-pilot allocation only. G5-G8 control "
            "which rows, if any, may be exposed to the policy."
        ),
    }
    return reset_registry, policy_seed_registry


def _translated_xy(position: tuple[float, float], dx: float, dy: float) -> list[float]:
    return [float(position[0]) + dx, float(position[1]) + dy]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEED_MANIFEST)
    parser.add_argument("--base-reset-registry", type=Path, default=DEFAULT_BASE_RESET)
    parser.add_argument("--reset-output", type=Path, default=DEFAULT_RESET_OUTPUT)
    parser.add_argument("--seed-output", type=Path, default=DEFAULT_SEED_OUTPUT)
    args = parser.parse_args()
    for output in (args.reset_output, args.seed_output):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite pilot registry: {output}")
    reset, seeds = build_payloads(
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
