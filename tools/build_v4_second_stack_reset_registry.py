#!/usr/bin/env python3
"""Build the model-blind C8 SimplerEnv/WidowX reset candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "artifacts/online_correction_v4/setup"
    / "second_stack_reset_registry.candidate.json"
)
SCHEMA = "v4-simpler-env-widowx-reset-registry-v1"
STATUS = "model_blind_candidate_not_released_for_inference"
FAMILY_ID = "C8"
FIXTURE_ID = "second_stack"
ENV_NAME = "simpler_env_widowx/widowx_stack_cube"
SOURCE_OBJECT = "baked_green_cube_3cm"
REFERENCE_OBJECT = "baked_yellow_cube_3cm"
GR00T_REPO = "https://github.com/NVIDIA/Isaac-GR00T.git"
GR00T_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
SIMPLER_ENV_REPO = "https://github.com/squarefk/SimplerEnv.git"
SIMPLER_ENV_COMMIT = "8a2d286c926c1371927caa7651a412b4cc331756"
MANISKILL_COMMIT = "c2a9e87c186300b694da6f2497dd68d2c347a4b7"
CHECKPOINT_REPO = "nvidia/GR00T-N1.7-SimplerEnv-Bridge"
CHECKPOINT_REVISION = "940134b3c2948ccfdf8e7393f2d2ca869dc42833"
JITTER_HALF_RANGE_M = 0.01
JITTER_ALGORITHM = "sha256_u64_independent_axes_common_scene_translation_v1"
BASE_POSITIONS_SCENE_XY_M = {
    SOURCE_OBJECT: (-0.21, -0.05),
    REFERENCE_OBJECT: (-0.11, 0.05),
}
FIXTURE_FILES = {
    "environment_registry": {
        "path": "simpler_env/__init__.py",
        "sha256": "d12b2be350fb5ff2671e4b1334d67c7afc6bef90919db6e9e15d42abad393c61",
    },
    "task_implementation": {
        "path": (
            "ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/"
            "put_on_in_scene.py"
        ),
        "sha256": "977fc830b8fb3903b72862d7082bacf06fb3d092f867a93be503d00671bf5a74",
    },
    "model_metadata": {
        "path": (
            "ManiSkill2_real2sim/data/custom/"
            "info_bridge_custom_baked_tex_v0.json"
        ),
        "sha256": "a25bdb72746bd51e7cb612d95ace97b535623ba6b79367084c4fa3ba62d08eda",
    },
    "source_collision": {
        "path": (
            "ManiSkill2_real2sim/data/custom/models/"
            "baked_green_cube_3cm/collision.obj"
        ),
        "sha256": "b57ba99c4a72caa36e99329c604f412834014726f2fc077eab6b07629abbc237",
    },
    "source_visual": {
        "path": (
            "ManiSkill2_real2sim/data/custom/models/"
            "baked_green_cube_3cm/textured.dae"
        ),
        "sha256": "90a1904406d3c4d303b583f65220b1e91a8bacd3731da9ae25ce9c1c3a7297c9",
    },
    "reference_collision": {
        "path": (
            "ManiSkill2_real2sim/data/custom/models/"
            "baked_yellow_cube_3cm/collision.obj"
        ),
        "sha256": "b57ba99c4a72caa36e99329c604f412834014726f2fc077eab6b07629abbc237",
    },
    "reference_visual": {
        "path": (
            "ManiSkill2_real2sim/data/custom/models/"
            "baked_yellow_cube_3cm/textured.dae"
        ),
        "sha256": "0eedc10a63594cd2b6a2af67faffbbcdbfb29b77edc9e0e65884e532bb53a364",
    },
}


class SecondStackRegistryError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SecondStackRegistryError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SecondStackRegistryError(
                f"{path}:{line_number} must contain a JSON object"
            )
        rows.append(value)
    return rows


def _axis_jitter(env_seed: int, axis: str) -> float:
    if axis not in {"x", "y"}:
        raise SecondStackRegistryError("jitter axis must be x or y")
    digest = hashlib.sha256(
        f"online-correction-v4:{FIXTURE_ID}:{env_seed}:{axis}:{JITTER_ALGORITHM}".encode(
            "utf-8"
        )
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)
    value = (2.0 * unit - 1.0) * JITTER_HALF_RANGE_M
    if not math.isfinite(value):
        raise SecondStackRegistryError("reset jitter is non-finite")
    return value


def _translated_xy(position: Iterable[float], dx: float, dy: float) -> list[float]:
    x, y = (float(value) for value in position)
    return [x + dx, y + dy]


def _family_blocks(
    *,
    campaign: Mapping[str, Any],
    queue_rows: Iterable[Mapping[str, Any]],
) -> dict[int, dict[str, int]]:
    family_rows = [
        row
        for row in campaign.get("families", [])
        if isinstance(row, Mapping) and row.get("id") == FAMILY_ID
    ]
    fixture_cfg = (campaign.get("fixtures") or {}).get(FIXTURE_ID)
    seed_cfg = campaign.get("seed_reservation")
    if (
        len(family_rows) != 1
        or not isinstance(fixture_cfg, Mapping)
        or not isinstance(seed_cfg, Mapping)
    ):
        raise SecondStackRegistryError("campaign C8 allocation is missing")
    block_count = family_rows[0].get("blocks")
    base = seed_cfg.get("environment_base")
    stride = seed_cfg.get("fixture_stride")
    slot = fixture_cfg.get("seed_slot")
    if any(type(value) is not int for value in (block_count, base, stride, slot)):
        raise SecondStackRegistryError("campaign C8 seed allocation must be integral")
    observed: dict[int, dict[str, int]] = {}
    for row in queue_rows:
        if row.get("family") != FAMILY_ID:
            continue
        if row.get("fixture") != FIXTURE_ID:
            raise SecondStackRegistryError("C8 queue fixture differs")
        block_id = row.get("block_id")
        env_seed = row.get("env_seed")
        counterbalance = row.get("counterbalance")
        sign = (
            counterbalance.get("physical_translation_sign")
            if isinstance(counterbalance, Mapping)
            else None
        )
        if (
            type(block_id) is not int
            or type(env_seed) is not int
            or sign not in {-1, 1}
        ):
            raise SecondStackRegistryError("C8 reset identity is invalid")
        selected = {"env_seed": env_seed, "physical_translation_sign": int(sign)}
        prior = observed.setdefault(block_id, selected)
        if prior != selected:
            raise SecondStackRegistryError(
                f"C8 block {block_id} reset identity differs across cells"
            )
    if set(observed) != set(range(block_count)):
        raise SecondStackRegistryError("C8 queue must contain every allocated block")
    expected = [base + stride * slot + index for index in range(block_count)]
    if [observed[index]["env_seed"] for index in range(block_count)] != expected:
        raise SecondStackRegistryError("C8 environment seed namespace differs")
    if {
        sum(row["physical_translation_sign"] == sign for row in observed.values())
        for sign in (-1, 1)
    } != {block_count // 2}:
        raise SecondStackRegistryError("C8 physical translation signs are not balanced")
    return observed


def build_registry(
    *,
    campaign_path: Path,
    queue_path: Path,
) -> dict[str, Any]:
    campaign = load_json(campaign_path)
    fixture = (campaign.get("fixtures") or {}).get(FIXTURE_ID)
    if (
        not isinstance(fixture, Mapping)
        or fixture.get("seed_slot") != 5
        or float(fixture.get("nominal_translation_m", 0.0)) != 0.08
    ):
        raise SecondStackRegistryError("campaign second_stack fixture differs")
    blocks = _family_blocks(
        campaign=campaign,
        queue_rows=load_jsonl(queue_path),
    )
    resets: dict[str, dict[str, Any]] = {}
    for block_id in sorted(blocks):
        row = blocks[block_id]
        env_seed = row["env_seed"]
        dx = _axis_jitter(env_seed, "x")
        dy = _axis_jitter(env_seed, "y")
        resets[str(env_seed)] = {
            "block_index": block_id,
            "official_episode_id": block_id % 24,
            "jitter_scene_xy_m": [dx, dy],
            "positions_scene_xy_m": {
                name: _translated_xy(position, dx, dy)
                for name, position in BASE_POSITIONS_SCENE_XY_M.items()
            },
            "physical_translation_sign": row["physical_translation_sign"],
        }
    return {
        "schema_version": SCHEMA,
        "campaign_id": "online_correction_v4",
        "family_id": FAMILY_ID,
        "fixture_id": FIXTURE_ID,
        "status": STATUS,
        "qualification_scope": "confirmatory",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "stack": "simplerenv_bridge_widowx",
        "environment_name": ENV_NAME,
        "control_frequency_hz": 5,
        "native_control_dt_s": 0.2,
        "object_roles": {
            "target": {
                "scene_object": SOURCE_OBJECT,
                "prompt_name": "green block",
            },
            "reference": {
                "scene_object": REFERENCE_OBJECT,
                "prompt_name": "yellow block",
            },
        },
        "object_specs": {
            SOURCE_OBJECT: {
                "role": "target",
                "primitive": "box_mesh",
                "dimensions_m": [0.03, 0.03, 0.03],
                "density_kg_m3": 1000,
                "base_position_scene_xy_m": list(
                    BASE_POSITIONS_SCENE_XY_M[SOURCE_OBJECT]
                ),
            },
            REFERENCE_OBJECT: {
                "role": "reference",
                "primitive": "box_mesh",
                "dimensions_m": [0.03, 0.03, 0.03],
                "density_kg_m3": 1000,
                "base_position_scene_xy_m": list(
                    BASE_POSITIONS_SCENE_XY_M[REFERENCE_OBJECT]
                ),
            },
        },
        "registered_env_seed_count": len(resets),
        "registered_env_seed_min": min(int(seed) for seed in resets),
        "registered_env_seed_max": max(int(seed) for seed in resets),
        "resets_by_env_seed": resets,
        "reset_jitter": {
            "algorithm": JITTER_ALGORITHM,
            "x_half_range_m": JITTER_HALF_RANGE_M,
            "y_half_range_m": JITTER_HALF_RANGE_M,
            "z_jitter_m": 0.0,
            "application": (
                "one common SimplerEnv scene-frame x/y translation is applied to "
                "both cubes after the official reset and before the settling dwell; "
                "each actor retains its official live reset z coordinate"
            ),
            "z_binding": "official_live_reset_actor_z",
            "policy_outcome_used": False,
        },
        "external_stack_identity": {
            "gr00t_repository": GR00T_REPO,
            "gr00t_commit": GR00T_COMMIT,
            "simpler_env_repository": SIMPLER_ENV_REPO,
            "simpler_env_commit": SIMPLER_ENV_COMMIT,
            "maniskill2_real2sim_commit": MANISKILL_COMMIT,
            "checkpoint_repository": CHECKPOINT_REPO,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "embodiment_tag": "SIMPLER_ENV_WIDOWX",
            "fixture_files": FIXTURE_FILES,
        },
        "source_identity": {
            "campaign_path": portable_path(campaign_path),
            "campaign_sha256": sha256_file(campaign_path),
            "queue_path": portable_path(queue_path),
            "queue_sha256": sha256_file(queue_path),
        },
        "release_boundary": (
            "Prospective model-blind C8 reset candidate only. The pinned "
            "SimplerEnv/WidowX stack, all reset rows, scene axes, camera, stability, "
            "swept paths, scripted placement, runtime interface, and scorer must "
            "qualify before any GR00T policy inference."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=ROOT / "docs/online_correction_v4/campaign.json",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=ROOT / "artifacts/online_correction_v4/queue.jsonl",
    )
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = build_registry(
        campaign_path=args.campaign.resolve(),
        queue_path=args.queue.resolve(),
    )
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(payload)
    destination.write_bytes(body)
    print(
        json.dumps(
            {
                "path": str(destination),
                "sha256": hashlib.sha256(body).hexdigest(),
                "registered_env_seed_count": payload["registered_env_seed_count"],
                "status": payload["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
