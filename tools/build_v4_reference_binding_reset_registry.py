#!/usr/bin/env python3
"""Build the model-blind C2 two-reference reset candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCENE = (
    ROOT
    / "experiments/online_correction_v4/droid_task_files/scene_assets"
    / "cube_two_bowls_reference_binding.usda"
)
OUTPUT = (
    ROOT
    / "artifacts/online_correction_v4/setup"
    / "reference_binding_reset_registry.candidate.json"
)
SCHEMA = "v4-droid-reference-binding-reset-registry-v1"
STATUS = "model_blind_candidate_not_released_for_inference"
SEED_MIN = 2_100_010_000
SEED_MAX = 2_100_010_127
JITTER_HALF_RANGE_M = 0.015
BASE_POSITIONS = {
    "cube": (0.36, 0.0, 0.073),
    "left_bowl": (0.48, 0.18, 0.0605),
    "right_bowl": (0.48, -0.18, 0.0605),
}


class ReferenceBindingRegistryError(ValueError):
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


def _axis_jitter(env_seed: int, axis: str) -> float:
    digest = hashlib.sha256(
        f"online_correction_v4:reference_binding:{env_seed}:{axis}".encode("utf-8")
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)
    value = (2.0 * unit - 1.0) * JITTER_HALF_RANGE_M
    if not math.isfinite(value):
        raise ReferenceBindingRegistryError("reset jitter is non-finite")
    return value


def _load_c2_blocks(queue_path: Path) -> dict[int, dict[str, Any]]:
    blocks: dict[int, dict[str, Any]] = {}
    with queue_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("family") != "C2":
                continue
            if row.get("fixture") != "reference_binding":
                raise ReferenceBindingRegistryError(
                    f"queue line {line_number}: C2 fixture differs"
                )
            block_id = row.get("block_id")
            env_seed = row.get("env_seed")
            counterbalance = row.get("counterbalance")
            if (
                not isinstance(block_id, int)
                or not isinstance(env_seed, int)
                or not isinstance(counterbalance, dict)
            ):
                raise ReferenceBindingRegistryError(
                    f"queue line {line_number}: C2 reset identity is invalid"
                )
            selected = {
                "env_seed": env_seed,
                "physical_A_color": counterbalance.get("physical_A_color"),
                "physical_A_start_side": counterbalance.get(
                    "physical_A_start_side"
                ),
                "physical_A_diagonal_signs": counterbalance.get(
                    "physical_A_diagonal_signs"
                ),
                "state_index": counterbalance.get("state_index"),
            }
            prior = blocks.setdefault(block_id, selected)
            if prior != selected:
                raise ReferenceBindingRegistryError(
                    f"queue block {block_id}: reset counterbalance differs across cells"
                )
    if set(blocks) != set(range(128)):
        raise ReferenceBindingRegistryError("C2 queue must contain blocks 0 through 127")
    if {row["env_seed"] for row in blocks.values()} != set(
        range(SEED_MIN, SEED_MAX + 1)
    ):
        raise ReferenceBindingRegistryError("C2 environment seed allocation differs")
    return blocks


def _translated(position: Iterable[float], dx: float, dy: float) -> list[float]:
    x, y, z = (float(value) for value in position)
    return [x + dx, y + dy, z]


def build_registry(
    *,
    campaign_path: Path,
    queue_path: Path,
) -> dict[str, Any]:
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    fixture = campaign.get("fixtures", {}).get("reference_binding")
    if (
        not isinstance(fixture, dict)
        or fixture.get("seed_slot") != 1
        or float(fixture.get("nominal_translation_m", 0.0)) != 0.12
    ):
        raise ReferenceBindingRegistryError(
            "campaign reference_binding allocation differs"
        )
    blocks = _load_c2_blocks(queue_path)
    scene_sha256 = sha256_file(SCENE)
    resets: dict[str, dict[str, Any]] = {}
    for block_id in range(128):
        row = blocks[block_id]
        env_seed = int(row["env_seed"])
        color = row["physical_A_color"]
        start_side = row["physical_A_start_side"]
        signs = row["physical_A_diagonal_signs"]
        if color not in {"blue", "yellow"}:
            raise ReferenceBindingRegistryError(
                f"block {block_id}: physical A color differs"
            )
        if start_side not in {"left", "right"}:
            raise ReferenceBindingRegistryError(
                f"block {block_id}: physical A start side differs"
            )
        if (
            not isinstance(signs, list)
            or len(signs) != 2
            or any(value not in {-1, 1} for value in signs)
        ):
            raise ReferenceBindingRegistryError(
                f"block {block_id}: physical A diagonal signs differ"
            )
        dx = _axis_jitter(env_seed, "x")
        dy = _axis_jitter(env_seed, "y")
        a_slot = BASE_POSITIONS[f"{start_side}_bowl"]
        b_side = "right" if start_side == "left" else "left"
        b_slot = BASE_POSITIONS[f"{b_side}_bowl"]
        positions = {
            "cube": _translated(BASE_POSITIONS["cube"], dx, dy),
            f"{color}_bowl": _translated(a_slot, dx, dy),
            f"{'yellow' if color == 'blue' else 'blue'}_bowl": _translated(
                b_slot,
                dx,
                dy,
            ),
        }
        resets[str(env_seed)] = {
            "block_index": block_id,
            "jitter_robot_base_xy_m": [dx, dy],
            "positions_robot_base_m": positions,
            "physical_A_color": color,
            "physical_A_scene_object": f"{color}_bowl",
            "physical_B_color": "yellow" if color == "blue" else "blue",
            "physical_B_scene_object": (
                "yellow_bowl" if color == "blue" else "blue_bowl"
            ),
            "physical_A_start_side": start_side,
            "physical_A_diagonal_signs": list(signs),
            "state_index": row["state_index"],
        }
    object_specs = {
        "cube": {
            "role": "target",
            "primitive": "box",
            "dimensions_m": [0.04, 0.04, 0.04],
            "mass_kg": 0.04,
            "base_position_robot_m": list(BASE_POSITIONS["cube"]),
        },
        "blue_bowl": {
            "role": "candidate_reference",
            "color": "blue",
            "primitive": "open_box_visual_with_colliding_base",
            "outer_dimensions_m": [0.10, 0.10, 0.035],
            "mass_kg": 0.20,
        },
        "yellow_bowl": {
            "role": "candidate_reference",
            "color": "yellow",
            "primitive": "open_box_visual_with_colliding_base",
            "outer_dimensions_m": [0.10, 0.10, 0.035],
            "mass_kg": 0.20,
        },
    }
    return {
        "schema_version": SCHEMA,
        "campaign_id": "online_correction_v4",
        "fixture_id": "reference_binding",
        "status": STATUS,
        "qualification_scope": "confirmatory",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "scene_asset": portable_path(SCENE),
        "scene_metadata_sha256": scene_sha256,
        "contact_objects": ["cube", "blue_bowl", "yellow_bowl", "table"],
        "movable_objects": ["cube", "blue_bowl", "yellow_bowl"],
        "object_roles": {
            "target": {
                "scene_object": "cube",
                "asset_identity": (
                    f"{portable_path(SCENE)}::cube@{scene_sha256}"
                ),
            },
            "reference": {
                "scene_object": "blue_bowl",
                "asset_identity": (
                    f"{portable_path(SCENE)}::blue_bowl@{scene_sha256}"
                ),
            },
            "distractor": {
                "scene_object": "yellow_bowl",
                "asset_identity": (
                    f"{portable_path(SCENE)}::yellow_bowl@{scene_sha256}"
                ),
            },
        },
        "object_specs": object_specs,
        "registered_env_seed_count": 128,
        "registered_env_seed_min": SEED_MIN,
        "registered_env_seed_max": SEED_MAX,
        "resets_by_env_seed": resets,
        "reset_jitter": {
            "algorithm": "sha256_u64_independent_axes_common_scene_translation_v1",
            "x_half_range_m": JITTER_HALF_RANGE_M,
            "y_half_range_m": JITTER_HALF_RANGE_M,
            "z_jitter_m": 0.0,
            "application": (
                "one common robot-base x/y translation is applied to the cube and "
                "both bowls, preserving the registered two-reference layout"
            ),
            "policy_outcome_used": False,
        },
        "counterbalance_contract": {
            "physical_A_identity": "bound by physical_A_color per reset",
            "physical_B_identity": "the other colored bowl",
            "physical_A_motion": "registered diagonal signs at the selected scale",
            "named_reference_resolution": (
                "A resolves to physical_A_scene_object and B resolves to "
                "physical_B_scene_object without changing the episode prompt"
            ),
        },
        "scene_receipt": {
            "status": "model_blind_procedural_asset_candidate",
            "source": portable_path(SCENE),
            "sha256": scene_sha256,
            "object_specs": object_specs,
            "support_surface": {
                "construction": "qualified_dynamic_table_payload",
                "initial_clearance_m": 0.003,
            },
            "policy_outcome_used": False,
        },
        "source_identity": {
            "campaign_path": portable_path(campaign_path),
            "campaign_sha256": sha256_file(campaign_path),
            "queue_path": portable_path(queue_path),
            "queue_sha256": sha256_file(queue_path),
        },
        "release_boundary": (
            "Prospective model-blind C2 two-reference reset candidate only. "
            "G2/G3 qualification and verified same-prompt prefix replay remain "
            "required before policy inference."
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
                "scene_sha256": payload["scene_metadata_sha256"],
                "registered_env_seed_count": payload[
                    "registered_env_seed_count"
                ],
                "status": payload["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
