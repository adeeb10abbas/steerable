"""Frozen constants for V4 DROID simulator task registrations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

STUDY_ID = "online_correction_v4"

ENV_QUEUE_ROW = "ONLINE_CORRECTION_V4_QUEUE_ROW"
ENV_QUEUE_ROW_SHA256 = "ONLINE_CORRECTION_V4_QUEUE_ROW_SHA256"
ENV_RESET_REGISTRY = "ONLINE_CORRECTION_V4_RESET_REGISTRY"
ENV_RESET_REGISTRY_SHA256 = "ONLINE_CORRECTION_V4_RESET_REGISTRY_SHA256"
ENV_ACTIVE_GOAL = "ONLINE_CORRECTION_V4_ACTIVE_GOAL"

RESET_REGISTRY_SCHEMA = "v4-droid-horizontal-reset-registry-v1"
OBJECT_PAIR_RESET_REGISTRY_SCHEMA = "v4-droid-object-pair-reset-registry-v1"
VERTICAL_RESET_REGISTRY_SCHEMA = "v4-droid-vertical-reset-registry-v1"
CONTAINMENT_RESET_REGISTRY_SCHEMA = "v4-droid-containment-reset-registry-v1"
QUEUE_ROW_REQUIRED_KEYS = (
    "episode_id",
    "fixture",
    "prompt_text",
    "prompt_sha256",
    "env_seed",
    "factors",
)

SCENE_ASSET = "rubiks_cube_banana_bowl.usda"
SCENE_METADATA_SHA256 = "83ecf76a1fde9091b5db9012b76790aca36c2fe6b2c36a8885f4f98d7c4b7e1c"
CONTACT_OBJECT_LIST = ("rubiks_cube", "banana", "bowl", "table")
MOVABLE_OBJECTS = ("rubiks_cube", "bowl", "banana")

TARGET_OBJECT = "rubiks_cube"
REFERENCE_OBJECT = "bowl"
DISTRACTOR_OBJECT = "banana"

HORIZONTAL_RELATIONS = ("left", "right", "front", "behind")
OBJECT_PAIR_RELATIONS = HORIZONTAL_RELATIONS
VERTICAL_RELATIONS = ("above", "below")
CONTAINMENT_RELATIONS = ("inside",)


@dataclass(frozen=True)
class FixtureObjectSpec:
    fixture_id: str
    reset_registry_schema: str
    scene_asset: str
    scene_metadata_sha256: str
    contact_objects: tuple[str, ...]
    movable_objects: tuple[str, ...]
    target_object: str
    reference_object: str
    distractor_object: str | None = None


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OBJECT_PAIR_SCENE_ASSET = (
    "experiments/online_correction_v4/droid_task_files/scene_assets/"
    "sponge_tray_object_pair.usda"
)
OBJECT_PAIR_SCENE_PATH = str(_REPOSITORY_ROOT / OBJECT_PAIR_SCENE_ASSET)
OBJECT_PAIR_SCENE_METADATA_SHA256 = (
    "154780bea1505cc760b1ebac22bbec3b85cbede2c5b5e5cd495ec15a1e7d1cc5"
)
OBJECT_PAIR_CONTACT_OBJECTS = ("sponge", "tray", "table")
OBJECT_PAIR_MOVABLE_OBJECTS = ("sponge", "tray")
OBJECT_PAIR_TARGET_OBJECT = "sponge"
OBJECT_PAIR_REFERENCE_OBJECT = "tray"

VERTICAL_SCENE_ASSET = (
    "experiments/online_correction_v4/droid_task_files/scene_assets/"
    "cube_shelves_vertical.usda"
)
VERTICAL_SCENE_PATH = str(_REPOSITORY_ROOT / VERTICAL_SCENE_ASSET)
VERTICAL_SCENE_METADATA_SHA256 = (
    "0ff17eda0af15e99f01ea2c90e326f40b8926b2e05dc3850fa068e8ccd1ebe66"
)
VERTICAL_CONTACT_OBJECTS = (
    "cube",
    "bowl",
    "shelf_bottom",
    "shelf_middle",
    "shelf_top",
    "table",
)
VERTICAL_MOVABLE_OBJECTS = ("cube", "bowl")
VERTICAL_TARGET_OBJECT = "cube"
VERTICAL_REFERENCE_OBJECT = "bowl"

CONTAINMENT_SCENE_ASSET = (
    "experiments/online_correction_v4/droid_task_files/scene_assets/"
    "cube_container_containment.usda"
)
CONTAINMENT_SCENE_PATH = str(_REPOSITORY_ROOT / CONTAINMENT_SCENE_ASSET)
CONTAINMENT_SCENE_METADATA_SHA256 = (
    "c27d53b44708fbe0ff445ca31fa1a973923e1e94908c4d6f5e956a017ad6c273"
)
CONTAINMENT_CONTACT_OBJECTS = ("cube", "bowl", "table")
CONTAINMENT_MOVABLE_OBJECTS = ("cube", "bowl")
CONTAINMENT_TARGET_OBJECT = "cube"
CONTAINMENT_REFERENCE_OBJECT = "bowl"

FIXTURE_OBJECT_SPECS: dict[str, FixtureObjectSpec] = {
    "horizontal": FixtureObjectSpec(
        fixture_id="horizontal",
        reset_registry_schema=RESET_REGISTRY_SCHEMA,
        scene_asset=SCENE_ASSET,
        scene_metadata_sha256=SCENE_METADATA_SHA256,
        contact_objects=CONTACT_OBJECT_LIST,
        movable_objects=MOVABLE_OBJECTS,
        target_object=TARGET_OBJECT,
        reference_object=REFERENCE_OBJECT,
        distractor_object=DISTRACTOR_OBJECT,
    ),
    "object_pair": FixtureObjectSpec(
        fixture_id="object_pair",
        reset_registry_schema=OBJECT_PAIR_RESET_REGISTRY_SCHEMA,
        scene_asset=OBJECT_PAIR_SCENE_ASSET,
        scene_metadata_sha256=OBJECT_PAIR_SCENE_METADATA_SHA256,
        contact_objects=OBJECT_PAIR_CONTACT_OBJECTS,
        movable_objects=OBJECT_PAIR_MOVABLE_OBJECTS,
        target_object=OBJECT_PAIR_TARGET_OBJECT,
        reference_object=OBJECT_PAIR_REFERENCE_OBJECT,
    ),
    "vertical": FixtureObjectSpec(
        fixture_id="vertical",
        reset_registry_schema=VERTICAL_RESET_REGISTRY_SCHEMA,
        scene_asset=VERTICAL_SCENE_ASSET,
        scene_metadata_sha256=VERTICAL_SCENE_METADATA_SHA256,
        contact_objects=VERTICAL_CONTACT_OBJECTS,
        movable_objects=VERTICAL_MOVABLE_OBJECTS,
        target_object=VERTICAL_TARGET_OBJECT,
        reference_object=VERTICAL_REFERENCE_OBJECT,
    ),
    "containment": FixtureObjectSpec(
        fixture_id="containment",
        reset_registry_schema=CONTAINMENT_RESET_REGISTRY_SCHEMA,
        scene_asset=CONTAINMENT_SCENE_ASSET,
        scene_metadata_sha256=CONTAINMENT_SCENE_METADATA_SHA256,
        contact_objects=CONTAINMENT_CONTACT_OBJECTS,
        movable_objects=CONTAINMENT_MOVABLE_OBJECTS,
        target_object=CONTAINMENT_TARGET_OBJECT,
        reference_object=CONTAINMENT_REFERENCE_OBJECT,
    ),
}


def fixture_object_spec(fixture_id: str) -> FixtureObjectSpec:
    try:
        return FIXTURE_OBJECT_SPECS[fixture_id]
    except KeyError as exc:
        raise ValueError(f"unsupported DROID fixture object specification: {fixture_id}") from exc

# 60 s active policy cap plus 1.0 s passive adjudication outside the cap.
EPISODE_LENGTH_S = 61
ACTIVE_POLICY_CAP_S = 60.0
PASSIVE_ADJUDICATION_S = 1.0

EXTERNAL_SCORER_MODE = "external_v4_first_placement"
ROBOLAB_SUCCESS_TERMINATION_FORBIDDEN = True

SUPPORTED_FIXTURE_IDS = frozenset({"horizontal", "object_pair"})

BLOCKED_FIXTURE_IDS: dict[str, str] = {
    "reference_binding": "named_reference_asset_receipt_pending",
    "vertical": "vertical_fixture_asset_receipt_pending",
    "containment": "containment_fixture_asset_receipt_pending",
    "second_stack": "second_stack_bridge_fixture_asset_receipt_pending",
}
