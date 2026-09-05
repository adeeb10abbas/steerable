"""Frozen constants for V4 DROID simulator task registrations."""

from __future__ import annotations

STUDY_ID = "online_correction_v4"

ENV_QUEUE_ROW = "ONLINE_CORRECTION_V4_QUEUE_ROW"
ENV_QUEUE_ROW_SHA256 = "ONLINE_CORRECTION_V4_QUEUE_ROW_SHA256"
ENV_RESET_REGISTRY = "ONLINE_CORRECTION_V4_RESET_REGISTRY"
ENV_RESET_REGISTRY_SHA256 = "ONLINE_CORRECTION_V4_RESET_REGISTRY_SHA256"
ENV_ACTIVE_GOAL = "ONLINE_CORRECTION_V4_ACTIVE_GOAL"

RESET_REGISTRY_SCHEMA = "v4-droid-horizontal-reset-registry-v1"
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

# 60 s active policy cap plus 1.0 s passive adjudication outside the cap.
EPISODE_LENGTH_S = 61
ACTIVE_POLICY_CAP_S = 60.0
PASSIVE_ADJUDICATION_S = 1.0

EXTERNAL_SCORER_MODE = "external_v4_first_placement"
ROBOLAB_SUCCESS_TERMINATION_FORBIDDEN = True

SUPPORTED_FIXTURE_IDS = frozenset({"horizontal"})

BLOCKED_FIXTURE_IDS: dict[str, str] = {
    "reference_binding": "named_reference_asset_receipt_pending",
    "vertical": "vertical_fixture_asset_receipt_pending",
    "containment": "containment_fixture_asset_receipt_pending",
    "object_pair": "object_pair_fixture_asset_receipt_pending",
    "second_stack": "second_stack_bridge_fixture_asset_receipt_pending",
}
