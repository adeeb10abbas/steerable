"""Fail-closed fixture registry for V4 DROID simulator task registrations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from experiments.online_correction_v4.droid_task_files.constants import (
    BLOCKED_FIXTURE_IDS,
    CONTAINMENT_RELATIONS,
    CONTACT_OBJECT_LIST,
    EPISODE_LENGTH_S,
    EXTERNAL_SCORER_MODE,
    HORIZONTAL_RELATIONS,
    OBJECT_PAIR_CONTACT_OBJECTS,
    OBJECT_PAIR_REFERENCE_OBJECT,
    OBJECT_PAIR_SCENE_METADATA_SHA256,
    OBJECT_PAIR_SCENE_PATH,
    OBJECT_PAIR_TARGET_OBJECT,
    ROBOLAB_SUCCESS_TERMINATION_FORBIDDEN,
    SCENE_ASSET,
    SCENE_METADATA_SHA256,
    SUPPORTED_FIXTURE_IDS,
    TARGET_OBJECT,
    REFERENCE_OBJECT,
    VERTICAL_RELATIONS,
    fixture_object_spec,
)


class FixtureRegistryError(ValueError):
    """Raised when a fixture is unsupported or lacks required asset receipts."""


@dataclass(frozen=True)
class FixtureRegistration:
    fixture_id: str
    relation: str | None
    scene_asset: str
    scene_metadata_sha256: str
    target_object: str
    reference_object: str
    contact_object_list: tuple[str, ...]
    episode_length_s: int
    timeout_only: bool
    external_scorer_mode: str
    robolab_success_termination_forbidden: bool
    task_module: str
    task_class: str
    attributes: tuple[str, ...]


_TASK_ROOT = Path(__file__).resolve().parent / "task_files"

_ACTIVE_TASKS = {
    "horizontal": ("horizontal_active.py", "V4HorizontalActiveTask"),
    "object_pair": ("object_pair_active.py", "V4ObjectPairActiveTask"),
    "vertical": ("vertical_active.py", "V4VerticalActiveTask"),
    "containment": ("containment_active.py", "V4ContainmentActiveTask"),
}

_HORIZONTAL_TASKS: dict[str, tuple[str, str]] = {
    "left": ("horizontal_left.py", "V4HorizontalLeftTask"),
    "right": ("horizontal_right.py", "V4HorizontalRightTask"),
    "front": ("horizontal_front.py", "V4HorizontalFrontTask"),
    "behind": ("horizontal_behind.py", "V4HorizontalBehindTask"),
}


def _fail(message: str) -> None:
    raise FixtureRegistryError(message)


def supported_fixture_ids() -> tuple[str, ...]:
    return tuple(sorted(SUPPORTED_FIXTURE_IDS))


def blocked_fixture_ids() -> dict[str, str]:
    return dict(BLOCKED_FIXTURE_IDS)


def list_registered_horizontal_relations() -> tuple[str, ...]:
    return HORIZONTAL_RELATIONS


def resolve_active_registration(
    fixture_id: str = "horizontal",
    *,
    allow_model_blind_candidate: bool = False,
) -> FixtureRegistration:
    if fixture_id in {"object_pair", "vertical", "containment"}:
        if (
            fixture_id in BLOCKED_FIXTURE_IDS
            and not allow_model_blind_candidate
        ):
            _fail(
                f"fixture {fixture_id!r} is blocked until asset receipts exist: "
                f"{BLOCKED_FIXTURE_IDS[fixture_id]}"
            )
        spec = fixture_object_spec(fixture_id)
        module_name, class_name = _ACTIVE_TASKS[fixture_id]
        attributes = [
            "spatial",
            "online_correction_v4",
            fixture_id,
            "timeout_only",
            "external_v4_scorer",
            "active_episode_bound",
        ]
        if allow_model_blind_candidate:
            attributes.append("model_blind_candidate")
        return FixtureRegistration(
            fixture_id=fixture_id,
            relation=None,
            scene_asset=str(Path(__file__).resolve().parents[3] / spec.scene_asset),
            scene_metadata_sha256=spec.scene_metadata_sha256,
            target_object=spec.target_object,
            reference_object=spec.reference_object,
            contact_object_list=spec.contact_objects,
            episode_length_s=EPISODE_LENGTH_S,
            timeout_only=True,
            external_scorer_mode=EXTERNAL_SCORER_MODE,
            robolab_success_termination_forbidden=ROBOLAB_SUCCESS_TERMINATION_FORBIDDEN,
            task_module=str(_TASK_ROOT / module_name),
            task_class=class_name,
            attributes=tuple(attributes),
        )
    if fixture_id != "horizontal":
        if fixture_id in BLOCKED_FIXTURE_IDS:
            _fail(
                f"fixture {fixture_id!r} is blocked until asset receipts exist: "
                f"{BLOCKED_FIXTURE_IDS[fixture_id]}"
            )
        _fail(f"unsupported active fixture {fixture_id!r}")
    module_name, class_name = _ACTIVE_TASKS[fixture_id]
    return FixtureRegistration(
        fixture_id="horizontal",
        relation=None,
        scene_asset=SCENE_ASSET,
        scene_metadata_sha256=SCENE_METADATA_SHA256,
        target_object=TARGET_OBJECT,
        reference_object=REFERENCE_OBJECT,
        contact_object_list=CONTACT_OBJECT_LIST,
        episode_length_s=EPISODE_LENGTH_S,
        timeout_only=True,
        external_scorer_mode=EXTERNAL_SCORER_MODE,
        robolab_success_termination_forbidden=ROBOLAB_SUCCESS_TERMINATION_FORBIDDEN,
        task_module=str(_TASK_ROOT / module_name),
        task_class=class_name,
        attributes=(
            "spatial",
            "online_correction_v4",
            "horizontal",
            "timeout_only",
            "external_v4_scorer",
            "active_episode_bound",
        ),
    )


def resolve_fixture_registration(
    fixture_id: str,
    *,
    relation: str | None = None,
) -> FixtureRegistration:
    if fixture_id in BLOCKED_FIXTURE_IDS:
        _fail(
            f"fixture {fixture_id!r} is blocked until asset receipts exist: "
            f"{BLOCKED_FIXTURE_IDS[fixture_id]}"
        )
    if fixture_id not in SUPPORTED_FIXTURE_IDS:
        _fail(f"unsupported fixture {fixture_id!r}")

    if fixture_id == "horizontal":
        if relation is None:
            _fail("horizontal fixture registration requires an explicit relation label")
        if relation not in HORIZONTAL_RELATIONS:
            _fail(
                f"unsupported horizontal relation {relation!r}; "
                f"expected one of {list(HORIZONTAL_RELATIONS)}"
            )
        module_name, class_name = _HORIZONTAL_TASKS[relation]
        return FixtureRegistration(
            fixture_id="horizontal",
            relation=relation,
            scene_asset=SCENE_ASSET,
            scene_metadata_sha256=SCENE_METADATA_SHA256,
            target_object=TARGET_OBJECT,
            reference_object=REFERENCE_OBJECT,
            contact_object_list=CONTACT_OBJECT_LIST,
            episode_length_s=EPISODE_LENGTH_S,
            timeout_only=True,
            external_scorer_mode=EXTERNAL_SCORER_MODE,
            robolab_success_termination_forbidden=ROBOLAB_SUCCESS_TERMINATION_FORBIDDEN,
            task_module=str(_TASK_ROOT / module_name),
            task_class=class_name,
            attributes=(
                "spatial",
                "online_correction_v4",
                "horizontal",
                relation,
                "timeout_only",
                "external_v4_scorer",
            ),
        )

    if fixture_id == "object_pair":
        if relation is None:
            _fail("object_pair fixture registration requires an explicit relation label")
        if relation not in HORIZONTAL_RELATIONS:
            _fail(
                f"unsupported object_pair relation {relation!r}; "
                f"expected one of {list(HORIZONTAL_RELATIONS)}"
            )
        module_name, class_name = _ACTIVE_TASKS[fixture_id]
        return FixtureRegistration(
            fixture_id=fixture_id,
            relation=relation,
            scene_asset=OBJECT_PAIR_SCENE_PATH,
            scene_metadata_sha256=OBJECT_PAIR_SCENE_METADATA_SHA256,
            target_object=OBJECT_PAIR_TARGET_OBJECT,
            reference_object=OBJECT_PAIR_REFERENCE_OBJECT,
            contact_object_list=OBJECT_PAIR_CONTACT_OBJECTS,
            episode_length_s=EPISODE_LENGTH_S,
            timeout_only=True,
            external_scorer_mode=EXTERNAL_SCORER_MODE,
            robolab_success_termination_forbidden=ROBOLAB_SUCCESS_TERMINATION_FORBIDDEN,
            task_module=str(_TASK_ROOT / module_name),
            task_class=class_name,
            attributes=(
                "spatial",
                "online_correction_v4",
                "object_pair",
                relation,
                "timeout_only",
                "external_v4_scorer",
            ),
        )

    if fixture_id in {"vertical", "containment"}:
        allowed = VERTICAL_RELATIONS if fixture_id == "vertical" else CONTAINMENT_RELATIONS
        if relation is None or relation not in allowed:
            _fail(
                f"{fixture_id} fixture registration requires one of "
                f"{list(allowed)}, got {relation!r}"
            )
        spec = fixture_object_spec(fixture_id)
        module_name, class_name = _ACTIVE_TASKS[fixture_id]
        return FixtureRegistration(
            fixture_id=fixture_id,
            relation=relation,
            scene_asset=str(Path(__file__).resolve().parents[3] / spec.scene_asset),
            scene_metadata_sha256=spec.scene_metadata_sha256,
            target_object=spec.target_object,
            reference_object=spec.reference_object,
            contact_object_list=spec.contact_objects,
            episode_length_s=EPISODE_LENGTH_S,
            timeout_only=True,
            external_scorer_mode=EXTERNAL_SCORER_MODE,
            robolab_success_termination_forbidden=ROBOLAB_SUCCESS_TERMINATION_FORBIDDEN,
            task_module=str(_TASK_ROOT / module_name),
            task_class=class_name,
            attributes=(
                "spatial",
                "online_correction_v4",
                fixture_id,
                relation,
                "timeout_only",
                "external_v4_scorer",
            ),
        )

    _fail(f"fixture {fixture_id!r} has no registration handler")
    raise AssertionError("unreachable")


def iter_horizontal_registrations() -> Iterable[FixtureRegistration]:
    for relation in HORIZONTAL_RELATIONS:
        yield resolve_fixture_registration("horizontal", relation=relation)
