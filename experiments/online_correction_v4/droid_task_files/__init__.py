"""Simulator-side V4 DROID RoboLab fixture task registrations."""

from experiments.online_correction_v4.droid_task_files.registry import (
    FixtureRegistration,
    FixtureRegistryError,
    HORIZONTAL_RELATIONS,
    list_registered_horizontal_relations,
    resolve_fixture_registration,
    supported_fixture_ids,
)

__all__ = [
    "FixtureRegistration",
    "FixtureRegistryError",
    "HORIZONTAL_RELATIONS",
    "list_registered_horizontal_relations",
    "resolve_fixture_registration",
    "supported_fixture_ids",
]
