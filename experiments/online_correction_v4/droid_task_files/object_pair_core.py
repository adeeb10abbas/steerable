"""RoboLab-free helpers for the V4 C7 object-pair task."""

from __future__ import annotations

from experiments.online_correction_v4.droid_task_files.constants import EXTERNAL_SCORER_MODE


def task_attributes(relation: str) -> list[str]:
    return [
        "spatial",
        "online_correction_v4",
        "object_pair",
        relation,
        "timeout_only",
        EXTERNAL_SCORER_MODE,
        "external_v4_scorer",
        "no_robolab_success_termination",
        "no_robolab_subtasks",
    ]


def termination_field_names() -> frozenset[str]:
    return frozenset({"time_out"})
