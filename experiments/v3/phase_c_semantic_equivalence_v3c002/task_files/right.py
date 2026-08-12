"""Exact E004 s=1 RIGHT task with a V3-C002 registered prompt byte string."""

from dataclasses import dataclass
import json
import os

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004 import fixture_tasks as fixture
from robolab.core.task.task import Task


def _prompt() -> str:
    value = os.environ["VLA_WAM_V3C002_PROMPT"]
    expected_hex = os.environ["VLA_WAM_V3C002_PROMPT_UTF8_HEX"]
    if value.encode("utf-8").hex() != expected_hex:
        raise RuntimeError("V3-C002 task prompt bytes differ from the registered queue cell")
    return value


def _contact_objects() -> list[str]:
    level = fixture._level_from_environment()
    logical = fixture._CANDIDATE.layout(level)
    mapping = json.loads(os.environ["VLA_WAM_V3E004_SCENE_OBJECT_MAPPING"])
    return [str(mapping[name]) for name in logical] + ["table"]


@dataclass
class V3C002DroidRightTask(Task):
    contact_object_list = _contact_objects()
    scene = fixture._scene()
    terminations = fixture._RightTermination
    instruction = {"default": _prompt()}
    attributes = ["spatial", "vla_wam_v3c002", "exact_e004_s1_object_layout"]
    episode_length_s: int = 30
    subtasks = fixture._subtask("right")
