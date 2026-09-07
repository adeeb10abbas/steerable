"""Tests for fixture-specific natural-grasp live-control runners."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_runner(name: str):
    path = ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vertical_runner_constants() -> None:
    mod = _load_runner("run_v4_vertical_natural_grasp_live_positive_control.py")
    assert mod.FIXTURE_ID == "vertical"
    assert mod.CANONICAL_ENV_SEED == 2100020000
    assert mod.DEFAULT_GOAL == "above"


def test_containment_runner_constants() -> None:
    mod = _load_runner("run_v4_containment_natural_grasp_live_positive_control.py")
    assert mod.FIXTURE_ID == "containment"
    assert mod.CANONICAL_ENV_SEED == 2100030000
    assert mod.DEFAULT_GOAL == "inside"


def test_shared_scripted_controller_applies_robotiq_flange_offset_for_all_fixtures() -> None:
    from run_v4_horizontal_g3_scripted_seed import frozen_scripted_controller_config

    for fixture_id in (
        "horizontal",
        "reference_binding",
        "vertical",
        "containment",
        "object_pair",
    ):
        config = frozen_scripted_controller_config(fixture_id)
        assert config["eef_tool_length_m"] == 0.14, fixture_id
