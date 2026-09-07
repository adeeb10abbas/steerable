#!/usr/bin/env python3
"""Tests for V4 policy episode throughput analysis."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import analyze_v4_policy_episode_throughput as throughput  # noqa: E402


def test_throughput_analysis_reports_6400_achievable() -> None:
    payload = throughput.analyze_policy_throughput()
    assert payload["achievable_episode_total"] == 6400
    assert payload["blocked_episode_total"] == 11264
    assert payload["c7_object_pair"]["healthy_lane_pairs"] == 15
    assert payload["concurrency_limit_root_cause"]["primary"] == (
        "rendered_lane_pairs_and_frozen_hardware_stratum"
    )


def test_c6_parallel_stratum_expansion_owned_by_agent_b() -> None:
    payload = throughput.analyze_policy_throughput()
    c6 = payload["per_stratum_g4_expansion"]["c6_containment"]
    assert c6["would_legitimately_expand_throughput"] is True
    assert c6["owner"] == "Agent B (containment)"
