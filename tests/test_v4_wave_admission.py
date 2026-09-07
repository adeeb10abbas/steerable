#!/usr/bin/env python3
"""Tests for serialized V4 wave admission."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import v4_wave_admission as admission  # noqa: E402


def test_superseded_attempts_are_recognized() -> None:
    assert admission.is_superseded_attempt("g3rb20260908r2")
    assert admission.is_superseded_attempt("g2r20260908a10080w")
    assert not admission.is_superseded_attempt("g3r20260908g")


def test_reference_binding_live_control_matches_tier_zero() -> None:
    tier = admission.tier_for_attempt("g3ngrb20260908p")
    assert tier is not None
    assert tier.tier_id == "natural_grasp_live_control"
    assert admission.tier_for_attempt("g3ngrb20260908neg").tier_id == "natural_grasp_live_control"


def test_c7_attempts_never_blocked_by_tier() -> None:
    assert admission.is_c7_attempt("attempt0351")
    summary = {
        "g3r20260908g": {"total": 128, "active": 10, "succeeded": 0, "failed": 0, "pending": 118, "suspended": 0},
        "attempt0351": {"total": 2, "active": 2, "succeeded": 0, "failed": 0, "pending": 0, "suspended": 0},
        "g2r20260908g": {"total": 128, "active": 128, "succeeded": 0, "failed": 0, "pending": 0, "suspended": 0},
        "g3rb20260908v": {"total": 128, "active": 128, "succeeded": 0, "failed": 0, "pending": 0, "suspended": 0},
    }
    admitted, report = admission.compute_admitted_attempts(summary)
    assert admitted == ["g3r20260908g"]
    assert "attempt0351" in report["c7_always_admitted"]


def test_only_first_incomplete_tier_admitted() -> None:
    summary = {
        "g3r20260908g": {"total": 128, "active": 0, "succeeded": 128, "failed": 0, "pending": 0, "suspended": 0},
        "g2r20260908g": {"total": 128, "active": 0, "succeeded": 0, "failed": 0, "pending": 128, "suspended": 128},
        "g3rb20260908v": {"total": 128, "active": 0, "succeeded": 0, "failed": 0, "pending": 128, "suspended": 128},
    }
    admitted, _ = admission.compute_admitted_attempts(summary)
    assert admitted == ["g2r20260908g"]


def test_dispatch_gate_blocks_non_admitted_attempt() -> None:
    summary = {
        "g3r20260908g": {"total": 128, "active": 50, "succeeded": 0, "failed": 0, "pending": 78, "suspended": 0},
        "g2r20260908g": {"total": 128, "active": 0, "succeeded": 0, "failed": 0, "pending": 128, "suspended": 0},
    }
    with pytest.raises(admission.WaveAdmissionError, match="wave admission gate blocked"):
        admission.require_dispatch_admission(attempt_id="g2r20260908g", attempt_summary=summary)


def test_mixed_pin_c5_c6_without_g_suffix_superseded() -> None:
    assert admission.is_superseded_attempt("g3c5p20260908a10080")
    assert admission.is_superseded_attempt("g3c6p20260908a10080c")
    assert not admission.is_superseded_attempt("g3c5p20260908a10080g")


def test_wall_clock_estimates_are_monotonic() -> None:
    payload = admission.build_wall_clock_estimates()
    cumulative = [item["cumulative_minutes_from_now"] for item in payload["tier_estimates"]]
    assert cumulative == sorted(cumulative)
    assert payload["qualification_completion_hours"] > 0


def test_wall_clock_estimates_use_partial_wave_job_counts() -> None:
    summary = {
        "g2r20260908g": {"total": 128, "succeeded": 32, "active": 0, "failed": 0, "pending": 96, "suspended": 128},
        "g2gpu20260908g": {"total": 1, "succeeded": 0, "active": 0, "failed": 0, "pending": 1, "suspended": 1},
    }
    payload = admission.build_wall_clock_estimates(summary)
    g2 = next(t for t in payload["tier_estimates"] if t["tier_id"] == "horizontal_g2")
    assert g2["remaining_seeds"] == 97
