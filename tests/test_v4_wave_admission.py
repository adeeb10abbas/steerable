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


def test_horizontal_g3_is_scientifically_blocked() -> None:
    assert admission.is_scientifically_blocked_attempt("g3r20260908g")
    assert not admission.is_scientifically_blocked_attempt("g3rb20260908v")


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
    assert "g3r20260908g" not in admitted
    assert "g3rb20260908v" in admitted
    assert "g2r20260908g" in admitted
    assert "attempt0351" in report["c7_always_admitted"]
    assert "g3r20260908g" in report["scientifically_blocked"]


def test_promoted_c2_and_c6_admit_in_parallel_with_g2_backfill() -> None:
    summary = {
        "g3rb20260908v": {"total": 128, "active": 50, "succeeded": 13, "failed": 0, "pending": 65, "suspended": 0},
        "g3c6p20260908a10040g": {"total": 1, "active": 0, "succeeded": 0, "failed": 0, "pending": 1, "suspended": 1},
        "g2r20260908g": {"total": 128, "active": 0, "succeeded": 103, "failed": 0, "pending": 25, "suspended": 25},
        "g3c5p20260908a10080g": {"total": 1, "active": 0, "succeeded": 0, "failed": 0, "pending": 1, "suspended": 1},
        "g3c6p20260908a10080g": {"total": 1, "active": 0, "succeeded": 0, "failed": 0, "pending": 1, "suspended": 1},
    }
    admitted, report = admission.compute_admitted_attempts(summary)
    assert set(admitted) == {
        "g3rb20260908v",
        "g3c6p20260908a10040g",
        "g2r20260908g",
    }
    assert "g3c5p20260908a10080g" not in admitted
    assert "g3c6p20260908a10080g" not in admitted
    assert "g3c6p20260908a10080g" in report["wrong_stratum_parallel_suspended"]


def test_wrong_stratum_c6_a10080g_never_admitted() -> None:
    assert admission.is_wrong_stratum_parallel_attempt("g3c6p20260908a10080g")
    summary = {
        "g3rb20260908v": {"total": 128, "active": 0, "succeeded": 128, "failed": 0, "pending": 0, "suspended": 0},
        "g3c6p20260908a10080g": {"total": 1, "active": 1, "succeeded": 0, "failed": 0, "pending": 0, "suspended": 0},
    }
    admitted, _ = admission.compute_admitted_attempts(summary)
    assert "g3c6p20260908a10080g" not in admitted


def test_c5_deprioritized_only_after_achievable_tiers_complete() -> None:
    summary = {
        "g3rb20260908v": {"total": 128, "active": 0, "succeeded": 128, "failed": 0, "pending": 0, "suspended": 0},
        "g3c6p20260908a10080g": {"total": 128, "active": 0, "succeeded": 128, "failed": 0, "pending": 0, "suspended": 0},
        "g2r20260908g": {"total": 128, "active": 0, "succeeded": 128, "failed": 0, "pending": 0, "suspended": 0},
        "g3c5p20260908a10080g": {"total": 1, "active": 0, "succeeded": 0, "failed": 0, "pending": 1, "suspended": 1},
    }
    admitted, _ = admission.compute_admitted_attempts(summary)
    assert admitted == ["g3c5p20260908a10080g"]


def test_dispatch_gate_blocks_scientifically_blocked_horizontal_g3() -> None:
    with pytest.raises(admission.WaveAdmissionError, match="scientifically blocked"):
        admission.require_dispatch_admission(attempt_id="g3r20260908g", attempt_summary={})


def test_dispatch_gate_blocks_non_admitted_attempt() -> None:
    summary = {
        "g3rb20260908v": {"total": 128, "active": 50, "succeeded": 0, "failed": 0, "pending": 78, "suspended": 0},
        "g3c5p20260908a10080g": {"total": 1, "active": 0, "succeeded": 0, "failed": 0, "pending": 1, "suspended": 1},
    }
    with pytest.raises(admission.WaveAdmissionError, match="wave admission gate blocked"):
        admission.require_dispatch_admission(attempt_id="g3c5p20260908a10080g", attempt_summary=summary)


def test_parallel_stratum_new_dispatch_allowed_while_primary_runs() -> None:
    summary = {
        "g3rb20260908v": {"total": 128, "active": 50, "succeeded": 0, "failed": 0, "pending": 78, "suspended": 0},
    }
    admission.require_dispatch_admission(
        attempt_id="g3c6p20260908a10040i",
        attempt_summary=summary,
    )


def test_mixed_pin_c5_c6_without_g_suffix_superseded() -> None:
    assert admission.is_superseded_attempt("g3c5p20260908a10080")
    assert admission.is_superseded_attempt("g3c6p20260908a10080c")
    assert not admission.is_superseded_attempt("g3c5p20260908a10080g")


def test_wall_clock_estimates_are_monotonic_for_primary_tiers() -> None:
    payload = admission.build_wall_clock_estimates()
    primary = [
        item["cumulative_minutes_from_now"]
        for item in payload["tier_estimates"]
        if item.get("admission_class") == "primary"
    ]
    assert primary == sorted(primary)
    assert payload["qualification_completion_hours"] > 0


def test_wall_clock_estimates_use_partial_wave_job_counts() -> None:
    summary = {
        "g2r20260908g": {"total": 128, "succeeded": 32, "active": 0, "failed": 0, "pending": 96, "suspended": 128},
        "g2gpu20260908g": {"total": 1, "succeeded": 0, "active": 0, "failed": 0, "pending": 1, "suspended": 1},
    }
    payload = admission.build_wall_clock_estimates(summary)
    g2 = next(t for t in payload["tier_estimates"] if t["tier_id"] == "horizontal_g2_evidence")
    assert g2["remaining_seeds"] == 97


def test_achievable_episode_estimates_cover_6400_episodes() -> None:
    payload = admission.build_achievable_episode_estimates()
    assert payload["achievable_episode_total"] == 6400
    assert payload["blocked_episode_total"] == 11264
    assert payload["parallel_strata_materially_faster"] is True
    assert payload["qualification_wall_clock_hours"]["parallel_a10040_c6"] < (
        payload["qualification_wall_clock_hours"]["serialized"]
    )
