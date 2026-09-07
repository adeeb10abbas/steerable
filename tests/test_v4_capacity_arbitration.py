#!/usr/bin/env python3
"""Tests for V4 GPU capacity arbitration."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import v4_capacity_arbitration as arbitration  # noqa: E402


def _lane_job(lane_id: str, attempt_id: str, role: str, *, active: int = 1, failed: int = 0) -> arbitration.LaneJobRef:
    return arbitration.LaneJobRef(
        name=f"v4-{lane_id}-{attempt_id}-abc123-{role}",
        lane_id=lane_id,
        attempt_id=attempt_id,
        role=role,
        suspend=False,
        active=active,
        succeeded=0,
        failed=failed,
        labels={},
    )


def test_canonical_c6_pilot_attempt_mapping() -> None:
    assert arbitration.canonical_c6_pilot_attempt("g7c6p00") == "attempt0057"


def test_stale_pilot_jobs_are_identified() -> None:
    deleted = arbitration.reclaim_stale_pilot_lane_jobs(
        [_lane_job("g7c6p00", "attempt0041", "policy"), _lane_job("g7c6p00", "attempt0057", "policy")],
        kube_context="ctx",
        namespace="ns",
        dry_run=True,
    )
    assert deleted[0]["attempt_id"] == "attempt0041"


def test_c7_throttle_keeps_lowest_index_lanes() -> None:
    jobs = []
    for index in range(30):
        lane = f"c7m{index:02d}"
        jobs.extend([_lane_job(lane, f"attempt{500+index:04d}", "policy"), _lane_job(lane, f"attempt{500+index:04d}", "sim")])
    throttle_jobs, report = arbitration.select_c7_throttle_jobs(jobs, ceiling=24)
    assert "c7m29" in {job.lane_id for job in throttle_jobs}
    assert len(report["lanes_kept"]) == 24


def test_pilot_priority_detects_c8_failure() -> None:
    assert arbitration.pilots_need_priority([_lane_job("g7c8p00", "attempt0020", "policy", active=0, failed=1)]) is True
