#!/usr/bin/env python3
"""Tests for repeatable V4 GPU orphan/zombie pool sweep."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import v4_gpu_pool_sweep as sweep  # noqa: E402


def _pod(
    lane_id: str,
    attempt_id: str,
    role: str,
    *,
    phase: str = "Running",
    gpu_count: int = 1,
    gpu_product: str = "NVIDIA-A40",
) -> sweep.PodRef:
    return sweep.PodRef(
        name=f"v4-{lane_id}-{attempt_id}-abc123-{role}-xyz",
        lane_id=lane_id,
        attempt_id=attempt_id,
        role=role,
        phase=phase,
        gpu_count=gpu_count,
        gpu_product=gpu_product,
        node_name="node-1",
        job_name=f"v4-{lane_id}-{attempt_id}-abc123-{role}",
    )


def test_orphan_policy_dead_sim_detected() -> None:
    pods = [
        _pod("c7m03", "attempt0100", "policy", gpu_product="NVIDIA-A100-SXM4-80GB"),
        _pod("c7m03", "attempt0100", "sim", phase="Failed"),
    ]
    detection = sweep.detect_lane_mismatches(pods)
    assert len(detection["orphan_policies"]) == 1
    assert detection["orphan_policies"][0]["reason_code"] == "orphan_policy_dead_sim"
    assert detection["orphan_policies"][0]["gpu_product"] == "NVIDIA-A100-SXM4-80GB"


def test_between_episodes_healthy_not_reclaimed() -> None:
    pods = [
        _pod("c7m03", "attempt0100", "policy", gpu_product="NVIDIA-A100-SXM4-80GB"),
        _pod("c7m03", "attempt0100", "sim", phase="Pending"),
    ]
    detection = sweep.detect_lane_mismatches(pods)
    assert detection["orphan_policies"] == []
    assert detection["zombie_sims"] == []


def test_zombie_sim_idle_policy_dead_detected() -> None:
    pods = [
        _pod("c6m05", "attempt0170", "sim", gpu_product="NVIDIA-A100-SXM4-40GB"),
        _pod("c6m05", "attempt0170", "policy", phase="Pending", gpu_product="NVIDIA-B200"),
    ]
    detection = sweep.detect_lane_mismatches(pods)
    assert len(detection["zombie_sims"]) == 1
    assert detection["zombie_sims"][0]["reason_code"] == "zombie_sim_idle_policy_dead"


def test_a40_assessment_mixed_when_free_and_pending() -> None:
    pods = [
        _pod("g7c8p00", "attempt0001", "policy"),
        _pod("c7m01", "attempt0100", "sim", gpu_product="NVIDIA-A40"),
    ]
    orphan = [
        {
            "gpu_product": "NVIDIA-A40",
            "gpu_count": 1,
            "reason_code": "orphan_policy_dead_sim",
        }
    ]
    assessment = sweep.assess_a40_fragmentation(pods, orphan)
    assert assessment["diagnosis"] in {
        "mixed_orphans_and_per_node_fragmentation",
        "orphan_held_gpus_primary",
    }
    assert "placement_recommendation" in assessment


def test_healthy_lane_pair_not_flagged() -> None:
    pods = [
        _pod("c8m13", "attempt0010", "policy"),
        _pod("c8m13", "attempt0010", "sim"),
    ]
    detection = sweep.detect_lane_mismatches(pods)
    assert detection["healthy_lane_pairs"] == ["c8m13"]
    assert detection["orphan_policies"] == []
    assert detection["zombie_sims"] == []
