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
    assert detection["orphan_policies"][0]["reason_code"] == "split_pair_orphan_policy_half"
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


def test_split_pair_orphan_detected_when_partner_failed() -> None:
    pods = [
        _pod("c8m00", "attempt0001", "policy"),
        _pod("c8m00", "attempt0001", "sim", phase="Failed"),
    ]
    detection = sweep.detect_lane_mismatches(pods)
    assert len(detection["split_pair_orphans"]) == 1
    assert detection["split_pair_orphans"][0]["reason_code"] == "split_pair_orphan_policy_half"


def test_split_pair_not_detected_when_sim_pending_between_episodes() -> None:
    pods = [
        _pod("c8m13", "attempt0014", "policy"),
        _pod("c8m13", "attempt0014", "sim", phase="Pending"),
    ]
    detection = sweep.detect_lane_mismatches(pods)
    assert detection["split_pair_orphans"] == []
    assert detection["orphan_policies"] == []


def test_startup_grace_pending_sim_not_orphan() -> None:
    pods = [
        _pod("c8m00", "attempt0001", "policy"),
        sweep.PodRef(
            name="v4-c8m00-attempt0001-abc123-sim-xyz",
            lane_id="c8m00",
            attempt_id="attempt0001",
            role="sim",
            phase="Pending",
            gpu_count=1,
            gpu_product="NVIDIA-A40",
            node_name="node-1",
            job_name="v4-c8m00-attempt0001-abc123-sim",
            age_seconds=120.0,
            container_ready=False,
        ),
    ]
    detection = sweep.detect_lane_mismatches(pods)
    assert detection["orphan_policies"] == []
    assert detection["split_pair_orphans"] == []


def test_finished_c7_running_policy_succeeded_sim_flagged_healthy_by_detector() -> None:
    """Standard sweep marks terminal C7 pairs healthy; finished-family reclaim deletes them."""
    pods = [
        _pod("c7m03", "attempt0618", "policy", gpu_product="NVIDIA-A100-SXM4-80GB"),
        _pod("c7m03", "attempt0618", "sim", phase="Succeeded", gpu_product="NVIDIA-A40"),
    ]
    detection = sweep.detect_lane_mismatches(pods)
    assert "c7m03" in detection["healthy_lane_pairs"]


def test_finished_c8_lane_ids_cover_all_confirmatory_lanes() -> None:
    import v4_gpu_scheduling as gpu_scheduling  # noqa: E402

    assert len(gpu_scheduling.FINISHED_C8_LANE_IDS) == 20
    assert gpu_scheduling.PRODUCTIVE_C8_LANE_IDS == frozenset()


def test_startup_grace_expired_sim_failed_is_orphan() -> None:
    pods = [
        _pod("c8m00", "attempt0001", "policy"),
        sweep.PodRef(
            name="v4-c8m00-attempt0001-abc123-sim-xyz",
            lane_id="c8m00",
            attempt_id="attempt0001",
            role="sim",
            phase="Failed",
            gpu_count=0,
            gpu_product="NVIDIA-A40",
            node_name="node-1",
            job_name="v4-c8m00-attempt0001-abc123-sim",
            age_seconds=400.0,
            container_ready=False,
        ),
    ]
    detection = sweep.detect_lane_mismatches(pods)
    assert len(detection["split_pair_orphans"]) == 1
