#!/usr/bin/env python3
"""Tests for V4 GPU placement spread policies."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import v4_gpu_placement_enforce as placement  # noqa: E402
import v4_gpu_pool_sweep as sweep  # noqa: E402
import v4_gpu_scheduling as gpu_scheduling  # noqa: E402


def _job(lane_id: str, role: str, phase: str, *, has_spread: bool = False) -> placement.JobRef:
    return placement.JobRef(
        name=f"v4-{lane_id}-attempt0001-abc123-{role}",
        lane_id=lane_id,
        attempt_id="attempt0001",
        role=role,
        pod_phase=phase,
        has_spread=has_spread,
    )


def test_c8_spread_yaml_includes_topology_and_protect_affinity() -> None:
    rows, labels = gpu_scheduling.render_pod_placement_yaml(
        placement_policy=gpu_scheduling.PLACEMENT_POLICIES["c8_a40_spread"],
        role="policy",
        protected_c7_lanes=["c7m00", "c7m09"],
        indent="      ",
    )
    joined = "\n".join(rows)
    assert "topologySpreadConstraints" in joined
    assert "v4-gpu-spread-family: c8-a40" in joined or 'c8-a40' in joined
    assert "podAntiAffinity" in joined
    assert "c7m00" in joined
    assert labels["v4-gpu-spread-family"] == "c8-a40"


def test_c6_spread_applies_to_simulator_role_only() -> None:
    policy_rows, _ = gpu_scheduling.render_pod_placement_yaml(
        placement_policy=gpu_scheduling.PLACEMENT_POLICIES["c6_a10040_spread"],
        role="policy",
        indent="      ",
    )
    sim_rows, labels = gpu_scheduling.render_pod_placement_yaml(
        placement_policy=gpu_scheduling.PLACEMENT_POLICIES["c6_a10040_spread"],
        role="simulator",
        indent="      ",
    )
    assert policy_rows == []
    assert "topologySpreadConstraints" in "\n".join(sim_rows)
    assert labels["v4-gpu-spread-family"] == "c6-a10040-sim"


def test_protect_list_skips_reclaim_for_productive_c7_pair() -> None:
    protect_list = {
        "productive_c7_lane_pairs": [{"lane_id": "c7m00", "attempt_id": "attempt0473"}],
        "c7_retry_shards_in_flight_do_not_preempt": [],
    }
    rows = [
        {
            "lane_id": "c7m00",
            "attempt_id": "attempt0473",
            "job": "v4-c7m00-attempt0473-abc-policy",
            "gpu_product": "NVIDIA-A100-SXM4-80GB",
            "gpu_count": 1,
            "reason_code": "orphan_policy_dead_sim",
        }
    ]
    allowed, skipped = sweep.filter_protected_rows(rows, protect_list=protect_list)
    assert allowed == []
    assert len(skipped) == 1


def test_lane_pair_binding_is_planning_not_scheduler_atom() -> None:
    finding = gpu_scheduling.analyze_lane_pair_capacity_binding(family="c8")
    assert finding["policy_sim_co_location_required"] is False
    assert finding["scheduler_gpu_request_per_pod"] == 1
    assert finding["planning_gpus_per_lane_pair"] == 2


def test_placement_skips_missing_sim_when_policy_still_running() -> None:
    lanes = {
        "c8m13": [
            _job("c8m13", "policy", "Running"),
            _job("c8m13", "sim", "Missing"),
        ]
    }
    policy = gpu_scheduling.PLACEMENT_POLICIES["c8_a40_spread"]
    protect_list = {"productive_c7_lane_pairs": [], "c7_retry_shards_in_flight_do_not_preempt": []}
    sim = lanes["c8m13"][1]
    assert placement.job_has_running_partner(sim, lanes_by_id=lanes) is True
    assert (
        placement.job_needs_placement(
            sim,
            placement_policy=policy,
            protect_list=protect_list,
            lanes_by_id=lanes,
        )
        is False
    )
