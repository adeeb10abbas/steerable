#!/usr/bin/env python3
"""Tests for isolated V4 cluster study checkouts."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import v4_study_checkout_isolation as isolation  # noqa: E402


def test_isolated_study_root_is_unique_per_attempt() -> None:
    a = isolation.isolated_study_root(
        workstream_id="g2_repair_v2",
        pin_commit="96ccf4ad95b292e61da28eafc2ad9a7b6dfa5e65",
        attempt_id="g2r20260908a10080r",
    )
    b = isolation.isolated_study_root(
        workstream_id="g2_repair_v2",
        pin_commit="96ccf4ad95b292e61da28eafc2ad9a7b6dfa5e65",
        attempt_id="g2gpu20260908a40c",
    )
    assert a != b
    assert "steerable-v4-g2_repair_v2-96ccf4ad-" in a
    assert a.startswith(isolation.CLUSTER_SRC_PARENT)


def test_assert_checkout_mutable_blocks_legacy_with_live_jobs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"schema_version":"v4-cluster-study-checkout-manifest-v2","isolated_checkouts":[]}',
        encoding="utf-8",
    )
    legacy = next(iter(isolation.LEGACY_SHARED_CHECKOUTS))
    with mock.patch.object(isolation, "list_live_attempt_ids", return_value={"g2r20260908a10080r"}):
        with pytest.raises(isolation.CheckoutIsolationError, match="refusing to mutate legacy"):
            isolation.assert_checkout_mutable(
                study_root=legacy,
                kube_context="ctx",
                namespace="ns",
                manifest_path=manifest_path,
            )


def test_rewrite_spec_study_root_paths_rewrites_cluster_paths() -> None:
    legacy = isolation.WORKSTREAM_TEMPLATE_ROOTS["c2_g3"]
    isolated = isolation.isolated_study_root(
        workstream_id="c2_g3",
        pin_commit="c401fb4577d8003a019ecf7ff7be549f2c0a5931",
        attempt_id="g3rb20260908v",
    )
    spec = {
        "study_root": legacy,
        "runner_path": f"{legacy}/tools/run_v4_horizontal_g3_path_seed.py",
        "runtime": {"pythonpath": f"{legacy}:/other"},
    }
    patched = isolation.rewrite_spec_study_root_paths(
        spec,
        legacy_study_root=legacy,
        isolated_study_root=isolated,
    )
    assert patched["study_root"] == isolated
    assert patched["runner_path"] == f"{isolated}/tools/run_v4_horizontal_g3_path_seed.py"
    assert patched["runtime"]["pythonpath"] == f"{isolated}:/other"


def test_resolve_workstream_id_from_legacy_and_isolated_paths() -> None:
    legacy = isolation.WORKSTREAM_TEMPLATE_ROOTS["g2_repair_v2"]
    isolated = isolation.isolated_study_root(
        workstream_id="g2_repair_v2",
        pin_commit="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        attempt_id="attempt",
    )
    assert isolation.resolve_workstream_id(legacy) == "g2_repair_v2"
    assert isolation.resolve_workstream_id(isolated) == "g2_repair_v2"
