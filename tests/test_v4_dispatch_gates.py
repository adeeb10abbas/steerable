#!/usr/bin/env python3
"""Tests for V4 dispatch gate preconditions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import v4_dispatch_gates as gates  # noqa: E402


def test_smoke_key_format() -> None:
    assert gates.smoke_key(fixture_id="horizontal", gate="G2", gpu_product="NVIDIA-A40") == (
        "horizontal:G2:NVIDIA-A40"
    )


def test_require_passed_smoke_blocks_unknown_stratum(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"schema_version": "v4-gpu-smoke-registry-v1", "passed_smokes": []}),
        encoding="utf-8",
    )
    with pytest.raises(gates.DispatchGateError, match="smoke-before-wave gate blocked"):
        gates.require_passed_smoke(
            fixture_id="horizontal",
            gate="G2",
            gpu_product="NVIDIA-B200",
            registry_path=registry,
        )


def test_require_passed_smoke_allows_registered_stratum(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    gates.record_passed_smoke(
        fixture_id="horizontal",
        gate="G2",
        gpu_product="NVIDIA-A100-SXM4-80GB",
        attempt_id="g2r20260908a10080a",
        receipt_path="artifacts/example.json",
        receipt_sha256="abc123",
        registry_path=registry,
    )
    entry = gates.require_passed_smoke(
        fixture_id="horizontal",
        gate="G2",
        gpu_product="NVIDIA-A100-SXM4-80GB",
        registry_path=registry,
    )
    assert entry["attempt_id"] == "g2r20260908a10080a"


def test_collect_output_parents_from_fixture_bundle() -> None:
    bundle = ROOT / (
        "artifacts/online_correction_v4/execution/g2_horizontal_repair_v2_20260908/"
        "rendered-a10080-smoke/v4-g2-horizontal-g2r20260908a10080a-11feca0806"
    )
    if not bundle.is_dir():
        pytest.skip("fixture bundle not present")
    parents = gates.collect_output_parents(bundle)
    assert parents
    assert all(parent.startswith("/data/") for parent in parents)


def test_enforce_dispatch_gates_smoke_skips_registry(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "s000-job.yaml").write_text(
        'name: OUTPUT_PARENT\n  value: "/data/users/ali/vla_wam/out/test"\n',
        encoding="utf-8",
    )
    (bundle / "s000-configmap.yaml").write_text(
        "simulator-launch.json: |\n"
        '    {"experiment_argv":["--study-root","/data/study","--expected-study-commit","deadbeef"],'
        '"file_bindings":[{"path":"/data/study/tools/x.py","sha256":"aa","bytes":1}]}\n'
        "  image.digest: sha256:abc\n",
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"schema_version": "v4-gpu-smoke-registry-v1", "passed_smokes": []}),
        encoding="utf-8",
    )

    def fake_kubectl(*, kube_context, namespace, publisher_pod, script):  # noqa: ANN001
        if "git -C" in script:
            return '{"head":"deadbeef","dirty_count":"0"}\n'
        if "sha256sum" in script:
            return '{"bytes":"1","sha256":"aa"}\n'
        return "prepared=1\n"

    with mock.patch.object(gates, "_kubectl_exec", side_effect=fake_kubectl):
        report = gates.enforce_dispatch_gates(
            bundle_root=bundle,
            fixture_id="horizontal",
            gate="G2",
            gpu_product="NVIDIA-B200",
            kube_context="ctx",
            namespace="ns",
            publisher_pod="pod",
            expected_study_commit="deadbeef",
            is_smoke=True,
            registry_path=registry,
        )
    assert report["is_smoke"] is True
    assert report["output_parents_prepared"] == 1


def test_verify_file_bindings_skips_configmap_lane_scripts() -> None:
    calls: list[str] = []

    def fake_kubectl(*, kube_context, namespace, publisher_pod, script):  # noqa: ANN001
        calls.append(script)
        return '{"bytes":"1","sha256":"aa"}\n'

    with mock.patch.object(gates, "_kubectl_exec", side_effect=fake_kubectl):
        checked = gates.verify_file_bindings_on_cluster(
            [
                {"path": "/data/study/tools/x.py", "sha256": "aa", "bytes": 1},
                {
                    "path": "/opt/v4-lane/scripts/lane_entrypoint.py",
                    "sha256": "bb",
                    "bytes": 2,
                },
            ],
            kube_context="ctx",
            namespace="ns",
            publisher_pod="pod",
        )
    assert len(checked) == 1
    assert checked[0]["path"] == "/data/study/tools/x.py"
    assert len(calls) == 1


def test_enforce_dispatch_gates_wave_requires_smoke(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "s000-job.yaml").write_text(
        'name: OUTPUT_PARENT\n  value: "/data/out"\n',
        encoding="utf-8",
    )
    (bundle / "s000-configmap.yaml").write_text(
        "simulator-launch.json: |\n"
        '    {"experiment_argv":["--study-root","/data/study","--expected-study-commit","deadbeef"],'
        '"file_bindings":[]}\n'
        "  image.digest: sha256:abc\n",
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"schema_version": "v4-gpu-smoke-registry-v1", "passed_smokes": []}),
        encoding="utf-8",
    )
    with pytest.raises(gates.DispatchGateError, match="smoke-before-wave"):
        gates.enforce_dispatch_gates(
            bundle_root=bundle,
            fixture_id="horizontal",
            gate="G2",
            gpu_product="NVIDIA-B200",
            kube_context="ctx",
            namespace="ns",
            publisher_pod="pod",
            expected_study_commit="deadbeef",
            is_smoke=False,
            registry_path=registry,
        )
