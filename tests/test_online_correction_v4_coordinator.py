"""Unit tests for the V4 campaign coordinator."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any
from unittest import mock

from experiments.online_correction_v4.coordinator import (
    ClusterBinding,
    CoordinatorBlockedError,
    CoordinatorError,
    CoordinatorInputs,
    ExecutionConfig,
    GroupReceipt,
    assert_behavioral_lane_spec,
    episode_retry_exhausted,
    load_group_receipts,
    parse_k8s_objects,
    plan_campaign,
    shard_group_units,
    storage_budget_allows,
)
from experiments.online_correction_v4.droid_contract import PrefixMode, sha256_file
from experiments.online_correction_v4.registry import CampaignRegistry, ExecutionGroup


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "docs/online_correction_v4/campaign.json"
TEMPLATE_LOCK = ROOT / "docs/online_correction_v4/runtime_lock.template.json"
QUEUE_PATH = ROOT / "artifacts/online_correction_v4/queue.jsonl"
QUEUE_MANIFEST_PATH = ROOT / "artifacts/online_correction_v4/queue_manifest.json"
SPEC_TEMPLATE = ROOT / "deploy/k8s/v4_lane_bundle/spec.example.json"
CLI = ROOT / "tools/launch_online_correction_v4.py"


def _released_lock(*, manifest_sha256: str, config_sha256: str, families: list[str]) -> dict[str, Any]:
    template = json.loads(TEMPLATE_LOCK.read_text(encoding="utf-8"))
    blocked = {fid: "blocked for test" for fid in template["blocked_families"] if fid not in families}
    template["release_status"] = "RELEASED"
    template["released_families"] = families
    template["blocked_families"] = blocked
    template["manifest_sha256"] = manifest_sha256
    template["config_sha256"] = config_sha256
    template["prefix_mode"] = PrefixMode.FRESH_SESSION_DETERMINISTIC_REPLAY.value
    template["prefix_mode_receipt_sha256"] = "a" * 64
    template["runner"] = {
        "commit": "c" * 40,
        "entrypoint": "tools/run_online_correction_v4.py",
        "sha256": "b" * 64,
    }
    template["writer_contract"] = {
        "schema_version": "v4-droid-writer-contract-v1",
        "output_parent_uri": "file:///persistent/v4/attempts",
        "viewport_video_required": True,
        "write_once_attempt_directories": True,
        "incremental_fsync_required": True,
        "required_streams": ["viewport_video", "trajectory", "requests"],
    }
    for name in ("cosmos3_nano_droid", "pi05_droid"):
        template["policies"][name].update(
            {
                "checkpoint_sha256": "1" * 64,
                "checkpoint_uri": f"file:///persistent/v4/checkpoints/{name}",
                "runtime_image_digest": "sha256:" + ("2" * 64),
                "integration_commit": "c" * 40,
                "native_control_dt_s": 0.05,
                "achieved_delay_s": 0.10,
                "achieved_standard_query_period_s": 0.50,
                "achieved_fast_query_period_s": 0.25,
                "prediction_horizon_actions": 32 if name.startswith("cosmos") else 15,
                "policy_reset_and_history_contract_uri": f"file:///persistent/v4/contracts/{name}",
            }
        )
    for fixture in template["fixtures"].values():
        fixture.update(
            {
                "geometry_sha256": "3" * 64,
                "scorer_sha256": "4" * 64,
                "reset_registry_sha256": "5" * 64,
                "geometry_uri": "file:///persistent/v4/geometry.json",
                "scorer_uri": "file:///persistent/v4/scorer.json",
                "reset_registry_uri": "file:///persistent/v4/resets.jsonl",
                "frame_transform_uri": "file:///persistent/v4/frame.json",
                "goal_geometry_and_tolerances_uri": "file:///persistent/v4/goals.json",
                "trigger_release_detector_uri": "file:///persistent/v4/detectors.json",
                "intervention_trajectory_registry_uri": "file:///persistent/v4/motion.jsonl",
                "scoring_and_visibility_thresholds_uri": "file:///persistent/v4/thresholds.json",
                "calibration_scale": 0.12,
                "D_cap_m": 0.12,
            }
        )
    for receipt in template["receipts"].values():
        receipt.update(
            {
                "passed": True,
                "family_ids": families,
                "uri": "file:///persistent/v4/receipts/gate.json",
                "sha256": "6" * 64,
            }
        )
    return template


def _launch_matrix(*, template_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "release_status": "RELEASED",
        "qualified_lanes": [
            {
                "lane_id": "lane00",
                "hardware_stratum": "a40-a100-test",
                "lane_spec_template_path": str(template_path),
            },
            {
                "lane_id": "lane01",
                "hardware_stratum": "a40-a100-test",
                "lane_spec_template_path": str(template_path),
            },
        ],
        "resource_budget": {
            "authorized_storage_bytes": 10_000_000_000_000,
            "estimated_bytes_per_episode": 1000,
            "estimated_bytes_per_infra_retry": 100,
        },
        "dispatch": {
            "max_infra_retries_per_episode": 3,
            "lane_quarantine_threshold": 5,
        },
    }


def _write_group_receipt(
    directory: Path,
    *,
    group_id: str,
    manifest_sha256: str,
    accepted: list[str],
    partial: list[str] | None = None,
    status: str = "complete",
) -> None:
    payload = {
        "group_id": group_id,
        "manifest_sha256": manifest_sha256,
        "accepted_episode_ids": accepted,
        "partial_episode_ids": partial or [],
        "status": status,
    }
    safe = group_id.replace(":", "_")
    path = directory / f"{safe}.group_receipt.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _simulator_launch_from_configmap(bundle_root: Path) -> dict[str, Any]:
    configmap = (bundle_root / "configmap.yaml").read_text(encoding="utf-8")
    marker = "simulator-launch.json: |"
    start = configmap.index(marker) + len(marker)
    lines: list[str] = []
    for line in configmap[start:].splitlines():
        if line.startswith("  ") and not line.startswith("    ") and line.strip():
            break
        if line.startswith("    "):
            lines.append(line[4:])
    return json.loads("\n".join(lines))


def _pvc_output_parent(tmp_path: Path) -> str:
    path = tmp_path / "pvc"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


class CoordinatorHelperTests(unittest.TestCase):
    def test_shard_groups_is_deterministic_and_balanced(self) -> None:
        from experiments.online_correction_v4.coordinator import LaneStratum

        lanes = [
            LaneStratum("lane00", "stratum-a", {}, ROOT / "deploy/k8s/v4_lane_bundle"),
            LaneStratum("lane01", "stratum-a", {}, ROOT / "deploy/k8s/v4_lane_bundle"),
        ]
        groups = [
            ExecutionGroup(group_id=f"policy:fixture{i}", policy="policy", fixture=f"fixture{i}")
            for i in range(6)
        ]
        units = [(group, [f"ep-{i}"]) for i, group in enumerate(groups)]
        first = shard_group_units(units, lanes)
        second = shard_group_units(units, lanes)
        self.assertEqual(first, second)
        self.assertEqual(sum(len(v) for v in first.values()), 6)
        self.assertLessEqual(
            max(len(v) for v in first.values())
            - min(len(v) for v in first.values()),
            1,
        )

    def test_shard_groups_never_mixes_policy_servers_on_one_lane(self) -> None:
        from experiments.online_correction_v4.coordinator import LaneStratum

        lanes = [
            LaneStratum(f"lane{i:02d}", "stratum-a", {}, ROOT)
            for i in range(4)
        ]
        units = [
            (
                ExecutionGroup(
                    group_id=f"{policy}:fixture{i}",
                    policy=policy,
                    fixture=f"fixture{i}",
                ),
                [f"{policy}-ep-{i}"],
            )
            for policy in ("nano", "pi05")
            for i in range(4)
        ]
        buckets = shard_group_units(units, lanes)
        for assignments in buckets.values():
            self.assertLessEqual(
                len({group.policy for group, _episode_ids in assignments}),
                1,
            )

    def test_storage_budget_blocks_when_exceeded(self) -> None:
        cfg = ExecutionConfig(
            authorized_storage_bytes=1000,
            estimated_bytes_per_episode=500,
            estimated_bytes_per_infra_retry=100,
        )
        allowed, summary = storage_budget_allows(
            pending_episodes=3,
            pending_infra_retries=0,
            bytes_already_used=0,
            execution_config=cfg,
        )
        self.assertFalse(allowed)
        self.assertEqual(summary["status"], "exceeded")

    def test_infra_retry_exhaustion_does_not_apply_to_valid_failures(self) -> None:
        cfg = ExecutionConfig(max_infra_retries=2)
        self.assertTrue(
            episode_retry_exhausted("ep-1", episode_infra_failures=Counter({"ep-1": 2}), execution_config=cfg)
        )
        self.assertFalse(
            episode_retry_exhausted("ep-2", episode_infra_failures=Counter({"ep-2": 1}), execution_config=cfg)
        )


class CoordinatorPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_sha = sha256_file(CONFIG_PATH)
        cls.queue_sha = sha256_file(QUEUE_PATH)
        cls.manifest_sha = json.loads(QUEUE_MANIFEST_PATH.read_text())["planning_manifest_sha256"]

    def _inputs(
        self,
        tmp_path: Path,
        *,
        families: list[str],
        receipts_dir: Path | None = None,
        lane_failures: Counter[str] | None = None,
        episode_failures: Counter[str] | None = None,
        render_root: Path | None = None,
        execution_config: ExecutionConfig | None = None,
        output_parent: str | None = None,
    ) -> CoordinatorInputs:
        lock_path = tmp_path / "runtime_lock.json"
        lock_path.write_text(
            json.dumps(
                _released_lock(
                    manifest_sha256=self.manifest_sha,
                    config_sha256=self.config_sha,
                    families=families,
                )
            ),
            encoding="utf-8",
        )
        matrix_path = tmp_path / "launch_matrix.json"
        matrix_path.write_text(json.dumps(_launch_matrix(template_path=SPEC_TEMPLATE)), encoding="utf-8")
        pvc_parent = output_parent or _pvc_output_parent(tmp_path)
        return CoordinatorInputs(
            runtime_lock_path=lock_path,
            queue_path=QUEUE_PATH,
            queue_manifest_path=QUEUE_MANIFEST_PATH,
            launch_matrix_path=matrix_path,
            campaign_config_path=CONFIG_PATH,
            group_receipts_dir=receipts_dir,
            render_output_root=render_root,
            cluster_binding=ClusterBinding(
                kube_context="test-context",
                namespace="test-namespace",
                pvc="test-pvc",
                output_parent=pvc_parent,
            )
            if render_root is not None
            else None,
            lane_infra_failures=lane_failures or Counter(),
            episode_infra_failures=episode_failures or Counter(),
            execution_config=execution_config or ExecutionConfig(),
            repo_root=ROOT,
        )

    def test_template_runtime_lock_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            matrix_path = tmp_path / "launch_matrix.json"
            matrix_path.write_text(json.dumps(_launch_matrix(template_path=SPEC_TEMPLATE)), encoding="utf-8")
            inputs = CoordinatorInputs(
                runtime_lock_path=TEMPLATE_LOCK,
                queue_path=QUEUE_PATH,
                queue_manifest_path=QUEUE_MANIFEST_PATH,
                launch_matrix_path=matrix_path,
                campaign_config_path=CONFIG_PATH,
                repo_root=ROOT,
            )
            with self.assertRaises(CoordinatorBlockedError):
                plan_campaign(inputs)

    def test_plan_c1_only_assigns_released_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inputs = self._inputs(Path(tmp), families=["C1"])
            plan = plan_campaign(inputs)
            self.assertEqual(plan.release_status, "RELEASED")
            self.assertGreater(plan.dispatchable_group_count, 0)
            self.assertFalse(plan.effect_size_peeking)
            assigned_groups = {gid for lane in plan.lane_assignments for gid in lane.group_ids}
            registry = CampaignRegistry.from_manifest_path(QUEUE_PATH)
            for group_id in assigned_groups:
                group = registry.by_execution_group[group_id]
                for episode_id in [
                    eid
                    for lane in plan.lane_assignments
                    if group_id in lane.group_ids
                    for eid in lane.remaining_episode_ids
                ]:
                    self.assertEqual(registry.get(episode_id).family, "C1")

    def test_pilot_release_rejects_confirmatory_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inputs = self._inputs(Path(tmp), families=["C1"])
            lock = json.loads(inputs.runtime_lock_path.read_text(encoding="utf-8"))
            lock["release_status"] = "PILOT_RELEASED"
            inputs.runtime_lock_path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaises(CoordinatorBlockedError) as ctx:
                plan_campaign(inputs)
            self.assertIn("engineering_pilot-only", str(ctx.exception))

    def test_pilot_release_dispatches_engineering_pilot_only_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            row = next(
                json.loads(line)
                for line in QUEUE_PATH.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["family"] == "C1"
            )
            row["cohort"] = "engineering_pilot"
            queue_path = tmp_path / "pilot.jsonl"
            queue_path.write_text(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            queue_sha = sha256_file(queue_path)
            queue_manifest_path = tmp_path / "pilot-manifest.json"
            queue_manifest_path.write_text(
                json.dumps(
                    {
                        "queue_sha256": queue_sha,
                        "frozen_queue_sha256": queue_sha,
                        "planning_manifest_sha256": queue_sha,
                    }
                ),
                encoding="utf-8",
            )
            lock = _released_lock(
                manifest_sha256=queue_sha,
                config_sha256=self.config_sha,
                families=["C1"],
            )
            lock["release_status"] = "PILOT_RELEASED"
            lock_path = tmp_path / "runtime_lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            matrix_path = tmp_path / "launch_matrix.json"
            matrix_path.write_text(
                json.dumps(_launch_matrix(template_path=SPEC_TEMPLATE)),
                encoding="utf-8",
            )
            plan = plan_campaign(
                CoordinatorInputs(
                    runtime_lock_path=lock_path,
                    queue_path=queue_path,
                    queue_manifest_path=queue_manifest_path,
                    launch_matrix_path=matrix_path,
                    campaign_config_path=CONFIG_PATH,
                    repo_root=ROOT,
                )
            )
            self.assertEqual(plan.release_status, "PILOT_RELEASED")
            self.assertEqual(plan.behavioral_episode_count, 1)

    def test_resume_from_group_receipts_skips_accepted_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry = CampaignRegistry.from_manifest_path(QUEUE_PATH)
            group = next(
                g
                for g in registry.iter_groups()
                if g.policy == "cosmos3_nano_droid" and g.fixture == "horizontal"
            )
            c1_episodes = [row.episode_id for row in group.rows if row.family == "C1"]
            receipts_dir = tmp_path / "receipts"
            receipts_dir.mkdir()
            _write_group_receipt(
                receipts_dir,
                group_id=group.group_id,
                manifest_sha256=self.manifest_sha,
                accepted=c1_episodes[:4],
                partial=c1_episodes[4:6],
                status="partial",
            )
            inputs = self._inputs(tmp_path, families=["C1"], receipts_dir=receipts_dir)
            plan = plan_campaign(inputs)
            self.assertEqual(plan.resume_receipt_count, 1)
            self.assertGreaterEqual(plan.partial_group_count, 1)
            for lane in plan.lane_assignments:
                for episode_id in lane.remaining_episode_ids:
                    self.assertNotIn(episode_id, set(c1_episodes[:4]))

    def test_c3_blocked_until_c1_controls_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inputs = self._inputs(Path(tmp), families=["C1", "C3"])
            plan = plan_campaign(inputs)
            self.assertTrue(plan.blocked_groups)
            self.assertTrue(
                any(item["reason"] == "control_dependencies_unsatisfied" for item in plan.blocked_groups)
            )

    def test_quarantined_lane_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inputs = self._inputs(
                Path(tmp),
                families=["C1"],
                lane_failures=Counter({"lane00": 5}),
            )
            plan = plan_campaign(inputs)
            self.assertIn("lane00", plan.quarantined_lanes)
            self.assertTrue(all(item.lane_id != "lane00" for item in plan.lane_assignments))

    def test_dry_run_render_produces_teardown_inventory_without_kubectl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            render_root = tmp_path / "rendered"
            output_parent = _pvc_output_parent(tmp_path)
            inputs = self._inputs(Path(tmp), families=["C1"], render_root=render_root, output_parent=output_parent)
            with mock.patch("experiments.online_correction_v4.coordinator.kubectl_create_bundle") as create_mock:
                plan = plan_campaign(inputs, render_bundles=True, create_on_cluster=False)
                create_mock.assert_not_called()
            self.assertTrue(plan.lane_assignments)
            assignment = plan.lane_assignments[0]
            self.assertIsNotNone(assignment.bundle_root)
            self.assertTrue(assignment.pvc_binding_root)
            self.assertTrue(assignment.pvc_binding_root.startswith(output_parent))
            inventory = assignment.teardown_inventory
            assert inventory is not None
            kinds = {item["kind"] for item in inventory["objects"]}
            self.assertEqual(kinds, {"ConfigMap", "Job", "Service"})
            self.assertEqual(len(inventory["objects"]), 4)
            self.assertEqual(sum(1 for item in inventory["objects"] if item["kind"] == "Job"), 2)
            bundle_root = Path(assignment.bundle_root)
            inventory_obj = parse_k8s_objects(bundle_root, kube_context="test-context")
            self.assertEqual(len(inventory_obj.objects), 4)
            configmap = (bundle_root / "configmap.yaml").read_text(encoding="utf-8")
            self.assertNotIn("infrastructure_qualification_only_no_scientific_behavior", configmap)
            self.assertNotIn('"/usr/bin/true"', configmap)
            self.assertIn("online_correction_v4", configmap)
            self.assertGreater(plan.behavioral_episode_count, 0)
            self.assertEqual(plan.dispatch_mode, "behavioral")
            sim_launch = _simulator_launch_from_configmap(bundle_root)
            argv = sim_launch["experiment_argv"]
            self.assertIn("--launch-config", argv)
            self.assertIn("/opt/v4-lane/config/simulator-launch.json", argv)
            manifest_flag = argv.index("--dispatch-manifest")
            manifest_path = argv[manifest_flag + 1]
            self.assertTrue(manifest_path.startswith(output_parent))
            self.assertIn(".coord-bindings/", manifest_path)
            self.assertTrue(argv[1].startswith(output_parent))
            for binding in sim_launch["file_bindings"]:
                if binding["path"].endswith("lane_dispatch_manifest.json") or "/.coord-bindings/" in binding["path"]:
                    self.assertTrue(str(binding["path"]).startswith(output_parent))
            local_manifest = json.loads(
                (bundle_root / ".bindings" / "lane_dispatch_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(local_manifest["episode_ids"], list(assignment.remaining_episode_ids))
            self.assertEqual(local_manifest["group_ids"], list(assignment.group_ids))
            self.assertTrue(local_manifest.get("one_episode_per_process"))
            self.assertEqual(local_manifest["queue_path"], f"{assignment.pvc_binding_root}/queue.jsonl")


class CoordinatorCreateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_sha = sha256_file(CONFIG_PATH)
        cls.manifest_sha = json.loads(QUEUE_MANIFEST_PATH.read_text())["planning_manifest_sha256"]

    def test_create_invokes_kubectl_only_in_create_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lock_path = tmp_path / "runtime_lock.json"
            lock_path.write_text(
                json.dumps(
                    _released_lock(
                        manifest_sha256=self.manifest_sha,
                        config_sha256=sha256_file(CONFIG_PATH),
                        families=["C1"],
                    )
                ),
                encoding="utf-8",
            )
            matrix_path = tmp_path / "launch_matrix.json"
            matrix_path.write_text(
                json.dumps(_launch_matrix(template_path=SPEC_TEMPLATE)),
                encoding="utf-8",
            )
            render_root = tmp_path / "rendered"
            lease_root = tmp_path / "leases"
            output_parent = _pvc_output_parent(tmp_path)
            inputs = CoordinatorInputs(
                runtime_lock_path=lock_path,
                queue_path=QUEUE_PATH,
                queue_manifest_path=QUEUE_MANIFEST_PATH,
                launch_matrix_path=matrix_path,
                campaign_config_path=CONFIG_PATH,
                render_output_root=render_root,
                group_lease_root=lease_root,
                cluster_binding=ClusterBinding(
                    kube_context="test-context",
                    namespace="test-namespace",
                    pvc="test-pvc",
                    output_parent=output_parent,
                ),
                repo_root=ROOT,
            )

            def _fake_kubectl(**kwargs: Any) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(args=[], returncode=0, stdout="created", stderr="")

            plan = plan_campaign(
                inputs,
                render_bundles=True,
                create_on_cluster=True,
                kubectl_runner=lambda *args, **kwargs: _fake_kubectl(**kwargs),
            )
            self.assertTrue(plan.lane_assignments)
            self.assertTrue(all(item.acquired_group_leases for item in plan.lane_assignments))


class CoordinatorRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_sha = sha256_file(CONFIG_PATH)
        cls.manifest_sha = json.loads(QUEUE_MANIFEST_PATH.read_text())["planning_manifest_sha256"]

    def _released_inputs(
        self,
        tmp_path: Path,
        *,
        families: list[str],
        render_root: Path | None = None,
        lease_root: Path | None = None,
        coordination_state: Path | None = None,
        qualification_only: bool = False,
        lane_failures: Counter[str] | None = None,
        output_parent: str | None = None,
    ) -> CoordinatorInputs:
        lock_path = tmp_path / "runtime_lock.json"
        lock_path.write_text(
            json.dumps(
                _released_lock(
                    manifest_sha256=self.manifest_sha,
                    config_sha256=self.config_sha,
                    families=families,
                )
            ),
            encoding="utf-8",
        )
        matrix_path = tmp_path / "launch_matrix.json"
        matrix_path.write_text(json.dumps(_launch_matrix(template_path=SPEC_TEMPLATE)), encoding="utf-8")
        pvc_parent = output_parent or _pvc_output_parent(tmp_path)
        return CoordinatorInputs(
            runtime_lock_path=lock_path,
            queue_path=QUEUE_PATH,
            queue_manifest_path=QUEUE_MANIFEST_PATH,
            launch_matrix_path=matrix_path,
            campaign_config_path=CONFIG_PATH,
            render_output_root=render_root,
            group_lease_root=lease_root,
            coordination_state_path=coordination_state,
            cluster_binding=ClusterBinding(
                kube_context="test-context",
                namespace="test-namespace",
                pvc="test-pvc",
                output_parent=pvc_parent,
            )
            if render_root is not None
            else None,
            lane_infra_failures=lane_failures or Counter(),
            qualification_only=qualification_only,
            repo_root=ROOT,
        )

    def test_qualification_only_reports_zero_behavioral_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            render_root = tmp_path / "rendered"
            inputs = self._released_inputs(
                tmp_path,
                families=["C1"],
                render_root=render_root,
                qualification_only=True,
            )
            plan = plan_campaign(inputs, render_bundles=True, create_on_cluster=False)
            self.assertEqual(plan.dispatch_mode, "qualification_only")
            self.assertEqual(plan.behavioral_episode_count, 0)
            self.assertTrue(plan.lane_assignments)
            configmap = (Path(plan.lane_assignments[0].bundle_root) / "configmap.yaml").read_text(
                encoding="utf-8"
            )
            self.assertIn("infrastructure_qualification_only_no_scientific_behavior", configmap)
            self.assertIn('"/usr/bin/true"', configmap)

    def test_create_blocked_when_all_lanes_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            render_root = tmp_path / "rendered"
            lease_root = tmp_path / "leases"
            lane_failures = Counter({"lane00": 5, "lane01": 5})
            for create in (False, True):
                inputs = self._released_inputs(
                    tmp_path,
                    families=["C1"],
                    render_root=render_root,
                    lease_root=lease_root,
                    lane_failures=lane_failures,
                )
                with self.assertRaises(CoordinatorBlockedError) as ctx:
                    plan_campaign(inputs, render_bundles=True, create_on_cluster=create)
                self.assertIn("quarantined", str(ctx.exception).lower())

    def test_durable_group_lease_blocks_overlapping_wave(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            render_root = tmp_path / "rendered"
            lease_root = tmp_path / "leases"
            inputs = self._released_inputs(
                tmp_path,
                families=["C1"],
                render_root=render_root,
                lease_root=lease_root,
            )

            def _fake_kubectl(**kwargs: Any) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(args=[], returncode=0, stdout="created", stderr="")

            first = plan_campaign(
                inputs,
                render_bundles=True,
                create_on_cluster=True,
                kubectl_runner=lambda *args, **kwargs: _fake_kubectl(**kwargs),
            )
            self.assertTrue(first.lane_assignments)
            inputs_second = self._released_inputs(
                tmp_path,
                families=["C1"],
                render_root=tmp_path / "rendered-second",
                lease_root=lease_root,
            )
            second = plan_campaign(
                inputs_second,
                render_bundles=True,
                create_on_cluster=True,
                kubectl_runner=lambda *args, **kwargs: _fake_kubectl(**kwargs),
            )
            self.assertEqual(second.lane_assignments, ())
            self.assertEqual(second.behavioral_episode_count, 0)
            leased_groups = {
                group_id
                for path in (lease_root / "leases").glob("*.lease")
                for group_id in [json.loads(path.read_text())["group_id"]]
            }
            self.assertTrue(leased_groups)

    def test_create_surfaces_durable_lease_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            render_root = tmp_path / "rendered"
            lease_root = tmp_path / "leases"
            inputs = self._released_inputs(
                tmp_path,
                families=["C1"],
                render_root=render_root,
                lease_root=lease_root,
            )

            def _fake_kubectl(**kwargs: Any) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(args=[], returncode=0, stdout="created", stderr="")

            from experiments.online_correction_v4.leases import LeaseConflict

            with mock.patch(
                "experiments.online_correction_v4.coordinator.GroupLeaseStore.acquire",
                side_effect=LeaseConflict("group 'x' is leased"),
            ):
                with self.assertRaises(CoordinatorBlockedError) as ctx:
                    plan_campaign(
                        inputs,
                        render_bundles=True,
                        create_on_cluster=True,
                        kubectl_runner=lambda *args, **kwargs: _fake_kubectl(**kwargs),
                    )
            self.assertIn("leased", str(ctx.exception).lower())

    def test_coordination_state_wires_retry_and_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry = CampaignRegistry.from_manifest_path(QUEUE_PATH)
            c1_group = next(
                g for g in registry.iter_groups() if any(row.family == "C1" for row in g.rows)
            )
            exhausted_episode = next(row.episode_id for row in c1_group.rows if row.family == "C1")
            state_path = tmp_path / "coordination_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": "v4-coordination-state-v1",
                        "lane_infra_failures": {"lane00": 5},
                        "episode_infra_failures": {exhausted_episode: 3},
                        "reserved_attempt_ids": ["attempt0001"],
                        "attempt_index": 2,
                    }
                ),
                encoding="utf-8",
            )
            inputs = self._released_inputs(
                tmp_path,
                families=["C1"],
                coordination_state=state_path,
            )
            plan = plan_campaign(inputs)
            self.assertIn("lane00", plan.quarantined_lanes)
            self.assertTrue(plan.scheduling_inputs["coordination_state_loaded"])
            assigned_episodes = {
                episode_id
                for lane in plan.lane_assignments
                for episode_id in lane.remaining_episode_ids
            }
            self.assertNotIn(exhausted_episode, assigned_episodes)
            if plan.lane_assignments:
                self.assertNotEqual(plan.lane_assignments[0].attempt_id, "attempt0001")

    def test_attempt_identity_skips_reserved_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            render_root = tmp_path / "rendered"
            (render_root / "lane00-attempt0001-a40-a100-test").mkdir(parents=True)
            inputs = self._released_inputs(tmp_path, families=["C1"], render_root=render_root)
            plan = plan_campaign(inputs, render_bundles=True, create_on_cluster=False)
            self.assertTrue(plan.lane_assignments)
            self.assertNotEqual(plan.lane_assignments[0].attempt_id, "attempt0001")

    def test_launch_matrix_not_released_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inputs = self._released_inputs(tmp_path, families=["C1"])
            matrix = json.loads(inputs.launch_matrix_path.read_text(encoding="utf-8"))
            matrix["release_status"] = "NOT_RELEASED"
            inputs.launch_matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
            with self.assertRaises(CoordinatorBlockedError) as ctx:
                plan_campaign(inputs)
            self.assertIn("NOT_RELEASED", str(ctx.exception))

    def test_qualification_template_cannot_masquerade_as_behavioral(self) -> None:
        template = json.loads(SPEC_TEMPLATE.read_text(encoding="utf-8"))
        with self.assertRaises(CoordinatorBlockedError):
            assert_behavioral_lane_spec(template)

    def test_kubectl_failure_rolls_back_new_leases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            render_root = tmp_path / "rendered"
            lease_root = tmp_path / "leases"
            inputs = self._released_inputs(
                tmp_path,
                families=["C1"],
                render_root=render_root,
                lease_root=lease_root,
            )

            def _fail_kubectl(**kwargs: Any) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")

            with self.assertRaises(CoordinatorError):
                plan_campaign(
                    inputs,
                    render_bundles=True,
                    create_on_cluster=True,
                    kubectl_runner=lambda *args, **kwargs: _fail_kubectl(**kwargs),
                )
            self.assertFalse(list((lease_root / "leases").glob("*.lease")))

    def test_partial_multi_lane_create_records_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            render_root = tmp_path / "rendered"
            lease_root = tmp_path / "leases"
            inputs = self._released_inputs(
                tmp_path,
                families=["C1"],
                render_root=render_root,
                lease_root=lease_root,
            )
            calls = {"count": 0}

            def _kubectl_every_other(**kwargs: Any) -> subprocess.CompletedProcess[str]:
                calls["count"] += 1
                if calls["count"] == 1:
                    return subprocess.CompletedProcess(args=[], returncode=0, stdout="created", stderr="")
                return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="second lane failed")

            plan = plan_campaign(
                inputs,
                render_bundles=True,
                create_on_cluster=True,
                kubectl_runner=lambda *args, **kwargs: _kubectl_every_other(**kwargs),
            )
            self.assertTrue(plan.partial_create_wave)
            assert plan.create_wave_receipt is not None
            self.assertEqual(plan.create_wave_receipt["status"], "partial_create_wave")
            self.assertEqual(len(plan.create_wave_receipt["created_assignments"]), 1)
            self.assertIsNotNone(plan.create_wave_receipt["failed_assignment"])
            created = [item for item in plan.lane_assignments if item.create_status == "created"]
            self.assertEqual(len(created), 1)
            self.assertTrue(list((lease_root / "leases").glob("*.lease")))

    def test_coordination_state_written_after_successful_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state_path = tmp_path / "coordination_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": "v4-coordination-state-v1",
                        "lane_infra_failures": {},
                        "episode_infra_failures": {},
                        "reserved_attempt_ids": [],
                        "attempt_index": 1,
                    }
                ),
                encoding="utf-8",
            )
            render_root = tmp_path / "rendered"
            inputs = self._released_inputs(
                tmp_path,
                families=["C1"],
                render_root=render_root,
                coordination_state=state_path,
            )
            plan = plan_campaign(inputs, render_bundles=True, create_on_cluster=False)
            self.assertTrue(plan.lane_assignments)
            updated = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertGreater(updated["attempt_index"], 1)
            self.assertIn(plan.lane_assignments[0].attempt_id, updated["reserved_attempt_ids"])

    def test_create_publishes_bindings_under_output_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            render_root = tmp_path / "rendered"
            lease_root = tmp_path / "leases"
            output_parent = _pvc_output_parent(tmp_path)
            inputs = self._released_inputs(
                tmp_path,
                families=["C1"],
                render_root=render_root,
                lease_root=lease_root,
                output_parent=output_parent,
            )

            def _fake_kubectl(**kwargs: Any) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(args=[], returncode=0, stdout="created", stderr="")

            plan = plan_campaign(
                inputs,
                render_bundles=True,
                create_on_cluster=True,
                kubectl_runner=lambda *args, **kwargs: _fake_kubectl(**kwargs),
            )
            assignment = plan.lane_assignments[0]
            assert assignment.pvc_binding_root is not None
            pvc_manifest = Path(assignment.pvc_binding_root) / "lane_dispatch_manifest.json"
            self.assertTrue(pvc_manifest.is_file())
            payload = json.loads(pvc_manifest.read_text(encoding="utf-8"))
            self.assertTrue(payload["queue_path"].startswith(output_parent))


class CoordinatorCliTests(unittest.TestCase):
    def test_cli_dry_run_blocked_on_template_lock(self) -> None:
        result = subprocess.run(
            [
                "python3",
                str(CLI),
                "--dry-run",
                "--runtime-lock",
                str(TEMPLATE_LOCK),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("blocked", result.stderr.lower())

    def test_cli_create_requires_cluster_binding(self) -> None:
        result = subprocess.run(
            [
                "python3",
                str(CLI),
                "--create",
                "--runtime-lock",
                str(TEMPLATE_LOCK),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertTrue(
            "kube-context" in result.stderr or "group-lease-root" in result.stderr
        )

    def test_cli_behavioral_create_requires_group_lease_root(self) -> None:
        result = subprocess.run(
            [
                "python3",
                str(CLI),
                "--create",
                "--runtime-lock",
                str(TEMPLATE_LOCK),
                "--kube-context",
                "test",
                "--namespace",
                "test",
                "--pvc",
                "test",
                "--output-parent",
                "/data/users/test/v4/raw",
                "--render-output-root",
                "/tmp/v4-render",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("group-lease-root", result.stderr)


class CoordinatorReceiptLoaderTests(unittest.TestCase):
    def test_load_group_receipts_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_group_receipt(
                tmp_path,
                group_id="cosmos3_nano_droid:horizontal",
                manifest_sha256="a" * 64,
                accepted=["ep-1"],
                status="complete",
            )
            receipts = load_group_receipts(tmp_path)
            self.assertEqual(len(receipts), 1)
            self.assertIsInstance(receipts[0], GroupReceipt)
            self.assertEqual(receipts[0].accepted_episode_ids, ("ep-1",))


if __name__ == "__main__":
    unittest.main()
