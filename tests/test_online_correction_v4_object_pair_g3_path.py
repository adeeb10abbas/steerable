"""Focused tests for C7 object-pair G3 path contracts and plan builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

from experiments.online_correction_v4.droid_g3 import (
    classify_contacts,
    fixture_object_spec,
    geometry_from_scene_for_fixture,
    horizontal_geometry_from_scene,
)
from experiments.online_correction_v4.model_blind_g3 import (
    G3GateError,
    build_counterbalance_index,
    build_plan_payload,
    path_checks_per_scale_for_seed_count,
    plan_schema,
    resolve_geometry_contract,
    validate_plan_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "docs/online_correction_v4/campaign.json"
QUEUE = ROOT / "artifacts/online_correction_v4/queue.jsonl"
MOTION = ROOT / "artifacts/online_correction_v4/motion_manifest.json"
REGISTRY = (
    ROOT
    / "artifacts/online_correction_v4/setup/object_pair_reset_registry.candidate.json"
)
G2_AGGREGATE = (
    ROOT
    / "artifacts/online_correction_v4/qualification/20260906_object_pair_g2_aggregate_g2c7q20260905ap.json"
)

BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_v4_horizontal_g3_plan",
    ROOT / "tools/build_v4_horizontal_g3_plan.py",
)
builder = importlib.util.module_from_spec(BUILDER_SPEC)
assert BUILDER_SPEC.loader is not None
BUILDER_SPEC.loader.exec_module(builder)


class ObjectPairG3PathTests(unittest.TestCase):
    def test_resolve_geometry_contract_uses_horizontal_fallback(self) -> None:
        campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
        contract = resolve_geometry_contract(campaign, "object_pair")
        horizontal = campaign["fixtures"]["horizontal"]["model_blind_g3_geometry"]
        self.assertEqual(contract["relation_clearance_m"], horizontal["relation_clearance_m"])
        self.assertEqual(
            contract["stationary_object_drift_max_m"],
            horizontal["stationary_object_drift_max_m"],
        )
        self.assertFalse(contract["policy_outcome_used"])

    def test_build_object_pair_plan(self) -> None:
        report = builder.build(
            campaign_path=CAMPAIGN,
            queue_path=QUEUE,
            motion_path=MOTION,
            registry_path=REGISTRY,
            g2_aggregate_path=G2_AGGREGATE,
            output_path=ROOT / "artifacts/online_correction_v4/setup/object_pair_g3_plan.candidate.json",
            fixture_id="object_pair",
        )
        self.assertEqual(report["registered_reset_count"], 64)
        self.assertEqual(report["path_checks_per_scale"], 1536)
        self.assertEqual(report["model_request_count"], 0)
        self.assertEqual(report["behavioral_episode_count"], 0)

        plan = json.loads(
            (
                ROOT
                / "artifacts/online_correction_v4/setup/object_pair_g3_plan.candidate.json"
            ).read_text(encoding="utf-8")
        )
        validate_plan_payload(plan)
        self.assertEqual(plan["schema_version"], plan_schema("object_pair"))
        self.assertEqual(plan["fixture_id"], "object_pair")
        self.assertEqual(plan["path_sweep"]["checks_per_scale"], 1536)

    def test_counterbalance_index_uses_c7_rows(self) -> None:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        seeds = sorted(int(seed) for seed in registry["resets_by_env_seed"])
        rows = [
            json.loads(line)
            for line in QUEUE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        counterbalance = build_counterbalance_index(
            rows,
            expected_env_seeds=seeds,
            counterbalance_family="C7",
            counterbalance_fixture="object_pair",
        )
        self.assertEqual(len(counterbalance), 64)
        self.assertEqual(path_checks_per_scale_for_seed_count(len(seeds)), 1536)

    def test_classify_contacts_without_distractor(self) -> None:
        spec = fixture_object_spec("object_pair")
        result = classify_contacts(
            {
                "sponge__table": 0.2,
                "tray__table": 0.1,
            },
            active_force_threshold_n=0.05,
            fixture_spec=spec,
        )
        self.assertTrue(result["support_valid"])
        self.assertFalse(result["unmodeled_collision"])
        self.assertNotIn("banana", result["supported_by_object"])

    def test_horizontal_geometry_wrapper_unchanged(self) -> None:
        task_frame = {
            "u_left_world": [0.0, 1.0, 0.0],
            "u_front_world": [-1.0, 0.0, 0.0],
            "u_up_world": [0.0, 0.0, 1.0],
            "origin_world": [0.0, 0.0, 0.0],
        }
        scene = {
            "objects": {
                "rubiks_cube": {
                    "world_aabb_m": {
                        "min_xyz": [0.4, -0.15, 0.05],
                        "max_xyz": [0.46, -0.09, 0.11],
                    }
                },
                "bowl": {
                    "world_aabb_m": {
                        "min_xyz": [0.4, 0.1, 0.05],
                        "max_xyz": [0.52, 0.22, 0.08],
                    }
                },
            },
            "table_world_aabb_m": {
                "min_xyz": [0.2, -0.4, 0.0],
                "max_xyz": [0.8, 0.4, 0.05],
            },
        }
        horizontal = horizontal_geometry_from_scene(
            task_frame_evidence=task_frame,
            scene_state=scene,
            support_edge_margin_m=0.005,
        )
        object_pair = geometry_from_scene_for_fixture(
            fixture_id="object_pair",
            task_frame_evidence=task_frame,
            scene_state={
                **scene,
                "objects": {
                    "sponge": scene["objects"]["rubiks_cube"],
                    "tray": scene["objects"]["bowl"],
                },
            },
            support_edge_margin_m=0.005,
        )
        self.assertEqual(
            horizontal["target_footprint"].half_left,
            object_pair["target_footprint"].half_left,
        )

    def test_g2_prerequisite_rejects_wrong_fixture(self) -> None:
        campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
        motion = campaign["motion"]
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in QUEUE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        with self.assertRaises(G3GateError):
            build_plan_payload(
                fixture_id="object_pair",
                source_identity={},
                g2_prerequisite={
                    "schema_version": "v4-horizontal-g2-aggregate-receipt-v1",
                    "status": "passed",
                    "passed": True,
                    "axis_review_passed": True,
                    "expected_seed_count": 128,
                    "observed_seed_count": 128,
                    "model_request_count": 0,
                    "behavioral_episode_count": 0,
                    "receipt": {"path": "x", "sha256": "a" * 64},
                },
                geometry_contract=resolve_geometry_contract(campaign, "object_pair"),
                reset_registry=registry,
                queue_rows=rows,
                scale_candidates=motion["calibration_scale_candidates"],
                nominal_displacement_m=0.12,
                minimum_shrinking_area_fraction=0.2,
            )


if __name__ == "__main__":
    unittest.main()
