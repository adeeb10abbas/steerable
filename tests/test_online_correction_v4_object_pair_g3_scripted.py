"""Focused tests for C7 object-pair G3 scripted contracts and render spec."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

from experiments.online_correction_v4.droid_g3 import fixture_object_spec
from experiments.online_correction_v4.droid_g3_scripted import (
    _object_dropped,
    _object_grabbed,
    run_scripted_horizontal_check,
    trajectory_schema,
)
from experiments.online_correction_v4.model_blind_g3 import (
    PATH_SAMPLE_INTERVAL_S,
    compile_path_scale_receipt,
    compile_path_seed_receipt,
    compile_scripted_check_receipt,
    expected_path_check_keys,
    expected_scripted_check_keys,
    scripted_receipt_schema,
    sha256_file,
    validate_path_scale_receipt,
    validate_plan_payload,
    validate_scripted_check_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
OBJECT_PAIR_PLAN = (
    ROOT
    / "artifacts/online_correction_v4/setup/object_pair_g3_plan.candidate.json"
)
OBJECT_PAIR_SPEC = (
    ROOT
    / "deploy/k8s/v4_lane_bundle/g3-scripted-object-pair-spec.example.json"
)

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_v4_horizontal_g3_scripted_seed",
    ROOT / "tools/run_v4_horizontal_g3_scripted_seed.py",
)
runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(runner)

RENDERER_SPEC = importlib.util.spec_from_file_location(
    "render_v4_horizontal_g3_scripted_k8s_jobs",
    ROOT / "tools/render_v4_horizontal_g3_scripted_k8s_jobs.py",
)
renderer = importlib.util.module_from_spec(RENDERER_SPEC)
assert RENDERER_SPEC.loader is not None
RENDERER_SPEC.loader.exec_module(renderer)


def _evidence(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _passing_path_observations(*, suffix: str) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for goal, scenario in expected_path_check_keys():
        measured = f"evidence/measured/{goal}_{scenario}_{suffix}.json".encode("utf-8")
        reference = f"evidence/reference/{goal}_{scenario}_{suffix}.json".encode("utf-8")
        observations.append(
            {
                "planned_duration_s": 1.0,
                "sample_interval_s": PATH_SAMPLE_INTERVAL_S,
                "sample_count": 51,
                "measured_pose_evidence": _evidence(
                    f"artifacts/g3/measured/{goal}_{scenario}_{suffix}.json",
                    measured,
                ),
                "reference_pose_evidence": _evidence(
                    f"artifacts/g3/reference/{goal}_{scenario}_{suffix}.json",
                    reference,
                ),
                "path_conformance": True,
                "collision_free": True,
                "support_valid": True,
                "reachable_workspace": True,
                "legal_goal_nonempty": True,
                "reference_robot_contact": False,
                "unmodeled_collision": False,
                "reasons": [],
            }
        )
    return observations


def _goal_area_cases() -> list[dict[str, object]]:
    return [
        {
            "relation": goal,
            "original_area_m2": 0.10,
            "destination_area_m2": 0.09,
            "shrinking_direction": True,
            "removed_area_fraction": 0.10,
            "minimum_shrinking_area_fraction": 0.20,
            "original_goal_empty": False,
            "destination_goal_empty": False,
            "passes_information_gate": True,
        }
        for goal in ("left", "right", "front", "behind")
    ]


class ObjectPairG3ScriptedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(OBJECT_PAIR_PLAN.read_text(encoding="utf-8"))
        validate_plan_payload(cls.plan)
        cls.plan_receipt = {
            "path": str(OBJECT_PAIR_PLAN),
            "sha256": sha256_file(OBJECT_PAIR_PLAN),
        }

    def test_fixture_object_spec_is_sponge_tray_no_distractor(self) -> None:
        spec = fixture_object_spec("object_pair")
        self.assertEqual(spec.target_object, "sponge")
        self.assertEqual(spec.reference_object, "tray")
        self.assertIsNone(spec.distractor_object)

    def test_object_contact_probes_use_fixture_target(self) -> None:
        observed: list[tuple[str, str]] = []

        def grabbed(_env, *, object: str, env_id: int) -> bool:
            observed.append(("grabbed", object))
            return env_id == 0

        def dropped(_env, *, object: str, env_id: int) -> bool:
            observed.append(("dropped", object))
            return env_id == 0

        env = SimpleNamespace(
            backend=SimpleNamespace(
                env=object(),
                modules={"object_grabbed": grabbed, "object_dropped": dropped},
            )
        )
        self.assertTrue(_object_grabbed(env, "sponge"))
        self.assertTrue(_object_dropped(env, "sponge"))
        self.assertEqual(observed, [("grabbed", "sponge"), ("dropped", "sponge")])

    def test_scripted_check_compile_uses_object_pair_schema(self) -> None:
        receipt = compile_scripted_check_receipt(
            check_kind="moving",
            environment_seed=int(self.plan["registered_env_seeds"][0]),
            goal="left",
            reference_position="endpoint",
            scale=0.5,
            displacement_m=0.06,
            fixture_id="object_pair",
            observation={
                "grasped": True,
                "transported": True,
                "released": True,
                "stably_placed": True,
                "goal_satisfied": True,
                "evidence": _evidence("artifacts/t.json", b"{}"),
                "reasons": [],
                "passed": True,
            },
        )
        validate_scripted_check_receipt(receipt)
        self.assertEqual(
            receipt["schema_version"],
            scripted_receipt_schema("object_pair"),
        )

    def test_expected_scripted_keys_count_one_hundred_twelve(self) -> None:
        keys = expected_scripted_check_keys(self.plan)
        self.assertEqual(len(keys), 112)

    def test_stationary_seed_order_unchanged(self) -> None:
        keys = runner.expected_scripted_seed_checks(plan=self.plan, mode="stationary")
        self.assertEqual(len(keys), 12)

    def test_horizontal_scripted_wrapper_unchanged(self) -> None:
        self.assertTrue(callable(run_scripted_horizontal_check))
        self.assertEqual(trajectory_schema("horizontal"), "v4-horizontal-g3-scripted-trajectory-v1")

    def test_render_spec_bindings(self) -> None:
        spec = json.loads(OBJECT_PAIR_SPEC.read_text(encoding="utf-8"))
        self.assertEqual(spec["fixture_id"], "object_pair")
        self.assertEqual(spec["scale"], 0.5)
        self.assertEqual(
            spec["plan_sha256"],
            "143c8d1bcec8997c1ee1c47bcba4ae8108cc0ea90295a1a230441e7337a6f9de",
        )
        self.assertEqual(
            spec["reset_registry_sha256"],
            "6272ef4de4c6188e65198cd5f1a7f35e1ccbaf3d8022d5a1b6d030cf0eb1b84c",
        )
        self.assertTrue(spec["output_parent_must_exist_on_pvc"])
        self.assertFalse(spec["path_scale_receipt_binding_only"])
        self.assertTrue(spec["path_scale_receipt_source"].endswith(".json"))
        self.assertTrue(len(spec["launch_prerequisites"]) >= 2)

    def test_render_object_pair_bundle_with_local_path_scale_receipt(self) -> None:
        scale = 0.5
        receipts = [
            compile_path_seed_receipt(
                plan=self.plan,
                plan_receipt=self.plan_receipt,
                environment_seed=int(seed),
                scale=scale,
                check_observations=_passing_path_observations(suffix=str(seed)),
                goal_area_cases=_goal_area_cases(),
            )
            for seed in self.plan["registered_env_seeds"]
        ]
        path_scale = compile_path_scale_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            scale=scale,
            path_seed_receipts=receipts,
        )
        validate_path_scale_receipt(path_scale, plan=self.plan)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            receipt_path = tmp_path / "path-scale-receipt.json"
            from experiments.online_correction_v4.model_blind_g3 import canonical_json_bytes

            receipt_path.write_bytes(canonical_json_bytes(path_scale))
            spec = json.loads(OBJECT_PAIR_SPEC.read_text(encoding="utf-8"))
            spec_dir = OBJECT_PAIR_SPEC.parent
            for key in (
                "marker_wrapper_source",
                "runner_source",
                "gate_core_source",
                "campaign_source",
                "plan_source",
                "reset_registry_source",
            ):
                value = spec.get(key)
                if isinstance(value, str) and value and not Path(value).is_absolute():
                    spec[key] = str((spec_dir / value).resolve())
            spec["authorization_status"] = "authorized_by_passing_path_scale_receipt"
            spec["path_scale_receipt_binding_only"] = False
            spec["path_scale_receipt_source"] = str(receipt_path)
            spec["path_scale_receipt_path"] = str(receipt_path)
            spec["path_scale_receipt_sha256"] = sha256_file(receipt_path)
            spec_path = tmp_path / "object-pair-scripted-spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            report = renderer.render(
                spec_path,
                tmp_path / "rendered",
            )
            self.assertEqual(report["job_count"], 10)
            self.assertEqual(report["scale"], 0.5)


if __name__ == "__main__":
    unittest.main()
