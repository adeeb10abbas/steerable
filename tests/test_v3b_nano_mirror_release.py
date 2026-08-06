#!/usr/bin/env python3
"""Tests for the fail-closed V3-B001 Nano position-reflection release."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from experiments.v3.cosmos_nano_phase_b import design


ROOT = Path(__file__).resolve().parents[1]


def file_record(path: Path, **extra: object) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        **extra,
    }


def write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def candidate(root: Path) -> tuple[dict, Path]:
    scene_metadata = write(root / "scene_metadata.json", b"exact scene metadata fixture\n")
    quaternions = {
        name: [1.0, 0.0, 0.0, 0.0] for name in design.CONTROL_POSITIONS_WORLD_XYZ
    }
    control = design.control_positions()
    mirrored = design.mirrored_positions()
    delta = [
        control["rubiks_cube"][index] - control["bowl"][index]
        for index in range(2)
    ]
    mirror_delta = [
        mirrored["rubiks_cube"][index] - mirrored["bowl"][index]
        for index in range(2)
    ]
    value = {
        "schema_version": design.CANDIDATE_SCHEMA,
        "study_id": design.STUDY_ID,
        "model_id": design.MODEL_ID,
        "phase": design.PHASE,
        "status": "model_blind_candidate_not_released_for_inference",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "exact_prompts": design.PROMPTS,
        "source_identity": {
            "robolab_commit": design.ROBOLAB_REPOSITORY_COMMIT,
            "scene": design.SCENE_ASSET,
            "scene_metadata": file_record(scene_metadata),
            "neutral_left_task": file_record(
                ROOT
                / "experiments/groot_droid/robolab_v2_tasks/"
                "rubiks_cube_left_of_bowl_matched.py"
            ),
            "neutral_right_task": file_record(
                ROOT
                / "experiments/groot_droid/robolab_v2_tasks/"
                "rubiks_cube_right_of_bowl_matched.py"
            ),
        },
        "factor": {
            "name": "movable_object_center_position_reflection_about_robot_sagittal_plane",
            "transform": "for rubiks_cube, bowl, and banana only: (x,y,z) -> (x,-y,z)",
            "robot_base_plane_y_m": 0.0,
            "changed": "initial center positions of the three movable objects",
            "held_fixed": [
                "object identities",
                "object quaternions",
                "nonmovable scene geometry",
                "robot base and controller",
                "camera poses",
                "prompt bytes and scorer axes",
            ],
            "claim_boundary": (
                "This is a position-mirrored movable-object layout, not a full geometric "
                "reflection; an improper reflection is not represented by a quaternion."
            ),
        },
        "layouts": {
            "control": {
                "positions_robot_base_m": control,
                "quaternions_wxyz_unchanged": quaternions,
            },
            "position_mirrored": {
                "positions_robot_base_m": mirrored,
                "quaternions_wxyz_unchanged": quaternions,
            },
        },
        "analytic_neutrality_precheck": {
            "control_cube_minus_bowl_xy_m": delta,
            "position_mirrored_cube_minus_bowl_xy_m": mirror_delta,
            "outside_left_and_right_45deg_cones": True,
            "live_simulator_reset_check_still_required": True,
        },
        "release_boundary": (
            "Candidate coordinates are not an inference release. A live model-blind reset, "
            "RTX renderer, fixture-settle, and raw-writer calibration report must pass and be "
            "hash-bound by a new amendment before any Nano model request."
        ),
    }
    path = root / "candidate.json"
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")
    return value, path


def live_reset(layout: dict, repeat: int) -> dict:
    return {
        "repeat": repeat,
        "positions_robot_base_m": copy.deepcopy(layout["positions_robot_base_m"]),
        "quaternions_wxyz": copy.deepcopy(layout["quaternions_wxyz_unchanged"]),
        "velocities": {
            name: [0.0] * 6 for name in design.CONTROL_POSITIONS_WORLD_XYZ
        },
        "stability_window": {
            name: {
                "max_linear_speed_m_s": 0.019,
                "max_angular_speed_rad_s": 0.19,
            }
            for name in design.CONTROL_POSITIONS_WORLD_XYZ
        },
        "left_predicate_at_reset": False,
        "right_predicate_at_reset": False,
        "input_views": {
            name: {"shape": [96, 128, 3], "dtype": "uint8", "pixel_range": 255}
            for name in (
                "over_shoulder_left_camera",
                "over_shoulder_right_camera",
                "head_camera",
                "wrist_cam",
            )
        },
    }


def quaternion_diagnostics(tasks: list[dict]) -> list[dict]:
    by_label = {task["label"]: task for task in tasks}
    controls = by_label["control_left"]["repeat_resets"]
    mirrored = by_label["position_mirrored_left"]["repeat_resets"]
    output = []
    for repeat, (control, mirror) in enumerate(zip(controls, mirrored)):
        objects = {}
        for name in design.CONTROL_POSITIONS_WORLD_XYZ:
            left = control["quaternions_wxyz"][name]
            right = mirror["quaternions_wxyz"][name]
            dot = min(1.0, abs(sum(a * b for a, b in zip(left, right))))
            objects[name] = {
                "max_abs_component_difference": max(
                    abs(a - b) for a, b in zip(left, right)
                ),
                "absolute_quaternion_dot": dot,
                "angular_distance_rad": 2.0 * math.acos(dot),
            }
        output.append({"repeat": repeat, "objects": objects})
    return output


def valid_report(root: Path) -> tuple[dict, Path]:
    candidate_value, candidate_path = candidate(root)
    robolab_import = write(root / "robolab_init.py", b"# exact effective import\n")
    vulkan_icd = write(root / "nvidia_icd.json", b'{"library_path":"libGLX_nvidia.so"}\n')
    videos = {}
    for label in design.TASK_LABELS:
        video = write(root / f"{label}.mp4", f"persisted {label} frames".encode())
        videos[label] = file_record(video, decoded_frame_count=3)
    tasks = []
    for label in design.TASK_LABELS:
        arm, relation = label.rsplit("_", 1)
        tasks.append(
            {
                "label": label,
                "arm": arm,
                "relation": relation,
                "task_name": (
                    "V3BNano"
                    + "".join(part.title() for part in label.split("_"))
                    + "CalibrationTask"
                ),
                "prompt": design.PROMPTS[relation],
                "repeat_resets": [
                    live_reset(candidate_value["layouts"][arm], repeat)
                    for repeat in range(3)
                ],
            }
        )
    report = {
        "schema_version": design.CALIBRATION_SCHEMA,
        "study_id": design.STUDY_ID,
        "model_id": design.MODEL_ID,
        "phase": design.PHASE,
        "status": "complete_model_blind_calibration_not_yet_released",
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "environment_seed": 9400,
        "pod": "raytrace-rtxpro6000-ali",
        "pod_uid": "test-pod-uid",
        "gpu_uuid": "GPU-test-uuid",
        "gpu_query": "0, GPU-test-uuid, RTX PRO 6000, test-driver",
        "candidate": file_record(candidate_path),
        "calibration_driver_source": file_record(
            ROOT / "experiments/v3/cosmos_nano_phase_b/model_blind_fixture_gate.py"
        ),
        "factor_task_source": file_record(
            ROOT / "experiments/v3/cosmos_nano_phase_b/fixture_tasks.py"
        ),
        "factor_task_wrappers": {
            label: file_record(ROOT / relative)
            for label, relative in design.TASK_WRAPPER_PATHS.items()
        },
        "robolab": {
            "commit": design.ROBOLAB_REPOSITORY_COMMIT,
            "tracked_diff_empty": True,
            "effective_import": file_record(robolab_import),
            "versions": {
                "isaacsim": "test-isaacsim",
                "isaaclab": "test-isaaclab",
                "robolab": "test-robolab",
            },
        },
        "study_checkout": {
            "commit": design.CALIBRATION_STUDY_COMMIT,
            "tracked_diff_empty": True,
        },
        "renderer": {
            "backend": "realtime RTX Vulkan",
            "quality": "balanced",
            "nvidia_icd": file_record(vulkan_icd),
            "all_required_rgb_views_nonblank": True,
        },
        "reset_gate": {
            "repeat_count_per_task": 3,
            "settle_steps": 60,
            "settle_steps_basis": (
                "The model-blind 60-step probe reduced movable-object translation below "
                "0.004 m/s while preserving a neutral reset; a longer 180-step probe was "
                "rejected after free settling crossed a task termination boundary"
            ),
            "stable_window_steps": 15,
            "position_tolerance_m": 0.003,
            "linear_speed_tolerance_m_s": 0.02,
            "angular_speed_tolerance_rad_s": 0.20,
            "angular_speed_tolerance_basis": (
                "0.02 m/s linear tolerance divided by a conservative 0.10 m object "
                "radius gives 0.20 rad/s, bounding rotational surface speed at the "
                "same scale as translation"
            ),
            "left_right_physical_fingerprints_equal_within_each_arm": True,
            "neither_predicate_true_at_every_reset": True,
            "live_position_reflection_passed_at_every_repeat": True,
            "initial_quaternion_sources_identical_across_layouts": True,
            "post_settle_quaternion_differences_recorded_not_gated": True,
        },
        "post_settle_cross_layout_quaternion_differences": quaternion_diagnostics(tasks),
        "tasks": tasks,
        "viewport_write_gate": videos,
        "claim_boundary": (
            "Model-blind calibration of a positions-only movable-object reflection. "
            "Initial quaternion sources are identical, while any recorded post-settle "
            "orientation difference is a downstream physical consequence of the position "
            "intervention. It is not behavioral evidence, a full scene mirror, or a "
            "reachability claim."
        ),
    }
    report_path = root / "model_blind_calibration_report.json"
    report_path.write_text(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")
    return report, report_path


class NanoMirrorReleaseTest(unittest.TestCase):
    def build(
        self,
        mutate: Callable[[dict, Path], None] | None = None,
    ) -> design.ReleasePayloads:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        report, report_path = valid_report(root)
        if mutate is not None:
            mutate(report, root)
            report_path.write_text(
                json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
            )
        return design.build_release(
            ROOT,
            report_path,
            recorded_at_utc="2026-08-06T01:00:00Z",
        )

    def test_exact_108_cell_matched_design_and_prompts(self) -> None:
        release = self.build()
        self.assertEqual(len(release.rows), 108)
        self.assertEqual(
            Counter(row["environment_seed"] for row in release.rows),
            {seed: 4 for seed in design.SEEDS},
        )
        by_seed = defaultdict(list)
        for row in release.rows:
            by_seed[row["environment_seed"]].append(row)
            self.assertEqual(row["environment_seed"], row["sampling_seed"])
            self.assertEqual(row["prompt"], design.PROMPTS[row["relation"]])
            self.assertEqual(row["factor"], design.FACTOR_NAME)
            self.assertEqual(
                row["amendment_sha256"],
                hashlib.sha256(release.amendment_bytes).hexdigest(),
            )
        for seed, rows in by_seed.items():
            self.assertEqual(
                {(row["arm"], row["relation"]) for row in rows},
                {(arm, relation) for arm in design.ARMS for relation in design.RELATIONS},
                seed,
            )
            self.assertEqual(
                {row["execution_order_index_within_seed"] for row in rows},
                {1, 2, 3, 4},
            )
        self.assertEqual(release.manifest["counts"]["control_cells"], 54)
        self.assertEqual(release.manifest["counts"]["position_mirrored_cells"], 54)
        self.assertEqual(set(design.ARMS), {"control", "position_mirrored"})

    def test_randomization_is_deterministic_and_position_balanced(self) -> None:
        first = design.randomized_orders()
        self.assertEqual(first, design.randomized_orders())
        counts = Counter()
        for order in first.values():
            for position, condition in enumerate(order, 1):
                counts[(position, condition)] += 1
        self.assertEqual(set(counts.values()), {6, 7})

    def test_analysis_contract_uses_full_sample_offset_and_named_success_subset(self) -> None:
        plan = self.build().amendment["analysis_plan"]
        primary = plan["full_sample_primary"]
        conditional = plan["success_conditional_secondary"]
        self.assertEqual(
            primary["per_arm_steering_separation"],
            "D[a,i] = s[a,i,left] - s[a,i,right]",
        )
        self.assertEqual(
            primary["directional_bias_contrast"],
            "B[a,i] = (-s[a,i,right]) - s[a,i,left]",
        )
        self.assertEqual(
            primary["position_reflection_interaction"],
            "I[i] = B[position_mirrored,i] - B[control,i]",
        )
        self.assertEqual(
            conditional["complete_case_subset_id"],
            "nano_v3b001_all_four_cells_correct",
        )

    def test_rejects_any_model_request_or_behavioral_episode(self) -> None:
        for field in ("model_request_count", "behavioral_episode_count"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(design.ReleaseError, field):
                    self.build(lambda report, _: report.__setitem__(field, 1))

    def test_rejects_dirty_or_wrong_source_identity(self) -> None:
        with self.assertRaisesRegex(design.ReleaseError, "study checkout"):
            self.build(
                lambda report, _: report["study_checkout"].__setitem__(
                    "tracked_diff_empty", False
                )
            )
        with self.assertRaisesRegex(design.ReleaseError, "RoboLab"):
            self.build(
                lambda report, _: report["robolab"].__setitem__(
                    "commit", "f" * 40
                )
            )

    def test_rejects_inexact_mirror_or_quaternion_change(self) -> None:
        def move_bowl(report: dict, _: Path) -> None:
            candidate_path = Path(report["candidate"]["path"])
            value = json.loads(candidate_path.read_text())
            value["layouts"]["position_mirrored"]["positions_robot_base_m"]["bowl"][0] += 0.001
            candidate_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            report["candidate"] = file_record(candidate_path)

        with self.assertRaisesRegex(design.ReleaseError, "bowl position changed"):
            self.build(move_bowl)

    def test_post_settle_orientation_is_recorded_as_a_mediator(self) -> None:
        settled = [0.99995, 0.0, 0.0, 0.01]

        def move_all_settled_quaternions(report: dict, _: Path) -> None:
            for task in report["tasks"]:
                for reset in task["repeat_resets"]:
                    reset["quaternions_wxyz"]["rubiks_cube"] = settled
            report["post_settle_cross_layout_quaternion_differences"] = (
                quaternion_diagnostics(report["tasks"])
            )

        release = self.build(move_all_settled_quaternions)
        self.assertEqual(len(release.rows), 108)

        def add_mirror_orientation_mediator(report: dict, _: Path) -> None:
            for task in report["tasks"]:
                if task["arm"] == "position_mirrored":
                    task["repeat_resets"][0]["quaternions_wxyz"]["rubiks_cube"] = [
                        0.9998,
                        0.0,
                        0.0,
                        0.02,
                    ]
            report["post_settle_cross_layout_quaternion_differences"] = (
                quaternion_diagnostics(report["tasks"])
            )

        mediated = self.build(add_mirror_orientation_mediator)
        diagnostic = mediated.amendment["calibration_evidence"][
            "post_settle_cross_layout_quaternion_differences"
        ][0]["objects"]["rubiks_cube"]
        self.assertGreater(diagnostic["angular_distance_rad"], 0.0)

        def falsify_diagnostic(report: dict, _: Path) -> None:
            report["post_settle_cross_layout_quaternion_differences"][0]["objects"][
                "rubiks_cube"
            ]["angular_distance_rad"] = 0.5

        with self.assertRaisesRegex(design.ReleaseError, "diagnostic mismatch"):
            self.build(falsify_diagnostic)

    def test_accepts_one_ulp_dot_roundoff_but_not_inconsistent_angle(self) -> None:
        def one_ulp_roundoff(report: dict, _: Path) -> None:
            recorded = report["post_settle_cross_layout_quaternion_differences"][0][
                "objects"
            ]["banana"]
            recorded_dot = math.nextafter(1.0, 0.0)
            recorded["absolute_quaternion_dot"] = recorded_dot
            recorded["angular_distance_rad"] = 2.0 * math.acos(recorded_dot)

        release = self.build(one_ulp_roundoff)
        self.assertEqual(len(release.rows), 108)

        def inconsistent_angle(report: dict, root: Path) -> None:
            one_ulp_roundoff(report, root)
            report["post_settle_cross_layout_quaternion_differences"][0]["objects"][
                "banana"
            ]["angular_distance_rad"] += 1e-8

        with self.assertRaisesRegex(design.ReleaseError, "angular_distance_rad"):
            self.build(inconsistent_angle)

    def test_rejects_failed_neutral_or_sustained_stability_gate(self) -> None:
        with self.assertRaisesRegex(design.ReleaseError, "starts LEFT"):
            self.build(
                lambda report, _: report["tasks"][0]["repeat_resets"][0].__setitem__(
                    "left_predicate_at_reset", True
                )
            )
        with self.assertRaisesRegex(design.ReleaseError, "sustained angular"):
            self.build(
                lambda report, _: report["tasks"][0]["repeat_resets"][0][
                    "stability_window"
                ]["rubiks_cube"].__setitem__("max_angular_speed_rad_s", 0.201)
            )

    def test_rejects_renderer_or_persisted_writer_tamper(self) -> None:
        with self.assertRaisesRegex(design.ReleaseError, "renderer gate"):
            self.build(
                lambda report, _: report["renderer"].__setitem__(
                    "all_required_rgb_views_nonblank", False
                )
            )

        def corrupt_video(report: dict, _: Path) -> None:
            Path(report["viewport_write_gate"]["control_left"]["path"]).write_bytes(
                b"corrupted after calibration"
            )

        with self.assertRaisesRegex(design.ReleaseError, "byte count changed|changed after calibration"):
            self.build(corrupt_video)

    def test_invalid_calibration_creates_no_release_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, report_path = valid_report(root)
            report["passed"] = False
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            output = root / "release"
            with self.assertRaises(design.ReleaseError):
                payloads = design.build_release(
                    ROOT,
                    report_path,
                    recorded_at_utc="2026-08-06T01:00:00Z",
                )
                design.write_release(output, payloads)
            self.assertFalse(output.exists())

    def test_atomic_write_refuses_overwrite(self) -> None:
        release = self.build()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "release"
            paths = design.write_release(output, release)
            self.assertEqual(paths["cells"].read_bytes(), release.cells_bytes)
            with self.assertRaisesRegex(design.ReleaseError, "refusing to overwrite"):
                design.write_release(output, release)


if __name__ == "__main__":
    unittest.main()
