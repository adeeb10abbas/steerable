from __future__ import annotations

from types import SimpleNamespace
import unittest

from experiments.online_correction_v4.second_stack import (
    REFERENCE_OBJECT,
    RELATION_AXES_SCENE_XY,
    SOURCE_OBJECT,
    SecondStackBindingError,
    apply_registered_reset,
    direct_prompt,
    fixture_actors,
    reference_destination_xy,
    task_axes_from_camera_extrinsic,
)


class _Pose:
    def __init__(self, p, q):
        self.p = list(p)
        self.q = list(q)


class _Actor:
    def __init__(self, name, position):
        self.name = name
        self.pose = _Pose(position, [1.0, 0.0, 0.0, 0.0])
        self.velocity = None
        self.angular_velocity = None

    def set_pose(self, pose):
        self.pose = pose

    def set_velocity(self, velocity):
        self.velocity = list(velocity)

    def set_angular_velocity(self, velocity):
        self.angular_velocity = list(velocity)


class _Scene:
    def __init__(self):
        self.steps = 0
        self.render_updates = 0

    def step(self):
        self.steps += 1

    def update_render(self):
        self.render_updates += 1


def _env():
    source = _Actor(SOURCE_OBJECT, [-0.26, 0.1, -1.0514])
    reference = _Actor(REFERENCE_OBJECT, [-0.06, 0.1, -1.0514])
    scene = _Scene()
    raw = SimpleNamespace(
        episode_source_obj=source,
        episode_target_obj=reference,
        _scene=scene,
    )
    inner = SimpleNamespace(unwrapped=raw)
    wrapper = SimpleNamespace(env=inner)
    return SimpleNamespace(unwrapped=wrapper), source, reference, scene


class SecondStackBindingTests(unittest.TestCase):
    def test_direct_prompt_uses_fixed_viewpoint(self) -> None:
        self.assertEqual(
            direct_prompt("left"),
            "Place the green block so that it is left of the yellow block. "
            "Use the robot's fixed viewpoint for left, right, front, and behind.",
        )
        with self.assertRaises(SecondStackBindingError):
            direct_prompt("above")

    def test_registered_reset_preserves_live_z_and_zeros_velocity(self) -> None:
        env, source, reference, scene = _env()
        result = apply_registered_reset(
            env,
            {
                "positions_scene_xy_m": {
                    SOURCE_OBJECT: [-0.21, -0.05],
                    REFERENCE_OBJECT: [-0.11, 0.05],
                }
            },
            settle_steps=3,
        )
        self.assertEqual(source.pose.p, [-0.21, -0.05, -1.0514])
        self.assertEqual(reference.pose.p, [-0.11, 0.05, -1.0514])
        self.assertEqual(source.velocity, [0.0, 0.0, 0.0])
        self.assertEqual(reference.angular_velocity, [0.0, 0.0, 0.0])
        self.assertEqual(scene.steps, 3)
        self.assertEqual(scene.render_updates, 1)
        self.assertEqual(result[SOURCE_OBJECT], source.pose.p)

    def test_live_object_name_binding_is_fail_closed(self) -> None:
        env, source, _reference, _scene = _env()
        source.name = "green_cube_3cm"
        with self.assertRaises(SecondStackBindingError):
            fixture_actors(env)

    def test_relation_axes_follow_fixed_camera_registration(self) -> None:
        origin = (-0.11, 0.05)
        left = reference_destination_xy(
            initial_xy=origin,
            relation="left",
            displacement_m=0.04,
            physical_translation_sign=1,
        )
        front_negative = reference_destination_xy(
            initial_xy=origin,
            relation="front",
            displacement_m=0.04,
            physical_translation_sign=-1,
        )
        for actual, expected in zip(
            left,
            (
                origin[0] + 0.04 * RELATION_AXES_SCENE_XY["left"][0],
                origin[1] + 0.04 * RELATION_AXES_SCENE_XY["left"][1],
            ),
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            front_negative,
            (
                origin[0] - 0.04 * RELATION_AXES_SCENE_XY["front"][0],
                origin[1] - 0.04 * RELATION_AXES_SCENE_XY["front"][1],
            ),
        ):
            self.assertAlmostEqual(actual, expected)

    def test_live_camera_matrix_reconstructs_registered_task_axes(self) -> None:
        extrinsic = [
            [-0.4839331805706024, 0.875105619430542, 0.0, -0.09338153898715973],
            [0.6025577783584595, 0.3332144320011139, -0.725184440612793, 0.7407573461532593],
            [-0.634613037109375, -0.35094085335731506, -0.6885547637939453, 1.0061874389648438],
            [0.0, 0.0, 0.0, 1.0],
        ]
        axes = task_axes_from_camera_extrinsic(extrinsic)
        for relation in ("left", "right", "front", "behind"):
            for actual, expected in zip(
                axes[relation],
                RELATION_AXES_SCENE_XY[relation],
            ):
                self.assertAlmostEqual(actual, expected, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
