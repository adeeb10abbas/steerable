"""Tests for fixture-parameterized V4 qualification metadata."""

from __future__ import annotations

import unittest

from experiments.online_correction_v4.fixture_qualification import (
    qualification_profile,
)
from experiments.online_correction_v4.geometry import build_prompt


class FixtureQualificationProfileTests(unittest.TestCase):
    def test_object_pair_profile_matches_legacy_prompts(self) -> None:
        profile = qualification_profile("object_pair")
        prompts = profile.g4_prompt_pair
        self.assertEqual(
            prompts.primary_prompt,
            build_prompt("sponge", "tray", "left", "direct"),
        )
        self.assertEqual(
            prompts.alternate_prompt,
            build_prompt("sponge", "tray", "right", "direct"),
        )
        self.assertEqual(profile.g3_basis, "scripted_aggregate")

    def test_containment_profile_uses_inside_prompts(self) -> None:
        profile = qualification_profile("containment")
        prompts = profile.g4_prompt_pair
        self.assertEqual(
            prompts.primary_prompt,
            build_prompt("cube", "bowl", "inside", "direct", horizontal=False),
        )
        self.assertEqual(
            prompts.alternate_prompt,
            build_prompt("cube", "bowl", "inside", "inverse", horizontal=False),
        )
        self.assertEqual(profile.g3_basis, "path_scale")
        self.assertEqual(profile.family_id, "C6")


if __name__ == "__main__":
    unittest.main()
