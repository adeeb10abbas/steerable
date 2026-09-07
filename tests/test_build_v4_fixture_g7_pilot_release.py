from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.build_v4_fixture_g7_pilot_release import (
    DEFAULT_POLICY_CHECKPOINT_REGISTRY,
    validate_policy_checkpoint_registry,
)


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "artifacts/online_correction_v4/setup"


class FixtureG7PilotReleaseTests(unittest.TestCase):
    def test_accepts_model_checkpoint_registry(self) -> None:
        validate_policy_checkpoint_registry(DEFAULT_POLICY_CHECKPOINT_REGISTRY)

    def test_rejects_g4_nano_seed_registry_as_checkpoint(self) -> None:
        seed_registry = SETUP / "containment_g4_nano_seed_registry.candidate.json"
        with self.assertRaisesRegex(ValueError, "nano policy seed registry"):
            validate_policy_checkpoint_registry(seed_registry)

    def test_rejects_g7_nano_seed_registry_as_checkpoint(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write(
                '{"schema_version":"v4-nano-policy-seed-registry-v1",'
                '"allowed_sampling_seeds":[1]}'
            )
            path = Path(handle.name)
        try:
            with self.assertRaisesRegex(ValueError, "nano policy seed registry"):
                validate_policy_checkpoint_registry(path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
