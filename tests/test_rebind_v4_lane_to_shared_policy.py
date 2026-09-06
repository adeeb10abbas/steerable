from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.rebind_v4_lane_to_shared_policy import (
    bind_shared_policy,
    canonical_json,
    sha256_bytes,
    yaml_block,
)


def _configmap(simulator: dict) -> str:
    return (
        "apiVersion: v1\nkind: ConfigMap\ndata:\n"
        "  simulator-launch.json: |\n"
        + "".join(
            f"    {line}"
            for line in canonical_json(simulator).splitlines(keepends=True)
        )
    )


class SharedPolicyBindingTests(unittest.TestCase):
    def test_rebinds_host_identity_and_launch_hash(self) -> None:
        donor = {
            "policy_wait": {
                "host": "donor-service",
                "port": 18127,
                "service_identity": {"immutable_identity_sha256": "a" * 64},
            }
        }
        target = {
            "policy_wait": {
                "host": "target-service",
                "port": 18127,
                "service_identity": {"immutable_identity_sha256": "b" * 64},
            }
        }
        old_sha = sha256_bytes(canonical_json(target).encode())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            donor_path = root / "donor.yaml"
            target_path = root / "target.yaml"
            job_path = root / "job.yaml"
            scripts_path = root / "scripts.yaml"
            donor_path.write_text(_configmap(donor))
            target_path.write_text(_configmap(target))
            job_path.write_text(f'launch:\n  value: "{old_sha}"\n')
            scripts_path.write_text("kind: ConfigMap\n")

            receipt = bind_shared_policy(
                donor_configmap=donor_path,
                target_configmap=target_path,
                target_simulator_job=job_path,
                target_scripts_configmap=scripts_path,
                output_dir=root / "out",
            )

            rebound, _start, _end = yaml_block(
                (root / "out/configmap.yaml").read_text(),
                "simulator-launch.json",
            )
            self.assertEqual(
                rebound["policy_wait"]["host"],
                "donor-service",
            )
            self.assertEqual(
                rebound["policy_wait"]["service_identity"],
                donor["policy_wait"]["service_identity"],
            )
            job = (root / "out/simulator-job.yaml").read_text()
            self.assertIn(
                receipt["target_simulator_launch_sha256_after"],
                job,
            )
            json.loads((root / "out/shared-policy-binding.json").read_text())


if __name__ == "__main__":
    unittest.main()
