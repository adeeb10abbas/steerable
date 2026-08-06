#!/usr/bin/env python3
"""Regressions for the released V3-B001 Nano live queue boundary."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.v3.cosmos_droid.contract import compute_adapter_contract_hash
from experiments.v3.cosmos_nano_phase_b.live_support import (
    bind_live_stack_runtime,
    validate_pinned_server_cli,
    verify_live_runtime_identity,
)
from experiments.v3.cosmos_nano_phase_b.queue_launcher import (
    FROZEN_LD_LIBRARY_PATH,
    FROZEN_VK_ICD,
    _candidate_hash,
    cell_plan,
    ordered_cells,
)
from experiments.v3.cosmos_nano_phase_b.runtime_adapter import (
    CHECKPOINT_REVISION,
    MODEL_REPOSITORY,
    RuntimeContractError,
    canonical_json_bytes,
    load_release_bundle,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
RELEASE_MANIFEST = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/"
    "nano_mirror_v3b001_manifest.json"
)
RELEASE_SHA256 = "5c82268739feb41281435a51dcd848b575218cd9fbe5839d9ad130d1a7888830"
PHASE_A_RUNTIME_SHA256 = "d4bc4ab7d03fd1d1041f0bcc384d34321f3bd7b16c0c4cf517b62b8a1a2160e2"


def _real_phase_a_runtime() -> dict:
    payload = {
        "adapter_contract_hash": compute_adapter_contract_hash(ROOT),
        "checkpoint_hash_gate_passed": True,
        "checkpoint_identifier": MODEL_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": "cf76fcba7008061ecf95ec08b1b21815a6ffcb2ae9878fa11fb64a5eafb2e246",
        "environment_lock_hash": "0c7339fef2598af251b314a8d3c41f896a0f4680eb77a0baa90fd8152fd55c63",
        "external_repository_commit": "411d25b2e35bc441126f48c44a4b93e1c0564274",
        "external_repository_diff_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "model_id": "cosmos3_nano_policy_droid",
        "renderer_backend": (
            "Isaac Sim viewport RTX Vulkan; NVIDIA ICD /etc/vulkan/icd.d/nvidia_icd.json; "
            "driver 580.105.08; NVIDIA RTX PRO 6000 Blackwell Server Edition"
        ),
        "repository_pins": {
            "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_v2a011_registry.json": "4b7fc1f7a98d73b3cd2995d32b926fc2bba4175f9b03a8131d81a78873b03eba",
            "experiments/cosmos/serve_nano_robolab_v2a011.py": "4d78af3e1fb4705b40ac36a803d49756ca6c98e512861f7f8d05b58ebc04b6f4",
            "experiments/cosmos/serve_robolab_without_guardrails.py": "02bc8836bd2a2ec009287487ee03e8bb810da0c1e07c94794faed84d3dc8f93b",
            "experiments/cosmos/v2_robolab_client.py": "c9936139ee6192f6647db16a8a58a3080c5d3c5ceb64c702286c85dea2009afa",
            "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_left_of_bowl_matched.py": "9c4d90be770266bac3ba5242b743098348c565ee622179b5e88fa2af0c4891bc",
            "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_right_of_bowl_matched.py": "ba0eb879590960c57976dd1b749c4ebbd3e86054e152ca5af014ac1bc2b6d02a",
        },
        "schema_version": "vla-wam-shared-v3-cosmos-runtime-identity-v1",
        "simulator_repository_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
        "simulator_repository_diff_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "simulator_version": "Isaac Sim 5.0.0.0 / Isaac Lab 2.2.0 / RoboLab 0.2.1",
        "study_id": "vla_wam_language_steerability_v3",
    }
    payload["runtime_identity_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    if payload["runtime_identity_sha256"] != PHASE_A_RUNTIME_SHA256:
        raise AssertionError("real Phase-A runtime fixture drifted")
    return payload


def _server_argv() -> list[str]:
    return [
        "--checkpoint-path", "/data/users/ali/vla_wam/checkpoints/cosmos3_nano_policy_droid",
        "--hf-revision", CHECKPOINT_REVISION,
        "--host", "0.0.0.0", "--port", "18011",
        "--domain-name", "droid_lerobot", "--decode-video",
        "--action-chunk-size", "32", "--action-dim", "8",
        "--action-space", "joint_pos", "--history-length", "1", "--use-state",
        "--conditioning-fps", "15", "--resolution", "480",
        "--guidance", "3", "--num-steps", "4", "--shift", "5",
    ]


class NanoPhaseBLiveQueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.release = load_release_bundle(
            RELEASE_MANIFEST, expected_manifest_sha256=RELEASE_SHA256
        )

    def test_real_phase_a_runtime_maps_to_new_live_bound_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "phase_a_runtime_identity.json"
            base.write_text(json.dumps(_real_phase_a_runtime(), indent=2, sort_keys=True) + "\n")
            live = bind_live_stack_runtime(
                study_root=ROOT,
                release=self.release,
                base_runtime_manifest=base,
            )
            self.assertEqual(live["environment_lock_sha256"], _real_phase_a_runtime()["environment_lock_hash"])
            self.assertEqual(live["phase_a_runtime_identity_sha256"], PHASE_A_RUNTIME_SHA256)
            output = root / "phase_b_live_runtime_identity.json"
            output.write_text(json.dumps(live, indent=2, sort_keys=True) + "\n")
            self.assertEqual(
                verify_live_runtime_identity(output, study_root=ROOT, release=self.release),
                live,
            )

    def test_server_cli_is_exact_and_wrong_guidance_is_rejected(self) -> None:
        self.assertEqual(validate_pinned_server_cli(_server_argv())["guidance"], 3.0)
        changed = _server_argv()
        changed[changed.index("--guidance") + 1] = "1"
        with self.assertRaisesRegex(RuntimeContractError, "guidance"):
            validate_pinned_server_cli(changed)

    def test_first_released_smoke_plan_has_exact_sim_and_eula_contract(self) -> None:
        cells = ordered_cells(self.release)
        self.assertEqual(cells[0].cell_id, "v3b001:nano:seed9400:position_mirrored:right")
        with tempfile.TemporaryDirectory() as directory:
            plan = cell_plan(
                study_root=ROOT,
                release_manifest=RELEASE_MANIFEST,
                release_manifest_sha256=RELEASE_SHA256,
                runtime_manifest=Path(directory) / "runtime.json",
                fixture_candidate=Path(directory) / "fixture.json",
                fixture_candidate_sha256=_candidate_hash(self.release),
                raw_root=Path(directory) / "raw",
                cell=cells[0],
                remote_host="10.0.0.1",
                remote_port=18011,
            )
        command = plan["bridge_command"]
        for flag, value in (
            ("--renderer", "realtime"),
            ("--rendering-type", "balanced"),
            ("--device", "cuda:0"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--headless", command)
        self.assertEqual(plan["environment"]["OMNI_KIT_ACCEPT_EULA"], "YES")
        self.assertEqual(plan["environment"]["VK_ICD_FILENAMES"], FROZEN_VK_ICD)
        self.assertEqual(plan["environment"]["LD_LIBRARY_PATH"], FROZEN_LD_LIBRARY_PATH)
        self.assertEqual(plan["thermal_guard"], "not_used")

    def test_bridge_does_not_redeclare_robolab_output_dir_argument(self) -> None:
        source = (
            ROOT / "experiments/v3/cosmos_nano_phase_b/robolab_bridge.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('BOOTSTRAP.add_argument("--output-dir"', source)
        self.assertIn("add_common_eval_args(parser)", source)


if __name__ == "__main__":
    unittest.main()
