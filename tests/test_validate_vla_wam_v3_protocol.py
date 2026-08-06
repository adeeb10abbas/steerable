"""Tests for the VLA/WAM v3 fail-closed protocol validator."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools" / "validate_vla_wam_v3_protocol.py"
SPEC = importlib.util.spec_from_file_location("validate_v3", VALIDATOR_PATH)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class ValidateV3ProtocolTest(unittest.TestCase):
    def copy_protocol_root(self, root: Path) -> None:
        shutil.copytree(ROOT / "artifacts" / "vla_wam_shared_v3", root / "artifacts" / "vla_wam_shared_v3")
        measurement_audit = json.loads(
            (ROOT / "artifacts" / "vla_wam_shared_v3" / "measurement_coverage_audit.json").read_text()
        )
        for cohort in measurement_audit["cohorts"]:
            for source in cohort["sources"]:
                source_path = ROOT / source["path"]
                destination = root / source["path"]
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
        (root / "docs").mkdir()
        for name in (
            "VLA_WAM_STEERABILITY_V3_PROTOCOL.md",
            "VLA_WAM_V3_CONTINUATION.md",
            "WORK_LAPTOP_B200_HANDOFF.md",
        ):
            shutil.copy2(ROOT / "docs" / name, root / "docs")
        shutil.copytree(
            ROOT / "experiments" / "v3" / "pi0_fast_old_name_config_bridge",
            root / "experiments" / "v3" / "pi0_fast_old_name_config_bridge",
        )
        shutil.copytree(
            ROOT / "experiments" / "v3" / "cosmos_nano_phase_b",
            root / "experiments" / "v3" / "cosmos_nano_phase_b",
        )
        shutil.copytree(
            ROOT / "experiments" / "v3" / "pi05_phase_b",
            root / "experiments" / "v3" / "pi05_phase_b",
        )
        shutil.copytree(
            ROOT / "experiments" / "v3" / "cosmos_nano_lateral_sweep",
            root / "experiments" / "v3" / "cosmos_nano_lateral_sweep",
        )
        shutil.copytree(
            ROOT / "experiments" / "v3" / "dreamzero_phase_b",
            root / "experiments" / "v3" / "dreamzero_phase_b",
        )
        (root / "experiments" / "groot_droid" / "robolab_v2_tasks").mkdir(
            parents=True
        )
        for name in (
            "rubiks_cube_left_of_bowl_matched.py",
            "rubiks_cube_right_of_bowl_matched.py",
        ):
            shutil.copy2(
                ROOT / "experiments" / "groot_droid" / "robolab_v2_tasks" / name,
                root / "experiments" / "groot_droid" / "robolab_v2_tasks" / name,
            )
        (root / "tools").mkdir()
        for name in (
            "build_v3a002_pi0_fast_media.py",
            "compile_nano_v3b001_results.py",
            "finalize_nano_v3b001_evidence.py",
            "render_nano_v3b001_results.py",
            "build_nano_v3b001_publication_media.py",
            "build_pi05_v3b002_registration.py",
            "build_dreamzero_v3b003_registration.py",
            "build_dreamzero_v3b003_release_gate.py",
            "build_nano_v3b005_queue.py",
            "analyze_v3_failure_mode_split.py",
        ):
            shutil.copy2(ROOT / "tools" / name, root / "tools" / name)
        (root / "tests").mkdir()
        for name in (
            "test_v3b_nano_runtime_adapter.py",
            "test_v3b_nano_live_queue.py",
            "test_compile_nano_v3b001_results.py",
            "test_finalize_nano_v3b001_evidence.py",
            "test_render_nano_v3b001_results.py",
            "test_build_nano_v3b001_publication_media.py",
            "test_build_pi05_v3b002_registration.py",
            "test_build_dreamzero_v3b003_registration.py",
            "test_pi05_v3b002_runtime.py",
            "test_pi05_v3b002_compiler.py",
            "test_dreamzero_v3b003_runtime.py",
            "test_build_nano_v3b005_queue.py",
        ):
            shutil.copy2(ROOT / "tests" / name, root / "tests" / name)

    def test_checked_in_protocol_passes(self) -> None:
        checks = VALIDATOR.validate(ROOT)
        self.assertGreaterEqual(len(checks), 250)

    def test_tampered_sampling_seed_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            path = root / "artifacts" / "vla_wam_shared_v3" / "stochastic_rollout_registry.json"
            value = json.loads(path.read_text())
            value["shared_sampling_seed_indices"][15] = 9999
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "sixteen shared"):
                VALIDATOR.validate(root)

    def test_tampered_phase_a_queue_fails_hash_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            path = root / "artifacts" / "vla_wam_shared_v3" / "phase_a_cells.jsonl"
            path.write_text(path.read_text().replace('"status":"authorized_new"', '"status":"tampered"', 1))
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "queue manifest hash"):
                VALIDATOR.validate(root)

    def test_tampered_detached_release_contract_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            path = root / "artifacts" / "vla_wam_shared_v3" / "failure_taxonomy.json"
            value = json.loads(path.read_text())
            value["scorer_consistency_rules"]["prohibited_inference"] = "Use requested_success as detached release."
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "detached release"):
                VALIDATOR.validate(root)

    def test_tampered_bridge_historical_pooling_flag_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            path = root / "artifacts" / "vla_wam_shared_v3" / "results" / "pi0_fast_old_name_config_v3a002_summary.json"
            value = json.loads(path.read_text())
            value["historical_pooling_prohibited"] = False
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "separate nonhistorical denominator"):
                VALIDATOR.validate(root)

    def test_tampered_bridge_seed_accounting_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            path = root / "artifacts" / "vla_wam_shared_v3" / "results" / "pi0_fast_old_name_config_v3a002_summary.json"
            value = json.loads(path.read_text())
            value["pairs"][1]["seed"] = 8310
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "each exact seed 8310-8329 once"):
                VALIDATOR.validate(root)

    def test_tampered_bridge_endpoint_sign_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            path = root / "artifacts" / "vla_wam_shared_v3" / "results" / "pi0_fast_old_name_config_v3a002_summary.json"
            value = json.loads(path.read_text())
            value["pairs"][0]["endpoint_ordering"] = "aligned"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "strict RIGHT-minus-LEFT sign rule"):
                VALIDATOR.validate(root)

    def test_tampered_bridge_summary_manifest_digest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            path = root / "artifacts" / "vla_wam_shared_v3" / "results" / "pi0_fast_old_name_config_v3a002_evidence_hash_manifest.json"
            value = json.loads(path.read_text())
            value["summary"]["sha256"] = "0" * 64
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "summary digest"):
                VALIDATOR.validate(root)

    def test_tampered_bridge_raw_read_only_flag_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            path = root / "artifacts" / "vla_wam_shared_v3" / "results" / "pi0_fast_old_name_config_v3a002_evidence_hash_manifest.json"
            value = json.loads(path.read_text())
            value["raw_inputs_read_only"] = False
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "read-only and prohibits historical pooling"):
                VALIDATOR.validate(root)

    def test_tampered_trace_amendment_scope_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            path = root / "artifacts" / "vla_wam_shared_v3" / "post_result_pi0_fast_token_trace_validation_amendment.json"
            value = json.loads(path.read_text())
            value["replacement_validation"]["only_removed_requirement"] = "Remove every token validation rule."
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "removes only within-episode"):
                VALIDATOR.validate(root)

    def test_tampered_continuation_bridge_digest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            path = root / "artifacts" / "vla_wam_shared_v3" / "continuation_state.json"
            value = json.loads(path.read_text())
            bridge = value["phase_a_results"]["droid_robolab"]["post_result_bridge_cohorts"]["pi0_fast_old_name_config_v3a002"]
            bridge["summary"]["sha256"] = "0" * 64
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "binds bridge result paths"):
                VALIDATOR.validate(root)

    def test_tampered_nano_mirror_queue_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            path = (
                root
                / "artifacts"
                / "vla_wam_shared_v3"
                / "phase_b"
                / "nano_mirror_v3b001"
                / "nano_mirror_v3b001_cells.jsonl"
            )
            rows = path.read_text().splitlines()
            first = json.loads(rows[0])
            first["prompt"] = "Put the Rubik's cube somewhere."
            rows[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(rows) + "\n")
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError,
                "manifest hash-binds calibration, amendment, and exact cell queue",
            ):
                VALIDATOR.validate(root)

    def test_tampered_bridge_publication_video_digest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            path = root / "artifacts" / "vla_wam_shared_v3" / "media" / "pi0_fast_old_name_config_v3a002" / "media_manifest.json"
            value = json.loads(path.read_text())
            value["publication_video"]["sha256"] = "0" * 64
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "H.264 video"):
                VALIDATOR.validate(root)

    def test_tampered_pi05_b002_result_fails_output_manifest_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            path = (
                root
                / "artifacts"
                / "vla_wam_shared_v3"
                / "phase_b"
                / "pi05_mirror_v3b002"
                / "results"
                / "pi05_v3b002_report.json"
            )
            value = json.loads(path.read_text())
            value["analysis"]["H3_binary_success"]["mean_DiD"] = 0
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError,
                "output manifest byte/hash-binds every copied result",
            ):
                VALIDATOR.validate(root)

    def test_tampered_pi05_b002_h3_fails_recomputation_even_if_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            results = (
                root
                / "artifacts"
                / "vla_wam_shared_v3"
                / "phase_b"
                / "pi05_mirror_v3b002"
                / "results"
            )
            report_path = results / "pi05_v3b002_report.json"
            report = json.loads(report_path.read_text())
            report["analysis"]["H3_binary_success"]["mean_DiD"] = 0
            report_path.write_text(json.dumps(report))
            output_path = results / "pi05_v3b002_output_manifest.json"
            output = json.loads(output_path.read_text())
            output["files"]["report"]["bytes"] = report_path.stat().st_size
            output["files"]["report"]["sha256"] = hashlib.sha256(
                report_path.read_bytes()
            ).hexdigest()
            output_path.write_text(json.dumps(output))
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError,
                "H3 binds four n=27 cells",
            ):
                VALIDATOR.validate(root)


if __name__ == "__main__":
    unittest.main()
