#!/usr/bin/env python3
"""Synthetic tests for the fail-closed Nano V3-B001 aggregate compiler."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.v3.cosmos_nano_phase_b.runtime_adapter import (
    CHECKPOINT_REVISION,
    MODEL_ID,
    MODEL_REPOSITORY,
    load_release_bundle,
    sha256_file,
)
from tools.compile_nano_v3b001_results import (
    AggregateCompilationError,
    compile_nano_v3b001_results,
)
from tools.vla_wam_v3_episode_schema import (
    BEHAVIORAL_SCHEMA_VERSION,
    INFRASTRUCTURE_SCHEMA_VERSION,
    MEASUREMENT_FRAME_DESCRIPTION,
    MEASUREMENT_FRAME_ID,
    derive_frozen_failure_stage,
    derive_initial_state_sha256,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
RELEASE_MANIFEST = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/"
    "nano_mirror_v3b001_manifest.json"
)


def _behavioral_record(*, path: Path, release, cell, success: bool) -> dict:
    margin_by_condition = {
        ("control", "left"): 0.10,
        ("control", "right"): 0.16,
        ("position_mirrored", "left"): 0.12,
        ("position_mirrored", "right"): 0.24,
    }
    margin = margin_by_condition[(cell.arm, cell.relation)]
    signed_offset = margin if cell.relation == "left" else -margin
    steps = [
        {
            "action_step": index,
            "object_xyz": [0.10, 0.0, 0.70],
            "reference_xyz": [0.0, 0.0, 0.70],
            "grippers_open": index == 6,
        }
        for index in range(7)
    ]
    if success:
        for index in range(1, 4):
            steps[index]["object_xyz"][2] = 0.74
        for index in range(4, 7):
            steps[index]["object_xyz"][0] = 0.0
            steps[index]["object_xyz"][1] = signed_offset
    else:
        # A retained behavioral failure still contributes its final offset.
        # One final cone sample is transient and no verified pickup occurs.
        steps[6]["object_xyz"][0] = 0.0
        steps[6]["object_xyz"][1] = signed_offset
    record = {
        "schema_version": BEHAVIORAL_SCHEMA_VERSION,
        "record_type": "behavioral_episode",
        "behavioral_result_valid": True,
        "study_id": "vla_wam_language_steerability_v3",
        "registered_cell_id": cell.cell_id,
        "attempt_id": cell.cell_id + ":attempt01",
        "model_id": MODEL_ID,
        "pair_id": cell.row["matched_block_id"],
        "arena": "droid_robolab",
        "environment_seed": cell.seed,
        "policy_seed": cell.seed,
        "requested_relation": cell.relation,
        "prompt": cell.row["prompt"],
        "prompt_family": "direct_command",
        "predicate_id": cell.row["success_predicate_id"],
        "reset_id": cell.cell_id + ":reset",
        "measurement_frame": MEASUREMENT_FRAME_ID,
        "measurement_frame_description": MEASUREMENT_FRAME_DESCRIPTION,
        "checkpoint": {"id": MODEL_REPOSITORY, "revision": CHECKPOINT_REVISION},
        "runtime_identity": {"id": "synthetic-v3b001", "sha256": "a" * 64},
        "artifacts": {
            "viewport_video": {
                "path": "raw/synthetic_viewport.mp4",
                "sha256": "b" * 64,
                "bytes": 1,
            },
            "executed_action_trace": {
                "path": "raw/synthetic_actions.npy",
                "sha256": "c" * 64,
                "bytes": 1,
            },
            "raw_result_jsonl": {
                "path": str(path.resolve()),
                "integrity_scope": "batch_manifest_after_close",
            },
        },
        "future_interface": "decoded_rgb_uint8_33_frames_per_policy_request",
        "future_requests": [],
        "missing_future_policy": "infrastructure_invalid_never_zero",
        "requested_success": success,
        "failure_stage": "placeholder",
        "frozen_failure_stage": "placeholder",
        "failure_taxonomy": "correct" if success else "pick_failed",
        "steps": steps,
        "actions_executed": 6,
        "action_cap": 6,
        "right_censored": not success,
        "wall_time_s": 1.0,
        "operational_wall_time_valid": True,
        "final_detached_release": success,
        "first_contact_step": None,
        "first_contact_unavailable_reason": "synthetic contact stream unavailable",
        "event_timeline": (
            [
                {"event": "episode_start", "action_step": 0},
                {"event": "verified_pickup", "action_step": 1},
                {"event": "requested_region_entry", "action_step": 4},
                {"event": "episode_end", "action_step": 6},
            ]
            if success
            else [
                {"event": "episode_start", "action_step": 0},
                {"event": "requested_region_entry", "action_step": 6},
                {"event": "episode_end", "action_step": 6},
            ]
        ),
        "amendment_id": "V3-B001",
        "phase_b_arm": cell.arm,
        "release_manifest_sha256": release.manifest_sha256,
        "release_fingerprint_sha256": release.release_fingerprint(cell),
    }
    record["initial_state_sha256"] = derive_initial_state_sha256(record)
    stage = derive_frozen_failure_stage(record)
    record["failure_stage"] = stage
    record["frozen_failure_stage"] = stage
    return record


def _infrastructure_record(*, path: Path, cell) -> dict:
    return {
        "schema_version": INFRASTRUCTURE_SCHEMA_VERSION,
        "record_type": "infrastructure_attempt",
        "behavioral_result_valid": False,
        "classification": "partial",
        "arena": "droid_robolab",
        "study_id": "vla_wam_language_steerability_v3",
        "registered_cell_id": cell.cell_id,
        "attempt_id": cell.cell_id + ":infra01",
        "model_id": MODEL_ID,
        "checkpoint": {"id": MODEL_REPOSITORY, "revision": CHECKPOINT_REVISION},
        "runtime_identity": {"id": "synthetic-v3b001", "sha256": "a" * 64},
        "pair_id": cell.row["matched_block_id"],
        "environment_seed": cell.seed,
        "policy_seed": cell.seed,
        "prompt": cell.row["prompt"],
        "prompt_family": "direct_command",
        "predicate_id": cell.row["success_predicate_id"],
        "reset_id": cell.cell_id + ":infra-reset",
        "measurement_frame": MEASUREMENT_FRAME_ID,
        "measurement_frame_description": MEASUREMENT_FRAME_DESCRIPTION,
        "artifacts": {
            "raw_result_jsonl": {
                "path": str(path.resolve()),
                "integrity_scope": "batch_manifest_after_close",
            }
        },
        "stage": "simulator_export",
        "error": "synthetic export interruption",
        "log_hash": "d" * 64,
        "runtime_intervention": False,
        "repair_attempt_id": cell.cell_id + ":attempt01",
        "event_timeline": [
            {"sequence": 0, "stage": "launch"},
            {"sequence": 1, "stage": "partial"},
        ],
    }


class NanoV3B001AggregateCompilerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.release_hash = sha256_file(RELEASE_MANIFEST)
        self.release = load_release_bundle(
            RELEASE_MANIFEST,
            expected_manifest_sha256=self.release_hash,
        )
        self.behavioral: list[Path] = []
        for cell in self.release.cells:
            path = self.root / "raw" / cell.cell_id.replace(":", "__") / "raw_episode.jsonl"
            path.parent.mkdir(parents=True)
            success = not (
                cell.seed == 9426 and cell.arm == "control" and cell.relation == "left"
            )
            write_jsonl(path, [_behavioral_record(path=path, release=self.release, cell=cell, success=success)])
            self.behavioral.append(path)

    def compile(self, output_name: str, **extra):
        return compile_nano_v3b001_results(
            release_manifest=RELEASE_MANIFEST,
            release_manifest_sha256=self.release_hash,
            behavioral_jsonls=self.behavioral,
            output_directory=self.root / output_name,
            bootstrap_replicates=49,
            bootstrap_seed=1234,
            **extra,
        )

    def test_complete_analysis_is_deterministic_and_uses_frozen_formulas(self) -> None:
        first = self.compile("compiled_a")
        second = self.compile("compiled_b")
        summary = json.loads(first["summary"].read_text())
        self.assertEqual(first["summary"].read_bytes(), second["summary"].read_bytes())
        self.assertEqual(
            first["episodes_manifest"].read_bytes(),
            second["episodes_manifest"].read_bytes(),
        )
        self.assertEqual(summary["behavioral_evidence"]["valid_episode_count"], 108)
        self.assertEqual(summary["full_sample_primary"]["population"]["matched_seed_count"], 27)
        self.assertEqual(
            summary["success_conditional_secondary"]["realized_matched_seed_count"],
            26,
        )
        self.assertAlmostEqual(
            summary["full_sample_primary"]["B_by_arm"]["control"]["mean_m"],
            0.06,
        )
        self.assertAlmostEqual(
            summary["full_sample_primary"]["B_by_arm"]["position_mirrored"]["mean_m"],
            0.12,
        )
        self.assertAlmostEqual(
            summary["full_sample_primary"]["I_position_reflection_interaction"]["mean_m"],
            0.06,
        )
        self.assertAlmostEqual(
            summary["full_sample_primary"]["J_redirection_interaction"]["mean_m"],
            0.10,
        )
        self.assertEqual(len(first["episodes"].read_text().splitlines()), 108)
        manifest = json.loads(first["episodes_manifest"].read_text())
        self.assertEqual(manifest["jsonl_sha256"], sha256_file(first["episodes"]))
        self.assertEqual(len(manifest["source_batches"]), 108)

    def test_missing_and_duplicate_cells_fail_closed(self) -> None:
        with self.assertRaisesRegex(AggregateCompilationError, "exactly 108"):
            compile_nano_v3b001_results(
                release_manifest=RELEASE_MANIFEST,
                release_manifest_sha256=self.release_hash,
                behavioral_jsonls=self.behavioral[:-1],
                output_directory=self.root / "missing",
                bootstrap_replicates=5,
            )
        duplicate = list(self.behavioral)
        duplicate[-1] = duplicate[0]
        with self.assertRaisesRegex(AggregateCompilationError, "duplicate behavioral cell"):
            compile_nano_v3b001_results(
                release_manifest=RELEASE_MANIFEST,
                release_manifest_sha256=self.release_hash,
                behavioral_jsonls=duplicate,
                output_directory=self.root / "duplicate",
                bootstrap_replicates=5,
            )

    def test_jsonl_tampering_is_rejected_by_post_close_hash(self) -> None:
        target = self.behavioral[0]
        target.write_bytes(target.read_bytes() + b" ")
        with self.assertRaisesRegex(AggregateCompilationError, "manifest mismatch"):
            self.compile("tampered_hash")

    def test_semantically_tampered_cell_is_rejected_even_with_updated_hash(self) -> None:
        target = self.behavioral[0]
        row = json.loads(target.read_text())
        row["prompt"] = "Put the Rubik's cube somewhere."
        target.write_text(json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
        manifest_path = target.with_name(target.name + ".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        manifest["jsonl_sha256"] = sha256_file(target)
        manifest["jsonl_bytes"] = target.stat().st_size
        manifest_path.write_text(json.dumps(manifest, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
        with self.assertRaisesRegex(AggregateCompilationError, "released prompt"):
            self.compile("tampered_semantics")

    def test_infrastructure_attempts_are_emitted_separately(self) -> None:
        cell = self.release.cells[0]
        path = self.root / "infra" / "infrastructure_attempts.jsonl"
        path.parent.mkdir(parents=True)
        write_jsonl(path, [_infrastructure_record(path=path, cell=cell)])
        outputs = self.compile("with_infra", infrastructure_jsonls=[path])
        summary = json.loads(outputs["summary"].read_text())
        self.assertEqual(summary["behavioral_evidence"]["valid_episode_count"], 108)
        self.assertEqual(summary["infrastructure_evidence"]["provided_attempt_count"], 1)
        self.assertFalse(summary["infrastructure_evidence"]["included_in_behavioral_denominator"])
        self.assertEqual(len(outputs["infrastructure"].read_text().splitlines()), 1)
        self.assertEqual(
            json.loads(outputs["infrastructure_manifest"].read_text())["row_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
