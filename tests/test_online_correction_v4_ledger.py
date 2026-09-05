"""Tests for the V4 attempt-to-accepted-ledger compiler."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from experiments.online_correction_v4.analysis import validate_accepted_ledger
from experiments.online_correction_v4.contracts import ATTEMPT_SELECTION_RULE
from experiments.online_correction_v4.ledger import (
    compile_accepted_ledger_from_attempts,
    discover_finalized_attempts,
    load_finalized_attempt,
    verify_evidence_manifest,
    write_ledger_outputs,
)
from experiments.online_correction_v4.recorder import EpisodeEvidenceRecorder
from experiments.online_correction_v4.leases import AttemptFinalizer

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("online_correction_v4", ROOT / "tools/online_correction_v4.py")
v4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v4)

PROTO_SHA = "d" * 64
SCORER_SHA = "c" * 64
VIDEO_SHA = "b" * 64
GOAL_SET_HASH = "1" * 64
PREFIX_RECEIPT_SHA = "2" * 64
PREFIX_IDENTITY_SHA = "3" * 64
RESPONSE_SCORER_SHA = "4" * 64


def _subset_manifest(*, families: tuple[str, ...], blocks: range) -> list[dict]:
    config, config_sha = v4.load_json(ROOT / "docs/online_correction_v4/campaign.json")
    rows = v4.build_manifest(config, config_sha)
    selected = set(families)
    return [row for row in rows if row["family"] in selected and row["block_id"] in blocks]


def _c2_outcome(response_m: float = 0.10) -> dict:
    return {
        "goal_violation_capped_m": response_m,
        "response_goal_violation_capped_m": response_m,
        "response_horizon_s": 2.0,
        "response_anchor": "t_event_planned+2.0s",
        "response_goal_set_branch": "move",
        "response_goal_set_hash_sha256": GOAL_SET_HASH,
        "response_projection": "planar",
        "response_scorer_sha256": RESPONSE_SCORER_SHA,
        "goal_set_empty": False,
        "goal_violation_cap_applied": False,
        "failure_stage": "pickup",
    }


def _c1_outcome(*, success: bool) -> dict:
    capped = 0.0 if success else 0.12
    return {
        "goal_violation_capped_m": capped,
        "goal_set_empty": False,
        "goal_violation_cap_applied": False,
        "failure_stage": "none" if success else "pickup",
    }


class AttemptFixtureBuilder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.finalizer = AttemptFinalizer(root)

    def write_attempt(
        self,
        *,
        manifest_row: dict,
        attempt_id: str,
        status: str,
        success: bool = False,
        trigger_eligible: bool = True,
        corrupt_blob: bool = False,
        include_c2_contract: bool = False,
        infra_reason: str = "simulator_crash",
    ) -> Path:
        episode_id = manifest_row["episode_id"]
        family = manifest_row["family"]
        episode = {
            "episode_id": episode_id,
            "attempt_id": attempt_id,
            "prefix_group_id": manifest_row["prefix_group_id"],
            "success": success,
            "trigger_eligible": trigger_eligible,
            "event_delivered": trigger_eligible,
            "event_observed": False,
            "timing": {"t_event_planned_s": 1.0},
            "provenance": {
                "protocol_sha256": PROTO_SHA,
                "scorer_sha256": SCORER_SHA,
                "config_sha256": manifest_row["config_sha256"],
            },
        }
        if status == "valid":
            if family == "C2" and include_c2_contract:
                episode["outcome"] = _c2_outcome()
                episode.update(
                    {
                        "common_prefix_verification_mode": "deterministic_fresh_session_replay",
                        "common_prefix_verification_receipt_sha256": PREFIX_RECEIPT_SHA,
                        "common_prefix_identity_hash_sha256": PREFIX_IDENTITY_SHA,
                    }
                )
            elif family == "C2":
                episode["outcome"] = _c1_outcome(success=success)
            else:
                episode["outcome"] = _c1_outcome(success=success)
        path = self.finalizer.begin_attempt(
            episode_id=episode_id,
            attempt_id=attempt_id,
            metadata={"episode_id": episode_id, "attempt_id": attempt_id},
        )
        recorder = EpisodeEvidenceRecorder(
            attempt_path=path,
            finalizer=self.finalizer,
            episode_id=episode_id,
            attempt_id=attempt_id,
            episode_record=dict(episode),
        )
        recorder.record_trajectory_row({"sim_time_s": 0.0, "control_tick": 0})
        payload = b"viewport-bytes"
        if corrupt_blob:
            recorder._store_blob("viewport.bin", payload)
            blob_path = path / recorder.blobs[0].relative_path
            blob_path.write_bytes(b"corrupted")
        else:
            recorder._store_blob("viewport.bin", payload)
        recorder.episode_record["viewport_video"] = {
            "video_uri": recorder.blobs[0].relative_path,
            "video_sha256": recorder.blobs[0].sha256 if not corrupt_blob else VIDEO_SHA,
        }
        if status == "infra_invalid":
            receipt = {
                "status": "infra_invalid",
                "infra_invalid_reason": infra_reason,
                "end_reason": "infra_invalid",
            }
        else:
            receipt = {
                "status": "valid",
                "success": success,
                "failure_label": "success" if success else "no_grasp",
                "failure_stage": "none" if success else "pickup",
            }
        recorder.finalize(terminal_receipt=receipt)
        return path


class LedgerCompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.config_sha = v4.load_json(ROOT / "docs/online_correction_v4/campaign.json")
        cls.c1_manifest = _subset_manifest(families=("C1",), blocks=range(1))
        cls.c2_manifest = _subset_manifest(families=("C2",), blocks=range(1))
        cls.c1_row = next(row for row in cls.c1_manifest if row["factors"]["policy"] == "cosmos3_nano_droid")
        cls.c2_row = next(row for row in cls.c2_manifest if row["factors"]["policy"] == "cosmos3_nano_droid")

    def _compile(self, manifest, attempts_root):
        attempts = discover_finalized_attempts(attempts_root)
        return compile_accepted_ledger_from_attempts(
            manifest=manifest,
            attempts=attempts,
            attempts_root=attempts_root,
            protocol_sha256=PROTO_SHA,
            scorer_sha256=SCORER_SHA,
            config=self.config,
        )

    def test_valid_failure_is_retained_in_accepted_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = AttemptFixtureBuilder(Path(tmp))
            builder.write_attempt(
                manifest_row=self.c1_row,
                attempt_id=f"{self.c1_row['episode_id']}--a001",
                status="valid",
                success=False,
            )
            result = self._compile([self.c1_row], Path(tmp))
            self.assertTrue(result.ok)
            self.assertEqual(len(result.accepted_rows), 1)
            self.assertFalse(result.accepted_rows[0]["success"])
            self.assertEqual(result.accepted_rows[0]["status"], "valid")

    def test_infra_retry_selects_latest_verified_valid_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = AttemptFixtureBuilder(Path(tmp))
            builder.write_attempt(
                manifest_row=self.c1_row,
                attempt_id=f"{self.c1_row['episode_id']}--a001",
                status="infra_invalid",
            )
            builder.write_attempt(
                manifest_row=self.c1_row,
                attempt_id=f"{self.c1_row['episode_id']}--a002",
                status="valid",
                success=False,
            )
            result = self._compile([self.c1_row], Path(tmp))
            self.assertTrue(result.ok)
            self.assertEqual(result.accepted_rows[0]["attempt_id"], f"{self.c1_row['episode_id']}--a002")
            self.assertEqual(len(result.rejected_rows), 1)
            self.assertEqual(result.rejected_rows[0]["status"], "infra_invalid")
            self.assertEqual(result.rejected_rows[0]["reason"], "simulator_crash")

    def test_duplicate_valid_attempts_keep_latest_without_outcome_peeking(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = AttemptFixtureBuilder(Path(tmp))
            builder.write_attempt(
                manifest_row=self.c1_row,
                attempt_id=f"{self.c1_row['episode_id']}--a001",
                status="valid",
                success=False,
            )
            builder.write_attempt(
                manifest_row=self.c1_row,
                attempt_id=f"{self.c1_row['episode_id']}--a002",
                status="valid",
                success=True,
            )
            result = self._compile([self.c1_row], Path(tmp))
            self.assertTrue(result.ok)
            self.assertEqual(result.accepted_rows[0]["attempt_id"], f"{self.c1_row['episode_id']}--a002")
            superseded = [row for row in result.rejected_rows if row.get("rejection_class") == "superseded_valid_attempt"]
            self.assertEqual(len(superseded), 1)
            self.assertEqual(superseded[0]["superseded_by_attempt_id"], f"{self.c1_row['episode_id']}--a002")

    def test_corrupted_evidence_is_rejected_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = AttemptFixtureBuilder(Path(tmp))
            builder.write_attempt(
                manifest_row=self.c1_row,
                attempt_id=f"{self.c1_row['episode_id']}--a001",
                status="valid",
                corrupt_blob=True,
            )
            result = self._compile([self.c1_row], Path(tmp))
            self.assertTrue(result.ok)
            self.assertEqual(result.accepted_rows, [])
            self.assertEqual(len(result.rejected_rows), 1)
            self.assertEqual(result.rejected_rows[0]["rejection_class"], "evidence_corruption")

    def test_c2_incomplete_contract_is_emitted_without_synthesis_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = AttemptFixtureBuilder(Path(tmp))
            builder.write_attempt(
                manifest_row=self.c2_row,
                attempt_id=f"{self.c2_row['episode_id']}--a001",
                status="valid",
                include_c2_contract=False,
            )
            result = self._compile([self.c2_row], Path(tmp))
            self.assertTrue(result.ok)
            row = result.accepted_rows[0]
            self.assertNotIn("common_prefix_verification_mode", row)
            self.assertNotIn("response_goal_violation_capped_m", row["outcome"])
            self.assertFalse(result.manifest_payload["validation_preview"]["ok"])
            self.assertIn(
                self.c2_row["episode_id"],
                result.manifest_payload["validation_preview"]["c2_incomplete_episode_ids"],
            )
            validation = validate_accepted_ledger([self.c2_row], result.accepted_rows, config=self.config)
            self.assertFalse(validation["ok"])

    def test_c2_complete_contract_passes_validation_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = AttemptFixtureBuilder(Path(tmp))
            builder.write_attempt(
                manifest_row=self.c2_row,
                attempt_id=f"{self.c2_row['episode_id']}--a001",
                status="valid",
                include_c2_contract=True,
            )
            result = self._compile([self.c2_row], Path(tmp))
            self.assertTrue(result.ok)
            accepted = result.accepted_rows[0]
            self.assertIn("common_prefix_verification_mode", accepted)
            self.assertIn("response_goal_violation_capped_m", accepted["outcome"])
            validation = validate_accepted_ledger([self.c2_row], result.accepted_rows, config=self.config)
            self.assertTrue(validation["ok"])

    def test_control_reuse_requires_accepted_source_episodes(self):
        c3_manifest = _subset_manifest(families=("C1", "C3"), blocks=range(1))
        c3_row = next(
            row
            for row in c3_manifest
            if row["family"] == "C3"
            and row["block_id"] == 0
            and row["factors"]["policy"] == "cosmos3_nano_droid"
        )
        c1_sources = [row for row in c3_manifest if row["episode_id"] in c3_row["reuse_episode_ids"]]
        self.assertEqual(len(c1_sources), 2)
        with tempfile.TemporaryDirectory() as tmp:
            builder = AttemptFixtureBuilder(Path(tmp))
            builder.write_attempt(
                manifest_row=c3_row,
                attempt_id=f"{c3_row['episode_id']}--a001",
                status="valid",
                success=False,
            )
            result = self._compile([c3_row], Path(tmp))
            self.assertFalse(result.ok)
            self.assertTrue(any("control reuse" in error for error in result.errors))

            for source in c1_sources:
                builder.write_attempt(
                    manifest_row=source,
                    attempt_id=f"{source['episode_id']}--a001",
                    status="valid",
                    success=True,
                )
            result = self._compile([c3_row] + c1_sources, Path(tmp))
            self.assertTrue(result.ok)
            c3_accepted = next(row for row in result.accepted_rows if row["episode_id"] == c3_row["episode_id"])
            self.assertEqual(c3_accepted["reuse_episode_ids"], c3_row["reuse_episode_ids"])
            link = next(
                item
                for item in result.reconciliation["control_reuse_links"]
                if item["episode_id"] == c3_row["episode_id"]
            )
            self.assertTrue(link["reuse_sources_accepted"])

    def test_atomic_outputs_include_provenance_hashes(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            builder = AttemptFixtureBuilder(Path(tmp))
            builder.write_attempt(
                manifest_row=self.c1_row,
                attempt_id=f"{self.c1_row['episode_id']}--a001",
                status="valid",
                success=False,
            )
            result = self._compile([self.c1_row], Path(tmp))
            outputs = write_ledger_outputs(
                result,
                Path(out),
                attempts_root=Path(tmp),
                manifest_path=ROOT / "artifacts/online_correction_v4/queue.jsonl",
            )
            manifest = json.loads(Path(outputs["accepted_ledger_manifest"]).read_text())
            self.assertEqual(manifest["attempt_selection_rule"], ATTEMPT_SELECTION_RULE)
            self.assertIn("accepted_ledger_sha256", manifest["outputs"])
            accepted = [
                json.loads(line)
                for line in Path(outputs["accepted_ledger"]).read_text().splitlines()
                if line.strip()
            ]
            self.assertIn("provenance", accepted[0])
            self.assertEqual(len(accepted[0]["provenance"]["complete_receipt_sha256"]), 64)

    def test_verify_evidence_manifest_detects_episode_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = AttemptFixtureBuilder(Path(tmp))
            path = builder.write_attempt(
                manifest_row=self.c1_row,
                attempt_id=f"{self.c1_row['episode_id']}--a001",
                status="valid",
            )
            episode_path = path / "episode.json"
            payload = json.loads(episode_path.read_text())
            payload["tampered"] = True
            episode_path.write_text(json.dumps(payload, indent=2) + "\n")
            parsed = load_finalized_attempt(path)
            errors = verify_evidence_manifest(path, parsed.evidence_manifest)
            self.assertTrue(any("episode_sha256 mismatch" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
