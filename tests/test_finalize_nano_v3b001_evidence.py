#!/usr/bin/env python3
"""Synthetic tests for the Nano V3-B001 compact evidence closer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.v3.cosmos_nano_phase_b.runtime_adapter import (
    load_release_bundle,
    sha256_file,
)
from tests.test_compile_nano_v3b001_results import (
    _behavioral_record,
    _infrastructure_record,
)
from tools.compile_nano_v3b001_results import compile_nano_v3b001_results
from tools.finalize_nano_v3b001_evidence import (
    NORMALIZED_INFRASTRUCTURE_FILENAME,
    EvidenceFinalizationError,
    finalize_nano_v3b001_evidence,
)
from tools.vla_wam_v3_episode_schema import (
    INFRASTRUCTURE_SCHEMA_VERSION,
    validate_raw_episode_record,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
RELEASE_MANIFEST = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/"
    "nano_mirror_v3b001_manifest.json"
)


def _write_blob(path: Path, payload: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


class NanoV3B001EvidenceFinalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.release_hash = sha256_file(RELEASE_MANIFEST)
        self.release = load_release_bundle(
            RELEASE_MANIFEST, expected_manifest_sha256=self.release_hash
        )
        self.behavioral: list[Path] = []
        self.artifact_paths: list[Path] = []
        for ordinal, cell in enumerate(self.release.cells):
            cell_root = self.root / "raw" / cell.cell_id.replace(":", "__")
            raw_path = cell_root / "raw_episode.jsonl"
            raw_path.parent.mkdir(parents=True)
            record = _behavioral_record(
                path=raw_path,
                release=self.release,
                cell=cell,
                success=True,
            )
            video_path = cell_root / "viewport.mp4"
            action_path = cell_root / "executed_actions.npy"
            future_path = cell_root / "decoded_future.npy"
            source_path = cell_root / "simulator_export.json"
            record["artifacts"]["viewport_video"] = _write_blob(
                video_path, f"video-{ordinal}".encode()
            )
            record["artifacts"]["executed_action_trace"] = _write_blob(
                action_path, f"action-{ordinal}".encode()
            )
            record["future_requests"] = [
                {
                    "request_index": 0,
                    "decoded_future": _write_blob(
                        future_path, f"future-{ordinal}".encode()
                    ),
                }
            ]
            record["source_artifacts"] = {
                "simulator_export": _write_blob(
                    source_path, f"source-{ordinal}".encode()
                )
            }
            self.artifact_paths.extend(
                [video_path, action_path, future_path, source_path]
            )
            write_jsonl(raw_path, [record])
            self.behavioral.append(raw_path)
        self.compiled = self.root / "compiled"
        compile_nano_v3b001_results(
            release_manifest=RELEASE_MANIFEST,
            release_manifest_sha256=self.release_hash,
            behavioral_jsonls=self.behavioral,
            output_directory=self.compiled,
            bootstrap_replicates=7,
            bootstrap_seed=123,
        )

    def _attempt_events(self) -> Path:
        path = self.root / "attempts" / "attempt_events.jsonl"
        path.parent.mkdir(parents=True)
        infrastructure = _infrastructure_record(
            path=path, cell=self.release.cells[0]
        )
        rows = [
            {
                "schema_version": "vla-wam-shared-v3b-nano-attempt-event-v1",
                "status": "bridge_started",
                "cell_id": self.release.cells[0].cell_id,
            },
            {"infrastructure_record": infrastructure},
        ]
        path.write_text(
            "".join(
                json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":"))
                + "\n"
                for row in rows
            )
        )
        return path

    def test_complete_closure_is_deterministic_and_normalizes_only_valid_infrastructure(self) -> None:
        first = finalize_nano_v3b001_evidence(
            compiled_output_directory=self.compiled,
            behavioral_jsonls=self.behavioral,
            output_directory=self.root / "final_a",
        )
        second = finalize_nano_v3b001_evidence(
            compiled_output_directory=self.compiled,
            behavioral_jsonls=self.behavioral,
            output_directory=self.root / "final_b",
        )
        self.assertEqual(
            first["final_evidence_manifest"].read_bytes(),
            second["final_evidence_manifest"].read_bytes(),
        )
        manifest = json.loads(first["final_evidence_manifest"].read_text())
        self.assertEqual(manifest["counts"]["behavioral_episode_count"], 108)
        self.assertEqual(manifest["counts"]["raw_batch_count"], 108)
        self.assertEqual(manifest["counts"]["unique_referenced_artifact_count"], 432)
        self.assertEqual(len(manifest["behavioral_sources"]), 108)
        self.assertEqual(len(manifest["referenced_artifacts"]), 432)

        event_path = self._attempt_events()
        with_infrastructure = finalize_nano_v3b001_evidence(
            compiled_output_directory=self.compiled,
            behavioral_jsonls=self.behavioral,
            attempt_event_jsonls=[event_path],
            output_directory=self.root / "final_with_infrastructure",
        )
        final_manifest = json.loads(
            with_infrastructure["final_evidence_manifest"].read_text()
        )
        self.assertEqual(
            final_manifest["counts"]["normalized_infrastructure_attempt_count"], 1
        )
        self.assertEqual(
            final_manifest["counts"]["unconvertible_attempt_event_count"], 1
        )
        normalized_path = with_infrastructure["normalized_infrastructure"]
        normalized_lines = normalized_path.read_text().splitlines()
        self.assertEqual(len(normalized_lines), 1)
        normalized = validate_raw_episode_record(json.loads(normalized_lines[0]))
        self.assertEqual(normalized["schema_version"], INFRASTRUCTURE_SCHEMA_VERSION)
        self.assertEqual(
            Path(normalized["artifacts"]["raw_result_jsonl"]["path"]),
            normalized_path,
        )
        batch = json.loads(
            with_infrastructure["normalized_infrastructure_manifest"].read_text()
        )
        self.assertEqual(batch["jsonl_sha256"], sha256_file(normalized_path))
        self.assertEqual(batch["row_count"], 1)
        self.assertEqual(
            normalized_path.name, NORMALIZED_INFRASTRUCTURE_FILENAME
        )

    def test_tampered_referenced_artifact_fails_before_output(self) -> None:
        self.artifact_paths[0].write_bytes(b"tampered")
        output = self.root / "tampered_output"
        with self.assertRaisesRegex(EvidenceFinalizationError, "tampered"):
            finalize_nano_v3b001_evidence(
                compiled_output_directory=self.compiled,
                behavioral_jsonls=self.behavioral,
                output_directory=output,
            )
        self.assertFalse(output.exists())

    def test_duplicate_behavioral_input_and_nonempty_output_fail_closed(self) -> None:
        with self.assertRaisesRegex(EvidenceFinalizationError, "duplicate behavioral"):
            finalize_nano_v3b001_evidence(
                compiled_output_directory=self.compiled,
                behavioral_jsonls=[*self.behavioral, self.behavioral[0]],
                output_directory=self.root / "duplicate_output",
            )
        output = self.root / "occupied"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("preserve")
        with self.assertRaisesRegex(EvidenceFinalizationError, "must be empty"):
            finalize_nano_v3b001_evidence(
                compiled_output_directory=self.compiled,
                behavioral_jsonls=self.behavioral,
                output_directory=output,
            )
        self.assertEqual(marker.read_text(), "preserve")

    def test_tampered_compiled_summary_is_not_closed(self) -> None:
        summary_path = self.compiled / "nano_v3b001_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["full_sample_primary"]["I_position_reflection_interaction"][
            "mean_m"
        ] = 999.0
        summary_path.write_text(json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n")
        with self.assertRaisesRegex(
            EvidenceFinalizationError, "summary is tampered or not reproducible"
        ):
            finalize_nano_v3b001_evidence(
                compiled_output_directory=self.compiled,
                behavioral_jsonls=self.behavioral,
                output_directory=self.root / "summary_tamper_output",
            )


if __name__ == "__main__":
    unittest.main()
