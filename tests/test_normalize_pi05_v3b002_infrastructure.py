#!/usr/bin/env python3
"""Focused tests for V3-B002 retained-infrastructure normalization."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.v3.pi05_phase_b.contract import (
    CHECKPOINT_MANIFEST_SHA256,
    OPENPI_COMMIT,
    OPENPI_CONFIG,
    ROBOLAB_COMMIT,
    RUNTIME_SCHEMA,
    canonical_json_bytes,
    cells_for_lane,
    load_release_bundle,
    sha256_bytes,
)
from tools.normalize_pi05_v3b002_infrastructure import (
    NormalizationError,
    normalize_attempt,
)
from tools.vla_wam_v3_episode_schema import parse_jsonl_record


ROOT = Path(__file__).resolve().parents[1]
RELEASE = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/"
    "pi05_mirror_v3b002_manifest.json"
)
RELEASE_SHA256 = "8aaaa38302f6a654090250b3e12cd8735fab28a74027d50193325ffa9d0dddea"


def _runtime(lane_count: int = 6) -> dict:
    topology = {
        "schema_version": "vla-wam-shared-v3b-pi05-live-topology-v1",
        "policy_server": {
            "owner": "ali",
            "pod": "policy-ali",
            "pod_uid": "policy-uid",
            "pod_ip": "10.0.0.1",
            "gpu_uuid": "GPU-policy",
            "gpu_model": "B200",
            "driver_version": "580",
            "port": 8001,
            "gpu_index": 2,
            "model_request_count_at_capture": 0,
        },
        "simulator_lanes": [
            {
                "owner": "ali",
                "pod": f"lane-{index}-ali",
                "pod_uid": f"lane-uid-{index}",
                "pod_ip": f"10.0.1.{index + 1}",
                "gpu_uuid": f"GPU-lane-{index}",
                "gpu_model": "RTX PRO 6000",
                "driver_version": "580",
            }
            for index in range(lane_count)
        ],
    }
    payload = {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-B002",
        "model_id": "pi05_current_stack_droid",
        "openpi_commit": OPENPI_COMMIT,
        "robolab_commit": ROBOLAB_COMMIT,
        "openpi_config": OPENPI_CONFIG,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "checkpoint_sha256": "1" * 64,
        "environment_lock_sha256": "2" * 64,
        "base_runtime_identity_sha256": "3" * 64,
        "external_repository_diff_hash": "4" * 64,
        "openpi_dir_status_sha256": "5" * 64,
        "robolab_dir_status_sha256": "6" * 64,
        "release_manifest_sha256": RELEASE_SHA256,
        "phase_b_adapter_contract_sha256": "7" * 64,
        "study_git_commit": "8" * 40,
        "live_topology": topology,
        "live_topology_sha256": sha256_bytes(canonical_json_bytes(topology)),
        "action_space": "joint_position_8d",
        "action_chunk_shape": [15, 8],
        "open_loop_horizon": 15,
        "action_cap": 450,
        "instruction_controller": "static_episode_prompt",
        "future_interface": "actions_only",
        "missing_future_policy": "action_only_interface_not_applicable_never_zero",
        "renderer_backend": "realtime RTX Vulkan",
        "simulator_version": "Isaac Sim 5.0",
    }
    payload["runtime_identity_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class NormalizePi05V3B002InfrastructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.release = load_release_bundle(
            ROOT, RELEASE, expected_manifest_sha256=RELEASE_SHA256
        )

    def _workspace(self) -> tuple[tempfile.TemporaryDirectory, Path, Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        base = Path(temporary.name)
        raw = base / "behavioral_attempt"
        raw.mkdir()
        runtime = base / "runtime.json"
        _write_json(runtime, _runtime())
        output = base / "normalized" / "infrastructure_attempts.jsonl"
        return temporary, raw, runtime, output

    def _normalize(self, raw: Path, runtime: Path, output: Path, label: str = "behavioral_attempt01") -> dict:
        return normalize_attempt(
            repo_root=ROOT,
            release_manifest=RELEASE,
            release_manifest_sha256=RELEASE_SHA256,
            raw_attempt_root=raw,
            attempt_label=label,
            runtime_manifest=runtime,
            output_jsonl=output,
        )

    def test_per_cell_failures_emit_six_unique_common_schema_rows(self) -> None:
        temporary, raw, runtime, output = self._workspace()
        self.addCleanup(temporary.cleanup)
        expected_cells = []
        for lane in range(6):
            cells = sorted(
                cells_for_lane(self.release.cells, lane_index=lane, lane_count=6),
                key=lambda cell: (cell.seed, cell.row["execution_order_index_within_seed"]),
            )
            cell = cells[0]
            expected_cells.append(cell.cell_id)
            attempt = (
                raw
                / "V3-B002_pi05_position_mirror"
                / cell.cell_id.replace(":", "__")
                / "attempt01"
            )
            _write_json(
                attempt / "bridge_failure.json",
                {
                    "registered_cell_id": cell.cell_id,
                    "denominator_eligible": False,
                    "error_type": "ImportError",
                    "error": "cannot import name 'Empty' from partially initialized module 'queue'",
                    "traceback": "Traceback retained",
                },
            )
            events = [
                {"status": "bridge_started"},
                {
                    "status": "infrastructure_failed_excluded_from_denominator",
                    "error": "ImportError: standard-library queue was shadowed",
                },
            ]
            (attempt / "attempt_events.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
                encoding="utf-8",
            )
            (raw / f"queue_lane{lane}.log").write_text(
                "ImportError: cannot import name 'Empty' from partially initialized module 'queue'\n",
                encoding="utf-8",
            )

        result = self._normalize(raw, runtime, output)
        self.assertEqual(result["attempt_count"], 6)
        rows = [parse_jsonl_record(line) for line in output.read_text().splitlines()]
        self.assertEqual({row["registered_cell_id"] for row in rows}, set(expected_cells))
        self.assertEqual(len({row["attempt_id"] for row in rows}), 6)
        self.assertTrue(all(row["classification"] == "technical_invalid" for row in rows))
        self.assertTrue(all(row["behavioral_result_valid"] is False for row in rows))
        self.assertTrue(
            all(row["stage"] == "bridge_module_import_before_model_request" for row in rows)
        )
        manifest = json.loads(output.with_name(output.name + ".manifest.json").read_text())
        self.assertEqual(manifest["row_count"], 6)
        self.assertEqual(
            manifest["record_schema_versions"],
            ["vla-wam-shared-v3-infrastructure-attempt-v1"],
        )

    def test_root_only_lane_logs_map_to_each_frozen_lane_first_cell(self) -> None:
        temporary, raw, runtime, output = self._workspace()
        self.addCleanup(temporary.cleanup)
        expected = set()
        for lane in range(6):
            cell = sorted(
                cells_for_lane(self.release.cells, lane_index=lane, lane_count=6),
                key=lambda item: (item.seed, item.row["execution_order_index_within_seed"]),
            )[0]
            expected.add(cell.cell_id)
            (raw / f"queue_lane{lane}.log").write_text(
                "Traceback (most recent call last):\n"
                "FileExistsError: [Errno 17] File exists: pod-local temp namespace\n",
                encoding="utf-8",
            )

        result = self._normalize(raw, runtime, output, "behavioral_attempt02")
        self.assertEqual(result["attempt_count"], 6)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual({row["registered_cell_id"] for row in rows}, expected)
        self.assertTrue(
            all(row["stage"] == "queue_attempt_directory_creation_before_bridge" for row in rows)
        )
        self.assertTrue(
            all("queue_log" in row["artifacts"] for row in rows)
        )

    def test_missing_lane_failure_evidence_is_rejected(self) -> None:
        temporary, raw, runtime, output = self._workspace()
        self.addCleanup(temporary.cleanup)
        for lane in range(5):
            (raw / f"queue_lane{lane}.log").write_text(
                "FileExistsError: retained failure\n", encoding="utf-8"
            )
        with self.assertRaisesRegex(NormalizationError, "lane 5"):
            self._normalize(raw, runtime, output, "behavioral_attempt02")
        self.assertFalse(output.exists())

    def test_behavioral_jsonl_in_infrastructure_root_is_rejected(self) -> None:
        temporary, raw, runtime, output = self._workspace()
        self.addCleanup(temporary.cleanup)
        (raw / "raw_episode.jsonl").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(NormalizationError, "contains behavioral JSONL"):
            self._normalize(raw, runtime, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
