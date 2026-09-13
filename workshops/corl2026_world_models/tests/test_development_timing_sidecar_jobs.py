from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/forecast_layout/development_timing_sidecar_jobs.py"
)
SPEC = importlib.util.spec_from_file_location("development_timing_sidecar_jobs", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sidecar = importlib.util.module_from_spec(SPEC)
import sys

sys.modules[SPEC.name] = sidecar
SPEC.loader.exec_module(sidecar)

TIMING_PATH = Path(__file__).resolve().parents[1] / "analysis/qualify_forecast_timing.py"
TIMING_SPEC = importlib.util.spec_from_file_location(
    "development_timing_sidecar_test_timing", TIMING_PATH
)
assert TIMING_SPEC is not None and TIMING_SPEC.loader is not None
timing = importlib.util.module_from_spec(TIMING_SPEC)
sys.modules[TIMING_SPEC.name] = timing
TIMING_SPEC.loader.exec_module(timing)


def write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(sidecar.queue.canonical_bytes(value))
    return path


class DevelopmentTimingSidecarTests(unittest.TestCase):
    def test_source_contains_no_duplicate_literal_dictionary_keys(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        duplicates: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            duplicates.extend((node.lineno, key) for key in set(keys) if keys.count(key) > 1)
        self.assertEqual(duplicates, [])

    def _fake_authority_spec(self, model: str) -> sidecar.AuthoritySpec:
        lower = model.lower()
        return sidecar.AuthoritySpec(
            model=model,
            job_id=f"timing-{lower}-native-authority-test",
            mode=f"{lower}-native-authority",
            role=f"wmf-forecast-0912-worker-{lower}",
            receipt_sha256="0" * 64,
            receipt_bytes=0,
            authority_sha256=("1" if model == "N3" else "2") * 64,
            authority_bytes=1234,
            target_count=32 if model == "N3" else 2,
        )

    @staticmethod
    def _d1_cache_diagnostic_value() -> dict:
        by_index = {}
        for request_index in range(57):
            full = request_index % 4 == 0
            before = (
                (0 if request_index == 0 else 9)
                if full
                else {1: 3, 2: 5, 3: 7}[request_index % 4]
            )
            after = 3 if full else before + 2
            latent = "[1,16,3,44,80]" if full else "[1,16,2,44,80]"
            rgb = "[9,352,640,3]" if full else "[5,352,640,3]"
            tensor = "[1,3,9,352,640]" if full else "[1,3,5,352,640]"
            by_index[str(request_index)] = {
                "count": 16,
                "latent_shapes": {latent: 16},
                "rank0_start_after": {str(after): 16},
                "rank0_start_before": {str(before): 16},
                "rank1_start_after": {str(after): 16},
                "rank1_start_before": {str(before): 16},
                "rgb_shapes": {rgb: 16},
                "tensor_shapes": {tensor: 16},
            }
        controls = {
            json.dumps(
                {
                    "offline_decode": True,
                    "probe_id": None,
                    "probe_plan_sha256": digest,
                },
                sort_keys=True,
                separators=(",", ":"),
            ): 228
            for digest in (
                "5964f153db83c4cd23522098f1646747d5a14c28d54b8d510089e70e4911529a",
                "695e8cb5d34f3a9cddc31bf5356aa02603b74385664ba4cc38bdd629f61c90dd",
                "6ad5f8928ea83ec6359d30ae802e53c43bdfe339e7dccf5dcbe69c57947ddf70",
                "94288f027985c0069f065cf439c4ab90a2e88018597bfca42597ab0b296a6e19",
            )
        }
        return {
            "by_request_index": by_index,
            "cell_count": 16,
            "claim_boundary": (
                "Read-only cache/decode schedule summary of immutable D1 development "
                "receipts; not timing qualification, annotation, or forecast accuracy."
            ),
            "completed_at_utc": "2026-09-13T11:56:08.550482+00:00",
            "decoded_rgb_shapes_by_request_index_modulo_four": {
                "0": {"[9,352,640,3]": 240},
                "1": {"[5,352,640,3]": 224},
                "2": {"[5,352,640,3]": 224},
                "3": {"[5,352,640,3]": 224},
            },
            "exact_modulo_four_shape_schedule_observed": True,
            "inventory": {
                "bytes": sidecar.D1_INVENTORY_BYTES,
                "path": str(sidecar.D1_DIAGNOSTIC_INVENTORY_PATH),
                "sha256": sidecar.D1_INVENTORY_SHA256,
            },
            "inventory_payload_sha256_verified": True,
            "measurement_control_distribution": controls,
            "official_action_path_distribution": {
                json.dumps("GrootSimPolicy.lazy_joint_forward_causal"): 912
            },
            "request_count": 912,
            "requests_per_cell_distribution": {"57": 16},
            "schema_version": "wmf-d1-development-cache-schedule-diagnostic-v1",
            "science_counts": {
                "behavioral_actions": 0,
                "behavioral_cells": 0,
                "model_requests": 0,
                "physical_resets": 0,
                "robot_episodes": 0,
            },
            "scientific_qualification": False,
            "source_attempt": "timing-d1-development-sidecar-001",
            "status": "read_only_summary_complete",
            "study_id": sidecar.STUDY_ID,
            "unique_request_receipt_hashes": 912,
        }

    def _d1_cache_diagnostic(self, root: Path) -> Path:
        return write_json(root / "diagnostic.json", self._d1_cache_diagnostic_value())

    def _authority_receipt(
        self, root: Path, model: str, *, safe: bool = False
    ) -> tuple[Path, str, sidecar.AuthoritySpec]:
        spec = self._fake_authority_spec(model)
        source = sidecar.CONTROL_ROOT / "sources" / sidecar.AUTHORITY_SOURCE_COMMIT
        receipt = sidecar.queue.signed_document(
            {
                "schema_version": sidecar.queue.TIMING_JOB_SCHEMA,
                "namespace": sidecar.NAMESPACE,
                "study_id": sidecar.STUDY_ID,
                "status": "passed",
                "decision": "go",
                "mode": spec.mode,
                "job_id": spec.job_id,
                "job_dir": str(spec.job_dir),
                "study_commit": sidecar.AUTHORITY_SOURCE_COMMIT,
                "queue_role": spec.role,
                "worker_id": spec.role,
                "physical_time_qualified": True,
                "behavioral_policy_skill_evaluated": False,
                "safe_to_release_confirmation": safe,
                "science_counts": sidecar.queue._expected_science_counts(
                    issued=0, referenced=6
                ),
                "evidence_counts": {
                    "qualified_generated_targets": spec.target_count,
                    "referenced_recorder_actions": 450,
                    "referenced_recorder_observations": 451,
                },
                "implementation": {
                    "timing_validator": {
                        "path": str(source / sidecar.TOOL_RELATIVE),
                        "sha256": sidecar.TOOL_SHA256,
                        "bytes": 116377,
                    },
                    "timing_contract": {
                        "path": str(source / sidecar.CONTRACT_RELATIVE),
                        "sha256": sidecar.CONTRACT_SHA256,
                        "bytes": 11765,
                    },
                },
                "outputs": {
                    "timing_authority": {
                        "path": str(spec.raw_authority_path),
                        "sha256": spec.authority_sha256,
                        "bytes": spec.authority_bytes,
                    },
                    "published_timing_authority": {
                        "path": str(spec.published_authority_path),
                        "sha256": spec.authority_sha256,
                        "bytes": spec.authority_bytes,
                    },
                },
            }
        )
        path = write_json(root / f"{model.lower()}-authority.json", receipt)
        digest = sidecar.queue.sha256_file(path)
        return path, digest, replace(
            spec, receipt_sha256=digest, receipt_bytes=path.stat().st_size
        )

    def test_authority_receipt_accepts_exact_pass_and_rejects_release_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, digest, spec = self._authority_receipt(root, "N3")
            evidence = sidecar.validate_local_authority_receipt(path, digest, spec)
            self.assertEqual(evidence["receipt"]["sha256"], digest)
            self.assertEqual(
                evidence["raw_authority"]["sha256"], spec.authority_sha256
            )

            unsafe, unsafe_digest, unsafe_spec = self._authority_receipt(
                root, "D1", safe=True
            )
            with self.assertRaisesRegex(
                sidecar.DevelopmentTimingJobError,
                "safe_to_release_confirmation",
            ):
                sidecar.validate_local_authority_receipt(
                    unsafe, unsafe_digest, unsafe_spec
                )

    def test_cache_schedule_diagnostic_is_exact_and_semantically_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._d1_cache_diagnostic(root)
            observed = sidecar.validate_local_d1_cache_schedule_diagnostic(
                path, sidecar.D1_CACHE_DIAGNOSTIC_SHA256
            )
            self.assertEqual(
                observed,
                {
                    "path": str(sidecar.D1_CACHE_DIAGNOSTIC_PATH),
                    "sha256": sidecar.D1_CACHE_DIAGNOSTIC_SHA256,
                    "bytes": sidecar.D1_CACHE_DIAGNOSTIC_BYTES,
                },
            )
            with self.assertRaisesRegex(
                sidecar.DevelopmentTimingJobError, "hash is not frozen"
            ):
                sidecar.validate_local_d1_cache_schedule_diagnostic(path, "f" * 64)

            malformed_value = self._d1_cache_diagnostic_value()
            malformed_value["by_request_index"]["1"]["rgb_shapes"] = {
                "[9,352,640,3]": 16
            }
            malformed = write_json(root / "malformed.json", malformed_value)
            malformed_identity = sidecar.queue.file_identity(malformed)
            with mock.patch.object(
                sidecar,
                "D1_CACHE_DIAGNOSTIC_SHA256",
                malformed_identity["sha256"],
            ), mock.patch.object(
                sidecar,
                "D1_CACHE_DIAGNOSTIC_BYTES",
                malformed_identity["bytes"],
            ):
                with self.assertRaisesRegex(
                    sidecar.DevelopmentTimingJobError, "request index 1 changed"
                ):
                    sidecar.validate_local_d1_cache_schedule_diagnostic(
                        malformed, malformed_identity["sha256"]
                    )

    def _dynamic_d1_aggregate(
        self, root: Path, *, status: str = "passed", wrong_path: bool = False
    ) -> tuple[Path, str, sidecar.AggregateSpec]:
        spec = sidecar.AggregateSpec(
            "D1",
            "D01",
            "d1-development-d01-simulator-003",
            sidecar.D1_ATTEMPT003_SOURCE_COMMIT,
            None,
            None,
            None,
        )
        cells = []
        for index, cell_id in enumerate(spec.cell_ids):
            path = (
                spec.raw_root
                / "cells"
                / f"{index:02d}-{cell_id.replace('__', '-')}"
                / "cell_receipt.json"
            )
            if wrong_path and index == 2:
                path = Path("/tmp/substituted-cell.json")
            cells.append(
                {
                    "path": str(path),
                    "sha256": f"{index + 1}" * 64,
                    "bytes": 70000 + index,
                }
            )
        receipt = {
            "schema_version": sidecar.D1_AGGREGATE_SCHEMA,
            "status": status,
            "exit_code": 0,
            "failure": None,
            "study_id": sidecar.STUDY_ID,
            "namespace": sidecar.NAMESPACE,
            "phase": "development",
            "layout_pair_id": "D01",
            "model_config": "D1",
            "block_id": "wmf_ablation_001_20260912__development__D01__D1",
            "study_commit": sidecar.D1_ATTEMPT003_SOURCE_COMMIT,
            "condition_order": list(sidecar.CONDITION_ORDER["D01"]),
            "cell_ids": list(spec.cell_ids),
            "counts": sidecar._expected_counts("D1"),
            "raw_attempt_root": str(spec.raw_root),
            "raw_attempt_recoverable_on_gm_pvc": True,
            "run_id": "d1-development-d01-003",
            "simulator_job_id": spec.job_id,
            "server_job_id": spec.job_id.replace("-simulator-", "-server-"),
            "all_simulator_children_reaped": True,
            "effective_model_noise_seed": 1140,
            "environment_seed": sidecar.ENVIRONMENT_SEED["D01"],
            "noise_semantics": "fixed; cells and requests are not independent noise draws",
            "queue_descriptor": {
                "job_id": spec.job_id,
                "role": "wmf-forecast-0912-worker-00",
                "path": str(sidecar.CONTROL_ROOT / "jobs" / spec.job_id / "descriptor.json"),
                "sha256": "9" * 64,
                "bytes": 2000,
            },
            "cell_receipts": cells,
            "resumed_cell_receipts": [],
            "new_cell_receipts": cells,
        }
        path = write_json(root / "aggregate.json", receipt)
        return path, sidecar.queue.sha256_file(path), spec

    def test_dynamic_d1_aggregate_requires_terminal_exact_cell_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, digest, spec = self._dynamic_d1_aggregate(root)
            evidence = sidecar.validate_local_aggregate_receipt(path, digest, spec)
            self.assertEqual(len(evidence["cell_receipts"]), 4)
            self.assertEqual(
                evidence["aggregate_receipt"]["path"], str(spec.receipt_path)
            )

            bad, bad_digest, bad_spec = self._dynamic_d1_aggregate(
                root / "bad", wrong_path=True
            )
            with self.assertRaisesRegex(
                sidecar.DevelopmentTimingJobError, "cell 2 descriptor path"
            ):
                sidecar.validate_local_aggregate_receipt(
                    bad, bad_digest, bad_spec
                )

            failed, failed_digest, failed_spec = self._dynamic_d1_aggregate(
                root / "failed", status="technical_failure"
            )
            with self.assertRaisesRegex(
                sidecar.DevelopmentTimingJobError, "aggregate changed: status"
            ):
                sidecar.validate_local_aggregate_receipt(
                    failed, failed_digest, failed_spec
                )

    @staticmethod
    def _mock_authorities() -> dict[str, dict]:
        return {
            model: {
                "receipt": {
                    "path": str(spec.receipt_path),
                    "sha256": spec.receipt_sha256,
                    "bytes": spec.receipt_bytes,
                },
                "raw_authority": {
                    "path": str(spec.raw_authority_path),
                    "sha256": spec.authority_sha256,
                    "bytes": spec.authority_bytes,
                },
                "published_authority": {
                    "path": str(spec.published_authority_path),
                    "sha256": spec.authority_sha256,
                    "bytes": spec.authority_bytes,
                },
            }
            for model, spec in sidecar.AUTHORITIES.items()
        }

    @staticmethod
    def _mock_aggregates(model: str) -> dict[str, dict]:
        return {
            layout: {
                "aggregate_receipt": {
                    "path": str(spec.receipt_path),
                    "sha256": spec.receipt_sha256 or (str(index + 3) * 64),
                    "bytes": spec.receipt_bytes or 17000,
                },
                "cell_receipts": [],
            }
            for index, (layout, spec) in enumerate(sidecar.AGGREGATES[model].items())
        }

    def test_n3_wave_is_one_zero_science_worker09_descriptor(self) -> None:
        authorities = self._mock_authorities()
        aggregates = self._mock_aggregates("N3")
        with mock.patch.object(
            sidecar,
            "validate_local_authority_receipt",
            side_effect=[authorities["N3"], authorities["D1"]],
        ), mock.patch.object(
            sidecar,
            "validate_local_aggregate_receipt",
            side_effect=[aggregates[layout] for layout in sidecar.LAYOUTS],
        ):
            wave = sidecar.build_sidecar_wave(
                model="N3",
                study_commit="a" * 40,
                n3_authority_receipt_path=Path("n3.json"),
                n3_authority_receipt_sha256=sidecar.AUTHORITIES["N3"].receipt_sha256,
                d1_authority_receipt_path=Path("d1.json"),
                d1_authority_receipt_sha256=sidecar.AUTHORITIES["D1"].receipt_sha256,
                aggregate_inputs={
                    layout: (Path(f"{layout}.json"), aggregates[layout]["aggregate_receipt"]["sha256"])
                    for layout in sidecar.LAYOUTS
                },
            )
        self.assertEqual([job["job_id"] for job in wave["jobs"]], [sidecar.JOB_ID["N3"]])
        self.assertEqual(wave["jobs"][0]["role"], sidecar.WORKER_ROLE)
        self.assertEqual(wave["jobs"][0]["publish_log_tail_bytes"], 8192)
        self.assertEqual(wave["science_counts"], sidecar._zero_science_counts())
        self.assertEqual(wave["referenced_behavioral_model_requests"], 240)
        self.assertIs(wave["safe_to_release_confirmation"], False)
        command = " ".join(wave["jobs"][0]["argv"])
        self.assertIn(sidecar.AUTHORITIES["N3"].receipt_sha256, command)
        self.assertIn(sidecar.AUTHORITIES["D1"].receipt_sha256, command)
        self.assertIn(sidecar.N3_AGGREGATES["D04"].receipt_sha256, command)
        self.assertNotIn("simulator", command)
        self.assertNotIn("n3_first_live", command)

    def test_d1_wave_cannot_emit_with_missing_attempt003_aggregate(self) -> None:
        authorities = self._mock_authorities()
        with mock.patch.object(
            sidecar,
            "validate_local_authority_receipt",
            side_effect=[authorities["N3"], authorities["D1"]],
        ):
            with self.assertRaisesRegex(
                sidecar.queue.TimingQueueError, "evidence file is missing"
            ):
                sidecar.build_sidecar_wave(
                    model="D1",
                    study_commit="b" * 40,
                    n3_authority_receipt_path=Path("n3.json"),
                    n3_authority_receipt_sha256=sidecar.AUTHORITIES["N3"].receipt_sha256,
                    d1_authority_receipt_path=Path("d1.json"),
                    d1_authority_receipt_sha256=sidecar.AUTHORITIES["D1"].receipt_sha256,
                    aggregate_inputs={
                        layout: (Path(f"missing-{layout}.json"), "f" * 64)
                        for layout in sidecar.LAYOUTS
                    },
                )

    def test_d1_attempt002_wave_binds_exact_cache_diagnostic_and_holds_confirmation(self) -> None:
        authorities = self._mock_authorities()
        aggregates = self._mock_aggregates("D1")
        with tempfile.TemporaryDirectory() as temporary:
            diagnostic = self._d1_cache_diagnostic(Path(temporary))
            with mock.patch.object(
                sidecar,
                "validate_local_authority_receipt",
                side_effect=[authorities["N3"], authorities["D1"]],
            ), mock.patch.object(
                sidecar,
                "validate_local_aggregate_receipt",
                side_effect=[aggregates[layout] for layout in sidecar.LAYOUTS],
            ):
                wave = sidecar.build_sidecar_wave(
                    model="D1",
                    study_commit="b" * 40,
                    n3_authority_receipt_path=Path("n3.json"),
                    n3_authority_receipt_sha256=(
                        sidecar.AUTHORITIES["N3"].receipt_sha256
                    ),
                    d1_authority_receipt_path=Path("d1.json"),
                    d1_authority_receipt_sha256=(
                        sidecar.AUTHORITIES["D1"].receipt_sha256
                    ),
                    aggregate_inputs={
                        layout: (
                            Path(f"{layout}.json"),
                            aggregates[layout]["aggregate_receipt"]["sha256"],
                        )
                        for layout in sidecar.LAYOUTS
                    },
                    d1_cache_schedule_diagnostic_path=diagnostic,
                    d1_cache_schedule_diagnostic_sha256=(
                        sidecar.D1_CACHE_DIAGNOSTIC_SHA256
                    ),
                )
        job = wave["jobs"][0]
        self.assertEqual(job["job_id"], "timing-d1-development-sidecar-002")
        self.assertEqual(sidecar.JOB_ID["D1"], job["job_id"])
        command = " ".join(job["argv"])
        self.assertIn(str(sidecar.D1_CACHE_DIAGNOSTIC_PATH), command)
        self.assertIn(sidecar.D1_CACHE_DIAGNOSTIC_SHA256, command)
        self.assertIn(sidecar.SIDECAR_TIMING_TOOL_SHA256, command)
        self.assertNotIn("timing-d1-development-sidecar-001 --", command)
        self.assertIs(wave["physical_time_coverage_complete"], False)
        self.assertEqual(
            wave["request_timing_coverage"], sidecar.D1_REQUEST_TIMING_COVERAGE
        )
        self.assertEqual(
            wave["physical_time_qualified_scope"],
            "source-proven full conditioning-origin decodes only",
        )
        self.assertIs(wave["safe_to_release_confirmation"], False)
        self.assertEqual(wave["science_counts"], sidecar._zero_science_counts())

    def test_d1_attempt002_runtime_rejects_old_id_and_missing_diagnostic(self) -> None:
        authorities = self._mock_authorities()
        aggregates = self._mock_aggregates("D1")
        base = dict(
            command="d1-sidecar",
            job_id=sidecar.JOB_ID["D1"],
            expected_role=sidecar.WORKER_ROLE,
            timing_tool_sha256=sidecar.SIDECAR_TIMING_TOOL_SHA256,
            contract_sha256=sidecar.CONTRACT_SHA256,
            expected_request_count=912,
            study_commit="c" * 40,
            sidecar_builder_sha256=sidecar.queue.sha256_file(MODULE_PATH),
            n3_authority_receipt_sha256=sidecar.AUTHORITIES["N3"].receipt_sha256,
            d1_authority_receipt_sha256=sidecar.AUTHORITIES["D1"].receipt_sha256,
            aggregate_receipt=[
                Path(aggregates[layout]["aggregate_receipt"]["path"])
                for layout in sidecar.LAYOUTS
            ],
            aggregate_receipt_sha256=[
                aggregates[layout]["aggregate_receipt"]["sha256"]
                for layout in sidecar.LAYOUTS
            ],
            d1_cache_schedule_diagnostic=sidecar.D1_CACHE_DIAGNOSTIC_PATH,
            d1_cache_schedule_diagnostic_sha256=sidecar.D1_CACHE_DIAGNOSTIC_SHA256,
        )
        with self.assertRaisesRegex(sidecar.DevelopmentTimingJobError, "job ID"):
            sidecar._runtime_descriptor(
                argparse.Namespace(
                    **{**base, "job_id": "timing-d1-development-sidecar-001"}
                )
            )
        with self.assertRaisesRegex(
            sidecar.DevelopmentTimingJobError, "cache schedule diagnostic"
        ):
            sidecar._runtime_descriptor(
                argparse.Namespace(
                    **{**base, "d1_cache_schedule_diagnostic": None}
                )
            )

    def test_runtime_descriptor_reconstructs_builder_exactly(self) -> None:
        authorities = self._mock_authorities()
        aggregates = self._mock_aggregates("D1")
        self_hash = sidecar.queue.sha256_file(MODULE_PATH)
        expected = sidecar._job_descriptor(
            model="D1",
            study_commit="c" * 40,
            self_sha256=self_hash,
            authorities=authorities,
            aggregates=aggregates,
            d1_cache_schedule_diagnostic={
                "path": str(sidecar.D1_CACHE_DIAGNOSTIC_PATH),
                "sha256": sidecar.D1_CACHE_DIAGNOSTIC_SHA256,
                "bytes": sidecar.D1_CACHE_DIAGNOSTIC_BYTES,
            },
        )
        args = argparse.Namespace(
            command="d1-sidecar",
            job_id=sidecar.JOB_ID["D1"],
            expected_role=sidecar.WORKER_ROLE,
            timing_tool_sha256=sidecar.SIDECAR_TIMING_TOOL_SHA256,
            contract_sha256=sidecar.CONTRACT_SHA256,
            expected_request_count=912,
            study_commit="c" * 40,
            sidecar_builder_sha256=self_hash,
            n3_authority_receipt_sha256=sidecar.AUTHORITIES["N3"].receipt_sha256,
            d1_authority_receipt_sha256=sidecar.AUTHORITIES["D1"].receipt_sha256,
            aggregate_receipt=[
                Path(aggregates[layout]["aggregate_receipt"]["path"])
                for layout in sidecar.LAYOUTS
            ],
            aggregate_receipt_sha256=[
                aggregates[layout]["aggregate_receipt"]["sha256"]
                for layout in sidecar.LAYOUTS
            ],
            d1_cache_schedule_diagnostic=sidecar.D1_CACHE_DIAGNOSTIC_PATH,
            d1_cache_schedule_diagnostic_sha256=(
                sidecar.D1_CACHE_DIAGNOSTIC_SHA256
            ),
        )
        model, observed = sidecar._runtime_descriptor(args)
        self.assertEqual(model, "D1")
        self.assertEqual(observed, expected)

    def test_staged_sidecar_implementation_uses_fresh_validator_pin(self) -> None:
        source = Path("/staged/source")
        exact = [
            {
                "path": str((source / sidecar.TOOL_RELATIVE).resolve()),
                "sha256": sidecar.SIDECAR_TIMING_TOOL_SHA256,
                "bytes": sidecar.SIDECAR_TIMING_TOOL_BYTES,
            },
            {
                "path": str((source / sidecar.CONTRACT_RELATIVE).resolve()),
                "sha256": sidecar.CONTRACT_SHA256,
                "bytes": 11765,
            },
        ]
        with mock.patch.object(sidecar.queue, "file_identity", side_effect=exact):
            observed = sidecar._validate_staged_sidecar_implementation(source)
        self.assertEqual(
            observed["timing_validator"]["sha256"],
            sidecar.SIDECAR_TIMING_TOOL_SHA256,
        )
        wrong = [
            {**exact[0], "sha256": sidecar.TOOL_SHA256},
            exact[1],
        ]
        with mock.patch.object(sidecar.queue, "file_identity", side_effect=wrong):
            with self.assertRaisesRegex(
                sidecar.DevelopmentTimingJobError, "validator bytes/hash changed"
            ):
                sidecar._validate_staged_sidecar_implementation(source)

    def test_recorder_transport_extracts_exact_request_and_rejects_cell_swap(self) -> None:
        cell_id = "wmf1__development__D01__N3__original__left"
        request = {
            "path": "/data/evidence/request.json",
            "sha256": "a" * 64,
            "bytes": 123,
        }

        def frozen(value):
            if isinstance(value, dict):
                return {
                    "__type__": "mapping",
                    "items": {key: frozen(child) for key, child in value.items()},
                }
            return value

        artifact = {
            "structure": frozen(
                {
                    "raw_response": {
                        "wmf_server_request_receipt": request,
                        "wmf_request_index": 0,
                        "wmf_cell_id": cell_id,
                    }
                }
            )
        }
        response = {"request_index": 0, "response_artifact": artifact}
        observed = sidecar._extract_request_descriptor(
            timing,
            response,
            model="N3",
            cell_id=cell_id,
            request_index=0,
        )
        self.assertEqual(observed, request)
        with self.assertRaisesRegex(
            sidecar.DevelopmentTimingJobError, "cell identity"
        ):
            sidecar._extract_request_descriptor(
                timing,
                response,
                model="N3",
                cell_id=cell_id + "-other",
                request_index=0,
            )

    def test_failure_receipt_reports_zero_science_and_confirmation_hold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary) / "jobs" / sidecar.JOB_ID["N3"]
            job.mkdir(parents=True)
            try:
                raise sidecar.DevelopmentTimingJobError("before evidence binding")
            except sidecar.DevelopmentTimingJobError as error:
                sidecar._failure_receipt(
                    context=None,
                    job_dir=job,
                    model="N3",
                    error=error,
                )
            receipt = sidecar.queue.load_json(
                job / "publish" / "timing_job_failure.json", "failure receipt"
            )
            self.assertEqual(receipt["science_counts"], sidecar._zero_science_counts())
            self.assertIs(receipt["safe_to_release_confirmation"], False)
            self.assertIs(receipt["behavioral_policy_skill_evaluated"], False)


if __name__ == "__main__":
    unittest.main()
