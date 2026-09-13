from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


WORKSHOP = Path(__file__).resolve().parents[1]
MODULE_PATH = WORKSHOP / "analysis/compile_development_resources.py"
SPEC = importlib.util.spec_from_file_location("compile_development_resources_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
resources = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = resources
SPEC.loader.exec_module(resources)


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resources.pretty_bytes(value))
    return path


def cuda_memory(*, allocated: int = 10, reserved: int = 20) -> dict[str, object]:
    return {
        "cuda_available": True,
        "allocated_bytes_before": 1,
        "reserved_bytes_before": 2,
        "allocated_bytes_after": 3,
        "reserved_bytes_after": 4,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
    }


class FakeEvidence:
    def __init__(self, rows: list[dict], tail: str = "tail") -> None:
        self.rows = rows
        self.tail = tail

    def _verify_journal(self, path: Path) -> tuple[list[dict], str]:
        del path
        return self.rows, self.tail


class DevelopmentResourceCompilerTests(unittest.TestCase):
    def test_contract_pins_definitions_zero_science_and_confirmation_hold(self) -> None:
        contract = resources._validate_contract()
        self.assertEqual(contract["science_counts"], resources.SCIENCE_COUNTS)
        self.assertEqual(
            contract["measurement_definitions"]["d1_source_cost_definition"],
            resources.D1_SOURCE_COST_DEFINITION,
        )
        self.assertFalse(contract["release_boundary"]["safe_to_release_confirmation"])

    def _journal_rows(self, request_count: int) -> list[dict]:
        rows: list[dict] = [{"kind": "attempt_started", "monotonic_ns": 100}]
        for request_index in range(request_count):
            base = 1_000 + request_index * 100
            rows.append({
                "kind": "model_request_sent",
                "monotonic_ns": base,
                "payload": {"request_index": request_index, "send_monotonic_ns": base},
            })
            rows.append({
                "kind": "model_response_received",
                "monotonic_ns": base + 50,
                "payload": {
                    "request_index": request_index,
                    "receive_monotonic_ns": base + 50,
                },
            })
        rows.append({"kind": "attempt_finalized", "monotonic_ns": 10_100})
        return rows

    def _cell(self, root: Path, model: str, requests: list[dict]) -> dict:
        request_descriptors = []
        for index, request in enumerate(requests):
            path = write_json(root / f"request-{index:03d}.json", request)
            request_descriptors.append({"path": str(path)})
        cell_id = f"wmf1__development__D01__{model}__original__left"
        return {
            "cell": {
                "cell_id": cell_id,
                "model_config": model,
                "layout_pair_id": "D01",
            },
            "cell_descriptor": {"path": str(root / "cell_receipt.json")},
            "completion_descriptor": {"path": str(root / "completion.json")},
            "journal_descriptor": {
                "path": str(root / "journal.jsonl"),
                "tail_sha256": "tail",
            },
            "request_descriptors": request_descriptors,
        }

    @staticmethod
    def _d1_request() -> dict:
        rank0_seconds = 0.4
        rank1_seconds = 0.6
        artifacts = [11, 13, 17, 19]
        return {
            "cost": {
                "rank_count": 2,
                "inference_wall_seconds_rank0_wrapper": 0.5,
                "summed_rank_forward_gpu_seconds_proxy": rank0_seconds + rank1_seconds,
                "offline_decode_wall_seconds": 0.25,
                "retained_output_bytes": sum(artifacts),
                "definition": resources.D1_SOURCE_COST_DEFINITION,
            },
            "temporal_and_cache_rank_metrics": [
                {"rank": 0, "wall_seconds": rank0_seconds, "cuda_memory": cuda_memory(allocated=100, reserved=200)},
                {"rank": 1, "wall_seconds": rank1_seconds, "cuda_memory": cuda_memory(allocated=300, reserved=400)},
            ],
            "official_returned_action": {"bytes": artifacts[0]},
            "latent_video": {"bytes": artifacts[1]},
            "offline_decode": {
                "requested": True,
                "performed": True,
                "wall_seconds": 0.25,
                "cuda_memory": cuda_memory(allocated=500, reserved=600),
                "decoded_tensor": {"bytes": artifacts[2]},
                "decoded_rgb": {"bytes": artifacts[3]},
            },
        }

    def test_n3_measurement_uses_distinct_monotonic_clocks_and_keeps_peaks_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = [
                {"started_monotonic_ns": index * 100, "completed_monotonic_ns": index * 100 + 20}
                for index in range(15)
            ]
            cell = self._cell(root, "N3", requests)
            measured = resources.measure_cell(
                cell, evidence=FakeEvidence(self._journal_rows(15))
            )
        self.assertEqual(measured["episode_wall"]["elapsed_nanoseconds"], 10_000)
        self.assertEqual(
            measured["inference_and_request_time"]["comparable_policy_request_roundtrip"]["nanoseconds_total"],
            750,
        )
        self.assertEqual(
            measured["inference_and_request_time"]["n3_server_request_service"]["nanoseconds_total"],
            300,
        )
        self.assertIsNone(
            measured["gpu_memory"]["model_server_forward"]["peak_allocated_bytes"]
        )
        self.assertFalse(measured["gpu_memory"]["measurement_complete_for_resource_release"])

    def test_d1_measurement_preserves_rank_proxy_and_allocator_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = [self._d1_request() for _ in range(57)]
            measured = resources.measure_cell(
                self._cell(root, "D1", requests),
                evidence=FakeEvidence(self._journal_rows(57)),
            )
        timing = measured["inference_and_request_time"]
        self.assertAlmostEqual(timing["d1_rank0_inference_wrapper"]["seconds_total"], 28.5)
        self.assertAlmostEqual(
            timing["d1_summed_rank_forward_gpu_seconds_proxy"]["seconds_total"], 57.0
        )
        self.assertEqual(timing["d1_retained_model_output"]["bytes_total"], 57 * 60)
        memory = measured["gpu_memory"]
        self.assertEqual(memory["model_server_rank_forward"]["max_device_peak_allocated_bytes"], 300)
        self.assertEqual(memory["offline_decode_rank0"]["peak_reserved_bytes"], 600)
        self.assertIsNone(memory["simulator_process"]["peak_allocated_bytes"])

    def test_d1_rejects_changed_source_cost_definition_and_rank_proxy(self) -> None:
        request = self._d1_request()
        changed = copy.deepcopy(request)
        changed["cost"]["definition"] = "invented billing seconds"
        with self.assertRaisesRegex(resources.ResourceCompilerError, "cost definition changed"):
            resources._measure_d1_requests([changed], cell_id="cell")
        changed = copy.deepcopy(request)
        changed["cost"]["summed_rank_forward_gpu_seconds_proxy"] = 9.0
        with self.assertRaisesRegex(resources.ResourceCompilerError, "differs from rank metrics"):
            resources._measure_d1_requests([changed], cell_id="cell")

    def test_cuda_allocator_rejects_impossible_ordering(self) -> None:
        value = cuda_memory(allocated=1, reserved=2)
        value["allocated_bytes_after"] = 3
        with self.assertRaisesRegex(resources.ResourceCompilerError, "ordering changed"):
            resources._validate_cuda_memory(value, "bad")

    def test_scan_tree_rejects_symlink_and_nonregular_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary).resolve()
            tree = raw / "tree"
            tree.mkdir()
            (tree / "regular").write_bytes(b"x")
            (tree / "link").symlink_to(tree / "regular")
            with self.assertRaisesRegex(resources.ResourceCompilerError, "symlink"):
                resources._scan_tree(tree, raw, "test tree")
            (tree / "link").unlink()
            os.mkfifo(tree / "fifo")
            with self.assertRaisesRegex(resources.ResourceCompilerError, "non-regular"):
                resources._scan_tree(tree, raw, "test tree")

    def test_full_scoped_inventory_covers_32_cells_and_1152_request_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary).resolve()
            aggregates: dict[tuple[str, str], dict] = {}
            servers: dict[str, dict] = {}
            compiled: dict[str, list[dict]] = {"N3": [], "D1": []}
            block_roots: list[Path] = []
            for model in resources.MODELS:
                for layout in resources.LAYOUTS:
                    block = raw / "blocks" / model / layout
                    block.mkdir(parents=True)
                    (block / "shared.json").write_bytes(b"shared")
                    block_roots.append(block)
                    aggregates[(model, layout)] = {"raw_attempt_root": str(block)}
                    if model == "D1":
                        server = raw / "servers" / layout
                        server.mkdir(parents=True)
                        (server / "server.json").write_bytes(b"server")
                        block_roots.append(server)
                        servers[layout] = {"raw_attempt_root": str(server)}
                    for condition_index in range(4):
                        cell_id = (
                            f"wmf1__development__{layout}__{model}__"
                            f"condition__{condition_index}"
                        )
                        cell_root = block / "cells" / cell_id
                        cell_root.mkdir(parents=True)
                        cell_receipt = cell_root / "cell_receipt.json"
                        completion = cell_root / "completion.json"
                        journal = cell_root / "journal.jsonl"
                        for path in (cell_receipt, completion, journal):
                            path.write_bytes(path.name.encode())
                        requests = []
                        request_count = resources.MODEL_REQUESTS_PER_CELL[model]
                        for request_index in range(request_count):
                            request_root = (
                                cell_root / "requests" / str(request_index)
                                if model == "N3"
                                else Path(servers[layout]["raw_attempt_root"])
                                / "requests" / cell_id / str(request_index)
                            )
                            request_root.mkdir(parents=True)
                            request = request_root / "request_receipt.json"
                            request.write_bytes(b"request")
                            requests.append({"path": str(request)})
                        compiled[model].append({
                            "cell": {"cell_id": cell_id, "layout_pair_id": layout},
                            "cell_descriptor": {"path": str(cell_receipt)},
                            "completion_descriptor": {"path": str(completion)},
                            "journal_descriptor": {"path": str(journal)},
                            "request_descriptors": requests,
                        })

            first = block_roots[0] / "shared-hardlink.bin"
            first.write_bytes(b"one storage object")
            os.link(first, block_roots[1] / "shared-hardlink.bin")
            inventory, per_cell = resources.build_file_inventory(
                raw_root=raw,
                aggregate_receipts=aggregates,
                d1_servers=servers,
                compiled_by_model=compiled,
            )
        resources.verify_signed(inventory, "inventory")
        self.assertEqual(len(per_cell), 32)
        self.assertEqual(
            inventory["scope_totals"]["model_request_tree"]["file_count"], 1152
        )
        self.assertEqual(
            inventory["scope_totals"]["recorder_cell_tree"]["file_count"], 96
        )
        self.assertLess(
            inventory["unique_storage_object_count"], inventory["file_count"]
        )
        self.assertTrue(
            all(row["attributable_deduplicated_bytes"] > 0 for row in per_cell.values())
        )

    def _write_valid_bundle(self, root: Path) -> tuple[dict, dict]:
        rows = []
        for model in resources.MODELS:
            for layout in resources.LAYOUTS:
                for arm in ("original", "reflected"):
                    for command in ("left", "right"):
                        cell_id = (
                            f"wmf1__development__{layout}__{model}__{arm}__{command}"
                        )
                        relative = f"cells/{model.lower()}/{cell_id}.json"
                        request_count = resources.MODEL_REQUESTS_PER_CELL[model]
                        inference = {
                            "comparable_policy_request_roundtrip": {
                                "status": "measured",
                                "definition": resources.REQUEST_ROUNDTRIP_DEFINITION,
                                "seconds_total": 1e-6,
                                "nanoseconds_total": 1_000,
                                "request_count": request_count,
                                "pure_gpu_forward_time": False,
                            }
                        }
                        gpu = {
                            "definition": resources.GPU_PEAK_DEFINITION,
                            "simulator_process": {
                                "status": "missing_not_instrumented",
                                "peak_allocated_bytes": None,
                                "peak_reserved_bytes": None,
                            },
                            "all_development_gpu_processes_peak": {
                                "status": "missing_not_measured",
                                "peak_allocated_bytes": None,
                                "peak_reserved_bytes": None,
                            },
                            "measurement_complete_for_resource_release": False,
                        }
                        if model == "N3":
                            inference["n3_server_request_service"] = {
                                "status": "measured",
                                "definition": resources.N3_SERVICE_DEFINITION,
                                "seconds_total": 1e-6,
                                "nanoseconds_total": 1_000,
                                "request_count": request_count,
                                "pure_gpu_forward_time": False,
                            }
                            gpu["model_server_forward"] = {
                                "status": "missing_not_instrumented_in_n3_behavioral_runtime",
                                "peak_allocated_bytes": None,
                                "peak_reserved_bytes": None,
                            }
                        else:
                            inference["d1_rank0_inference_wrapper"] = {
                                "status": "measured",
                                "definition": resources.D1_RANK0_DEFINITION,
                                "seconds_total": 1.0,
                            }
                            inference["d1_summed_rank_forward_gpu_seconds_proxy"] = {
                                "status": "measured_proxy_not_billing",
                                "definition": resources.D1_RANK_PROXY_DEFINITION,
                                "seconds_total": 2.0,
                            }
                            gpu["model_server_rank_forward"] = {
                                "status": "measured_per_rank_per_request",
                                "simultaneous_cross_device_peak_measured": False,
                            }
                        measured = {
                            "cell_id": cell_id,
                            "model_id": model,
                            "layout_pair_id": layout,
                            "source_cell_receipt": {
                                "path": "/data/source/cell.json",
                                "sha256": "0" * 64,
                                "bytes": 1,
                            },
                            "adapter_completion": {
                                "path": "/data/source/completion.json",
                                "sha256": "1" * 64,
                                "bytes": 1,
                            },
                            "adapter_journal": {
                                "path": "/data/source/journal.jsonl",
                                "sha256": "2" * 64,
                                "bytes": 1,
                            },
                            "official_request_receipts": [
                                {
                                    "path": f"/data/source/{cell_id}/{index}.json",
                                    "sha256": f"{index % 16:x}" * 64,
                                    "bytes": 1,
                                }
                                for index in range(request_count)
                            ],
                            "episode_wall": {
                                "status": "measured",
                                "definition": resources.EPISODE_WALL_DEFINITION,
                                "start_monotonic_ns": 100,
                                "end_monotonic_ns": 1_100,
                                "elapsed_nanoseconds": 1_000,
                                "elapsed_seconds": 1e-6,
                            },
                            "inference_and_request_time": inference,
                            "gpu_memory": gpu,
                        }
                        cell = resources._attach_storage(
                            measured,
                            {
                                "recorder_cell_tree_bytes": 1,
                                "model_request_tree_bytes": 1,
                                "attributable_deduplicated_bytes": 2,
                                "recorder_cell_tree_file_count": 1,
                                "model_request_tree_file_count": 1,
                            },
                        )
                        payload = resources.pretty_bytes(cell)
                        path = root / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(payload)
                        rows.append({
                            "model_id": model,
                            "layout_pair_id": layout,
                            "cell_id": cell_id,
                            "measurement": resources.bytes_descriptor(relative, payload),
                        })
        block_roots = []
        for model, components in (
            ("N3", ("n3_joint_attempt",)),
            ("D1", ("d1_simulator_attempt", "d1_server_attempt")),
        ):
            for layout in resources.LAYOUTS:
                for component in components:
                    block_roots.append({
                        "path": str(resources.RAW_ROOT / "synthetic" / model / layout / component),
                        "model_id": model,
                        "layout_pair_id": layout,
                        "block_component": component,
                    })
        files = []
        for index in range(resources.EXPECTED_REQUESTS + resources.EXPECTED_CELLS * 3):
            scope = (
                "model_request_tree"
                if index < resources.EXPECTED_REQUESTS
                else "recorder_cell_tree"
            )
            files.append({
                "path": str(resources.RAW_ROOT / "synthetic/files" / f"{index:04d}"),
                "bytes": 1,
                "sha256": f"{index % 16:x}" * 64,
                "scope": scope,
                "model_id": "N3",
                "layout_pair_id": "D01",
                "cell_id": "synthetic-cell",
                "block_component": "n3_joint_attempt",
                "counted_once_by_storage_object": True,
            })
        inventory = resources.sign_document({
            "schema_version": resources.FILE_INVENTORY_SCHEMA,
            "study_id": resources.STUDY_ID,
            "status": "complete_immutable_snapshot",
            "definition": {
                "path_deduplication": "resolved_absolute_regular_file_path",
                "storage_object_deduplication": "st_dev_and_st_ino_within_this_single_pvc_scan",
                "mutation_check": "pre_scan_and_post_scan_metadata_plus_open_fd_hash_stability",
                "symlinks": "rejected",
                "non_regular_files": "rejected",
                "scope_priority": [
                    "model_request_tree",
                    "recorder_cell_tree",
                    "shared_block_overhead",
                ],
            },
            "raw_root": str(resources.RAW_ROOT),
            "block_roots": block_roots,
            "file_count": len(files),
            "unique_resolved_path_bytes": len(files),
            "unique_storage_object_count": len(files),
            "unique_storage_object_bytes": len(files),
            "scope_totals": {
                "model_request_tree": {
                    "file_count": resources.EXPECTED_REQUESTS,
                    "unique_path_bytes": resources.EXPECTED_REQUESTS,
                },
                "recorder_cell_tree": {
                    "file_count": resources.EXPECTED_CELLS * 3,
                    "unique_path_bytes": resources.EXPECTED_CELLS * 3,
                },
                "shared_block_overhead": {"file_count": 0, "unique_path_bytes": 0},
            },
            "files": files,
        })
        inventory_payload = resources.pretty_bytes(inventory)
        (root / "resource_file_inventory.json").write_bytes(inventory_payload)
        aggregate = resources.sign_document({
            "schema_version": resources.AGGREGATE_SCHEMA,
            "study_id": resources.STUDY_ID,
            "mode": resources.MODE,
            "status": "complete_resource_audit_with_declared_missingness",
            "formal_development_cohort_authenticated": True,
            "resource_release_gate_complete": False,
            "safe_to_release_confirmation": False,
            "counts": {
                "cells": 32,
                "source_behavioral_requests": 1152,
                "source_behavioral_actions": 14400,
                "new_model_requests": 0,
                "new_behavioral_actions": 0,
            },
            "definitions": {
                "episode_wall": resources.EPISODE_WALL_DEFINITION,
                "request_roundtrip": resources.REQUEST_ROUNDTRIP_DEFINITION,
                "n3_server_service": resources.N3_SERVICE_DEFINITION,
                "d1_rank0_wrapper": resources.D1_RANK0_DEFINITION,
                "d1_summed_rank_proxy": resources.D1_RANK_PROXY_DEFINITION,
                "d1_offline_decode": resources.D1_DECODE_DEFINITION,
                "raw_attributable_bytes": resources.RAW_ATTRIBUTABLE_DEFINITION,
                "gpu_allocator_peak": resources.GPU_PEAK_DEFINITION,
            },
            "inputs": {},
            "science_activity": resources.SCIENCE_COUNTS,
            "cell_measurements": rows,
            "resource_file_inventory": resources.bytes_descriptor(
                "resource_file_inventory.json", inventory_payload
            ),
            "models": {
                model: {
                    "cell_count": 16,
                    "request_count": resources.MODEL_REQUEST_COUNTS[model],
                    "behavioral_action_count": 7200,
                    "resource_release_complete": False,
                }
                for model in resources.MODELS
            },
            "annotation_time": resources._missing_annotation(),
            "operator_release_fields": resources.EXECUTION_AUTHORIZATION,
            "missing_release_requirements": list(
                resources.MISSING_RELEASE_REQUIREMENTS
            ),
            "claim_boundary": "test retained evidence only",
        })
        aggregate_payload = resources.pretty_bytes(aggregate)
        (root / "resource_aggregate.json").write_bytes(aggregate_payload)
        receipt = resources.sign_document({
            "schema_version": resources.COMPILER_RECEIPT_SCHEMA,
            "study_id": resources.STUDY_ID,
            "mode": resources.MODE,
            "status": "compiled_complete_with_declared_missingness",
            "input_manifest": {},
            "resource_compiler": {},
            "resource_contract": {},
            "resource_release_gate_complete": False,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "behavioral_policy_skill_evaluated": False,
            "science_activity": resources.SCIENCE_COUNTS,
            "outputs": {
                "resource_aggregate": resources.bytes_descriptor(
                    "resource_aggregate.json", aggregate_payload
                ),
                "resource_file_inventory": resources.bytes_descriptor(
                    "resource_file_inventory.json", inventory_payload
                ),
                "cell_measurement_count": 32,
                "cell_measurements": rows,
            },
            "counts": {
                "authenticated_cells": 32,
                "authenticated_source_behavioral_requests": 1152,
                "authenticated_source_behavioral_actions": 14400,
                "inventoried_raw_files": len(files),
                "inventoried_unique_resolved_path_bytes": len(files),
            },
            "claim_boundary": "test retained evidence only",
        })
        (root / "resource_compiler_receipt.json").write_bytes(
            resources.pretty_bytes(receipt)
        )
        return aggregate, receipt

    def test_bundle_validation_accepts_exact_missingness_and_rejects_release_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_valid_bundle(root)
            validated = resources.validate_resource_bundle(root)
            aggregate = resources.load_json(
                root / "resource_aggregate.json", "resource aggregate"
            )
            authorization = aggregate["operator_release_fields"]
            self.assertEqual(authorization["selected_execution_host"], "GM cluster")
            self.assertEqual(authorization["authorized_block_counts"]["maximum_core"], 58)
            self.assertEqual(
                authorization["authorized_behavioral_cell_counts"]["maximum_core"], 232
            )
            self.assertFalse(authorization["user_authorization_pending"])
            self.assertIsNone(authorization["measured_safe_parallel_blocks_by_model"])
            self.assertFalse(validated["safe_to_release_confirmation"])
            path = root / "resource_aggregate.json"
            changed = resources.load_json(path, "aggregate")
            changed.pop("payload_sha256")
            changed["safe_to_release_confirmation"] = True
            path.write_bytes(resources.pretty_bytes(resources.sign_document(changed)))
            with self.assertRaises(resources.ResourceCompilerError):
                resources.validate_resource_bundle(root)

    def test_bundle_validation_rejects_nonregular_extra_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_valid_bundle(root)
            os.mkfifo(root / "unexpected-fifo")
            with self.assertRaisesRegex(resources.ResourceCompilerError, "non-regular"):
                resources.validate_resource_bundle(root)

    def test_json_loader_rejects_duplicate_keys_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text('{"x":1,"x":2}', encoding="utf-8")
            with self.assertRaisesRegex(resources.ResourceCompilerError, "duplicate JSON key"):
                resources.load_json(path, "bad")
            path.write_text('{"x":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(resources.ResourceCompilerError, "non-finite"):
                resources.load_json(path, "bad")

    def test_manifest_requires_exact_hash_fields_and_nonsymlink_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary).resolve()
            manifest = {
                "schema_version": resources.INPUT_SCHEMA,
                "study_id": resources.STUDY_ID,
                "mode": resources.MODE,
                "raw_root": str(raw),
                "planned_cells": {},
                "timing_sidecar_receipts": [],
                "aggregate_receipts": [],
                "d1_server_receipts": [],
                "queue_wrapper_snapshots": [],
            }
            path = write_json(raw / "manifest.json", manifest)
            digest = resources.sha256_file(path)
            with mock.patch.object(resources, "RAW_ROOT", raw):
                observed, resolved = resources._validate_manifest(path, digest)
                self.assertEqual(observed, manifest)
                self.assertEqual(resolved, path)
                with self.assertRaisesRegex(resources.ResourceCompilerError, "hash changed"):
                    resources._validate_manifest(path, "0" * 64)
                manifest["unexpected"] = True
                changed = write_json(raw / "changed.json", manifest)
                with self.assertRaisesRegex(resources.ResourceCompilerError, "fields changed"):
                    resources._validate_manifest(changed, resources.sha256_file(changed))
                link = raw / "link.json"
                link.symlink_to(path)
                with self.assertRaisesRegex(resources.ResourceCompilerError, "symlink"):
                    resources._validate_manifest(link, digest)


if __name__ == "__main__":
    unittest.main()
