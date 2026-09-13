from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/forecast_layout/development_evidence_compiler_jobs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "development_evidence_compiler_jobs", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
jobs = importlib.util.module_from_spec(SPEC)
import sys

sys.modules[SPEC.name] = jobs
SPEC.loader.exec_module(jobs)


class DevelopmentEvidenceCompilerJobTests(unittest.TestCase):
    def _aggregates(self) -> dict[tuple[str, str], dict]:
        result = {}
        index = 1
        for model in jobs.MODELS:
            for layout in jobs.LAYOUTS:
                spec = jobs.aggregate_support.AGGREGATES[model][layout]
                result[(model, layout)] = {
                    "aggregate_receipt": {
                        "path": str(spec.receipt_path),
                        "sha256": f"{index:064x}",
                        "bytes": 12000 + index,
                    },
                    "cell_receipts": [
                        {
                            "path": f"/data/cell-{index}-{cell}.json",
                            "sha256": f"{index + cell + 16:064x}",
                            "bytes": 70000 + cell,
                        }
                        for cell in range(4)
                    ],
                }
                index += 1
        return result

    def _aggregate_inputs(self) -> dict[tuple[str, str], tuple[Path, str]]:
        return {
            key: (Path(f"/fetched/{key[0]}-{key[1]}.json"), value["aggregate_receipt"]["sha256"])
            for key, value in self._aggregates().items()
        }

    def _implementation(self) -> dict[str, dict]:
        return jobs._local_implementation()

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
            duplicates.extend(
                (node.lineno, key) for key in set(keys) if keys.count(key) > 1
            )
        self.assertEqual(duplicates, [])

    def test_contract_is_exact_formal_only_and_confirmation_closed(self) -> None:
        path = jobs.REPOSITORY_ROOT / jobs.CONTRACT_RELATIVE
        identity = jobs.queue.file_identity(path)
        observed = jobs._validate_contract(path, identity["sha256"])
        self.assertEqual(observed, identity)
        self.assertEqual(
            jobs.queue.load_json(path, "contract"), jobs._expected_contract()
        )
        contract = jobs._expected_contract()
        self.assertEqual(contract["compiler"]["supported_production_modes"], ["formal_full"])
        self.assertIs(contract["output_policy"]["diagnostic_release_capable"], False)
        self.assertIs(contract["safe_to_release_confirmation"], False)
        self.assertEqual(contract["science_counts"], jobs._zero_science_counts())

    def test_builder_requires_exactly_eight_explicit_receipts(self) -> None:
        with self.assertRaisesRegex(
            jobs.DevelopmentCompilerQueueError, "exactly eight explicit"
        ):
            jobs.validate_local_aggregate_inputs({})

        incomplete = self._aggregate_inputs()
        incomplete.pop(("D1", "D04"))
        with mock.patch.object(jobs, "_local_implementation", return_value=self._implementation()):
            with self.assertRaisesRegex(
                jobs.DevelopmentCompilerQueueError, "exactly eight explicit"
            ):
                jobs.build_formal_wave(
                    study_commit="a" * 40, aggregate_inputs=incomplete
                )

    def test_explicit_receipts_cannot_collapse_to_an_ambiguous_path(self) -> None:
        evidence = {
            "aggregate_receipt": {
                "path": "/data/duplicate-aggregate.json",
                "sha256": "a" * 64,
                "bytes": 12000,
            },
            "cell_receipts": [{"path": "/data/cell", "sha256": "b" * 64, "bytes": 1}] * 4,
        }
        with mock.patch.object(
            jobs.aggregate_support,
            "validate_local_aggregate_receipt",
            return_value=evidence,
        ):
            with self.assertRaisesRegex(
                jobs.DevelopmentCompilerQueueError, "paths are ambiguous"
            ):
                jobs.validate_local_aggregate_inputs(self._aggregate_inputs())

    def test_formal_wave_is_deterministic_hash_bound_and_zero_science(self) -> None:
        aggregates = self._aggregates()
        implementation = self._implementation()
        inputs = self._aggregate_inputs()
        with mock.patch.object(
            jobs, "_local_implementation", return_value=implementation
        ), mock.patch.object(
            jobs, "validate_local_aggregate_inputs", return_value=aggregates
        ):
            first = jobs.build_formal_wave(
                study_commit="b" * 40, aggregate_inputs=inputs
            )
            second = jobs.build_formal_wave(
                study_commit="b" * 40, aggregate_inputs=inputs
            )
        self.assertEqual(first, second)
        self.assertEqual(first["mode"], "formal_full")
        self.assertEqual(first["science_counts"], jobs._zero_science_counts())
        self.assertIs(first["safe_to_release_confirmation"], False)
        self.assertIs(first["diagnostic_release_capable"], False)
        self.assertEqual(len(first["aggregate_receipts"]), 8)
        descriptor = first["jobs"][0]
        self.assertEqual(descriptor["job_id"], jobs.JOB_ID)
        self.assertEqual(descriptor["role"], jobs.WORKER_ROLE)
        command = " ".join(descriptor["argv"])
        self.assertIn("formal-full", command)
        self.assertNotIn("diagnostic", command)
        for identity in implementation.values():
            self.assertIn(identity["sha256"], command)
        for evidence in aggregates.values():
            aggregate = evidence["aggregate_receipt"]
            self.assertIn(aggregate["sha256"], command)
            self.assertIn(str(aggregate["bytes"]), command)

    def test_runtime_descriptor_reconstructs_builder_and_rejects_wrong_role(self) -> None:
        aggregates = self._aggregates()
        implementation = self._implementation()
        expected = jobs._job_descriptor(
            study_commit="c" * 40,
            implementation=implementation,
            aggregates=aggregates,
        )
        args = jobs._parser().parse_args(expected["argv"][2:])
        observed, observed_implementation, observed_aggregates = jobs._runtime_descriptor(args)
        self.assertEqual(observed, expected)
        self.assertEqual(
            observed_implementation["compiler"]["sha256"],
            implementation["compiler"]["sha256"],
        )
        self.assertEqual(
            observed_aggregates[("D1", "D04")]["aggregate_receipt"],
            aggregates[("D1", "D04")]["aggregate_receipt"],
        )

        wrong = argparse.Namespace(**vars(args))
        wrong.expected_role = "wmf-forecast-0912-worker-06"
        with self.assertRaisesRegex(
            jobs.DevelopmentCompilerQueueError, "worker role changed"
        ):
            jobs._runtime_descriptor(wrong)

        wrong_hash = argparse.Namespace(**vars(args))
        wrong_hash.planned_cells_sha256 = "f" * 64
        with self.assertRaisesRegex(
            jobs.DevelopmentCompilerQueueError, "planned-cell hash changed"
        ):
            jobs._runtime_descriptor(wrong_hash)

    def test_runtime_uses_queue_context_for_descriptor_claim_role_and_pod(self) -> None:
        source = inspect_source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("queue.validate_queue_context(", source)
        tree = ast.parse(inspect_source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "validate_queue_context"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg for item in calls[0].keywords}
        self.assertEqual(
            keywords,
            {
                "source_root",
                "job_dir",
                "study_commit",
                "job_id",
                "expected_role",
                "expected_descriptor",
            },
        )
        # The shared validator is the source of the strict owner/hostname/POD_UID
        # checks; ensure this staged dependency has not lost any of them.
        queue_source = (jobs.REPOSITORY_ROOT / jobs.QUEUE_RELATIVE).read_text(
            encoding="utf-8"
        )
        self.assertIn('claim.get("worker_id") == worker_id', queue_source)
        self.assertIn('os.environ.get("POD_UID", "")', queue_source)
        self.assertIn("queue claim does not bind the exact descriptor", queue_source)
        self.assertIn("runtime hostname does not match worker identity", queue_source)

    def test_queue_context_actually_rejects_missing_pod_and_wrong_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "checkout"
            checkout.mkdir()
            subprocess.run(["git", "init", "-q", str(checkout)], check=True)
            subprocess.run(
                ["git", "-C", str(checkout), "config", "user.email", "test@example.invalid"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(checkout), "config", "user.name", "Queue Test"],
                check=True,
            )
            (checkout / "source.txt").write_text("immutable\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(checkout), "add", "source.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(checkout), "commit", "-q", "-m", "fixture"],
                check=True,
            )
            commit = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            state = root / "control"
            source_root = state / "sources" / commit
            source_root.parent.mkdir(parents=True)
            shutil.move(str(checkout), source_root)
            job_dir = state / "jobs" / jobs.JOB_ID
            (job_dir / "claim").mkdir(parents=True)
            descriptor = jobs._job_descriptor(
                study_commit=commit,
                implementation=self._implementation(),
                aggregates=self._aggregates(),
            )
            (job_dir / "descriptor.json").write_bytes(
                jobs.queue.canonical_bytes(jobs.queue._normalized_descriptor(descriptor))
            )
            descriptor_identity = jobs.queue.file_identity(job_dir / "descriptor.json")

            def write_claim(worker_id: str) -> None:
                (job_dir / "claim" / "owner.json").write_bytes(
                    jobs.queue.canonical_bytes(
                        {
                            "worker_id": worker_id,
                            "worker_pid": 1,
                            "descriptor_sha256": descriptor_identity["sha256"],
                            "release_boundary": "claim_committed_under_shared_release_lock",
                            "control_generation": 1,
                            "control_commit": "d" * 40,
                        }
                    )
                )

            write_claim(jobs.WORKER_ROLE)
            with mock.patch.object(
                jobs.queue.socket,
                "gethostname",
                return_value=jobs.WORKER_ROLE + "-fixture",
            ), mock.patch.dict(os.environ, {"POD_UID": ""}):
                with self.assertRaisesRegex(
                    jobs.queue.TimingQueueError, "POD_UID is missing"
                ):
                    jobs.queue.validate_queue_context(
                        source_root=source_root,
                        job_dir=job_dir,
                        study_commit=commit,
                        job_id=jobs.JOB_ID,
                        expected_role=jobs.WORKER_ROLE,
                        expected_descriptor=descriptor,
                    )

            write_claim("wmf-forecast-0912-worker-06")
            with mock.patch.object(
                jobs.queue.socket,
                "gethostname",
                return_value=jobs.WORKER_ROLE + "-fixture",
            ), mock.patch.dict(os.environ, {"POD_UID": "pod-fixture"}):
                with self.assertRaisesRegex(
                    jobs.queue.TimingQueueError, "claim worker identity changed"
                ):
                    jobs.queue.validate_queue_context(
                        source_root=source_root,
                        job_dir=job_dir,
                        study_commit=commit,
                        job_id=jobs.JOB_ID,
                        expected_role=jobs.WORKER_ROLE,
                        expected_descriptor=descriptor,
                    )

            write_claim(jobs.WORKER_ROLE)
            with mock.patch.object(
                jobs.queue.socket,
                "gethostname",
                return_value=jobs.WORKER_ROLE + "-fixture",
            ), mock.patch.dict(os.environ, {"POD_UID": "pod-fixture"}):
                context = jobs.queue.validate_queue_context(
                    source_root=source_root,
                    job_dir=job_dir,
                    study_commit=commit,
                    job_id=jobs.JOB_ID,
                    expected_role=jobs.WORKER_ROLE,
                    expected_descriptor=descriptor,
                )
            self.assertEqual(context.worker_id, jobs.WORKER_ROLE)
            self.assertEqual(context.pod_uid, "pod-fixture")

    def test_manifest_is_formal_sorted_and_uses_exact_aggregate_descriptors(self) -> None:
        aggregates = self._aggregates()
        manifest = jobs._compiler_manifest(
            source_root=jobs.REPOSITORY_ROOT, aggregates=aggregates
        )
        self.assertEqual(manifest["mode"], "formal_full")
        self.assertEqual(manifest["raw_root"], str(jobs.RAW_ROOT))
        self.assertEqual(manifest["planned_cells"]["sha256"], jobs.PLANNED_CELLS_SHA256)
        self.assertEqual(
            [
                (row["model_id"], row["layout_pair_id"])
                for row in manifest["aggregate_receipts"]
            ],
            [(model, layout) for model in jobs.MODELS for layout in jobs.LAYOUTS],
        )
        self.assertEqual(
            manifest["aggregate_receipts"][-1]["receipt"],
            aggregates[("D1", "D04")]["aggregate_receipt"],
        )

    def test_staged_implementation_rehashes_contract_compiler_and_freeze(self) -> None:
        expected = self._implementation()
        observed, compiler, timing, annotation = jobs._validate_staged_implementation(
            source_root=jobs.REPOSITORY_ROOT, expected=expected
        )
        self.assertEqual(observed, expected)
        self.assertEqual(compiler.INPUT_SCHEMA, jobs.COMPILER_INPUT_SCHEMA)
        self.assertEqual(timing.REQUEST_INVENTORY_SCHEMA, compiler.TIMING_INVENTORY_SCHEMA)
        self.assertEqual(annotation.ACTION_MANIFEST_SCHEMA, compiler.ACTION_MANIFEST_SCHEMA)

        wrong = {name: dict(value) for name, value in expected.items()}
        wrong["compiler"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(
            jobs.DevelopmentCompilerQueueError, "staged compiler hash changed"
        ):
            jobs._validate_staged_implementation(
                source_root=jobs.REPOSITORY_ROOT, expected=wrong
            )

    def test_context_paths_are_exactly_bound_to_control_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "control"
            commit = "a" * 40
            exact = SimpleNamespace(
                study_commit=commit,
                job_dir=control / "jobs" / jobs.JOB_ID,
                source_root=control / "sources" / commit,
            )
            with mock.patch.object(jobs, "CONTROL_ROOT", control):
                jobs._exact_context_paths(exact)
                with self.assertRaisesRegex(
                    jobs.DevelopmentCompilerQueueError, "exact control-root job"
                ):
                    jobs._exact_context_paths(
                        SimpleNamespace(
                            **{
                                **vars(exact),
                                "job_dir": control / "other" / jobs.JOB_ID,
                            }
                        )
                    )
                with self.assertRaisesRegex(
                    jobs.DevelopmentCompilerQueueError, "exact control-root checkout"
                ):
                    jobs._exact_context_paths(
                        SimpleNamespace(
                            **{
                                **vars(exact),
                                "source_root": control / "sources" / ("b" * 40),
                            }
                        )
                    )

    def _write_bundle_skeleton(self, bundle: Path) -> None:
        for relative in jobs._expected_bundle_paths() - {"compiler_receipt.json"}:
            path = bundle / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

    def _relative_descriptor(self, bundle: Path, relative: str) -> dict:
        identity = jobs.queue.file_identity(bundle / relative)
        return {
            "path": relative,
            "sha256": identity["sha256"],
            "bytes": identity["bytes"],
        }

    def _compiler_receipt(
        self,
        *,
        bundle: Path,
        manifest_identity: dict,
        implementation: dict,
        aggregates: dict,
        unsafe_confirmation: bool = False,
    ) -> dict:
        model_outputs = {}
        for model in jobs.MODELS:
            lower = model.lower()
            cells = []
            for layout in jobs.LAYOUTS:
                for cell_id in jobs.aggregate_support.AGGREGATES[model][layout].cell_ids:
                    cells.append(
                        {
                            "cell_id": cell_id,
                            "official_request_count": jobs.aggregate_support.REQUESTS_PER_CELL[model],
                            "action_manifest": self._relative_descriptor(
                                bundle, f"cells/{lower}/{cell_id}/action_manifest.json"
                            ),
                            "recording_receipt": self._relative_descriptor(
                                bundle, f"cells/{lower}/{cell_id}/recording_receipt.json"
                            ),
                        }
                    )
            model_outputs[model] = {
                "complete": True,
                "cell_count": 16,
                "request_count": jobs.MODEL_REQUESTS[model],
                "action_count": jobs.MODEL_ACTIONS[model],
                "aggregate_receipts": [
                    aggregates[(model, layout)]["aggregate_receipt"]
                    for layout in jobs.LAYOUTS
                ],
                "timing_request_inventory": self._relative_descriptor(
                    bundle, f"{lower}_development_timing_request_inventory.json"
                ),
                "request_provenance": self._relative_descriptor(
                    bundle, f"{lower}_development_request_provenance.json"
                ),
                "freeze_cell_evidence": self._relative_descriptor(
                    bundle, f"{lower}_development_freeze_cells.json"
                ),
                "cells": cells,
            }
        return jobs.queue.signed_document(
            {
                "schema_version": jobs.COMPILER_RECEIPT_SCHEMA,
                "study_id": jobs.STUDY_ID,
                "mode": jobs.MODE,
                "status": "compiled_complete",
                "formal_cohort_complete": True,
                "safe_for_timing_binding": True,
                "safe_for_confirmation_release": unsafe_confirmation,
                "input_manifest": manifest_identity,
                "planned_cells": jobs.queue.file_identity(
                    jobs.REPOSITORY_ROOT / jobs.PLANNED_CELLS_RELATIVE
                ),
                "compiler_source": {
                    "path": str(jobs.COMPILER_RELATIVE),
                    "sha256": implementation["compiler"]["sha256"],
                    "bytes": implementation["compiler"]["bytes"],
                },
                "freeze_validator_dependency": {
                    "path": str(jobs.FREEZE_RELATIVE),
                    "sha256": implementation["freeze_validator"]["sha256"],
                    "bytes": implementation["freeze_validator"]["bytes"],
                },
                "raw_root": str(jobs.RAW_ROOT),
                "camera_id": jobs.CAMERA_ID,
                "models": model_outputs,
                "counts": {
                    "compiled_cells": jobs.EXPECTED_CELLS,
                    "compiled_source_behavioral_requests": jobs.EXPECTED_REQUESTS,
                    "compiled_source_behavioral_actions": jobs.EXPECTED_ACTIONS,
                },
                "compiler_science_activity": jobs._compiler_zero_science(),
                "unsupported_outputs": {
                    "resource_metrics": "not_synthesized",
                    "camera_crop_contract": "not_created",
                    "human_pixel_blindness_receipt": "not_created",
                    "confirmation_release": "not_created",
                    "labels": "not_created",
                    "movement_threshold": "not_created",
                },
            }
        )

    def _write_semantic_bundle(self, bundle: Path) -> tuple[dict, dict, dict]:
        """Write a full output-level fixture; only upstream raw replay is mocked."""

        _, compiler, _, annotation = jobs._validate_staged_implementation(
            source_root=jobs.REPOSITORY_ROOT,
            expected=self._implementation(),
        )
        bundle.mkdir()
        model_outputs = {}
        freeze_replays = {}
        source_counter = 1
        for model in jobs.MODELS:
            lower = model.lower()
            request_count = compiler.MODEL_LIMITS[model]["request_count"]
            prefix = compiler.MODEL_LIMITS[model]["executed_prefix"]
            cell_ids = sorted(
                cell_id
                for layout in jobs.LAYOUTS
                for cell_id in jobs.aggregate_support.AGGREGATES[model][layout].cell_ids
            )
            roster = []
            provenance_requests = []
            timing_entries = []
            freeze_entries = []
            cell_outputs = []
            for cell_id in cell_ids:
                parts = cell_id.split("__")
                layout = parts[2]
                condition = f"{parts[-2]}_{parts[-1]}"
                recording_id = f"attempt-{source_counter:04d}"
                source_sha = f"{source_counter:064x}"
                action_relative = f"cells/{lower}/{cell_id}/action_manifest.json"
                recording_relative = f"cells/{lower}/{cell_id}/recording_receipt.json"
                action_path = bundle / action_relative
                recording_path = bundle / recording_relative
                action_path.parent.mkdir(parents=True)
                action_manifest = annotation.sign_document(
                    {
                        "schema_version": compiler.ACTION_MANIFEST_SCHEMA,
                        "study_id": jobs.STUDY_ID,
                        "cell_id": cell_id,
                        "recording_id": recording_id,
                        "model_id": model,
                        "executed_action_count": 450,
                        "actions": [
                            {
                                "action_index": action_index,
                                "request_index": min(action_index // prefix, request_count - 1),
                                "executed_action_sha256": f"{action_index + 1:064x}",
                                "control_timestamp": "2026-09-13T10:00:00Z",
                                "physics_step_id": f"physics-step:{action_index + 1}",
                                "camera_frame_id": f"camera-frame:{action_index + 1}",
                            }
                            for action_index in range(450)
                        ],
                    }
                )
                action_path.write_text(
                    json.dumps(action_manifest, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                action_descriptor = self._relative_descriptor(bundle, action_relative)
                recording_receipt = annotation.sign_document(
                    {
                        "schema_version": compiler.RECORDING_RECEIPT_SCHEMA,
                        "study_id": jobs.STUDY_ID,
                        "receipt_id": f"recording_{source_sha[:24]}",
                        "stage": "development",
                        "cell_id": cell_id,
                        "recording_id": recording_id,
                        "model_id": model,
                        "layout_pair_id": layout,
                        "condition_id": condition,
                        "recording_status": "valid_complete",
                        "executed_action_count": 450,
                        "censor_reason": None,
                        "source_video_id": f"video_{source_sha[:24]}",
                        "source_video_sha256": source_sha,
                        "action_manifest_path": "action_manifest.json",
                        "action_manifest_sha256": action_descriptor["sha256"],
                    }
                )
                recording_path.write_text(
                    json.dumps(recording_receipt, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                recording_descriptor = self._relative_descriptor(
                    bundle, recording_relative
                )
                source_cell = {
                    "path": f"/raw/{cell_id}/cell_receipt.json",
                    "sha256": source_sha,
                    "bytes": 1000 + source_counter,
                }
                completion = {
                    "path": f"/raw/{cell_id}/completion.json",
                    "sha256": f"{source_counter + 100:064x}",
                    "bytes": 2000 + source_counter,
                }
                journal = {
                    "path": f"/raw/{cell_id}/journal.jsonl",
                    "sha256": f"{source_counter + 200:064x}",
                    "bytes": 3000 + source_counter,
                    "event_count": 2000,
                    "tail_sha256": f"{source_counter + 300:064x}",
                }
                official_requests = [
                    {
                        "path": f"/raw/{cell_id}/request-{index:04d}.json",
                        "sha256": (
                            f"{source_counter:016x}{index:016x}" + "a" * 32
                        ),
                        "bytes": 4000 + index,
                    }
                    for index in range(request_count)
                ]
                freeze_entry = {
                    "cell_receipt": source_cell,
                    "server_request_receipts": official_requests,
                    "resource_receipt": None,
                }
                freeze_entries.append(freeze_entry)
                freeze_replays[cell_id] = {
                    "entry": freeze_entry,
                    "validated": {
                        "request_receipt_sha256s": [
                            row["sha256"] for row in official_requests
                        ],
                        "adapter_completion": completion,
                        "adapter_journal": journal,
                    },
                }
                source_video = {
                    "path": f"/raw/{cell_id}/viewport.mp4",
                    "sha256": source_sha,
                    "bytes": 5000 + source_counter,
                }
                roster.append(
                    {
                        "cell_id": cell_id,
                        "recording_id": recording_id,
                        "model_id": model,
                        "layout_pair_id": layout,
                        "condition_id": condition,
                        "recording_status": "valid_complete",
                        "executed_action_count": 450,
                        "censor_reason": None,
                        "recording_receipt_path": recording_relative,
                        "recording_receipt_sha256": recording_descriptor["sha256"],
                        "action_manifest_path": action_relative,
                        "action_manifest_sha256": action_descriptor["sha256"],
                        "source_video_id": f"video_{source_sha[:24]}",
                        "source_video_sha256": source_sha,
                    }
                )
                pins = compiler.MODEL_PINS[model]
                source_pins = {
                    "study_commit": "a" * 40,
                    "robolab_commit": pins["robolab_commit"],
                }
                if model == "N3":
                    source_pins["cosmos_commit"] = pins["cosmos_commit"]
                    source_fragment = f"cosmos:{pins['cosmos_commit']}"
                else:
                    source_pins.update(
                        {
                            "dreamzero_commit": pins["dreamzero_commit"],
                            "dreamzero_tree": pins["dreamzero_tree"],
                        }
                    )
                    source_fragment = f"dreamzero:{pins['dreamzero_commit']}"
                pose_sha = f"{source_counter + 400:064x}"
                model_identity = {
                    "source_pins": source_pins,
                    "checkpoint_pin": {
                        "revision": pins["checkpoint_revision"],
                        "aggregate_sha256": pins["checkpoint_aggregate_sha256"],
                    },
                    "source_identity": (
                        f"study:{'a' * 40};robolab:{pins['robolab_commit']};"
                        f"{source_fragment};pose:{pose_sha}"
                    ),
                    "checkpoint_identity": (
                        f"revision:{pins['checkpoint_revision']};"
                        f"aggregate:{pins['checkpoint_aggregate_sha256']}"
                    ),
                    "pose_manifest_sha256": pose_sha,
                }
                reset_binding = {
                    "role": "context_reset",
                    "payload_sha256": f"{source_counter + 500:064x}",
                    "array_count": 0,
                    "structure_sha256": f"{source_counter + 600:064x}",
                    "artifact": None,
                    "artifact_base": f"/raw/{cell_id}",
                }
                model_context = (
                    {
                        "server_context_id": f"context-{source_counter}",
                        "server_begin_receipt": {"passed": True},
                        "server_end_receipt": {"passed": True},
                        "recorder_context_reset": reset_binding,
                    }
                    if model == "N3"
                    else {
                        "server_ready": {"path": "/raw/ready"},
                        "runtime_identity": {"path": "/raw/runtime"},
                        "server_contract": {"path": "/raw/contract"},
                        "simulator_claim": {"path": "/raw/claim"},
                        "server_context_id": f"context-{source_counter}",
                        "server_reset_receipt": {"path": "/raw/reset"},
                        "server_episode_manifest": {"path": "/raw/episode"},
                        "recorder_context_reset": reset_binding,
                    }
                )
                for request_index, official in enumerate(official_requests):
                    start = request_index * prefix
                    preceding = None if request_index == 0 else start - 1

                    def observation(control_step: int) -> dict:
                        return {
                            "observation_id": f"obs_{control_step:06d}",
                            "control_step": control_step,
                            "physics_step": control_step + 10,
                            "physics_time_s": float(control_step) / 60.0,
                            "camera_frame_native_id": control_step + 20,
                            "camera_frame_id": (
                                f"{jobs.CAMERA_ID}:native-int:{control_step + 20}"
                            ),
                            "camera_capture_time_ns": control_step * 16_000_000,
                            "camera_timestamp_source": "fixture_native_clock",
                            "payload_sha256": f"{source_counter + request_index + 700:064x}",
                            "payload_artifact": {
                                "path": f"/raw/{cell_id}/obs-{control_step}.npz",
                                "sha256": f"{source_counter + request_index + 800:064x}",
                                "bytes": 6000,
                            },
                        }

                    request = {
                        "source_request_id": "request_" + official["sha256"][:32],
                        "cell_id": cell_id,
                        "recording_id": recording_id,
                        "model_id": model,
                        "layout_pair_id": layout,
                        "condition_id": condition,
                        "request_index": request_index,
                        "action_step_start": start,
                        "executed_prefix_actions": min(prefix, 450 - start),
                        "current_observation_id": f"obs_{start:06d}",
                        "preceding_observation_id": (
                            None if preceding is None else f"obs_{preceding:06d}"
                        ),
                        "history_mode": (
                            "persistence_at_initial_request"
                            if request_index == 0
                            else "preceding_observation"
                        ),
                        "current_observation": observation(start),
                        "preceding_observation": (
                            None if preceding is None else observation(preceding)
                        ),
                        "official_request_receipt": official,
                        "recorder_model_request": {
                            "role": "model_request",
                            "payload_sha256": f"{source_counter + request_index + 900:064x}",
                            "array_count": 0,
                            "structure_sha256": f"{source_counter + request_index + 1000:064x}",
                            "artifact": None,
                            "artifact_base": f"/raw/{cell_id}",
                        },
                        "recorder_response_payload_sha256": (
                            f"{source_counter + request_index + 1100:064x}"
                        ),
                        "adapter_completion": completion,
                        "adapter_journal": journal,
                        "source_video_id": f"video_{source_sha[:24]}",
                        "source_video": source_video,
                        "model_output_or_action_modified": False,
                        "model_identity": model_identity,
                        "model_context": model_context,
                        "action_manifest": action_descriptor,
                        "recording_receipt": recording_descriptor,
                    }
                    provenance_requests.append(request)
                    timing_entries.append(
                        {
                            "cell_id": cell_id,
                            "request_index": request_index,
                            "request_receipt": official,
                            "adapter_completion": completion,
                            "adapter_journal": journal,
                        }
                    )
                cell_outputs.append(
                    {
                        "cell_id": cell_id,
                        "source_cell_receipt": source_cell,
                        "action_manifest": action_descriptor,
                        "recording_receipt": recording_descriptor,
                        "official_request_count": request_count,
                        "source_video": source_video,
                    }
                )
                source_counter += 1

            timing_document = {
                "schema_version": compiler.TIMING_INVENTORY_SCHEMA,
                "study_id": jobs.STUDY_ID,
                "model_id": model,
                "request_receipts": timing_entries,
            }
            provenance_document = compiler.sign_document(
                {
                    "schema_version": compiler.PROVENANCE_SCHEMA,
                    "study_id": jobs.STUDY_ID,
                    "mode": jobs.MODE,
                    "model_id": model,
                    "stage": "development",
                    "status": "complete",
                    "safe_for_formal_release": True,
                    "annotation_state": "not_started",
                    "camera_id": jobs.CAMERA_ID,
                    "camera_crop_contract": None,
                    "resource_measurements": None,
                    "labels": None,
                    "episode_roster": roster,
                    "requests": provenance_requests,
                }
            )
            freeze_document = compiler.sign_document(
                {
                    "schema_version": compiler.FREEZE_FRAGMENT_SCHEMA,
                    "study_id": jobs.STUDY_ID,
                    "mode": jobs.MODE,
                    "model_id": model,
                    "status": "complete",
                    "safe_for_alignment_input": True,
                    "resource_receipts_synthesized": False,
                    "development_cells": freeze_entries,
                }
            )
            for relative, document in (
                (f"{lower}_development_timing_request_inventory.json", timing_document),
                (f"{lower}_development_request_provenance.json", provenance_document),
                (f"{lower}_development_freeze_cells.json", freeze_document),
            ):
                (bundle / relative).write_text(
                    json.dumps(document, sort_keys=True) + "\n", encoding="utf-8"
                )
            model_outputs[model] = {"cells": cell_outputs}
        (bundle / "compiler_receipt.json").write_text("{}\n", encoding="utf-8")
        bundle_files = {
            relative: jobs.queue.file_identity(bundle / relative)
            for relative in jobs._expected_bundle_paths()
        }
        return {"models": model_outputs}, bundle_files, freeze_replays

    def test_compiler_receipt_and_bundle_are_strictly_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            self._write_bundle_skeleton(bundle)
            manifest = root / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            manifest_identity = jobs.queue.file_identity(manifest)
            implementation = self._implementation()
            _, compiler, _, _ = jobs._validate_staged_implementation(
                source_root=jobs.REPOSITORY_ROOT, expected=implementation
            )
            aggregates = self._aggregates()
            receipt = self._compiler_receipt(
                bundle=bundle,
                manifest_identity=manifest_identity,
                implementation=implementation,
                aggregates=aggregates,
            )
            (bundle / "compiler_receipt.json").write_bytes(
                jobs.queue.canonical_bytes(receipt)
            )
            observed = jobs._validate_compiler_receipt(
                compiler=compiler,
                returned=receipt,
                bundle=bundle,
                expected_bundle=bundle,
                manifest_identity=manifest_identity,
                implementation=implementation,
                aggregates=aggregates,
            )
            self.assertEqual(observed[0], receipt)
            self.assertEqual(len(observed[2]), jobs.EXPECTED_BUNDLE_FILES)

            dependency_mutations = {
                "path": "workshops/corl2026_world_models/analysis/other.py",
                "sha256": "f" * 64,
                "bytes": implementation["freeze_validator"]["bytes"] + 1,
            }
            for field, changed in dependency_mutations.items():
                with self.subTest(freeze_dependency_field=field):
                    wrong_dependency = dict(receipt)
                    wrong_dependency.pop("payload_sha256")
                    wrong_dependency["freeze_validator_dependency"] = {
                        **wrong_dependency["freeze_validator_dependency"],
                        field: changed,
                    }
                    wrong_dependency = jobs.queue.signed_document(wrong_dependency)
                    (bundle / "compiler_receipt.json").write_bytes(
                        jobs.queue.canonical_bytes(wrong_dependency)
                    )
                    with self.assertRaisesRegex(
                        jobs.DevelopmentCompilerQueueError,
                        "freeze-validator dependency changed",
                    ):
                        jobs._validate_compiler_receipt(
                            compiler=compiler,
                            returned=wrong_dependency,
                            bundle=bundle,
                            expected_bundle=bundle,
                            manifest_identity=manifest_identity,
                            implementation=implementation,
                            aggregates=aggregates,
                        )

            extra = bundle / "unexpected.json"
            extra.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                jobs.DevelopmentCompilerQueueError, "file inventory changed"
            ):
                jobs._inventory_bundle(bundle, expected_bundle=bundle)

    def test_bundle_root_rejects_lexical_aliases_and_symlink_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "job" / "raw" / "compiler_bundle"
            expected.mkdir(parents=True)
            self._write_bundle_skeleton(expected)
            (expected / "compiler_receipt.json").write_text("{}\n", encoding="utf-8")
            observed, _ = jobs._inventory_bundle(expected, expected_bundle=expected)
            self.assertEqual(len(observed), jobs.EXPECTED_BUNDLE_FILES)

            with self.assertRaisesRegex(
                jobs.DevelopmentCompilerQueueError, "lexical root changed"
            ):
                jobs._inventory_bundle(expected, expected_bundle=root / "other")

            alias = root / "bundle-alias"
            alias.symlink_to(expected, target_is_directory=True)
            with self.assertRaisesRegex(
                jobs.DevelopmentCompilerQueueError, "contains a symlink"
            ):
                jobs._inventory_bundle(alias, expected_bundle=alias)

            ancestor = root / "ancestor-alias"
            ancestor.symlink_to(root / "job", target_is_directory=True)
            through_ancestor = ancestor / "raw" / "compiler_bundle"
            with self.assertRaisesRegex(
                jobs.DevelopmentCompilerQueueError, "contains a symlink"
            ):
                jobs._inventory_bundle(
                    through_ancestor, expected_bundle=through_ancestor
                )

    def test_semantic_validation_rejects_empty_compiler_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            self._write_bundle_skeleton(bundle)
            (bundle / "compiler_receipt.json").write_text("{}\n", encoding="utf-8")
            files = {
                relative: jobs.queue.file_identity(bundle / relative)
                for relative in jobs._expected_bundle_paths()
            }
            _, compiler, timing, annotation = jobs._validate_staged_implementation(
                source_root=jobs.REPOSITORY_ROOT,
                expected=self._implementation(),
            )
            with self.assertRaisesRegex(
                jobs.DevelopmentCompilerQueueError,
                "timing inventory fields changed",
            ):
                jobs._validate_semantic_bundle_outputs(
                    compiler=compiler,
                    timing=timing,
                    annotation=annotation,
                    bundle=bundle,
                    bundle_files=files,
                    receipt={"models": {}},
                )

    def test_all_seventy_nonreceipt_outputs_pass_semantic_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            receipt, files, freeze_replays = self._write_semantic_bundle(bundle)
            _, compiler, timing, annotation = jobs._validate_staged_implementation(
                source_root=jobs.REPOSITORY_ROOT,
                expected=self._implementation(),
            )

            def replay(entry, *, model, expected_cell_id, evidence_base, require_resource):
                self.assertEqual(entry, freeze_replays[expected_cell_id]["entry"])
                self.assertEqual(evidence_base, bundle)
                self.assertIs(require_resource, False)
                return freeze_replays[expected_cell_id]["validated"]

            with mock.patch.object(
                compiler.freeze, "_validate_cell", side_effect=replay
            ) as validator:
                validated = jobs._validate_semantic_bundle_outputs(
                    compiler=compiler,
                    timing=timing,
                    annotation=annotation,
                    bundle=bundle,
                    bundle_files=files,
                    receipt=receipt,
                )
            self.assertEqual(
                set(validated),
                jobs._expected_bundle_paths() - {"compiler_receipt.json"},
            )
            self.assertEqual(len(validated), 70)
            self.assertEqual(validator.call_count, 32)

    def test_confirmation_capability_in_compiler_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            self._write_bundle_skeleton(bundle)
            manifest = root / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            manifest_identity = jobs.queue.file_identity(manifest)
            implementation = self._implementation()
            _, compiler, _, _ = jobs._validate_staged_implementation(
                source_root=jobs.REPOSITORY_ROOT, expected=implementation
            )
            aggregates = self._aggregates()
            receipt = self._compiler_receipt(
                bundle=bundle,
                manifest_identity=manifest_identity,
                implementation=implementation,
                aggregates=aggregates,
                unsafe_confirmation=True,
            )
            (bundle / "compiler_receipt.json").write_bytes(
                jobs.queue.canonical_bytes(receipt)
            )
            with self.assertRaisesRegex(
                jobs.DevelopmentCompilerQueueError,
                "safe_for_confirmation_release",
            ):
                jobs._validate_compiler_receipt(
                    compiler=compiler,
                    returned=receipt,
                    bundle=bundle,
                    expected_bundle=bundle,
                    manifest_identity=manifest_identity,
                    implementation=implementation,
                    aggregates=aggregates,
                )

    def test_failure_receipt_is_zero_science_and_cannot_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "control"
            job = control / "jobs" / jobs.JOB_ID
            job.mkdir(parents=True)
            with mock.patch.object(jobs, "CONTROL_ROOT", control):
                try:
                    raise jobs.DevelopmentCompilerQueueError("aggregate gate failed")
                except jobs.DevelopmentCompilerQueueError as error:
                    jobs._write_failure_receipt(
                        context=None, job_dir=job, error=error
                    )
            receipt = jobs.queue.load_json(
                job / "publish" / "development_evidence_compiler_job_failure.json",
                "failure receipt",
            )
            self.assertEqual(receipt["science_counts"], jobs._zero_science_counts())
            self.assertIs(receipt["safe_for_timing_binding"], False)
            self.assertIs(receipt["safe_to_release_confirmation"], False)
            self.assertIs(receipt["confirmation_released"], False)

    def test_failure_publication_refuses_success_or_unknown_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "control"
            job = control / "jobs" / jobs.JOB_ID
            publish = job / "publish"
            publish.mkdir(parents=True)
            success = publish / jobs.PUBLISH_RECEIPT_NAME
            success.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(jobs, "CONTROL_ROOT", control):
                jobs._write_failure_receipt(
                    context=None,
                    job_dir=job,
                    error=jobs.DevelopmentCompilerQueueError("after success"),
                )
            self.assertEqual({path.name for path in publish.iterdir()}, {success.name})

            success.unlink()
            unknown = publish / "unexpected.json"
            unknown.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(jobs, "CONTROL_ROOT", control):
                jobs._write_failure_receipt(
                    context=None,
                    job_dir=job,
                    error=jobs.DevelopmentCompilerQueueError("unknown inventory"),
                )
            self.assertEqual({path.name for path in publish.iterdir()}, {unknown.name})

    def test_success_receipt_write_is_the_final_fallible_statement(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_formal_job"
        )
        protected = next(node for node in function.body if isinstance(node, ast.Try))
        self.assertIsInstance(protected.body[-1], ast.Return)
        final_operation = protected.body[-2]
        self.assertIsInstance(final_operation, ast.Expr)
        self.assertIsInstance(final_operation.value, ast.Call)
        self.assertIsInstance(final_operation.value.func, ast.Attribute)
        self.assertEqual(final_operation.value.func.attr, "immutable_json")

    def test_parser_exposes_no_diagnostic_runtime(self) -> None:
        parser = jobs._parser()
        action = next(
            item
            for item in parser._actions
            if isinstance(item, argparse._SubParsersAction)
        )
        self.assertEqual(set(action.choices), {"emit-formal", "formal-full"})


if __name__ == "__main__":
    unittest.main()
