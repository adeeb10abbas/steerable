#!/usr/bin/env python3
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
LAYOUT = ROOT / "workshops/corl2026_world_models/experiments/forecast_layout"
MODULE_PATH = LAYOUT / "confirmation_release_wave_jobs.py"
spec = importlib.util.spec_from_file_location("wmf_test_confirmation_release_wave", MODULE_PATH)
assert spec is not None and spec.loader is not None
wave = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = wave
spec.loader.exec_module(wave)

COMMIT = "a" * 40
_NOW = datetime.now(timezone.utc).replace(microsecond=0)
AS_OF = _NOW.isoformat().replace("+00:00", "Z")
CONSUME_BY = (_NOW + timedelta(minutes=4)).isoformat().replace("+00:00", "Z")
ORDER = (
    "C02", "C13", "C11", "C10", "C24", "C12", "C16", "C05",
    "C17", "C19", "C09", "C06", "C08", "C01", "C04", "C23",
    "C03", "C20", "C18", "C15", "C14", "C22", "C07", "C21",
)


def write_json(path: Path, value: object) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return wave.file_identity(path)


def sign(value: dict) -> dict:
    return wave.signed_document(value)


def blocks(models=("N3", "D1")) -> dict:
    result = {}
    for model in models:
        for layout in ORDER:
            result[(model, layout)] = SimpleNamespace(
                condition_order=("original-left", "original-right", "reflected-left", "reflected-right"),
                cell_ids=tuple(f"{model}-{layout}-cell-{index}" for index in range(4)),
            )
    return result


def empty_ledger_blocks(models=("N3",), *, overrides=None) -> dict:
    overrides = overrides or {}
    result = {}
    for model in models:
        for layout in ORDER:
            state = overrides.get((model, layout), "not_run")
            result[(model, layout)] = {
                "model_config": model,
                "layout_pair_id": layout,
                "condition_order": ["original-left", "original-right", "reflected-left", "reflected-right"],
                "cell_ids": [f"{model}-{layout}-cell-{index}" for index in range(4)],
                "state": state,
                "completed_prefix_cells": 0,
                "terminal_attempt_id": None,
                "attempts": [],
            }
    return result


def fixture_rows() -> list[dict]:
    rows = []
    digest = "1" * 64
    for layout in (f"C{index:02d}" for index in range(1, 25)):
        rows.append({
            "layout_pair_id": layout,
            "candidate_id": f"{layout}__candidate_00",
            "gate_receipt": {"path": f"/evidence/{layout}/gate.json", "bytes": 1, "sha256": digest},
            "pose_manifest": {"path": f"/evidence/{layout}/pose.json", "bytes": 1, "sha256": digest},
            "capture_receipt": {"path": f"/evidence/{layout}/capture.json", "bytes": 1, "sha256": digest},
        })
    return rows


def prerequisite_identities() -> dict:
    result = {}
    for name in wave.PREREQUISITE_NAMES:
        result[name] = {"path": f"/evidence/{name}.json", "bytes": 1, "sha256": "2" * 64}
    return result


def native_ledger_replay(value: dict) -> dict:
    by_block = {
        (row["model_config"], row["layout_pair_id"]): dict(row)
        for row in value["blocks"]
    }
    attempt_ids: set[str] = set()
    job_ids: set[str] = set()
    claims: dict[str, dict] = {}
    for row in value["blocks"]:
        for attempt in row["attempts"]:
            attempt_ids.add(attempt["attempt_id"])
            job_ids.update(attempt["job_ids"])
            for job_id, descriptor in zip(
                attempt["job_ids"], attempt["queue_claims"], strict=True
            ):
                claims[job_id] = json.loads(
                    Path(descriptor["path"]).read_text(encoding="utf-8")
                )
    return {
        "document": value,
        "blocks": by_block,
        "attempt_ids": attempt_ids,
        "job_ids": job_ids,
        "queue_claims": claims,
    }


FAKE_EVIDENCE = SimpleNamespace(
    validate_result_attempt_ledger=lambda value, **_kwargs: native_ledger_replay(value),
    validate_publication_pending=lambda value, **_kwargs: dict(value),
)


def worker(worker_id: str, role: str, gpu_count: int, deployment_sha256: str) -> dict:
    deadline = "2026-09-15T16:20:00Z"
    return {
        "worker_id": worker_id,
        "role": role,
        "gpu_count": gpu_count,
        "idle": True,
        "active_job_ids": [],
        "deployment_sha256": deployment_sha256,
        "admission_deadline_utc": deadline,
        "pod_uid": f"pod-{worker_id}",
        "hostname": f"host-{worker_id}",
        "observed_at_utc": AS_OF,
        "consume_by_utc": CONSUME_BY,
        "attestation": {
            "path": f"/evidence/{worker_id}-attestation.json",
            "bytes": 1,
            "sha256": "9" * 64,
        },
    }


class FakeFixture:
    def __init__(self, rows):
        self.rows = {row["layout_pair_id"]: row for row in rows}

    def validate_fixture_freeze(self, _path, _sha, **kwargs):
        layout = kwargs.get("expected_layout_pair_id")
        return {"layout_count": 24, "selected_layout": None if layout is None else self.rows[layout]}


class ConfirmationReleaseWaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = wave.load_contract(ROOT)
        cls.runtime = wave.load_runtime_modules(ROOT)
        cls.runtime["contract"] = cls.contract

    def test_contract_authenticates_schedule_order_and_deployment_topology(self) -> None:
        identity, order, loaded = wave.validate_schedule(
            source_root=ROOT, contract=self.contract, runtime=self.runtime
        )
        self.assertEqual(identity["sha256"], self.contract["prepared_schedule"]["sha256"])
        self.assertEqual(order, ORDER)
        self.assertEqual(len(loaded), 48)
        topology = wave.deployment_topology(source_root=ROOT, contract=self.contract)
        self.assertEqual(topology["wmf-forecast-0912-worker-n3-00"]["gpu_count"], 2)
        self.assertEqual(topology["wmf-forecast-0912-worker-d1-00"]["role"], "d1")
        self.assertEqual(topology["wmf-forecast-0912-worker-00"]["gpu_count"], 1)

    def test_native_release_evidence_api_and_schemas_match_consumer(self) -> None:
        evidence = wave.load_release_evidence_module(ROOT)
        self.assertEqual(evidence.STUDY_ID, wave.STUDY_ID)
        self.assertEqual(evidence.NAMESPACE, wave.NAMESPACE)
        self.assertEqual(evidence.LEDGER_SCHEMA, wave.LEDGER_SCHEMA)
        self.assertEqual(evidence.RECONCILIATION_SCHEMA, wave.RECONCILIATION_SCHEMA)
        self.assertEqual(evidence.PENDING_SCHEMA, wave.PUBLICATION_PENDING_SCHEMA)
        self.assertEqual(
            evidence.VERIFICATION_SCHEMA,
            wave.PUBLICATION_VERIFICATION_SCHEMA,
        )
        self.assertEqual(
            self.contract["runtime_admission"]["fresh_h1_stage"], "queue_start"
        )
        self.assertTrue(
            self.contract["runtime_admission"][
                "downstream_replay_allows_consume_by_expiry_after_fresh_admission"
            ]
        )
        self.assertTrue(
            self.contract["runtime_admission"]["science_must_not_start_before_gate"]
        )
        self.assertNotIn("as_of_utc", inspect.signature(wave.build_pending_release_wave).parameters)
        self.assertNotIn("as_of_utc", inspect.signature(wave.verify_published_release_wave).parameters)
        parser = wave.build_parser()
        for mode in ("build-pending", "verify-published"):
            choice = next(action for action in parser._actions if action.dest == "mode")
            self.assertNotIn("as_of_utc", {action.dest for action in choice.choices[mode]._actions})

    def _runtime_admission_fixture(self, directory: Path):
        row = fixture_rows()[1]
        finalizer_id = "confirmation-release-finalizer-n3-a001"
        job_id, raw, normalized = wave._n3_descriptor(
            layout="C02", attempt_number=1, study_commit=COMMIT, fixture=row,
            release_identity={"path": "/release.json", "bytes": 1, "sha256": "3" * 64},
            fixture_identity={"path": "/fixture.json", "bytes": 1, "sha256": "4" * 64},
            prerequisites=prerequisite_identities(), finalizer_job_id=finalizer_id,
            consume_by_utc=CONSUME_BY, contract=self.contract, queue=self.runtime["queue"],
        )
        self.assertEqual(job_id, raw["job_id"])
        state = directory / "control"
        job_root = state / "jobs" / job_id
        descriptor_identity = write_json(job_root / "descriptor.json", normalized)
        claim = {
            "worker_id": "wmf-forecast-0912-worker-n3-00",
            "claimed_at": AS_OF,
            "claimed_unix": _NOW.timestamp(),
            "worker_pid": 123,
            "control_commit": "d" * 40,
            "control_generation": 1001,
            "descriptor_sha256": wave.sha256_bytes(self.runtime["queue"].encode(normalized)),
            "release_boundary": "claim_committed_under_shared_release_lock",
        }
        claim_identity = write_json(job_root / "claim" / "owner.json", claim)
        control = {
            "namespace": wave.NAMESPACE,
            "control_commit": claim["control_commit"],
            "control_generation": 1002,
            "admission_deadline_unix": _NOW.timestamp() + 200000,
            "shutdown": False,
            "active_job_ids": [job_id],
        }
        write_json(state / "control.json", control)
        gpu = {
            "index": 0, "uuid": "GPU-0001", "name": "NVIDIA B200",
            "memory_total_mib": 192000, "memory_used_mib": 0,
            "utilization_percent": 0,
        }
        gpu2 = {**gpu, "index": 1, "uuid": "GPU-0002"}
        attestation = sign({
            "queue_job_id": finalizer_id,
            "worker_id": claim["worker_id"],
            "role": "n3",
            "hostname": "n3-host",
            "pod_uid": "11111111-1111-1111-1111-111111111111",
            "gpu_inventory": {
                "inventory_complete": True, "gpus": [gpu, gpu2],
                "compute_processes": [], "idle": True,
            },
        })
        attestation_identity = write_json(directory / "worker_attestation.json", attestation)
        worker_row = {
            "worker_id": claim["worker_id"], "role": "n3", "gpu_count": 2,
            "idle": True, "active_job_ids": [finalizer_id],
            "deployment_sha256": "8" * 64,
            "admission_deadline_utc": "2026-09-16T00:00:00Z",
            "attestation": attestation_identity,
            "pod_uid": attestation["pod_uid"], "hostname": attestation["hostname"],
            "observed_at_utc": AS_OF, "consume_by_utc": CONSUME_BY,
        }
        topology = {
            claim["worker_id"]: {"role": "n3", "gpu_count": 2},
            **{
                f"wmf-test-worker-{index:02d}": {"role": "n3", "gpu_count": 2}
                for index in range(31)
            },
        }
        task_gpu_state = {
            "inventory_complete": True,
            "fixed_deployment_worker_count": 32,
            "fixed_deployment_worker_ids": sorted(topology),
            "deployment_inventory_is_scientific_sample_size": False,
            "selected_attestation_worker_ids": [claim["worker_id"]],
            "workers": [worker_row],
            "lane": "N3",
            "authorized_post_exit_capacity_worker_ids": [claim["worker_id"]],
            "finalizer_uses_no_gpu": True,
            "post_exit_capacity_requires_terminal_reaped_outer_result": True,
        }
        reconciliation = {
            "task_gpu_state": task_gpu_state,
        }
        pending = {"producer_queue_job": {"worker_id": claim["worker_id"]}}
        fragment = {"jobs": [raw]}
        fragment_bytes = wave.pretty_bytes(fragment)
        verification = {
            "artifacts": {
                "reconciliation": reconciliation,
                "publication_pending": pending,
                "worker_attestation": attestation,
            },
            "artifact_descriptors": {
                "queue_fragment": {
                    "commit": "b" * 40,
                    "git_path": "results/jobs/finalizer/queue.json",
                    "bytes": len(fragment_bytes),
                    "sha256": wave.sha256_bytes(fragment_bytes),
                }
            },
        }
        plan = {
            "consume_by_utc": CONSUME_BY,
            "release_finalizer_job_id": finalizer_id,
        }
        receipt = dict(plan)
        document = {
            "verified_at_utc": AS_OF,
            "consume_by_utc": CONSUME_BY,
            "pending_publication_commit": "b" * 40,
            "verified_remote_head": "c" * 40,
            "payload_sha256": "6" * 64,
        }
        triplet = {
            "descriptor": descriptor_identity,
            "descriptor_value": normalized,
            "claim": claim_identity,
            "claim_value": claim,
            "result": None,
            "claim_control": {
                "job_id": job_id, "control_commit": claim["control_commit"],
                "control_generation": claim["control_generation"],
                "control_queue_blob_sha256": "5" * 64,
                "descriptor_sha256": claim["descriptor_sha256"],
            },
        }

        class Evidence:
            @staticmethod
            def load_contract(_root):
                return {"queue": {"state_dir": str(state), "publication_grace_seconds": 600}}

            @staticmethod
            def verify_published_finalizer(**_kwargs):
                return verification

            @staticmethod
            def queue_triplet(**kwargs):
                self.assertEqual(kwargs["job_id"], job_id)
                return triplet

            @staticmethod
            def deployment_topology(_root):
                return topology

            @staticmethod
            def _pod_uid(_environ, _proc):
                return attestation["pod_uid"], "downward_api_metadata_uid"

            @staticmethod
            def gpu_inventory():
                return [gpu, gpu2], []

            @staticmethod
            def validate_worker_attestation(*_args, **_kwargs):
                raise AssertionError("finalizer worker must use H1 attestation")

            @staticmethod
            def validate_queue_claim(value, *, descriptor, result, queue):
                self.assertIsNone(result)
                self.assertEqual(value["worker_id"], claim["worker_id"])
                self.assertEqual(
                    value["descriptor_sha256"],
                    wave.sha256_bytes(queue.encode(descriptor)),
                )

        return {
            "state": state, "job_root": job_root, "job_id": job_id,
            "finalizer_id": finalizer_id, "plan": plan, "fragment": fragment,
            "receipt": receipt, "document": document,
            "verification": verification, "evidence": Evidence,
            "gpu": [gpu, gpu2], "attestation": attestation,
            "raw": raw, "normalized": normalized, "claim": claim,
            "control": control, "triplet": triplet,
            "task_gpu_state": task_gpu_state, "topology": topology,
        }

    def _invoke_runtime_admission(
        self, fixture: dict, *, now: datetime = _NOW, evidence=None,
        hostname: str = "n3-host",
    ) -> dict:
        evidence = fixture["evidence"] if evidence is None else evidence
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(wave, "load_contract", return_value=self.contract))
            stack.enter_context(mock.patch.object(wave, "validate_study_commit"))
            stack.enter_context(mock.patch.object(wave, "load_runtime_modules", return_value=self.runtime))
            stack.enter_context(mock.patch.object(wave, "load_release_evidence_module", return_value=evidence))
            stack.enter_context(mock.patch.object(
                wave, "_validate_published_wave_artifacts",
                return_value=(
                    fixture["plan"], fixture["fragment"], fixture["receipt"],
                    fixture["document"],
                ),
            ))
            stack.enter_context(mock.patch.object(wave, "_current_utc", return_value=now))
            stack.enter_context(mock.patch.object(wave.socket, "gethostname", return_value=hostname))
            stack.enter_context(mock.patch.dict(wave.os.environ, {"HOSTNAME": hostname}, clear=False))
            return wave.validate_runtime_release_admission(
                source_root=ROOT, state_dir=fixture["state"],
                job_dir=fixture["job_root"], job_id=fixture["job_id"],
                study_commit=COMMIT, expected_role="n3",
                finalizer_job_id=fixture["finalizer_id"],
                consume_by_utc=CONSUME_BY,
            )

    def _replay_runtime_admission(
        self, fixture: dict, *, receipt_path: Path, receipt_sha256: str,
        evidence=None, now: datetime = _NOW, hostname: str = "n3-host",
        verify_local_worker: bool = True,
    ) -> dict:
        evidence = fixture["evidence"] if evidence is None else evidence
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                wave, "load_contract", return_value=self.contract
            ))
            stack.enter_context(mock.patch.object(wave, "validate_study_commit"))
            stack.enter_context(mock.patch.object(
                wave, "load_runtime_modules", return_value=self.runtime
            ))
            stack.enter_context(mock.patch.object(
                wave, "load_release_evidence_module", return_value=evidence
            ))
            stack.enter_context(mock.patch.object(wave, "_current_utc", return_value=now))
            stack.enter_context(mock.patch.object(
                wave.socket, "gethostname", return_value=hostname
            ))
            stack.enter_context(mock.patch.dict(
                wave.os.environ, {"HOSTNAME": hostname}, clear=False
            ))
            return wave.validate_runtime_admission_receipt(
                receipt_path=receipt_path, receipt_sha256=receipt_sha256,
                source_root=ROOT, study_commit=COMMIT,
                expected_job_id=fixture["job_id"], expected_role="n3",
                expected_finalizer_job_id=fixture["finalizer_id"],
                expected_consume_by_utc=CONSUME_BY,
                verify_local_worker=verify_local_worker,
            )

    def test_runtime_admission_is_fresh_h1_bound_and_survives_later_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._runtime_admission_fixture(Path(temporary))
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(wave, "load_contract", return_value=self.contract))
                stack.enter_context(mock.patch.object(wave, "validate_study_commit"))
                stack.enter_context(mock.patch.object(wave, "load_runtime_modules", return_value=self.runtime))
                stack.enter_context(mock.patch.object(wave, "load_release_evidence_module", return_value=fixture["evidence"]))
                stack.enter_context(mock.patch.object(
                    wave, "_validate_published_wave_artifacts",
                    return_value=(fixture["plan"], fixture["fragment"], fixture["receipt"], fixture["document"]),
                ))
                stack.enter_context(mock.patch.object(wave, "_current_utc", return_value=_NOW))
                stack.enter_context(mock.patch.object(wave.socket, "gethostname", return_value="n3-host"))
                stack.enter_context(mock.patch.dict(wave.os.environ, {"HOSTNAME": "n3-host"}, clear=False))
                admitted = wave.validate_runtime_release_admission(
                    source_root=ROOT, state_dir=fixture["state"],
                    job_dir=fixture["job_root"], job_id=fixture["job_id"],
                    study_commit=COMMIT, expected_role="n3",
                    finalizer_job_id=fixture["finalizer_id"],
                    consume_by_utc=CONSUME_BY,
                )
            path = Path(temporary) / "release_admission.json"
            identity = write_json(path, admitted)
            later = _NOW + timedelta(hours=1)

            class LoadedModelEvidence(fixture["evidence"]):
                @staticmethod
                def gpu_inventory():
                    return fixture["gpu"], [{"pid": 456, "command": "model-server"}]

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(wave, "load_contract", return_value=self.contract))
                stack.enter_context(mock.patch.object(wave, "validate_study_commit"))
                stack.enter_context(mock.patch.object(
                    wave, "load_release_evidence_module", return_value=LoadedModelEvidence
                ))
                stack.enter_context(mock.patch.object(wave, "_current_utc", return_value=later))
                stack.enter_context(mock.patch.object(wave.socket, "gethostname", return_value="n3-host"))
                stack.enter_context(mock.patch.dict(wave.os.environ, {"HOSTNAME": "n3-host"}, clear=False))
                replayed = wave.validate_runtime_admission_receipt(
                    receipt_path=path, receipt_sha256=identity["sha256"],
                    source_root=ROOT, study_commit=COMMIT,
                    expected_job_id=fixture["job_id"], expected_role="n3",
                    expected_finalizer_job_id=fixture["finalizer_id"],
                    expected_consume_by_utc=CONSUME_BY, verify_local_worker=True,
                )
            self.assertEqual(replayed["checked_at_utc"], AS_OF)

    def test_runtime_admission_rejects_expiry_and_current_gpu_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._runtime_admission_fixture(Path(temporary))

            def invoke(now, evidence=fixture["evidence"]):
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(wave, "load_contract", return_value=self.contract))
                    stack.enter_context(mock.patch.object(wave, "validate_study_commit"))
                    stack.enter_context(mock.patch.object(wave, "load_runtime_modules", return_value=self.runtime))
                    stack.enter_context(mock.patch.object(wave, "load_release_evidence_module", return_value=evidence))
                    stack.enter_context(mock.patch.object(
                        wave, "_validate_published_wave_artifacts",
                        return_value=(fixture["plan"], fixture["fragment"], fixture["receipt"], fixture["document"]),
                    ))
                    stack.enter_context(mock.patch.object(wave, "_current_utc", return_value=now))
                    stack.enter_context(mock.patch.object(wave.socket, "gethostname", return_value="n3-host"))
                    stack.enter_context(mock.patch.dict(wave.os.environ, {"HOSTNAME": "n3-host"}, clear=False))
                    return wave.validate_runtime_release_admission(
                        source_root=ROOT, state_dir=fixture["state"],
                        job_dir=fixture["job_root"], job_id=fixture["job_id"],
                        study_commit=COMMIT, expected_role="n3",
                        finalizer_job_id=fixture["finalizer_id"],
                        consume_by_utc=CONSUME_BY,
                    )

            with self.assertRaisesRegex(wave.ConfirmationReleaseWaveError, "expired"):
                invoke(_NOW + timedelta(minutes=5))

            class ChangedGpu(fixture["evidence"]):
                @staticmethod
                def gpu_inventory():
                    rows = json.loads(json.dumps(fixture["gpu"]))
                    rows[1]["uuid"] = "GPU-substituted"
                    return rows, []

            with self.assertRaisesRegex(wave.ConfirmationReleaseWaveError, "pod/GPU"):
                invoke(_NOW, ChangedGpu)

    def test_runtime_admission_rejects_absent_duplicate_or_tampered_descriptor(self) -> None:
        mutations = {
            "absent": lambda fixture: fixture["fragment"]["jobs"][0].update(
                job_id="confirmation-c99-n3-a001"
            ),
            "duplicated": lambda fixture: fixture["fragment"]["jobs"].append(
                json.loads(json.dumps(fixture["fragment"]["jobs"][0]))
            ),
            "tampered": lambda fixture: fixture["fragment"]["jobs"][0]["argv"].__setitem__(
                fixture["fragment"]["jobs"][0]["argv"].index(
                    "--confirmation-release-finalizer-job-id"
                ) + 1,
                "confirmation-release-finalizer-n3-a999",
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                fixture = self._runtime_admission_fixture(Path(temporary))
                mutate(fixture)
                with self.assertRaisesRegex(
                    wave.ConfirmationReleaseWaveError,
                    "absent or duplicated|descriptor authority",
                ):
                    self._invoke_runtime_admission(fixture)

    def test_runtime_admission_rejects_unselected_claim_and_live_control_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._runtime_admission_fixture(Path(temporary))
            fixture["triplet"]["claim_value"] = {
                **fixture["claim"], "worker_id": "wmf-test-worker-00",
            }
            with self.assertRaisesRegex(
                wave.ConfirmationReleaseWaveError, "outside published capacity"
            ):
                self._invoke_runtime_admission(fixture)

        control_cases = ("extra", "missing", "reordered")
        for case in control_cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                fixture = self._runtime_admission_fixture(Path(temporary))
                control = dict(fixture["control"])
                if case == "extra":
                    control["active_job_ids"] = [fixture["job_id"], "unpublished-job"]
                elif case == "missing":
                    control["active_job_ids"] = []
                else:
                    other = json.loads(json.dumps(fixture["raw"]))
                    other["job_id"] = "confirmation-c13-n3-a001"
                    fixture["fragment"]["jobs"].append(other)
                    control["active_job_ids"] = [other["job_id"], fixture["job_id"]]
                write_json(fixture["state"] / "control.json", control)
                with self.assertRaisesRegex(
                    wave.ConfirmationReleaseWaveError, "exact releasable wave"
                ):
                    self._invoke_runtime_admission(fixture)

    def test_runtime_admission_rejects_hostname_pod_gpu_count_or_compute_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._runtime_admission_fixture(Path(temporary))
            with self.assertRaisesRegex(
                wave.ConfirmationReleaseWaveError, "hostname differs"
            ):
                self._invoke_runtime_admission(fixture, hostname="other-host")

        for case in ("pod", "gpu_count", "compute"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                fixture = self._runtime_admission_fixture(Path(temporary))

                class ChangedRuntime(fixture["evidence"]):
                    @staticmethod
                    def _pod_uid(_environ, _proc):
                        uid = (
                            "22222222-2222-2222-2222-222222222222"
                            if case == "pod" else fixture["attestation"]["pod_uid"]
                        )
                        return uid, "downward_api_metadata_uid"

                    @staticmethod
                    def gpu_inventory():
                        rows = fixture["gpu"][:1] if case == "gpu_count" else fixture["gpu"]
                        processes = [{"pid": 123}] if case == "compute" else []
                        return rows, processes

                with self.assertRaisesRegex(
                    wave.ConfirmationReleaseWaveError, "pod/GPU identity or idle"
                ):
                    self._invoke_runtime_admission(fixture, evidence=ChangedRuntime)

    def test_runtime_receipt_replay_rejects_descriptor_or_claim_substitution(self) -> None:
        for target in ("descriptor", "claim"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                fixture = self._runtime_admission_fixture(Path(temporary))
                admitted = self._invoke_runtime_admission(fixture)
                admission_path = Path(temporary) / "release_admission.json"
                admission_identity = write_json(admission_path, admitted)
                target_path = (
                    fixture["job_root"] / "descriptor.json"
                    if target == "descriptor"
                    else fixture["job_root"] / "claim" / "owner.json"
                )
                changed = json.loads(target_path.read_text(encoding="utf-8"))
                if target == "descriptor":
                    changed["max_wall_seconds"] -= 1
                else:
                    changed["worker_pid"] += 1
                write_json(target_path, changed)
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        wave, "load_contract", return_value=self.contract
                    ))
                    stack.enter_context(mock.patch.object(wave, "validate_study_commit"))
                    stack.enter_context(mock.patch.object(
                        wave, "load_runtime_modules", return_value=self.runtime
                    ))
                    stack.enter_context(mock.patch.object(
                        wave, "load_release_evidence_module",
                        return_value=fixture["evidence"],
                    ))
                    stack.enter_context(mock.patch.object(
                        wave, "_current_utc", return_value=_NOW
                    ))
                    with self.assertRaisesRegex(
                        wave.ConfirmationReleaseWaveError, "identity changed"
                    ):
                        wave.validate_runtime_admission_receipt(
                            receipt_path=admission_path,
                            receipt_sha256=admission_identity["sha256"],
                            source_root=ROOT, study_commit=COMMIT,
                            expected_job_id=fixture["job_id"], expected_role="n3",
                            expected_finalizer_job_id=fixture["finalizer_id"],
                            expected_consume_by_utc=CONSUME_BY,
                            verify_local_worker=False,
                        )

    def test_runtime_receipt_replay_rejects_worker_identity_drift_after_admission(self) -> None:
        for case in ("hostname", "pod", "gpu"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                fixture = self._runtime_admission_fixture(Path(temporary))
                admitted = self._invoke_runtime_admission(fixture)
                path = Path(temporary) / "release_admission.json"
                identity = write_json(path, admitted)

                class ChangedWorker(fixture["evidence"]):
                    @staticmethod
                    def _pod_uid(_environ, _proc):
                        uid = (
                            "22222222-2222-2222-2222-222222222222"
                            if case == "pod" else fixture["attestation"]["pod_uid"]
                        )
                        return uid, "downward_api_metadata_uid"

                    @staticmethod
                    def gpu_inventory():
                        rows = json.loads(json.dumps(fixture["gpu"]))
                        if case == "gpu":
                            rows[0]["uuid"] = "GPU-drifted-after-admission"
                        return rows, [{"pid": 456, "command": "model-server"}]

                with self.assertRaisesRegex(
                    wave.ConfirmationReleaseWaveError,
                    "current worker identity changed",
                ):
                    self._replay_runtime_admission(
                        fixture, receipt_path=path,
                        receipt_sha256=identity["sha256"], evidence=ChangedWorker,
                        now=_NOW + timedelta(hours=1),
                        hostname="other-host" if case == "hostname" else "n3-host",
                    )

    def test_study_commit_binds_builder_and_contract_and_rejects_dirty_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            builder = root / wave.BUILDER_RELATIVE
            contract_path = root / wave.CONTRACT_RELATIVE
            builder.parent.mkdir(parents=True)
            builder.write_bytes(MODULE_PATH.read_bytes())
            write_json(contract_path, {"source_dependencies": {}})
            wave.subprocess.run(["git", "init", "-q", str(root)], check=True)
            wave.subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "wave@example.invalid"],
                check=True,
            )
            wave.subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Wave Test"],
                check=True,
            )
            wave.subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            wave.subprocess.run(
                ["git", "-C", str(root), "commit", "-q", "-m", "freeze builder"],
                check=True,
            )
            commit = wave.subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            minimal_contract = {"source_dependencies": {}}
            wave.validate_study_commit(root, commit, minimal_contract)

            builder.write_bytes(builder.read_bytes() + b"# dirty builder\n")
            with self.assertRaisesRegex(
                wave.ConfirmationReleaseWaveError, "pinned source.*release_wave_jobs"
            ):
                wave.validate_study_commit(root, commit, minimal_contract)
            builder.write_bytes(MODULE_PATH.read_bytes())

            contract_path.write_text('{"source_dependencies":{"dirty":true}}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                wave.ConfirmationReleaseWaveError, "pinned source.*release_wave_contract"
            ):
                wave.validate_study_commit(root, commit, minimal_contract)

    def test_schedule_row_reordering_cannot_change_signed_hash_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = json.loads(
                (
                    ROOT
                    / self.contract["prepared_schedule"]["path"]
                ).read_text(encoding="utf-8")
            )
            source["jobs"] = list(reversed(source["jobs"]))
            schedule_path = root / "schedule.json"
            schedule_identity = write_json(schedule_path, source)
            contract = json.loads(json.dumps(self.contract))
            contract["prepared_schedule"]["path"] = "schedule.json"
            contract["prepared_schedule"]["sha256"] = schedule_identity["sha256"]

            def schedule_block(_root, layout, model):
                return SimpleNamespace(
                    schedule_sha256=schedule_identity["sha256"],
                    block_id=f"{model}-{layout}",
                    condition_order=("a", "b", "c", "d"),
                    cell_ids=("0", "1", "2", "3"),
                )

            def runtime_block(_root, layout, *, model):
                return schedule_block(_root, layout, model)

            runtime = {
                "common": SimpleNamespace(load_confirmation_schedule_block=schedule_block),
                "n3": SimpleNamespace(
                    load_confirmation_block=lambda source_root, layout: runtime_block(
                        source_root, layout, model="N3"
                    )
                ),
                "d1": SimpleNamespace(
                    load_confirmation_block=lambda source_root, layout: runtime_block(
                        source_root, layout, model="D1"
                    )
                ),
            }
            _identity, order, _blocks = wave.validate_schedule(
                source_root=root, contract=contract, runtime=runtime
            )
            self.assertEqual(order, ORDER)

            source["order_assignment"]["confirmation_blocks_in_hash_order"] = list(
                reversed(ORDER)
            )
            changed_identity = write_json(schedule_path, source)
            contract["prepared_schedule"]["sha256"] = changed_identity["sha256"]
            with self.assertRaisesRegex(
                wave.ConfirmationReleaseWaveError, "hash order changed"
            ):
                wave.validate_schedule(source_root=root, contract=contract, runtime=runtime)

    def test_missing_development_annotation_keeps_confirmation_held(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            value = {
                "schema_version": "wmf-development-confirmation-release-freeze-v1",
                "status": "held_missing_development_evidence",
                "cohort_branch": "reduced_n3",
                "qualified_model_ids": ["N3"],
                "release_decision": {
                    "eligible": False,
                    "blockers": ["two authenticated human annotation passes are missing"],
                },
                "annotation": {
                    "usability_decision": {"measurement_usable": False},
                    "movement_resolution": {
                        "status": "pending_duplicate_development_labels"
                    },
                },
                "resource_budget": {
                    "status": "held_pending_complete_development_measurements",
                    "by_model": {"N3": {"max_parallel_blocks": 1}},
                },
            }
            identity = write_json(
                Path(temporary) / "held-confirmation-freeze.json", value
            )
            runtime = {
                "common": SimpleNamespace(
                    verify_confirmation_freeze=lambda *_args, **_kwargs: {
                        "unexpected": "a held document must not reach native admission"
                    }
                )
            }
            with self.assertRaisesRegex(
                wave.ConfirmationReleaseWaveError,
                "still held by development evidence",
            ):
                wave.validate_release_freeze(
                    descriptor=identity,
                    source_root=ROOT,
                    contract=self.contract,
                    runtime=runtime,
                )

    def test_runtime_publication_hash_binds_terminal_sources_and_prerequisites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prerequisites = {}
            for name in wave.PREREQUISITE_NAMES:
                prerequisites[name] = write_json(directory / f"{name}.json", {"name": name})
            prefix = "workshops/corl2026_world_models/experiments/forecast_layout/"
            runtime_files = {
                name: {
                    "path": prefix + name,
                    "sha256": self.contract["source_dependencies"][prefix + name],
                }
                for name in wave.RUNTIME_IDENTITY_FILES
            }
            value = sign({
                "schema_version": wave.RUNTIME_IDENTITIES_SCHEMA,
                "status": "published_terminal_context_runtime",
                "study_id": wave.STUDY_ID,
                "namespace": wave.NAMESPACE,
                "study_commit": COMMIT,
                "published_at_utc": AS_OF,
                "runtime_files": runtime_files,
                "terminal_context_schemas": self.contract["terminal_context_schemas"],
                "prerequisite_receipts": prerequisites,
                "claim_boundary": "published runtime and prerequisite identities only",
            })
            identity = write_json(directory / "runtime-identities.json", value)
            native_validate = mock.Mock(return_value={
                "receipt": value,
                "identity": identity,
                "prerequisite_receipts": prerequisites,
            })
            native = SimpleNamespace(
                validate_terminal_runtime_identities=native_validate,
            )
            with mock.patch.object(
                wave, "load_terminal_runtime_identities_module", return_value=native,
            ):
                loaded, observed = wave.validate_runtime_identities(
                    descriptor=identity, source_root=ROOT, study_commit=COMMIT,
                    contract=self.contract,
                )
            self.assertEqual(loaded["terminal_context_schemas"]["N3"], "wmf-n3-terminal-context-receipt-v1")
            self.assertEqual(set(observed), wave.PREREQUISITE_NAMES)
            native_validate.assert_called_once_with(
                receipt_path=Path(identity["path"]),
                expected_sha256=identity["sha256"],
                source_root=ROOT,
                source_commit=COMMIT,
            )
            tampered = json.loads(Path(identity["path"]).read_text())
            tampered["runtime_files"]["n3_confirmation_block_job.py"]["sha256"] = "0" * 64
            tampered.pop("payload_sha256")
            bad = write_json(directory / "runtime-identities-bad.json", sign(tampered))
            native.validate_terminal_runtime_identities = mock.Mock(return_value={
                "receipt": sign(tampered),
                "identity": bad,
                "prerequisite_receipts": prerequisites,
            })
            with mock.patch.object(
                wave, "load_terminal_runtime_identities_module", return_value=native,
            ):
                with self.assertRaisesRegex(wave.ConfirmationReleaseWaveError, "runtime identity"):
                    wave.validate_runtime_identities(
                        descriptor=bad, source_root=ROOT, study_commit=COMMIT,
                        contract=self.contract,
                    )

            native.validate_terminal_runtime_identities = mock.Mock(
                side_effect=RuntimeError("unpublished source"),
            )
            with mock.patch.object(
                wave, "load_terminal_runtime_identities_module", return_value=native,
            ):
                with self.assertRaisesRegex(
                    wave.ConfirmationReleaseWaveError, "failed native validation",
                ):
                    wave.validate_runtime_identities(
                        descriptor=identity, source_root=ROOT, study_commit=COMMIT,
                        contract=self.contract,
                    )

    def _resource(self, directory: Path, *, n3_limit=1, global_limit=1) -> tuple[dict, dict]:
        raw = write_json(directory / "raw-resource.json", {"raw": True})
        topology = {
            "global_max_parallel_blocks": global_limit,
            "cross_model_simultaneous_blocks_allowed": False,
            "max_parallel_blocks_by_model": {"N3": n3_limit, "D1": 1},
            "N3": {"model_worker_role": "n3", "gpu_count": 2},
            "D1": {
                "model_worker_role": "d1", "model_gpu_count": 2,
                "simulator_worker_role": "wmf-forecast-0912-worker-00",
                "simulator_gpu_count": 1,
            },
            "measurement_status": "passed_at_explicit_topology",
            "higher_concurrency_qualified": global_limit > 1 or n3_limit > 1,
            "scheduling_rule": "signed measured cap",
        }
        value = sign({
            "schema_version": "wmf-development-resource-machine-freeze-v1",
            "status": "machine_resource_gate_passed_annotation_time_pending",
            "decision": "serial_gm_topology_frozen_confirmation_held",
            "namespace": wave.NAMESPACE, "study_id": wave.STUDY_ID,
            "study_commit": COMMIT, "job_id": "resource-finalize",
            "queue_role": "any", "worker_id": "any", "runtime_identity": {},
            "queue_descriptor": {}, "queue_claim": {}, "implementation": {},
            "inputs": {}, "measurement": {},
            "model_envelopes": {"N3": {"measured": True}, "D1": {"measured": True}},
            "safe_execution_topology": topology, "science_counts": {},
            "machine_resource_gate_complete": True,
            "resource_measurement_freeze_complete": True,
            "annotation_time": {}, "remaining_resource_release_requirements": [],
            "safe_to_release_confirmation": False, "confirmation_released": False,
            "behavioral_policy_skill_evaluated": False, "claim_boundary": "capacity only",
            "raw_machine_resource_freeze": raw,
        })
        return write_json(directory / "resource.json", value), value

    def test_resource_cap_must_be_jointly_proved_by_release_and_machine_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            identity, _value = self._resource(directory, n3_limit=3, global_limit=3)
            release = {"resource_budget": {"by_model": {"N3": {"max_parallel_blocks": 2}}}}
            _loaded, limits = wave.validate_resource_qualification(
                descriptor=identity, study_commit=COMMIT, models=("N3",),
                release=release, contract=self.contract,
            )
            self.assertEqual(limits, {"N3": 2})
            release["resource_budget"]["by_model"]["N3"]["max_parallel_blocks"] = 4
            _loaded, limits = wave.validate_resource_qualification(
                descriptor=identity, study_commit=COMMIT, models=("N3",),
                release=release, contract=self.contract,
            )
            self.assertEqual(limits, {"N3": 3})
            serial_identity, _ = self._resource(directory / "serial", n3_limit=1, global_limit=1)
            _loaded, limits = wave.validate_resource_qualification(
                descriptor=serial_identity, study_commit=COMMIT, models=("N3",),
                release=release, contract=self.contract,
            )
            self.assertEqual(limits, {"N3": 1})

    def test_new_descriptors_round_trip_runtime_parsers_and_queue_normalizer(self) -> None:
        row = fixture_rows()[1]
        prerequisites = prerequisite_identities()
        release = {"path": "/evidence/release.json", "bytes": 1, "sha256": "3" * 64}
        fixture = {"path": "/evidence/fixture.json", "bytes": 1, "sha256": "4" * 64}
        _, n3_raw, n3_normalized = wave._n3_descriptor(
            layout="C02", attempt_number=1, study_commit=COMMIT, fixture=row,
            release_identity=release, fixture_identity=fixture,
            prerequisites=prerequisites,
            finalizer_job_id="confirmation-release-finalizer-n3-a001",
            consume_by_utc=CONSUME_BY,
            contract=self.contract, queue=self.runtime["queue"],
        )
        wave._round_trip_new_jobs(
            raw_jobs=[n3_raw], normalized_jobs=[n3_normalized], model="N3",
            layout="C02", study_commit=COMMIT, runtime=self.runtime,
        )
        _, d1_raw, d1_normalized = wave._d1_descriptors(
            layout="C02", attempt_number=2, study_commit=COMMIT,
            simulator_role="wmf-forecast-0912-worker-00", fixture=row,
            release_identity=release, fixture_identity=fixture,
            prerequisites=prerequisites,
            finalizer_job_id="confirmation-release-finalizer-d1-a001",
            consume_by_utc=CONSUME_BY,
            contract=self.contract, queue=self.runtime["queue"],
        )
        wave._round_trip_new_jobs(
            raw_jobs=d1_raw, normalized_jobs=d1_normalized, model="D1",
            layout="C02", study_commit=COMMIT, runtime=self.runtime,
        )
        self.assertEqual([item["role"] for item in d1_raw], ["d1", "wmf-forecast-0912-worker-00"])
        self.assertTrue(all(item["publish_log_tail_bytes"] == 0 for item in [n3_raw, *d1_raw]))
        self.assertIn(d1_raw[1]["job_id"], d1_raw[0]["argv"])
        self.assertIn(d1_raw[0]["job_id"], d1_raw[1]["argv"])
        forbidden = ("password", "private-key", "secret", "token")
        self.assertFalse(any(any(term in arg.lower() for term in forbidden) for job in [n3_raw, *d1_raw] for arg in job["argv"]))
        tampered = json.loads(json.dumps(n3_normalized))
        index = tampered["argv"].index("--confirmation-freeze-sha256") + 1
        tampered["argv"][index] = "0" * 64
        with self.assertRaisesRegex(wave.ConfirmationReleaseWaveError, "evidence binding"):
            wave._parse_descriptor(
                tampered, model="N3", layout="C02", study_commit=COMMIT,
                runtime=self.runtime, expected_job_ids=[tampered["job_id"]],
                expected_attempt_id=tampered["job_id"],
                expected_options=wave._historical_evidence_options(
                    model="N3", fixture=row, release_identity=release,
                    fixture_identity=fixture, prerequisites=prerequisites,
                ),
            )
        duplicate = json.loads(json.dumps(n3_normalized))
        duplicate["argv"].extend(["--job-id", duplicate["job_id"]])
        with self.assertRaisesRegex(wave.ConfirmationReleaseWaveError, "duplicate options"):
            wave._parse_descriptor(
                duplicate, model="N3", layout="C02", study_commit=COMMIT,
                runtime=self.runtime, expected_job_ids=[duplicate["job_id"]],
            )
        secret = json.loads(json.dumps(n3_normalized))
        secret["argv"][secret["argv"].index("--candidate-id") + 1] = "api-token"
        with self.assertRaisesRegex(wave.ConfirmationReleaseWaveError, "secret-bearing"):
            wave._parse_descriptor(
                secret, model="N3", layout="C02", study_commit=COMMIT,
                runtime=self.runtime, expected_job_ids=[secret["job_id"]],
            )

    def test_ledger_preserves_technical_predecessor_and_terminal_safety_without_successor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture = fixture_rows()[1]
            release_identity = {"path": "/release.json", "bytes": 1, "sha256": "3" * 64}
            fixture_identity = {"path": "/fixture.json", "bytes": 1, "sha256": "4" * 64}

            def historical_attempt(
                number: int, states: str | list[str], predecessor,
                *, queue_status: str = "failed", evidence_root: Path | None = None,
            ):
                states = [states] if isinstance(states, str) else states
                attempt_directory = evidence_root or directory
                attempt_id, raw, normalized = wave._n3_descriptor(
                    layout="C02", attempt_number=number, study_commit=COMMIT,
                    fixture=fixture, release_identity=release_identity,
                    fixture_identity=fixture_identity, prerequisites=prerequisite_identities(),
                    finalizer_job_id="confirmation-release-finalizer-n3-a001",
                    consume_by_utc=CONSUME_BY,
                    contract=self.contract, queue=self.runtime["queue"],
                )
                job_root = attempt_directory / "control" / "jobs" / raw["job_id"]
                descriptor_identity = write_json(job_root / "descriptor.json", normalized)
                stream = {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
                state_root = job_root.parents[1]
                executed_argv = [
                    item.replace(
                        "{source_root}", str(state_root / "sources" / COMMIT)
                    ).replace("{job_dir}", str(job_root)).replace(
                        "{state_dir}", str(state_root)
                    )
                    for item in normalized["argv"]
                ]
                result = {
                    "schema_version": "wmf-cluster-result-v1", "namespace": wave.NAMESPACE,
                    "job_id": raw["job_id"], "worker_id": "n3-live-01",
                    "source_commit": COMMIT,
                    "descriptor_sha256": hashlib.sha256(self.runtime["queue"].encode(normalized)).hexdigest(),
                    "started_at": AS_OF, "argv": executed_argv,
                    "job_dir": str(job_root.resolve()), "status": queue_status,
                    "returncode": 0 if queue_status == "succeeded" else 1,
                    "error_type": None, "ended_at": AS_OF,
                    "wall_seconds": 1.0, "child_pid": 123, "child_reaped": True,
                    "stdout": stream, "stderr": stream,
                }
                result_identity = write_json(job_root / "result.json", result)
                claim = {
                    "worker_id": "n3-live-01", "claimed_at": AS_OF,
                    "claimed_unix": datetime.fromisoformat(AS_OF.replace("Z", "+00:00")).timestamp(),
                    "worker_pid": 122, "control_commit": "c" * 40,
                    "control_generation": 997,
                    "descriptor_sha256": hashlib.sha256(self.runtime["queue"].encode(normalized)).hexdigest(),
                    "release_boundary": "claim_committed_under_shared_release_lock",
                }
                claim_identity = write_json(job_root / "claim" / "owner.json", claim)
                cell_rows = []
                for condition_index, state in enumerate(states):
                    receipt_identity = write_json(
                        attempt_directory / raw["job_id"] / "cells"
                        / f"{condition_index:02d}-cell" / f"native-{number}.json",
                        {"native_state": state},
                    )
                    cell_rows.append({
                        "condition_index": condition_index,
                        "state": state,
                        "receipt": receipt_identity,
                    })
                return {
                    "attempt_id": attempt_id, "attempt_number": number,
                    "predecessor_attempt_id": predecessor, "start_cell_index": 0,
                    "job_ids": [raw["job_id"]], "queue_descriptors": [descriptor_identity],
                    "queue_claims": [claim_identity], "queue_results": [result_identity],
                    "cells": cell_rows,
                    "runtime_evidence": {
                        "form": "native_cell_prefix",
                        "zero_behavioral_cells_launched": False,
                    },
                }

            attempt1 = historical_attempt(1, "technical_invalid", None)
            attempt2 = historical_attempt(2, "safety_censored", attempt1["attempt_id"])
            ledger_rows = list(empty_ledger_blocks(("N3",)).values())
            ledger_rows[0] = {
                **ledger_rows[0], "state": "safety_censored", "completed_prefix_cells": 0,
                "terminal_attempt_id": attempt2["attempt_id"], "attempts": [attempt1, attempt2],
            }
            inputs = {
                "prepared_schedule": {"path": "/schedule.json", "bytes": 1, "sha256": "5" * 64},
                "confirmation_freeze": release_identity,
                "fixture_freeze": fixture_identity,
                "resource_qualification": {"path": "/resource.json", "bytes": 1, "sha256": "6" * 64},
                "terminal_runtime_identities": {"path": "/runtime.json", "bytes": 1, "sha256": "7" * 64},
            }

            def ledger_document(rows):
                return sign({
                    "schema_version": wave.LEDGER_SCHEMA,
                    "status": "authenticated_native_attempt_inventory",
                    "study_id": wave.STUDY_ID, "namespace": wave.NAMESPACE,
                    "study_commit": COMMIT, "cohort_branch": "reduced_n3",
                    "qualified_model_ids": ["N3"],
                    **inputs, "layout_order": list(ORDER), "blocks": rows,
                    "attempt_inventory_complete": True,
                    "claim_boundary": "all native attempts retained",
                })

            ledger_identity = write_json(directory / "attempt-ledger.json", ledger_document(ledger_rows))
            native = lambda **kwargs: json.loads(Path(kwargs["receipt_path"]).read_text())["native_state"]
            with mock.patch.object(wave, "_native_cell_state", side_effect=native):
                _value, derived, attempt_ids, job_ids, claims = wave.validate_result_ledger(
                    descriptor=ledger_identity, study_commit=COMMIT, branch="reduced_n3",
                    source_root=ROOT, state_dir=directory, evidence=FAKE_EVIDENCE,
                    models=("N3",), layout_order=ORDER, blocks=blocks(("N3",)),
                    schedule_identity=inputs["prepared_schedule"],
                    release_identity=inputs["confirmation_freeze"],
                    fixture_identity=inputs["fixture_freeze"],
                    resource_identity=inputs["resource_qualification"],
                    runtime_identity=inputs["terminal_runtime_identities"], runtime=self.runtime,
                    fixture_rows={row["layout_pair_id"]: row for row in fixture_rows()},
                    prerequisites=prerequisite_identities(),
                )
            self.assertEqual(derived[("N3", "C02")]["state"], "safety_censored")
            self.assertEqual(attempt_ids, {attempt1["attempt_id"], attempt2["attempt_id"]})
            self.assertEqual(len(job_ids), 2)
            self.assertEqual(set(claims), job_ids)

            # A wrapper can fail after the last durable native receipt.  The
            # native four-cell block remains terminal and must not be selected
            # for a scientific rerun.
            passed_attempt = historical_attempt(
                1, ["passed", "passed", "passed", "passed"], None,
                queue_status="failed", evidence_root=directory / "passed-evidence",
            )
            passed_rows = list(empty_ledger_blocks(("N3",)).values())
            passed_rows[0] = {
                **passed_rows[0], "state": "passed", "completed_prefix_cells": 4,
                "terminal_attempt_id": passed_attempt["attempt_id"],
                "attempts": [passed_attempt],
            }
            passed_ledger = write_json(
                directory / "outer-postprocessing-failed.json",
                ledger_document(passed_rows),
            )
            with mock.patch.object(wave, "_native_cell_state", side_effect=native):
                _value, passed_derived, *_rest = wave.validate_result_ledger(
                    descriptor=passed_ledger, study_commit=COMMIT, branch="reduced_n3",
                    source_root=ROOT, state_dir=directory, evidence=FAKE_EVIDENCE,
                    models=("N3",), layout_order=ORDER, blocks=blocks(("N3",)),
                    schedule_identity=inputs["prepared_schedule"],
                    release_identity=inputs["confirmation_freeze"],
                    fixture_identity=inputs["fixture_freeze"],
                    resource_identity=inputs["resource_qualification"],
                    runtime_identity=inputs["terminal_runtime_identities"], runtime=self.runtime,
                    fixture_rows={row["layout_pair_id"]: row for row in fixture_rows()},
                    prerequisites=prerequisite_identities(),
                )
            self.assertEqual(passed_derived[("N3", "C02")]["state"], "passed")
            lane, remaining = wave._select_lane(
                models=("N3",), layout_order=ORDER, ledger_blocks=passed_derived,
                lane_priority=("N3", "D1"),
            )
            self.assertEqual((lane, remaining[0]), ("N3", "C13"))

            # If the queue wrapper dies between cells, the durable passed
            # prefix is retained and the terminal/reaped outer failure makes
            # the attempt technical-invalid rather than erasing that prefix.
            partial_attempt = historical_attempt(
                1, ["passed"], None, queue_status="failed",
                evidence_root=directory / "partial-evidence",
            )
            partial_rows = list(empty_ledger_blocks(("N3",)).values())
            partial_rows[0] = {
                **partial_rows[0], "state": "technical_invalid",
                "completed_prefix_cells": 1, "terminal_attempt_id": None,
                "attempts": [partial_attempt],
            }
            partial_ledger = write_json(
                directory / "outer-failed-after-prefix.json",
                ledger_document(partial_rows),
            )
            with mock.patch.object(wave, "_native_cell_state", side_effect=native):
                _value, partial_derived, *_rest = wave.validate_result_ledger(
                    descriptor=partial_ledger, study_commit=COMMIT,
                    branch="reduced_n3", source_root=ROOT, state_dir=directory,
                    evidence=FAKE_EVIDENCE, models=("N3",), layout_order=ORDER,
                    blocks=blocks(("N3",)), schedule_identity=inputs["prepared_schedule"],
                    release_identity=inputs["confirmation_freeze"],
                    fixture_identity=inputs["fixture_freeze"],
                    resource_identity=inputs["resource_qualification"],
                    runtime_identity=inputs["terminal_runtime_identities"],
                    runtime=self.runtime,
                    fixture_rows={row["layout_pair_id"]: row for row in fixture_rows()},
                    prerequisites=prerequisite_identities(),
                )
            self.assertEqual(
                (
                    partial_derived[("N3", "C02")]["state"],
                    partial_derived[("N3", "C02")]["completed_prefix_cells"],
                ),
                ("technical_invalid", 1),
            )

            reordered = [ledger_rows[1], ledger_rows[0], *ledger_rows[2:]]
            bad_order = write_json(directory / "bad-order.json", ledger_document(reordered))
            with self.assertRaisesRegex(wave.ConfirmationReleaseWaveError, "block order"):
                wave.validate_result_ledger(
                    descriptor=bad_order, study_commit=COMMIT, branch="reduced_n3",
                    source_root=ROOT, state_dir=directory, evidence=FAKE_EVIDENCE,
                    models=("N3",), layout_order=ORDER, blocks=blocks(("N3",)),
                    schedule_identity=inputs["prepared_schedule"],
                    release_identity=inputs["confirmation_freeze"], fixture_identity=inputs["fixture_freeze"],
                    resource_identity=inputs["resource_qualification"],
                    runtime_identity=inputs["terminal_runtime_identities"], runtime=self.runtime,
                    fixture_rows={row["layout_pair_id"]: row for row in fixture_rows()},
                    prerequisites=prerequisite_identities(),
                )

            attempt3 = historical_attempt(3, "technical_invalid", attempt2["attempt_id"])
            successor_rows = json.loads(json.dumps(ledger_rows))
            successor_rows[0]["attempts"].append(attempt3)
            bad_successor = write_json(directory / "bad-successor.json", ledger_document(successor_rows))
            with mock.patch.object(wave, "_native_cell_state", side_effect=native):
                with self.assertRaisesRegex(wave.ConfirmationReleaseWaveError, "attempt follows terminal"):
                    wave.validate_result_ledger(
                        descriptor=bad_successor, study_commit=COMMIT, branch="reduced_n3",
                        source_root=ROOT, state_dir=directory, evidence=FAKE_EVIDENCE,
                        models=("N3",), layout_order=ORDER, blocks=blocks(("N3",)),
                        schedule_identity=inputs["prepared_schedule"],
                        release_identity=inputs["confirmation_freeze"], fixture_identity=inputs["fixture_freeze"],
                        resource_identity=inputs["resource_qualification"],
                        runtime_identity=inputs["terminal_runtime_identities"], runtime=self.runtime,
                        fixture_rows={row["layout_pair_id"]: row for row in fixture_rows()},
                        prerequisites=prerequisite_identities(),
                    )

            tampered_claim = json.loads(
                Path(attempt2["queue_claims"][0]["path"]).read_text(encoding="utf-8")
            )
            tampered_claim["descriptor_sha256"] = "0" * 64
            attempt2["queue_claims"][0] = write_json(
                Path(attempt2["queue_claims"][0]["path"]), tampered_claim
            )
            claim_tampered_ledger = write_json(
                directory / "claim-tampered-ledger.json", ledger_document(ledger_rows)
            )
            with mock.patch.object(wave, "_native_cell_state", side_effect=native):
                with self.assertRaisesRegex(
                    wave.ConfirmationReleaseWaveError, "queue claim owner"
                ):
                    wave.validate_result_ledger(
                        descriptor=claim_tampered_ledger, study_commit=COMMIT,
                        source_root=ROOT, state_dir=directory, evidence=FAKE_EVIDENCE,
                        branch="reduced_n3", models=("N3",), layout_order=ORDER,
                        blocks=blocks(("N3",)), schedule_identity=inputs["prepared_schedule"],
                        release_identity=inputs["confirmation_freeze"],
                        fixture_identity=inputs["fixture_freeze"],
                        resource_identity=inputs["resource_qualification"],
                        runtime_identity=inputs["terminal_runtime_identities"], runtime=self.runtime,
                        fixture_rows={row["layout_pair_id"]: row for row in fixture_rows()},
                        prerequisites=prerequisite_identities(),
                    )

    def _mocked_build(self, directory: Path, *, models=("N3",), limits=None,
                      workers=None, ledger_overrides=None, custom_ledger=None):
        limits = limits or {model: 1 for model in models}
        rows = fixture_rows()
        input_paths = {}
        for name in (
            "confirmation_freeze", "resource_qualification", "terminal_runtime_identities",
            "result_attempt_ledger", "cluster_reconciliation",
        ):
            input_paths[name] = write_json(directory / f"{name}.json", {"name": name})
        input_paths["fixture_freeze"] = write_json(directory / "fixture_freeze.json", {"layouts": rows})
        fake_runtime = dict(self.runtime)
        fake_runtime["fixture"] = FakeFixture(rows)
        fake_runtime["contract"] = self.contract
        fake_blocks = blocks(models)
        ledger_blocks = custom_ledger or empty_ledger_blocks(models, overrides=ledger_overrides)
        branch = {("N3",): "reduced_n3", ("D1",): "reduced_d1", ("N3", "D1"): "full_two_model"}[tuple(models)]
        release = {
            "cohort_branch": branch,
            "resource_budget": {"by_model": {model: {"max_parallel_blocks": limits[model]} for model in models}},
        }
        finalizer_lane = (
            "d1" if workers and workers[0].get("role") == "d1" else "n3"
        )
        finalizer_id = f"confirmation-release-finalizer-{finalizer_lane}-a001"
        reconciliation = {
            "created_at_utc": AS_OF,
            "consume_by_utc": CONSUME_BY,
            "observed_control_generation_lower_bound": 999,
            "observed_control_generation": 999,
            "control_semantics": {"active_job_ids": [finalizer_id]},
        }
        input_paths["cluster_reconciliation"] = write_json(
            directory / "cluster_reconciliation.json", reconciliation
        )
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(wave, "validate_study_commit"))
            stack.enter_context(mock.patch.object(wave, "load_runtime_modules", return_value=fake_runtime))
            stack.enter_context(mock.patch.object(
                wave, "load_release_evidence_module", return_value=FAKE_EVIDENCE,
            ))
            stack.enter_context(mock.patch.object(
                wave, "validate_schedule",
                return_value=({"path": "/schedule.json", "bytes": 1, "sha256": "5" * 64}, ORDER, fake_blocks),
            ))
            stack.enter_context(mock.patch.object(
                wave, "validate_release_freeze", return_value=(release, {}, tuple(models)),
            ))
            stack.enter_context(mock.patch.object(
                wave, "validate_resource_qualification", return_value=({}, limits),
            ))
            stack.enter_context(mock.patch.object(
                wave, "validate_runtime_identities", return_value=({}, prerequisite_identities()),
            ))
            stack.enter_context(mock.patch.object(
                wave, "validate_result_ledger", return_value=({}, ledger_blocks, set(), set(), {}),
            ))
            stack.enter_context(mock.patch.object(wave, "_validate_selected_prerequisites"))
            stack.enter_context(mock.patch.object(
                wave, "validate_pending_reconciliation",
                return_value=(reconciliation, workers or [], set()),
            ))
            result = wave.build_pending_release_wave(
                source_root=ROOT, state_dir=directory, study_commit=COMMIT,
                confirmation_freeze=Path(input_paths["confirmation_freeze"]["path"]),
                confirmation_freeze_sha256=input_paths["confirmation_freeze"]["sha256"],
                fixture_freeze=Path(input_paths["fixture_freeze"]["path"]),
                fixture_freeze_sha256=input_paths["fixture_freeze"]["sha256"],
                resource_qualification=Path(input_paths["resource_qualification"]["path"]),
                resource_qualification_sha256=input_paths["resource_qualification"]["sha256"],
                terminal_runtime_identities=Path(input_paths["terminal_runtime_identities"]["path"]),
                terminal_runtime_identities_sha256=input_paths["terminal_runtime_identities"]["sha256"],
                result_attempt_ledger=Path(input_paths["result_attempt_ledger"]["path"]),
                result_attempt_ledger_sha256=input_paths["result_attempt_ledger"]["sha256"],
                cluster_reconciliation=Path(input_paths["cluster_reconciliation"]["path"]),
                cluster_reconciliation_sha256=input_paths["cluster_reconciliation"]["sha256"],
            )
        return result

    def _published_verification(self, directory: Path, result: dict) -> tuple[dict, object]:
        publication_commit = "b" * 40
        verified_head = "c" * 40
        published = directory / "published"
        identities = {
            "plan": write_json(published / "confirmation_release_plan.json", result["plan"]),
            "queue_fragment": write_json(
                published / "confirmation_release_queue_fragment.json",
                result["queue_fragment"],
            ),
            "wave_receipt": write_json(
                published / "confirmation_release_wave_receipt.json", result["receipt"]
            ),
        }
        ledger_descriptor = result["plan"]["input_identities"]["result_attempt_ledger"]
        reconciliation_descriptor = result["plan"]["input_identities"]["cluster_reconciliation"]
        ledger = json.loads(Path(ledger_descriptor["path"]).read_text(encoding="utf-8"))
        reconciliation = json.loads(
            Path(reconciliation_descriptor["path"]).read_text(encoding="utf-8")
        )
        artifacts = {
            "ledger": ledger_descriptor,
            "reconciliation": reconciliation_descriptor,
            **identities,
        }
        evidence_receipt = sign({"status": "pending_coordinator_publication_no_science"})
        worker_attestation = sign({"status": "idle_no_gpu_compute_process"})
        artifact_paths = {
            name: f"results/jobs/confirmation-release-finalizer-n3-a001/{name}.json"
            for name in (*artifacts, "evidence_receipt", "worker_attestation")
        }
        expected_paths = {
            **artifact_paths,
            "pending": (
                "results/jobs/confirmation-release-finalizer-n3-a001/"
                "publication_pending.json"
            ),
            "cluster_status": "results/wmf_ablation_001_20260912/status.json",
            "producer_job_status": (
                "results/wmf_ablation_001_20260912/jobs/"
                "confirmation-release-finalizer-n3-a001.json"
            ),
            "producer_publish_manifest": (
                "results/jobs/confirmation-release-finalizer-n3-a001/"
                "publish_manifest.json"
            ),
        }
        pending = sign({
            "study_commit": COMMIT,
            "consume_by_utc": CONSUME_BY,
            "producer_queue_job": {"job_id": "confirmation-release-finalizer-n3-a001"},
            "artifacts": artifacts,
            "expected_h1_paths": expected_paths,
        })
        remote = {}
        serialized = {
            "ledger": ledger,
            "reconciliation": reconciliation,
            "plan": result["plan"],
            "queue_fragment": result["queue_fragment"],
            "wave_receipt": result["receipt"],
            "publication_pending": pending,
            "evidence_receipt": evidence_receipt,
            "worker_attestation": worker_attestation,
        }
        for name, value in serialized.items():
            blob = wave.pretty_bytes(value)
            remote[name] = {
                "commit": publication_commit,
                "git_path": expected_paths["pending" if name == "publication_pending" else name],
                "bytes": len(blob),
                "sha256": wave.sha256_bytes(blob),
            }
        for name in ("cluster_status", "producer_job_status", "producer_publish_manifest"):
            remote[name] = {
                "commit": publication_commit,
                "git_path": expected_paths[name],
                "bytes": 1,
                "sha256": "e" * 64,
            }
        producer_result = {"job_id": "confirmation-release-finalizer-n3-a001"}
        document = sign({
            "schema_version": wave.PUBLICATION_VERIFICATION_SCHEMA,
            "status": "coordinator_publication_verified_read_only",
            "study_id": wave.STUDY_ID,
            "namespace": wave.NAMESPACE,
            "study_commit": COMMIT,
            "verified_at_utc": AS_OF,
            "consume_by_utc": CONSUME_BY,
            "results_remote": self.contract["trusted_publication"]["repository_identity"],
            "results_ref": self.contract["trusted_publication"]["results_ref"],
            "observed_results_ancestor_commit": "d" * 40,
            "pending_publication_commit": publication_commit,
            "verified_remote_head": verified_head,
            "intervening_commit_count": 0,
            "later_status_only_commit_count": 0,
            "finalizer_job_id": "confirmation-release-finalizer-n3-a001",
            "artifact_descriptors": remote,
            "producer_outer_result": producer_result,
            "semantic_control": {"active_job_ids": ["confirmation-release-finalizer-n3-a001"]},
            "queue_fragment_release_gate_passed": True,
            "results_branch_written_by_verifier": False,
            "study_worktree_mutated": False,
            "claim_boundary": "read-only coordinator publication verification",
        })
        verification = {
            "document": document,
            "publication_commit": publication_commit,
            "verified_remote_head": verified_head,
            "artifact_descriptors": remote,
            "cluster_status": {},
            "producer_result": producer_result,
            "artifacts": {**serialized},
        }
        evidence = SimpleNamespace(
            validate_publication_pending=lambda value, **_kwargs: dict(value),
            verify_published_finalizer=lambda **_kwargs: verification,
        )
        return verification, evidence

    def test_pending_reconciliation_is_native_replayed_and_generation_998_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = sign({
                "created_at_utc": AS_OF,
                "consume_by_utc": CONSUME_BY,
                "observed_control_generation_lower_bound": 999,
                "observed_control_generation": 1002,
                "control_semantics": {
                    "active_job_ids": ["confirmation-release-finalizer-n3-a001"]
                },
            })
            identity = write_json(directory / "reconciliation.json", value)
            ledger_identity = write_json(directory / "ledger.json", {"ledger": True})
            ledger_validation = {
                "document": {"ledger": True}, "blocks": {}, "attempt_ids": set(),
                "job_ids": set(), "queue_claims": {},
            }
            idle = worker("n3-live-01", "n3", 2, "1" * 64)

            class Evidence:
                @staticmethod
                def queue_triplet(**_kwargs):
                    return {"job_id": "confirmation-release-finalizer-n3-a001"}

                @staticmethod
                def validate_pending_reconciliation(document, **_kwargs):
                    return {
                        "document": document,
                        "post_exit_workers": [idle],
                        "control_semantics": document["control_semantics"],
                        "all_prior_job_ids": ["confirmation-release-finalizer-n3-a001"],
                        "authorized_capacity_worker_ids": ["n3-live-01"],
                        "ledger_validation": ledger_validation,
                        "all_queue_jobs": {
                            "confirmation-release-finalizer-n3-a001": {}
                        },
                    }

            loaded, workers, prior = wave.validate_pending_reconciliation(
                descriptor=identity, ledger_identity=ledger_identity,
                ledger_validation=ledger_validation, all_attempt_ids=set(),
                all_job_ids=set(), source_root=ROOT, state_dir=directory,
                study_commit=COMMIT, contract=self.contract,
                runtime=self.runtime, evidence=Evidence,
            )
            self.assertEqual(loaded["observed_control_generation"], 1002)
            self.assertEqual([row["worker_id"] for row in workers], ["n3-live-01"])
            self.assertEqual(prior, {"confirmation-release-finalizer-n3-a001"})

            stale = json.loads(json.dumps(value))
            stale["observed_control_generation_lower_bound"] = 998
            stale.pop("payload_sha256")
            stale_identity = write_json(directory / "stale.json", sign(stale))
            with self.assertRaisesRegex(wave.ConfirmationReleaseWaveError, "generation"):
                wave.validate_pending_reconciliation(
                    descriptor=stale_identity, ledger_identity=ledger_identity,
                    ledger_validation=ledger_validation, all_attempt_ids=set(),
                    all_job_ids=set(), source_root=ROOT, state_dir=directory,
                    study_commit=COMMIT, contract=self.contract,
                    runtime=self.runtime, evidence=Evidence,
                )

    def test_n3_wave_uses_minimum_of_signed_cap_idle_workers_and_remaining_hash_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            workers = [
                worker("n3-live-01", "n3", 2, "1" * 64),
                worker("n3-live-02", "n3", 2, "1" * 64),
            ]
            result = self._mocked_build(
                directory, models=("N3",), limits={"N3": 3}, workers=workers
            )
            self.assertEqual(result["plan"]["selected_layouts"], ["C02", "C13"])
            self.assertEqual(len(result["queue_fragment"]["jobs"]), 2)
            self.assertEqual(result["plan"]["wave_model"], "N3")
            self.assertEqual(result["plan"]["status"], "pending_h1_verification")
            self.assertFalse(result["plan"]["confirmation_release"])
            self.assertFalse(result["plan"]["execution_release_authorized"])
            self.assertTrue(result["plan"]["post_publication_verification_required"])
            self.assertFalse(result["receipt"]["confirmation_release"])
            self.assertFalse(result["receipt"]["execution_release_authorized"])
            self.assertFalse(result["plan"]["outcome_dependent_ordering"])
            self.assertEqual(len(result["plan"]["layout_accounting_by_model"]["N3"]), 24)
            self.assertEqual(
                result["plan"]["input_identities"]["release_wave_builder"],
                wave.file_identity(MODULE_PATH),
            )
            self.assertEqual(
                result["receipt"]["input_identities"]["release_wave_contract"],
                wave.file_identity(LAYOUT / wave.CONTRACT_FILENAME),
            )
            self.assertTrue(all(job["publish_log_tail_bytes"] == 0 for job in result["queue_fragment"]["jobs"]))
            repeat = self._mocked_build(
                directory, models=("N3",), limits={"N3": 3}, workers=workers
            )
            self.assertEqual(result["plan"], repeat["plan"])
            self.assertEqual(result["queue_fragment"], repeat["queue_fragment"])
            self.assertEqual(result["receipt"], repeat["receipt"])

    def test_current_conservative_n3_cap_emits_one_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workers = [
                worker("n3-live-01", "n3", 2, "1" * 64),
                worker("n3-live-02", "n3", 2, "1" * 64),
            ]
            result = self._mocked_build(
                Path(temporary), models=("N3",), limits={"N3": 1}, workers=workers
            )
            self.assertEqual(result["plan"]["selected_layouts"], ["C02"])
            self.assertEqual(len(result["queue_fragment"]["jobs"]), 1)

    def test_post_publication_gate_is_the_only_authorized_descriptor_return(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = self._mocked_build(
                directory, models=("N3",), limits={"N3": 1},
                workers=[worker("n3-live-01", "n3", 2, "1" * 64)],
            )
            verification, evidence = self._published_verification(directory, result)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(wave, "validate_study_commit"))
                stack.enter_context(
                    mock.patch.object(wave, "load_release_evidence_module", return_value=evidence)
                )
                released = wave.verify_published_release_wave(
                    source_root=ROOT,
                    finalizer_job_id="confirmation-release-finalizer-n3-a001",
                    study_commit=COMMIT,
                )
            self.assertTrue(released["authorization"]["confirmation_release"])
            self.assertTrue(released["authorization"]["execution_release_authorized"])
            self.assertFalse(released["authorization"]["queue_mutated"])
            self.assertEqual(released["queue_fragment"], result["queue_fragment"])
            self.assertEqual(
                released["authorization"]["publication_verification_payload_sha256"],
                verification["document"]["payload_sha256"],
            )
            active_queue = (
                ROOT
                / "workshops/corl2026_world_models/execution/20260912/autonomy/cluster_queue.json"
            )
            active_before = active_queue.read_bytes()
            (directory / "authorized").mkdir()
            written = wave.write_authorized_release_wave(
                output_dir=directory / "authorized", source_root=ROOT, result=released
            )
            self.assertEqual(
                set(written),
                {
                    "authorization", "plan", "queue_fragment", "receipt",
                    "publication_verification",
                },
            )
            self.assertEqual(active_queue.read_bytes(), active_before)

            forged = json.loads(json.dumps(released))
            forged["authorization"]["confirmation_release"] = False
            forged["authorization"].pop("payload_sha256")
            forged["authorization"] = sign(forged["authorization"])
            with self.assertRaisesRegex(
                wave.ConfirmationReleaseWaveError,
                "authorization is detached or not executable",
            ):
                wave.write_authorized_release_wave(
                    output_dir=directory / "forged-authorized",
                    source_root=ROOT,
                    result=forged,
                )
            self.assertFalse((directory / "forged-authorized").exists())
            self.assertEqual(active_queue.read_bytes(), active_before)

            unavailable = SimpleNamespace(
                validate_publication_pending=lambda value, **_kwargs: dict(value),
                verify_published_finalizer=mock.Mock(side_effect=RuntimeError("H1 absent")),
            )
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(wave, "validate_study_commit"))
                stack.enter_context(
                    mock.patch.object(wave, "load_release_evidence_module", return_value=unavailable)
                )
                with self.assertRaisesRegex(
                    wave.ConfirmationReleaseWaveError,
                    "coordinator publication failed native read-only verification",
                ):
                    wave.verify_published_release_wave(
                        source_root=ROOT,
                        finalizer_job_id="confirmation-release-finalizer-n3-a001",
                        study_commit=COMMIT,
                    )

    def test_post_publication_gate_rejects_coordinated_queue_and_verification_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = self._mocked_build(
                directory, models=("N3",), limits={"N3": 1},
                workers=[worker("n3-live-01", "n3", 2, "1" * 64)],
            )
            verification, evidence = self._published_verification(directory, result)

            changed = json.loads(json.dumps(verification))
            changed["artifacts"]["queue_fragment"]["jobs"][0]["argv"].extend(
                ["--job-id", changed["artifacts"]["queue_fragment"]["jobs"][0]["job_id"]]
            )
            queue_blob = wave.pretty_bytes(changed["artifacts"]["queue_fragment"])
            queue_sha = wave.sha256_bytes(queue_blob)
            changed["artifacts"]["plan"]["queue_fragment_sha256"] = queue_sha
            changed["artifacts"]["plan"].pop("payload_sha256")
            changed["artifacts"]["plan"] = sign(changed["artifacts"]["plan"])
            changed["artifacts"]["wave_receipt"]["queue_fragment_sha256"] = queue_sha
            changed["artifacts"]["wave_receipt"]["plan_sha256"] = wave.sha256_bytes(
                wave.pretty_bytes(changed["artifacts"]["plan"])
            )
            changed["artifacts"]["wave_receipt"].pop("payload_sha256")
            changed["artifacts"]["wave_receipt"] = sign(
                changed["artifacts"]["wave_receipt"]
            )
            for name in ("queue_fragment", "plan", "wave_receipt"):
                blob = wave.pretty_bytes(changed["artifacts"][name])
                changed["artifacts"]["publication_pending"]["artifacts"][name]["bytes"] = len(blob)
                changed["artifacts"]["publication_pending"]["artifacts"][name]["sha256"] = wave.sha256_bytes(blob)
                changed["artifact_descriptors"][name]["bytes"] = len(blob)
                changed["artifact_descriptors"][name]["sha256"] = wave.sha256_bytes(blob)
            changed["artifacts"]["publication_pending"].pop("payload_sha256")
            changed["artifacts"]["publication_pending"] = sign(
                changed["artifacts"]["publication_pending"]
            )
            pending_blob = wave.pretty_bytes(changed["artifacts"]["publication_pending"])
            changed["artifact_descriptors"]["publication_pending"]["bytes"] = len(pending_blob)
            changed["artifact_descriptors"]["publication_pending"]["sha256"] = wave.sha256_bytes(pending_blob)
            changed["document"]["artifact_descriptors"] = changed["artifact_descriptors"]
            changed["document"].pop("payload_sha256")
            changed["document"] = sign(changed["document"])
            with self.assertRaisesRegex(wave.ConfirmationReleaseWaveError, "duplicate option"):
                wave._validate_published_wave_artifacts(
                    source_root=ROOT, study_commit=COMMIT, contract=self.contract,
                    runtime=self.runtime, evidence=evidence, verification=changed,
                )

            forged_gate = json.loads(json.dumps(verification))
            forged_gate["document"]["queue_fragment_release_gate_passed"] = False
            forged_gate["document"].pop("payload_sha256")
            forged_gate["document"] = sign(forged_gate["document"])
            with self.assertRaisesRegex(
                wave.ConfirmationReleaseWaveError, "does not authorize"
            ):
                wave._validate_published_wave_artifacts(
                    source_root=ROOT, study_commit=COMMIT, contract=self.contract,
                    runtime=self.runtime, evidence=evidence, verification=forged_gate,
                )

    def test_passed_and_safety_censored_blocks_never_rerun_technical_retries_are_fresh(self) -> None:
        ledger = empty_ledger_blocks(("N3",))
        ledger[("N3", "C02")]["state"] = "safety_censored"
        ledger[("N3", "C02")]["terminal_attempt_id"] = "old-censor"
        ledger[("N3", "C13")]["state"] = "passed"
        ledger[("N3", "C13")]["completed_prefix_cells"] = 4
        ledger[("N3", "C13")]["terminal_attempt_id"] = "old-pass"
        technical = ledger[("N3", "C11")]
        technical["state"] = "technical_invalid"
        technical["completed_prefix_cells"] = 2
        technical["attempts"] = [{"attempt_id": "confirmation-c11-n3-a001"}]
        with tempfile.TemporaryDirectory() as temporary:
            result = self._mocked_build(
                Path(temporary), models=("N3",), limits={"N3": 1},
                workers=[worker("n3-live-01", "n3", 2, "1" * 64)],
                custom_ledger=ledger,
            )
            attempt = result["plan"]["prospective_attempts"][0]
            self.assertEqual(attempt["layout_pair_id"], "C11")
            self.assertEqual(attempt["attempt_number"], 2)
            self.assertEqual(attempt["predecessor_attempt_id"], "confirmation-c11-n3-a001")
            self.assertEqual(attempt["start_cell_index"], 2)
            self.assertEqual(attempt["attempt_id"], "confirmation-c11-n3-a002")

    def test_d1_wave_is_one_noninterleaved_server_simulator_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self._mocked_build(
                Path(temporary), models=("D1",), limits={"D1": 8},
                workers=[
                    worker("d1-live-01", "d1", 2, "2" * 64),
                    worker("sim-live-00", "wmf-forecast-0912-worker-00", 1, "3" * 64),
                    worker("sim-live-05", "wmf-forecast-0912-worker-05", 1, "3" * 64),
                ],
            )
            jobs = result["queue_fragment"]["jobs"]
            self.assertEqual(result["plan"]["selected_layouts"], ["C02"])
            self.assertEqual(len(jobs), 2)
            self.assertEqual([job["role"] for job in jobs], ["d1", "wmf-forecast-0912-worker-00"])
            self.assertEqual(len(result["plan"]["prospective_attempts"]), 1)
            self.assertIn(jobs[1]["job_id"], jobs[0]["argv"])
            self.assertIn(jobs[0]["job_id"], jobs[1]["argv"])

    def test_full_branch_lane_priority_is_fixed_and_all_terminal_emits_no_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            both = empty_ledger_blocks(("N3", "D1"))
            first = self._mocked_build(
                directory / "n3", models=("N3", "D1"),
                workers=[worker("n3-live-01", "n3", 2, "1" * 64)],
                custom_ledger=both,
            )
            self.assertEqual((first["plan"]["wave_model"], first["plan"]["selected_layouts"]), ("N3", ["C02"]))

            for layout in ORDER:
                both[("N3", layout)]["state"] = "passed"
                both[("N3", layout)]["completed_prefix_cells"] = 4
                both[("N3", layout)]["terminal_attempt_id"] = f"n3-{layout}"
            second = self._mocked_build(
                directory / "d1", models=("N3", "D1"),
                workers=[
                    worker("d1-live-01", "d1", 2, "2" * 64),
                    worker(
                        "sim-live-00", "wmf-forecast-0912-worker-00", 1,
                        "3" * 64,
                    ),
                ],
                custom_ledger=both,
            )
            self.assertEqual((second["plan"]["wave_model"], second["plan"]["selected_layouts"]), ("D1", ["C02"]))

            for layout in ORDER:
                both[("D1", layout)]["state"] = "safety_censored"
                both[("D1", layout)]["terminal_attempt_id"] = f"d1-{layout}"
            complete = self._mocked_build(
                directory / "complete", models=("N3", "D1"),
                workers=[], custom_ledger=both,
            )
            self.assertIsNone(complete["plan"]["wave_model"])
            self.assertEqual(complete["queue_fragment"]["jobs"], [])
            self.assertEqual(complete["plan"]["status"], "pending_h1_verification")

    def test_writer_is_immutable_and_never_mutates_active_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = self._mocked_build(
                directory, models=("N3",), limits={"N3": 1},
                workers=[worker("n3-live-01", "n3", 2, "1" * 64)],
            )
            queue_path = ROOT / "workshops/corl2026_world_models/execution/20260912/autonomy/cluster_queue.json"
            before = queue_path.read_bytes()
            (directory / "out").mkdir()
            identities = wave.write_release_wave(
                output_dir=directory / "out", source_root=ROOT, result=result
            )
            self.assertEqual(set(identities), {"plan", "queue_fragment", "receipt"})
            self.assertEqual(queue_path.read_bytes(), before)
            self.assertEqual(
                identities,
                wave.write_release_wave(output_dir=directory / "out", source_root=ROOT, result=result),
            )
            changed = json.loads(json.dumps(result))
            changed["plan"]["selected_layouts"] = ["C13"]
            with self.assertRaisesRegex(wave.ConfirmationReleaseWaveError, "immutable output differs"):
                wave.write_release_wave(output_dir=directory / "out", source_root=ROOT, result=changed)
            with self.assertRaisesRegex(
                wave.ConfirmationReleaseWaveError, "pre-existing directory",
            ):
                wave.write_release_wave(
                    output_dir=directory / "missing", source_root=ROOT, result=result,
                )
            self.assertFalse((directory / "missing").exists())
            self.assertEqual(queue_path.read_bytes(), before)

    def test_json_loader_is_strict_and_binds_identity_to_same_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "input.json"
            original = b'{"nested":{"value":1}}\n'
            path.write_bytes(original)
            real_loads = json.loads

            def mutate_after_read(payload, *args, **kwargs):
                path.write_bytes(b'{"nested":{"value":2}}\n')
                return real_loads(payload, *args, **kwargs)

            with mock.patch.object(wave.json, "loads", side_effect=mutate_after_read):
                identity, value = wave.load_json_with_identity(path, "strict input")
            self.assertEqual(value, {"nested": {"value": 1}})
            self.assertEqual(identity["bytes"], len(original))
            self.assertEqual(identity["sha256"], hashlib.sha256(original).hexdigest())

            malformed = {
                "duplicate": b'{"value":1,"value":2}\n',
                "nested duplicate": b'{"nested":{"value":1,"value":2}}\n',
                "NaN": b'{"value":NaN}\n',
                "Infinity": b'{"value":Infinity}\n',
                "negative Infinity": b'{"value":-Infinity}\n',
                "overflow float": b'{"value":1e999}\n',
            }
            for label, payload in malformed.items():
                with self.subTest(label=label):
                    path.write_bytes(payload)
                    with self.assertRaises(wave.ConfirmationReleaseWaveError):
                        wave.load_json(path, label)

    def test_json_loader_rejects_symlinked_parent_component(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            real = directory / "real"
            real.mkdir()
            write_json(real / "input.json", {"value": 1})
            linked = directory / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(
                wave.ConfirmationReleaseWaveError, "path component is a symlink",
            ):
                wave.load_json(linked / "input.json", "linked input")

    def test_json_loader_rejects_parent_replacement_during_component_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            parent = directory / "parent"
            parent.mkdir()
            write_json(parent / "input.json", {"value": 1})
            replacement = directory / "replacement"
            replacement.mkdir()
            write_json(replacement / "input.json", {"value": 2})
            displaced = directory / "displaced"
            original_open = wave.os.open
            replaced = False

            def replace_before_open(path, flags, mode=0o777, *, dir_fd=None):
                nonlocal replaced
                if path == parent.name and dir_fd is not None and not replaced:
                    replaced = True
                    parent.rename(displaced)
                    replacement.rename(parent)
                return original_open(path, flags, mode, dir_fd=dir_fd)

            with mock.patch.object(wave.os, "open", side_effect=replace_before_open):
                with self.assertRaisesRegex(
                    wave.ConfirmationReleaseWaveError, "changed during open",
                ):
                    wave.load_json(parent / "input.json", "raced input")
            self.assertTrue(replaced)

    def test_writer_parent_swap_rolls_back_without_output_residue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            parent = directory / "parent"
            parent.mkdir()
            (parent / "out").mkdir()
            swapped = directory / "swapped"
            result = {
                "plan": {"kind": "plan"},
                "queue_fragment": {"kind": "fragment"},
                "receipt": {"kind": "receipt"},
            }
            queue_path = (
                ROOT / "workshops/corl2026_world_models/execution/20260912/"
                "autonomy/cluster_queue.json"
            )
            queue_before = queue_path.read_bytes()
            original_write = wave._write_regular_at
            swapped_once = False

            def write_then_swap(*args, **kwargs):
                nonlocal swapped_once
                observed = original_write(*args, **kwargs)
                if not swapped_once:
                    swapped_once = True
                    parent.rename(swapped)
                    parent.mkdir()
                return observed

            with mock.patch.object(wave, "_write_regular_at", side_effect=write_then_swap):
                with self.assertRaises(wave.ConfirmationReleaseWaveError):
                    wave.write_release_wave(
                        output_dir=parent / "out", source_root=ROOT, result=result,
                    )
            self.assertTrue(swapped_once)
            self.assertTrue((swapped / "out").is_dir())
            self.assertFalse((parent / "out").exists())
            self.assertEqual(list(swapped.rglob("*.json")), [])
            self.assertEqual(list(parent.rglob("*.json")), [])
            self.assertEqual(queue_path.read_bytes(), queue_before)

    def test_input_hash_and_payload_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            identity = write_json(path, {"value": 1})
            path.write_text('{"value":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(wave.ConfirmationReleaseWaveError, "SHA-256 mismatch"):
                wave.verify_input(path, identity["sha256"], "input")
            signed = sign({"value": 1})
            signed["value"] = 2
            with self.assertRaisesRegex(wave.ConfirmationReleaseWaveError, "payload hash mismatch"):
                wave.verify_signed_document(signed, "input")


if __name__ == "__main__":
    unittest.main()
