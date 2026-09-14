"""Synthetic full-ledger assembly integration; this is not recording qualification.

The scientific-context seam supplies the real 24-layout schedule with temporary
PVC roots and explicitly synthetic prerequisite documents. The remote history
fetch is replaced by a local Git commit descended from the published source.
N3's base recorder validator is replaced by loading the synthetic cell document;
its confirmation validator and server-terminal protocol/validator remain real.
D1's base passed-recorder, deep model-ready and deep server-terminal validators
load synthetic sidecars. Its confirmation passed/failure validators still bind
the real claim, ready, run/pair, prerequisites, session and terminal descriptors.
Assembler, filesystem inventory, queue triplets, Git claim-history validation,
runtime-admission validation, native cell-state and both ledger replays are real.
No production prerequisites, labels, model requests or behavioral cells are produced.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
import dataclasses
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
LAYOUT = ROOT / "workshops/corl2026_world_models/experiments/forecast_layout"
STUDY_COMMIT = "1af6e4228b5c213718e002a840f9a7784eca5c76"
CHECKED = "2026-09-14T01:00:00Z"
CONSUME_BY = "2026-09-14T01:04:00Z"
FINALIZER = "confirmation-release-finalizer-n3-a001"
QUEUE_PATH = "workshops/corl2026_world_models/execution/20260912/autonomy/cluster_queue.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evidence = load(LAYOUT / "confirmation_release_evidence_jobs.py", "wmf_assembly_evidence")


def write_json(path: Path, document: dict, *, queue_encoding=False) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(document, sort_keys=True, indent=2).encode() + b"\n"
        if queue_encoding else evidence.pretty_bytes(document)
    )
    path.write_bytes(payload)
    return evidence.file_identity(path)


class ConfirmationLedgerAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = evidence.load_modules(ROOT)
        cls.wave = cls.modules["wave"]
        cls.queue = cls.modules["queue"]
        cls.contract = cls.wave.load_contract(ROOT)
        cls.sim_role = cls.contract["runtime"]["D1"]["simulator_roles"][0]
        cls.runtime = cls.wave.load_runtime_modules(ROOT)
        cls.runtime["contract"] = cls.contract
        cls.schedule, cls.order, cls.blocks = cls.wave.validate_schedule(
            source_root=ROOT, contract=cls.contract, runtime=cls.runtime,
        )

    def _git(self, repository: Path, *args: str, input=None) -> str:
        environment = {
            **os.environ,
            "GIT_AUTHOR_NAME": "Synthetic ledger test",
            "GIT_AUTHOR_EMAIL": "synthetic@example.invalid",
            "GIT_COMMITTER_NAME": "Synthetic ledger test",
            "GIT_COMMITTER_EMAIL": "synthetic@example.invalid",
            "GIT_AUTHOR_DATE": CHECKED,
            "GIT_COMMITTER_DATE": CHECKED,
        }
        return subprocess.run(
            ["git", "-C", str(repository), *args], input=input,
            text=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environment,
        ).stdout.strip()

    def _fixture(self, root: Path, *, passed_cells: int, outer_status: str,
                 model="N3", states=None, attempt_specs=None) -> dict:
        wave, queue, n3 = self.wave, self.queue, self.runtime["n3"]
        state = root / "state"
        (state / "jobs").mkdir(parents=True)
        (state / "sources" / STUDY_COMMIT).mkdir(parents=True)
        blocks = {
            key: dataclasses.replace(block, raw_root=root / "raw" / key[0] / key[1])
            for key, block in self.blocks.items()
        }
        block = blocks[(model, self.order[0])]
        identities = {
            name: write_json(root / "synthetic_inputs" / (name + ".json"), {
                "synthetic_test_only": True, "prerequisite": name,
            })
            for name in (
                "confirmation_freeze", "fixture_freeze", "resource_qualification",
                "terminal_runtime_identities",
            )
        }
        prerequisites = {
            name: write_json(root / "synthetic_inputs" / (name + ".json"), {
                "synthetic_test_only": True, "prerequisite": name,
            })
            for name in wave.PREREQUISITE_NAMES
        }
        fixture_rows = {
            layout: {
                "layout_pair_id": layout, "candidate_id": layout + "__candidate_00",
                **{
                    name: write_json(root / "synthetic_inputs" / layout / (name + ".json"), {
                        "synthetic_test_only": True, "layout": layout, "kind": name,
                    })
                    for name in ("gate_receipt", "pose_manifest", "capture_receipt")
                },
            }
            for layout in self.order
        }
        finalizer = FINALIZER.replace("-n3-", f"-{model.lower()}-")
        attempt_specs = attempt_specs or [(states or ["passed"] * passed_cells, outer_status)]
        attempts = []
        raw_jobs = []
        for number, (cell_states, status) in enumerate(attempt_specs, start=1):
            args = dict(
                layout=block.layout_pair_id, attempt_number=number, study_commit=STUDY_COMMIT,
                fixture=fixture_rows[block.layout_pair_id],
                release_identity=identities["confirmation_freeze"],
                fixture_identity=identities["fixture_freeze"], prerequisites=prerequisites,
                finalizer_job_id=finalizer, consume_by_utc=CONSUME_BY,
                contract=self.contract, queue=queue,
            )
            if model == "N3":
                attempt_id, raw, descriptor = wave._n3_descriptor(**args)
                raws, normalized = [raw], [descriptor]
            else:
                attempt_id, raws, normalized = wave._d1_descriptors(simulator_role=self.sim_role, **args)
            attempts.append({"attempt_id": attempt_id, "raws": raws, "descriptors": normalized,
                             "states": cell_states, "outer_status": status})
            raw_jobs.extend(raws)
        repository = root / "local_control_history"
        subprocess.run(
            ["git", "clone", "--shared", "--no-checkout", str(ROOT), str(repository)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self._git(repository, "read-tree", STUDY_COMMIT)
        queue_manifest = {
            "schema_version": evidence.QUEUE_SCHEMA, "namespace": evidence.NAMESPACE,
            "shutdown": False, "jobs": raw_jobs,
        }
        queue_identity = write_json(repository / QUEUE_PATH, queue_manifest)
        self._git(repository, "add", "--", QUEUE_PATH)
        tree = self._git(repository, "write-tree")
        control_commit = self._git(repository, "commit-tree", tree, "-p", STUDY_COMMIT,
                                   input="Synthetic ledger test control only\n")
        control_identity = write_json(state / "control.json", {
            "namespace": evidence.NAMESPACE, "control_commit": control_commit,
            "control_generation": 1002, "admission_deadline_unix": 1789479999,
            "shutdown": False, "active_job_ids": [finalizer],
        })
        prefix = 0
        for attempt in attempts:
            admissions = []
            job_ids = [row["job_id"] for row in attempt["descriptors"]]
            attempt_id = attempt["attempt_id"]
            if model == "N3":
                roots = [Path(block.raw_root) / attempt_id]
            else:
                roots = [Path(block.raw_root) / "server_attempts" / job_ids[0],
                         Path(block.raw_root) / "simulator_attempts" / job_ids[1]]
            for descriptor, attempt_root in zip(attempt["descriptors"], roots, strict=True):
                job_id = descriptor["job_id"]
                job_root = state / "jobs" / job_id
                descriptor_identity = write_json(job_root / "descriptor.json", descriptor, queue_encoding=True)
                self.assertEqual(descriptor_identity["sha256"], evidence.sha256_bytes(queue.encode(descriptor)))
                role = descriptor["role"]
                worker = {"n3": "wmf-forecast-0912-worker-n3-00", "d1": "wmf-forecast-0912-worker-d1-00",
                          self.sim_role: "wmf-forecast-0912-worker-00"}[role]
                claim = {
                    "worker_id": worker, "claimed_at": CHECKED,
                    "claimed_unix": datetime.fromisoformat(CHECKED.replace("Z", "+00:00")).timestamp(),
                    "worker_pid": 123, "control_commit": control_commit,
                    "control_generation": 1001, "descriptor_sha256": descriptor_identity["sha256"],
                    "release_boundary": "claim_committed_under_shared_release_lock",
                }
                claim_identity = write_json(job_root / "claim" / "owner.json", claim)
                status = attempt["outer_status"]
                result = {
                    "schema_version": evidence.RESULT_SCHEMA, "namespace": evidence.NAMESPACE,
                    "job_id": job_id, "worker_id": worker, "source_commit": STUDY_COMMIT,
                    "descriptor_sha256": descriptor_identity["sha256"], "started_at": CHECKED,
                    "argv": evidence._expanded_argv(
                        descriptor, source_root=state / "sources" / STUDY_COMMIT,
                        job_dir=job_root, state_dir=state,
                    ),
                    "job_dir": str(job_root), "status": status,
                    "returncode": 0 if status == "succeeded" else 1,
                    "error_type": None if status == "succeeded" else "SyntheticFailure",
                    "ended_at": "2026-09-14T01:30:00Z", "wall_seconds": 1800,
                    "child_pid": 456, "child_reaped": True,
                }
                for stream in ("stdout", "stderr"):
                    log = job_root / (stream + ".log")
                    log.write_bytes(b"")
                    identity = evidence.file_identity(log)
                    result[stream] = {key: identity[key] for key in ("bytes", "sha256")}
                write_json(job_root / "result.json", result)
                publication_hash = evidence.sha256_bytes(evidence.pretty_bytes(queue_manifest))
                admission = wave.signed_document({
                    "schema_version": wave.RUNTIME_ADMISSION_SCHEMA,
                    "status": "authenticated_before_runtime_start", "study_id": wave.STUDY_ID,
                    "namespace": wave.NAMESPACE, "study_commit": STUDY_COMMIT,
                    "checked_at_utc": CHECKED, "consume_by_utc": CONSUME_BY, "stage": "queue_start",
                    "release_finalizer_job_id": finalizer,
                    "publication_verification_payload_sha256": "b" * 64,
                    "pending_publication_commit": "c" * 40, "verified_remote_head": "d" * 40,
                    "published_queue_fragment": {
                        "commit": "c" * 40, "git_path": "results/synthetic/queue.json",
                        "bytes": len(evidence.pretty_bytes(queue_manifest)), "sha256": publication_hash,
                    },
                    "published_queue_fragment_sha256": publication_hash,
                    "queue_job_ids": job_ids, "job_id": job_id, "worker_id": worker,
                    "role": role, "queue_descriptor": descriptor_identity, "queue_claim": claim_identity,
                    "claim_control": {
                        "job_id": job_id, "control_commit": control_commit, "control_generation": 1001,
                        "control_queue_blob_sha256": queue_identity["sha256"],
                        "descriptor_sha256": descriptor_identity["sha256"],
                    },
                    "live_control": control_identity, "hostname": f"synthetic-{role}-host",
                    "pod_uid": "11111111-1111-1111-1111-111111111111",
                    "pod_uid_source": "downward_api_metadata_uid",
                    "gpu_identity": [
                        {"index": index, "uuid": f"GPU-synthetic-{role}-{index}", "name": "NVIDIA B200",
                         "memory_total_mib": 192000}
                        for index in range(1 if role == self.sim_role else 2)
                    ],
                    "idle_gpu_required": True, "gpu_compute_processes_observed": False,
                    "science_reset_request_action_started": False, "queue_mutated": False,
                    "jobs_dispatched": 0, "claim_boundary": wave.RUNTIME_ADMISSION_CLAIM_BOUNDARY,
                })
                admission_identity = write_json(attempt_root / "release_admission.json", admission)
                admissions.append((admission, admission_identity))
                filename, schema = {
                    "n3": ("n3_behavioral_confirmation_receipt.json", "wmf-n3-behavioral-confirmation-job-v1"),
                    "d1": ("d1_behavioral_server_receipt.json", "wmf-d1-behavioral-confirmation-server-job-v1"),
                    self.sim_role: ("d1_behavioral_confirmation_receipt.json", "wmf-d1-behavioral-confirmation-simulator-job-v1"),
                }[role]
                write_json(attempt_root / "publish" / filename, {
                    "schema_version": schema,
                    "status": "passed" if status == "succeeded" else "technical_failure",
                    "release_admission": admission_identity,
                    "all_server_children_reaped": True, "server_process_exit": {"status": "reaped", "reaped": True},
                    "all_simulator_children_reaped": True,
                })
            if model == "N3":
                write_json(roots[0] / "cleanup.json", {
                    "schema_version": "wmf-n3-behavioral-confirmation-cleanup-v1",
                    "all_children_reaped": True, "remaining_compute_processes": [],
                })
                with n3.configured_pilot(block):
                    for index, cell_state in enumerate(attempt["states"], start=prefix):
                        self._native_n3_passed(roots[0], block, index, state=cell_state)
            else:
                with self.runtime["d1"].configured_pilot(block, self.sim_role):
                    self._native_d1_attempt(block, attempt, roots, admissions, finalizer, prefix)
            prefix += attempt["states"].count("passed")
        return {
            "state": state, "context": {
                "wave": wave, "contract": self.contract, "runtime": self.runtime,
                "schedule_identity": self.schedule, "layout_order": self.order, "blocks": blocks,
                "release": {"cohort_branch": f"reduced_{model.lower()}"}, "models": (model,),
                "fixture_rows": fixture_rows, "prerequisites": prerequisites, "identities": identities,
            },
            "history": {"repo": str(repository), "head": control_commit},
            "block": block, "attempt": attempts[0]["attempt_id"], "attempts": attempts,
            "model": model, "finalizer": finalizer,
        }

    def _native_n3_passed(self, attempt_root: Path, block, index: int, state="passed") -> None:
        n3 = self.runtime["n3"]
        protocol = n3.pilot.ServerProtocol(start_cell_index=index)
        arm, command, _task = block.conditions[index]
        begin = protocol.begin({
            "study_id": n3.pilot.STUDY_ID, "block_id": block.block_id,
            "cell_id": block.cell_ids[index], "condition_index": index, "layout_arm": arm,
            "command": command, "prompt": n3.pilot.PROMPTS[command],
            "effective_seed": block.effective_seed, "expected_actions": 450,
            "expected_requests": n3.pilot.REQUEST_COUNT, "client_session_id": f"synthetic-client-{index}",
        }, server_context_id=f"synthetic-context-{index}", temporal_reset_evidence={
            "passed": True, "unresolved_mutable_temporal_fields": [],
        })
        requests = n3.pilot.REQUEST_COUNT if state == "passed" else 1
        actions = 450 if state == "passed" else 17
        for _ in range(requests):
            protocol.complete_behavioral()
        end = protocol.end({
            "study_id": n3.pilot.STUDY_ID, "block_id": block.block_id,
            "cell_id": block.cell_ids[index], "condition_index": index,
            "server_context_id": begin["server_context_id"], "client_session_id": begin["client_session_id"],
            "status": "completed" if state == "passed" else state,
            "stop_reason": "action_cap" if state == "passed" else state, "actions_executed": actions,
            "request_count": requests,
            "final_chunk_executed_actions": n3.pilot.FINAL_EXECUTED_ACTIONS if state == "passed" else None,
        })
        terminal = n3.pilot.persist_server_terminal_receipt(
            attempt_root=attempt_root, end_response=end, protocol_context_active=False,
            model_capture_active=False,
        )
        receipt = {
            "synthetic_test_only": True, "phase": "confirmation",
            "source_pins": {"study_commit": STUDY_COMMIT},
            "transport_contract": deepcopy(n3.NO_REPLAY_TRANSPORT_CONTRACT),
            "confirmation_prerequisites": {}, "server_begin_receipt": begin,
            "server_end_receipt": {**end, "server_context_terminal": terminal},
            "server_context_terminal": terminal,
        }
        if state != "passed":
            receipt.update({
                "schema_version": n3.CELL_RECEIPT_SCHEMA, "study_id": n3.pilot.STUDY_ID,
                "block_id": block.block_id, "layout_pair_id": block.layout_pair_id,
                "model_config": "N3", "cell_id": block.cell_ids[index], "condition_index": index,
                "status": state, "recorded_stop_reason": state,
                "actions_executed": actions, "request_count": requests,
                "episode_context_id": begin["server_context_id"], "server_context_id": begin["server_context_id"],
                "client_session_id": begin["client_session_id"],
            })
        write_json(attempt_root / "cells" / f"{index:02d}-{n3.pilot.safe_cell_component(block.cell_ids[index])}"
                   / ("cell_receipt.json" if state == "passed" else "technical_failure.json"), receipt)

    def _native_d1_attempt(self, block, attempt, roots, admissions, finalizer, prefix):
        """Native-shaped sidecars matching the existing terminal-failure fixtures."""
        d1 = self.runtime["d1"]
        server, simulator = roots
        run_id = attempt["attempt_id"]
        coordination = Path(block.raw_root) / "coordination" / run_id
        future = server / "future"
        contract = write_json(future / "server_contract.json", {"synthetic_test_only": True})
        execution, execution_sha = d1.write_execution_prerequisites(
            simulator / "execution_prerequisites.json", block=block, prerequisites={},
        )
        ready = {
            "schema_version": d1.pilot.SERVER_READY_SCHEMA, "status": "ready",
            "study_id": d1.pilot.STUDY_ID, "phase": "confirmation",
            "block_id": block.block_id, "layout_pair_id": block.layout_pair_id, "model_config": "D1",
            "run_id": run_id, "server_job_id": server.name, "paired_simulator_job_id": simulator.name,
            "study_commit": STUDY_COMMIT, "service_host": d1.pilot.SERVICE_HOST,
            "service_port": d1.pilot.SERVICE_PORT, "pilot_contract_sha256": block.contract_sha256,
            "confirmation_contract_sha256": block.contract_sha256, "expected_cell_ids": list(block.cell_ids),
            "returned_action_shape": [d1.pilot.RETURNED_ACTION_HORIZON, d1.pilot.ACTION_DIM],
            "executed_prefix_horizon": d1.pilot.EXECUTED_PREFIX_HORIZON,
            "effective_model_noise_seed": d1.pilot.EFFECTIVE_MODEL_NOISE_SEED,
            "global_state_noninterleaving": True, "future_root": str(future),
            "server_contract": contract, "confirmation_prerequisites_sha256": execution_sha,
        }
        ready_identity = write_json(coordination / "server_ready.json", ready)
        claim = {
            "schema_version": d1.pilot.SIMULATOR_CLAIM_SCHEMA, "status": "claimed",
            "run_id": run_id, "simulator_job_id": simulator.name, "server_job_id": server.name,
            "server_ready_sha256": ready_identity["sha256"], "study_commit": STUDY_COMMIT,
            "block_id": block.block_id, "worker_role": self.sim_role, "pilot_contract_sha256": block.contract_sha256,
            "lease_token": "synthetic-lease", "start_cell_index": prefix,
        }
        claim_identity = write_json(coordination / "simulator_claim.json", claim)
        write_json(simulator / "resume.json", {
            "schema_version": d1.RESUME_SCHEMA, "block_id": block.block_id,
            "layout_pair_id": block.layout_pair_id, "start_cell_index": prefix,
        })
        write_json(coordination / "simulator_terminal.json", {
            "schema_version": "wmf-d1-behavioral-confirmation-simulator-terminal-v1",
            "all_simulator_children_reaped": True, "safe_for_server_shutdown": True,
        })
        server_admission, simulator_admission = admissions
        ack = self.wave.signed_document({
            "schema_version": "wmf-d1-confirmation-runtime-admission-ack-v1",
            "status": "distinct_server_simulator_admitted_before_science", "study_id": evidence.STUDY_ID,
            "namespace": evidence.NAMESPACE, "study_commit": STUDY_COMMIT,
            "run_id": run_id, "block_id": block.block_id, "layout_pair_id": block.layout_pair_id,
            "release_finalizer_job_id": finalizer, "consume_by_utc": CONSUME_BY,
            "server_job_id": server.name, "simulator_job_id": simulator.name,
            "server_worker_id": server_admission[0]["worker_id"],
            "simulator_worker_id": simulator_admission[0]["worker_id"],
            "server_admission": server_admission[1], "simulator_admission": simulator_admission[1],
            "server_ready": ready_identity, "simulator_claim": claim_identity,
            "server_publication_verification_payload_sha256": server_admission[0]["publication_verification_payload_sha256"],
            "simulator_publication_verification_payload_sha256": simulator_admission[0]["publication_verification_payload_sha256"],
            "pending_publication_commit": server_admission[0]["pending_publication_commit"],
            "server_verified_remote_head": server_admission[0]["verified_remote_head"],
            "simulator_verified_remote_head": simulator_admission[0]["verified_remote_head"],
            "published_queue_fragment_sha256": server_admission[0]["published_queue_fragment_sha256"],
            "science_reset_request_action_started": False, "queue_mutated": False, "jobs_dispatched": 0,
            "completed_at_utc": CHECKED, "claim_boundary": "native D1 pair admitted before science",
        })
        write_json(coordination / "release_admission_ack.json", ack)
        for index, cell_state in enumerate(attempt["states"], start=prefix):
            episode = f"synthetic-d1-context-{index}"
            session = f"synthetic-d1-client-{index}"
            cell_root = simulator / "cells" / f"{index:02d}-{d1.pilot.safe_component(block.cell_ids[index])}"
            requests = d1.pilot.REQUEST_COUNT if cell_state == "passed" else 1
            actions = 450 if cell_state == "passed" else 7
            terminal = write_json(future / "episodes" / episode / "terminal_context_receipt.json", {
                "synthetic_test_only": True, "terminal_state": "context_closed",
                "actions_executed": actions, "request_count": requests,
            })
            manifest = write_json(future / "episodes" / episode / "episode_manifest.json", {
                "synthetic_test_only": True,
            })
            pose_sha = "7" * 64
            completion = write_json(cell_root / "adapter_completion.json", {"identity": {
                "cell_id": block.cell_ids[index], "stage": "confirmation",
                "layout_pair_id": block.layout_pair_id, "model_config": "D1",
                "effective_seed": d1.pilot.EFFECTIVE_MODEL_NOISE_SEED,
                "source_identity": f"study:{STUDY_COMMIT};pose:{pose_sha}",
            }})
            receipt = {
                "synthetic_test_only": True, "schema_version": d1.CELL_RECEIPT_SCHEMA,
                "status": cell_state, "phase": "confirmation", "study_id": d1.pilot.STUDY_ID,
                "block_id": block.block_id, "layout_pair_id": block.layout_pair_id, "model_config": "D1",
                "cell_id": block.cell_ids[index], "condition_index": index,
                "confirmation_contract_sha256": block.contract_sha256,
                "transport_contract": deepcopy(d1.NO_REPLAY_TRANSPORT_CONTRACT),
                "server_context_terminal": terminal, "server_episode_manifest": manifest,
                "server_terminal_reset_scan": {"passed": True, "world_size": 2},
                "server_ready": ready_identity, "simulator_claim": claim_identity,
                "adapter_completion": completion,
                "confirmation_fixture": {"execution_prerequisites": execution,
                                         "execution_prerequisites_sha256": execution_sha,
                                         "pose_manifest": {"sha256": pose_sha}},
                "run_id": run_id, "server_job_id": server.name, "simulator_job_id": simulator.name,
                "study_commit": STUDY_COMMIT, "recorded_stop_reason": cell_state,
                "actions_executed": actions, "request_count": requests,
                "episode_id": episode, "episode_context_id": episode, "server_context_id": episode,
                "client_session_id": session,
                "server_begin_receipt": {"passed": True, "episode_context_id": episode,
                                         "client_session_id": session},
            }
            write_json(cell_root / ("cell_receipt.json" if cell_state == "passed" else "technical_failure.json"), receipt)

    @contextmanager
    def _fixture_interfaces(self, fixture: dict):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(evidence, "load_modules", return_value=self.modules))
            stack.enter_context(mock.patch.object(evidence, "_scientific_context", return_value=fixture["context"]))
            stack.enter_context(mock.patch.object(
                evidence, "_fetch_authorized_control_history",
                side_effect=lambda **_kwargs: nullcontext(fixture["history"]),
            ))
            stack.enter_context(mock.patch.object(
                self.runtime["n3"].development, "_validate_cell_receipt",
                side_effect=lambda path, **_kwargs: (
                    evidence.load_json(path, "synthetic recorder boundary"), evidence.file_identity(path),
                ),
            ))
            d1 = self.runtime["d1"]
            stack.enter_context(mock.patch.object(
                d1.development, "_PILOT_VALIDATE_PASSED_CELL",
                side_effect=lambda path, **_kwargs: (
                    evidence.load_json(path, "synthetic D1 recorder boundary"), evidence.file_identity(path),
                ),
            ))
            stack.enter_context(mock.patch.object(
                d1.development, "_validate_live_ready_descriptor",
                side_effect=lambda descriptor, _label: evidence.verify_descriptor(descriptor, "synthetic D1 ready boundary"),
            ))

            def terminal_loader(descriptor, **kwargs):
                identity, document = evidence.verify_descriptor(descriptor, "synthetic D1 terminal boundary")
                expected = kwargs["expected_finalize_control"]
                self.assertEqual(document["actions_executed"], expected["actions_executed"])
                self.assertEqual(document["request_count"], expected["request_count"])
                manifest = Path(identity["path"]).parent / "episode_manifest.json"
                return document, identity, {}, evidence.file_identity(manifest), [], {"passed": True, "world_size": 2}

            stack.enter_context(mock.patch.object(d1.pilot, "validate_server_terminal_receipt", side_effect=terminal_loader))
            configured = (
                self.runtime["n3"].configured_pilot(fixture["block"])
                if fixture["model"] == "N3" else d1.configured_pilot(fixture["block"], self.sim_role)
            )
            stack.enter_context(configured)
            yield

    def _assemble(self, fixture: dict):
        return evidence.assemble_result_attempt_ledger(
            source_root=ROOT, state_dir=fixture["state"], study_commit=STUDY_COMMIT,
            inputs=fixture["context"]["identities"], running_finalizer_job_id=fixture["finalizer"],
        )

    def _consume(self, fixture: dict, descriptor: dict):
        context = fixture["context"]
        identities = context["identities"]
        return self.wave.validate_result_ledger(
            descriptor=descriptor, study_commit=STUDY_COMMIT, branch=fixture["context"]["release"]["cohort_branch"],
            source_root=ROOT, state_dir=fixture["state"], evidence=evidence,
            models=context["models"], layout_order=context["layout_order"], blocks=context["blocks"],
            schedule_identity=context["schedule_identity"], release_identity=identities["confirmation_freeze"],
            fixture_identity=identities["fixture_freeze"], resource_identity=identities["resource_qualification"],
            runtime_identity=identities["terminal_runtime_identities"], runtime=context["runtime"],
            fixture_rows=context["fixture_rows"], prerequisites=context["prerequisites"],
        )

    def test_partial_passed_successful_outer_is_rejected_during_assembly(self):
        for model in ("N3", "D1"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temporary:
                fixture = self._fixture(Path(temporary).resolve(), model=model,
                                        passed_cells=1, outer_status="succeeded")
                with self._fixture_interfaces(fixture), self.assertRaisesRegex(
                    evidence.ConfirmationReleaseEvidenceError,
                    "partial passed prefix has no terminal outer technical failure",
                ):
                    self._assemble(fixture)

    def test_completed_blocks_stay_terminal_with_success_or_wrapper_failure(self):
        for model in ("N3", "D1"):
            for status in ("succeeded", "failed"):
                with self.subTest(model=model, outer_status=status), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary).resolve()
                    fixture = self._fixture(root, model=model, passed_cells=4, outer_status=status)
                    with self._fixture_interfaces(fixture):
                        ledger, jobs = self._assemble(fixture)
                        self.assertEqual(len(ledger["blocks"]), 24)
                        self.assertEqual(len(jobs), 1 if model == "N3" else 2)
                        block = ledger["blocks"][0]
                        self.assertEqual(block["state"], "passed")
                        self.assertEqual(block["completed_prefix_cells"], 4)
                        self.assertEqual(block["terminal_attempt_id"], fixture["attempt"])
                        self.assertEqual(sum(len(row["attempts"]) for row in ledger["blocks"]), 1)
                        self._consume(fixture, write_json(root / "ledger.json", ledger))

    def test_partial_passed_outer_failure_is_preserved_without_invented_failed_cell(self):
        for model in ("N3", "D1"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                fixture = self._fixture(root, model=model, passed_cells=1, outer_status="failed")
                with self._fixture_interfaces(fixture):
                    ledger, _jobs = self._assemble(fixture)
                    block = ledger["blocks"][0]
                    self.assertEqual((block["state"], block["completed_prefix_cells"]), ("technical_invalid", 1))
                    self.assertIsNone(block["terminal_attempt_id"])
                    self.assertEqual([cell["state"] for cell in block["attempts"][0]["cells"]], ["passed"])
                    self._consume(fixture, write_json(root / "ledger.json", ledger))

    def test_technical_prefix_retry_resumes_without_duplicate_valid_cells(self):
        for model in ("N3", "D1"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                fixture = self._fixture(root, model=model, passed_cells=0, outer_status="failed", attempt_specs=[
                    (["passed", "technical_failure"], "failed"), (["passed"] * 3, "succeeded"),
                ])
                with self._fixture_interfaces(fixture):
                    ledger, jobs = self._assemble(fixture)
                    block = ledger["blocks"][0]
                    self.assertEqual((block["state"], block["completed_prefix_cells"]), ("passed", 4))
                    self.assertEqual(len(jobs), 2 if model == "N3" else 4)
                    attempts = block["attempts"]
                    self.assertEqual([row["start_cell_index"] for row in attempts], [0, 1])
                    self.assertEqual(attempts[1]["predecessor_attempt_id"], attempts[0]["attempt_id"])
                    self.assertEqual([cell["condition_index"] for cell in attempts[1]["cells"]], [1, 2, 3])
                    self.assertEqual(sum(cell["state"] == "passed" for row in attempts for cell in row["cells"]), 4)
                    self.assertEqual(sum(cell["state"] == "technical_invalid" for row in attempts for cell in row["cells"]), 1)
                    self._consume(fixture, write_json(root / "ledger.json", ledger))

    def test_safety_censor_preserves_prefix_and_rejects_retry(self):
        for model in ("N3", "D1"):
            for retry in (False, True):
                with self.subTest(model=model, retry=retry), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary).resolve()
                    specs = [(["passed", "safety_abort"], "failed")]
                    if retry:
                        specs.append((["passed"] * 3, "succeeded"))
                    fixture = self._fixture(root, model=model, passed_cells=0, outer_status="failed", attempt_specs=specs)
                    with self._fixture_interfaces(fixture):
                        if retry:
                            with self.assertRaisesRegex(evidence.ConfirmationReleaseEvidenceError, "retry follows terminal block"):
                                self._assemble(fixture)
                        else:
                            ledger, _jobs = self._assemble(fixture)
                            block = ledger["blocks"][0]
                            self.assertEqual((block["state"], block["completed_prefix_cells"]), ("safety_censored", 1))
                            self.assertEqual(block["terminal_attempt_id"], fixture["attempt"])
                            self.assertEqual([cell["state"] for cell in block["attempts"][0]["cells"]], ["passed", "safety_censored"])
                            self._consume(fixture, write_json(root / "ledger.json", ledger))

    def test_completed_block_rejects_another_attempt(self):
        for model in ("N3", "D1"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temporary:
                fixture = self._fixture(Path(temporary).resolve(), model=model,
                                        passed_cells=0, outer_status="failed", attempt_specs=[
                                            (["passed"] * 4, "failed"), ([], "failed"),
                                        ])
                with self._fixture_interfaces(fixture), self.assertRaisesRegex(
                    evidence.ConfirmationReleaseEvidenceError, "retry follows terminal block",
                ):
                    self._assemble(fixture)

    def test_producer_replay_independently_rejects_changed_successful_outer_result(self):
        """Re-sign valid ledgers after changing outer receipts, without assembly."""
        for model in ("N3", "D1"):
            for states in (["passed"], ["passed", "technical_failure"], ["passed", "safety_abort"]):
                with self.subTest(model=model, states=states), tempfile.TemporaryDirectory() as temporary:
                    fixture = self._fixture(Path(temporary).resolve(), model=model,
                                            passed_cells=0, states=states, outer_status="failed")
                    with self._fixture_interfaces(fixture):
                        ledger, jobs = self._assemble(fixture)
                        changed = deepcopy(ledger)
                        attempt = changed["blocks"][0]["attempts"][0]
                        for index, job_id in enumerate(attempt["job_ids"]):
                            result = deepcopy(jobs[job_id]["result_value"])
                            result.update(status="succeeded", returncode=0, error_type=None)
                            attempt["queue_results"][index] = write_json(
                                Path(jobs[job_id]["result"]["path"]), result,
                            )
                        changed.pop("payload_sha256")
                        changed = evidence.signed_document(changed)
                        with self.assertRaisesRegex(
                            evidence.ConfirmationReleaseEvidenceError,
                            "partial passed prefix has no terminal outer technical failure|failed/censored native attempt has only successful queue results",
                        ):
                            evidence.validate_result_attempt_ledger(
                                changed, source_root=ROOT, state_dir=fixture["state"],
                                study_commit=STUDY_COMMIT,
                            )


if __name__ == "__main__":
    unittest.main()
