#!/usr/bin/env python3
"""Compile retained confirmation recordings into analysis-ready evidence.

This command is deliberately CPU-only.  It authenticates the terminal block
receipts, passed cell receipts, native recorder completion/journal/payload
chain, returned-versus-executed actions, original-camera identities, and the
frozen development-to-confirmation release.  It never loads a model, starts a
simulator, creates a label, or releases a queue job.

The compiler emits the exact request-inventory schema consumed by
``forecast_annotation_workflow.py`` plus signed receipts for every artifact it
creates.  ``assemble-analysis-manifest`` is a separate, late operation: it can
only join the compiled native evidence to already-existing locked human-label
artifacts, and it replays the final analyzer's own evidence gate before making
the manifest visible.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Sequence


sys.dont_write_bytecode = True

PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[1]
FORECAST = PACKAGE / "experiments" / "forecast_layout"
DEVELOPMENT_COMPILER_PATH = Path(__file__).with_name("compile_development_evidence.py")
ANNOTATION_PATH = Path(__file__).with_name("forecast_annotation_workflow.py")
ANALYZER_PATH = Path(__file__).with_name("forecast_evidence_analysis.py")
FREEZE_PATH = Path(__file__).with_name("freeze_development_release.py")
FIXTURE_FREEZE_PATH = FORECAST / "confirmation_fixture_freeze.py"
N3_CONFIRMATION_PATH = FORECAST / "n3_confirmation_block_job.py"
D1_CONFIRMATION_PATH = FORECAST / "d1_confirmation_block_jobs.py"
N3_PILOT_PATH = FORECAST / "n3_behavioral_pilot_job.py"
D1_PILOT_PATH = FORECAST / "d1_behavioral_pilot_jobs.py"
D1_SERVER_PATH = FORECAST / "d1_instrumented_server.py"
RESOURCE_QUALIFICATION_CONTRACT_PATH = (
    FORECAST / "development_resource_qualification_contract.json"
)
CONTRACT_PATH = FORECAST / "confirmation_evidence_compiler_contract.json"
SPEC_PATH = FORECAST / "ablation_spec.json"

# The existing confirmation validators intentionally use workshop-local
# imports so their cluster entry points remain standalone scripts.
if str(FORECAST) not in sys.path:
    sys.path.insert(0, str(FORECAST))

STUDY_ID = "WMF-ABLATION-001"
NAMESPACE = "wmf_ablation_001_20260912"
INPUT_SCHEMA = "wmf-confirmation-evidence-compiler-input-v1"
COHORT_CLOSE_SCHEMA = "wmf-confirmation-cohort-close-v1"
COHORT_SEAL_PLAN_SCHEMA = "wmf-confirmation-cohort-seal-plan-v1"
COMPILER_SCHEMA = "wmf-confirmation-evidence-compiler-receipt-v1"
REQUEST_INVENTORY_RECEIPT_SCHEMA = "wmf-confirmation-request-inventory-receipt-v1"
PROVENANCE_SCHEMA = "wmf-confirmation-request-provenance-v1"
PRIVATE_VIDEO_SCHEMA = "wmf-confirmation-private-video-inventory-v1"
ZERO_SCIENCE_SCHEMA = "wmf-confirmation-evidence-zero-science-v1"
ACTION_MANIFEST_SCHEMA = "wmf-forecast-action-manifest-v1"
RECORDING_RECEIPT_SCHEMA = "wmf-forecast-recording-receipt-v1"
ENDPOINT_SCHEMA = "wmf-forecast-endpoint-trace-receipt-v1"
HISTORY_SCHEMA = "wmf-forecast-request-history-receipt-v1"
# Safety-censored evidence is admissible only through the model-native deep
# validators pinned below to the exact published terminal-context runtime.
N3_CONTEXT_TERMINAL_SCHEMA = "wmf-n3-terminal-context-receipt-v1"
D1_CONTEXT_TERMINAL_SCHEMA = "wmf-d1-terminal-context-receipt-v1"
ANALYSIS_MANIFEST_SCHEMA = "wmf-forecast-analysis-evidence-manifest-v1"
REQUEST_INVENTORY_SCHEMA = "wmf-forecast-request-inventory-v1"
SEAL_JOB_RECEIPT_SCHEMA = "wmf-confirmation-cohort-seal-job-v1"
SEAL_JOB_RECEIPT_NAME = "confirmation_cohort_seal_job_receipt.json"
TERMINAL_RUNTIME_SOURCE_COMMIT = "03732d3c6fa37c26a3ab2e8608a80d076fdcc33a"
TERMINAL_RUNTIME_SHA256 = {
    "d1_pilot_dependency": "2cdeb2fa8008088351fec6d68a75617dad2833367c384ff9aa67d949daf281bf",
    "d1_confirmation_validator_dependency": "2fb7113fc3cf475d8d3e7c4ef3232012730eadb74f3995e7ae91c04f3fef5a68",
    "d1_server_dependency": "085aec66d7deb06cac2c50e61b3205899476185d70720330cb0c586416532801",
    "resource_qualification_contract_dependency": "dfb203fb2bca8fe89f5c6b9c73f4daa8eb836f37917ecf7f9e8a5aa5f2571803",
    "n3_pilot_dependency": "f116ceb1a3a9022642a735293b5b86188ce83dc4075cc00040a7795642de75a4",
    "n3_confirmation_validator_dependency": "cf2c4bb31b646a5e601d6dbaeab696d135c3d7593f11c221bebafecb1f3e4beb",
}

MODELS_BY_BRANCH = {
    "full_two_model": ("N3", "D1"),
    "reduced_n3": ("N3",),
    "reduced_d1": ("D1",),
}
MODELS = ("N3", "D1")
LAYOUTS = tuple(f"C{index:02d}" for index in range(1, 25))
STATUS_VALUES = ("valid_complete", "valid_censored", "technical_invalid", "not_run")
MODEL_PREFIX = {"N3": 32, "D1": 8}
MODEL_FULL_REQUESTS = {"N3": 15, "D1": 57}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}\Z")

INPUT_KEYS = {
    "schema_version",
    "study_id",
    "cohort_branch",
    "inventory_finalized_at",
    "source_root",
    "raw_root",
    "camera_id",
    "study_commit",
    "development_release_freeze",
    "cohort_close_receipt",
    "block_receipts",
    "payload_sha256",
}
BLOCK_INPUT_KEYS = {
    "model_id",
    "layout_pair_id",
    "block_id",
    "schedule_row_sha256",
    "evidence_form",
    "receipt",
    "failure_cells",
    "queue_jobs",
    "selected_queue_job_ids",
    "d1_pair",
}
FULL_AGGREGATE_FORM = "full_terminal_aggregate"
D1_ZERO_LAUNCH_FORM = "d1_minimal_no_behavioral_cells_launched"
FAILURE_CELL_KEYS = {
    "cell_id",
    "condition_index",
    "status",
    "actions_executed",
    "request_count",
    "artifact_state",
    "failure_receipt",
    "absence_reason",
    "adapter_completion",
    "adapter_journal",
    "source_video",
    "context_terminal",
}
QUEUE_JOB_KEYS = {
    "job_id",
    "model_id",
    "layout_pair_id",
    "block_id",
    "mode",
    "run_id",
    "selected_for_block_evidence",
    "descriptor",
    "claim_owner",
    "result",
    "live_descendant_pids",
    "runtime_terminal_evidence",
}
QUEUE_RUNTIME_KEYS = {
    "runtime_receipt",
    "cleanup_receipt",
    "nested_process_terminal",
    "protocol_terminal",
    "failure_cells",
}
D1_PAIR_KEYS = {
    "run_id",
    "server_job_id",
    "simulator_job_id",
    "server_receipt",
    "simulator_terminal",
}
COHORT_CLOSE_KEYS = {
    "schema_version",
    "study_id",
    "namespace",
    "status",
    "cohort_branch",
    "study_commit",
    "sealed_at_utc",
    "development_release_freeze",
    "prepared_schedule",
    "seal_plan",
    "sealer_source",
    "producer_queue_job",
    "raw_root",
    "compiler_camera_id",
    "queue_state_dir",
    "control_state",
    "confirmation_admission_barrier",
    "release_lock_held_during_seal",
    "all_confirmation_queue_job_ids",
    "blocks",
    "nonterminal_claim_job_ids",
    "live_descendant_processes",
    "d1_global_server_lock_path",
    "unexpired_d1_lease_paths",
    "unterminated_d1_protocol_claim_paths",
    "payload_sha256",
}
SEAL_PRODUCER_KEYS = {
    "job_id",
    "job_dir",
    "role",
    "worker_id",
    "runtime_identity",
    "descriptor",
    "claim_owner",
    "queue_result_lineage",
}
RUNTIME_IDENTITY_KEYS = {"hostname", "pod_uid", "pid"}
QUEUE_RESULT_LINEAGE_KEYS = {
    "path",
    "availability",
    "job_id",
    "source_commit",
    "descriptor_sha256",
    "worker_id",
}
SEAL_JOB_RECEIPT_KEYS = {
    "schema_version",
    "study_id",
    "status",
    "job_id",
    "job_dir",
    "study_commit",
    "queue_role",
    "worker_id",
    "runtime_identity",
    "queue_descriptor",
    "queue_claim",
    "queue_result_lineage",
    "seal_plan",
    "cohort_close_receipt",
    "terminal_block_index",
    "compiler_input_manifest",
    "compiler_receipt",
    "compiler_counts",
    "science_counts",
    "queue_mutations",
    "confirmation_released",
    "labels_created",
    "scientific_results_computed",
    "claim_boundary",
    "payload_sha256",
}
COHORT_SEAL_PLAN_KEYS = {
    "schema_version",
    "study_id",
    "cohort_branch",
    "study_commit",
    "development_release_freeze",
    "prepared_schedule",
    "selected_blocks",
    "payload_sha256",
}
COHORT_SEAL_SELECTION_KEYS = {
    "model_id",
    "layout_pair_id",
    "selected_queue_job_ids",
}
ADMISSION_BARRIER_KEYS = {
    "kind",
    "control_commit",
    "control_generation",
    "active_job_ids",
    "admission_deadline_unix",
    "verified_at_utc",
}
REFERENCE_KEYS = {"path", "sha256"}
ANALYSIS_SOURCE_KEYS = {
    "ablation_spec",
    "development_release_freeze",
    "request_selection",
    "annotation_freeze",
    "restricted_map",
    "final_consensus",
}
ANALYSIS_MANIFEST_KEYS = {
    "schema_version",
    "study_id",
    "stage",
    "cohort_branch",
    "sources",
    "endpoint_trace_receipts",
    "request_history_receipts",
    "payload_sha256",
}


class ConfirmationCompilerError(RuntimeError):
    """The supplied confirmation evidence cannot be compiled safely."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfirmationCompilerError(message)


def _load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    require(specification is not None and specification.loader is not None,
            f"cannot load required validator: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


development = _load_module(DEVELOPMENT_COMPILER_PATH, "wmf_confirmation_development_compiler")
annotation = _load_module(ANNOTATION_PATH, "wmf_confirmation_annotation_validator")
analyzer = _load_module(ANALYZER_PATH, "wmf_confirmation_final_analyzer")
freeze = _load_module(FREEZE_PATH, "wmf_confirmation_release_validator")
fixture_freeze = _load_module(FIXTURE_FREEZE_PATH, "wmf_confirmation_fixture_validator")
n3_confirmation = _load_module(N3_CONFIRMATION_PATH, "wmf_n3_confirmation_compiler_validator")
d1_confirmation = _load_module(D1_CONFIRMATION_PATH, "wmf_d1_confirmation_compiler_validator")


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ConfirmationCompilerError("value is not finite canonical JSON") from error


def pretty_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ConfirmationCompilerError("value is not finite JSON") from error


def sha256_bytes(payload: bytes) -> str:
    return development.sha256_bytes(payload)


def sha256_file(path: Path) -> str:
    return development.sha256_file(path)


def sign_document(value: Mapping[str, Any]) -> dict[str, Any]:
    require("payload_sha256" not in value, "document is already signed")
    result = dict(value)
    result["payload_sha256"] = sha256_bytes(canonical_bytes(result))
    return result


def verify_signed(value: Mapping[str, Any], label: str) -> None:
    observed = value.get("payload_sha256")
    require(isinstance(observed, str) and SHA256_RE.fullmatch(observed) is not None,
            f"{label} payload SHA-256 is invalid")
    unsigned = dict(value)
    unsigned.pop("payload_sha256")
    require(sha256_bytes(canonical_bytes(unsigned)) == observed,
            f"{label} payload hash mismatch")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    require(isinstance(value, Mapping), f"{label} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    require(not missing and not extra,
            f"{label} fields changed (missing={sorted(missing)}, extra={sorted(extra)})")


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return development.load_json(path, label)
    except ConfirmationCompilerError:
        raise
    except Exception as error:
        raise ConfirmationCompilerError(f"{label}: {error}") from error


def _reference(descriptor: Mapping[str, Any]) -> dict[str, str]:
    return {"path": str(descriptor["path"]), "sha256": str(descriptor["sha256"])}


def _bytes_descriptor(relative: str, payload: bytes) -> dict[str, Any]:
    return {"path": relative, "sha256": sha256_bytes(payload), "bytes": len(payload)}


def _validate_rfc3339_utc(value: Any, label: str) -> str:
    require(isinstance(value, str) and value.endswith("Z"), f"{label} must be RFC3339 UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ConfirmationCompilerError(f"{label} must be RFC3339 UTC") from error
    require(parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0,
            f"{label} must be UTC")
    return value


def _descriptor(
    value: Any,
    *,
    base: Path,
    label: str,
    raw_root: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    try:
        return development._descriptor(value, base=base, label=label, raw_root=raw_root)
    except ConfirmationCompilerError:
        raise
    except Exception as error:
        raise ConfirmationCompilerError(f"{label}: {error}") from error


def _safe_component(value: str) -> str:
    require(SAFE_COMPONENT_RE.fullmatch(value) is not None, f"unsafe cell identity: {value}")
    return value


def _verify_staged_source(source_root: Path, study_commit: str) -> None:
    """Require this compiler to execute from the exact immutable study tree."""

    require(Path(__file__).resolve().is_relative_to(source_root),
            "confirmation compiler is not executing from its supplied source root")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source_root, check=True,
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=source_root, check=True, capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ConfirmationCompilerError(f"cannot verify confirmation staged source: {error}") from error
    require(head == study_commit, "confirmation staged source HEAD differs from study commit")
    require(not dirty.strip(), "confirmation staged source is dirty")


def _validate_compiler_dependencies(receipt: Mapping[str, Any]) -> None:
    """Bind every local interpreter dependency before consumer replay."""

    source_root_value = receipt.get("source_root")
    study_commit = receipt.get("study_commit")
    require(
        isinstance(source_root_value, str)
        and Path(source_root_value).is_absolute()
        and isinstance(study_commit, str)
        and COMMIT_RE.fullmatch(study_commit) is not None,
        "compiled dependency source identity is invalid",
    )
    source_root = Path(source_root_value).resolve()
    require(
        source_root == REPOSITORY.resolve(),
        "compiled dependency source root is not this staged repository",
    )
    require(
        receipt.get("terminal_runtime_source_commit")
        == TERMINAL_RUNTIME_SOURCE_COMMIT,
        "compiled terminal-runtime source pin changed",
    )
    _verify_staged_source(source_root, study_commit)
    dependencies = {
        "compiler_source": Path(__file__).resolve(),
        "contract": CONTRACT_PATH.resolve(),
        "ablation_spec_dependency": SPEC_PATH.resolve(),
        "development_compiler_dependency": DEVELOPMENT_COMPILER_PATH.resolve(),
        "annotation_validator_dependency": ANNOTATION_PATH.resolve(),
        "final_analyzer_dependency": ANALYZER_PATH.resolve(),
        "release_validator_dependency": FREEZE_PATH.resolve(),
        "fixture_freeze_dependency": FIXTURE_FREEZE_PATH.resolve(),
        "n3_confirmation_validator_dependency": N3_CONFIRMATION_PATH.resolve(),
        "d1_confirmation_validator_dependency": D1_CONFIRMATION_PATH.resolve(),
        "n3_pilot_dependency": N3_PILOT_PATH.resolve(),
        "d1_pilot_dependency": D1_PILOT_PATH.resolve(),
        "d1_server_dependency": D1_SERVER_PATH.resolve(),
        "resource_qualification_contract_dependency": (
            RESOURCE_QUALIFICATION_CONTRACT_PATH.resolve()
        ),
    }
    for key, expected_path in dependencies.items():
        descriptor, path = _exact_file_reference(
            receipt.get(key),
            base=source_root,
            label=f"compiled {key}",
        )
        require(
            path == expected_path
            and descriptor == development.file_descriptor(expected_path),
            f"compiled {key} differs from the active staged dependency",
        )
        if key in TERMINAL_RUNTIME_SHA256:
            require(
                descriptor["sha256"] == TERMINAL_RUNTIME_SHA256[key],
                f"compiled {key} differs from terminal-runtime commit "
                f"{TERMINAL_RUNTIME_SOURCE_COMMIT}",
            )


def _planned_cells(branch: str) -> dict[str, tuple[str, str, str]]:
    try:
        planned = annotation._planned_cells("confirmation", branch)
    except Exception as error:
        raise ConfirmationCompilerError(f"cannot load confirmation cell roster: {error}") from error
    expected = len(MODELS_BY_BRANCH[branch]) * 24 * 4
    require(len(planned) == expected, "confirmation planned-cell cardinality changed")
    return dict(planned)


def _validate_release_freeze(
    descriptor_value: Any,
    *,
    manifest_base: Path,
    branch: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    descriptor, path = _descriptor(
        descriptor_value,
        base=manifest_base,
        label="development confirmation-release freeze",
    )
    try:
        value = freeze.validate_release_freeze(path, descriptor["sha256"])
    except Exception as error:
        raise ConfirmationCompilerError(f"development confirmation-release freeze failed: {error}") from error
    require(value.get("cohort_branch") == branch,
            "compiler branch differs from the development confirmation-release freeze")
    require(value.get("qualified_model_ids") == list(MODELS_BY_BRANCH[branch]),
            "compiler models differ from the development confirmation-release freeze")
    return descriptor, path, dict(value)


def _exact_file_reference(
    value: Any,
    *,
    base: Path,
    label: str,
    raw_root: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Authenticate one exact three-field file descriptor."""

    _exact_keys(value, {"path", "sha256", "bytes"}, label)
    return _descriptor(value, base=base, raw_root=raw_root, label=label)


def _descriptor_matches(
    observed: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    require(
        all(observed.get(key) == expected.get(key) for key in ("path", "sha256", "bytes")),
        f"{label} descriptor changed",
    )


def _option(argv: Sequence[Any], name: str, label: str) -> str:
    indices = [index for index, value in enumerate(argv) if value == name]
    require(len(indices) == 1 and indices[0] + 1 < len(argv), f"{label} lacks unique {name}")
    value = argv[indices[0] + 1]
    require(isinstance(value, str) and value, f"{label} {name} is invalid")
    return value


def _validate_queue_job(
    value: Any,
    *,
    close_path: Path,
    model: str,
    layout: str,
    block_id: str,
    study_commit: str,
    expected_state_dir: Path | None = None,
) -> dict[str, Any]:
    """Authenticate one terminal cluster-queue claim and its reaped child."""

    _exact_keys(value, QUEUE_JOB_KEYS, f"{model} {layout} cohort-close queue job")
    require(value.get("model_id") == model and value.get("layout_pair_id") == layout,
            f"{model} {layout} queue-job model/layout changed")
    require(value.get("block_id") == block_id,
            f"{model} {layout} queue-job block changed")
    job_id = value.get("job_id")
    require(isinstance(job_id, str) and SAFE_COMPONENT_RE.fullmatch(job_id) is not None,
            f"{model} {layout} queue-job ID is invalid")
    mode = value.get("mode")
    allowed_modes = {"queue"} if model == "N3" else {"server-job", "simulator-job"}
    require(mode in allowed_modes, f"{model} {layout} queue-job mode is invalid")
    run_id = value.get("run_id")
    if model == "N3":
        require(run_id is None, f"{model} {layout} queue job fabricates a D1 run ID")
    else:
        require(isinstance(run_id, str) and SAFE_COMPONENT_RE.fullmatch(run_id) is not None,
                f"{model} {layout} queue-job run ID is invalid")
    require(type(value.get("selected_for_block_evidence")) is bool,
            f"{model} {layout} selected-job flag is invalid")
    descriptor, descriptor_path = _exact_file_reference(
        value.get("descriptor"), base=close_path.parent,
        label=f"{model} {layout} queue descriptor {job_id}",
    )
    claim_descriptor, claim_path = _exact_file_reference(
        value.get("claim_owner"), base=close_path.parent,
        label=f"{model} {layout} queue claim {job_id}",
    )
    result_descriptor, result_path = _exact_file_reference(
        value.get("result"), base=close_path.parent,
        label=f"{model} {layout} queue result {job_id}",
    )
    queue = load_json(descriptor_path, f"{model} {layout} queue descriptor {job_id}")
    _exact_keys(
        queue,
        {
            "schema_version", "namespace", "job_id", "released", "source_commit",
            "role", "argv", "max_wall_seconds", "publish_log_tail_bytes",
        },
        f"{model} {layout} queue descriptor {job_id}",
    )
    require(
        queue.get("schema_version") == "wmf-cluster-job-v1"
        and queue.get("namespace") == NAMESPACE
        and queue.get("job_id") == job_id
        and queue.get("released") is True
        and queue.get("source_commit") == study_commit,
        f"{model} {layout} queue descriptor identity changed",
    )
    argv = queue.get("argv")
    require(isinstance(argv, list) and len(argv) >= 3 and argv[2] == mode,
            f"{model} {layout} queue descriptor mode changed")
    expected_runner = (
        n3_confirmation.RUNNER_FILENAME
        if model == "N3" else d1_confirmation.RUNNER_FILENAME
    )
    expected_runner_path = (
        "{source_root}/workshops/corl2026_world_models/experiments/forecast_layout/"
        + expected_runner
    )
    require(argv[0] == "/usr/bin/python3" and argv[1] == expected_runner_path,
            f"{model} {layout} queue descriptor runner changed")
    require(_option(argv, "--layout-pair-id", str(job_id)) == layout
            and _option(argv, "--study-commit", str(job_id)) == study_commit
            and _option(argv, "--job-id", str(job_id)) == job_id,
            f"{model} {layout} queue descriptor arguments changed")
    if model == "D1":
        require(_option(argv, "--run-id", str(job_id)) == run_id,
                f"{model} {layout} queue descriptor run ID changed")
        simulator_role = _option(argv, "--simulator-worker-role", str(job_id))
        expected_role = (
            d1_confirmation.pilot.SERVER_QUEUE_ROLE
            if mode == "server-job" else simulator_role
        )
        require(
            simulator_role in d1_confirmation.ALLOWED_SIMULATOR_ROLES
            and queue.get("role") == expected_role,
            f"{model} {layout} queue descriptor worker role changed",
        )
    else:
        require(queue.get("role") == n3_confirmation.pilot.QUEUE_ROLE,
                f"{model} {layout} queue descriptor worker role changed")
    claim = load_json(claim_path, f"{model} {layout} queue claim {job_id}")
    _exact_keys(
        claim,
        {
            "worker_id", "claimed_at", "claimed_unix", "worker_pid",
            "control_commit", "control_generation", "descriptor_sha256",
            "release_boundary",
        },
        f"{model} {layout} queue claim {job_id}",
    )
    require(
        claim.get("descriptor_sha256") == descriptor["sha256"]
        and claim.get("release_boundary") == "claim_committed_under_shared_release_lock"
        and isinstance(claim.get("claimed_at"), str)
        and type(claim.get("claimed_unix")) in {int, float}
        and math.isfinite(float(claim["claimed_unix"]))
        and float(claim["claimed_unix"]) > 0
        and isinstance(claim.get("control_commit"), str)
        and COMMIT_RE.fullmatch(claim["control_commit"]) is not None
        and type(claim.get("control_generation")) is int
        and claim["control_generation"] >= 0
        and type(claim.get("worker_pid")) is int
        and claim["worker_pid"] > 0,
        f"{model} {layout} queue claim is detached from its release boundary",
    )
    result = load_json(result_path, f"{model} {layout} queue result {job_id}")
    _exact_keys(
        result,
        {
            "schema_version", "namespace", "job_id", "worker_id", "source_commit",
            "descriptor_sha256", "started_at", "argv", "job_dir", "status",
            "returncode", "error_type", "ended_at", "wall_seconds", "child_pid",
            "child_reaped", "stdout", "stderr",
        },
        f"{model} {layout} queue result {job_id}",
    )
    terminal_status = result.get("status")
    require(
        result.get("schema_version") == "wmf-cluster-result-v1"
        and result.get("namespace") == NAMESPACE
        and result.get("job_id") == job_id
        and result.get("worker_id") == claim.get("worker_id")
        and result.get("source_commit") == study_commit
        and result.get("descriptor_sha256") == descriptor["sha256"]
        and terminal_status in {"succeeded", "failed", "timed_out", "interrupted"}
        and result.get("child_reaped") is True,
        f"{model} {layout} queue result is nonterminal or detached",
    )
    if terminal_status == "succeeded":
        require(result.get("returncode") == 0,
                f"{model} {layout} succeeded queue result has a nonzero return code")
    job_dir = descriptor_path.parent
    require(
        descriptor_path.name == "descriptor.json"
        and job_dir.name == job_id
        and job_dir.parent.name == "jobs",
        f"{model} {layout} queue descriptor is outside its canonical job directory",
    )
    state_dir = job_dir.parent.parent
    if expected_state_dir is not None:
        require(state_dir == Path(expected_state_dir).resolve(),
                f"{model} {layout} queue job is outside the sealed queue state")
    source_worktree = state_dir / "sources" / study_commit
    expected_argv = [
        item.replace("{source_root}", str(source_worktree))
        .replace("{job_dir}", str(job_dir))
        .replace("{state_dir}", str(state_dir))
        for item in argv
    ]
    require(
        result.get("argv") == expected_argv
        and result.get("job_dir") == str(job_dir),
        f"{model} {layout} executed argv/job directory differs from its queue descriptor",
    )
    start_time = _queue_time(result.get("started_at"), f"{job_id} queue start")
    end_time = _queue_time(result.get("ended_at"), f"{job_id} queue end")
    wall_seconds = result.get("wall_seconds")
    require(
        start_time < end_time
        and type(wall_seconds) in {int, float}
        and math.isfinite(float(wall_seconds))
        and float(wall_seconds) >= 0,
        f"{model} {layout} queue result timing is invalid",
    )
    for stream_name in ("stdout", "stderr"):
        stream = result.get(stream_name)
        _exact_keys(stream, {"bytes", "sha256"}, f"{job_id} queue {stream_name} identity")
        stream_path = job_dir / f"{stream_name}.log"
        development._reject_symlink_components(stream_path, f"{job_id} queue {stream_name}")
        require(stream_path.is_file(), f"{job_id} queue {stream_name} log is missing")
        require(
            type(stream.get("bytes")) is int
            and stream["bytes"] >= 0
            and isinstance(stream.get("sha256"), str)
            and SHA256_RE.fullmatch(stream["sha256"]) is not None
            and stream_path.stat().st_size == stream["bytes"]
            and sha256_file(stream_path) == stream["sha256"],
            f"{job_id} queue {stream_name} log changed after terminal result",
        )
    require(value.get("live_descendant_pids") == [],
            f"{model} {layout} queue job retained a live descendant")
    runtime = _validate_queue_runtime_evidence(
        value.get("runtime_terminal_evidence"),
        close_path=close_path,
        model=model,
        layout=layout,
        block_id=block_id,
        study_commit=study_commit,
        job_id=str(job_id),
        mode=str(mode),
        run_id=run_id,
        queue_descriptor=descriptor,
        queue_result=result,
    )
    return {
        **dict(value),
        "descriptor": descriptor,
        "descriptor_value": queue,
        "claim_owner": claim_descriptor,
        "claim_value": claim,
        "result": result_descriptor,
        "result_value": result,
        "runtime_terminal_evidence": runtime,
    }


def _validate_queue_runtime_evidence(
    value: Any,
    *,
    close_path: Path,
    model: str,
    layout: str,
    block_id: str,
    study_commit: str,
    job_id: str,
    mode: str,
    run_id: Any,
    queue_descriptor: Mapping[str, Any],
    queue_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate deep child cleanup for every queue attempt, selected or not."""

    _exact_keys(value, QUEUE_RUNTIME_KEYS, f"{job_id} runtime terminal evidence")
    failure_cells = value.get("failure_cells")
    require(isinstance(failure_cells, list),
            f"{job_id} runtime failure inventory is invalid")

    def optional_descriptor(raw: Any, label: str) -> tuple[dict[str, Any] | None, Path | None]:
        if raw is None:
            return None, None
        return _exact_file_reference(raw, base=close_path.parent, label=label)

    runtime_descriptor, runtime_path = optional_descriptor(
        value.get("runtime_receipt"), f"{job_id} runtime receipt"
    )
    cleanup_descriptor, cleanup_path = optional_descriptor(
        value.get("cleanup_receipt"), f"{job_id} cleanup receipt"
    )
    process_descriptor, process_path = optional_descriptor(
        value.get("nested_process_terminal"),
        f"{job_id} nested-process terminal receipt",
    )
    terminal_descriptor, terminal_path = optional_descriptor(
        value.get("protocol_terminal"), f"{job_id} protocol terminal"
    )
    require(runtime_descriptor is not None and runtime_path is not None,
            f"{job_id} lacks a deep terminal runtime receipt")
    runtime = load_json(runtime_path, f"{job_id} runtime receipt")
    runtime_status = runtime.get("status")
    runtime_exit = runtime.get("exit_code")
    require(
        runtime_status in {"passed", "technical_failure"}
        and runtime_exit == (0 if runtime_status == "passed" else 1)
        and queue_result.get("status")
        == ("succeeded" if runtime_status == "passed" else "failed")
        and queue_result.get("returncode") == runtime_exit,
        f"{job_id} queue result contradicts its runtime terminal receipt",
    )
    if model == "N3":
        require(
            mode == "queue"
            and runtime.get("schema_version") == n3_confirmation.QUEUE_RECEIPT_SCHEMA
            and runtime.get("source_commit") == study_commit
            and runtime.get("layout_pair_id") == layout
            and runtime.get("block_id") == block_id,
            f"{job_id} N3 runtime receipt identity changed",
        )
        _descriptor_matches(
            runtime.get("queue_descriptor", {}), queue_descriptor,
            f"{job_id} N3 runtime queue descriptor",
        )
        require(cleanup_descriptor is not None and cleanup_path is not None
                and process_descriptor is not None and process_path is not None
                and terminal_descriptor is None,
                f"{job_id} N3 runtime cleanup inventory changed")
        cleanup = load_json(cleanup_path, f"{job_id} N3 cleanup receipt")
        require(
            cleanup.get("schema_version")
            == "wmf-n3-behavioral-confirmation-cleanup-v1"
            and cleanup.get("all_children_reaped") is True
            and cleanup.get("remaining_compute_processes") == [],
            f"{job_id} N3 nested scientific children were not closed",
        )
        server_exit = runtime.get("server_exit")
        process_exit = load_json(process_path, f"{job_id} N3 server exit receipt")
        require(
            isinstance(server_exit, Mapping)
            and process_exit == server_exit
            and process_exit.get("child_reaped") is True,
            f"{job_id} N3 server child lacks a reaped exit receipt",
        )
        terminal = None
    elif mode == "server-job":
        require(
            runtime.get("schema_version") == d1_confirmation.SERVER_RECEIPT_SCHEMA
            and runtime.get("run_id") == run_id
            and runtime.get("server_job_id") == job_id
            and runtime.get("study_commit") == study_commit
            and runtime.get("block_id") == block_id
            and runtime.get("all_server_children_reaped") is True,
            f"{job_id} D1 server runtime receipt identity/cleanup changed",
        )
        _descriptor_matches(
            runtime.get("queue_descriptor", {}), queue_descriptor,
            f"{job_id} D1 server runtime queue descriptor",
        )
        process_exit = runtime.get("server_process_exit")
        require(
            process_exit is None
            or (
                isinstance(process_exit, Mapping)
                and process_exit.get("status") == "reaped"
                and process_exit.get("reaped") is True
            ),
            f"{job_id} D1 server child was not reaped",
        )
        require(cleanup_descriptor is None and process_descriptor is None
                and terminal_descriptor is None,
                f"{job_id} D1 server terminal inventory changed")
        cleanup = None
        terminal = None
    else:
        require(
            model == "D1"
            and mode == "simulator-job"
            and runtime.get("schema_version") == d1_confirmation.SIMULATOR_RECEIPT_SCHEMA
            and runtime.get("run_id") == run_id
            and runtime.get("simulator_job_id") == job_id
            and runtime.get("study_commit") == study_commit
            and runtime.get("block_id") == block_id
            and runtime.get("all_simulator_children_reaped") is True,
            f"{job_id} D1 simulator runtime receipt identity/cleanup changed",
        )
        embedded_queue = runtime.get("queue_descriptor")
        if embedded_queue is not None:
            _descriptor_matches(
                embedded_queue, queue_descriptor,
                f"{job_id} D1 simulator runtime queue descriptor",
            )
        else:
            require(
                runtime_status == "technical_failure"
                and runtime.get("counts") is None
                and runtime.get("cell_receipts") is None,
                f"{job_id} D1 simulator omits its queue descriptor outside a minimal failure",
            )
        require(cleanup_descriptor is None and process_descriptor is None
                and terminal_descriptor is not None
                and terminal_path is not None,
                f"{job_id} D1 simulator protocol-terminal inventory changed")
        terminal = load_json(terminal_path, f"{job_id} D1 simulator protocol terminal")
        require(
            terminal.get("schema_version") == d1_confirmation.pilot.SIMULATOR_TERMINAL_SCHEMA
            and terminal.get("run_id") == run_id
            and terminal.get("simulator_job_id") == job_id
            and terminal.get("block_id") == block_id
            and terminal.get("status") == runtime_status
            and terminal.get("all_simulator_children_reaped") is True
            and terminal.get("safe_for_server_shutdown") is True,
            f"{job_id} D1 simulator protocol terminal changed",
        )
        _descriptor_matches(
            terminal.get("simulator_receipt", {}), runtime_descriptor,
            f"{job_id} D1 simulator terminal/runtime receipt",
        )
        cleanup = None
    return {
        "runtime_receipt": runtime_descriptor,
        "runtime_value": runtime,
        "cleanup_receipt": cleanup_descriptor,
        "cleanup_value": cleanup,
        "nested_process_terminal": process_descriptor,
        "protocol_terminal": terminal_descriptor,
        "protocol_terminal_value": terminal,
        "failure_cells": list(failure_cells),
    }


def _validate_seal_producer(
    value: Any,
    *,
    close_path: Path,
    queue_state_dir: Path,
    source_root: Path,
    raw_root: Path,
    compiler_camera_id: str,
    study_commit: str,
    sealed_at_utc: str,
    seal_plan_descriptor: Mapping[str, Any],
    active_producer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the detached queue job that produced the cohort seal.

    The queue worker can only write ``result.json`` after this wrapper exits,
    so the signed close records deterministic result lineage.  A later
    compiler run must find that immutable outer result and bind it back to the
    signed descriptor, claim, wrapper PID, exact executed argv, and logs.
    """

    _exact_keys(value, SEAL_PRODUCER_KEYS, "cohort-close seal producer")
    job_id = value.get("job_id")
    role = value.get("role")
    worker_id = value.get("worker_id")
    require(
        isinstance(job_id, str)
        and SAFE_COMPONENT_RE.fullmatch(job_id) is not None
        and isinstance(role, str)
        and role.startswith("wmf-forecast-0912-worker-")
        and SAFE_COMPONENT_RE.fullmatch(role) is not None
        and isinstance(worker_id, str)
        and SAFE_COMPONENT_RE.fullmatch(worker_id) is not None,
        "cohort-close seal producer identity is invalid",
    )
    job_dir = Path(str(value.get("job_dir")))
    require(job_dir.is_absolute(), "cohort-close seal producer job directory is invalid")
    development._reject_symlink_components(job_dir, "cohort-close seal producer job directory")
    job_dir = job_dir.resolve()
    require(
        job_dir == Path(queue_state_dir).resolve() / "jobs" / job_id,
        "cohort-close seal producer is outside the sealed queue state",
    )
    seal_bundle = job_dir / "raw" / "cohort_seal"
    require(
        close_path.resolve() == seal_bundle / "cohort_close_receipt.json",
        "cohort-close receipt is outside the producer's canonical seal bundle",
    )
    descriptor, descriptor_path = _exact_file_reference(
        value.get("descriptor"), base=close_path.parent,
        label="cohort-close seal-producer descriptor",
    )
    claim_descriptor, claim_path = _exact_file_reference(
        value.get("claim_owner"), base=close_path.parent,
        label="cohort-close seal-producer claim",
    )
    require(
        descriptor_path == job_dir / "descriptor.json"
        and claim_path == job_dir / "claim" / "owner.json",
        "cohort-close seal producer descriptor/claim paths changed",
    )
    queue_value = load_json(descriptor_path, "cohort-close seal-producer descriptor")
    _exact_keys(
        queue_value,
        {
            "schema_version", "namespace", "job_id", "released", "source_commit",
            "role", "argv", "max_wall_seconds", "publish_log_tail_bytes",
        },
        "cohort-close seal-producer descriptor",
    )
    argv = queue_value.get("argv")
    expected_runner = (
        "{source_root}/workshops/corl2026_world_models/experiments/forecast_layout/"
        "confirmation_evidence_compiler_jobs.py"
    )
    expected_options = [
        "--source-root", "--study-commit", "--job-dir", "--job-id",
        "--seal-plan", "--seal-plan-sha256", "--raw-root", "--camera-id",
    ]
    require(
        queue_value.get("schema_version") == "wmf-cluster-job-v1"
        and queue_value.get("namespace") == NAMESPACE
        and queue_value.get("job_id") == job_id
        and queue_value.get("released") is True
        and queue_value.get("source_commit") == study_commit
        and queue_value.get("role") == role
        and queue_value.get("max_wall_seconds") == 21600
        and queue_value.get("publish_log_tail_bytes") == 8192
        and isinstance(argv, list)
        and len(argv) == 3 + 2 * len(expected_options)
        and argv[:3] == ["/usr/bin/python3", expected_runner, "seal-cohort"]
        and argv[3::2] == expected_options
        and _option(argv, "--source-root", str(job_id)) == "{source_root}"
        and _option(argv, "--study-commit", str(job_id)) == study_commit
        and _option(argv, "--job-dir", str(job_id)) == "{job_dir}"
        and _option(argv, "--job-id", str(job_id)) == job_id,
        "cohort-close seal producer descriptor execution changed",
    )
    expanded_seal_plan = (
        _option(argv, "--seal-plan", str(job_id))
        .replace("{source_root}", str(Path(source_root).resolve()))
        .replace("{job_dir}", str(job_dir))
        .replace("{state_dir}", str(Path(queue_state_dir).resolve()))
    )
    expanded_raw_root = (
        _option(argv, "--raw-root", str(job_id))
        .replace("{source_root}", str(Path(source_root).resolve()))
        .replace("{job_dir}", str(job_dir))
        .replace("{state_dir}", str(Path(queue_state_dir).resolve()))
    )
    require(
        _option(argv, "--seal-plan-sha256", str(job_id))
        == seal_plan_descriptor["sha256"]
        and development.file_descriptor(Path(expanded_seal_plan).resolve())["sha256"]
        == seal_plan_descriptor["sha256"]
        and development.file_descriptor(Path(expanded_seal_plan).resolve())["bytes"]
        == seal_plan_descriptor["bytes"]
        and Path(expanded_raw_root).resolve() == Path(raw_root).resolve(),
        "cohort-close seal producer input arguments changed",
    )
    require(
        _option(argv, "--camera-id", str(job_id)) == compiler_camera_id
        and compiler_camera_id in development.PRIMARY_CAMERA_CHOICES,
        "cohort-close compiler camera changed",
    )
    claim = load_json(claim_path, "cohort-close seal-producer claim")
    _exact_keys(
        claim,
        {
            "worker_id", "claimed_at", "claimed_unix", "worker_pid",
            "control_commit", "control_generation", "descriptor_sha256",
            "release_boundary",
        },
        "cohort-close seal-producer claim",
    )
    require(
        claim.get("worker_id") == worker_id
        and claim.get("descriptor_sha256") == descriptor["sha256"]
        and claim.get("release_boundary") == "claim_committed_under_shared_release_lock"
        and isinstance(claim.get("control_commit"), str)
        and COMMIT_RE.fullmatch(claim["control_commit"]) is not None
        and type(claim.get("control_generation")) is int
        and claim["control_generation"] >= 0
        and type(claim.get("worker_pid")) is int
        and claim["worker_pid"] > 0,
        "cohort-close seal producer claim is detached from its descriptor",
    )
    runtime = value.get("runtime_identity")
    _exact_keys(runtime, RUNTIME_IDENTITY_KEYS, "cohort-close seal-producer runtime")
    require(
        isinstance(runtime.get("hostname"), str)
        and runtime["hostname"].startswith(worker_id + "-")
        and isinstance(runtime.get("pod_uid"), str)
        and bool(runtime["pod_uid"])
        and type(runtime.get("pid")) is int
        and runtime["pid"] > 0,
        "cohort-close seal producer runtime identity is invalid",
    )
    lineage = value.get("queue_result_lineage")
    _exact_keys(lineage, QUEUE_RESULT_LINEAGE_KEYS, "cohort-close seal result lineage")
    result_path = job_dir / "result.json"
    require(
        lineage == {
            "path": str(result_path),
            "availability": "written_by_queue_worker_after_wrapper_exit",
            "job_id": job_id,
            "source_commit": study_commit,
            "descriptor_sha256": descriptor["sha256"],
            "worker_id": worker_id,
        },
        "cohort-close seal result lineage changed",
    )
    source_worktree = Path(queue_state_dir).resolve() / "sources" / study_commit
    require(
        Path(source_root).resolve() == source_worktree.resolve(),
        "cohort-close seal producer source differs from compiler source",
    )
    claim_time = _queue_time(
        claim.get("claimed_at"), "cohort-close seal producer claim time"
    )
    require(
        type(claim.get("claimed_unix")) in {int, float}
        and math.isfinite(float(claim["claimed_unix"]))
        and abs(claim_time.timestamp() - float(claim["claimed_unix"])) <= 2.0
        and claim_time
        <= _queue_time(sealed_at_utc, "cohort-close seal time"),
        "cohort-close seal producer claim time is invalid",
    )
    if not result_path.exists() and not result_path.is_symlink():
        require(
            active_producer is not None
            and dict(active_producer) == dict(value)
            and runtime["pid"] == os.getpid()
            and runtime["hostname"] == socket.gethostname()
            and runtime["pod_uid"] == os.environ.get("POD_UID", ""),
            "cohort-close seal producer outer result is unavailable without its exact live context",
        )
        return {
            **dict(value),
            "descriptor": descriptor,
            "claim_owner": claim_descriptor,
            "_outer_result_state": "pending_current_wrapper_exit",
        }
    result = load_json(result_path, "cohort-close seal-producer outer queue result")
    _exact_keys(
        result,
        {
            "schema_version", "namespace", "job_id", "worker_id", "source_commit",
            "descriptor_sha256", "started_at", "argv", "job_dir", "status",
            "returncode", "error_type", "ended_at", "wall_seconds", "child_pid",
            "child_reaped", "stdout", "stderr",
        },
        "cohort-close seal-producer outer queue result",
    )
    executed_argv = [
        item.replace("{source_root}", str(source_worktree))
        .replace("{job_dir}", str(job_dir))
        .replace("{state_dir}", str(Path(queue_state_dir).resolve()))
        for item in argv
    ]
    started = _queue_time(result.get("started_at"), "cohort-close seal queue start")
    ended = _queue_time(result.get("ended_at"), "cohort-close seal queue end")
    sealed = _queue_time(sealed_at_utc, "cohort-close seal time")
    wall_seconds = result.get("wall_seconds")
    require(
        result.get("schema_version") == "wmf-cluster-result-v1"
        and result.get("namespace") == NAMESPACE
        and result.get("job_id") == job_id
        and result.get("worker_id") == worker_id
        and result.get("source_commit") == study_commit
        and result.get("descriptor_sha256") == descriptor["sha256"]
        and result.get("argv") == executed_argv
        and result.get("job_dir") == str(job_dir)
        and result.get("status") == "succeeded"
        and result.get("returncode") == 0
        and result.get("error_type") is None
        and result.get("child_pid") == runtime["pid"]
        and result.get("child_reaped") is True
        and type(wall_seconds) in {int, float}
        and not isinstance(wall_seconds, bool)
        and math.isfinite(float(wall_seconds))
        and float(wall_seconds) >= 0
        and claim_time <= started <= sealed <= ended,
        "cohort-close seal producer outer queue result is contradictory",
    )
    for stream_name in ("stdout", "stderr"):
        stream = result.get(stream_name)
        _exact_keys(
            stream, {"bytes", "sha256"},
            f"cohort-close seal queue {stream_name} identity",
        )
        stream_path = job_dir / f"{stream_name}.log"
        development._reject_symlink_components(
            stream_path, f"cohort-close seal queue {stream_name}"
        )
        require(
            stream_path.is_file()
            and type(stream.get("bytes")) is int
            and stream["bytes"] >= 0
            and isinstance(stream.get("sha256"), str)
            and SHA256_RE.fullmatch(stream["sha256"]) is not None
            and stream_path.stat().st_size == stream["bytes"]
            and sha256_file(stream_path) == stream["sha256"],
            f"cohort-close seal queue {stream_name} log changed",
        )
    seal_receipt_path = job_dir / "publish" / SEAL_JOB_RECEIPT_NAME
    seal_receipt_descriptor = development.file_descriptor(seal_receipt_path)
    seal_receipt = load_json(seal_receipt_path, "cohort-close seal job receipt")
    _exact_keys(
        seal_receipt, SEAL_JOB_RECEIPT_KEYS, "cohort-close seal job receipt"
    )
    require(
        seal_receipt.get("schema_version") == SEAL_JOB_RECEIPT_SCHEMA
        and seal_receipt.get("study_id") == STUDY_ID
        and seal_receipt.get("status")
        == "cohort_terminal_no_live_claims_or_descendants"
        and seal_receipt.get("job_id") == job_id
        and seal_receipt.get("job_dir") == str(job_dir)
        and seal_receipt.get("study_commit") == study_commit
        and seal_receipt.get("queue_role") == role
        and seal_receipt.get("worker_id") == worker_id
        and seal_receipt.get("runtime_identity") == runtime
        and seal_receipt.get("queue_descriptor") == descriptor
        and seal_receipt.get("queue_claim") == claim_descriptor
        and seal_receipt.get("queue_result_lineage") == lineage
        and seal_receipt.get("seal_plan") == seal_plan_descriptor
        and seal_receipt.get("queue_mutations") == 0
        and seal_receipt.get("confirmation_released") is False
        and seal_receipt.get("labels_created") is False
        and seal_receipt.get("scientific_results_computed") is False,
        "cohort-close seal job receipt identity or zero-science boundary changed",
    )
    verify_signed(seal_receipt, "cohort-close seal job receipt")
    science_counts = seal_receipt.get("science_counts")
    require(
        science_counts
        == {
            "model_runtime_loads": 0,
            "model_servers_started": 0,
            "model_requests_issued_by_job": 0,
            "simulator_processes_started": 0,
            "physical_resets": 0,
            "robot_episodes": 0,
            "behavioral_actions_executed_by_job": 0,
            "behavioral_cells_launched_by_job": 0,
            "labels_created_by_job": 0,
            "confirmation_jobs_released_by_job": 0,
        },
        "cohort-close seal job receipt contains scientific activity",
    )
    output_paths = {
        "cohort_close_receipt": close_path.resolve(),
        "terminal_block_index": seal_bundle / "terminal_block_index.json",
        "compiler_input_manifest": job_dir / "raw" / "compiler_input_manifest.json",
        "compiler_receipt": job_dir / "raw" / "compiler_bundle" / "compiler_receipt.json",
    }
    output_descriptors: dict[str, dict[str, Any]] = {}
    for name, expected_path in output_paths.items():
        output_descriptor, output_path = _exact_file_reference(
            seal_receipt.get(name),
            base=seal_receipt_path.parent,
            label=f"cohort-close seal job {name}",
        )
        require(
            output_path == expected_path.resolve(),
            f"cohort-close seal job {name} escaped its canonical output path",
        )
        output_descriptors[name] = output_descriptor
    terminal_index = load_json(
        output_paths["terminal_block_index"], "cohort-close terminal block index"
    )
    require(
        terminal_index.get("schema_version")
        == "wmf-confirmation-terminal-block-index-v1"
        and terminal_index.get("cohort_close_receipt")
        == output_descriptors["cohort_close_receipt"],
        "cohort-close terminal index is detached from the seal receipt",
    )
    verify_signed(terminal_index, "cohort-close terminal block index")
    compiler_input = load_json(
        output_paths["compiler_input_manifest"], "cohort-close compiler input manifest"
    )
    require(
        compiler_input.get("schema_version") == INPUT_SCHEMA
        and compiler_input.get("cohort_close_receipt")
        == output_descriptors["cohort_close_receipt"],
        "cohort-close compiler input is detached from the seal receipt",
    )
    verify_signed(compiler_input, "cohort-close compiler input manifest")
    compiler_receipt = load_json(
        output_paths["compiler_receipt"], "cohort-close compiler receipt"
    )
    require(
        compiler_receipt.get("schema_version") == COMPILER_SCHEMA
        and compiler_receipt.get("cohort_close_receipt")
        == output_descriptors["cohort_close_receipt"]
        and compiler_receipt.get("input_manifest")
        == output_descriptors["compiler_input_manifest"]
        and compiler_receipt.get("counts") == seal_receipt.get("compiler_counts")
        and compiler_receipt.get("labels_created") is False
        and compiler_receipt.get("scientific_results_computed") is False
        and compiler_receipt.get("confirmation_released") is False,
        "cohort-close compiler receipt is detached from the seal receipt",
    )
    verify_signed(compiler_receipt, "cohort-close compiler receipt")
    try:
        stdout_lines = [
            line.strip()
            for line in (job_dir / "stdout.log").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        stdout_summary = json.loads(stdout_lines[-1])
    except (OSError, UnicodeDecodeError, IndexError, json.JSONDecodeError) as error:
        raise ConfirmationCompilerError(
            "cohort-close seal queue stdout lacks its final signed-receipt summary"
        ) from error
    expected_summary = {
        "status": seal_receipt["status"],
        "science_counts": science_counts,
        "confirmation_released": False,
        "output": str(seal_receipt_path),
        "sha256": seal_receipt_descriptor["sha256"],
    }
    require(
        stdout_summary == expected_summary,
        "cohort-close seal queue stdout does not bind its signed job receipt",
    )
    return {
        **dict(value),
        "descriptor": descriptor,
        "claim_owner": claim_descriptor,
        "seal_job_receipt": seal_receipt_descriptor,
        "_outer_result_state": "validated_succeeded_reaped",
    }


def _validate_staged_sealer_copy(
    value: Any, *, close_path: Path, source_root: Path
) -> tuple[dict[str, Any], Path]:
    descriptor, copied_path = _exact_file_reference(
        value, base=close_path.parent,
        label="cohort-close immutable sealer source",
    )
    require(
        copied_path == (close_path.parent / "sealer_source.py").resolve(),
        "cohort-close sealer copy is outside the canonical seal bundle",
    )
    expected = (
        Path(source_root).resolve()
        / "workshops/corl2026_world_models/experiments/forecast_layout/"
        "confirmation_evidence_compiler_jobs.py"
    )
    expected_descriptor = development.file_descriptor(expected)
    require(
        descriptor["sha256"] == expected_descriptor["sha256"]
        and descriptor["bytes"] == expected_descriptor["bytes"]
        and copied_path.read_bytes() == expected.read_bytes(),
        "cohort-close was not produced by the staged authoritative sealer source",
    )
    return descriptor, copied_path


def _load_cohort_close(
    value: Any,
    *,
    manifest_base: Path,
    branch: str,
    study_commit: str,
    release_descriptor: Mapping[str, Any],
    source_root: Path,
    active_seal_producer: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    descriptor, path = _exact_file_reference(
        value, base=manifest_base, label="confirmation cohort-close receipt"
    )
    close = load_json(path, "confirmation cohort-close receipt")
    _exact_keys(close, COHORT_CLOSE_KEYS, "confirmation cohort-close receipt")
    require(close.get("schema_version") == COHORT_CLOSE_SCHEMA,
            "confirmation cohort-close schema changed")
    verify_signed(close, "confirmation cohort-close receipt")
    require(
        close.get("study_id") == STUDY_ID
        and close.get("namespace") == NAMESPACE
        and close.get("status") == "cohort_terminal_no_live_claims_or_descendants"
        and close.get("cohort_branch") == branch
        and close.get("study_commit") == study_commit,
        "confirmation cohort-close identity/status changed",
    )
    _validate_rfc3339_utc(close.get("sealed_at_utc"), "cohort-close sealed_at_utc")
    close_release, _ = _exact_file_reference(
        close.get("development_release_freeze"), base=path.parent,
        label="cohort-close development release freeze",
    )
    _descriptor_matches(close_release, release_descriptor, "cohort-close release")
    schedule_descriptor, _ = _exact_file_reference(
        close.get("prepared_schedule"), base=path.parent,
        label="cohort-close prepared schedule",
    )
    seal_plan_descriptor, seal_plan_path = _exact_file_reference(
        close.get("seal_plan"), base=path.parent,
        label="cohort-close immutable seal plan",
    )
    sealer_descriptor, sealer_path = _validate_staged_sealer_copy(
        close.get("sealer_source"), close_path=path, source_root=source_root
    )
    require(
        seal_plan_path == (path.parent / "seal_plan.json").resolve()
        and sealer_path == (path.parent / "sealer_source.py").resolve(),
        "cohort-close producer artifacts are outside the canonical seal bundle",
    )
    queue_state_dir = close.get("queue_state_dir")
    require(isinstance(queue_state_dir, str) and Path(queue_state_dir).is_absolute(),
            "cohort-close queue-state root is invalid")
    development._reject_symlink_components(Path(queue_state_dir), "cohort-close queue-state root")
    require(Path(queue_state_dir).resolve().is_dir(),
            "cohort-close queue-state root is unavailable")
    close_raw_root = close.get("raw_root")
    require(isinstance(close_raw_root, str) and Path(close_raw_root).is_absolute(),
            "cohort-close raw root is invalid")
    development._reject_symlink_components(Path(close_raw_root), "cohort-close raw root")
    require(Path(close_raw_root).resolve().is_dir(),
            "cohort-close raw root is unavailable")
    compiler_camera_id = close.get("compiler_camera_id")
    require(compiler_camera_id in development.PRIMARY_CAMERA_CHOICES,
            "cohort-close compiler camera is invalid")
    producer = _validate_seal_producer(
        close.get("producer_queue_job"),
        close_path=path,
        queue_state_dir=Path(queue_state_dir),
        source_root=source_root,
        raw_root=Path(close_raw_root),
        compiler_camera_id=str(compiler_camera_id),
        study_commit=study_commit,
        sealed_at_utc=str(close.get("sealed_at_utc")),
        seal_plan_descriptor=seal_plan_descriptor,
        active_producer=active_seal_producer,
    )
    seal_plan = load_json(seal_plan_path, "cohort-close immutable seal plan")
    _exact_keys(seal_plan, COHORT_SEAL_PLAN_KEYS, "cohort-close immutable seal plan")
    verify_signed(seal_plan, "cohort-close immutable seal plan")
    require(
        seal_plan.get("schema_version") == COHORT_SEAL_PLAN_SCHEMA
        and seal_plan.get("study_id") == STUDY_ID
        and seal_plan.get("cohort_branch") == branch
        and seal_plan.get("study_commit") == study_commit
        and seal_plan.get("development_release_freeze") == close.get("development_release_freeze")
        and seal_plan.get("prepared_schedule") == close.get("prepared_schedule"),
        "cohort-close seal plan identity/bindings changed",
    )
    control_descriptor, control_path = _exact_file_reference(
        close.get("control_state"), base=path.parent,
        label="cohort-close immutable control snapshot",
    )
    require(control_path == (path.parent / "control_state.json").resolve(),
            "cohort-close control snapshot is outside the canonical seal bundle")
    control = load_json(control_path, "cohort-close immutable control snapshot")
    _exact_keys(
        control,
        {
            "namespace", "control_commit", "control_generation",
            "admission_deadline_unix", "shutdown", "active_job_ids",
        },
        "cohort-close immutable control snapshot",
    )
    require(
        control.get("namespace") == NAMESPACE
        and isinstance(control.get("control_commit"), str)
        and COMMIT_RE.fullmatch(control["control_commit"]) is not None
        and type(control.get("control_generation")) is int
        and control["control_generation"] >= 0
        and type(control.get("shutdown")) is bool
        and isinstance(control.get("active_job_ids"), list)
        and all(isinstance(item, str) for item in control["active_job_ids"]),
        "cohort-close control snapshot is invalid",
    )
    active_ids = control["active_job_ids"]
    require(
        active_ids == list(dict.fromkeys(active_ids))
        and all(SAFE_COMPONENT_RE.fullmatch(item) is not None for item in active_ids),
        "cohort-close control active-job inventory is invalid",
    )
    barrier = close.get("confirmation_admission_barrier")
    _exact_keys(barrier, ADMISSION_BARRIER_KEYS, "cohort-close admission barrier")
    require(
        barrier.get("control_commit") == control.get("control_commit")
        and barrier.get("control_generation") == control.get("control_generation")
        and barrier.get("active_job_ids") == active_ids
        and barrier.get("admission_deadline_unix")
        == control.get("admission_deadline_unix")
        and barrier.get("verified_at_utc") == close.get("sealed_at_utc"),
        "cohort-close admission barrier differs from its control snapshot",
    )
    seal_time = _queue_time(close.get("sealed_at_utc"), "cohort-close seal time").timestamp()
    barrier_kind = barrier.get("kind")
    deadline = control.get("admission_deadline_unix")
    if barrier_kind == "global_shutdown":
        require(control.get("shutdown") is True,
                "cohort-close shutdown barrier was not active")
    else:
        require(
            barrier_kind == "expired_absolute_admission_deadline"
            and control.get("shutdown") is False
            and type(deadline) in {int, float}
            and math.isfinite(float(deadline))
            and float(deadline) > 0
            and float(deadline) <= seal_time,
            "cohort-close lacks a no-future-admission barrier",
        )
    require(close.get("release_lock_held_during_seal") is True,
            "cohort-close was not sealed under the queue release lock")
    all_ids = close.get("all_confirmation_queue_job_ids")
    require(
        isinstance(all_ids, list)
        and all(isinstance(item, str) and SAFE_COMPONENT_RE.fullmatch(item) is not None for item in all_ids)
        and all_ids == sorted(set(all_ids)),
        "cohort-close confirmation job inventory is invalid",
    )
    require(close.get("nonterminal_claim_job_ids") == [],
            "cohort-close retains nonterminal confirmation claims")
    require(close.get("live_descendant_processes") == [],
            "cohort-close retains live confirmation descendants")
    if "D1" in MODELS_BY_BRANCH[branch]:
        require(
            close.get("d1_global_server_lock_path")
            == str(d1_confirmation.GLOBAL_D1_SERVER_LOCK.resolve()),
            "cohort-close D1 global-server lock identity changed",
        )
    else:
        require(close.get("d1_global_server_lock_path") is None,
                "N3-only cohort-close fabricates a D1 server-lock audit")
    require(close.get("unexpired_d1_lease_paths") == [],
            "cohort-close retains an unexpired D1 lease")
    require(close.get("unterminated_d1_protocol_claim_paths") == [],
            "cohort-close retains an unterminated D1 protocol claim")
    rows = close.get("blocks")
    require(isinstance(rows, list), "cohort-close block inventory is invalid")
    selections = seal_plan.get("selected_blocks")
    require(isinstance(selections, list), "cohort-close seal plan selections are invalid")
    expected_selection_keys = {
        (model, layout) for model in MODELS_BY_BRANCH[branch] for layout in LAYOUTS
    }
    observed_selection_keys: set[tuple[str, str]] = set()
    for index, selection in enumerate(selections):
        _exact_keys(
            selection, COHORT_SEAL_SELECTION_KEYS,
            f"cohort-close seal selection {index}",
        )
        selection_key = (selection.get("model_id"), selection.get("layout_pair_id"))
        selected_ids = selection.get("selected_queue_job_ids")
        require(
            selection_key in expected_selection_keys
            and selection_key not in observed_selection_keys
            and isinstance(selected_ids, list)
            and selected_ids == sorted(set(selected_ids))
            and len(selected_ids) == (1 if selection_key[0] == "N3" else 2)
            and all(isinstance(item, str) and SAFE_COMPONENT_RE.fullmatch(item)
                    for item in selected_ids),
            f"cohort-close seal selection {index} is invalid",
        )
        observed_selection_keys.add((str(selection_key[0]), str(selection_key[1])))
    require(observed_selection_keys == expected_selection_keys,
            "cohort-close seal selections do not exactly cover the cohort")
    require(
        sorted(
            (
                row.get("model_id"), row.get("layout_pair_id"),
                row.get("selected_queue_job_ids"),
            )
            for row in rows if isinstance(row, Mapping)
        )
        == sorted(
            (
                row.get("model_id"), row.get("layout_pair_id"),
                row.get("selected_queue_job_ids"),
            )
            for row in selections if isinstance(row, Mapping)
        ),
        "cohort-close selected jobs differ from its immutable seal plan",
    )
    return descriptor, path, {
        **close,
        "prepared_schedule": schedule_descriptor,
        "seal_plan": seal_plan_descriptor,
        "sealer_source": sealer_descriptor,
        "producer_queue_job": producer,
        "_producer_outer_result_state": producer["_outer_result_state"],
        "control_state": control_descriptor,
    }


def _aggregate_schema(model: str) -> str:
    return (
        n3_confirmation.QUEUE_RECEIPT_SCHEMA
        if model == "N3"
        else d1_confirmation.SIMULATOR_RECEIPT_SCHEMA
    )


def _aggregate_commit(receipt: Mapping[str, Any], model: str) -> Any:
    return receipt.get("source_commit" if model == "N3" else "study_commit")


def _validate_count_object(
    value: Any,
    *,
    model: str,
    layout: str,
) -> dict[str, int]:
    expected_keys = {
        "planned_behavioral_cells",
        "launched_behavioral_cells",
        "resumed_valid_behavioral_cells",
        "newly_launched_behavioral_cells",
        "completed_valid_behavioral_cells",
        "technically_invalid_behavioral_cells",
        "right_censored_behavioral_cells",
        "unrun_behavioral_cells",
        "actual_behavioral_actions",
        "actual_behavioral_model_requests",
        "new_generation_qualification_requests",
        "reused_prerequisite_generation_qualification_requests",
        "recorder_only_episodes_counted_as_behavioral",
    }
    _exact_keys(value, expected_keys, f"{model} {layout} aggregate counts")
    counts: dict[str, int] = {}
    for key in expected_keys:
        item = value.get(key)
        require(type(item) is int and item >= 0, f"{model} {layout} aggregate count {key} is invalid")
        counts[key] = item
    launched = counts["launched_behavioral_cells"]
    completed = counts["completed_valid_behavioral_cells"]
    invalid = counts["technically_invalid_behavioral_cells"]
    censored = counts["right_censored_behavioral_cells"]
    require(counts["planned_behavioral_cells"] == 4, f"{model} {layout} planned count changed")
    require(0 <= launched <= 4, f"{model} {layout} launched count is invalid")
    require(
        completed + invalid + censored == launched
        and counts["unrun_behavioral_cells"] == 4 - launched,
        f"{model} {layout} aggregate status counts do not close",
    )
    require(
        counts["resumed_valid_behavioral_cells"]
        + counts["newly_launched_behavioral_cells"] == launched,
        f"{model} {layout} resumed/newly-launched counts do not close",
    )
    require(launched - completed <= 1,
            f"{model} {layout} aggregate claims execution after a failed indivisible cell")
    require(counts["new_generation_qualification_requests"] == 0,
            f"{model} {layout} aggregate mixes generation probes into confirmation")
    require(counts["reused_prerequisite_generation_qualification_requests"] == 6,
            f"{model} {layout} prerequisite generation count changed")
    require(counts["recorder_only_episodes_counted_as_behavioral"] == 0,
            f"{model} {layout} counts recorder-only work as behavior")
    return counts


def _zero_launch_counts() -> dict[str, int]:
    """The only synthesized count vector allowed for a minimal D1 receipt.

    The values are not inferred from missing artifacts.  They are used only
    after the cohort sealer has explicitly selected the zero-launch evidence
    form and authenticated the paired queue/runtime lifecycle.
    """

    return {
        "planned_behavioral_cells": 4,
        "launched_behavioral_cells": 0,
        "resumed_valid_behavioral_cells": 0,
        "newly_launched_behavioral_cells": 0,
        "completed_valid_behavioral_cells": 0,
        "technically_invalid_behavioral_cells": 0,
        "right_censored_behavioral_cells": 0,
        "unrun_behavioral_cells": 4,
        "actual_behavioral_actions": 0,
        "actual_behavioral_model_requests": 0,
        "new_generation_qualification_requests": 0,
        "reused_prerequisite_generation_qualification_requests": 6,
        "recorder_only_episodes_counted_as_behavioral": 0,
    }


def _release_binding(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    prerequisites = receipt.get("prerequisites")
    require(isinstance(prerequisites, Mapping), "confirmation aggregate lacks prerequisites")
    release = prerequisites.get("confirmation_release")
    require(isinstance(release, Mapping), "confirmation aggregate lacks its release binding")
    return release


def _validate_aggregate(
    *,
    descriptor: Mapping[str, Any],
    path: Path,
    model: str,
    layout: str,
    schedule: Any,
    study_commit: str,
    source_root: Path,
    raw_root: Path,
    release_descriptor: Mapping[str, Any],
    release_freeze: Mapping[str, Any],
    evidence_form: str = FULL_AGGREGATE_FORM,
) -> dict[str, Any]:
    receipt = load_json(path, f"{model} {layout} confirmation aggregate")
    if evidence_form == D1_ZERO_LAUNCH_FORM:
        require(model == "D1", f"{model} {layout} uses the D1-only zero-launch form")
        require(
            receipt.get("counts") is None
            and receipt.get("cell_receipts") is None
            and receipt.get("prerequisites") is None,
            f"D1 {layout} full aggregate cannot be downgraded to the zero-launch form",
        )
        required = {
            "schema_version": d1_confirmation.SIMULATOR_RECEIPT_SCHEMA,
            "status": "technical_failure",
            "exit_code": 1,
            "run_id": receipt.get("run_id"),
            "server_job_id": receipt.get("server_job_id"),
            "simulator_job_id": receipt.get("simulator_job_id"),
            "study_commit": study_commit,
            "block_id": schedule.block_id,
            "all_simulator_children_reaped": True,
        }
        for key, wanted in required.items():
            require(wanted is not None and receipt.get(key) == wanted,
                    f"D1 {layout} minimal aggregate {key} changed")
        raw_attempt_value = receipt.get("raw_attempt_root")
        require(isinstance(raw_attempt_value, str) and Path(raw_attempt_value).is_absolute(),
                f"D1 {layout} minimal aggregate raw-attempt root is invalid")
        raw_attempt = development._under(
            Path(raw_attempt_value), raw_root, f"D1 {layout} minimal raw attempt"
        )
        require(raw_attempt.is_dir(), f"D1 {layout} minimal raw attempt is unavailable")
        # The close ledger, not this sparse runtime receipt, proves the exact
        # schedule/release binding and the absence of cell artifacts.
        return {
            "descriptor": dict(descriptor),
            "path": path,
            "receipt": receipt,
            "counts": _zero_launch_counts(),
            "raw_attempt": raw_attempt,
            "cell_descriptors": [],
            "cell_paths": [],
            "selected_fixture": None,
            "evidence_form": evidence_form,
        }
    require(evidence_form == FULL_AGGREGATE_FORM,
            f"{model} {layout} aggregate evidence form is invalid")
    expected = {
        "schema_version": _aggregate_schema(model),
        "study_id": STUDY_ID,
        "phase": "confirmation",
        "layout_pair_id": layout,
        "model_config": model,
        "block_id": schedule.block_id,
    }
    for key, wanted in expected.items():
        require(receipt.get(key) == wanted, f"{model} {layout} aggregate {key} changed")
    require(_aggregate_commit(receipt, model) == study_commit,
            f"{model} {layout} aggregate study commit changed")
    status = receipt.get("status")
    exit_code = receipt.get("exit_code")
    require(
        (status == "passed" and exit_code == 0 and receipt.get("failure") is None)
        or (status == "technical_failure" and exit_code == 1 and isinstance(receipt.get("failure"), Mapping)),
        f"{model} {layout} aggregate terminal status is invalid",
    )
    require(receipt.get("cell_ids") == list(schedule.cell_ids),
            f"{model} {layout} aggregate cell order changed")
    require(receipt.get("condition_order") == list(schedule.condition_order),
            f"{model} {layout} aggregate condition order changed")
    require(
        receipt.get("schedule")
        == {"path": str(schedule.schedule_path), "sha256": schedule.schedule_sha256},
        f"{model} {layout} aggregate prepared-schedule binding changed",
    )
    counts = _validate_count_object(receipt.get("counts"), model=model, layout=layout)
    if status == "passed":
        require(counts["completed_valid_behavioral_cells"] == 4,
                f"{model} {layout} passed aggregate is incomplete")
    release = _release_binding(receipt)
    require(release.get("confirmation_freeze") == dict(release_descriptor),
            f"{model} {layout} aggregate binds another confirmation freeze")
    require(release.get("cohort_branch") == release_freeze.get("cohort_branch"),
            f"{model} {layout} aggregate branch differs from release")
    require(release.get("qualified_model_ids") == release_freeze.get("qualified_model_ids"),
            f"{model} {layout} aggregate model set differs from release")
    expected_alignment = release_freeze["alignment_contracts_by_model"][model]
    require(release.get("alignment_contract") == expected_alignment,
            f"{model} {layout} aggregate alignment binding differs from release")
    prerequisites = receipt.get("prerequisites")
    require(isinstance(prerequisites, Mapping),
            f"{model} {layout} aggregate prerequisites are missing")
    fixture_descriptor = prerequisites.get("confirmation_fixture_freeze")
    require(isinstance(fixture_descriptor, Mapping),
            f"{model} {layout} aggregate fixture-freeze binding is missing")
    try:
        fixtures = fixture_freeze.validate_fixture_freeze(
            Path(str(fixture_descriptor.get("path"))),
            str(fixture_descriptor.get("sha256")),
            source_root=source_root,
            expected_layout_pair_id=layout,
            expected_study_commit=study_commit,
        )
    except Exception as error:
        raise ConfirmationCompilerError(
            f"{model} {layout} confirmation fixture freeze failed: {error}"
        ) from error
    require(
        prerequisites.get("confirmation_fixture_freeze") == fixtures["fixture_freeze"]
        and prerequisites.get("selected_fixture") == fixtures["selected_layout"],
        f"{model} {layout} aggregate selected fixture differs from its freeze",
    )
    raw_attempt_value = receipt.get("raw_attempt_root")
    require(isinstance(raw_attempt_value, str) and Path(raw_attempt_value).is_absolute(),
            f"{model} {layout} aggregate raw-attempt root is invalid")
    raw_attempt = development._under(Path(raw_attempt_value), raw_root, f"{model} {layout} raw attempt")
    require(raw_attempt.is_dir(), f"{model} {layout} raw attempt is unavailable")
    require(receipt.get("raw_attempt_recoverable_on_gm_pvc") is True,
            f"{model} {layout} aggregate does not preserve its raw attempt")
    raw_cells = receipt.get("cell_receipts")
    require(isinstance(raw_cells, list), f"{model} {layout} aggregate cell receipts are invalid")
    require(len(raw_cells) == counts["completed_valid_behavioral_cells"],
            f"{model} {layout} aggregate passed-cell count differs")
    cell_descriptors: list[dict[str, Any]] = []
    cell_paths: list[Path] = []
    for index, raw_descriptor in enumerate(raw_cells):
        observed, cell_path = _descriptor(
            raw_descriptor,
            base=path.parent,
            raw_root=raw_root,
            label=f"{model} {layout} passed cell {index}",
        )
        cell = load_json(cell_path, f"{model} {layout} passed cell {index}")
        require(cell.get("cell_id") == schedule.cell_ids[index],
                f"{model} {layout} passed cells are not the executed prefix")
        require(cell.get("condition_index") == index,
                f"{model} {layout} passed cell condition index changed")
        try:
            if model == "N3":
                n3_confirmation.validate_cells_bind_prerequisites(
                    [cell], prerequisites=prerequisites
                )
            else:
                d1_confirmation.validate_cells_bind_prerequisites(
                    [cell], prerequisites=prerequisites, block=schedule
                )
        except Exception as error:
            raise ConfirmationCompilerError(
                f"{model} {layout} passed cell prerequisite binding failed: {error}"
            ) from error
        cell_descriptors.append(observed)
        cell_paths.append(cell_path)
    return {
        "descriptor": dict(descriptor),
        "path": path,
        "receipt": receipt,
        "counts": counts,
        "raw_attempt": raw_attempt,
        "cell_descriptors": cell_descriptors,
        "cell_paths": cell_paths,
        "selected_fixture": dict(fixtures["selected_layout"]),
        "evidence_form": evidence_form,
    }


def _validate_failure_cells(
    value: Any,
    *,
    close_path: Path,
    model: str,
    layout: str,
    schedule: Any,
    aggregate: Mapping[str, Any],
    raw_root: Path,
    d1_runtime: Mapping[str, str] | None = None,
) -> dict[int, dict[str, Any]]:
    """Validate the close-time presence/absence declaration for failed cells."""

    require(isinstance(value, list), f"{model} {layout} failure inventory is invalid")
    counts = aggregate["counts"]
    first_failed = counts["completed_valid_behavioral_cells"]
    failed_count = (
        counts["technically_invalid_behavioral_cells"]
        + counts["right_censored_behavioral_cells"]
    )
    require(len(value) == failed_count == counts["launched_behavioral_cells"] - first_failed,
            f"{model} {layout} failure inventory does not close launched cells")
    output: dict[int, dict[str, Any]] = {}
    for ordinal, raw in enumerate(value):
        _exact_keys(raw, FAILURE_CELL_KEYS, f"{model} {layout} failure row {ordinal}")
        index = raw.get("condition_index")
        require(type(index) is int and index == first_failed + ordinal and index < 4,
                f"{model} {layout} failure rows are not the executed non-complete prefix")
        cell_id = schedule.cell_ids[index]
        require(raw.get("cell_id") == cell_id,
                f"{model} {layout} failure cell identity changed")
        status = raw.get("status")
        require(status in {"safety_abort", "technical_failure"},
                f"{cell_id} close-time failure status is invalid")
        actions = raw.get("actions_executed")
        requests = raw.get("request_count")
        require(
            type(actions) is int
            and 0 <= actions <= 450
            and (status == "technical_failure" or actions < 450),
            f"{cell_id} close-time action count is invalid",
        )
        require(type(requests) is int and 0 <= requests <= MODEL_FULL_REQUESTS[model],
                f"{cell_id} close-time request count is invalid")
        state = raw.get("artifact_state")
        cell_root = _cell_root(aggregate["raw_attempt"], index, cell_id)
        canonical = cell_root / "technical_failure.json"
        if state == "present_hash_bound":
            require(raw.get("absence_reason") is None,
                    f"{cell_id} present failure has an absence reason")
            descriptor, path = _exact_file_reference(
                raw.get("failure_receipt"), base=close_path.parent, raw_root=raw_root,
                label=f"{cell_id} hash-bound technical failure",
            )
            require(path == canonical.resolve(),
                    f"{cell_id} failure descriptor is outside its canonical cell")
            failure = load_json(path, f"{cell_id} technical failure")
            require(
                failure.get("schema_version")
                == (n3_confirmation.CELL_RECEIPT_SCHEMA if model == "N3"
                    else d1_confirmation.CELL_RECEIPT_SCHEMA)
                and failure.get("cell_id") == cell_id
                and failure.get("condition_index") == index
                and failure.get("status") == status
                and failure.get("actions_executed") == actions
                and failure.get("request_count") == requests,
                f"{cell_id} failure receipt differs from its close-time declaration",
            )
            try:
                if model == "N3":
                    validated_failure = n3_confirmation.validate_failed_confirmation_cell(
                        path, condition_index=index, block=schedule
                    )
                    n3_confirmation.validate_cells_bind_prerequisites(
                        [validated_failure],
                        prerequisites=aggregate["receipt"]["prerequisites"],
                    )
                else:
                    require(
                        d1_runtime is not None,
                        f"{cell_id} D1 failure lacks its selected queue lifecycle",
                    )
                    with d1_confirmation._configured_for_validation(
                        schedule, d1_runtime["simulator_worker_role"]
                    ):
                        validated_failure = (
                            d1_confirmation.validate_failed_confirmation_cell(
                                path, condition_index=index, block=schedule
                            )
                        )
                        d1_confirmation.validate_cells_bind_prerequisites(
                            [validated_failure],
                            prerequisites=aggregate["receipt"]["prerequisites"],
                            block=schedule,
                        )
                    require(
                        failure.get("study_commit") == d1_runtime["study_commit"]
                        and failure.get("run_id") == d1_runtime["run_id"]
                        and failure.get("server_job_id")
                        == d1_runtime["server_job_id"]
                        and failure.get("simulator_job_id")
                        == d1_runtime["simulator_job_id"],
                        f"{cell_id} D1 failure is detached from its selected queue pair",
                    )
            except ConfirmationCompilerError:
                raise
            except Exception as error:
                raise ConfirmationCompilerError(
                    f"{cell_id} failed native confirmation-failure validation: {error}"
                ) from error
            require(
                validated_failure == failure,
                f"{cell_id} native confirmation-failure validator changed its source bytes",
            )
        else:
            require(
                state == "absent_verified_at_cohort_close"
                and status == "technical_failure"
                and actions == 0
                and requests == 0
                and raw.get("failure_receipt") is None
                and raw.get("absence_reason") == "child_terminated_before_failure_receipt",
                f"{cell_id} failure absence declaration is invalid",
            )
            require(not canonical.exists() and not canonical.is_symlink(),
                    f"{cell_id} declared-absent failure artifact now exists")
            cell_receipt = canonical.with_name("cell_receipt.json")
            require(not cell_receipt.exists() and not cell_receipt.is_symlink(),
                    f"{cell_id} absent-failure declaration contradicts a cell receipt")
            descriptor = None
            path = None
            failure = None
        completion_path = cell_root / "recording" / "completion.json"
        journal_path = cell_root / "recording" / "events.partial.jsonl"

        def close_sidecar(
            supplied: Any, canonical_path: Path, sidecar_label: str
        ) -> tuple[dict[str, Any] | None, Path | None]:
            if supplied is None:
                require(not canonical_path.exists() and not canonical_path.is_symlink(),
                        f"{cell_id} close ledger omits existing {sidecar_label}")
                return None, None
            observed, observed_path = _exact_file_reference(
                supplied,
                base=close_path.parent,
                raw_root=raw_root,
                label=f"{cell_id} close-time {sidecar_label}",
            )
            require(observed_path == canonical_path.resolve(),
                    f"{cell_id} {sidecar_label} is outside its canonical cell")
            return observed, observed_path

        completion_descriptor, completion_observed_path = close_sidecar(
            raw.get("adapter_completion"), completion_path, "adapter completion"
        )
        journal_descriptor, journal_observed_path = close_sidecar(
            raw.get("adapter_journal"), journal_path, "adapter journal"
        )
        require(completion_descriptor is None or journal_descriptor is not None,
                f"{cell_id} completion lacks its hash-bound journal")
        require(
            not (actions > 0 or requests > 0)
            or (completion_descriptor is not None and journal_descriptor is not None),
            f"{cell_id} nonzero technical prefix lacks native completion/journal evidence",
        )

        native_root = cell_root / "native_simulator"
        if native_root.exists() or native_root.is_symlink():
            require(native_root.is_dir() and not native_root.is_symlink(),
                    f"{cell_id} native simulator root is invalid")
            native_items = list(native_root.rglob("*"))
            require(not any(item.is_symlink() for item in native_items),
                    f"{cell_id} native simulator evidence contains a symlink")
            videos = sorted(
                item.resolve() for item in native_items
                if item.is_file() and item.suffix.lower() == ".mp4" and item.stat().st_size > 0
            )
        else:
            videos = []
        require(len(videos) <= 1, f"{cell_id} has ambiguous close-time viewport videos")
        supplied_video = raw.get("source_video")
        if supplied_video is None:
            require(not videos, f"{cell_id} close ledger omits an existing source video")
            video_descriptor = None
            video_path = None
        else:
            video_descriptor, video_path = _exact_file_reference(
                supplied_video,
                base=close_path.parent,
                raw_root=raw_root,
                label=f"{cell_id} close-time source video",
            )
            require(videos == [video_path],
                    f"{cell_id} source-video inventory changed after cohort close")
        context_value = raw.get("context_terminal")
        if context_value is None:
            context_descriptor = None
            context_path = None
        else:
            context_descriptor, context_path = _exact_file_reference(
                context_value,
                base=close_path.parent,
                raw_root=raw_root,
                label=f"{cell_id} close-time model-context terminal",
            )
        if failure is not None:
            require(failure.get("server_context_terminal") == context_descriptor,
                    f"{cell_id} failure/context-terminal descriptor changed")
        if status == "safety_abort":
            require(
                completion_descriptor is not None
                and journal_descriptor is not None
                and video_descriptor is not None,
                f"{cell_id} safety-censored evidence is incomplete at cohort close",
            )
            require(context_descriptor is not None,
                    f"{cell_id} safety-censored evidence lacks terminal model-context proof")
        output[index] = {
            **dict(raw),
            "failure_receipt": descriptor,
            "failure_path": path,
            "failure_value": failure,
            "adapter_completion": completion_descriptor,
            "completion_path": completion_observed_path,
            "adapter_journal": journal_descriptor,
            "journal_path": journal_observed_path,
            "source_video": video_descriptor,
            "video_path": video_path,
            "context_terminal": context_descriptor,
            "context_terminal_path": context_path,
            "declaration_sha256": sha256_bytes(canonical_bytes(raw)),
        }
    invalid = sum(row["status"] == "technical_failure" for row in output.values())
    censored = sum(row["status"] == "safety_abort" for row in output.values())
    actions = 450 * first_failed + sum(row["actions_executed"] for row in output.values())
    requests = (
        MODEL_FULL_REQUESTS[model] * first_failed
        + sum(row["request_count"] for row in output.values())
    )
    require(
        invalid == counts["technically_invalid_behavioral_cells"]
        and censored == counts["right_censored_behavioral_cells"]
        and actions == counts["actual_behavioral_actions"]
        and requests == counts["actual_behavioral_model_requests"],
        f"{model} {layout} failure categories/actions/requests do not close the aggregate",
    )
    return output


def _validate_d1_pair(
    value: Any,
    *,
    close_path: Path,
    layout: str,
    block_id: str,
    study_commit: str,
    aggregate: Mapping[str, Any],
    aggregate_descriptor: Mapping[str, Any],
    selected_jobs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _exact_keys(value, D1_PAIR_KEYS, f"D1 {layout} paired lifecycle")
    run_id = value.get("run_id")
    server_id = value.get("server_job_id")
    simulator_id = value.get("simulator_job_id")
    require(
        isinstance(run_id, str)
        and isinstance(server_id, str)
        and isinstance(simulator_id, str)
        and set(selected_jobs) == {server_id, simulator_id}
        and selected_jobs[server_id]["mode"] == "server-job"
        and selected_jobs[simulator_id]["mode"] == "simulator-job"
        and selected_jobs[server_id]["run_id"] == run_id
        and selected_jobs[simulator_id]["run_id"] == run_id,
        f"D1 {layout} selected queue jobs do not form one pair",
    )
    server_descriptor, server_path = _exact_file_reference(
        value.get("server_receipt"), base=close_path.parent,
        label=f"D1 {layout} server terminal receipt",
    )
    terminal_descriptor, terminal_path = _exact_file_reference(
        value.get("simulator_terminal"), base=close_path.parent,
        label=f"D1 {layout} simulator terminal receipt",
    )
    server = load_json(server_path, f"D1 {layout} server terminal receipt")
    terminal = load_json(terminal_path, f"D1 {layout} simulator terminal receipt")
    aggregate_receipt = aggregate["receipt"]
    aggregate_status = aggregate_receipt.get("status")
    require(
        server.get("schema_version") == d1_confirmation.SERVER_RECEIPT_SCHEMA
        and server.get("run_id") == run_id
        and server.get("server_job_id") == server_id
        and server.get("paired_simulator_job_id") == simulator_id
        and server.get("study_commit") == study_commit
        and server.get("block_id") == block_id
        and server.get("status") == aggregate_status
        and server.get("exit_code") == (0 if aggregate_status == "passed" else 1)
        and server.get("all_server_children_reaped") is True,
        f"D1 {layout} server terminal receipt is incomplete",
    )
    server_process = server.get("server_process_exit")
    if server_process is None:
        require(
            server.get("status") == "technical_failure"
            and server.get("server_ready") is None
            and aggregate["counts"]["launched_behavioral_cells"] == 0,
            f"D1 {layout} lacks an explicit safely-unstarted server lifecycle",
        )
    else:
        require(
            isinstance(server_process, Mapping)
            and server_process.get("status") == "reaped"
            and server_process.get("reaped") is True,
            f"D1 {layout} server child was not reaped",
        )
    require(
        terminal.get("schema_version") == d1_confirmation.pilot.SIMULATOR_TERMINAL_SCHEMA
        and terminal.get("run_id") == run_id
        and terminal.get("server_job_id") == server_id
        and terminal.get("simulator_job_id") == simulator_id
        and terminal.get("block_id") == block_id
        and terminal.get("status") == aggregate_status
        and terminal.get("all_simulator_children_reaped") is True
        and terminal.get("safe_for_server_shutdown") is True,
        f"D1 {layout} simulator terminal receipt is incomplete",
    )
    require(
        aggregate_receipt.get("run_id") == run_id
        and aggregate_receipt.get("server_job_id") == server_id
        and aggregate_receipt.get("simulator_job_id") == simulator_id
        and aggregate_receipt.get("all_simulator_children_reaped") is True,
        f"D1 {layout} simulator aggregate differs from its paired lifecycle",
    )
    _descriptor_matches(
        selected_jobs[server_id]["descriptor"], server.get("queue_descriptor", {}),
        f"D1 {layout} server queue",
    )
    simulator_queue = aggregate_receipt.get("queue_descriptor")
    if isinstance(simulator_queue, Mapping):
        _descriptor_matches(
            selected_jobs[simulator_id]["descriptor"], simulator_queue,
            f"D1 {layout} simulator queue",
        )
    else:
        require(
            aggregate.get("evidence_form") == D1_ZERO_LAUNCH_FORM
            and aggregate_receipt.get("status") == "technical_failure",
            f"D1 {layout} simulator queue binding is absent outside zero-launch evidence",
        )
    _descriptor_matches(
        terminal.get("simulator_receipt", {}), aggregate_descriptor,
        f"D1 {layout} simulator terminal/aggregate",
    )
    _descriptor_matches(
        server.get("simulator_terminal", {}), terminal_descriptor,
        f"D1 {layout} server/simulator terminal",
    )
    claim = aggregate_receipt.get("simulator_claim")
    if isinstance(claim, Mapping):
        _descriptor_matches(claim, server.get("simulator_claim", {}), f"D1 {layout} paired claim")
        require(terminal.get("simulator_claim_sha256") == claim.get("sha256"),
                f"D1 {layout} terminal claim binding changed")
    else:
        require(
            aggregate.get("evidence_form") == D1_ZERO_LAUNCH_FORM
            and aggregate["counts"]["launched_behavioral_cells"] == 0
            and server.get("simulator_claim") is None
            and terminal.get("simulator_claim_sha256") is None,
            f"D1 {layout} lacks a paired claim outside the explicit zero-launch form",
        )
    for job_id, runtime_status, runtime_exit, role in (
        (server_id, server.get("status"), server.get("exit_code"), "server"),
        (
            simulator_id,
            aggregate_receipt.get("status"),
            aggregate_receipt.get("exit_code"),
            "simulator",
        ),
    ):
        queue_result = selected_jobs[job_id]["result_value"]
        expected_queue_status = "succeeded" if runtime_status == "passed" else "failed"
        require(
            queue_result.get("status") == expected_queue_status
            and queue_result.get("returncode") == runtime_exit,
            f"D1 {layout} {role} queue result contradicts its runtime receipt",
        )
    return {
        **dict(value),
        "server_receipt": server_descriptor,
        "server_value": server,
        "simulator_terminal": terminal_descriptor,
        "simulator_terminal_value": terminal,
    }


def _queue_time(value: Any, label: str) -> datetime:
    require(isinstance(value, str), f"{label} is not an RFC3339 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ConfirmationCompilerError(f"{label} is not an RFC3339 timestamp") from error
    require(
        parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0,
        f"{label} is not UTC",
    )
    return parsed


def _attempt_prefix(
    *,
    aggregate: Mapping[str, Any],
    close_path: Path,
    raw_root: Path,
    schedule: Any,
    model: str,
    layout: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Authenticate one attempt's resume/new/final contiguous cell prefix."""

    if aggregate.get("evidence_form") == D1_ZERO_LAUNCH_FORM:
        return [], [], []
    receipt = aggregate["receipt"]

    def descriptor_list(raw: Any, name: str) -> list[dict[str, Any]]:
        require(isinstance(raw, list), f"{model} {layout} attempt {name} is invalid")
        output: list[dict[str, Any]] = []
        for index, value in enumerate(raw):
            observed, path = _exact_file_reference(
                value,
                base=close_path.parent,
                raw_root=raw_root,
                label=f"{model} {layout} attempt {name} {index}",
            )
            cell = load_json(path, f"{model} {layout} attempt {name} {index}")
            require(
                index < len(schedule.cell_ids)
                and cell.get("cell_id") == schedule.cell_ids[index]
                and cell.get("condition_index") == index,
                f"{model} {layout} attempt {name} is not the frozen contiguous prefix",
            )
            output.append(observed)
        return output

    before = descriptor_list(receipt.get("resumed_cell_receipts"), "resumed prefix")
    new_raw = receipt.get("new_cell_receipts")
    require(isinstance(new_raw, list), f"{model} {layout} attempt new-cell inventory is invalid")
    new: list[dict[str, Any]] = []
    for offset, value in enumerate(new_raw):
        index = len(before) + offset
        observed, path = _exact_file_reference(
            value,
            base=close_path.parent,
            raw_root=raw_root,
            label=f"{model} {layout} attempt new cell {index}",
        )
        cell = load_json(path, f"{model} {layout} attempt new cell {index}")
        require(
            index < len(schedule.cell_ids)
            and cell.get("cell_id") == schedule.cell_ids[index]
            and cell.get("condition_index") == index,
            f"{model} {layout} attempt new cells are not a contiguous extension",
        )
        new.append(observed)
    after = descriptor_list(receipt.get("cell_receipts"), "final prefix")
    counts = aggregate["counts"]
    require(
        before + new == after
        and len(before) == counts["resumed_valid_behavioral_cells"]
        and len(after) == counts["completed_valid_behavioral_cells"]
        and after == aggregate["cell_descriptors"],
        f"{model} {layout} attempt resume/new/final prefix does not close",
    )
    resume_descriptor, resume_path = _exact_file_reference(
        receipt.get("resume"),
        base=close_path.parent,
        raw_root=raw_root,
        label=f"{model} {layout} attempt resume receipt",
    )
    resume = load_json(resume_path, f"{model} {layout} attempt resume receipt")
    require(
        resume.get("schema_version")
        == (n3_confirmation.RESUME_SCHEMA if model == "N3" else d1_confirmation.RESUME_SCHEMA)
        and resume.get("block_id") == schedule.block_id
        and resume.get("layout_pair_id") == layout
        and resume.get("start_cell_index") == len(before)
        and resume.get("completed_prefix_cell_ids") == list(schedule.cell_ids[:len(before)])
        and resume.get("prior_cell_receipts") == receipt.get("resumed_cell_receipts"),
        f"{model} {layout} attempt resume receipt differs from its prefix",
    )
    require(receipt.get("resume") == resume_descriptor,
            f"{model} {layout} attempt resume descriptor changed")
    return before, new, after


def _derive_attempt_tail(
    attempts: Sequence[Mapping[str, Any]], *, model: str, layout: str
) -> Mapping[str, Any]:
    """Replay a unique chronological retry chain and return its terminal tail.

    Callers cannot select an attempt.  The only eligible lifecycle is the
    chronological tail after exact prefix replay.  Overlap/equal boundaries
    are rejected because the legacy queue schema has no predecessor ordinal.
    """

    require(attempts, f"{model} {layout} has no terminal attempt lifecycle")
    ordered = sorted(attempts, key=lambda item: item["started_at"])
    require(
        len({item["started_at"] for item in ordered}) == len(ordered),
        f"{model} {layout} retry chronology has equal start times",
    )
    current: list[dict[str, Any]] = []
    produced: set[tuple[str, str, int]] = set()
    prior_end: datetime | None = None
    prior_terminal = False
    prior_launched = 0
    for ordinal, attempt in enumerate(ordered):
        start = attempt["started_at"]
        end = attempt["ended_at"]
        require(start < end, f"{model} {layout} attempt interval is empty or reversed")
        if prior_end is not None:
            require(prior_end < start,
                    f"{model} {layout} retry lifecycles overlap or have ambiguous ordering")
        require(not prior_terminal,
                f"{model} {layout} has a retry after passed or safety-censored evidence")
        before = list(attempt["before"])
        new = list(attempt["new"])
        after = list(attempt["after"])
        require(before == current and after == before + new,
                f"{model} {layout} retry does not resume the exact prior valid prefix")
        for descriptor in new:
            identity = (
                str(descriptor.get("path")),
                str(descriptor.get("sha256")),
                int(descriptor.get("bytes", -1)),
            )
            require(identity not in produced,
                    f"{model} {layout} retry duplicates a produced valid cell")
            produced.add(identity)
        counts = attempt["aggregate"]["counts"]
        launched = counts["launched_behavioral_cells"]
        require(ordinal == 0 or launched >= prior_launched,
                f"{model} {layout} retry regresses launched-prefix evidence")
        censored = counts["right_censored_behavioral_cells"]
        invalid = counts["technically_invalid_behavioral_cells"]
        status = attempt["aggregate"]["receipt"].get("status")
        if censored:
            require(censored == 1 and invalid == 0,
                    f"{model} {layout} safety-censored attempt categories changed")
            prior_terminal = True
        elif status == "passed":
            require(len(after) == 4 and invalid == 0,
                    f"{model} {layout} passed retry is not a complete prefix")
            prior_terminal = True
        else:
            require(status == "technical_failure" and invalid in {0, 1},
                    f"{model} {layout} retry outcome is not retryable technical evidence")
            if len(after) == 4:
                prior_terminal = True
        current = after
        prior_end = end
        prior_launched = launched
    maximal = max(len(item["after"]) for item in ordered)
    require(len(current) == maximal,
            f"{model} {layout} terminal retry is not the maximal valid prefix")
    return ordered[-1]


def _validate_block_attempt_lifecycles(
    *,
    queue_rows: Sequence[Mapping[str, Any]],
    close_path: Path,
    model: str,
    layout: str,
    schedule: Any,
    study_commit: str,
    source_root: Path,
    raw_root: Path,
    release_descriptor: Mapping[str, Any],
    release_freeze: Mapping[str, Any],
    queue_state_dir: Path | None = None,
) -> dict[str, Any]:
    """Deep-validate every retry and derive the sole selected lifecycle."""

    require(queue_rows, f"{model} {layout} cohort-close queue inventory is empty")
    jobs: dict[str, dict[str, Any]] = {}
    for queue_raw in queue_rows:
        job = _validate_queue_job(
            queue_raw,
            close_path=close_path,
            model=model,
            layout=layout,
            block_id=schedule.block_id,
            study_commit=study_commit,
            expected_state_dir=queue_state_dir,
        )
        job_id = job["job_id"]
        require(job_id not in jobs, f"{model} {layout} queue job is duplicated: {job_id}")
        jobs[job_id] = job

    lifecycle_rows: list[tuple[list[str], dict[str, Any] | None]] = []
    if model == "N3":
        require(all(job["mode"] == "queue" for job in jobs.values()),
                f"N3 {layout} queue lifecycle mode changed")
        lifecycle_rows = [([job_id], None) for job_id in sorted(jobs)]
    else:
        by_run: dict[str, dict[str, dict[str, Any]]] = {}
        for job in jobs.values():
            run_id = job.get("run_id")
            require(isinstance(run_id, str), f"D1 {layout} queue lifecycle lacks a run ID")
            roles = by_run.setdefault(run_id, {})
            require(job["mode"] not in roles,
                    f"D1 {layout} run {run_id} duplicates a queue role")
            roles[job["mode"]] = job
        for run_id, roles in sorted(by_run.items()):
            require(set(roles) == {"server-job", "simulator-job"},
                    f"D1 {layout} run {run_id} is an orphaned queue lifecycle")
            server = roles["server-job"]
            simulator = roles["simulator-job"]
            server_id = server["job_id"]
            simulator_id = simulator["job_id"]
            require(
                _option(server["descriptor_value"]["argv"], "--simulator-job-id", server_id)
                == simulator_id
                and _option(simulator["descriptor_value"]["argv"], "--server-job-id", simulator_id)
                == server_id,
                f"D1 {layout} run {run_id} queue peers are not reciprocal",
            )
            lifecycle_rows.append((sorted([server_id, simulator_id]), {"run_id": run_id}))

    attempts: list[dict[str, Any]] = []
    for job_ids, pair_seed in lifecycle_rows:
        aggregate_job = (
            jobs[job_ids[0]] if model == "N3"
            else next(jobs[item] for item in job_ids if jobs[item]["mode"] == "simulator-job")
        )
        runtime = aggregate_job["runtime_terminal_evidence"]
        aggregate_descriptor = runtime["runtime_receipt"]
        aggregate_path = Path(str(aggregate_descriptor["path"]))
        aggregate_value = runtime["runtime_value"]
        evidence_form = FULL_AGGREGATE_FORM
        if model == "D1" and not isinstance(aggregate_value.get("counts"), Mapping):
            evidence_form = D1_ZERO_LAUNCH_FORM
        aggregate = _validate_aggregate(
            descriptor=aggregate_descriptor,
            path=aggregate_path,
            model=model,
            layout=layout,
            schedule=schedule,
            study_commit=study_commit,
            source_root=source_root,
            raw_root=raw_root,
            release_descriptor=release_descriptor,
            release_freeze=release_freeze,
            evidence_form=evidence_form,
        ) | {"schedule": schedule}
        d1_runtime = None
        if model == "D1":
            d1_server = next(
                jobs[item] for item in job_ids if jobs[item]["mode"] == "server-job"
            )
            d1_simulator = next(
                jobs[item] for item in job_ids if jobs[item]["mode"] == "simulator-job"
            )
            simulator_role = d1_simulator["descriptor_value"].get("role")
            require(
                isinstance(simulator_role, str)
                and SAFE_COMPONENT_RE.fullmatch(simulator_role) is not None,
                f"D1 {layout} simulator queue role is invalid",
            )
            d1_runtime = {
                "study_commit": study_commit,
                "run_id": str(pair_seed["run_id"]),
                "server_job_id": d1_server["job_id"],
                "simulator_job_id": d1_simulator["job_id"],
                "simulator_worker_role": simulator_role,
            }
        failures = _validate_failure_cells(
            runtime["failure_cells"],
            close_path=close_path,
            model=model,
            layout=layout,
            schedule=schedule,
            aggregate=aggregate,
            raw_root=raw_root,
            d1_runtime=d1_runtime,
        )
        before, new, after = _attempt_prefix(
            aggregate=aggregate,
            close_path=close_path,
            raw_root=raw_root,
            schedule=schedule,
            model=model,
            layout=layout,
        )
        queue_jobs = [jobs[item] for item in job_ids]
        starts = [_queue_time(job["result_value"].get("started_at"), f"{job['job_id']} start")
                  for job in queue_jobs]
        ends = [_queue_time(job["result_value"].get("ended_at"), f"{job['job_id']} end")
                for job in queue_jobs]
        for job, start in zip(queue_jobs, starts):
            claim_time = float(job["claim_value"]["claimed_unix"])
            claim_text = _queue_time(
                job["claim_value"].get("claimed_at"), f"{job['job_id']} claim time"
            )
            require(abs(claim_text.timestamp() - claim_time) <= 2.0,
                    f"{job['job_id']} numeric/text claim times disagree")
            require(claim_time <= start.timestamp(),
                    f"{job['job_id']} queue result starts before its claim")
        if model == "D1":
            require(max(starts) < min(ends),
                    f"D1 {layout} server/simulator pair did not overlap")
            server_job = next(job for job in queue_jobs if job["mode"] == "server-job")
            simulator_job = next(job for job in queue_jobs if job["mode"] == "simulator-job")
            pair = {
                "run_id": pair_seed["run_id"],
                "server_job_id": server_job["job_id"],
                "simulator_job_id": simulator_job["job_id"],
                "server_receipt": server_job["runtime_terminal_evidence"]["runtime_receipt"],
                "simulator_terminal": simulator_job["runtime_terminal_evidence"]["protocol_terminal"],
            }
            _validate_d1_pair(
                pair,
                close_path=close_path,
                layout=layout,
                block_id=schedule.block_id,
                study_commit=study_commit,
                aggregate=aggregate,
                aggregate_descriptor=aggregate_descriptor,
                selected_jobs={job["job_id"]: job for job in queue_jobs},
            )
        else:
            pair = None
            _descriptor_matches(
                aggregate_job["descriptor"], aggregate["receipt"].get("queue_descriptor", {}),
                f"N3 {layout} aggregate queue",
            )
        attempts.append({
            "job_ids": list(job_ids),
            "started_at": min(starts),
            "ended_at": max(ends),
            "before": before,
            "new": new,
            "after": after,
            "aggregate": aggregate,
            "failures_by_index": failures,
            "d1_pair": pair,
        })
    tail = _derive_attempt_tail(attempts, model=model, layout=layout)
    derived_ids = list(tail["job_ids"])
    flagged_ids = sorted(
        job_id for job_id, job in jobs.items()
        if job.get("selected_for_block_evidence") is True
    )
    require(flagged_ids == derived_ids,
            f"{model} {layout} caller-selected lifecycle is not the derived retry tail")
    return {
        "jobs": jobs,
        "selected_queue_job_ids": derived_ids,
        "aggregate": tail["aggregate"],
        "failures_by_index": tail["failures_by_index"],
        "d1_pair": tail["d1_pair"],
        "attempt_count": len(attempts),
    }


def _validate_cohort_close_blocks(
    *,
    close: Mapping[str, Any],
    close_path: Path,
    raw_blocks: Sequence[Mapping[str, Any]],
    aggregates: Mapping[tuple[str, str], Mapping[str, Any]],
    study_commit: str,
    source_root: Path,
    raw_root: Path,
    release_descriptor: Mapping[str, Any],
    release_freeze: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    close_rows = close.get("blocks")
    require(isinstance(close_rows, list), "cohort-close block inventory is invalid")
    close_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in close_rows:
        require(isinstance(row, Mapping), "cohort-close block row is invalid")
        _exact_keys(row, BLOCK_INPUT_KEYS, "cohort-close block row")
        key = (row.get("model_id"), row.get("layout_pair_id"))
        require(key in aggregates and key not in close_by_key,
                "cohort-close block identity is invalid or duplicated")
        close_by_key[(str(key[0]), str(key[1]))] = row
    manifest_by_key = {
        (str(row.get("model_id")), str(row.get("layout_pair_id"))): row
        for row in raw_blocks
    }
    require(set(close_by_key) == set(aggregates) == set(manifest_by_key),
            "cohort-close blocks do not exactly cover the compiled cohort")
    require(
        all(dict(close_by_key[key]) == dict(manifest_by_key[key]) for key in close_by_key),
        "compiler block inventory differs from the cohort-close ledger",
    )
    first = next(iter(aggregates.values()))
    schedule_path = first["schedule"].schedule_path
    schedule_descriptor = development.file_descriptor(schedule_path)
    _descriptor_matches(close["prepared_schedule"], schedule_descriptor,
                        "cohort-close prepared schedule")
    all_job_ids: set[str] = set()
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key in sorted(close_by_key):
        model, layout = key
        row = close_by_key[key]
        aggregate = aggregates[key]
        schedule = aggregate["schedule"]
        require(row.get("evidence_form") == aggregate.get("evidence_form"),
                f"{model} {layout} cohort-close evidence form changed")
        require(row.get("block_id") == schedule.block_id,
                f"{model} {layout} cohort-close block ID changed")
        expected_row_sha = sha256_bytes(canonical_bytes(schedule.schedule_row))
        require(row.get("schedule_row_sha256") == expected_row_sha,
                f"{model} {layout} cohort-close schedule-row hash changed")
        aggregate_descriptor, _ = _exact_file_reference(
            row.get("receipt"), base=close_path.parent,
            label=f"{model} {layout} cohort-close aggregate receipt",
        )
        _descriptor_matches(aggregate_descriptor, aggregate["descriptor"],
                            f"{model} {layout} cohort-close aggregate")
        queue_rows = row.get("queue_jobs")
        require(isinstance(queue_rows, list) and queue_rows,
                f"{model} {layout} cohort-close queue inventory is empty")
        lifecycles = _validate_block_attempt_lifecycles(
            queue_rows=queue_rows,
            close_path=close_path,
            model=model,
            layout=layout,
            schedule=schedule,
            study_commit=study_commit,
            source_root=source_root,
            raw_root=raw_root,
            release_descriptor=release_descriptor,
            release_freeze=release_freeze,
            queue_state_dir=Path(str(close["queue_state_dir"])),
        )
        jobs = lifecycles["jobs"]
        for job_id in jobs:
            require(job_id not in all_job_ids,
                    f"cohort-close queue job is duplicated: {job_id}")
            all_job_ids.add(job_id)
        selected_ids = lifecycles["selected_queue_job_ids"]
        selected_aggregate = lifecycles["aggregate"]
        _descriptor_matches(
            selected_aggregate["descriptor"], aggregate_descriptor,
            f"{model} {layout} selected retry-tail aggregate",
        )
        require(
            row.get("evidence_form") == selected_aggregate.get("evidence_form")
            and row.get("selected_queue_job_ids") == selected_ids,
            f"{model} {layout} block evidence is not its derived retry tail",
        )
        selected_failure_rows = (
            next(
                job["runtime_terminal_evidence"]["failure_cells"]
                for job in jobs.values()
                if job["job_id"] in selected_ids and job["mode"] != "server-job"
            )
        )
        require(row.get("failure_cells") == selected_failure_rows,
                f"{model} {layout} selected failure inventory differs from retry tail")
        failures = lifecycles["failures_by_index"]
        pair = lifecycles["d1_pair"]
        require(row.get("d1_pair") == (
            None if pair is None else {
                key: pair[key] for key in D1_PAIR_KEYS
            }
        ), f"{model} {layout} paired lifecycle differs from retry tail")
        if aggregate.get("evidence_form") == D1_ZERO_LAUNCH_FORM:
            cell_tree = aggregate["raw_attempt"] / "cells"
            source_artifacts = [] if not cell_tree.exists() else [
                item for item in cell_tree.rglob("*")
                if item.is_symlink() or item.is_file()
            ]
            require(not source_artifacts,
                    f"D1 {layout} zero-launch form contradicts cell/failure artifacts")
        result[key] = {
            "failures_by_index": failures,
            "queue_jobs": jobs,
            "selected_queue_job_ids": list(selected_ids),
            "d1_pair": pair,
        }
    require(sorted(all_job_ids) == close.get("all_confirmation_queue_job_ids"),
            "cohort-close omitted or added confirmation queue jobs")
    return result


def _cell_root(raw_attempt: Path, index: int, cell_id: str) -> Path:
    _safe_component(cell_id)
    component = cell_id.replace("__", "-").replace("_", "-")
    require(
        component == n3_confirmation.pilot.safe_cell_component(cell_id)
        and component == d1_confirmation.pilot.safe_component(cell_id),
        f"runtime cell-path normalization changed: {cell_id}",
    )
    return raw_attempt / "cells" / f"{index:02d}-{component}"


def _completion_and_journal(
    *,
    cell_root: Path,
    cell: Mapping[str, Any] | None,
    raw_root: Path,
    required: bool,
) -> tuple[dict[str, Any] | None, Path | None, dict[str, Any] | None, Path | None]:
    if cell is not None:
        completion_value = cell.get("adapter_completion")
        journal_value = cell.get("adapter_journal")
    else:
        completion_path = cell_root / "recording" / "completion.json"
        journal_path = cell_root / "recording" / "events.partial.jsonl"
        if not completion_path.is_file() or not journal_path.is_file():
            require(not required, f"{cell_root.name} lacks a finalized native recording")
            return None, None, None, None
        completion_value = development.file_descriptor(completion_path)
        journal_value = development.file_descriptor(journal_path)
    completion_descriptor, completion_path = _descriptor(
        completion_value,
        base=cell_root,
        raw_root=raw_root,
        label=f"{cell_root.name} adapter completion",
    )
    journal_descriptor, journal_path = _descriptor(
        journal_value,
        base=cell_root,
        raw_root=raw_root,
        label=f"{cell_root.name} adapter journal",
    )
    return completion_descriptor, completion_path, journal_descriptor, journal_path


def _validate_completion(
    *,
    completion_path: Path,
    journal_path: Path,
    expected_cell_id: str,
    model: str,
    layout: str,
    condition: str,
    status: str,
    expected_actions: int,
    expected_prompt: str,
    expected_effective_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    completion = load_json(completion_path, f"{expected_cell_id} adapter completion")
    require(completion.get("schema_version") == "wmf-forecast-recording-attempt-v1",
            f"{expected_cell_id} adapter completion schema changed")
    require(completion.get("study_id") == STUDY_ID,
            f"{expected_cell_id} adapter completion study changed")
    identity = completion.get("identity")
    require(isinstance(identity, Mapping), f"{expected_cell_id} adapter identity is missing")
    arm, command = condition.split("_", 1)
    expected_identity = {
        "cell_id": expected_cell_id,
        "stage": "confirmation",
        "layout_pair_id": layout,
        "layout_arm": arm,
        "command": command,
        "prompt": expected_prompt,
        "model_config": model,
        "effective_seed": expected_effective_seed,
    }
    for key, wanted in expected_identity.items():
        require(identity.get(key) == wanted,
                f"{expected_cell_id} adapter identity {key} changed")
    attempt_id = identity.get("attempt_id")
    require(isinstance(attempt_id, str) and attempt_id,
            f"{expected_cell_id} adapter attempt identity is missing")
    require(completion.get("actions_executed") == expected_actions,
            f"{expected_cell_id} adapter/failure action counts differ")
    observation_count = completion.get("observation_count")
    require(type(observation_count) is int and 0 <= observation_count <= expected_actions + 1,
            f"{expected_cell_id} adapter observation count is invalid")
    request_count = completion.get("request_count")
    require(type(request_count) is int and request_count >= 0,
            f"{expected_cell_id} adapter request count is invalid")
    require(completion.get("success_configured_as_termination") is False,
            f"{expected_cell_id} adapter retained success termination")
    if status == "valid_complete":
        require(
            completion.get("behavioral_result_valid") is True
            and completion.get("stop_reason") == "action_cap"
            and completion.get("right_censored") is False
            and completion.get("technical_invalid") is False
            and completion.get("validation_errors") == []
            and observation_count == expected_actions + 1
            and expected_actions == 450,
            f"{expected_cell_id} is not a valid complete adapter recording",
        )
    elif status == "valid_censored":
        require(
            completion.get("behavioral_result_valid") is False
            and completion.get("stop_reason") == "safety_abort"
            and completion.get("right_censored") is True
            and completion.get("technical_invalid") is False
            and completion.get("validation_errors") == []
            and observation_count == expected_actions + 1
            and 0 < expected_actions < 450,
            f"{expected_cell_id} is not a valid safety-censored adapter recording",
        )
    else:
        require(
            status == "technical_invalid"
            and completion.get("behavioral_result_valid") is False
            and completion.get("technical_invalid") is True
            and completion.get("right_censored") is False
            and completion.get("stop_reason") == "technical_failure"
            and 0 <= expected_actions <= 450,
            f"{expected_cell_id} technical status differs from adapter completion",
        )
    rows, tail = development._verify_journal(journal_path)
    require(completion.get("event_count") == len(rows)
            and completion.get("journal_tail_sha256") == tail,
            f"{expected_cell_id} completion does not bind its full journal")
    recorded_journal = completion.get("journal_path")
    require(isinstance(recorded_journal, str)
            and not Path(recorded_journal).is_symlink()
            and Path(recorded_journal).resolve(strict=True) == journal_path,
            f"{expected_cell_id} completion cites another journal")
    require(rows and rows[0].get("kind") == "attempt_started"
            and rows[-1].get("kind") == "attempt_finalized",
            f"{expected_cell_id} journal attempt boundary is incomplete")
    finalized = development._event_payload(rows[-1], f"{expected_cell_id} finalized event")
    for key in (
        "schema_version", "study_id", "identity", "stop_reason",
        "behavioral_result_valid", "technical_invalid", "right_censored",
        "actions_executed", "observation_count", "request_count",
        "request_execution", "validation_errors",
    ):
        require(finalized.get(key) == completion.get(key),
                f"{expected_cell_id} finalized event differs at {key}")
    return completion, rows, dict(identity)


def _video_descriptor(
    *,
    cell_root: Path,
    cell: Mapping[str, Any] | None,
    raw_root: Path,
    required: bool,
) -> tuple[dict[str, Any] | None, Path | None]:
    if cell is not None:
        value = cell.get("viewport_video")
        descriptor, path = _descriptor(
            value, base=cell_root, raw_root=raw_root,
            label=f"{cell_root.name} viewport video",
        )
        return descriptor, path
    native = cell_root / "native_simulator"
    paths = [] if not native.is_dir() else sorted(
        path for path in native.rglob("*.mp4") if path.is_file() and path.stat().st_size > 0
    )
    require(len(paths) <= 1, f"{cell_root.name} has ambiguous viewport videos")
    require(not required or len(paths) == 1, f"{cell_root.name} lacks its safety-censored video")
    if not paths:
        return None, None
    descriptor, path = _descriptor(
        development.file_descriptor(paths[0]),
        base=cell_root,
        raw_root=raw_root,
        label=f"{cell_root.name} viewport video",
    )
    return descriptor, path


def _transported_request_descriptors(
    *,
    rows: Sequence[Mapping[str, Any]],
    completion_path: Path,
    raw_root: Path,
    cell_id: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, event in enumerate(development._event_rows(rows, "model_response_received")):
        payload = development._event_payload(event, f"{cell_id} response {index}")
        response = development._verify_payload_descriptor(
            payload.get("response_artifact"),
            base=completion_path.parent,
            raw_root=raw_root,
            label=f"{cell_id} response {index}",
            required_role="model_response",
        )
        raw_response = development._mapping_item(
            response["structure"], "raw_response", f"{cell_id} response {index}"
        )
        transported = development._thaw_scalar(
            development._mapping_item(
                raw_response, "wmf_server_request_receipt", f"{cell_id} response {index}"
            ),
            f"{cell_id} transported request descriptor {index}",
        )
        descriptor, _ = _descriptor(
            transported,
            base=completion_path.parent,
            raw_root=raw_root,
            label=f"{cell_id} official request {index}",
        )
        output.append(descriptor)
    return output


@contextmanager
def _temporary_limits(model: str, *, actions: int, observations: int, requests: int) -> Iterator[None]:
    original = development.MODEL_LIMITS[model]
    development.MODEL_LIMITS[model] = {
        **original,
        "action_cap": actions,
        "observation_count": observations,
        "request_count": requests,
    }
    try:
        yield
    finally:
        development.MODEL_LIMITS[model] = original


def _recorded_action_and_request_evidence(
    *,
    completion: Mapping[str, Any],
    completion_path: Path,
    rows: Sequence[Mapping[str, Any]],
    cell: Mapping[str, Any],
    model: str,
    camera_id: str,
    raw_root: Path,
) -> dict[str, Any]:
    actions = int(completion["actions_executed"])
    observations = int(completion["observation_count"])
    requests = int(completion["request_count"])
    require(requests == math.ceil(actions / MODEL_PREFIX[model]),
            f"{cell['cell_id']} requests do not reconstruct the executed prefix")
    working_cell = dict(cell)
    if model == "D1" and "server_request_receipts" not in working_cell:
        working_cell["server_request_receipts"] = _transported_request_descriptors(
            rows=rows,
            completion_path=completion_path,
            raw_root=raw_root,
            cell_id=str(cell["cell_id"]),
        )
    with _temporary_limits(
        model, actions=actions, observations=observations, requests=requests
    ):
        try:
            (
                request_descriptors,
                request_values,
                response_descriptors,
                executable_action_hashes,
                packed_request_bindings,
            ) = development._discover_requests(
                rows=rows,
                completion_path=completion_path,
                completion=completion,
                cell=working_cell,
                model=model,
                raw_root=raw_root,
            )
            observation_evidence, action_rows = development._observation_and_action_evidence(
                rows=rows,
                completion_path=completion_path,
                completion=completion,
                cell=working_cell,
                model=model,
                camera_id=camera_id,
                raw_root=raw_root,
                executable_action_hashes=executable_action_hashes,
            )
        except Exception as error:
            raise ConfirmationCompilerError(
                f"{cell['cell_id']} failed retained request/action replay: {error}"
            ) from error
    return {
        "request_descriptors": request_descriptors,
        "request_values": request_values,
        "response_descriptors": response_descriptors,
        "packed_request_bindings": packed_request_bindings,
        "observations": observation_evidence,
        "actions": action_rows,
    }


REQUEST_LIFECYCLE_KINDS = {
    "model_request_packed",
    "model_request_sent",
    "model_response_received",
    "model_request_completed",
}


def _indexed_request_events(
    *,
    rows: Sequence[Mapping[str, Any]],
    cell_id: str,
    kind: str,
    request_count: int,
) -> dict[int, tuple[dict[str, Any], dict[str, Any]]]:
    output: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in development._event_rows(rows, kind):
        payload = development._event_payload(row, f"{cell_id} {kind}")
        index = payload.get("request_index")
        require(
            type(index) is int
            and 0 <= index < request_count
            and index not in output,
            f"{cell_id} {kind} request indices are invalid or duplicated",
        )
        output[index] = (row, payload)
    require(
        sorted(output) == list(range(len(output))),
        f"{cell_id} {kind} request events are not one contiguous prefix",
    )
    return output


def _technical_response_equivalence(
    *,
    model: str,
    cell: Mapping[str, Any],
    request_index: int,
    request_descriptor: Mapping[str, Any],
    request_path: Path,
    request: Mapping[str, Any],
    response_descriptor: Mapping[str, Any],
    completion_path: Path,
    raw_root: Path,
    chunk_descriptor: Mapping[str, Any] | None,
) -> None:
    """Reuse the development deep artifact readers for a zero-action request."""

    require(
        dict(request_descriptor) == development.file_descriptor(request_path),
        f"{cell['cell_id']} technical official request descriptor changed",
    )
    raw_response = development._mapping_item(
        response_descriptor["structure"], "raw_response",
        f"{cell['cell_id']} technical response {request_index}",
    )
    if model == "N3":
        recorder_action_node = development._mapping_item(
            raw_response, "action", "N3 technical recorder raw response"
        )
        recorder_video_node = development._mapping_item(
            raw_response, "video", "N3 technical recorder raw response"
        )
        recorder_future = development._mapping_item(
            response_descriptor["structure"], "future_evidence",
            "N3 technical recorder future evidence",
        )
        recorder_decoded_node = development._mapping_item(
            recorder_future, "decoded", "N3 technical recorder decoded future"
        )
        recorder_action_identity = development._recorder_numpy_value_identity(
            response_descriptor, recorder_action_node,
            base=completion_path.parent, raw_root=raw_root,
            label=f"{cell['cell_id']} technical request {request_index} N3 action",
        )
        recorder_video_identity = development._recorder_numpy_value_identity(
            response_descriptor, recorder_video_node,
            base=completion_path.parent, raw_root=raw_root,
            label=f"{cell['cell_id']} technical request {request_index} N3 video",
        )
        require(
            development._recorder_numpy_value_identity(
                response_descriptor, recorder_decoded_node,
                base=completion_path.parent, raw_root=raw_root,
                label=f"{cell['cell_id']} technical request {request_index} decoded future",
            ) == recorder_video_identity,
            f"{cell['cell_id']} technical N3 raw/decoded future differs",
        )
        official_reference = request.get("official_returned_response")
        require(isinstance(official_reference, Mapping),
                f"{cell['cell_id']} technical N3 official response is missing")
        official_candidate = Path(str(official_reference.get("manifest_path")))
        if not official_candidate.is_absolute():
            official_candidate = request_path.parent / official_candidate
        official_path = development._under(
            official_candidate, raw_root,
            f"{cell['cell_id']} technical request {request_index} official N3 response",
        )
        require(
            sha256_file(official_path) == official_reference.get("manifest_sha256"),
            f"{cell['cell_id']} technical N3 response manifest changed",
        )
        official = load_json(official_path, "technical N3 official response")
        official_action = development._mapping_item(
            official.get("structure"), "action", "technical N3 official action"
        )
        official_video = development._mapping_item(
            official.get("structure"), "video", "technical N3 official video"
        )
        require(
            isinstance(official_action, Mapping)
            and official_action.get("value_sha256") == recorder_action_identity
            and isinstance(official_video, Mapping)
            and official_video.get("value_sha256") == recorder_video_identity,
            f"{cell['cell_id']} technical N3 recorder/official response differs",
        )
    else:
        episode_id = development._thaw_scalar(
            development._mapping_item(
                raw_response, "wmf_episode_context_id", "technical D1 raw response"
            ),
            "technical D1 episode context",
        )
        require(
            episode_id == request.get("episode_id"),
            f"{cell['cell_id']} technical D1 episode context changed",
        )
        raw_future = development._thaw_scalar(
            development._mapping_item(
                raw_response, "future_evidence", "technical D1 raw response"
            ),
            "technical D1 raw future evidence",
        )
        outer_future = development._thaw_scalar(
            development._mapping_item(
                response_descriptor["structure"], "future_evidence",
                "technical D1 recorder response",
            ),
            "technical D1 recorder future evidence",
        )
        require(
            isinstance(raw_future, Mapping) and raw_future == outer_future,
            f"{cell['cell_id']} technical D1 raw/recorder future differs",
        )
        expected_future = {
            "latent": request.get("latent_video"),
            "decoded.tensor": request.get("offline_decode", {}).get("decoded_tensor"),
            "decoded.rgb": request.get("offline_decode", {}).get("decoded_rgb"),
        }
        observed_future = {
            "latent": raw_future.get("latent"),
            "decoded.tensor": raw_future.get("decoded", {}).get("tensor")
            if isinstance(raw_future.get("decoded"), Mapping) else None,
            "decoded.rgb": raw_future.get("decoded", {}).get("rgb")
            if isinstance(raw_future.get("decoded"), Mapping) else None,
        }
        for key, official in expected_future.items():
            observed = observed_future[key]
            require(
                isinstance(official, Mapping) and isinstance(observed, Mapping),
                f"{cell['cell_id']} technical D1 {key} evidence is missing",
            )
            official_descriptor, _ = development._descriptor(
                {
                    "path": official.get("path"),
                    "sha256": official.get("file_sha256"),
                    "bytes": official.get("bytes"),
                },
                base=request_path.parent, raw_root=raw_root,
                label=f"technical D1 official {key}",
            )
            observed_descriptor, _ = development._descriptor(
                observed, base=completion_path.parent, raw_root=raw_root,
                label=f"technical D1 recorder {key}",
            )
            require(
                observed_descriptor == official_descriptor,
                f"{cell['cell_id']} technical D1 recorder/official {key} differs",
            )
    if chunk_descriptor is None:
        return
    returned_header, returned_raw = development._small_payload_array(
        chunk_descriptor, member_key="returned_action_chunk",
        base=completion_path.parent, raw_root=raw_root,
        label=f"{cell['cell_id']} technical request {request_index} returned action chunk",
    )
    limits = development.MODEL_LIMITS[model]
    require(
        list(returned_header["shape"]) == [limits["returned_actions"], 8],
        f"{cell['cell_id']} technical returned-action shape changed",
    )
    if model == "D1":
        official_action = request.get("official_returned_action")
        require(
            isinstance(official_action, Mapping)
            and sha256_bytes(returned_raw) == official_action.get("data_sha256"),
            f"{cell['cell_id']} technical D1 returned action differs",
        )
    else:
        official_reference = request["official_returned_response"]
        official_candidate = Path(str(official_reference["manifest_path"]))
        if not official_candidate.is_absolute():
            official_candidate = request_path.parent / official_candidate
        official_path = development._under(
            official_candidate, raw_root,
            f"{cell['cell_id']} technical request {request_index} official N3 action",
        )
        official = load_json(official_path, "technical N3 official response")
        official_node = development._mapping_item(
            official.get("structure"), "action", "technical N3 official action"
        )
        returned_identity = sha256_bytes(
            canonical_bytes(
                {
                    "kind": "numpy",
                    "dtype": returned_header["descr"],
                    "shape": list(returned_header["shape"]),
                },
                ensure_ascii=True,
            ) + returned_raw
        )
        require(
            isinstance(official_node, Mapping)
            and official_node.get("__type__") == "numpy"
            and official_node.get("value_sha256") == returned_identity,
            f"{cell['cell_id']} technical N3 returned action differs",
        )


def _technical_retained_evidence(
    *,
    completion: Mapping[str, Any],
    completion_path: Path,
    rows: Sequence[Mapping[str, Any]],
    cell: Mapping[str, Any],
    model: str,
    camera_id: str,
    raw_root: Path,
) -> dict[str, Any]:
    """Authenticate complete action-bearing requests and any final partial request."""

    cell_id = str(cell["cell_id"])
    action_count = int(completion["actions_executed"])
    observation_count = int(completion["observation_count"])
    request_count = int(completion["request_count"])
    execution = completion.get("request_execution")
    require(
        isinstance(execution, list) and len(execution) == request_count,
        f"{cell_id} technical completion request inventory changed",
    )
    event_maps = {
        kind: _indexed_request_events(
            rows=rows, cell_id=cell_id, kind=kind, request_count=request_count
        )
        for kind in REQUEST_LIFECYCLE_KINDS
    }
    require(
        len(event_maps["model_request_packed"]) == request_count,
        f"{cell_id} technical request inputs are not fully retained",
    )
    stage_counts = [
        len(event_maps[kind])
        for kind in (
            "model_request_packed", "model_request_sent",
            "model_response_received", "model_request_completed",
        )
    ]
    require(
        stage_counts == sorted(stage_counts, reverse=True),
        f"{cell_id} technical request lifecycle is not a nested prefix",
    )
    require(
        request_count <= 1
        or len(event_maps["model_request_completed"]) >= request_count - 1,
        f"{cell_id} started another request after an incomplete request",
    )
    limits = development.MODEL_LIMITS[model]
    expected_start = 0
    action_request_count = 0
    stage_inventory: list[dict[str, Any]] = []
    manual_requests: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for index, raw_execution in enumerate(execution):
        require(isinstance(raw_execution, Mapping),
                f"{cell_id} technical request execution {index} is invalid")
        executed = raw_execution.get("executed_actions")
        require(
            raw_execution.get("request_index") == index
            and raw_execution.get("action_step_start") == expected_start
            and type(executed) is int
            and 0 <= executed <= limits["executed_prefix"]
            and raw_execution.get("returned_actions") == limits["returned_actions"]
            and raw_execution.get("eligible_executable_prefix_actions")
            == limits["executed_prefix"]
            and raw_execution.get("unused_executable_prefix_actions")
            == limits["executed_prefix"] - executed
            and raw_execution.get("returned_actions_outside_executable_prefix")
            == limits["returned_actions"] - limits["executed_prefix"]
            and raw_execution.get("current_observation_id")
            == f"obs_{expected_start:06d}"
            and raw_execution.get("preceding_observation_id")
            == (None if expected_start == 0 else f"obs_{expected_start - 1:06d}"),
            f"{cell_id} technical request execution {index} changed",
        )
        if index < request_count - 1:
            require(
                executed == limits["executed_prefix"],
                f"{cell_id} continued after a short technical request prefix",
            )
        if executed > 0:
            require(
                index == action_request_count
                and index in event_maps["model_request_completed"],
                f"{cell_id} action-bearing technical request is incomplete",
            )
            action_request_count += 1
        else:
            require(
                index == request_count - 1,
                f"{cell_id} zero-action request is not the terminal request",
            )
        expected_start += executed

        packed_row, packed = event_maps["model_request_packed"][index]
        packed_descriptor = development._verify_payload_descriptor(
            packed.get("model_request_artifact"),
            base=completion_path.parent, raw_root=raw_root,
            label=f"{cell_id} technical packed request {index}",
            required_role="model_request",
        )
        development._mapping_item(
            packed_descriptor["structure"], "extracted_preprocessing_output",
            f"{cell_id} technical packed request {index}",
        )
        development._mapping_item(
            packed_descriptor["structure"], "wire_request",
            f"{cell_id} technical packed request {index}",
        )
        require(
            packed.get("action_step_start") == raw_execution["action_step_start"]
            and packed.get("current_observation_id")
            == raw_execution["current_observation_id"]
            and packed.get("preceding_observation_id")
            == raw_execution["preceding_observation_id"]
            and packed.get("returned_action_horizon") == limits["returned_actions"]
            and packed.get("executed_prefix_horizon") == limits["executed_prefix"]
            and packed.get("required_future_evidence")
            == limits["required_future_evidence"],
            f"{cell_id} technical packed request {index} differs from completion",
        )
        sent_pair = event_maps["model_request_sent"].get(index)
        response_pair = event_maps["model_response_received"].get(index)
        completed_pair = event_maps["model_request_completed"].get(index)
        require(
            (response_pair is None or sent_pair is not None)
            and (completed_pair is None or response_pair is not None),
            f"{cell_id} technical request {index} skips a lifecycle stage",
        )
        prior_sequence = packed_row["sequence"]
        sent_payload = response_payload = completed_payload = None
        if sent_pair is not None:
            sent_row, sent_payload = sent_pair
            require(
                sent_row["sequence"] > prior_sequence
                and type(packed.get("pack_monotonic_ns")) is int
                and type(sent_payload.get("send_monotonic_ns")) is int
                and sent_payload["send_monotonic_ns"] >= packed["pack_monotonic_ns"],
                f"{cell_id} technical request {index} send clocks/order changed",
            )
            prior_sequence = sent_row["sequence"]
        request_descriptor = request_value = response_descriptor = None
        request_path = None
        if response_pair is not None:
            response_row, response_payload = response_pair
            require(
                response_row["sequence"] > prior_sequence
                and sent_payload is not None
                and type(response_payload.get("receive_monotonic_ns")) is int
                and response_payload["receive_monotonic_ns"]
                >= sent_payload["send_monotonic_ns"],
                f"{cell_id} technical request {index} response clocks/order changed",
            )
            response_descriptor = development._verify_payload_descriptor(
                response_payload.get("response_artifact"),
                base=completion_path.parent, raw_root=raw_root,
                label=f"{cell_id} technical response {index}",
                required_role="model_response",
            )
            raw_response = development._mapping_item(
                response_descriptor["structure"], "raw_response",
                f"{cell_id} technical raw response {index}",
            )
            transported = development._thaw_scalar(
                development._mapping_item(
                    raw_response, "wmf_server_request_receipt",
                    f"{cell_id} technical raw response {index}",
                ),
                f"{cell_id} technical transported request {index}",
            )
            request_descriptor, request_path = development._descriptor(
                transported, base=completion_path.parent, raw_root=raw_root,
                label=f"{cell_id} technical official request {index}",
            )
            request_value = development._verify_request_receipt(
                descriptor=request_descriptor, request_path=request_path,
                model=model, cell=cell, request_index=index, raw_root=raw_root,
            )
            require(
                response_payload.get("future_kinds") == raw_execution.get("future_kinds"),
                f"{cell_id} technical response future inventory changed",
            )
            prior_sequence = response_row["sequence"]
        chunk_descriptor = None
        if completed_pair is not None:
            completed_row, completed_payload = completed_pair
            require(
                completed_row["sequence"] > prior_sequence,
                f"{cell_id} technical request {index} completed out of order",
            )
            chunk_descriptor = development._verify_payload_descriptor(
                completed_payload.get("action_chunks_artifact"),
                base=completion_path.parent, raw_root=raw_root,
                label=f"{cell_id} technical request {index} action chunks",
                required_role="action_chunks",
            )
            require(
                raw_execution.get("action_chunks_artifact") == chunk_descriptor
                and completed_payload.get("returned_action_shape")
                == [limits["returned_actions"], 8]
                and isinstance(completed_payload.get("executable_action_shape"), list)
                and completed_payload["executable_action_shape"][:1]
                and completed_payload["executable_action_shape"][0]
                >= limits["executed_prefix"]
                and completed_payload["executable_action_shape"][1:] == [8]
                and completed_payload.get("missing_future_evidence") == [],
                f"{cell_id} technical request {index} completion/chunk changed",
            )
        else:
            require(
                raw_execution.get("action_chunks_artifact") is None
                and executed == 0,
                f"{cell_id} incomplete technical request owns actions or a chunk",
            )
        if request_descriptor is not None:
            assert request_path is not None and request_value is not None
            _technical_response_equivalence(
                model=model, cell=cell, request_index=index,
                request_descriptor=request_descriptor, request_path=request_path,
                request=request_value, response_descriptor=response_descriptor,
                completion_path=completion_path, raw_root=raw_root,
                chunk_descriptor=chunk_descriptor,
            )
        manual_requests.append((
            {} if request_descriptor is None else request_descriptor,
            {} if request_value is None else request_value,
            {} if response_descriptor is None else response_descriptor,
        ))
        stage_inventory.append({
            "request_index": index,
            "stage": (
                "completed" if completed_pair is not None
                else "response_received" if response_pair is not None
                else "sent" if sent_pair is not None else "packed"
            ),
            "action_step_start": raw_execution["action_step_start"],
            "executed_actions": executed,
            "packed_model_request": development._payload_binding(
                packed_descriptor, base=completion_path.parent, raw_root=raw_root,
                label=f"{cell_id} technical packed request {index}",
            ),
            "official_request_receipt": request_descriptor,
            "recorder_response": (
                None if response_descriptor is None
                else development._payload_binding(
                    response_descriptor, base=completion_path.parent, raw_root=raw_root,
                    label=f"{cell_id} technical response {index}",
                )
            ),
            "action_chunks": (
                None if chunk_descriptor is None
                else development._payload_binding(
                    chunk_descriptor, base=completion_path.parent, raw_root=raw_root,
                    label=f"{cell_id} technical request {index} action chunks",
                )
            ),
        })
    require(
        expected_start == action_count,
        f"{cell_id} technical request allocations do not close executed actions",
    )

    if action_request_count:
        deep_rows = []
        for row in rows:
            if row.get("kind") not in REQUEST_LIFECYCLE_KINDS:
                deep_rows.append(row)
                continue
            payload = development._event_payload(row, f"{cell_id} technical request filter")
            if payload.get("request_index") < action_request_count:
                deep_rows.append(row)
        deep_completion = {
            **dict(completion),
            "request_count": action_request_count,
            "request_execution": [dict(row) for row in execution[:action_request_count]],
        }
        recorded = _recorded_action_and_request_evidence(
            completion=deep_completion, completion_path=completion_path,
            rows=deep_rows, cell=cell, model=model, camera_id=camera_id,
            raw_root=raw_root,
        )
    else:
        with _temporary_limits(
            model, actions=0, observations=observation_count, requests=0
        ):
            try:
                observations, actions = development._observation_and_action_evidence(
                    rows=rows, completion_path=completion_path,
                    completion={**dict(completion), "request_count": 0, "request_execution": []},
                    cell=cell, model=model, camera_id=camera_id, raw_root=raw_root,
                    executable_action_hashes=[],
                )
            except Exception as error:
                raise ConfirmationCompilerError(
                    f"{cell_id} failed retained zero-action observation replay: {error}"
                ) from error
        recorded = {
            "request_descriptors": [], "request_values": [],
            "response_descriptors": [], "packed_request_bindings": [],
            "observations": observations, "actions": actions,
        }
    # The general lifecycle pass independently validates every request, including
    # a terminal zero-action request that the action-bearing replay omits.
    return {
        "request_descriptors": [
            item[0] for item in manual_requests if item[0]
        ],
        "request_values": [item[1] for item in manual_requests if item[1]],
        "response_descriptors": [item[2] for item in manual_requests if item[2]],
        "packed_request_bindings": [
            item["packed_model_request"] for item in stage_inventory
        ],
        "observations": recorded["observations"],
        "actions": recorded["actions"],
        "technical_request_evidence": stage_inventory,
    }


def _validate_partial_context(
    *,
    completion: Mapping[str, Any],
    completion_path: Path,
    rows: Sequence[Mapping[str, Any]],
    raw_root: Path,
    model: str,
    cell_id: str,
) -> dict[str, Any]:
    events = development._event_rows(rows, "model_context_reset")
    require(len(events) == 1, f"{cell_id} lacks exactly one context reset")
    payload = development._event_payload(events[0], f"{cell_id} context reset")
    require(payload.get("artifact") == completion.get("context_reset_artifact"),
            f"{cell_id} context reset differs from completion")
    descriptor = development._verify_payload_descriptor(
        payload.get("artifact"),
        base=completion_path.parent,
        raw_root=raw_root,
        label=f"{cell_id} context reset",
        required_role="context_reset",
    )
    value = development._thaw_scalar(descriptor["structure"], f"{cell_id} context reset")
    require(isinstance(value, Mapping), f"{cell_id} context reset payload is invalid")
    require(value.get("passed") is True
            and value.get("reset_scope") == development.CONTEXT_RESET_SCOPE,
            f"{cell_id} context reset did not pass the full temporal/cache scope")
    if model == "N3":
        cache = value.get("cache_reset_evidence")
        require(isinstance(cache, Mapping)
                and cache.get("passed") is True
                and cache.get("exclusive_active_episode") == cell_id
                and cache.get("wrapper_request_index_reset_to_zero") is True
                and cache.get("unresolved_mutable_temporal_fields") == [],
                f"{cell_id} N3 context reset is incomplete")
    else:
        cache = value.get("cache_reset_evidence")
        require(value.get("service_route_proved_by_reset_artifact") is True
                and isinstance(cache, Mapping)
                and cache.get("source") == "validated_official_d1_two_rank_reset_receipt"
                and cache.get("world_size") == 2
                and cache.get("unresolved_mutable_temporal_fields") == [],
                f"{cell_id} D1 context reset is incomplete")
    later = development._event_rows(rows, "model_request_packed")
    require(later and events[0]["sequence"] < later[0]["sequence"],
            f"{cell_id} context reset did not precede inference")
    binding = development._payload_binding(
        descriptor,
        base=completion_path.parent,
        raw_root=raw_root,
        label=f"{cell_id} context reset",
    )
    return {"binding": binding, "value": dict(value)}


def _validate_censored_context_terminal(
    *,
    model: str,
    cell_id: str,
    layout: str,
    condition_index: int,
    condition: str,
    schedule: Any,
    failure_path: Path,
    failure_value: Mapping[str, Any],
    study_commit: str,
    d1_runtime: Mapping[str, str] | None,
    completion: Mapping[str, Any],
    context_reset: Mapping[str, Any],
    terminal_descriptor: Mapping[str, Any] | None,
    terminal_path: Path | None,
    request_descriptors: Sequence[Mapping[str, Any]],
    request_values: Sequence[Mapping[str, Any]],
    raw_root: Path,
) -> dict[str, Any]:
    """Deep-validate the frozen server-authored post-terminal context chain."""

    expected_schema = (
        N3_CONTEXT_TERMINAL_SCHEMA if model == "N3" else D1_CONTEXT_TERMINAL_SCHEMA
    )
    require(
        schedule.condition_order[condition_index].replace("-", "_") == condition,
        f"{cell_id} safety condition differs from its frozen schedule",
    )
    require(
        terminal_descriptor is not None
        and terminal_path is not None
        and failure_value.get("status") == "safety_abort"
        and failure_value.get("recorded_stop_reason") == "safety_abort"
        and failure_value.get("server_context_terminal") == terminal_descriptor,
        f"{cell_id} safety_abort lacks its close-bound {expected_schema} evidence",
    )
    observed_terminal, observed_terminal_path = _exact_file_reference(
        terminal_descriptor,
        base=failure_path.parent,
        raw_root=raw_root,
        label=f"{cell_id} frozen server-context terminal",
    )
    require(
        observed_terminal == terminal_descriptor
        and observed_terminal_path == terminal_path.resolve(),
        f"{cell_id} server-context terminal descriptor/path changed",
    )
    try:
        if model == "N3":
            validated_failure = n3_confirmation.validate_failed_confirmation_cell(
                failure_path, condition_index=condition_index, block=schedule
            )
            validator_name = "n3_confirmation.validate_failed_confirmation_cell"
        else:
            require(
                d1_runtime is not None,
                f"{cell_id} D1 safety_abort lacks its selected queue lifecycle",
            )
            with d1_confirmation._configured_for_validation(
                schedule, d1_runtime["simulator_worker_role"]
            ):
                validated_failure = d1_confirmation.validate_failed_confirmation_cell(
                    failure_path, condition_index=condition_index, block=schedule
                )
            require(
                failure_value.get("study_commit") == study_commit
                == d1_runtime["study_commit"]
                and failure_value.get("run_id") == d1_runtime["run_id"]
                and failure_value.get("server_job_id")
                == d1_runtime["server_job_id"]
                and failure_value.get("simulator_job_id")
                == d1_runtime["simulator_job_id"],
                f"{cell_id} D1 safety_abort is detached from its selected queue pair",
            )
            validator_name = "d1_confirmation.validate_failed_confirmation_cell"
    except ConfirmationCompilerError:
        raise
    except Exception as error:
        raise ConfirmationCompilerError(
            f"{cell_id} failed frozen native {expected_schema} validation: {error}"
        ) from error
    require(
        validated_failure == failure_value,
        f"{cell_id} frozen safety validator changed its source bytes",
    )
    terminal = load_json(terminal_path, f"{cell_id} server context terminal")
    require(
        terminal.get("schema_version") == expected_schema
        and terminal.get("status") == "passed"
        and terminal.get("terminal_state") == "context_closed"
        and terminal.get("model_config") == model
        and terminal.get("study_id") == STUDY_ID
        and terminal.get("phase") == "confirmation"
        and terminal.get("block_id") == schedule.block_id
        and terminal.get("layout_pair_id") == layout
        and terminal.get("cell_id") == cell_id
        and terminal.get("condition_index") == condition_index
        and terminal.get("stop_reason") == "safety_abort"
        and terminal.get("actions_executed") == completion.get("actions_executed")
        and terminal.get("request_count") == completion.get("request_count")
        and terminal.get("server_request_count") == completion.get("request_count")
        and len(request_descriptors) == len(request_values)
        == completion.get("request_count"),
        f"{cell_id} server terminal identity/counts differ from native recording",
    )
    if model == "N3":
        require(
            terminal.get("context_active_after_end") is False
            and terminal.get("model_capture_active_after_end") is False,
            f"{cell_id} N3 context remained active after terminal end",
        )
        terminal_reset_sha256 = None
    else:
        require(
            terminal.get("episode_bookkeeping_cleared") is True
            and terminal.get("context_active_after_finalize") is False
            and isinstance(terminal.get("episode_manifest"), Mapping)
            and isinstance(terminal.get("final_two_rank_reset"), Mapping),
            f"{cell_id} D1 terminal finalize/reset evidence is incomplete",
        )
        terminal_reset_sha256 = sha256_bytes(
            canonical_bytes(terminal["final_two_rank_reset"])
        )
    return {
        "schema_version": expected_schema,
        "descriptor": dict(terminal_descriptor),
        "validated_by": validator_name,
        "terminal_runtime_source_commit": TERMINAL_RUNTIME_SOURCE_COMMIT,
        "terminal_state": "context_closed",
        "stop_reason": "safety_abort",
        "actions_executed": terminal["actions_executed"],
        "request_count": terminal["request_count"],
        "server_context_id": terminal["server_context_id"],
        "client_session_id": terminal["client_session_id"],
        "initial_context_reset": dict(context_reset["binding"]),
        "final_two_rank_reset_sha256": terminal_reset_sha256,
    }


def _alignment_objects(
    *,
    release_path: Path,
    release_freeze: Mapping[str, Any],
    models: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result = {}
    entries = release_freeze.get("alignment_contracts_by_model")
    require(isinstance(entries, Mapping) and set(entries) == set(models),
            "confirmation release alignment model coverage changed")
    for model in models:
        entry = entries[model]
        require(isinstance(entry, Mapping), f"{model} alignment entry is invalid")
        alignment_descriptor, alignment_path = _descriptor(
            {"path": entry.get("path"), "sha256": entry.get("sha256"),
             "bytes": (release_path.parent / str(entry.get("path"))).resolve().stat().st_size},
            base=release_path.parent,
            label=f"{model} alignment contract",
        )
        alignment = load_json(alignment_path, f"{model} alignment contract")
        unsigned = dict(alignment)
        contract_sha = unsigned.pop("contract_sha256", None)
        require(contract_sha == entry.get("contract_sha256")
                and sha256_bytes(canonical_bytes(unsigned)) == contract_sha,
                f"{model} alignment contract hash changed")
        mapping_value = entry.get("mapping_receipt")
        require(isinstance(mapping_value, Mapping), f"{model} mapping receipt entry is invalid")
        mapping_candidate = Path(str(mapping_value.get("path")))
        if not mapping_candidate.is_absolute():
            mapping_candidate = release_path.parent / mapping_candidate
        mapping_descriptor, mapping_path = _descriptor(
            {"path": str(mapping_value.get("path")), "sha256": mapping_value.get("sha256"),
             "bytes": mapping_candidate.resolve().stat().st_size},
            base=release_path.parent,
            label=f"{model} physical mapping receipt",
        )
        mapping = load_json(mapping_path, f"{model} physical mapping receipt")
        require(mapping.get("receipt_id") == alignment.get("mapping_receipt_id")
                and alignment.get("mapping_receipt_sha256") == mapping_descriptor["sha256"],
                f"{model} alignment/mapping binding changed")
        result[model] = {
            "entry": dict(entry),
            "alignment": alignment,
            "alignment_descriptor": alignment_descriptor,
            "mapping": mapping,
            "mapping_descriptor": mapping_descriptor,
        }
    return result


def _request_inventory_row(
    *,
    model: str,
    cell_id: str,
    layout: str,
    condition: str,
    recording_id: str,
    source_video_id: str,
    video_sha256: str,
    action_sha256: str,
    request_index: int,
    execution: Mapping[str, Any],
    request_descriptor: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Any]],
    alignment_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    alignment = alignment_bundle["alignment"]
    mapping_descriptor = alignment_bundle["mapping_descriptor"]
    start = execution.get("action_step_start")
    executed = execution.get("executed_actions")
    require(type(start) is int and type(executed) is int and executed > 0,
            f"{cell_id} request {request_index} execution prefix is invalid")
    target_offset = alignment.get("target_executed_action_offset")
    require(type(target_offset) is int and target_offset > 0,
            f"{model} frozen target action offset is invalid")
    within = target_offset <= executed
    mapping_applies = model == "N3" or request_index % 4 == 0
    timestamp_error: float | None = None
    current = observations.get(str(execution.get("current_observation_id")))
    require(isinstance(current, Mapping), f"{cell_id} request {request_index} current observation is absent")
    camera_match = current.get("camera_frame_id", "").startswith(str(alignment["camera_id"]) + ":")
    if mapping_applies and within:
        target = observations.get(f"obs_{start + target_offset:06d}")
        require(isinstance(target, Mapping), f"{cell_id} request {request_index} target observation is absent")
        camera_elapsed = (
            int(target["camera_capture_time_ns"]) - int(current["camera_capture_time_ns"])
        ) / 1_000_000_000
        physics_elapsed = float(target["physics_time_s"]) - float(current["physics_time_s"])
        require(camera_elapsed > 0 and physics_elapsed > 0,
                f"{cell_id} request {request_index} native clocks did not advance")
        target_time = float(alignment["primary_horizon_s"])
        timestamp_error = max(abs(camera_elapsed - target_time), abs(physics_elapsed - target_time))
    return {
        "cell_id": cell_id,
        "source_request_id": "request_" + str(request_descriptor["sha256"])[:32],
        "source_video_id": source_video_id,
        "source_video_sha256": video_sha256,
        "model_id": model,
        "layout_pair_id": layout,
        "condition_id": condition,
        "episode_id": recording_id,
        "request_index": request_index,
        "request_start_action_index": start,
        "action_manifest_sha256": action_sha256,
        "camera_id": alignment["camera_id"],
        "camera_crop_id": alignment["camera_crop_id"],
        "camera_crop_sha256": alignment["camera_crop_sha256"],
        "alignment_contract_id": alignment["contract_id"],
        "alignment_contract_sha256": alignment["contract_sha256"],
        "technical_valid": mapping_applies,
        "technical_invalid_reason": None if mapping_applies else "forecast_timing_unavailable",
        "camera_identity_match": camera_match,
        "target_within_executed_prefix": within,
        "executed_prefix_actions": executed,
        "target_executed_action_offset": target_offset,
        "generated_frame_index": alignment["generated_frame_index"],
        "target_physical_time_s": alignment["primary_horizon_s"],
        "timestamp_error_s": timestamp_error,
        "timestamp_tolerance_s": alignment["timestamp_tolerance_s"],
        "history_mode": (
            "persistence_at_initial_request" if request_index == 0 else "preceding_observation"
        ),
        "early_horizon_supported": alignment.get("early_horizon") is not None,
        "alignment_receipt_id": alignment["mapping_receipt_id"],
        "alignment_receipt_sha256": mapping_descriptor["sha256"],
    }


def _source_identity_check(
    *, completion: Mapping[str, Any], model: str, study_commit: str, cell_id: str,
    expected_pose_sha256: str,
) -> dict[str, Any]:
    identity = completion["identity"]
    pins = development.MODEL_PINS[model]
    source = identity.get("source_identity")
    checkpoint = identity.get("checkpoint_identity")
    model_fragment = (
        f"cosmos:{pins['cosmos_commit']}"
        if model == "N3"
        else f"dreamzero:{pins['dreamzero_commit']}"
    )
    prefix = f"study:{study_commit};robolab:{pins['robolab_commit']};{model_fragment};pose:"
    require(isinstance(source, str) and source == prefix + expected_pose_sha256
            and SHA256_RE.fullmatch(expected_pose_sha256) is not None,
            f"{cell_id} source identity differs from pinned confirmation sources")
    expected_checkpoint = (
        f"revision:{pins['checkpoint_revision']};aggregate:{pins['checkpoint_aggregate_sha256']}"
    )
    require(checkpoint == expected_checkpoint,
            f"{cell_id} checkpoint identity differs from the selected model")
    return {
        "source_identity": source,
        "checkpoint_identity": checkpoint,
        "pose_manifest_sha256": source[len(prefix):],
    }


def _expected_cell_identity(model: str, condition: str, schedule: Any) -> tuple[str, int]:
    _arm, command = condition.split("_", 1)
    prompts = n3_confirmation.pilot.PROMPTS if model == "N3" else d1_confirmation.pilot.PROMPTS
    require(command in prompts, f"{model} confirmation command has no frozen prompt")
    effective_seed = (
        int(schedule.effective_seed)
        if model == "N3"
        else int(d1_confirmation.pilot.EFFECTIVE_MODEL_NOISE_SEED)
    )
    return str(prompts[command]), effective_seed


def _require_optional_descriptor_match(
    observed: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    label: str,
) -> None:
    require((observed is None) == (expected is None), f"{label} presence changed after cohort close")
    if observed is not None and expected is not None:
        _descriptor_matches(observed, expected, label)


def _compile_recorded_cell(
    *,
    model: str,
    layout: str,
    condition: str,
    condition_index: int,
    cell_id: str,
    status: str,
    action_count: int,
    source_descriptor: Mapping[str, Any],
    source_cell_path: Path | None,
    source_failure_path: Path | None,
    source_root: Path,
    study_commit: str,
    raw_root: Path,
    camera_id: str,
    schedule: Any,
    alignment_bundle: Mapping[str, Any],
    expected_pose_sha256: str,
    failure_declaration: Mapping[str, Any] | None = None,
    d1_runtime: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    cell_root = _cell_root(
        (source_cell_path.parents[2] if source_cell_path is not None else source_failure_path.parents[2]),
        condition_index,
        cell_id,
    )
    # Above reconstruction must equal the actual cell directory.  It also
    # rejects a descriptor smuggled from another attempt or condition.
    actual_parent = (source_cell_path if source_cell_path is not None else source_failure_path).parent
    require(cell_root.resolve() == actual_parent.resolve(),
            f"{cell_id} source artifact is outside its canonical cell directory")
    cell = None if source_cell_path is None else load_json(source_cell_path, f"{cell_id} cell receipt")
    if status == "valid_complete":
        try:
            if model == "N3":
                validated, identity = n3_confirmation.validate_passed_confirmation_cell(
                    source_cell_path,
                    condition_index=condition_index,
                    study_commit=study_commit,
                    block=schedule,
                )
            else:
                validated, identity = d1_confirmation.validate_passed_confirmation_cell(
                    source_cell_path,
                    condition_index=condition_index,
                    study_commit=study_commit,
                    block=schedule,
                )
        except Exception as error:
            raise ConfirmationCompilerError(f"{cell_id} failed confirmation cell validation: {error}") from error
        require(validated == cell and identity["sha256"] == source_descriptor["sha256"],
                f"{cell_id} confirmation validator/source descriptor differs")
    completion_descriptor, completion_path, journal_descriptor, journal_path = _completion_and_journal(
        cell_root=actual_parent,
        cell=cell,
        raw_root=raw_root,
        required=True,
    )
    assert completion_descriptor is not None and completion_path is not None
    assert journal_descriptor is not None and journal_path is not None
    expected_prompt, expected_seed = _expected_cell_identity(model, condition, schedule)
    completion, rows, identity = _validate_completion(
        completion_path=completion_path,
        journal_path=journal_path,
        expected_cell_id=cell_id,
        model=model,
        layout=layout,
        condition=condition,
        status=status,
        expected_actions=action_count,
        expected_prompt=expected_prompt,
        expected_effective_seed=expected_seed,
    )
    source_identity = _source_identity_check(
        completion=completion, model=model, study_commit=study_commit, cell_id=cell_id,
        expected_pose_sha256=expected_pose_sha256,
    )
    context_reset = None
    if status == "valid_censored":
        context_reset = _validate_partial_context(
            completion=completion,
            completion_path=completion_path,
            rows=rows,
            raw_root=raw_root,
            model=model,
            cell_id=cell_id,
        )
    source_cell = cell if cell is not None else {
        "cell_id": cell_id,
        "prompt": identity.get("prompt"),
        "effective_seed": identity.get("effective_seed"),
    }
    recorded = _recorded_action_and_request_evidence(
        completion=completion,
        completion_path=completion_path,
        rows=rows,
        cell=source_cell,
        model=model,
        camera_id=camera_id,
        raw_root=raw_root,
    )
    video_descriptor, video_path = _video_descriptor(
        cell_root=actual_parent,
        cell=cell,
        raw_root=raw_root,
        required=True,
    )
    assert video_descriptor is not None and video_path is not None
    context_terminal = None
    if failure_declaration is not None:
        _require_optional_descriptor_match(
            completion_descriptor, failure_declaration.get("adapter_completion"),
            f"{cell_id} close-time adapter completion",
        )
        _require_optional_descriptor_match(
            journal_descriptor, failure_declaration.get("adapter_journal"),
            f"{cell_id} close-time adapter journal",
        )
        _require_optional_descriptor_match(
            video_descriptor, failure_declaration.get("source_video"),
            f"{cell_id} close-time source video",
        )
        if status == "valid_censored":
            assert context_reset is not None
            require(
                source_failure_path is not None
                and isinstance(failure_declaration.get("failure_value"), Mapping),
                f"{cell_id} safety_abort lacks its validated failure receipt",
            )
            context_terminal = _validate_censored_context_terminal(
                model=model,
                cell_id=cell_id,
                layout=layout,
                condition_index=condition_index,
                condition=condition,
                schedule=schedule,
                failure_path=source_failure_path,
                failure_value=failure_declaration["failure_value"],
                study_commit=study_commit,
                d1_runtime=d1_runtime,
                completion=completion,
                context_reset=context_reset,
                terminal_descriptor=failure_declaration.get("context_terminal"),
                terminal_path=failure_declaration.get("context_terminal_path"),
                request_descriptors=recorded["request_descriptors"],
                request_values=recorded["request_values"],
                raw_root=raw_root,
            )
    return {
        "source_artifact": dict(source_descriptor),
        "source_cell": cell,
        "source_failure": None if source_failure_path is None else str(source_failure_path),
        "completion_descriptor": completion_descriptor,
        "completion_path": completion_path,
        "journal_descriptor": journal_descriptor,
        "journal_path": journal_path,
        "completion": completion,
        "rows": rows,
        "recording_id": identity["attempt_id"],
        "video_descriptor": video_descriptor,
        "video_path": video_path,
        "source_video_id": "video_" + str(source_descriptor["sha256"])[:24],
        "source_identity": source_identity,
        "context_terminal": context_terminal,
        **recorded,
    }


def _compile_technical_cell(
    *,
    model: str,
    layout: str,
    condition: str,
    condition_index: int,
    cell_id: str,
    action_count: int,
    request_count: int,
    failure_descriptor: Mapping[str, Any] | None,
    failure_path: Path | None,
    cell_root: Path,
    failure_declaration: Mapping[str, Any],
    study_commit: str,
    raw_root: Path,
    camera_id: str,
    schedule: Any,
    expected_pose_sha256: str,
    d1_runtime: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    failure: Mapping[str, Any] | None = None
    if failure_path is not None:
        require(failure_path.parent == cell_root.resolve(),
                f"{cell_id} failure receipt is outside its canonical cell")
        failure = load_json(failure_path, f"{cell_id} technical failure")
        require(
            failure == failure_declaration.get("failure_value"),
            f"{cell_id} failure bytes differ from their close-time validation",
        )
        try:
            if model == "N3":
                replayed_failure = n3_confirmation.validate_failed_confirmation_cell(
                    failure_path, condition_index=condition_index, block=schedule
                )
            else:
                require(
                    d1_runtime is not None,
                    f"{cell_id} D1 failure lacks its selected queue lifecycle",
                )
                with d1_confirmation._configured_for_validation(
                    schedule, d1_runtime["simulator_worker_role"]
                ):
                    replayed_failure = d1_confirmation.validate_failed_confirmation_cell(
                        failure_path, condition_index=condition_index, block=schedule
                    )
                require(
                    failure.get("study_commit") == d1_runtime["study_commit"]
                    and failure.get("run_id") == d1_runtime["run_id"]
                    and failure.get("server_job_id") == d1_runtime["server_job_id"]
                    and failure.get("simulator_job_id")
                    == d1_runtime["simulator_job_id"],
                    f"{cell_id} D1 failure is detached from its selected queue pair",
                )
        except ConfirmationCompilerError:
            raise
        except Exception as error:
            raise ConfirmationCompilerError(
                f"{cell_id} failed repeated confirmation-failure validation: {error}"
            ) from error
        require(
            replayed_failure == failure,
            f"{cell_id} repeated confirmation-failure validator changed source bytes",
        )
    source_hash = (
        str(failure_descriptor["sha256"])
        if failure_descriptor is not None
        else str(failure_declaration["declaration_sha256"])
    )
    completion_descriptor, completion_path, journal_descriptor, journal_path = _completion_and_journal(
        cell_root=cell_root, cell=None, raw_root=raw_root,
        required=action_count > 0 or request_count > 0,
    )
    video_descriptor, video_path = _video_descriptor(
        cell_root=cell_root, cell=None, raw_root=raw_root, required=False
    )
    if completion_path is None:
        require(action_count == 0 and request_count == 0,
                f"{cell_id} lacks native evidence for executed actions or requests")
        _require_optional_descriptor_match(
            None, failure_declaration.get("adapter_completion"),
            f"{cell_id} close-time adapter completion",
        )
        # A startup-killed child can leave a partial journal but no completion.
        # It is retained as source provenance, never parsed into action evidence.
        _require_optional_descriptor_match(
            video_descriptor, failure_declaration.get("source_video"),
            f"{cell_id} close-time source video",
        )
        return {
            "source_artifact": None if failure_descriptor is None else dict(failure_descriptor),
            "completion_descriptor": None,
            "completion_path": None,
            "journal_descriptor": failure_declaration.get("adapter_journal"),
            "journal_path": failure_declaration.get("journal_path"),
            "completion": None,
            "rows": [],
            "recording_id": "technical_" + source_hash[:24],
            "video_descriptor": video_descriptor,
            "video_path": video_path,
            "source_video_id": (
                None if video_descriptor is None else "video_" + source_hash[:24]
            ),
            "source_identity": None,
            "context_terminal": None,
            "request_descriptors": [],
            "request_values": [],
            "response_descriptors": [],
            "packed_request_bindings": [],
            "observations": [],
            "actions": [],
            "technical_request_evidence": [],
        }
    assert completion_descriptor is not None and journal_descriptor is not None and journal_path is not None
    expected_prompt, expected_seed = _expected_cell_identity(model, condition, schedule)
    completion, rows, identity = _validate_completion(
        completion_path=completion_path,
        journal_path=journal_path,
        expected_cell_id=cell_id,
        model=model,
        layout=layout,
        condition=condition,
        status="technical_invalid",
        expected_actions=action_count,
        expected_prompt=expected_prompt,
        expected_effective_seed=expected_seed,
    )
    source_identity = _source_identity_check(
        completion=completion, model=model, study_commit=study_commit, cell_id=cell_id,
        expected_pose_sha256=expected_pose_sha256,
    )
    try:
        retained = _technical_retained_evidence(
            completion=completion,
            completion_path=completion_path,
            rows=rows,
            cell={
                **dict(failure or {}),
                "cell_id": cell_id,
                "prompt": expected_prompt,
                "effective_seed": expected_seed,
            },
            model=model,
            camera_id=camera_id,
            raw_root=raw_root,
        )
    except ConfirmationCompilerError:
        raise
    except Exception as error:
        raise ConfirmationCompilerError(
            f"{cell_id} failed retained technical-prefix replay: {error}"
        ) from error
    _require_optional_descriptor_match(
        completion_descriptor, failure_declaration.get("adapter_completion"),
        f"{cell_id} close-time adapter completion",
    )
    _require_optional_descriptor_match(
        journal_descriptor, failure_declaration.get("adapter_journal"),
        f"{cell_id} close-time adapter journal",
    )
    _require_optional_descriptor_match(
        video_descriptor, failure_declaration.get("source_video"),
        f"{cell_id} close-time source video",
    )
    return {
        "source_artifact": None if failure_descriptor is None else dict(failure_descriptor),
        "completion_descriptor": completion_descriptor,
        "completion_path": completion_path,
        "journal_descriptor": journal_descriptor,
        "journal_path": journal_path,
        "completion": completion,
        "rows": rows,
        "recording_id": identity["attempt_id"],
        "video_descriptor": video_descriptor,
        "video_path": video_path,
        "source_video_id": (
            None if video_descriptor is None else "video_" + source_hash[:24]
        ),
        "source_identity": source_identity,
        "context_terminal": failure_declaration.get("context_terminal"),
        **retained,
    }


def _derive_endpoint(
    *,
    compiled: Mapping[str, Any],
    roster: Mapping[str, Any],
    recording_receipt_sha256: str,
    action_manifest_sha256: str,
) -> dict[str, Any]:
    source_completion = _reference(compiled["completion_descriptor"])
    source_journal = _reference(compiled["journal_descriptor"])
    seed = {
        "source_adapter_completion": source_completion,
        "source_adapter_journal": source_journal,
    }
    try:
        adapter = analyzer._read_adapter_source(seed, receipt_path=Path.cwd() / "endpoint.json", roster=roster)
        observations = analyzer._observation_events(adapter)
        adapter["observations"] = observations
        command = str(roster["condition_id"]).split("_", 1)[1]
        first = analyzer._derive_first_success(adapter, command)
        first_index = None if first is None else first["action_step"]
        zero = analyzer._point_from_observation(adapter, observations, "obs_000000")
        if roster["recording_status"] == "valid_complete":
            action_450 = analyzer._point_from_observation(adapter, observations, "obs_000450")
        else:
            action_450 = None
        if first_index is not None:
            selected_id = f"obs_{first_index:06d}"
            selected = analyzer._point_from_observation(adapter, observations, selected_id)
        elif roster["recording_status"] == "valid_complete":
            selected_id = "obs_000450"
            selected = action_450
        else:
            selected_id = None
            selected = None
    except Exception as error:
        raise ConfirmationCompilerError(f"{roster['cell_id']} endpoint replay failed: {error}") from error
    endpoint = sign_document({
        "schema_version": ENDPOINT_SCHEMA,
        "study_id": STUDY_ID,
        "stage": "confirmation",
        "cell_id": roster["cell_id"],
        "recording_id": roster["recording_id"],
        "model_id": roster["model_id"],
        "recording_status": roster["recording_status"],
        "recording_receipt_sha256": recording_receipt_sha256,
        "action_manifest_sha256": action_manifest_sha256,
        "executed_action_count": roster["executed_action_count"],
        "first_success_action_index": first_index,
        "action_zero": zero,
        "action_450": action_450,
        "first_success_or_action_450": selected,
        "action_zero_observation_id": "obs_000000",
        "action_450_observation_id": (
            "obs_000450" if roster["recording_status"] == "valid_complete" else None
        ),
        "first_success_or_action_450_observation_id": selected_id,
        "source_adapter_completion": source_completion,
        "source_adapter_journal": source_journal,
    })
    return endpoint


def _derive_history(
    *, request: Mapping[str, Any], compiled: Mapping[str, Any]
) -> dict[str, Any]:
    execution = compiled["completion"]["request_execution"][request["request_index"]]
    preceding_id = execution["preceding_observation_id"]
    current_id = execution["current_observation_id"]
    observations = {row["observation_id"]: row for row in compiled["observations"]}
    require(isinstance(preceding_id, str) and isinstance(current_id, str),
            f"{request['source_request_id']} has no real preceding observation")
    preceding = observations[preceding_id]
    current = observations[current_id]
    interval = (
        int(current["camera_capture_time_ns"]) - int(preceding["camera_capture_time_ns"])
    ) / 1_000_000_000
    return sign_document({
        "schema_version": HISTORY_SCHEMA,
        "study_id": STUDY_ID,
        "stage": "confirmation",
        "source_request_id": request["source_request_id"],
        "cell_id": request["cell_id"],
        "alignment_receipt_id": request["alignment_receipt_id"],
        "alignment_receipt_sha256": request["alignment_receipt_sha256"],
        "camera_id": request["camera_id"],
        "preceding_observation_id": preceding_id,
        "current_observation_id": current_id,
        "preceding_camera_capture_time_ns": preceding["camera_capture_time_ns"],
        "current_camera_capture_time_ns": current["camera_capture_time_ns"],
        "preceding_physics_time_s": preceding["physics_time_s"],
        "current_physics_time_s": current["physics_time_s"],
        "preceding_observation_interval_s": interval,
        "source_adapter_completion": _reference(compiled["completion_descriptor"]),
        "source_adapter_journal": _reference(compiled["journal_descriptor"]),
    })


def _write_atomic_directory_validated(
    target: Path,
    files: Mapping[str, bytes],
    validator: Callable[[Path], None],
) -> None:
    target = Path(target)
    require(not target.exists(), f"refusing to overwrite compiler output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for relative, payload in sorted(files.items()):
            path = temporary / relative
            require(path.resolve().is_relative_to(temporary.resolve()), "compiler output path escaped")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        validator(temporary)
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _validate_compiled_bundle(
    bundle: Path,
    *,
    active_seal_producer: Mapping[str, Any] | None = None,
) -> None:
    compiler_receipt_path = bundle / "compiler_receipt.json"
    compiler_receipt = load_json(compiler_receipt_path, "compiled compiler receipt")
    require(compiler_receipt.get("schema_version") == COMPILER_SCHEMA,
            "compiled compiler receipt schema changed")
    verify_signed(compiler_receipt, "compiled compiler receipt")
    _validate_compiler_dependencies(compiler_receipt)
    outer_state = compiler_receipt.get("cohort_close_outer_result_state")
    require(
        outer_state in {"pending_current_wrapper_exit", "validated_succeeded_reaped"},
        "compiled cohort-close outer-result state is invalid",
    )
    manifest_descriptor, manifest_path = _exact_file_reference(
        compiler_receipt.get("input_manifest"), base=bundle,
        label="compiled input manifest",
    )
    manifest = load_json(manifest_path, "compiled input manifest")
    _exact_keys(manifest, INPUT_KEYS, "compiled input manifest")
    verify_signed(manifest, "compiled input manifest")
    require(
        manifest_descriptor == compiler_receipt.get("input_manifest")
        and manifest.get("cohort_close_receipt")
        == compiler_receipt.get("cohort_close_receipt"),
        "compiled input/cohort-close lineage changed",
    )
    replayed_descriptor, _close_path, replayed_close = _load_cohort_close(
        manifest.get("cohort_close_receipt"),
        manifest_base=manifest_path.parent,
        branch=str(compiler_receipt.get("cohort_branch")),
        study_commit=str(compiler_receipt.get("study_commit")),
        release_descriptor=compiler_receipt.get("confirmation_release_freeze"),
        source_root=Path(str(compiler_receipt.get("source_root"))),
        active_seal_producer=active_seal_producer,
    )
    require(
        replayed_descriptor == compiler_receipt.get("cohort_close_receipt"),
        "compiled receipt is detached from the replayed cohort close",
    )
    replayed_state = replayed_close.get("_producer_outer_result_state")
    if active_seal_producer is not None:
        require(
            outer_state == "pending_current_wrapper_exit"
            and replayed_state == "pending_current_wrapper_exit",
            "active compiler did not replay its exact pending sealer context",
        )
    else:
        require(
            replayed_state == "validated_succeeded_reaped",
            "compiled cohort close still lacks a successful reaped outer result",
        )
        producer = replayed_close.get("producer_queue_job")
        require(isinstance(producer, Mapping),
                "replayed cohort close lacks its validated seal producer")
        seal_receipt_descriptor, seal_receipt_path = _exact_file_reference(
            producer.get("seal_job_receipt"),
            base=_close_path.parent,
            label="replayed cohort-close seal job receipt",
        )
        seal_receipt = load_json(
            seal_receipt_path, "replayed cohort-close seal job receipt"
        )
        verify_signed(seal_receipt, "replayed cohort-close seal job receipt")
        require(
            seal_receipt.get("compiler_receipt")
            == development.file_descriptor(compiler_receipt_path)
            and seal_receipt_descriptor
            == producer.get("seal_job_receipt"),
            "compiled receipt is not the exact output authenticated by the sealer job",
        )
    outputs = compiler_receipt.get("outputs")
    require(isinstance(outputs, Mapping), "compiled compiler receipt lacks outputs")

    def output_file(key: str) -> tuple[Path, dict[str, Any]]:
        value = outputs.get(key)
        require(isinstance(value, Mapping), f"compiled output {key} descriptor is missing")
        path = bundle / str(value.get("path"))
        require(path.resolve().is_relative_to(bundle.resolve())
                and path.is_file()
                and path.stat().st_size == value.get("bytes")
                and sha256_file(path) == value.get("sha256"),
                f"compiled output {key} is missing or changed")
        return path, dict(value)

    inventory_path = bundle / "request_inventory.json"
    inventory = load_json(inventory_path, "compiled request inventory")
    declared_inventory_path, inventory_descriptor = output_file("request_inventory")
    require(declared_inventory_path == inventory_path
            and inventory_descriptor["sha256"] == sha256_file(inventory_path),
            "compiled request inventory descriptor changed")
    rows = inventory.get("requests")
    require(isinstance(rows, list), "compiled request inventory requests are invalid")
    try:
        annotation._validate_episode_roster(
            inventory.get("episode_roster"),
            stage="confirmation",
            cohort_branch=inventory.get("cohort_branch"),
            base=inventory_path.parent,
        )
        annotation._validate_alignment_contracts(
            inventory.get("alignment_contracts"),
            stage="confirmation",
            cohort_branch=inventory.get("cohort_branch"),
        )
    except Exception as error:
        raise ConfirmationCompilerError(
            f"compiled roster/alignment inventory failed consumer replay: {error}"
        ) from error
    if rows:
        try:
            annotation.select_requests(
                inventory,
                inventory_sha256=sha256_file(inventory_path),
                inventory_path=inventory_path,
            )
        except Exception as error:
            raise ConfirmationCompilerError(f"compiled request inventory failed consumer replay: {error}") from error

    wrapper_path, _ = output_file("request_inventory_receipt")
    wrapper = load_json(wrapper_path, "compiled request inventory receipt")
    require(wrapper.get("schema_version") == REQUEST_INVENTORY_RECEIPT_SCHEMA
            and wrapper.get("request_inventory") == inventory_descriptor,
            "compiled request inventory signature wrapper changed")
    verify_signed(wrapper, "compiled request inventory receipt")
    provenance_path, _ = output_file("request_provenance")
    provenance = load_json(provenance_path, "compiled request provenance")
    require(provenance.get("schema_version") == PROVENANCE_SCHEMA
            and provenance.get("request_inventory_sha256") == inventory_descriptor["sha256"],
            "compiled request provenance binding changed")
    verify_signed(provenance, "compiled request provenance")
    require(
        [row.get("source_request_id") for row in provenance.get("requests", [])]
        == [row.get("source_request_id") for row in rows],
        "compiled request provenance order/coverage differs from inventory",
    )
    video_path, _ = output_file("private_video_inventory")
    videos = load_json(video_path, "compiled private video inventory")
    require(videos.get("schema_version") == PRIVATE_VIDEO_SCHEMA
            and videos.get("visibility") == "private_source_evidence_not_blind_annotation_media",
            "compiled private video inventory boundary changed")
    verify_signed(videos, "compiled private video inventory")
    for row in videos.get("videos", []):
        require(isinstance(row, Mapping) and row.get("private_not_for_blind_raters") is True,
                "compiled private video row is invalid")
        source = row.get("source_video")
        if source is not None:
            descriptor, _ = _descriptor(
                source, base=video_path.parent,
                label=f"private video {row.get('cell_id')}",
            )
            require(descriptor == source, "private video descriptor changed")
    zero_path, _ = output_file("zero_science_receipt")
    zero = load_json(zero_path, "compiled zero-science receipt")
    require(zero.get("schema_version") == ZERO_SCIENCE_SCHEMA,
            "compiled zero-science receipt schema changed")
    verify_signed(zero, "compiled zero-science receipt")
    require(all(
        item == 0 for key, item in zero.items()
        if key not in {"schema_version", "study_id", "stage", "payload_sha256"}
    ), "compiled zero-science receipt contains activity")
    roster = {row["cell_id"]: row for row in inventory["episode_roster"]}
    endpoint_adapters = {}
    for path in sorted((bundle / "cells").glob("*/*/endpoint_trace_receipt.json")):
        receipt = load_json(path, "compiled endpoint receipt")
        cell_id = receipt.get("cell_id")
        require(cell_id in roster and cell_id not in endpoint_adapters,
                f"compiled endpoint identity is unknown or duplicated: {cell_id}")
        try:
            endpoint_adapters[cell_id] = analyzer.validate_endpoint_receipt(
                receipt, receipt_path=path, roster=roster[cell_id]
            )["adapter_source"]
        except Exception as error:
            raise ConfirmationCompilerError(f"compiled endpoint failed analyzer replay: {error}") from error
    expected = {
        cell_id for cell_id, row in roster.items()
        if row["recording_status"] in {"valid_complete", "valid_censored"}
    }
    require(set(endpoint_adapters) == expected,
            "compiled endpoint receipts do not exactly cover valid/censored cells")
    requests = {row["source_request_id"]: row for row in rows}
    seen_histories = set()
    for path in sorted((bundle / "histories").glob("*.json")):
        receipt = load_json(path, "compiled history receipt")
        request_id = receipt.get("source_request_id")
        require(request_id in requests and request_id not in seen_histories,
                f"compiled history references an unknown or duplicate request: {request_id}")
        seen_histories.add(request_id)
        request = requests[request_id]
        try:
            analyzer.validate_history_receipt(
                receipt,
                receipt_path=path,
                request=request,
                adapter=endpoint_adapters[request["cell_id"]],
            )
        except Exception as error:
            raise ConfirmationCompilerError(f"compiled history failed analyzer replay: {error}") from error
    expected_histories = {
        request_id for request_id, request in requests.items()
        if request["history_mode"] == "preceding_observation"
    }
    require(seen_histories == expected_histories,
            "compiled history receipts do not exactly cover real preceding-observation requests")


def compile_manifest(
    manifest_path: Path,
    manifest_sha256: str,
    output_dir: Path,
    *,
    active_seal_producer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    supplied = Path(manifest_path)
    development._reject_symlink_components(supplied, "confirmation compiler input")
    manifest_path = supplied.resolve()
    require(manifest_path.is_file(), "confirmation compiler input is missing")
    require(isinstance(manifest_sha256, str) and SHA256_RE.fullmatch(manifest_sha256) is not None,
            "confirmation compiler input SHA-256 is invalid")
    require(sha256_file(manifest_path) == manifest_sha256,
            "confirmation compiler input file hash changed")
    manifest = load_json(manifest_path, "confirmation compiler input")
    _exact_keys(manifest, INPUT_KEYS, "confirmation compiler input")
    require(manifest.get("schema_version") == INPUT_SCHEMA,
            "confirmation compiler input schema changed")
    require(manifest.get("study_id") == STUDY_ID,
            "confirmation compiler input study changed")
    verify_signed(manifest, "confirmation compiler input")
    branch = manifest.get("cohort_branch")
    require(branch in MODELS_BY_BRANCH, "confirmation compiler branch is invalid")
    models = MODELS_BY_BRANCH[str(branch)]
    _validate_rfc3339_utc(manifest.get("inventory_finalized_at"), "inventory_finalized_at")
    camera_id = manifest.get("camera_id")
    require(camera_id in development.PRIMARY_CAMERA_CHOICES,
            "confirmation compiler camera is not an original recorded camera")
    study_commit = manifest.get("study_commit")
    require(isinstance(study_commit, str) and COMMIT_RE.fullmatch(study_commit) is not None,
            "confirmation compiler study commit is invalid")
    source_root_value = manifest.get("source_root")
    raw_root_value = manifest.get("raw_root")
    require(isinstance(source_root_value, str) and Path(source_root_value).is_absolute(),
            "confirmation compiler source root must be absolute")
    require(isinstance(raw_root_value, str) and Path(raw_root_value).is_absolute(),
            "confirmation compiler raw root must be absolute")
    development._reject_symlink_components(Path(source_root_value), "confirmation source root")
    development._reject_symlink_components(Path(raw_root_value), "confirmation raw root")
    source_root = Path(source_root_value).resolve()
    raw_root = Path(raw_root_value).resolve()
    require(source_root.is_dir() and raw_root.is_dir(),
            "confirmation compiler source/raw root is unavailable")
    _verify_staged_source(source_root, study_commit)
    output_dir = Path(output_dir)
    require(not output_dir.exists(), f"refusing to overwrite compiler output: {output_dir}")
    require(development._under(output_dir.parent, raw_root, "confirmation compiler output parent").is_dir(),
            "confirmation compiler output parent is unavailable")
    release_descriptor, release_path, release_freeze = _validate_release_freeze(
        manifest.get("development_release_freeze"),
        manifest_base=manifest_path.parent,
        branch=str(branch),
    )
    close_descriptor, close_path, cohort_close = _load_cohort_close(
        manifest.get("cohort_close_receipt"),
        manifest_base=manifest_path.parent,
        branch=str(branch),
        study_commit=study_commit,
        release_descriptor=release_descriptor,
        source_root=source_root,
        active_seal_producer=active_seal_producer,
    )
    require(
        Path(str(cohort_close.get("raw_root"))).resolve() == raw_root,
        "confirmation manifest raw root differs from the authoritative cohort close",
    )
    require(
        manifest.get("inventory_finalized_at") == cohort_close.get("sealed_at_utc"),
        "inventory_finalized_at must equal the authoritative cohort-close time",
    )
    alignments = _alignment_objects(
        release_path=release_path,
        release_freeze=release_freeze,
        models=models,
    )
    raw_blocks = manifest.get("block_receipts")
    require(isinstance(raw_blocks, list), "confirmation block receipt inventory must be a list")
    expected_keys = {(model, layout) for model in models for layout in LAYOUTS}
    blocks: dict[tuple[str, str], tuple[dict[str, Any], Path, str]] = {}
    raw_block_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_blocks):
        require(isinstance(raw, Mapping), f"confirmation block input {index} must be an object")
        _exact_keys(raw, BLOCK_INPUT_KEYS, f"confirmation block input {index}")
        key = (raw.get("model_id"), raw.get("layout_pair_id"))
        require(key in expected_keys and key not in blocks,
                f"confirmation block input {index} identity is invalid or duplicated")
        descriptor, path = _descriptor(
            raw.get("receipt"),
            base=manifest_path.parent,
            label=f"{key[0]} {key[1]} terminal block receipt",
        )
        evidence_form = raw.get("evidence_form")
        require(evidence_form in {FULL_AGGREGATE_FORM, D1_ZERO_LAUNCH_FORM},
                f"confirmation block input {index} evidence form is invalid")
        blocks[(str(key[0]), str(key[1]))] = (descriptor, path, str(evidence_form))
        raw_block_rows.append(dict(raw))
    require(set(blocks) == expected_keys,
            "confirmation terminal receipts do not exactly cover the cohort branch")

    planned = _planned_cells(str(branch))
    aggregate_results: dict[tuple[str, str], dict[str, Any]] = {}
    for model in models:
        for layout in LAYOUTS:
            try:
                schedule = (
                    n3_confirmation.load_confirmation_block(source_root, layout)
                    if model == "N3"
                    else d1_confirmation.load_confirmation_block(source_root, layout)
                )
            except Exception as error:
                raise ConfirmationCompilerError(f"{model} {layout} schedule validation failed: {error}") from error
            descriptor, path, evidence_form = blocks[(model, layout)]
            aggregate_results[(model, layout)] = _validate_aggregate(
                descriptor=descriptor,
                path=path,
                model=model,
                layout=layout,
                schedule=schedule,
                study_commit=study_commit,
                source_root=source_root,
                raw_root=raw_root,
                release_descriptor=release_descriptor,
                release_freeze=release_freeze,
                evidence_form=evidence_form,
            ) | {"schedule": schedule}

    close_blocks = _validate_cohort_close_blocks(
        close=cohort_close,
        close_path=close_path,
        raw_blocks=raw_block_rows,
        aggregates=aggregate_results,
        study_commit=study_commit,
        source_root=source_root,
        raw_root=raw_root,
        release_descriptor=release_descriptor,
        release_freeze=release_freeze,
    )
    for key, closure in close_blocks.items():
        aggregate_results[key].update(closure)

    files: dict[str, bytes] = {}
    roster: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    endpoint_descriptors: list[dict[str, Any]] = []
    history_descriptors: list[dict[str, Any]] = []
    status_counts = {status: 0 for status in STATUS_VALUES}
    observed_actions = 0
    observed_source_requests = 0
    seen_recording_ids: set[str] = set()
    seen_request_ids: set[str] = set()
    source_artifact_rows: list[dict[str, Any]] = []

    for model in models:
        for layout in LAYOUTS:
            block = aggregate_results[(model, layout)]
            schedule = block["schedule"]
            counts = block["counts"]
            if model == "D1":
                pair = block.get("d1_pair")
                queue_jobs = block.get("queue_jobs")
                require(
                    isinstance(pair, Mapping)
                    and isinstance(queue_jobs, Mapping)
                    and pair.get("simulator_job_id") in queue_jobs,
                    f"D1 {layout} compiled block lacks its selected queue pair",
                )
                simulator_job = queue_jobs[pair["simulator_job_id"]]
                simulator_role = simulator_job.get("descriptor_value", {}).get("role")
                require(
                    isinstance(simulator_role, str)
                    and SAFE_COMPONENT_RE.fullmatch(simulator_role) is not None,
                    f"D1 {layout} selected simulator role is invalid",
                )
                d1_runtime_binding = {
                    "study_commit": study_commit,
                    "run_id": str(pair["run_id"]),
                    "server_job_id": str(pair["server_job_id"]),
                    "simulator_job_id": str(pair["simulator_job_id"]),
                    "simulator_worker_role": simulator_role,
                }
            else:
                d1_runtime_binding = None
            completed_count = counts["completed_valid_behavioral_cells"]
            launched_count = counts["launched_behavioral_cells"]
            for index, cell_id in enumerate(schedule.cell_ids):
                condition = schedule.condition_order[index].replace("-", "_")
                require(planned.get(cell_id) == (model, layout, condition),
                        f"{cell_id} schedule/planned roster binding changed")
                if index < completed_count:
                    status = "valid_complete"
                    action_count = 450
                    source_request_count = MODEL_FULL_REQUESTS[model]
                    source_descriptor = block["cell_descriptors"][index]
                    source_cell_path = block["cell_paths"][index]
                    failure_descriptor = None
                    failure_path = None
                elif index < launched_count:
                    source_cell_path = None
                    failure = block["failures_by_index"][index]
                    status = (
                        "valid_censored"
                        if failure["status"] == "safety_abort"
                        else "technical_invalid"
                    )
                    action_count = failure["actions_executed"]
                    source_request_count = failure["request_count"]
                    failure_descriptor = failure["failure_receipt"]
                    failure_path = failure["failure_path"]
                    source_descriptor = failure_descriptor
                else:
                    status = "not_run"
                    action_count = 0
                    source_request_count = 0
                    source_descriptor = None
                    source_cell_path = None
                    failure_descriptor = None
                    failure_path = None
                status_counts[status] += 1
                if status == "not_run":
                    roster.append({
                        "cell_id": cell_id,
                        "recording_id": None,
                        "model_id": model,
                        "layout_pair_id": layout,
                        "condition_id": condition,
                        "recording_status": status,
                        "executed_action_count": None,
                        "censor_reason": None,
                        "recording_receipt_path": None,
                        "recording_receipt_sha256": None,
                        "action_manifest_path": None,
                        "action_manifest_sha256": None,
                        "source_video_id": None,
                        "source_video_sha256": None,
                    })
                    continue
                if status in {"valid_complete", "valid_censored"}:
                    selected_fixture = block.get("selected_fixture")
                    require(isinstance(selected_fixture, Mapping),
                            f"{model} {layout} recorded cell lacks selected fixture evidence")
                    expected_pose = selected_fixture.get("pose_manifest", {}).get("sha256")
                    require(isinstance(expected_pose, str) and SHA256_RE.fullmatch(expected_pose) is not None,
                            f"{model} {layout} selected fixture pose hash is invalid")
                    compiled = _compile_recorded_cell(
                        model=model,
                        layout=layout,
                        condition=condition,
                        condition_index=index,
                        cell_id=cell_id,
                        status=status,
                        action_count=action_count,
                        source_descriptor=source_descriptor,
                        source_cell_path=source_cell_path,
                        source_failure_path=failure_path,
                        source_root=source_root,
                        study_commit=study_commit,
                        raw_root=raw_root,
                        camera_id=camera_id,
                        schedule=schedule,
                        alignment_bundle=alignments[model],
                        expected_pose_sha256=expected_pose,
                        failure_declaration=(
                            None if index < completed_count
                            else block["failures_by_index"][index]
                        ),
                        d1_runtime=d1_runtime_binding,
                    )
                    censor_reason = None if status == "valid_complete" else "safety_abort"
                else:
                    selected_fixture = block.get("selected_fixture")
                    require(isinstance(selected_fixture, Mapping),
                            f"{model} {layout} technical cell lacks selected fixture evidence")
                    expected_pose = selected_fixture.get("pose_manifest", {}).get("sha256")
                    require(isinstance(expected_pose, str) and SHA256_RE.fullmatch(expected_pose) is not None,
                            f"{model} {layout} selected fixture pose hash is invalid")
                    compiled = _compile_technical_cell(
                        model=model,
                        layout=layout,
                        condition=condition,
                        condition_index=index,
                        cell_id=cell_id,
                        action_count=action_count,
                        request_count=source_request_count,
                        failure_descriptor=failure_descriptor,
                        failure_path=failure_path,
                        cell_root=_cell_root(block["raw_attempt"], index, cell_id),
                        failure_declaration=block["failures_by_index"][index],
                        study_commit=study_commit,
                        raw_root=raw_root,
                        camera_id=camera_id,
                        schedule=schedule,
                        expected_pose_sha256=expected_pose,
                        d1_runtime=d1_runtime_binding,
                    )
                    censor_reason = "recording_integrity_failure"
                source_artifact_rows.append({
                    "cell_id": cell_id,
                    "model_id": model,
                    "layout_pair_id": layout,
                    "condition_id": condition,
                    "recording_status": status,
                    "source_artifact": source_descriptor,
                    "failure_declaration_sha256": (
                        None if index < completed_count
                        else block["failures_by_index"][index]["declaration_sha256"]
                    ),
                    "adapter_completion": compiled["completion_descriptor"],
                    "adapter_journal": compiled["journal_descriptor"],
                    "source_video": compiled["video_descriptor"],
                    "server_context_terminal": compiled.get("context_terminal"),
                    "technical_request_evidence": compiled.get(
                        "technical_request_evidence"
                    ),
                })
                recording_id = compiled["recording_id"]
                require(recording_id not in seen_recording_ids,
                        f"recording identity is duplicated: {recording_id}")
                seen_recording_ids.add(recording_id)
                observed_actions += action_count
                if compiled["completion"] is not None:
                    require(compiled["completion"]["request_count"] == source_request_count,
                            f"{cell_id} failure/completion request counts differ")
                    observed_source_requests += source_request_count
                else:
                    require(source_request_count == 0,
                            f"{cell_id} lacks native evidence for observed source requests")
                cell_rel = Path("cells") / model.lower() / _safe_component(cell_id)
                action_rel = cell_rel / "action_manifest.json"
                recording_rel = cell_rel / "recording_receipt.json"
                endpoint_rel = cell_rel / "endpoint_trace_receipt.json"
                action_manifest = sign_document({
                    "schema_version": ACTION_MANIFEST_SCHEMA,
                    "study_id": STUDY_ID,
                    "cell_id": cell_id,
                    "recording_id": recording_id,
                    "model_id": model,
                    "executed_action_count": action_count,
                    "actions": compiled["actions"],
                })
                action_payload = pretty_bytes(action_manifest)
                action_descriptor = _bytes_descriptor(str(action_rel), action_payload)
                source_video_id = compiled["source_video_id"]
                video_descriptor = compiled["video_descriptor"]
                recording_source_hash = (
                    str(source_descriptor["sha256"])
                    if source_descriptor is not None
                    else str(block["failures_by_index"][index]["declaration_sha256"])
                )
                recording_unsigned = {
                    "schema_version": RECORDING_RECEIPT_SCHEMA,
                    "study_id": STUDY_ID,
                    "receipt_id": "recording_" + recording_source_hash[:24],
                    "stage": "confirmation",
                    "cell_id": cell_id,
                    "recording_id": recording_id,
                    "model_id": model,
                    "layout_pair_id": layout,
                    "condition_id": condition,
                    "recording_status": status,
                    "executed_action_count": action_count,
                    "censor_reason": censor_reason,
                    "source_video_id": source_video_id,
                    "source_video_sha256": None if video_descriptor is None else video_descriptor["sha256"],
                    "action_manifest_path": "action_manifest.json",
                    "action_manifest_sha256": action_descriptor["sha256"],
                }
                recording_receipt = sign_document(recording_unsigned)
                recording_payload = pretty_bytes(recording_receipt)
                recording_descriptor = _bytes_descriptor(str(recording_rel), recording_payload)
                files[str(action_rel)] = action_payload
                files[str(recording_rel)] = recording_payload
                roster_row = {
                    "cell_id": cell_id,
                    "recording_id": recording_id,
                    "model_id": model,
                    "layout_pair_id": layout,
                    "condition_id": condition,
                    "recording_status": status,
                    "executed_action_count": action_count,
                    "censor_reason": censor_reason,
                    "recording_receipt_path": recording_descriptor["path"],
                    "recording_receipt_sha256": recording_descriptor["sha256"],
                    "action_manifest_path": action_descriptor["path"],
                    "action_manifest_sha256": action_descriptor["sha256"],
                    "source_video_id": source_video_id,
                    "source_video_sha256": None if video_descriptor is None else video_descriptor["sha256"],
                }
                roster.append(roster_row)
                video_rows.append({
                    "cell_id": cell_id,
                    "recording_id": recording_id,
                    "model_id": model,
                    "layout_pair_id": layout,
                    "condition_id": condition,
                    "recording_status": status,
                    "private_not_for_blind_raters": True,
                    "source_video_id": source_video_id,
                    "source_video": video_descriptor,
                })
                if status in {"valid_complete", "valid_censored"}:
                    observation_map = {row["observation_id"]: row for row in compiled["observations"]}
                    for request_index, request_descriptor in enumerate(compiled["request_descriptors"]):
                        execution = compiled["completion"]["request_execution"][request_index]
                        request = _request_inventory_row(
                            model=model,
                            cell_id=cell_id,
                            layout=layout,
                            condition=condition,
                            recording_id=recording_id,
                            source_video_id=source_video_id,
                            video_sha256=video_descriptor["sha256"],
                            action_sha256=action_descriptor["sha256"],
                            request_index=request_index,
                            execution=execution,
                            request_descriptor=request_descriptor,
                            observations=observation_map,
                            alignment_bundle=alignments[model],
                        )
                        require(request["source_request_id"] not in seen_request_ids,
                                f"source request identity is duplicated: {request['source_request_id']}")
                        seen_request_ids.add(request["source_request_id"])
                        request_rows.append(request)
                        provenance_rows.append({
                            "source_request_id": request["source_request_id"],
                            "cell_id": cell_id,
                            "recording_id": recording_id,
                            "model_id": model,
                            "layout_pair_id": layout,
                            "condition_id": condition,
                            "request_index": request_index,
                            "action_step_start": execution["action_step_start"],
                            "executed_prefix_actions": execution["executed_actions"],
                            "current_observation": observation_map[execution["current_observation_id"]],
                            "preceding_observation": (
                                None if execution["preceding_observation_id"] is None
                                else observation_map[execution["preceding_observation_id"]]
                            ),
                            "official_request_receipt": request_descriptor,
                            "recorder_model_request": compiled["packed_request_bindings"][request_index],
                            "recorder_response": compiled["response_descriptors"][request_index],
                            "adapter_completion": compiled["completion_descriptor"],
                            "adapter_journal": compiled["journal_descriptor"],
                            "source_video": video_descriptor,
                            "model_output_or_action_modified": False,
                            "source_identity": compiled["source_identity"],
                        })
                    endpoint = _derive_endpoint(
                        compiled=compiled,
                        roster=roster_row,
                        recording_receipt_sha256=recording_descriptor["sha256"],
                        action_manifest_sha256=action_descriptor["sha256"],
                    )
                    endpoint_payload = pretty_bytes(endpoint)
                    files[str(endpoint_rel)] = endpoint_payload
                    endpoint_descriptors.append(_bytes_descriptor(str(endpoint_rel), endpoint_payload))
                    for request in request_rows[-len(compiled["request_descriptors"]):]:
                        if request["history_mode"] != "preceding_observation":
                            continue
                        history = _derive_history(request=request, compiled=compiled)
                        history_rel = Path("histories") / f"{request['source_request_id']}.json"
                        history_payload = pretty_bytes(history)
                        files[str(history_rel)] = history_payload
                        history_descriptors.append(_bytes_descriptor(str(history_rel), history_payload))

    expected_roster_count = len(models) * 24 * 4
    require(len(roster) == expected_roster_count and {row["cell_id"] for row in roster} == set(planned),
            "compiled confirmation roster is incomplete or duplicated")
    aggregate_actions = sum(
        row["counts"]["actual_behavioral_actions"] for row in aggregate_results.values()
    )
    aggregate_requests = sum(
        row["counts"]["actual_behavioral_model_requests"] for row in aggregate_results.values()
    )
    require(observed_actions == aggregate_actions,
            "compiled source action count differs from terminal block receipts")
    require(observed_source_requests == aggregate_requests,
            "compiled source request count differs from terminal block receipts")
    request_inventory = {
        "schema_version": REQUEST_INVENTORY_SCHEMA,
        "study_id": STUDY_ID,
        "stage": "confirmation",
        "cohort_branch": branch,
        "inventory_complete": True,
        "inventory_finalized_at": manifest["inventory_finalized_at"],
        "annotation_state": "not_started",
        "episode_roster": roster,
        "alignment_contracts": [dict(alignments[model]["alignment"]) for model in models],
        "requests": request_rows,
    }
    request_inventory_payload = pretty_bytes(request_inventory)
    files["request_inventory.json"] = request_inventory_payload
    inventory_descriptor = _bytes_descriptor("request_inventory.json", request_inventory_payload)
    provenance = sign_document({
        "schema_version": PROVENANCE_SCHEMA,
        "study_id": STUDY_ID,
        "stage": "confirmation",
        "cohort_branch": branch,
        "annotation_state": "not_started",
        "request_inventory_sha256": inventory_descriptor["sha256"],
        "requests": provenance_rows,
    })
    provenance_payload = pretty_bytes(provenance)
    files["request_provenance.json"] = provenance_payload
    private_video = sign_document({
        "schema_version": PRIVATE_VIDEO_SCHEMA,
        "study_id": STUDY_ID,
        "stage": "confirmation",
        "cohort_branch": branch,
        "visibility": "private_source_evidence_not_blind_annotation_media",
        "videos": video_rows,
    })
    private_video_payload = pretty_bytes(private_video)
    files["private_video_inventory.json"] = private_video_payload
    zero_science = sign_document({
        "schema_version": ZERO_SCIENCE_SCHEMA,
        "study_id": STUDY_ID,
        "stage": "confirmation_evidence_compilation",
        "model_runtime_loads": 0,
        "model_servers_started": 0,
        "model_requests_issued_by_compiler": 0,
        "simulator_processes_started": 0,
        "physical_resets": 0,
        "robot_episodes": 0,
        "behavioral_actions_executed_by_compiler": 0,
        "behavioral_cells_launched_by_compiler": 0,
        "labels_created_by_compiler": 0,
        "confirmation_jobs_released_by_compiler": 0,
    })
    zero_science_payload = pretty_bytes(zero_science)
    files["zero_science_receipt.json"] = zero_science_payload
    request_inventory_receipt = sign_document({
        "schema_version": REQUEST_INVENTORY_RECEIPT_SCHEMA,
        "study_id": STUDY_ID,
        "stage": "confirmation",
        "cohort_branch": branch,
        "request_inventory": inventory_descriptor,
        "inventory_consumer_schema": REQUEST_INVENTORY_SCHEMA,
        "inventory_consumer_validation": (
            "forecast_annotation_workflow.select_requests"
            if request_rows else "not_applicable_empty_inventory"
        ),
        "episode_roster_count": len(roster),
        "status_counts": status_counts,
        "request_count": len(request_rows),
        "block_receipts": [
            block["descriptor"] for _, block in sorted(aggregate_results.items())
        ],
        "annotation_state": "not_started",
        "labels": None,
    })
    request_inventory_receipt_payload = pretty_bytes(request_inventory_receipt)
    files["request_inventory_receipt.json"] = request_inventory_receipt_payload
    output_inventory = {
        "request_inventory": inventory_descriptor,
        "request_inventory_receipt": _bytes_descriptor(
            "request_inventory_receipt.json", request_inventory_receipt_payload
        ),
        "request_provenance": _bytes_descriptor("request_provenance.json", provenance_payload),
        "private_video_inventory": _bytes_descriptor(
            "private_video_inventory.json", private_video_payload
        ),
        "zero_science_receipt": _bytes_descriptor("zero_science_receipt.json", zero_science_payload),
        "endpoint_trace_receipts": endpoint_descriptors,
        "request_history_receipts": history_descriptors,
    }
    compiler_receipt = sign_document({
        "schema_version": COMPILER_SCHEMA,
        "study_id": STUDY_ID,
        "status": "compiled_complete_roster",
        "stage": "confirmation",
        "cohort_branch": branch,
        "study_commit": study_commit,
        "input_manifest": development.file_descriptor(manifest_path),
        "contract": development.file_descriptor(CONTRACT_PATH),
        "compiler_source": development.file_descriptor(Path(__file__).resolve()),
        "ablation_spec_dependency": development.file_descriptor(SPEC_PATH),
        "development_compiler_dependency": development.file_descriptor(DEVELOPMENT_COMPILER_PATH),
        "annotation_validator_dependency": development.file_descriptor(ANNOTATION_PATH),
        "final_analyzer_dependency": development.file_descriptor(ANALYZER_PATH),
        "release_validator_dependency": development.file_descriptor(FREEZE_PATH),
        "fixture_freeze_dependency": development.file_descriptor(FIXTURE_FREEZE_PATH),
        "n3_confirmation_validator_dependency": development.file_descriptor(
            N3_CONFIRMATION_PATH
        ),
        "d1_confirmation_validator_dependency": development.file_descriptor(
            D1_CONFIRMATION_PATH
        ),
        "terminal_runtime_source_commit": TERMINAL_RUNTIME_SOURCE_COMMIT,
        "n3_pilot_dependency": development.file_descriptor(N3_PILOT_PATH),
        "d1_pilot_dependency": development.file_descriptor(D1_PILOT_PATH),
        "d1_server_dependency": development.file_descriptor(D1_SERVER_PATH),
        "resource_qualification_contract_dependency": development.file_descriptor(
            RESOURCE_QUALIFICATION_CONTRACT_PATH
        ),
        "confirmation_release_freeze": release_descriptor,
        "cohort_close_receipt": close_descriptor,
        "cohort_close_outer_result_state": cohort_close[
            "_producer_outer_result_state"
        ],
        "source_root": str(source_root),
        "raw_root": str(raw_root),
        "camera_id": camera_id,
        "counts": {
            "planned_cells": expected_roster_count,
            "valid_complete": status_counts["valid_complete"],
            "valid_censored": status_counts["valid_censored"],
            "technical_invalid": status_counts["technical_invalid"],
            "not_run": status_counts["not_run"],
            "source_behavioral_actions": observed_actions,
            "source_behavioral_model_requests": observed_source_requests,
            "annotation_inventory_requests": len(request_rows),
            "endpoint_trace_receipts": len(endpoint_descriptors),
            "request_history_receipts": len(history_descriptors),
        },
        "outputs": output_inventory,
        "source_confirmation_artifacts": source_artifact_rows,
        "safe_for_request_selection": bool(request_rows),
        "safe_for_analysis_manifest_assembly": False,
        "labels_created": False,
        "scientific_results_computed": False,
        "confirmation_released": False,
        "claim_boundary": (
            "Authenticated source-only confirmation evidence and an exact status roster. "
            "This receipt contains no labels, forecast metric, policy competence claim, or queue release."
        ),
    })
    files["compiler_receipt.json"] = pretty_bytes(compiler_receipt)
    _write_atomic_directory_validated(
        output_dir,
        files,
        lambda bundle: _validate_compiled_bundle(
            bundle, active_seal_producer=active_seal_producer
        ),
    )
    return compiler_receipt


def _resolve_reference(base: Path, value: Any, label: str) -> tuple[dict[str, str], Path]:
    _exact_keys(value, REFERENCE_KEYS, label)
    path_value = value.get("path")
    digest = value.get("sha256")
    require(isinstance(path_value, str) and path_value,
            f"{label} path is missing")
    require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
            f"{label} SHA-256 is invalid")
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = base / candidate
    development._reject_symlink_components(candidate, label)
    path = candidate.resolve()
    require(path.is_file() and sha256_file(path) == digest,
            f"{label} file is missing or changed")
    return {"path": str(path), "sha256": digest}, path


def _load_compiler_bundle(bundle: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    supplied = Path(bundle)
    development._reject_symlink_components(supplied, "confirmation compiler bundle")
    bundle = supplied.resolve()
    require(bundle.is_dir(), "confirmation compiler bundle is missing")
    receipt_path = bundle / "compiler_receipt.json"
    receipt = load_json(receipt_path, "confirmation compiler receipt")
    require(receipt.get("schema_version") == COMPILER_SCHEMA
            and receipt.get("status") == "compiled_complete_roster"
            and receipt.get("scientific_results_computed") is False
            and receipt.get("labels_created") is False
            and receipt.get("confirmation_released") is False,
            "confirmation compiler receipt boundary changed")
    verify_signed(receipt, "confirmation compiler receipt")
    inventory_path = bundle / "request_inventory.json"
    inventory = load_json(inventory_path, "confirmation request inventory")
    expected = receipt.get("outputs", {}).get("request_inventory", {}).get("sha256")
    require(sha256_file(inventory_path) == expected,
            "confirmation request inventory differs from compiler receipt")
    return receipt, inventory, receipt_path


def assemble_analysis_manifest(
    *,
    compiler_bundle: Path,
    ablation_spec: Mapping[str, Any],
    development_release_freeze: Mapping[str, Any],
    request_selection: Mapping[str, Any],
    annotation_freeze: Mapping[str, Any],
    restricted_map: Mapping[str, Any],
    final_consensus: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    receipt, inventory, _ = _load_compiler_bundle(compiler_bundle)
    bundle = Path(compiler_bundle).resolve()
    # Reopen the native sources and replay both downstream consumer contracts;
    # a stale compiler receipt alone is never sufficient for final assembly.
    _validate_compiled_bundle(bundle)
    provided = {
        "ablation_spec": ablation_spec,
        "development_release_freeze": development_release_freeze,
        "request_selection": request_selection,
        "annotation_freeze": annotation_freeze,
        "restricted_map": restricted_map,
        "final_consensus": final_consensus,
    }
    require(set(provided) == ANALYSIS_SOURCE_KEYS, "analysis source inventory changed")
    sources: dict[str, dict[str, str]] = {}
    paths: dict[str, Path] = {}
    for key, value in provided.items():
        sources[key], paths[key] = _resolve_reference(Path.cwd(), value, f"analysis source {key}")
    require(paths["ablation_spec"] == SPEC_PATH.resolve(),
            "analysis manifest must use the authoritative machine-readable ablation spec")
    anchored_spec, anchored_spec_path = _exact_file_reference(
        receipt.get("ablation_spec_dependency"),
        base=bundle,
        label="compiled ablation-spec dependency",
    )
    require(
        anchored_spec_path == paths["ablation_spec"]
        and sources["ablation_spec"] == _reference(anchored_spec),
        "analysis ablation spec is not the exact source authenticated by the compiler",
    )
    try:
        analyzer.validate_machine_spec(load_json(paths["ablation_spec"], "ablation spec"))
        validated_release = freeze.validate_release_freeze(
            paths["development_release_freeze"], sources["development_release_freeze"]["sha256"]
        )
    except Exception as error:
        raise ConfirmationCompilerError(f"analysis immutable source validation failed: {error}") from error
    branch = receipt["cohort_branch"]
    require(validated_release.get("cohort_branch") == branch,
            "analysis release branch differs from compiled confirmation evidence")
    compiled_release = receipt.get("confirmation_release_freeze")
    _exact_keys(
        compiled_release, {"path", "sha256", "bytes"},
        "compiled confirmation-release freeze",
    )
    require(
        development.file_descriptor(paths["development_release_freeze"])
        == compiled_release,
        "analysis release is not the exact freeze used to compile confirmation evidence",
    )
    selection = load_json(paths["request_selection"], "confirmation request selection")
    try:
        expected_selection = annotation.select_requests(
            inventory,
            inventory_sha256=sha256_file(bundle / "request_inventory.json"),
            inventory_path=bundle / "request_inventory.json",
        )
        require(selection == expected_selection,
                "request selection is not the exact deterministic selection of compiled evidence")
        selected = annotation._selected_requests(selection)
    except Exception as error:
        raise ConfirmationCompilerError(f"confirmation request selection failed replay: {error}") from error
    require(selection.get("stage") == "confirmation" and selection.get("cohort_branch") == branch,
            "request selection does not match the compiled confirmation cohort")
    provenance = selection.get("provenance")
    require(isinstance(provenance, Mapping)
            and provenance.get("request_inventory_sha256") == sha256_file(bundle / "request_inventory.json"),
            "request selection does not bind the compiled request inventory")
    inventory_rows = {row["source_request_id"]: row for row in inventory["requests"]}
    require(set(selected) <= set(inventory_rows),
            "request selection contains a request outside compiled confirmation evidence")
    roster = {row["cell_id"]: row for row in inventory["episode_roster"]}
    endpoint_refs = []
    for descriptor in receipt["outputs"]["endpoint_trace_receipts"]:
        path = bundle / descriptor["path"]
        require(path.is_file() and sha256_file(path) == descriptor["sha256"],
                "compiled endpoint receipt changed")
        endpoint_refs.append({"path": str(path.resolve()), "sha256": descriptor["sha256"]})
    expected_endpoint_cells = {
        cell_id for cell_id, row in roster.items()
        if row["recording_status"] in {"valid_complete", "valid_censored"}
    }
    observed_endpoint_cells = {
        load_json(Path(row["path"]), "compiled endpoint")["cell_id"] for row in endpoint_refs
    }
    require(observed_endpoint_cells == expected_endpoint_cells,
            "compiled endpoint coverage changed before analysis assembly")
    history_by_request: dict[str, dict[str, str]] = {}
    for descriptor in receipt["outputs"]["request_history_receipts"]:
        path = bundle / descriptor["path"]
        require(path.is_file() and sha256_file(path) == descriptor["sha256"],
                "compiled history receipt changed")
        request_id = load_json(path, "compiled history")["source_request_id"]
        require(request_id not in history_by_request, "compiled history request is duplicated")
        history_by_request[request_id] = {"path": str(path.resolve()), "sha256": descriptor["sha256"]}
    expected_history = {
        request_id for request_id, row in selected.items()
        if row["history_mode"] == "preceding_observation"
    }
    require(expected_history <= set(history_by_request),
            "selected request lacks a compiled native history receipt")
    history_refs = [history_by_request[request_id] for request_id in sorted(expected_history)]
    manifest = sign_document({
        "schema_version": ANALYSIS_MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "stage": "confirmation",
        "cohort_branch": branch,
        "sources": sources,
        "endpoint_trace_receipts": sorted(endpoint_refs, key=lambda row: row["path"]),
        "request_history_receipts": history_refs,
    })
    _exact_keys(manifest, ANALYSIS_MANIFEST_KEYS, "assembled analysis manifest")
    output = Path(output)
    require(not output.exists(), f"refusing to overwrite analysis manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(pretty_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            analyzer.load_evidence(temporary)
        except Exception as error:
            raise ConfirmationCompilerError(f"assembled manifest failed final analyzer replay: {error}") from error
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def _reference_arg(path: Path, digest: str) -> dict[str, str]:
    require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
            f"invalid SHA-256 for {path}")
    return {"path": str(Path(path).resolve()), "sha256": digest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--manifest", type=Path, required=True)
    compile_parser.add_argument("--manifest-sha256", required=True)
    compile_parser.add_argument("--output-dir", type=Path, required=True)
    assemble = subparsers.add_parser("assemble-analysis-manifest")
    assemble.add_argument("--compiler-bundle", type=Path, required=True)
    for name in sorted(ANALYSIS_SOURCE_KEYS):
        option = name.replace("_", "-")
        assemble.add_argument(f"--{option}", type=Path, required=True)
        assemble.add_argument(f"--{option}-sha256", required=True)
    assemble.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "compile":
            result = compile_manifest(args.manifest, args.manifest_sha256, args.output_dir)
            output = Path(args.output_dir).resolve() / "compiler_receipt.json"
            summary = {
                "status": result["status"],
                "planned_cells": result["counts"]["planned_cells"],
                "valid_complete": result["counts"]["valid_complete"],
                "valid_censored": result["counts"]["valid_censored"],
                "technical_invalid": result["counts"]["technical_invalid"],
                "not_run": result["counts"]["not_run"],
                "scientific_results_computed": False,
                "labels_created": False,
                "confirmation_released": False,
                "output": str(output),
                "sha256": sha256_file(output),
            }
        else:
            references = {
                key: _reference_arg(
                    getattr(args, key), getattr(args, f"{key}_sha256")
                )
                for key in ANALYSIS_SOURCE_KEYS
            }
            result = assemble_analysis_manifest(
                compiler_bundle=args.compiler_bundle,
                output=args.output,
                **references,
            )
            summary = {
                "status": "analysis_manifest_assembled_and_validated",
                "scientific_results_computed": False,
                "output": str(Path(args.output).resolve()),
                "sha256": sha256_file(Path(args.output).resolve()),
                "schema_version": result["schema_version"],
            }
    except ConfirmationCompilerError as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
