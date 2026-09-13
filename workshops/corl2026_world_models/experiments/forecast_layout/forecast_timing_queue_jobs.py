#!/usr/bin/env python3
"""Detached queue wrappers and descriptor builders for forecast timing evidence.

The initial wave performs only source auditing, input packaging, and
normalization of an already completed D1 generation qualification.  It starts
no simulator, model server, model request, physical reset, or behavioral
action.  A separate N3 descriptor builder is deliberately receipt-gated: it
cannot emit the six-request generation job until the live P00 input preparation
job has published a hash-bound passing receipt.

The native-authority wave is likewise receipt-gated.  Its builder accepts the
passed N3 generation attempt instead of predicting that outcome, and binds the
already-published source audits, D1 generation normalization, and recorder
trace by exact PVC path and hash.  Those jobs qualify timing only; confirmation
remains held.

Every runtime command reopens its immutable queue descriptor and claim, proves
the exact staged Git checkout and worker role, verifies fixed input hashes, and
writes only beneath its unique queue job directory on the shared PVC.  The
large/raw chain stays there; ``publish`` contains bounded JSON receipts only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import traceback
from types import ModuleType
from typing import Any, Mapping, Sequence


# Never leave __pycache__ dirt in an immutable staged queue checkout.
sys.dont_write_bytecode = True

NAMESPACE = "wmf_ablation_001_20260912"
STUDY_ID = "WMF-ABLATION-001"
QUEUE_JOB_SCHEMA = "wmf-cluster-job-v1"
TIMING_JOB_SCHEMA = "wmf-forecast-timing-queue-job-v1"
INITIAL_DESCRIPTOR_SCHEMA = "wmf-forecast-timing-initial-wave-descriptors-v1"
AUTHORITY_DESCRIPTOR_SCHEMA = "wmf-forecast-timing-authority-wave-descriptors-v1"

COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")

FORECAST_RELATIVE = Path("workshops/corl2026_world_models")
THIS_RELATIVE = FORECAST_RELATIVE / "experiments/forecast_layout/forecast_timing_queue_jobs.py"
TOOL_RELATIVE = FORECAST_RELATIVE / "analysis/qualify_forecast_timing.py"
CONTRACT_RELATIVE = (
    FORECAST_RELATIVE / "experiments/forecast_layout/forecast_timing_lineage_contract.json"
)
N3_RUNNER_RELATIVE = FORECAST_RELATIVE / "experiments/forecast_layout/n3_first_live.py"
N3_RUNTIME_CONTRACT_RELATIVE = (
    FORECAST_RELATIVE / "experiments/forecast_layout/n3_first_live_contract.json"
)

# Prior prerequisite receipts were produced by the original reviewed validator.
# New authority attempts use the staged-prefix-safe validator while retaining
# the exact same lineage contract and qualified N3 generation runner.
PRIOR_TOOL_SHA256 = "805fc1bc5d6eeeb174ac880ba034cc9eb7198292ffacce8fc350ea5bc7ffc262"
TOOL_SHA256 = "aab99105478b983c0e58fba11642f394d08c09de855367806474452763a5965b"
CONTRACT_SHA256 = "7a9cca8b2d3c0057ab6022b8ce0612fba0e84e4ccacaa4258f6cad266e48490e"
N3_RUNNER_SHA256 = "9655047f322e7488d3bbf27839cf5ae6bf0ea91c351bbfe4ec45950883c42f8d"
N3_RUNTIME_CONTRACT_SHA256 = (
    "a9503011b8f7236dd3e8eb9c2e4328f2ab95021b4f72e5900c01b51cee743e61"
)

ROBOLAB_PYTHON = Path("/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python")
COSMOS_PYTHON = Path(
    "/data/users/ali/vla_wam/envs/cosmos-nano-411d25b-v3-exact/bin/python"
)
N3_SOURCE = Path("/data/users/ali/vla_wam/external/v3-clean/cosmos-nano-411d25b")
D1_SOURCE = Path("/data/users/ali/vla_wam/external/DreamZero-v3e004-clean-ab790c1")
N3_CHECKPOINT = Path("/data/users/ali/vla_wam/checkpoints/cosmos3_nano_policy_droid")
CONTROL_ROOT = Path("/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control")
CAPTURE_RECEIPT = Path(
    "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/fixed_observations/"
    "fixed-observation-p00-001/capture/capture_receipt.json"
)
CAPTURE_RECEIPT_SHA256 = (
    "8d42bc36fe57747f5ce008c54f14e64f03bbe27be756b80c2b431d50aef7f663"
)
D1_QUALIFICATION_RECEIPT = CONTROL_ROOT / (
    "jobs/d1-first-live-005/publish/d1_qualification_job_receipt.json"
)
D1_QUALIFICATION_RECEIPT_SHA256 = (
    "3c856549999b9145dc07c30853a4c6d2968d09eb31c2db883f5eb1655d31627b"
)
RECORDER_RECEIPT = CONTROL_ROOT / (
    "jobs/recorder-qualification-p00-003/publish/recorder_qualification_receipt.json"
)
RECORDER_RECEIPT_SHA256 = (
    "0e3f02f37a2548e36ae3a45a38a1fac63c56cd8798732056f24b03d103991fde"
)
CAMERA_ID = "over_shoulder_left_camera"
EFFECTIVE_SEED = 2026091000

N3_COMPAT_BIN = Path("/data/users/ali/vla_wam/raw/cosmos3_nano_droid/v2_a011/compat-bin")
N3_HF_HOME = Path("/data/users/ali/vla_wam/cache/huggingface-cosmos")
N3_LD_LIBRARY_PATH = ":".join(
    (
        "/data/users/ali/vla_wam/envs/cosmos-nano-411d25b-v3-exact/"
        "lib/python3.13/site-packages/nvidia/cudnn/lib",
        "/data/users/ali/vla_wam/envs/isaac-system-libs/lib",
        "/data/users/jsalfity/glvnd/lib",
    )
)
SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class TimingQueueError(RuntimeError):
    """A queue, input, or output binding failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TimingQueueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TimingQueueError("value is not canonical finite JSON") from error


def compact_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TimingQueueError("value is not canonical finite JSON") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    supplied = Path(path)
    require(not supplied.is_symlink(), f"evidence is a symlink: {supplied}")
    resolved = supplied.resolve()
    require(resolved.is_file(), f"evidence file is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise TimingQueueError(f"non-finite JSON token: {value}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    supplied = Path(path)
    require(not supplied.is_symlink(), f"{label} is a symlink")
    try:
        value = json.loads(
            supplied.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except TimingQueueError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TimingQueueError(f"{label} is unreadable JSON") from error
    require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def signed_document(value: Mapping[str, Any]) -> dict[str, Any]:
    require("payload_sha256" not in value, "document is already signed")
    result = dict(value)
    result["payload_sha256"] = sha256_bytes(compact_bytes(result))
    return result


def verify_signed_document(value: Mapping[str, Any], label: str) -> None:
    observed = value.get("payload_sha256")
    require(isinstance(observed, str) and SHA256_RE.fullmatch(observed) is not None,
            f"{label} signature is invalid")
    unsigned = dict(value)
    unsigned.pop("payload_sha256")
    require(sha256_bytes(compact_bytes(unsigned)) == observed, f"{label} signature changed")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def immutable_bytes(path: Path, payload: bytes, *, maximum_bytes: int | None = None) -> None:
    target = Path(path)
    require(not target.exists() and not target.is_symlink(), f"refusing to replace output: {target}")
    if maximum_bytes is not None:
        require(len(payload) <= maximum_bytes, f"output exceeds {maximum_bytes} bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.parent.is_symlink(), f"output parent is a symlink: {target.parent}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise TimingQueueError(f"refusing to replace output: {target}") from error
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def immutable_json(path: Path, value: Mapping[str, Any], *, maximum_bytes: int = 512 * 1024) -> None:
    immutable_bytes(path, canonical_bytes(value), maximum_bytes=maximum_bytes)


def _verified_sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"{label} is not SHA-256")
    return value


def _verified_commit(value: Any) -> str:
    require(isinstance(value, str) and COMMIT_RE.fullmatch(value) is not None,
            "study commit is not a full lowercase commit")
    return value


def _verified_safe_id(value: Any, label: str) -> str:
    require(isinstance(value, str) and SAFE_ID_RE.fullmatch(value) is not None,
            f"{label} is not a safe queue ID")
    return value


@dataclass(frozen=True)
class InitialJob:
    mode: str
    job_id: str
    role: str
    model: str | None
    max_wall_seconds: int
    referenced_generation_requests: int = 0


INITIAL_JOBS: tuple[InitialJob, ...] = (
    InitialJob(
        "n3-source-audit",
        "timing-n3-source-audit-001",
        "wmf-forecast-0912-worker-05",
        "N3",
        900,
    ),
    InitialJob(
        "d1-source-audit",
        "timing-d1-source-audit-001",
        "wmf-forecast-0912-worker-06",
        "D1",
        900,
    ),
    InitialJob(
        "n3-live-input",
        "timing-n3-live-input-p00-001",
        "wmf-forecast-0912-worker-09",
        "N3",
        900,
    ),
    InitialJob(
        "d1-normalize-generation",
        "timing-d1-normalize-generation-001",
        "wmf-forecast-0912-worker-09",
        "D1",
        1800,
        referenced_generation_requests=6,
    ),
)
INITIAL_BY_MODE = {job.mode: job for job in INITIAL_JOBS}

N3_GENERATION_JOB_ID = "timing-n3-live-generation-p00-002"
N3_GENERATION_ROLE = "n3"
N3_GENERATION_WORKER_ID = "wmf-forecast-0912-worker-n3-00"
PREPARATION_JOB = INITIAL_BY_MODE["n3-live-input"]
PREPARATION_CLUSTER_JOB_DIR = CONTROL_ROOT / "jobs" / PREPARATION_JOB.job_id
PREPARATION_CLUSTER_JOB_RECEIPT = (
    PREPARATION_CLUSTER_JOB_DIR / "publish" / "timing_job_receipt.json"
)
PREPARATION_CLUSTER_JOB_RECEIPT_SHA256 = (
    "e1119ea9bc30cb27ec54736569e71d02f692ed5ba2b37f9fde25404729114d07"
)
PREPARATION_RAW_ROOT = PREPARATION_CLUSTER_JOB_DIR / "raw" / "n3_live_input"
PREPARATION_RECEIPT_SHA256 = (
    "b4c8a220dd3c936875b6112bbd8fbb0afb32b687fbb4ab2e7262b7695f328dca"
)
PREPARATION_MANIFEST_SHA256 = (
    "ab4dc15aa64722f37cf42b0f9ef813e0eb4279245ce7aba1698f96f3f2e096a9"
)
PREPARATION_PAYLOAD_SHA256 = (
    "a6cf00507e5a2fff6837f9d3a55846830c9ca7306b4bfe580ebd2b78d31a827f"
)


@dataclass(frozen=True)
class PriorTimingJob:
    model: str
    mode: str
    job_id: str
    role: str
    worker_id: str
    study_commit: str
    receipt_sha256: str
    output_key: str
    published_output_key: str
    artifact_name: str
    artifact_bytes: int
    artifact_sha256: str
    referenced_generation_requests: int

    @property
    def job_dir(self) -> Path:
        return CONTROL_ROOT / "jobs" / self.job_id

    @property
    def receipt_path(self) -> Path:
        return self.job_dir / "publish" / "timing_job_receipt.json"

    @property
    def artifact_path(self) -> Path:
        return self.job_dir / "raw" / self.artifact_name


N3_SOURCE_AUDIT_JOB = PriorTimingJob(
    "N3",
    "n3-source-audit",
    "timing-n3-source-audit-001",
    "wmf-forecast-0912-worker-05",
    "wmf-forecast-0912-worker-05",
    "25ff299ca0d2b9964eb48286990ee2301cb207b8",
    "9807b7b9fd895eb080740b2d135fb8d5877020a53c6ee2ac9a3796f0fe144619",
    "primary",
    "published_primary",
    "n3_source_audit.json",
    7394,
    "98fe2660232c479db1f98b552915c98f5c5695df6b4090bc6dffb50952e2f8c3",
    0,
)
D1_SOURCE_AUDIT_JOB = PriorTimingJob(
    "D1",
    "d1-source-audit",
    "timing-d1-source-audit-001",
    "wmf-forecast-0912-worker-06",
    "wmf-forecast-0912-worker-06",
    "25ff299ca0d2b9964eb48286990ee2301cb207b8",
    "b669f9576a32b2ca8fa5127d82d4ba4d6f765d44ab2f7ce31eb24d0f2ae25dd9",
    "primary",
    "published_primary",
    "d1_source_audit.json",
    5459,
    "39181fecb7d473b52b2b661c29b61cce653e3ec11edd437c74f1fefbefa76466",
    0,
)
D1_GENERATION_JOB = PriorTimingJob(
    "D1",
    "d1-normalize-generation",
    "timing-d1-normalize-generation-001",
    "wmf-forecast-0912-worker-09",
    "wmf-forecast-0912-worker-09",
    "25ff299ca0d2b9964eb48286990ee2301cb207b8",
    "dc490e78f4290a319e05c76b4bebd1ab7d10b5519cfb747656d1f1d4905acaf7",
    "primary",
    "published_primary",
    "d1_generation_probe.json",
    4764,
    "e7eddb7c3cc81b2c352e57dda7cb7a19ef5cf4bec8ca33309bfc3a58ad724127",
    6,
)


@dataclass(frozen=True)
class AuthorityJob:
    model: str
    mode: str
    job_id: str
    role: str
    expected_target_count: int


AUTHORITY_JOBS: tuple[AuthorityJob, ...] = (
    AuthorityJob(
        "N3",
        "n3-native-authority",
        "timing-n3-native-authority-002",
        "wmf-forecast-0912-worker-05",
        32,
    ),
    AuthorityJob(
        "D1",
        "d1-native-authority",
        "timing-d1-native-authority-002",
        "wmf-forecast-0912-worker-06",
        2,
    ),
)
AUTHORITY_BY_MODE = {job.mode: job for job in AUTHORITY_JOBS}

N3_GENERATION_STUDY_COMMIT = "35e969b7fae02592a9ac45025bd0c64c5432e3ac"
N3_GENERATION_CLUSTER_JOB_DIR = CONTROL_ROOT / "jobs" / N3_GENERATION_JOB_ID
N3_GENERATION_CLUSTER_JOB_RECEIPT = (
    N3_GENERATION_CLUSTER_JOB_DIR / "publish" / "timing_job_receipt.json"
)


def _runtime_base_argv(job: InitialJob, study_commit: str) -> list[str]:
    return [
        str(ROBOLAB_PYTHON),
        "{source_root}/" + str(THIS_RELATIVE),
        job.mode,
        "--source-root",
        "{source_root}",
        "--study-commit",
        study_commit,
        "--job-dir",
        "{job_dir}",
        "--job-id",
        job.job_id,
        "--expected-role",
        job.role,
        "--contract-sha256",
        CONTRACT_SHA256,
    ]


def _initial_argv(job: InitialJob, study_commit: str) -> list[str]:
    argv = _runtime_base_argv(job, study_commit)
    if job.mode in {"n3-source-audit", "d1-source-audit"}:
        source = N3_SOURCE if job.model == "N3" else D1_SOURCE
        argv.extend(("--external-source-root", str(source)))
    elif job.mode == "n3-live-input":
        argv.extend(
            (
                "--capture-receipt",
                str(CAPTURE_RECEIPT),
                "--capture-receipt-sha256",
                CAPTURE_RECEIPT_SHA256,
                "--camera-id",
                CAMERA_ID,
            )
        )
    elif job.mode == "d1-normalize-generation":
        argv.extend(
            (
                "--qualification-receipt",
                str(D1_QUALIFICATION_RECEIPT),
                "--qualification-receipt-sha256",
                D1_QUALIFICATION_RECEIPT_SHA256,
            )
        )
    else:  # pragma: no cover - the frozen inventory makes this unreachable.
        raise TimingQueueError(f"unsupported initial timing mode: {job.mode}")
    return argv


def build_initial_descriptor(job: InitialJob, study_commit: str) -> dict[str, Any]:
    commit = _verified_commit(study_commit)
    require(job in INITIAL_JOBS, "job is outside the frozen initial timing wave")
    return {
        "job_id": job.job_id,
        "released": True,
        "source_commit": commit,
        "role": job.role,
        "argv": _initial_argv(job, commit),
        "max_wall_seconds": job.max_wall_seconds,
        "publish_log_tail_bytes": 4096,
    }


def build_initial_wave(study_commit: str) -> dict[str, Any]:
    """Return descriptors only; this function never edits the active queue."""

    commit = _verified_commit(study_commit)
    return {
        "schema_version": INITIAL_DESCRIPTOR_SCHEMA,
        "namespace": NAMESPACE,
        "source_commit": commit,
        "status": "descriptor_only_not_dispatched",
        "generation_requests_issued": 0,
        "behavioral_actions_executed": 0,
        "jobs": [build_initial_descriptor(job, commit) for job in INITIAL_JOBS],
        "claim_boundary": (
            "Four detached CPU-safe evidence jobs only. The returned descriptors are not an active "
            "queue and this command does not dispatch them."
        ),
    }


def _n3_generation_shell_argv(
    study_commit: str,
    *,
    preparation_job_receipt_sha256: str,
    preparation_receipt: Mapping[str, Any],
    observation_manifest: Mapping[str, Any],
) -> list[str]:
    wrapper = "'{source_root}/" + str(THIS_RELATIVE) + "'"
    command = " ".join(
        (
            "mkdir -p '{job_dir}/torchinductor' '{job_dir}/tmp' &&",
            f"cd {N3_SOURCE} && exec {COSMOS_PYTHON}",
            wrapper,
            "n3-generate",
            "--source-root '{source_root}'",
            f"--study-commit {study_commit}",
            "--job-dir '{job_dir}'",
            f"--job-id {N3_GENERATION_JOB_ID}",
            f"--expected-role {N3_GENERATION_ROLE}",
            f"--expected-worker-id {N3_GENERATION_WORKER_ID}",
            f"--contract-sha256 {CONTRACT_SHA256}",
            f"--preparation-job-receipt {PREPARATION_CLUSTER_JOB_RECEIPT}",
            f"--preparation-job-receipt-sha256 {preparation_job_receipt_sha256}",
            f"--preparation-receipt {preparation_receipt['path']}",
            f"--preparation-receipt-sha256 {preparation_receipt['sha256']}",
            f"--observation-manifest {observation_manifest['path']}",
            f"--observation-manifest-sha256 {observation_manifest['sha256']}",
        )
    )
    return [
        "/usr/bin/env",
        "CUDA_VISIBLE_DEVICES=0",
        "DS_IGNORE_CUDA_DETECTION=1",
        f"HF_HOME={N3_HF_HOME}",
        f"PATH={N3_COMPAT_BIN}:{COSMOS_PYTHON.parent}:{SYSTEM_PATH}",
        f"LD_LIBRARY_PATH={N3_LD_LIBRARY_PATH}",
        f"PYTHONPATH={N3_SOURCE}:{{source_root}}",
        "TORCHINDUCTOR_CACHE_DIR={job_dir}/torchinductor",
        "TMPDIR={job_dir}/tmp",
        "/bin/bash",
        "-c",
        command,
    ]


def _require_descriptor(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} descriptor is missing")
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    require(isinstance(path, str) and Path(path).is_absolute(), f"{label} path is not absolute")
    require(type(size) is int and size > 0, f"{label} byte count is invalid")
    _verified_sha(digest, f"{label} digest")
    return {"path": path, "bytes": size, "sha256": digest}


def validate_preparation_job_receipt(
    path: Path, expected_sha256: str
) -> dict[str, dict[str, Any] | str]:
    """Validate the published copy before materializing the N3 descriptor.

    The receipt may be read from the fetched results branch on the workstation;
    all artifact paths embedded in it must still name the exact PVC attempt.
    Runtime validation later reopens the original receipt and every raw input.
    """

    expected = _verified_sha(expected_sha256, "preparation job receipt digest")
    identity = file_identity(path)
    require(identity["sha256"] == expected, "preparation job receipt hash mismatch")
    receipt = load_json(path, "preparation job receipt")
    verify_signed_document(receipt, "preparation job receipt")
    exact = {
        "schema_version": TIMING_JOB_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "status": "passed",
        "decision": "go",
        "mode": PREPARATION_JOB.mode,
        "job_id": PREPARATION_JOB.job_id,
        "queue_role": PREPARATION_JOB.role,
        "job_dir": str(PREPARATION_CLUSTER_JOB_DIR),
        "physical_time_qualified": False,
        "behavioral_policy_skill_evaluated": False,
    }
    for key, wanted in exact.items():
        require(receipt.get(key) == wanted, f"preparation job receipt changed: {key}")
    counts = receipt.get("science_counts")
    require(
        counts
        == {
            "model_runtime_loads": 0,
            "model_servers_started": 0,
            "model_requests_issued_by_job": 0,
            "referenced_generation_requests": 0,
            "physical_resets": 0,
            "robot_episodes": 0,
            "behavioral_actions": 0,
            "behavioral_cells": 0,
        },
        "preparation job receipt claims unexpected science",
    )
    outputs = receipt.get("outputs")
    require(isinstance(outputs, Mapping), "preparation outputs are missing")
    primary = _require_descriptor(outputs.get("primary"), "preparation receipt")
    manifest = _require_descriptor(outputs.get("observation_manifest"), "observation manifest")
    payload = _require_descriptor(outputs.get("observation_payload"), "observation payload")
    expected_root = PREPARATION_CLUSTER_JOB_DIR / "raw" / "n3_live_input"
    require(Path(primary["path"]) == expected_root / "preparation_receipt.json",
            "preparation receipt escaped its immutable attempt")
    require(Path(manifest["path"]) == expected_root / "observation_manifest.json",
            "observation manifest escaped its immutable attempt")
    require(Path(payload["path"]) == expected_root / "observation.npz",
            "observation payload escaped its immutable attempt")
    _verified_commit(receipt.get("study_commit"))
    return {
        "job_receipt_sha256": expected,
        "preparation_receipt": primary,
        "observation_manifest": manifest,
        "observation_payload": payload,
    }


def build_n3_generation_descriptor(
    study_commit: str,
    *,
    preparation_job_receipt_path: Path,
    preparation_job_receipt_sha256: str,
) -> dict[str, Any]:
    """Build, but do not dispatch, the receipt-gated six-request N3 job."""

    commit = _verified_commit(study_commit)
    evidence = validate_preparation_job_receipt(
        preparation_job_receipt_path, preparation_job_receipt_sha256
    )
    preparation = evidence["preparation_receipt"]
    manifest = evidence["observation_manifest"]
    assert isinstance(preparation, Mapping) and isinstance(manifest, Mapping)
    return {
        "job_id": N3_GENERATION_JOB_ID,
        "released": True,
        "source_commit": commit,
        "role": N3_GENERATION_ROLE,
        "argv": _n3_generation_shell_argv(
            commit,
            preparation_job_receipt_sha256=str(evidence["job_receipt_sha256"]),
            preparation_receipt=preparation,
            observation_manifest=manifest,
        ),
        "max_wall_seconds": 7200,
        "publish_log_tail_bytes": 8192,
    }


def _require_bound_descriptor(
    value: Any,
    *,
    expected_path: Path,
    expected_sha256: str,
    label: str,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    descriptor = _require_descriptor(value, label)
    require(Path(descriptor["path"]) == expected_path, f"{label} path changed")
    require(
        descriptor["sha256"] == _verified_sha(expected_sha256, f"{label} digest"),
        f"{label} hash changed",
    )
    if expected_bytes is not None:
        require(descriptor["bytes"] == expected_bytes, f"{label} byte count changed")
    return descriptor


def _validate_receipt_implementation(
    receipt: Mapping[str, Any], *, study_commit: str, include_n3_runner: bool
) -> dict[str, dict[str, Any]]:
    implementation = receipt.get("implementation")
    require(isinstance(implementation, Mapping), "timing receipt implementation is missing")
    expected: dict[str, tuple[Path, str]] = {
        "timing_validator": (TOOL_RELATIVE, PRIOR_TOOL_SHA256),
        "timing_contract": (CONTRACT_RELATIVE, CONTRACT_SHA256),
    }
    if include_n3_runner:
        expected.update(
            {
                "n3_generation_runner": (N3_RUNNER_RELATIVE, N3_RUNNER_SHA256),
                "n3_runtime_contract": (
                    N3_RUNTIME_CONTRACT_RELATIVE,
                    N3_RUNTIME_CONTRACT_SHA256,
                ),
            }
        )
    require(set(implementation) == set(expected), "timing receipt implementation inventory changed")
    source_root = CONTROL_ROOT / "sources" / study_commit
    return {
        key: _require_bound_descriptor(
            implementation[key],
            expected_path=source_root / relative,
            expected_sha256=digest,
            label=f"timing receipt {key}",
        )
        for key, (relative, digest) in expected.items()
    }


def _validate_timing_receipt_header(
    receipt: Mapping[str, Any],
    *,
    mode: str,
    job_id: str,
    job_dir: Path,
    role: str,
    worker_id: str,
    study_commit: str,
) -> None:
    exact = {
        "schema_version": TIMING_JOB_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "status": "passed",
        "decision": "go",
        "mode": mode,
        "job_id": job_id,
        "job_dir": str(job_dir),
        "study_commit": study_commit,
        "queue_role": role,
        "worker_id": worker_id,
        "physical_time_qualified": False,
        "behavioral_policy_skill_evaluated": False,
        "safe_to_release_confirmation": False,
    }
    for key, wanted in exact.items():
        require(receipt.get(key) == wanted, f"timing job receipt changed: {key}")
    runtime = receipt.get("runtime_identity")
    require(isinstance(runtime, Mapping), "timing receipt runtime identity is missing")
    require(
        isinstance(runtime.get("hostname"), str)
        and runtime["hostname"].startswith(worker_id + "-"),
        "timing receipt hostname differs from its worker identity",
    )
    require(
        isinstance(runtime.get("pod_uid"), str) and bool(runtime["pod_uid"]),
        "timing receipt pod UID is missing",
    )
    require(type(runtime.get("pid")) is int and runtime["pid"] > 0, "timing receipt PID is invalid")
    queue_descriptor = receipt.get("queue_descriptor")
    require(isinstance(queue_descriptor, Mapping), "timing queue descriptor is missing")
    _require_bound_descriptor(
        queue_descriptor,
        expected_path=job_dir / "descriptor.json",
        expected_sha256=str(queue_descriptor.get("sha256", "")),
        label="timing queue descriptor",
    )
    queue_claim = receipt.get("queue_claim")
    require(isinstance(queue_claim, Mapping), "timing queue claim is missing")
    _require_bound_descriptor(
        queue_claim,
        expected_path=job_dir / "claim" / "owner.json",
        expected_sha256=str(queue_claim.get("sha256", "")),
        label="timing queue claim",
    )


def _expected_science_counts(*, issued: int, referenced: int) -> dict[str, int]:
    return {
        "model_runtime_loads": 1 if issued else 0,
        "model_servers_started": 0,
        "model_requests_issued_by_job": issued,
        "referenced_generation_requests": referenced,
        "physical_resets": 0,
        "robot_episodes": 0,
        "behavioral_actions": 0,
        "behavioral_cells": 0,
    }


def validate_n3_generation_job_receipt(
    path: Path, expected_sha256: str
) -> dict[str, Any]:
    """Validate passed attempt 002 without requiring its PVC artifacts locally.

    The builder may read the deliberately published receipt from a fetched
    results branch.  The detached runtime later reopens the exact PVC receipt,
    normalized probe, qualification, and their transitive raw evidence.
    """

    expected = _verified_sha(expected_sha256, "N3 generation job receipt digest")
    identity = file_identity(path)
    require(identity["sha256"] == expected, "N3 generation job receipt hash mismatch")
    receipt = load_json(path, "N3 generation job receipt")
    verify_signed_document(receipt, "N3 generation job receipt")
    _validate_timing_receipt_header(
        receipt,
        mode="n3-generate",
        job_id=N3_GENERATION_JOB_ID,
        job_dir=N3_GENERATION_CLUSTER_JOB_DIR,
        role=N3_GENERATION_ROLE,
        worker_id=N3_GENERATION_WORKER_ID,
        study_commit=N3_GENERATION_STUDY_COMMIT,
    )
    require(
        receipt.get("science_counts") == _expected_science_counts(issued=6, referenced=0),
        "N3 generation job receipt science counts changed",
    )
    _validate_receipt_implementation(
        receipt, study_commit=N3_GENERATION_STUDY_COMMIT, include_n3_runner=True
    )
    inputs = receipt.get("inputs")
    require(isinstance(inputs, Mapping), "N3 generation inputs are missing")
    _require_bound_descriptor(
        inputs.get("preparation_job_receipt"),
        expected_path=PREPARATION_CLUSTER_JOB_RECEIPT,
        expected_sha256=PREPARATION_CLUSTER_JOB_RECEIPT_SHA256,
        expected_bytes=4125,
        label="N3 preparation job receipt input",
    )
    _require_bound_descriptor(
        inputs.get("preparation_receipt"),
        expected_path=PREPARATION_RAW_ROOT / "preparation_receipt.json",
        expected_sha256=PREPARATION_RECEIPT_SHA256,
        expected_bytes=1787,
        label="N3 preparation receipt input",
    )
    _require_bound_descriptor(
        inputs.get("observation_manifest"),
        expected_path=PREPARATION_RAW_ROOT / "observation_manifest.json",
        expected_sha256=PREPARATION_MANIFEST_SHA256,
        expected_bytes=2128,
        label="N3 observation manifest input",
    )
    _require_bound_descriptor(
        inputs.get("observation_payload"),
        expected_path=PREPARATION_RAW_ROOT / "observation.npz",
        expected_sha256=PREPARATION_PAYLOAD_SHA256,
        expected_bytes=1037592,
        label="N3 observation payload input",
    )
    child = receipt.get("child")
    require(
        isinstance(child, Mapping)
        and child.get("returncode") == 0
        and child.get("reaped") is True,
        "N3 generation child did not exit cleanly",
    )
    expected_child = [
        str(COSMOS_PYTHON),
        str(CONTROL_ROOT / "sources" / N3_GENERATION_STUDY_COMMIT / N3_RUNNER_RELATIVE),
        "--source-root",
        str(N3_SOURCE),
        "--checkpoint-root",
        str(N3_CHECKPOINT),
        "--observation-manifest",
        str(
            PREPARATION_CLUSTER_JOB_DIR
            / "raw"
            / "n3_live_input"
            / "observation_manifest.json"
        ),
        "--output-dir",
        str(N3_GENERATION_CLUSTER_JOB_DIR / "raw" / "n3_generation_raw"),
        "--publish-dir",
        str(N3_GENERATION_CLUSTER_JOB_DIR / "raw" / "n3_generation_publish"),
        "--effective-seed",
        str(EFFECTIVE_SEED),
    ]
    require(child.get("argv") == expected_child, "N3 generation child command changed")
    outputs = receipt.get("outputs")
    require(isinstance(outputs, Mapping), "N3 generation outputs are missing")
    normalized = _require_descriptor(
        outputs.get("normalized_generation_probe"), "N3 normalized generation probe"
    )
    require(
        Path(normalized["path"])
        == N3_GENERATION_CLUSTER_JOB_DIR / "raw" / "n3_generation_probe.json",
        "N3 normalized generation probe escaped attempt 002",
    )
    published = _require_bound_descriptor(
        outputs.get("published_generation_probe"),
        expected_path=N3_GENERATION_CLUSTER_JOB_DIR / "publish" / "n3_generation_probe.json",
        expected_sha256=str(normalized["sha256"]),
        expected_bytes=int(normalized["bytes"]),
        label="published N3 generation probe",
    )
    qualification = _require_descriptor(outputs.get("qualification"), "N3 qualification")
    require(
        Path(qualification["path"])
        == N3_GENERATION_CLUSTER_JOB_DIR
        / "raw"
        / "n3_generation_publish"
        / "n3_qualification.json",
        "N3 qualification escaped attempt 002",
    )
    published_qualification = _require_bound_descriptor(
        outputs.get("published_qualification"),
        expected_path=N3_GENERATION_CLUSTER_JOB_DIR / "publish" / "n3_qualification.json",
        expected_sha256=str(qualification["sha256"]),
        expected_bytes=int(qualification["bytes"]),
        label="published N3 qualification",
    )
    return {
        "job_receipt": {
            "path": str(N3_GENERATION_CLUSTER_JOB_RECEIPT),
            "bytes": identity["bytes"],
            "sha256": expected,
        },
        "generation_probe": normalized,
        "published_generation_probe": published,
        "qualification": qualification,
        "published_qualification": published_qualification,
    }


def _prior_receipt_descriptor(job: PriorTimingJob) -> dict[str, Any]:
    return {
        "path": str(job.receipt_path),
        "sha256": job.receipt_sha256,
    }


def _prior_artifact_descriptor(job: PriorTimingJob) -> dict[str, Any]:
    return {
        "path": str(job.artifact_path),
        "bytes": job.artifact_bytes,
        "sha256": job.artifact_sha256,
    }


def _authority_argv(
    job: AuthorityJob,
    study_commit: str,
    *,
    generation_job_receipt: Mapping[str, Any],
    generation_probe: Mapping[str, Any],
) -> list[str]:
    source_job = N3_SOURCE_AUDIT_JOB if job.model == "N3" else D1_SOURCE_AUDIT_JOB
    return [
        str(ROBOLAB_PYTHON),
        "{source_root}/" + str(THIS_RELATIVE),
        job.mode,
        "--source-root",
        "{source_root}",
        "--study-commit",
        study_commit,
        "--job-dir",
        "{job_dir}",
        "--job-id",
        job.job_id,
        "--expected-role",
        job.role,
        "--contract-sha256",
        CONTRACT_SHA256,
        "--source-audit-job-receipt",
        str(source_job.receipt_path),
        "--source-audit-job-receipt-sha256",
        source_job.receipt_sha256,
        "--source-audit",
        str(source_job.artifact_path),
        "--source-audit-sha256",
        source_job.artifact_sha256,
        "--generation-job-receipt",
        str(generation_job_receipt["path"]),
        "--generation-job-receipt-sha256",
        str(generation_job_receipt["sha256"]),
        "--generation-probe",
        str(generation_probe["path"]),
        "--generation-probe-sha256",
        str(generation_probe["sha256"]),
        "--recorder-receipt",
        str(RECORDER_RECEIPT),
        "--recorder-receipt-sha256",
        RECORDER_RECEIPT_SHA256,
        "--camera-id",
        CAMERA_ID,
        "--expected-target-count",
        str(job.expected_target_count),
    ]


def build_authority_descriptor(
    job: AuthorityJob,
    study_commit: str,
    *,
    n3_generation: Mapping[str, Any],
) -> dict[str, Any]:
    commit = _verified_commit(study_commit)
    require(job in AUTHORITY_JOBS, "job is outside the frozen native-authority wave")
    if job.model == "N3":
        generation_job_receipt = n3_generation["job_receipt"]
        generation_probe = n3_generation["generation_probe"]
    else:
        generation_job_receipt = _prior_receipt_descriptor(D1_GENERATION_JOB)
        generation_probe = _prior_artifact_descriptor(D1_GENERATION_JOB)
    require(isinstance(generation_job_receipt, Mapping), "generation receipt binding is missing")
    require(isinstance(generation_probe, Mapping), "generation probe binding is missing")
    return {
        "job_id": job.job_id,
        "released": True,
        "source_commit": commit,
        "role": job.role,
        "argv": _authority_argv(
            job,
            commit,
            generation_job_receipt=generation_job_receipt,
            generation_probe=generation_probe,
        ),
        "max_wall_seconds": 3600,
        "publish_log_tail_bytes": 8192,
    }


def build_authority_wave(
    study_commit: str,
    *,
    n3_generation_job_receipt_path: Path,
    n3_generation_job_receipt_sha256: str,
) -> dict[str, Any]:
    """Build the two timing-only jobs after attempt 002 actually passes."""

    commit = _verified_commit(study_commit)
    n3_generation = validate_n3_generation_job_receipt(
        n3_generation_job_receipt_path, n3_generation_job_receipt_sha256
    )
    return {
        "schema_version": AUTHORITY_DESCRIPTOR_SCHEMA,
        "namespace": NAMESPACE,
        "source_commit": commit,
        "status": "descriptor_only_not_dispatched_receipt_gate_passed",
        "generation_requests_issued_by_wave": 0,
        "behavioral_actions_executed": 0,
        "safe_to_release_confirmation": False,
        "evidence_gate": {
            "n3_generation_attempt": dict(n3_generation["job_receipt"]),
            "n3_source_audit_job": _prior_receipt_descriptor(N3_SOURCE_AUDIT_JOB),
            "d1_source_audit_job": _prior_receipt_descriptor(D1_SOURCE_AUDIT_JOB),
            "d1_generation_normalization_job": _prior_receipt_descriptor(
                D1_GENERATION_JOB
            ),
            "recorder_receipt": {
                "path": str(RECORDER_RECEIPT),
                "sha256": RECORDER_RECEIPT_SHA256,
            },
        },
        "jobs": [
            build_authority_descriptor(job, commit, n3_generation=n3_generation)
            for job in AUTHORITY_JOBS
        ],
        "claim_boundary": (
            "Two detached source/probe/native-clock timing authority jobs only. They issue no "
            "new model request or robot action, and confirmation remains held after success."
        ),
    }


def _normalized_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": QUEUE_JOB_SCHEMA,
        "namespace": NAMESPACE,
        "job_id": value["job_id"],
        "released": True,
        "source_commit": value["source_commit"],
        "role": value["role"],
        "argv": list(value["argv"]),
        "max_wall_seconds": value["max_wall_seconds"],
        "publish_log_tail_bytes": value["publish_log_tail_bytes"],
    }


def _run_git(root: Path, *argv: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *argv],
            capture_output=True,
            text=True,
            timeout=60,
            env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never"),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise TimingQueueError("Git source verification is unavailable") from error
    require(result.returncode == 0, f"Git source verification failed: {argv[0]}")
    return result.stdout


@dataclass(frozen=True)
class QueueContext:
    source_root: Path
    job_dir: Path
    study_commit: str
    job_id: str
    role: str
    descriptor: dict[str, Any]
    descriptor_identity: dict[str, Any]
    claim_identity: dict[str, Any]
    worker_id: str
    hostname: str
    pod_uid: str


def validate_queue_context(
    *,
    source_root: Path,
    job_dir: Path,
    study_commit: str,
    job_id: str,
    expected_role: str,
    expected_descriptor: Mapping[str, Any],
    expected_worker_id: str | None = None,
) -> QueueContext:
    commit = _verified_commit(study_commit)
    _verified_safe_id(job_id, "job ID")
    _verified_safe_id(expected_role, "queue role")
    worker_id = _verified_safe_id(
        expected_role if expected_worker_id is None else expected_worker_id,
        "worker ID",
    )
    require(not Path(source_root).is_symlink(), "queue source root is a symlink")
    require(not Path(job_dir).is_symlink(), "queue job directory is a symlink")
    try:
        source = Path(source_root).resolve(strict=True)
        job = Path(job_dir).resolve(strict=True)
    except OSError as error:
        raise TimingQueueError("queue source or job directory is missing") from error
    require(source.is_dir() and job.is_dir(), "queue source or job path is not a directory")
    state = job.parent.parent
    require(job.parent == state / "jobs", "job directory is outside queue state")
    require(source.parent == state / "sources", "source checkout is outside queue state")
    require(source.name == commit, "source path is not named by the exact study commit")
    require(job.name == job_id, "job path is not named by the exact job ID")

    descriptor_path = job / "descriptor.json"
    descriptor = load_json(descriptor_path, "queue descriptor")
    wanted = _normalized_descriptor(expected_descriptor)
    require(descriptor == wanted, "queue descriptor differs from the frozen timing job")
    descriptor_identity = file_identity(descriptor_path)

    claim_path = job / "claim" / "owner.json"
    claim = load_json(claim_path, "queue claim owner")
    require(claim.get("worker_id") == worker_id, "queue claim worker identity changed")
    require(
        claim.get("descriptor_sha256") == descriptor_identity["sha256"],
        "queue claim does not bind the exact descriptor",
    )
    require(
        claim.get("release_boundary") == "claim_committed_under_shared_release_lock",
        "queue claim lacks the serialized release boundary",
    )
    require(type(claim.get("worker_pid")) is int and claim["worker_pid"] > 0,
            "queue claim worker PID is invalid")
    require(type(claim.get("control_generation")) is int and claim["control_generation"] >= 1,
            "queue claim generation is invalid")
    require(isinstance(claim.get("control_commit"), str)
            and COMMIT_RE.fullmatch(claim["control_commit"]) is not None,
            "queue claim control commit is invalid")

    hostname = socket.gethostname()
    pod_uid = os.environ.get("POD_UID", "")
    require(hostname.startswith(worker_id + "-"), "runtime hostname does not match worker identity")
    require(bool(pod_uid), "runtime POD_UID is missing")
    require(_run_git(source, "rev-parse", "HEAD").strip() == commit,
            "staged source checkout commit changed")
    require(not _run_git(source, "status", "--porcelain=v1", "--untracked-files=all"),
            "staged source checkout is dirty")
    return QueueContext(
        source_root=source,
        job_dir=job,
        study_commit=commit,
        job_id=job_id,
        role=expected_role,
        descriptor=descriptor,
        descriptor_identity=descriptor_identity,
        claim_identity=file_identity(claim_path),
        worker_id=claim["worker_id"],
        hostname=hostname,
        pod_uid=pod_uid,
    )


def _validate_staged_implementation(source_root: Path, *, include_n3_runner: bool = False) -> dict[str, Any]:
    tool = file_identity(source_root / TOOL_RELATIVE)
    contract = file_identity(source_root / CONTRACT_RELATIVE)
    require(tool["sha256"] == TOOL_SHA256, "staged timing validator hash changed")
    require(contract["sha256"] == CONTRACT_SHA256, "staged timing contract hash changed")
    result = {"timing_validator": tool, "timing_contract": contract}
    if include_n3_runner:
        runner = file_identity(source_root / N3_RUNNER_RELATIVE)
        runtime_contract = file_identity(source_root / N3_RUNTIME_CONTRACT_RELATIVE)
        require(runner["sha256"] == N3_RUNNER_SHA256, "staged N3 generation runner hash changed")
        require(runtime_contract["sha256"] == N3_RUNTIME_CONTRACT_SHA256,
                "staged N3 runtime contract hash changed")
        result["n3_generation_runner"] = runner
        result["n3_runtime_contract"] = runtime_contract
    return result


def _validate_exact_file(path: Path, expected_path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    require(Path(path) == expected_path, f"{label} path changed")
    identity = file_identity(path)
    require(identity["sha256"] == _verified_sha(expected_sha256, f"{label} digest"),
            f"{label} hash changed")
    return identity


def _validate_prior_timing_job(job: PriorTimingJob) -> dict[str, Any]:
    receipt_identity = _validate_exact_file(
        job.receipt_path, job.receipt_path, job.receipt_sha256, f"{job.model} prior job receipt"
    )
    receipt = load_json(job.receipt_path, f"{job.model} prior job receipt")
    verify_signed_document(receipt, f"{job.model} prior job receipt")
    _validate_timing_receipt_header(
        receipt,
        mode=job.mode,
        job_id=job.job_id,
        job_dir=job.job_dir,
        role=job.role,
        worker_id=job.worker_id,
        study_commit=job.study_commit,
    )
    require(
        receipt.get("science_counts")
        == _expected_science_counts(issued=0, referenced=job.referenced_generation_requests),
        f"{job.model} prior job receipt science counts changed",
    )
    _validate_receipt_implementation(
        receipt, study_commit=job.study_commit, include_n3_runner=False
    )
    outputs = receipt.get("outputs")
    require(isinstance(outputs, Mapping), f"{job.model} prior job outputs are missing")
    _require_bound_descriptor(
        outputs.get(job.output_key),
        expected_path=job.artifact_path,
        expected_sha256=job.artifact_sha256,
        expected_bytes=job.artifact_bytes,
        label=f"{job.model} prior raw artifact",
    )
    published_path = job.job_dir / "publish" / job.artifact_name
    _require_bound_descriptor(
        outputs.get(job.published_output_key),
        expected_path=published_path,
        expected_sha256=job.artifact_sha256,
        expected_bytes=job.artifact_bytes,
        label=f"{job.model} prior published artifact",
    )
    artifact_identity = _validate_exact_file(
        job.artifact_path,
        job.artifact_path,
        job.artifact_sha256,
        f"{job.model} prior raw artifact",
    )
    require(
        artifact_identity["bytes"] == job.artifact_bytes,
        f"{job.model} prior raw artifact byte count changed",
    )
    published_identity = _validate_exact_file(
        published_path,
        published_path,
        job.artifact_sha256,
        f"{job.model} prior published artifact",
    )
    require(
        published_identity["bytes"] == job.artifact_bytes,
        f"{job.model} prior published artifact byte count changed",
    )
    return {
        "job_receipt": receipt_identity,
        "artifact": artifact_identity,
        "published_artifact": published_identity,
    }


def _runtime_authority_descriptor(
    args: argparse.Namespace,
) -> tuple[AuthorityJob, dict[str, Any]]:
    job = AUTHORITY_BY_MODE.get(args.command)
    require(job is not None, "runtime mode is outside the frozen native-authority wave")
    require(args.job_id == job.job_id, "runtime authority job ID changed")
    require(args.expected_role == job.role, "runtime authority queue role changed")
    require(args.contract_sha256 == CONTRACT_SHA256, "runtime timing contract hash changed")
    require(args.camera_id == CAMERA_ID, "runtime authority camera changed")
    require(
        args.expected_target_count == job.expected_target_count,
        "runtime authority target count changed",
    )
    source_job = N3_SOURCE_AUDIT_JOB if job.model == "N3" else D1_SOURCE_AUDIT_JOB
    require(
        Path(args.source_audit_job_receipt) == source_job.receipt_path,
        "runtime source-audit job receipt path changed",
    )
    require(
        args.source_audit_job_receipt_sha256 == source_job.receipt_sha256,
        "runtime source-audit job receipt hash changed",
    )
    require(Path(args.source_audit) == source_job.artifact_path, "runtime source audit path changed")
    require(
        args.source_audit_sha256 == source_job.artifact_sha256,
        "runtime source audit hash changed",
    )
    require(Path(args.recorder_receipt) == RECORDER_RECEIPT, "runtime recorder receipt path changed")
    require(
        args.recorder_receipt_sha256 == RECORDER_RECEIPT_SHA256,
        "runtime recorder receipt hash changed",
    )
    if job.model == "N3":
        require(
            Path(args.generation_job_receipt) == N3_GENERATION_CLUSTER_JOB_RECEIPT,
            "runtime N3 generation job receipt path changed",
        )
        _verified_sha(args.generation_job_receipt_sha256, "N3 generation job receipt digest")
        expected_probe = N3_GENERATION_CLUSTER_JOB_DIR / "raw" / "n3_generation_probe.json"
        require(Path(args.generation_probe) == expected_probe, "runtime N3 generation probe path changed")
        _verified_sha(args.generation_probe_sha256, "N3 generation probe digest")
    else:
        require(
            Path(args.generation_job_receipt) == D1_GENERATION_JOB.receipt_path,
            "runtime D1 generation job receipt path changed",
        )
        require(
            args.generation_job_receipt_sha256 == D1_GENERATION_JOB.receipt_sha256,
            "runtime D1 generation job receipt hash changed",
        )
        require(
            Path(args.generation_probe) == D1_GENERATION_JOB.artifact_path,
            "runtime D1 generation probe path changed",
        )
        require(
            args.generation_probe_sha256 == D1_GENERATION_JOB.artifact_sha256,
            "runtime D1 generation probe hash changed",
        )
    descriptor = {
        "job_id": job.job_id,
        "released": True,
        "source_commit": _verified_commit(args.study_commit),
        "role": job.role,
        "argv": _authority_argv(
            job,
            args.study_commit,
            generation_job_receipt={
                "path": str(Path(args.generation_job_receipt)),
                "sha256": args.generation_job_receipt_sha256,
            },
            generation_probe={
                "path": str(Path(args.generation_probe)),
                "sha256": args.generation_probe_sha256,
            },
        ),
        "max_wall_seconds": 3600,
        "publish_log_tail_bytes": 8192,
    }
    return job, descriptor


def _authority_prerequisites(
    args: argparse.Namespace, job: AuthorityJob
) -> dict[str, Any]:
    source_job = N3_SOURCE_AUDIT_JOB if job.model == "N3" else D1_SOURCE_AUDIT_JOB
    source = _validate_prior_timing_job(source_job)
    require(
        source["job_receipt"]["sha256"] == args.source_audit_job_receipt_sha256
        and source["artifact"]["sha256"] == args.source_audit_sha256,
        "runtime source-audit inputs differ from their prior receipt",
    )
    if job.model == "N3":
        receipt_identity = _validate_exact_file(
            Path(args.generation_job_receipt),
            N3_GENERATION_CLUSTER_JOB_RECEIPT,
            args.generation_job_receipt_sha256,
            "N3 generation job receipt",
        )
        generation = validate_n3_generation_job_receipt(
            Path(args.generation_job_receipt), args.generation_job_receipt_sha256
        )
        probe = generation["generation_probe"]
        require(isinstance(probe, Mapping), "N3 generation probe binding is missing")
        require(
            Path(args.generation_probe) == Path(str(probe["path"]))
            and args.generation_probe_sha256 == probe["sha256"],
            "runtime N3 probe differs from attempt 002 receipt",
        )
        probe_identity = _validate_exact_file(
            Path(args.generation_probe),
            Path(str(probe["path"])),
            args.generation_probe_sha256,
            "N3 generation probe",
        )
        require(
            probe_identity["bytes"] == probe["bytes"],
            "runtime N3 generation probe byte count changed",
        )
        generation_input = {
            "job_receipt": receipt_identity,
            "generation_probe": probe_identity,
            "qualification": dict(generation["qualification"]),
        }
    else:
        generation_input = _validate_prior_timing_job(D1_GENERATION_JOB)
        require(
            generation_input["job_receipt"]["sha256"]
            == args.generation_job_receipt_sha256
            and generation_input["artifact"]["sha256"] == args.generation_probe_sha256,
            "runtime D1 probe differs from its normalization receipt",
        )
    recorder = _validate_exact_file(
        Path(args.recorder_receipt),
        RECORDER_RECEIPT,
        args.recorder_receipt_sha256,
        "recorder qualification receipt",
    )
    return {
        "source_audit_job": source["job_receipt"],
        "source_audit": source["artifact"],
        "generation_job": generation_input["job_receipt"],
        "generation_probe": (
            generation_input["generation_probe"]
            if job.model == "N3"
            else generation_input["artifact"]
        ),
        "recorder_receipt": recorder,
        "camera_id": CAMERA_ID,
    }


def _load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    require(specification is not None and specification.loader is not None,
            f"cannot load staged module: {path.name}")
    module = importlib.util.module_from_spec(specification)
    # dataclasses with postponed annotations resolve their module through
    # sys.modules while the class decorator runs.
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _prepare_output_directories(job_dir: Path) -> tuple[Path, Path]:
    raw = Path(job_dir) / "raw"
    publish = Path(job_dir) / "publish"
    for path in (raw, publish):
        require(not path.exists() and not path.is_symlink(),
                f"immutable timing output already exists: {path.name}")
    raw.mkdir()
    publish.mkdir()
    _fsync_directory(Path(job_dir))
    return raw, publish


def _copy_compact(source: Path, destination: Path) -> dict[str, Any]:
    source_identity = file_identity(source)
    require(source_identity["bytes"] <= 8 * 1024 * 1024,
            "compact timing artifact exceeds the publication bound")
    immutable_bytes(destination, Path(source).read_bytes(), maximum_bytes=8 * 1024 * 1024)
    copied = file_identity(destination)
    require(copied["sha256"] == source_identity["sha256"]
            and copied["bytes"] == source_identity["bytes"],
            "published timing artifact differs from raw evidence")
    return copied


def _science_counts(*, referenced_generation_requests: int = 0,
                    model_requests_issued_by_job: int = 0) -> dict[str, int]:
    return {
        "model_runtime_loads": 1 if model_requests_issued_by_job else 0,
        "model_servers_started": 0,
        "model_requests_issued_by_job": model_requests_issued_by_job,
        "referenced_generation_requests": referenced_generation_requests,
        "physical_resets": 0,
        "robot_episodes": 0,
        "behavioral_actions": 0,
        "behavioral_cells": 0,
    }


def _receipt_base(context: QueueContext, mode: str, implementation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": TIMING_JOB_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "mode": mode,
        "job_id": context.job_id,
        "job_dir": str(context.job_dir),
        "study_commit": context.study_commit,
        "queue_role": context.role,
        "worker_id": context.worker_id,
        "runtime_identity": {
            "hostname": context.hostname,
            "pod_uid": context.pod_uid,
            "pid": os.getpid(),
        },
        "queue_descriptor": context.descriptor_identity,
        "queue_claim": context.claim_identity,
        "implementation": dict(implementation),
        "physical_time_qualified": False,
        "behavioral_policy_skill_evaluated": False,
    }


def _write_failure_receipt(
    *, context: QueueContext | None, job_dir: Path, mode: str, error: BaseException,
    science_counts: Mapping[str, Any] | None = None,
) -> None:
    job = Path(job_dir).resolve()
    publish = job / "publish"
    try:
        if publish.exists() or publish.is_symlink():
            require(publish.is_dir() and not publish.is_symlink(), "failure publish path is invalid")
        else:
            publish.mkdir()
        target = publish / "timing_job_failure.json"
        if target.exists() or target.is_symlink():
            return
        value = signed_document({
            "schema_version": TIMING_JOB_SCHEMA,
            "namespace": NAMESPACE,
            "study_id": STUDY_ID,
            "status": "technical_invalid",
            "decision": "no_go",
            "mode": mode,
            "job_id": context.job_id if context is not None else job.name,
            "job_dir": str(job),
            "study_commit": context.study_commit if context is not None else None,
            "queue_role": context.role if context is not None else None,
            "queue_descriptor": context.descriptor_identity if context is not None else None,
            "queue_claim": context.claim_identity if context is not None else None,
            "failure": {
                "error_type": type(error).__name__,
                "detail": str(error)[:2000],
                "traceback": traceback.format_exc(limit=20)[-12000:],
            },
            "science_counts": dict(science_counts or _science_counts()),
            "physical_time_qualified": False,
            "behavioral_policy_skill_evaluated": False,
            "safe_to_release_confirmation": False,
            "completed_at_utc": utc_now(),
            "claim_boundary": (
                "Technical-invalid timing queue attempt only; no behavioral evidence or "
                "forecast accuracy is claimed."
            ),
        })
        immutable_json(target, value)
    except BaseException:
        # The outer queue result and full stderr still retain the failure.
        return


def _expected_initial_descriptor(args: argparse.Namespace) -> tuple[InitialJob, dict[str, Any]]:
    job = INITIAL_BY_MODE.get(args.command)
    require(job is not None, "runtime mode is outside the frozen initial timing wave")
    require(args.job_id == job.job_id, "runtime job ID changed")
    require(args.expected_role == job.role, "runtime queue role changed")
    require(args.contract_sha256 == CONTRACT_SHA256, "runtime timing contract hash changed")
    if job.mode in {"n3-source-audit", "d1-source-audit"}:
        wanted_source = N3_SOURCE if job.model == "N3" else D1_SOURCE
        require(Path(args.external_source_root) == wanted_source, "external source path changed")
    elif job.mode == "n3-live-input":
        require(Path(args.capture_receipt) == CAPTURE_RECEIPT, "P00 capture path changed")
        require(args.capture_receipt_sha256 == CAPTURE_RECEIPT_SHA256, "P00 capture hash changed")
        require(args.camera_id == CAMERA_ID, "selected original camera changed")
    elif job.mode == "d1-normalize-generation":
        require(Path(args.qualification_receipt) == D1_QUALIFICATION_RECEIPT,
                "D1 qualification path changed")
        require(args.qualification_receipt_sha256 == D1_QUALIFICATION_RECEIPT_SHA256,
                "D1 qualification hash changed")
    return job, build_initial_descriptor(job, args.study_commit)


def run_initial_job(args: argparse.Namespace) -> dict[str, Any]:
    job, expected_descriptor = _expected_initial_descriptor(args)
    context: QueueContext | None = None
    try:
        context = validate_queue_context(
            source_root=args.source_root,
            job_dir=args.job_dir,
            study_commit=args.study_commit,
            job_id=args.job_id,
            expected_role=args.expected_role,
            expected_descriptor=expected_descriptor,
        )
        implementation = _validate_staged_implementation(context.source_root)
        timing = _load_module(
            context.source_root / TOOL_RELATIVE,
            f"wmf_forecast_timing_{context.job_id.replace('-', '_')}",
        )
        raw, publish = _prepare_output_directories(context.job_dir)
        outputs: dict[str, Any]
        inputs: dict[str, Any]

        if job.mode in {"n3-source-audit", "d1-source-audit"}:
            model = str(job.model)
            external_source = Path(args.external_source_root)
            result = timing.audit_source(
                model_id=model,
                source_root=external_source,
                contract_path=context.source_root / CONTRACT_RELATIVE,
                contract_sha256=CONTRACT_SHA256,
            )
            require(result.get("physical_time_qualified") is False,
                    "source audit improperly qualified physical time")
            primary_path = raw / f"{model.lower()}_source_audit.json"
            timing.atomic_json(primary_path, result)
            inputs = {"external_source_root": str(external_source.resolve())}
            outputs = {
                "primary": file_identity(primary_path),
                "published_primary": _copy_compact(primary_path, publish / primary_path.name),
            }
        elif job.mode == "n3-live-input":
            capture = _validate_exact_file(
                Path(args.capture_receipt), CAPTURE_RECEIPT,
                args.capture_receipt_sha256, "P00 capture receipt",
            )
            bundle = raw / "n3_live_input"
            timing.prepare_n3_live_input(
                capture_path=Path(args.capture_receipt),
                capture_sha256=args.capture_receipt_sha256,
                camera_id=args.camera_id,
                output_dir=bundle,
            )
            primary_path = bundle / "preparation_receipt.json"
            primary_value = load_json(primary_path, "N3 preparation receipt")
            timing.verify_signed(primary_value, "N3 preparation receipt")
            require(primary_value.get("model_request_count") == 0
                    and primary_value.get("behavioral_action_count") == 0,
                    "N3 preparation performed model or behavioral work")
            manifest_path = bundle / "observation_manifest.json"
            payload_path = bundle / "observation.npz"
            inputs = {"capture_receipt": capture, "camera_id": args.camera_id}
            outputs = {
                "primary": file_identity(primary_path),
                "observation_manifest": file_identity(manifest_path),
                "observation_payload": file_identity(payload_path),
                "published_primary": _copy_compact(
                    primary_path, publish / "preparation_receipt.json"
                ),
            }
        elif job.mode == "d1-normalize-generation":
            qualification = _validate_exact_file(
                Path(args.qualification_receipt), D1_QUALIFICATION_RECEIPT,
                args.qualification_receipt_sha256, "D1 qualification receipt",
            )
            result = timing.normalize_generation_probe(
                model_id="D1",
                qualification_path=Path(args.qualification_receipt),
                qualification_sha256=args.qualification_receipt_sha256,
                contract_path=context.source_root / CONTRACT_RELATIVE,
                contract_sha256=CONTRACT_SHA256,
            )
            require(result.get("model_returned_action_executed") is False,
                    "D1 normalization reports an executed model action")
            primary_path = raw / "d1_generation_probe.json"
            timing.atomic_json(primary_path, result)
            timing.validate_generation_probe(
                primary_path, sha256_file(primary_path), expected_model="D1"
            )
            inputs = {"qualification_receipt": qualification}
            outputs = {
                "primary": file_identity(primary_path),
                "published_primary": _copy_compact(
                    primary_path, publish / "d1_generation_probe.json"
                ),
            }
        else:  # pragma: no cover
            raise TimingQueueError(f"unsupported initial timing mode: {job.mode}")

        receipt = signed_document({
            **_receipt_base(context, job.mode, implementation),
            "status": "passed",
            "decision": "go",
            "inputs": inputs,
            "outputs": outputs,
            "science_counts": _science_counts(
                referenced_generation_requests=job.referenced_generation_requests
            ),
            "safe_to_release_confirmation": False,
            "completed_at_utc": utc_now(),
            "claim_boundary": (
                "Detached timing prerequisite evidence only. This job did not issue a model "
                "request, execute a model action, or produce behavioral evidence."
            ),
        })
        immutable_json(publish / "timing_job_receipt.json", receipt)
        return receipt
    except BaseException as error:
        _write_failure_receipt(
            context=context,
            job_dir=Path(args.job_dir),
            mode=getattr(args, "command", "unknown"),
            error=error,
        )
        raise


def _n3_prerequisite_from_runtime(args: argparse.Namespace) -> dict[str, Any]:
    require(Path(args.preparation_job_receipt) == PREPARATION_CLUSTER_JOB_RECEIPT,
            "preparation job receipt path changed")
    receipt_identity = _validate_exact_file(
        Path(args.preparation_job_receipt), PREPARATION_CLUSTER_JOB_RECEIPT,
        args.preparation_job_receipt_sha256, "preparation job receipt",
    )
    evidence = validate_preparation_job_receipt(
        Path(args.preparation_job_receipt), args.preparation_job_receipt_sha256
    )
    preparation = evidence["preparation_receipt"]
    manifest = evidence["observation_manifest"]
    assert isinstance(preparation, Mapping) and isinstance(manifest, Mapping)
    require(Path(args.preparation_receipt) == Path(str(preparation["path"])),
            "runtime preparation receipt path differs from its job receipt")
    require(args.preparation_receipt_sha256 == preparation["sha256"],
            "runtime preparation receipt hash differs from its job receipt")
    require(Path(args.observation_manifest) == Path(str(manifest["path"])),
            "runtime observation manifest path differs from its job receipt")
    require(args.observation_manifest_sha256 == manifest["sha256"],
            "runtime observation manifest hash differs from its job receipt")
    preparation_identity = _validate_exact_file(
        Path(args.preparation_receipt), Path(str(preparation["path"])),
        args.preparation_receipt_sha256, "N3 preparation receipt",
    )
    manifest_identity = _validate_exact_file(
        Path(args.observation_manifest), Path(str(manifest["path"])),
        args.observation_manifest_sha256, "N3 observation manifest",
    )
    return {
        "preparation_job_receipt": receipt_identity,
        "preparation_receipt": preparation_identity,
        "observation_manifest": manifest_identity,
        "observation_payload": evidence["observation_payload"],
    }


def _runtime_n3_descriptor(args: argparse.Namespace) -> dict[str, Any]:
    require(args.job_id == N3_GENERATION_JOB_ID, "runtime N3 generation job ID changed")
    require(args.expected_role == N3_GENERATION_ROLE, "runtime N3 generation role changed")
    require(args.expected_worker_id == N3_GENERATION_WORKER_ID,
            "runtime N3 generation worker identity changed")
    require(args.contract_sha256 == CONTRACT_SHA256, "runtime timing contract hash changed")
    preparation = {
        "path": str(Path(args.preparation_receipt)),
        "sha256": args.preparation_receipt_sha256,
    }
    manifest = {
        "path": str(Path(args.observation_manifest)),
        "sha256": args.observation_manifest_sha256,
    }
    return {
        "job_id": N3_GENERATION_JOB_ID,
        "released": True,
        "source_commit": _verified_commit(args.study_commit),
        "role": N3_GENERATION_ROLE,
        "argv": _n3_generation_shell_argv(
            args.study_commit,
            preparation_job_receipt_sha256=args.preparation_job_receipt_sha256,
            preparation_receipt=preparation,
            observation_manifest=manifest,
        ),
        "max_wall_seconds": 7200,
        "publish_log_tail_bytes": 8192,
    }


def run_n3_generation_job(args: argparse.Namespace) -> dict[str, Any]:
    expected_descriptor = _runtime_n3_descriptor(args)
    context: QueueContext | None = None
    child: subprocess.CompletedProcess[bytes] | None = None
    try:
        context = validate_queue_context(
            source_root=args.source_root,
            job_dir=args.job_dir,
            study_commit=args.study_commit,
            job_id=args.job_id,
            expected_role=args.expected_role,
            expected_descriptor=expected_descriptor,
            expected_worker_id=args.expected_worker_id,
        )
        implementation = _validate_staged_implementation(
            context.source_root, include_n3_runner=True
        )
        prerequisites = _n3_prerequisite_from_runtime(args)
        timing = _load_module(
            context.source_root / TOOL_RELATIVE,
            f"wmf_forecast_timing_{context.job_id.replace('-', '_')}",
        )
        n3 = _load_module(
            context.source_root / N3_RUNNER_RELATIVE,
            f"wmf_n3_generation_{context.job_id.replace('-', '_')}",
        )
        _, fixed = n3.load_fixed_observation(Path(args.observation_manifest))
        preparation_value = load_json(Path(args.preparation_receipt), "N3 preparation receipt")
        timing.verify_signed(preparation_value, "N3 preparation receipt")
        require(preparation_value.get("schema_version") == "wmf-n3-live-zero-policy-input-v1"
                and preparation_value.get("status")
                == "live_zero_policy_input_prepared_generation_not_yet_run",
                "N3 preparation receipt did not pass")
        require(preparation_value.get("model_request_count") == 0
                and preparation_value.get("behavioral_action_count") == 0,
                "N3 preparation receipt contains scientific actions")
        require(fixed.get("wire_observation_sha256")
                == preparation_value.get("wire_observation_sha256"),
                "N3 runtime input differs from the preparation receipt")

        raw, publish = _prepare_output_directories(context.job_dir)
        generation_raw = raw / "n3_generation_raw"
        generation_publish = raw / "n3_generation_publish"
        child_stdout = raw / "n3_generation.stdout.log"
        child_stderr = raw / "n3_generation.stderr.log"
        command = [
            str(COSMOS_PYTHON),
            str(context.source_root / N3_RUNNER_RELATIVE),
            "--source-root",
            str(N3_SOURCE),
            "--checkpoint-root",
            str(N3_CHECKPOINT),
            "--observation-manifest",
            str(Path(args.observation_manifest)),
            "--output-dir",
            str(generation_raw),
            "--publish-dir",
            str(generation_publish),
            "--effective-seed",
            str(EFFECTIVE_SEED),
        ]
        with child_stdout.open("xb") as stdout, child_stderr.open("xb") as stderr:
            child = subprocess.run(
                command,
                cwd=N3_SOURCE,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1"),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
        require(child.returncode == 0, "N3 six-request generation child failed")

        qualification_path = generation_publish / "n3_qualification.json"
        qualification = load_json(qualification_path, "N3 compact qualification")
        require(qualification.get("schema_version") == "wmf-n3-runtime-qualification-v1"
                and qualification.get("status") == "passed"
                and qualification.get("qualified") is True,
                "N3 compact qualification did not pass")
        require(qualification.get("generation_request_count") == 6
                and qualification.get("robot_episode_count") == 0,
                "N3 qualification is not a six-request zero-episode probe")

        normalized = timing.normalize_generation_probe(
            model_id="N3",
            qualification_path=qualification_path,
            qualification_sha256=sha256_file(qualification_path),
            contract_path=context.source_root / CONTRACT_RELATIVE,
            contract_sha256=CONTRACT_SHA256,
        )
        require(normalized.get("model_returned_action_executed") is False,
                "N3 generation normalization reports an executed action")
        normalized_path = raw / "n3_generation_probe.json"
        timing.atomic_json(normalized_path, normalized)
        timing.validate_generation_probe(
            normalized_path, sha256_file(normalized_path), expected_model="N3"
        )
        outputs = {
            "qualification": file_identity(qualification_path),
            "normalized_generation_probe": file_identity(normalized_path),
            "child_stdout": file_identity(child_stdout),
            "child_stderr": file_identity(child_stderr),
            "published_qualification": _copy_compact(
                qualification_path, publish / "n3_qualification.json"
            ),
            "published_generation_probe": _copy_compact(
                normalized_path, publish / "n3_generation_probe.json"
            ),
        }
        receipt = signed_document({
            **_receipt_base(context, args.command, implementation),
            "status": "passed",
            "decision": "go",
            "inputs": prerequisites,
            "outputs": outputs,
            "child": {
                "argv": command,
                "returncode": child.returncode,
                "reaped": True,
            },
            "science_counts": _science_counts(model_requests_issued_by_job=6),
            "safe_to_release_confirmation": False,
            "completed_at_utc": utc_now(),
            "claim_boundary": (
                "Six N3 generation requests against one live P00 fixed input, with retained "
                "actions/latents/decodes and zero executed robot actions or behavioral episodes."
            ),
        })
        immutable_json(publish / "timing_job_receipt.json", receipt)
        return receipt
    except BaseException as error:
        failure_counts: dict[str, Any] = {
            "n3_generation_child_started": child is not None,
            "model_runtime_loads": None if child is not None else 0,
            "model_servers_started": 0,
            "model_requests_issued_by_job": None if child is not None else 0,
            "model_requests_completed_before_failure": None if child is not None else 0,
            "referenced_generation_requests": 0,
            "physical_resets": 0,
            "robot_episodes": 0,
            "behavioral_actions": 0,
            "behavioral_cells": 0,
        }
        failure_path = (
            Path(args.job_dir) / "raw" / "n3_generation_publish"
            / "n3_qualification_failure.json"
        )
        try:
            failure = load_json(failure_path, "N3 generation child failure")
            completed = failure.get("generation_requests_completed_before_failure")
            if type(completed) is int and 0 <= completed <= 6:
                failure_counts["model_requests_completed_before_failure"] = completed
        except BaseException:
            pass
        _write_failure_receipt(
            context=context,
            job_dir=Path(args.job_dir),
            mode=getattr(args, "command", "n3-generate"),
            error=error,
            science_counts=failure_counts,
        )
        raise


def run_authority_job(args: argparse.Namespace) -> dict[str, Any]:
    """Qualify source-mapped targets against native clocks; issue no requests/actions."""

    job, expected_descriptor = _runtime_authority_descriptor(args)
    context: QueueContext | None = None
    try:
        context = validate_queue_context(
            source_root=args.source_root,
            job_dir=args.job_dir,
            study_commit=args.study_commit,
            job_id=args.job_id,
            expected_role=args.expected_role,
            expected_descriptor=expected_descriptor,
        )
        implementation = _validate_staged_implementation(context.source_root)
        prerequisites = _authority_prerequisites(args, job)
        timing = _load_module(
            context.source_root / TOOL_RELATIVE,
            f"wmf_forecast_authority_{context.job_id.replace('-', '_')}",
        )
        raw, publish = _prepare_output_directories(context.job_dir)
        authority = timing.qualify_timing(
            model_id=job.model,
            contract_path=context.source_root / CONTRACT_RELATIVE,
            contract_sha256=CONTRACT_SHA256,
            source_audit_path=Path(args.source_audit),
            source_audit_sha256=args.source_audit_sha256,
            generation_probe_path=Path(args.generation_probe),
            generation_probe_sha256=args.generation_probe_sha256,
            recorder_receipt_path=Path(args.recorder_receipt),
            recorder_receipt_sha256=args.recorder_receipt_sha256,
            camera_id=args.camera_id,
        )
        targets = authority.get("generated_targets")
        require(
            isinstance(targets, list) and len(targets) == job.expected_target_count,
            f"{job.model} native authority target count changed",
        )
        authority_path = raw / f"{job.model.lower()}_timing_authority.json"
        timing.atomic_json(authority_path, authority)
        validated = timing.validate_timing_authority(
            authority_path,
            sha256_file(authority_path),
            expected_model=job.model,
        )
        require(
            len(validated.get("generated_targets", [])) == job.expected_target_count,
            f"{job.model} deep authority validation changed target count",
        )
        outputs = {
            "timing_authority": file_identity(authority_path),
            "published_timing_authority": _copy_compact(
                authority_path, publish / authority_path.name
            ),
        }
        receipt = signed_document(
            {
                **_receipt_base(context, job.mode, implementation),
                "status": "passed",
                "decision": "go",
                "inputs": prerequisites,
                "outputs": outputs,
                "science_counts": _science_counts(referenced_generation_requests=6),
                "evidence_counts": {
                    "qualified_generated_targets": job.expected_target_count,
                    "referenced_recorder_actions": 450,
                    "referenced_recorder_observations": 451,
                },
                "physical_time_qualified": True,
                "safe_to_release_confirmation": False,
                "completed_at_utc": utc_now(),
                "claim_boundary": (
                    "Source/probe-backed native physical timing authority only. This job issued "
                    "zero new model requests, executed zero robot actions, evaluated no policy "
                    "skill, and does not release confirmation."
                ),
            }
        )
        immutable_json(publish / "timing_job_receipt.json", receipt)
        return receipt
    except BaseException as error:
        _write_failure_receipt(
            context=context,
            job_dir=Path(args.job_dir),
            mode=getattr(args, "command", "native-authority"),
            error=error,
        )
        raise


def _write_descriptor_output(path: Path | None, value: Mapping[str, Any]) -> None:
    if path is None:
        print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
        return
    immutable_json(path, value, maximum_bytes=2 * 1024 * 1024)
    print(json.dumps({"output": str(Path(path).resolve()), "sha256": sha256_file(path)}, sort_keys=True))


def _add_runtime_base(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--study-commit", required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-role", required=True)
    parser.add_argument("--contract-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    initial = commands.add_parser("emit-initial", help="emit, but do not dispatch, four descriptors")
    initial.add_argument("--study-commit", required=True)
    initial.add_argument("--output", type=Path)

    follow_on = commands.add_parser(
        "emit-n3-generation", help="emit the N3 descriptor only after a passed preparation receipt"
    )
    follow_on.add_argument("--study-commit", required=True)
    follow_on.add_argument("--preparation-job-receipt", type=Path, required=True)
    follow_on.add_argument("--preparation-job-receipt-sha256", required=True)
    follow_on.add_argument("--output", type=Path)

    authority_wave = commands.add_parser(
        "emit-authority-wave",
        help="emit two timing-only authority jobs after N3 attempt 002 passes",
    )
    authority_wave.add_argument("--study-commit", required=True)
    authority_wave.add_argument("--n3-generation-job-receipt", type=Path, required=True)
    authority_wave.add_argument("--n3-generation-job-receipt-sha256", required=True)
    authority_wave.add_argument("--output", type=Path)

    for mode in ("n3-source-audit", "d1-source-audit"):
        command = commands.add_parser(mode)
        _add_runtime_base(command)
        command.add_argument("--external-source-root", type=Path, required=True)

    prepare = commands.add_parser("n3-live-input")
    _add_runtime_base(prepare)
    prepare.add_argument("--capture-receipt", type=Path, required=True)
    prepare.add_argument("--capture-receipt-sha256", required=True)
    prepare.add_argument("--camera-id", required=True)

    normalize = commands.add_parser("d1-normalize-generation")
    _add_runtime_base(normalize)
    normalize.add_argument("--qualification-receipt", type=Path, required=True)
    normalize.add_argument("--qualification-receipt-sha256", required=True)

    generation = commands.add_parser("n3-generate")
    _add_runtime_base(generation)
    generation.add_argument("--expected-worker-id", required=True)
    generation.add_argument("--preparation-job-receipt", type=Path, required=True)
    generation.add_argument("--preparation-job-receipt-sha256", required=True)
    generation.add_argument("--preparation-receipt", type=Path, required=True)
    generation.add_argument("--preparation-receipt-sha256", required=True)
    generation.add_argument("--observation-manifest", type=Path, required=True)
    generation.add_argument("--observation-manifest-sha256", required=True)

    for mode in ("n3-native-authority", "d1-native-authority"):
        authority = commands.add_parser(mode)
        _add_runtime_base(authority)
        authority.add_argument("--source-audit-job-receipt", type=Path, required=True)
        authority.add_argument("--source-audit-job-receipt-sha256", required=True)
        authority.add_argument("--source-audit", type=Path, required=True)
        authority.add_argument("--source-audit-sha256", required=True)
        authority.add_argument("--generation-job-receipt", type=Path, required=True)
        authority.add_argument("--generation-job-receipt-sha256", required=True)
        authority.add_argument("--generation-probe", type=Path, required=True)
        authority.add_argument("--generation-probe-sha256", required=True)
        authority.add_argument("--recorder-receipt", type=Path, required=True)
        authority.add_argument("--recorder-receipt-sha256", required=True)
        authority.add_argument("--camera-id", required=True)
        authority.add_argument("--expected-target-count", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "emit-initial":
            _write_descriptor_output(args.output, build_initial_wave(args.study_commit))
            return 0
        if args.command == "emit-n3-generation":
            descriptor = build_n3_generation_descriptor(
                args.study_commit,
                preparation_job_receipt_path=args.preparation_job_receipt,
                preparation_job_receipt_sha256=args.preparation_job_receipt_sha256,
            )
            _write_descriptor_output(args.output, descriptor)
            return 0
        if args.command == "emit-authority-wave":
            wave = build_authority_wave(
                args.study_commit,
                n3_generation_job_receipt_path=args.n3_generation_job_receipt,
                n3_generation_job_receipt_sha256=args.n3_generation_job_receipt_sha256,
            )
            _write_descriptor_output(args.output, wave)
            return 0
        if args.command == "n3-generate":
            receipt = run_n3_generation_job(args)
        elif args.command in AUTHORITY_BY_MODE:
            receipt = run_authority_job(args)
        else:
            receipt = run_initial_job(args)
    except BaseException as error:
        print(json.dumps({
            "status": "technical_invalid",
            "error_type": type(error).__name__,
            "detail": str(error)[:2000],
        }, sort_keys=True), flush=True)
        return 3
    print(json.dumps({
        "status": receipt["status"],
        "mode": receipt["mode"],
        "job_id": receipt["job_id"],
        "science_counts": receipt["science_counts"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
