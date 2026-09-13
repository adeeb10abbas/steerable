#!/usr/bin/env python3
"""Build and run detached development-timing sidecar jobs.

These jobs inventory already-completed development requests and bind them to
the passed native timing authorities.  They start no model, simulator, reset,
episode, request, or action.  The N3 job is releasable only from the exact four
passed N3 development aggregates.  The D1 descriptor is deliberately withheld
until D01, D02, and D04 attempt003 have passed alongside the frozen D03
attempt002 aggregate.

Raw inventories and timing sidecars remain on the GM PVC.  Only a compact,
signed receipt is placed in ``publish``.  Every runtime invocation reopens its
queue descriptor/claim, exact source commit, both authority receipts, every
aggregate and cell receipt, and all transitive request/recorder evidence.
Confirmation remains held even after these jobs pass.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

FORECAST_ROOT = Path(__file__).resolve().parent
QUEUE_MODULE_PATH = FORECAST_ROOT / "forecast_timing_queue_jobs.py"


def _load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load workshop module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


queue = _load_module(QUEUE_MODULE_PATH, "wmf_forecast_timing_queue_support")

NAMESPACE = queue.NAMESPACE
STUDY_ID = queue.STUDY_ID
QUEUE_JOB_SCHEMA = queue.QUEUE_JOB_SCHEMA
CONTROL_ROOT = queue.CONTROL_ROOT
ROBOLAB_PYTHON = queue.ROBOLAB_PYTHON
TOOL_RELATIVE = queue.TOOL_RELATIVE
CONTRACT_RELATIVE = queue.CONTRACT_RELATIVE
TOOL_SHA256 = queue.TOOL_SHA256
CONTRACT_SHA256 = queue.CONTRACT_SHA256

THIS_RELATIVE = (
    queue.FORECAST_RELATIVE
    / "experiments/forecast_layout/development_timing_sidecar_jobs.py"
)
WAVE_SCHEMA = "wmf-development-timing-sidecar-wave-v1"
JOB_RECEIPT_SCHEMA = "wmf-development-timing-sidecar-queue-job-v1"
N3_AGGREGATE_SCHEMA = "wmf-n3-behavioral-development-job-v1"
D1_AGGREGATE_SCHEMA = "wmf-d1-behavioral-development-simulator-job-v1"
N3_CELL_SCHEMA = "wmf-n3-behavioral-development-cell-v1"
D1_CELL_SCHEMA = "wmf-d1-behavioral-development-cell-v1"
AUTHORITY_SOURCE_COMMIT = "950468266b05a2c429e4f13d7a5fb103728f206e"
D1_ATTEMPT003_SOURCE_COMMIT = "25ff299ca0d2b9964eb48286990ee2301cb207b8"
WORKER_ROLE = "wmf-forecast-0912-worker-09"
LAYOUTS = ("D01", "D02", "D03", "D04")
CONDITION_ORDER: dict[str, tuple[str, ...]] = {
    "D01": ("original-left", "reflected-right", "original-right", "reflected-left"),
    "D02": ("reflected-right", "original-left", "reflected-left", "original-right"),
    "D03": ("reflected-right", "original-right", "original-left", "reflected-left"),
    "D04": ("original-left", "reflected-left", "reflected-right", "original-right"),
}
ENVIRONMENT_SEED = {
    "D01": 2026091101,
    "D02": 2026091102,
    "D03": 2026091103,
    "D04": 2026091104,
}


@dataclass(frozen=True)
class AuthoritySpec:
    model: str
    job_id: str
    mode: str
    role: str
    receipt_sha256: str
    receipt_bytes: int
    authority_sha256: str
    authority_bytes: int
    target_count: int

    @property
    def job_dir(self) -> Path:
        return CONTROL_ROOT / "jobs" / self.job_id

    @property
    def receipt_path(self) -> Path:
        return self.job_dir / "publish" / "timing_job_receipt.json"

    @property
    def raw_authority_path(self) -> Path:
        return self.job_dir / "raw" / f"{self.model.lower()}_timing_authority.json"

    @property
    def published_authority_path(self) -> Path:
        return self.job_dir / "publish" / f"{self.model.lower()}_timing_authority.json"


AUTHORITIES: dict[str, AuthoritySpec] = {
    "N3": AuthoritySpec(
        model="N3",
        job_id="timing-n3-native-authority-002",
        mode="n3-native-authority",
        role="wmf-forecast-0912-worker-05",
        receipt_sha256="c57866fd5e223e7e893e53aa4e5d05e9b53b8219339eb1ed49fba16c9b570f13",
        receipt_bytes=4892,
        authority_sha256="fe4bc0056489bbc752824212f7fed96df99a857c84736c767e1cca278f2adb22",
        authority_bytes=23676,
        target_count=32,
    ),
    "D1": AuthoritySpec(
        model="D1",
        job_id="timing-d1-native-authority-002",
        mode="d1-native-authority",
        role="wmf-forecast-0912-worker-06",
        receipt_sha256="36c791a3ca58fb11aa4e9731d5bd0003c5080b6332ab03e76888d2c539713884",
        receipt_bytes=4893,
        authority_sha256="77265480e4779c6231b90a0f8bdbe91f73846bf7b9c1655656da39400fab76bc",
        authority_bytes=7515,
        target_count=2,
    ),
}


@dataclass(frozen=True)
class CellSpec:
    path: str
    sha256: str
    bytes: int

    def descriptor(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True)
class AggregateSpec:
    model: str
    layout: str
    job_id: str
    source_commit: str
    receipt_sha256: str | None
    receipt_bytes: int | None
    cells: tuple[CellSpec, ...] | None

    @property
    def receipt_filename(self) -> str:
        return (
            "n3_behavioral_development_receipt.json"
            if self.model == "N3"
            else "d1_behavioral_development_receipt.json"
        )

    @property
    def receipt_path(self) -> Path:
        return CONTROL_ROOT / "jobs" / self.job_id / "publish" / self.receipt_filename

    @property
    def raw_root(self) -> Path:
        behavioral = CONTROL_ROOT.parent / "behavioral" / "development" / self.model
        if self.model == "N3":
            return behavioral / self.layout / self.job_id
        return behavioral / self.layout / "simulator_attempts" / self.job_id

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return tuple(
            f"wmf1__development__{self.layout}__{self.model}__{condition.replace('-', '__')}"
            for condition in CONDITION_ORDER[self.layout]
        )


def _fixed_cells(
    *, model: str, layout: str, job_id: str, rows: Sequence[tuple[int, str]]
) -> tuple[CellSpec, ...]:
    behavioral = CONTROL_ROOT.parent / "behavioral" / "development" / model / layout
    root = behavioral / job_id if model == "N3" else behavioral / "simulator_attempts" / job_id
    output = []
    for index, (size, digest) in enumerate(rows):
        cell_id = (
            f"wmf1__development__{layout}__{model}__"
            f"{CONDITION_ORDER[layout][index].replace('-', '__')}"
        )
        slug = cell_id.replace("__", "-")
        output.append(CellSpec(str(root / "cells" / f"{index:02d}-{slug}" / "cell_receipt.json"), digest, size))
    return tuple(output)


N3_AGGREGATES: dict[str, AggregateSpec] = {
    "D01": AggregateSpec(
        "N3", "D01", "n3-development-d01-002",
        "91aaf8f5337bbf3a03e7cb65062ae1df77b6bb12",
        "a9ca815c6e99b270357220fa5ccb97562c50d667dd3c23745d460e5ec8c053de", 12551,
        _fixed_cells(model="N3", layout="D01", job_id="n3-development-d01-002", rows=(
            (36058, "607e5bf86b08023d2d09317069910dd6f8470904806161b3bea98fd596612f77"),
            (36114, "fa5c2ab16aafb7bd5b4182990d16735c12aab0a1fc2f60966d96bc4e6690cc94"),
            (36071, "201b391a6f96fa99f3712d4363b08aa2fb85d9fd2bba50e8202fd6b483906a23"),
            (36097, "6d35d660dc588e550f4fbd88498d6a8f64d01319866d0636df774a092393dda0"),
        )),
    ),
    "D02": AggregateSpec(
        "N3", "D02", "n3-development-d02-002",
        "c5b5368d852ecc9356b9df90cc5e6f62bfaecde8",
        "efd2ea9f2f015c671a9d9d2d9dc22c1c1f9f5f3ffd87e4810acd5e3767b2f73e", 12551,
        _fixed_cells(model="N3", layout="D02", job_id="n3-development-d02-002", rows=(
            (36080, "7d62073b04e9c762c9b543f4de30ecfd4c38122cf8284ee8a68597dff9a16736"),
            (36056, "3a448afb85b355d1eb12b6bc933b26c367bfa53803659b5663b6284a611b6fe3"),
            (36063, "06eeaa04de46931f2f6e2985edab41371611f93f0e377b56333a7ea09178ccb0"),
            (36071, "b8a28bb9f7511868656caf5505b540e8620a6e00ec58afa4847f1bfc5e8e1dcc"),
        )),
    ),
    "D03": AggregateSpec(
        "N3", "D03", "n3-development-d03-001",
        "91aaf8f5337bbf3a03e7cb65062ae1df77b6bb12",
        "d3e85f2087d7e24d57974a9344c333408e9769552715d4669e16f1e059082d0e", 12551,
        _fixed_cells(model="N3", layout="D03", job_id="n3-development-d03-001", rows=(
            (36087, "62cf5769cdfe35c0b9b13225c84b55961621e760e87c87170ca5d591deeedc32"),
            (36070, "ee7c868b75bc25980c1f692866834bbf5650bbf65318a82eca19b016602a97be"),
            (36056, "312fc085413183d048c05c77640b7273207809228b26276f7e327f7a8c3cefbb"),
            (36069, "d598820cd9fe62e3c90f7669524b8b6aad9475bf70a987fd51fcb1e7dd8c5ccf"),
        )),
    ),
    "D04": AggregateSpec(
        "N3", "D04", "n3-development-d04-001",
        "91aaf8f5337bbf3a03e7cb65062ae1df77b6bb12",
        "e5f3206c6da2b9caa2ca9cd4c3ed82142157656ac9a1d1744486d14fa6e57318", 12551,
        _fixed_cells(model="N3", layout="D04", job_id="n3-development-d04-001", rows=(
            (36044, "e85c372f87aac66e997683a82ea030d22e58061367023518c44a53b5aae455d0"),
            (36060, "e70662a9291bafe41571251d2c7c127263635d5cd71d9cc9bff856a41b37c231"),
            (36078, "ef384c70f84913803a5555fb1ed968094c0b511c2796f1e035c07b8e5f2472e8"),
            (36059, "194d4ac61c2098263360e60419d983505f9be9bc5119f0c2d23e2de71a52141d"),
        )),
    ),
}

D1_AGGREGATES: dict[str, AggregateSpec] = {
    layout: AggregateSpec(
        "D1", layout, f"d1-development-{layout.lower()}-simulator-003",
        D1_ATTEMPT003_SOURCE_COMMIT, None, None, None,
    )
    for layout in ("D01", "D02", "D04")
}
D1_AGGREGATES["D03"] = AggregateSpec(
    "D1", "D03", "d1-development-d03-simulator-002",
    "64e84d1f7634e2a7fbaaf0e1ffd05bf021614517",
    "eb7f4f894b1e56c1b8971ee1155d77e7aeb1b62c9c6cfc41de0483e3baf2dd4f", 16510,
    _fixed_cells(model="D1", layout="D03", job_id="d1-development-d03-simulator-002", rows=(
        (70902, "a755f3f806ea73a395268d7cda38748b1bb6a7cdd60c7d4ee5e9872d7f912e26"),
        (70892, "6560cf4ea7b11187e9fd525a2eac722004bf7945283654257d4a4cbd47ee034f"),
        (70879, "c3af55faf13a6947493435c89f88adb9af71b475ee6f3e1fc24256abd9d7e6fb"),
        (70888, "f942a38a88d6293a58519594d329089f8f33b86de23f2bcd702b6f20fbd12b03"),
    )),
)

AGGREGATES = {"N3": N3_AGGREGATES, "D1": D1_AGGREGATES}
JOB_ID = {
    "N3": "timing-n3-development-sidecar-001",
    "D1": "timing-d1-development-sidecar-001",
}
REQUEST_COUNT = {"N3": 240, "D1": 912}
REQUESTS_PER_CELL = {"N3": 15, "D1": 57}


class DevelopmentTimingJobError(RuntimeError):
    """A release or immutable evidence binding failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentTimingJobError(message)


def _descriptor(
    value: Any, *, path: Path, sha256: str, bytes_: int, label: str
) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} descriptor is missing")
    expected = {"path": str(path), "sha256": sha256, "bytes": bytes_}
    require(
        {key: value.get(key) for key in expected} == expected,
        f"{label} descriptor identity changed",
    )
    return expected


def _free_descriptor(value: Any, *, path: Path, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} descriptor is missing")
    digest = value.get("sha256")
    size = value.get("bytes")
    require(
        isinstance(digest, str) and queue.SHA256_RE.fullmatch(digest) is not None,
        f"{label} descriptor hash is invalid",
    )
    require(type(size) is int and size > 0, f"{label} descriptor byte count is invalid")
    require(Path(str(value.get("path"))) == path, f"{label} descriptor path changed")
    return {"path": str(path), "sha256": digest, "bytes": size}


def validate_local_authority_receipt(
    path: Path, expected_sha256: str, spec: AuthoritySpec
) -> dict[str, Any]:
    """Validate a deliberately published authority receipt without PVC access."""

    require(expected_sha256 == spec.receipt_sha256, f"{spec.model} authority receipt hash is not frozen")
    identity = queue.file_identity(path)
    require(
        identity["sha256"] == spec.receipt_sha256
        and identity["bytes"] == spec.receipt_bytes,
        f"{spec.model} authority receipt bytes/hash changed",
    )
    receipt = queue.load_json(path, f"{spec.model} authority receipt")
    queue.verify_signed_document(receipt, f"{spec.model} authority receipt")
    exact = {
        "schema_version": queue.TIMING_JOB_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "status": "passed",
        "decision": "go",
        "mode": spec.mode,
        "job_id": spec.job_id,
        "job_dir": str(spec.job_dir),
        "study_commit": AUTHORITY_SOURCE_COMMIT,
        "queue_role": spec.role,
        "worker_id": spec.role,
        "physical_time_qualified": True,
        "behavioral_policy_skill_evaluated": False,
        "safe_to_release_confirmation": False,
    }
    for key, wanted in exact.items():
        require(receipt.get(key) == wanted, f"{spec.model} authority receipt changed: {key}")
    require(
        receipt.get("science_counts")
        == queue._expected_science_counts(issued=0, referenced=6),
        f"{spec.model} authority receipt science counts changed",
    )
    require(
        receipt.get("evidence_counts")
        == {
            "qualified_generated_targets": spec.target_count,
            "referenced_recorder_actions": 450,
            "referenced_recorder_observations": 451,
        },
        f"{spec.model} authority evidence counts changed",
    )
    implementation = receipt.get("implementation")
    require(isinstance(implementation, Mapping), f"{spec.model} authority implementation is missing")
    source = CONTROL_ROOT / "sources" / AUTHORITY_SOURCE_COMMIT
    _descriptor(
        implementation.get("timing_validator"),
        path=source / TOOL_RELATIVE,
        sha256=TOOL_SHA256,
        bytes_=116377,
        label=f"{spec.model} authority validator",
    )
    _descriptor(
        implementation.get("timing_contract"),
        path=source / CONTRACT_RELATIVE,
        sha256=CONTRACT_SHA256,
        bytes_=11765,
        label=f"{spec.model} authority contract",
    )
    outputs = receipt.get("outputs")
    require(isinstance(outputs, Mapping), f"{spec.model} authority outputs are missing")
    raw = _descriptor(
        outputs.get("timing_authority"),
        path=spec.raw_authority_path,
        sha256=spec.authority_sha256,
        bytes_=spec.authority_bytes,
        label=f"{spec.model} raw authority",
    )
    published = _descriptor(
        outputs.get("published_timing_authority"),
        path=spec.published_authority_path,
        sha256=spec.authority_sha256,
        bytes_=spec.authority_bytes,
        label=f"{spec.model} published authority",
    )
    return {
        "receipt": {
            "path": str(spec.receipt_path),
            "sha256": spec.receipt_sha256,
            "bytes": spec.receipt_bytes,
        },
        "raw_authority": raw,
        "published_authority": published,
    }


def _expected_counts(model: str) -> dict[str, int]:
    return {
        "actual_behavioral_actions": 1800,
        "actual_behavioral_model_requests": 60 if model == "N3" else 228,
        "completed_valid_behavioral_cells": 4,
        "launched_behavioral_cells": 4,
        "new_generation_qualification_requests": 0,
        "newly_launched_behavioral_cells": 4,
        "planned_behavioral_cells": 4,
        "recorder_only_episodes_counted_as_behavioral": 0,
        "resumed_valid_behavioral_cells": 0,
        "reused_prerequisite_generation_qualification_requests": 6,
        "right_censored_behavioral_cells": 0,
        "technically_invalid_behavioral_cells": 0,
        "unrun_behavioral_cells": 0,
    }


def validate_local_aggregate_receipt(
    path: Path, expected_sha256: str, spec: AggregateSpec
) -> dict[str, Any]:
    """Validate one fetched aggregate and its exact embedded cell inventory."""

    require(
        isinstance(expected_sha256, str)
        and queue.SHA256_RE.fullmatch(expected_sha256) is not None,
        f"{spec.model} {spec.layout} aggregate expected hash is invalid",
    )
    if spec.receipt_sha256 is not None:
        require(expected_sha256 == spec.receipt_sha256, f"{spec.model} {spec.layout} aggregate hash is not frozen")
    identity = queue.file_identity(path)
    require(identity["sha256"] == expected_sha256, f"{spec.model} {spec.layout} aggregate hash changed")
    if spec.receipt_bytes is not None:
        require(identity["bytes"] == spec.receipt_bytes, f"{spec.model} {spec.layout} aggregate byte count changed")
    receipt = queue.load_json(path, f"{spec.model} {spec.layout} aggregate")
    exact = {
        "schema_version": N3_AGGREGATE_SCHEMA if spec.model == "N3" else D1_AGGREGATE_SCHEMA,
        "status": "passed",
        "exit_code": 0,
        "failure": None,
        "study_id": STUDY_ID,
        "namespace": NAMESPACE,
        "phase": "development",
        "layout_pair_id": spec.layout,
        "model_config": spec.model,
        "block_id": f"wmf_ablation_001_20260912__development__{spec.layout}__{spec.model}",
        "condition_order": list(CONDITION_ORDER[spec.layout]),
        "cell_ids": list(spec.cell_ids),
        "counts": _expected_counts(spec.model),
        "raw_attempt_root": str(spec.raw_root),
        "raw_attempt_recoverable_on_gm_pvc": True,
    }
    exact["source_commit" if spec.model == "N3" else "study_commit"] = spec.source_commit
    for key, wanted in exact.items():
        require(receipt.get(key) == wanted, f"{spec.model} {spec.layout} aggregate changed: {key}")
    if spec.model == "D1":
        require(
            receipt.get("run_id") == f"d1-development-{spec.layout.lower()}-{spec.job_id.rsplit('-', 1)[-1]}"
            and receipt.get("simulator_job_id") == spec.job_id
            and receipt.get("server_job_id")
            == spec.job_id.replace("-simulator-", "-server-")
            and receipt.get("all_simulator_children_reaped") is True,
            f"D1 {spec.layout} terminal pair identity changed",
        )
        require(
            receipt.get("effective_model_noise_seed") == 1140
            and receipt.get("environment_seed") == ENVIRONMENT_SEED[spec.layout]
            and receipt.get("noise_semantics")
            == "fixed; cells and requests are not independent noise draws",
            f"D1 {spec.layout} seed semantics changed",
        )
    else:
        server_exit = receipt.get("server_exit")
        require(
            isinstance(server_exit, Mapping)
            and server_exit.get("child_reaped") is True
            and server_exit.get("terminated_by_queue_supervisor") is True,
            f"N3 {spec.layout} server was not cleanly reaped",
        )
        require(
            receipt.get("effective_seed") == ENVIRONMENT_SEED[spec.layout],
            f"N3 {spec.layout} effective seed changed",
        )
    queue_descriptor = receipt.get("queue_descriptor")
    descriptor_path = CONTROL_ROOT / "jobs" / spec.job_id / "descriptor.json"
    require(
        isinstance(queue_descriptor, Mapping)
        and queue_descriptor.get("job_id") == spec.job_id
        and queue_descriptor.get("role")
        == ("n3" if spec.model == "N3" else "wmf-forecast-0912-worker-00"),
        f"{spec.model} {spec.layout} queue descriptor changed",
    )
    _free_descriptor(
        queue_descriptor,
        path=descriptor_path,
        label=f"{spec.model} {spec.layout} queue descriptor",
    )
    cells_value = receipt.get("cell_receipts")
    require(isinstance(cells_value, list) and len(cells_value) == 4, f"{spec.model} {spec.layout} cell inventory changed")
    cells: list[dict[str, Any]] = []
    for index, value in enumerate(cells_value):
        cell_id = spec.cell_ids[index]
        expected_path = spec.raw_root / "cells" / f"{index:02d}-{cell_id.replace('__', '-')}" / "cell_receipt.json"
        if spec.cells is None:
            cells.append(_free_descriptor(value, path=expected_path, label=f"{spec.model} {spec.layout} cell {index}"))
        else:
            fixed = spec.cells[index]
            require(Path(fixed.path) == expected_path, f"{spec.model} {spec.layout} frozen cell path is inconsistent")
            cells.append(_descriptor(value, path=expected_path, sha256=fixed.sha256, bytes_=fixed.bytes, label=f"{spec.model} {spec.layout} cell {index}"))
    require(len({row["sha256"] for row in cells}) == 4, f"{spec.model} {spec.layout} cell hashes are duplicated")
    resumed = receipt.get("resumed_cell_receipts")
    new = receipt.get("new_cell_receipts")
    require(isinstance(resumed, list) and isinstance(new, list) and resumed + new == cells_value, f"{spec.model} {spec.layout} resume/new inventory changed")
    return {
        "aggregate_receipt": {
            "path": str(spec.receipt_path),
            "sha256": expected_sha256,
            "bytes": identity["bytes"],
        },
        "cell_receipts": cells,
    }


def _job_descriptor(
    *, model: str, study_commit: str, self_sha256: str,
    authorities: Mapping[str, Mapping[str, Any]],
    aggregates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    require(model in ("N3", "D1"), "sidecar model is unsupported")
    commit = queue._verified_commit(study_commit)
    queue._verified_sha(self_sha256, "sidecar builder digest")
    argv = [
        str(ROBOLAB_PYTHON),
        "{source_root}/" + str(THIS_RELATIVE),
        f"{model.lower()}-sidecar",
        "--source-root", "{source_root}",
        "--study-commit", commit,
        "--job-dir", "{job_dir}",
        "--job-id", JOB_ID[model],
        "--expected-role", WORKER_ROLE,
        "--sidecar-builder-sha256", self_sha256,
        "--timing-tool-sha256", TOOL_SHA256,
        "--contract-sha256", CONTRACT_SHA256,
        "--expected-request-count", str(REQUEST_COUNT[model]),
    ]
    for authority_model in ("N3", "D1"):
        evidence = authorities[authority_model]
        argv.extend(
            [
                f"--{authority_model.lower()}-authority-receipt",
                str(evidence["receipt"]["path"]),
                f"--{authority_model.lower()}-authority-receipt-sha256",
                str(evidence["receipt"]["sha256"]),
            ]
        )
    for layout in LAYOUTS:
        aggregate = aggregates[layout]["aggregate_receipt"]
        argv.extend(
            [
                "--aggregate-receipt", str(aggregate["path"]),
                "--aggregate-receipt-sha256", str(aggregate["sha256"]),
            ]
        )
    return {
        "job_id": JOB_ID[model],
        "released": True,
        "source_commit": commit,
        "role": WORKER_ROLE,
        "argv": argv,
        "max_wall_seconds": 14400,
        "publish_log_tail_bytes": 8192,
    }


def build_sidecar_wave(
    *, model: str, study_commit: str,
    n3_authority_receipt_path: Path, n3_authority_receipt_sha256: str,
    d1_authority_receipt_path: Path, d1_authority_receipt_sha256: str,
    aggregate_inputs: Mapping[str, tuple[Path, str]],
) -> dict[str, Any]:
    """Emit one descriptor only after every exact local gate validates."""

    require(model in AGGREGATES, "sidecar model is unsupported")
    require(set(aggregate_inputs) == set(LAYOUTS), f"{model} requires exactly D01-D04 aggregates")
    authorities = {
        "N3": validate_local_authority_receipt(
            n3_authority_receipt_path, n3_authority_receipt_sha256, AUTHORITIES["N3"]
        ),
        "D1": validate_local_authority_receipt(
            d1_authority_receipt_path, d1_authority_receipt_sha256, AUTHORITIES["D1"]
        ),
    }
    aggregates = {
        layout: validate_local_aggregate_receipt(
            aggregate_inputs[layout][0], aggregate_inputs[layout][1], AGGREGATES[model][layout]
        )
        for layout in LAYOUTS
    }
    self_sha256 = queue.sha256_file(Path(__file__).resolve())
    descriptor = _job_descriptor(
        model=model,
        study_commit=study_commit,
        self_sha256=self_sha256,
        authorities=authorities,
        aggregates=aggregates,
    )
    return {
        "schema_version": WAVE_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "model_id": model,
        "source_commit": descriptor["source_commit"],
        "status": "descriptor_only_not_dispatched_all_evidence_gates_passed",
        "jobs": [descriptor],
        "evidence_gate": {
            "authorities": authorities,
            "development_aggregates": aggregates,
        },
        "science_counts": _zero_science_counts(),
        "referenced_behavioral_cells": 16,
        "referenced_behavioral_model_requests": REQUEST_COUNT[model],
        "safe_to_release_confirmation": False,
        "claim_boundary": (
            "One CPU-side inventory/bind/validate job over immutable completed development "
            "evidence. It issues no request or action and cannot release confirmation."
        ),
    }


def _zero_science_counts() -> dict[str, int]:
    return {
        "model_runtime_loads": 0,
        "model_servers_started": 0,
        "model_requests_issued_by_job": 0,
        "physical_resets": 0,
        "robot_episodes": 0,
        "behavioral_actions_executed_by_job": 0,
        "behavioral_cells_launched_by_job": 0,
    }


def _runtime_evidence(args: argparse.Namespace, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    authority_args = {
        "N3": (Path(args.n3_authority_receipt), args.n3_authority_receipt_sha256),
        "D1": (Path(args.d1_authority_receipt), args.d1_authority_receipt_sha256),
    }
    authorities: dict[str, Any] = {}
    for authority_model, spec in AUTHORITIES.items():
        supplied_path, supplied_sha = authority_args[authority_model]
        require(supplied_path == spec.receipt_path, f"{authority_model} runtime authority receipt path changed")
        authorities[authority_model] = validate_local_authority_receipt(
            supplied_path, supplied_sha, spec
        )
    require(
        len(args.aggregate_receipt) == len(LAYOUTS)
        and len(args.aggregate_receipt_sha256) == len(LAYOUTS),
        f"{model} runtime aggregate inventory is incomplete",
    )
    aggregates: dict[str, Any] = {}
    for layout, supplied_path, supplied_sha in zip(
        LAYOUTS, args.aggregate_receipt, args.aggregate_receipt_sha256
    ):
        spec = AGGREGATES[model][layout]
        require(Path(supplied_path) == spec.receipt_path, f"{model} {layout} runtime aggregate path changed")
        aggregates[layout] = validate_local_aggregate_receipt(
            Path(supplied_path), supplied_sha, spec
        )
    return authorities, aggregates


def _runtime_descriptor(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    model = {"n3-sidecar": "N3", "d1-sidecar": "D1"}.get(args.command)
    require(model is not None, "runtime sidecar mode is unsupported")
    require(args.job_id == JOB_ID[model], "runtime sidecar job ID changed")
    require(args.expected_role == WORKER_ROLE, "runtime sidecar worker role changed")
    require(args.timing_tool_sha256 == TOOL_SHA256, "runtime timing validator hash changed")
    require(args.contract_sha256 == CONTRACT_SHA256, "runtime timing contract hash changed")
    require(args.expected_request_count == REQUEST_COUNT[model], "runtime request count changed")
    authorities = {
        name: {
            "receipt": {
                "path": str(spec.receipt_path),
                "sha256": getattr(args, f"{name.lower()}_authority_receipt_sha256"),
                "bytes": spec.receipt_bytes,
            }
        }
        for name, spec in AUTHORITIES.items()
    }
    aggregates = {
        layout: {
            "aggregate_receipt": {
                "path": str(path), "sha256": digest, "bytes": 0,
            }
        }
        for layout, path, digest in zip(
            LAYOUTS, args.aggregate_receipt, args.aggregate_receipt_sha256
        )
    }
    descriptor = _job_descriptor(
        model=model,
        study_commit=args.study_commit,
        self_sha256=args.sidecar_builder_sha256,
        authorities=authorities,
        aggregates=aggregates,
    )
    return model, descriptor


def _normalize_actual_descriptor(value: Any, *, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} descriptor is missing")
    path = Path(str(value.get("path")))
    require(path.is_absolute(), f"{label} path is not absolute")
    return _free_descriptor(value, path=path, label=label)


def _extract_request_descriptor(
    timing: ModuleType, response: Mapping[str, Any], *, model: str,
    cell_id: str, request_index: int,
) -> dict[str, Any]:
    require(response.get("request_index") == request_index, "journal response request index changed")
    artifact = response.get("response_artifact")
    require(isinstance(artifact, Mapping), "journal response artifact is missing")
    raw_response = timing._mapping_item(artifact.get("structure"), "raw_response")
    descriptor = timing._thaw_scalar_structure(
        timing._mapping_item(raw_response, "wmf_server_request_receipt")
    )
    normalized = _normalize_actual_descriptor(
        descriptor, label=f"{model} {cell_id} request {request_index}"
    )
    transported_index = timing._thaw_scalar_structure(
        timing._mapping_item(raw_response, "wmf_request_index")
    )
    require(transported_index == request_index, "transported request index changed")
    if model == "N3":
        transported_cell = timing._thaw_scalar_structure(
            timing._mapping_item(raw_response, "wmf_cell_id")
        )
        require(transported_cell == cell_id, "transported N3 cell identity changed")
    return normalized


def build_request_inventory(
    *, timing: ModuleType, model: str,
    aggregates: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    entries: list[dict[str, Any]] = []
    aggregate_receipts: list[dict[str, Any]] = []
    for layout in LAYOUTS:
        evidence = aggregates[layout]
        aggregate_receipts.append(dict(evidence["aggregate_receipt"]))
        spec = AGGREGATES[model][layout]
        for cell_index, descriptor in enumerate(evidence["cell_receipts"]):
            identity = queue.file_identity(Path(descriptor["path"]))
            require(identity == descriptor, f"{model} {layout} cell {cell_index} bytes/hash changed")
            cell = queue.load_json(Path(descriptor["path"]), f"{model} {layout} cell {cell_index}")
            cell_id = spec.cell_ids[cell_index]
            exact = {
                "schema_version": N3_CELL_SCHEMA if model == "N3" else D1_CELL_SCHEMA,
                "status": "passed",
                "study_id": STUDY_ID,
                "layout_pair_id": layout,
                "model_config": model,
                "condition_index": cell_index,
                "cell_id": cell_id,
                "actions_executed": 450,
                "observation_count": 451,
                "behavioral_model_request_count": REQUESTS_PER_CELL[model],
                "behavioral_episode_count": 1,
                "generation_qualification_request_count": 0,
            }
            if model == "D1":
                exact["phase"] = "development"
            for key, wanted in exact.items():
                require(cell.get(key) == wanted, f"{model} {layout} cell {cell_index} changed: {key}")
            completion = _normalize_actual_descriptor(
                cell.get("adapter_completion"), label=f"{model} {layout} completion {cell_index}"
            )
            journal = _normalize_actual_descriptor(
                cell.get("adapter_journal"), label=f"{model} {layout} journal {cell_index}"
            )
            rows, tail = timing._verify_journal(Path(journal["path"]))
            require(
                cell["adapter_journal"].get("event_count") == len(rows)
                and cell["adapter_journal"].get("tail_sha256") == tail,
                f"{model} {layout} cell {cell_index} journal summary changed",
            )
            journal.update({"event_count": len(rows), "tail_sha256": tail})
            responses = timing._event_payloads(rows, "model_response_received")
            require(
                len(responses) == REQUESTS_PER_CELL[model],
                f"{model} {layout} cell {cell_index} response inventory changed",
            )
            requests = [
                _extract_request_descriptor(
                    timing, response, model=model, cell_id=cell_id, request_index=index
                )
                for index, response in enumerate(responses)
            ]
            if model == "D1":
                declared = cell.get("server_request_receipts")
                require(
                    isinstance(declared, list)
                    and [
                        _normalize_actual_descriptor(
                            value, label=f"D1 {layout} declared request {index}"
                        )
                        for index, value in enumerate(declared)
                    ]
                    == requests,
                    f"D1 {layout} cell {cell_index} request inventory differs from recorder transport",
                )
            for request_index, request in enumerate(requests):
                entries.append(
                    {
                        "cell_id": cell_id,
                        "request_index": request_index,
                        "request_receipt": request,
                        "adapter_completion": completion,
                        "adapter_journal": journal,
                    }
                )
    require(len(entries) == REQUEST_COUNT[model], f"{model} complete request inventory count changed")
    hashes = [entry["request_receipt"]["sha256"] for entry in entries]
    require(len(set(hashes)) == len(hashes), f"{model} request receipt hashes are duplicated")
    return timing.sign_document(
        {
            "schema_version": timing.REQUEST_INVENTORY_SCHEMA,
            "study_id": STUDY_ID,
            "model_id": model,
            "status": "complete_exact_development_inventory",
            "aggregate_receipts": aggregate_receipts,
            "request_receipts": entries,
            "old_request_receipts_modified": False,
            "model_requests_issued_by_inventory_job": 0,
            "behavioral_actions_executed_by_inventory_job": 0,
        }
    ), hashes


def _validate_runtime_authorities(
    timing: ModuleType, authorities: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    validated: dict[str, Any] = {}
    for model, spec in AUTHORITIES.items():
        evidence = authorities[model]
        receipt_identity = queue.file_identity(spec.receipt_path)
        require(receipt_identity == evidence["receipt"], f"{model} runtime authority receipt identity changed")
        raw_identity = queue.file_identity(spec.raw_authority_path)
        published_identity = queue.file_identity(spec.published_authority_path)
        require(
            raw_identity == evidence["raw_authority"]
            and published_identity == evidence["published_authority"],
            f"{model} runtime authority artifact identity changed",
        )
        authority = timing.validate_timing_authority(
            spec.raw_authority_path, spec.authority_sha256, expected_model=model
        )
        require(
            len(authority.get("generated_targets", [])) == spec.target_count,
            f"{model} runtime authority target count changed",
        )
        validated[model] = authority
    return validated


def _failure_receipt(
    *, context: Any | None, job_dir: Path, model: str, error: BaseException
) -> None:
    publish = Path(job_dir).resolve() / "publish"
    try:
        if publish.exists() or publish.is_symlink():
            require(publish.is_dir() and not publish.is_symlink(), "failure publish path is invalid")
        else:
            publish.mkdir()
        target = publish / "timing_job_failure.json"
        if target.exists() or target.is_symlink():
            return
        receipt = queue.signed_document(
            {
                "schema_version": JOB_RECEIPT_SCHEMA,
                "namespace": NAMESPACE,
                "study_id": STUDY_ID,
                "model_id": model,
                "status": "technical_invalid",
                "decision": "no_go",
                "job_id": context.job_id if context is not None else Path(job_dir).name,
                "study_commit": context.study_commit if context is not None else None,
                "queue_role": context.role if context is not None else None,
                "queue_descriptor": context.descriptor_identity if context is not None else None,
                "queue_claim": context.claim_identity if context is not None else None,
                "failure": {
                    "error_type": type(error).__name__,
                    "detail": str(error)[:2000],
                    "traceback": traceback.format_exc(limit=20)[-12000:],
                },
                "science_counts": _zero_science_counts(),
                "physical_time_qualified": False,
                "behavioral_policy_skill_evaluated": False,
                "safe_to_release_confirmation": False,
                "claim_boundary": (
                    "Failed CPU-side evidence validation only; no model, simulator, reset, "
                    "request, action, episode, or behavioral cell was started."
                ),
                "completed_at_utc": queue.utc_now(),
            }
        )
        queue.immutable_json(target, receipt, maximum_bytes=512 * 1024)
    except BaseException:
        return


def run_sidecar_job(args: argparse.Namespace) -> dict[str, Any]:
    model, expected_descriptor = _runtime_descriptor(args)
    context = None
    try:
        context = queue.validate_queue_context(
            source_root=args.source_root,
            job_dir=args.job_dir,
            study_commit=args.study_commit,
            job_id=args.job_id,
            expected_role=args.expected_role,
            expected_descriptor=expected_descriptor,
        )
        self_identity = queue.file_identity(context.source_root / THIS_RELATIVE)
        require(
            self_identity["sha256"] == args.sidecar_builder_sha256,
            "staged sidecar builder hash changed",
        )
        implementation = queue._validate_staged_implementation(context.source_root)
        require(
            implementation["timing_validator"]["sha256"] == args.timing_tool_sha256
            and implementation["timing_contract"]["sha256"] == args.contract_sha256,
            "staged timing implementation differs from descriptor",
        )
        authorities, aggregates = _runtime_evidence(args, model)
        timing = _load_module(
            context.source_root / TOOL_RELATIVE,
            f"wmf_development_timing_{model.lower()}_{context.job_id.replace('-', '_')}",
        )
        _validate_runtime_authorities(timing, authorities)
        raw, publish = queue._prepare_output_directories(context.job_dir)
        inventory, request_hashes = build_request_inventory(
            timing=timing, model=model, aggregates=aggregates
        )
        inventory_path = raw / f"{model.lower()}_development_request_inventory.json"
        timing.atomic_json(inventory_path, inventory)
        inventory_identity = queue.file_identity(inventory_path)
        sidecar = timing.bind_development_requests(
            authority_path=AUTHORITIES[model].raw_authority_path,
            authority_sha256=AUTHORITIES[model].authority_sha256,
            inventory_path=inventory_path,
            inventory_sha256=inventory_identity["sha256"],
        )
        sidecar_path = raw / f"{model.lower()}_development_timing_sidecar.json"
        timing.atomic_json(sidecar_path, sidecar)
        sidecar_identity = queue.file_identity(sidecar_path)
        validated = timing.validate_development_timing(
            sidecar_path,
            sidecar_identity["sha256"],
            expected_model=model,
            expected_request_hashes=request_hashes,
        )
        require(
            len(validated.get("request_timing_bindings", [])) == REQUEST_COUNT[model],
            f"{model} validated sidecar request count changed",
        )
        receipt = queue.signed_document(
            {
                "schema_version": JOB_RECEIPT_SCHEMA,
                "namespace": NAMESPACE,
                "study_id": STUDY_ID,
                "model_id": model,
                "status": "passed",
                "decision": "go",
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
                "implementation": {**implementation, "sidecar_builder": self_identity},
                "inputs": {
                    "authorities": authorities,
                    "development_aggregates": aggregates,
                },
                "outputs": {
                    "raw_request_inventory": inventory_identity,
                    "raw_development_timing_sidecar": sidecar_identity,
                },
                "referenced_behavioral_cells": 16,
                "referenced_behavioral_model_requests": REQUEST_COUNT[model],
                "source_request_receipt_sha256s": request_hashes,
                "science_counts": _zero_science_counts(),
                "physical_time_qualified": True,
                "development_timing_sidecar_valid": True,
                "behavioral_policy_skill_evaluated": False,
                "safe_to_release_confirmation": False,
                "raw_outputs_recoverable_on_gm_pvc": True,
                "claim_boundary": (
                    "Binds native timing to immutable completed development requests only. "
                    "It issues no new request/action and does not release confirmation."
                ),
                "completed_at_utc": queue.utc_now(),
            }
        )
        queue.immutable_json(
            publish / "timing_job_receipt.json", receipt, maximum_bytes=512 * 1024
        )
        return receipt
    except BaseException as error:
        _failure_receipt(
            context=context,
            job_dir=Path(args.job_dir),
            model=model,
            error=error,
        )
        raise


def _aggregate_inputs(args: argparse.Namespace) -> dict[str, tuple[Path, str]]:
    return {
        layout: (
            Path(getattr(args, f"{layout.lower()}_aggregate")),
            getattr(args, f"{layout.lower()}_aggregate_sha256"),
        )
        for layout in LAYOUTS
    }


def _add_builder_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--study-commit", required=True)
    parser.add_argument("--n3-authority-receipt", type=Path, required=True)
    parser.add_argument("--n3-authority-receipt-sha256", required=True)
    parser.add_argument("--d1-authority-receipt", type=Path, required=True)
    parser.add_argument("--d1-authority-receipt-sha256", required=True)
    for layout in LAYOUTS:
        parser.add_argument(f"--{layout.lower()}-aggregate", type=Path, required=True)
        parser.add_argument(f"--{layout.lower()}-aggregate-sha256", required=True)
    parser.add_argument("--output", type=Path)


def _add_runtime_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--study-commit", required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-role", required=True)
    parser.add_argument("--sidecar-builder-sha256", required=True)
    parser.add_argument("--timing-tool-sha256", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--expected-request-count", type=int, required=True)
    parser.add_argument("--n3-authority-receipt", type=Path, required=True)
    parser.add_argument("--n3-authority-receipt-sha256", required=True)
    parser.add_argument("--d1-authority-receipt", type=Path, required=True)
    parser.add_argument("--d1-authority-receipt-sha256", required=True)
    parser.add_argument("--aggregate-receipt", type=Path, action="append", required=True)
    parser.add_argument("--aggregate-receipt-sha256", action="append", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for model in ("n3", "d1"):
        emit = commands.add_parser(f"emit-{model}-sidecar")
        _add_builder_inputs(emit)
        runtime = commands.add_parser(f"{model}-sidecar")
        _add_runtime_inputs(runtime)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command.startswith("emit-"):
        model = "N3" if args.command == "emit-n3-sidecar" else "D1"
        wave = build_sidecar_wave(
            model=model,
            study_commit=args.study_commit,
            n3_authority_receipt_path=args.n3_authority_receipt,
            n3_authority_receipt_sha256=args.n3_authority_receipt_sha256,
            d1_authority_receipt_path=args.d1_authority_receipt,
            d1_authority_receipt_sha256=args.d1_authority_receipt_sha256,
            aggregate_inputs=_aggregate_inputs(args),
        )
        queue._write_descriptor_output(args.output, wave)
        return 0
    receipt = run_sidecar_job(args)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            raise
        print(
            json.dumps(
                {
                    "status": "technical_failure",
                    "error_type": type(error).__name__,
                    "detail": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
