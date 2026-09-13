#!/usr/bin/env python3
"""Build and run the detached formal development-resource audit job.

The descriptor-only emitter reads exact immutable blobs from a fetched results
commit and emits one CPU-safe queue descriptor.  It does not edit the active
queue.  The detached runtime repeats every gate from the original PVC files,
authenticates the complete formal development cohort, and invokes the
zero-science resource compiler below its unique job directory.

Only one compact signed terminal receipt is publishable.  Raw per-cell
measurements and the de-duplicated file inventory stay on the PVC.  This job is
missingness-aware and can never release confirmation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import traceback
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence


sys.dont_write_bytecode = True

FORECAST_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = FORECAST_ROOT.parents[1]
QUEUE_SUPPORT_PATH = Path(__file__).with_name("forecast_timing_queue_jobs.py")
AGGREGATE_SUPPORT_PATH = Path(__file__).with_name(
    "development_timing_sidecar_jobs.py"
)


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


queue = _load_module(QUEUE_SUPPORT_PATH, "wmf_resource_queue_support")
aggregate_support = _load_module(
    AGGREGATE_SUPPORT_PATH, "wmf_resource_aggregate_support"
)

NAMESPACE = queue.NAMESPACE
STUDY_ID = queue.STUDY_ID
CONTROL_ROOT = queue.CONTROL_ROOT
RAW_ROOT = CONTROL_ROOT.parent
ROBOLAB_PYTHON = queue.ROBOLAB_PYTHON

THIS_RELATIVE = (
    queue.FORECAST_RELATIVE
    / "experiments/forecast_layout/development_resource_jobs.py"
)
QUEUE_SUPPORT_RELATIVE = queue.THIS_RELATIVE
AGGREGATE_SUPPORT_RELATIVE = (
    queue.FORECAST_RELATIVE
    / "experiments/forecast_layout/development_timing_sidecar_jobs.py"
)
RESOURCE_COMPILER_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/compile_development_resources.py"
)
RESOURCE_CONTRACT_RELATIVE = (
    queue.FORECAST_RELATIVE
    / "experiments/forecast_layout/development_resource_contract.json"
)
EVIDENCE_COMPILER_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/compile_development_evidence.py"
)
FREEZE_VALIDATOR_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/freeze_development_release.py"
)
TIMING_VALIDATOR_RELATIVE = (
    queue.FORECAST_RELATIVE / "analysis/qualify_forecast_timing.py"
)
PLANNED_CELLS_RELATIVE = (
    queue.FORECAST_RELATIVE / "experiments/forecast_layout/planned_cells.csv"
)

WAVE_SCHEMA = "wmf-development-resource-wave-v1"
CONTRACT_SCHEMA = "wmf-development-resource-contract-v1"
JOB_RECEIPT_SCHEMA = "wmf-development-resource-queue-job-v1"
OUTPUT_INVENTORY_SCHEMA = "wmf-development-resource-output-inventory-v1"
COMPILER_INPUT_SCHEMA = "wmf-development-resource-compiler-input-v1"
COMPILER_RECEIPT_SCHEMA = "wmf-development-resource-compiler-receipt-v1"
RESOURCE_AGGREGATE_SCHEMA = "wmf-development-resource-aggregate-v1"
RESOURCE_CELL_SCHEMA = "wmf-development-resource-cell-measurement-v1"
RESOURCE_FILE_INVENTORY_SCHEMA = "wmf-development-resource-file-inventory-v1"

JOB_ID = "development-resource-compiler-formal-001"
WORKER_ROLE = "wmf-forecast-0912-worker-09"
MODE = "formal_full"
MAX_WALL_SECONDS = 43200
PUBLISH_LOG_TAIL_BYTES = 8192
MANIFEST_RELATIVE = Path("raw/resource_input_manifest.json")
BUNDLE_RELATIVE = Path("raw/resource_bundle")
PUBLISH_RECEIPT_NAME = "development_resource_job_receipt.json"
PUBLISH_FAILURE_NAME = "development_resource_job_failure.json"
MODELS = ("N3", "D1")
LAYOUTS = ("D01", "D02", "D03", "D04")
EXPECTED_CELLS = 32
EXPECTED_REQUESTS = 1152
EXPECTED_ACTIONS = 14400
EXPECTED_BUNDLE_FILES = 35
PLANNED_CELLS_SHA256 = (
    "7d06120a56419877d1acdfdc498c6dc054bf6860ce6bda5b2a25f55ae3b4166e"
)
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class ResourceQueueError(RuntimeError):
    """The resource audit descriptor or detached runtime failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResourceQueueError(message)


def _zero_science_counts() -> dict[str, int]:
    return {
        "model_runtime_loads": 0,
        "model_servers_started": 0,
        "model_requests_issued_by_job": 0,
        "simulator_processes_started": 0,
        "physical_resets": 0,
        "robot_episodes": 0,
        "behavioral_actions_executed_by_job": 0,
        "behavioral_cells_launched_by_job": 0,
        "labels_created_by_job": 0,
    }


class TimingReceiptSpec:
    def __init__(self, model: str, job_id: str, digest: str, size: int) -> None:
        self.model = model
        self.job_id = job_id
        self.sha256 = digest
        self.bytes = size

    @property
    def receipt_path(self) -> Path:
        return CONTROL_ROOT / "jobs" / self.job_id / "publish/timing_job_receipt.json"

    @property
    def results_path(self) -> str:
        return f"results/jobs/{self.job_id}/publish/timing_job_receipt.json"


TIMING_RECEIPTS = {
    "N3": TimingReceiptSpec(
        "N3",
        "timing-n3-development-sidecar-001",
        "7c4aa556d37f170308ef5529489936e3fd7fbce819909d8e1d5937c16965f9d9",
        29935,
    ),
    "D1": TimingReceiptSpec(
        "D1",
        "timing-d1-development-sidecar-002",
        "75490aab357d8290afa10f0e1a85437794a37c81f6a9a0524c94c746198da68d",
        79824,
    ),
}


class D1ServerSpec:
    def __init__(
        self,
        layout: str,
        job_id: str,
        simulator_job_id: str,
        run_id: str,
        study_commit: str,
        digest: str,
        size: int,
    ) -> None:
        self.layout = layout
        self.job_id = job_id
        self.simulator_job_id = simulator_job_id
        self.run_id = run_id
        self.study_commit = study_commit
        self.sha256 = digest
        self.bytes = size

    @property
    def receipt_path(self) -> Path:
        return CONTROL_ROOT / "jobs" / self.job_id / "publish/d1_behavioral_server_receipt.json"

    @property
    def results_path(self) -> str:
        return f"results/jobs/{self.job_id}/publish/d1_behavioral_server_receipt.json"


D1_SERVERS = {
    "D01": D1ServerSpec(
        "D01", "d1-development-d01-server-003", "d1-development-d01-simulator-003",
        "d1-development-d01-003", "25ff299ca0d2b9964eb48286990ee2301cb207b8",
        "e64c90d479ff5579114c5631e1a7c975a794640013e721f2daee149ab8c0cb58", 12362,
    ),
    "D02": D1ServerSpec(
        "D02", "d1-development-d02-server-003", "d1-development-d02-simulator-003",
        "d1-development-d02-003", "25ff299ca0d2b9964eb48286990ee2301cb207b8",
        "2ec55cbd1237a2e58082a6991b5a0a4c44121ae6835ed01d1ee1043fc002fcfe", 12362,
    ),
    "D03": D1ServerSpec(
        "D03", "d1-development-d03-server-002", "d1-development-d03-simulator-002",
        "d1-development-d03-002", "64e84d1f7634e2a7fbaaf0e1ffd05bf021614517",
        "6ffd934c4d05f815ebc06dfe788f8fbbbd12d755c691654cd05aa64da2346ec0", 12362,
    ),
    "D04": D1ServerSpec(
        "D04", "d1-development-d04-server-003", "d1-development-d04-simulator-003",
        "d1-development-d04-003", "25ff299ca0d2b9964eb48286990ee2301cb207b8",
        "043424b7cbde31c7616ee7c9873ccc0b42f5077561005e9800654a483460dfcb", 12362,
    ),
}


def _fixed_d1_aggregate(
    layout: str,
    job_id: str,
    source_commit: str,
    receipt_sha256: str,
    rows: Sequence[tuple[int, str]],
) -> Any:
    raw_root = (
        RAW_ROOT
        / "behavioral/development/D1"
        / layout
        / "simulator_attempts"
        / job_id
    )
    cells = []
    for index, (size, digest) in enumerate(rows):
        condition = aggregate_support.CONDITION_ORDER[layout][index]
        cell_id = (
            f"wmf1__development__{layout}__D1__"
            f"{condition.replace('-', '__')}"
        )
        slug = cell_id.replace("__", "-")
        cells.append(
            aggregate_support.CellSpec(
                str(raw_root / "cells" / f"{index:02d}-{slug}" / "cell_receipt.json"),
                digest,
                size,
            )
        )
    return aggregate_support.AggregateSpec(
        "D1",
        layout,
        job_id,
        source_commit,
        receipt_sha256,
        16510,
        tuple(cells),
    )


RESOURCE_AGGREGATES = {
    "N3": dict(aggregate_support.N3_AGGREGATES),
    "D1": {
        "D01": _fixed_d1_aggregate(
            "D01",
            "d1-development-d01-simulator-003",
            "25ff299ca0d2b9964eb48286990ee2301cb207b8",
            "9b1fd7d7479ad275d4baa943b9d8d6a4ed1d223e2d9ca20b3bff0342711ec5eb",
            (
                (70885, "aee559b64e10a8cceb53dc624f604a1d7f73954c0b669fb3190b14e7648645fc"),
                (70935, "f4cb1f1a1b38dc99c5505e446a8596417718bc34108142cf88e8adda57eac754"),
                (70899, "838675702a1da02fa38481917c0f07c46cbabb878bd86e1a5a5ddd4370b67bc3"),
                (70923, "a7ec5e05f368beaa90a19c0a1bd1d768f0d2c1917659b8df25c810aa7f873fdc"),
            ),
        ),
        "D02": _fixed_d1_aggregate(
            "D02",
            "d1-development-d02-simulator-003",
            "25ff299ca0d2b9964eb48286990ee2301cb207b8",
            "d1dc66c57d36cb3e620731c69a6e41fdf52c3cc2a797c2dae7ee36d63d10c6c9",
            (
                (70892, "225fd27f2411d0e21ceb8ef8eee8d47891f586fd5190f015550270d071b4de78"),
                (70873, "0b2dfa0e7d163c9b89539289291a2eb9c959bfa5f9724d64e5df8b3377993d4c"),
                (70878, "dc72aa1a6a08d50393f81a65e6fd7a777877b28e1448a3341abc77aa7c4c7a32"),
                (70887, "2b7b7fa8fc9e2dd2cbe040727a0994f364ba75f23ad6ca63c714e6e6d1d1f15f"),
            ),
        ),
        "D03": _fixed_d1_aggregate(
            "D03",
            "d1-development-d03-simulator-002",
            "64e84d1f7634e2a7fbaaf0e1ffd05bf021614517",
            "eb7f4f894b1e56c1b8971ee1155d77e7aeb1b62c9c6cfc41de0483e3baf2dd4f",
            (
                (70902, "a755f3f806ea73a395268d7cda38748b1bb6a7cdd60c7d4ee5e9872d7f912e26"),
                (70892, "6560cf4ea7b11187e9fd525a2eac722004bf7945283654257d4a4cbd47ee034f"),
                (70879, "c3af55faf13a6947493435c89f88adb9af71b475ee6f3e1fc24256abd9d7e6fb"),
                (70888, "f942a38a88d6293a58519594d329089f8f33b86de23f2bcd702b6f20fbd12b03"),
            ),
        ),
        "D04": _fixed_d1_aggregate(
            "D04",
            "d1-development-d04-simulator-003",
            "25ff299ca0d2b9964eb48286990ee2301cb207b8",
            "78a9b8b7095e48ce5412a902a7120f3c9ad71e3edb6eeeddb228720fe1b458fa",
            (
                (70864, "5ee363b92407a957b1ec5190643780722fdf6a55c2e7a5a9a72d7b2adaf8b1f5"),
                (70876, "b7172bccec6f9d4d72c0da204401e59c45787449780b6dba397539348666b632"),
                (70888, "9f72821183a135430026d34b2fa996b0a0062ef8d2f7f705e8c6e0a4782df050"),
                (70877, "cc45bb066436bd510938137448216450ed6de1e10a7f3741bd84a25f80e65154"),
            ),
        ),
    },
}


class QueueSnapshotSpec:
    def __init__(
        self,
        job_id: str,
        worker_id: str,
        source_commit: str,
        descriptor_sha256: str,
        digest: str,
        size: int,
    ) -> None:
        self.job_id = job_id
        self.worker_id = worker_id
        self.source_commit = source_commit
        self.descriptor_sha256 = descriptor_sha256
        self.sha256 = digest
        self.bytes = size

    @property
    def results_path(self) -> str:
        return f"results/{NAMESPACE}/jobs/{self.job_id}.json"

    @property
    def pvc_snapshot_path(self) -> Path:
        return CONTROL_ROOT / "results-git" / self.results_path


QUEUE_SNAPSHOTS = {
    row.job_id: row
    for row in (
        QueueSnapshotSpec("n3-development-d01-002", "wmf-forecast-0912-worker-n3-00", "91aaf8f5337bbf3a03e7cb65062ae1df77b6bb12", "48ab701dcfd44e815ef924797129bed68ec9ba2a8a5f124cea7437474930f7bb", "ffac647702ff4bb71ad1130934ecaac7caaf0ba85e12947fb783e997e43ad410", 1502),
        QueueSnapshotSpec("n3-development-d02-002", "wmf-forecast-0912-worker-n3-00", "c5b5368d852ecc9356b9df90cc5e6f62bfaecde8", "4caf24ee91f57a43710320fb648a3dc0e347d9f7c1719b4dae5e2438c3fb1892", "f78a85d2851e33f3d913241141ca686d4d5bc5af5aa26be5555b02a8706e305d", 1504),
        QueueSnapshotSpec("n3-development-d03-001", "wmf-forecast-0912-worker-n3-00", "91aaf8f5337bbf3a03e7cb65062ae1df77b6bb12", "1767323f4cf6911c8fe664da8b71937162663c6f13ab98dca67ad58edb3a48f9", "5288dc7fa33ece7d49e7d7c2d3f4336a39e2a59f68888d5338300800ecbf4b6b", 1503),
        QueueSnapshotSpec("n3-development-d04-001", "wmf-forecast-0912-worker-n3-00", "91aaf8f5337bbf3a03e7cb65062ae1df77b6bb12", "ba030c9c27ae4f0c35c4b5e68b397ad5eabf749d8cb77322d36ac75a7186c04e", "482063eadf2bbe6404af509e135287cba1e1d2a37ed171048ea8c9195080892f", 1504),
        QueueSnapshotSpec("d1-development-d01-simulator-003", "wmf-forecast-0912-worker-00", "25ff299ca0d2b9964eb48286990ee2301cb207b8", "d0d7c0fd5b0e74a91927766f749a27939c2c912778efbbee3b93b3e350392b3c", "d712d6b9986607f71d56fd111f6e1b26ea882b1ceb17dec77fd65b1b2f68cd88", 1540),
        QueueSnapshotSpec("d1-development-d02-simulator-003", "wmf-forecast-0912-worker-00", "25ff299ca0d2b9964eb48286990ee2301cb207b8", "b1f0272a7f3baa3940d125d129e4c2732dbb127e14b4b8bf19db3ec22071835f", "4f9f8bd2fb3fa61ace949ad9d09435963c4e95fca57cc3f71e3db96f81ee4a0b", 1540),
        QueueSnapshotSpec("d1-development-d03-simulator-002", "wmf-forecast-0912-worker-00", "64e84d1f7634e2a7fbaaf0e1ffd05bf021614517", "deeacda4c09b490b0d9aad8d1bff71825308e6e0c0aa316f6da7208959c64d66", "a268555a780a7d9cd3328bd370fbe91c17845f6a621d9c1d95c4141db1bdc749", 1539),
        QueueSnapshotSpec("d1-development-d04-simulator-003", "wmf-forecast-0912-worker-00", "25ff299ca0d2b9964eb48286990ee2301cb207b8", "574b376a27f630c5f710e014c89f2217d0a353ab6794305731b33368cba52291", "6c54d1e99468eae12f08c10532b96b8b248803f40b4f318648820af5b5d2e334", 1541),
        QueueSnapshotSpec("d1-development-d01-server-003", "wmf-forecast-0912-worker-d1-00", "25ff299ca0d2b9964eb48286990ee2301cb207b8", "1082e92d6bc78c947ae144950fbc77e3293e967711b06b0f09121c7c45df1406", "3b803bc2a8f264e5e25770e59c1dd0c609badb6e6189abac4ae9be7e5b6d75fa", 1092),
        QueueSnapshotSpec("d1-development-d02-server-003", "wmf-forecast-0912-worker-d1-00", "25ff299ca0d2b9964eb48286990ee2301cb207b8", "987048a602744d25be79dbf41c88f37e1298af9c749a6dcf89c6e4abda808c31", "199e3b7a1ae66bd754c43a322da857831701958ebca7ab5f0b8f4eb065e8c0a6", 1092),
        QueueSnapshotSpec("d1-development-d03-server-002", "wmf-forecast-0912-worker-d1-00", "64e84d1f7634e2a7fbaaf0e1ffd05bf021614517", "851e139ab3002fd7ee386e13179c4f4384155693f8b4e26890f7dbc351d368d7", "bfa5c8070b13b1e4bfbe3abfa69d30541ed9200cf92b61750498e0e9fb72eeac", 1092),
        QueueSnapshotSpec("d1-development-d04-server-003", "wmf-forecast-0912-worker-d1-00", "25ff299ca0d2b9964eb48286990ee2301cb207b8", "ef0f8d9296903765f3d1ba84f0471597655f0cc31a84c9fea32e8b951e642c83", "3c2b19b6eccc7a229605c97a49b8e0c148cb1e80bd75680db7a7a79b30bca91a", 1093),
        QueueSnapshotSpec("timing-n3-development-sidecar-001", "wmf-forecast-0912-worker-09", "1f25ecf74b2342aa36e6d168fc1f34a5df273d85", "a6bcfcd6169e0221ad947421ecdbbd6cfdb0c88bc26781b7c662740d2bcd3d37", "736e075955d165fb8afee0ba9048656b9ba289ca36ee492d9dab47c1f6a48ffe", 5025),
        QueueSnapshotSpec("timing-d1-development-sidecar-002", "wmf-forecast-0912-worker-09", "fcaa098cc45a3f8e32fa0a72203fd64795fa456a", "45327d7bbfb4e1f4bd9ec1f6d673f0e76f561dc12c3a4f6a2348845a0b1b1a6d", "5d44b4953617692ed5f701b2e013086f0b6687a589d74d02765a701369589906", 5025),
    )
}


def _expected_contract() -> dict[str, Any]:
    value = queue.load_json(REPOSITORY_ROOT / RESOURCE_CONTRACT_RELATIVE, "resource contract")
    require(value.get("schema_version") == CONTRACT_SCHEMA, "resource contract schema changed")
    require(value.get("study_id") == STUDY_ID, "resource contract study changed")
    require(value.get("mode") == MODE, "resource contract mode changed")
    require(value.get("cohort") == {
        "models": ["N3", "D1"],
        "layout_pair_ids": list(LAYOUTS),
        "cells": EXPECTED_CELLS,
        "source_behavioral_requests": EXPECTED_REQUESTS,
        "source_behavioral_actions": EXPECTED_ACTIONS,
        "aggregate_receipts": 8,
        "d1_server_receipts": 4,
        "timing_sidecar_receipts": 2,
        "queue_wrapper_snapshots": 14,
    }, "resource contract cohort changed")
    require(value.get("science_counts") == _zero_science_counts(), "resource contract claims science")
    boundary = value.get("release_boundary")
    require(
        isinstance(boundary, Mapping)
        and boundary.get("safe_to_release_confirmation") is False
        and boundary.get("legacy_resource_receipts_emitted") is False
        and boundary.get("confirmation_jobs_emitted") is False,
        "resource contract release boundary changed",
    )
    return value


def _source_paths(root: Path) -> dict[str, Path]:
    return {
        "queue_wrapper": root / THIS_RELATIVE,
        "queue_support": root / QUEUE_SUPPORT_RELATIVE,
        "aggregate_support": root / AGGREGATE_SUPPORT_RELATIVE,
        "resource_compiler": root / RESOURCE_COMPILER_RELATIVE,
        "resource_contract": root / RESOURCE_CONTRACT_RELATIVE,
        "evidence_compiler": root / EVIDENCE_COMPILER_RELATIVE,
        "freeze_validator": root / FREEZE_VALIDATOR_RELATIVE,
        "timing_validator": root / TIMING_VALIDATOR_RELATIVE,
        "planned_cells": root / PLANNED_CELLS_RELATIVE,
    }


def _local_implementation() -> dict[str, dict[str, Any]]:
    identities = {name: queue.file_identity(path) for name, path in _source_paths(REPOSITORY_ROOT).items()}
    require(identities["planned_cells"]["sha256"] == PLANNED_CELLS_SHA256, "planned-cell CSV hash changed")
    _expected_contract()
    return identities


def _run_git(repository: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True,
            timeout=60,
            env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never"),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ResourceQueueError("local Git results read failed") from error
    require(result.returncode == 0, f"local Git results command failed: {arguments[0]}")
    return result.stdout


def _verified_results_commit(repository: Path, commit: str) -> str:
    require(isinstance(commit, str) and COMMIT_RE.fullmatch(commit) is not None, "results commit is invalid")
    kind = _run_git(repository, "cat-file", "-t", commit).decode("ascii", "strict").strip()
    require(kind == "commit", "results object is not a commit")
    return commit


def _result_paths() -> list[str]:
    paths = [spec.results_path for spec in TIMING_RECEIPTS.values()]
    paths.extend(
        f"results/jobs/{RESOURCE_AGGREGATES[model][layout].job_id}/publish/"
        + ("n3_behavioral_development_receipt.json" if model == "N3" else "d1_behavioral_development_receipt.json")
        for model in MODELS for layout in LAYOUTS
    )
    paths.extend(spec.results_path for spec in D1_SERVERS.values())
    paths.extend(spec.results_path for spec in QUEUE_SNAPSHOTS.values())
    require(len(paths) == 28 and len(set(paths)) == 28, "results input path inventory changed")
    return paths


@contextmanager
def _materialized_results(
    repository: Path, commit: str
) -> Iterator[tuple[Path, dict[str, Path]]]:
    repository = Path(repository).resolve()
    require(repository.is_dir(), "results repository is unavailable")
    commit = _verified_results_commit(repository, commit)
    with tempfile.TemporaryDirectory(prefix="wmf-resource-results-") as temporary:
        root = Path(temporary)
        paths: dict[str, Path] = {}
        for index, relative in enumerate(_result_paths()):
            payload = _run_git(repository, "show", f"{commit}:{relative}")
            target = root / f"{index:02d}.json"
            target.write_bytes(payload)
            paths[relative] = target
        yield root, paths


def _fixed_file(path: Path, *, digest: str, size: int, label: str) -> dict[str, Any]:
    identity = queue.file_identity(path)
    require(identity["sha256"] == digest, f"{label} hash changed")
    require(identity["bytes"] == size, f"{label} byte count changed")
    return identity


def _semantic_timing_receipt(path: Path, spec: TimingReceiptSpec) -> dict[str, Any]:
    _fixed_file(path, digest=spec.sha256, size=spec.bytes, label=f"{spec.model} timing receipt")
    value = queue.load_json(path, f"{spec.model} timing receipt")
    queue.verify_signed_document(value, f"{spec.model} timing receipt")
    exact = {
        "schema_version": "wmf-development-timing-sidecar-queue-job-v1",
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "model_id": spec.model,
        "job_id": spec.job_id,
        "status": "passed",
        "decision": "go",
        "physical_time_qualified": True,
        "development_timing_sidecar_valid": True,
        "safe_to_release_confirmation": False,
        "referenced_behavioral_cells": 16,
        "referenced_behavioral_model_requests": 240 if spec.model == "N3" else 912,
    }
    for key, wanted in exact.items():
        require(value.get(key) == wanted, f"{spec.model} timing receipt changed: {key}")
    require(value.get("science_counts") == {
        "model_runtime_loads": 0,
        "model_servers_started": 0,
        "model_requests_issued_by_job": 0,
        "physical_resets": 0,
        "robot_episodes": 0,
        "behavioral_actions_executed_by_job": 0,
        "behavioral_cells_launched_by_job": 0,
    }, f"{spec.model} timing receipt claims science")
    outputs = value.get("outputs")
    require(
        isinstance(outputs, Mapping)
        and isinstance(outputs.get("raw_request_inventory"), Mapping)
        and isinstance(outputs.get("raw_development_timing_sidecar"), Mapping),
        f"{spec.model} timing outputs are missing",
    )
    return {"path": str(spec.receipt_path), "sha256": spec.sha256, "bytes": spec.bytes}


def _semantic_d1_server(path: Path, spec: D1ServerSpec) -> dict[str, Any]:
    _fixed_file(path, digest=spec.sha256, size=spec.bytes, label=f"D1 {spec.layout} server receipt")
    value = queue.load_json(path, f"D1 {spec.layout} server receipt")
    exact = {
        "schema_version": "wmf-d1-behavioral-development-server-job-v1",
        "status": "passed",
        "exit_code": 0,
        "failure": None,
        "phase": "development",
        "layout_pair_id": spec.layout,
        "run_id": spec.run_id,
        "server_job_id": spec.job_id,
        "paired_simulator_job_id": spec.simulator_job_id,
        "study_commit": spec.study_commit,
        "all_server_children_reaped": True,
    }
    for key, wanted in exact.items():
        require(value.get(key) == wanted, f"D1 {spec.layout} server receipt changed: {key}")
    process = value.get("server_process_exit")
    require(
        isinstance(process, Mapping)
        and process.get("status") == "reaped"
        and process.get("reaped") is True,
        f"D1 {spec.layout} server child was not reaped",
    )
    for key in ("queue_descriptor", "topology", "server_ready"):
        descriptor = value.get(key)
        require(
            isinstance(descriptor, Mapping)
            and isinstance(descriptor.get("path"), str)
            and type(descriptor.get("bytes")) is int
            and descriptor["bytes"] > 0
            and isinstance(descriptor.get("sha256"), str)
            and queue.SHA256_RE.fullmatch(descriptor["sha256"]) is not None,
            f"D1 {spec.layout} server {key} descriptor changed",
        )
    return {"path": str(spec.receipt_path), "sha256": spec.sha256, "bytes": spec.bytes}


def _semantic_queue_snapshot(path: Path, spec: QueueSnapshotSpec) -> dict[str, Any]:
    _fixed_file(path, digest=spec.sha256, size=spec.bytes, label=f"{spec.job_id} queue snapshot")
    value = queue.load_json(path, f"{spec.job_id} queue snapshot")
    exact = {
        "job_id": spec.job_id,
        "worker_id": spec.worker_id,
        "source_commit": spec.source_commit,
        "descriptor_sha256": spec.descriptor_sha256,
        "status": "succeeded",
        "returncode": 0,
        "error_type": None,
        "child_reaped": True,
    }
    for key, wanted in exact.items():
        require(value.get(key) == wanted, f"{spec.job_id} queue snapshot changed: {key}")
    require(
        type(value.get("wall_seconds")) in (int, float)
        and not isinstance(value.get("wall_seconds"), bool)
        and 0 < float(value["wall_seconds"]),
        f"{spec.job_id} queue wall seconds are invalid",
    )
    return {"path": str(spec.pvc_snapshot_path), "sha256": spec.sha256, "bytes": spec.bytes}


def validate_results_inputs(
    repository: Path, results_commit: str
) -> dict[str, Any]:
    with _materialized_results(repository, results_commit) as (_, paths):
        timing = {
            model: _semantic_timing_receipt(paths[spec.results_path], spec)
            for model, spec in TIMING_RECEIPTS.items()
        }
        aggregates: dict[tuple[str, str], dict[str, Any]] = {}
        for model in MODELS:
            for layout in LAYOUTS:
                spec = RESOURCE_AGGREGATES[model][layout]
                name = (
                    "n3_behavioral_development_receipt.json"
                    if model == "N3" else "d1_behavioral_development_receipt.json"
                )
                relative = f"results/jobs/{spec.job_id}/publish/{name}"
                try:
                    evidence = aggregate_support.validate_local_aggregate_receipt(
                        paths[relative], str(spec.receipt_sha256), spec
                    )
                except BaseException as error:
                    raise ResourceQueueError(f"{model} {layout} aggregate gate failed: {error}") from error
                descriptor = evidence.get("aggregate_receipt")
                require(isinstance(descriptor, Mapping), f"{model} {layout} aggregate descriptor is missing")
                aggregates[(model, layout)] = dict(descriptor)
        servers = {
            layout: _semantic_d1_server(paths[spec.results_path], spec)
            for layout, spec in D1_SERVERS.items()
        }
        snapshots = {
            job_id: _semantic_queue_snapshot(paths[spec.results_path], spec)
            for job_id, spec in QUEUE_SNAPSHOTS.items()
        }
    return {
        "results_commit": results_commit,
        "timing_sidecar_receipts": timing,
        "aggregate_receipts": aggregates,
        "d1_server_receipts": servers,
        "queue_wrapper_snapshots": snapshots,
    }


def _hash_argv(implementation: Mapping[str, Mapping[str, Any]]) -> list[str]:
    argv: list[str] = []
    for name in _source_paths(REPOSITORY_ROOT):
        argv.extend((f"--{name.replace('_', '-')}-sha256", str(implementation[name]["sha256"])))
    return argv


def _input_argv(inputs: Mapping[str, Any]) -> list[str]:
    argv: list[str] = ["--results-commit", str(inputs["results_commit"])]
    for model in MODELS:
        row = inputs["timing_sidecar_receipts"][model]
        argv.extend(("--timing-receipt", model, row["path"], row["sha256"], str(row["bytes"])))
    for model in MODELS:
        for layout in LAYOUTS:
            row = inputs["aggregate_receipts"][(model, layout)]
            argv.extend(("--aggregate-receipt", model, layout, row["path"], row["sha256"], str(row["bytes"])))
    for layout in LAYOUTS:
        row = inputs["d1_server_receipts"][layout]
        argv.extend(("--d1-server-receipt", layout, row["path"], row["sha256"], str(row["bytes"])))
    for job_id in sorted(QUEUE_SNAPSHOTS):
        row = inputs["queue_wrapper_snapshots"][job_id]
        argv.extend(("--queue-snapshot", job_id, row["path"], row["sha256"], str(row["bytes"])))
    return argv


def _job_descriptor(
    *,
    study_commit: str,
    implementation: Mapping[str, Mapping[str, Any]],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    commit = queue._verified_commit(study_commit)
    require(set(implementation) == set(_source_paths(REPOSITORY_ROOT)), "resource implementation inventory changed")
    argv = [
        str(ROBOLAB_PYTHON),
        "{source_root}/" + str(THIS_RELATIVE),
        "formal-full",
        "--source-root", "{source_root}",
        "--study-commit", commit,
        "--job-dir", "{job_dir}",
        "--job-id", JOB_ID,
        "--expected-role", WORKER_ROLE,
    ]
    argv.extend(_hash_argv(implementation))
    argv.extend(_input_argv(inputs))
    require(len(argv) <= 256, "resource descriptor argv exceeds queue bound")
    return {
        "job_id": JOB_ID,
        "released": True,
        "source_commit": commit,
        "role": WORKER_ROLE,
        "argv": argv,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "publish_log_tail_bytes": PUBLISH_LOG_TAIL_BYTES,
    }


def build_formal_wave(
    *, study_commit: str, repository: Path, results_commit: str
) -> dict[str, Any]:
    implementation = _local_implementation()
    inputs = validate_results_inputs(repository, results_commit)
    descriptor = _job_descriptor(
        study_commit=study_commit, implementation=implementation, inputs=inputs
    )
    return {
        "schema_version": WAVE_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "mode": MODE,
        "source_commit": descriptor["source_commit"],
        "results_commit": results_commit,
        "status": "descriptor_only_not_dispatched_all_immutable_evidence_gates_passed",
        "jobs": [descriptor],
        "implementation": implementation,
        "inputs": {
            "timing_sidecar_receipts": [
                {"model_id": model, "receipt": inputs["timing_sidecar_receipts"][model]}
                for model in MODELS
            ],
            "aggregate_receipts": [
                {"model_id": model, "layout_pair_id": layout, "receipt": inputs["aggregate_receipts"][(model, layout)]}
                for model in MODELS for layout in LAYOUTS
            ],
            "d1_server_receipts": [
                {"layout_pair_id": layout, "receipt": inputs["d1_server_receipts"][layout]}
                for layout in LAYOUTS
            ],
            "queue_wrapper_snapshots": [
                {"job_id": job_id, "snapshot": inputs["queue_wrapper_snapshots"][job_id]}
                for job_id in sorted(QUEUE_SNAPSHOTS)
            ],
        },
        "expected_authenticated_cells": EXPECTED_CELLS,
        "expected_source_behavioral_requests": EXPECTED_REQUESTS,
        "expected_source_behavioral_actions": EXPECTED_ACTIONS,
        "science_counts": _zero_science_counts(),
        "resource_release_gate_complete_after_pass": False,
        "safe_to_release_confirmation": False,
        "claim_boundary": (
            "One detached CPU-only audit of immutable completed development evidence. "
            "It preserves unavailable measurements as missing and cannot release confirmation."
        ),
    }


def _parse_runtime_rows(
    rows: Sequence[Sequence[str]],
    *,
    width: int,
    label: str,
) -> list[list[str]]:
    output: list[list[str]] = []
    for row in rows:
        require(len(row) == width, f"{label} row width changed")
        output.append(list(row))
    return output


def _runtime_implementation(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        name: {"sha256": queue._verified_sha(getattr(args, f"{name}_sha256"), f"{name} digest")}
        for name in _source_paths(REPOSITORY_ROOT)
    }


def _runtime_inputs(args: argparse.Namespace) -> dict[str, Any]:
    results_commit = args.results_commit
    require(isinstance(results_commit, str) and COMMIT_RE.fullmatch(results_commit), "runtime results commit changed")
    timing_rows = _parse_runtime_rows(args.timing_receipt, width=4, label="timing receipt")
    timing: dict[str, dict[str, Any]] = {}
    for model, path, digest, size_raw in timing_rows:
        require(model in TIMING_RECEIPTS and model not in timing, "runtime timing receipt identity changed")
        spec = TIMING_RECEIPTS[model]
        size = int(size_raw)
        require(Path(path) == spec.receipt_path and digest == spec.sha256 and size == spec.bytes, f"runtime {model} timing receipt changed")
        timing[model] = {"path": path, "sha256": digest, "bytes": size}
    require(set(timing) == set(MODELS), "runtime timing receipt cohort changed")

    aggregate_rows = _parse_runtime_rows(args.aggregate_receipt, width=5, label="aggregate receipt")
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    for model, layout, path, digest, size_raw in aggregate_rows:
        require(model in MODELS and layout in LAYOUTS and (model, layout) not in aggregates, "runtime aggregate identity changed")
        spec = RESOURCE_AGGREGATES[model][layout]
        size = int(size_raw)
        require(
            Path(path) == spec.receipt_path
            and digest == spec.receipt_sha256
            and size == spec.receipt_bytes,
            f"runtime {model} {layout} aggregate changed",
        )
        aggregates[(model, layout)] = {"path": path, "sha256": digest, "bytes": size}
    require(set(aggregates) == {(model, layout) for model in MODELS for layout in LAYOUTS}, "runtime aggregate cohort changed")

    server_rows = _parse_runtime_rows(args.d1_server_receipt, width=4, label="D1 server receipt")
    servers: dict[str, dict[str, Any]] = {}
    for layout, path, digest, size_raw in server_rows:
        require(layout in D1_SERVERS and layout not in servers, "runtime D1 server identity changed")
        spec = D1_SERVERS[layout]
        size = int(size_raw)
        require(Path(path) == spec.receipt_path and digest == spec.sha256 and size == spec.bytes, f"runtime D1 {layout} server receipt changed")
        servers[layout] = {"path": path, "sha256": digest, "bytes": size}
    require(set(servers) == set(LAYOUTS), "runtime D1 server cohort changed")

    snapshot_rows = _parse_runtime_rows(args.queue_snapshot, width=4, label="queue snapshot")
    snapshots: dict[str, dict[str, Any]] = {}
    for job_id, path, digest, size_raw in snapshot_rows:
        require(job_id in QUEUE_SNAPSHOTS and job_id not in snapshots, "runtime queue snapshot identity changed")
        spec = QUEUE_SNAPSHOTS[job_id]
        size = int(size_raw)
        require(Path(path) == spec.pvc_snapshot_path and digest == spec.sha256 and size == spec.bytes, f"runtime {job_id} queue snapshot changed")
        snapshots[job_id] = {"path": path, "sha256": digest, "bytes": size}
    require(set(snapshots) == set(QUEUE_SNAPSHOTS), "runtime queue snapshot cohort changed")
    return {
        "results_commit": results_commit,
        "timing_sidecar_receipts": timing,
        "aggregate_receipts": aggregates,
        "d1_server_receipts": servers,
        "queue_wrapper_snapshots": snapshots,
    }


def _runtime_descriptor(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    require(args.command == "formal-full", "runtime resource mode changed")
    require(args.job_id == JOB_ID, "runtime resource job ID changed")
    require(args.expected_role == WORKER_ROLE, "runtime resource role changed")
    implementation = _runtime_implementation(args)
    require(implementation["planned_cells"]["sha256"] == PLANNED_CELLS_SHA256, "runtime planned-cell hash changed")
    inputs = _runtime_inputs(args)
    descriptor = _job_descriptor(
        study_commit=args.study_commit, implementation=implementation, inputs=inputs
    )
    return descriptor, implementation, inputs


def _exact_context_paths(context: Any) -> None:
    require(
        context.job_dir == CONTROL_ROOT / "jobs" / JOB_ID,
        "queue context job directory is not exact",
    )
    require(
        context.source_root == CONTROL_ROOT / "sources" / context.study_commit,
        "queue context source root is not exact",
    )


def _validate_staged_implementation(
    source_root: Path,
    expected: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], ModuleType]:
    paths = _source_paths(source_root)
    require(set(paths) == set(expected), "staged resource implementation inventory changed")
    observed: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        identity = queue.file_identity(path)
        require(identity["sha256"] == expected[name]["sha256"], f"staged {name} hash changed")
        observed[name] = identity
    require(observed["planned_cells"]["sha256"] == PLANNED_CELLS_SHA256, "staged planned cells changed")
    contract = queue.load_json(observed["resource_contract"]["path"], "staged resource contract")
    require(contract.get("schema_version") == CONTRACT_SCHEMA, "staged resource contract schema changed")
    require(contract.get("science_counts") == _zero_science_counts(), "staged resource contract claims science")
    require(contract.get("release_boundary", {}).get("safe_to_release_confirmation") is False, "staged contract release boundary changed")
    compiler = _load_module(
        Path(observed["resource_compiler"]["path"]),
        "wmf_detached_development_resource_compiler",
    )
    exact_interface = {
        "INPUT_SCHEMA": COMPILER_INPUT_SCHEMA,
        "COMPILER_RECEIPT_SCHEMA": COMPILER_RECEIPT_SCHEMA,
        "AGGREGATE_SCHEMA": RESOURCE_AGGREGATE_SCHEMA,
        "CELL_SCHEMA": RESOURCE_CELL_SCHEMA,
        "FILE_INVENTORY_SCHEMA": RESOURCE_FILE_INVENTORY_SCHEMA,
        "STUDY_ID": STUDY_ID,
        "MODE": MODE,
        "MODELS": MODELS,
        "LAYOUTS": LAYOUTS,
        "EXPECTED_CELLS": EXPECTED_CELLS,
        "EXPECTED_REQUESTS": EXPECTED_REQUESTS,
        "EXPECTED_ACTIONS": EXPECTED_ACTIONS,
        "EXECUTION_AUTHORIZATION": {
            "status": "authorized_bounded_core_study",
            "selected_execution_host": "GM cluster",
            "selected_kubernetes_namespace": "211247-prod",
            "authorized_block_counts": {
                "recording_pilot_max": 2,
                "development": 8,
                "confirmation": 48,
                "maximum_core": 58,
            },
            "authorized_behavioral_cell_counts": {
                "recording_pilot_max": 8,
                "development": 32,
                "confirmation": 192,
                "maximum_core": 232,
            },
            "confirmation_cells_authorized_by_model": {"N3": 96, "D1": 96},
            "measured_safe_parallel_blocks_by_model": None,
            "measured_gpu_memory_envelope_bytes_per_gpu": None,
            "measurement_freeze_complete": False,
            "user_authorization_pending": False,
            "boundary": (
                "Authorization does not substitute for measured GPU peaks, annotation time, "
                "or the pre-confirmation safe topology/concurrency freeze."
            ),
        },
    }
    for name, wanted in exact_interface.items():
        require(getattr(compiler, name, None) == wanted, f"resource compiler interface changed: {name}")
    require(
        Path(getattr(compiler, "EVIDENCE_COMPILER_PATH", "missing")).resolve()
        == Path(observed["evidence_compiler"]["path"])
        and Path(getattr(compiler, "TIMING_VALIDATOR_PATH", "missing")).resolve()
        == Path(observed["timing_validator"]["path"])
        and Path(getattr(compiler, "CONTRACT_PATH", "missing")).resolve()
        == Path(observed["resource_contract"]["path"]),
        "resource compiler loaded a different staged dependency",
    )
    return observed, compiler


def _reopen_runtime_inputs(inputs: Mapping[str, Any]) -> None:
    for model in MODELS:
        spec = TIMING_RECEIPTS[model]
        _semantic_timing_receipt(spec.receipt_path, spec)
    for model in MODELS:
        for layout in LAYOUTS:
            spec = RESOURCE_AGGREGATES[model][layout]
            try:
                aggregate_support.validate_local_aggregate_receipt(
                    spec.receipt_path, str(spec.receipt_sha256), spec
                )
            except BaseException as error:
                raise ResourceQueueError(f"runtime {model} {layout} aggregate gate failed: {error}") from error
    for layout, spec in D1_SERVERS.items():
        _semantic_d1_server(spec.receipt_path, spec)
    for job_id, spec in QUEUE_SNAPSHOTS.items():
        _semantic_queue_snapshot(spec.pvc_snapshot_path, spec)
    require(inputs["results_commit"], "runtime results commit provenance is missing")


def _compiler_manifest(
    *, source_root: Path, inputs: Mapping[str, Any]
) -> dict[str, Any]:
    planned = queue.file_identity(source_root / PLANNED_CELLS_RELATIVE)
    require(planned["sha256"] == PLANNED_CELLS_SHA256, "manifest planned cells changed")
    return {
        "schema_version": COMPILER_INPUT_SCHEMA,
        "study_id": STUDY_ID,
        "mode": MODE,
        "raw_root": str(RAW_ROOT),
        "planned_cells": planned,
        "timing_sidecar_receipts": [
            {"model_id": model, "receipt": inputs["timing_sidecar_receipts"][model]}
            for model in MODELS
        ],
        "aggregate_receipts": [
            {"model_id": model, "layout_pair_id": layout, "receipt": inputs["aggregate_receipts"][(model, layout)]}
            for model in MODELS for layout in LAYOUTS
        ],
        "d1_server_receipts": [
            {"layout_pair_id": layout, "receipt": inputs["d1_server_receipts"][layout]}
            for layout in LAYOUTS
        ],
        "queue_wrapper_snapshots": [
            {"job_id": job_id, "snapshot": inputs["queue_wrapper_snapshots"][job_id]}
            for job_id in sorted(QUEUE_SNAPSHOTS)
        ],
    }


def _inventory_bundle(bundle: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    require(bundle == CONTROL_ROOT / "jobs" / JOB_ID / BUNDLE_RELATIVE, "resource bundle path changed")
    require(bundle.is_dir() and not bundle.is_symlink(), "resource bundle is unavailable")
    descriptors: list[dict[str, Any]] = []
    by_relative: dict[str, dict[str, Any]] = {}
    for path in sorted(bundle.rglob("*")):
        require(not path.is_symlink(), f"resource bundle contains symlink: {path}")
        if path.is_dir():
            continue
        require(path.is_file(), f"resource bundle contains non-file: {path}")
        relative = str(path.relative_to(bundle))
        require(relative not in by_relative, "resource bundle path duplicated")
        identity = queue.file_identity(path)
        descriptor = {"path": relative, "sha256": identity["sha256"], "bytes": identity["bytes"]}
        descriptors.append(descriptor)
        by_relative[relative] = descriptor
    require(len(descriptors) == EXPECTED_BUNDLE_FILES, "resource bundle file count changed")
    return descriptors, by_relative


def _validate_compiler_output(
    *, compiler: ModuleType, returned: Mapping[str, Any], bundle: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    validated = compiler.validate_resource_bundle(bundle)
    require(validated == returned, "returned resource receipt differs from bundle")
    receipt_path = bundle / "resource_compiler_receipt.json"
    receipt = queue.load_json(receipt_path, "resource compiler receipt")
    queue.verify_signed_document(receipt, "resource compiler receipt")
    require(receipt == returned, "resource compiler receipt changed after validation")
    exact = {
        "schema_version": COMPILER_RECEIPT_SCHEMA,
        "study_id": STUDY_ID,
        "mode": MODE,
        "status": "compiled_complete_with_declared_missingness",
        "resource_release_gate_complete": False,
        "safe_to_release_confirmation": False,
        "confirmation_released": False,
        "behavioral_policy_skill_evaluated": False,
        "science_activity": _zero_science_counts(),
    }
    for key, wanted in exact.items():
        require(receipt.get(key) == wanted, f"resource compiler receipt changed: {key}")
    counts = receipt.get("counts")
    require(
        isinstance(counts, Mapping)
        and counts.get("authenticated_cells") == EXPECTED_CELLS
        and counts.get("authenticated_source_behavioral_requests") == EXPECTED_REQUESTS
        and counts.get("authenticated_source_behavioral_actions") == EXPECTED_ACTIONS
        and type(counts.get("inventoried_raw_files")) is int
        and counts["inventoried_raw_files"] > 0
        and type(counts.get("inventoried_unique_resolved_path_bytes")) is int
        and counts["inventoried_unique_resolved_path_bytes"] > 0,
        "resource compiler counts changed",
    )
    inventory, files = _inventory_bundle(bundle)
    for key, relative in (
        ("resource_aggregate", "resource_aggregate.json"),
        ("resource_file_inventory", "resource_file_inventory.json"),
    ):
        observed = receipt.get("outputs", {}).get(key)
        require(isinstance(observed, Mapping), f"resource compiler {key} descriptor is missing")
        require(
            {name: observed.get(name) for name in ("path", "sha256", "bytes")}
            == files[relative],
            f"resource compiler {key} descriptor changed",
        )
    return receipt, queue.file_identity(receipt_path), inventory


def _write_failure_receipt(
    *, context: Any | None, job_dir: Path, error: BaseException
) -> None:
    expected_job = CONTROL_ROOT / "jobs" / JOB_ID
    try:
        supplied = Path(job_dir)
        if supplied != expected_job or supplied.resolve() != expected_job:
            return
        publish = supplied / "publish"
        if publish.exists() or publish.is_symlink():
            require(publish.is_dir() and not publish.is_symlink(), "failure publish path is invalid")
        else:
            publish.mkdir()
        success = publish / PUBLISH_RECEIPT_NAME
        failure = publish / PUBLISH_FAILURE_NAME
        entries = {path.name for path in publish.iterdir()}
        if success.exists() or success.is_symlink():
            return
        if entries == {PUBLISH_FAILURE_NAME} and failure.is_file() and not failure.is_symlink():
            return
        require(not entries, "failure publication inventory is not empty")
        receipt = queue.signed_document({
            "schema_version": JOB_RECEIPT_SCHEMA,
            "namespace": NAMESPACE,
            "study_id": STUDY_ID,
            "mode": MODE,
            "status": "technical_invalid",
            "decision": "no_go",
            "job_id": context.job_id if context is not None else JOB_ID,
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
            "resource_release_gate_complete": False,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "behavioral_policy_skill_evaluated": False,
            "claim_boundary": (
                "Failed CPU-only retained-evidence/resource validation. No model, simulator, "
                "reset, request, action, episode, label, or confirmation job was started."
            ),
            "completed_at_utc": queue.utc_now(),
        })
        queue.immutable_json(failure, receipt, maximum_bytes=512 * 1024)
    except BaseException:
        return


def run_formal_job(args: argparse.Namespace) -> dict[str, Any]:
    descriptor, expected_implementation, inputs = _runtime_descriptor(args)
    context = None
    try:
        context = queue.validate_queue_context(
            source_root=args.source_root,
            job_dir=args.job_dir,
            study_commit=args.study_commit,
            job_id=args.job_id,
            expected_role=args.expected_role,
            expected_descriptor=descriptor,
        )
        _exact_context_paths(context)
        implementation, compiler = _validate_staged_implementation(
            context.source_root, expected_implementation
        )
        _reopen_runtime_inputs(inputs)
        raw, publish = queue._prepare_output_directories(context.job_dir)
        manifest_path = context.job_dir / MANIFEST_RELATIVE
        bundle = context.job_dir / BUNDLE_RELATIVE
        require(manifest_path.parent == raw and bundle.parent == raw, "resource output paths changed")
        manifest = _compiler_manifest(source_root=context.source_root, inputs=inputs)
        queue.immutable_json(manifest_path, manifest, maximum_bytes=4 * 1024 * 1024)
        manifest_identity = queue.file_identity(manifest_path)
        returned = compiler.compile_resources(
            manifest_path, manifest_identity["sha256"], bundle
        )
        compiler_receipt, compiler_receipt_identity, bundle_inventory = _validate_compiler_output(
            compiler=compiler, returned=returned, bundle=bundle
        )
        output_inventory = queue.signed_document({
            "schema_version": OUTPUT_INVENTORY_SCHEMA,
            "namespace": NAMESPACE,
            "study_id": STUDY_ID,
            "mode": MODE,
            "job_id": context.job_id,
            "study_commit": context.study_commit,
            "bundle_root": str(bundle),
            "file_count": len(bundle_inventory),
            "files": bundle_inventory,
            "resource_release_gate_complete": False,
            "safe_to_release_confirmation": False,
        })
        inventory_path = raw / "resource_output_inventory.json"
        queue.immutable_json(inventory_path, output_inventory, maximum_bytes=2 * 1024 * 1024)
        inventory_identity = queue.file_identity(inventory_path)
        receipt = queue.signed_document({
            "schema_version": JOB_RECEIPT_SCHEMA,
            "namespace": NAMESPACE,
            "study_id": STUDY_ID,
            "mode": MODE,
            "status": "passed_with_declared_missingness",
            "decision": "no_go_resource_gate_incomplete",
            "job_id": context.job_id,
            "job_dir": str(context.job_dir),
            "study_commit": context.study_commit,
            "results_commit": inputs["results_commit"],
            "queue_role": context.role,
            "worker_id": context.worker_id,
            "runtime_identity": {
                "hostname": context.hostname,
                "pod_uid": context.pod_uid,
                "pid": os.getpid(),
            },
            "queue_descriptor": context.descriptor_identity,
            "queue_claim": context.claim_identity,
            "implementation": implementation,
            "inputs": {
                "resource_manifest": manifest_identity,
                "timing_sidecar_receipts": manifest["timing_sidecar_receipts"],
                "aggregate_receipts": manifest["aggregate_receipts"],
                "d1_server_receipts": manifest["d1_server_receipts"],
                "queue_wrapper_snapshots": manifest["queue_wrapper_snapshots"],
            },
            "outputs": {
                "resource_compiler_receipt": compiler_receipt_identity,
                "resource_output_inventory": inventory_identity,
                "resource_aggregate": compiler_receipt["outputs"]["resource_aggregate"],
                "resource_file_inventory": compiler_receipt["outputs"]["resource_file_inventory"],
                "cell_measurement_count": EXPECTED_CELLS,
            },
            "counts": compiler_receipt["counts"],
            "science_counts": _zero_science_counts(),
            "resource_release_gate_complete": False,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "behavioral_policy_skill_evaluated": False,
            "raw_outputs_recoverable_on_gm_pvc": True,
            "published_files": [PUBLISH_RECEIPT_NAME],
            "execution_authorization": compiler.EXECUTION_AUTHORIZATION,
            "missing_release_requirements": [
                "N3 behavioral model-server and simulator GPU peaks",
                "D1 simulator and simultaneous all-process GPU peaks",
                "two-rater annotation and adjudication time",
                "measured confirmation execution topology, GPU-memory envelope, and safe concurrency freeze",
            ],
            "claim_boundary": (
                "CPU-only resource audit over 32 immutable completed cells. Measured fields "
                "and declared missingness are recoverable on the PVC; confirmation remains held."
            ),
            "completed_at_utc": queue.utc_now(),
        })
        require(
            publish.is_dir() and not publish.is_symlink() and not any(publish.iterdir()),
            "resource success publish inventory is not empty",
        )
        # Final fallible operation: the exception handler cannot publish a
        # contradictory failure beside an immutable success receipt.
        queue.immutable_json(
            publish / PUBLISH_RECEIPT_NAME, receipt, maximum_bytes=2 * 1024 * 1024
        )
        return receipt
    except BaseException as error:
        _write_failure_receipt(context=context, job_dir=Path(args.job_dir), error=error)
        raise


def _add_runtime_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--study-commit", required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-role", required=True)
    for name in _source_paths(REPOSITORY_ROOT):
        parser.add_argument(f"--{name.replace('_', '-')}-sha256", dest=f"{name}_sha256", required=True)
    parser.add_argument("--results-commit", required=True)
    parser.add_argument("--timing-receipt", nargs=4, action="append", required=True)
    parser.add_argument("--aggregate-receipt", nargs=5, action="append", required=True)
    parser.add_argument("--d1-server-receipt", nargs=4, action="append", required=True)
    parser.add_argument("--queue-snapshot", nargs=4, action="append", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    emit = commands.add_parser("emit-formal")
    emit.add_argument("--study-commit", required=True)
    emit.add_argument("--repository", type=Path, default=REPOSITORY_ROOT)
    emit.add_argument("--results-commit", required=True)
    emit.add_argument("--output", type=Path)
    runtime = commands.add_parser("formal-full")
    _add_runtime_inputs(runtime)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "emit-formal":
        wave = build_formal_wave(
            study_commit=args.study_commit,
            repository=args.repository,
            results_commit=args.results_commit,
        )
        queue._write_descriptor_output(args.output, wave)
        return 0
    receipt = run_formal_job(args)
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
        print(json.dumps({
            "status": "technical_failure",
            "error_type": type(error).__name__,
            "detail": str(error),
        }, sort_keys=True), file=sys.stderr, flush=True)
        raise
