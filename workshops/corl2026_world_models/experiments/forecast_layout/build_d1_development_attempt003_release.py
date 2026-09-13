#!/usr/bin/env python3
"""Build the exact D01/D02/D04 attempt003 queue fragment without dispatching it.

The attempt002 queue release is immutable evidence.  This builder reproduces
its fixture, capture, recorder, qualification, and P00 pilot bindings while
changing only the source/attempt identities and adding the attempt003 pair
admission timeout required by ``d1_development_block_jobs.py``.

The returned jobs are raw ``wmf-cluster-queue-v1`` entries.  Descriptor hashes
are calculated after applying the queue controller's deterministic
normalization (schema and namespace fields plus sorted, indented JSON).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


NAMESPACE = "wmf_ablation_001_20260912"
STUDY_ID = "WMF-ABLATION-001"
SOURCE_COMMIT = "25ff299ca0d2b9964eb48286990ee2301cb207b8"
RUNTIME_SHA256 = "ffc342087f1caf3837da1715d76b3b002eae1a40e709554e71284a4e4e28c632"
BASE_RELEASE_COMMIT = "7ba946fe7db3cc12d3999f94c2033ec95ec0d015"
BASE_RELEASE_RECORD_SHA256 = (
    "2a25e362fddc2dc9945f3510f37b14e4b8e29f8cf6113d7347e5ea7af78e02d6"
)
BASE_SOURCE_COMMIT = "64e84d1f7634e2a7fbaaf0e1ffd05bf021614517"
BASE_DESCRIPTOR_SHA256 = {
    "d1-development-d01-server-002": (
        "8500c2d292729fea40bd1d5745333bb432909d685feae812ac02accc500a1434"
    ),
    "d1-development-d01-simulator-002": (
        "c3aceacc7e56572a1223e009d09cc6c5e953ec9a96c7c3a38280e146966a8c3e"
    ),
    "d1-development-d02-server-002": (
        "7f9081ce17b8e8dc3cc4bcce3945c84e2d53437b98fb9abe8f895d6da7906e74"
    ),
    "d1-development-d02-simulator-002": (
        "23469894af5b1123efb87dbfc651afb35c274bb692a04ba6450eec4ae99fd380"
    ),
    "d1-development-d04-server-002": (
        "24789ab48609be3d7023351c4878c778677b9a1d5e155a45f152cba44905929a"
    ),
    "d1-development-d04-simulator-002": (
        "2bb3160b6a7ad684ade23d1c8a6861a17c449b03c5c7b598c4b34e1a05f06e5d"
    ),
}

ATTEMPT = "003"
LAYOUT_PAIR_IDS = ("D01", "D02", "D04")
SERVER_ROLE = "d1"
SIMULATOR_ROLE = "wmf-forecast-0912-worker-00"
SERVICE_HOST = "wmf-forecast-0912-d1"
SERVICE_PORT = 18021
PAIR_ADMISSION_TIMEOUT_SECONDS = 900
RUNNER = (
    "{source_root}/workshops/corl2026_world_models/experiments/forecast_layout/"
    "d1_development_block_jobs.py"
)
MAX_WALL_SECONDS = 60000
PUBLISH_LOG_TAIL_BYTES = 8192

SHARED_BINDINGS = {
    "recorder_receipt": (
        "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control/jobs/"
        "recorder-qualification-p00-003/publish/recorder_qualification_receipt.json"
    ),
    "recorder_receipt_sha256": (
        "0e3f02f37a2548e36ae3a45a38a1fac63c56cd8798732056f24b03d103991fde"
    ),
    "d1_qualification_receipt": (
        "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control/jobs/"
        "d1-first-live-005/publish/d1_qualification_job_receipt.json"
    ),
    "d1_qualification_receipt_sha256": (
        "3c856549999b9145dc07c30853a4c6d2968d09eb31c2db883f5eb1655d31627b"
    ),
    "pilot_simulator_receipt": (
        "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control/jobs/"
        "d1-p00-simulator-003/publish/d1_behavioral_pilot_receipt.json"
    ),
    "pilot_simulator_receipt_sha256": (
        "20b4068ee9748daad7ae5e8350e8b7cf36d62c686af51a0d4b9653f9cecb2ab0"
    ),
    "pilot_server_receipt": (
        "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control/jobs/"
        "d1-p00-server-003/publish/d1_behavioral_server_receipt.json"
    ),
    "pilot_server_receipt_sha256": (
        "95ef4351ffa494194ae6d9c4525acd8191b2d35438680f9284cad798d71f3201"
    ),
}

LAYOUT_BINDINGS = {
    "D01": {
        "candidate_id": "D01__candidate_00",
        "gate_receipt": (
            "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control/jobs/"
            "fixture-d01-candidate-00/publish/fixture_gate_receipt.json"
        ),
        "gate_receipt_sha256": (
            "ae001a2ff7c945ebb6f0fa20aec656257cdefff2195745e150d6c1b6a4a5bad9"
        ),
        "pose_manifest": (
            "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/"
            "fixed_observations/d01_pose_manifest.json"
        ),
        "pose_manifest_sha256": (
            "4b6a69592c0053d5117ef88a56c4e9e1f9037714d212c37f4b27c1fbf98f04fc"
        ),
        "capture_receipt": (
            "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control/jobs/"
            "fixed-observation-d01-001/publish/fixed_observation_job_receipt.json"
        ),
        "capture_receipt_sha256": (
            "8531d11584a36f2326074b287e42394a1f9cf3b04fd1016855f472152b547077"
        ),
    },
    "D02": {
        "candidate_id": "D02__candidate_00",
        "gate_receipt": (
            "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control/jobs/"
            "fixture-d02-candidate-00/publish/fixture_gate_receipt.json"
        ),
        "gate_receipt_sha256": (
            "e30a3dd1ad3d8621661a406a7c1d6c30034383f9b5ad6bed8a96c63015a477d6"
        ),
        "pose_manifest": (
            "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/"
            "fixed_observations/d02_pose_manifest.json"
        ),
        "pose_manifest_sha256": (
            "701c2d0acdd9ccb546673d1ae028de2838eb425ee42cd79df3cbc82829d760af"
        ),
        "capture_receipt": (
            "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control/jobs/"
            "fixed-observation-d02-001/publish/fixed_observation_job_receipt.json"
        ),
        "capture_receipt_sha256": (
            "a700280562d3640349fe8ec5a2594decf4a7b2b9f30450c9e9af55420bb907ac"
        ),
    },
    "D04": {
        "candidate_id": "D04__candidate_00",
        "gate_receipt": (
            "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control/jobs/"
            "fixture-d04-candidate-00/publish/fixture_gate_receipt.json"
        ),
        "gate_receipt_sha256": (
            "c700f4ab196d5aee19476ca9eb1f144651353f3767cb2524c0a210d77a225d6b"
        ),
        "pose_manifest": (
            "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/"
            "fixed_observations/d04_pose_manifest.json"
        ),
        "pose_manifest_sha256": (
            "e0b9d2eb2f9ff374b7828a51941ff11b22f190b381919e802740906f92b8f15e"
        ),
        "capture_receipt": (
            "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control/jobs/"
            "fixed-observation-d04-001/publish/fixed_observation_job_receipt.json"
        ),
        "capture_receipt_sha256": (
            "f943e9fa2a968bc34fea9ed7cc0cbc4508f86f1a28071a82972863dc602807bc"
        ),
    },
}

PREREQUISITE_PREFLIGHTS = {
    "D01": {
        "layout_pair_id": "D01",
        "job_id": "d1-development-d01-prerequisite-preflight-002",
        "receipt_results_path": (
            "results/jobs/d1-development-d01-prerequisite-preflight-002/publish/"
            "d1_development_prerequisite_preflight_receipt.json"
        ),
        "receipt_sha256": (
            "9da254c289e8aa1dbd340ab37ba52b32a76fd98503681fb0dcff6ac24b8a1ab3"
        ),
        "queue_result_status": "succeeded",
        "queue_returncode": 0,
        "child_reaped": True,
        "receipt_status": "passed",
        "receipt_decision": "go",
        "safe_to_release_behavioral_pair": True,
    },
    "D02": {
        "layout_pair_id": "D02",
        "job_id": "d1-development-d02-prerequisite-preflight-001",
        "receipt_results_path": (
            "results/jobs/d1-development-d02-prerequisite-preflight-001/publish/"
            "d1_development_prerequisite_preflight_receipt.json"
        ),
        "receipt_sha256": (
            "418365e8376bc29cd0afbda4b53361d829b7c5e0c685b48a480a1f1f15abe39f"
        ),
        "queue_result_status": "succeeded",
        "queue_returncode": 0,
        "child_reaped": True,
        "receipt_status": "passed",
        "receipt_decision": "go",
        "safe_to_release_behavioral_pair": True,
    },
    "D03": {
        "layout_pair_id": "D03",
        "job_id": "d1-development-d03-prerequisite-preflight-001",
        "receipt_results_path": (
            "results/jobs/d1-development-d03-prerequisite-preflight-001/publish/"
            "d1_development_prerequisite_preflight_receipt.json"
        ),
        "receipt_sha256": (
            "106f7b4eb3f774aaf759110c058dbdca74630f19b415b268fdd045934e621325"
        ),
        "queue_result_status": "succeeded",
        "queue_returncode": 0,
        "child_reaped": True,
        "receipt_status": "passed",
        "receipt_decision": "go",
        "safe_to_release_behavioral_pair": True,
    },
    "D04": {
        "layout_pair_id": "D04",
        "job_id": "d1-development-d04-prerequisite-preflight-001",
        "receipt_results_path": (
            "results/jobs/d1-development-d04-prerequisite-preflight-001/publish/"
            "d1_development_prerequisite_preflight_receipt.json"
        ),
        "receipt_sha256": (
            "6a51fa6abfa0c1ee1e534aa36fa3429e8c0fcd3077eb68770c6f72400df845a6"
        ),
        "queue_result_status": "succeeded",
        "queue_returncode": 0,
        "child_reaped": True,
        "receipt_status": "passed",
        "receipt_decision": "go",
        "safe_to_release_behavioral_pair": True,
    },
}

EXTERNAL_D03_TERMINAL = {
    "source_commit": BASE_SOURCE_COMMIT,
    "run_id": "d1-development-d03-002",
    "server_job_id": "d1-development-d03-server-002",
    "server_descriptor_sha256": (
        "851e139ab3002fd7ee386e13179c4f4384155693f8b4e26890f7dbc351d368d7"
    ),
    "server_receipt_sha256": (
        "6ffd934c4d05f815ebc06dfe788f8fbbbd12d755c691654cd05aa64da2346ec0"
    ),
    "simulator_job_id": "d1-development-d03-simulator-002",
    "simulator_descriptor_sha256": (
        "deeacda4c09b490b0d9aad8d1bff71825308e6e0c0aa316f6da7208959c64d66"
    ),
    "simulator_receipt_sha256": (
        "eb7f4f894b1e56c1b8971ee1155d77e7aeb1b62c9c6cfc41de0483e3baf2dd4f"
    ),
}

EXPECTED_DESCRIPTOR_SHA256 = {
    "d1-development-d01-server-003": (
        "1082e92d6bc78c947ae144950fbc77e3293e967711b06b0f09121c7c45df1406"
    ),
    "d1-development-d01-simulator-003": (
        "d0d7c0fd5b0e74a91927766f749a27939c2c912778efbbee3b93b3e350392b3c"
    ),
    "d1-development-d02-server-003": (
        "987048a602744d25be79dbf41c88f37e1298af9c749a6dcf89c6e4abda808c31"
    ),
    "d1-development-d02-simulator-003": (
        "b1f0272a7f3baa3940d125d129e4c2732dbb127e14b4b8bf19db3ec22071835f"
    ),
    "d1-development-d04-server-003": (
        "ef0f8d9296903765f3d1ba84f0471597655f0cc31a84c9fea32e8b951e642c83"
    ),
    "d1-development-d04-simulator-003": (
        "574b376a27f630c5f710e014c89f2217d0a353ab6794305731b33368cba52291"
    ),
}

SAFE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _identity(layout_pair_id: str) -> dict[str, str]:
    require(layout_pair_id in LAYOUT_PAIR_IDS, "layout is outside attempt003 release")
    stem = f"d1-development-{layout_pair_id.lower()}"
    return {
        "server_job_id": f"{stem}-server-{ATTEMPT}",
        "simulator_job_id": f"{stem}-simulator-{ATTEMPT}",
        "run_id": f"{stem}-{ATTEMPT}",
    }


def _common_argv(layout_pair_id: str, mode: str) -> list[str]:
    require(mode in {"server-job", "simulator-job"}, "unsupported mode")
    identity = _identity(layout_pair_id)
    own = (
        identity["server_job_id"]
        if mode == "server-job"
        else identity["simulator_job_id"]
    )
    peer = (
        identity["simulator_job_id"]
        if mode == "server-job"
        else identity["server_job_id"]
    )
    return [
        "/usr/bin/python3",
        RUNNER,
        mode,
        "--layout-pair-id",
        layout_pair_id,
        "--simulator-worker-role",
        SIMULATOR_ROLE,
        "--source-root",
        "{source_root}",
        "--study-commit",
        SOURCE_COMMIT,
        "--job-dir",
        "{job_dir}",
        "--job-id",
        own,
        "--simulator-job-id" if mode == "server-job" else "--server-job-id",
        peer,
        "--run-id",
        identity["run_id"],
    ]


def _binding_argv(layout_pair_id: str) -> list[str]:
    binding = LAYOUT_BINDINGS[layout_pair_id]
    return [
        "--candidate-id",
        binding["candidate_id"],
        "--gate-receipt",
        binding["gate_receipt"],
        "--gate-receipt-sha256",
        binding["gate_receipt_sha256"],
        "--pose-manifest",
        binding["pose_manifest"],
        "--pose-manifest-sha256",
        binding["pose_manifest_sha256"],
        "--capture-receipt",
        binding["capture_receipt"],
        "--capture-receipt-sha256",
        binding["capture_receipt_sha256"],
        "--recorder-receipt",
        SHARED_BINDINGS["recorder_receipt"],
        "--d1-qualification-receipt",
        SHARED_BINDINGS["d1_qualification_receipt"],
        "--d1-qualification-receipt-sha256",
        SHARED_BINDINGS["d1_qualification_receipt_sha256"],
        "--pilot-simulator-receipt",
        SHARED_BINDINGS["pilot_simulator_receipt"],
        "--pilot-simulator-receipt-sha256",
        SHARED_BINDINGS["pilot_simulator_receipt_sha256"],
        "--pilot-server-receipt",
        SHARED_BINDINGS["pilot_server_receipt"],
        "--pilot-server-receipt-sha256",
        SHARED_BINDINGS["pilot_server_receipt_sha256"],
        "--recorder-receipt-sha256",
        SHARED_BINDINGS["recorder_receipt_sha256"],
    ]


def build_descriptor(layout_pair_id: str, mode: str) -> dict[str, Any]:
    identity = _identity(layout_pair_id)
    argv = _common_argv(layout_pair_id, mode) + _binding_argv(layout_pair_id)
    if mode == "server-job":
        job_id = identity["server_job_id"]
        role = SERVER_ROLE
        argv.extend(("--port", str(SERVICE_PORT)))
    else:
        require(mode == "simulator-job", "unsupported mode")
        job_id = identity["simulator_job_id"]
        role = SIMULATOR_ROLE
        argv.extend(("--remote-host", SERVICE_HOST, "--remote-port", str(SERVICE_PORT)))
    argv.extend(("--pair-admission-timeout-seconds", str(PAIR_ADMISSION_TIMEOUT_SECONDS)))
    return {
        "job_id": job_id,
        "released": True,
        "source_commit": SOURCE_COMMIT,
        "role": role,
        "argv": argv,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "publish_log_tail_bytes": PUBLISH_LOG_TAIL_BYTES,
    }


def normalized_descriptor(job: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror ``cluster_queue.normalize_job`` for exact pre-release hashing."""

    allowed = {
        "job_id",
        "released",
        "source_commit",
        "role",
        "argv",
        "max_wall_seconds",
        "publish_log_tail_bytes",
    }
    require(set(job) == allowed, "raw descriptor fields changed")
    require(job["released"] is True, "descriptor is not released")
    require(
        isinstance(job["source_commit"], str)
        and SAFE_COMMIT.fullmatch(job["source_commit"]) is not None,
        "descriptor source commit is invalid",
    )
    return {
        "schema_version": "wmf-cluster-job-v1",
        "namespace": NAMESPACE,
        "job_id": job["job_id"],
        "released": True,
        "source_commit": job["source_commit"],
        "role": job["role"],
        "argv": list(job["argv"]),
        "max_wall_seconds": job["max_wall_seconds"],
        "publish_log_tail_bytes": job["publish_log_tail_bytes"],
    }


def descriptor_bytes(job: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(normalized_descriptor(job), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def descriptor_sha256(job: Mapping[str, Any]) -> str:
    return hashlib.sha256(descriptor_bytes(job)).hexdigest()


def build_release_fragment() -> dict[str, Any]:
    """Return the reviewed descriptor fragment; never edit the active queue."""

    jobs = [
        build_descriptor(layout, mode)
        for layout in LAYOUT_PAIR_IDS
        for mode in ("server-job", "simulator-job")
    ]
    hashes = {job["job_id"]: descriptor_sha256(job) for job in jobs}
    if EXPECTED_DESCRIPTOR_SHA256:
        require(hashes == EXPECTED_DESCRIPTOR_SHA256, "attempt003 descriptor hash drift")
    return {
        "schema_version": "wmf-d1-development-attempt003-descriptor-release-v1",
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "status": "descriptor_only_not_dispatched",
        "runtime_source_commit": SOURCE_COMMIT,
        "runtime_sha256": RUNTIME_SHA256,
        "base_attempt002_release_commit": BASE_RELEASE_COMMIT,
        "base_attempt002_release_record_sha256": BASE_RELEASE_RECORD_SHA256,
        "base_attempt002_descriptor_sha256": dict(BASE_DESCRIPTOR_SHA256),
        "pair_admission_timeout_seconds": PAIR_ADMISSION_TIMEOUT_SECONDS,
        "layout_pair_ids": list(LAYOUT_PAIR_IDS),
        "maximum_new_behavioral_cells": 12,
        "confirmation_inference_authorized": False,
        "prerequisite_preflights": dict(PREREQUISITE_PREFLIGHTS),
        "required_external_d03_terminal": dict(EXTERNAL_D03_TERMINAL),
        "descriptor_sha256": hashes,
        "jobs": jobs,
        "release_boundary": (
            "Descriptor fragment only. This builder does not edit cluster_queue.json, "
            "commit, push, dispatch, or start scientific work."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(
        json.dumps(build_release_fragment(), indent=2, sort_keys=True, allow_nan=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
