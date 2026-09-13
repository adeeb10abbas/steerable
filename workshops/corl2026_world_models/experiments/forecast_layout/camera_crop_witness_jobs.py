#!/usr/bin/env python3
"""Build and run the detached, CPU-only camera-crop replay witness.

``build-wave`` is descriptor-only.  It requires five explicit compact passed
receipts fetched from the results branch and never edits the active queue.
``run`` is the queue entry point: it revalidates the immutable descriptor,
claim, staged source, prerequisite PVC receipts and implementation hashes,
then launches the N3 and D1 byte replays in isolated process groups.  Only two
signed crop contracts and one signed zero-science job receipt are publishable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

FORECAST_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = FORECAST_ROOT.parents[1]
QUEUE_PATH = Path(__file__).with_name("forecast_timing_queue_jobs.py")
REPLAY_PATH = FORECAST_ROOT / "analysis/camera_crop_replay_witness.py"


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


queue = _load_module(QUEUE_PATH, "wmf_camera_crop_queue_support")
replay = _load_module(REPLAY_PATH, "wmf_camera_crop_replay_support")

NAMESPACE = queue.NAMESPACE
STUDY_ID = queue.STUDY_ID
CONTROL_ROOT = queue.CONTROL_ROOT
FORECAST_RELATIVE = queue.FORECAST_RELATIVE

THIS_RELATIVE = FORECAST_RELATIVE / "experiments/forecast_layout/camera_crop_witness_jobs.py"
QUEUE_RELATIVE = queue.THIS_RELATIVE
REPLAY_RELATIVE = FORECAST_RELATIVE / "analysis/camera_crop_replay_witness.py"
CONTRACT_RELATIVE = (
    FORECAST_RELATIVE / "experiments/forecast_layout/camera_crop_witness_contract.json"
)
N3_CONTRACT_RELATIVE = (
    FORECAST_RELATIVE / "experiments/forecast_layout/n3_first_live_contract.json"
)
D1_IDENTITY_RELATIVE = (
    FORECAST_RELATIVE / "experiments/forecast_layout/d1_identity_contract.json"
)
D1_CAPTURE_OVERLAY_RELATIVE = Path("experiments/dreamzero_droid/v2_robolab_client.py")

WAVE_SCHEMA = "wmf-camera-crop-witness-wave-v1"
JOB_RECEIPT_SCHEMA = "wmf-camera-crop-witness-queue-job-v1"
FAILURE_SCHEMA = "wmf-camera-crop-witness-queue-job-v1"
JOB_ID = "camera-crop-replay-witness-001"
WORKER_ROLE = "wmf-forecast-0912-worker-06"
MAX_WALL_SECONDS = 10800
PUBLISH_LOG_TAIL_BYTES = 8192
RAW_SUBDIR = Path("raw/camera_crop_witness")
SUCCESS_RECEIPT = "camera_crop_witness_job_receipt.json"
FAILURE_RECEIPT = "camera_crop_witness_job_failure.json"
MODEL_CONTRACT_NAMES = {
    "N3": "n3_camera_crop_contract.json",
    "D1": "d1_camera_crop_contract.json",
}

PREREQUISITE_CLUSTER_PATHS = {
    "n3_source_audit": CONTROL_ROOT / "jobs/timing-n3-source-audit-001/raw/n3_source_audit.json",
    "d1_source_audit": CONTROL_ROOT / "jobs/timing-d1-source-audit-001/raw/d1_source_audit.json",
    "n3_generation": CONTROL_ROOT / (
        "jobs/timing-n3-live-generation-p00-002/raw/n3_generation_publish/n3_qualification.json"
    ),
    "d1_generation": CONTROL_ROOT / (
        "jobs/timing-d1-normalize-generation-001/raw/d1_generation_probe.json"
    ),
    "d1_qualification": CONTROL_ROOT / (
        "jobs/d1-first-live-005/publish/d1_qualification_job_receipt.json"
    ),
}


class CameraCropQueueError(RuntimeError):
    """A descriptor, prerequisite, detached child, or result gate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CameraCropQueueError(message)


def zero_science_counts() -> dict[str, int]:
    return dict(replay.ZERO_SCIENCE_COUNTS)


def _source_paths(root: Path) -> dict[str, Path]:
    return {
        "queue_wrapper": root / THIS_RELATIVE,
        "queue_support": root / QUEUE_RELATIVE,
        "replay": root / REPLAY_RELATIVE,
        "runtime_contract": root / CONTRACT_RELATIVE,
        "n3_runtime_contract": root / N3_CONTRACT_RELATIVE,
        "d1_identity_contract": root / D1_IDENTITY_RELATIVE,
        "d1_capture_overlay": root / D1_CAPTURE_OVERLAY_RELATIVE,
    }


def _local_implementation() -> dict[str, dict[str, Any]]:
    return {
        name: queue.file_identity(path)
        for name, path in _source_paths(REPOSITORY_ROOT).items()
    }


def _runtime_contract(root: Path) -> dict[str, Any]:
    value = replay.load_json(root / CONTRACT_RELATIVE, "camera witness runtime contract")
    require(value.get("schema_version") == replay.RUNTIME_SCHEMA, "runtime contract schema changed")
    require(value.get("study_id") == STUDY_ID, "runtime contract study changed")
    require(value.get("job") == {
        "job_id": JOB_ID,
        "worker_role": WORKER_ROLE,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "cpu_only": True,
    }, "runtime contract queue identity changed")
    require(value.get("science_counts") == zero_science_counts(),
            "runtime contract science counts changed")
    require(value.get("simulator_state_render_used") is False
            and value.get("whole_frame_identity") is False
            and value.get("safe_to_release_confirmation") is False,
            "runtime contract authority changed")
    return value


def _validate_prerequisite_value(
    name: str,
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    require(value.get("schema_version") == expected["schema_version"],
            f"{name} schema changed")
    if expected.get("model_id") is not None:
        require(value.get("model_id") == expected["model_id"], f"{name} model changed")
    if expected.get("status") is not None:
        require(value.get("status") == expected["status"], f"{name} status changed")
    if expected.get("decision") is not None:
        require(value.get("decision") == expected["decision"], f"{name} decision changed")
    require(value.get("study_id", STUDY_ID) == STUDY_ID, f"{name} study changed")
    if name == "n3_source_audit":
        require(value.get("source", {}).get("commit")
                == "411d25b2e35bc441126f48c44a4b93e1c0564274", "N3 audit source changed")
    elif name == "d1_source_audit":
        require(value.get("source", {}).get("commit")
                == "ab790c198fbce33503358efbbd4187ce9a89adf3", "D1 audit source changed")
    elif name == "n3_generation":
        require(value.get("qualified") is True
                and value.get("generation_request_count") == 6
                and value.get("robot_episode_count") == 0,
                "N3 generation prerequisite is not the passed zero-policy probe")
    elif name == "d1_generation":
        require(value.get("model_returned_action_executed") is False
                and value.get("selected_request", {}).get("decoded_rgb_shape") == [9, 352, 640, 3],
                "D1 generation prerequisite changed")
    elif name == "d1_qualification":
        require(value.get("behavioral_episode_count") == 0
                and value.get("generation_request_count") == 6,
                "D1 qualification is not the passed zero-policy probe")


def validate_prerequisites(
    paths: Mapping[str, tuple[Path, str]], *, root: Path = REPOSITORY_ROOT
) -> dict[str, dict[str, Any]]:
    runtime = _runtime_contract(root)
    expected = runtime["prerequisite_receipts"]
    require(set(paths) == set(expected), "exactly five explicit prerequisite receipts are required")
    result: dict[str, dict[str, Any]] = {}
    seen: set[Path] = set()
    for name in sorted(expected):
        path, supplied_hash = paths[name]
        require(path not in seen, "prerequisite receipt paths are ambiguous")
        seen.add(path)
        wanted = expected[name]
        require(supplied_hash == wanted["sha256"], f"{name} supplied SHA-256 changed")
        identity = replay.file_identity(
            Path(path),
            label=f"builder prerequisite {name}",
            expected_sha256=wanted["sha256"],
            expected_bytes=wanted["bytes"],
        )
        value = replay.load_json(Path(path), f"builder prerequisite {name}")
        _validate_prerequisite_value(name, value, wanted)
        result[name] = identity
    return result


def _hash_argv(implementation: Mapping[str, Mapping[str, Any]]) -> list[str]:
    argv: list[str] = []
    for name in sorted(implementation):
        argv.extend(["--implementation", name, str(implementation[name]["sha256"])])
    return argv


def _job_descriptor(
    *, study_commit: str, implementation: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    commit = queue._verified_commit(study_commit)
    require(set(implementation) == set(_source_paths(REPOSITORY_ROOT)),
            "implementation inventory changed")
    argv = [
        "/usr/bin/python3",
        "{source_root}/" + str(THIS_RELATIVE),
        "run",
        "--source-root", "{source_root}",
        "--study-commit", commit,
        "--job-dir", "{job_dir}",
        "--job-id", JOB_ID,
        "--expected-role", WORKER_ROLE,
    ]
    argv.extend(_hash_argv(implementation))
    return {
        "job_id": JOB_ID,
        "released": True,
        "source_commit": commit,
        "role": WORKER_ROLE,
        "argv": argv,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "publish_log_tail_bytes": PUBLISH_LOG_TAIL_BYTES,
    }


def build_wave(
    *,
    study_commit: str,
    prerequisite_inputs: Mapping[str, tuple[Path, str]],
) -> dict[str, Any]:
    """Return one receipt-gated descriptor without dispatching it."""

    implementation = _local_implementation()
    prerequisites = validate_prerequisites(prerequisite_inputs)
    descriptor = _job_descriptor(study_commit=study_commit, implementation=implementation)
    return {
        "schema_version": WAVE_SCHEMA,
        "namespace": NAMESPACE,
        "study_id": STUDY_ID,
        "source_commit": descriptor["source_commit"],
        "status": "descriptor_only_not_dispatched_all_five_receipt_gates_passed",
        "jobs": [descriptor],
        "implementation": implementation,
        "prerequisite_receipts": prerequisites,
        "expected_outputs": {
            "job_receipt": SUCCESS_RECEIPT,
            "N3": MODEL_CONTRACT_NAMES["N3"],
            "D1": MODEL_CONTRACT_NAMES["D1"],
        },
        "science_counts": zero_science_counts(),
        "simulator_state_render_used": False,
        "whole_frame_identity": False,
        "safe_to_release_confirmation": False,
        "confirmation_released": False,
        "claim_boundary": (
            "One CPU-only retained-pixel replay job. Descriptor creation and job success "
            "issue no model request, simulator action, behavior, label, or confirmation release."
        ),
    }


def _runtime_implementation(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    rows = args.implementation or []
    result: dict[str, dict[str, str]] = {}
    for name, digest in rows:
        require(name not in result, f"duplicate implementation argument: {name}")
        queue._verified_sha(digest, f"{name} implementation digest")
        result[name] = {"sha256": digest}
    require(set(result) == set(_source_paths(REPOSITORY_ROOT)),
            "runtime implementation arguments changed")
    return result


def _runtime_descriptor(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    implementation = _runtime_implementation(args)
    require(args.job_id == JOB_ID and args.expected_role == WORKER_ROLE,
            "runtime queue identity changed")
    return _job_descriptor(
        study_commit=args.study_commit, implementation=implementation
    ), implementation


def _validate_staged_implementation(
    source_root: Path, expected: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    paths = _source_paths(source_root)
    require(set(paths) == set(expected), "staged implementation inventory changed")
    result = {}
    for name, path in paths.items():
        identity = replay.file_identity(path, label=f"staged {name}", within=source_root)
        require(identity["sha256"] == expected[name]["sha256"],
                f"staged {name} hash changed")
        result[name] = identity
    _runtime_contract(source_root)
    return result


def _validate_cluster_prerequisites(runtime: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    expected = runtime["prerequisite_receipts"]
    require(set(expected) == set(PREREQUISITE_CLUSTER_PATHS),
            "cluster prerequisite inventory changed")
    result = {}
    for name, path in PREREQUISITE_CLUSTER_PATHS.items():
        wanted = expected[name]
        identity = replay.file_identity(
            path,
            label=f"PVC prerequisite {name}",
            expected_path=path,
            expected_sha256=wanted["sha256"],
            expected_bytes=wanted["bytes"],
        )
        value = replay.load_json(path, f"PVC prerequisite {name}")
        _validate_prerequisite_value(name, value, wanted)
        result[name] = identity
    return result


def _child_command(
    *, model: str, context: Any, implementation: Mapping[str, Mapping[str, Any]], raw: Path
) -> tuple[list[str], Path, Path, Path]:
    runtime = replay.load_json(
        context.source_root / CONTRACT_RELATIVE, "camera witness runtime contract"
    )
    python_key = "robolab_python" if model == "N3" else "d1_python"
    executable = Path(runtime["paths"][python_key])
    replay.file_identity(executable.resolve(), label=f"{model} Python executable")
    model_root = raw / model
    contract_path = raw / MODEL_CONTRACT_NAMES[model]
    log_path = raw / f"{model.lower()}_replay.log"
    command = [
        str(executable),
        str(context.source_root / REPLAY_RELATIVE),
        "witness",
        "--model", model,
        "--runtime-contract", str(context.source_root / CONTRACT_RELATIVE),
        "--output-root", str(model_root),
        "--output-contract", str(contract_path),
    ]
    require(implementation["replay"]["sha256"]
            == replay.sha256_file(context.source_root / REPLAY_RELATIVE),
            "child replay implementation changed")
    return command, model_root, contract_path, log_path


def _launch_children(
    *, context: Any, implementation: Mapping[str, Mapping[str, Any]], raw: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    processes: dict[str, tuple[subprocess.Popen[bytes], Any, Path, Path]] = {}
    launches: dict[str, Any] = {}
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    for model in ("N3", "D1"):
        command, model_root, contract_path, log_path = _child_command(
            model=model, context=context, implementation=implementation, raw=raw
        )
        log = log_path.open("xb")
        try:
            process = subprocess.Popen(
                command,
                cwd=context.source_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            log.close()
            raise
        processes[model] = (process, log, contract_path, log_path)
        launches[model] = {
            "pid": process.pid,
            "process_group_id": process.pid,
            "start_new_session": True,
            "cuda_visible_devices": "",
            "contract_path": str(contract_path),
            "log_path": str(log_path),
        }
    queue.immutable_json(raw / "child_launches.json", {
        "schema_version": "wmf-camera-crop-witness-child-launches-v1",
        "job_id": JOB_ID,
        "children": launches,
        "science_counts": zero_science_counts(),
        "safe_to_release_confirmation": False,
    })
    deadline = time.monotonic() + MAX_WALL_SECONDS - 60
    exits: dict[str, int] = {}
    try:
        for model, (process, log, _, _) in processes.items():
            remaining = max(0.0, deadline - time.monotonic())
            try:
                exits[model] = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                exits[model] = 124
            finally:
                log.close()
        if any(code == 124 for code in exits.values()):
            for process, _, _, _ in processes.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for process, _, _, _ in processes.values():
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
    finally:
        for _, log, _, _ in processes.values():
            if not log.closed:
                log.close()
    require(exits == {"N3": 0, "D1": 0}, f"camera replay child exits changed: {exits}")
    contracts = {}
    for model, (_, _, contract_path, log_path) in processes.items():
        value = replay.load_json(contract_path, f"{model} camera crop contract")
        replay.validate_camera_crop_contract(value, model)
        contracts[model] = {
            "value": value,
            "identity": replay.file_identity(contract_path, label=f"raw {model} contract"),
            "log": replay.file_identity(log_path, label=f"{model} replay log"),
        }
    return contracts, launches


def _publish_contracts(
    publish: Path, contracts: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    outputs = {}
    created: list[Path] = []
    try:
        for model in ("N3", "D1"):
            raw_identity = contracts[model]["identity"]
            target = publish / MODEL_CONTRACT_NAMES[model]
            created.append(target)
            queue.immutable_bytes(
                target,
                Path(raw_identity["path"]).read_bytes(),
                maximum_bytes=2 * 1024 * 1024,
            )
            identity = replay.file_identity(target, label=f"published {model} camera contract")
            require(identity["sha256"] == raw_identity["sha256"]
                    and identity["bytes"] == raw_identity["bytes"],
                    f"published {model} contract differs from raw contract")
            value = contracts[model]["value"]
            outputs[model] = {
                **identity,
                "schema_version": replay.SCHEMA,
                "model_id": model,
                "camera_crop_id": value["camera_crop_id"],
                "payload_sha256": value["payload_sha256"],
            }
    except BaseException:
        _cleanup_partial_publication(publish, expected=created)
        raise
    return outputs


def _cleanup_partial_publication(
    publish: Path, *, expected: Sequence[Path] | None = None
) -> bool:
    """Remove only this job's derived contract copies before a terminal receipt.

    Raw witnesses are never touched.  Unknown entries or a terminal success
    receipt make cleanup fail closed instead of broadening the deletion scope.
    """

    directory = Path(publish)
    require(directory.is_dir() and not directory.is_symlink(),
            "partial publication directory changed")
    success = directory / SUCCESS_RECEIPT
    if success.exists() or success.is_symlink():
        return False
    allowed = {directory / name for name in MODEL_CONTRACT_NAMES.values()}
    entries = set(directory.iterdir())
    if expected is not None:
        expected_set = set(expected)
        require(expected_set.issubset(allowed), "partial publication cleanup target changed")
        require(entries.issubset(expected_set), "unexpected partial publication artifact")
    else:
        require(entries.issubset(allowed), "unexpected partial publication artifact")
    for path in sorted(entries):
        require(path in allowed and path.is_file() and not path.is_symlink(),
                "partial publication artifact is not a regular derived contract")
        path.unlink()
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _write_failure(job_dir: Path, context: Any | None, error: BaseException) -> None:
    expected = CONTROL_ROOT / "jobs" / JOB_ID
    try:
        if Path(job_dir) != expected or Path(job_dir).resolve() != expected:
            return
        publish = expected / "publish"
        if not publish.exists():
            publish.mkdir()
        require(publish.is_dir() and not publish.is_symlink(), "failure publish path changed")
        if (publish / SUCCESS_RECEIPT).exists() or (publish / SUCCESS_RECEIPT).is_symlink():
            return
        existing_failure = publish / FAILURE_RECEIPT
        if existing_failure.is_file() and not existing_failure.is_symlink():
            return
        _cleanup_partial_publication(publish)
        require(not any(publish.iterdir()), "failure publication inventory is not empty")
        value = queue.signed_document({
            "schema_version": FAILURE_SCHEMA,
            "namespace": NAMESPACE,
            "study_id": STUDY_ID,
            "status": "technical_invalid",
            "decision": "no_go",
            "job_id": JOB_ID,
            "study_commit": context.study_commit if context is not None else None,
            "queue_role": context.role if context is not None else WORKER_ROLE,
            "failure": {
                "error_type": type(error).__name__,
                "detail": str(error)[:2000],
                "traceback": traceback.format_exc(limit=20)[-12000:],
            },
            "science_counts": zero_science_counts(),
            "simulator_state_render_used": False,
            "whole_frame_identity": False,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "claim_boundary": (
                "Technical-invalid CPU replay. No model, simulator, request, action, behavior, "
                "label, or confirmation job was started; no crop contract is published."
            ),
            "completed_at_utc": queue.utc_now(),
        })
        queue.immutable_json(publish / FAILURE_RECEIPT, value)
    except BaseException:
        return


def run_job(args: argparse.Namespace) -> dict[str, Any]:
    descriptor, expected_implementation = _runtime_descriptor(args)
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
        implementation = _validate_staged_implementation(
            context.source_root, expected_implementation
        )
        runtime = _runtime_contract(context.source_root)
        prerequisites = _validate_cluster_prerequisites(runtime)
        raw, publish = queue._prepare_output_directories(context.job_dir)
        require(context.job_dir / RAW_SUBDIR == raw / "camera_crop_witness",
                "camera witness raw path changed")
        witness_raw = raw / "camera_crop_witness"
        witness_raw.mkdir()
        contracts, launches = _launch_children(
            context=context, implementation=implementation, raw=witness_raw
        )
        outputs = _publish_contracts(publish, contracts)
        receipt = queue.signed_document({
            "schema_version": JOB_RECEIPT_SCHEMA,
            "namespace": NAMESPACE,
            "study_id": STUDY_ID,
            "status": "passed",
            "decision": "qualified_from_original_camera_pixels",
            "job_id": context.job_id,
            "study_commit": context.study_commit,
            "queue_role": context.role,
            "queue_descriptor": context.descriptor_identity,
            "queue_claim": context.claim_identity,
            "worker": {
                "worker_id": context.worker_id,
                "hostname": context.hostname,
                "pod_uid": context.pod_uid,
            },
            "implementation": implementation,
            "prerequisite_receipts": prerequisites,
            "children": launches,
            "outputs": outputs,
            "science_counts": zero_science_counts(),
            "simulator_state_render_used": False,
            "whole_frame_identity": False,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "behavioral_policy_skill_evaluated": False,
            "claim_boundary": (
                "Two CPU-only retained-pixel replays. Passing qualifies only the signed "
                "selected-camera crops; it is not timing, prediction accuracy, policy skill, "
                "label evidence, or authority to release confirmation."
            ),
            "completed_at_utc": queue.utc_now(),
        })
        queue.immutable_json(publish / SUCCESS_RECEIPT, receipt)
        return receipt
    except BaseException as error:
        _write_failure(Path(args.job_dir), context, error)
        raise


def _add_prerequisite_arguments(parser: argparse.ArgumentParser) -> None:
    for name in sorted(PREREQUISITE_CLUSTER_PATHS):
        flag = name.replace("_", "-")
        parser.add_argument(f"--{flag}", type=Path, required=True)
        parser.add_argument(f"--{flag}-sha256", required=True)


def _builder_inputs(args: argparse.Namespace) -> dict[str, tuple[Path, str]]:
    return {
        name: (
            getattr(args, name),
            getattr(args, name + "_sha256"),
        )
        for name in PREREQUISITE_CLUSTER_PATHS
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-wave", help="emit one gated descriptor without dispatch")
    build.add_argument("--study-commit", required=True)
    build.add_argument("--output", type=Path)
    _add_prerequisite_arguments(build)
    run = sub.add_parser("run", help="run inside the claimed detached queue job")
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--study-commit", required=True)
    run.add_argument("--job-dir", type=Path, required=True)
    run.add_argument("--job-id", required=True)
    run.add_argument("--expected-role", required=True)
    run.add_argument("--implementation", nargs=2, action="append", metavar=("NAME", "SHA256"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        run_job(args)
        return 0
    result = build_wave(
        study_commit=args.study_commit, prerequisite_inputs=_builder_inputs(args)
    )
    payload = queue.canonical_bytes(result)
    if args.output is None:
        sys.stdout.buffer.write(payload)
    else:
        target = Path(args.output)
        require(not target.exists() and not target.is_symlink(), "refusing to replace wave output")
        queue.immutable_bytes(target, payload, maximum_bytes=2 * 1024 * 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
