"""Execute the six user-directed MAIN P cells with existing native components.

This narrowly scoped execution does not assert a passed pre-run integration or
future-time-map qualification. Raw prediction scores remain unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
import uuid

from .contract import Cell, Release, canonical_bytes, sha256_file
from .family_campaign_executor import _fsync_json
from .main_p_fixture_assignment import SCHEMA
from .paper_engineering import bound_file, record


def wait_json(path: Path, root: Path, timeout: float = 1800):
    deadline = time.monotonic() + timeout
    while not path.is_file():
        failures = list(root.glob("failure-*.json"))
        if failures:
            raise RuntimeError(f"peer failed: {failures[0].read_text()}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.2)
    return json.loads(path.read_bytes())


def load_plan(path: Path):
    plan = json.loads(path.read_bytes())
    direction = json.loads(bound_file(plan["direction"]).read_bytes())
    assignment = json.loads(bound_file(plan["assignment"]).read_bytes())
    completion = json.loads(bound_file(plan["fixed_input_completion"]).read_bytes())
    if (direction["id"] != "SGW-MAIN-N3-LAT-P-20260923DM"
            or direction["maximum_behavioral_episodes"] != 6
            or direction["maximum_policy_requests_this_execution"] != 90
            or direction["prediction_physical_scores_permitted"] is not False
            or assignment["schema_version"] != SCHEMA
            or assignment["assignment_sha256"] != hashlib.sha256(canonical_bytes({
                key: value for key, value in assignment.items() if key != "assignment_sha256"
            })).hexdigest()
            or plan["assignment"]["sha256"] != direction["assignment_file_sha256"]
            or plan["fixed_input_completion"]["sha256"] != direction["fixed_input_completion_sha256"]
            or completion["model_requests"] != 6
            or completion["comparisons"]["repeat_equal"] != [True, True]
            or completion["comparisons"]["opposite_prompt_distinct"] != [True, True]):
        raise ValueError("not the six authorized MAIN P cells with the completed direct gate")
    bound_file(plan["candidate"])
    rows = assignment["main_p_frozen_queue"]["cells"]
    if len(rows) != 6 or len({row["cell_id"] for row in rows}) != 6:
        raise ValueError("MAIN P requires six unique original cells")
    cells = tuple(Cell({
        **row, "model": "N3", "family": "LAT", "stage": "P",
        "release_id": direction["id"], "status": "RELEASED",
        "fixture_sha256": assignment["assignment_sha256"],
        "sampling_seed": int(row["effective_policy_seed"]),
        "prediction_time_mapping_status": "unqualified",
    }) for row in rows)
    return plan, cells


def prepare(plan_path: Path, root: Path):
    source = Path(__file__).resolve().parents[3]
    artifacts = source / "artifacts/workshops/spatial_grounding_v1"
    direction = artifacts / "main_p_execution_20260923dm.json"
    directive = json.loads(direction.read_bytes())
    assignment = source / directive["assignment_path"]
    completion = source / directive["fixed_input_completion_path"]
    engineering = json.loads((artifacts / "infrastructure/paper-engineering-20260923cj/registration.json").read_bytes())
    proposal = Path("/data/users/ali/sgw-01/qualification/paper-engineering-20260923cs/evidence/engineering-proposal.json")
    candidates = json.loads(proposal.read_bytes())["candidates"]
    if len(candidates) != 1 or candidates[0]["candidate_id"] != "SGW-ENG-008-LAT-057":
        raise ValueError("expected the one already qualified redesigned candidate")
    root.mkdir(parents=True, exist_ok=False)
    (root / "claims").mkdir()
    (root / "registration").mkdir()
    candidate = root / "registration/candidate.json"
    _fsync_json(candidate, candidates[0])
    plan = {
        "source_root": str(source),
        "source_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "robolab_root": "/data/users/ali/vla_wam/external/RoboLab-pi05-v3-0aef241",
        "direction": record(direction), "assignment": record(assignment),
        "fixed_input_completion": record(completion),
        "original_proposal": record(proposal), "candidate": record(candidate),
        "assets": engineering["assets"],
        "future_physical_scores_permitted": False,
    }
    _fsync_json(plan_path, plan)
    load_plan(plan_path)
    print(json.dumps({"status": "prepared_six_MAIN_P_cells", "plan": record(plan_path)}))


def simulator(plan_path: Path, root: Path):
    from .robolab_jointpos_environment import create_environment
    from .simulator_mailbox import MailboxReceiver
    from isaaclab.app import AppLauncher

    plan, cells = load_plan(plan_path)
    owner = json.loads((root / "claims/simulator/owner.json").read_bytes())
    binding_path = root / "environment-binding.json"
    binding = {
        "source_root": plan["source_root"], "source_commit": plan["source_commit"],
        "robolab_root": plan["robolab_root"],
        "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
        "assets_manifest": plan["assets"]["path"],
        "assets_manifest_sha256": plan["assets"]["sha256"],
        "cells": {cell.cell_id: {
            "family": "LAT", "layout_id": cell.row["layout_id"],
            "fixture_sha256": cell.row["fixture_sha256"],
            "prompt_sha256": cell.row["prompt_sha256"],
            "candidate_path": plan["candidate"]["path"],
            "candidate_file_sha256": plan["candidate"]["sha256"],
            "scene_seed": int(cell.row["environment_seed"]),
        } for cell in cells},
    }
    _fsync_json(binding_path, binding)
    os.environ["SGW01_ENV_BINDING"] = str(binding_path)
    os.environ["SGW01_ENV_BINDING_SHA256"] = sha256_file(binding_path)
    os.environ["SGW01_SIMULATOR_DEVICE"] = "cuda:0"
    app = AppLauncher({
        "headless": True, "enable_cameras": True, "device": "cuda:0",
        "rendering_mode": "balanced",
        "kit_args": f"--portable-root={root}/kit --/rtx/verifyDriverVersion/enabled=false",
    }).app
    try:
        for cell in cells:
            channel = root / "mailboxes" / cell.cell_id
            channel.mkdir(parents=True, exist_ok=False)
            for name in ("requests", "responses", "faults"):
                (channel / name).mkdir()
            identity = {
                "release_id": cell.row["release_id"], "cell_id": cell.cell_id,
                "attempt_id": "attempt-001", "channel_nonce": uuid.uuid4().hex,
                "candidate_sha256": plan["candidate"]["sha256"],
                "binding_sha256": sha256_file(binding_path),
                "simulator_job_uid": owner["job_uid"],
                "simulator_pod_uid": owner["pod_uid"],
            }
            environment = create_environment(cell=cell, evidence_root=channel / "evidence")
            receiver = MailboxReceiver(root=channel, identity=identity, environment=environment)
            _fsync_json(channel / "ready.json", identity)
            deadline = time.monotonic() + 5400
            try:
                while not receiver.closed:
                    if list(root.glob("failure-*.json")):
                        raise RuntimeError("peer failed; preserving the incomplete episode")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("simulator episode deadline expired")
                    for request in sorted((channel / "requests").glob("*.json")):
                        if int(request.name[:4]) > receiver.last:
                            receiver.serve_one(request)
                    time.sleep(0.01)
            finally:
                receiver.close_environment()
        _fsync_json(root / "simulator-complete.json", {"cells": 6})
    except BaseException:
        _fsync_json(root / "failure-simulator.json", {
            "traceback": traceback.format_exc(), "partial_evidence_preserved": True,
            "automatic_retry": False,
        })
        raise
    finally:
        app.close()


def policy(plan_path: Path, root: Path):
    from .adapters import NANO_CONFIG, NanoPolicyAdapter, ProductionAdapter
    from .nano_backend import build_pinned_nano_backend
    from .producer import NanoEvidenceProducer, make_nano_http_server
    from .recorder import AttemptRecorder
    from .runtime import _NanoHttpTransport
    from .simulator_mailbox import MailboxClient
    from .trace import read_trace_sidecar
    from .worker import _canonical_outcome, _run_with_deadline

    plan, cells = load_plan(plan_path)
    wait_json(root / "claims/simulator/owner.json", root)
    output = root / "policy"
    output.mkdir()
    os.environ["SGW01_TRACE_SIDECAR"] = str(output / "trace.jsonl")
    os.environ["SGW01_CAMERA_NAME"] = "over_shoulder_left_camera"
    backend = build_pinned_nano_backend()
    producer = NanoEvidenceProducer(
        backend, trace_path=output / "trace.jsonl", future_dir=output / "futures",
        attestation_path=output / "attestation.json",
    )
    server = make_nano_http_server(producer, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    transport = _NanoHttpTransport("127.0.0.1", server.server_port, read_trace_sidecar, producer.attestation)

    def environment_factory(*, cell, evidence_root):
        channel = root / "mailboxes" / cell.cell_id
        identity = wait_json(channel / "ready.json", root)
        return MailboxClient(root=channel, identity=identity, timeout_s=180)

    adapter = ProductionAdapter(
        NanoPolicyAdapter, transport=transport, transport_factory=lambda **_: transport,
        environment_factory=environment_factory, runtime_handle=transport,
    )
    release = Release(
        root=root / "registration", release_id=cells[0].row["release_id"],
        hashes={"execution-plan.json": sha256_file(plan_path)},
        cells=cells, binding={"prediction_time_mapping_status": "unqualified"},
    )
    _fsync_json(output / "ready.json", {"config": NANO_CONFIG, "maximum_requests": 90})
    completed = []
    try:
        for cell in cells:
            recorder = AttemptRecorder(release, cell, "attempt-001")
            recorder.begin()
            _fsync_json(root / "progress.json", {
                "state": "executing_main_P", "current_cell": cell.cell_id,
                "completed_cells": completed,
            })
            try:
                reset = adapter.reset(cell, recorder)
                outcome = _run_with_deadline(
                    3600, "MAIN P episode", lambda: adapter.run_episode(cell, recorder, reset),
                )
                outcome = _canonical_outcome({**outcome, "attempt_id": recorder.attempt_id}, cell, None)
                outcome["prediction_time_mapping_status"] = "unqualified"
                outcome["prediction_physical_score"] = None
                recorder.complete(outcome)
                completed.append({"cell_id": cell.cell_id, "status": outcome["status"]})
            except BaseException as error:
                recorder.event("execution_stopped_preserve_partial", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                if adapter.environment is not None:
                    adapter.environment.close()
                    adapter.environment = None
        _fsync_json(root / "policy-complete.json", {
            "completed_cells": completed, "prediction_physical_scores_permitted": False,
            "model_requests": len((output / "trace.jsonl").read_text().splitlines()),
        })
    finally:
        try:
            adapter.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=10)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--role", choices=("prepare", "simulator", "policy"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        {"prepare": prepare, "simulator": simulator, "policy": policy}[args.role](args.plan, args.root)
    except BaseException:
        _fsync_json(args.root / f"failure-{args.role}.json", {
            "role": args.role, "traceback": traceback.format_exc(),
            "partial_evidence_preserved": True, "automatic_retry": False,
        })
        raise


if __name__ == "__main__":
    main()
