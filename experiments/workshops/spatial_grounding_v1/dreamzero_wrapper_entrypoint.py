"""Entry point for the owned SGW D1 HTTP evidence wrapper.

The native DreamZero server factory is intentionally explicit because its
websocket/model constructor is not part of this repository.  The default
factory verifies the pinned source/checkpoint identity and fails closed until
the reviewed native binding is supplied.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

from .dreamzero_backend import (
    _find_action_head,
    _observe_action_head,
    build_pinned_dreamzero_backend,
    configure_official_startup,
)
from .dreamzero_producer import DreamZeroEvidenceProducer, make_dreamzero_http_server
from .dreamzero_rank_lifecycle import OwnedD1RankLifecycle


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for the SGW D1 wrapper")
    return value


def _proc_start_time(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split(") ", 1)[-1].split()[19]
    except (OSError, IndexError):
        return str(pid)


def _write_ready(payload: dict[str, object]) -> None:
    path = Path(_required("SGW01_D1_RANK_READY_DIR")) / f"rank-{os.environ['RANK']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"stale D1 rank readiness file exists: {path}") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def run_native_rank_worker() -> None:
    """Run the exact exported non-zero-rank conditional worker loop."""
    if os.environ.get("RANK") == "0":
        raise RuntimeError("rank worker entrypoint cannot run as rank zero")
    configure_official_startup()
    source_root = Path(_required("SGW01_D1_SERVER_SOURCE_ROOT")).resolve()
    model_path = _required("SGW01_D1_MODEL_PATH")
    sys.path.insert(0, str(source_root))
    module = importlib.import_module("socket_test_optimized_AR")
    origin = Path(str(getattr(module, "__file__", ""))).resolve()
    try:
        origin.relative_to(source_root)
    except ValueError as exc:
        raise RuntimeError("D1 rank worker imported AR source outside pinned checkout") from exc
    device_mesh = module.init_mesh()
    signal_group = module.dist.new_group(
        backend="gloo",
        timeout=datetime.timedelta(seconds=int(os.environ.get("SGW01_D1_COLLECTIVE_TIMEOUT", "50000"))),
    )
    policy = module.GrootSimPolicy(
        embodiment_tag=module.EmbodimentTag("oxe_droid"),
        model_path=model_path,
        device="cuda" if module.torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )
    head = _observe_action_head(_find_action_head(policy))
    native_config = {
        "num_inference_steps": head["num_inference_steps"],
        "seed": head["seed"],
        "cfg_scale": head["cfg_scale"],
        "action_output_dim": 8,
        "checkpoint_padded_action_dim": 32,
    }
    server = module.WebsocketPolicyServer(
        policy=policy,
        host="127.0.0.1",
        port=int(os.environ.get("SGW01_D1_PORT", "0")),
        metadata={"model_name": "dreamzero", "model_path": model_path},
        output_dir=None,
        signal_group=signal_group,
    )
    _write_ready({
        "rank": int(os.environ["RANK"]),
        "pid": os.getpid(),
        "start_time": _proc_start_time(os.getpid()),
        "run_nonce": os.environ["SGW01_D1_RUN_NONCE"],
        "source_commit": os.environ["SGW01_D1_SOURCE_COMMIT"],
        "checkpoint_revision": os.environ["SGW01_D1_CHECKPOINT_REVISION"],
        "native_config": native_config,
        "native_config_sha256": hashlib.sha256(
            (json.dumps(native_config, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
    })
    asyncio.run(server._worker_loop())


def main() -> None:
    if "--rank-worker" in sys.argv:
        run_native_rank_worker()
        return
    world_size = int(os.environ.get("SGW01_D1_WORLD_SIZE", "1"))
    worker_spec = os.environ.get("SGW01_D1_RANK_WORKER_ARGV", "").strip()
    if world_size > 1 and not worker_spec:
        worker_spec = json.dumps([
            sys.executable,
            "-m",
            "experiments.workshops.spatial_grounding_v1.dreamzero_wrapper_entrypoint",
            "--rank-worker",
        ])
    lifecycle = None
    if worker_spec:
        try:
            worker_argv = json.loads(worker_spec)
        except json.JSONDecodeError as exc:
            raise RuntimeError("SGW01_D1_RANK_WORKER_ARGV must be JSON argv") from exc
        lifecycle = OwnedD1RankLifecycle(
            worker_argv=worker_argv,
            world_size=world_size,
            log_dir=Path(_required("SGW01_D1_RANK_LOG_DIR")),
            ready_dir=Path(_required("SGW01_D1_RANK_READY_DIR")),
            source_commit=os.environ.get("SGW01_D1_SOURCE_COMMIT", ""),
            checkpoint_revision=os.environ.get("SGW01_D1_CHECKPOINT_REVISION", ""),
            master_port=int(os.environ.get("SGW01_D1_MASTER_PORT", "29591")),
        )
        lifecycle.configure_rank_zero()
        lifecycle.install_signal_cleanup()
    host = _required("SGW01_D1_HOST")
    port = int(_required("SGW01_D1_PORT"))
    if lifecycle is None:
        backend = build_pinned_dreamzero_backend()
        producer = DreamZeroEvidenceProducer(
            backend,
            trace_path=Path(_required("SGW01_TRACE_SIDECAR")),
            future_dir=Path(_required("SGW01_FUTURE_DIR")),
            attestation_path=Path(_required("SGW01_SERVER_ATTESTATION")),
        )
        make_dreamzero_http_server(
            producer,
            host=host,
            port=port,
        ).serve_forever()
        return
    with lifecycle:
        backend = build_pinned_dreamzero_backend()
        lifecycle.await_ready()
        producer = DreamZeroEvidenceProducer(
            backend,
            trace_path=Path(_required("SGW01_TRACE_SIDECAR")),
            future_dir=Path(_required("SGW01_FUTURE_DIR")),
            attestation_path=Path(_required("SGW01_SERVER_ATTESTATION")),
        )
        make_dreamzero_http_server(
            producer,
            host=host,
            port=port,
            healthcheck=lifecycle.assert_healthy,
        ).serve_forever()


if __name__ == "__main__":
    main()
