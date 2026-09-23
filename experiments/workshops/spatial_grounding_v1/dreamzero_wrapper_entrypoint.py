"""Entry point for the owned SGW D1 HTTP evidence wrapper.

The native DreamZero server factory is intentionally explicit because its
websocket/model constructor is not part of this repository.  The default
factory verifies the pinned source/checkpoint identity and fails closed until
the reviewed native binding is supplied.
"""

from __future__ import annotations

import os
import json
from pathlib import Path

from .dreamzero_backend import build_pinned_dreamzero_backend
from .dreamzero_producer import DreamZeroEvidenceProducer, make_dreamzero_http_server
from .dreamzero_rank_lifecycle import OwnedD1RankLifecycle


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for the SGW D1 wrapper")
    return value


def main() -> None:
    world_size = int(os.environ.get("SGW01_D1_WORLD_SIZE", "1"))
    worker_spec = os.environ.get("SGW01_D1_RANK_WORKER_ARGV", "").strip()
    if world_size > 1 and not worker_spec:
        raise RuntimeError("SGW01_D1_RANK_WORKER_ARGV is required for distributed D1")
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
            host=_required("SGW01_D1_HOST"),
            port=int(_required("SGW01_D1_PORT")),
        ).serve_forever()
        return
    with lifecycle:
        backend = build_pinned_dreamzero_backend()
        producer = DreamZeroEvidenceProducer(
            backend,
            trace_path=Path(_required("SGW01_TRACE_SIDECAR")),
            future_dir=Path(_required("SGW01_FUTURE_DIR")),
            attestation_path=Path(_required("SGW01_SERVER_ATTESTATION")),
        )
        make_dreamzero_http_server(
            producer,
            host=_required("SGW01_D1_HOST"),
            port=int(_required("SGW01_D1_PORT")),
        ).serve_forever()


if __name__ == "__main__":
    main()
