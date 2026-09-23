"""Entry point for the owned SGW D1 HTTP evidence wrapper.

The native DreamZero server factory is intentionally explicit because its
websocket/model constructor is not part of this repository.  The default
factory verifies the pinned source/checkpoint identity and fails closed until
the reviewed native binding is supplied.
"""

from __future__ import annotations

import os
from pathlib import Path

from .dreamzero_backend import build_pinned_dreamzero_backend
from .dreamzero_producer import DreamZeroEvidenceProducer, make_dreamzero_http_server


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for the SGW D1 wrapper")
    return value


def main() -> None:
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
