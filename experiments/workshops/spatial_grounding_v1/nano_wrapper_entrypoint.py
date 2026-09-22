"""Entry point for an owned SGW Nano HTTP server.

The backend factory is deliberately explicit.  It must load the pinned Cosmos
service and expose ``resolved_config``, ``source_root``, ``checkpoint_path``,
and ``predict(observation, prompt, sampling_seed)``.  This module never
silently falls back to a fake or OpenPI websocket backend.
"""

from __future__ import annotations

import os
from pathlib import Path

from .producer import serve_nano_wrapper
from .runtime import _load_callable


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for the SGW Nano wrapper")
    return value


def main() -> None:
    factory_spec = os.environ.get(
        "SGW01_NANO_BACKEND_FACTORY",
        "experiments.workshops.spatial_grounding_v1.nano_backend:build_pinned_nano_backend",
    )
    factory = _load_callable(factory_spec, "Nano backend factory")
    backend = factory()
    serve_nano_wrapper(
        backend,
        host=_required("SGW01_NANO_HOST"),
        port=int(_required("SGW01_NANO_PORT")),
        trace_path=Path(_required("SGW01_TRACE_SIDECAR")),
        future_dir=Path(_required("SGW01_FUTURE_DIR")),
        attestation_path=Path(_required("SGW01_SERVER_ATTESTATION")),
    )


if __name__ == "__main__":
    main()
