from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .io import sha256_file


def project_root() -> Path:
    """Return the source checkout root for this package."""

    return Path(__file__).resolve().parents[2]


def implementation_manifest(root: Path | None = None) -> dict[str, Any]:
    """Fingerprint the executable RES-1 implementation and environment lock."""

    checkout = (root or project_root()).resolve()
    candidates = sorted((checkout / "src" / "steerable_bridge").glob("*.py"))
    candidates.extend(
        path for path in (checkout / "pyproject.toml", checkout / "uv.lock") if path.exists()
    )
    records = [
        {
            "path": str(path.relative_to(checkout)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(candidates)
    ]
    canonical = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "combined_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": records,
    }
