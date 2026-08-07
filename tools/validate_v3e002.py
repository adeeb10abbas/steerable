#!/usr/bin/env python3
"""Fail-closed registration/output validator for V3-E002."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "artifacts/vla_wam_shared_v3/phase_e/reference_controller_symmetry_v3e002/registration.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    reg = json.loads(REG.read_text(encoding="utf-8"))
    assert reg["schema_version"] == "vla-wam-shared-v3e002-registration-v1"
    assert reg["status"] == "registered_before_inference"
    assert reg["model_blind"] is True
    assert reg["learned_model_request_count"] == 0
    assert len(reg["queue"]) == 108
    assert len({row["cell_id"] for row in reg["queue"]}) == 108
    for item in reg["parent_bindings"]:
        path = ROOT / item["path"]
        assert path.is_file(), path
        assert path.stat().st_size == item["bytes"], path
        assert sha256(path) == item["sha256"], path
    print(json.dumps({"status": "valid", "registration_sha256": sha256(REG), "behavioral_episodes": 108, "learned_model_requests": 0}, indent=2))


if __name__ == "__main__":
    main()
