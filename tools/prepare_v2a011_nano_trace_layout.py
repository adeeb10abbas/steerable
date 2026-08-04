#!/usr/bin/env python3
"""Reproduce the V2-A011 zero-copy action-trace layout compatibility bridge.

This helper is intentionally Nano-specific.  It validates the six canonical
V2-A011 trace metadata records, then exposes only same-file symlinks under the
frozen shared compiler's ``simulator_attempt001/actions`` layout.  It never
loads a model, starts a policy request, changes a rollout, or copies action or
future bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MODEL_ID = "cosmos3_nano_policy_droid"
REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
SEEDS = (8300, 8301, 8302)
DIRECTIONS = ("left", "right")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_metadata(path: Path, seed: int, direction: str) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if (
        data.get("model_id") != MODEL_ID
        or data.get("checkpoint_revision") != REVISION
        or data.get("amendment_id") != "V2-A011"
        or data.get("sampling_seed_base") != seed
        or len(data.get("requests", [])) < 1
    ):
        raise RuntimeError(f"V2-A011 trace identity mismatch: {path}")
    executed = Path(data["executed_actions"]["path"])
    if not executed.is_file() or sha256(executed) != data["executed_actions"]["sha256"]:
        raise RuntimeError(f"V2-A011 executed-action hash mismatch: {path}")
    for request in data["requests"]:
        for field, hash_field in (("action_path", "action_sha256"), ("future_path", "future_sha256")):
            payload = Path(request[field])
            if not payload.is_file() or sha256(payload) != request[hash_field]:
                raise RuntimeError(f"V2-A011 retained request hash mismatch: {path}:{request.get('request_index')}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--event-output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for seed in SEEDS:
        for direction in DIRECTIONS:
            source = args.raw_root / f"seed{seed}/actions/seed{seed}_{direction}_executed_actions.json"
            if not source.is_file():
                raise RuntimeError(f"missing canonical V2-A011 trace metadata: {source}")
            verify_metadata(source, seed, direction)
            target = args.raw_root / f"seed{seed}/simulator_attempt001/actions/{source.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                if not target.is_symlink() or target.resolve() != source.resolve():
                    raise RuntimeError(f"refusing non-identical existing compatibility target: {target}")
            else:
                os.symlink(source, target)
            records.append({
                "seed": seed,
                "direction": direction,
                "source": file_record(source),
                "compatibility_symlink": str(target),
                "target_resolves_to_source": target.resolve() == source.resolve(),
                "copied_bytes": 0,
            })

    payload = {
        "schema_version": "vla-wam-shared-v2-cosmos3-nano-policy-droid-trace-layout-helper-v1",
        "status": "passed_zero_copy_measurement_only",
        "model_id": MODEL_ID,
        "checkpoint_revision": REVISION,
        "amendment_id": "V2-A011",
        "records": records,
        "behavioral_data_modified": False,
        "episodes_rerun": 0,
        "model_requests_started_by_resolution": 0,
        "total_copied_bytes": 0,
        "claim_boundary": "This helper only exposes verified canonical metadata through a zero-copy legacy path for measurement compilation. It does not change behavioral evidence.",
    }
    args.event_output.parent.mkdir(parents=True, exist_ok=True)
    args.event_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "record_count": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
