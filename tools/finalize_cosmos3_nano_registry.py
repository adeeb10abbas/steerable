#!/usr/bin/env python3
"""Finalize the V2-A011 Nano checkpoint registry without loading the model."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    checkpoint = args.checkpoint.expanduser().resolve()
    if registry["checkpoint"]["revision"] != REVISION:
        raise RuntimeError("Registry revision is not the frozen V2-A011 revision")
    incomplete = sorted((checkpoint / ".cache").rglob("*.incomplete"))
    nonempty_incomplete = [path for path in incomplete if path.stat().st_size > 0]
    if nonempty_incomplete:
        raise RuntimeError(f"Checkpoint still has partial payloads: {nonempty_incomplete}")
    index_path = checkpoint / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text()).get("weight_map", {})
    indexed_weights = sorted(set(weight_map.values()))
    if indexed_weights != registry["checkpoint"]["required_indexed_weight_files"]:
        raise RuntimeError("Resolved weight index differs from the frozen registry")
    for relative in indexed_weights:
        path = checkpoint / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required indexed weight is absent: {path}")
    for relative in registry["checkpoint"]["required_component_files"]:
        path = checkpoint / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required model component is absent: {path}")
    files = sorted(
        path for path in checkpoint.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(checkpoint).parts
    )
    if not files:
        raise RuntimeError("Checkpoint has no payload files")
    revision_metadata = checkpoint / ".cache/huggingface/download/config.json.metadata"
    resolved_revision = revision_metadata.read_text().splitlines()[0]
    if resolved_revision != REVISION:
        raise RuntimeError(f"Hugging Face metadata revision mismatch: {resolved_revision}")
    file_records = {
        str(path.relative_to(checkpoint)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in files
    }
    payload_bytes = sum(item["bytes"] for item in file_records.values())
    if payload_bytes < 20_000_000_000:
        raise RuntimeError(f"Nano payload is implausibly small: {payload_bytes} bytes")
    registry["status"] = "checkpoint_hash_verified_fixed_observation_gate_pending"
    registry["checkpoint"].update(
        download_status="complete_exact_revision_and_hashed",
        present_non_cache_file_count=len(file_records),
        present_non_cache_bytes=payload_bytes,
        files=file_records,
        hash_gate_passed=True,
        hash_completed_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    registry["fixed_observation_gate"]["status"] = "released_pending_three_request_gate"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"files": len(file_records), "bytes": payload_bytes, "output": str(args.output)}))


if __name__ == "__main__":
    main()
