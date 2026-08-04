#!/usr/bin/env python3
"""Verify the frozen Cosmos3-Super checkpoint payload before any model load."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


MODEL_ID = "nvidia/Cosmos3-Super"
REVISION = "e0262be9d8f7586bc24c069a2aed2b665bdff266"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text())
    snapshot = json.loads(args.source_snapshot.read_text())
    checkpoint = args.checkpoint.expanduser().resolve()
    if registry["amendment_id"] != "V2-A012":
        raise RuntimeError("Registry is not V2-A012")
    if registry["checkpoint"]["id"] != MODEL_ID or registry["checkpoint"]["revision"] != REVISION:
        raise RuntimeError("Registry model identity differs from V2-A012")
    if snapshot["model_repository"] != MODEL_ID or snapshot["model_revision"] != REVISION:
        raise RuntimeError("Source snapshot identity differs from V2-A012")
    if sha256(args.source_snapshot) != registry["checkpoint"]["source_snapshot_manifest_sha256"]:
        raise RuntimeError("Source snapshot hash differs from the frozen registry")

    incomplete = sorted((checkpoint / ".cache").rglob("*.incomplete"))
    nonempty_incomplete = [path for path in incomplete if path.stat().st_size > 0]
    if nonempty_incomplete:
        raise RuntimeError(f"Checkpoint still has partial payloads: {nonempty_incomplete}")
    metadata_path = checkpoint / ".cache/huggingface/download/config.json.metadata"
    if not metadata_path.is_file() or metadata_path.read_text().splitlines()[0] != REVISION:
        raise RuntimeError("Hugging Face resolved revision metadata is absent or incorrect")

    local_files = {
        str(path.relative_to(checkpoint)): path
        for path in checkpoint.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(checkpoint).parts
    }
    expected_files = snapshot["files"]
    if set(local_files) != set(expected_files):
        missing = sorted(set(expected_files) - set(local_files))
        unexpected = sorted(set(local_files) - set(expected_files))
        raise RuntimeError(f"Checkpoint file layout differs; missing={missing}, unexpected={unexpected}")

    local_records = {}
    for relative, expected in sorted(expected_files.items()):
        path = local_files[relative]
        if path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"Byte count differs for {relative}")
        digest = sha256(path)
        if expected["lfs_sha256"] and digest != expected["lfs_sha256"]:
            raise RuntimeError(f"LFS SHA-256 differs for {relative}")
        local_records[relative] = {"bytes": path.stat().st_size, "sha256": digest}

    for relative, expected_digest in snapshot["metadata_sha256"].items():
        if local_records[relative]["sha256"] != expected_digest:
            raise RuntimeError(f"Frozen metadata SHA-256 differs for {relative}")
    payload_bytes = sum(record["bytes"] for record in local_records.values())
    if payload_bytes != snapshot["snapshot_total_bytes"]:
        raise RuntimeError("Checkpoint total bytes differ from source snapshot")

    registry["checkpoint"].update(
        download_status="complete_exact_revision_and_hashed",
        hash_gate_passed=True,
        hash_completed_at_utc=datetime.now(timezone.utc).isoformat(),
        present_non_cache_file_count=len(local_records),
        present_non_cache_bytes=payload_bytes,
        local_files=local_records,
    )
    registry["status"] = "checkpoint_hash_verified_other_pre_inference_gates_blocked"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"files": len(local_records), "bytes": payload_bytes, "output": str(args.output)}))


if __name__ == "__main__":
    main()
