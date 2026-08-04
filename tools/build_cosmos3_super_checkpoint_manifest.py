#!/usr/bin/env python3
"""Freeze Hugging Face metadata for Cosmos3-Super without downloading weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


MODEL_ID = "nvidia/Cosmos3-Super"
REVISION = "e0262be9d8f7586bc24c069a2aed2b665bdff266"
INDEX_TOTAL_BYTES = 129_230_007_264


def fetch_json(url: str) -> tuple[Any, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "steerable-v2a012-freeze"})
    with urllib.request.urlopen(request, timeout=30) as response:
        headers = {key.lower(): value for key, value in response.headers.items()}
        return json.load(response), headers


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "steerable-v2a012-freeze"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def tree_rows() -> list[dict[str, Any]]:
    url = (
        f"https://huggingface.co/api/models/{MODEL_ID}/tree/{REVISION}"
        "?recursive=true&expand=true"
    )
    rows: list[dict[str, Any]] = []
    while url:
        page, headers = fetch_json(url)
        if not isinstance(page, list):
            raise RuntimeError("Hugging Face tree endpoint did not return a list")
        rows.extend(page)
        match = re.search(r"<([^>]+)>; rel=\"next\"", headers.get("link", ""))
        url = match.group(1) if match else ""
    return rows


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_info, _ = fetch_json(
        f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}"
    )
    if model_info.get("sha") != REVISION:
        raise RuntimeError("Resolved Hugging Face revision does not match V2-A012")
    if model_info.get("gated") is not False or model_info.get("private") is not False:
        raise RuntimeError("V2-A012 requires the public, ungated Cosmos3-Super snapshot")

    files = [row for row in tree_rows() if row.get("type") == "file"]
    if len(files) != 88:
        raise RuntimeError(f"Unexpected snapshot file count: {len(files)}")
    paths = [row["path"] for row in files]
    if len(paths) != len(set(paths)):
        raise RuntimeError("Snapshot contains duplicate file paths")

    records: dict[str, dict[str, Any]] = {}
    for row in sorted(files, key=lambda item: item["path"]):
        lfs = row.get("lfs")
        records[row["path"]] = {
            "bytes": row["size"],
            "git_blob_oid": row["oid"],
            "lfs_sha256": lfs["oid"] if lfs else None,
        }

    metadata_paths = (
        "config.json",
        "model_index.json",
        "model.safetensors.index.json",
        "transformer/config.json",
    )
    metadata_sha256: dict[str, str] = {}
    for relative in metadata_paths:
        payload = fetch_bytes(
            f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{relative}?download=true"
        )
        metadata_sha256[relative] = sha256_bytes(payload)

    index_payload = fetch_bytes(
        f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/model.safetensors.index.json?download=true"
    )
    weight_index = json.loads(index_payload)
    if weight_index.get("metadata", {}).get("total_size") != INDEX_TOTAL_BYTES:
        raise RuntimeError("Cosmos3-Super weight index total differs from V2-A012")
    indexed_weights = sorted(set(weight_index.get("weight_map", {}).values()))
    if len(indexed_weights) != 28 or any(path not in records for path in indexed_weights):
        raise RuntimeError("Unexpected Cosmos3-Super indexed-weight layout")
    if any(records[path]["lfs_sha256"] is None for path in indexed_weights):
        raise RuntimeError("Indexed weight is not an LFS payload")

    total_bytes = sum(record["bytes"] for record in records.values())
    lfs_bytes = sum(record["bytes"] for record in records.values() if record["lfs_sha256"])
    manifest = {
        "schema_version": "vla-wam-shared-v2-cosmos3-super-hf-snapshot-v1",
        "amendment_id": "V2-A012",
        "model_repository": MODEL_ID,
        "model_revision": REVISION,
        "resolved_public": True,
        "resolved_gated": False,
        "model_index_total_bytes": INDEX_TOTAL_BYTES,
        "snapshot_file_count": len(records),
        "snapshot_total_bytes": total_bytes,
        "snapshot_lfs_payload_bytes": lfs_bytes,
        "metadata_sha256": metadata_sha256,
        "indexed_weight_files": indexed_weights,
        "files": records,
        "claim_boundary": "This is remote metadata only. It performs no checkpoint download, model load, or inference.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "files": len(records),
                "snapshot_total_bytes": total_bytes,
                "model_index_total_bytes": INDEX_TOTAL_BYTES,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
