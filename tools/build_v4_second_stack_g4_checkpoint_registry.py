#!/usr/bin/env python3
"""Materialize a frozen GR00T Bridge checkpoint registry from a passing G4 receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_exclusive(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def build_registry(g4: dict[str, Any], *, g4_path: Path) -> dict[str, Any]:
    if g4.get("fixture_id") != "second_stack":
        raise ValueError("G4 receipt fixture mismatch")
    if g4.get("passed") is not True or g4.get("status") != "passed":
        raise ValueError("G4 receipt is not passing")
    manifest = g4.get("checkpoint_content_manifest")
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("G4 receipt lacks checkpoint_content_manifest")
    revision = g4.get("checkpoint_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("G4 receipt lacks checkpoint_revision")
    return {
        "schema_version": "v4-groot-bridge-checkpoint-registry-v1",
        "campaign_id": "online_correction_v4",
        "policy_id": "groot_bridge_widowx",
        "fixture_id": "second_stack",
        "checkpoint_revision": revision,
        "repository": "nvidia/GR00T-N1.7-SimplerEnv-Bridge",
        "files": manifest,
        "qualification_basis": {
            "g4_policy_session": {
                "path": str(g4_path),
                "bytes": g4_path.stat().st_size,
                "sha256": sha256_file(g4_path),
            }
        },
        "release_boundary": (
            "Frozen checkpoint content manifest extracted from the passing C8 G4 "
            "policy-session receipt. It does not authorize behavioral episodes."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g4", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint registry: {args.out}")
    g4_path = args.g4.resolve()
    registry = build_registry(load_json(g4_path), g4_path=g4_path)
    write_exclusive(args.out.resolve(), canonical_json_bytes(registry))
    print(
        json.dumps(
            {
                "checkpoint_revision": registry["checkpoint_revision"],
                "file_count": len(registry["files"]),
                "registry": {
                    "path": str(args.out),
                    "bytes": args.out.stat().st_size,
                    "sha256": sha256_file(args.out),
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
