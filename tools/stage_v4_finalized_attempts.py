#!/usr/bin/env python3
"""Index finalized V4 attempts into the compiler's two-level layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_manifest_ids(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            rows = payload
        elif isinstance(payload.get("episodes"), list):
            rows = payload["episodes"]
        else:
            queue_value = payload.get("queue_path")
            if not isinstance(queue_value, str) or not queue_value:
                raise ValueError("manifest has neither episodes nor queue_path")
            queue_path = Path(queue_value)
            if not queue_path.is_absolute():
                adjacent = path.parent / queue_path
                queue_path = adjacent if adjacent.is_file() else ROOT / queue_path
            queue_bytes = queue_path.read_bytes()
            expected_sha256 = payload.get("queue_sha256")
            if (
                isinstance(expected_sha256, str)
                and hashlib.sha256(queue_bytes).hexdigest() != expected_sha256
            ):
                raise ValueError("manifest queue hash differs")
            rows = [
                json.loads(line)
                for line in queue_bytes.decode("utf-8").splitlines()
                if line.strip()
            ]
    ids = {str(row["episode_id"]) for row in rows}
    if not ids:
        raise ValueError("manifest has no episode IDs")
    return ids


def stage_attempts(
    *,
    source_roots: list[Path],
    manifest_ids: set[str],
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    candidates: dict[tuple[str, str], Path] = {}
    for source_root in source_roots:
        if not source_root.is_dir():
            raise ValueError(f"source root is not a directory: {source_root}")
        for receipt_path in source_root.rglob("COMPLETE.json"):
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            episode_id = str(receipt.get("episode_id", ""))
            attempt_id = str(receipt.get("attempt_id", ""))
            if episode_id not in manifest_ids:
                continue
            if not attempt_id:
                raise ValueError(f"COMPLETE receipt lacks attempt_id: {receipt_path}")
            key = (episode_id, attempt_id)
            previous = candidates.get(key)
            if previous is not None and previous.resolve() != receipt_path.parent.resolve():
                raise ValueError(
                    f"duplicate finalized attempt {episode_id}/{attempt_id}: "
                    f"{previous} and {receipt_path.parent}"
                )
            candidates[key] = receipt_path.parent
    output_root.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    for (episode_id, attempt_id), source in sorted(candidates.items()):
        episode_dir = output_root / episode_id
        episode_dir.mkdir(exist_ok=True)
        destination = episode_dir / attempt_id
        os.symlink(source.resolve(), destination, target_is_directory=True)
        rows.append(
            {
                "episode_id": episode_id,
                "attempt_id": attempt_id,
                "source": str(source.resolve()),
                "staged_path": str(destination.relative_to(output_root)),
            }
        )
    inventory = {
        "schema_version": "v4-finalized-attempt-index-v1",
        "manifest_episode_count": len(manifest_ids),
        "indexed_episode_count": len({row["episode_id"] for row in rows}),
        "indexed_attempt_count": len(rows),
        "attempts": rows,
    }
    (output_root / "index.json").write_text(
        json.dumps(inventory, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    inventory = stage_attempts(
        source_roots=[path.resolve() for path in args.source_root],
        manifest_ids=load_manifest_ids(args.manifest),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(inventory, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
