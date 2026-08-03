#!/usr/bin/env python3
"""Verify and hash a locally staged immutable Hugging Face model snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve()
    required = {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    }
    missing = sorted(name for name in required if not (snapshot / name).is_file())
    if missing:
        raise FileNotFoundError(f"Snapshot is missing required files: {missing}")

    files = sorted(
        path for path in snapshot.iterdir() if path.is_file() and not path.is_symlink()
    )
    metadata_dir = snapshot / ".cache/huggingface/download"
    revision_evidence: dict[str, str] = {}
    for path in files:
        metadata_path = metadata_dir / f"{path.name}.metadata"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing Hugging Face metadata for {path.name}")
        revision = metadata_path.read_text().splitlines()[0].strip()
        if revision != args.expected_revision:
            raise ValueError(
                f"Revision mismatch for {path.name}: {revision} != {args.expected_revision}"
            )
        revision_evidence[path.name] = revision

    config = json.loads((snapshot / "config.json").read_text())
    if config.get("model_type") != "qwen3_vl":
        raise ValueError(f"Unexpected Cosmos model_type: {config.get('model_type')!r}")
    if config.get("architectures") != ["Qwen3VLForConditionalGeneration"]:
        raise ValueError(f"Unexpected Cosmos architecture: {config.get('architectures')!r}")

    manifest = {
        "schema_version": "vla-wam-shared-v2-local-hf-snapshot-v1",
        "repo_id": args.repo_id,
        "expected_revision": args.expected_revision,
        "snapshot": str(snapshot),
        "revision_metadata_verified_for_every_top_level_file": True,
        "revision_evidence": revision_evidence,
        "config_contract": {
            "model_type": config["model_type"],
            "architectures": config["architectures"],
            "transformers_version": config.get("transformers_version"),
        },
        "files": {
            path.name: {"size": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
