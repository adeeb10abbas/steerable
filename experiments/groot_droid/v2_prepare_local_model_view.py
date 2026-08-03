#!/usr/bin/env python3
"""Create a symlink-only GR00T checkpoint view with a pinned local backbone path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REMOTE_BACKBONE = "nvidia/Cosmos-Reason2-2B"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone-revision", required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    backbone = args.backbone.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty model view: {output}")
    for required in (
        checkpoint / "config.json",
        checkpoint / "processor_config.json",
        backbone / "config.json",
        backbone / "model.safetensors",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    output.mkdir(parents=True, exist_ok=True)
    local_backbone = output / "_local_backbones/nvidia/Cosmos-Reason2-2B"
    local_backbone.parent.mkdir(parents=True, exist_ok=True)
    local_backbone.symlink_to(backbone, target_is_directory=True)

    symlinks: dict[str, str] = {}
    for source in sorted(checkpoint.iterdir()):
        if source.name in {"config.json", "processor_config.json"}:
            continue
        target = output / source.name
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        symlinks[source.name] = str(source.resolve())

    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("model_name") != REMOTE_BACKBONE:
        raise ValueError(f"Unexpected GR00T backbone identifier: {config.get('model_name')!r}")
    config["model_name"] = str(local_backbone)

    processor_config = json.loads((checkpoint / "processor_config.json").read_text())
    processor_kwargs = processor_config["processor_kwargs"]
    existing_processor_model = processor_kwargs.get("model_name")
    if existing_processor_model not in {None, REMOTE_BACKBONE}:
        raise ValueError(
            f"Unexpected processor backbone identifier: {existing_processor_model!r}"
        )
    processor_kwargs["model_name"] = str(local_backbone)

    config_path = output / "config.json"
    processor_path = output / "processor_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    processor_path.write_text(
        json.dumps(processor_config, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "schema_version": "vla-wam-shared-v2-groot-local-backbone-view-v1",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_config_sha256": _sha256(checkpoint / "config.json"),
        "source_processor_config_sha256": _sha256(
            checkpoint / "processor_config.json"
        ),
        "backbone": str(backbone),
        "backbone_revision": args.backbone_revision,
        "backbone_config_sha256": _sha256(backbone / "config.json"),
        "local_backbone_symlink": str(local_backbone),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "processor_config_path": str(processor_path),
        "processor_config_sha256": _sha256(processor_path),
        "only_semantic_override": {
            "field": "model_name",
            "from": REMOTE_BACKBONE,
            "to": str(local_backbone),
            "reason": "Resolve the verified exact gated dependency without credentials or egress.",
        },
        "symlinks": symlinks,
    }
    manifest_path = output / "local_dependency_view_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
