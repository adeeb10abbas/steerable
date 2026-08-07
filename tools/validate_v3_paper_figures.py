#!/usr/bin/env python3
"""Validate hash-bearing V3 paper-figure manifests and image integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


MANIFESTS = (
    "artifacts/vla_wam_shared_v3/analysis/paper_figures/figure1_nano_instrument_sensitivity.manifest.json",
    "artifacts/vla_wam_shared_v3/analysis/paper_figures/figure2_three_checkpoint_position_reflection.manifest.json",
    "artifacts/vla_wam_shared_v3/analysis/paper_figures/figure5_cross_arena_directional_success.manifest.json",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_record(root: Path, record: dict[str, Any], label: str, checks: list[str]) -> Path:
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path escapes repository: {relative}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside repository: {relative}") from exc
    if not path.is_file():
        raise ValueError(f"{label} is missing: {relative}")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{label} byte count differs: {relative}")
    if sha256_path(path) != record["sha256"]:
        raise ValueError(f"{label} SHA-256 differs: {relative}")
    checks.append(f"{label} exists and matches bytes/SHA-256")
    return path


def validate_manifest(root: Path, relative: str, checks: list[str]) -> None:
    path = root / relative
    data = json.loads(path.read_text())
    if data.get("study_id") != "vla_wam_language_steerability_v3":
        raise ValueError(f"unexpected study id in {relative}")
    if data.get("status") != "complete_retrospective_visualization_no_new_inference":
        raise ValueError(f"unexpected status in {relative}")
    checks.append(f"{relative} fixes the V3 study and retrospective/no-inference status")

    inputs = data.get("inputs")
    outputs = data.get("outputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError(f"{relative} has no inputs")
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise ValueError(f"{relative} must bind one SVG and one PNG")
    for index, record in enumerate(inputs):
        validate_record(root, record, f"{relative} input[{index}]", checks)
    validate_record(root, data["renderer"], f"{relative} renderer", checks)
    output_paths = [validate_record(root, record, f"{relative} output[{index}]", checks) for index, record in enumerate(outputs)]
    suffixes = sorted(path.suffix for path in output_paths)
    if suffixes != [".png", ".svg"]:
        raise ValueError(f"{relative} output formats differ: {suffixes}")
    checks.append(f"{relative} binds exactly one SVG and one PNG")
    png = next(path for path in output_paths if path.suffix == ".png")
    with Image.open(png) as image:
        image.verify()
    with Image.open(png) as image:
        width, height = image.size
    if width < 2400 or height < 1200:
        raise ValueError(f"{relative} PNG is below publication resolution: {width}x{height}")
    checks.append(f"{relative} PNG decodes at publication resolution ({width}x{height})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    checks: list[str] = []
    for relative in MANIFESTS:
        validate_manifest(root, relative, checks)
    print(json.dumps({"status": "valid", "check_count": len(checks), "checks": checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
