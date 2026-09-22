"""Hash actual SGW-01 RoboLab scene inputs without importing Isaac or a policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_git_worktree(path: Path) -> bool:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() == "true"
    except subprocess.CalledProcessError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("--scene", default="rubiks_cube_banana_bowl.usda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def referenced_usd_assets(path: Path, root: Path, seen: set[Path] | None = None) -> list[Path]:
    """Resolve direct USDA ``@asset@`` references inside the pinned checkout."""

    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        return []
    seen.add(path)
    if path.suffix.lower() not in {".usd", ".usda"}:
        return [path]
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [path]
    children: list[Path] = [path]
    for reference in re.findall(r"@([^@]+)@", text):
        candidate = (path.parent / reference).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"scene reference escapes pinned RoboLab checkout: {reference}")
        if not candidate.is_file():
            raise FileNotFoundError(f"scene reference is missing: {candidate}")
        children.extend(referenced_usd_assets(candidate, root, seen))
    return children


def main() -> None:
    args = parse_args()
    root = args.robolab_root.resolve()
    if not is_git_worktree(root):
        raise ValueError("asset manifest requires the pinned RoboLab checkout")
    scene = root / "assets/scenes" / args.scene
    paths = referenced_usd_assets(scene, root)
    if not paths:
        raise ValueError("scene dependency resolution produced no assets")
    records = []
    for path in paths:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"asset is not a file: {path}")
        records.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    output = {
        "schema_version": "sgw-01-robolab-asset-manifest-v1",
        "status": "measured_asset_files_not_fixture_qualified",
        "robolab_root": str(root),
        "robolab_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "scene": records[0],
        "assets": records[1:],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, allow_nan=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
