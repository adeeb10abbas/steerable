"""Hash actual SGW-01 RoboLab scene inputs without importing Isaac or a policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("--scene", default="rubiks_cube_banana_bowl.usda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.robolab_root.resolve()
    if not (root / ".git").is_dir():
        raise ValueError("asset manifest requires the pinned RoboLab checkout")
    paths = [
        root / "assets/scenes" / args.scene,
        root / "assets/objects/ycb/bowl.usd",
        root / "assets/objects/ycb/banana.usd",
        root / "assets/objects/hot3d/rubiks_cube.usd",
        root / "assets/objects/ycb/textures/obj_000013.png",
        root / "assets/objects/ycb/textures/obj_000010.png",
        root / "assets/objects/hot3d/textures/obj_000030.png",
        root / "assets/fixtures/franka_table.usd",
        root / "assets/fixtures/table_maple.usd",
        root / "assets/fixtures/Props/instaceable_meshes.usd",
        root / "assets/backgrounds/default/home_office.exr",
        root / "assets/robots/franka_robotiq_2f_85_flattened.usd",
    ]
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
        "scene": records[0],
        "assets": records[1:],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, allow_nan=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
