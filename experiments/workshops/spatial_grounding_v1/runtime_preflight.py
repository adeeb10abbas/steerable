"""Exercise a real assigned RoboLab/Isaac renderer without importing a policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from .simulator_bridge import load_factory


def file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("--assets-manifest", type=Path, required=True)
    parser.add_argument("--scene-factory", required=True, help="Pinned module:factory; creates one Isaac scene")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--renderer", required=True, choices=("realtime",))
    parser.add_argument("--rendering-type", required=True, choices=("balanced",))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite preflight receipt: {args.output}")
    if not (args.robolab_root / ".git").is_dir() or not args.assets_manifest.is_file():
        raise ValueError("preflight requires a pinned RoboLab checkout and measured asset manifest")
    renderer_factory = load_factory(args.scene_factory)
    scene = renderer_factory(
        robolab_root=args.robolab_root,
        assets_manifest=args.assets_manifest,
        device=args.device,
        renderer=args.renderer,
        rendering_type=args.rendering_type,
        model_request_count=0,
    )
    if not all(hasattr(scene, method) for method in ("render", "close")):
        raise RuntimeError("scene factory must return an object with render() and close()")
    try:
        frame = scene.render()
        if not isinstance(frame, (bytes, bytearray)) or not frame:
            raise RuntimeError("Isaac renderer did not produce a nonempty RGB frame")
    finally:
        scene.close()
    commit = subprocess.check_output(
        ["git", "-C", str(args.robolab_root), "rev-parse", "HEAD"], text=True
    ).strip()
    output = {
        "schema_version": "sgw-01-robolab-isaac-renderer-preflight-v1",
        "status": "passed_zero_model_renderer_preflight",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "robolab_commit": commit,
        "robolab_root": str(args.robolab_root.resolve()),
        "assets_manifest": file_record(args.assets_manifest),
        "renderer": args.renderer,
        "rendering_type": args.rendering_type,
        "device": args.device,
        "scene_factory": args.scene_factory,
        "rendered_frame_bytes": len(frame),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, allow_nan=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
