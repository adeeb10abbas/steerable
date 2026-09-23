"""Verify exact CPU replay pixels without asserting live model-input parity."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


SOURCE_MANIFEST_SHA256 = "065a192aef51b65870ed885d176292ef15b8e7198d8b17ac66ef77c217169b41"
SCENES = ("0-height-left", "1-height-right", "2-dist-left", "3-dist-right")
SLOTS = (
    "D1-over_shoulder_left_camera", "D1-over_shoulder_right_camera", "D1-wrist_cam",
    "N3-server-540x640", "N3-spatial-480",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(root):
    manifest = json.loads((root / "manifest.json").read_text())
    keyed = {(row["scene"], row["slot"]): row for row in manifest["images"]}
    assert len(keyed) == len(manifest["images"]) == 20
    assert set(keyed) == {(scene, slot) for scene in SCENES for slot in SLOTS}
    for key, row in keyed.items():
        filename = f"{key[0]}--{key[1]}.png"
        assert Path(row["path"]).name == filename
        path = root / filename
        pixels = np.asarray(Image.open(path))
        assert pixels.dtype == np.uint8 and list(pixels.shape) == row["shape"]
        assert [int(pixels.min()), int(pixels.max())] == row["range"]
        assert sha256(path) == row["png_sha256"]
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == row["array_sha256"]
    return manifest, keyed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-images", required=True, type=Path)
    parser.add_argument("--replayed-images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert sha256(args.source_images / "manifest.json") == SOURCE_MANIFEST_SHA256
    source, source_rows = rows(args.source_images)
    replay, replay_rows = rows(args.replayed_images)
    assert {key: value for key, value in source.items() if key != "images"} == {
        key: value for key, value in replay.items() if key != "images"
    }
    comparisons = []
    for key, expected in source_rows.items():
        actual = replay_rows[key]
        assert {k: v for k, v in expected.items() if k != "path"} == {
            k: v for k, v in actual.items() if k != "path"
        }
        comparisons.append({
            "scene": key[0], "slot": key[1], "shape": expected["shape"],
            "source_png_sha256": expected["png_sha256"],
            "replayed_png_sha256": actual["png_sha256"],
            "source_rgb_sha256": expected["array_sha256"],
            "replayed_rgb_sha256": actual["array_sha256"],
            "all_metadata_equal_except_output_path": True,
        })
    padding = []
    for scene in SCENES:
        server = np.asarray(Image.open(args.source_images / f"{scene}--N3-server-540x640.png"))
        spatial = np.asarray(Image.open(args.source_images / f"{scene}--N3-spatial-480.png"))
        assert server.shape == (540, 640, 3) and spatial.shape == (544, 736, 3)
        assert np.array_equal(spatial, np.pad(server, ((0, 4), (0, 96), (0, 0)), mode="reflect"))
        padding.append({
            "scene": scene, "content_rectangle_xyxy": [0, 0, 640, 540],
            "padding_right_px": 96, "padding_bottom_px": 4,
            "independent_numpy_reflection_matches_every_rgb_pixel": True,
        })
    result = {
        "schema_version": "sgw-01-independent-bi-policy-pixel-verification-v1",
        "claim_boundary": (
            "Exact offline CPU replay of supplied image functions and configuration; "
            "not live request-time model-tensor parity, model recognition, predictions, "
            "runtime qualification, or behavioral release."
        ),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "independent_replay_manifest_sha256": sha256(args.replayed_images / "manifest.json"),
        "producer_reproducer_sha256": source["reproducer_sha256"],
        "verifier_sha256": sha256(Path(__file__)),
        "python_version": sys.version,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "numpy", "Pillow")
        },
        "images_bit_identical": len(comparisons),
        "image_comparisons": comparisons,
        "reflection_padding_checks": padding,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "verified_images": len(comparisons), "output": str(args.output),
        "sha256": sha256(args.output), "versions": result["versions"],
    }, indent=2))


if __name__ == "__main__":
    main()
