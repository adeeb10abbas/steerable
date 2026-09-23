import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any

import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as transforms_F

ROOT = Path(__file__).parent
OUTPUT = ROOT / "sgw-bi-policy-inputs"
ARCHIVE = ROOT / "sgw-bi-review-evidence.tar.gz"
torch.set_num_threads(2)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def source(name):
    row = json.loads((ROOT / name).read_text())
    assert digest(row["text"].encode()) == row["sha256"]
    return row


image_utils = source("sgw-native-image-utils-ao.json")
nano = source("sgw-native-nano-observation-at.json")
transform = source("sgw-native-nano-transform-bi.json")
sources = json.loads((ROOT / "sgw-native-nano-spatial-config-bi.json").read_text())
utils = sources["files"]["cosmos_framework/data/vfm/utils.py"]
assert digest(utils["text"].encode()) == utils["sha256"]
config = json.loads((ROOT / "sgw-native-nano-checkpoint-config-bi.json").read_text())
assert "dataloader_train" not in config["config"]

d1 = {}
exec(compile(image_utils["text"], image_utils["path"], "exec"), d1)
namespace = {"np": np, "torch": torch, "F": torch.nn.functional, "Any": Any}
names = {"_ensure_rgb_uint8_image", "_resize_rgb_uint8", "_compose_roboarena_views", "_extract_observation_image"}
nodes = [n for n in ast.parse(nano["text"]).body if isinstance(n, ast.FunctionDef) and n.name in names]
assert len(nodes) == len(names)
exec(compile(ast.Module(body=nodes, type_ignores=[]), nano["path"], "exec"), namespace)
table = next(n for n in ast.parse(utils["text"]).body
             if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
             and n.target.id == "VIDEO_RES_SIZE_INFO")
namespace.update({"transforms_F": transforms_F})
exec(compile(ast.Module(body=[table], type_ignores=[]), utils["path"], "exec"), namespace)
names = {"find_closest_target_size", "reflection_pad_to_target", "VideoResize"}
nodes = [n for n in ast.parse(transform["text"]).body
         if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
assert len(nodes) == len(names)
exec(compile(ast.Module(body=nodes, type_ignores=[]), transform["path"], "exec"), namespace)
resize = namespace["VideoResize"]()

manifest = {
    "schema_version": "sgw-01-model-free-policy-image-replay-v1",
    "claim_boundary": "Exact source CPU image preprocessing replay of real bi views; no model/server was loaded. D1 images are client API camera inputs, not a qualified final internal model tensor. N3 spatial stage uses the source default selected when the pinned checkpoint config has no training dataloader. Native runtime qualification remains required.",
    "model_requests": 0,
    "behavioral_episodes": 0,
    "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__},
    "sources": [{key: row[key] for key in ("path", "sha256")} for row in (image_utils, nano, transform, utils)],
    "checkpoint_config": {key: config[key] for key in ("path", "sha256", "download_metadata")},
    "d1_config": {"height": 180, "width": 320, "resize": "pad", "cam2_source": "right"},
    "n3_config": {"server_height": 540, "server_width": 640, "resolution": "480", "keep_aspect_ratio": True, "frames": 33},
    "images": [],
}


def save(scene, slot, array, source_rows, **extra):
    assert array.dtype == np.uint8 and array.ndim == 3 and array.shape[-1] == 3
    path = OUTPUT / f"{scene}--{slot}.png"
    Image.fromarray(array).save(path)
    manifest["images"].append({
        "scene": scene, "slot": slot, "path": str(path),
        "shape": list(array.shape), "dtype": str(array.dtype),
        "range": [int(array.min()), int(array.max())],
        "array_sha256": digest(array.tobytes()), "png_sha256": digest(path.read_bytes()),
        "source_views": source_rows, **extra,
    })


OUTPUT.mkdir(exist_ok=False)
with tarfile.open(ARCHIVE) as archive:
    export = json.load(archive.extractfile("export_manifest.json"))
    records = {row["export_path"]: row for row in export["files"]}
    for scene in ("0-height-left", "1-height-right", "2-dist-left", "3-dist-right"):
        frames, rows = {}, {}
        for camera in ("over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam"):
            name = f"{scene}/views/{camera}.npy"
            raw = archive.extractfile(name).read()
            assert digest(raw) == records[name]["sha256"]
            frames[camera] = np.load(io.BytesIO(raw), allow_pickle=False)
            rows[camera] = records[name]
            save(scene, f"D1-{camera}", d1["resize_with_pad"](frames[camera], 180, 320), [rows[camera]])
        wire = {
            "observation/wrist_image_left": frames["wrist_cam"],
            "observation/exterior_image_1_left": frames["over_shoulder_left_camera"],
            "observation/exterior_image_2_left": frames["over_shoulder_right_camera"],
        }
        composite = namespace["_extract_observation_image"](wire)
        image = namespace["_resize_rgb_uint8"](composite, (540, 640))
        order = [rows[name] for name in ("wrist_cam", "over_shoulder_left_camera", "over_shoulder_right_camera")]
        save(scene, "N3-server-540x640", image, order, layout="wrist above left/right exterior views")
        video = torch.zeros((3, 33, 540, 640), dtype=torch.uint8)
        video[:, 0] = torch.from_numpy(image.copy()).permute(2, 0, 1)
        transformed = resize({"video": video}, "480")
        final = transformed["video"][:, 0].permute(1, 2, 0).contiguous().numpy()
        save(scene, "N3-spatial-480", final, order,
             image_size=transformed["image_size"].tolist(),
             layout="wrist above left/right exterior views; source-defined resize and reflection padding")
manifest["reproducer_sha256"] = digest(Path(__file__).read_bytes())
(OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps({"output": str(OUTPUT), "images": len(manifest["images"]),
                  "manifest_sha256": digest((OUTPUT / "manifest.json").read_bytes()),
                  "n3_shapes": [r["shape"] for r in manifest["images"] if r["slot"] == "N3-spatial-480"]}, indent=2))
