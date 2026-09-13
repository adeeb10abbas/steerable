#!/usr/bin/env python3
"""Prepare an exact existing Nano packed diagnostic input; never launch a model."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import socket
import time

from PIL import Image
import numpy as np

SOURCE = Path("/data/users/ali/vla_wam/raw/cosmos_v2/fixed_observation_gate")
OUTPUT = Path("/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/qualification_inputs/n3_historical_packed_5100")
EXPECTED = {
    "conditioning.png": "2a431b0fa288890b3509b314c0351c91123d5f64b237678fed972848e29cd55b",
    "source_plan.json": "c5025b51f57224308d16338db38ab57b06e7025ba9e585a05993e01a79952fc3",
}
EXPECTED_RGB = "6261ce5ab21383342c2012c14f7ff97d3dcd74e5f4202f2b3444355cc7ba3332"
AUTHORITY = {
    "commit": "ce561e66f82e95055e39d3d7711691982f6b2086",
    "path": "artifacts/vla_wam_shared_v3/phase_c/four_phrasings_v3c001/gates/cosmos3_nano_policy_droid/evidence_manifest.json",
    "input_scope": "observation.conditioning_image_path and observation.source_plan_path",
}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def descriptor(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}

def identity(value):
    value = np.ascontiguousarray(value)
    header = {"kind": "numpy", "dtype": value.dtype.str, "shape": list(value.shape)}
    return dict(header, value_sha256=hashlib.sha256(canonical(header) + value.tobytes(order="C")).hexdigest())

def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

def main():
    for name, expected in EXPECTED.items():
        if digest(SOURCE / name) != expected:
            raise ValueError("Historical source hash mismatch: " + name)
    plan = json.loads((SOURCE / "source_plan.json").read_text())
    image = np.asarray(Image.open(SOURCE / "conditioning.png").convert("RGB"))
    if image.shape != (540, 640, 3) or image.dtype != np.uint8:
        raise ValueError("Historical packed RGB shape/dtype mismatch")
    if hashlib.sha256(image.tobytes()).hexdigest() != EXPECTED_RGB:
        raise ValueError("Historical decoded RGB bytes differ from pinned diagnostic input")
    arrays = {
        "image": image,
        "joint_position": np.asarray(plan["source"]["joint_position"], dtype=np.float32),
        "gripper_position": np.asarray(plan["source"]["gripper_position"], dtype=np.float32),
    }
    if arrays["joint_position"].shape != (7,) or arrays["gripper_position"].shape != (1,):
        raise ValueError("Historical proprioceptive shape mismatch")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Non-finite historical payload")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name in EXPECTED:
        shutil.copyfile(SOURCE / name, OUTPUT / name)
        if digest(OUTPUT / name) != EXPECTED[name]:
            raise ValueError("Copied source hash mismatch")
    payload_path = OUTPUT / "observation.npz"
    with payload_path.open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    with np.load(payload_path, allow_pickle=False) as saved:
        for name, array in arrays.items():
            if not np.array_equal(saved[name], array) or saved[name].dtype != array.dtype:
                raise ValueError("Round-trip array mismatch: " + name)
    capture_receipt = {
        "schema_version": "wmf-n3-historical-packed-observation-capture-v1",
        "preparation_wall_time_ns": time.time_ns(),
        "preparation_host": socket.gethostname(),
        "preparation_uid": os.getuid(),
        "source_artifacts": {name: descriptor(SOURCE / name) for name in EXPECTED},
        "authority": AUTHORITY,
        "historical_source": plan["source"],
        "payload_arrays": {key: identity(array) for key, array in arrays.items()},
        "image_raw_rgb_sha256": EXPECTED_RGB,
        "transform": "Decode the existing grounded PNG to uint8 RGB using Pillow; decoded RGB bytes must match the pinned historical RGB hash. Read recorded joint/gripper arrays from source_plan as float32, matching the pinned historical Nano probe. No resize or image editing.",
        "timing_provenance": "Native camera capture timestamps and frame counters were not preserved in these two historical source files.",
        "grounded_input_disclosure": "The frozen grounded diagnostic PNG differs from the original source conditioning PNG and RGB hashes recorded inside source_plan. Its own bytes are verified against the later pinned Nano fixed-observation evidence; this is not claimed to be an untouched original camera frame.",
        "historical_episode_identity": "excluded calibration seed 5100, episode 0, replan 0",
        "is_H01_seed9400_observation": False,
        "settled_reset_attestation_available": False,
        "physical_time_mapping_qualified": False,
        "behavioral_episode_count": 0,
        "model_requests_executed_by_preparer": 0,
    }
    receipt_path = OUTPUT / "capture_receipt.json"
    write_json(receipt_path, capture_receipt)
    capture = {
        "kind": "historical_packed_policy_input",
        "provenance_kind": "archived_policy_input",
        "historical_timing_unavailable": True,
        "capture_id": "v1_calibration_cosmos_left_5100:episode0:replan0:grounded_diagnostic",
        "settled_reset_identity": None,
        "simulator_observation_id": "RubiksCubeLeftOfBowlMatchedTask:seed5100:episode0:replan0",
        "camera_frame_ids": {"observation/image": "historical_packed_diagnostic_frame"},
        "camera_capture_time_ns": {"observation/image": None},
        "capture_receipt": {"path": receipt_path.name, "sha256": digest(receipt_path)},
        "native_capture_timing_available": False,
        "physical_time_mapping_qualified": False,
        "unavailable_reason": "The historical source plan binds episode/replan identity and real recorded state, but does not preserve native camera capture timestamps, camera counters, or a settled-reset attestation.",
    }
    manifest = {
        "schema_version": "wmf-n3-fixed-observation-v1",
        "observation_id": capture["capture_id"],
        "payload": {
            "path": payload_path.name,
            "sha256": digest(payload_path),
            "arrays": {"observation/" + key: dict(archive_key=key, **identity(value)) for key, value in arrays.items()},
        },
        "source_capture": capture,
        "source_authority": AUTHORITY,
        "input_role": "nonbehavioral_fixed_observation_generation_diagnostic_only",
    }
    manifest_path = OUTPUT / "observation_manifest.json"
    write_json(manifest_path, manifest)
    result = {
        "schema_version": "wmf-n3-archived-observation-preparation-v1",
        "status": "prepared_and_payload_validated",
        "manifest": descriptor(manifest_path),
        "payload": descriptor(payload_path),
        "capture_receipt": descriptor(receipt_path),
        "source_artifacts": capture_receipt["source_artifacts"],
        "array_identities": manifest["payload"]["arrays"],
        "wire_observation_sha256": hashlib.sha256(canonical({"observation/" + k: identity(v) for k, v in arrays.items()})).hexdigest(),
        "physical_time_mapping_qualified": False,
        "is_H01_seed9400_observation": False,
        "model_requests_executed_by_preparer": 0,
        "runner_loader_validation": "pending parent correction to official three-field Nano wire contract and explicit historical unavailable timing",
    }
    write_json(OUTPUT / "preparation_receipt.json", result)
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__":
    main()
