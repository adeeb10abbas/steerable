from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "analysis/camera_crop_replay_witness.py"
)
SPEC = importlib.util.spec_from_file_location("camera_crop_replay_witness", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
witness = importlib.util.module_from_spec(SPEC)
import sys

sys.modules[SPEC.name] = witness
SPEC.loader.exec_module(witness)

RUNTIME_CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "experiments/forecast_layout/camera_crop_witness_contract.json"
)


def crop_contract(model: str) -> dict:
    generated = {
        "N3": {
            "operation": "crop_half_open_thwc",
            "canvas_shape": [33, 528, 640, 3],
            "y": [360, 528],
            "x": [0, 320],
            "output_shape": [33, 168, 320, 3],
        },
        "D1": {
            "operation": "crop_half_open_thwc",
            "canvas_shape": [9, 352, 640, 3],
            "y": [176, 352],
            "x": [0, 320],
            "output_shape": [9, 176, 320, 3],
        },
    }[model]
    output_shape = [168, 320, 3] if model == "N3" else [176, 320, 3]
    return witness.signed_document(
        {
            "schema_version": witness.SCHEMA,
            "study_id": witness.STUDY_ID,
            "contract_id": f"{model.lower()}-test",
            "model_id": model,
            "status": "qualified_from_original_camera_pixels",
            "camera_id": witness.CAMERA_ID,
            "camera_crop_id": f"{model.lower()}-test",
            "crop_operation": witness._expected_crop_operation(model),
            "image_width_px": 320,
            "image_height_px": output_shape[0],
            "generated_decoded_crop": {
                **generated,
                "selected_camera": witness.CAMERA_ID,
                "output_value_sha256": "a" * 64,
                "witness_artifact": {},
            },
            "original_camera_replay": {
                "source_camera_shape": [720, 1280, 3],
                "transform_chain": witness._expected_transform_chain(model),
                "output_shape": output_shape,
                "witness_input_value_sha256": witness.array_data_sha256(
                    np.zeros((720, 1280, 3), dtype=np.uint8)
                ),
                "output_value_sha256": witness.array_data_sha256(
                    np.zeros(output_shape, dtype=np.uint8)
                ),
                "witness_artifact": {},
            },
            "dependency_chain": {},
            "runtime_dependencies": {},
            "pinned_dependencies": {"n3": {}, "d1": {}},
            "retained_artifacts": {},
            "replay_checks": {"byte_exact": True},
            "science_counts": dict(witness.ZERO_SCIENCE_COUNTS),
            "simulator_state_render_used": False,
            "whole_frame_identity": False,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "behavioral_policy_skill_evaluated": False,
            "claim_boundary": "test",
        }
    )


class CameraCropReplayWitnessTests(unittest.TestCase):
    def test_runtime_contract_freezes_required_geometry_and_zero_authority(self) -> None:
        runtime = witness.load_json(RUNTIME_CONTRACT, "runtime contract")
        self.assertEqual(runtime["schema_version"], witness.RUNTIME_SCHEMA)
        self.assertEqual(runtime["job"]["worker_role"], "wmf-forecast-0912-worker-06")
        self.assertEqual(
            runtime["models"]["N3"]["transformed_image_size"],
            [544, 736, 540, 640],
        )
        self.assertEqual(runtime["models"]["N3"]["latent_shape"][-2:], [33, 40])
        self.assertEqual(runtime["models"]["N3"]["decoded_shape"], [33, 528, 640, 3])
        self.assertEqual(runtime["models"]["D1"]["decoded_shape"], [9, 352, 640, 3])
        self.assertEqual(runtime["science_counts"], witness.ZERO_SCIENCE_COUNTS)
        self.assertIs(runtime["simulator_state_render_used"], False)
        self.assertIs(runtime["whole_frame_identity"], False)
        self.assertIs(runtime["safe_to_release_confirmation"], False)
        self.assertIs(runtime["confirmation_released"], False)

    def test_runtime_rejects_rehashed_confirmation_release(self) -> None:
        runtime = json.loads(RUNTIME_CONTRACT.read_text(encoding="utf-8"))
        runtime["confirmation_released"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "runtime.json"
            path.write_text(json.dumps(runtime), encoding="utf-8")
            with mock.patch.dict(witness.os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                with self.assertRaisesRegex(
                    witness.CameraCropWitnessError, "authority boundary"
                ):
                    witness.run_witness("N3", path, path.parent / "output")

    def test_extract_generated_n3_crop_is_exact_half_open_geometry(self) -> None:
        contract = crop_contract("N3")
        frames = np.arange(33 * 528 * 640 * 3, dtype=np.uint32).reshape(33, 528, 640, 3)
        frames = (frames % 251).astype(np.uint8)
        result = witness.extract_generated_crop(frames, contract)
        self.assertEqual(result.shape, (33, 168, 320, 3))
        np.testing.assert_array_equal(result, frames[:, 360:528, 0:320, :])

    def test_extract_generated_d1_crop_is_exact_half_open_geometry(self) -> None:
        contract = crop_contract("D1")
        frames = np.arange(9 * 352 * 640 * 3, dtype=np.uint32).reshape(9, 352, 640, 3)
        frames = (frames % 253).astype(np.uint8)
        result = witness.extract_generated_crop(frames, contract)
        self.assertEqual(result.shape, (9, 176, 320, 3))
        np.testing.assert_array_equal(result, frames[:, 176:352, 0:320, :])

    def test_original_replay_api_uses_model_chain_and_witness_hash(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        contract = crop_contract("N3")
        downsampled = np.zeros((180, 320, 3), dtype=np.uint8)
        downsampled[168:] = 91
        with mock.patch.object(witness, "_verify_runtime_dependency_chain"), mock.patch.object(
            witness,
            "_n3_replay_stages",
            return_value={"torch_downsampled": downsampled},
        ):
            result = witness.replay_original_camera_frame(frame, contract)
        self.assertEqual(result.shape, (168, 320, 3))
        self.assertTrue(np.all(result == 0))

    def test_contract_rejects_tamper_whole_frame_and_nonzero_science(self) -> None:
        contract = crop_contract("D1")
        contract["generated_decoded_crop"]["y"] = [0, 352]
        with self.assertRaisesRegex(witness.CameraCropWitnessError, "signature changed"):
            witness.validate_camera_crop_contract(contract)

        contract = crop_contract("D1")
        unsigned = dict(contract)
        unsigned.pop("payload_sha256")
        unsigned["whole_frame_identity"] = True
        contract = witness.signed_document(unsigned)
        with self.assertRaisesRegex(witness.CameraCropWitnessError, "whole_frame_identity"):
            witness.validate_camera_crop_contract(contract)

        contract = crop_contract("N3")
        unsigned = dict(contract)
        unsigned.pop("payload_sha256")
        unsigned["science_counts"] = dict(witness.ZERO_SCIENCE_COUNTS)
        unsigned["science_counts"]["labels_created_by_job"] = 1
        contract = witness.signed_document(unsigned)
        with self.assertRaisesRegex(witness.CameraCropWitnessError, "science counts"):
            witness.validate_camera_crop_contract(contract)

        contract = crop_contract("N3")
        unsigned = dict(contract)
        unsigned.pop("payload_sha256")
        unsigned["image_height_px"] = 540
        contract = witness.signed_document(unsigned)
        with self.assertRaisesRegex(witness.CameraCropWitnessError, "image dimensions"):
            witness.validate_camera_crop_contract(contract)

    def test_secure_path_rejects_final_and_parent_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            real = root / "real"
            real.mkdir()
            payload = real / "payload.bin"
            payload.write_bytes(b"pixels")
            final_link = root / "payload-link.bin"
            final_link.symlink_to(payload)
            with self.assertRaisesRegex(witness.CameraCropWitnessError, "symlink"):
                witness.file_identity(final_link, label="final link")
            parent_link = root / "parent-link"
            parent_link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(witness.CameraCropWitnessError, "symlink"):
                witness.file_identity(parent_link / "payload.bin", label="parent link")

    def test_nested_artifact_loader_rejects_escape_and_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            request = root / "request"
            request.mkdir()
            array = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
            data = request / "0000_value.npy"
            with data.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            identity = witness.n3_value_identity(array)
            node = {
                "__type__": "numpy",
                "artifact": {
                    "path": data.name,
                    "bytes": data.stat().st_size,
                    "sha256": witness.sha256_file(data),
                },
                **identity,
            }
            structure = {"__type__": "mapping", "items": {"value": node}}
            manifest = {
                "schema_version": "wmf-lossless-nested-payload-v1",
                "role": "unit_artifact",
                "structure": structure,
                "logical_sha256": witness.sha256_bytes(
                    witness.n3_canonical_bytes(witness._logical_nested(structure))
                ),
            }
            manifest_path = request / "unit_artifact.manifest.json"
            manifest_path.write_bytes(witness.pretty_bytes(manifest))
            descriptor = {
                "manifest_path": str(manifest_path),
                "manifest_sha256": witness.sha256_file(manifest_path),
                "logical_sha256": manifest["logical_sha256"],
            }
            loaded, retained = witness.load_n3_artifact(
                descriptor, role="unit_artifact", allowed_root=request
            )
            np.testing.assert_array_equal(loaded["value"], array)
            self.assertEqual(len(retained), 2)

            data.write_bytes(b"changed")
            with self.assertRaisesRegex(witness.CameraCropWitnessError, "SHA-256 changed"):
                witness.load_n3_artifact(
                    descriptor, role="unit_artifact", allowed_root=request
                )

    def test_generated_crop_rejects_wrong_dtype_and_canvas(self) -> None:
        contract = crop_contract("D1")
        with self.assertRaisesRegex(witness.CameraCropWitnessError, "canvas shape"):
            witness.extract_generated_crop(np.zeros((9, 351, 640, 3), dtype=np.uint8), contract)
        with self.assertRaisesRegex(witness.CameraCropWitnessError, "dtype"):
            witness.extract_generated_crop(np.zeros((9, 352, 640, 3), dtype=np.float32), contract)

    def test_runtime_preflight_is_publish_safe_and_contains_no_argv(self) -> None:
        buffer = io.BytesIO()
        stdout = SimpleNamespace(buffer=buffer)
        with mock.patch.object(witness.sys, "stdout", stdout), mock.patch.dict(
            witness.os.environ,
            {
                "PYTHONPATH": "/exact/one:/exact/two:relative-redacted",
                "CUDA_VISIBLE_DEVICES": "",
            },
            clear=True,
        ):
            witness.emit_runtime_preflight("N3")
        value = json.loads(buffer.getvalue())
        self.assertEqual(value["schema_version"], "wmf-camera-crop-child-runtime-preflight-v1")
        self.assertEqual(value["phase"], "before_replay")
        self.assertEqual(value["model_id"], "N3")
        self.assertIn("openpi_client.image_tools", value["module_specs"])
        self.assertEqual(
            value["path_environment"]["PYTHONPATH"][:2],
            [{"absolute_path": "/exact/one"}, {"absolute_path": "/exact/two"}],
        )
        self.assertIn(
            "redacted_nonabsolute_value_sha256",
            value["path_environment"]["PYTHONPATH"][2],
        )
        self.assertIs(value["argv_published"], False)
        self.assertIs(value["secret_environment_published"], False)
        self.assertNotIn("argv", " ".join(value.keys()).replace("argv_published", ""))


if __name__ == "__main__":
    unittest.main()
