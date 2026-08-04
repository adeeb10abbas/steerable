import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_v2a015_cfg_media as media  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}


def write_npy_rgb(path: Path, shape=(33, 2, 3, 3)) -> None:
    header = repr(
        {"descr": "|u1", "fortran_order": False, "shape": shape}
    ).encode("latin1")
    padding = 64 - ((10 + len(header) + 1) % 64)
    header = header + b" " * padding + b"\n"
    payload = bytes(index % 251 for index in range(media.math.prod(shape)))
    path.write_bytes(b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(header)) + header + payload)


class V2A015MediaCompilerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _make_evidence(self, arm: str):
        contract = media.ARM_CONTRACTS[arm]
        episodes = []
        pair_paths = []
        expected_prediction_count = 0
        for seed in media.SEEDS:
            pair_dir = self.root / f"{arm}-{seed}"
            pair_dir.mkdir()
            cells = []
            for relation in media.RELATIONS:
                viewport = pair_dir / f"{relation}-viewport.mp4"
                viewport.write_bytes(f"viewport-{seed}-{relation}".encode())
                simulator = {"viewport_video": record(viewport)}
                base = {
                    "environment_seed": seed,
                    "requested_relation": relation,
                    "prompt": media.PROMPTS[relation],
                    "simulator_artifacts": simulator,
                }
                episode = {
                    **base,
                    "requested_success": relation == "right",
                    "actions_executed": 217 if relation == "right" else 450,
                }
                if arm == "dreamzero_action_cfg_s2":
                    decode = pair_dir / f"{relation}-official-decode.mp4"
                    decode.write_bytes(f"decode-{seed}-{relation}".encode())
                    future = pair_dir / f"{relation}-future.json"
                    future.write_text(
                        json.dumps({"official_reset_decode": [record(decode)]}) + "\n"
                    )
                    future_record = {**record(future), "official_decode_count": 1}
                    base["future_manifest"] = future_record
                    episode.update(
                        {
                            "future_manifest": future_record,
                            "official_decoded_future_count": 1,
                            "official_decoded_futures": [record(decode)],
                        }
                    )
                    expected_prediction_count += 1
                else:
                    requests = []
                    compiled_requests = []
                    for index in range(2):
                        future = pair_dir / f"{relation}-future-{index}.npy"
                        write_npy_rgb(future)
                        request_record = {
                            "request_index": index,
                            "decoded_future": record(future),
                        }
                        requests.append(request_record)
                        compiled_requests.append(
                            {
                                **request_record,
                                "prompt": media.PROMPTS[relation],
                            }
                        )
                        expected_prediction_count += 1
                    base.update(
                        {"model_requests": requests, "decoded_future_count": len(requests)}
                    )
                    episode.update(
                        {
                            "imagined_future_requests": compiled_requests,
                            "decoded_future_count": len(requests),
                        }
                    )
                cells.append(base)
                episodes.append(episode)
            pair = {
                "schema_version": contract.pair_schema,
                "status": "complete_behavioral_pair_candidate",
                "amendment_id": "V2-A015",
                "arm_id": contract.arm_id,
                "model_id": contract.model_id,
                "environment_seed": seed,
                "cells": cells,
            }
            pair_path = pair_dir / "pair_manifest.json"
            pair_path.write_text(json.dumps(pair, indent=2) + "\n")
            pair_paths.append(pair_path)
        result = {
            "schema_version": contract.result_schema,
            "status": "complete",
            "amendment_id": "V2-A015",
            "arm_id": contract.arm_id,
            "model_id": contract.model_id,
            "exact_prompts": media.PROMPTS,
            "summary": {"valid_episode_count": 6},
            "episodes": episodes,
            "provenance": {"pair_manifests": [record(path) for path in pair_paths]},
        }
        result_path = self.root / f"{arm}-result.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        return result_path, pair_paths, expected_prediction_count

    def test_npy_parser_and_extractor_are_numpy_independent(self):
        source = self.root / "future.npy"
        output = self.root / "future.rgb"
        write_npy_rgb(source)
        shape, offset, payload = media.inspect_npy_uint8_rgb(source)
        self.assertEqual(shape, (33, 2, 3, 3))
        self.assertEqual(payload, 33 * 2 * 3 * 3)
        self.assertGreater(offset, 0)
        self.assertEqual(media.extract_npy_payload(source, output), shape)
        self.assertEqual(output.stat().st_size, payload)

        source.write_bytes(source.read_bytes() + b"trailing")
        with self.assertRaisesRegex(RuntimeError, "payload length"):
            media.inspect_npy_uint8_rgb(source)

    def test_dreamzero_requires_and_retains_every_official_decode(self):
        result, pairs, expected_count = self._make_evidence("dreamzero_action_cfg_s2")
        contract, _, _, cells = media.collect_evidence(
            "dreamzero_action_cfg_s2", result, pairs
        )
        self.assertEqual(contract.arm_id, "dreamzero_action_cfg_s2")
        self.assertEqual(len(cells), 6)
        self.assertEqual(sum(len(cell.prediction_sources) for cell in cells), expected_count)
        self.assertTrue(all(not cell.prediction_shapes for cell in cells))

    def test_cosmos_requires_every_request_in_exact_order(self):
        result, pairs, expected_count = self._make_evidence("cosmos3_nano_g1")
        contract, _, _, cells = media.collect_evidence("cosmos3_nano_g1", result, pairs)
        self.assertEqual(contract.arm_id, "cosmos3_nano_no_cfg_g1")
        self.assertEqual(sum(len(cell.prediction_sources) for cell in cells), expected_count)
        self.assertTrue(
            all(shape == (33, 2, 3, 3) for cell in cells for shape in cell.prediction_shapes)
        )

        payload = json.loads(result.read_text())
        payload["episodes"][0]["imagined_future_requests"].reverse()
        result.write_text(json.dumps(payload, indent=2) + "\n")
        with self.assertRaisesRegex(RuntimeError, "request order"):
            media.collect_evidence("cosmos3_nano_g1", result, pairs)

    def test_substituted_pair_manifest_fails_compiled_provenance(self):
        result, pairs, _ = self._make_evidence("dreamzero_action_cfg_s2")
        substitute = self.root / "substitute.json"
        substitute.write_bytes(pairs[0].read_bytes())
        with self.assertRaisesRegex(RuntimeError, "provenance"):
            media.collect_evidence(
                "dreamzero_action_cfg_s2", result, [substitute, pairs[1], pairs[2]]
            )

    def test_publication_encoder_contract_is_explicit(self):
        command = media._encoder_command(Path("ffmpeg"), 15, Path("output.mp4"))
        self.assertIn("libx264", command)
        self.assertIn("yuv420p", command)
        self.assertIn("+faststart", command)
        self.assertEqual(command[command.index("-threads") + 1], "1")
        self.assertEqual(command[command.index("-map_metadata") + 1], "-1")


if __name__ == "__main__":
    unittest.main()
