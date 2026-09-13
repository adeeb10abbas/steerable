from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


WORKSHOP = Path(__file__).resolve().parents[1]
LAYOUT = WORKSHOP / "experiments/forecast_layout"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SERVER = load_module("wmf_d1_server", LAYOUT / "d1_instrumented_server.py")
PROBE = load_module("wmf_d1_probe", LAYOUT / "d1_probe.py")


class FakeHead:
    def __init__(self) -> None:
        self.current_start_frame = 17
        self.kv_cache1 = [torch.ones(1)]
        self.kv_cache_neg = [torch.ones(1)]
        self.crossattn_cache = [torch.ones(1)]
        self.crossattn_cache_neg = [torch.ones(1)]
        self.clip_feas = torch.ones(1)
        self.ys = torch.ones(1)
        self.language = torch.ones(1)
        self.seed = 1140
        self.cfg_scale = 5.0
        self.num_inference_steps = 16
        self.num_frame_per_block = 4
        self.action_horizon = 24
        self.ip_rank = 0
        self.ip_size = 2
        self.dynamic_cache_schedule = False
        self.trt_engine = None
        self.dit_step_mask = list(SERVER.EXPECTED_DIT_STEP_MASK)

    def _create_kv_caches(self):
        self.kv_cache1 = [torch.zeros(1)]
        self.kv_cache_neg = [torch.zeros(1)]
        return self.kv_cache1, self.kv_cache_neg

    def _create_crossattn_caches(self):
        self.crossattn_cache = [torch.zeros(1)]
        self.crossattn_cache_neg = [torch.zeros(1)]
        return self.crossattn_cache, self.crossattn_cache_neg


class FakePolicy:
    def __init__(self) -> None:
        self.trained_model = type("Model", (), {})()
        self.trained_model.action_head = FakeHead()


class FakeVAE:
    def decode(self, latent, **kwargs):
        del kwargs
        return torch.zeros((1, 3, latent.shape[2], 2, 2), dtype=torch.float32)


class FakeRuntimeHead(FakeHead):
    def __init__(self) -> None:
        super().__init__()
        self.vae = FakeVAE()
        self.tiled = False
        self.tile_size_height = 2
        self.tile_size_width = 2
        self.tile_stride_height = 1
        self.tile_stride_width = 1


class FakeTrainedModel:
    def __init__(self) -> None:
        self.action_head = FakeRuntimeHead()

    def lazy_joint_video_action_causal(self, normalized_input, latent_video=None):
        del normalized_input, latent_video
        head = self.action_head
        if head.current_start_frame == 0:
            head._create_kv_caches()
            head._create_crossattn_caches()
        head.current_start_frame = 5
        head.clip_feas = torch.zeros(1)
        head.ys = torch.zeros(1)
        head.language = torch.zeros(1)
        return "result", torch.ones((1, 4, 5, 2, 2), dtype=torch.bfloat16)


class FakeGrootPolicy:
    def __init__(self) -> None:
        self.trained_model = FakeTrainedModel()

    def lazy_joint_forward_causal(self, batch):
        normalized = {
            "images": torch.from_numpy(batch.obs["video.exterior_image_1_left"]).clone(),
            "state": torch.from_numpy(batch.obs["state.joint_position"]).clone(),
            "text": batch.obs["annotation.language.action_text"],
        }
        return self.trained_model.lazy_joint_video_action_causal(normalized)


class FakeOfficialPolicy:
    def __init__(self, groot_policy, signal_group, output_dir=None):
        del signal_group, output_dir
        self._policy = groot_policy
        self._frame_buffers = {"video.exterior_image_1_left": []}
        self._call_count = 0
        self._is_first_call = True
        self.video_across_time = []
        self._current_session_id = None
        self.action = np.arange(24 * 8, dtype=np.float32).reshape(24, 8)

    def reset(self, reset_info):
        del reset_info
        self._frame_buffers = {"video.exterior_image_1_left": []}
        self._call_count = 0
        self._is_first_call = True
        self.video_across_time = []

    def infer(self, obs):
        self._call_count += 1
        self._is_first_call = False
        self._current_session_id = obs["session_id"]
        converted = {
            "video.exterior_image_1_left": obs["observation/exterior_image_0_left"][None],
            "state.joint_position": obs["observation/joint_position"][None],
            "annotation.language.action_text": obs["prompt"],
        }
        batch = type("Batch", (), {"obs": converted})()
        _, latent = self._policy.lazy_joint_forward_causal(batch)
        self.video_across_time.append(latent)
        return self.action


class D1ContractTests(unittest.TestCase):
    def test_frozen_identity_and_probe_plan(self) -> None:
        identity = json.loads((LAYOUT / "d1_identity_contract.json").read_text())
        plan = json.loads((LAYOUT / "d1_probe_plan.json").read_text())
        self.assertEqual(
            identity["source"]["commit"],
            "ab790c198fbce33503358efbbd4187ce9a89adf3",
        )
        self.assertEqual(
            identity["checkpoint"]["revision"],
            "96ad344138c66e82536422432ad742f015784942",
        )
        self.assertEqual(
            identity["tokenizer"]["revision"],
            "66cb9e7e85526fe440a945569e42c72fb6cbc0ad",
        )
        self.assertFalse(identity["action_path"]["custom_s2_allowed"])
        self.assertEqual(identity["runtime"]["distributed_world_size"], 2)
        self.assertEqual(identity["runtime"]["returned_action_shape"], [24, 8])
        self.assertEqual(identity["runtime"]["effective_noise_seed"], 1140)
        self.assertEqual(len(identity["checkpoint"]["files"]), 25)
        self.assertEqual(len(identity["tokenizer"]["files"]), 4)
        parsed, digest = PROBE.load_probe_plan(LAYOUT / "d1_probe_plan.json")
        self.assertEqual(parsed, plan)
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            [probe["offline_decode"] for probe in parsed["probes"]],
            [False, False, False, True, True, True],
        )

    def test_full_temporal_reset_and_official_head_gate(self) -> None:
        policy = FakePolicy()
        receipt = SERVER.reset_temporal_state(policy, rank=0, reset_id="unit-reset")
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["after"]["current_start_frame"], 0)
        for field in SERVER.RESET_FIELDS_TO_NONE:
            self.assertTrue(receipt["after"]["fields"][field]["is_none"])
        head_receipt = SERVER.validate_official_head(policy, rank=0)
        self.assertEqual(head_receipt["status"], "passed")
        policy.trained_model.action_head.action_cfg_scale = 2.0
        with self.assertRaisesRegex(RuntimeError, "action_cfg_scale"):
            SERVER.validate_official_head(policy, rank=0)

    def test_cache_reinitialization_evidence_requires_both_ranks(self) -> None:
        metrics = []
        for rank in (0, 1):
            policy = FakePolicy()
            policy.trained_model.action_head.ip_rank = rank
            SERVER.reset_temporal_state(policy, rank=rank, reset_id="unit-reset")
            before = SERVER.temporal_snapshot(policy, rank=rank)
            head = policy.trained_model.action_head
            with SERVER.capture_cache_reinitialization(head) as events:
                head._create_kv_caches()
                head._create_crossattn_caches()
                head.current_start_frame = 5
            after = SERVER.temporal_snapshot(policy, rank=rank)
            metrics.append(
                {
                    "rank": rank,
                    "temporal_before": before,
                    "temporal_after": after,
                    "cache_reinitialization": events,
                }
            )
        SERVER.validate_first_request_cache_evidence(metrics)
        with self.assertRaisesRegex(RuntimeError, "ranks 0 and 1"):
            SERVER.validate_first_request_cache_evidence(metrics[:1])

    def test_exact_mapping_hash_is_path_independent(self) -> None:
        values = {
            "array": np.arange(6, dtype=np.float64).reshape(2, 3),
            "tensor": torch.arange(4, dtype=torch.bfloat16),
            "text": "fixed",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = SERVER.save_exact_mapping(values, root / "first", prefix="value")
            second = SERVER.save_exact_mapping(values, root / "second", prefix="value")
            self.assertEqual(first["content_sha256"], second["content_sha256"])
            PROBE._verify_mapping(first, root / "manifest.json")

    def test_wrapper_returns_official_object_and_finalizes_probe(self) -> None:
        Policy = SERVER.make_instrumented_policy_class(FakeOfficialPolicy)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            future_root = root / "future"
            future_root.mkdir()
            policy = FakeGrootPolicy()

            def gathered(local, *, group):
                del group
                peer = json.loads(json.dumps(local))
                peer["rank"] = 1
                if "before" in peer:
                    peer["before"]["rank"] = 1
                    peer["after"]["rank"] = 1
                if "temporal_before" in peer:
                    peer["temporal_before"]["rank"] = 1
                    peer["temporal_after"]["rank"] = 1
                return [local, peer]

            with (
                mock.patch.object(SERVER.dist, "broadcast"),
                mock.patch.object(SERVER.dist, "broadcast_object_list"),
                mock.patch.object(SERVER, "_all_gather_object", side_effect=gathered),
            ):
                wrapper = Policy(
                    groot_policy=policy,
                    signal_group=object(),
                    future_root=future_root,
                    server_contract_sha256="a" * 64,
                )
                wrapper.reset(
                    {
                        SERVER.RESET_KEY: {
                            "episode_id": "unit_probe",
                            "expected_session_id": "fixed-session",
                        }
                    }
                )
                images = np.zeros((180, 320, 3), dtype=np.uint8)
                request = {
                    "observation/exterior_image_0_left": images,
                    "observation/exterior_image_1_left": images.copy(),
                    "observation/wrist_image_left": images.copy(),
                    "observation/joint_position": np.zeros(7, dtype=np.float64),
                    "observation/cartesian_position": np.zeros(6, dtype=np.float64),
                    "observation/gripper_position": np.zeros(1, dtype=np.float64),
                    "prompt": SERVER.LEFT,
                    "session_id": "fixed-session",
                    SERVER.MEASUREMENT_KEY: {
                        "probe_id": "unit_probe",
                        "offline_decode": True,
                    },
                }
                returned = wrapper.infer(request)
                self.assertIs(returned, wrapper.action)
                wrapper.reset({SERVER.RESET_KEY: {"finalize_only": True}})

            manifest_path = future_root / "episodes/unit_probe/episode_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["request_count"], 1)
            record = manifest["requests"][0]
            self.assertFalse(record["custom_s2_used"])
            self.assertTrue(record["offline_decode"]["performed"])
            self.assertEqual(record["official_returned_action"]["shape"], [24, 8])


class D1EvaluatorTests(unittest.TestCase):
    def _reset_receipt(self) -> dict:
        fields = {
            field: {"is_none": True} for field in SERVER.RESET_FIELDS_TO_NONE
        }
        return {
            "status": "passed",
            "world_size": 2,
            "rank_receipts": [
                {
                    "rank": rank,
                    "status": "passed",
                    "after": {"current_start_frame": 0, "fields": fields},
                }
                for rank in (0, 1)
            ],
        }

    def _rank_metrics(self) -> list[dict]:
        pre_fields = {
            field: {"is_none": True} for field in SERVER.RESET_FIELDS_TO_NONE
        }
        post_fields = dict(pre_fields)
        for field in SERVER.CACHE_FIELDS:
            post_fields[field] = {"is_none": False}
        return [
            {
                "rank": rank,
                "wall_seconds": 1.0,
                "temporal_before": {"current_start_frame": 0, "fields": pre_fields},
                "temporal_after": {"current_start_frame": 5, "fields": post_fields},
                "cache_reinitialization": {
                    "_create_kv_caches": [{}],
                    "_create_crossattn_caches": [{}],
                },
            }
            for rank in (0, 1)
        ]

    def test_evaluator_accepts_exact_six_request_contract(self) -> None:
        plan, plan_hash = PROBE.load_probe_plan(LAYOUT / "d1_probe_plan.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            future_root = root / "future"
            episodes = future_root / "episodes"
            episodes.mkdir(parents=True)
            server_contract_path = future_root / "server_contract.json"
            server_contract = {
                "schema_version": SERVER.SCHEMA_VERSION,
                "status": "passed",
                "configuration_id": "D1",
                "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
                "custom_s2_used": False,
                "patched_s1_used": False,
                "world_size": 2,
                "returned_action_shape": [24, 8],
                "effective_official_model_noise_seed": 1140,
                "official_repository_commit": SERVER.OFFICIAL_COMMIT,
                "checkpoint_root": "/checkpoint",
                "tokenizer_root": "/tokenizer",
                "port": 8123,
            }
            PROBE.atomic_write_json(server_contract_path, server_contract)
            run_receipt = {"returned_actions": {}}

            for probe in plan["probes"]:
                probe_id = probe["id"]
                episode_dir = episodes / probe_id
                request_dir = episode_dir / "request_0000"
                request_dir.mkdir(parents=True)
                is_right = probe["relation"] == "right"
                action = np.full((24, 8), float(is_right), dtype=np.float32)
                latent = torch.full((1, 4, 5, 2, 2), float(is_right), dtype=torch.bfloat16)
                action_path = request_dir / "action.npy"
                latent_path = request_dir / "latent.pt"
                np.save(action_path, action, allow_pickle=False)
                torch.save(latent, latent_path)
                raw = SERVER.save_exact_mapping(
                    {"pixels": np.arange(4, dtype=np.uint8)},
                    request_dir / "raw",
                    prefix="raw",
                )
                converted = SERVER.save_exact_mapping(
                    {"pixels": np.arange(4, dtype=np.uint8), "prompt": probe["prompt"]},
                    request_dir / "converted",
                    prefix="converted",
                )
                normalized = SERVER.save_exact_mapping(
                    {"pixels": torch.arange(4), "prompt": probe["prompt"]},
                    request_dir / "normalized",
                    prefix="normalized",
                )
                if probe["offline_decode"]:
                    decoded_tensor_path = request_dir / "decoded.pt"
                    decoded_rgb_path = request_dir / "decoded.npy"
                    decoded_tensor = torch.zeros((1, 3, 2, 2, 2))
                    decoded_rgb = np.zeros((2, 2, 2, 3), dtype=np.uint8)
                    torch.save(decoded_tensor, decoded_tensor_path)
                    np.save(decoded_rgb_path, decoded_rgb, allow_pickle=False)
                    decode = {
                        "requested": True,
                        "performed": True,
                        "latent_data_sha256_before": SERVER.tensor_data_sha256(latent),
                        "latent_data_sha256_after": SERVER.tensor_data_sha256(latent),
                        "wall_seconds": 1.0,
                        "decoded_tensor": {
                            "path": str(decoded_tensor_path),
                            "file_sha256": SERVER.sha256_file(decoded_tensor_path),
                            "data_sha256": SERVER.tensor_data_sha256(decoded_tensor),
                        },
                        "decoded_rgb": {
                            "path": str(decoded_rgb_path),
                            "file_sha256": SERVER.sha256_file(decoded_rgb_path),
                            "data_sha256": SERVER.array_data_sha256(decoded_rgb),
                        },
                    }
                else:
                    decode = {"requested": False, "performed": False}
                request = {
                    "schema_version": SERVER.REQUEST_SCHEMA_VERSION,
                    "probe_id": probe_id,
                    "prompt": probe["prompt"],
                    "effective_official_model_noise_seed": 1140,
                    "official_forward_call_count": 1,
                    "raw_inputs": raw,
                    "converted_inputs": converted,
                    "normalized_model_inputs": normalized,
                    "official_returned_action": {
                        "path": str(action_path),
                        "file_sha256": SERVER.sha256_file(action_path),
                        "data_sha256": SERVER.array_data_sha256(action),
                    },
                    "latent_video": {
                        "path": str(latent_path),
                        "file_sha256": SERVER.sha256_file(latent_path),
                        "data_sha256": SERVER.tensor_data_sha256(latent),
                    },
                    "offline_decode": decode,
                    "temporal_and_cache_rank_metrics": self._rank_metrics(),
                    "cost": {
                        "inference_wall_seconds_rank0_wrapper": 1.0,
                        "summed_rank_forward_gpu_seconds_proxy": 2.0,
                    },
                }
                manifest = {
                    "schema_version": SERVER.EPISODE_SCHEMA_VERSION,
                    "status": "complete",
                    "request_count": 1,
                    "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
                    "custom_s2_used": False,
                    "patched_s1_used": False,
                    "two_rank_reset": self._reset_receipt(),
                    "requests": [request],
                }
                PROBE.atomic_write_json(episode_dir / "episode_manifest.json", manifest)
                client_path = root / f"client_{probe_id}.npy"
                np.save(client_path, action, allow_pickle=False)
                run_receipt["returned_actions"][probe_id] = {
                    "path": str(client_path),
                    "file_sha256": SERVER.sha256_file(client_path),
                    "data_sha256": SERVER.array_data_sha256(action),
                }

            report = PROBE.evaluate(
                plan=plan,
                plan_sha256=plan_hash,
                future_root=future_root,
                run_receipt=run_receipt,
                fixture_path=root / "fixture.npz",
                fixture_sha256="0" * 64,
                server_contract_path=server_contract_path,
            )
            self.assertTrue(report["passed"], report["failed_checks"])
            self.assertEqual(report["generation_request_count"], 6)
            self.assertEqual(report["behavioral_episode_count"], 0)


if __name__ == "__main__":
    unittest.main()
