#!/usr/bin/env python3
"""V2-A015 provenance overlay for the frozen DreamZero RoboLab client."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from v2_robolab_client import PROMPTS, V2DreamZeroDroidClient, _sha256


AMENDMENT_ID = "V2-A015"
ARM_ID = "dreamzero_action_cfg_s2"
MODEL_ID = "dreamzero_droid_action_cfg"
CHECKPOINT_ID = "GEAR-Dreams/DreamZero-DROID"
CHECKPOINT_REVISION = "96ad344138c66e82536422432ad742f015784942"
OFFICIAL_SOURCE_COMMIT = "ab790c198fbce33503358efbbd4187ce9a89adf3"
ACTION_CFG_STYLE_SCALE = 2.0
VIDEO_CFG_SCALE = 5.0
BASELINE_ACTION_CFG_EQUIVALENT = 1.0
BASELINE_RESULT_ARTIFACT = (
    "artifacts/vla_wam_shared_v2/pilot/expansion/"
    "dreamzero_droid_direct_gate.json"
)
BASELINE_RESULT_SHA256 = (
    "4c76cdc3ca9eaf227d21d160199408f22e1b3dd7a71176a5a5dbe22223714461"
)


class V2A015DreamZeroDroidClient(V2DreamZeroDroidClient):
    """Keep the V2 controller unchanged and label every trace as s=2."""

    def __init__(
        self,
        *,
        remote_host: str,
        remote_port: int,
        environment_seed: int,
        sampling_seed_label: int,
        action_trace_dir: Path,
        amendment_path: Path,
        fixed_observation_gate_path: Path,
        server_contract_path: Path,
        future_root: Path,
    ) -> None:
        if environment_seed != sampling_seed_label:
            raise ValueError("V2-A015 environment and sampling seed labels must match")
        self.environment_seed = int(environment_seed)
        self.amendment_path = Path(amendment_path).resolve()
        self.fixed_observation_gate_path = Path(
            fixed_observation_gate_path
        ).resolve()
        self.server_contract_path = Path(server_contract_path).resolve()
        self.future_root = Path(future_root).resolve()
        self._v2a015_trace_written = False
        super().__init__(
            remote_host=remote_host,
            remote_port=remote_port,
            sampling_seed_label=sampling_seed_label,
            action_trace_dir=action_trace_dir,
        )

    def _metadata_path(self) -> Path:
        if self.prompt not in PROMPTS:
            raise ValueError("Cannot identify the frozen V2-A015 relation")
        relation = PROMPTS[self.prompt]
        return self.action_trace_dir / (
            f"seed{self.sampling_seed_label}_{relation}_executed_actions.json"
        )

    def _write_trace(self) -> None:
        if self._v2a015_trace_written or not self.executed_actions:
            return
        metadata_path = self._metadata_path()
        relation = PROMPTS[self.prompt]
        expected_paths = (
            metadata_path,
            self.action_trace_dir
            / f"seed{self.sampling_seed_label}_{relation}_executed_actions.npy",
            self.action_trace_dir
            / f"seed{self.sampling_seed_label}_{relation}_returned_raw_chunks.npy",
            self.action_trace_dir
            / (
                f"seed{self.sampling_seed_label}_{relation}_"
                "returned_executable_chunks.npy"
            ),
        )
        existing = [str(path) for path in expected_paths if path.exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite V2-A015 action evidence: {existing}"
            )

        super()._write_trace()
        metadata = json.loads(metadata_path.read_text())
        metadata.update(
            {
                "schema_version": (
                    "vla-wam-shared-v2-dreamzero-v2a015-action-trace-v1"
                ),
                "amendment_id": AMENDMENT_ID,
                "arm_id": ARM_ID,
                "model_id": MODEL_ID,
                "checkpoint": CHECKPOINT_ID,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "official_repository_commit": OFFICIAL_SOURCE_COMMIT,
                "environment_seed": self.environment_seed,
                "action_cfg_style_scale": ACTION_CFG_STYLE_SCALE,
                "baseline_action_cfg_equivalent": BASELINE_ACTION_CFG_EQUIVALENT,
                "video_cfg_scale": VIDEO_CFG_SCALE,
                "guidance_formula": (
                    "negative_action + scale * (conditional_action - negative_action)"
                ),
                "guidance_branch": (
                    "released_fixed_visual_quality_negative_prompt"
                ),
                "negative_branch_caveat": (
                    "This is CFG-style negative-branch action guidance, not strict "
                    "empty-text classifier-free guidance and not an official "
                    "DreamZero action-CFG feature."
                ),
                "baseline_result_artifact": BASELINE_RESULT_ARTIFACT,
                "baseline_result_sha256": BASELINE_RESULT_SHA256,
                "amendment": {
                    "path": str(self.amendment_path),
                    "sha256": _sha256(self.amendment_path),
                },
                "fixed_observation_release_gate": {
                    "path": str(self.fixed_observation_gate_path),
                    "sha256": _sha256(self.fixed_observation_gate_path),
                },
                "server_contract": {
                    "path": str(self.server_contract_path),
                    "sha256": _sha256(self.server_contract_path),
                    "future_root": str(self.future_root),
                },
                "claim_boundary": (
                    "This trace is one behavioral cell in the six-cell V2-A015 "
                    "DreamZero s=2 denominator. The preserved s=1 V2-A007 baseline "
                    "is referenced by hash and is not rerun or overwritten."
                ),
            }
        )
        metadata["returned_raw_chunks"]["definition"] = (
            "Derived V2-A015 s=2 24x8 absolute joint-position/gripper response "
            "before the unchanged client gripper binarization."
        )
        metadata["returned_executable_chunks"]["definition"] = (
            "V2-A015 s=2 returned chunks after the unchanged official >0.5 "
            "gripper binarization."
        )
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        self._v2a015_trace_written = True

    def reset(self, *, env_id: int | None = None) -> None:
        retain_episode = env_id is None and bool(self.executed_actions)
        manifests_before = (
            set(self.future_root.glob("episode_*/future_manifest.json"))
            if retain_episode
            else set()
        )
        super().reset(env_id=env_id)
        if not retain_episode:
            return

        deadline = time.monotonic() + 30.0
        new_manifests: set[Path] = set()
        while time.monotonic() < deadline:
            manifests_after = set(
                self.future_root.glob("episode_*/future_manifest.json")
            )
            new_manifests = manifests_after - manifests_before
            if new_manifests:
                break
            time.sleep(0.2)
        if len(new_manifests) != 1:
            raise RuntimeError(
                "Expected exactly one finalized V2-A015 future manifest after "
                f"episode reset, found {sorted(str(path) for path in new_manifests)}"
            )

        future_manifest_path = next(iter(new_manifests)).resolve()
        future_manifest = json.loads(future_manifest_path.read_text())
        if (
            future_manifest.get("schema_version")
            != "vla-wam-shared-v2-dreamzero-v2a015-future-retention-v1"
            or future_manifest.get("amendment_id") != AMENDMENT_ID
            or future_manifest.get("action_cfg_style_scale")
            != ACTION_CFG_STYLE_SCALE
            or future_manifest.get("video_cfg_scale") != VIDEO_CFG_SCALE
        ):
            raise ValueError("Behavioral future manifest is not the V2-A015 s=2 arm")
        requests = future_manifest.get("requests", [])
        if len(requests) != self.request_count:
            raise ValueError(
                "V2-A015 client/server request count mismatch: "
                f"client={self.request_count}, server={len(requests)}"
            )
        if any(record.get("prompt") != self.prompt for record in requests):
            raise ValueError("V2-A015 future manifest contains a non-static prompt")
        for index, (record, raw_chunk) in enumerate(
            zip(requests, self.returned_raw_chunks, strict=True)
        ):
            if record.get("action_cfg_style_scale") != ACTION_CFG_STYLE_SCALE:
                raise ValueError(f"Request {index} is not labeled DreamZero s=2")
            action_entry = record.get("returned_action", {})
            action_path = Path(action_entry.get("path", ""))
            if (
                not action_path.is_file()
                or action_entry.get("sha256") != _sha256(action_path)
            ):
                raise ValueError(f"Request {index} returned-action evidence is invalid")
            server_action = np.load(action_path, allow_pickle=False)
            if not np.array_equal(server_action, raw_chunk):
                raise ValueError(
                    f"Request {index} retained action differs from the client response"
                )
            latent_entry = record.get("latent_video", {})
            latent_path = Path(latent_entry.get("path", ""))
            if (
                not latent_path.is_file()
                or latent_entry.get("sha256") != _sha256(latent_path)
            ):
                raise ValueError(f"Request {index} latent-future evidence is invalid")

        decoded = future_manifest.get("official_reset_decode", [])
        if not decoded:
            raise ValueError("V2-A015 behavioral episode has no official decoded future")
        for entry in decoded:
            decoded_path = Path(entry.get("path", ""))
            if (
                not decoded_path.is_file()
                or entry.get("sha256") != _sha256(decoded_path)
            ):
                raise ValueError("V2-A015 official decoded-future evidence is invalid")

        metadata_path = self._metadata_path()
        metadata = json.loads(metadata_path.read_text())
        metadata["future_manifest"] = {
            "path": str(future_manifest_path),
            "sha256": _sha256(future_manifest_path),
            "request_count": len(requests),
            "official_decode_count": len(decoded),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
