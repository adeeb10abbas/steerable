"""Content-addressed episode evidence recorder with write-once finalization."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from experiments.online_correction_v4.adapters import EncodedVideoArtifact
from experiments.online_correction_v4.leases import AttemptFinalizer, WriteOnceViolation


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass
class ContentAddressedBlob:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass
class EpisodeEvidenceRecorder:
    attempt_path: Path
    finalizer: AttemptFinalizer
    episode_id: str
    attempt_id: str
    episode_record: dict[str, Any] = field(default_factory=dict)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    future_artifacts: list[dict[str, Any]] = field(default_factory=list)
    viewport_frames: list[dict[str, Any]] = field(default_factory=list)
    blobs: list[ContentAddressedBlob] = field(default_factory=list)
    _finalized: bool = False
    _trajectory_rows_since_flush: int = 0

    @classmethod
    def open(
        cls,
        *,
        finalizer: AttemptFinalizer,
        episode_id: str,
        attempt_id: str,
        metadata: dict[str, Any],
    ) -> EpisodeEvidenceRecorder:
        attempt_path = finalizer.begin_attempt(
            episode_id=episode_id,
            attempt_id=attempt_id,
            metadata=metadata,
        )
        recorder = cls(
            attempt_path=attempt_path,
            finalizer=finalizer,
            episode_id=episode_id,
            attempt_id=attempt_id,
            episode_record=dict(metadata),
        )
        return recorder

    def _store_blob(self, name: str, payload: bytes) -> ContentAddressedBlob:
        digest = digest_bytes(payload)
        rel = f"blobs/{digest[:16]}_{name}"
        target = self.attempt_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(payload)
            os.replace(tmp, target)
            self._fsync_path(target)
        elif target.read_bytes() != payload:
            raise WriteOnceViolation(f"content-addressed blob collision with differing bytes: {rel}")
        blob = ContentAddressedBlob(relative_path=rel, sha256=digest, size_bytes=len(payload))
        if not any(existing.relative_path == blob.relative_path for existing in self.blobs):
            self.blobs.append(blob)
        return blob

    @staticmethod
    def _fsync_path(path: Path) -> None:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())

    def set_episode_fields(self, **fields: Any) -> None:
        self.episode_record.update(fields)

    def record_timing(self, timing: dict[str, Any]) -> None:
        self.episode_record["timing"] = dict(timing)

    def record_observation(
        self,
        *,
        observation_id: str,
        capture_time_s: float,
        payload: bytes,
        camera_ids: Iterable[str] = (),
        state_hash: str = "",
        native_input_present: bool = False,
    ) -> dict[str, Any]:
        if not payload:
            raise WriteOnceViolation("policy-input payload bytes are required")
        blob = self._store_blob(f"obs_{observation_id}.bin", payload)
        row = {
            "observation_id": observation_id,
            "capture_time_s": capture_time_s,
            "payload_uri": blob.relative_path,
            "payload_sha256": blob.sha256,
            "camera_ids": list(camera_ids),
            "state_hash": state_hash,
            "native_input_present": native_input_present,
        }
        self.observations.append(row)
        return row

    def record_request(self, row: dict[str, Any]) -> None:
        self.requests.append(dict(row))

    def record_future_artifact(
        self,
        *,
        request_id: str,
        kind: str,
        payload: bytes,
        payload_sha256: str = "",
    ) -> dict[str, Any]:
        if not payload:
            raise WriteOnceViolation("future artifact payload bytes are required")
        digest = payload_sha256 or digest_bytes(payload)
        blob = self._store_blob(f"future_{request_id}_{kind}.bin", payload)
        if blob.sha256 != digest:
            raise WriteOnceViolation("future artifact digest mismatch")
        row = {
            "request_id": request_id,
            "kind": kind,
            "payload_uri": blob.relative_path,
            "payload_sha256": blob.sha256,
        }
        self.future_artifacts.append(row)
        return row

    def record_viewport_frame(
        self,
        *,
        frame_index: int,
        sim_time_s: float,
        control_tick: int,
        fps: float,
        payload: bytes,
        payload_sha256: str = "",
        format_kind: str = "encoded_image",
        width: int = 0,
        height: int = 0,
        channels: int = 3,
    ) -> dict[str, Any]:
        blob = self._store_blob(f"viewport_f{frame_index:06d}.bin", payload)
        row = {
            "frame_index": frame_index,
            "sim_time_s": sim_time_s,
            "control_tick": control_tick,
            "fps": fps,
            "payload_uri": blob.relative_path,
            "payload_sha256": payload_sha256 or blob.sha256,
            "format_kind": format_kind,
            "width": width,
            "height": height,
            "channels": channels,
            "evidence_mode": "raw_blob",
        }
        self.viewport_frames.append(row)
        return row

    def record_viewport_frame_index(
        self,
        *,
        frame_index: int,
        sim_time_s: float,
        control_tick: int,
        fps: float,
        payload_sha256: str,
        format_kind: str,
        width: int,
        height: int,
        channels: int = 3,
    ) -> dict[str, Any]:
        row = {
            "frame_index": frame_index,
            "sim_time_s": sim_time_s,
            "control_tick": control_tick,
            "fps": fps,
            "payload_sha256": payload_sha256,
            "format_kind": format_kind,
            "width": width,
            "height": height,
            "channels": channels,
            "evidence_mode": "video_index",
        }
        self.viewport_frames.append(row)
        return row

    def record_viewport_video(self, artifact: EncodedVideoArtifact) -> dict[str, Any]:
        row = {
            "video_uri": artifact.relative_path,
            "video_sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
            "fps": artifact.fps,
            "frame_count": artifact.frame_count,
            "codec": artifact.codec,
        }
        self.episode_record["viewport_video"] = row
        return row

    def record_trajectory_row(self, row: dict[str, Any]) -> None:
        self.trajectory.append(dict(row))
        self._trajectory_rows_since_flush += 1

    def record_event(self, row: dict[str, Any]) -> None:
        self.events.append(dict(row))

    def should_flush_trajectory(self, *, interval: int) -> bool:
        return self._trajectory_rows_since_flush >= interval

    def flush_incremental(self, *, fsync: bool = True) -> None:
        if self._finalized:
            raise WriteOnceViolation("cannot flush after finalization")
        self.finalizer.write_incremental(self.attempt_path, "episode.json", self.episode_record)
        self.finalizer.write_incremental(
            self.attempt_path,
            "trajectory.json",
            {"schema_version": 1, "rows": self.trajectory},
        )
        self.finalizer.write_incremental(
            self.attempt_path,
            "requests.json",
            {"schema_version": 1, "rows": self.requests},
        )
        self.finalizer.write_incremental(
            self.attempt_path,
            "observations.json",
            {"schema_version": 1, "rows": self.observations},
        )
        self.finalizer.write_incremental(
            self.attempt_path,
            "events.json",
            {"schema_version": 1, "rows": self.events},
        )
        if self.future_artifacts:
            self.finalizer.write_incremental(
                self.attempt_path,
                "future_artifacts.json",
                {"schema_version": 1, "rows": self.future_artifacts},
            )
        if self.viewport_frames:
            self.finalizer.write_incremental(
                self.attempt_path,
                "viewport_frames.json",
                {"schema_version": 1, "rows": self.viewport_frames},
            )
        if fsync:
            for name in (
                "episode.json",
                "trajectory.json",
                "requests.json",
                "observations.json",
                "events.json",
                "future_artifacts.json",
                "viewport_frames.json",
            ):
                path = self.attempt_path / name
                if path.exists():
                    self._fsync_path(path)
        self._trajectory_rows_since_flush = 0

    def flush_partial(self, *, reason: str) -> None:
        """Persist incremental evidence without finalizing the attempt."""
        self.episode_record.setdefault("partial_flush_reasons", [])
        reasons = self.episode_record["partial_flush_reasons"]
        if isinstance(reasons, list):
            reasons.append(reason)
        self.flush_incremental(fsync=True)

    def finalize(self, *, terminal_receipt: dict[str, Any]) -> Path:
        if self._finalized:
            raise WriteOnceViolation("episode evidence already finalized")
        self.flush_incremental()
        manifest = {
            "episode_id": self.episode_id,
            "attempt_id": self.attempt_id,
            "episode_sha256": digest_bytes(canonical_json_bytes(self.episode_record)),
            "trajectory_sha256": digest_bytes(canonical_json_bytes(self.trajectory)),
            "requests_sha256": digest_bytes(canonical_json_bytes(self.requests)),
            "observations_sha256": digest_bytes(canonical_json_bytes(self.observations)),
            "events_sha256": digest_bytes(canonical_json_bytes(self.events)),
            "future_artifacts_sha256": digest_bytes(canonical_json_bytes(self.future_artifacts)),
            "viewport_frames_sha256": digest_bytes(canonical_json_bytes(self.viewport_frames)),
            "blob_count": len(self.blobs),
            "blobs": [blob.__dict__ for blob in self.blobs],
        }
        self.finalizer.write_incremental(self.attempt_path, "evidence_manifest.json", manifest)
        self._fsync_path(self.attempt_path / "evidence_manifest.json")
        receipt = {
            **terminal_receipt,
            "episode_id": self.episode_id,
            "attempt_id": self.attempt_id,
            "evidence_manifest_sha256": digest_bytes(canonical_json_bytes(manifest)),
        }
        marker = self.finalizer.finalize(self.attempt_path, receipt)
        self._finalized = True
        return marker
