"""Viewport capture attestation and ffmpeg/OpenCV video encoding for live DROID runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import subprocess
from pathlib import Path
from typing import Any

from experiments.online_correction_v4.adapters import EncodedVideoArtifact, ViewportFrame, ViewportVideoRequiredError
from experiments.online_correction_v4.recorder import digest_bytes


@dataclass(frozen=True)
class ViewportCapture:
    """Attested simulator viewport payload with an explicit wire format."""

    payload: bytes
    format_kind: str
    width: int
    height: int
    channels: int = 3

    @property
    def payload_sha256(self) -> str:
        return digest_bytes(self.payload)


def _require_cv2_np():
    try:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError as exc:  # pragma: no cover - cluster runtime owns deps
        raise ViewportVideoRequiredError(
            "OpenCV/numpy are required for viewport decode and encoding"
        ) from exc
    import cv2
    import numpy as np

    return cv2, np


def _blank_rejected(image: Any) -> None:
    _cv2, np = _require_cv2_np()
    if not np.ptp(image):
        raise ViewportVideoRequiredError("viewport frame is blank")


def attest_viewport_capture(
    payload: bytes,
    *,
    format_kind: str | None = None,
    width: int | None = None,
    height: int | None = None,
    channels: int | None = None,
) -> ViewportCapture:
    """Fail closed unless the payload format is explicitly attested."""
    if not payload:
        raise ViewportVideoRequiredError("empty viewport frame payload")
    if format_kind in {"raw_rgb24", "raw_bgr24"}:
        if width is None or height is None or width <= 0 or height <= 0:
            raise ViewportVideoRequiredError(
                f"{format_kind} viewport capture requires positive width and height metadata"
            )
        use_channels = int(channels or 3)
        if use_channels != 3:
            raise ViewportVideoRequiredError(f"{format_kind} viewport capture requires 3 channels")
        expected = width * height * use_channels
        if len(payload) != expected:
            raise ViewportVideoRequiredError(
                f"{format_kind} payload length {len(payload)} != expected {expected}"
            )
        if len(set(payload)) <= 1:
            raise ViewportVideoRequiredError("viewport frame is blank")
        try:
            _cv2, np = _require_cv2_np()
            array = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, use_channels)
            _blank_rejected(array)
        except ViewportVideoRequiredError as exc:
            if "OpenCV/numpy are required" not in str(exc):
                raise
        except Exception:
            pass
        return ViewportCapture(
            payload=payload,
            format_kind=format_kind,
            width=width,
            height=height,
            channels=use_channels,
        )
    if format_kind in {"encoded_image", "encoded_png", "encoded_jpeg"} or format_kind is None:
        cv2, np = _require_cv2_np()
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            if format_kind is not None and format_kind.startswith("encoded"):
                raise ViewportVideoRequiredError(
                    f"viewport bytes are not decodable as {format_kind}"
                )
        else:
            if image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
                raise ViewportVideoRequiredError("encoded viewport frame has invalid RGB shape")
            _blank_rejected(image)
            kind = format_kind if format_kind is not None else "encoded_image"
            return ViewportCapture(
                payload=payload,
                format_kind=kind,
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                channels=3,
            )
    if format_kind is None and width is not None and height is not None:
        return attest_viewport_capture(
            payload,
            format_kind="raw_rgb24",
            width=width,
            height=height,
            channels=channels,
        )
    raise ViewportVideoRequiredError(
        f"viewport format cannot be attested: format_kind={format_kind!r}, "
        f"width={width}, height={height}, payload_bytes={len(payload)}"
    )


def capture_from_ndarray(array: Any, *, format_kind: str = "raw_rgb24") -> ViewportCapture:
    """Attest a simulator-native RGB/BGR ndarray viewport."""
    _cv2, np = _require_cv2_np()
    image = np.asarray(array)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ViewportVideoRequiredError("viewport ndarray must be HxWx3")
    height, width = int(image.shape[0]), int(image.shape[1])
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    _blank_rejected(image)
    if format_kind == "raw_bgr24":
        payload = np.ascontiguousarray(image).tobytes()
    else:
        payload = np.ascontiguousarray(image).tobytes()
        if format_kind != "raw_rgb24":
            raise ViewportVideoRequiredError(f"unsupported ndarray viewport format: {format_kind}")
    return ViewportCapture(
        payload=payload,
        format_kind=format_kind,
        width=width,
        height=height,
        channels=3,
    )


def viewport_frame_from_capture(
    *,
    frame_index: int,
    sim_time_s: float,
    control_tick: int,
    capture: ViewportCapture,
) -> ViewportFrame:
    return ViewportFrame(
        frame_index=frame_index,
        sim_time_s=sim_time_s,
        control_tick=control_tick,
        payload=capture.payload,
        payload_sha256=capture.payload_sha256,
        format_kind=capture.format_kind,
        width=capture.width,
        height=capture.height,
        channels=capture.channels,
    )


def decode_viewport_capture(capture: ViewportCapture) -> Any:
    """Return a BGR uint8 image suitable for ffmpeg/OpenCV writers."""
    cv2, np = _require_cv2_np()
    if capture.format_kind in {"encoded_image", "encoded_png", "encoded_jpeg"}:
        image = cv2.imdecode(np.frombuffer(capture.payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ViewportVideoRequiredError("encoded viewport frame is not decodable")
        _blank_rejected(image)
        return image
    if capture.format_kind == "raw_bgr24":
        image = np.frombuffer(capture.payload, dtype=np.uint8).reshape(
            capture.height, capture.width, capture.channels
        )
        _blank_rejected(image)
        return np.ascontiguousarray(image)
    if capture.format_kind == "raw_rgb24":
        image = np.frombuffer(capture.payload, dtype=np.uint8).reshape(
            capture.height, capture.width, capture.channels
        )
        _blank_rejected(image)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    raise ViewportVideoRequiredError(f"unsupported attested viewport format: {capture.format_kind}")


def decode_viewport_frame(frame: ViewportFrame) -> Any:
    capture = ViewportCapture(
        payload=frame.payload,
        format_kind=frame.format_kind or "encoded_image",
        width=int(frame.width or 0),
        height=int(frame.height or 0),
        channels=int(frame.channels or 3),
    )
    if capture.format_kind == "encoded_image" and (capture.width <= 0 or capture.height <= 0):
        return decode_viewport_capture(attest_viewport_capture(frame.payload))
    return decode_viewport_capture(capture)


def _resolve_ffmpeg_bin() -> str:
    raw = os.environ.get("FFMPEG_BIN", "ffmpeg")
    path = Path(raw)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ViewportVideoRequiredError(f"FFMPEG_BIN must be an executable absolute path: {raw!r}")
    return str(path)


@dataclass
class FfmpegViewportVideoWriter:
    """Stream-attested BGR frames into ffmpeg (yuv420p/mp4), matching lane preflight."""

    fps: float
    relative_path: str = "viewport_video.mp4"
    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    _ffmpeg_bin: str = field(default="", repr=False)
    _attempt_path: Path | None = field(default=None, init=False, repr=False)
    _process: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)
    _frame_size: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _frame_count: int = field(default=0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self._ffmpeg_bin:
            object.__setattr__(self, "_ffmpeg_bin", _resolve_ffmpeg_bin())

    def bind_attempt_path(self, attempt_path: Path) -> None:
        if self._closed:
            raise ViewportVideoRequiredError("viewport writer already finalized")
        self._attempt_path = attempt_path

    def append_frame(self, frame: ViewportFrame) -> None:
        if self._closed:
            raise ViewportVideoRequiredError("viewport writer already finalized")
        if self._attempt_path is None:
            raise ViewportVideoRequiredError("attempt_path must be bound before viewport capture")
        image = decode_viewport_frame(frame)
        height, width = image.shape[:2]
        if self._frame_size is None:
            target = self._attempt_path / self.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            command = [
                self._ffmpeg_bin,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(float(self.fps)),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                self.codec,
                "-pix_fmt",
                self.pixel_format,
                "-y",
                str(target),
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if process.stdin is None:
                raise ViewportVideoRequiredError("ffmpeg viewport encoder stdin unavailable")
            self._process = process
            self._frame_size = (width, height)
        elif self._frame_size != (width, height):
            raise ViewportVideoRequiredError(
                f"viewport frame size changed from {self._frame_size} to {(width, height)}"
            )
        assert self._process is not None and self._process.stdin is not None
        try:
            self._process.stdin.write(image.tobytes())
            self._process.stdin.flush()
        except BrokenPipeError as exc:
            stderr = self._process.stderr.read().decode("utf-8", errors="replace") if self._process.stderr else ""
            raise ViewportVideoRequiredError(f"ffmpeg viewport encoder closed early: {stderr[-400:]}") from exc
        self._frame_count += 1

    def finalize_video(self, *, attempt_path: Path) -> EncodedVideoArtifact:
        if self._closed:
            raise ViewportVideoRequiredError("viewport writer already finalized")
        if self._frame_count == 0:
            raise ViewportVideoRequiredError("no viewport frames captured")
        if self._process is not None:
            if self._process.stdin is not None:
                self._process.stdin.close()
            returncode = self._process.wait()
            stderr = self._process.stderr.read().decode("utf-8", errors="replace") if self._process.stderr else ""
            self._process = None
            if returncode != 0:
                raise ViewportVideoRequiredError(
                    f"ffmpeg viewport encoder failed ({returncode}): {stderr[-400:]}"
                )
        target = attempt_path / self.relative_path
        if not target.is_file():
            raise ViewportVideoRequiredError("encoded viewport video path is missing")
        payload = target.read_bytes()
        if not payload:
            raise ViewportVideoRequiredError("encoded viewport video is empty")
        self._closed = True
        digest = digest_bytes(payload)
        return EncodedVideoArtifact(
            relative_path=self.relative_path,
            sha256=digest,
            size_bytes=len(payload),
            fps=float(self.fps),
            frame_count=self._frame_count,
            codec=f"ffmpeg/{self.codec}/{self.pixel_format}",
        )

    def close(self) -> None:
        if self._process is not None:
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except OSError:
                    pass
            if self._process.poll() is None:
                self._process.kill()
            self._process = None


@dataclass
class OpenCVViewportVideoWriter:
    """Fallback encoder when ffmpeg is unavailable in local/dev environments."""

    fps: float
    codec: str = "mp4v"
    relative_path: str = "viewport_video.mp4"
    _attempt_path: Path | None = field(default=None, init=False, repr=False)
    _writer: Any = field(default=None, init=False, repr=False)
    _frame_size: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _frame_count: int = field(default=0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def bind_attempt_path(self, attempt_path: Path) -> None:
        if self._closed:
            raise ViewportVideoRequiredError("viewport writer already finalized")
        self._attempt_path = attempt_path

    def append_frame(self, frame: ViewportFrame) -> None:
        if self._closed:
            raise ViewportVideoRequiredError("viewport writer already finalized")
        if self._attempt_path is None:
            raise ViewportVideoRequiredError("attempt_path must be bound before viewport capture")
        image = decode_viewport_frame(frame)
        height, width = image.shape[:2]
        if self._frame_size is None:
            self._frame_size = (width, height)
            cv2, _np = _require_cv2_np()
            target = self._attempt_path / self.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            writer = cv2.VideoWriter(str(target), fourcc, float(self.fps), (width, height))
            if not writer.isOpened():
                raise ViewportVideoRequiredError("failed to open viewport video writer")
            self._writer = writer
        elif self._frame_size != (width, height):
            raise ViewportVideoRequiredError(
                f"viewport frame size changed from {self._frame_size} to {(width, height)}"
            )
        if not self._writer.write(image):
            raise ViewportVideoRequiredError("viewport frame write failed")
        self._frame_count += 1

    def finalize_video(self, *, attempt_path: Path) -> EncodedVideoArtifact:
        if self._closed:
            raise ViewportVideoRequiredError("viewport writer already finalized")
        if self._frame_count == 0:
            raise ViewportVideoRequiredError("no viewport frames captured")
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        target = attempt_path / self.relative_path
        if not target.exists():
            raise ViewportVideoRequiredError("encoded viewport video path is missing")
        payload = target.read_bytes()
        if not payload:
            raise ViewportVideoRequiredError("encoded viewport video is empty")
        self._closed = True
        digest = digest_bytes(payload)
        return EncodedVideoArtifact(
            relative_path=self.relative_path,
            sha256=digest,
            size_bytes=len(payload),
            fps=float(self.fps),
            frame_count=self._frame_count,
            codec=self.codec,
        )

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def build_live_viewport_writer(*, fps: float) -> FfmpegViewportVideoWriter | OpenCVViewportVideoWriter:
    """Prefer ffmpeg to match k8s lane preflight; fall back to OpenCV locally."""
    try:
        return FfmpegViewportVideoWriter(fps=fps)
    except ViewportVideoRequiredError:
        return OpenCVViewportVideoWriter(fps=fps)
