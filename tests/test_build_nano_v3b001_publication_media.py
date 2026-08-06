import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_render_nano_v3b001_results import SyntheticEvidence  # noqa: E402
from tools import build_nano_v3b001_publication_media as media  # noqa: E402


def _find_ffmpeg() -> Path | None:
    candidates: list[Path] = []
    if os.environ.get("FFMPEG_BINARY"):
        candidates.append(Path(os.environ["FFMPEG_BINARY"]))
    if shutil.which("ffmpeg"):
        candidates.append(Path(shutil.which("ffmpeg") or ""))
    candidates.extend(
        [
            Path.home()
            / "LIBERO-plus/venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1",
            Path.home() / "Library/Caches/ms-playwright/ffmpeg-1011/ffmpeg-mac",
        ]
    )
    cache = Path.home() / ".cache/uv/archive-v0"
    if cache.is_dir():
        candidates.extend(sorted(cache.glob("*/imageio_ffmpeg/binaries/ffmpeg-*")))
    return next((path.resolve() for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)


def _find_font() -> Path | None:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


FFMPEG = _find_ffmpeg()
FONT = _find_font()
try:
    import PIL  # noqa: F401

    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_npy(path: Path, *, salt: int, shape: tuple[int, int, int, int] = (33, 6, 8, 3)) -> None:
    header = repr(
        {"descr": "|u1", "fortran_order": False, "shape": shape}
    ).encode("latin1")
    padding = 64 - ((10 + len(header) + 1) % 64)
    header += b" " * padding + b"\n"
    payload = bytearray()
    frame_bytes = shape[1] * shape[2] * shape[3]
    for frame in range(shape[0]):
        payload.extend(
            ((salt + frame * 7 + pixel) % 256 for pixel in range(frame_bytes))
        )
    path.write_bytes(
        b"\x93NUMPY"
        + bytes((1, 0))
        + struct.pack("<H", len(header))
        + header
        + bytes(payload)
    )


def _make_mp4(ffmpeg: Path, path: Path) -> None:
    completed = subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=5:duration=0.6",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)


class MediaEvidence:
    def __init__(self, root: Path, ffmpeg: Path) -> None:
        self.root = root
        self.fixture = SyntheticEvidence(root)
        self.actual: dict[str, Path] = {}
        self.predictions: dict[tuple[str, int], Path] = {}
        base_video = root / "tiny-source.mp4"
        _make_mp4(ffmpeg, base_video)
        rows = copy.deepcopy(self.fixture.rows)
        for row_index, row in enumerate(rows):
            if row["environment_seed"] != min(media.result_renderer.SEEDS):
                continue
            cell_id = str(row["registered_cell_id"])
            actual = root / f"{cell_id.replace(':', '_')}-actual.mp4"
            shutil.copyfile(base_video, actual)
            row["artifacts"]["viewport_video"] = _record(actual)
            self.actual[cell_id] = actual
            futures = []
            for request_index, action_step in enumerate((0, 16)):
                future = root / f"{cell_id.replace(':', '_')}-request-{request_index}.npy"
                _write_npy(future, salt=row_index * 19 + request_index * 37)
                futures.append(
                    {
                        "request_index": request_index,
                        "action_step_start": action_step,
                        "decoded_future": _record(future),
                        "decoded_future_shape": [33, 6, 8, 3],
                        "future_evidence_status": "exposed_and_retained",
                    }
                )
                self.predictions[(cell_id, request_index)] = future
            row["future_requests"] = futures
        self.fixture.rows = rows
        self.fixture._write_episodes(rows)
        self.fixture.rewrite_summary_binding()


@unittest.skipUnless(FFMPEG and FONT and HAS_PILLOW, "ffmpeg, Pillow, and a font are required")
class NanoV3B001PublicationMediaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.evidence = MediaEvidence(self.root, FFMPEG)  # type: ignore[arg-type]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self, name: str) -> dict[str, object]:
        return media.build_publication_media(
            summary_path=self.evidence.fixture.summary_path,
            episodes_path=self.evidence.fixture.episodes_path,
            output_directory=self.root / name,
            actual_rollout_assets=self.evidence.actual,
            decoded_prediction_assets=self.evidence.predictions,
            ffmpeg_path=FFMPEG,  # type: ignore[arg-type]
            font_file=FONT,  # type: ignore[arg-type]
        )

    def test_npy_and_selection_contract_retain_every_horizon(self) -> None:
        evidence, cells = media.collect_inputs(
            summary_path=self.evidence.fixture.summary_path,
            episodes_path=self.evidence.fixture.episodes_path,
            actual_rollout_assets=self.evidence.actual,
            decoded_prediction_assets=self.evidence.predictions,
        )
        self.assertEqual(evidence.selected_seed, 9400)
        self.assertEqual(len(cells), 4)
        self.assertTrue(all(len(cell.futures) == 2 for cell in cells))
        for cell in cells:
            self.assertEqual([future.request_index for future in cell.futures], [0, 1])
            self.assertEqual([future.action_step_start for future in cell.futures], [0, 16])
            self.assertTrue(all(future.array.shape == (33, 6, 8, 3) for future in cell.futures))

        source = next(iter(self.evidence.predictions.values()))
        parsed = media.inspect_npy_uint8_rgb(source)
        self.assertEqual(sum(1 for _ in media._iter_npy_rgb(parsed)), 33)

    def test_deterministic_complete_publication_build_and_overwrite_refusal(self) -> None:
        first = self._build("first")
        second = self._build("second")
        first_dir = Path(first["manifest"]).parent
        second_dir = Path(second["manifest"]).parent
        first_files = {path.name: path for path in first_dir.iterdir()}
        second_files = {path.name: path for path in second_dir.iterdir()}
        self.assertEqual(set(first_files), set(second_files))
        self.assertEqual(len(list(first_dir.glob("*.mp4"))), 4)
        self.assertEqual(len(list(first_dir.glob("*.png"))), 4)
        for name in first_files:
            self.assertEqual(first_files[name].read_bytes(), second_files[name].read_bytes(), name)

        manifest = json.loads(Path(first["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], media.MANIFEST_SCHEMA)
        self.assertEqual(manifest["selection"]["selected_seed"], 9400)
        self.assertFalse(manifest["selection"]["outcome_used_for_selection"])
        self.assertEqual(manifest["selection"]["selected_cell_count"], 4)
        self.assertEqual(
            manifest["media_semantics"]["visible_labels"],
            {
                "actual": media.ACTUAL_LABEL,
                "prediction": media.PREDICTION_LABEL,
                "continuity_notice": media.CONTINUITY_NOTICE,
            },
        )
        self.assertIn("one continuous", manifest["media_semantics"]["continuity_boundary"])

        for cell in manifest["selected_media"]:
            self.assertIn(cell["exact_prompt"], media.result_renderer.PROMPTS.values())
            self.assertEqual(len(cell["source_local_prediction_horizons_in_order"]), 2)
            self.assertEqual(
                [item["request_index"] for item in cell["source_local_prediction_horizons_in_order"]],
                [0, 1],
            )
            timeline = cell["timeline"]
            self.assertEqual(timeline["future_frames_written"], 66)
            self.assertEqual(timeline["separator_frames_written"], 16)
            self.assertEqual(timeline["prediction_timeline_frames"], 82)
            self.assertGreater(timeline["actual_held_frames"], 0)
            self.assertEqual(timeline["prediction_held_frames"], 0)
            self.assertEqual(timeline["output_frame_count"], 82)
            validation = cell["output_validation"]
            self.assertEqual(validation["codec_name"], "h264")
            self.assertEqual(validation["pixel_format"], "yuv420p")
            self.assertFalse(validation["has_audio"])
            self.assertEqual(validation["frame_count"], 82)
            self.assertEqual(len(validation["decoded_frame_samples"]), 3)
            self.assertLess(
                validation["faststart_atom_offsets"]["moov"],
                validation["faststart_atom_offsets"]["mdat"],
            )
            video = first_dir / cell["publication_video"]["path"]
            poster = first_dir / cell["poster"]["path"]
            self.assertEqual(cell["publication_video"]["sha256"], _sha256(video))
            self.assertEqual(cell["poster"]["sha256"], _sha256(poster))
            self.assertTrue(poster.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

        with self.assertRaisesRegex(media.NanoPublicationMediaError, "must be empty"):
            self._build("first")

    def test_missing_or_hash_mismatched_horizon_fails_before_output(self) -> None:
        incomplete = dict(self.evidence.predictions)
        incomplete.pop(next(iter(incomplete)))
        with self.assertRaisesRegex(media.NanoPublicationMediaError, "every selected exposed horizon"):
            media.build_publication_media(
                summary_path=self.evidence.fixture.summary_path,
                episodes_path=self.evidence.fixture.episodes_path,
                output_directory=self.root / "missing",
                actual_rollout_assets=self.evidence.actual,
                decoded_prediction_assets=incomplete,
                ffmpeg_path=FFMPEG,  # type: ignore[arg-type]
                font_file=FONT,  # type: ignore[arg-type]
            )
        self.assertFalse((self.root / "missing").exists())

        source = next(iter(self.evidence.predictions.values()))
        original = source.read_bytes()
        tampered = bytearray(original)
        tampered[-1] ^= 1
        source.write_bytes(tampered)
        with self.assertRaisesRegex(media.NanoPublicationMediaError, "SHA-256"):
            media.build_publication_media(
                summary_path=self.evidence.fixture.summary_path,
                episodes_path=self.evidence.fixture.episodes_path,
                output_directory=self.root / "tampered",
                actual_rollout_assets=self.evidence.actual,
                decoded_prediction_assets=self.evidence.predictions,
                ffmpeg_path=FFMPEG,  # type: ignore[arg-type]
                font_file=FONT,  # type: ignore[arg-type]
            )
        self.assertFalse((self.root / "tampered").exists())


class NpyParserFailureTests(unittest.TestCase):
    def test_trailing_or_wrong_shape_npy_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = root / "good.npy"
            _write_npy(good, salt=3)
            good.write_bytes(good.read_bytes() + b"trailing")
            with self.assertRaisesRegex(media.NanoPublicationMediaError, "payload length"):
                media.inspect_npy_uint8_rgb(good)

            wrong = root / "wrong.npy"
            _write_npy(wrong, salt=4, shape=(32, 6, 8, 3))
            with self.assertRaisesRegex(media.NanoPublicationMediaError, r"\[33,H,W,3\]"):
                media.inspect_npy_uint8_rgb(wrong)


if __name__ == "__main__":
    unittest.main()
