import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compile_vla_wam_v2_dreamzero import (  # noqa: E402
    sha256,
    validate_checkpoint_payloads,
)


class DreamZeroCheckpointManifestTest(unittest.TestCase):
    def _selected(self, count: int, size: int) -> dict:
        return {
            "checkpoint": "GEAR-Dreams/DreamZero-DROID",
            "checkpoint_revision": "revision-under-test",
            "checkpoint_observed_file_count": count,
            "checkpoint_observed_payload_bytes": size,
        }

    def _manifest(self, path: Path, files: list[dict], size: int) -> None:
        path.write_text(json.dumps({
            "schema_version": "vla-wam-shared-v2-dreamzero-official-source-checkpoint-manifest-v1",
            "status": "verified",
            "checkpoint": {
                "repository": "GEAR-Dreams/DreamZero-DROID",
                "revision": "revision-under-test",
                "payload_file_count": len(files),
                "payload_bytes": size,
                "files": files,
            },
        }))

    def test_nested_manifest_resolves_only_under_explicit_payload_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            payload_root = base / "checkpoint"
            payload_root.mkdir()
            payload = payload_root / "model.safetensors"
            payload.write_bytes(b"exact payload")
            manifest = base / "committed_manifest.json"
            record = {
                "path": payload.name,
                "bytes": payload.stat().st_size,
                "sha256": sha256(payload),
            }
            self._manifest(manifest, [record], payload.stat().st_size)
            resolved_manifest, resolved_root = validate_checkpoint_payloads(
                {
                    "checkpoint_payload_manifest": str(manifest),
                    "checkpoint_payload_root": str(payload_root),
                },
                base,
                self._selected(1, payload.stat().st_size),
            )
            self.assertEqual(resolved_manifest, manifest.resolve())
            self.assertEqual(resolved_root, payload_root.resolve())

    def test_rejects_payload_parent_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            payload_root = base / "checkpoint"
            payload_root.mkdir()
            outside = base / "outside.bin"
            outside.write_bytes(b"outside")
            manifest = base / "committed_manifest.json"
            record = {
                "path": "../outside.bin",
                "bytes": outside.stat().st_size,
                "sha256": sha256(outside),
            }
            self._manifest(manifest, [record], outside.stat().st_size)
            with self.assertRaisesRegex(RuntimeError, "escapes explicit root"):
                validate_checkpoint_payloads(
                    {
                        "checkpoint_payload_manifest": str(manifest),
                        "checkpoint_payload_root": str(payload_root),
                    },
                    base,
                    self._selected(1, outside.stat().st_size),
                )


if __name__ == "__main__":
    unittest.main()
