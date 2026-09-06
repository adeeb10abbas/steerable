from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.run_v4_second_stack_g4 import (
    SecondStackG4Error,
    _verify_backbone_cache,
)


class SecondStackG4Tests(unittest.TestCase):
    def test_backbone_cache_binds_revision_and_snapshot(self) -> None:
        revision = "9" * 40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backbone = root / "backbone"
            metadata = backbone / ".cache/huggingface/download"
            metadata.mkdir(parents=True)
            for filename in ("config.json", "model.safetensors"):
                (backbone / filename).write_bytes(filename.encode())
                (metadata / f"{filename}.metadata").write_text(
                    f"{revision}\ncontent-id\n0\n",
                    encoding="utf-8",
                )
            hf_home = root / "hf"
            cache = (
                hf_home
                / "hub"
                / "models--nvidia--Cosmos-Reason2-2B"
            )
            (cache / "refs").mkdir(parents=True)
            (cache / "snapshots").mkdir()
            (cache / "refs/main").write_text(revision, encoding="utf-8")
            (cache / f"snapshots/{revision}").symlink_to(backbone)

            receipt = _verify_backbone_cache(
                backbone_path=backbone,
                backbone_revision=revision,
                hf_home=hf_home,
            )

            self.assertEqual(receipt["revision"], revision)
            self.assertIn("model.safetensors", receipt["content_manifest"])

    def test_backbone_cache_rejects_different_metadata_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backbone = root / "backbone"
            metadata = backbone / ".cache/huggingface/download"
            metadata.mkdir(parents=True)
            (metadata / "config.json.metadata").write_text(
                f"{'a' * 40}\ncontent-id\n0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                SecondStackG4Error,
                "revision differs",
            ):
                _verify_backbone_cache(
                    backbone_path=backbone,
                    backbone_revision="b" * 40,
                    hf_home=root / "hf",
                )


if __name__ == "__main__":
    unittest.main()
