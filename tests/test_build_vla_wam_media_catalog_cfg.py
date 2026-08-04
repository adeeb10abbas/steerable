import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_vla_wam_media_catalog as catalog  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


class CFGAblationMediaCatalogTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.media = self.root / "media"
        self.pilot = self.root / "pilot"
        self.cfg = self.media / "cfg_v2a015"
        self.media.mkdir(parents=True)
        self.pilot.mkdir(parents=True)
        self.gallery = self.media / "video_first_gallery_manifest.json"
        self.originals = {
            "REPO_ROOT": catalog.REPO_ROOT,
            "MEDIA_ROOT": catalog.MEDIA_ROOT,
            "PILOT_MEDIA_ROOT": catalog.PILOT_MEDIA_ROOT,
            "GALLERY_MANIFEST": catalog.GALLERY_MANIFEST,
            "CFG_MEDIA_ROOT": catalog.CFG_MEDIA_ROOT,
        }
        catalog.REPO_ROOT = self.root
        catalog.MEDIA_ROOT = self.media
        catalog.PILOT_MEDIA_ROOT = self.pilot
        catalog.GALLERY_MANIFEST = self.gallery
        catalog.CFG_MEDIA_ROOT = self.cfg

        self.write_json("behavioral.json", {"gallery_entries": []})
        self.write_json(
            "imagination.json", {"gallery_entries": [], "official_decodes": []}
        )

    def tearDown(self):
        for key, value in self.originals.items():
            setattr(catalog, key, value)
        self.temporary.cleanup()

    def write_json(self, name: str, value: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(value, indent=2) + "\n")
        return path

    def gallery_payload(self, contracts=None) -> dict:
        payload = {
            "entries": [],
            "dreamzero_manifest_contract": {"path": "behavioral.json"},
            "additional_manifest_contracts": [],
            "dreamzero_imagination_manifest_contract": {
                "path": "imagination.json"
            },
            "prediction_only_manifest_contracts": [],
        }
        if contracts is not None:
            payload["cfg_ablation_media_contracts"] = contracts
        return payload

    def make_cfg_manifest(self, arm_id: str) -> Path:
        metadata = catalog.CFG_ARM_METADATA[arm_id]
        arm_dir = self.cfg / arm_id
        arm_dir.mkdir(parents=True)
        outputs = {}
        for key, suffix in (
            ("actual_video", ".mp4"),
            ("actual_poster", ".jpg"),
            ("prediction_or_imagination_video", ".mp4"),
            ("prediction_or_imagination_poster", ".jpg"),
        ):
            path = arm_dir / f"{key}{suffix}"
            path.write_bytes(f"{arm_id}-{key}".encode())
            outputs[key] = record(path, self.root)
        manifest = {
            "schema_version": "vla-wam-shared-v2-v2a015-cfg-media-v1",
            "status": "complete_all_six_cells_actual_and_prediction_media",
            "amendment_id": "V2-A015",
            "arm_id": arm_id,
            "model_id": metadata["model_id"],
            "outputs": outputs,
        }
        path = arm_dir / "media_manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return path

    def test_absent_contract_key_preserves_old_scan_and_ignores_cfg_folder(self):
        ordinary = self.media / "ordinary.mp4"
        ordinary.write_bytes(b"ordinary-video")
        uncontracted = self.cfg / "uncontracted" / "new.mp4"
        uncontracted.parent.mkdir(parents=True)
        uncontracted.write_bytes(b"must-not-leak")
        self.gallery.write_text(json.dumps(self.gallery_payload()) + "\n")

        rows, manifests = catalog.build_rows()
        self.assertEqual([row["path"] for row in rows], ["media/ordinary.mp4"])
        self.assertNotIn("media/cfg_v2a015/uncontracted/new.mp4", [row["path"] for row in rows])
        self.assertEqual(
            manifests,
            ["behavioral.json", "imagination.json", "media/video_first_gallery_manifest.json"],
        )

    def test_complete_contract_adds_two_videos_per_arm_and_not_posters(self):
        dream = self.make_cfg_manifest("dreamzero_action_cfg_s2")
        cosmos = self.make_cfg_manifest("cosmos3_nano_no_cfg_g1")
        contracts = [
            {"path": str(dream.relative_to(self.root))},
            {"path": str(cosmos.relative_to(self.root))},
        ]
        self.gallery.write_text(json.dumps(self.gallery_payload(contracts)) + "\n")

        rows, manifests = catalog.build_rows()
        self.assertEqual(len(rows), 4)
        by_role = {row["publication_role"]: row for row in rows}
        self.assertIn("CFG ablation complete actual execution", by_role)
        self.assertIn("CFG ablation complete imagination", by_role)
        self.assertIn("CFG ablation complete local prediction", by_role)
        self.assertTrue(all(Path(row["path"]).suffix == ".mp4" for row in rows))
        for row in rows:
            self.assertEqual(row["arena"], "DROID / RoboLab")
            self.assertEqual(row["model_class"], "WAM")
            self.assertNotIn("episode", row["publication_role"].lower())
        self.assertIn(str(dream.relative_to(self.root)), manifests)
        self.assertIn(str(cosmos.relative_to(self.root)), manifests)

    def test_output_hash_or_byte_mismatch_fails_closed(self):
        dream = self.make_cfg_manifest("dreamzero_action_cfg_s2")
        payload = json.loads(dream.read_text())
        # Posters remain required, hash-bearing contract outputs even though the
        # MP4-only catalog does not emit rows for them.
        payload["outputs"]["actual_poster"]["bytes"] += 1
        dream.write_text(json.dumps(payload, indent=2) + "\n")
        self.gallery.write_text(
            json.dumps(
                self.gallery_payload([{"path": str(dream.relative_to(self.root))}])
            )
            + "\n"
        )
        with self.assertRaisesRegex(ValueError, "path/hash/bytes"):
            catalog.build_rows()

    def test_written_catalog_explicitly_contributes_zero_episodes(self):
        dream = self.make_cfg_manifest("dreamzero_action_cfg_s2")
        self.gallery.write_text(
            json.dumps(
                self.gallery_payload([{"path": str(dream.relative_to(self.root))}])
            )
            + "\n"
        )
        rows, manifests = catalog.build_rows()
        catalog.write_outputs(rows, manifests)
        output = json.loads((self.media / "media_catalog.json").read_text())
        self.assertEqual(output["video_count"], 2)
        self.assertEqual(len(output["videos"]), 2)
        self.assertTrue(all(item["path"].endswith(".mp4") for item in output["videos"]))
        self.assertEqual(output["cfg_ablation_video_count"], 2)
        self.assertEqual(output["cfg_ablation_episode_count_contribution"], 0)
        self.assertNotIn("poster_count", output)
        readme = (self.media / "README.md").read_text()
        self.assertIn("Catalog total: **2 committed MP4s**", readme)


if __name__ == "__main__":
    unittest.main()
