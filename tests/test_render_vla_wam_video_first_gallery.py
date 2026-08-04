import copy
import importlib.util
import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools/render_vla_wam_video_first_gallery.py"
SPEC = importlib.util.spec_from_file_location("video_first_gallery", MODULE_PATH)
gallery = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gallery)


def file_record(path: str) -> dict:
    target = REPO_ROOT / path
    return {"path": path, "bytes": target.stat().st_size, "sha256": gallery.sha256(target)}


def cfg_contracts() -> list[dict]:
    comparison_record = file_record(
        "artifacts/vla_wam_shared_v2/pilot/expansion/cfg_ablation_v2a015_comparison.json"
    )
    contracts = []
    for arm_id in reversed(gallery.CFG_ARM_ORDER):
        spec = gallery.CFG_ARM_SPECS[arm_id]
        contract = file_record(spec["media_manifest_path"])
        contract.update(
            {
                "schema_version": gallery.CFG_MEDIA_SCHEMA,
                "status": gallery.CFG_MEDIA_STATUS,
                "amendment_id": "V2-A015",
                "arm_id": arm_id,
                "model_id": spec["model_id"],
                "baseline_result": file_record(spec["baseline_result_path"]),
                "intervention_result": file_record(spec["intervention_result_path"]),
                "comparison": {
                    **comparison_record,
                    "schema_version": gallery.CFG_COMPARISON_SCHEMA,
                    "status": "complete",
                    "comparison_key": spec["comparison_key"],
                },
            }
        )
        contracts.append(contract)
    return contracts


class CfgGalleryTests(unittest.TestCase):
    def test_absent_contract_keeps_loader_optional(self) -> None:
        self.assertEqual(gallery.load_cfg_ablation_entries({}), [])

    def test_valid_contracts_render_complete_media_and_raw_metrics(self) -> None:
        entries = gallery.load_cfg_ablation_entries(
            {"cfg_ablation_media_contracts": cfg_contracts()}
        )
        self.assertEqual([entry["arm_id"] for entry in entries], list(gallery.CFG_ARM_ORDER))
        manifest = {
            "title": "Test",
            "display_policy": "Test policy.",
            "claim_boundary": "Test boundary.",
            "missing_publication_media": [],
        }
        rendered = gallery.render_html(
            manifest, [], True, [], [], True, [], entries
        )
        self.assertIn("ALL SIX COMPLETE ACTUAL EXECUTIONS", rendered)
        self.assertIn("Put the Rubik&#x27;s cube to the left of the bowl.", rendered)
        self.assertIn("6/6 → 4/6", rendered)
        self.assertIn("3/6 → 4/6", rendered)
        self.assertIn("n=3 per direction per setting", rendered)
        self.assertIn(
            "conditional-action equivalent s=1 → derived negative-branch action guidance s=2",
            rendered,
        )
        self.assertIn("Open complete imaginations", rendered)
        self.assertIn("Absolute mean-margin gap", rendered)
        self.assertIn("derived CFG-style negative-branch action guidance", rendered)
        self.assertIn("not a continuous imagined rollout", rendered)
        self.assertIn("both requested-side margins fell", rendered)
        self.assertIn("rather than establishing direction-independent robustness", rendered)
        self.assertIn(".cfg-media,.cfg-prompts,.cfg-metrics", rendered)
        self.assertLess(
            rendered.index("v2a015-cosmos3-nano-g1-complete-media"),
            rendered.index('<div class="grid">'),
        )
        markdown = gallery.render_markdown(
            manifest, [], True, [], [], True, [], entries
        )
        self.assertIn("Prompt asks LEFT", markdown)
        self.assertIn("Mean-margin gap (RIGHT − LEFT)", markdown)
        self.assertIn("Complete local-prediction composite", markdown)
        self.assertIn("Complete imagination composite", markdown)

    def test_contract_hash_mismatch_fails_closed(self) -> None:
        contracts = cfg_contracts()
        contracts[0]["sha256"] = "0" * 64
        with self.assertRaises(SystemExit):
            gallery.load_cfg_ablation_entries({"cfg_ablation_media_contracts": contracts})

    def test_incomplete_arm_set_fails_closed(self) -> None:
        with self.assertRaises(SystemExit):
            gallery.load_cfg_ablation_entries(
                {"cfg_ablation_media_contracts": cfg_contracts()[:1]}
            )

    def test_comparison_identity_mismatch_fails_closed(self) -> None:
        contracts = copy.deepcopy(cfg_contracts())
        contracts[0]["comparison"]["comparison_key"] = "not_the_bound_arm"
        with self.assertRaises(SystemExit):
            gallery.load_cfg_ablation_entries({"cfg_ablation_media_contracts": contracts})


if __name__ == "__main__":
    unittest.main()
