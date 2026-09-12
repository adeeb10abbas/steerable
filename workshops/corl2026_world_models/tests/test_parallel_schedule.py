"""Contract tests for unreleased four-condition experiment jobs."""

import copy
import csv
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


WORKSHOP = Path(__file__).resolve().parents[1]
MODULE = WORKSHOP / "experiments/forecast_layout/prepare_parallel_schedule.py"


def load_module():
    if not MODULE.exists():
        raise AssertionError("The schedule preparation implementation is missing")
    spec = importlib.util.spec_from_file_location("parallel_schedule", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.schedule = load_module()
        folder = WORKSHOP / "experiments/forecast_layout"
        self.spec = json.loads((folder / "ablation_spec.json").read_text())
        with (folder / "planned_cells.csv").open() as stream:
            self.rows = list(csv.DictReader(stream))

    def build(self, rows=None, spec=None):
        return self.schedule.build_schedule(
            self.spec if spec is None else spec,
            self.rows if rows is None else rows,
            "wmf_ablation_001_20260912",
        )

    def test_jobs_preserve_complete_factorial_without_releasing_or_allocating(self):
        result = self.build()
        jobs = result["jobs"]
        self.assertEqual(len(jobs), 58)
        self.assertEqual(len({cell for job in jobs for cell in job["ordered_cell_ids"]}), 232)
        for phase, count in [("pilot", 2), ("development", 8), ("confirmation", 48)]:
            selected = [job for job in jobs if job["phase"] == phase]
            self.assertEqual(len(selected), count)
            self.assertTrue(all(len(job["ordered_cell_ids"]) == 4 for job in selected))
        self.assertEqual({job["model_config"] for job in jobs}, {"N3", "D1"})
        self.assertFalse(result["launch_ready"])
        for job in jobs:
            self.assertFalse(job["released"])
            self.assertTrue(job["indivisible"])
            self.assertNotIn("worker", job)
            self.assertNotIn("command", job)

    def test_confirmation_uses_all_permutations_and_same_order_for_both_models(self):
        jobs = self.build()["jobs"]
        confirmation = [job for job in jobs if job["phase"] == "confirmation"]
        wanted = set(itertools.permutations([
            "original-left", "original-right", "reflected-left", "reflected-right"
        ]))
        for model in ("N3", "D1"):
            orders = [tuple(job["condition_order"]) for job in confirmation if job["model_config"] == model]
            self.assertEqual(len(orders), 24)
            self.assertEqual(set(orders), wanted)
        for block in {job["layout_pair_id"] for job in jobs}:
            pair = [job for job in jobs if job["layout_pair_id"] == block]
            self.assertEqual(pair[0]["condition_order"], pair[1]["condition_order"])

    def test_order_is_deterministic_and_independent_of_inventory_row_order(self):
        self.assertEqual(self.build(), self.build(list(reversed(self.rows))))
        self.assertEqual(self.build(), self.build())

    def test_namespace_hash_assignment_matches_fixed_reference_endpoints(self):
        result = self.build()
        order = result["order_assignment"]["confirmation_blocks_in_hash_order"]
        self.assertEqual(order[:3], ["C02", "C13", "C11"])
        self.assertEqual(order[-3:], ["C22", "C07", "C21"])
        for job in result["jobs"]:
            if job["layout_pair_id"] == "C02":
                self.assertEqual(job["condition_order"], [
                    "original-left", "original-right", "reflected-left", "reflected-right"
                ])
            if job["layout_pair_id"] == "C21":
                self.assertEqual(job["condition_order"], [
                    "reflected-right", "reflected-left", "original-right", "original-left"
                ])

    def test_cli_retains_source_hashes_is_repeatable_and_refuses_changed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(MODULE), "--output-dir", directory]
            subprocess.run(command, check=True, capture_output=True)
            path = Path(directory) / "parallel_schedule.json"
            first = path.read_bytes()
            result = json.loads(first)
            self.assertEqual(len(result["source_sha256"]), 3)
            self.assertEqual(result["upstream_pin"], self.spec["upstream_pin"])
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(path.read_bytes(), first)
            path.write_text("changed evidence\n")
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("refusing to replace", failed.stderr)
            self.assertEqual(path.read_text(), "changed evidence\n")

    def test_dependency_barriers_prevent_confirmation_before_development(self):
        jobs = self.build()["jobs"]
        pilot = {job["model_config"]: job["job_id"] for job in jobs if job["phase"] == "pilot"}
        development = {job["job_id"] for job in jobs if job["phase"] == "development"}
        for job in jobs:
            if job["phase"] == "pilot":
                self.assertEqual(job["depends_on_jobs"], [])
            elif job["phase"] == "development":
                self.assertEqual(job["depends_on_jobs"], [pilot[job["model_config"]]])
            else:
                self.assertEqual(set(job["depends_on_jobs"]), development)
                self.assertIn("post_development_confirmation_freeze", job["required_gates"])

    def test_duplicate_missing_and_unbalanced_conditions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.build(self.rows + [self.rows[0]])
        with self.assertRaisesRegex(ValueError, "232|missing"):
            self.build(self.rows[:-1])
        broken = copy.deepcopy(self.rows)
        broken[0]["layout_arm"] = "reflected"
        with self.assertRaisesRegex(ValueError, "factorial|condition|cell"):
            self.build(broken)
        broken = copy.deepcopy(self.rows)
        broken[8]["phase"] = "confirmation"
        with self.assertRaisesRegex(ValueError, "phase|stage|cell"):
            self.build(broken)

    def test_guidance_released_rows_and_changed_stop_rule_are_rejected(self):
        for field, value in [("model_config", "D2"), ("status", "RELEASED"),
                             ("stop_on_success", "true"), ("selected", "false"),
                             ("candidate_effective_policy_seed", "1")]:
            broken = copy.deepcopy(self.rows)
            broken[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.build(broken)
        changed_spec = copy.deepcopy(self.spec)
        changed_spec["stages"]["confirmation"]["layout_pairs"] = 12
        with self.assertRaises(ValueError):
            self.build(spec=changed_spec)


class SeedAuditTests(unittest.TestCase):
    def test_pinned_artifact_collision_audit_ignores_working_tree_and_discloses_scope(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "artifacts/cohort").mkdir(parents=True)
            (repo / "artifacts/cohort/rows.json").write_text(json.dumps({
                "episodes": [{"sampling_seed": 2026091000}, {"bytes": 2026091201}]
            }))
            (repo / "artifacts/cohort/rows.csv").write_text("policy_seed,value\n2026091101,3\n")
            (repo / "artifacts/cohort/other.jsonl").write_text('{"seeds": [2026091102]}\n')
            for command in (["git", "init", "-q"], ["git", "add", "artifacts"],
                            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                             "commit", "-qm", "fixture"]):
                subprocess.run(command, cwd=repo, check=True, capture_output=True)
            (repo / "artifacts/cohort/rows.json").write_text("{}")
            result = module.audit_candidate_seeds(repo, "HEAD", [2026091000, 2026091101, 2026091102, 2026091201, 2026091224])
            self.assertEqual({hit["seed"] for hit in result["seed_field_collisions"]},
                             {2026091000, 2026091101, 2026091102})
            self.assertEqual({hit["seed"] for hit in result["other_numeric_occurrences"]}, {2026091201})
            self.assertEqual(result["candidate_results"]["2026091224"], "not_found_in_scanned_seed_fields")
            self.assertEqual(result["artifact_file_count"], 3)
            self.assertFalse(result["proves_runtime_seed_compatibility"])
            self.assertFalse(result["full_collision_proof"])


if __name__ == "__main__":
    unittest.main()
