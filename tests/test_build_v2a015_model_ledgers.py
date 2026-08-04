import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_v2a015_model_ledgers as ledgers  # noqa: E402


def setup_row(stage: str, suffix: str = "") -> dict[str, str]:
    return {
        "stage": stage,
        "error": f"error {stage} {suffix}".strip(),
        "cause": f"cause {stage} {suffix}".strip(),
        "effect": f"effect {stage} {suffix}; excluded from every denominator".strip(),
    }


def preflight_payload(rows: list[dict[str, str]]) -> dict:
    return {
        "schema_version": ledgers.PREFLIGHT_SCHEMA,
        "amendment_id": ledgers.AMENDMENT_ID,
        "setup_invalid_attempts": rows,
        "inference_accounting": {
            "setup_invalid_attempt_count": len(rows),
            "setup_invalid_attempts_in_behavioral_denominator": 0,
            "behavioral_denominator_count": 0,
        },
    }


def native_event(attempt: int, relation: str) -> dict:
    return {
        "id": f"native-dream-attempt{attempt:02d}-{relation}",
        "model_id": ledgers.DREAMZERO_MODEL_ID,
        "pair_id": "droid_pair_seed_8300",
        "environment_seed": 8300,
        "sampling_seed": 8300,
        "requested_relation": relation,
        "started_at_utc": f"2026-08-04T18:{attempt:02d}:00Z",
        "completed_at_utc": f"2026-08-04T18:{attempt:02d}:30Z",
        "status": "worker_exit_nonzero",
        "classification": "partial",
        "behavioral_result_valid": False,
        "wall_latency_valid": False,
        "worker_pid": 49000 + attempt,
        "worker_pgid": 49000 + attempt,
        "worker_exit_code": 1,
        "gpu_index": 0,
        "max_temperature_c": None,
        "events": [{"event": "worker_started"}],
        "raw_event_log": f"/pvc/dreamzero_seed8300_attempt{attempt:02d}.jsonl",
    }


class V2A015ModelLedgerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_json(self, name: str, value: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(value, indent=2) + "\n")
        return path

    def outputs(self) -> dict[str, Path]:
        return {
            "dreamzero_invalid_output": self.root / "dream-invalid.json",
            "cosmos_invalid_output": self.root / "cosmos-invalid.json",
            "dreamzero_runtime_output": self.root / "dream-runtime.json",
            "cosmos_runtime_output": self.root / "cosmos-runtime.json",
        }

    def build(self, preflight: Path, native: list[Path] | None = None, **values):
        arguments = {
            "preflight": preflight,
            "native_invalid_ledgers": native or [],
            **self.outputs(),
            "assert_dreamzero_no_runtime_interventions": True,
            "assert_cosmos_no_runtime_interventions": True,
        }
        arguments.update(values)
        return ledgers.build_ledgers(**arguments)

    def test_split_deduplicates_native_left_right_rows_and_writes_zero_ledgers(self):
        rows = [
            setup_row("source_patch_apply_check", "first real launch"),
            setup_row("source_patch_apply_check", "second real launch"),
            setup_row("bounded_loader_static_audit"),
            setup_row("dreamzero_action_s2_behavior_isaac_eula_gate"),
            setup_row("dreamzero_action_s2_behavior_vulkan_native_library_gate"),
            setup_row("dreamzero_action_s2_behavior_glvnd_and_warp_cache_gate"),
            setup_row("cosmos3_nano_g1_server_import"),
        ]
        preflight = self.write_json("preflight.json", preflight_payload(rows))
        events = [
            native_event(attempt, relation)
            for attempt in (2, 3, 4)
            for relation in ("left", "right")
        ]
        native = self.write_json(
            "dream-native-invalid.json",
            {"schema_version": ledgers.NATIVE_INVALID_SCHEMA, "events": events},
        )

        summary = self.build(preflight, [native])
        self.assertEqual(summary["dreamzero_setup_invalid_attempt_count"], 6)
        self.assertEqual(summary["cosmos_setup_invalid_attempt_count"], 1)
        self.assertEqual(summary["native_source_event_count"], 6)
        self.assertEqual(summary["native_deduplicated_launch_count"], 3)

        outputs = self.outputs()
        dream = json.loads(outputs["dreamzero_invalid_output"].read_text())
        cosmos = json.loads(outputs["cosmos_invalid_output"].read_text())
        self.assertEqual(dream["setup_invalid_attempt_count"], 6)
        self.assertEqual(dream["provenance"]["native_source_event_count"], 6)
        self.assertEqual(
            dream["provenance"]["deduplicated_native_launch_count"], 3
        )
        self.assertEqual(dream["provenance"]["corroborated_setup_attempt_count"], 3)
        self.assertEqual(
            dream["provenance"]["native_events_counted_as_additional_setup_attempts"],
            0,
        )
        self.assertEqual(cosmos["provenance"]["native_invalid_ledgers"], [])
        self.assertEqual(cosmos["provenance"]["native_source_event_count"], 0)
        self.assertEqual(
            sum(
                "corroborating_native_launch" in row["provenance"]
                for row in dream["attempts"]
            ),
            3,
        )
        for row in dream["attempts"] + cosmos["attempts"]:
            self.assertEqual(row["classification"], "setup_invalid")
            self.assertIs(row["behavioral_result_valid"], False)
            self.assertEqual(row["denominator_status"], "excluded")

        for key, model_id, arm_id in (
            (
                "dreamzero_runtime_output",
                ledgers.DREAMZERO_MODEL_ID,
                ledgers.DREAMZERO_ARM_ID,
            ),
            (
                "cosmos_runtime_output",
                ledgers.COSMOS_MODEL_ID,
                ledgers.COSMOS_ARM_ID,
            ),
        ):
            runtime = json.loads(outputs[key].read_text())
            self.assertEqual(runtime["model_id"], model_id)
            self.assertEqual(runtime["arm_id"], arm_id)
            self.assertEqual(runtime["events"], [])
            self.assertEqual(runtime["runtime_intervention_count"], 0)
            self.assertIs(
                runtime["zero_event_basis"]["explicit_caller_assertion"], True
            )
            self.assertIs(
                runtime["zero_event_basis"]["missing_file_inference_used"], False
            )

        before = {key: path.read_bytes() for key, path in outputs.items()}
        self.build(preflight, [native], overwrite=True)
        after = {key: path.read_bytes() for key, path in outputs.items()}
        self.assertEqual(before, after)

    def test_unknown_preflight_stage_fails_closed(self):
        preflight = self.write_json(
            "unknown.json", preflight_payload([setup_row("unowned_model_import")])
        )
        with self.assertRaisesRegex(RuntimeError, "Unrecognized V2-A015"):
            self.build(preflight)
        for path in self.outputs().values():
            self.assertFalse(path.exists())

    def test_duplicate_canonical_row_is_not_silently_deduplicated(self):
        row = setup_row("source_patch_apply_check")
        preflight = self.write_json(
            "duplicate.json", preflight_payload([row, dict(row)])
        )
        with self.assertRaisesRegex(RuntimeError, "Duplicate canonical"):
            self.build(preflight)

    def test_native_launch_requires_exactly_one_left_and_right_row(self):
        rows = [setup_row("dreamzero_action_s2_behavior_isaac_eula_gate")]
        preflight = self.write_json("preflight.json", preflight_payload(rows))
        native = self.write_json(
            "one-relation.json",
            {
                "schema_version": ledgers.NATIVE_INVALID_SCHEMA,
                "events": [native_event(2, "left")],
            },
        )
        with self.assertRaisesRegex(RuntimeError, "exactly one LEFT and one RIGHT"):
            self.build(preflight, [native])

    def test_unknown_native_attempt_does_not_invent_an_event(self):
        rows = [setup_row("dreamzero_action_s2_behavior_isaac_eula_gate")]
        preflight = self.write_json("preflight.json", preflight_payload(rows))
        native = self.write_json(
            "unknown-attempt.json",
            {
                "schema_version": ledgers.NATIVE_INVALID_SCHEMA,
                "events": [
                    native_event(9, "left"),
                    native_event(9, "right"),
                ],
            },
        )
        with self.assertRaisesRegex(RuntimeError, "No fail-closed preflight mapping"):
            self.build(preflight, [native])

    def test_zero_intervention_output_requires_explicit_assertions(self):
        preflight = self.write_json(
            "preflight.json",
            preflight_payload([setup_row("cosmos3_nano_g1_server_import")]),
        )
        with self.assertRaisesRegex(RuntimeError, "DreamZero zero interventions"):
            self.build(
                preflight,
                assert_dreamzero_no_runtime_interventions=False,
            )
        with self.assertRaisesRegex(RuntimeError, "Cosmos zero interventions"):
            self.build(
                preflight,
                assert_cosmos_no_runtime_interventions=False,
            )

    def test_completed_arm_can_compile_before_the_other_arm_runs(self):
        payload = preflight_payload(
            [
                setup_row("dreamzero_action_s1_server_import"),
                setup_row("cosmos3_nano_g1_server_import"),
            ]
        )
        # Valid DreamZero behaviors can already be in the denominator while
        # setup-invalid rows remain excluded.  This must not block the split.
        payload["inference_accounting"]["behavioral_denominator_count"] = 6
        preflight = self.write_json("partial-sequence.json", payload)
        dream_invalid = self.root / "only-dream-invalid.json"
        dream_runtime = self.root / "only-dream-runtime.json"
        summary = ledgers.build_ledgers(
            preflight=preflight,
            native_invalid_ledgers=[],
            dreamzero_invalid_output=dream_invalid,
            dreamzero_runtime_output=dream_runtime,
            assert_dreamzero_no_runtime_interventions=True,
        )
        self.assertEqual(summary["compiled_arms"], [ledgers.DREAMZERO_ARM_ID])
        self.assertTrue(dream_invalid.is_file())
        self.assertTrue(dream_runtime.is_file())
        self.assertFalse((self.root / "cosmos-invalid.json").exists())
        self.assertFalse((self.root / "cosmos-runtime.json").exists())

    def test_preflight_accounting_mismatch_is_rejected(self):
        payload = preflight_payload(
            [setup_row("cosmos3_nano_g1_server_import")]
        )
        payload["inference_accounting"]["setup_invalid_attempt_count"] = 2
        preflight = self.write_json("bad-count.json", payload)
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            self.build(preflight)


if __name__ == "__main__":
    unittest.main()
