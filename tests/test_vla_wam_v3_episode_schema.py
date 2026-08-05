from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from vla_wam_v3_episode_schema import (  # noqa: E402
    BEHAVIORAL_SCHEMA_VERSION,
    INFRASTRUCTURE_SCHEMA_VERSION,
    EpisodeSchemaError,
    derive_frozen_failure_stage,
    derive_initial_state_sha256,
    encode_jsonl_record,
    validate_behavioral_record,
    validate_infrastructure_record,
    write_jsonl,
)


def behavioral_record(*, arena: str = "droid_robolab", relation: str = "left", success: bool = False) -> dict:
    """A six-action trace; caller sets regions/taxonomy for the desired outcome."""

    steps = []
    for action_step in range(7):
        lateral = 0.0
        forward = 0.1
        if arena == "robotwin_place_a2b":
            step = {
                "action_step": action_step,
                "object_xyz": [0.01 * action_step, 0.0, 0.70],
                "reference_xyz": [0.0, 0.0, 0.70],
                "grippers_open": True,
                "requested_region": False,
                "opposite_region": False,
            }
        else:
            step = {
                "action_step": action_step,
                "object_xyz": [0.01 * action_step, 0.0, 0.70],
                "reference_xyz": [0.0, 0.0, 0.70],
                "grippers_open": True,
            }
        steps.append(step)
    record = {
        "schema_version": BEHAVIORAL_SCHEMA_VERSION,
        "record_type": "behavioral_episode",
        "behavioral_result_valid": True,
        "arena": arena,
        "requested_relation": relation,
        "requested_success": success,
        "final_detached_release": success,
        "failure_stage": "object_moved_no_verified_pickup",
        "frozen_failure_stage": "object_moved_no_verified_pickup",
        "failure_taxonomy": "pick_failed",
        "study_id": "vla_wam_language_steerability_v3",
        "registered_cell_id": "v3-droid-pair-8300-left",
        "attempt_id": "v3-droid-pair-8300-left-attempt-01",
        "model_id": "synthetic_policy",
        "checkpoint": {"id": "synthetic/checkpoint", "revision": "deadbeef"},
        "runtime_identity": {"id": "synthetic-runtime", "sha256": "a" * 64},
        "pair_id": "droid_pair_seed_8300",
        "environment_seed": 8300,
        "policy_seed": 8300,
        "prompt": "Put the Rubik's cube to the left of the bowl.",
        "prompt_family": "direct_command",
        "predicate_id": "frozen_relation_release_v3",
        "reset_id": "neutral_reset_seed_8300",
        "measurement_frame": "robot_base_object_minus_reference_xyz_m",
        "measurement_frame_description": (
            "Object and reference XYZ samples are expressed in the frozen robot-base frame; "
            "forward is object-minus-reference x and lateral is object-minus-reference y, "
            "with positive lateral denoting robot LEFT."
        ),
        "artifacts": {
            "raw_result_jsonl": {
                "path": "raw/episode_results.jsonl",
                "integrity_scope": "batch_manifest_after_close",
            },
            **{
                key: {"path": f"raw/{key}.bin", "sha256": "b" * 64, "bytes": 1}
                for key in ("executed_action_trace", "viewport_video")
            },
        },
        "actions_executed": 6,
        "action_cap": 6,
        "right_censored": not success,
        "wall_time_s": 1.25,
        "operational_wall_time_valid": True,
        "first_contact_step": None,
        "first_contact_unavailable_reason": "contact instrumentation was not enabled",
        "event_timeline": [
            {"event": "episode_start", "action_step": 0},
            {"event": "episode_end", "action_step": 6},
        ],
        "steps": steps,
    }
    record["initial_state_sha256"] = derive_initial_state_sha256(record)
    return record


def set_droid_region(record: dict, indices: range, *, requested: bool) -> None:
    # DROID robot-frame LEFT cone: positive y with negligible forward x.
    for index in indices:
        record["steps"][index]["object_xyz"][1] = 0.10 if requested else -0.10
        record["steps"][index]["object_xyz"][0] = 0.0


def set_robotwin_region(record: dict, indices: range, *, requested: bool) -> None:
    for index in indices:
        record["steps"][index]["requested_region"] = requested
        record["steps"][index]["opposite_region"] = not requested
        record["steps"][index]["object_xyz"][1] = 0.10 if requested else -0.10


def set_pickup(record: dict, indices: range = range(2, 5)) -> None:
    for index in indices:
        record["steps"][index]["object_xyz"][2] = 0.74


def add_derived_events(record: dict) -> None:
    """Populate the required ordered event evidence after a synthetic mutation."""

    events = [{"event": "episode_start", "action_step": 0}]
    if any(step.get("contact_detected") for step in record["steps"]):
        events.append({"event": "first_contact", "action_step": record["first_contact_step"]})
    lifted = [step["object_xyz"][2] >= 0.73 for step in record["steps"]]
    for index in range(len(lifted) - 2):
        if all(lifted[index:index + 3]):
            events.append({"event": "verified_pickup", "action_step": index})
            break
    if record["arena"] == "robotwin_place_a2b":
        for name, field in (("requested_region_entry", "requested_region"), ("opposite_region_entry", "opposite_region")):
            first = next((index for index, step in enumerate(record["steps"]) if step[field]), None)
            if first is not None:
                events.append({"event": name, "action_step": first})
    else:
        for name, requested in (("requested_region_entry", True), ("opposite_region_entry", False)):
            first = next(
                (index for index, step in enumerate(record["steps"])
                 if (
                     ((step["object_xyz"][1] - step["reference_xyz"][1]) if requested
                      else (step["reference_xyz"][1] - step["object_xyz"][1]))
                     / max(1e-9, ((step["object_xyz"][0] - step["reference_xyz"][0]) ** 2
                                  + (step["object_xyz"][1] - step["reference_xyz"][1]) ** 2) ** 0.5)
                     >= 2**-0.5
                 )),
                None,
            )
            if first is not None:
                events.append({"event": name, "action_step": first})
    events.sort(key=lambda event: event["action_step"])
    events.append({"event": "episode_end", "action_step": record["actions_executed"]})
    record["event_timeline"] = events
    stage = derive_frozen_failure_stage(record)
    record["failure_stage"] = stage
    record["frozen_failure_stage"] = stage
    record["initial_state_sha256"] = derive_initial_state_sha256(record)


class V3EpisodeSchemaTest(unittest.TestCase):
    def test_all_five_frozen_outcomes(self) -> None:
        correct = behavioral_record(success=True)
        set_pickup(correct)
        set_droid_region(correct, range(4, 7), requested=True)
        correct["failure_taxonomy"] = "correct"

        wrong_side = behavioral_record()
        set_pickup(wrong_side)
        set_droid_region(wrong_side, range(4, 7), requested=False)
        wrong_side["failure_taxonomy"] = "wrong_side"

        release_failed = behavioral_record()
        set_pickup(release_failed)
        set_droid_region(release_failed, range(4, 7), requested=True)
        release_failed["failure_taxonomy"] = "release_failed"

        pick_failed = behavioral_record()
        pick_failed["failure_taxonomy"] = "pick_failed"

        transport_failed = behavioral_record()
        set_pickup(transport_failed)
        transport_failed["failure_taxonomy"] = "transport_failed"

        for record, expected in (
            (correct, "correct"), (wrong_side, "wrong_side"),
            (release_failed, "release_failed"), (pick_failed, "pick_failed"),
            (transport_failed, "transport_failed"),
        ):
            add_derived_events(record)
            validated = validate_behavioral_record(record)
            self.assertEqual(validated["failure_taxonomy"], expected)

    def test_transient_entry_and_path_length_are_preserved(self) -> None:
        record = behavioral_record(arena="robotwin_place_a2b")
        set_pickup(record)
        set_robotwin_region(record, range(2, 3), requested=True)
        record["failure_taxonomy"] = "transport_failed"
        add_derived_events(record)
        validated = validate_behavioral_record(record)
        measurements = validated["measurements"]
        self.assertEqual(measurements["region_kind"], "robotwin_native_relation_region")
        self.assertEqual(measurements["requested_entry_kind"], "transient")
        self.assertEqual(measurements["first_requested_entry_step"], 2)
        self.assertIsNone(measurements["first_sustained_requested_entry_step"])
        self.assertAlmostEqual(measurements["object_path_length_m"], 0.2798963507)
        self.assertEqual(measurements["episode_length_steps"], 6)
        self.assertEqual(measurements["scored_state_sample_count"], 7)

    def test_null_contact_requires_reason_and_contact_stream_must_match(self) -> None:
        record = behavioral_record()
        self.assertEqual(
            validate_behavioral_record(record)["measurements"]["first_contact_status"],
            "instrumentation_unavailable",
        )
        with self.assertRaisesRegex(EpisodeSchemaError, "requires a non-empty"):
            bad = copy.deepcopy(record)
            bad["first_contact_unavailable_reason"] = ""
            validate_behavioral_record(bad)

        contact = behavioral_record()
        for step in contact["steps"]:
            step["contact_detected"] = step["action_step"] == 3
        contact["first_contact_step"] = 3
        contact["first_contact_unavailable_reason"] = None
        add_derived_events(contact)
        validated = validate_behavioral_record(contact)
        self.assertEqual(validated["measurements"]["first_contact_step"], 3)
        self.assertEqual(validated["measurements"]["first_contact_status"], "observed")

        not_observed = behavioral_record()
        for step in not_observed["steps"]:
            step["contact_detected"] = False
        not_observed["first_contact_unavailable_reason"] = None
        validated = validate_behavioral_record(not_observed)
        self.assertEqual(validated["measurements"]["first_contact_status"], "not_observed")

    def test_initial_state_and_legacy_stage_are_derived_not_trusted(self) -> None:
        record = behavioral_record()
        tampered_state = copy.deepcopy(record)
        tampered_state["initial_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(EpisodeSchemaError, "retained initial physical state"):
            validate_behavioral_record(tampered_state)

        tampered_stage = copy.deepcopy(record)
        tampered_stage["failure_stage"] = "success"
        tampered_stage["frozen_failure_stage"] = "success"
        with self.assertRaisesRegex(EpisodeSchemaError, "v2 arena classifier"):
            validate_behavioral_record(tampered_stage)

    def test_infrastructure_attempts_are_excluded_from_behavioral_taxonomy(self) -> None:
        infra = {
            "schema_version": INFRASTRUCTURE_SCHEMA_VERSION,
            "record_type": "infrastructure_attempt",
            "behavioral_result_valid": False,
            "classification": "partial",
            "arena": "droid_robolab",
            "study_id": "vla_wam_language_steerability_v3",
            "registered_cell_id": "v3-droid-pair-8300-left",
            "attempt_id": "seed8300-left-stall-01",
            "model_id": "synthetic_policy",
            "checkpoint": {"id": "synthetic/checkpoint", "revision": "deadbeef"},
            "runtime_identity": {"id": "synthetic-runtime", "sha256": "a" * 64},
            "pair_id": "droid_pair_seed_8300",
            "environment_seed": 8300,
            "policy_seed": 8300,
            "prompt": "Put the Rubik's cube to the left of the bowl.",
            "prompt_family": "direct_command",
            "predicate_id": "frozen_relation_release_v3",
            "reset_id": "neutral_reset_seed_8300",
            "measurement_frame": "robot_base_object_minus_reference_xyz_m",
            "measurement_frame_description": (
                "Object and reference XYZ samples are expressed in the frozen robot-base frame; "
                "forward is object-minus-reference x and lateral is object-minus-reference y, "
                "with positive lateral denoting robot LEFT."
            ),
            "artifacts": {
                "raw_result_jsonl": {
                    "path": "raw/episode_results.jsonl",
                    "integrity_scope": "batch_manifest_after_close",
                },
            },
            "stage": "policy_server",
            "error": "worker exited before behavioral completion",
            "log_hash": "c" * 64,
            "runtime_intervention": False,
            "repair_attempt_id": None,
            "event_timeline": [{"sequence": 0, "stage": "launch"}, {"sequence": 1, "stage": "partial"}],
        }
        self.assertEqual(validate_infrastructure_record(infra)["attempt_id"], infra["attempt_id"])
        malformed_optional_artifact = copy.deepcopy(infra)
        malformed_optional_artifact["artifacts"]["viewport_video"] = {
            "path": "raw/partial.mp4",
            "sha256": "not-a-sha256",
            "bytes": 1,
        }
        with self.assertRaisesRegex(EpisodeSchemaError, "lowercase SHA-256"):
            validate_infrastructure_record(malformed_optional_artifact)
        empty_optional_artifact = copy.deepcopy(infra)
        empty_optional_artifact["artifacts"]["viewport_video"] = {
            "path": "raw/partial.mp4",
            "sha256": "d" * 64,
            "bytes": 0,
        }
        with self.assertRaisesRegex(EpisodeSchemaError, ">= 1"):
            validate_infrastructure_record(empty_optional_artifact)
        infra["failure_taxonomy"] = "pick_failed"
        with self.assertRaisesRegex(EpisodeSchemaError, "must not carry"):
            validate_infrastructure_record(infra)

    def test_rejects_nan_and_taxonomy_mismatch(self) -> None:
        record = behavioral_record()
        record["steps"][0]["object_xyz"][1] = float("nan")
        with self.assertRaisesRegex(EpisodeSchemaError, "finite"):
            validate_behavioral_record(record)

        mismatch = behavioral_record()
        mismatch["failure_taxonomy"] = "transport_failed"
        add_derived_events(mismatch)
        with self.assertRaisesRegex(EpisodeSchemaError, "disagrees"):
            validate_behavioral_record(mismatch)

    def test_requires_provenance_and_orders_events(self) -> None:
        missing_provenance = behavioral_record()
        del missing_provenance["checkpoint"]
        with self.assertRaisesRegex(EpisodeSchemaError, "checkpoint"):
            validate_behavioral_record(missing_provenance)

        record = behavioral_record()
        set_pickup(record)
        add_derived_events(record)
        record["event_timeline"] = [
            {"event": "episode_start", "action_step": 0},
            {"event": "episode_end", "action_step": 6},
            {"event": "verified_pickup", "action_step": 2},
        ]
        with self.assertRaisesRegex(EpisodeSchemaError, "nondecreasing"):
            validate_behavioral_record(record)

    def test_droid_is_named_cone_and_jsonl_is_deterministic(self) -> None:
        record = behavioral_record()
        set_droid_region(record, range(4, 7), requested=True)
        record["failure_taxonomy"] = "pick_failed"  # pickup failure precedes side classification.
        add_derived_events(record)
        encoded = encode_jsonl_record(record)
        self.assertTrue(encoded.endswith("\n"))
        decoded = json.loads(encoded)
        self.assertEqual(decoded["measurements"]["region_kind"], "droid_45_degree_cone")
        self.assertEqual(decoded["measurements"]["requested_final_lateral_margin_m"], 0.1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episodes.jsonl"
            manifest = write_jsonl(path, [record])
            self.assertEqual(path.read_text(), encoded)
            manifest_path = path.with_name(path.name + ".manifest.json")
            self.assertEqual(json.loads(manifest_path.read_text()), manifest)
            self.assertEqual(manifest["row_count"], 1)
            self.assertEqual(manifest["jsonl_bytes"], len(encoded.encode()))
            self.assertEqual(len(manifest["jsonl_sha256"]), 64)
            with self.assertRaisesRegex(EpisodeSchemaError, "refusing to overwrite"):
                write_jsonl(path, [record])

    def test_pick_failure_precedes_wrong_side_and_rejects_spoofed_offsets(self) -> None:
        record = behavioral_record()
        set_droid_region(record, range(0, 7), requested=False)
        record["failure_taxonomy"] = "pick_failed"
        add_derived_events(record)
        self.assertEqual(validate_behavioral_record(record)["failure_taxonomy"], "pick_failed")

        spoofed = behavioral_record()
        spoofed["steps"][0]["lateral_offset_m"] = -99.0
        with self.assertRaisesRegex(EpisodeSchemaError, "derived from robot-base XYZ"):
            validate_behavioral_record(spoofed)

        self_hash = behavioral_record()
        self_hash["artifacts"]["raw_result_jsonl"]["sha256"] = "d" * 64
        with self.assertRaisesRegex(EpisodeSchemaError, "inline self-hash"):
            validate_behavioral_record(self_hash)

    def test_detached_release_is_a_separate_scorer_predicate(self) -> None:
        release_failed = behavioral_record()
        set_pickup(release_failed)
        set_droid_region(release_failed, range(4, 7), requested=True)
        release_failed["final_detached_release"] = False
        release_failed["failure_taxonomy"] = "release_failed"
        add_derived_events(release_failed)
        validated = validate_behavioral_record(release_failed)
        self.assertFalse(validated["measurements"]["final_detached_release"])

        inconsistent = copy.deepcopy(release_failed)
        inconsistent["final_detached_release"] = True
        with self.assertRaisesRegex(EpisodeSchemaError, "scorer inconsistency"):
            validate_behavioral_record(inconsistent)

        success_without_detach = behavioral_record(success=True)
        success_without_detach["final_detached_release"] = False
        with self.assertRaisesRegex(EpisodeSchemaError, "requested_success requires"):
            validate_behavioral_record(success_without_detach)


if __name__ == "__main__":
    unittest.main()
