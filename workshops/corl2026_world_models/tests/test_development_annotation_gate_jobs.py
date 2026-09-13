import hashlib
import importlib.util
import json
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest import mock


WORKSHOP = Path(__file__).resolve().parents[1]
MODULE_PATH = WORKSHOP / "experiments/forecast_layout/development_annotation_gate_jobs.py"
WORKFLOW_TEST_PATH = Path(__file__).with_name("test_forecast_annotation_workflow.py")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gate = load_module(MODULE_PATH, "development_annotation_gate_jobs_under_test")
workflow_test = load_module(WORKFLOW_TEST_PATH, "annotation_workflow_fixture_for_gate")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class DevelopmentAnnotationGateJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow_test.AnnotationWorkflowTests.setUpClass()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = workflow_test.AnnotationWorkflowTests(methodName="runTest")
        self.fixture.setUp()
        selection, selection_path = self.fixture.make_selection()
        image_inventory, image_path = self.fixture.make_images(selection, selection_path)
        self.selection_path = selection_path
        self.image_inventory = image_inventory
        self.image_path = image_path
        self.private = self.root / "private"
        self.private.mkdir(mode=0o700)
        packet_root = self.private / "packets"
        restricted = self.private / "restricted" / "identity_map.json"
        result = gate.annotation.package_packets(
            selection_path=selection_path,
            image_inventory_path=image_path,
            freeze_path=self.fixture.freeze_path,
            packet_root=packet_root,
            restricted_map_path=restricted,
        )
        mapping = json.loads(restricted.read_text())
        streams = {
            slot: {
                "batch_count": mapping["packets"][slot]["batch_count"],
                "packet_ids": [row["packet_id"] for row in mapping["packets"][slot]["batches"]],
                "packet_manifest_sha256": [
                    row["packet_manifest_sha256"] for row in mapping["packets"][slot]["batches"]
                ],
            }
            for slot in ("rater_a", "rater_b")
        }
        reference = gate._descriptor(self.fixture.freeze_path)
        receipt = gate._sign({
            "schema_version": gate.PACKAGE_SCHEMA,
            "study_id": gate.STUDY_ID,
            "stage": "development",
            "status": "private_packets_ready_for_single_batch_gate",
            "media_job_receipt": reference,
            "preparation_receipt": reference,
            "pixel_blindness_receipt": reference,
            "development_freeze": reference,
            "image_inventory": gate._descriptor(image_path),
            "packet_tree": {"path": str(packet_root), **gate._tree_descriptor(packet_root)},
            "restricted_map": gate._descriptor(restricted),
            "rater_streams": streams,
            "counts": {
                "selected_requests": result["selected_requests"],
                "source_image_records": result["source_image_records"],
                "unique_packet_images": result["unique_packet_images"],
                "first_pass_raters_required": 2,
                "human_pixel_blindness_reviews_consumed": 1,
                "human_responses_consumed": 0,
                "human_labels_consumed": 0,
            },
            "science_counts": gate._zero_science_counts(),
            "labels_created_by_job": 0,
            "restricted_map_private": True,
            "safe_to_release_confirmation": False,
            "claim_boundary": "test-only synthetic fixture; no scientific claim",
            "completed_at_utc": "2026-09-13T00:00:00Z",
        })
        write_json(self.private / "package_receipt.json", receipt)
        self.public = self.root / "handoff"
        self.public.mkdir()
        self.inbox = self.root / "inbox"
        self.inbox.mkdir()
        self.base = datetime(2026, 9, 13, tzinfo=timezone.utc)

    def _real_finalize_inputs(self):
        preparation = self.root / "real-finalize-preparation"
        preparation.mkdir(exist_ok=True)
        pre_review = gate.media.sign_document({
            "schema_version": gate.media.PRE_REVIEW_SCHEMA,
            "study_id": gate.STUDY_ID,
            "stage": "development",
            "status": "pending_human_pixel_blindness_review",
            "selection_manifest_sha256": self.image_inventory["selection_manifest_sha256"],
            "images": self.image_inventory["images"],
            "safe_for_rater_distribution": False,
        })
        write_json(preparation / "image_inventory_pre_review.json", pre_review)
        review_path = Path(self.image_inventory["pixel_blindness_receipt_path"])
        if not review_path.is_absolute():
            review_path = self.image_path.parent / review_path
        return preparation, review_path.resolve()

    def tearDown(self):
        self.fixture.tearDown()
        self.temporary.cleanup()

    def _next_times(self):
        sequence = len(gate._lifecycle(self.private)) + 1
        released = self.base + timedelta(minutes=10 * sequence)
        return {
            "released": utc(released),
            "started": utc(released + timedelta(minutes=1)),
            "completed": utc(released + timedelta(minutes=4)),
            "locked": utc(released + timedelta(minutes=5)),
            "collected": utc(released + timedelta(minutes=6)),
            "revoked": utc(released + timedelta(minutes=7)),
        }

    def _response(self, *, rater_code, offset, times, impossible_seconds=False):
        template_path = self.public / "active" / "batch" / "response_template.json"
        response = json.loads(template_path.read_text())
        response["rater_code"] = rater_code
        required = (
            gate.annotation.ADJUDICATOR_ATTESTATIONS
            if response["rater_slot"] == "adjudicator"
            else gate.annotation.RESPONSE_ATTESTATIONS
        )
        response["attestations"] = {key: True for key in required}
        response["started_at"] = times["started"]
        response["completed_at"] = times["completed"]
        response["locked"] = True
        response["locked_at"] = times["locked"]
        for item in response["annotations"]:
            item.update({
                "cube_resolvability": "resolvable",
                "bowl_resolvability": "resolvable",
                "cube_identity": "rubiks_cube",
                "bowl_identity": "bowl",
                "cube_center_px": [8.0 + offset, 6.0],
                "bowl_center_px": [3.0, 4.0],
                "bowl_width_px": 5.0,
                "ambiguity_codes": ["none"],
                "source_guess": "unsure",
                "annotation_seconds": 10000.0 if impossible_seconds else 1.25,
                "notes": "",
            })
        path = self.inbox / f"{response['packet_id']}_{rater_code}.json"
        write_json(path, response)
        return path

    def _cycle(self, slot, ordinal, rater_code, offset):
        times = self._next_times()
        delivery = gate.release_batch(
            private_root=self.private,
            public_root=self.public,
            slot=slot,
            ordinal=ordinal,
            released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        response = self._response(rater_code=rater_code, offset=offset, times=times)
        collection = gate.collect_response(
            private_root=self.private,
            public_root=self.public,
            response=gate._descriptor(response),
            collector_code="authorized-collector",
            collected_at_utc=times["collected"],
        )
        revocation = gate.revoke_delivery(
            private_root=self.private,
            public_root=self.public,
            revoked_by="authorized-operator",
            revoked_at_utc=times["revoked"],
        )
        return delivery, collection, revocation

    def _replace_json(self, path, value):
        path.unlink()
        write_json(path, value)

    def _resign_candidate_delivery(self, delivery_id, **updates):
        pending = self.private / "pending_collections" / delivery_id
        active_path = self.public / "active" / "delivery_receipt.json"
        witness_path = self.private / "delivery_intents" / f"{delivery_id}.json"
        pending_delivery_path = pending / "delivery_receipt.json"
        delivery = json.loads(active_path.read_text())
        delivery.pop("payload_sha256")
        delivery.update(updates)
        delivery = gate._sign(delivery)
        for path in (active_path, witness_path, pending_delivery_path):
            self._replace_json(path, delivery)
        collection_path = pending / "collection_receipt.json"
        collection = json.loads(collection_path.read_text())
        collection.pop("payload_sha256")
        collection["delivery_receipt_sha256"] = gate._sha256_file(active_path)
        collection = gate._sign(collection)
        self._replace_json(collection_path, collection)
        return delivery, collection

    def test_contract_is_exact_and_zero_science(self):
        contract = gate.validate_contract()
        self.assertEqual(contract, gate._expected_contract())
        self.assertEqual(set(contract["science_counts"].values()), {0})
        self.assertFalse(contract["confirmation_policy"]["safe_to_release_confirmation"])
        self.assertFalse(contract["human_input_policy"]["job_may_create_or_fill_labels"])

    def test_real_finalize_reviewed_inventory_promotes_exact_valid_human_receipt(self):
        preparation, review = self._real_finalize_inputs()
        output = self.root / "real-reviewed-image-inventory.json"
        returned = gate.media.finalize_reviewed_inventory(
            preparation_dir=preparation,
            review_path=review,
            review_sha256=gate._sha256_file(review),
            output_path=output,
        )
        self.assertTrue(output.is_file())
        self.assertEqual(returned, json.loads(output.read_text()))
        self.assertEqual(returned["schema_version"], gate.media.IMAGE_INVENTORY_SCHEMA)
        self.assertEqual(returned["stage"], "development")
        self.assertEqual(returned["images"], self.image_inventory["images"])
        self.assertEqual(returned["pixel_blindness_receipt_sha256"], gate._sha256_file(review))

    def test_real_finalize_rejects_stale_partial_stage_scope_asset_signature_and_output(self):
        preparation, original_review = self._real_finalize_inputs()
        original = json.loads(original_review.read_text())

        stale = self.root / "stale-review.json"
        write_json(stale, original)
        stale_sha = gate._sha256_file(stale)
        stale.write_text(stale.read_text() + " ")
        with self.assertRaisesRegex(gate.media.AnnotationMediaBridgeError, "hash changed"):
            gate.media.finalize_reviewed_inventory(
                preparation_dir=preparation, review_path=stale,
                review_sha256=stale_sha, output_path=self.root / "stale-output.json",
            )

        cases = []
        partial = json.loads(json.dumps(original))
        partial["assets"] = partial["assets"][:-1]
        cases.append(("partial", partial, "exact rendered media population"))
        wrong_stage = json.loads(json.dumps(original))
        wrong_stage["stage"] = "confirmation"
        cases.append(("wrong-stage", wrong_stage, "stage mismatch"))
        wrong_scope = json.loads(json.dumps(original))
        wrong_scope["artifact_scope"] = "illustrated_examples"
        cases.append(("wrong-scope", wrong_scope, "artifact scope mismatch"))
        wrong_asset = json.loads(json.dumps(original))
        wrong_asset["assets"][0]["media_sha256"] = "f" * 64
        cases.append(("wrong-asset", wrong_asset, "exact rendered media population"))
        for name, value, error in cases:
            path = self.root / f"{name}.json"
            value.pop("payload_sha256")
            write_json(path, gate.media.sign_document(value))
            with self.subTest(name=name), self.assertRaisesRegex(
                gate.media.AnnotationMediaBridgeError, error
            ):
                gate.media.finalize_reviewed_inventory(
                    preparation_dir=preparation, review_path=path,
                    review_sha256=gate._sha256_file(path),
                    output_path=self.root / f"{name}-output.json",
                )

        unsigned = json.loads(json.dumps(original))
        unsigned["reviewer_code"] = "tampered-reviewer"
        unsigned_path = self.root / "bad-signature.json"
        write_json(unsigned_path, unsigned)
        with self.assertRaisesRegex(gate.media.AnnotationMediaBridgeError, "payload hash mismatch"):
            gate.media.finalize_reviewed_inventory(
                preparation_dir=preparation, review_path=unsigned_path,
                review_sha256=gate._sha256_file(unsigned_path),
                output_path=self.root / "bad-signature-output.json",
            )

        existing = self.root / "already-exists.json"
        existing.write_text("do not overwrite\n")
        before = existing.read_bytes()
        with self.assertRaisesRegex(gate.media.AnnotationMediaBridgeError, "refusing to overwrite"):
            gate.media.finalize_reviewed_inventory(
                preparation_dir=preparation, review_path=original_review,
                review_sha256=gate._sha256_file(original_review), output_path=existing,
            )
        self.assertEqual(existing.read_bytes(), before)

    def test_single_batch_publication_is_atomic_source_free_and_private_map_absent(self):
        times = self._next_times()
        receipt = gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        self.assertEqual([path.name for path in self.public.iterdir()], ["active"])
        self.assertFalse((self.public / "active" / "restricted").exists())
        self.assertNotIn(b"source_request_id", (self.public / "active" / "batch" / "packet.json").read_bytes())
        self.assertEqual(receipt["science_counts"], gate._zero_science_counts())
        self.assertEqual(receipt["labels_created_by_job"], 0)
        self.assertFalse(receipt["safe_to_release_confirmation"])
        with self.assertRaisesRegex(gate.AnnotationGateError, "already active"):
            gate.release_batch(
                private_root=self.private, public_root=self.public,
                slot="rater_b", ordinal=1, released_by="authorized-operator",
                released_at_utc=utc(self.base + timedelta(hours=1)),
            )

    def test_release_fails_closed_on_skipped_ordinal_and_extra_public_file(self):
        with self.assertRaisesRegex(gate.AnnotationGateError, "next frozen ordinal"):
            gate.release_batch(
                private_root=self.private, public_root=self.public,
                slot="rater_a", ordinal=2, released_by="authorized-operator",
                released_at_utc=utc(self.base + timedelta(hours=1)),
            )
        (self.public / "unowned.txt").write_text("x")
        with self.assertRaisesRegex(gate.AnnotationGateError, "outside the sole active batch"):
            gate.release_batch(
                private_root=self.private, public_root=self.public,
                slot="rater_a", ordinal=1, released_by="authorized-operator",
                released_at_utc=utc(self.base + timedelta(hours=1)),
            )

    def test_release_rejects_extra_or_identity_bearing_packet_payload(self):
        mapping = json.loads((self.private / "restricted" / "identity_map.json").read_text())
        batch = mapping["packets"]["rater_a"]["batches"][0]
        packet_root = Path(mapping["packet_root"])
        batch_dir = (packet_root / batch["packet_relative_path"]).parent
        (batch_dir / "extra.json").write_text('{"model_id":"N3"}\n')
        # Rebind the private package tree so the failure is specifically the
        # source-free inventory gate, not the outer immutable-tree checksum.
        package_path = self.private / "package_receipt.json"
        package = json.loads(package_path.read_text())
        package["packet_tree"] = {
            "path": str(packet_root), **gate._tree_descriptor(packet_root)
        }
        package = gate._sign(package)
        write_json(package_path, package)
        with self.assertRaisesRegex(gate.AnnotationGateError, "file inventory"):
            gate.release_batch(
                private_root=self.private, public_root=self.public,
                slot="rater_a", ordinal=1, released_by="authorized-operator",
                released_at_utc=utc(self.base + timedelta(hours=1)),
            )

    def test_duplicate_json_keys_and_release_before_package_time_fail_closed(self):
        malformed = self.root / "duplicate.json"
        malformed.write_text('{"a":1,"a":2}\n')
        with self.assertRaisesRegex(gate.AnnotationGateError, "duplicate JSON key"):
            gate._load_json(malformed, "malformed fixture")
        with self.assertRaisesRegex(gate.AnnotationGateError, "precedes private packet completion"):
            gate.release_batch(
                private_root=self.private, public_root=self.public,
                slot="rater_a", ordinal=1, released_by="authorized-operator",
                released_at_utc="2026-09-12T23:59:59Z",
            )
        future = datetime.now(timezone.utc) + timedelta(
            seconds=gate.MAX_FUTURE_CLOCK_SKEW_SECONDS + 60
        )
        with self.assertRaisesRegex(gate.AnnotationGateError, "future clock skew"):
            gate._event_timestamp(utc(future), "future event")

    def test_private_delivery_witness_blocks_rerelease_if_public_tree_disappears(self):
        times = self._next_times()
        receipt = gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        witness = self.private / "delivery_intents" / f"{receipt['delivery_id']}.json"
        self.assertTrue(witness.is_file())
        # Simulate out-of-band handoff deletion. The private immutable witness
        # must prevent treating that deletion as a completed revocation.
        shutil.rmtree(self.public / "active")
        with self.assertRaisesRegex(gate.AnnotationGateError, "unreconciled delivery witness"):
            gate.release_batch(
                private_root=self.private, public_root=self.public,
                slot="rater_a", ordinal=1, released_by="authorized-operator",
                released_at_utc=utc(self.base + timedelta(hours=1)),
            )

    def test_regular_file_cannot_replace_archived_delivery_and_reset_ordinal(self):
        _, _, revocation = self._cycle("rater_a", 1, "human-alpha", 0.0)
        archive = self.private / "revoked_deliveries" / revocation["delivery_id"]
        shutil.rmtree(archive)
        archive.write_text("not an archived delivery\n")
        with self.assertRaisesRegex(gate.AnnotationGateError, "non-directory"):
            gate.release_batch(
                private_root=self.private, public_root=self.public,
                slot="rater_a", ordinal=1, released_by="authorized-operator",
                released_at_utc=utc(self.base + timedelta(hours=1)),
            )

    def test_dangling_archive_root_symlink_blocks_release_before_publication(self):
        archive = self.private / "revoked_deliveries"
        dangling_target = self.root / "missing-archive-target"
        archive.symlink_to(dangling_target, target_is_directory=True)
        self.assertTrue(archive.is_symlink())
        self.assertFalse(archive.exists())
        with self.assertRaisesRegex(gate.AnnotationGateError, "revoked-delivery archive changed"):
            gate.release_batch(
                private_root=self.private, public_root=self.public,
                slot="rater_a", ordinal=1, released_by="authorized-operator",
                released_at_utc=utc(self.base + timedelta(minutes=10)),
            )
        self.assertTrue(archive.is_symlink())
        self.assertFalse(dangling_target.exists())
        self.assertEqual(list(self.public.iterdir()), [])
        self.assertFalse((self.private / "delivery_intents").exists())

    def test_active_and_archived_publication_inventories_are_exact(self):
        times = self._next_times()
        gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        response = self._response(rater_code="human-alpha", offset=0.0, times=times)
        shutil.copyfile(
            self.private / "restricted" / "identity_map.json",
            self.public / "active" / "restricted_identity_map.json",
        )
        with self.assertRaisesRegex(gate.AnnotationGateError, "active annotation handoff inventory"):
            gate.collect_response(
                private_root=self.private, public_root=self.public,
                response=gate._descriptor(response), collector_code="authorized-collector",
                collected_at_utc=times["collected"],
            )
        (self.public / "active" / "restricted_identity_map.json").unlink()
        gate.collect_response(
            private_root=self.private, public_root=self.public,
            response=gate._descriptor(response), collector_code="authorized-collector",
            collected_at_utc=times["collected"],
        )
        revocation = gate.revoke_delivery(
            private_root=self.private, public_root=self.public,
            revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
        )
        published = (
            self.private / "revoked_deliveries" / revocation["delivery_id"]
            / "published_batch"
        )
        (published / "source_key.json").write_text('{"source_request_id":"leak"}\n')
        with self.assertRaisesRegex(gate.AnnotationGateError, "archived published batch inventory"):
            gate.release_batch(
                private_root=self.private, public_root=self.public,
                slot="rater_a", ordinal=2, released_by="authorized-operator",
                released_at_utc=utc(self.base + timedelta(hours=1)),
            )

    def test_partial_collection_crash_residue_blocks_second_collection(self):
        times = self._next_times()
        delivery = gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        response = self._response(rater_code="human-alpha", offset=0.0, times=times)
        residue = (
            self.private / "pending_collections"
            / f".{delivery['delivery_id']}.crash-residue"
        )
        residue.mkdir(parents=True)
        shutil.copyfile(response, residue / "response.json")
        with self.assertRaisesRegex(gate.AnnotationGateError, "unreconciled pending collection"):
            gate.collect_response(
                private_root=self.private, public_root=self.public,
                response=gate._descriptor(response), collector_code="authorized-collector",
                collected_at_utc=times["collected"],
            )
        self.assertTrue((self.public / "active").is_dir())
        self.assertTrue(residue.is_dir())
        self.assertFalse(
            (self.private / "pending_collections" / delivery["delivery_id"]).exists()
        )
        self.assertFalse((self.private / "revoked_deliveries").exists())

    def test_foreign_pending_entry_blocks_collection(self):
        times = self._next_times()
        delivery = gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        response = self._response(rater_code="human-alpha", offset=0.0, times=times)
        pending_root = self.private / "pending_collections"
        pending_root.mkdir()
        (pending_root / "foreign-entry").write_text("not a transaction\n")
        with self.assertRaisesRegex(gate.AnnotationGateError, "unexpected content"):
            gate.collect_response(
                private_root=self.private, public_root=self.public,
                response=gate._descriptor(response), collector_code="authorized-collector",
                collected_at_utc=times["collected"],
            )
        self.assertTrue((self.public / "active").is_dir())
        self.assertFalse((pending_root / delivery["delivery_id"]).exists())

    def test_completed_collection_retry_is_rejected_without_mutation(self):
        times = self._next_times()
        delivery = gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        response = self._response(rater_code="human-alpha", offset=0.0, times=times)
        gate.collect_response(
            private_root=self.private, public_root=self.public,
            response=gate._descriptor(response), collector_code="authorized-collector",
            collected_at_utc=times["collected"],
        )
        pending = self.private / "pending_collections" / delivery["delivery_id"]
        active_sha = gate.annotation.artifact_sha256(self.public / "active")
        pending_sha = gate.annotation.artifact_sha256(pending)
        with self.assertRaisesRegex(gate.AnnotationGateError, "unreconciled pending collection"):
            gate.collect_response(
                private_root=self.private, public_root=self.public,
                response=gate._descriptor(response), collector_code="authorized-collector",
                collected_at_utc=times["collected"],
            )
        self.assertEqual(gate.annotation.artifact_sha256(self.public / "active"), active_sha)
        self.assertEqual(gate.annotation.artifact_sha256(pending), pending_sha)

    def test_full_collection_crash_residue_blocks_revocation_without_false_success(self):
        times = self._next_times()
        delivery = gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        response = self._response(rater_code="human-alpha", offset=0.0, times=times)
        gate.collect_response(
            private_root=self.private, public_root=self.public,
            response=gate._descriptor(response), collector_code="authorized-collector",
            collected_at_utc=times["collected"],
        )
        pending = self.private / "pending_collections" / delivery["delivery_id"]
        residue = pending.parent / f".{delivery['delivery_id']}.crash-residue"
        shutil.copytree(pending, residue)
        active_sha = gate.annotation.artifact_sha256(self.public / "active")
        pending_sha = gate.annotation.artifact_sha256(pending)
        residue_sha = gate.annotation.artifact_sha256(residue)
        witness = self.private / "delivery_intents" / f"{delivery['delivery_id']}.json"
        witness_sha = gate._sha256_file(witness)
        with self.assertRaisesRegex(gate.AnnotationGateError, "beyond the active delivery"):
            gate.revoke_delivery(
                private_root=self.private, public_root=self.public,
                revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
            )
        self.assertTrue((self.public / "active").is_dir())
        self.assertTrue(pending.is_dir())
        self.assertTrue(residue.is_dir())
        self.assertEqual(gate.annotation.artifact_sha256(self.public / "active"), active_sha)
        self.assertEqual(gate.annotation.artifact_sha256(pending), pending_sha)
        self.assertEqual(gate.annotation.artifact_sha256(residue), residue_sha)
        self.assertEqual(gate._sha256_file(witness), witness_sha)
        self.assertFalse(
            (self.private / "revoked_deliveries" / delivery["delivery_id"]).exists()
        )
        self.assertEqual(list(self.private.rglob("revocation_receipt.json")), [])
        self.assertEqual(list((self.private / "revoked_deliveries").glob("*.revoking")), [])

    def test_pending_collection_inventory_types_and_bindings_fail_before_revocation(self):
        times = self._next_times()
        delivery = gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        response = self._response(rater_code="human-alpha", offset=0.0, times=times)
        gate.collect_response(
            private_root=self.private, public_root=self.public,
            response=gate._descriptor(response), collector_code="authorized-collector",
            collected_at_utc=times["collected"],
        )
        pending = self.private / "pending_collections" / delivery["delivery_id"]
        active_sha = gate.annotation.artifact_sha256(self.public / "active")
        witness = self.private / "delivery_intents" / f"{delivery['delivery_id']}.json"
        witness_sha = gate._sha256_file(witness)

        collection_path = pending / "collection_receipt.json"
        collection_bytes = collection_path.read_bytes()
        collection_path.unlink()
        with self.assertRaisesRegex(gate.AnnotationGateError, "inventory changed"):
            gate.revoke_delivery(
                private_root=self.private, public_root=self.public,
                revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
            )
        collection_path.write_bytes(collection_bytes)

        extra = pending / "extra.json"
        extra.write_text("{}\n")
        with self.assertRaisesRegex(gate.AnnotationGateError, "inventory changed"):
            gate.revoke_delivery(
                private_root=self.private, public_root=self.public,
                revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
            )
        extra.unlink()

        response_path = pending / "response.json"
        response_bytes = response_path.read_bytes()
        response_path.unlink()
        os.mkfifo(response_path)
        try:
            with self.assertRaisesRegex(gate.AnnotationGateError, "not a regular file"):
                gate.revoke_delivery(
                    private_root=self.private, public_root=self.public,
                    revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
                )
        finally:
            response_path.unlink()
            response_path.write_bytes(response_bytes)

        response_path.unlink()
        response_path.write_bytes(response_bytes + b"tamper")
        with self.assertRaisesRegex(gate.AnnotationGateError, "response bytes"):
            gate.revoke_delivery(
                private_root=self.private, public_root=self.public,
                revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
            )
        response_path.unlink()
        response_path.write_bytes(response_bytes)

        private_delivery = pending / "delivery_receipt.json"
        private_delivery_bytes = private_delivery.read_bytes()
        private_delivery.unlink()
        private_delivery.write_bytes(private_delivery_bytes + b"tamper")
        with self.assertRaisesRegex(gate.AnnotationGateError, "private delivery copy"):
            gate.revoke_delivery(
                private_root=self.private, public_root=self.public,
                revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
            )
        private_delivery.unlink()
        private_delivery.write_bytes(private_delivery_bytes)

        self.assertEqual(gate.annotation.artifact_sha256(self.public / "active"), active_sha)
        self.assertEqual(gate._sha256_file(witness), witness_sha)
        self.assertTrue(pending.is_dir())
        self.assertEqual(list(self.private.rglob("revocation_receipt.json")), [])

    def test_resigned_candidate_rater_hash_fails_complete_preflight_without_mutation(self):
        times = self._next_times()
        delivery = gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        response = self._response(rater_code="human-alpha", offset=0.0, times=times)
        gate.collect_response(
            private_root=self.private, public_root=self.public,
            response=gate._descriptor(response), collector_code="authorized-collector",
            collected_at_utc=times["collected"],
        )
        pending = self.private / "pending_collections" / delivery["delivery_id"]
        collection_path = pending / "collection_receipt.json"
        collection = json.loads(collection_path.read_text())
        collection.pop("payload_sha256")
        collection["rater_code_sha256"] = "0" * 64
        self._replace_json(collection_path, gate._sign(collection))
        witness = self.private / "delivery_intents" / f"{delivery['delivery_id']}.json"
        active_sha = gate.annotation.artifact_sha256(self.public / "active")
        pending_sha = gate.annotation.artifact_sha256(pending)
        witness_sha = gate._sha256_file(witness)

        with self.assertRaisesRegex(gate.AnnotationGateError, "rater identity hash"):
            gate.revoke_delivery(
                private_root=self.private, public_root=self.public,
                revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
            )
        self.assertEqual(gate.annotation.artifact_sha256(self.public / "active"), active_sha)
        self.assertEqual(gate.annotation.artifact_sha256(pending), pending_sha)
        self.assertEqual(gate._sha256_file(witness), witness_sha)
        archive = self.private / "revoked_deliveries"
        self.assertFalse((archive / delivery["delivery_id"]).exists())
        self.assertEqual(list(archive.glob("*.revoking")), [])
        self.assertEqual(list(self.private.rglob("revocation_receipt.json")), [])

    def test_candidate_preflight_rejects_self_consistent_cross_history_rater_change(self):
        self._cycle("rater_a", 1, "human-alpha", 0.0)
        times = self._next_times()
        delivery = gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=2, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        response = self._response(rater_code="human-alpha", offset=0.0, times=times)
        gate.collect_response(
            private_root=self.private, public_root=self.public,
            response=gate._descriptor(response), collector_code="authorized-collector",
            collected_at_utc=times["collected"],
        )
        pending = self.private / "pending_collections" / delivery["delivery_id"]
        response_path = pending / "response.json"
        response_value = json.loads(response_path.read_text())
        response_value["rater_code"] = "human-beta"
        self._replace_json(response_path, response_value)
        collection_path = pending / "collection_receipt.json"
        collection = json.loads(collection_path.read_text())
        collection.pop("payload_sha256")
        collection["response"] = {
            "file": "response.json",
            "sha256": gate._sha256_file(response_path),
            "bytes": response_path.stat().st_size,
        }
        collection["rater_code_sha256"] = hashlib.sha256(b"human-beta").hexdigest()
        self._replace_json(collection_path, gate._sign(collection))
        witness = self.private / "delivery_intents" / f"{delivery['delivery_id']}.json"
        active_sha = gate.annotation.artifact_sha256(self.public / "active")
        pending_sha = gate.annotation.artifact_sha256(pending)
        witness_sha = gate._sha256_file(witness)

        with self.assertRaisesRegex(gate.AnnotationGateError, "multiple raters"):
            gate.revoke_delivery(
                private_root=self.private, public_root=self.public,
                revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
            )
        self.assertEqual(gate.annotation.artifact_sha256(self.public / "active"), active_sha)
        self.assertEqual(gate.annotation.artifact_sha256(pending), pending_sha)
        self.assertEqual(gate._sha256_file(witness), witness_sha)
        archive = self.private / "revoked_deliveries"
        self.assertFalse((archive / delivery["delivery_id"]).exists())
        self.assertEqual(list(archive.glob(f".{delivery['delivery_id']}*.revoking")), [])
        self.assertEqual(len(list(archive.glob("*/revocation_receipt.json"))), 1)

    def test_candidate_preflight_rejects_cross_history_time_and_frozen_stream_changes(self):
        self._cycle("rater_a", 1, "human-alpha", 0.0)
        times = self._next_times()
        delivery = gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=2, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        response = self._response(rater_code="human-alpha", offset=0.0, times=times)
        gate.collect_response(
            private_root=self.private, public_root=self.public,
            response=gate._descriptor(response), collector_code="authorized-collector",
            collected_at_utc=times["collected"],
        )
        pending = self.private / "pending_collections" / delivery["delivery_id"]
        original_delivery = json.loads(
            (self.public / "active" / "delivery_receipt.json").read_text()
        )

        self._resign_candidate_delivery(
            delivery["delivery_id"],
            released_at_utc=utc(self.base + timedelta(minutes=16, seconds=30)),
        )
        witness = self.private / "delivery_intents" / f"{delivery['delivery_id']}.json"
        time_active_sha = gate.annotation.artifact_sha256(self.public / "active")
        time_pending_sha = gate.annotation.artifact_sha256(pending)
        time_witness_sha = gate._sha256_file(witness)
        with self.assertRaisesRegex(gate.AnnotationGateError, "precedes its prior revocation"):
            gate.revoke_delivery(
                private_root=self.private, public_root=self.public,
                revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
            )
        self.assertEqual(gate.annotation.artifact_sha256(self.public / "active"), time_active_sha)
        self.assertEqual(gate.annotation.artifact_sha256(pending), time_pending_sha)
        self.assertEqual(gate._sha256_file(witness), time_witness_sha)

        self._resign_candidate_delivery(
            delivery["delivery_id"],
            released_at_utc=original_delivery["released_at_utc"],
            batch_count=original_delivery["batch_count"] + 1,
        )
        active_sha = gate.annotation.artifact_sha256(self.public / "active")
        pending_sha = gate.annotation.artifact_sha256(pending)
        witness_sha = gate._sha256_file(witness)
        with self.assertRaisesRegex(gate.AnnotationGateError, "frozen packet stream"):
            gate.revoke_delivery(
                private_root=self.private, public_root=self.public,
                revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
            )
        self.assertEqual(gate.annotation.artifact_sha256(self.public / "active"), active_sha)
        self.assertEqual(gate.annotation.artifact_sha256(pending), pending_sha)
        self.assertEqual(gate._sha256_file(witness), witness_sha)

        self._resign_candidate_delivery(
            delivery["delivery_id"],
            batch_count=original_delivery["batch_count"],
            package_receipt_sha256="f" * 64,
        )
        with self.assertRaisesRegex(gate.AnnotationGateError, "frozen packet stream"):
            gate.revoke_delivery(
                private_root=self.private, public_root=self.public,
                revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
            )

        self._resign_candidate_delivery(
            delivery["delivery_id"],
            package_receipt_sha256=original_delivery["package_receipt_sha256"],
            packet_artifact_sha256="f" * 64,
        )
        artifact_active_sha = gate.annotation.artifact_sha256(self.public / "active")
        artifact_pending_sha = gate.annotation.artifact_sha256(pending)
        artifact_witness_sha = gate._sha256_file(witness)
        with self.assertRaisesRegex(gate.AnnotationGateError, "active packet bytes changed"):
            gate.revoke_delivery(
                private_root=self.private, public_root=self.public,
                revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
            )
        self.assertEqual(gate.annotation.artifact_sha256(self.public / "active"), artifact_active_sha)
        self.assertEqual(gate.annotation.artifact_sha256(pending), artifact_pending_sha)
        self.assertEqual(gate._sha256_file(witness), artifact_witness_sha)
        archive = self.private / "revoked_deliveries"
        self.assertFalse((archive / delivery["delivery_id"]).exists())
        self.assertEqual(list(archive.glob(f".{delivery['delivery_id']}*.revoking")), [])

    def test_source_free_batch_preflights_nested_special_entry_before_json_open(self):
        times = self._next_times()
        delivery = gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        packet = self.public / "active" / "batch" / "packet.json"
        packet.unlink()
        os.mkfifo(packet)
        with self.assertRaisesRegex(gate.AnnotationGateError, "source-free packet contains a non-file"):
            gate._active_delivery(self.public, private_root=self.private)
        self.assertTrue((self.public / "active").is_dir())
        self.assertTrue(packet.exists())
        self.assertFalse((self.private / "revoked_deliveries").exists())
        self.assertTrue(
            (self.private / "delivery_intents" / f"{delivery['delivery_id']}.json").is_file()
        )

    def test_existing_revoking_transaction_blocks_revocation_before_mutation(self):
        times = self._next_times()
        delivery = gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        response = self._response(rater_code="human-alpha", offset=0.0, times=times)
        gate.collect_response(
            private_root=self.private, public_root=self.public,
            response=gate._descriptor(response), collector_code="authorized-collector",
            collected_at_utc=times["collected"],
        )
        pending = self.private / "pending_collections" / delivery["delivery_id"]
        active_sha = gate.annotation.artifact_sha256(self.public / "active")
        pending_sha = gate.annotation.artifact_sha256(pending)
        staging = (
            self.private / "revoked_deliveries"
            / f".{delivery['delivery_id']}.revoking"
        )
        staging.mkdir(parents=True)
        with self.assertRaisesRegex(gate.AnnotationGateError, "archived delivery inventory changed"):
            gate.revoke_delivery(
                private_root=self.private, public_root=self.public,
                revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
            )
        self.assertEqual(gate.annotation.artifact_sha256(self.public / "active"), active_sha)
        self.assertEqual(gate.annotation.artifact_sha256(pending), pending_sha)
        self.assertTrue(staging.is_dir())
        self.assertEqual(list(self.private.rglob("revocation_receipt.json")), [])

    def test_foreign_archive_entries_block_revocation_without_mutating_live_transaction(self):
        times = self._next_times()
        delivery = gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        response = self._response(rater_code="human-alpha", offset=0.0, times=times)
        gate.collect_response(
            private_root=self.private, public_root=self.public,
            response=gate._descriptor(response), collector_code="authorized-collector",
            collected_at_utc=times["collected"],
        )
        pending = self.private / "pending_collections" / delivery["delivery_id"]
        witness = self.private / "delivery_intents" / f"{delivery['delivery_id']}.json"
        active_sha = gate.annotation.artifact_sha256(self.public / "active")
        pending_sha = gate.annotation.artifact_sha256(pending)
        witness_sha = gate._sha256_file(witness)
        archive = self.private / "revoked_deliveries"
        archive.mkdir()

        foreign_file = archive / "foreign-file"
        foreign_file.write_text("unowned archive content\n")
        with self.assertRaisesRegex(gate.AnnotationGateError, "non-directory"):
            gate.revoke_delivery(
                private_root=self.private, public_root=self.public,
                revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
            )
        self.assertEqual(gate.annotation.artifact_sha256(self.public / "active"), active_sha)
        self.assertEqual(gate.annotation.artifact_sha256(pending), pending_sha)
        self.assertEqual(gate._sha256_file(witness), witness_sha)
        self.assertEqual(list(self.private.rglob("revocation_receipt.json")), [])
        foreign_file.unlink()

        residue = archive / ".foreign-revocation-residue"
        residue.mkdir()
        (residue / "response.json").write_text("partial archive transaction\n")
        with self.assertRaisesRegex(gate.AnnotationGateError, "archived delivery inventory changed"):
            gate.revoke_delivery(
                private_root=self.private, public_root=self.public,
                revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
            )
        self.assertEqual(gate.annotation.artifact_sha256(self.public / "active"), active_sha)
        self.assertEqual(gate.annotation.artifact_sha256(pending), pending_sha)
        self.assertEqual(gate._sha256_file(witness), witness_sha)
        self.assertTrue(residue.is_dir())
        self.assertFalse((archive / delivery["delivery_id"]).exists())
        self.assertEqual(list(self.private.rglob("revocation_receipt.json")), [])
        self.assertEqual(list(archive.glob(f".{delivery['delivery_id']}*.revoking")), [])

    def test_active_and_archive_inventories_reject_special_file_type_substitutions(self):
        times = self._next_times()
        gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        active_receipt = self.public / "active" / "delivery_receipt.json"
        receipt_bytes = active_receipt.read_bytes()
        active_receipt.unlink()
        os.mkfifo(active_receipt)
        try:
            with self.assertRaisesRegex(gate.AnnotationGateError, "not a regular file"):
                gate._active_delivery(self.public, private_root=self.private)
        finally:
            active_receipt.unlink()
            active_receipt.write_bytes(receipt_bytes)

        active_batch = self.public / "active" / "batch"
        saved_batch = self.root / "saved-active-batch"
        os.replace(active_batch, saved_batch)
        active_batch.write_text("not a directory\n")
        try:
            with self.assertRaisesRegex(gate.AnnotationGateError, "not a directory"):
                gate._active_delivery(self.public, private_root=self.private)
        finally:
            active_batch.unlink()
            os.replace(saved_batch, active_batch)

        response = self._response(rater_code="human-alpha", offset=0.0, times=times)
        gate.collect_response(
            private_root=self.private, public_root=self.public,
            response=gate._descriptor(response), collector_code="authorized-collector",
            collected_at_utc=times["collected"],
        )
        revocation = gate.revoke_delivery(
            private_root=self.private, public_root=self.public,
            revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
        )
        archive = self.private / "revoked_deliveries" / revocation["delivery_id"]
        archived_response = archive / "response.json"
        response_bytes = archived_response.read_bytes()
        archived_response.unlink()
        response_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        response_socket.bind(str(archived_response))
        try:
            with self.assertRaisesRegex(gate.AnnotationGateError, "not a regular file"):
                gate._lifecycle(self.private)
        finally:
            response_socket.close()
            archived_response.unlink()
            archived_response.write_bytes(response_bytes)

        archived_receipt = archive / "published_batch" / "delivery_receipt.json"
        archived_receipt_bytes = archived_receipt.read_bytes()
        archived_receipt.unlink()
        archived_receipt.mkdir()
        try:
            with self.assertRaisesRegex(gate.AnnotationGateError, "not a regular file"):
                gate._lifecycle(self.private)
        finally:
            archived_receipt.rmdir()
            archived_receipt.write_bytes(archived_receipt_bytes)

        archived_batch = archive / "published_batch" / "batch"
        saved_archived_batch = self.root / "saved-archived-batch"
        os.replace(archived_batch, saved_archived_batch)
        os.mkfifo(archived_batch)
        try:
            with self.assertRaisesRegex(gate.AnnotationGateError, "not a directory"):
                gate._lifecycle(self.private)
        finally:
            archived_batch.unlink()
            os.replace(saved_archived_batch, archived_batch)

    def test_collection_binds_exact_hash_timestamp_and_does_not_create_labels(self):
        times = self._next_times()
        delivery = gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        response = self._response(rater_code="human-rater-a", offset=0.0, times=times)
        receipt = gate.collect_response(
            private_root=self.private, public_root=self.public,
            response=gate._descriptor(response), collector_code="authorized-collector",
            collected_at_utc=times["collected"],
        )
        self.assertEqual(receipt["delivery_id"], delivery["delivery_id"])
        self.assertEqual(receipt["response"]["sha256"], hashlib.sha256(response.read_bytes()).hexdigest())
        self.assertEqual(receipt["labels_created_by_job"], 0)
        self.assertGreater(receipt["annotation_count_received"], 0)
        self.assertLessEqual(receipt["response_timing"]["annotation_seconds"],
                             receipt["response_timing"]["wall_seconds"])

    def test_collection_rejects_hash_change_impossible_time_and_predelivery_start(self):
        times = self._next_times()
        gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        response = self._response(rater_code="human-rater-a", offset=0.0, times=times)
        bad_descriptor = gate._descriptor(response)
        response.write_text(response.read_text() + " ")
        with self.assertRaisesRegex(gate.AnnotationGateError, "bytes or SHA-256 changed"):
            gate.collect_response(
                private_root=self.private, public_root=self.public,
                response=bad_descriptor, collector_code="authorized-collector",
                collected_at_utc=times["collected"],
            )
        response = self._response(
            rater_code="human-rater-a", offset=0.0, times=times, impossible_seconds=True
        )
        with self.assertRaisesRegex(gate.AnnotationGateError, "session duration"):
            gate.collect_response(
                private_root=self.private, public_root=self.public,
                response=gate._descriptor(response), collector_code="authorized-collector",
                collected_at_utc=times["collected"],
            )
        response_value = json.loads(response.read_text())
        response_value["annotations"] = json.loads(
            (self.public / "active" / "batch" / "response_template.json").read_text()
        )["annotations"]
        for item in response_value["annotations"]:
            item.update({
                "cube_resolvability": "resolvable", "bowl_resolvability": "resolvable",
                "cube_identity": "rubiks_cube", "bowl_identity": "bowl",
                "cube_center_px": [8.0, 6.0], "bowl_center_px": [3.0, 4.0],
                "bowl_width_px": 5.0, "ambiguity_codes": ["none"],
                "source_guess": "unsure", "annotation_seconds": 1.0, "notes": "",
            })
        response_value["started_at"] = utc(self.base)
        response_value["completed_at"] = utc(self.base + timedelta(minutes=1))
        response_value["locked_at"] = utc(self.base + timedelta(minutes=2))
        write_json(response, response_value)
        with self.assertRaisesRegex(gate.AnnotationGateError, "precedes delivery"):
            gate.collect_response(
                private_root=self.private, public_root=self.public,
                response=gate._descriptor(response), collector_code="authorized-collector",
                collected_at_utc=times["collected"],
            )

    def test_revoke_requires_collection_then_archives_all_immutable_receipts(self):
        times = self._next_times()
        gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_a", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        with self.assertRaisesRegex(gate.AnnotationGateError, "no collected response"):
            gate.revoke_delivery(
                private_root=self.private, public_root=self.public,
                revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
            )
        response = self._response(rater_code="human-rater-a", offset=0.0, times=times)
        gate.collect_response(
            private_root=self.private, public_root=self.public,
            response=gate._descriptor(response), collector_code="authorized-collector",
            collected_at_utc=times["collected"],
        )
        receipt = gate.revoke_delivery(
            private_root=self.private, public_root=self.public,
            revoked_by="authorized-operator", revoked_at_utc=times["revoked"],
        )
        self.assertEqual(list(self.public.iterdir()), [])
        archive = self.private / "revoked_deliveries" / receipt["delivery_id"]
        self.assertTrue((archive / "published_batch" / "delivery_receipt.json").is_file())
        self.assertTrue((archive / "collection_receipt.json").is_file())
        self.assertTrue((archive / "revocation_receipt.json").is_file())
        self.assertTrue((archive / "response.json").is_file())
        self.assertEqual(len(gate._lifecycle(self.private)), 1)

    def test_distinct_raters_and_same_slot_identity_are_enforced(self):
        self._cycle("rater_a", 1, "human-alpha", 0.0)
        times = self._next_times()
        gate.release_batch(
            private_root=self.private, public_root=self.public,
            slot="rater_b", ordinal=1, released_by="authorized-operator",
            released_at_utc=times["released"],
        )
        response = self._response(rater_code="human-alpha", offset=0.25, times=times)
        with self.assertRaisesRegex(gate.AnnotationGateError, "same human"):
            gate.collect_response(
                private_root=self.private, public_root=self.public,
                response=gate._descriptor(response), collector_code="authorized-collector",
                collected_at_utc=times["collected"],
            )

    def test_tampered_immutable_response_blocks_later_release(self):
        _, _, revocation = self._cycle("rater_a", 1, "human-alpha", 0.0)
        archived = self.private / "revoked_deliveries" / revocation["delivery_id"] / "response.json"
        archived.chmod(0o600)
        archived.write_text(archived.read_text() + " ")
        with self.assertRaisesRegex(gate.AnnotationGateError, "response hash binding changed"):
            gate.release_batch(
                private_root=self.private, public_root=self.public,
                slot="rater_a", ordinal=2, released_by="authorized-operator",
                released_at_utc=utc(self.base + timedelta(hours=10)),
            )

    def test_full_first_pass_adjudication_and_time_accounting_include_adjudicator(self):
        mapping = json.loads((self.private / "restricted" / "identity_map.json").read_text())
        for slot, rater_code, offset in (
            ("rater_a", "human-alpha", 0.0),
            ("rater_b", "human-beta", 0.5),
        ):
            for ordinal in range(1, mapping["packets"][slot]["batch_count"] + 1):
                self._cycle(slot, ordinal, rater_code, offset)
        completed = utc(self.base + timedelta(minutes=10 * (len(gate._lifecycle(self.private)) + 1)))
        adjudication = gate.prepare_adjudication(
            private_root=self.private, public_root=self.public,
            completed_at_utc=completed,
        )
        self.assertGreater(adjudication["counts"]["adjudication_required"], 0)
        self.assertEqual(adjudication["labels_created_by_job"], 0)
        adjudication_map = json.loads(
            (self.private / "restricted" / "adjudication_map.json").read_text()
        )
        for ordinal in range(1, adjudication_map["packets"]["adjudicator"]["batch_count"] + 1):
            self._cycle("adjudicator", ordinal, "human-gamma", 0.25)
        output = self.private / "annotation_time_accounting.json"
        receipt = gate.write_time_accounting(
            private_root=self.private, output_path=output,
            completed_at_utc=utc(self.base + timedelta(hours=4)),
        )
        self.assertGreater(receipt["annotation_seconds"]["rater_a_total"], 0)
        self.assertGreater(receipt["annotation_seconds"]["rater_b_total"], 0)
        self.assertGreater(receipt["annotation_seconds"]["adjudicator_total"], 0)
        self.assertEqual(receipt["labels_created_by_job"], 0)
        self.assertFalse(receipt["safe_to_release_confirmation"])

    def test_no_disagreement_records_zero_adjudicator_total_without_a_response(self):
        mapping = json.loads((self.private / "restricted" / "identity_map.json").read_text())
        for slot, rater_code in (
            ("rater_a", "human-alpha"),
            ("rater_b", "human-beta"),
        ):
            for ordinal in range(1, mapping["packets"][slot]["batch_count"] + 1):
                self._cycle(slot, ordinal, rater_code, 0.0)
        completed = utc(self.base + timedelta(minutes=10 * (len(gate._lifecycle(self.private)) + 1)))
        adjudication = gate.prepare_adjudication(
            private_root=self.private, public_root=self.public,
            completed_at_utc=completed,
        )
        self.assertEqual(adjudication["counts"]["adjudication_required"], 0)
        self.assertIsNone(adjudication["packet_tree"])
        output = self.private / "zero_adjudicator_time.json"
        timing = gate.write_time_accounting(
            private_root=self.private, output_path=output,
            completed_at_utc=utc(self.base + timedelta(hours=4)),
        )
        self.assertEqual(timing["annotation_seconds"]["adjudicator_total"], 0.0)
        self.assertEqual(timing["by_slot"]["adjudicator"]["response_count"], 0)

    def test_package_wrapper_calls_all_three_authoritative_functions_and_stays_zero_label(self):
        sandbox = self.root / "package-wrapper"
        prep = sandbox / "preparation"
        prep.mkdir(parents=True)
        write_json(prep / "preparation_receipt.json", {"receipt": "prepared"})
        write_json(prep / "request_selection.json", {"selection": "placeholder"})
        media_receipt = sandbox / "media_job_receipt.json"
        write_json(media_receipt, {
            "outputs": {
                "output_root": str(prep.resolve()),
                "preparation_receipt": gate._descriptor(prep / "preparation_receipt.json"),
                "publish_tranche_index": {"path": str((prep / "index.json").resolve()), "sha256": "a" * 64, "bytes": 1},
            },
            "completed_at_utc": "2026-09-13T00:00:00Z",
        })
        review = sandbox / "review.json"
        write_json(review, {"reviewed_at": "2026-09-13T00:01:00Z"})
        freeze = sandbox / "freeze.json"
        write_json(freeze, {})
        examples = sandbox / "illustrated_examples.json"
        write_json(examples, {"reviewed_at": "2026-09-12T23:58:00Z"})
        examples_blindness = sandbox / "examples_blindness.json"
        write_json(examples_blindness, {"reviewed_at": "2026-09-12T23:59:00Z"})
        output = sandbox / "private"

        def fake_finalize(**kwargs):
            write_json(kwargs["output_path"], {"inventory": "human-reviewed"})

        def fake_packets(**kwargs):
            (kwargs["packet_root"] / "rater_a" / "batch_x").mkdir(parents=True)
            (kwargs["packet_root"] / "rater_a" / "batch_x" / "packet.json").write_text("{}\n")
            (kwargs["packet_root"] / "rater_b" / "batch_y").mkdir(parents=True)
            (kwargs["packet_root"] / "rater_b" / "batch_y" / "packet.json").write_text("{}\n")
            mapping = gate.annotation.sign_document({
                "schema_version": gate.annotation.RESTRICTED_MAP_SCHEMA,
                "study_id": gate.STUDY_ID,
                "stage": "development",
                "cohort_branch": "full_two_model",
                "qualified_model_ids": ["N3", "D1"],
                "visibility": "RESTRICTED ANALYST IDENTITY MAP; NEVER DISTRIBUTE TO RATERS OR BLIND ADJUDICATORS",
                "packet_root": str(kwargs["packet_root"].resolve()),
                "packets": {
                    "rater_a": {"batch_count": 1, "batches": [{"packet_id": "packet_x", "packet_manifest_sha256": "a" * 64}]},
                    "rater_b": {"batch_count": 1, "batches": [{"packet_id": "packet_y", "packet_manifest_sha256": "b" * 64}]},
                },
            })
            write_json(kwargs["restricted_map_path"], mapping)
            return {"selected_requests": 128, "source_image_records": 768, "unique_packet_images": 640}

        with (
            mock.patch.object(gate.media_jobs, "_validate_terminal_preparation_job") as validate_media,
            mock.patch.object(gate.annotation, "validate_freeze", return_value={
                "cohort_branch": "full_two_model", "qualified_model_ids": ["N3", "D1"],
                "examples_path": examples, "examples_blindness_path": examples_blindness,
            }) as validate_freeze,
            mock.patch.object(gate.media, "finalize_reviewed_inventory", side_effect=fake_finalize) as finalize,
            mock.patch.object(gate.annotation, "package_packets", side_effect=fake_packets) as package,
        ):
            receipt = gate.package_private_packets(
                study_commit="a" * 40,
                media_job_receipt=gate._descriptor(media_receipt),
                pixel_blindness_receipt=gate._descriptor(review),
                development_freeze=gate._descriptor(freeze),
                private_root=output,
                completed_at_utc="2026-09-13T00:02:00Z",
            )
        validate_media.assert_called_once()
        validate_freeze.assert_called_once()
        finalize.assert_called_once()
        package.assert_called_once()
        self.assertEqual(receipt["labels_created_by_job"], 0)
        self.assertEqual(receipt["science_counts"], gate._zero_science_counts())
        self.assertFalse(receipt["safe_to_release_confirmation"])


if __name__ == "__main__":
    unittest.main()
