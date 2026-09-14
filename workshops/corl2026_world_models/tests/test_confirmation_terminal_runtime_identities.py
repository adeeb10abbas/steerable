"""Adversarial tests for terminal-runtime identity publication."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
FORECAST = ROOT / "workshops/corl2026_world_models/experiments/forecast_layout"
SOURCE_PATH = FORECAST / "confirmation_terminal_runtime_identities.py"
CONTRACT_PATH = FORECAST / "confirmation_terminal_runtime_identities_contract.json"
SOURCE_BYTES = SOURCE_PATH.read_bytes()
BASE_CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
SOURCE_RELATIVE = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "confirmation_terminal_runtime_identities.py"
)
CONTRACT_RELATIVE = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "confirmation_terminal_runtime_identities_contract.json"
)
BRANCH = "codex/forecast-layout-gm-20260912"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


class IdentityFixture:
    """One tiny, real Git graph with synthetic non-production evidence."""

    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.checkout = self.base / "checkout"
        self.remote = self.base / "remote.git"
        self.evidence = self.base / "evidence"
        self.output_parent = self.base / "output"
        self.checkout.mkdir()
        self.evidence.mkdir()
        self.output_parent.mkdir()
        self._run("git", "init", "-q", "-b", BRANCH, cwd=self.checkout)
        self._run("git", "config", "user.email", "test@example.invalid", cwd=self.checkout)
        self._run("git", "config", "user.name", "WMF Test", cwd=self.checkout)

        source = self.checkout / SOURCE_RELATIVE
        source.parent.mkdir(parents=True)
        source.write_bytes(SOURCE_BYTES)
        self.contract = copy.deepcopy(BASE_CONTRACT)
        self.contract["status"] = "frozen_for_final_published_control_runtime"
        self.runtime_paths: dict[str, Path] = {}
        for name, row in self.contract["runtime_files"].items():
            path = self.checkout / row["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"# synthetic test runtime: {name}\n".encode()
            path.write_bytes(payload)
            row["sha256"] = digest(payload)
            self.runtime_paths[name] = path

        self.prerequisite_paths: dict[str, Path] = {}
        for name, row in self.contract["prerequisite_receipts"].items():
            parent = self.evidence / name
            parent.mkdir()
            path = parent / "receipt.json"
            value = {"schema_version": f"test-{name}-v1", "status": "passed"}
            payload = json_bytes(value)
            path.write_bytes(payload)
            row.clear()
            row.update(
                {
                    "path": str(path),
                    "bytes": len(payload),
                    "sha256": digest(payload),
                    "expected_fields": value,
                }
            )
            self.prerequisite_paths[name] = path

        self.contract_path = self.checkout / CONTRACT_RELATIVE
        self._write_contract()
        self._run("git", "add", ".", cwd=self.checkout)
        self._run("git", "commit", "-q", "-m", "initial", cwd=self.checkout)
        self._run("git", "init", "-q", "--bare", str(self.remote), cwd=self.base)
        self._run("git", "remote", "add", "test", self.remote.as_uri(), cwd=self.checkout)
        self._run(
            "git", "push", "-q", "test", f"HEAD:refs/heads/{BRANCH}", cwd=self.checkout
        )
        self.commit = self._text("git", "rev-parse", "HEAD", cwd=self.checkout)
        self.module = self._load_module()
        self.output = self.output_parent / "terminal-runtime-identities.json"
        self.commit_counter = 0

    def close(self) -> None:
        self.temporary.cleanup()

    def __enter__(self) -> "IdentityFixture":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def _run(*arguments: str, cwd: Path) -> None:
        subprocess.run(arguments, cwd=cwd, check=True, capture_output=True)

    @staticmethod
    def _text(*arguments: str, cwd: Path) -> str:
        return subprocess.run(
            arguments, cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()

    def _load_module(self):
        name = f"wmf_terminal_runtime_test_{id(self)}"
        specification = importlib.util.spec_from_file_location(
            name, self.checkout / SOURCE_RELATIVE
        )
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module

    def _write_contract(self) -> None:
        self.contract_path.write_bytes(json_bytes(self.contract))

    def update_runtime_contract(self, name: str) -> None:
        payload = self.runtime_paths[name].read_bytes()
        self.contract["runtime_files"][name]["sha256"] = digest(payload)
        self._write_contract()

    def commit_changes(self, *, push: bool) -> str:
        self.commit_counter += 1
        self._run("git", "add", ".", cwd=self.checkout)
        self._run(
            "git", "commit", "-q", "-m", f"change-{self.commit_counter}", cwd=self.checkout
        )
        commit = self._text("git", "rev-parse", "HEAD", cwd=self.checkout)
        if push:
            self._run(
                "git", "push", "-q", "test", f"HEAD:refs/heads/{BRANCH}", cwd=self.checkout
            )
        self.commit = commit
        return commit

    def advance_remote(self) -> str:
        marker = self.checkout / "control-marker.txt"
        marker.write_text(f"advance {self.commit_counter + 1}\n", encoding="utf-8")
        return self.commit_changes(push=True)

    def state(self) -> dict:
        with self.module._authenticated_graph(
            self.commit, remote_url=self.remote.as_uri(), protocol="file"
        ) as (graph, _authentication):
            return self.module._validate_source_and_evidence(
                source_root=self.checkout,
                source_commit=self.commit,
                graph=graph,
                protocol="file",
            )

    def produce(self, *, before_rename=None, after_rename=None) -> dict:
        return self.module._produce_impl(
            source_root=self.checkout,
            source_commit=self.commit,
            output=self.output,
            remote_url=self.remote.as_uri(),
            protocol="file",
            published_at_utc="2026-09-14T01:00:00Z",
            before_rename=before_rename,
            after_rename=after_rename,
        )


class ConfirmationTerminalRuntimeIdentityTests(unittest.TestCase):
    def test_checked_in_contract_is_frozen_to_current_confirmation_runtime_bytes(self) -> None:
        specification = importlib.util.spec_from_file_location("wmf_pending_contract", SOURCE_PATH)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        value = module.load_json_bytes(CONTRACT_PATH.read_bytes(), "contract")
        self.assertEqual(value["status"], "frozen_for_final_published_control_runtime")
        module._validate_contract(value)
        for name, relative in module.RUNTIME_FILES.items():
            self.assertEqual(
                value["runtime_files"][name]["sha256"],
                digest((ROOT / relative).read_bytes()),
            )
        pending = copy.deepcopy(value)
        pending["status"] = "awaiting_final_published_confirmation_runtime_hashes"
        with self.assertRaisesRegex(module.TerminalRuntimeIdentityError, "not frozen"):
            module._validate_contract(pending)

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(self) -> None:
        with IdentityFixture() as fixture:
            with self.assertRaisesRegex(fixture.module.TerminalRuntimeIdentityError, "duplicate"):
                fixture.module.load_json_bytes(b'{"a":1,"a":2}', "duplicate")
            with self.assertRaisesRegex(fixture.module.TerminalRuntimeIdentityError, "non-finite"):
                fixture.module.load_json_bytes(b'{"a":NaN}', "nonfinite")
            with self.assertRaisesRegex(fixture.module.TerminalRuntimeIdentityError, "non-finite"):
                fixture.module.load_json_bytes(b'{"a":1e999}', "overflow")
            with self.assertRaisesRegex(fixture.module.TerminalRuntimeIdentityError, "UTF-8"):
                fixture.module.load_json_bytes('{"a":1}'.encode("utf-16"), "utf16")

    def test_authenticated_state_binds_all_runtime_blobs_and_prerequisites(self) -> None:
        with IdentityFixture() as fixture:
            state = fixture.state()
            self.assertEqual(set(state["runtime_files"]), set(fixture.module.RUNTIME_FILES))
            self.assertEqual(
                set(state["prerequisite_receipts"]), fixture.module.PREREQUISITE_NAMES
            )
            for name, row in state["runtime_files"].items():
                self.assertEqual(
                    row["sha256"], fixture.contract["runtime_files"][name]["sha256"]
                )

    def test_unpublished_descendant_is_rejected(self) -> None:
        with IdentityFixture() as fixture:
            (fixture.checkout / "unpublished.txt").write_text("local only\n", encoding="utf-8")
            fixture.commit_changes(push=False)
            with self.assertRaisesRegex(
                fixture.module.TerminalRuntimeIdentityError, "published immutable source commit"
            ):
                fixture.produce()
            self.assertFalse(fixture.output.exists())

    def test_dirty_runtime_substitution_is_rejected(self) -> None:
        with IdentityFixture() as fixture:
            fixture.runtime_paths["n3_confirmation_block_job.py"].write_text(
                "# substituted runtime\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                fixture.module.TerminalRuntimeIdentityError, "(byte count|final published Git blob)"
            ):
                fixture.produce()
            self.assertFalse(fixture.output.exists())

    def test_coordinated_runtime_and_contract_substitution_is_rejected_when_unpublished(self) -> None:
        with IdentityFixture() as fixture:
            name = "d1_confirmation_block_jobs.py"
            fixture.runtime_paths[name].write_text("# coordinated substitute\n", encoding="utf-8")
            fixture.update_runtime_contract(name)
            fixture.commit_changes(push=False)
            with self.assertRaisesRegex(
                fixture.module.TerminalRuntimeIdentityError, "published immutable source commit"
            ):
                fixture.produce()
            self.assertFalse(fixture.output.exists())

    def test_dirty_producer_source_is_rejected(self) -> None:
        with IdentityFixture() as fixture:
            source = fixture.checkout / SOURCE_RELATIVE
            source.write_bytes(source.read_bytes() + b"\n# dirty\n")
            with self.assertRaisesRegex(
                fixture.module.TerminalRuntimeIdentityError, "producer source (byte count|differs)"
            ):
                fixture.produce()

    def test_prerequisite_receipt_substitution_is_rejected(self) -> None:
        with IdentityFixture() as fixture:
            path = fixture.prerequisite_paths["d1_qualification_receipt"]
            path.write_bytes(json_bytes({"schema_version": "substitute", "status": "passed"}))
            with self.assertRaisesRegex(
                fixture.module.TerminalRuntimeIdentityError, "(byte count|descriptor) changed"
            ):
                fixture.produce()
            self.assertFalse(fixture.output.exists())

    def test_prerequisite_leaf_symlink_is_rejected_without_reading_target(self) -> None:
        with IdentityFixture() as fixture:
            path = fixture.prerequisite_paths["n3_pilot_receipt"]
            sentinel = fixture.base / "private-sentinel"
            sentinel.write_text("must not be read\n", encoding="utf-8")
            path.unlink()
            path.symlink_to(sentinel)
            with self.assertRaisesRegex(
                fixture.module.TerminalRuntimeIdentityError, "symlink"
            ):
                fixture.produce()
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "must not be read\n")
            self.assertFalse(fixture.output.exists())

    def test_prerequisite_hard_link_is_rejected_before_publication(self) -> None:
        with IdentityFixture() as fixture:
            path = fixture.prerequisite_paths["n3_pilot_receipt"]
            alternate = fixture.base / "alternate-receipt.json"
            alternate.write_bytes(path.read_bytes())
            path.unlink()
            os.link(alternate, path)
            with self.assertRaisesRegex(
                fixture.module.TerminalRuntimeIdentityError, "singly linked"
            ):
                fixture.produce()
            self.assertFalse(fixture.output.exists())

    def test_prerequisite_parent_swap_during_same_fd_read_is_rejected(self) -> None:
        with IdentityFixture() as fixture:
            target = fixture.prerequisite_paths["recorder_receipt"].parent
            moved = fixture.base / "moved-receipt-parent"
            original = fixture.module._require_same_open_directory
            swapped = False

            def swap(path: Path, descriptor: int, label: str) -> None:
                nonlocal swapped
                if label == "prerequisite recorder_receipt parent" and not swapped:
                    swapped = True
                    target.rename(moved)
                    target.mkdir()
                original(path, descriptor, label)

            fixture.module._require_same_open_directory = swap
            with self.assertRaisesRegex(
                fixture.module.TerminalRuntimeIdentityError, "changed after it was opened"
            ):
                fixture.produce()
            self.assertTrue(swapped)
            self.assertFalse(fixture.output.exists())

    def test_output_symlink_parent_is_rejected_without_outside_residue(self) -> None:
        with IdentityFixture() as fixture:
            outside = fixture.base / "outside"
            outside.mkdir()
            fixture.output_parent.rmdir()
            fixture.output_parent.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(fixture.module.TerminalRuntimeIdentityError, "symlink"):
                fixture.produce()
            self.assertEqual(list(outside.iterdir()), [])

    def test_output_parent_swap_before_rename_cleans_held_staging_inode(self) -> None:
        with IdentityFixture() as fixture:
            moved = fixture.base / "moved-output"

            def swap() -> None:
                fixture.output_parent.rename(moved)
                fixture.output_parent.mkdir()

            with self.assertRaisesRegex(
                fixture.module.TerminalRuntimeIdentityError, "changed after it was opened"
            ):
                fixture.produce(before_rename=swap)
            self.assertFalse(fixture.output.exists())
            self.assertEqual(list(moved.iterdir()), [])
            self.assertEqual(list(fixture.output_parent.iterdir()), [])

    def test_output_parent_swap_after_rename_rolls_back_published_inode(self) -> None:
        with IdentityFixture() as fixture:
            moved = fixture.base / "moved-output"

            def swap() -> None:
                fixture.output_parent.rename(moved)
                fixture.output_parent.mkdir()

            with self.assertRaisesRegex(
                fixture.module.TerminalRuntimeIdentityError, "changed after it was opened"
            ):
                fixture.produce(after_rename=swap)
            self.assertFalse(fixture.output.exists())
            self.assertEqual(list(moved.iterdir()), [])
            self.assertEqual(list(fixture.output_parent.iterdir()), [])

    def test_prerequisite_drift_after_rename_rolls_back_output(self) -> None:
        with IdentityFixture() as fixture:
            path = fixture.prerequisite_paths["d1_pilot_server_receipt"]

            def drift() -> None:
                path.write_text('{"status":"changed"}\n', encoding="utf-8")

            with self.assertRaisesRegex(
                fixture.module.TerminalRuntimeIdentityError, "(byte count|descriptor) changed"
            ):
                fixture.produce(after_rename=drift)
            self.assertFalse(fixture.output.exists())
            self.assertEqual(list(fixture.output_parent.iterdir()), [])

    def test_remote_head_drift_after_rename_rolls_back_output(self) -> None:
        with IdentityFixture() as fixture:
            old_commit = fixture.commit

            def drift() -> None:
                fixture.advance_remote()

            with self.assertRaisesRegex(
                fixture.module.TerminalRuntimeIdentityError, "control head drifted"
            ):
                fixture.module._produce_impl(
                    source_root=fixture.checkout,
                    source_commit=old_commit,
                    output=fixture.output,
                    remote_url=fixture.remote.as_uri(),
                    protocol="file",
                    published_at_utc="2026-09-14T01:00:00Z",
                    after_rename=drift,
                )
            self.assertFalse(fixture.output.exists())
            self.assertEqual(list(fixture.output_parent.iterdir()), [])

    def test_published_source_ancestor_of_release_head_is_accepted(self) -> None:
        with IdentityFixture() as fixture:
            immutable_source = fixture.commit
            release_head = fixture.advance_remote()
            self.assertNotEqual(immutable_source, release_head)
            fixture.commit = immutable_source
            result = fixture.produce()
            receipt = json.loads(fixture.output.read_bytes())
            self.assertEqual(result["study_commit"], immutable_source)
            self.assertEqual(receipt["study_commit"], immutable_source)

    def test_success_is_exact_immutable_receipt_and_native_validator_replays(self) -> None:
        with IdentityFixture() as fixture:
            result = fixture.produce()
            payload = fixture.output.read_bytes()
            receipt = json.loads(payload)
            self.assertEqual(result["sha256"], digest(payload))
            self.assertEqual(set(receipt), fixture.module.OUTPUT_KEYS)
            self.assertEqual(receipt["schema_version"], fixture.module.OUTPUT_SCHEMA)
            self.assertEqual(receipt["study_commit"], fixture.commit)
            self.assertEqual(set(receipt["runtime_files"]), set(fixture.module.RUNTIME_FILES))
            self.assertEqual(
                set(receipt["prerequisite_receipts"]), fixture.module.PREREQUISITE_NAMES
            )
            self.assertEqual(os.stat(fixture.output).st_mode & 0o777, 0o444)
            validated = fixture.module._validate_published_impl(
                receipt_path=fixture.output,
                expected_sha256=result["sha256"],
                source_root=fixture.checkout,
                source_commit=fixture.commit,
                remote_url=fixture.remote.as_uri(),
                protocol="file",
            )
            self.assertEqual(validated["receipt"], receipt)
            self.assertEqual(validated["identity"]["sha256"], result["sha256"])

    def test_existing_output_is_never_replaced(self) -> None:
        with IdentityFixture() as fixture:
            fixture.output.write_text("preexisting\n", encoding="utf-8")
            with self.assertRaisesRegex(
                fixture.module.TerminalRuntimeIdentityError, "refusing to replace"
            ):
                fixture.produce()
            self.assertEqual(fixture.output.read_text(encoding="utf-8"), "preexisting\n")

    def test_native_validator_rejects_receipt_substitution(self) -> None:
        with IdentityFixture() as fixture:
            result = fixture.produce()
            os.chmod(fixture.output, 0o644)
            fixture.output.write_bytes(fixture.output.read_bytes() + b" ")
            with self.assertRaisesRegex(
                fixture.module.TerminalRuntimeIdentityError, "SHA-256 changed"
            ):
                fixture.module._validate_published_impl(
                    receipt_path=fixture.output,
                    expected_sha256=result["sha256"],
                    source_root=fixture.checkout,
                    source_commit=fixture.commit,
                    remote_url=fixture.remote.as_uri(),
                    protocol="file",
                )

    def test_receipt_validator_rejects_extra_claim_even_when_resigned(self) -> None:
        with IdentityFixture() as fixture:
            state = fixture.state()
            value = fixture.module.sign_document(
                {
                    "schema_version": fixture.module.OUTPUT_SCHEMA,
                    "status": fixture.module.OUTPUT_STATUS,
                    "study_id": fixture.module.STUDY_ID,
                    "namespace": fixture.module.NAMESPACE,
                    "study_commit": fixture.commit,
                    "published_at_utc": "2026-09-14T01:00:00Z",
                    "runtime_files": state["runtime_files"],
                    "terminal_context_schemas": fixture.module.TERMINAL_CONTEXT_SCHEMAS,
                    "prerequisite_receipts": state["prerequisite_receipts"],
                    "claim_boundary": fixture.module.CLAIM_BOUNDARY,
                    "execution_authorized": True,
                }
            )
            with self.assertRaisesRegex(
                fixture.module.TerminalRuntimeIdentityError, "disallowed keys"
            ):
                fixture.module._validate_receipt(value, state, fixture.commit)


if __name__ == "__main__":
    unittest.main()
