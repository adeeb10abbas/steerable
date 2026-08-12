#!/usr/bin/env python3
"""Validate the non-mutating V3-C002/V3-E006 frozen-validator reconciliation.

The two closed experiments bind incompatible bytes at one historical path,
``tools/validate_v3e_publication_bundle.py``.  This integration validator keeps
the pre-experiment mainline blob at that shared path, validates exact archival
copies against their source-branch blobs, and runs each experiment's own
closure validators in a disposable detached worktree where only that path is
overlaid.  It never edits an experiment manifest, source, artifact, or the
checked-out integration tree.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_RELATIVE = Path(
    "artifacts/vla_wam_shared_v3/final_merge_reconciliation/validator_reconciliation.json"
)
CONFLICT_PATH = "tools/validate_v3e_publication_bundle.py"
C002_CLOSURE_RELATIVE = Path(
    "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002/active/closure/evidence_manifest.json"
)
C002_REGISTRATION_RELATIVE = Path(
    "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002/active/registration.json"
)
E006_EVIDENCE_RELATIVE = Path(
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/results/evidence_manifest.json"
)
E006_RESULTS_RELATIVE = Path(
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/results/results.json"
)
E006_RELEASE_RELATIVE = Path(
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006/release_gate.json"
)


class ReconciliationError(RuntimeError):
    """Raised when the merged checkout cannot prove both frozen identities."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReconciliationError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconciliationError(f"invalid JSON at {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def run_git(
    repo_root: Path, args: Sequence[str], *, text: bool = True
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )


def git_text(repo_root: Path, args: Sequence[str]) -> str:
    completed = run_git(repo_root, args)
    if completed.returncode != 0:
        raise ReconciliationError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return str(completed.stdout)


def git_blob(repo_root: Path, commit: str, path: str) -> bytes:
    completed = run_git(repo_root, ["show", f"{commit}:{path}"], text=False)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise ReconciliationError(f"missing frozen blob {commit}:{path}: {stderr.strip()}")
    return bytes(completed.stdout)


def ensure_commit(repo_root: Path, commit: str, label: str) -> None:
    completed = run_git(repo_root, ["cat-file", "-e", f"{commit}^{{commit}}"])
    require(completed.returncode == 0, f"{label} commit is unavailable: {commit}")


def ensure_ancestor(repo_root: Path, commit: str, label: str) -> None:
    completed = run_git(repo_root, ["merge-base", "--is-ancestor", commit, "HEAD"])
    require(completed.returncode == 0, f"merged HEAD does not descend {label}: {commit}")


def verify_binding(repo_root: Path, binding: Mapping[str, Any], label: str) -> None:
    path_value = binding.get("path")
    require(isinstance(path_value, str) and path_value, f"{label} lacks a path")
    path = repo_root / path_value
    require(path.is_file(), f"{label} file is absent: {path_value}")
    require(type(binding.get("bytes")) is int, f"{label} lacks an integer byte count")
    require(binding["bytes"] == path.stat().st_size, f"{label} bytes differ: {path_value}")
    require(
        binding.get("sha256") == sha256_file(path),
        f"{label} SHA-256 differs: {path_value}",
    )


def verify_frozen_conflict_binding(
    binding: Mapping[str, Any], experiment: Mapping[str, Any], label: str
) -> None:
    frozen = experiment["frozen_validator"]
    require(binding.get("path") == CONFLICT_PATH, f"{label} conflict path changed")
    require(
        binding.get("bytes") == frozen["bytes"]
        and binding.get("sha256") == frozen["sha256"],
        f"{label} frozen validator binding differs",
    )


def branch_changed_paths(repo_root: Path, base: str, tip: str) -> set[str]:
    return {
        line
        for line in git_text(repo_root, ["diff", "--name-only", f"{base}..{tip}"]).splitlines()
        if line
    }


def assert_only_allowed_conflict(
    c002_paths: set[str], e006_paths: set[str], allowed: set[str]
) -> None:
    actual = c002_paths & e006_paths
    require(
        actual == allowed,
        f"cross-experiment changed-path intersection differs: {sorted(actual)}",
    )


def worktree_paths(repo_root: Path) -> set[str]:
    output = git_text(repo_root, ["worktree", "list", "--porcelain"])
    return {
        line.split(" ", 1)[1]
        for line in output.splitlines()
        if line.startswith("worktree ")
    }


def run_checked(command: Sequence[str], *, cwd: Path, label: str) -> dict[str, Any]:
    completed = subprocess.run(
        list(command), cwd=cwd, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise ReconciliationError(
            f"{label} failed ({completed.returncode}):\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return {"label": label, "command": list(command), "returncode": completed.returncode}


@contextmanager
def temporary_overlaid_worktree(
    repo_root: Path, *, label: str, archived_relative: str, expected_sha256: str
) -> Iterator[Path]:
    """Yield a detached checkout with only the historical conflict path overlaid."""

    parent = Path(tempfile.mkdtemp(prefix=f"v3-final-{label}-"))
    worktree = parent / "checkout"
    added = False
    try:
        completed = run_git(
            repo_root, ["worktree", "add", "--detach", str(worktree), "HEAD"]
        )
        if completed.returncode != 0:
            raise ReconciliationError(
                f"could not create {label} detached audit worktree: {completed.stderr.strip()}"
            )
        added = True
        source = repo_root / archived_relative
        target = worktree / CONFLICT_PATH
        require(source.is_file(), f"{label} archival copy is absent: {archived_relative}")
        shutil.copyfile(source, target)
        os.chmod(target, stat.S_IMODE(source.stat().st_mode))
        require(
            sha256_file(target) == expected_sha256,
            f"{label} overlay digest differs from its archival copy",
        )
        yield worktree
    finally:
        if added:
            completed = run_git(repo_root, ["worktree", "remove", "--force", str(worktree)])
            if completed.returncode != 0:
                raise ReconciliationError(
                    f"could not remove {label} detached audit worktree: {completed.stderr.strip()}"
                )
            prune = run_git(repo_root, ["worktree", "prune"])
            if prune.returncode != 0:
                raise ReconciliationError(
                    f"could not prune {label} detached audit worktree: {prune.stderr.strip()}"
                )
        shutil.rmtree(parent, ignore_errors=True)


def experiment_by_id(manifest: Mapping[str, Any], experiment_id: str) -> dict[str, Any]:
    rows = manifest.get("experiments")
    require(isinstance(rows, list), "integration manifest lacks experiment records")
    matches = [row for row in rows if isinstance(row, dict) and row.get("experiment_id") == experiment_id]
    require(len(matches) == 1, f"integration manifest lacks exactly one {experiment_id} record")
    return matches[0]


def validate_manifest_and_archives(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = read_json(repo_root / MANIFEST_RELATIVE)
    require(
        manifest.get("schema_version")
        == "vla-wam-shared-v3-final-merge-validator-reconciliation-v1",
        "integration manifest schema changed",
    )
    require(
        manifest.get("status") == "requires_nonmutating_parent_blob_validation",
        "integration manifest status changed",
    )
    base = str(manifest.get("required_base_commit", ""))
    ensure_commit(repo_root, base, "required base")
    ensure_ancestor(repo_root, base, "required base")

    allowed_rows = manifest.get("allowed_cross_experiment_changed_path_intersection")
    require(isinstance(allowed_rows, list) and all(isinstance(path, str) for path in allowed_rows), "allowed conflict list changed")
    allowed = set(allowed_rows)
    require(allowed == {CONFLICT_PATH}, "integration allows an unexpected conflict path")

    canonical = manifest.get("canonical_resolution")
    require(isinstance(canonical, dict), "canonical resolution is absent")
    require(canonical.get("path") == CONFLICT_PATH, "canonical resolution path changed")
    require(canonical.get("source_commit") == base, "canonical resolution source commit changed")
    base_blob = git_blob(repo_root, base, CONFLICT_PATH)
    require(len(base_blob) == canonical.get("bytes"), "base canonical validator bytes changed")
    require(sha256_bytes(base_blob) == canonical.get("sha256"), "base canonical validator SHA changed")
    verify_binding(repo_root, canonical, "canonical resolution")
    require(
        (repo_root / CONFLICT_PATH).read_bytes() == base_blob,
        "canonical validator differs from its pre-experiment base blob",
    )

    c002 = experiment_by_id(manifest, "V3-C002")
    e006 = experiment_by_id(manifest, "V3-E006")
    for experiment in (c002, e006):
        tip = str(experiment.get("branch_tip_commit", ""))
        source = str(experiment.get("frozen_validator_source_commit", ""))
        ensure_commit(repo_root, tip, f"{experiment['experiment_id']} branch tip")
        ensure_ancestor(repo_root, tip, f"{experiment['experiment_id']} branch tip")
        ensure_commit(repo_root, source, f"{experiment['experiment_id']} frozen source")
        frozen = experiment.get("frozen_validator")
        require(isinstance(frozen, dict), f"{experiment['experiment_id']} frozen validator is absent")
        require(frozen.get("canonical_path") == CONFLICT_PATH, f"{experiment['experiment_id']} conflict path changed")
        for commit, label in ((tip, "branch tip"), (source, "frozen source")):
            blob = git_blob(repo_root, commit, CONFLICT_PATH)
            require(
                len(blob) == frozen.get("bytes") and sha256_bytes(blob) == frozen.get("sha256"),
                f"{experiment['experiment_id']} {label} frozen blob changed",
            )
        archived = frozen.get("archival_copy")
        require(isinstance(archived, dict), f"{experiment['experiment_id']} archival copy is absent")
        require(
            archived.get("bytes") == frozen.get("bytes") and archived.get("sha256") == frozen.get("sha256"),
            f"{experiment['experiment_id']} archival metadata changed",
        )
        verify_binding(repo_root, archived, f"{experiment['experiment_id']} archival copy")
        require(
            (repo_root / str(archived["path"])).read_bytes()
            == git_blob(repo_root, tip, CONFLICT_PATH),
            f"{experiment['experiment_id']} archival copy differs from its branch-tip blob",
        )

    assert_only_allowed_conflict(
        branch_changed_paths(repo_root, base, str(c002["branch_tip_commit"])),
        branch_changed_paths(repo_root, base, str(e006["branch_tip_commit"])),
        allowed,
    )
    return manifest, c002, e006


def validate_c002_nonconflicting_bindings(repo_root: Path, experiment: Mapping[str, Any]) -> None:
    registration = read_json(repo_root / C002_REGISTRATION_RELATIVE)
    source_bindings = registration.get("source_bindings")
    require(isinstance(source_bindings, dict), "C002 registration lacks source bindings")
    for path, binding in source_bindings.items():
        require(isinstance(path, str) and isinstance(binding, dict), "C002 source binding is malformed")
        if path == CONFLICT_PATH:
            verify_frozen_conflict_binding(binding, experiment, "C002 registration")
        else:
            verify_binding(repo_root, binding, f"C002 source {path}")

    closure = read_json(repo_root / C002_CLOSURE_RELATIVE)
    require(
        closure.get("semantic_result") is False
        and closure.get("behavioral_execution_status") == "not_executed"
        and closure.get("behavioral_episode_count") == 0
        and closure.get("excluded_model_request_count") == 2
        and closure.get("full_queue_launched") is False
        and closure.get("release_authorized") is False
        and closure.get("retry_performed") is False,
        "C002 closure status/count boundary changed",
    )
    artifacts = closure.get("closure_artifacts")
    require(isinstance(artifacts, dict), "C002 closure lacks artifact bindings")
    for label, binding in artifacts.items():
        require(isinstance(binding, dict), f"C002 closure binding is malformed: {label}")
        if binding.get("path") == CONFLICT_PATH:
            verify_frozen_conflict_binding(binding, experiment, f"C002 closure {label}")
        else:
            verify_binding(repo_root, binding, f"C002 closure {label}")
    raw = closure.get("raw_evidence_rehash")
    require(
        isinstance(raw, dict)
        and raw.get("passed") is True
        and raw.get("unique_raw_bindings_rehashed") == 81
        and raw.get("raw_bytes_rehashed") == 26_518_210,
        "C002 target raw rehash summary changed",
    )


def validate_e006_nonconflicting_bindings(repo_root: Path, experiment: Mapping[str, Any]) -> None:
    evidence = read_json(repo_root / E006_EVIDENCE_RELATIVE)
    require(
        evidence.get("status") == "gate_failed_no_valid_candidate_stop_before_registration"
        and evidence.get("model_request_count") == 0
        and evidence.get("behavioral_episode_count") == 0
        and evidence.get("state_candidate_count") == 0
        and evidence.get("registration_created") is False,
        "E006 evidence stop boundary changed",
    )
    local_files = evidence.get("local_files")
    require(isinstance(local_files, list), "E006 evidence lacks local bindings")
    for index, binding in enumerate(local_files):
        require(isinstance(binding, dict), f"E006 local binding is malformed: {index}")
        if binding.get("path") == CONFLICT_PATH:
            verify_frozen_conflict_binding(binding, experiment, "E006 evidence")
        else:
            verify_binding(repo_root, binding, f"E006 local file {index}")

    results = read_json(repo_root / E006_RESULTS_RELATIVE)
    release = read_json(repo_root / E006_RELEASE_RELATIVE)
    for payload, label in ((results, "results"), (release, "release")):
        require(
            payload.get("model_request_count") == 0
            and payload.get("behavioral_episode_count") == 0
            and payload.get("state_candidate_count") == 0,
            f"E006 {label} count boundary changed",
        )
    require(
        results.get("status") == "gate_failed_no_valid_candidate_stop_before_registration"
        and release.get("release_for_inference") is False,
        "E006 release boundary changed",
    )


def run_overlay_checks(
    repo_root: Path, c002: Mapping[str, Any], e006: Mapping[str, Any], *, verify_raw: bool
) -> list[dict[str, Any]]:
    before = worktree_paths(repo_root)
    records: list[dict[str, Any]] = []
    c002_frozen = c002["frozen_validator"]
    with temporary_overlaid_worktree(
        repo_root,
        label="c002",
        archived_relative=str(c002_frozen["archival_copy"]["path"]),
        expected_sha256=str(c002_frozen["sha256"]),
    ) as worktree:
        records.append(
            run_checked(
                [sys.executable, str(worktree / "tools/validate_v3c002_isolation_closure.py")],
                cwd=worktree,
                label="C002 frozen-identity closure validator",
            )
        )
        records.append(
            run_checked(
                [sys.executable, str(worktree / "tools/validate_v3c002_publication_bundle.py")],
                cwd=worktree,
                label="C002 frozen-identity publication bundle",
            )
        )

    e006_frozen = e006["frozen_validator"]
    with temporary_overlaid_worktree(
        repo_root,
        label="e006",
        archived_relative=str(e006_frozen["archival_copy"]["path"]),
        expected_sha256=str(e006_frozen["sha256"]),
    ) as worktree:
        e006_command = [sys.executable, str(worktree / "tools/validate_v3e006.py")]
        infra_command = [
            sys.executable,
            str(worktree / "tools/validate_v3e006_infrastructure_evidence.py"),
            "--repo-root",
            str(worktree),
        ]
        if verify_raw:
            e006_command.append("--verify-raw")
            infra_command.append("--verify-raw")
        records.append(
            run_checked(
                e006_command,
                cwd=worktree,
                label="E006 frozen-identity stop validator",
            )
        )
        records.append(
            run_checked(
                infra_command,
                cwd=worktree,
                label="E006 frozen-identity infrastructure validator",
            )
        )
        records.append(
            run_checked(
                [sys.executable, str(worktree / CONFLICT_PATH)],
                cwd=worktree,
                label="E006 frozen-identity publication bundle",
            )
        )
    after = worktree_paths(repo_root)
    require(after == before, "detached overlay worktrees were not completely removed")
    return records


def validate_reconciliation(
    repo_root: Path = REPO_ROOT, *, run_overlays: bool = True, verify_raw: bool = False
) -> dict[str, Any]:
    root = repo_root.resolve()
    manifest, c002, e006 = validate_manifest_and_archives(root)
    validate_c002_nonconflicting_bindings(root, c002)
    validate_e006_nonconflicting_bindings(root, e006)
    overlays = run_overlay_checks(root, c002, e006, verify_raw=verify_raw) if run_overlays else []
    return {
        "status": "valid_nonmutating_v3_final_merge_reconciliation",
        "canonical_validator_sha256": manifest["canonical_resolution"]["sha256"],
        "c002_frozen_validator_sha256": c002["frozen_validator"]["sha256"],
        "e006_frozen_validator_sha256": e006["frozen_validator"]["sha256"],
        "verified_nonconflicting_bindings": True,
        "overlay_validators_run": [record["label"] for record in overlays],
        "raw_verified": verify_raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument(
        "--verify-raw",
        action="store_true",
        help="Require E006 target-PVC raw paths while running its disposable overlay.",
    )
    args = parser.parse_args()
    try:
        value = validate_reconciliation(
            args.repo_root, run_overlays=not args.no_overlay, verify_raw=args.verify_raw
        )
    except ReconciliationError as exc:
        raise SystemExit(f"final merge reconciliation invalid: {exc}") from exc
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
