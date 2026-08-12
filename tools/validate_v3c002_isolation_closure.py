#!/usr/bin/env python3
"""Validate the portable V3-C002 failed-isolation closure metadata."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE = (
    REPO_ROOT
    / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002/active"
)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def same_target_binding(record: object, local_path: Path, label: str) -> None:
    require(isinstance(record, dict), f"{label} binding is absent")
    require(record.get("bytes") == local_path.stat().st_size, f"{label} bytes changed")
    require(record.get("sha256") == sha256_file(local_path), f"{label} digest changed")


def same_repo_binding(record: object, local_path: Path, label: str) -> None:
    require(isinstance(record, dict), f"{label} binding is absent")
    expected = local_path.relative_to(REPO_ROOT).as_posix()
    require(record.get("path") == expected, f"{label} path changed")
    same_target_binding(record, local_path, label)


def main() -> None:
    gates = ACTIVE / "gates"
    report_path = gates / "isolation_failure_report.json"
    receipt_path = gates / "isolation_target_raw_rehash_receipt.json"
    ledger_path = gates / "isolation_prelaunch_invalid_attempts.jsonl"
    report = read_json(report_path)
    receipt = read_json(receipt_path)

    require(
        report.get("schema_version")
        == "vla-wam-shared-v3c002-two-lane-isolation-failure-v1"
        and report.get("status") == "failed_registered_exact_action_equality_gate"
        and report.get("passed") is False,
        "registered isolation failure record changed",
    )
    require(
        report.get("model_request_count") == 2
        and report.get("behavioral_episode_count") == 0
        and report.get("excluded_from_behavioral_denominators") is True,
        "isolation counts or denominator status changed",
    )
    require(report.get("no_retry_performed") is True, "isolation retry status changed")
    require(
        report.get("release_authorized") is False
        and report.get("full_behavior_authorized") is False,
        "failed isolation authorized behavior",
    )
    require(
        all(report.get(key) is True for key in ("fixed_observation_equal", "fixed_prompt_equal", "request_seed_equal")),
        "isolation inputs were not fixed",
    )
    hashes = report.get("action_sha256_by_lane")
    require(
        report.get("actions_exactly_equal") is False
        and isinstance(hashes, list)
        and len(hashes) == 2
        and len(set(hashes)) == 2,
        "action inequality is not retained",
    )
    require(
        report.get("max_absolute_action_difference") == 0.0013794898986816406
        and report.get("mean_absolute_action_difference") == 0.0002258223103126511,
        "registered action-difference summary changed",
    )

    require(
        receipt.get("schema_version")
        == "vla-wam-shared-v3c002-isolation-target-raw-rehash-v1"
        and receipt.get("status") == "passed_target_rehash_of_failed_isolation_evidence"
        and receipt.get("passed") is True,
        "target raw rehash receipt did not pass",
    )
    require(
        receipt.get("isolation_gate_passed") is False
        and receipt.get("release_authorized") is False
        and receipt.get("full_behavior_authorized") is False,
        "target receipt misstates release status",
    )
    require(
        receipt.get("model_request_count") == 2
        and receipt.get("behavioral_episode_count") == 0
        and receipt.get("action_sha256_by_lane") == hashes,
        "target receipt differs from the failure report",
    )
    same_target_binding(receipt.get("failure_report"), report_path, "failure report")
    require(
        receipt.get("unique_raw_bindings_rehashed") == 81
        and receipt.get("raw_bytes_rehashed") == 26_518_210,
        "target raw rehash coverage changed",
    )

    local_sources = {
        "registration": ACTIVE / "registration.json",
        "queue": ACTIVE / "queue.jsonl",
        "excluded_smoke_gate": gates / "excluded_smoke_gate.json",
        "registered_isolation_runner": REPO_ROOT
        / "experiments/v3/phase_c_semantic_equivalence_v3c002/fixed_observation_isolation.py",
        "registered_isolation_compiler": REPO_ROOT / "tools/compile_v3c002_isolation_gate.py",
    }
    for key, path in local_sources.items():
        same_target_binding(report.get(key), path, key)
    require(not (gates / "two_lane_isolation_gate.json").exists(), "passed isolation gate must remain absent")
    release = read_json(ACTIVE / "release_gate.json")
    require(
        release.get("passed") is False
        and release.get("status") == "blocked_pending_committed_source_and_runtime_preflight_gates",
        "fail-closed release record changed",
    )
    require(not (ACTIVE / "results").exists(), "behavioral results exist after failed isolation")

    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    require(len(rows) == 4, "isolation prelaunch invalid ledger changed")
    require(
        all(
            row.get("model_request_count") == 0
            and row.get("behavioral_episode_count") == 0
            and row.get("denominator_eligible") is False
            for row in rows
        ),
        "isolation prelaunch invalid ledger contains behavior",
    )
    full_ledger = [
        json.loads(line)
        for line in (ACTIVE / "infrastructure_attempts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    require(len(full_ledger) == 7 and full_ledger[-4:] == rows, "active infrastructure ledger omits isolation invalids")

    closure = (ACTIVE / "ISOLATION_CLOSURE_MEMO.md").read_text(encoding="utf-8")
    routing = (ACTIVE / "C002_MANUSCRIPT_ROUTING.md").read_text(encoding="utf-8")
    closure_dir = ACTIVE / "closure"
    insert_path = closure_dir / "MANUSCRIPT_INSERT.md"
    manifest_path = closure_dir / "evidence_manifest.json"
    insert = insert_path.read_text(encoding="utf-8")
    manifest = read_json(manifest_path)
    for phrase in (
        "not a semantic result",
        "No behavioral episode was run",
        "No passed two-lane isolation gate",
    ):
        require(phrase in closure, f"closure memo omits: {phrase}")
    for phrase in ("was not behaviorally executed", "not a semantic result", "never launched"):
        require(phrase in routing, f"manuscript routing omits: {phrase}")
    for phrase in (
        "no manuscript result",
        "no behavioral episodes were run",
        "methods or limitations",
        "It is not a semantic result",
        sha256_file(report_path),
        sha256_file(receipt_path),
    ):
        require(phrase in insert, f"manuscript insert omits: {phrase}")
    require("|" not in insert, "manuscript insert must not contain a result table")

    require(
        manifest.get("schema_version")
        == "vla-wam-shared-v3c002-prebehavior-isolation-closure-evidence-manifest-v1"
        and manifest.get("status") == "closed_before_behavior_after_failed_registered_isolation",
        "closure evidence manifest schema or status changed",
    )
    require(
        manifest.get("semantic_result") is False
        and manifest.get("behavioral_execution_status") == "not_executed"
        and manifest.get("behavioral_episode_count") == 0
        and manifest.get("excluded_model_request_count") == 2
        and manifest.get("full_queue_launched") is False
        and manifest.get("release_authorized") is False
        and manifest.get("retry_performed") is False,
        "closure evidence manifest counts or authorization changed",
    )
    require(
        manifest.get("source_commit") == "e2d9ae3904b4a08e549c784903c167a4213d3d47",
        "closure evidence manifest source commit changed",
    )
    publication = manifest.get("publication_routing")
    require(
        isinstance(publication, dict)
        and publication.get("main_results_insert_authorized") is False
        and publication.get("result_table_authorized") is False
        and publication.get("semantic_claim_authorized") is False
        and publication.get("allowed_scope")
        == "methods_or_limitations_failed_pre_release_infrastructure_check_only",
        "closure evidence manifest publication routing changed",
    )
    raw_rehash = manifest.get("raw_evidence_rehash")
    require(
        isinstance(raw_rehash, dict)
        and raw_rehash.get("receipt") == "isolation_target_raw_rehash_receipt"
        and raw_rehash.get("unique_raw_bindings_rehashed") == 81
        and raw_rehash.get("raw_bytes_rehashed") == 26_518_210
        and raw_rehash.get("passed") is True,
        "closure evidence manifest raw rehash summary changed",
    )
    artifacts = manifest.get("closure_artifacts")
    require(isinstance(artifacts, dict), "closure evidence manifest bindings are absent")
    bound_paths = {
        "registration": ACTIVE / "registration.json",
        "queue": ACTIVE / "queue.jsonl",
        "release_gate": ACTIVE / "release_gate.json",
        "physical_gate": gates / "model_blind_physical_gate.json",
        "excluded_smoke_gate": gates / "excluded_smoke_gate.json",
        "isolation_failure_report": report_path,
        "isolation_target_raw_rehash_receipt": receipt_path,
        "isolation_prelaunch_invalid_attempts": ledger_path,
        "infrastructure_attempts": ACTIVE / "infrastructure_attempts.jsonl",
        "closure_memo": ACTIVE / "ISOLATION_CLOSURE_MEMO.md",
        "manuscript_routing": ACTIVE / "C002_MANUSCRIPT_ROUTING.md",
        "manuscript_insert": insert_path,
        "frozen_publication_bundle_validator": REPO_ROOT / "tools/validate_v3e_publication_bundle.py",
        "closure_validator": REPO_ROOT / "tools/validate_v3c002_isolation_closure.py",
        "additive_publication_bundle_validator": REPO_ROOT / "tools/validate_v3c002_publication_bundle.py",
    }
    require(set(artifacts) == set(bound_paths), "closure evidence manifest binding set changed")
    for label, path in bound_paths.items():
        same_repo_binding(artifacts.get(label), path, label)

    print(
        json.dumps(
            {
                "status": "valid_registered_c002_closed_before_behavior_after_failed_isolation",
                "model_requests_excluded": 2,
                "behavioral_episodes": 0,
                "failure_report_sha256": sha256_file(report_path),
                "target_raw_rehash_receipt_sha256": sha256_file(receipt_path),
                "evidence_manifest_sha256": sha256_file(manifest_path),
                "manuscript_insert_sha256": sha256_file(insert_path),
                "full_queue_launched": False,
                "semantic_result": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
