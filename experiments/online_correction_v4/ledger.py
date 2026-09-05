"""Compile write-once attempt directories into the V4 accepted ledger."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.online_correction_v4.analysis import (
    VERIFIED_COMMON_PREFIX_MODES,
    digest,
    digest_bytes,
    read_jsonl,
    validate_accepted_ledger,
    validate_c2_result_record,
)
from experiments.online_correction_v4.contracts import (
    ACCEPTED_LEDGER_MANIFEST_SCHEMA_VERSION,
    ACCEPTED_LEDGER_SCHEMA_VERSION,
    ATTEMPT_SELECTION_RULE,
    REJECTED_ATTEMPT_SCHEMA_VERSION,
)
from experiments.online_correction_v4.recorder import canonical_json_bytes, digest_bytes as recorder_digest_bytes

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
C2_PREFIX_FIELDS: tuple[str, ...] = (
    "common_prefix_verification_mode",
    "common_prefix_verification_receipt_sha256",
    "common_prefix_identity_hash_sha256",
)
C2_RESPONSE_OUTCOME_FIELDS: tuple[str, ...] = (
    "response_goal_violation_capped_m",
    "response_horizon_s",
    "response_anchor",
    "response_goal_set_branch",
    "response_goal_set_hash_sha256",
    "response_projection",
    "response_scorer_sha256",
)


class LedgerError(ValueError):
    """Raised when attempt evidence cannot be compiled into an accepted ledger."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LedgerError(message)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LedgerError(f"{path}: expected JSON object")
    return payload


def _load_json_rows(path: Path, *, key: str) -> list[dict[str, Any]]:
    payload = _load_json(path)
    rows = payload.get(key, payload.get("rows"))
    if not isinstance(rows, list):
        raise LedgerError(f"{path}: expected list under {key!r} or 'rows'")
    return [dict(row) for row in rows if isinstance(row, dict)]


@dataclass
class ParsedAttempt:
    attempt_path: Path
    episode_id: str
    attempt_id: str
    complete: dict[str, Any]
    episode: dict[str, Any]
    evidence_manifest: dict[str, Any]
    verification_errors: list[str] = field(default_factory=list)

    @property
    def terminal_status(self) -> str:
        return str(self.complete.get("status") or self.episode.get("status") or "")

    @property
    def evidence_verified(self) -> bool:
        return not self.verification_errors


@dataclass
class LedgerCompileResult:
    accepted_rows: list[dict[str, Any]]
    rejected_rows: list[dict[str, Any]]
    manifest_payload: dict[str, Any]
    reconciliation: dict[str, Any]
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def verify_evidence_manifest(attempt_path: Path, manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    episode_path = attempt_path / "episode.json"
    if not episode_path.is_file():
        errors.append("episode.json is missing")
        return errors

    episode = _load_json(episode_path)
    expected_episode_sha = manifest.get("episode_sha256")
    if expected_episode_sha != recorder_digest_bytes(canonical_json_bytes(episode)):
        errors.append("episode_sha256 mismatch")

    component_checks = (
        ("trajectory.json", "rows", "trajectory_sha256"),
        ("requests.json", "rows", "requests_sha256"),
        ("observations.json", "rows", "observations_sha256"),
        ("events.json", "rows", "events_sha256"),
    )
    for filename, row_key, manifest_key in component_checks:
        path = attempt_path / filename
        if not path.is_file():
            errors.append(f"{filename} is missing")
            continue
        rows = _load_json_rows(path, key=row_key)
        expected = manifest.get(manifest_key)
        actual = recorder_digest_bytes(canonical_json_bytes(rows))
        if expected != actual:
            errors.append(f"{manifest_key} mismatch for {filename}")

    for optional_file, manifest_key in (
        ("future_artifacts.json", "future_artifacts_sha256"),
        ("viewport_frames.json", "viewport_frames_sha256"),
    ):
        path = attempt_path / optional_file
        if not path.is_file():
            continue
        rows = _load_json_rows(path, key="rows")
        expected = manifest.get(manifest_key)
        actual = recorder_digest_bytes(canonical_json_bytes(rows))
        if expected != actual:
            errors.append(f"{manifest_key} mismatch for {optional_file}")

    blobs = manifest.get("blobs")
    if not isinstance(blobs, list):
        errors.append("evidence_manifest.blobs must be a list")
    else:
        for index, blob in enumerate(blobs):
            if not isinstance(blob, dict):
                errors.append(f"blob {index}: must be an object")
                continue
            rel = blob.get("relative_path")
            expected_sha = blob.get("sha256")
            if not isinstance(rel, str) or not rel:
                errors.append(f"blob {index}: relative_path required")
                continue
            blob_path = attempt_path / rel
            if not blob_path.is_file():
                errors.append(f"blob {index}: missing file {rel}")
                continue
            payload = blob_path.read_bytes()
            actual_sha = recorder_digest_bytes(payload)
            if expected_sha != actual_sha:
                errors.append(f"blob {index}: sha256 mismatch for {rel}")

    complete_path = attempt_path / "COMPLETE.json"
    if complete_path.is_file():
        complete = _load_json(complete_path)
        expected_manifest_sha = complete.get("evidence_manifest_sha256")
        actual_manifest_sha = recorder_digest_bytes(canonical_json_bytes(dict(manifest)))
        if expected_manifest_sha != actual_manifest_sha:
            errors.append("COMPLETE.json evidence_manifest_sha256 mismatch")

    return errors


def load_finalized_attempt(attempt_path: Path) -> ParsedAttempt:
    complete_path = attempt_path / "COMPLETE.json"
    manifest_path = attempt_path / "evidence_manifest.json"
    episode_path = attempt_path / "episode.json"
    _require(complete_path.is_file(), f"{attempt_path}: COMPLETE.json is required")
    _require(manifest_path.is_file(), f"{attempt_path}: evidence_manifest.json is required")
    _require(episode_path.is_file(), f"{attempt_path}: episode.json is required")

    complete = _load_json(complete_path)
    evidence_manifest = _load_json(manifest_path)
    episode = _load_json(episode_path)
    episode_id = str(complete.get("episode_id") or episode.get("episode_id") or attempt_path.parent.name)
    attempt_id = str(complete.get("attempt_id") or episode.get("attempt_id") or attempt_path.name)
    verification_errors = verify_evidence_manifest(attempt_path, evidence_manifest)
    return ParsedAttempt(
        attempt_path=attempt_path,
        episode_id=episode_id,
        attempt_id=attempt_id,
        complete=complete,
        episode=episode,
        evidence_manifest=evidence_manifest,
        verification_errors=verification_errors,
    )


def discover_finalized_attempts(attempts_root: Path) -> list[ParsedAttempt]:
    if not attempts_root.is_dir():
        raise LedgerError(f"attempts root is not a directory: {attempts_root}")
    discovered: list[ParsedAttempt] = []
    for episode_dir in sorted(path for path in attempts_root.iterdir() if path.is_dir()):
        for attempt_dir in sorted(path for path in episode_dir.iterdir() if path.is_dir()):
            if (attempt_dir / "COMPLETE.json").is_file():
                discovered.append(load_finalized_attempt(attempt_dir))
    return discovered


def _provenance_block(
    parsed: ParsedAttempt,
    *,
    attempts_root: Path,
    selection_rule: str = ATTEMPT_SELECTION_RULE,
) -> dict[str, Any]:
    rel_path = parsed.attempt_path.relative_to(attempts_root)
    return {
        "attempt_root_uri": str(rel_path),
        "complete_receipt_sha256": digest_bytes((parsed.attempt_path / "COMPLETE.json").read_bytes()),
        "evidence_manifest_sha256": digest(parsed.evidence_manifest),
        "episode_record_sha256": digest(parsed.episode),
        "attempt_selection_rule": selection_rule,
    }


def _resolve_provenance_hashes(
    parsed: ParsedAttempt,
    *,
    protocol_sha256: str,
    scorer_sha256: str,
) -> tuple[str, str, str]:
    provenance = parsed.episode.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    episode_protocol = provenance.get("protocol_sha256") or parsed.episode.get("protocol_sha256") or protocol_sha256
    episode_scorer = provenance.get("scorer_sha256") or parsed.episode.get("scorer_sha256") or scorer_sha256
    config_sha256 = provenance.get("config_sha256") or parsed.episode.get("config_sha256") or ""
    return str(episode_protocol), str(episode_scorer), str(config_sha256)


def _video_fields(episode: Mapping[str, Any], *, attempt_path: Path, attempts_root: Path) -> tuple[str, str]:
    viewport = episode.get("viewport_video")
    if isinstance(viewport, dict):
        video_uri = str(viewport.get("video_uri") or "")
        video_sha256 = str(viewport.get("video_sha256") or "")
        if video_uri and not video_uri.startswith("/"):
            return video_uri, video_sha256
        if video_uri:
            return str((attempt_path / video_uri).relative_to(attempts_root)), video_sha256
    return str(attempt_path.relative_to(attempts_root) / "viewport_video.bin"), ""


def _outcome_from_episode(episode: Mapping[str, Any], *, family: str) -> dict[str, Any]:
    outcome = episode.get("outcome")
    if isinstance(outcome, dict):
        payload = dict(outcome)
    else:
        payload = {}
        terminal = episode.get("terminal_scoring")
        if isinstance(terminal, dict):
            for key in (
                "failure_label",
                "failure_stage",
                "goal_violation_capped_m",
                "goal_set_empty",
                "goal_violation_cap_applied",
            ):
                if key in terminal and key not in payload:
                    payload[key] = terminal[key]
    if "failure_stage" not in payload:
        payload["failure_stage"] = episode.get("failure_stage", "other")
    if "goal_violation_capped_m" not in payload and "goal_violation_capped_m" in episode:
        payload["goal_violation_capped_m"] = episode["goal_violation_capped_m"]
    if family == "C2":
        for key in C2_RESPONSE_OUTCOME_FIELDS:
            if key in episode and key not in payload:
                payload[key] = episode[key]
    return payload


def _c2_contract_complete(record: Mapping[str, Any], *, config: Mapping[str, Any] | None) -> bool:
    return not validate_c2_result_record(record, label="record", config=config)


def build_infra_invalid_inventory_row(
    parsed: ParsedAttempt,
    *,
    attempts_root: Path,
    manifest_row: Mapping[str, Any] | None = None,
    rejection_class: str = "infra_invalid",
) -> dict[str, Any]:
    reason = (
        parsed.complete.get("infra_invalid_reason")
        or parsed.complete.get("reason")
        or parsed.episode.get("infra_invalid_reason")
        or ""
    )
    row = {
        "schema_version": REJECTED_ATTEMPT_SCHEMA_VERSION,
        "record_type": "rejected_attempt",
        "episode_id": parsed.episode_id,
        "attempt_id": parsed.attempt_id,
        "status": "infra_invalid",
        "rejection_class": rejection_class,
        "reason": str(reason),
        "detail": parsed.complete.get("detail") or parsed.episode.get("detail") or "",
        "verification_errors": list(parsed.verification_errors),
        "provenance": _provenance_block(parsed, attempts_root=attempts_root),
    }
    if manifest_row is not None:
        row["family"] = manifest_row.get("family")
        row["block_id"] = manifest_row.get("block_id")
    return row


def build_accepted_row(
    parsed: ParsedAttempt,
    manifest_row: Mapping[str, Any],
    *,
    attempts_root: Path,
    protocol_sha256: str,
    scorer_sha256: str,
) -> dict[str, Any]:
    episode = parsed.episode
    protocol_sha, scorer_sha, _ = _resolve_provenance_hashes(
        parsed, protocol_sha256=protocol_sha256, scorer_sha256=scorer_sha256
    )
    video_uri, video_sha256 = _video_fields(episode, attempt_path=parsed.attempt_path, attempts_root=attempts_root)
    trace_uri = str(parsed.attempt_path.relative_to(attempts_root) / "trajectory.json")
    outcome = _outcome_from_episode(episode, family=str(manifest_row["family"]))
    row: dict[str, Any] = {
        "schema_version": ACCEPTED_LEDGER_SCHEMA_VERSION,
        "episode_id": parsed.episode_id,
        "attempt_id": parsed.attempt_id,
        "status": "valid",
        "family": manifest_row["family"],
        "config_sha256": manifest_row["config_sha256"],
        "prefix_group_id": manifest_row["prefix_group_id"],
        "reuse_episode_ids": list(manifest_row.get("reuse_episode_ids") or []),
        "success": bool(episode.get("success", False)),
        "trigger_eligible": bool(episode.get("trigger_eligible", False)),
        "event_delivered": bool(episode.get("event_delivered", False)),
        "event_observed": bool(episode.get("event_observed", False)),
        "motion_truncated_by_release": bool(episode.get("motion_truncated_by_release", False)),
        "outcome": outcome,
        "timing": dict(episode.get("timing") or {}),
        "trace_uri": trace_uri,
        "video_uri": video_uri,
        "trace_sha256": parsed.evidence_manifest["trajectory_sha256"],
        "video_sha256": video_sha256 or parsed.evidence_manifest.get("viewport_video_sha256", ""),
        "scorer_sha256": scorer_sha,
        "protocol_sha256": protocol_sha,
        "provenance": _provenance_block(parsed, attempts_root=attempts_root),
    }
    if episode.get("intervention_exposure"):
        row["intervention_exposure"] = dict(episode["intervention_exposure"])
    if episode.get("motion"):
        row["motion"] = dict(episode["motion"])
    if manifest_row["family"] == "C2":
        for key in C2_PREFIX_FIELDS:
            if key in episode:
                row[key] = episode[key]
    return row


def select_accepted_attempts(
    grouped: Mapping[str, Sequence[ParsedAttempt]],
) -> tuple[dict[str, ParsedAttempt], list[dict[str, Any]], list[str]]:
    """Select at most one verified valid attempt per episode without outcome peeking."""
    selected: dict[str, ParsedAttempt] = {}
    rejected: list[dict[str, Any]] = []
    errors: list[str] = []

    for episode_id in sorted(grouped):
        attempts = list(grouped[episode_id])
        attempt_ids = [attempt.attempt_id for attempt in attempts]
        if len(set(attempt_ids)) != len(attempt_ids):
            errors.append(f"{episode_id}: duplicate attempt_id values across directories")

        verified_valid = [
            attempt
            for attempt in attempts
            if attempt.terminal_status == "valid" and attempt.evidence_verified
        ]
        verified_infra = [
            attempt
            for attempt in attempts
            if attempt.terminal_status == "infra_invalid" and attempt.evidence_verified
        ]
        corrupted = [attempt for attempt in attempts if not attempt.evidence_verified]

        if verified_valid:
            chosen = max(verified_valid, key=lambda attempt: attempt.attempt_id)
            selected[episode_id] = chosen
            for attempt in verified_valid:
                if attempt.attempt_id != chosen.attempt_id:
                    rejected.append(
                        {
                            "episode_id": episode_id,
                            "attempt_id": attempt.attempt_id,
                            "status": "valid",
                            "rejection_class": "superseded_valid_attempt",
                            "reason": ATTEMPT_SELECTION_RULE,
                            "superseded_by_attempt_id": chosen.attempt_id,
                        }
                    )

        for attempt in verified_infra:
            rejected.append(
                {
                    "episode_id": episode_id,
                    "attempt_id": attempt.attempt_id,
                    "status": "infra_invalid",
                    "rejection_class": "infra_invalid",
                    "reason": attempt.complete.get("infra_invalid_reason")
                    or attempt.complete.get("reason")
                    or "",
                }
            )

        for attempt in corrupted:
            rejected.append(
                {
                    "episode_id": episode_id,
                    "attempt_id": attempt.attempt_id,
                    "status": attempt.terminal_status or "corrupted",
                    "rejection_class": "evidence_corruption",
                    "reason": "evidence verification failed",
                    "verification_errors": list(attempt.verification_errors),
                }
            )

        unverified_valid = [
            attempt
            for attempt in attempts
            if attempt.terminal_status == "valid" and not attempt.evidence_verified
        ]
        if unverified_valid and episode_id not in selected:
            for attempt in unverified_valid:
                if not any(
                    item.get("attempt_id") == attempt.attempt_id and item.get("rejection_class") == "evidence_corruption"
                    for item in rejected
                ):
                    rejected.append(
                        {
                            "episode_id": episode_id,
                            "attempt_id": attempt.attempt_id,
                            "status": "valid",
                            "rejection_class": "evidence_corruption",
                            "reason": "valid terminal status with failed evidence verification",
                            "verification_errors": list(attempt.verification_errors),
                        }
                    )

    return selected, rejected, errors


def reconcile_queue_and_controls(
    manifest: Sequence[Mapping[str, Any]],
    accepted_rows: Mapping[str, Mapping[str, Any]],
    *,
    queue_episode_ids: set[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    manifest_by_id = {row["episode_id"]: row for row in manifest}
    errors: list[str] = []
    if len(manifest_by_id) != len(manifest):
        errors.append("manifest contains duplicate episode_id values")

    missing = sorted(episode_id for episode_id in manifest_by_id if episode_id not in accepted_rows)
    unknown = sorted(episode_id for episode_id in accepted_rows if episode_id not in manifest_by_id)
    if unknown:
        errors.append(f"accepted rows reference unknown manifest episodes: {unknown}")

    if queue_episode_ids is not None:
        extra_attempts = sorted(set(accepted_rows) - queue_episode_ids)
        if extra_attempts:
            errors.append(f"accepted episodes missing from frozen queue: {extra_attempts[:5]}")

    control_links: list[dict[str, Any]] = []
    missing_controls: list[str] = []
    for episode_id, row in sorted(accepted_rows.items()):
        manifest_row = manifest_by_id[episode_id]
        reuse_ids = list(manifest_row.get("reuse_episode_ids") or [])
        unresolved = [control_id for control_id in reuse_ids if control_id not in accepted_rows]
        if unresolved:
            missing_controls.extend(f"{episode_id}->{control_id}" for control_id in unresolved)
        control_links.append(
            {
                "episode_id": episode_id,
                "family": manifest_row["family"],
                "reuse_episode_ids": reuse_ids,
                "reuse_sources_accepted": not unresolved,
                "missing_control_ids": unresolved,
            }
        )

    if missing_controls:
        errors.append(
            "control reuse requires accepted source episodes: "
            + ", ".join(sorted(set(missing_controls))[:8])
        )

    by_family = defaultdict(lambda: {"planned": 0, "accepted_valid": 0, "missing_valid": 0})
    for episode_id, manifest_row in manifest_by_id.items():
        family = manifest_row["family"]
        by_family[family]["planned"] += 1
        if episode_id in accepted_rows:
            by_family[family]["accepted_valid"] += 1
        else:
            by_family[family]["missing_valid"] += 1

    reconciliation = {
        "planned_episodes": len(manifest_by_id),
        "accepted_valid_unique": len(accepted_rows),
        "missing_valid": len(missing),
        "missing_episode_ids": missing,
        "unknown_accepted_episode_ids": unknown,
        "by_family": dict(sorted(by_family.items())),
        "control_reuse_links": control_links,
        "missing_control_links": sorted(set(missing_controls)),
    }
    return reconciliation, errors


def compile_accepted_ledger_from_attempts(
    *,
    manifest: Sequence[Mapping[str, Any]],
    attempts: Sequence[ParsedAttempt],
    attempts_root: Path,
    protocol_sha256: str,
    scorer_sha256: str,
    config: Mapping[str, Any] | None = None,
    queue_episode_ids: set[str] | None = None,
    require_full_coverage: bool = False,
) -> LedgerCompileResult:
    manifest_by_id = {row["episode_id"]: row for row in manifest}
    grouped: dict[str, list[ParsedAttempt]] = defaultdict(list)
    errors: list[str] = []
    warnings: list[str] = []

    for attempt in attempts:
        if attempt.episode_id not in manifest_by_id:
            errors.append(f"{attempt.attempt_id}: episode_id {attempt.episode_id!r} is not in manifest")
        grouped[attempt.episode_id].append(attempt)

    selected, selection_rejected, selection_errors = select_accepted_attempts(grouped)
    errors.extend(selection_errors)

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    c2_incomplete: list[str] = []

    for episode_id in sorted(selected):
        parsed = selected[episode_id]
        manifest_row = manifest_by_id[episode_id]
        accepted_rows.append(
            build_accepted_row(
                parsed,
                manifest_row,
                attempts_root=attempts_root,
                protocol_sha256=protocol_sha256,
                scorer_sha256=scorer_sha256,
            )
        )
        row = accepted_rows[-1]
        if manifest_row["family"] == "C2":
            if not _c2_contract_complete(row, config=config):
                c2_incomplete.append(episode_id)
                warnings.append(
                    f"{episode_id}: C2 accepted row lacks complete prefix/response contract; "
                    "confirmatory C2 analysis remains fail-closed"
                )

    accepted_by_id = {row["episode_id"]: row for row in accepted_rows}
    reconciliation, reconcile_errors = reconcile_queue_and_controls(
        manifest,
        accepted_by_id,
        queue_episode_ids=queue_episode_ids,
    )
    errors.extend(reconcile_errors)
    if require_full_coverage and reconciliation["missing_valid"]:
        errors.append(
            f"incomplete allocation: {reconciliation['missing_valid']} manifest cells lack accepted valid rows"
        )

    for item in selection_rejected:
        parsed = next(
            (
                attempt
                for attempt in grouped[item["episode_id"]]
                if attempt.attempt_id == item["attempt_id"]
            ),
            None,
        )
        manifest_row = manifest_by_id.get(item["episode_id"])
        if parsed is not None and item.get("rejection_class") == "infra_invalid":
            rejected_rows.append(
                build_infra_invalid_inventory_row(
                    parsed,
                    attempts_root=attempts_root,
                    manifest_row=manifest_row,
                )
            )
        else:
            rejected_rows.append(
                {
                    "schema_version": REJECTED_ATTEMPT_SCHEMA_VERSION,
                    "record_type": "rejected_attempt",
                    **item,
                    "provenance": _provenance_block(parsed, attempts_root=attempts_root)
                    if parsed is not None
                    else {},
                }
            )

    validation = validate_accepted_ledger(manifest, accepted_rows, config=config)
    validation_preview = {
        "ok": validation.get("ok"),
        "error_count": len(validation.get("errors", [])),
        "c2_incomplete_episode_ids": c2_incomplete,
    }
    if c2_incomplete:
        validation_preview["ok"] = False
        validation_preview["c2_fail_closed"] = True

    manifest_payload = {
        "schema_version": ACCEPTED_LEDGER_MANIFEST_SCHEMA_VERSION,
        "attempt_selection_rule": ATTEMPT_SELECTION_RULE,
        "inputs": {
            "attempts_root": str(attempts_root),
            "attempt_count": len(attempts),
            "manifest_episode_count": len(manifest_by_id),
            "protocol_sha256": protocol_sha256,
            "scorer_sha256": scorer_sha256,
        },
        "outputs": {
            "accepted_count": len(accepted_rows),
            "rejected_count": len(rejected_rows),
            "accepted_unique_episode_ids": sorted(accepted_by_id),
        },
        "reconciliation": reconciliation,
        "validation_preview": validation_preview,
        "verified_common_prefix_modes": sorted(VERIFIED_COMMON_PREFIX_MODES),
        "limitations": [
            "Accepted rows never synthesize C2 prefix verification or response metrics.",
            "C2 confirmatory analysis remains blocked until runtime emits the complete contract.",
        ],
    }

    return LedgerCompileResult(
        accepted_rows=accepted_rows,
        rejected_rows=rejected_rows,
        manifest_payload=manifest_payload,
        reconciliation=reconciliation,
        errors=errors,
        warnings=warnings,
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write_bytes(path, encoded.encode("utf-8"))


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) for row in rows]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    _atomic_write_bytes(path, payload)


def write_ledger_outputs(
    result: LedgerCompileResult,
    output_dir: Path,
    *,
    attempts_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    _require(result.ok, "; ".join(result.errors))
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "accepted_ledger.jsonl"
    rejected_path = output_dir / "rejected_attempts.jsonl"
    manifest_out = output_dir / "accepted_ledger_manifest.json"

    staging = output_dir / ".ledger_compile_staging"
    if staging.exists():
        raise LedgerError(f"staging directory already exists: {staging}")
    staging.mkdir()

    try:
        _atomic_write_jsonl(staging / accepted_path.name, result.accepted_rows)
        _atomic_write_jsonl(staging / rejected_path.name, result.rejected_rows)
        manifest_payload = {
            **result.manifest_payload,
            "inputs": {
                **result.manifest_payload["inputs"],
                "manifest_path": str(manifest_path),
                "manifest_sha256": digest_bytes(manifest_path.read_bytes()),
            },
            "outputs": {
                **result.manifest_payload["outputs"],
                "accepted_ledger_path": str(accepted_path),
                "accepted_ledger_sha256": digest_bytes((staging / accepted_path.name).read_bytes()),
                "rejected_attempts_path": str(rejected_path),
                "rejected_attempts_sha256": digest_bytes((staging / rejected_path.name).read_bytes()),
            },
        }
        _atomic_write_json(staging / manifest_out.name, manifest_payload)
        for name in (accepted_path.name, rejected_path.name, manifest_out.name):
            os.replace(staging / name, output_dir / name)
    finally:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink(missing_ok=True)
            staging.rmdir()

    return {
        "output_dir": str(output_dir),
        "accepted_ledger": str(accepted_path),
        "rejected_attempts": str(rejected_path),
        "accepted_ledger_manifest": str(manifest_out),
    }


def load_queue_episode_ids(queue_path: Path) -> set[str]:
    rows = read_jsonl(queue_path)
    return {str(row["episode_id"]) for row in rows if row.get("episode_id")}


def compile_accepted_ledger(
    *,
    manifest_path: Path,
    attempts_root: Path,
    output_dir: Path,
    protocol_sha256: str,
    scorer_sha256: str,
    config: Mapping[str, Any] | None = None,
    queue_path: Path | None = None,
    require_full_coverage: bool = False,
) -> dict[str, Any]:
    manifest = read_jsonl(manifest_path)
    attempts = discover_finalized_attempts(attempts_root)
    queue_ids = load_queue_episode_ids(queue_path) if queue_path is not None else None
    result = compile_accepted_ledger_from_attempts(
        manifest=manifest,
        attempts=attempts,
        attempts_root=attempts_root,
        protocol_sha256=protocol_sha256,
        scorer_sha256=scorer_sha256,
        config=config,
        queue_episode_ids=queue_ids,
        require_full_coverage=require_full_coverage,
    )
    outputs = write_ledger_outputs(
        result,
        output_dir,
        attempts_root=attempts_root,
        manifest_path=manifest_path,
    )
    return {
        "ok": True,
        "warnings": result.warnings,
        "reconciliation": result.reconciliation,
        "outputs": outputs,
        "manifest": result.manifest_payload,
    }
