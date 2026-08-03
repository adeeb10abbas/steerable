from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .constants import DEFAULT_SEED
from .io import load_json, sha256_file, write_json
from .provenance import implementation_manifest
from .text import stable_id


TRUE_VALUES = {"1", "true", "yes", "y", "pass", "passed"}
FALSE_VALUES = {"0", "false", "no", "n", "fail", "failed"}
MANUAL_CLASSES = {
    "task_level_language",
    "subtask_same_intent_paraphrase_candidate",
    "atomic_motion_or_gripper_command",
    "point_coordinate_grounding",
    "multi_point_gripper_trace",
    "hybrid_or_added_behavioral_semantics",
    "malformed_or_unclear",
}

COMMAND_REVIEW_FIELDS = frozenset(
    {
        "primary_manual_class",
        "primary_same_intent_with_canonical",
        "primary_reviewer",
        "primary_review_notes",
        "secondary_manual_class",
        "secondary_same_intent_with_canonical",
        "secondary_reviewer",
        "secondary_review_notes",
        "adjudicated_class",
        "adjudicated_same_intent_with_canonical",
        "adjudicator",
        "adjudication_notes",
    }
)
PARAPHRASE_REVIEW_FIELDS = frozenset(
    {
        "primary_all_surfaces_semantically_equivalent",
        "primary_language_natural_and_acceptable",
        "primary_reviewer",
        "primary_review_notes",
        "secondary_all_surfaces_semantically_equivalent",
        "secondary_language_natural_and_acceptable",
        "secondary_reviewer",
        "secondary_review_notes",
        "adjudicated_semantically_equivalent",
        "adjudicated_language_acceptable",
        "adjudicator",
        "adjudication_notes",
    }
)
SEQUENCE_REVIEW_FIELDS = frozenset(
    {
        "primary_reviewer",
        "primary_task_identity_believable",
        "primary_direct_step_i_alignment_believable",
        "primary_boundary_alignment_believable",
        "primary_review_notes",
        "secondary_reviewer",
        "secondary_task_identity_believable",
        "secondary_direct_step_i_alignment_believable",
        "secondary_boundary_alignment_believable",
        "secondary_review_notes",
        "adjudicated_task_identity_believable",
        "adjudicated_direct_step_i_alignment_believable",
        "adjudicated_boundary_alignment_believable",
        "adjudicator",
        "adjudication_notes",
    }
)

AUDIT_SPECS: dict[str, dict[str, Any]] = {
    "manual_command_audit.csv": {
        "identity_fields": ("pool_id", "slot_index"),
        "unit_fields": ("pool_id",),
        "expected_units": 100,
        "review_fields": COMMAND_REVIEW_FIELDS,
        "requires_seed": False,
    },
    "manual_paraphrase_group_audit.csv": {
        "identity_fields": ("intent_id",),
        "unit_fields": ("intent_id",),
        "expected_units": 100,
        "review_fields": PARAPHRASE_REVIEW_FIELDS,
        "requires_seed": True,
    },
    "visual_alignment_audit.csv": {
        "identity_fields": ("steering_trajectory_id",),
        "unit_fields": ("steering_trajectory_id",),
        "expected_units": 20,
        "review_fields": SEQUENCE_REVIEW_FIELDS,
        "requires_seed": True,
    },
    "manual_sequence_audit.csv": {
        "identity_fields": ("steering_trajectory_id",),
        "unit_fields": ("steering_trajectory_id",),
        "expected_units": 30,
        "review_fields": SEQUENCE_REVIEW_FIELDS,
        "requires_seed": True,
    },
}

AUDIT_RUN_PROVENANCE_PATHS = (
    "input_manifest.json",
    "target_split.json",
    "pilot_split.json",
    "split_validation.json",
    "manifest_validation.json",
    "pilot_manifest_validation.json",
    "paraphrase_eligibility_table.csv",
    "surface_diversity_report.csv",
    "manifests/master_manifest.parquet",
    "manifests/A_task_canonical.parquet",
    "manifests/B_task_paraphrases.parquet",
    "manifests/C_subtask_canonical.parquet",
    "manifests/D_subtask_paraphrases.parquet",
    "manifests/pilot/master_manifest.parquet",
    "manifests/pilot/A_task_canonical.parquet",
    "manifests/pilot/B_task_paraphrases.parquet",
    "manifests/pilot/C_subtask_canonical.parquet",
    "manifests/pilot/D_subtask_paraphrases.parquet",
)


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def collect_audit_run_provenance(
    artifact_dir: Path, *, base_seed: int
) -> dict[str, Any]:
    """Bind human sheets to the exact inputs, cohorts, and manifest bytes."""

    files: list[dict[str, Any]] = []
    missing: list[str] = []
    for relative in AUDIT_RUN_PROVENANCE_PATHS:
        path = artifact_dir / relative
        if not path.exists():
            missing.append(relative)
            continue
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    target_split_seed = None
    pilot_split_seed = None
    if (artifact_dir / "target_split.json").exists():
        target_split_seed = load_json(artifact_dir / "target_split.json").get("seed")
    if (artifact_dir / "pilot_split.json").exists():
        pilot_split_seed = load_json(artifact_dir / "pilot_split.json").get("seed")
    core = {
        "base_seed": int(base_seed),
        "expected_target_split_seed": int(base_seed) + 2,
        "expected_pilot_split_seed": int(base_seed) + 1,
        "observed_target_split_seed": target_split_seed,
        "observed_pilot_split_seed": pilot_split_seed,
        "files": files,
        "missing_files": missing,
    }
    return {
        "schema_version": 1,
        **core,
        "split_seeds_match_base": (
            target_split_seed == int(base_seed) + 2
            and pilot_split_seed == int(base_seed) + 1
        ),
        "combined_sha256": _canonical_digest(core),
    }


def _row_identity(row: Mapping[str, str], fields: Iterable[str]) -> str:
    return json.dumps(
        [str(row.get(field, "")) for field in fields],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _sheet_snapshot(
    rows: list[dict[str, str]],
    spec: Mapping[str, Any],
    *,
    immutable_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Audit sheet must contain at least one data row")
    review_fields = set(spec["review_fields"])
    fields = (
        tuple(immutable_fields)
        if immutable_fields is not None
        else tuple(field for field in rows[0] if field not in review_fields)
    )
    missing_fields = sorted(field for field in fields if field not in rows[0])
    identities = [_row_identity(row, spec["identity_fields"]) for row in rows]
    units = [_row_identity(row, spec["unit_fields"]) for row in rows]
    records = [
        {
            "identity": identity,
            "values": {field: str(row.get(field, "")) for field in fields},
        }
        for identity, row in zip(identities, rows)
    ]
    records.sort(key=lambda record: (record["identity"], _canonical_digest(record)))
    unique_identities = sorted(set(identities))
    unique_units = sorted(set(units))

    seed: int | None = None
    seed_valid = not bool(spec["requires_seed"])
    if spec["requires_seed"]:
        raw_seeds = {str(row.get("audit_seed", "")).strip() for row in rows}
        if len(raw_seeds) == 1:
            try:
                seed = int(next(iter(raw_seeds)))
                seed_valid = True
            except ValueError:
                seed_valid = False

    return {
        "immutable_fields": list(fields),
        "missing_immutable_fields": missing_fields,
        "row_count": len(rows),
        "identity_count": len(unique_identities),
        "unit_count": len(unique_units),
        "identity_values": unique_identities,
        "unit_identity_values": unique_units,
        "identity_rows_unique": len(unique_identities) == len(rows),
        "expected_unit_count": int(spec["expected_units"]),
        "expected_unit_count_exact": len(unique_units) == int(spec["expected_units"]),
        "audit_seed": seed,
        "audit_seed_valid": seed_valid,
        "immutable_sha256": _canonical_digest(records),
    }


def build_audit_lock(
    artifact_dir: Path, *, overwrite: bool, base_seed: int
) -> dict[str, Any]:
    """Persist immutable sample membership and non-review fields for all sheets."""

    path = artifact_dir / "audit_lock.json"
    if not overwrite:
        if not path.exists():
            raise RuntimeError(
                "Reviewed audit sheets exist without audit_lock.json; refusing "
                "to infer the original locked sample from human-edited files."
            )
        return load_json(path)

    sheets: dict[str, Any] = {}
    for filename, spec in AUDIT_SPECS.items():
        snapshot = _sheet_snapshot(_rows(artifact_dir / filename), spec)
        if (
            snapshot["missing_immutable_fields"]
            or not snapshot["identity_rows_unique"]
            or not snapshot["expected_unit_count_exact"]
            or not snapshot["audit_seed_valid"]
        ):
            raise ValueError(f"Refusing to lock invalid audit template: {filename}")
        sheets[filename] = snapshot

    implementation = implementation_manifest()
    run_provenance = collect_audit_run_provenance(
        artifact_dir, base_seed=base_seed
    )
    if run_provenance["missing_files"] or not run_provenance["split_seeds_match_base"]:
        raise ValueError(
            "Refusing to lock audits without complete, seed-consistent run provenance"
        )
    lock = {
        "schema_version": 1,
        "implementation_combined_sha256": implementation["combined_sha256"],
        "run_provenance": run_provenance,
        "sheets": sheets,
    }
    write_json(path, lock)
    return lock


def validate_audit_lock(artifact_dir: Path) -> dict[str, Any]:
    """Fail closed if a reviewer sheet or the generating code has drifted."""

    path = artifact_dir / "audit_lock.json"
    current_implementation = implementation_manifest()["combined_sha256"]
    if not path.exists():
        return {
            "lock_present": False,
            "implementation_match": False,
            "sheets": {},
            "all_pass": False,
        }

    lock = load_json(path)
    locked_implementation = str(lock.get("implementation_combined_sha256", ""))
    locked_run_provenance = lock.get("run_provenance", {})
    try:
        observed_run_provenance = collect_audit_run_provenance(
            artifact_dir,
            base_seed=int(locked_run_provenance.get("base_seed")),
        )
    except (TypeError, ValueError):
        observed_run_provenance = {
            "missing_files": list(AUDIT_RUN_PROVENANCE_PATHS),
            "split_seeds_match_base": False,
            "combined_sha256": "",
        }
    run_provenance_checks = {
        "lock_entry_present": bool(locked_run_provenance),
        "all_files_present": not observed_run_provenance.get("missing_files"),
        "split_seeds_match_locked_base": bool(
            observed_run_provenance.get("split_seeds_match_base")
        ),
        "combined_sha256_match": bool(
            observed_run_provenance.get("combined_sha256")
            and observed_run_provenance.get("combined_sha256")
            == locked_run_provenance.get("combined_sha256")
        ),
        "file_records_match": observed_run_provenance.get("files")
        == locked_run_provenance.get("files"),
    }
    run_provenance_match = all(run_provenance_checks.values())
    sheet_results: dict[str, Any] = {}
    locked_sheets = lock.get("sheets", {})
    for filename, spec in AUDIT_SPECS.items():
        expected = locked_sheets.get(filename)
        if not isinstance(expected, dict) or not (artifact_dir / filename).exists():
            sheet_results[filename] = {
                "lock_entry_present": isinstance(expected, dict),
                "sheet_present": (artifact_dir / filename).exists(),
                "all_pass": False,
            }
            continue
        try:
            observed = _sheet_snapshot(
                _rows(artifact_dir / filename),
                spec,
                immutable_fields=expected.get("immutable_fields", ()),
            )
        except (KeyError, TypeError, ValueError):
            sheet_results[filename] = {
                "lock_entry_present": True,
                "sheet_present": True,
                "all_pass": False,
            }
            continue
        checks = {
            "immutable_fields_present": not observed["missing_immutable_fields"],
            "row_count_match": observed["row_count"] == expected.get("row_count"),
            "identity_rows_unique": observed["identity_rows_unique"],
            "identity_count_match": observed["identity_count"]
            == expected.get("identity_count"),
            "unit_count_match": observed["unit_count"] == expected.get("unit_count"),
            "identity_values_match": observed["identity_values"]
            == expected.get("identity_values"),
            "unit_identity_values_match": observed["unit_identity_values"]
            == expected.get("unit_identity_values"),
            "audit_seed_valid": observed["audit_seed_valid"],
            "audit_seed_match": observed["audit_seed"] == expected.get("audit_seed"),
            "immutable_sha256_match": observed["immutable_sha256"]
            == expected.get("immutable_sha256"),
        }
        sheet_results[filename] = {
            **checks,
            "all_pass": all(checks.values()),
            "locked_row_count": expected.get("row_count"),
            "observed_row_count": observed["row_count"],
            "locked_unit_count": expected.get("unit_count"),
            "observed_unit_count": observed["unit_count"],
        }

    implementation_match = bool(
        locked_implementation and locked_implementation == current_implementation
    )
    sample_sheets_pass = (
        len(sheet_results) == len(AUDIT_SPECS)
        and all(result.get("all_pass", False) for result in sheet_results.values())
    )
    return {
        "lock_present": True,
        "locked_implementation_combined_sha256": locked_implementation,
        "current_implementation_combined_sha256": current_implementation,
        "implementation_match": implementation_match,
        "sample_sheets_pass": sample_sheets_pass,
        "run_provenance": {
            **run_provenance_checks,
            "all_pass": run_provenance_match,
            "locked_base_seed": locked_run_provenance.get("base_seed"),
            "locked_combined_sha256": locked_run_provenance.get(
                "combined_sha256"
            ),
            "observed_combined_sha256": observed_run_provenance.get(
                "combined_sha256"
            ),
        },
        "sheets": sheet_results,
        "all_pass": implementation_match
        and run_provenance_match
        and sample_sheets_pass,
    }


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truth(value: object) -> bool | None:
    normalized = str(value or "").strip().casefold()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def _effective_truth(row: Mapping[str, str], adjudicated: str, primary: str) -> bool | None:
    return (
        _truth(row.get(adjudicated))
        if str(row.get(adjudicated, "")).strip()
        else _truth(row.get(primary))
    )


def _effective_disputed_truth(
    row: Mapping[str, str],
    *,
    adjudicated: str,
    primary: str,
    secondary: str,
    secondary_complete: bool,
) -> bool | None:
    """Use adjudication only to resolve a completed independent disagreement."""

    primary_value = _truth(row.get(primary))
    secondary_value = _truth(row.get(secondary))
    if (
        secondary_complete
        and primary_value != secondary_value
        and str(row.get(adjudicated, "")).strip()
    ):
        return _truth(row.get(adjudicated))
    return primary_value


def _identity(value: object) -> str:
    return str(value or "").strip().casefold()


def _independent_reviewers(row: Mapping[str, str]) -> bool:
    primary = _identity(row.get("primary_reviewer"))
    secondary = _identity(row.get("secondary_reviewer"))
    return bool(primary and secondary and primary != secondary)


def _valid_adjudicator(row: Mapping[str, str]) -> bool:
    adjudicator = _identity(row.get("adjudicator"))
    return bool(
        adjudicator
        and adjudicator
        not in {
            _identity(row.get("primary_reviewer")),
            _identity(row.get("secondary_reviewer")),
        }
        and str(row.get("adjudication_notes", "")).strip()
    )


def _invalid_nonblank_truth(row: Mapping[str, str], fields: Iterable[str]) -> bool:
    return any(
        str(row.get(field, "")).strip() and _truth(row.get(field)) is None
        for field in fields
    )


def _rate(values: Iterable[bool | None]) -> float | None:
    observed = [value for value in values if value is not None]
    return sum(observed) / len(observed) if observed else None


def _command_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_pool: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_pool[row["pool_id"]].append(row)
    slot_identities = {
        (row.get("pool_id", ""), row.get("slot_index", "")) for row in rows
    }
    slot_identities_unique = len(slot_identities) == len(rows)
    reviewed_rows = [
        row
        for row in rows
        if row.get("primary_reviewer", "").strip()
        and row.get("primary_manual_class", "").strip() in MANUAL_CLASSES
        and _truth(row.get("primary_same_intent_with_canonical")) is not None
    ]
    reviewed_pools = {
        pool_id
        for pool_id, pool_rows in by_pool.items()
        if all(row in reviewed_rows for row in pool_rows)
    }
    secondary_required = [
        row for row in rows if _truth(row.get("secondary_review_required")) is True
    ]
    expected_secondary_pool_ids = set(
        sorted(
            by_pool,
            key=lambda pool_id: stable_id("secondary_review", pool_id),
        )[: max(1, round(0.20 * len(by_pool)))]
    )
    observed_secondary_pool_ids = {
        row["pool_id"] for row in secondary_required
    }
    secondary_assignment_exact = (
        observed_secondary_pool_ids == expected_secondary_pool_ids
        and all(
            (_truth(row.get("secondary_review_required")) is True)
            == (row["pool_id"] in expected_secondary_pool_ids)
            for row in rows
        )
    )
    secondary_complete = [
        row
        for row in secondary_required
        if _independent_reviewers(row)
        and row.get("secondary_manual_class", "").strip() in MANUAL_CLASSES
        and _truth(row.get("secondary_same_intent_with_canonical")) is not None
    ]
    secondary_complete_ids = {id(row) for row in secondary_complete}
    secondary_disagreements = [
        row
        for row in secondary_complete
        if (
            row.get("primary_manual_class", "").strip()
            != row.get("secondary_manual_class", "").strip()
            or _truth(row.get("primary_same_intent_with_canonical"))
            != _truth(row.get("secondary_same_intent_with_canonical"))
        )
    ]
    unresolved_disagreements = [
        row
        for row in secondary_disagreements
        if (
            (
                row.get("primary_manual_class", "").strip()
                != row.get("secondary_manual_class", "").strip()
                and not row.get("adjudicated_class", "").strip()
            )
            or (
                _truth(row.get("primary_same_intent_with_canonical"))
                != _truth(row.get("secondary_same_intent_with_canonical"))
                and _truth(row.get("adjudicated_same_intent_with_canonical"))
                is None
            )
            or not _valid_adjudicator(row)
        )
    ]

    def invalid_command_adjudication(row: Mapping[str, str]) -> bool:
        adjudicated_class = row.get("adjudicated_class", "").strip()
        adjudicated_intent = row.get(
            "adjudicated_same_intent_with_canonical", ""
        ).strip()
        class_disagreement = (
            id(row) in secondary_complete_ids
            and row.get("primary_manual_class", "").strip()
            != row.get("secondary_manual_class", "").strip()
        )
        intent_disagreement = (
            id(row) in secondary_complete_ids
            and _truth(row.get("primary_same_intent_with_canonical"))
            != _truth(row.get("secondary_same_intent_with_canonical"))
        )
        any_adjudication = bool(adjudicated_class or adjudicated_intent)
        metadata_without_adjudication = bool(
            row.get("adjudicator", "").strip()
            or row.get("adjudication_notes", "").strip()
        ) and not any_adjudication
        return bool(
            _invalid_nonblank_truth(
                row, ("adjudicated_same_intent_with_canonical",)
            )
            or (adjudicated_class and adjudicated_class not in MANUAL_CLASSES)
            or (adjudicated_class and not class_disagreement)
            or (adjudicated_intent and not intent_disagreement)
            or (any_adjudication and not _valid_adjudicator(row))
            or metadata_without_adjudication
        )

    invalid_adjudications = [
        row for row in rows if invalid_command_adjudication(row)
    ]

    def effective_manual_class(row: Mapping[str, str]) -> str:
        class_disagreement = (
            id(row) in secondary_complete_ids
            and row.get("primary_manual_class", "").strip()
            != row.get("secondary_manual_class", "").strip()
        )
        if (
            class_disagreement
            and row.get("adjudicated_class", "").strip()
            and not invalid_command_adjudication(row)
        ):
            return row.get("adjudicated_class", "").strip()
        return row.get("primary_manual_class", "").strip()

    taxonomy_agreement = [
        row["automatic_class"] == effective_manual_class(row)
        for row in reviewed_rows
    ]
    auto_candidates = [
        row
        for row in reviewed_rows
        if row["automatic_class"] == "subtask_same_intent_paraphrase_candidate"
    ]
    false_equivalence = [
        not bool(
            _effective_disputed_truth(
                row,
                adjudicated="adjudicated_same_intent_with_canonical",
                primary="primary_same_intent_with_canonical",
                secondary="secondary_same_intent_with_canonical",
                secondary_complete=(
                    id(row) in secondary_complete_ids
                    and not invalid_command_adjudication(row)
                ),
            )
        )
        for row in auto_candidates
    ]
    confusion: Counter[tuple[str, str]] = Counter()
    for row in reviewed_rows:
        manual_class = effective_manual_class(row)
        confusion[(row["automatic_class"], manual_class)] += 1
    per_automatic_class: dict[str, dict[str, Any]] = {}
    for automatic_class in sorted({automatic for automatic, _ in confusion}):
        support = sum(
            count
            for (automatic, _), count in confusion.items()
            if automatic == automatic_class
        )
        agreements = confusion[(automatic_class, automatic_class)]
        per_automatic_class[automatic_class] = {
            "support": support,
            "agreement_count": agreements,
            "precision": agreements / support if support else None,
        }
    manual_unclear = sum(
        count
        for (_, manual), count in confusion.items()
        if manual == "malformed_or_unclear"
    )
    invalid_secondary_flags = [
        row
        for row in rows
        if _invalid_nonblank_truth(row, ("secondary_review_required",))
    ]
    return {
        "template_pool_count": len(by_pool),
        "template_command_rows": len(rows),
        "slot_identity_rows_unique": slot_identities_unique,
        "fully_primary_reviewed_pools": len(reviewed_pools),
        "fully_primary_reviewed_command_rows": len(reviewed_rows),
        "secondary_required_rows": len(secondary_required),
        "expected_secondary_pool_count": len(expected_secondary_pool_ids),
        "observed_secondary_pool_count": len(observed_secondary_pool_ids),
        "secondary_assignment_exact": secondary_assignment_exact,
        "secondary_complete_rows": len(secondary_complete),
        "secondary_disagreement_rows": len(secondary_disagreements),
        "unresolved_disagreement_rows": len(unresolved_disagreements),
        "invalid_adjudication_rows": len(invalid_adjudications),
        "invalid_secondary_flag_rows": len(invalid_secondary_flags),
        "automatic_taxonomy_agreement_rate": _rate(taxonomy_agreement),
        "automatic_candidate_false_equivalence_rate": _rate(false_equivalence),
        "automatic_to_manual_confusion": {
            automatic: {
                manual: confusion[(automatic, manual)]
                for manual in sorted(
                    manual
                    for (observed_automatic, manual) in confusion
                    if observed_automatic == automatic
                )
            }
            for automatic in sorted({automatic for automatic, _ in confusion})
        },
        "per_automatic_class_precision": per_automatic_class,
        "manual_unclear_rate": (
            manual_unclear / len(reviewed_rows) if reviewed_rows else None
        ),
        "secondary_disagreement_rate": (
            len(secondary_disagreements) / len(secondary_complete)
            if secondary_complete
            else None
        ),
        "unresolved_secondary_disagreement_rate": (
            len(unresolved_disagreements) / len(secondary_required)
            if secondary_required
            else None
        ),
        "complete": (
            len(by_pool) == 100
            and slot_identities_unique
            and len(reviewed_pools) == len(by_pool)
            and secondary_assignment_exact
            and len(secondary_complete) == len(secondary_required)
            and not unresolved_disagreements
            and not invalid_adjudications
            and not invalid_secondary_flags
        ),
    }


def _paraphrase_summary(
    rows: list[dict[str, str]], *, secondary_seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    unique_intent_count = len({row.get("intent_id", "") for row in rows})
    intent_ids_unique = unique_intent_count == len(rows)
    reviewed = [
        row
        for row in rows
        if row.get("primary_reviewer", "").strip()
        and _truth(row.get("primary_all_surfaces_semantically_equivalent")) is not None
        and _truth(row.get("primary_language_natural_and_acceptable")) is not None
    ]
    secondary_required = [
        row for row in rows if _truth(row.get("secondary_review_required")) is True
    ]
    expected_secondary_ids = set(
        sorted(
            (row["intent_id"] for row in rows),
            key=lambda intent_id: stable_id(
                "group_secondary_review", secondary_seed, intent_id
            ),
        )[: max(1, round(0.20 * len(rows)))]
    )
    observed_secondary_ids = {row["intent_id"] for row in secondary_required}
    secondary_assignment_exact = (
        observed_secondary_ids == expected_secondary_ids
        and all(
            (_truth(row.get("secondary_review_required")) is True)
            == (row["intent_id"] in expected_secondary_ids)
            for row in rows
        )
    )
    secondary_complete = [
        row
        for row in secondary_required
        if _independent_reviewers(row)
        and _truth(row.get("secondary_all_surfaces_semantically_equivalent")) is not None
        and _truth(row.get("secondary_language_natural_and_acceptable")) is not None
    ]
    secondary_complete_ids = {id(row) for row in secondary_complete}
    secondary_disagreements = [
        row
        for row in secondary_complete
        if (
            _truth(row.get("primary_all_surfaces_semantically_equivalent"))
            != _truth(row.get("secondary_all_surfaces_semantically_equivalent"))
            or _truth(row.get("primary_language_natural_and_acceptable"))
            != _truth(row.get("secondary_language_natural_and_acceptable"))
        )
    ]
    unresolved_disagreements = [
        row
        for row in secondary_disagreements
        if (
            (
                _truth(row.get("primary_all_surfaces_semantically_equivalent"))
                != _truth(row.get("secondary_all_surfaces_semantically_equivalent"))
                and _truth(row.get("adjudicated_semantically_equivalent")) is None
            )
            or (
                _truth(row.get("primary_language_natural_and_acceptable"))
                != _truth(row.get("secondary_language_natural_and_acceptable"))
                and _truth(row.get("adjudicated_language_acceptable")) is None
            )
            or not _valid_adjudicator(row)
        )
    ]
    adjudication_fields = (
        "adjudicated_semantically_equivalent",
        "adjudicated_language_acceptable",
    )

    def invalid_paraphrase_adjudication(row: Mapping[str, str]) -> bool:
        semantic_adjudication = str(
            row.get("adjudicated_semantically_equivalent", "")
        ).strip()
        language_adjudication = str(
            row.get("adjudicated_language_acceptable", "")
        ).strip()
        semantic_disagreement = (
            id(row) in secondary_complete_ids
            and _truth(row.get("primary_all_surfaces_semantically_equivalent"))
            != _truth(row.get("secondary_all_surfaces_semantically_equivalent"))
        )
        language_disagreement = (
            id(row) in secondary_complete_ids
            and _truth(row.get("primary_language_natural_and_acceptable"))
            != _truth(row.get("secondary_language_natural_and_acceptable"))
        )
        any_adjudication = bool(semantic_adjudication or language_adjudication)
        metadata_without_adjudication = bool(
            row.get("adjudicator", "").strip()
            or row.get("adjudication_notes", "").strip()
        ) and not any_adjudication
        return bool(
            _invalid_nonblank_truth(row, adjudication_fields)
            or (semantic_adjudication and not semantic_disagreement)
            or (language_adjudication and not language_disagreement)
            or (any_adjudication and not _valid_adjudicator(row))
            or metadata_without_adjudication
        )

    invalid_adjudications = [
        row for row in rows if invalid_paraphrase_adjudication(row)
    ]
    equivalence = [
        _effective_disputed_truth(
            row,
            adjudicated="adjudicated_semantically_equivalent",
            primary="primary_all_surfaces_semantically_equivalent",
            secondary="secondary_all_surfaces_semantically_equivalent",
            secondary_complete=(
                id(row) in secondary_complete_ids
                and not invalid_paraphrase_adjudication(row)
            ),
        )
        for row in reviewed
    ]
    acceptability = [
        _effective_disputed_truth(
            row,
            adjudicated="adjudicated_language_acceptable",
            primary="primary_language_natural_and_acceptable",
            secondary="secondary_language_natural_and_acceptable",
            secondary_complete=(
                id(row) in secondary_complete_ids
                and not invalid_paraphrase_adjudication(row)
            ),
        )
        for row in reviewed
    ]
    equivalence_rate = _rate(equivalence)
    acceptability_rate = _rate(acceptability)
    invalid_secondary_flags = [
        row
        for row in rows
        if _invalid_nonblank_truth(row, ("secondary_review_required",))
    ]
    complete = (
        len(rows) == 100
        and intent_ids_unique
        and len(reviewed) == len(rows)
        and secondary_assignment_exact
        and len(secondary_complete) == len(secondary_required)
        and not unresolved_disagreements
        and not invalid_adjudications
        and not invalid_secondary_flags
    )
    return {
        "template_group_count": len(rows),
        "unique_intent_count": unique_intent_count,
        "intent_ids_unique": intent_ids_unique,
        "primary_reviewed_groups": len(reviewed),
        "secondary_required_groups": len(secondary_required),
        "expected_secondary_group_count": len(expected_secondary_ids),
        "observed_secondary_group_count": len(observed_secondary_ids),
        "secondary_assignment_exact": secondary_assignment_exact,
        "secondary_complete_groups": len(secondary_complete),
        "secondary_disagreement_groups": len(secondary_disagreements),
        "unresolved_disagreement_groups": len(unresolved_disagreements),
        "invalid_adjudication_groups": len(invalid_adjudications),
        "invalid_secondary_flag_groups": len(invalid_secondary_flags),
        "semantic_equivalence_pass_rate": equivalence_rate,
        "estimated_false_equivalence_rate": (
            1.0 - equivalence_rate if equivalence_rate is not None else None
        ),
        "language_acceptability_pass_rate": acceptability_rate,
        "complete": complete,
        "gate_pass": (
            complete
            and equivalence_rate is not None
            and equivalence_rate >= 0.98
            and acceptability_rate is not None
            and acceptability_rate >= 0.95
        ),
    }


def _sequence_summary(
    rows: list[dict[str, str]], minimum: int, *, secondary_seed: int
) -> dict[str, Any]:
    unique_trajectory_count = len(
        {row.get("steering_trajectory_id", "") for row in rows}
    )
    trajectory_ids_unique = unique_trajectory_count == len(rows)
    reviewed = [
        row
        for row in rows
        if row.get("primary_reviewer", "").strip()
        and _truth(row.get("primary_task_identity_believable")) is not None
        and _truth(row.get("primary_direct_step_i_alignment_believable")) is not None
        and _truth(row.get("primary_boundary_alignment_believable")) is not None
    ]
    secondary_required = [
        row for row in rows if _truth(row.get("secondary_review_required")) is True
    ]
    expected_secondary_ids = set(
        sorted(
            (int(row["steering_trajectory_id"]) for row in rows),
            key=lambda trajectory_id: stable_id(
                "sequence_secondary_review", secondary_seed, trajectory_id
            ),
        )[: max(1, round(0.20 * len(rows)))]
    )
    observed_secondary_ids = {
        int(row["steering_trajectory_id"]) for row in secondary_required
    }
    secondary_assignment_exact = (
        observed_secondary_ids == expected_secondary_ids
        and all(
            (_truth(row.get("secondary_review_required")) is True)
            == (int(row["steering_trajectory_id"]) in expected_secondary_ids)
            for row in rows
        )
    )
    secondary_complete = [
        row
        for row in secondary_required
        if _independent_reviewers(row)
        and _truth(row.get("secondary_task_identity_believable")) is not None
        and _truth(row.get("secondary_direct_step_i_alignment_believable")) is not None
        and _truth(row.get("secondary_boundary_alignment_believable")) is not None
    ]
    secondary_complete_ids = {id(row) for row in secondary_complete}
    dimensions = (
        ("task_identity_believable", "adjudicated_task_identity_believable"),
        (
            "direct_step_i_alignment_believable",
            "adjudicated_direct_step_i_alignment_believable",
        ),
        ("boundary_alignment_believable", "adjudicated_boundary_alignment_believable"),
    )
    secondary_disagreements = [
        row
        for row in secondary_complete
        if any(
            _truth(row.get(f"primary_{name}"))
            != _truth(row.get(f"secondary_{name}"))
            for name, _ in dimensions
        )
    ]
    unresolved_disagreements = [
        row
        for row in secondary_disagreements
        if any(
            _truth(row.get(f"primary_{name}"))
            != _truth(row.get(f"secondary_{name}"))
            and _truth(row.get(adjudicated)) is None
            for name, adjudicated in dimensions
        )
        or not _valid_adjudicator(row)
    ]
    adjudication_fields = tuple(adjudicated for _, adjudicated in dimensions)

    def invalid_sequence_adjudication(row: Mapping[str, str]) -> bool:
        any_adjudication = any(
            str(row.get(field, "")).strip() for field in adjudication_fields
        )
        unsolicited_dimension = any(
            str(row.get(adjudicated, "")).strip()
            and not (
                id(row) in secondary_complete_ids
                and _truth(row.get(f"primary_{name}"))
                != _truth(row.get(f"secondary_{name}"))
            )
            for name, adjudicated in dimensions
        )
        metadata_without_adjudication = bool(
            row.get("adjudicator", "").strip()
            or row.get("adjudication_notes", "").strip()
        ) and not any_adjudication
        return bool(
            _invalid_nonblank_truth(row, adjudication_fields)
            or unsolicited_dimension
            or (any_adjudication and not _valid_adjudicator(row))
            or metadata_without_adjudication
        )

    invalid_adjudications = [
        row for row in rows if invalid_sequence_adjudication(row)
    ]
    task_rate = _rate(
        _effective_disputed_truth(
            row,
            adjudicated="adjudicated_task_identity_believable",
            primary="primary_task_identity_believable",
            secondary="secondary_task_identity_believable",
            secondary_complete=(
                id(row) in secondary_complete_ids
                and not invalid_sequence_adjudication(row)
            ),
        )
        for row in reviewed
    )
    direct_rate = _rate(
        _effective_disputed_truth(
            row,
            adjudicated="adjudicated_direct_step_i_alignment_believable",
            primary="primary_direct_step_i_alignment_believable",
            secondary="secondary_direct_step_i_alignment_believable",
            secondary_complete=(
                id(row) in secondary_complete_ids
                and not invalid_sequence_adjudication(row)
            ),
        )
        for row in reviewed
    )
    boundary_rate = _rate(
        _effective_disputed_truth(
            row,
            adjudicated="adjudicated_boundary_alignment_believable",
            primary="primary_boundary_alignment_believable",
            secondary="secondary_boundary_alignment_believable",
            secondary_complete=(
                id(row) in secondary_complete_ids
                and not invalid_sequence_adjudication(row)
            ),
        )
        for row in reviewed
    )
    invalid_secondary_flags = [
        row
        for row in rows
        if _invalid_nonblank_truth(row, ("secondary_review_required",))
    ]
    complete = (
        len(rows) == minimum
        and trajectory_ids_unique
        and len(reviewed) == len(rows)
        and secondary_assignment_exact
        and len(secondary_complete) == len(secondary_required)
        and not unresolved_disagreements
        and not invalid_adjudications
        and not invalid_secondary_flags
    )
    return {
        "template_trajectory_count": len(rows),
        "unique_trajectory_count": unique_trajectory_count,
        "trajectory_ids_unique": trajectory_ids_unique,
        "primary_reviewed_trajectories": len(reviewed),
        "secondary_required_trajectories": len(secondary_required),
        "expected_secondary_trajectory_count": len(expected_secondary_ids),
        "observed_secondary_trajectory_count": len(observed_secondary_ids),
        "secondary_assignment_exact": secondary_assignment_exact,
        "secondary_complete_trajectories": len(secondary_complete),
        "secondary_disagreement_trajectories": len(secondary_disagreements),
        "unresolved_disagreement_trajectories": len(unresolved_disagreements),
        "invalid_adjudication_trajectories": len(invalid_adjudications),
        "invalid_secondary_flag_trajectories": len(invalid_secondary_flags),
        "task_identity_pass_rate": task_rate,
        "direct_mapping_pass_rate": direct_rate,
        "boundary_alignment_pass_rate": boundary_rate,
        "complete": complete,
        "gate_pass": bool(
            complete
            and task_rate == 1.0
            and direct_rate == 1.0
            and boundary_rate == 1.0
        ),
    }


def finalize_audits(artifact_dir: Path) -> dict[str, Any]:
    """Report on human-owned sheets and emit a fail-closed scientific gate.

    This command deliberately does not promote reviewed strings into the
    eligibility table or regenerate the A/B/C/D manifests.  Promotion is a
    separate curation-and-regeneration stage so a report-only command cannot
    silently change a treatment definition.
    """

    lock_validation = validate_audit_lock(artifact_dir)
    lock_path = artifact_dir / "audit_lock.json"
    lock = load_json(lock_path) if lock_path.exists() else {"sheets": {}}

    def locked_seed(filename: str, fallback: int) -> int:
        value = lock.get("sheets", {}).get(filename, {}).get("audit_seed")
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    command = _command_summary(_rows(artifact_dir / "manual_command_audit.csv"))
    paraphrase = _paraphrase_summary(
        _rows(artifact_dir / "manual_paraphrase_group_audit.csv"),
        secondary_seed=locked_seed(
            "manual_paraphrase_group_audit.csv", DEFAULT_SEED
        ),
    )
    visual = _sequence_summary(
        _rows(artifact_dir / "visual_alignment_audit.csv"),
        minimum=20,
        secondary_seed=locked_seed("visual_alignment_audit.csv", DEFAULT_SEED),
    )
    sequence = _sequence_summary(
        _rows(artifact_dir / "manual_sequence_audit.csv"),
        minimum=30,
        secondary_seed=locked_seed(
            "manual_sequence_audit.csv", DEFAULT_SEED + 30
        ),
    )
    manifest = load_json(artifact_dir / "manifest_validation.json")
    pilot_manifest = load_json(artifact_dir / "pilot_manifest_validation.json")
    split = load_json(artifact_dir / "split_validation.json")
    checks = {
        "structural_manifests_pass": bool(
            manifest.get("all_structural_assertions_pass")
        ),
        "pilot_structural_manifests_pass": bool(
            pilot_manifest.get("all_structural_assertions_pass")
        ),
        "group_safe_nested_splits_pass": bool(
            split.get("all_group_and_role_assertions_pass")
        ),
        "audit_sample_lock_pass": bool(lock_validation.get("sample_sheets_pass")),
        "audit_run_provenance_match": bool(
            lock_validation.get("run_provenance", {}).get("all_pass")
        ),
        "implementation_fingerprint_match": bool(
            lock_validation.get("implementation_match")
        ),
        "command_taxonomy_audit_complete": bool(command["complete"]),
        "paraphrase_audit_pass": bool(
            paraphrase["complete"] and paraphrase["gate_pass"]
        ),
        "visual_alignment_audit_pass": bool(
            visual["complete"] and visual["gate_pass"]
        ),
        "thirty_sequence_audit_pass": bool(
            sequence["complete"] and sequence["gate_pass"]
        ),
        "task_subtask_surface_distributions_matched": bool(
            manifest.get("surface_match_validation", {}).get("all_pass")
        ),
    }
    training_ready = all(checks.values())
    result = {
        "schema_version": 2,
        "mode": "report_only_scientific_gate",
        "decision": (
            "GO" if training_ready else "TRAINING NO-GO; CONDITIONAL GO TO CURATION"
        ),
        "training_ready": training_ready,
        "promotion_applied": False,
        "checks": checks,
        "audit_lock_validation": lock_validation,
        "command_taxonomy_audit": command,
        "paraphrase_group_audit": paraphrase,
        "visual_alignment_audit": visual,
        "sequence_audit": sequence,
        "note": (
            "This gate is fail-closed. Blank, malformed, or incomplete reviewer "
            "fields remain failures; the pipeline never fabricates judgments. "
            "Finalization is report-only: it does not mutate eligibility, "
            "verified surface pools, manifests, or the decision memo. Reviewed "
            "surfaces require an explicit promotion and full regeneration step."
        ),
    }
    write_json(artifact_dir / "human_audit_summary.json", result)
    return result
