from copy import deepcopy

from steerable_bridge.audits import (
    AUDIT_RUN_PROVENANCE_PATHS,
    _command_summary,
    _paraphrase_summary,
    _sequence_summary,
    build_audit_lock,
    validate_audit_lock,
)
from steerable_bridge.constants import DEFAULT_SEED
from steerable_bridge.io import write_csv, write_json
from steerable_bridge.text import stable_id


def _paraphrase_rows() -> list[dict[str, str]]:
    rows = [
        {
            "intent_id": f"intent_{index:03d}",
            "secondary_review_required": "False",
            "primary_all_surfaces_semantically_equivalent": "yes",
            "primary_language_natural_and_acceptable": "yes",
            "primary_reviewer": "primary",
            "secondary_all_surfaces_semantically_equivalent": "",
            "secondary_language_natural_and_acceptable": "",
            "secondary_reviewer": "",
            "adjudicated_semantically_equivalent": "",
            "adjudicated_language_acceptable": "",
            "adjudicator": "",
            "adjudication_notes": "",
        }
        for index in range(100)
    ]
    expected = set(
        sorted(
            (row["intent_id"] for row in rows),
            key=lambda intent_id: stable_id(
                "group_secondary_review", DEFAULT_SEED, intent_id
            ),
        )[:20]
    )
    for row in rows:
        if row["intent_id"] in expected:
            row["secondary_review_required"] = "True"
            row["secondary_all_surfaces_semantically_equivalent"] = "yes"
            row["secondary_language_natural_and_acceptable"] = "yes"
            row["secondary_reviewer"] = "secondary"
    return rows


def test_paraphrase_audit_requires_locked_independent_secondary_review() -> None:
    rows = _paraphrase_rows()
    assert _paraphrase_summary(rows)["gate_pass"] is True

    cleared = deepcopy(rows)
    for row in cleared:
        row["secondary_review_required"] = "False"
    assert _paraphrase_summary(cleared)["complete"] is False

    same_reviewer = deepcopy(rows)
    for row in same_reviewer:
        if row["secondary_review_required"] == "True":
            row["secondary_reviewer"] = "primary"
    assert _paraphrase_summary(same_reviewer)["complete"] is False

    duplicated = [deepcopy(rows[0]) for _ in rows]
    assert _paraphrase_summary(duplicated)["complete"] is False


def test_paraphrase_disagreement_requires_valid_third_party_adjudication() -> None:
    rows = _paraphrase_rows()
    disputed = next(row for row in rows if row["secondary_review_required"] == "True")
    disputed["secondary_all_surfaces_semantically_equivalent"] = "no"
    summary = _paraphrase_summary(rows)
    assert summary["unresolved_disagreement_groups"] == 1
    assert summary["gate_pass"] is False

    disputed["adjudicated_semantically_equivalent"] = "yes"
    disputed["adjudicator"] = "third-reviewer"
    disputed["adjudication_notes"] = "Resolved after comparing every surface."
    assert _paraphrase_summary(rows)["gate_pass"] is True

    disputed["adjudicated_semantically_equivalent"] = "maybe"
    assert _paraphrase_summary(rows)["complete"] is False


def test_paraphrase_unsolicited_adjudication_cannot_override_primary() -> None:
    rows = _paraphrase_rows()
    row = next(row for row in rows if row["secondary_review_required"] == "False")
    row["primary_all_surfaces_semantically_equivalent"] = "no"
    row["adjudicated_semantically_equivalent"] = "yes"
    row["adjudicator"] = "third-reviewer"
    row["adjudication_notes"] = "No secondary disagreement existed."
    summary = _paraphrase_summary(rows)
    assert summary["semantic_equivalence_pass_rate"] == 0.99
    assert summary["invalid_adjudication_groups"] == 1
    assert summary["complete"] is False


def test_sequence_audit_rejects_tampered_secondary_flags() -> None:
    rows = [
        {
            "steering_trajectory_id": str(index),
            "secondary_review_required": "False",
            "primary_reviewer": "primary",
            "primary_task_identity_believable": "yes",
            "primary_direct_step_i_alignment_believable": "yes",
            "primary_boundary_alignment_believable": "yes",
            "secondary_reviewer": "",
            "secondary_task_identity_believable": "",
            "secondary_direct_step_i_alignment_believable": "",
            "secondary_boundary_alignment_believable": "",
            "adjudicated_task_identity_believable": "",
            "adjudicated_direct_step_i_alignment_believable": "",
            "adjudicated_boundary_alignment_believable": "",
            "adjudicator": "",
            "adjudication_notes": "",
        }
        for index in range(20)
    ]
    expected = set(
        sorted(
            range(20),
            key=lambda trajectory_id: stable_id(
                "sequence_secondary_review", DEFAULT_SEED, trajectory_id
            ),
        )[:4]
    )
    for row in rows:
        if int(row["steering_trajectory_id"]) in expected:
            row["secondary_review_required"] = "True"
            row["secondary_reviewer"] = "secondary"
            row["secondary_task_identity_believable"] = "yes"
            row["secondary_direct_step_i_alignment_believable"] = "yes"
            row["secondary_boundary_alignment_believable"] = "yes"
    assert _sequence_summary(
        rows, 20, secondary_seed=DEFAULT_SEED
    )["gate_pass"] is True
    rows[0]["secondary_review_required"] = "True"
    assert _sequence_summary(
        rows, 20, secondary_seed=DEFAULT_SEED
    )["complete"] is False

    duplicated = [deepcopy(rows[0]) for _ in rows]
    assert _sequence_summary(
        duplicated, 20, secondary_seed=DEFAULT_SEED
    )["complete"] is False


def test_sequence_unsolicited_adjudication_cannot_override_primary() -> None:
    rows = [
        {
            "steering_trajectory_id": str(index),
            "secondary_review_required": "False",
            "primary_reviewer": "primary",
            "primary_task_identity_believable": "yes",
            "primary_direct_step_i_alignment_believable": "yes",
            "primary_boundary_alignment_believable": "yes",
            "secondary_reviewer": "",
            "secondary_task_identity_believable": "",
            "secondary_direct_step_i_alignment_believable": "",
            "secondary_boundary_alignment_believable": "",
            "adjudicated_task_identity_believable": "",
            "adjudicated_direct_step_i_alignment_believable": "",
            "adjudicated_boundary_alignment_believable": "",
            "adjudicator": "",
            "adjudication_notes": "",
        }
        for index in range(20)
    ]
    expected = set(
        sorted(
            range(20),
            key=lambda trajectory_id: stable_id(
                "sequence_secondary_review", DEFAULT_SEED, trajectory_id
            ),
        )[:4]
    )
    for row in rows:
        if int(row["steering_trajectory_id"]) in expected:
            row["secondary_review_required"] = "True"
            row["secondary_reviewer"] = "secondary"
            row["secondary_task_identity_believable"] = "yes"
            row["secondary_direct_step_i_alignment_believable"] = "yes"
            row["secondary_boundary_alignment_believable"] = "yes"
    row = next(row for row in rows if row["secondary_review_required"] == "False")
    row["primary_boundary_alignment_believable"] = "no"
    row["adjudicated_boundary_alignment_believable"] = "yes"
    row["adjudicator"] = "third-reviewer"
    row["adjudication_notes"] = "No secondary disagreement existed."
    summary = _sequence_summary(rows, 20, secondary_seed=DEFAULT_SEED)
    assert summary["boundary_alignment_pass_rate"] == 0.95
    assert summary["invalid_adjudication_trajectories"] == 1
    assert summary["gate_pass"] is False


def test_command_unsolicited_adjudication_cannot_change_confusion() -> None:
    rows = [
        {
            "pool_id": f"pool_{index:03d}",
            "slot_index": "0",
            "automatic_class": "atomic_motion_or_gripper_command",
            "secondary_review_required": "False",
            "primary_manual_class": "atomic_motion_or_gripper_command",
            "primary_same_intent_with_canonical": "yes",
            "primary_reviewer": "primary",
            "secondary_manual_class": "",
            "secondary_same_intent_with_canonical": "",
            "secondary_reviewer": "",
            "adjudicated_class": "",
            "adjudicated_same_intent_with_canonical": "",
            "adjudicator": "",
            "adjudication_notes": "",
        }
        for index in range(100)
    ]
    expected = set(
        sorted(
            (row["pool_id"] for row in rows),
            key=lambda pool_id: stable_id("secondary_review", pool_id),
        )[:20]
    )
    for row in rows:
        if row["pool_id"] in expected:
            row["secondary_review_required"] = "True"
            row["secondary_manual_class"] = "atomic_motion_or_gripper_command"
            row["secondary_same_intent_with_canonical"] = "yes"
            row["secondary_reviewer"] = "secondary"
    row = next(row for row in rows if row["secondary_review_required"] == "False")
    row["primary_manual_class"] = "malformed_or_unclear"
    row["adjudicated_class"] = "atomic_motion_or_gripper_command"
    row["adjudicator"] = "third-reviewer"
    row["adjudication_notes"] = "No secondary disagreement existed."
    summary = _command_summary(rows)
    assert summary["invalid_adjudication_rows"] == 1
    assert summary["automatic_taxonomy_agreement_rate"] == 0.99
    assert summary["complete"] is False


def test_audit_lock_rejects_sample_substitution_and_persists_custom_seed(
    tmp_path,
) -> None:
    command_rows = [
        {"pool_id": f"pool_{index:03d}", "slot_index": "0", "locked": "value"}
        for index in range(100)
    ]
    paraphrase_rows = [
        {
            "intent_id": f"intent_{index:03d}",
            "audit_seed": "42",
            "secondary_review_required": "False",
            "primary_reviewer": "",
        }
        for index in range(100)
    ]
    visual_rows = [
        {
            "steering_trajectory_id": str(index),
            "audit_seed": "42",
            "secondary_review_required": "False",
            "primary_reviewer": "",
        }
        for index in range(20)
    ]
    sequence_rows = [
        {
            "steering_trajectory_id": str(index),
            "audit_seed": "72",
            "secondary_review_required": "False",
            "primary_reviewer": "",
        }
        for index in range(30)
    ]
    for filename, rows in (
        ("manual_command_audit.csv", command_rows),
        ("manual_paraphrase_group_audit.csv", paraphrase_rows),
        ("visual_alignment_audit.csv", visual_rows),
        ("manual_sequence_audit.csv", sequence_rows),
    ):
        write_csv(tmp_path / filename, rows, list(rows[0]))

    write_json(tmp_path / "target_split.json", {"seed": 44})
    write_json(tmp_path / "pilot_split.json", {"seed": 43})
    for relative in AUDIT_RUN_PROVENANCE_PATHS:
        path = tmp_path / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"locked fixture: {relative}\n", encoding="utf-8")

    lock = build_audit_lock(tmp_path, overwrite=True, base_seed=42)
    assert lock["sheets"]["manual_paraphrase_group_audit.csv"]["audit_seed"] == 42
    assert validate_audit_lock(tmp_path)["all_pass"] is True

    paraphrase_rows[0]["primary_reviewer"] = "researcher-a"
    write_csv(
        tmp_path / "manual_paraphrase_group_audit.csv",
        paraphrase_rows,
        list(paraphrase_rows[0]),
    )
    assert validate_audit_lock(tmp_path)["all_pass"] is True

    write_json(tmp_path / "target_split.json", {"seed": 45})
    provenance_drift = validate_audit_lock(tmp_path)
    assert provenance_drift["all_pass"] is False
    assert provenance_drift["run_provenance"]["all_pass"] is False
    write_json(tmp_path / "target_split.json", {"seed": 44})
    assert validate_audit_lock(tmp_path)["all_pass"] is True

    split_validation_path = tmp_path / "split_validation.json"
    locked_split_validation = split_validation_path.read_text(encoding="utf-8")
    split_validation_path.write_text("changed split validation\n", encoding="utf-8")
    assert validate_audit_lock(tmp_path)["run_provenance"]["all_pass"] is False
    split_validation_path.write_text(locked_split_validation, encoding="utf-8")
    assert validate_audit_lock(tmp_path)["all_pass"] is True

    substituted = [deepcopy(paraphrase_rows[0]) for _ in paraphrase_rows]
    write_csv(
        tmp_path / "manual_paraphrase_group_audit.csv",
        substituted,
        list(substituted[0]),
    )
    validation = validate_audit_lock(tmp_path)
    assert validation["all_pass"] is False
    assert validation["sheets"]["manual_paraphrase_group_audit.csv"][
        "identity_rows_unique"
    ] is False
