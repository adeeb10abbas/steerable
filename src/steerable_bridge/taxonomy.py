from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Iterator, Mapping

from .text import (
    ANGLE_PLACEHOLDER_RE,
    BRACKET_LIKE_RE,
    coordinate_pairs,
    normalize_semantic_text,
    source_collection,
    stable_id,
)


ACTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "reach": re.compile(r"\b(?:reach|approach|go to|move (?:toward|towards|to)|position .* near)\b"),
    "grasp": re.compile(r"\b(?:grasp|grab|grip|close (?:the )?gripper on)\b"),
    "lift": re.compile(r"\b(?:lift|raise|bring up)\b"),
    "transport": re.compile(r"\b(?:carry|transport|bring|move .* (?:toward|towards|to|over|into))\b"),
    "place": re.compile(r"\b(?:place|put|set down|position .* (?:in|inside|on|onto))\b"),
    "release": re.compile(r"\b(?:release|drop|open (?:the )?gripper)\b"),
    "open": re.compile(r"\bopen\b"),
    "close": re.compile(r"\bclose\b"),
    "rotate": re.compile(r"\b(?:rotate|turn)\b"),
    "pull": re.compile(r"\b(?:pull|drag)\b"),
    "push": re.compile(r"\b(?:push|press)\b"),
    "stop": re.compile(r"\b(?:stop|halt|remain in place|hold position|terminate|task complete|end)\b"),
}

SEQUENCE_RE = re.compile(
    r"\b(?:then|and then|before|after|while|followed by|in preparation for|prepare for)\b"
)
ELLIPSIS_RE = re.compile(r"(?:\.\.\.|…)")
ATOMIC_RE = re.compile(
    r"^(?:"
    r"stop|halt|wait|end|task complete|stop moving|come to a stop|remain in place|hold position|"
    r"open (?:the )?gripper|close (?:the )?gripper|release|grasp|"
    r"(?:move|go|shift|rotate|turn)(?: (?:the )?(?:gripper|arm|end effector))?"
    r"(?: (?:slightly|slowly|further))?"
    r" (?:up|down|left|right|forward|back|backward|backwards|away|clockwise|counter clockwise)"
    r")$"
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "away",
    "back",
    "backward",
    "backwards",
    "before",
    "by",
    "down",
    "forward",
    "for",
    "from",
    "in",
    "inside",
    "into",
    "it",
    "left",
    "near",
    "of",
    "on",
    "onto",
    "over",
    "right",
    "slightly",
    "the",
    "then",
    "to",
    "toward",
    "towards",
    "up",
    "while",
    "with",
}
ACTION_WORDS = {
    "approach",
    "bring",
    "carry",
    "close",
    "drop",
    "end",
    "go",
    "grab",
    "grasp",
    "grip",
    "halt",
    "lift",
    "move",
    "open",
    "pick",
    "place",
    "position",
    "pull",
    "push",
    "put",
    "raise",
    "reach",
    "release",
    "reposition",
    "rotate",
    "set",
    "shift",
    "stop",
    "take",
    "transport",
    "turn",
}
GENERIC_ENTITY_WORDS = {
    "arm",
    "effector",
    "execution",
    "gripper",
    "location",
    "motion",
    "movement",
    "position",
    "scene",
    "space",
    "step",
    "task",
}
CONSTRAINT_WORDS = {
    "above",
    "after",
    "away",
    "back",
    "backward",
    "backwards",
    "before",
    "below",
    "behind",
    "clockwise",
    "counterclockwise",
    "counter",
    "down",
    "forward",
    "front",
    "in",
    "inside",
    "into",
    "left",
    "near",
    "off",
    "on",
    "onto",
    "over",
    "right",
    "through",
    "toward",
    "towards",
    "under",
    "up",
}


def action_families(normalized: str) -> set[str]:
    return {name for name, pattern in ACTION_PATTERNS.items() if pattern.search(normalized)}


def composition_action_families(normalized: str) -> set[str]:
    """Return action families with lexical aliases collapsed.

    ``open gripper`` is intentionally recognized by both the broad ``open``
    pattern and the semantically sharper ``release`` pattern.  That overlap is
    useful for canonical-family comparison, but it is not evidence that this
    phrase contains two distinct actions.  Rewrite only that lexical span so a
    separate action such as ``open drawer`` remains visible.
    """

    collapsed = re.sub(r"\bopen (?:the )?gripper\b", "release", normalized)
    collapsed = re.sub(r"\bclose (?:the )?gripper\b", "grasp", collapsed)
    return action_families(collapsed)


def entity_tokens(normalized: str) -> set[str]:
    return {
        token
        for token in normalized.split()
        if token not in STOPWORDS
        and token not in ACTION_WORDS
        and token not in GENERIC_ENTITY_WORDS
        and not token.isdigit()
    }


def constraint_tokens(normalized: str) -> set[str]:
    return set(constraint_sequence(normalized))


def constraint_sequence(normalized: str) -> tuple[str, ...]:
    return tuple(token for token in normalized.split() if token in CONSTRAINT_WORDS)


def _placeholder_flags(text: str) -> tuple[bool, int]:
    placeholders = ANGLE_PLACEHOLDER_RE.findall(text)
    if not placeholders:
        return False, 0
    numeric = re.compile(r"^<\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*>$")
    numeric_count = sum(bool(numeric.match(value)) for value in placeholders)
    has_unresolved = any(not numeric.match(value) for value in placeholders)
    return has_unresolved, numeric_count


def classify_command(
    command: object,
    *,
    canonical_task: str,
    canonical_subtask: str,
) -> dict[str, Any]:
    if not isinstance(command, str):
        return {
            "automatic_class": "malformed_or_unclear",
            "rule_reason": "non_string_command",
            "normalized_command": "",
            "coordinate_count": 0,
            "same_intent_candidate": False,
            "needs_semantic_review": False,
        }
    text = command.strip()
    normalized = normalize_semantic_text(text)
    if not normalized:
        return {
            "automatic_class": "malformed_or_unclear",
            "rule_reason": "empty_after_normalization",
            "normalized_command": normalized,
            "coordinate_count": 0,
            "same_intent_candidate": False,
            "needs_semantic_review": False,
        }

    task_norm = normalize_semantic_text(canonical_task)
    subtask_norm = normalize_semantic_text(canonical_subtask)
    coords = coordinate_pairs(text)
    unresolved_placeholder, numeric_angle_coordinate_count = _placeholder_flags(text)
    square_bracket_residue = bool(BRACKET_LIKE_RE.search(text)) and not coords

    common = {
        "normalized_command": normalized,
        "coordinate_count": len(coords),
        "has_unresolved_placeholder": unresolved_placeholder,
        "has_numeric_angle_coordinate": numeric_angle_coordinate_count > 0,
        "has_unparsed_square_bracket": square_bracket_residue,
    }
    if normalized == task_norm:
        return {
            **common,
            "automatic_class": "task_level_language",
            "rule_reason": "shared_trajectory_task_exact_normalized_match",
            "same_intent_candidate": False,
            "needs_semantic_review": False,
        }
    if unresolved_placeholder or square_bracket_residue or ELLIPSIS_RE.search(text):
        return {
            **common,
            "automatic_class": "malformed_or_unclear",
            "rule_reason": (
                "unresolved_placeholder"
                if unresolved_placeholder
                else "unparsed_square_bracket"
                if square_bracket_residue
                else "ellipsis_or_truncation"
            ),
            "same_intent_candidate": False,
            "needs_semantic_review": False,
        }
    effective_coordinate_count = len(coords) + numeric_angle_coordinate_count
    canonical_families = action_families(subtask_norm)
    command_families = action_families(normalized)
    command_composition_families = composition_action_families(normalized)
    movement_families = {"reach", "lift", "transport", "rotate", "pull", "push"}
    terminal_families = {"grasp", "place", "release", "open", "close"}
    coordinate_hybrid = bool(
        SEQUENCE_RE.search(normalized)
        or (
            command_composition_families & movement_families
            and command_composition_families & terminal_families
        )
        or len(command_composition_families & terminal_families) > 1
    )
    if effective_coordinate_count and coordinate_hybrid:
        return {
            **common,
            "coordinate_count": effective_coordinate_count,
            "automatic_class": "hybrid_or_added_behavioral_semantics",
            "rule_reason": "coordinate_plus_terminal_action_direction_or_sequence",
            "same_intent_candidate": False,
            "needs_semantic_review": False,
        }
    if effective_coordinate_count >= 2:
        return {
            **common,
            "coordinate_count": effective_coordinate_count,
            "automatic_class": "multi_point_gripper_trace",
            "rule_reason": "two_or_more_coordinate_pairs",
            "same_intent_candidate": False,
            "needs_semantic_review": False,
        }
    if effective_coordinate_count == 1:
        return {
            **common,
            "coordinate_count": effective_coordinate_count,
            "automatic_class": "point_coordinate_grounding",
            "rule_reason": "one_coordinate_pair",
            "same_intent_candidate": False,
            "needs_semantic_review": False,
        }
    strict_primitive_compound = (
        not entity_tokens(normalized)
        and bool(
            re.search(
                r"\b(?:move|go|shift|rotate|turn|open|close|release|stop|halt|wait)\b",
                normalized,
            )
        )
    )
    if ATOMIC_RE.fullmatch(normalized) or strict_primitive_compound:
        return {
            **common,
            "automatic_class": "atomic_motion_or_gripper_command",
            "rule_reason": "high_precision_atomic_pattern",
            "same_intent_candidate": False,
            "needs_semantic_review": False,
        }

    canonical_entities = entity_tokens(subtask_norm)
    command_entities = entity_tokens(normalized)
    canonical_constraints = constraint_tokens(subtask_norm)
    command_constraints = constraint_tokens(normalized)
    ordered_constraints_compatible = (
        constraint_sequence(subtask_norm) == constraint_sequence(normalized)
    )

    if SEQUENCE_RE.search(normalized) or len(command_families) > max(1, len(canonical_families)):
        return {
            **common,
            "automatic_class": "hybrid_or_added_behavioral_semantics",
            "rule_reason": "sequencing_marker_or_additional_action_family",
            "same_intent_candidate": False,
            "needs_semantic_review": False,
        }

    exact_canonical = normalized == subtask_norm
    family_compatible = bool(canonical_families) and command_families == canonical_families
    entity_compatible = canonical_entities == command_entities
    constraint_compatible = canonical_constraints == command_constraints
    if exact_canonical or (
        family_compatible
        and entity_compatible
        and constraint_compatible
        and ordered_constraints_compatible
        and len(canonical_entities) <= 1
    ):
        return {
            **common,
            "automatic_class": "subtask_same_intent_paraphrase_candidate",
            "rule_reason": (
                "canonical_subtask_exact_normalized_match"
                if exact_canonical
                else "matching_action_family_and_no_added_entities"
            ),
            "same_intent_candidate": True,
            "needs_semantic_review": not exact_canonical,
        }

    return {
        **common,
        "automatic_class": "malformed_or_unclear",
        "rule_reason": "semantic_class_unclear_requires_manual_review",
        "same_intent_candidate": False,
        "needs_semantic_review": True,
    }


def iter_taxonomy_rows(
    commands: Mapping[str, Any],
    steering_records: Mapping[int, Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    for raw_trajectory_id, raw_pools in commands.items():
        trajectory_id = int(raw_trajectory_id)
        steering = steering_records.get(trajectory_id, {})
        canonical_task = str(steering.get("canonical_task", ""))
        pools = raw_pools if isinstance(raw_pools, dict) else {}
        merged_pools: dict[str, dict[str, Any]] = {}
        for pool_index, (canonical_subtask, raw_commands) in enumerate(pools.items()):
            normalized_subtask = normalize_semantic_text(str(canonical_subtask))
            merged = merged_pools.setdefault(
                normalized_subtask,
                {
                    "canonical_subtask": str(canonical_subtask),
                    "pool_index": pool_index,
                    "commands": [],
                },
            )
            values = raw_commands if isinstance(raw_commands, list) else [raw_commands]
            merged["commands"].extend(values)
        for normalized_subtask, merged in merged_pools.items():
            canonical_subtask = merged["canonical_subtask"]
            pool_index = int(merged["pool_index"])
            pool_id = stable_id(
                "pool", trajectory_id, normalized_subtask
            )
            values = merged["commands"]
            for slot_index, command in enumerate(values):
                classification = classify_command(
                    command,
                    canonical_task=canonical_task,
                    canonical_subtask=str(canonical_subtask),
                )
                yield {
                    "steering_trajectory_id": trajectory_id,
                    "source_collection": steering.get("source_collection", "unknown"),
                    "pool_id": pool_id,
                    "pool_index": pool_index,
                    "canonical_task": canonical_task,
                    "canonical_subtask": canonical_subtask,
                    "slot_index": slot_index,
                    "command": command if isinstance(command, str) else json.dumps(command),
                    **classification,
                }


def make_manual_pool_audit(
    commands: Mapping[str, Any],
    steering_records: Mapping[int, Mapping[str, Any]],
    sample_size: int = 100,
) -> list[dict[str, Any]]:
    by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen_pool_ids: set[str] = set()
    for raw_trajectory_id, pools in commands.items():
        trajectory_id = int(raw_trajectory_id)
        if not isinstance(pools, dict):
            continue
        source = str(steering_records.get(trajectory_id, {}).get("source_collection", "unknown"))
        for subtask in pools:
            pool_id = stable_id(
                "pool", trajectory_id, normalize_semantic_text(str(subtask))
            )
            if pool_id in seen_pool_ids:
                continue
            seen_pool_ids.add(pool_id)
            by_source[source].append((str(raw_trajectory_id), str(subtask)))

    sources = sorted(by_source)
    locked_100_allocation = {
        "bridge_data_v2": 32,
        "bridge_data_v1": 23,
        "rss": 21,
        "icra": 13,
        "flap": 11,
    }
    if sample_size == 100 and set(sources) == set(locked_100_allocation):
        allocation = locked_100_allocation
    else:
        total_pools = sum(len(values) for values in by_source.values())
        exact = {
            source: sample_size * len(by_source[source]) / total_pools for source in sources
        }
        allocation = {source: int(exact[source]) for source in sources}
        remaining = sample_size - sum(allocation.values())
        for source in sorted(
            sources, key=lambda value: exact[value] - allocation[value], reverse=True
        )[:remaining]:
            allocation[source] += 1
    selected: list[tuple[str, str]] = []
    for source in sources:
        target = allocation[source]
        ranked = sorted(
            by_source[source],
            key=lambda item: stable_id(
                "rank", source, item[0], normalize_semantic_text(item[1])
            ),
        )
        selected.extend(ranked[:target])

    selected_pool_ids = {
        stable_id("pool", int(raw_trajectory_id), normalize_semantic_text(subtask))
        for raw_trajectory_id, subtask in selected
    }
    secondary_review_pool_ids = set(
        sorted(
            selected_pool_ids,
            key=lambda pool_id: stable_id("secondary_review", pool_id),
        )[: max(1, round(0.20 * len(selected_pool_ids)))]
    )

    rows: list[dict[str, Any]] = []
    for raw_trajectory_id, subtask in selected:
        trajectory_id = int(raw_trajectory_id)
        steering = steering_records[trajectory_id]
        canonical_task = str(steering.get("canonical_task", ""))
        pool = commands[raw_trajectory_id][subtask]
        values = pool if isinstance(pool, list) else [pool]
        for slot_index, command in enumerate(values):
            classification = classify_command(
                command,
                canonical_task=canonical_task,
                canonical_subtask=subtask,
            )
            pool_id = stable_id(
                "pool", trajectory_id, normalize_semantic_text(subtask)
            )
            rows.append(
                {
                    "pool_id": pool_id,
                    "steering_trajectory_id": trajectory_id,
                    "source_collection": steering.get("source_collection", "unknown"),
                    "canonical_task": canonical_task,
                    "canonical_subtask": subtask,
                    "slot_index": slot_index,
                    "command": command if isinstance(command, str) else json.dumps(command),
                    "automatic_class": classification["automatic_class"],
                    "rule_reason": classification["rule_reason"],
                    "secondary_review_required": pool_id in secondary_review_pool_ids,
                    "primary_manual_class": "",
                    "primary_same_intent_with_canonical": "",
                    "primary_reviewer": "",
                    "primary_review_notes": "",
                    "secondary_manual_class": "",
                    "secondary_same_intent_with_canonical": "",
                    "secondary_reviewer": "",
                    "secondary_review_notes": "",
                    "adjudicated_class": "",
                    "adjudicated_same_intent_with_canonical": "",
                    "adjudicator": "",
                    "adjudication_notes": "",
                }
            )
    return rows


def generated_wrapper_paraphrases(canonical: str) -> tuple[list[str], list[str]]:
    """Meaning-preserving but deliberately low-strength construction candidates.

    The canonical instruction remains verbatim inside every wrapper. These are
    structurally safe candidates, not a claim of broad lexical diversity.
    """
    base = canonical.strip().rstrip(".?!")
    variants = [
        f'Please follow this instruction: "{base}."',
        f'Execute the following instruction: "{base}."',
        f'The requested robot instruction is: "{base}."',
        f'Carry out this command: "{base}."',
        f'Robot command: "{base}."',
        f'Complete this instruction: "{base}."',
    ]
    return variants[:4], variants[4:]


def summarize_taxonomy(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    slot_counts: Counter[str] = Counter()
    normalized_context_classes: dict[str, set[str]] = defaultdict(set)
    normalized_seen: set[tuple[str, str]] = set()
    normalized_extra_duplicates = 0
    total = 0
    for row in rows:
        total += 1
        class_name = str(row["automatic_class"])
        slot_counts[class_name] += 1
        normalized = str(row.get("normalized_command", ""))
        if normalized:
            normalized_context_classes[normalized].add(class_name)
            key = (str(row["pool_id"]), normalized)
            if key in normalized_seen:
                normalized_extra_duplicates += 1
            else:
                normalized_seen.add(key)
    unique_counts: Counter[str] = Counter()
    for classes in normalized_context_classes.values():
        if len(classes) == 1:
            unique_counts[next(iter(classes))] += 1
        else:
            unique_counts["mixed_across_contexts"] += 1
    return {
        "command_slots": total,
        "slot_counts": dict(sorted(slot_counts.items())),
        "slot_shares": {
            name: count / total if total else 0.0 for name, count in sorted(slot_counts.items())
        },
        "normalized_unique_strings": len(normalized_context_classes),
        "normalized_unique_string_counts": dict(sorted(unique_counts.items())),
        "within_pool_normalized_extra_duplicate_slots": normalized_extra_duplicates,
    }
