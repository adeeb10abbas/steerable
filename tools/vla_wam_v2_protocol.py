#!/usr/bin/env python3
"""Shared protocol and prompt rendering utilities for v2 model adapters."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROMPT_IDS = (
    "direct_command",
    "short_command",
    "goal_as_outcome",
    "desired_plus_negated_opposite",
)
DIRECTIONS = ("left", "right")


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        protocol = json.load(handle)
    if protocol.get("study_id") != "vla_wam_language_steerability_v2":
        raise ValueError(f"Not the frozen v2 protocol: {path}")
    return protocol


def prompt_family(protocol: dict[str, Any], family_id: str) -> dict[str, Any]:
    matches = [family for family in protocol["prompt_families"] if family["id"] == family_id]
    if len(matches) != 1:
        raise KeyError(f"Expected one prompt family {family_id!r}, found {len(matches)}")
    return matches[0]


def canonical_short_object_name(raw_model_name: str) -> str:
    """Derive a modifier-light noun from a RoboTwin raw model identifier."""

    value = re.sub(r"^\d+[_-]", "", raw_model_name.strip().lower())
    value = value.replace("playingcards", "playing cards")
    value = value.replace("woodenblock", "wooden block")
    value = value.replace("toothbrush", "toothbrush")
    value = re.sub(r"[_-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        raise ValueError(f"Could not derive short object name from {raw_model_name!r}")
    return value


def normalize_object_description(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    value = value.rstrip(". ")
    if not value:
        raise ValueError("Object description must not be empty")
    return value


def first_seen_object_description(
    robotwin_root: Path, raw_model_name: str, model_id: int
) -> str:
    path = (
        robotwin_root
        / "description"
        / "objects_description"
        / raw_model_name
        / f"base{model_id}.json"
    )
    with path.open() as handle:
        payload = json.load(handle)
    descriptions = payload.get("seen") or []
    if descriptions:
        return normalize_object_description(str(descriptions[0]))
    fallback = payload.get("raw_description") or canonical_short_object_name(raw_model_name)
    return normalize_object_description(str(fallback))


def render_prompt(
    protocol: dict[str, Any],
    *,
    family_id: str,
    direction: str,
    movable: str,
    reference: str,
    movable_short: str | None = None,
    arena: str,
) -> str:
    if direction not in DIRECTIONS:
        raise ValueError(f"Unsupported direction: {direction}")
    family = prompt_family(protocol, family_id)
    if arena == "droid_robolab" and family_id == "short_command":
        exact_key = f"droid_exact_{direction}"
        prompt = family[exact_key]
    else:
        values = {
            "movable": normalize_object_description(movable),
            "reference": normalize_object_description(reference),
            "movable_short": normalize_object_description(movable_short or movable),
        }
        prompt = family[direction].format(**values)
    desired_count = len(re.findall(rf"\b{direction}\b", prompt.lower()))
    if desired_count != 1:
        raise ValueError(
            f"Rendered {family_id}/{direction} prompt has {desired_count} desired tokens: {prompt}"
        )
    opposite = "right" if direction == "left" else "left"
    opposite_count = len(re.findall(rf"\b{opposite}\b", prompt.lower()))
    expected_opposite = 1 if family_id == "desired_plus_negated_opposite" else 0
    if opposite_count != expected_opposite:
        raise ValueError(
            f"Rendered {family_id}/{direction} prompt has {opposite_count} opposite tokens: {prompt}"
        )
    if family_id == "desired_plus_negated_opposite" and "not" not in prompt.lower():
        raise ValueError(f"Contrastive prompt lost negation: {prompt}")
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--family", choices=PROMPT_IDS, required=True)
    parser.add_argument("--direction", choices=DIRECTIONS, required=True)
    parser.add_argument("--movable", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--movable-short")
    parser.add_argument("--arena", choices=("droid_robolab", "robotwin_place_a2b"), required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol.resolve())
    print(
        render_prompt(
            protocol,
            family_id=args.family,
            direction=args.direction,
            movable=args.movable,
            reference=args.reference,
            movable_short=args.movable_short,
            arena=args.arena,
        )
    )


if __name__ == "__main__":
    main()
