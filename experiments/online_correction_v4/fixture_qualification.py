"""Fixture-parameterized qualification metadata for V4 G4-G8 gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from experiments.online_correction_v4.droid_task_files.constants import (
    fixture_object_spec,
)
from experiments.online_correction_v4.geometry import build_prompt
from experiments.online_correction_v4.model_blind_g3 import (
    path_scale_receipt_schema,
)


PolicyId = Literal["cosmos3_nano_droid"]
G3BasisKind = Literal["path_scale", "scripted_aggregate"]


@dataclass(frozen=True)
class G4PromptPair:
    primary_label: str
    alternate_label: str
    primary_prompt: str
    alternate_prompt: str


@dataclass(frozen=True)
class PilotAllocationProfile:
    env_seed_start: int
    env_seed_end: int  # inclusive
    stationary_count: int = 16
    motion_count: int = 8
    goal_cycle: tuple[str, ...] = ("inside",)
    wording_cycle: tuple[str, ...] = ("direct", "inverse")
    scenario_stationary_first: str = "original_sham"
    scenario_stationary_second: str = "destination_static"
    scenario_motion: str = "move_stop"


@dataclass(frozen=True)
class FixtureQualificationProfile:
    fixture_id: str
    family_id: str
    policy_id: PolicyId
    g3_basis: G3BasisKind
    g3_path_scale_schema: str
    g3_scripted_aggregate_schema: str | None
    g4_receipt_schema: str
    g5_receipt_schema: str
    g6_receipt_schema: str
    g4_sampling_seed: int
    horizontal_prompt_suffix: bool
    g4_prompt_pair: G4PromptPair
    pilot_allocation: PilotAllocationProfile | None = None
    confirmatory_episode_count: int = 768
    confirmatory_block_seed_count: int = 64
    nano_seed_registry_schema: str = "v4-nano-policy-seed-registry-v1"
    pilot_seed_registry_schema: str | None = None
    hardware_stratum: str | None = None
    writer_output_parent: str | None = None


def _g4_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g4-nano-policy-session-receipt-v1"


def _g5_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g5-trigger-branch-receipt-v1"


def _g6_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g6-measurement-receipt-v1"


def _pilot_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-engineering-pilot-seed-registry-v1"


def _object_pair_pilot_allocation() -> PilotAllocationProfile:
    return PilotAllocationProfile(
        env_seed_start=2110000800,
        env_seed_end=2110000823,
        goal_cycle=("left", "right", "front", "behind"),
        wording_cycle=("direct",),
    )


def _containment_pilot_allocation() -> PilotAllocationProfile:
    return PilotAllocationProfile(
        env_seed_start=2110000600,
        env_seed_end=2110000623,
    )


def _object_pair_g4_prompts() -> G4PromptPair:
    left = build_prompt("sponge", "tray", "left", "direct")
    right = build_prompt("sponge", "tray", "right", "direct")
    return G4PromptPair("left", "right", left, right)


def _containment_g4_prompts() -> G4PromptPair:
    direct = build_prompt("cube", "bowl", "inside", "direct", horizontal=False)
    inverse = build_prompt("cube", "bowl", "inside", "inverse", horizontal=False)
    return G4PromptPair("direct", "inverse", direct, inverse)


FIXTURE_QUALIFICATION_PROFILES: dict[str, FixtureQualificationProfile] = {
    "object_pair": FixtureQualificationProfile(
        fixture_id="object_pair",
        family_id="C7",
        policy_id="cosmos3_nano_droid",
        g3_basis="scripted_aggregate",
        g3_path_scale_schema=path_scale_receipt_schema("object_pair"),
        g3_scripted_aggregate_schema="v4-object-pair-g3-aggregate-receipt-v1",
        g4_receipt_schema=_g4_schema("object_pair"),
        g5_receipt_schema=_g5_schema("object_pair"),
        g6_receipt_schema=_g6_schema("object_pair"),
        g4_sampling_seed=2110000800,
        horizontal_prompt_suffix=True,
        g4_prompt_pair=_object_pair_g4_prompts(),
        pilot_allocation=_object_pair_pilot_allocation(),
        pilot_seed_registry_schema="v4-object-pair-engineering-pilot-seed-registry-v1",
        hardware_stratum="a10080-policy_a40-simulator",
        writer_output_parent="/data/users/ali/vla_wam/raw/v4/g7-object-pair",
    ),
    "containment": FixtureQualificationProfile(
        fixture_id="containment",
        family_id="C6",
        policy_id="cosmos3_nano_droid",
        g3_basis="path_scale",
        g3_path_scale_schema=path_scale_receipt_schema("containment"),
        g3_scripted_aggregate_schema=None,
        g4_receipt_schema=_g4_schema("containment"),
        g5_receipt_schema=_g5_schema("containment"),
        g6_receipt_schema=_g6_schema("containment"),
        g4_sampling_seed=2110030800,
        horizontal_prompt_suffix=False,
        g4_prompt_pair=_containment_g4_prompts(),
        pilot_allocation=_containment_pilot_allocation(),
        pilot_seed_registry_schema=_pilot_schema("containment"),
        hardware_stratum="a10040-policy_b200-simulator",
        writer_output_parent="/data/users/ali/vla_wam/raw/v4/g7-containment",
    ),
}


def qualification_profile(fixture_id: str) -> FixtureQualificationProfile:
    try:
        return FIXTURE_QUALIFICATION_PROFILES[fixture_id]
    except KeyError as exc:
        raise ValueError(f"unsupported qualification fixture: {fixture_id!r}") from exc


def manipulated_and_reference(fixture_id: str) -> tuple[str, str]:
    spec = fixture_object_spec(fixture_id)
    return spec.target_object, spec.reference_object
