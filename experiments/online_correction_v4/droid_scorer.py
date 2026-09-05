"""Terminal scorer binding for the V4 horizontal cube/bowl fixture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.online_correction_v4 import geometry as geom
from experiments.online_correction_v4.adapters import (
    SimulatorSnapshot,
    TerminalPhysicalPredicates,
    TerminalScoringEvidence,
)
from experiments.online_correction_v4.contracts import EpisodeManifestRow, EpisodeRuntimeFlags, TimingConfig
from experiments.online_correction_v4.droid_contract import DroidContractError, FixtureRuntimeBinding
from experiments.online_correction_v4.droid_task_files.registry import (
    blocked_fixture_ids,
    resolve_fixture_registration,
)
from experiments.online_correction_v4.geometry import relation_from_wording
from experiments.online_correction_v4.scoring import ScoringContext, TerminalEvidence, score_terminal_first_placement


class TerminalScorerError(RuntimeError):
    """Raised when terminal scoring geometry cannot be loaded or applied."""


def resolve_file_uri(uri: str, *, label: str) -> Path:
    if not isinstance(uri, str) or not uri.strip():
        raise TerminalScorerError(f"{label} uri is required")
    if uri.startswith("TODO"):
        raise TerminalScorerError(f"{label} uri is not released: {uri}")
    if uri.startswith("file://"):
        return Path(uri[7:]).expanduser().resolve()
    return Path(uri).expanduser().resolve()


def load_scoring_context(
    geometry_path: Path,
    *,
    expected_sha256: str,
    relation: geom.RelationKind,
    d_cap_m: float,
) -> ScoringContext:
    from experiments.online_correction_v4.droid_contract import sha256_file

    if not geometry_path.is_file():
        raise TerminalScorerError(f"geometry file is missing: {geometry_path}")
    digest = sha256_file(geometry_path)
    if digest != expected_sha256:
        raise TerminalScorerError("geometry digest mismatch")
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TerminalScorerError("geometry payload must be an object")

    frame_raw = payload.get("task_frame")
    if not isinstance(frame_raw, dict):
        raise TerminalScorerError("geometry task_frame is required")
    if "u_left" in frame_raw:
        frame = geom.TaskFrame(
            u_left=tuple(float(v) for v in frame_raw["u_left"]),
            u_front=tuple(float(v) for v in frame_raw["u_front"]),
            u_up=tuple(float(v) for v in frame_raw["u_up"]),
            origin=tuple(float(v) for v in frame_raw.get("origin", (0.0, 0.0, 0.0))),
        )
    else:
        frame = geom.TaskFrame(
            u_left=tuple(float(v) for v in frame_raw["x_axis_world"]),
            u_front=tuple(float(v) for v in frame_raw["y_axis_world"]),
            u_up=tuple(float(v) for v in frame_raw["z_axis_world"]),
            origin=tuple(float(v) for v in frame_raw["origin_world"]),
        )

    workspace_raw = payload.get("workspace")
    if not isinstance(workspace_raw, dict):
        raise TerminalScorerError("geometry workspace is required")
    workspace = geom.AxisAlignedBox(
        float(workspace_raw["x_min"]),
        float(workspace_raw["x_max"]),
        float(workspace_raw["y_min"]),
        float(workspace_raw["y_max"]),
        float(workspace_raw["z_min"]),
        float(workspace_raw["z_max"]),
    )

    def _footprint(name: str) -> geom.ObjectFootprint:
        raw = payload.get(name)
        if not isinstance(raw, dict):
            raise TerminalScorerError(f"geometry {name} footprint is required")
        return geom.ObjectFootprint(
            half_left=float(raw["half_left"]),
            half_front=float(raw["half_front"]),
            half_up=float(raw["half_up"]),
        )

    clearance = float(payload.get("clearance_m", 0.01))
    planar = geom.PlanarRelationSpec(
        relation=relation,
        clearance_m=clearance,
        workspace=workspace,
        object_footprint=_footprint("object_footprint"),
        reference_footprint=_footprint("reference_footprint"),
    )
    return ScoringContext(frame=frame, d_cap_m=d_cap_m, planar_spec=planar)


def aggregate_settling_predicates(
    samples: Sequence[TerminalPhysicalPredicates],
    *,
    dwell_ticks: int,
) -> TerminalPhysicalPredicates:
    """Reduce passive-settling samples to one terminal predicate bundle."""
    if not samples:
        return TerminalPhysicalPredicates(
            available=False,
            missing_fields=("settling_samples",),
        )
    if any(not sample.available for sample in samples):
        missing = sorted({field for sample in samples for field in sample.missing_fields})
        return TerminalPhysicalPredicates(
            available=False,
            boundary_violation=any(sample.boundary_violation for sample in samples),
            collision_terminal_failure=any(sample.collision_terminal_failure for sample in samples),
            missing_fields=tuple(missing or ("terminal_predicates",)),
        )
    if any("support_contact_evidence" in sample.missing_fields for sample in samples):
        missing = sorted(
            {field for sample in samples for field in sample.missing_fields}
            | {"support_contact_evidence"}
        )
        return TerminalPhysicalPredicates(
            available=False,
            boundary_violation=any(sample.boundary_violation for sample in samples),
            collision_terminal_failure=any(sample.collision_terminal_failure for sample in samples),
            missing_fields=tuple(missing),
        )
    support_ok = all(sample.allowed_support or sample.allowed_containment for sample in samples)
    stable_window = max(1, int(dwell_ticks))
    stable_run = 0
    stable_for_dwell = False
    for sample in samples:
        if sample.stable_for_dwell:
            stable_run += 1
            if stable_run >= stable_window:
                stable_for_dwell = True
                break
        else:
            stable_run = 0
    return TerminalPhysicalPredicates(
        available=True,
        allowed_support=support_ok and any(sample.allowed_support for sample in samples),
        allowed_containment=any(sample.allowed_containment for sample in samples),
        stable_for_dwell=stable_for_dwell,
        boundary_violation=any(sample.boundary_violation for sample in samples),
        collision_terminal_failure=any(sample.collision_terminal_failure for sample in samples),
        moved_object_mask_pixels=max(sample.moved_object_mask_pixels for sample in samples),
    )


@dataclass
class HorizontalDroidTerminalScorer:
    """Score first placement for the released horizontal cube/bowl fixture."""

    relation: geom.RelationKind
    ctx: ScoringContext
    timing: TimingConfig | None = None

    def score_terminal(
        self,
        *,
        snapshot: SimulatorSnapshot,
        runtime_flags: EpisodeRuntimeFlags,
        passive_settling_reason: str | None,
        grasp_occurred: bool = False,
        carry_verified: bool = False,
        settling_predicates: tuple[TerminalPhysicalPredicates, ...] = (),
    ) -> TerminalScoringEvidence:
        state = snapshot.object_state
        p_obj = (state.object_x, state.object_y, state.object_z_pos)
        p_ref = snapshot.reference_position_world
        tick_predicates = snapshot.terminal_predicates
        dwell_ticks = (
            self.timing.release_detection_dwell_ticks
            if self.timing is not None
            else 2
        )
        if settling_predicates:
            physical = aggregate_settling_predicates(settling_predicates, dwell_ticks=dwell_ticks)
        elif tick_predicates is not None:
            physical = tick_predicates
        else:
            physical = TerminalPhysicalPredicates(
                available=False,
                missing_fields=("terminal_predicates",),
            )
        predicates_available = physical.available
        if passive_settling_reason == "release" and not predicates_available:
            return TerminalScoringEvidence(
                grasp_occurred=grasp_occurred,
                carry_verified=carry_verified,
                released=state.detached,
                unresolved_behavioral_failure=True,
                predicates_available=False,
                metadata={"failure_label": "unresolved_behavioral_failure", "missing_predicates": list(physical.missing_fields)},
            )
        evidence = TerminalEvidence(
            p_obj_world=p_obj,
            p_named_ref_world=p_ref,
            relation=self.relation,
            grasp_occurred=grasp_occurred,
            carry_verified=carry_verified,
            released=state.detached,
            stable_for_dwell=physical.stable_for_dwell if predicates_available else False,
            allowed_support=physical.allowed_support if predicates_available else False,
            allowed_containment=physical.allowed_containment if predicates_available else False,
            boundary_violation=physical.boundary_violation,
            collision_terminal_failure=physical.collision_terminal_failure,
            unresolved_behavioral_failure=not predicates_available,
            transport_incomplete=carry_verified
            and passive_settling_reason == "timeout"
            and not state.detached,
            timeout_without_completion=passive_settling_reason == "timeout" and not state.detached,
        )
        score = score_terminal_first_placement(self.ctx, evidence)
        return TerminalScoringEvidence(
            success=score.success,
            grasp_occurred=evidence.grasp_occurred,
            carry_verified=evidence.carry_verified,
            released=evidence.released,
            geometric_relation_correct=score.predicates["geometric_relation_correct"],
            allowed_support=score.predicates["allowed_support_or_containment"],
            allowed_containment=physical.allowed_containment,
            stable_for_dwell=score.predicates["stable_for_registered_dwell"],
            boundary_violation=evidence.boundary_violation,
            collision_terminal_failure=evidence.collision_terminal_failure,
            transport_incomplete=evidence.transport_incomplete,
            unresolved_behavioral_failure=evidence.unresolved_behavioral_failure,
            predicates_available=predicates_available,
            metadata={
                "failure_label": score.failure_label,
                "goal_violation_capped_m": score.goal_violation_capped_m,
                "settling_samples": len(settling_predicates),
            },
        )


def build_terminal_scorer(
    *,
    manifest: EpisodeManifestRow,
    fixture_binding: FixtureRuntimeBinding,
    scoring_context: ScoringContext | None = None,
    timing: TimingConfig | None = None,
) -> HorizontalDroidTerminalScorer:
    fixture_id = manifest.fixture
    if fixture_id in blocked_fixture_ids():
        raise DroidContractError(
            f"fixture {fixture_id!r} is blocked until asset receipts exist: "
            f"{blocked_fixture_ids()[fixture_id]}"
        )
    resolve_fixture_registration(fixture_id, relation=manifest.factors.get("goal"))
    relation = relation_from_wording(
        manifest.factors["goal"],
        manifest.factors.get("wording", "direct"),
    )
    if scoring_context is not None:
        return HorizontalDroidTerminalScorer(relation=relation, ctx=scoring_context, timing=timing)

    geometry_path = resolve_file_uri(fixture_binding.geometry_uri, label="geometry")
    ctx = load_scoring_context(
        geometry_path,
        expected_sha256=fixture_binding.geometry_sha256,
        relation=relation,
        d_cap_m=fixture_binding.d_cap_m,
    )
    return HorizontalDroidTerminalScorer(relation=relation, ctx=ctx, timing=timing)
