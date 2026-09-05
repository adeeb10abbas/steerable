"""Model-blind preparation gate framework (contracts and accounting only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, runtime_checkable


class GateOutcome(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    PENDING = "pending"


@dataclass(frozen=True)
class ScaleLadderCandidate:
    scale: float
    fixture_id: str
    jointly_feasible: Optional[bool] = None
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class GoalAreaGateCase:
    fixture_id: str
    reset_id: str
    relation: str
    motion_sign: int
    shrinking_direction: bool
    original_area_m2: float
    destination_area_m2: float
    overlap_fraction: float
    passes_information_gate: bool


@dataclass(frozen=True)
class ScriptedCheckReceipt:
    check_kind: str
    fixture_id: str
    goal: str
    reference_case: str
    reset_case: str
    reference_position: str
    passed: bool
    evidence_uri: Optional[str] = None
    evidence_sha256: Optional[str] = None


SCRIPTED_STATIONARY_CHECKS_PER_GEOMETRY = 621
SCRIPTED_MOVEMENT_CHECKS_PER_GEOMETRY = 23
SCRIPTED_TOTAL_PER_GEOMETRY = SCRIPTED_STATIONARY_CHECKS_PER_GEOMETRY + SCRIPTED_MOVEMENT_CHECKS_PER_GEOMETRY


@runtime_checkable
class GeometryFeasibilityProvider(Protocol):
    def evaluate_scale(self, fixture_id: str, scale: float) -> ScaleLadderCandidate:
        ...

    def goal_area_cases(self, fixture_id: str, scale: float) -> Iterable[GoalAreaGateCase]:
        ...

    def run_scripted_check(self, spec: Mapping[str, Any]) -> ScriptedCheckReceipt:
        ...


@runtime_checkable
class VisibilityDetectorContract(Protocol):
    """Evaluator-only visibility proxy; never policy input."""

    displacement_threshold_m: float

    def qualifies_changed_observation(
        self,
        *,
        observation_id: str,
        reference_displacement_m: float,
        moved_object_mask_pixels: int,
        camera_ids: Iterable[str],
        moved_object_mask_pixels_by_camera: Mapping[str, int] | None = None,
    ) -> bool:
        ...


@runtime_checkable
class ResponseDetectorContract(Protocol):
    displacement_threshold_m: float
    dwell_control_ticks: int
    sensitivity_thresholds_m: tuple[float, ...]

    def qualifies_response(
        self,
        *,
        object_displacement_m: float,
        sham_displacement_m: float,
        dwell_ticks: int,
        under_robot_control: bool,
    ) -> bool:
        ...


@dataclass
class PreparationGateFramework:
    scale_candidates: tuple[float, ...]
    minimum_shrinking_area_fraction: float = 0.20
    goal_area_gate_fixtures: tuple[str, ...] = ("horizontal", "reference_binding")
    geometry: Optional[GeometryFeasibilityProvider] = None
    visibility_detector: Optional[VisibilityDetectorContract] = None
    response_detector: Optional[ResponseDetectorContract] = None
    scripted_receipts: list[ScriptedCheckReceipt] = field(default_factory=list)
    selected_scales: dict[str, float] = field(default_factory=dict)
    gate_outcomes: dict[str, GateOutcome] = field(default_factory=dict)

    def select_fixture_scale(self, fixture_id: str) -> tuple[GateOutcome, Optional[float], list[str]]:
        if self.geometry is None:
            return GateOutcome.PENDING, None, ["geometry provider not bound"]
        errors: list[str] = []
        for scale in self.scale_candidates:
            candidate = self.geometry.evaluate_scale(fixture_id, scale)
            if candidate.jointly_feasible is not True:
                errors.append(candidate.rejection_reason or f"scale {scale} infeasible")
                continue
            if fixture_id in self.goal_area_gate_fixtures and not self._passes_goal_area_gate(fixture_id, scale):
                errors.append(f"scale {scale} fails 20% shrinking-direction information gate")
                continue
            self.selected_scales[fixture_id] = scale
            self.gate_outcomes[f"scale:{fixture_id}"] = GateOutcome.PASSED
            return GateOutcome.PASSED, scale, errors
        self.gate_outcomes[f"scale:{fixture_id}"] = GateOutcome.BLOCKED
        return GateOutcome.BLOCKED, None, errors

    def _passes_goal_area_gate(self, fixture_id: str, scale: float) -> bool:
        if self.geometry is None:
            return False
        shrinking_cases = [
            case
            for case in self.geometry.goal_area_cases(fixture_id, scale)
            if case.shrinking_direction
        ]
        if not shrinking_cases:
            return True
        return all(case.passes_information_gate for case in shrinking_cases)

    def account_scripted_checks(
        self,
        *,
        geometry_candidate_id: str,
        run_check: Callable[[Mapping[str, Any]], ScriptedCheckReceipt],
        check_specs: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        passed = 0
        failed = 0
        for spec in check_specs:
            receipt = run_check(spec)
            self.scripted_receipts.append(receipt)
            if receipt.passed:
                passed += 1
            else:
                failed += 1
        total = passed + failed
        outcome = GateOutcome.PASSED if failed == 0 and total > 0 else GateOutcome.FAILED if failed else GateOutcome.PENDING
        key = f"scripted:{geometry_candidate_id}"
        self.gate_outcomes[key] = outcome
        return {
            "geometry_candidate_id": geometry_candidate_id,
            "expected_stationary": SCRIPTED_STATIONARY_CHECKS_PER_GEOMETRY,
            "expected_movement": SCRIPTED_MOVEMENT_CHECKS_PER_GEOMETRY,
            "expected_total": SCRIPTED_TOTAL_PER_GEOMETRY,
            "observed_total": total,
            "passed": passed,
            "failed": failed,
            "outcome": outcome.value,
        }

    def bind_visibility_contract(self, detector: VisibilityDetectorContract) -> None:
        self.visibility_detector = detector

    def bind_response_contract(self, detector: ResponseDetectorContract) -> None:
        self.response_detector = detector

    def evaluate_visibility_contract(self) -> GateOutcome:
        if self.visibility_detector is None:
            self.gate_outcomes["visibility_detector"] = GateOutcome.PENDING
            return GateOutcome.PENDING
        self.gate_outcomes["visibility_detector"] = GateOutcome.PASSED
        return GateOutcome.PASSED

    def evaluate_response_contract(self) -> GateOutcome:
        if self.response_detector is None:
            self.gate_outcomes["response_detector"] = GateOutcome.PENDING
            return GateOutcome.PENDING
        self.gate_outcomes["response_detector"] = GateOutcome.PASSED
        return GateOutcome.PASSED


@dataclass(frozen=True)
class FrozenChangedObservationDetector:
    """Model-blind visibility proxy over evaluator-side observation audit fields."""

    displacement_threshold_m: float

    def qualifies_changed_observation(
        self,
        *,
        observation_id: str,
        reference_displacement_m: float,
        moved_object_mask_pixels: int,
        camera_ids: Iterable[str],
        moved_object_mask_pixels_by_camera: Mapping[str, int] | None = None,
    ) -> bool:
        from experiments.online_correction_v4.observation_audit import (
            ObservationAuditEvidence,
            evaluate_changed_observation_visibility,
        )

        mask_by_camera = dict(moved_object_mask_pixels_by_camera or {})
        if not mask_by_camera and moved_object_mask_pixels > 0:
            camera_list = tuple(str(camera_id) for camera_id in camera_ids)
            if len(camera_list) == 1:
                mask_by_camera[camera_list[0]] = moved_object_mask_pixels
        audit = ObservationAuditEvidence(
            reference_displacement_m=reference_displacement_m,
            camera_ids=tuple(str(camera_id) for camera_id in camera_ids),
            moved_object_mask_pixels_by_camera=mask_by_camera,
        )
        result = evaluate_changed_observation_visibility(
            audit,
            displacement_threshold_m=self.displacement_threshold_m,
            policy_camera_ids=audit.camera_ids,
        )
        return result.qualified
