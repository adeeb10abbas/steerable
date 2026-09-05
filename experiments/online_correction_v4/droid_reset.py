"""Two-reset / one-physical-reset attestation for V4 DROID/RoboLab simulators."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from experiments.online_correction_v4.droid_contract import (
    RESET_SCHEMA,
    STUDY_ID,
    canonical_json_bytes,
    sha256_bytes,
)


SETTLE_STEPS = 60
STABILITY_WINDOW_STEPS = 15
LINEAR_SPEED_TOLERANCE_M_S = 0.02
ANGULAR_SPEED_TOLERANCE_RAD_S = 0.20


class ResetAttestationError(RuntimeError):
    """Raised when RoboLab reset discipline diverges from the frozen contract."""


@runtime_checkable
class PhysicalEnv(Protocol):
    def reset(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def step(self, action: Any) -> Any:
        ...


@runtime_checkable
class SettleProbe(Protocol):
    def hold_action(self) -> Any:
        ...

    def sample_stability(self) -> dict[str, Any]:
        ...

    def physical_reset_payload(self) -> dict[str, Any]:
        ...

    def zero_episode_length_buf(self) -> tuple[list[float], list[int]]:
        ...

    def on_settle_complete(self, post_settle_obs: Any) -> None:
        ...


@dataclass
class ResetAttestationState:
    episode_id: str
    env_seed: int
    fixture_id: str
    reset_registry_sha256: str
    locked_native_control_dt_s: float | None = None
    reset_registry_path: str | None = None
    runner_pre_action_reset_calls: int = 0
    physical_reset_calls: int = 0
    settle_gate_runs: int = 0
    duplicate_second_reset_idempotent: bool = False
    settle_evidence: dict[str, Any] | None = None
    cached_reset_result: Any = None
    attestation_written: bool = False
    attestation_body: dict[str, Any] = field(default_factory=dict)

    def validate_reset_state(self) -> None:
        if self.runner_pre_action_reset_calls != 2:
            raise ResetAttestationError(
                "frozen RoboLab runner must perform exactly two pre-action reset calls"
            )
        if self.physical_reset_calls != 1 or self.settle_gate_runs != 1:
            raise ResetAttestationError(
                "duplicate runner reset must map to one physical reset and one settle gate"
            )
        if self.settle_evidence is None:
            raise ResetAttestationError("settle/stability evidence is missing")

    def validate_contract(self) -> dict[str, Any]:
        self.validate_reset_state()
        if self.attestation_body.get("schema_version") != RESET_SCHEMA:
            raise ResetAttestationError("reset attestation schema mismatch")
        return dict(self.attestation_body)


@dataclass
class TwoResetAttestationProxy:
    """Wrap a physical env to enforce the Phase-B reset discipline for V4."""

    env: PhysicalEnv
    probe: SettleProbe
    state: ResetAttestationState
    attestation_writer: Callable[[dict[str, Any]], None] | None = None
    attestation_validator: Callable[[dict[str, Any], Mapping[str, Any]], None] | None = None
    model_request_guard: Callable[[], None] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        if self.state.attestation_written:
            raise ResetAttestationError("RoboLab attempted reset after attestation was finalized")
        self.state.runner_pre_action_reset_calls += 1
        if self.state.runner_pre_action_reset_calls > 2:
            raise ResetAttestationError("frozen RoboLab runner performed more than two pre-action reset calls")
        if self.state.runner_pre_action_reset_calls == 2:
            if (
                self.state.cached_reset_result is None
                or self.state.physical_reset_calls != 1
                or self.state.settle_gate_runs != 1
                or self.state.settle_evidence is None
            ):
                raise ResetAttestationError(
                    "duplicate runner reset occurred before the physical reset gate completed"
                )
            self.state.duplicate_second_reset_idempotent = True
            if self.state.settle_evidence is not None:
                self.state.settle_evidence["duplicate_second_reset_idempotent"] = True
                self.state.settle_evidence["runner_pre_action_reset_calls"] = 2
            return self.state.cached_reset_result

        self.state.physical_reset_calls += 1
        result = self.env.reset(*args, **kwargs)
        hold = self.probe.hold_action()
        self.state.settle_gate_runs += 1
        post_settle_obs: Any = None
        for _ in range(SETTLE_STEPS):
            step = self.env.step(hold)
            post_settle_obs = _observation_from_step(step)
            if _terminated_or_truncated(step):
                raise ResetAttestationError("environment terminated during model-blind settling")
        maxima = self.probe.sample_stability()
        for _ in range(STABILITY_WINDOW_STEPS):
            step = self.env.step(hold)
            post_settle_obs = _observation_from_step(step)
            if _terminated_or_truncated(step):
                raise ResetAttestationError("environment terminated during stability window")
            window = self.probe.sample_stability()
            for name, row in window.items():
                if name not in maxima:
                    maxima[name] = row
                    continue
                maxima[name]["max_linear_component_speed_m_s"] = max(
                    maxima[name]["max_linear_component_speed_m_s"],
                    row["max_linear_component_speed_m_s"],
                )
                maxima[name]["max_angular_component_speed_rad_s"] = max(
                    maxima[name]["max_angular_component_speed_rad_s"],
                    row["max_angular_component_speed_rad_s"],
                )
        before, after = self.probe.zero_episode_length_buf()
        physical = self.probe.physical_reset_payload()
        if post_settle_obs is not None:
            self.probe.on_settle_complete(post_settle_obs)
        evidence = {
            "schema_version": "v4-droid-settle-stability-v1",
            "study_id": STUDY_ID,
            "registered_episode_id": self.state.episode_id,
            "settle_steps": SETTLE_STEPS,
            "stable_window_steps": STABILITY_WINDOW_STEPS,
            "linear_speed_tolerance_m_s": LINEAR_SPEED_TOLERANCE_M_S,
            "angular_speed_tolerance_rad_s": ANGULAR_SPEED_TOLERANCE_RAD_S,
            "stability_window_component_maxima": maxima,
            "episode_length_buf_before_reset": before,
            "episode_length_buf_after_reset": after,
            "episode_length_buf_reset_passed": after == [0],
            "model_request_count_during_gate": 0,
            "runner_pre_action_reset_calls": self.state.runner_pre_action_reset_calls,
            "physical_reset_calls": self.state.physical_reset_calls,
            "settle_gate_runs": self.state.settle_gate_runs,
            "duplicate_second_reset_idempotent": False,
        }
        self._validate_settle_evidence(evidence)
        self.state.settle_evidence = evidence
        post_info = result[1] if isinstance(result, tuple) and len(result) == 2 else {}
        if post_settle_obs is None:
            post_settle_obs = result[0] if isinstance(result, tuple) else result
        self.state.cached_reset_result = (post_settle_obs, post_info)
        return self.state.cached_reset_result

    def step(self, action: Any) -> Any:
        if not self.state.attestation_written:
            raise ResetAttestationError("env.step occurred before reset attestation was written")
        if self.model_request_guard is not None:
            self.model_request_guard()
        return self.env.step(action)

    def finalize_attestation(
        self,
        *,
        prompt_sha256: str,
        runtime_identity_sha256: str,
        initial_state_sha256: str,
    ) -> dict[str, Any]:
        self.state.validate_reset_state()
        attestation = {
            "schema_version": RESET_SCHEMA,
            "study_id": STUDY_ID,
            "registered_episode_id": self.state.episode_id,
            "environment_seed": self.state.env_seed,
            "fixture_id": self.state.fixture_id,
            "reset_registry_sha256": self.state.reset_registry_sha256,
            "prompt_sha256": prompt_sha256,
            "runtime_identity_sha256": runtime_identity_sha256,
            "model_request_count_before_attestation": 0,
            "runner_pre_action_reset_calls": self.state.runner_pre_action_reset_calls,
            "physical_reset_calls": self.state.physical_reset_calls,
            "settle_gate_runs": self.state.settle_gate_runs,
            "duplicate_second_reset_idempotent": self.state.duplicate_second_reset_idempotent,
            "settle_steps": SETTLE_STEPS,
            "stable_window_steps": STABILITY_WINDOW_STEPS,
            "physical_reset_sha256": sha256_bytes(
                canonical_json_bytes(self.probe.physical_reset_payload())
            ),
            "initial_state_sha256": initial_state_sha256,
            "settle_stability_evidence_sha256": sha256_bytes(
                canonical_json_bytes(self.state.settle_evidence or {})
            ),
        }
        if self.state.locked_native_control_dt_s is not None:
            attestation["locked_native_control_dt_s"] = self.state.locked_native_control_dt_s
        if self.attestation_validator is not None:
            self.attestation_validator(attestation, self.probe.physical_reset_payload())
        body = {key: value for key, value in attestation.items()}
        attestation["reset_fingerprint_sha256"] = sha256_bytes(canonical_json_bytes(body))
        self.state.attestation_body = attestation
        self.state.attestation_written = True
        if self.attestation_writer is not None:
            self.attestation_writer(attestation)
        return attestation

    @staticmethod
    def _validate_settle_evidence(evidence: Mapping[str, Any]) -> None:
        maxima = evidence.get("stability_window_component_maxima")
        if not isinstance(maxima, dict) or not maxima:
            raise ResetAttestationError("settle/stability evidence lacks component maxima")
        for name, row in maxima.items():
            linear = row.get("max_linear_component_speed_m_s")
            angular = row.get("max_angular_component_speed_rad_s")
            if linear is None or angular is None:
                raise ResetAttestationError(f"{name} stability maxima are incomplete")
            if linear > LINEAR_SPEED_TOLERANCE_M_S or angular > ANGULAR_SPEED_TOLERANCE_RAD_S:
                raise ResetAttestationError(f"{name} exceeded released stability thresholds")


def _terminated_or_truncated(step_result: Any) -> bool:
    if isinstance(step_result, tuple):
        if len(step_result) >= 4:
            terminated = step_result[2]
            truncated = step_result[3]
            if hasattr(terminated, "__getitem__"):
                return bool(terminated[0]) or bool(truncated[0])
            return bool(terminated) or bool(truncated)
    return False


def _observation_from_step(step_result: Any) -> Any:
    if isinstance(step_result, tuple) and step_result:
        return step_result[0]
    return step_result


def validate_reset_attestation_payload(payload: Mapping[str, Any], *, episode_id: str) -> dict[str, Any]:
    if payload.get("registered_episode_id") != episode_id:
        raise ResetAttestationError("reset attestation episode mismatch")
    if payload.get("schema_version") != RESET_SCHEMA:
        raise ResetAttestationError("reset attestation schema mismatch")
    body = {key: value for key, value in payload.items() if key != "reset_fingerprint_sha256"}
    if payload.get("reset_fingerprint_sha256") != sha256_bytes(canonical_json_bytes(body)):
        raise ResetAttestationError("reset attestation fingerprint mismatch")
    if payload.get("runner_pre_action_reset_calls") != 2:
        raise ResetAttestationError("reset attestation lacks two runner reset calls")
    if payload.get("physical_reset_calls") != 1:
        raise ResetAttestationError("reset attestation lacks one physical reset")
    return dict(payload)


def validate_reset_attestation_file(path: str | Any, *, episode_id: str) -> dict[str, Any]:
    payload = json.loads(Path(str(path)).read_text(encoding="utf-8"))
    return validate_reset_attestation_payload(payload, episode_id=episode_id)
