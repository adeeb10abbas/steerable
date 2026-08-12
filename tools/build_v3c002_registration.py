#!/usr/bin/env python3
"""Prepare the V3-C002 pre-registration package, without authorizing inference.

The protocol requires two independent human wording attestations *before* a
behavioral registration.  This command therefore writes a hash-bound draft and
an immutable queue draft.  ``registration_status`` is intentionally not
``registered_after_two_human_wording_agreements`` until the separate finalizer
has verified real independent reviewer records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (  # noqa: E402
    AMENDMENT_ID,
    ARENA,
    CELL_SCHEMA,
    LAYOUT_LEVEL,
    MODEL_ID,
    PROMPT_CONDITIONS,
    SEEDS,
    SUCCESS_PREDICATE_ID,
    STUDY_ID,
    canonical_json_sha256,
    file_binding,
    registered_prompts,
    sha256_file,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002.wording_gate import (  # noqa: E402
    SHEET_SCHEMA,
    build_blinded_sheet,
    pending_gate,
)


BASE_COMMIT = "18a2bf0200183647291cc7aeb1fe89997b3fb82f"
ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002/draft_v5"
E004_ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"
V1_ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002"
V2_ROOT = V1_ROOT / "draft_v2"
V3_ROOT = V1_ROOT / "draft_v3"
V4_ROOT = V1_ROOT / "draft_v4"

DEPENDENCY_PATHS = {
    "checkpoint": ("artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_checkpoint_manifest.json",),
    "runtime": (
        "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/gates/runtime_identity.json",
        "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/registration.json",
        "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/queue.jsonl",
    ),
    "policy_server": ("experiments/pi05_current_stack/v2a010_serve_policy.py",),
    "controller": (
        "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/droid_behavioral_bridge.py",
        "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/run_droid_queue.py",
    ),
    "action_interface": ("experiments/v3/phase_e/symmetric_layout_cohort_v3e004/droid_behavioral_contract.py",),
    "camera_configuration": (
        "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/live_snapshot_adapter.py",
        "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/layout/candidate.json",
    ),
    "horizon": ("artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/gates/runtime_identity.json",),
    "scorer": (
        "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/episode_compiler.py",
        "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/runtime_contract.py",
        "artifacts/vla_wam_shared_v3/failure_taxonomy.json",
    ),
    "raw_writer": ("experiments/v3/phase_e/symmetric_layout_cohort_v3e004/run_droid_queue.py",),
    "renderer": (
        "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/live_snapshot_adapter.py",
        "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/gates/runtime_identity.json",
    ),
}


def _relative_binding(path: Path) -> dict[str, Any]:
    binding = file_binding(path)
    try:
        binding["path"] = str(path.relative_to(REPO_ROOT))
    except ValueError:
        # Test or review packages may be prepared outside the repository; that
        # does not change the hash binding or authorize a behavioral run.
        binding["path"] = str(path.resolve())
    return binding


def _write_new(path: Path, value: Any) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing evidence/draft: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_new_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing queue draft: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _exact_runtime_contract(*, e004_cell: dict[str, Any], source_commit: str) -> dict[str, Any]:
    inherited_path = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/gates/runtime_identity.json"
    inherited = json.loads(inherited_path.read_text(encoding="utf-8"))
    dependencies = {
        group: [_relative_binding(REPO_ROOT / path) for path in paths]
        for group, paths in DEPENDENCY_PATHS.items()
    }
    values = {
        "model_id": MODEL_ID,
        "arena": ARENA,
        "checkpoint": e004_cell["runtime_identity_requirement"]["checkpoint"],
        "checkpoint_manifest_sha256": e004_cell["runtime_identity_requirement"]["checkpoint_manifest_sha256"],
        "checkpoint_digest": inherited["checkpoint_sha256"],
        "openpi_commit": e004_cell["runtime_identity_requirement"]["openpi_commit"],
        "robolab_commit": e004_cell["runtime_identity_requirement"]["robolab_commit"],
        "source_commit": source_commit,
        "action_dim": e004_cell["runtime_identity_requirement"]["action_dim"],
        "action_horizon": e004_cell["runtime_identity_requirement"]["action_horizon"],
        "action_cap": e004_cell["runtime_identity_requirement"]["action_cap"],
        "action_interface": {
            "action_dim": e004_cell["runtime_identity_requirement"]["action_dim"],
            "action_horizon": e004_cell["runtime_identity_requirement"]["action_horizon"],
            "action_cap": e004_cell["runtime_identity_requirement"]["action_cap"],
            "action_chunk_shape": inherited["action_chunk_shape"],
            "action_space": inherited["action_space"],
        },
        "policy_cameras": ["head_camera", "over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam"],
        "simulator_identity": inherited["simulator_version"],
        "renderer_backend": inherited["renderer_backend"],
        "runtime_identity_source_sha256": inherited["runtime_identity_sha256"],
        "policy_io_mode": inherited["future_interface"],
        "prompt_mode": inherited["instruction_controller"],
    }
    unsigned = {
        "schema_version": "vla-wam-shared-v3c002-exact-e004-pi05-runtime-contract-v1",
        "source_amendment": "V3-E004",
        "symmetry_level_s": 1.0,
        "identity_values": values,
        "dependency_bindings": dependencies,
        "component_digests": {group: canonical_json_sha256(records) for group, records in dependencies.items()},
    }
    return {**unsigned, "contract_sha256": canonical_json_sha256(unsigned)}


def _queue(candidate_sha256: str, runtime_contract: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = registered_prompts()
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        order = tuple(
            sorted(
                PROMPT_CONDITIONS,
                key=lambda condition: __import__("hashlib").sha256(
                    f"V3-C002|queue-order-v1|{seed}|{condition}".encode("utf-8")
                ).hexdigest(),
            )
        )
        for index, condition in enumerate(order):
            prompt = prompts[condition]
            rows.append(
                {
                    "schema_version": CELL_SCHEMA,
                    "study_id": STUDY_ID,
                    "amendment_id": AMENDMENT_ID,
                    "cell_id": f"v3c002:seed{seed}:{condition}",
                    "seed_block_id": f"v3c002:seed{seed}",
                    "episode_seed": seed,
                    "environment_seed": seed,
                    "sampling_seed": seed,
                    "model_id": MODEL_ID,
                    "arena": ARENA,
                    "execution_mode": "new_behavioral_episode",
                    "execution_order": list(order),
                    "execution_order_index": index,
                    "seed_block_indivisible": True,
                    "prompt_condition": condition,
                    **prompt,
                    "static_episode_prompt": True,
                    "prompt_goal_policy": "physical_goal is registered metadata; scorer must not parse prompt text",
                    "request_seed_formula": "episode_seed * 1000 + replan_index",
                    "layout_source_amendment": "V3-E004",
                    "symmetry_level_s": LAYOUT_LEVEL,
                    "layout_candidate_sha256": candidate_sha256,
                    "exact_runtime_contract_sha256": runtime_contract["contract_sha256"],
                    "success_predicate_id": SUCCESS_PREDICATE_ID,
                    "runtime_identity_requirement": runtime_contract["identity_values"],
                    "behavioral_failure_policy": "retain_in_denominator",
                    "infrastructure_failure_policy": "separate_stream_excluded_from_behavioral_denominator",
                    "missing_measurement_policy": "NR remains null and is never converted to zero",
                    "release_status": "pre_registration_draft_pending_two_human_wording_agreements",
                    "required_raw_outputs": [
                        "simulator_video",
                        "executed_action_trace",
                        "raw_episode_jsonl",
                        "final_state",
                        "state_trace",
                    ],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    candidate = E004_ROOT / "layout/candidate.json"
    e004_registration = E004_ROOT / "registration.json"
    e004_queue = E004_ROOT / "queue.jsonl"
    e004_results = E004_ROOT / "results/results.json"
    e004_cell = None
    for line in e004_queue.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("cell_id") == "v3e004:pi05:seed9400:s100:left":
            e004_cell = row
            break
    if e004_cell is None:
        raise SystemExit("cannot locate the E004 π0.5 s=1 runtime contract")
    candidate_sha = sha256_file(candidate)
    runtime_requirement = dict(e004_cell["runtime_identity_requirement"])
    source_commit = __import__("subprocess").run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    runtime_contract = _exact_runtime_contract(e004_cell=e004_cell, source_commit=source_commit)
    sheet_path = output_root / "prompt_comprehension_sheet.json"
    _write_new(sheet_path, build_blinded_sheet())
    response_template = {
        "schema_version": "vla-wam-shared-v3c002-human-prompt-attestation-v1",
        "instruction": "Copy outside the repository, obtain one independent authorized reader response, then supply the completed immutable record to the finalizer. This template contains no reader identity or decision and cannot pass the gate.",
        "sheet_sha256": sha256_file(sheet_path),
        "reader_id": None,
        "authorization_reference": None,
        "attested_at_utc": None,
        "signature_or_record_reference": None,
        "responses": [
            {"reader_pair_id": pair["reader_pair_id"], "decision": None}
            for pair in build_blinded_sheet()["pairs"]
        ],
    }
    _write_new(output_root / "human_reader_attestation_TEMPLATE.json", response_template)
    wording_gate_path = output_root / "wording_gate.json"
    _write_new(
        wording_gate_path,
        {
            **pending_gate(
            sheet_path=sheet_path,
            reason="Repository search found no existing authorized independent human-reader evidence. Two real signed/record-referenced attestations are required before behavioral registration.",
            ),
            "sheet": _relative_binding(sheet_path),
        },
    )
    queue_path = output_root / "queue.jsonl"
    rows = _queue(candidate_sha, runtime_contract)
    _write_new_jsonl(queue_path, rows)
    sources = {
        rel: _relative_binding(REPO_ROOT / rel)
        for rel in (
            "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/registration.json",
            "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/queue.jsonl",
            "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/layout/candidate.json",
            "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/results/results.json",
            "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/droid_behavioral_bridge.py",
            "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/droid_behavioral_contract.py",
            "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/episode_compiler.py",
            "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/runtime_contract.py",
            "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/run_droid_queue.py",
            "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/live_snapshot_adapter.py",
            "experiments/pi05_current_stack/v2a010_serve_policy.py",
            "artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_checkpoint_manifest.json",
            "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/gates/runtime_identity.json",
            "artifacts/vla_wam_shared_v3/failure_taxonomy.json",
            "experiments/v3/phase_c_semantic_equivalence_v3c002/__init__.py",
            "experiments/v3/phase_c_semantic_equivalence_v3c002/contract.py",
            "experiments/v3/phase_c_semantic_equivalence_v3c002/wording_gate.py",
            "experiments/v3/phase_c_semantic_equivalence_v3c002/runtime.py",
            "experiments/v3/phase_c_semantic_equivalence_v3c002/runner.py",
            "experiments/v3/phase_c_semantic_equivalence_v3c002/compiler.py",
            "tools/build_v3c002_registration.py",
            "tools/finalize_v3c002_registration.py",
            "tools/validate_v3c002.py",
            "tools/validate_v3c002_v1_historical.py",
            "tools/validate_v3c002_v2_historical.py",
            "tools/validate_v3c002_v3_historical.py",
            "tools/validate_v3c002_v4_historical.py",
            "tools/validate_v3c002_results.py",
            "tools/validate_v3e_publication_bundle.py",
            "tests/test_v3c002_semantic_equivalence.py",
        )
    }
    registration = {
        "schema_version": "vla-wam-shared-v3c002-registration-v4",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "title": "Semantically equivalent prompt control",
        "required_source_base_commit": BASE_COMMIT,
        "source_lineage": {
            "required_base_commit": BASE_COMMIT,
            "replacement_commit": source_commit,
            "replacement_reason": "Prospective C002 implementation and pre-run audit hardening were committed after the mandated clean base; no prior definition or experiment was changed.",
            "recorded_before_any_model_request": True,
            "model_requests_at_recording": 0,
            "behavioral_episodes_at_recording": 0,
        },
        "registration_status": "pre_registration_draft_pending_two_human_wording_agreements",
        "registered_at_utc": None,
        "model_request_count_before_registration": 0,
        "behavioral_episode_count_before_registration": 0,
        "model_requests_authorized": False,
        "behavioral_episodes_authorized": False,
        "pre_registration_wording_gate": _relative_binding(wording_gate_path),
        "wording_gate_requirement": "Two distinct authorized readers must independently mark both blinded pairs same_physical_endpoint before this draft can become a behavioral registration.",
        "design": {
            "seed_blocks": list(SEEDS),
            "seed_block_count": len(SEEDS),
            "conditions_per_seed_block": 4,
            "new_behavioral_episodes": len(rows),
            "block_integrity": "Each of the four prompt conditions stays on one serial execution lane and is never split across shards.",
            "execution_order": "SHA-256 ranked deterministic four-condition order, frozen in queue.jsonl.",
            "request_seed": "episode_seed * 1000 + replan_index",
            "no_reuse_of_e004_behavioral_outcomes": True,
        },
        "registered_prompts": registered_prompts(),
        "scoring": {
            "physical_goal_is_explicit_metadata": True,
            "prompt_parsing_prohibited": True,
            "frozen_success_predicate_id": SUCCESS_PREDICATE_ID,
            "requested_side_depth": "+signed final lateral offset for physical LEFT; negative signed final lateral offset for physical RIGHT",
            "sign_convention": "+Y = robot-left",
        },
        "e004_s1_layout": {
            "source_amendment": "V3-E004",
            "candidate": sources["artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/layout/candidate.json"],
            "candidate_sha256": candidate_sha,
            "symmetry_level_s": 1.0,
            "scope": "fully symmetric object layout only; robot, reset posture, camera rig, wrist mounting, and embodiment are not asserted bilaterally symmetric",
        },
        "exact_e004_pi05_runtime": runtime_contract,
        "analysis_plan": {
            "primary": {
                "estimand": "inverse-reference minus canonical requested-side depth, independently for each registered physical goal",
                "margin_m": 0.0415,
                "bootstrap": {"resamples": 20000, "confidence_level": 0.90, "cluster": "seed block"},
                "tost": "paired two one-sided test against [-0.0415,+0.0415] m",
                "claim_rule": "Both LEFT and RIGHT depth TOSTs and the inverse-reference endpoint positive control must pass; any failure withholds the model-level semantic equivalence claim.",
            },
            "secondary_binary": {"margin_probability": 0.1556, "interval": "paired seed-clustered 90% bootstrap", "claim_rule": "Both directional intervals must lie strictly inside the registered margin."},
            "positive_controls": {
                "canonical": "canonical physical LEFT minus canonical physical RIGHT endpoint redirection remains positive",
                "inverse_reference": "inverse physical LEFT minus inverse physical RIGHT endpoint redirection remains positive when scored by registered physical goal",
                "failure_rule": "If inverse-reference positive control fails, semantic redirection is unsupported; do not reinterpret the result as prompt equivalence.",
            },
            "descriptive": "Exact equality or inequality of action traces is descriptive only, not the primary grounding test.",
            "no_sample_extension_after_results": True,
        },
        "required_outputs": [
            "registration.json", "queue.jsonl", "release_gate.json", "infrastructure_attempts.jsonl",
            "results/episodes.jsonl", "results/pairs.jsonl", "results/results.json", "results/DECISION_MEMO.md",
            "results/evidence_manifest.json", "MANUSCRIPT_INSERT.md",
        ],
        "final_validation_requirement": "The committed final validator must reconstruct every count and estimand, rehash every retained raw artifact, verify infrastructure exclusion, and compare memo/manuscript claims to results.json.",
        "queue": _relative_binding(queue_path),
        "source_bindings": sources,
        "release_boundary": "No model request, action, or behavioral episode is authorized until the independent wording gate passes, this registration is activated with real attestations, the registration/queue/code are committed and pushed, and exact E004 runtime/lane/preflight gates pass.",
    }
    _write_new(output_root / "registration.json", registration)
    if V1_ROOT != output_root:
        superseded_bindings = {}
        for revision, prior_root in (("v1", V1_ROOT), ("v2", V2_ROOT), ("v3", V3_ROOT), ("v4", V4_ROOT)):
            revision_bindings = {}
            for name in ("registration.json", "queue.jsonl", "release_gate.json"):
                path = prior_root / name
                if path.is_file():
                    revision_bindings[name] = _relative_binding(path)
            if revision_bindings:
                superseded_bindings[revision] = revision_bindings
        _write_new(
            output_root / "supersession.json",
            {
                "schema_version": "vla-wam-shared-v3c002-preregistration-supersession-v4",
                "status": "prospective_v5_supersedes_unexecuted_v1_v2_v3_v4_drafts",
                "superseded_draft_bindings": superseded_bindings,
                "superseding_registration": _relative_binding(output_root / "registration.json"),
                "reason": "Independent pre-run audits required exact E004 scorer normalization/evaluation and checkout-portable committed evidence paths after V2 was already hash-bound.",
                "v1_model_requests": 0,
                "v1_behavioral_episodes": 0,
                "v2_model_requests": 0,
                "v2_behavioral_episodes": 0,
                "v3_model_requests": 0,
                "v3_behavioral_episodes": 0,
                "v4_model_requests": 0,
                "v4_behavioral_episodes": 0,
                "v1_v2_v3_v4_must_never_be_activated": True,
            },
        )
    release_gate = {
        "schema_version": "vla-wam-shared-v3c002-release-gate-v4",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "blocked_pre_registration_wording_gate",
        "passed": False,
        "registration": _relative_binding(output_root / "registration.json"),
        "queue": _relative_binding(queue_path),
        "wording_gate": _relative_binding(wording_gate_path),
        "required_bound_artifacts": ["source_push_gate", "physical_gate", "excluded_smoke_gate", "two_lane_isolation_gate", "lane_manifests"],
        "reason": "No authorized two-reader agreements are present. No runtime, lane, smoke, isolation, model request, or behavioral episode is authorized.",
    }
    _write_new(output_root / "release_gate.json", release_gate)
    _write_new_jsonl(output_root / "infrastructure_attempts.jsonl", [])
    print(json.dumps({"status": "prepared_fail_closed_before_behavioral_registration", "root": str(output_root), "queue_rows": len(rows), "queue_sha256": sha256_file(queue_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
