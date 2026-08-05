#!/usr/bin/env python3
"""Audit or compile the three frozen V3-A002 pi0-FAST RTX shards.

Raw shard, simulator, action, state, and video files are read-only inputs.
``audit`` inventories terminal pairs and reports active pairs without writing.
``compile`` first requires all three shard ledgers to be terminal, validates
the original launch-time adapter/runtime contract, and then applies only the
frozen V3-A003 post-result token-trace correction.  Technical or partial pair
attempts receive separate infrastructure records and can never enter the
behavioral denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import adapter as frozen_adapter  # noqa: E402
from adapter import (  # noqa: E402
    ACTION_SHAPE,
    ACTION_TRACE_SCHEMA,
    INFRA_CAPTURE_SCHEMA,
    MODEL_ID,
    build_infrastructure_record,
    build_behavioral_record,
    load_authorized_pair,
    preflight,
    sha256_file,
)


STUDY_ID = "vla_wam_language_steerability_v3"
SHARD_LEDGER_SCHEMA = (
    "vla-wam-shared-v3-pi0-fast-old-name-config-shard-ledger-v1"
)
AUDIT_SCHEMA = "vla-wam-shared-v3-pi0-fast-old-name-config-shard-audit-v2"
SUMMARY_SCHEMA = "vla-wam-shared-v3-pi0-fast-old-name-config-summary-v3"
HASH_MANIFEST_SCHEMA = (
    "vla-wam-shared-v3-pi0-fast-old-name-config-hash-manifest-v3"
)
INFRASTRUCTURE_LEDGER_SCHEMA = (
    "vla-wam-shared-v3-pi0-fast-old-name-config-infrastructure-ledger-v1"
)
PAIR_MANIFEST_SCHEMA = (
    "vla-wam-shared-v3-pi0-fast-old-name-config-pair-manifest-v3"
)
COMPILATION_AMENDMENT_ID = "V3-A003"
COMPILATION_AMENDMENT_SCHEMA = (
    "vla-wam-shared-v3-post-result-pi0-fast-token-trace-validation-"
    "amendment-v1"
)
COMPILATION_AMENDMENT_RELATIVE = Path(
    "artifacts/vla_wam_shared_v3/"
    "post_result_pi0_fast_token_trace_validation_amendment.json"
)
COMPILATION_AMENDMENT_SHA256 = (
    "8224173b0070244384b267251d256a0f37dd3682cf09086547bf13ddecdaea6d"
)
ORIGINAL_ADAPTER_SHA256 = (
    "f5fcb2b37c5e7582ec6d70ee6a7954e28edd3aa8dd3d7e7ae1bae66bd9b6eec6"
)
ORIGINAL_ADAPTER_CONTRACT_SHA256 = (
    "0c7937482824090e3033fa3f6822c8277aff7b0b7f1403b565bc13140b5461db"
)
FROZEN_TOKENIZER_SOURCE_SHA256 = (
    "a1b94e9e72849a18834778f229c6bb389a495eb7fbe0aa800edea728b9424ff4"
)
COMPILER_RELATIVE = Path(
    "experiments/v3/pi0_fast_old_name_config_bridge/compile_shards.py"
)
RELEASE_GATE_RELATIVE = Path("release_gate_attempt01/release_gate.json")
TERMINAL_SHARD_STATUSES = {
    "all_guard_launches_exit_zero_pending_compilation",
    "completed_with_technical_failures",
}
BEHAVIORAL_PAIR_STATUS = (
    "guard_exit_zero_pending_fail_closed_pair_compilation"
)
TECHNICAL_PAIR_STATUSES = {
    "technical_failure_guard_exit_nonzero",
    "technical_failure_launcher_exception",
}
LEFT_PROMPT = "Put the Rubik's cube to the left of the bowl."
RIGHT_PROMPT = "Put the Rubik's cube to the right of the bowl."
HEX64 = set("0123456789abcdef")
FIXED_OBSERVATION_TOKEN_SHA256 = {
    "left": "d282373e2576a5e7d74efeecd12c7cf5530160ea78fb54ab8166edc4975a659a",
    "right": "9ab3cb0ffc19e5a9e80bfd1822ef8d5c2f25446930e21743e020b6e057935cd9",
}


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    pod: str
    seed_start: int
    seed_end: int
    runtime_identity_name: str

    @property
    def seeds(self) -> list[int]:
        return list(range(self.seed_start, self.seed_end + 1))


SHARDS = (
    ShardSpec(
        shard_id="raytrace_rtxpro6000_ali_seed8310_8316_attempt01",
        pod="raytrace-rtxpro6000-ali",
        seed_start=8310,
        seed_end=8316,
        runtime_identity_name="runtime_identity_raytrace_rtxpro6000_ali.json",
    ),
    ShardSpec(
        shard_id="vla_wam_rtx_cosmos_ali_seed8317_8323_attempt01",
        pod="vla-wam-rtx-cosmos-ali",
        seed_start=8317,
        seed_end=8323,
        runtime_identity_name="runtime_identity_vla_wam_rtx_cosmos_ali.json",
    ),
    ShardSpec(
        shard_id="vla_wam_rtx_nano_ali_seed8324_8329_attempt01",
        pod="vla-wam-rtx-nano-ali",
        seed_start=8324,
        seed_end=8329,
        runtime_identity_name="runtime_identity_vla_wam_rtx_nano_ali.json",
    ),
)


class CompileError(ValueError):
    """The frozen shard inventory is incomplete, ambiguous, or inconsistent."""


def _fail(message: str) -> None:
    raise CompileError(message)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CompileError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        _fail(f"{path} must contain one JSON object")
    return value


def _load_single_jsonl(path: Path) -> dict[str, Any]:
    try:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
    except OSError as error:
        raise CompileError(f"cannot read JSONL {path}: {error}") from error
    if len(lines) != 1:
        _fail(f"expected exactly one record in {path}, found {len(lines)}")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise CompileError(f"cannot decode JSONL {path}: {error}") from error
    if not isinstance(value, dict):
        _fail(f"{path} record must be an object")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        _fail(f"required nonempty artifact is absent: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _file_record_as(actual_path: Path, advertised_path: Path) -> dict[str, Any]:
    record = _file_record(actual_path)
    record["path"] = str(advertised_path.resolve())
    return record


def _rehash_claimed_file(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(record.get("path", ""))).resolve()
    observed = _file_record(path)
    claimed = {
        "path": str(path),
        "sha256": record.get("sha256"),
        "bytes": record.get("bytes"),
    }
    if observed != claimed:
        _fail(f"stale hash claim at final closure: {path}")
    return observed


def _corrected_action_trace_validator(
    path: Path,
    cell: dict[str, Any],
    actions_executed: int,
) -> tuple[dict[str, Any], Path, Path]:
    """Validate V3-A002 traces without conflating prompt and observation tokens.

    The frozen server hashes the complete output of ``_input_transform``: token
    ids plus mask after the current observation has been incorporated.  Those
    bytes may therefore change between requests even though the raw episode
    prompt is static.  The original adapter incorrectly required the hashes to
    be identical within an episode.  This post-result validator preserves every
    other frozen check, requires the raw prompt and its SHA-256 to be constant,
    and requires every per-request transformed-token attestation to be a valid
    lowercase SHA-256.  It does not reinterpret or mutate the retained trace.
    """

    import numpy as np

    trace = _load_object(path)
    prompt = cell["prompt"]
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    expected = {
        "schema_version": ACTION_TRACE_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "bridge_id": "v3a002",
        "environment_seed": cell["environment_seed"],
        "sampling_seed_base": cell["sampling_seed"],
        "prompt": prompt,
        "prompt_sha256": prompt_sha,
        "requested_relation": cell["relation"],
        "prompt_controller": "episode_static",
        "open_loop_execution_horizon": 10,
    }
    for key, wanted in expected.items():
        if trace.get(key) != wanted:
            _fail(f"corrected action-trace validation mismatch for {key}")

    actions_entry = trace.get("executed_actions", {})
    actions_path = Path(str(actions_entry.get("path", ""))).resolve()
    if (
        not actions_path.is_file()
        or actions_entry.get("sha256") != sha256_file(actions_path)
        or actions_entry.get("bytes") != actions_path.stat().st_size
        or actions_entry.get("count") != actions_executed
        or actions_entry.get("shape") != [actions_executed, 8]
        or actions_entry.get("dtype") != "float32"
    ):
        _fail("corrected validator found invalid executed-action metadata")
    actions = np.load(actions_path, allow_pickle=False)
    if (
        actions.shape != (actions_executed, 8)
        or actions.dtype != np.float32
        or not np.isfinite(actions).all()
    ):
        _fail("corrected validator found an invalid executed-action tensor")

    request_seeds = trace.get("request_sampling_seeds")
    if (
        not isinstance(request_seeds, list)
        or not request_seeds
        or request_seeds
        != [
            cell["sampling_seed"] * 1000 + index
            for index in range(len(request_seeds))
        ]
    ):
        _fail("corrected validator found an invalid request-seed sequence")
    expected_requests = math.ceil(actions_executed / 10)
    if len(request_seeds) != expected_requests:
        _fail("corrected validator found an invalid ten-action request count")

    chunks_entry = trace.get("returned_action_chunks", {})
    chunks_path = Path(str(chunks_entry.get("path", ""))).resolve()
    if (
        not chunks_path.is_file()
        or chunks_entry.get("sha256") != sha256_file(chunks_path)
        or chunks_entry.get("bytes") != chunks_path.stat().st_size
        or chunks_entry.get("count") != len(request_seeds)
        or chunks_entry.get("shape") != [len(request_seeds), 10, 8]
        or chunks_entry.get("dtype") != "float32"
    ):
        _fail("corrected validator found invalid returned-chunk metadata")
    chunks = np.load(chunks_path, allow_pickle=False)
    if (
        chunks.shape != (len(request_seeds), *ACTION_SHAPE)
        or chunks.dtype != np.float32
        or not np.isfinite(chunks).all()
    ):
        _fail("corrected validator found an invalid returned-chunk tensor")

    attestations = trace.get("request_attestations")
    if not isinstance(attestations, list) or len(attestations) != len(request_seeds):
        _fail("corrected validator found incomplete request attestations")
    for index, attestation in enumerate(attestations):
        if not isinstance(attestation, dict):
            _fail(f"request attestation {index} is not an object")
        expected_attestation = {
            "request_index": index,
            "sampling_seed": request_seeds[index],
            "prompt_sha256": prompt_sha,
            "action_chunk_payload_sha256": hashlib.sha256(
                chunks[index].tobytes(order="C")
            ).hexdigest(),
        }
        for key, wanted in expected_attestation.items():
            if attestation.get(key) != wanted:
                _fail(
                    "corrected request attestation mismatch for "
                    f"{index}/{key}"
                )
        if not _is_sha256(attestation.get("tokenized_prompt_sha256")):
            _fail(
                f"request attestation {index} lacks a lowercase token SHA-256"
            )
    return trace, actions_path, chunks_path


def _token_trace_integrity_metadata(
    path: Path,
    cell: dict[str, Any],
    actions_executed: int,
) -> dict[str, Any]:
    trace, _, _ = _corrected_action_trace_validator(
        path,
        cell,
        actions_executed,
    )
    ordered = [
        row["tokenized_prompt_sha256"]
        for row in trace["request_attestations"]
    ]
    relation = cell["relation"]
    expected_first = FIXED_OBSERVATION_TOKEN_SHA256[relation]
    if ordered[0] != expected_first:
        _fail(
            f"{relation} first-request token hash differs from the validated "
            "fixed-observation release gate"
        )
    return {
        "schema_version": (
            "vla-wam-shared-v3-pi0-fast-token-trace-integrity-v1"
        ),
        "raw_prompt_sha256": trace["prompt_sha256"],
        "request_count": len(ordered),
        "first_request_matches_fixed_observation_gate": True,
        "fixed_observation_gate_token_sha256": expected_first,
        "ordered_tokenized_input_sha256": ordered,
        "token_hash_cardinality": len(set(ordered)),
        "cardinality_interpretation": (
            "Integrity metadata for observation-conditioned FAST input tokens; "
            "cardinality is not evidence of prompt switching."
        ),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl_staged(
    *,
    actual_path: Path,
    advertised_path: Path,
    records: list[dict[str, Any]],
    study_root: Path,
) -> dict[str, Any]:
    """Write staged JSONL whose embedded path already names the final cohort."""

    sys.path.insert(0, str(study_root / "tools"))
    from vla_wam_v3_episode_schema import (  # type: ignore
        validate_raw_episode_record,
    )

    normalized = [validate_raw_episode_record(record) for record in records]
    if not normalized:
        _fail("a staged JSONL batch must contain at least one record")
    actual_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = actual_path.with_name(actual_path.name + ".manifest.json")
    if actual_path.exists() or manifest_path.exists():
        _fail(f"refusing to overwrite staged JSONL evidence: {actual_path}")
    with actual_path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in normalized:
            handle.write(
                json.dumps(
                    record,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    study_ids = {record["study_id"] for record in normalized}
    schemas = {record["schema_version"] for record in normalized}
    if len(study_ids) != 1:
        _fail("a staged JSONL batch must contain exactly one study_id")
    manifest = {
        "schema_version": "vla-wam-shared-v3-jsonl-batch-manifest-v1",
        "study_id": next(iter(study_ids)),
        "jsonl_path": str(advertised_path.resolve()),
        "jsonl_sha256": sha256_file(actual_path),
        "jsonl_bytes": actual_path.stat().st_size,
        "row_count": len(normalized),
        "record_schema_versions": sorted(schemas),
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _require_exact_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        _fail(f"{label} is missing: {resolved}")
    return resolved


def _runtime_for_spec(
    *, study_root: Path, raw_root: Path, spec: ShardSpec, release_gate: Path
) -> tuple[Path, dict[str, Any]]:
    runtime_path = _require_exact_file(
        raw_root / spec.runtime_identity_name,
        f"{spec.pod} runtime identity",
    )
    authorization = preflight(
        study_root,
        spec.seed_start,
        runtime_path,
        release_gate,
    )
    runtime = authorization["runtime_identity"]
    target = runtime.get("target_kubernetes", {})
    simulator = target.get("simulator", {}) if isinstance(target, dict) else {}
    if simulator.get("pod") != spec.pod or simulator.get("gpu_index") != 0:
        _fail(f"runtime identity is not bound to {spec.pod} GPU 0")
    adapter_sources = runtime.get("adapter_source_sha256", {})
    if (
        runtime.get("adapter_contract_hash")
        != ORIGINAL_ADAPTER_CONTRACT_SHA256
        or not isinstance(adapter_sources, dict)
        or adapter_sources.get(
            "experiments/v3/pi0_fast_old_name_config_bridge/adapter.py"
        )
        != ORIGINAL_ADAPTER_SHA256
    ):
        _fail(f"{spec.pod} does not retain the original launch-time adapter")
    return runtime_path, runtime


def _exact_behavioral_paths(
    *,
    pair_dir: Path,
    robolab_root: Path,
    seed: int,
) -> dict[str, Path]:
    state_dir = pair_dir / "state_capture"
    action_dir = pair_dir / "action_trace"
    native_root = (
        robolab_root
        / "output"
        / f"v3a002_pi0_fast_old_name_config_bridge_seed{seed}_both_attempt01"
    )
    paths = {
        "left_capture": state_dir / f"seed{seed}_left.json",
        "right_capture": state_dir / f"seed{seed}_right.json",
        "left_state_jsonl": state_dir / f"seed{seed}_left_states.partial.jsonl",
        "right_state_jsonl": state_dir / f"seed{seed}_right_states.partial.jsonl",
        "left_action_trace": (
            action_dir / f"seed{seed}_direct_command_left_action_trace.json"
        ),
        "right_action_trace": (
            action_dir / f"seed{seed}_direct_command_right_action_trace.json"
        ),
        "left_video": (
            native_root
            / "RubiksCubeLeftOfBowlMatchedTask"
            / "Put_the_Rubiks_cube_to_the_left_of_the_bowl_0_viewport.mp4"
        ),
        "right_video": (
            native_root
            / "RubiksCubeRightOfBowlMatchedTask"
            / "Put_the_Rubiks_cube_to_the_right_of_the_bowl_0_viewport.mp4"
        ),
        "native_output_dir": native_root,
    }
    return paths


def _assert_unambiguous_behavioral_layout(
    *,
    study_root: Path,
    pair_dir: Path,
    paths: dict[str, Path],
    seed: int,
    pod: str,
) -> dict[str, Any]:
    required_files = [
        path for name, path in paths.items() if name != "native_output_dir"
    ]
    for path in required_files:
        _file_record(path)

    state_dir = pair_dir / "state_capture"
    expected_captures = {
        paths["left_capture"].resolve(),
        paths["right_capture"].resolve(),
    }
    observed_captures = {
        path.resolve() for path in state_dir.glob(f"seed{seed}_*.json")
    }
    if observed_captures != expected_captures:
        _fail(
            f"seed {seed} has ambiguous final state captures: "
            f"{sorted(map(str, observed_captures))}"
        )

    action_dir = pair_dir / "action_trace"
    expected_manifests = {
        paths["left_action_trace"].resolve(),
        paths["right_action_trace"].resolve(),
    }
    observed_manifests = {
        path.resolve() for path in action_dir.glob(f"seed{seed}_*_action_trace.json")
    }
    if observed_manifests != expected_manifests:
        _fail(
            f"seed {seed} has ambiguous action-trace manifests: "
            f"{sorted(map(str, observed_manifests))}"
        )

    native_root = paths["native_output_dir"]
    videos = {path.resolve() for path in native_root.rglob("*_viewport.mp4")}
    expected_videos = {
        paths["left_video"].resolve(),
        paths["right_video"].resolve(),
    }
    if videos != expected_videos:
        _fail(
            f"seed {seed} has ambiguous viewport videos: {sorted(map(str, videos))}"
        )

    cells = {row["relation"]: row for row in load_authorized_pair(study_root, seed)}
    token_integrity: dict[str, Any] = {}
    for relation, prompt in (("left", LEFT_PROMPT), ("right", RIGHT_PROMPT)):
        capture = _load_object(paths[f"{relation}_capture"])
        if (
            capture.get("registered_cell_id")
            != f"v3:droid:{MODEL_ID}:seed{seed}:{relation}"
            or capture.get("requested_relation") != relation
            or capture.get("prompt") != prompt
            or capture.get("environment_seed") != seed
            or capture.get("policy_seed") != seed
            or capture.get("simulator_pod") != pod
            or capture.get("behavioral_result_valid_candidate") is not True
        ):
            _fail(f"seed {seed}/{relation} final capture is not a valid candidate")
        contract = capture.get("capture_contract", {})
        if (
            not isinstance(contract, dict)
            or Path(str(contract.get("partial_state_stream", ""))).resolve()
            != paths[f"{relation}_state_jsonl"].resolve()
        ):
            _fail(f"seed {seed}/{relation} state stream path is ambiguous")

        trace = _load_object(paths[f"{relation}_action_trace"])
        if (
            trace.get("environment_seed") != seed
            or trace.get("sampling_seed_base") != seed
            or trace.get("requested_relation") != relation
            or trace.get("prompt") != prompt
        ):
            _fail(f"seed {seed}/{relation} action trace identity is ambiguous")
        token_integrity[relation] = _token_trace_integrity_metadata(
            paths[f"{relation}_action_trace"],
            cells[relation],
            int(capture["actions_executed"]),
        )
    return token_integrity


def _technical_partial_material_by_relation(
    *, pair_dir: Path, robolab_root: Path, seed: int
) -> dict[str, bool]:
    paths = _exact_behavioral_paths(
        pair_dir=pair_dir,
        robolab_root=robolab_root,
        seed=seed,
    )
    result: dict[str, bool] = {}
    for relation in ("left", "right"):
        exact_candidates = (
            paths[f"{relation}_capture"],
            paths[f"{relation}_state_jsonl"],
            paths[f"{relation}_action_trace"],
            paths[f"{relation}_video"],
        )
        action_matches = list(
            (pair_dir / "action_trace").glob(
                f"seed{seed}_*_{relation}_*"
            )
        )
        native_relation_dir = paths[f"{relation}_video"].parent
        result[relation] = any(path.is_file() for path in exact_candidates) or any(
            path.is_file() for path in action_matches
        ) or (
            native_relation_dir.is_dir()
            and any(path.is_file() for path in native_relation_dir.rglob("*"))
        )
    return result


def _inspect_shard(
    *,
    study_root: Path,
    raw_root: Path,
    spec: ShardSpec,
    release_gate: Path,
) -> dict[str, Any]:
    shard_dir = raw_root / "shards" / spec.shard_id
    if not shard_dir.is_dir():
        return {
            "shard_id": spec.shard_id,
            "pod": spec.pod,
            "status": "missing",
            "pairs": [],
            "pending_seeds": spec.seeds,
        }
    if shard_dir.is_symlink():
        _fail(f"frozen shard directory may not be a symlink: {shard_dir}")

    ledger_path = _require_exact_file(shard_dir / "shard_ledger.json", "shard ledger")
    ledger = _load_object(ledger_path)
    expected = {
        "schema_version": SHARD_LEDGER_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "shard_id": spec.shard_id,
        "simulator_pod_tag": spec.pod,
        "seed_range_inclusive": [spec.seed_start, spec.seed_end],
        "seeds": spec.seeds,
        "attempt": 1,
        "gpu_index": 0,
    }
    for key, wanted in expected.items():
        if ledger.get(key) != wanted:
            _fail(f"{spec.shard_id} ledger mismatch for {key}")
    pair_contract = ledger.get("matched_pair_contract", {})
    if (
        pair_contract.get("relations") != ["left", "right"]
        or pair_contract.get("condition") != "both"
        or pair_contract.get("left_right_may_be_split") is not False
        or pair_contract.get("environment_and_sampling_seed_equal") is not True
    ):
        _fail(f"{spec.shard_id} no longer preserves the matched-pair contract")

    runtime_path, runtime = _runtime_for_spec(
        study_root=study_root,
        raw_root=raw_root,
        spec=spec,
        release_gate=release_gate,
    )
    if (
        Path(str(ledger.get("runtime_identity", {}).get("path", ""))).resolve()
        != runtime_path
        or ledger.get("runtime_identity", {}).get("sha256")
        != sha256_file(runtime_path)
        or Path(str(ledger.get("release_gate", {}).get("path", ""))).resolve()
        != release_gate
        or ledger.get("release_gate", {}).get("sha256")
        != sha256_file(release_gate)
    ):
        _fail(f"{spec.shard_id} ledger runtime/gate identity changed")

    raw_pairs = ledger.get("pairs")
    if not isinstance(raw_pairs, list):
        _fail(f"{spec.shard_id} ledger pairs must be a list")
    observed_seeds = [row.get("seed") for row in raw_pairs if isinstance(row, dict)]
    if len(observed_seeds) != len(raw_pairs) or observed_seeds != spec.seeds[:len(raw_pairs)]:
        _fail(f"{spec.shard_id} pair rows are duplicated, reordered, or out of range")

    shard_status = ledger.get("status")
    if shard_status not in TERMINAL_SHARD_STATUSES | {"running"}:
        _fail(f"{spec.shard_id} has unknown shard status {shard_status!r}")
    if shard_status in TERMINAL_SHARD_STATUSES and observed_seeds != spec.seeds:
        _fail(f"terminal shard {spec.shard_id} is missing registered pairs")

    robolab_root = Path(str(runtime.get("robolab_dir", ""))).resolve()
    tokenizer_source = _file_record(
        Path(str(runtime.get("openpi_dir", ""))).resolve()
        / "src/openpi/transforms.py"
    )
    if tokenizer_source["sha256"] != FROZEN_TOKENIZER_SOURCE_SHA256:
        _fail(f"{spec.shard_id} frozen TokenizeFASTInputs source changed")
    pair_inventory: list[dict[str, Any]] = []
    completed_count = 0
    technical_count = 0
    launching_count = 0
    launching_positions = [
        index
        for index, row in enumerate(raw_pairs)
        if row.get("status") == "launching"
    ]
    if len(launching_positions) > 1 or (
        launching_positions and launching_positions[0] != len(raw_pairs) - 1
    ):
        _fail(f"{spec.shard_id} has ambiguous concurrent pair rows")
    for row in raw_pairs:
        seed = int(row["seed"])
        expected_pair_dir = shard_dir / f"seed{seed}_attempt01"
        pair_dir = Path(str(row.get("pair_dir", ""))).resolve()
        if pair_dir != expected_pair_dir.resolve() or not pair_dir.is_dir():
            _fail(f"seed {seed} pair directory is missing or redirected")
        if row.get("pair_id") != f"v3:droid:{MODEL_ID}:seed{seed}":
            _fail(f"seed {seed} pair identity changed")
        status = row.get("status")
        base = {
            "seed": seed,
            "pair_id": row["pair_id"],
            "pair_dir": str(pair_dir),
            "status": status,
            "runtime_identity": str(runtime_path),
            "pod": spec.pod,
        }
        if status == "launching":
            if row.get("guard_exit_code") is not None:
                _fail(f"active seed {seed} already claims a guard exit code")
            launching_count += 1
            pair_inventory.append({**base, "disposition": "in_progress"})
            continue
        if status == BEHAVIORAL_PAIR_STATUS:
            if row.get("guard_exit_code") != 0:
                _fail(f"behavioral-candidate seed {seed} lacks guard exit zero")
            paths = _exact_behavioral_paths(
                pair_dir=pair_dir,
                robolab_root=robolab_root,
                seed=seed,
            )
            token_trace_integrity = _assert_unambiguous_behavioral_layout(
                study_root=study_root,
                pair_dir=pair_dir,
                paths=paths,
                seed=seed,
                pod=spec.pod,
            )
            stdout_record = _file_record(pair_dir / "pair_stdout_stderr.log")
            claimed_log_sha = row.get("stdout_stderr_log_sha256")
            if claimed_log_sha != stdout_record["sha256"]:
                _fail(f"seed {seed} stdout log changed after shard completion")
            completed_count += 1
            thermal_path = pair_dir / "thermal_events.jsonl"
            pair_inventory.append(
                {
                    **base,
                    "disposition": "behavioral_candidate",
                    "paths": {name: str(path) for name, path in paths.items()},
                    "stdout_log": stdout_record,
                    "thermal_events": (
                        _file_record(thermal_path)
                        if thermal_path.is_file()
                        and thermal_path.stat().st_size > 0
                        else None
                    ),
                    "token_trace_integrity": token_trace_integrity,
                }
            )
            continue
        if status in TECHNICAL_PAIR_STATUSES:
            if row.get("guard_exit_code") in {0, None} and status == (
                "technical_failure_guard_exit_nonzero"
            ):
                _fail(f"technical seed {seed} has an inconsistent guard exit code")
            log_path = pair_dir / "pair_stdout_stderr.log"
            if not log_path.is_file() or log_path.stat().st_size <= 0:
                log_path = ledger_path
            partial_material = _technical_partial_material_by_relation(
                pair_dir=pair_dir,
                robolab_root=robolab_root,
                seed=seed,
            )
            classification_by_relation = {
                relation: (
                    "partial"
                    if partial_material[relation]
                    else "technical_invalid"
                )
                for relation in ("left", "right")
            }
            technical_count += 1
            pair_inventory.append(
                {
                    **base,
                    "disposition": "infrastructure_attempt",
                    "classification_by_relation": classification_by_relation,
                    "partial_material_by_relation": partial_material,
                    "error": row.get("error") or status,
                    "log": _file_record(log_path),
                    "thermal_events": (
                        _file_record(pair_dir / "thermal_events.jsonl")
                        if (pair_dir / "thermal_events.jsonl").is_file()
                        and (pair_dir / "thermal_events.jsonl").stat().st_size > 0
                        else None
                    ),
                }
            )
            continue
        _fail(f"seed {seed} has unknown pair status {status!r}")

    if ledger.get("pair_count_guard_exit_zero") != completed_count:
        _fail(f"{spec.shard_id} completed-pair count disagrees with its ledger")
    if ledger.get("pair_count_technical_failure") != technical_count:
        _fail(f"{spec.shard_id} technical-pair count disagrees with its ledger")
    if shard_status in TERMINAL_SHARD_STATUSES and launching_count:
        _fail(f"terminal shard {spec.shard_id} still contains an active pair")

    return {
        "shard_id": spec.shard_id,
        "pod": spec.pod,
        "status": shard_status,
        "ledger": (
            _file_record(ledger_path)
            if shard_status in TERMINAL_SHARD_STATUSES
            else {"path": str(ledger_path), "mutable_while_running": True}
        ),
        "runtime_identity": _file_record(runtime_path),
        "launch_time_contract": {
            "adapter_contract_sha256": runtime["adapter_contract_hash"],
            "adapter_source_sha256": runtime["adapter_source_sha256"][
                "experiments/v3/pi0_fast_old_name_config_bridge/adapter.py"
            ],
            "tokenizer_source": tokenizer_source,
        },
        "pairs": pair_inventory,
        "pending_seeds": spec.seeds[len(raw_pairs):],
    }


def _validate_compilation_amendment(
    *,
    study_root: Path,
    release_gate: Path,
    shards: list[dict[str, Any]],
) -> dict[str, Any]:
    amendment_path = _require_exact_file(
        study_root / COMPILATION_AMENDMENT_RELATIVE,
        "V3-A003 post-result trace-validation amendment",
    )
    amendment_record = _file_record(amendment_path)
    if amendment_record["sha256"] != COMPILATION_AMENDMENT_SHA256:
        _fail("V3-A003 post-result trace-validation amendment hash changed")
    amendment = _load_object(amendment_path)
    for key, wanted in {
        "schema_version": COMPILATION_AMENDMENT_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": COMPILATION_AMENDMENT_ID,
        "status": (
            "frozen_after_structural_compiler_failure_and_before_any_"
            "v3a002_cell_was_accepted"
        ),
    }.items():
        if amendment.get(key) != wanted:
            _fail(f"V3-A003 amendment mismatch for {key}")

    invalid = amendment.get("invalid_original_assumption", {})
    source_path = invalid.get("source_path")
    if (
        source_path
        != "experiments/v3/pi0_fast_old_name_config_bridge/adapter.py"
        or invalid.get("source_sha256") != ORIGINAL_ADAPTER_SHA256
        or sha256_file(study_root / str(source_path)) != ORIGINAL_ADAPTER_SHA256
    ):
        _fail("V3-A003 does not bind the original launch-time adapter source")

    source_evidence = amendment.get("frozen_source_evidence", {})
    if (
        source_evidence.get("openpi_commit")
        != "235044ed8a1502c0a18338eedc5d7adfe705af05"
        or source_evidence.get("path") != "src/openpi/transforms.py"
        or source_evidence.get("sha256") != FROZEN_TOKENIZER_SOURCE_SHA256
        or source_evidence.get("transform") != "TokenizeFASTInputs"
    ):
        _fail("V3-A003 frozen TokenizeFASTInputs source identity changed")

    scope = amendment.get("scope", {})
    runtime_hashes = [
        shard["runtime_identity"]["sha256"]
        for shard in shards
        if shard["status"] != "missing"
    ]
    declared_runtime_hashes = scope.get("runtime_identity_hashes")
    if (
        scope.get("model_id") != MODEL_ID
        or scope.get("environment_seeds_inclusive") != [8310, 8329]
        or scope.get("matched_pairs") != 20
        or scope.get("behavioral_cells") != 40
        or scope.get("original_adapter_contract_sha256")
        != ORIGINAL_ADAPTER_CONTRACT_SHA256
        or scope.get("release_gate_sha256") != sha256_file(release_gate)
        or not isinstance(declared_runtime_hashes, list)
        or any(value not in declared_runtime_hashes for value in runtime_hashes)
    ):
        _fail("V3-A003 scope does not match the frozen V3-A002 run")
    if len(runtime_hashes) == len(SHARDS) and runtime_hashes != declared_runtime_hashes:
        _fail("V3-A003 runtime-identity order changed")

    for shard in shards:
        if shard["status"] == "missing":
            continue
        launch = shard.get("launch_time_contract", {})
        if (
            launch.get("adapter_contract_sha256")
            != ORIGINAL_ADAPTER_CONTRACT_SHA256
            or launch.get("adapter_source_sha256") != ORIGINAL_ADAPTER_SHA256
            or launch.get("tokenizer_source", {}).get("sha256")
            != FROZEN_TOKENIZER_SOURCE_SHA256
        ):
            _fail(
                f"{shard['shard_id']} launch contract disagrees with V3-A003"
            )

    replacement = amendment.get("replacement_validation", {})
    matched_boundary = replacement.get("matched_pair_token_boundary", {})
    if (
        replacement.get("only_removed_requirement")
        != "Within-episode equality of tokenized_prompt_sha256 across changing robot states."
        or matched_boundary.get("left_first_request_sha256")
        != FIXED_OBSERVATION_TOKEN_SHA256["left"]
        or matched_boundary.get("right_first_request_sha256")
        != FIXED_OBSERVATION_TOKEN_SHA256["right"]
        or "Do not compare the full post-intervention episode token sets"
        not in str(matched_boundary.get("rule", ""))
        or amendment.get("compiler_rule")
        != (
            "Compilation must validate the original launch-time adapter and "
            "runtime identities first, then apply this narrowly scoped "
            "trace-validation amendment without mutating any raw artifact. "
            "Every compiled row and hash manifest must record this amendment "
            "path and SHA-256."
        )
    ):
        _fail("V3-A003 corrected compilation rule changed")
    return {
        "amendment_id": COMPILATION_AMENDMENT_ID,
        "schema_version": COMPILATION_AMENDMENT_SCHEMA,
        **amendment_record,
        "applied_after_original_runtime_validation": True,
        "raw_artifacts_mutated": False,
    }


def audit(*, study_root: Path, raw_root: Path) -> dict[str, Any]:
    study_root = study_root.resolve()
    raw_root = raw_root.resolve()
    if not (study_root / ".git").exists():
        _fail(f"study root is not a Git worktree: {study_root}")
    release_gate = _require_exact_file(
        raw_root / RELEASE_GATE_RELATIVE,
        "V3-A002 release gate",
    )
    shards_root = raw_root / "shards"
    if not shards_root.is_dir():
        _fail(f"shards root is missing: {shards_root}")
    expected_ids = {spec.shard_id for spec in SHARDS}
    observed_ids = {path.name for path in shards_root.iterdir() if path.is_dir()}
    unexpected = observed_ids - expected_ids
    if unexpected:
        _fail(f"unexpected shard directories are ambiguous evidence: {sorted(unexpected)}")

    shards = [
        _inspect_shard(
            study_root=study_root,
            raw_root=raw_root,
            spec=spec,
            release_gate=release_gate,
        )
        for spec in SHARDS
    ]
    amendment = _validate_compilation_amendment(
        study_root=study_root,
        release_gate=release_gate,
        shards=shards,
    )
    pairs = [pair for shard in shards for pair in shard["pairs"]]
    missing_shards = [shard["shard_id"] for shard in shards if shard["status"] == "missing"]
    in_progress = [pair["seed"] for pair in pairs if pair["disposition"] == "in_progress"]
    pending = [seed for shard in shards for seed in shard["pending_seeds"]]
    behavioral = [
        pair for pair in pairs if pair["disposition"] == "behavioral_candidate"
    ]
    infrastructure = [
        pair for pair in pairs if pair["disposition"] == "infrastructure_attempt"
    ]
    safe_to_compile = not missing_shards and not in_progress and not pending and all(
        shard["status"] in TERMINAL_SHARD_STATUSES for shard in shards
    )
    return {
        "schema_version": AUDIT_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "status": "terminal_ready" if safe_to_compile else "in_progress",
        "safe_to_compile": safe_to_compile,
        "frozen_shard_ids": [spec.shard_id for spec in SHARDS],
        "release_gate": _file_record(release_gate),
        "post_result_trace_validation_amendment": amendment,
        "planned_matched_pairs": 20,
        "planned_behavioral_cells": 40,
        "behavioral_candidate_pairs": len(behavioral),
        "infrastructure_attempt_pairs": len(infrastructure),
        "active_seeds": in_progress,
        "pending_seeds": pending,
        "missing_shards": missing_shards,
        "shards": shards,
    }


def _read_thermal_intervention(path: str | None) -> bool:
    if path is None:
        return False
    source = Path(path)
    if not source.is_file():
        return False
    intervention_events = {"cooldown_started", "cooldown_completed", "emergency_hold"}
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise CompileError(f"malformed thermal JSONL {source}: {error}") from error
        if isinstance(record, dict) and record.get("event") in intervention_events:
            return True
    return False


def _write_infrastructure_cells(
    *,
    study_root: Path,
    output_dir: Path,
    advertised_output_dir: Path,
    pair: dict[str, Any],
    amendment: dict[str, Any],
    write_outputs: bool = True,
) -> list[dict[str, Any]]:
    seed = int(pair["seed"])
    runtime_identity = Path(pair["runtime_identity"])
    cells = {row["relation"]: row for row in load_authorized_pair(study_root, seed)}
    records = []
    for relation in ("left", "right"):
        cell = cells[relation]
        destination = (
            output_dir / "infrastructure" / f"seed{seed}" / f"{relation}.jsonl"
        )
        advertised_destination = (
            advertised_output_dir
            / "infrastructure"
            / f"seed{seed}"
            / f"{relation}.jsonl"
        )
        capture = {
            "schema_version": INFRA_CAPTURE_SCHEMA,
            "registered_cell_id": cell["cell_id"],
            "attempt_id": f"{pair['pair_id']}:{relation}:attempt01",
            "environment_seed": seed,
            "policy_seed": seed,
            "prompt": cell["prompt"],
            "requested_relation": relation,
            "classification": pair["classification_by_relation"][relation],
            "stage": "paired_worker_guard",
            "error": pair["error"],
            "log_path": pair["log"]["path"],
            "log_hash": pair["log"]["sha256"],
            "runtime_intervention": _read_thermal_intervention(
                pair.get("thermal_events", {}).get("path")
                if isinstance(pair.get("thermal_events"), dict)
                else None
            ),
            "repair_attempt_id": None,
            "event_timeline": [
                {"sequence": 0, "stage": "attempt_started"},
                {"sequence": 1, "stage": "paired_worker_guard"},
            ],
        }
        record = build_infrastructure_record(
            study_root,
            cell,
            capture,
            runtime_identity,
            advertised_destination,
        )
        record["post_result_trace_validation_amendment"] = {
            "amendment_id": amendment["amendment_id"],
            "schema_version": amendment["schema_version"],
            "path": amendment["path"],
            "sha256": amendment["sha256"],
            "applied_after_original_runtime_validation": True,
            "behavioral_trace_rule_applied": False,
            "raw_artifacts_mutated": False,
        }
        if not write_outputs:
            records.append({"relation": relation, "record": record})
            continue
        manifest = _write_jsonl_staged(
            actual_path=destination,
            advertised_path=advertised_destination,
            records=[record],
            study_root=study_root,
        )
        records.append(
            {
                "relation": relation,
                "record": record,
                "jsonl": _file_record_as(destination, advertised_destination),
                "jsonl_manifest": _file_record_as(
                    destination.with_name(destination.name + ".manifest.json"),
                    advertised_destination.with_name(
                        advertised_destination.name + ".manifest.json"
                    ),
                ),
                "writer_manifest": manifest,
            }
        )
    return records


def _wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(
        math.comb(discordant, value) for value in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def _endpoint_sign_test_exact(aligned: int, anti_aligned: int) -> float:
    """Exact two-sided binomial sign test, excluding exact ties."""

    return _mcnemar_exact(aligned, anti_aligned)


def _final_signed_lateral_offset(record: dict[str, Any]) -> float:
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        _fail("compiled behavioral record has no retained state steps")
    final = steps[-1]
    return float(final["object_xyz"][1]) - float(final["reference_xyz"][1])


def _final_raw_object_robot_y(record: dict[str, Any]) -> float:
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        _fail("compiled behavioral record has no retained state steps")
    return float(steps[-1]["object_xyz"][1])


def _common_prefix_action_metrics(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np

    arrays = {}
    for relation, record in (("left", left), ("right", right)):
        artifact = record["artifacts"]["executed_action_trace"]
        _rehash_claimed_file(artifact)
        actions = np.load(Path(artifact["path"]), allow_pickle=False)
        expected_steps = int(record["actions_executed"])
        if (
            actions.shape != (expected_steps, 8)
            or actions.dtype != np.float32
            or not np.isfinite(actions).all()
        ):
            _fail(f"{relation} validated executed actions changed before summary")
        arrays[relation] = actions
    common = min(len(arrays["left"]), len(arrays["right"]))
    if common <= 0:
        _fail("matched behavioral pair has no common executed-action prefix")
    left_prefix = arrays["left"][:common]
    right_prefix = arrays["right"][:common]
    delta = left_prefix.astype(np.float64) - right_prefix.astype(np.float64)
    return {
        "action_rms_common_prefix": float(np.sqrt(np.mean(np.square(delta)))),
        "common_prefix_actions": int(common),
        "executed_actions_distinct": bool(
            not np.array_equal(left_prefix, right_prefix)
        ),
        "whole_file_hashes_differ_integrity_only": (
            left["artifacts"]["executed_action_trace"]["sha256"]
            != right["artifacts"]["executed_action_trace"]["sha256"]
        ),
    }


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "kind": "numeric",
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
            "observed_count": 0,
            "null_count": 0,
        }
    return {
        "kind": "numeric",
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "observed_count": len(values),
        "null_count": 0,
    }


def _add_raw_record(
    destination: dict[str, dict[str, Any]], record: dict[str, Any]
) -> None:
    path = str(Path(record["path"]).resolve())
    normalized = {
        "path": path,
        "sha256": record["sha256"],
        "bytes": record["bytes"],
    }
    existing = destination.get(path)
    if existing is not None and existing != normalized:
        _fail(f"inconsistent hashes claimed for raw artifact {path}")
    destination[path] = normalized


def _compile_behavioral_pair_amended(
    *,
    study_root: Path,
    seed: int,
    runtime_identity_path: Path,
    release_gate_path: Path,
    amendment: dict[str, Any],
    left_capture_path: Path,
    left_video_path: Path,
    left_action_trace_path: Path,
    left_output_jsonl: Path,
    advertised_left_output_jsonl: Path,
    right_capture_path: Path,
    right_video_path: Path,
    right_action_trace_path: Path,
    right_output_jsonl: Path,
    advertised_right_output_jsonl: Path,
    pair_manifest_path: Path,
    advertised_pair_manifest_path: Path,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Apply V3-A003 after revalidating the original launch-time contract."""

    compiler_record = _file_record(study_root / COMPILER_RELATIVE)
    cells = {row["relation"]: row for row in load_authorized_pair(study_root, seed)}
    if left_output_jsonl.resolve() == right_output_jsonl.resolve():
        _fail("LEFT and RIGHT require separate per-episode JSONL files")
    for path in (left_output_jsonl, right_output_jsonl):
        manifest = path.with_name(path.name + ".manifest.json")
        if path.exists() or manifest.exists():
            _fail(f"refusing to overwrite compiled V3-A003 output: {path}")
    if pair_manifest_path.exists():
        _fail(f"refusing to overwrite V3-A003 pair manifest: {pair_manifest_path}")

    captures = {
        "left": _load_object(left_capture_path),
        "right": _load_object(right_capture_path),
    }
    traces = {
        "left": left_action_trace_path,
        "right": right_action_trace_path,
    }
    outputs = {
        "left": left_output_jsonl,
        "right": right_output_jsonl,
    }
    advertised_outputs = {
        "left": advertised_left_output_jsonl,
        "right": advertised_right_output_jsonl,
    }
    videos = {"left": left_video_path, "right": right_video_path}
    capture_paths = {"left": left_capture_path, "right": right_capture_path}

    original_validator = frozen_adapter._validate_action_trace
    if original_validator.__name__ != "_validate_action_trace":
        _fail("original adapter validator is not the launch-time function")
    frozen_adapter._validate_action_trace = _corrected_action_trace_validator
    try:
        records = {
            relation: build_behavioral_record(
                study_root,
                cells[relation],
                captures[relation],
                capture_paths[relation],
                runtime_identity_path,
                release_gate_path,
                videos[relation],
                traces[relation],
                advertised_outputs[relation],
            )
            for relation in ("left", "right")
        }
    finally:
        frozen_adapter._validate_action_trace = original_validator

    if (
        records["left"]["initial_state_sha256"]
        != records["right"]["initial_state_sha256"]
    ):
        _fail("matched LEFT/RIGHT captures do not begin from an identical reset")
    if records["left"]["runtime_identity"] != records["right"]["runtime_identity"]:
        _fail("matched LEFT/RIGHT captures use different runtime identities")

    amendment_reference = {
        "amendment_id": amendment["amendment_id"],
        "schema_version": amendment["schema_version"],
        "path": amendment["path"],
        "sha256": amendment["sha256"],
        "applied_after_original_runtime_validation": True,
        "raw_artifacts_mutated": False,
    }
    for relation in ("left", "right"):
        records[relation][
            "post_result_trace_validation_amendment"
        ] = amendment_reference
        records[relation]["token_trace_integrity"] = (
            _token_trace_integrity_metadata(
                traces[relation],
                cells[relation],
                int(captures[relation]["actions_executed"]),
            )
        )
    left_first = records["left"]["token_trace_integrity"][
        "ordered_tokenized_input_sha256"
    ][0]
    right_first = records["right"]["token_trace_integrity"][
        "ordered_tokenized_input_sha256"
    ][0]
    if left_first == right_first:
        _fail("LEFT and RIGHT first-request token hashes unexpectedly match")

    if not write_outputs:
        return {
            "status": "validated_under_v3a003_without_writes",
            "initial_state_sha256": records["left"]["initial_state_sha256"],
            "post_result_trace_validation_amendment": amendment_reference,
        }

    left_manifest = _write_jsonl_staged(
        actual_path=left_output_jsonl,
        advertised_path=advertised_left_output_jsonl,
        records=[records["left"]],
        study_root=study_root,
    )
    right_manifest = _write_jsonl_staged(
        actual_path=right_output_jsonl,
        advertised_path=advertised_right_output_jsonl,
        records=[records["right"]],
        study_root=study_root,
    )
    pair_manifest = {
        "schema_version": PAIR_MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "pair_id": records["left"]["pair_id"],
        "environment_seed": seed,
        "initial_state_sha256": records["left"]["initial_state_sha256"],
        "runtime_identity": records["left"]["runtime_identity"],
        "release_gate_sha256": records["left"]["release_authorization"][
            "release_gate_sha256"
        ],
        "post_result_trace_validation_amendment": amendment_reference,
        "corrected_validator_implementation": {
            **compiler_record,
            "relative_path": str(COMPILER_RELATIVE),
        },
        "left": {
            "registered_cell_id": records["left"]["registered_cell_id"],
            "token_trace_integrity": records["left"]["token_trace_integrity"],
            "raw_jsonl": _file_record_as(
                left_output_jsonl,
                advertised_left_output_jsonl,
            ),
            "raw_jsonl_manifest": _file_record_as(
                left_output_jsonl.with_name(
                    left_output_jsonl.name + ".manifest.json"
                ),
                advertised_left_output_jsonl.with_name(
                    advertised_left_output_jsonl.name + ".manifest.json"
                ),
            ),
        },
        "right": {
            "registered_cell_id": records["right"]["registered_cell_id"],
            "token_trace_integrity": records["right"]["token_trace_integrity"],
            "raw_jsonl": _file_record_as(
                right_output_jsonl,
                advertised_right_output_jsonl,
            ),
            "raw_jsonl_manifest": _file_record_as(
                right_output_jsonl.with_name(
                    right_output_jsonl.name + ".manifest.json"
                ),
                advertised_right_output_jsonl.with_name(
                    advertised_right_output_jsonl.name + ".manifest.json"
                ),
            ),
        },
        "historical_pooling_prohibited": True,
    }
    pair_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(pair_manifest_path, pair_manifest)
    return {
        "status": "compiled_under_v3a003",
        "pair_manifest": _file_record_as(
            pair_manifest_path,
            advertised_pair_manifest_path,
        ),
        "left_jsonl_manifest": left_manifest,
        "right_jsonl_manifest": right_manifest,
        "initial_state_sha256": pair_manifest["initial_state_sha256"],
        "post_result_trace_validation_amendment": amendment_reference,
    }


def _verify_manifest_closure(
    *,
    manifest_path: Path,
    actual_root: Path,
    final_root: Path,
    forbidden_staging_path: Path,
    study_root: Path,
) -> None:
    manifest = _load_object(manifest_path)
    if (
        manifest.get("schema_version") != HASH_MANIFEST_SCHEMA
        or manifest.get("study_id") != STUDY_ID
        or manifest.get("model_id") != MODEL_ID
    ):
        _fail("final hash manifest identity changed")
    amendment = manifest.get("post_result_trace_validation_amendment", {})
    if amendment.get("sha256") != COMPILATION_AMENDMENT_SHA256:
        _fail("final hash manifest lost V3-A003")

    for record in manifest.get("raw_source_artifacts", []):
        if not isinstance(record, dict):
            _fail("raw-source manifest entry is not an object")
        _rehash_claimed_file(record)

    expected_relative: set[str] = set()
    for record in manifest.get("derived_artifacts", []):
        if not isinstance(record, dict):
            _fail("derived manifest entry is not an object")
        relative = record.get("relative_path")
        if not isinstance(relative, str) or not relative:
            _fail("derived manifest entry lacks relative_path")
        advertised = Path(str(record.get("path", ""))).resolve()
        if advertised != (final_root / relative).resolve():
            _fail(f"derived artifact does not advertise final path: {advertised}")
        actual = actual_root / relative
        observed = _file_record(actual)
        if (
            record.get("sha256") != observed["sha256"]
            or record.get("bytes") != observed["bytes"]
        ):
            _fail(f"derived artifact hash closure failed: {relative}")
        if relative in expected_relative:
            _fail(f"duplicate derived manifest path: {relative}")
        expected_relative.add(relative)

    observed_relative = {
        str(path.relative_to(actual_root))
        for path in actual_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if observed_relative != expected_relative:
        _fail("derived manifest does not close over the staged cohort")

    for label in ("summary", "infrastructure_intervention_ledger"):
        record = manifest.get(label)
        if not isinstance(record, dict):
            _fail(f"hash manifest lacks {label}")
        relative = str(
            Path(record["path"]).resolve().relative_to(final_root.resolve())
        )
        derived = next(
            (
                row
                for row in manifest["derived_artifacts"]
                if row["relative_path"] == relative
            ),
            None,
        )
        if derived is None or any(
            record.get(key) != derived.get(key) for key in ("path", "sha256", "bytes")
        ):
            _fail(f"{label} record disagrees with derived closure")

    ledger_record = manifest["infrastructure_intervention_ledger"]
    ledger_relative = Path(ledger_record["path"]).resolve().relative_to(
        final_root.resolve()
    )
    infrastructure_ledger = _load_object(actual_root / ledger_relative)
    for row in infrastructure_ledger.get("terminal_shard_ledgers", []):
        _rehash_claimed_file(row)
    for attempt in infrastructure_ledger.get("attempts", []):
        _rehash_claimed_file(attempt["technical_log"])
    for row in infrastructure_ledger.get("thermal_guard", {}).get("pairs", []):
        _rehash_claimed_file(row["ledger"])

    compiler = _file_record(study_root / COMPILER_RELATIVE)
    for pair_record in manifest.get("compiled_pair_manifests", []):
        if not isinstance(pair_record, dict):
            _fail("compiled pair-manifest record is not an object")
        relative = Path(pair_record["path"]).resolve().relative_to(
            final_root.resolve()
        )
        pair_path = actual_root / relative
        pair_manifest = _load_object(pair_path)
        implementation = pair_manifest.get(
            "corrected_validator_implementation", {}
        )
        if any(
            implementation.get(key) != compiler[key]
            for key in ("path", "sha256", "bytes")
        ):
            _fail("pair manifest does not retain the final compiler source hash")
        observed = _file_record(pair_path)
        if any(
            pair_record.get(key) != value
            for key, value in (
                ("sha256", observed["sha256"]),
                ("bytes", observed["bytes"]),
            )
        ):
            _fail("compiled pair-manifest hash record is stale")

    forbidden = str(forbidden_staging_path.resolve()).encode()
    for path in actual_root.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        payload = path.read_bytes()
        if forbidden in payload:
            _fail(f"derived artifact leaks staging path: {path}")
        if path.suffix == ".jsonl":
            for line in payload.decode().splitlines():
                row = json.loads(line)
                amendment = row.get(
                    "post_result_trace_validation_amendment", {}
                )
                if amendment.get("sha256") != COMPILATION_AMENDMENT_SHA256:
                    _fail(f"compiled row lacks V3-A003: {path}")
                for artifact in row.get("artifacts", {}).values():
                    if (
                        isinstance(artifact, dict)
                        and isinstance(artifact.get("sha256"), str)
                        and isinstance(artifact.get("bytes"), int)
                    ):
                        _rehash_claimed_file(artifact)


def compile_all(
    *, study_root: Path, raw_root: Path, output_dir: Path
) -> dict[str, Any]:
    inspection = audit(study_root=study_root, raw_root=raw_root)
    if not inspection["safe_to_compile"]:
        _fail(
            "refusing compilation while a frozen shard is missing or active; "
            f"active={inspection['active_seeds']} pending={inspection['pending_seeds']}"
        )
    output_dir = output_dir.resolve()
    if _is_within(output_dir, raw_root):
        _fail("compiled outputs must remain outside the raw evidence root")
    if output_dir.exists():
        _fail(f"refusing to overwrite compilation directory: {output_dir}")
    staging_dir = output_dir.with_name(f".{output_dir.name}.staging")
    if staging_dir.exists():
        _fail(f"refusing to overwrite prior cohort staging directory: {staging_dir}")

    behavioral_pairs = sorted(
        (
            pair
            for shard in inspection["shards"]
            for pair in shard["pairs"]
            if pair["disposition"] == "behavioral_candidate"
        ),
        key=lambda pair: pair["seed"],
    )
    infrastructure_pairs = sorted(
        (
            pair
            for shard in inspection["shards"]
            for pair in shard["pairs"]
            if pair["disposition"] == "infrastructure_attempt"
        ),
        key=lambda pair: pair["seed"],
    )

    def compile_pair(
        pair: dict[str, Any], *, write_outputs: bool
    ) -> dict[str, Any]:
        seed = int(pair["seed"])
        paths = {key: Path(value) for key, value in pair["paths"].items()}
        pair_output = staging_dir / "pairs" / f"seed{seed}"
        advertised_pair_output = output_dir / "pairs" / f"seed{seed}"
        return _compile_behavioral_pair_amended(
            study_root=study_root,
            seed=seed,
            runtime_identity_path=Path(pair["runtime_identity"]),
            release_gate_path=Path(inspection["release_gate"]["path"]),
            amendment=inspection["post_result_trace_validation_amendment"],
            left_capture_path=paths["left_capture"],
            left_video_path=paths["left_video"],
            left_action_trace_path=paths["left_action_trace"],
            left_output_jsonl=pair_output / "left.jsonl",
            advertised_left_output_jsonl=(
                advertised_pair_output / "left.jsonl"
            ),
            right_capture_path=paths["right_capture"],
            right_video_path=paths["right_video"],
            right_action_trace_path=paths["right_action_trace"],
            right_output_jsonl=pair_output / "right.jsonl",
            advertised_right_output_jsonl=(
                advertised_pair_output / "right.jsonl"
            ),
            pair_manifest_path=pair_output / "pair_manifest.json",
            advertised_pair_manifest_path=(
                advertised_pair_output / "pair_manifest.json"
            ),
            write_outputs=write_outputs,
        )

    # Phase one is deliberately write-free.  Every behavioral and technical
    # input must satisfy its complete compiler contract before the derived
    # destination is created, so a late invalid pair cannot leave a prefix of
    # the cohort looking compiled.
    _file_record(study_root / COMPILER_RELATIVE)
    for pair in behavioral_pairs:
        compile_pair(pair, write_outputs=False)
    for pair in infrastructure_pairs:
        _write_infrastructure_cells(
            study_root=study_root,
            output_dir=staging_dir,
            advertised_output_dir=output_dir,
            pair=pair,
            amendment=inspection["post_result_trace_validation_amendment"],
            write_outputs=False,
        )

    staging_dir.mkdir(parents=True, exist_ok=False)
    compiled_pairs: list[dict[str, Any]] = []
    behavioral_records: list[dict[str, Any]] = []
    raw_records: dict[str, dict[str, Any]] = {}
    for pair in behavioral_pairs:
        seed = int(pair["seed"])
        pair_output = staging_dir / "pairs" / f"seed{seed}"
        advertised_pair_output = output_dir / "pairs" / f"seed{seed}"
        left_jsonl = pair_output / "left.jsonl"
        right_jsonl = pair_output / "right.jsonl"
        pair_manifest = pair_output / "pair_manifest.json"
        result = compile_pair(pair, write_outputs=True)
        left_record = _load_single_jsonl(left_jsonl)
        right_record = _load_single_jsonl(right_jsonl)
        behavioral_records.extend((left_record, right_record))
        compiled_pairs.append(
            {
                "seed": seed,
                "pod": pair["pod"],
                "compile_result": result,
                "pair_manifest": _file_record_as(
                    pair_manifest,
                    advertised_pair_output / "pair_manifest.json",
                ),
            }
        )
        _add_raw_record(raw_records, pair["stdout_log"])
        if isinstance(pair.get("thermal_events"), dict):
            _add_raw_record(raw_records, pair["thermal_events"])
        for record in (left_record, right_record):
            for artifact in record.get("artifacts", {}).values():
                if (
                    isinstance(artifact, dict)
                    and isinstance(artifact.get("sha256"), str)
                    and isinstance(artifact.get("bytes"), int)
                ):
                    _add_raw_record(raw_records, artifact)

    infrastructure_records: list[dict[str, Any]] = []
    for pair in infrastructure_pairs:
        written = _write_infrastructure_cells(
            study_root=study_root,
            output_dir=staging_dir,
            advertised_output_dir=output_dir,
            pair=pair,
            amendment=inspection["post_result_trace_validation_amendment"],
        )
        infrastructure_records.extend(written)
        _add_raw_record(raw_records, pair["log"])
        if isinstance(pair.get("thermal_events"), dict):
            _add_raw_record(raw_records, pair["thermal_events"])

    by_relation = {
        relation: [
            record
            for record in behavioral_records
            if record["requested_relation"] == relation
        ]
        for relation in ("left", "right")
    }
    direction_summary = {}
    for relation, records in by_relation.items():
        successes = sum(record["requested_success"] is True for record in records)
        direction_summary[relation] = {
            "successes": successes,
            "episodes": len(records),
            "wilson_95": _wilson(successes, len(records)),
        }

    records_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for record in behavioral_records:
        records_by_seed.setdefault(int(record["environment_seed"]), {})[
            record["requested_relation"]
        ] = record
    discordance = Counter()
    endpoint_ordering = Counter()
    distinct_action_pairs = 0
    pair_rows = []
    for seed in sorted(records_by_seed):
        pair = records_by_seed[seed]
        if set(pair) != {"left", "right"}:
            _fail(f"compiled seed {seed} lost one matched relation")
        left = pair["left"]
        right = pair["right"]
        outcome = (
            "both"
            if left["requested_success"] and right["requested_success"]
            else "left_only"
            if left["requested_success"]
            else "right_only"
            if right["requested_success"]
            else "neither"
        )
        discordance[outcome] += 1
        left_offset = _final_signed_lateral_offset(left)
        right_offset = _final_signed_lateral_offset(right)
        shift = right_offset - left_offset
        left_raw_y = _final_raw_object_robot_y(left)
        right_raw_y = _final_raw_object_robot_y(right)
        ordering = (
            "aligned" if shift < 0 else "anti_aligned" if shift > 0 else "tie"
        )
        endpoint_ordering[ordering] += 1
        action_metrics = _common_prefix_action_metrics(left, right)
        distinct_action_pairs += int(action_metrics["executed_actions_distinct"])
        pair_rows.append(
            {
                "seed": seed,
                "left_success": left["requested_success"],
                "right_success": right["requested_success"],
                "left_signed_final_lateral_offset_m": left_offset,
                "right_signed_final_lateral_offset_m": right_offset,
                "right_minus_left_endpoint_shift_m": shift,
                "left_raw_robot_y_m": left_raw_y,
                "right_raw_robot_y_m": right_raw_y,
                "right_minus_left_raw_object_robot_y_shift_m": (
                    right_raw_y - left_raw_y
                ),
                "endpoint_ordering": ordering,
                **action_metrics,
                "left_token_hash_cardinality": left["token_trace_integrity"][
                    "token_hash_cardinality"
                ],
                "right_token_hash_cardinality": right["token_trace_integrity"][
                    "token_hash_cardinality"
                ],
            }
        )

    taxonomy = Counter(
        record["failure_taxonomy"] for record in behavioral_records
    )
    all_attempt_pairs = behavioral_pairs + infrastructure_pairs
    retained_thermal_ledgers = sum(
        isinstance(pair.get("thermal_events"), dict)
        for pair in all_attempt_pairs
    )
    thermal_intervention_pairs = sum(
        _read_thermal_intervention(pair["thermal_events"]["path"])
        for pair in all_attempt_pairs
        if isinstance(pair.get("thermal_events"), dict)
    )
    infrastructure_ledger = {
        "schema_version": INFRASTRUCTURE_LEDGER_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "status": "complete_terminal_guarded_cohort",
        "cohort": "V3-A002_public_old_name_config_bridge",
        "historical_pooling_prohibited": True,
        "completed_guarded_pairs": len(all_attempt_pairs),
        "behavioral_episode_count": len(behavioral_records),
        "excluded_attempt_count": len(infrastructure_pairs),
        "terminal_shard_ledgers": [
            {
                "shard_id": shard["shard_id"],
                "pod": shard["pod"],
                **shard["ledger"],
            }
            for shard in inspection["shards"]
        ],
        "post_result_trace_validation_amendment": inspection[
            "post_result_trace_validation_amendment"
        ],
        "behavioral_denominator_effect": "none",
        "technical_or_partial_pair_count": len(infrastructure_pairs),
        "technical_or_partial_cell_count": len(infrastructure_records),
        "attempts": [
            {
                "seed": pair["seed"],
                "pair_id": pair["pair_id"],
                "pod": pair["pod"],
                "classification_by_relation": pair[
                    "classification_by_relation"
                ],
                "partial_material_by_relation": pair[
                    "partial_material_by_relation"
                ],
                "error": pair["error"],
                "technical_log": pair["log"],
            }
            for pair in infrastructure_pairs
        ],
        "thermal_guard": {
            "retained_pair_ledger_count": retained_thermal_ledgers,
            "intervention_pair_count": thermal_intervention_pairs,
            "pairs": [
                {
                    "seed": pair["seed"],
                    "pair_id": pair["pair_id"],
                    "pod": pair["pod"],
                    "ledger": pair["thermal_events"],
                    "intervention": _read_thermal_intervention(
                        pair["thermal_events"]["path"]
                    ),
                }
                for pair in all_attempt_pairs
                if isinstance(pair.get("thermal_events"), dict)
            ],
        },
    }
    infrastructure_ledger_path = (
        staging_dir / "infrastructure_intervention_ledger.json"
    )
    advertised_infrastructure_ledger_path = (
        output_dir / "infrastructure_intervention_ledger.json"
    )
    _write_json_atomic(infrastructure_ledger_path, infrastructure_ledger)
    infrastructure_ledger_record = _file_record_as(
        infrastructure_ledger_path,
        advertised_infrastructure_ledger_path,
    )
    endpoint_shifts = [
        float(pair["right_minus_left_endpoint_shift_m"])
        for pair in pair_rows
    ]
    raw_object_y_shifts = [
        float(pair["right_minus_left_raw_object_robot_y_shift_m"])
        for pair in pair_rows
    ]
    action_rms_values = [
        float(pair["action_rms_common_prefix"])
        for pair in pair_rows
    ]
    common_prefix_counts = [
        int(pair["common_prefix_actions"])
        for pair in pair_rows
    ]
    whole_file_action_hash_differences = sum(
        pair["whole_file_hashes_differ_integrity_only"]
        for pair in pair_rows
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "status": "compiled_terminal_frozen_shards",
        "cohort": "V3-A002_public_old_name_config_bridge",
        "historical_pooling_prohibited": True,
        "historical_pi0_fast_denominator_included": False,
        "post_result_trace_validation_amendment": inspection[
            "post_result_trace_validation_amendment"
        ],
        "frozen_shard_ids": [spec.shard_id for spec in SHARDS],
        "planned_matched_pairs": 20,
        "behavioral_matched_pairs": len(records_by_seed),
        "behavioral_episodes": len(behavioral_records),
        "infrastructure_attempt_pairs": len(infrastructure_pairs),
        "infrastructure_attempt_cell_records": len(infrastructure_records),
        "thermal_guard": {
            "retained_pair_ledgers": retained_thermal_ledgers,
            "intervention_pairs": thermal_intervention_pairs,
            "denominator_effect": "none",
        },
        "infrastructure_intervention_ledger": infrastructure_ledger_record,
        "success_by_direction": direction_summary,
        "success_discordance": {
            "both": discordance["both"],
            "left_only": discordance["left_only"],
            "right_only": discordance["right_only"],
            "neither": discordance["neither"],
            "exact_two_sided_mcnemar_p": _mcnemar_exact(
                discordance["left_only"], discordance["right_only"]
            ),
        },
        "endpoint_ordering": {
            "aligned": endpoint_ordering["aligned"],
            "anti_aligned": endpoint_ordering["anti_aligned"],
            "ties": endpoint_ordering["tie"],
            "exact_two_sided_sign_test_p_excluding_ties": (
                _endpoint_sign_test_exact(
                    endpoint_ordering["aligned"],
                    endpoint_ordering["anti_aligned"],
                )
            ),
            "definition": (
                "RIGHT-condition signed final lateral offset minus LEFT-condition "
                "signed final lateral offset is aligned only when strictly "
                "negative; zero is a tie."
            ),
        },
        "right_minus_left_endpoint_shift_m": _numeric_summary(endpoint_shifts),
        "right_minus_left_raw_object_robot_y_shift_m_geometry_diagnostic": (
            _numeric_summary(raw_object_y_shifts)
        ),
        "action_rms_common_prefix": {
            **_numeric_summary(action_rms_values),
            "coordinate_definition": (
                "RMS over all 8 native mixed policy action coordinates and "
                "min(T_LEFT,T_RIGHT) common executed steps."
            ),
            "unit": "descriptive_mixed_native_action_coordinates",
            "not_meters_or_path_distance": True,
        },
        "common_prefix_action_count": {
            **_numeric_summary(common_prefix_counts),
            "definition": "min(T_LEFT,T_RIGHT) executed action steps per pair",
        },
        "distinct_executed_action_pairs": {
            "count": distinct_action_pairs,
            "pairs": len(records_by_seed),
            "definition": (
                "LEFT and RIGHT validated executed-action arrays differ over "
                "their min-length common executed prefix."
            ),
        },
        "whole_file_executed_action_hash_differences_integrity_only": {
            "count": whole_file_action_hash_differences,
            "pairs": len(records_by_seed),
            "not_a_behavioral_metric": True,
        },
        "failure_taxonomy": dict(sorted(taxonomy.items())),
        "total_actions_executed": sum(
            int(record["actions_executed"]) for record in behavioral_records
        ),
        "pairs": pair_rows,
        "claim_boundary": (
            "This bridge cohort uses public OpenPI commit 235044ed and is not "
            "the missing historical runtime. Its results remain a separate "
            "20-pair denominator."
        ),
    }
    summary_path = staging_dir / "summary.json"
    advertised_summary_path = output_dir / "summary.json"
    _write_json_atomic(summary_path, summary)

    for shard in inspection["shards"]:
        _add_raw_record(raw_records, shard["runtime_identity"])
        if isinstance(shard.get("ledger"), dict) and "sha256" in shard["ledger"]:
            _add_raw_record(raw_records, shard["ledger"])
    _add_raw_record(raw_records, inspection["release_gate"])
    _add_raw_record(
        raw_records,
        inspection["post_result_trace_validation_amendment"],
    )
    _add_raw_record(raw_records, _file_record(study_root / COMPILER_RELATIVE))
    raw_records = {
        path: _rehash_claimed_file(record)
        for path, record in sorted(raw_records.items())
    }

    derived = []
    manifest_path = staging_dir / "hash_manifest.json"
    for path in sorted(staging_dir.rglob("*")):
        if path.is_file() and path != manifest_path:
            relative = path.relative_to(staging_dir)
            record = _file_record_as(path, output_dir / relative)
            record["relative_path"] = str(relative)
            derived.append(record)
    manifest = {
        "schema_version": HASH_MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "historical_pooling_prohibited": True,
        "raw_inputs_read_only": True,
        "post_result_trace_validation_amendment": inspection[
            "post_result_trace_validation_amendment"
        ],
        "summary": _file_record_as(summary_path, advertised_summary_path),
        "infrastructure_intervention_ledger": (
            infrastructure_ledger_record
        ),
        "raw_source_artifacts": sorted(raw_records.values(), key=lambda row: row["path"]),
        "derived_artifacts": derived,
        "compiled_pair_manifests": [row["pair_manifest"] for row in compiled_pairs],
    }
    _write_json_atomic(manifest_path, manifest)
    _verify_manifest_closure(
        manifest_path=manifest_path,
        actual_root=staging_dir,
        final_root=output_dir,
        forbidden_staging_path=staging_dir,
        study_root=study_root,
    )
    os.replace(staging_dir, output_dir)
    final_manifest_path = output_dir / "hash_manifest.json"
    try:
        _verify_manifest_closure(
            manifest_path=final_manifest_path,
            actual_root=output_dir,
            final_root=output_dir,
            forbidden_staging_path=staging_dir,
            study_root=study_root,
        )
    except Exception:
        os.replace(output_dir, staging_dir)
        raise
    return {
        "status": "compiled",
        "summary": _file_record(output_dir / "summary.json"),
        "hash_manifest": _file_record(final_manifest_path),
        "infrastructure_intervention_ledger": _file_record(
            output_dir / "infrastructure_intervention_ledger.json"
        ),
        "behavioral_pairs": len(compiled_pairs),
        "infrastructure_pairs": len(infrastructure_pairs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for name in ("audit", "compile"):
        command = commands.add_parser(name)
        command.add_argument("--study-root", type=Path, required=True)
        command.add_argument("--raw-root", type=Path, required=True)
    commands.choices["compile"].add_argument(
        "--output-dir", type=Path, required=True
    )
    args = parser.parse_args()
    if args.mode == "audit":
        result = audit(
            study_root=args.study_root.resolve(),
            raw_root=args.raw_root.resolve(),
        )
    else:
        result = compile_all(
            study_root=args.study_root.resolve(),
            raw_root=args.raw_root.resolve(),
            output_dir=args.output_dir.resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
