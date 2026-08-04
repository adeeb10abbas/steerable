#!/usr/bin/env python3
"""Split V2-A015 setup failures and materialize explicit zero-event ledgers.

The V2-A015 preflight is the canonical count of setup-invalid launch attempts.
Native thermal-guard ledgers, when supplied by exact path, only corroborate a
canonical preflight row.  Their LEFT/RIGHT rows describe one paired launch and
must never become two behavioral or setup-attempt counts.

Successful native guards intentionally do not write an intervention ledger.
Consequently, this tool writes an empty runtime-intervention ledger only when
the caller explicitly asserts zero observed interventions for each arm.  A
missing native file is never interpreted as evidence of zero events.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


AMENDMENT_ID = "V2-A015"
PREFLIGHT_SCHEMA = "vla-wam-shared-v2-cfg-ablation-preflight-v1"
NATIVE_INVALID_SCHEMA = "vla-wam-shared-v2-native-thermal-invalid-attempts-v1"
INVALID_OUTPUT_SCHEMA = (
    "vla-wam-shared-v2-v2a015-model-setup-invalid-attempt-ledger-v1"
)
RUNTIME_OUTPUT_SCHEMA = (
    "vla-wam-shared-v2-v2a015-model-runtime-intervention-ledger-v1"
)

DREAMZERO_MODEL_ID = "dreamzero_droid_action_cfg"
DREAMZERO_ARM_ID = "dreamzero_action_cfg_s2"
COSMOS_MODEL_ID = "cosmos3_nano_policy_droid"
COSMOS_ARM_ID = "cosmos3_nano_no_cfg_g1"

MODEL_ARMS = {
    DREAMZERO_MODEL_ID: DREAMZERO_ARM_ID,
    COSMOS_MODEL_ID: COSMOS_ARM_ID,
}

DREAMZERO_GENERAL_STAGES = {
    "source_patch_apply_check",
    "bounded_loader_static_audit",
}

# These three failed paired launches predate an explicit preflight-stage field
# in the native thermal-guard schema.  The mapping binds their exact launch-log
# attempt numbers to the canonical rows already recorded in the preflight.
DREAMZERO_NATIVE_ATTEMPT_STAGE = {
    2: "dreamzero_action_s2_behavior_isaac_eula_gate",
    3: "dreamzero_action_s2_behavior_vulkan_native_library_gate",
    4: "dreamzero_action_s2_behavior_glvnd_and_warp_cache_gate",
}

_ATTEMPT_RE = re.compile(r"(?:^|[_-])attempt0*([0-9]+)(?:[^0-9]|$)", re.I)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing exact input file: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _load_json(path: Path) -> Any:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing exact JSON input: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON input {path}: {error}") from error


def _write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite output without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.staging")
    if staging.exists():
        raise RuntimeError(f"Refusing to overwrite stale staging file: {staging}")
    try:
        staging.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def _require_string(row: dict[str, Any], key: str, *, label: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label} requires a nonempty string {key!r}")
    return value


def _route_stage(stage: str) -> tuple[str, str]:
    if stage in DREAMZERO_GENERAL_STAGES or stage.startswith("dreamzero_"):
        return DREAMZERO_MODEL_ID, DREAMZERO_ARM_ID
    if stage.startswith("cosmos3_nano_g1_"):
        return COSMOS_MODEL_ID, COSMOS_ARM_ID
    raise RuntimeError(
        f"Unrecognized V2-A015 setup-invalid stage {stage!r}; refusing to omit it"
    )


def _normalize_preflight(
    path: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, int]],
    dict[str, Any],
]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError("V2-A015 preflight must be a JSON object")
    if payload.get("schema_version") != PREFLIGHT_SCHEMA:
        raise RuntimeError(
            "Unexpected V2-A015 preflight schema: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("amendment_id") != AMENDMENT_ID:
        raise RuntimeError("Preflight is not bound to amendment V2-A015")
    source_rows = payload.get("setup_invalid_attempts")
    if not isinstance(source_rows, list):
        raise RuntimeError("Preflight lacks setup_invalid_attempts[]")

    accounting = payload.get("inference_accounting")
    if not isinstance(accounting, dict):
        raise RuntimeError("Preflight lacks inference_accounting")
    if accounting.get("setup_invalid_attempt_count") != len(source_rows):
        raise RuntimeError(
            "Preflight setup-invalid row count disagrees with inference_accounting"
        )
    if accounting.get("setup_invalid_attempts_in_behavioral_denominator") != 0:
        raise RuntimeError(
            "Preflight accounting requires "
            "setup_invalid_attempts_in_behavioral_denominator=0"
        )

    routed: dict[str, list[dict[str, Any]]] = {
        DREAMZERO_MODEL_ID: [],
        COSMOS_MODEL_ID: [],
    }
    stage_index: dict[str, dict[str, int]] = {
        DREAMZERO_MODEL_ID: {},
        COSMOS_MODEL_ID: {},
    }
    seen_row_hashes: set[str] = set()
    for source_index, source_row in enumerate(source_rows):
        label = f"setup_invalid_attempts[{source_index}]"
        if not isinstance(source_row, dict):
            raise RuntimeError(f"{label} is not an object")
        stage = _require_string(source_row, "stage", label=label)
        _require_string(source_row, "error", label=label)
        _require_string(source_row, "cause", label=label)
        _require_string(source_row, "effect", label=label)
        model_id, arm_id = _route_stage(stage)
        source_hash = _value_sha256(source_row)
        if source_hash in seen_row_hashes:
            raise RuntimeError(
                f"Duplicate canonical preflight row at {label}; refusing to deduplicate attempts"
            )
        seen_row_hashes.add(source_hash)
        normalized = {
            "attempt_id": (
                f"v2a015-{model_id}-setup-{source_hash[:16]}"
            ),
            "amendment_id": AMENDMENT_ID,
            "model_id": model_id,
            "arm_id": arm_id,
            "classification": "setup_invalid",
            "behavioral_result_valid": False,
            "wall_latency_valid": False,
            "denominator_status": "excluded",
            "stage": stage,
            "error": source_row["error"],
            "cause": source_row["cause"],
            "effect": source_row["effect"],
            "provenance": {
                "canonical_collection": "setup_invalid_attempts",
                "canonical_source_index": source_index,
                "canonical_source_record_sha256": source_hash,
            },
        }
        routed[model_id].append(normalized)
        # Repeated stages can represent multiple real launch attempts (the two
        # patch-apply rows do).  Native corroboration therefore requires an
        # unambiguous, unique stage.
        if stage in stage_index[model_id]:
            stage_index[model_id][stage] = -1
        else:
            stage_index[model_id][stage] = len(routed[model_id]) - 1

    return routed, stage_index, _file_record(path)


def _native_launch_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"id", "requested_relation"}
    }


def _native_stage_for_launch(model_id: str, rows: list[dict[str, Any]]) -> str:
    declared = {row.get("preflight_stage") for row in rows}
    if len(declared) == 1 and next(iter(declared)) not in (None, ""):
        stage = next(iter(declared))
        if not isinstance(stage, str):
            raise RuntimeError("Native preflight_stage must be a string")
        return stage

    raw_log = rows[0].get("raw_event_log")
    if not isinstance(raw_log, str) or not raw_log:
        raise RuntimeError("Native invalid launch lacks raw_event_log")
    matches = _ATTEMPT_RE.findall(Path(raw_log).name)
    if len(matches) != 1:
        raise RuntimeError(
            f"Cannot identify one launch attempt from native raw_event_log: {raw_log}"
        )
    attempt = int(matches[0])
    if model_id == DREAMZERO_MODEL_ID and attempt in DREAMZERO_NATIVE_ATTEMPT_STAGE:
        return DREAMZERO_NATIVE_ATTEMPT_STAGE[attempt]
    raise RuntimeError(
        f"No fail-closed preflight mapping for native {model_id} attempt {attempt}; "
        "add an exact preflight_stage to the native rows before compilation"
    )


def _load_native_ledgers(
    paths: Iterable[Path],
    routed: dict[str, list[dict[str, Any]]],
    stage_index: dict[str, dict[str, int]],
) -> dict[str, Any]:
    source_records: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    group_sources: dict[str, list[dict[str, Any]]] = {}
    seen_paths: set[Path] = set()
    native_event_count = 0
    native_event_count_by_model = {model_id: 0 for model_id in MODEL_ARMS}

    for supplied_path in paths:
        path = supplied_path.expanduser().resolve()
        if path in seen_paths:
            raise RuntimeError(f"Native invalid ledger supplied more than once: {path}")
        seen_paths.add(path)
        payload = _load_json(path)
        if not isinstance(payload, dict) or payload.get("schema_version") != NATIVE_INVALID_SCHEMA:
            raise RuntimeError(f"Unexpected native invalid-ledger schema: {path}")
        events = payload.get("events")
        if not isinstance(events, list):
            raise RuntimeError(f"Native invalid ledger lacks events[]: {path}")
        source_record = {**_file_record(path), "source_event_count": len(events)}
        native_event_count += len(events)
        source_models: set[str] = set()

        for source_index, row in enumerate(events):
            label = f"{path}:events[{source_index}]"
            if not isinstance(row, dict):
                raise RuntimeError(f"{label} is not an object")
            model_id = _require_string(row, "model_id", label=label)
            if model_id not in MODEL_ARMS:
                raise RuntimeError(f"{label} has unrecognized model_id {model_id!r}")
            source_models.add(model_id)
            native_event_count_by_model[model_id] += 1
            relation = row.get("requested_relation")
            if relation not in {"left", "right"}:
                raise RuntimeError(f"{label} lacks exact LEFT/RIGHT relation identity")
            classification = str(row.get("classification", "")).lower()
            if not any(
                token in classification
                for token in ("partial", "invalid", "infrastructure", "setup")
            ):
                raise RuntimeError(f"{label} is not classified as an invalid attempt")
            if row.get("behavioral_result_valid") is not False:
                raise RuntimeError(f"{label} is not explicitly behavior-invalid")
            if row.get("wall_latency_valid") is not False:
                raise RuntimeError(f"{label} is not explicitly wall-latency-invalid")
            for key in (
                "pair_id",
                "started_at_utc",
                "completed_at_utc",
                "status",
                "raw_event_log",
            ):
                _require_string(row, key, label=label)
            for key in ("environment_seed", "sampling_seed", "gpu_index"):
                if not isinstance(row.get(key), int) or isinstance(row.get(key), bool):
                    raise RuntimeError(f"{label} requires integer {key!r}")

            launch_payload = _native_launch_payload(row)
            launch_hash = _value_sha256(launch_payload)
            grouped.setdefault(launch_hash, []).append(row)
            group_sources.setdefault(launch_hash, []).append(
                {
                    "ledger_sha256": source_record["sha256"],
                    "source_event_index": source_index,
                    "native_event_id": row.get("id"),
                    "requested_relation": relation,
                }
            )
        source_record["model_ids"] = sorted(source_models)
        source_records.append(source_record)

    seen_stage_launches: set[tuple[str, str]] = set()
    corroborated_by_model = {model_id: 0 for model_id in MODEL_ARMS}
    for launch_hash, rows in sorted(grouped.items()):
        relations = [row["requested_relation"] for row in rows]
        if sorted(relations) != ["left", "right"]:
            raise RuntimeError(
                "Each native paired launch must contain exactly one LEFT and one RIGHT row; "
                f"launch {launch_hash} has {relations!r}"
            )
        model_ids = {row["model_id"] for row in rows}
        if len(model_ids) != 1:
            raise RuntimeError(f"Native launch {launch_hash} mixes model identities")
        model_id = next(iter(model_ids))
        stage = _native_stage_for_launch(model_id, rows)
        index = stage_index[model_id].get(stage)
        if index is None:
            raise RuntimeError(
                f"Native launch maps to absent canonical preflight stage {stage!r}"
            )
        if index < 0:
            raise RuntimeError(
                f"Native launch maps ambiguously to repeated preflight stage {stage!r}"
            )
        stage_key = (model_id, stage)
        if stage_key in seen_stage_launches:
            raise RuntimeError(
                f"Multiple native launches map to one canonical preflight attempt {stage!r}"
            )
        seen_stage_launches.add(stage_key)
        launch = rows[0]
        corroboration = {
            "native_launch_id": f"native-launch-{launch_hash[:20]}",
            "native_launch_record_sha256": launch_hash,
            "pair_id": launch["pair_id"],
            "environment_seed": launch["environment_seed"],
            "sampling_seed": launch["sampling_seed"],
            "started_at_utc": launch["started_at_utc"],
            "completed_at_utc": launch["completed_at_utc"],
            "status": launch["status"],
            "worker_pid": launch.get("worker_pid"),
            "worker_pgid": launch.get("worker_pgid"),
            "worker_exit_code": launch.get("worker_exit_code"),
            "gpu_index": launch["gpu_index"],
            "raw_event_log": launch["raw_event_log"],
            "requested_relations": ["left", "right"],
            "source_events": sorted(
                group_sources[launch_hash], key=lambda row: row["requested_relation"]
            ),
            "counted_as_additional_setup_attempts": 0,
        }
        routed[model_id][index]["provenance"][
            "corroborating_native_launch"
        ] = corroboration
        corroborated_by_model[model_id] += 1

    return {
        "sources": source_records,
        "source_event_count": native_event_count,
        "source_event_count_by_model": native_event_count_by_model,
        "deduplicated_launch_count": len(grouped),
        "deduplicated_launch_count_by_model": corroborated_by_model,
        "corroborated_setup_attempt_count_by_model": corroborated_by_model,
        "counted_as_additional_setup_attempts": 0,
    }


def _invalid_payload(
    *,
    model_id: str,
    arm_id: str,
    attempts: list[dict[str, Any]],
    preflight_record: dict[str, Any],
    native_provenance: dict[str, Any],
) -> dict[str, Any]:
    model_sources = [
        source
        for source in native_provenance["sources"]
        if model_id in source["model_ids"]
    ]
    return {
        "schema_version": INVALID_OUTPUT_SCHEMA,
        "status": "complete_canonical_preflight_split",
        "amendment_id": AMENDMENT_ID,
        "model_id": model_id,
        "arm_id": arm_id,
        "setup_invalid_attempt_count": len(attempts),
        "attempts": attempts,
        "accounting": {
            "behavioral_result_count": 0,
            "behavioral_denominator_count": 0,
            "denominator_policy": (
                "Every row is setup-invalid and excluded from every behavioral "
                "success denominator. Native LEFT/RIGHT rows corroborate one paired "
                "launch and add zero setup-attempt counts."
            ),
        },
        "provenance": {
            "preflight": preflight_record,
            "canonical_collection": "setup_invalid_attempts",
            "canonical_source_row_indices": [
                row["provenance"]["canonical_source_index"] for row in attempts
            ],
            "native_invalid_ledgers": model_sources,
            "native_source_event_count": native_provenance[
                "source_event_count_by_model"
            ][model_id],
            "deduplicated_native_launch_count": native_provenance[
                "deduplicated_launch_count_by_model"
            ][model_id],
            "corroborated_setup_attempt_count": native_provenance[
                "corroborated_setup_attempt_count_by_model"
            ][model_id],
            "native_events_counted_as_additional_setup_attempts": 0,
        },
    }


def _runtime_payload(
    *, model_id: str, arm_id: str, preflight_record: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_OUTPUT_SCHEMA,
        "status": "complete_explicit_zero_observed",
        "amendment_id": AMENDMENT_ID,
        "model_id": model_id,
        "arm_id": arm_id,
        "runtime_intervention_count": 0,
        "events": [],
        "accounting": {
            "behavioral_denominator_count_removed": 0,
            "wall_latency_count_excluded": 0,
        },
        "zero_event_basis": {
            "explicit_caller_assertion": True,
            "assertion_scope": (
                "Completed native process-group guard launches for this V2-A015 arm "
                "observed no cooldown or emergency thermal intervention."
            ),
            "missing_file_inference_used": False,
            "note": (
                "The native guard emits no intervention JSON on an uninterrupted "
                "successful launch. This zero is recorded from the caller's explicit "
                "observation/assertion, never from file absence alone."
            ),
        },
        "provenance": {"preflight": preflight_record},
    }


def build_ledgers(
    *,
    preflight: Path,
    native_invalid_ledgers: Iterable[Path],
    dreamzero_invalid_output: Path | None = None,
    cosmos_invalid_output: Path | None = None,
    dreamzero_runtime_output: Path | None = None,
    cosmos_runtime_output: Path | None = None,
    assert_dreamzero_no_runtime_interventions: bool = False,
    assert_cosmos_no_runtime_interventions: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    dream_selected = (
        dreamzero_invalid_output is not None or dreamzero_runtime_output is not None
    )
    cosmos_selected = cosmos_invalid_output is not None or cosmos_runtime_output is not None
    if dream_selected and (
        dreamzero_invalid_output is None or dreamzero_runtime_output is None
    ):
        raise RuntimeError(
            "DreamZero compilation requires both invalid and runtime output paths"
        )
    if cosmos_selected and (cosmos_invalid_output is None or cosmos_runtime_output is None):
        raise RuntimeError("Cosmos compilation requires both invalid and runtime output paths")
    if not dream_selected and not cosmos_selected:
        raise RuntimeError("Select at least one complete model arm for ledger compilation")
    if dream_selected and not assert_dreamzero_no_runtime_interventions:
        raise RuntimeError(
            "Refusing to infer DreamZero zero interventions; pass the explicit assertion"
        )
    if cosmos_selected and not assert_cosmos_no_runtime_interventions:
        raise RuntimeError(
            "Refusing to infer Cosmos zero interventions; pass the explicit assertion"
        )

    outputs = [
        path
        for path in (
            dreamzero_invalid_output,
            cosmos_invalid_output,
            dreamzero_runtime_output,
            cosmos_runtime_output,
        )
        if path is not None
    ]
    resolved_outputs = [path.expanduser().resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise RuntimeError("All four output paths must be distinct")

    routed, stage_index, preflight_record = _normalize_preflight(preflight)
    native_provenance = _load_native_ledgers(
        native_invalid_ledgers, routed, stage_index
    )
    payloads: dict[Path, dict[str, Any]] = {}
    if dream_selected:
        assert dreamzero_invalid_output is not None
        assert dreamzero_runtime_output is not None
        payloads[dreamzero_invalid_output] = _invalid_payload(
            model_id=DREAMZERO_MODEL_ID,
            arm_id=DREAMZERO_ARM_ID,
            attempts=routed[DREAMZERO_MODEL_ID],
            preflight_record=preflight_record,
            native_provenance=native_provenance,
        )
        payloads[dreamzero_runtime_output] = _runtime_payload(
            model_id=DREAMZERO_MODEL_ID,
            arm_id=DREAMZERO_ARM_ID,
            preflight_record=preflight_record,
        )
    if cosmos_selected:
        assert cosmos_invalid_output is not None
        assert cosmos_runtime_output is not None
        payloads[cosmos_invalid_output] = _invalid_payload(
            model_id=COSMOS_MODEL_ID,
            arm_id=COSMOS_ARM_ID,
            attempts=routed[COSMOS_MODEL_ID],
            preflight_record=preflight_record,
            native_provenance=native_provenance,
        )
        payloads[cosmos_runtime_output] = _runtime_payload(
            model_id=COSMOS_MODEL_ID,
            arm_id=COSMOS_ARM_ID,
            preflight_record=preflight_record,
        )
    for path, payload in payloads.items():
        _write_json(path, payload, overwrite=overwrite)
    return {
        "dreamzero_setup_invalid_attempt_count": len(routed[DREAMZERO_MODEL_ID]),
        "cosmos_setup_invalid_attempt_count": len(routed[COSMOS_MODEL_ID]),
        "native_source_event_count": native_provenance["source_event_count"],
        "native_deduplicated_launch_count": native_provenance[
            "deduplicated_launch_count"
        ],
        "compiled_arms": [
            arm_id
            for selected, arm_id in (
                (dream_selected, DREAMZERO_ARM_ID),
                (cosmos_selected, COSMOS_ARM_ID),
            )
            if selected
        ],
        "runtime_intervention_count": {
            arm_id: 0
            for selected, arm_id in (
                (dream_selected, DREAMZERO_ARM_ID),
                (cosmos_selected, COSMOS_ARM_ID),
            )
            if selected
        },
        "outputs": [str(path.expanduser().resolve()) for path in outputs],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument(
        "--native-invalid-ledger",
        type=Path,
        action="append",
        default=[],
        help="Exact native invalid-ledger path; repeat explicitly, never use a glob.",
    )
    parser.add_argument("--dreamzero-invalid-output", type=Path)
    parser.add_argument("--cosmos-invalid-output", type=Path)
    parser.add_argument("--dreamzero-runtime-output", type=Path)
    parser.add_argument("--cosmos-runtime-output", type=Path)
    parser.add_argument(
        "--assert-dreamzero-no-runtime-interventions",
        action="store_true",
        help="Explicitly assert that no DreamZero runtime thermal intervention occurred.",
    )
    parser.add_argument(
        "--assert-cosmos-no-runtime-interventions",
        action="store_true",
        help="Explicitly assert that no Cosmos runtime thermal intervention occurred.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = build_ledgers(
            preflight=args.preflight,
            native_invalid_ledgers=args.native_invalid_ledger,
            dreamzero_invalid_output=args.dreamzero_invalid_output,
            cosmos_invalid_output=args.cosmos_invalid_output,
            dreamzero_runtime_output=args.dreamzero_runtime_output,
            cosmos_runtime_output=args.cosmos_runtime_output,
            assert_dreamzero_no_runtime_interventions=(
                args.assert_dreamzero_no_runtime_interventions
            ),
            assert_cosmos_no_runtime_interventions=(
                args.assert_cosmos_no_runtime_interventions
            ),
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
