#!/usr/bin/env python3
"""Validate Tier-C checkpoint provenance against its committed JSON Schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


EXPECTED_MODEL_IDS = {
    "pi0_fast_droid_vla",
    "pi0_fast_old_name_config_v3a002",
    "pi05_current_stack_droid",
    "groot_n17_droid_vla",
    "cosmos3_edge_policy_droid",
    "cosmos3_nano_policy_droid",
    "dreamzero_droid_action_cfg",
    "efficient_wam_rt_robotwin",
    "fastwam_robotwin",
    "lingbot_va_robotwin",
}


class ProvenanceValidationError(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_ref(schema: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ProvenanceValidationError(f"unsupported external schema reference: {ref}")
    node: Any = schema
    for token in ref[2:].split("/"):
        node = node[token.replace("~1", "/").replace("~0", "~")]
    return node


def _type_ok(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }[expected]


def validate_instance(value: Any, rule: dict[str, Any], schema: dict[str, Any], location: str = "$") -> None:
    if "$ref" in rule:
        validate_instance(value, _resolve_ref(schema, rule["$ref"]), schema, location)
        return
    if "oneOf" in rule:
        passes = 0
        for branch in rule["oneOf"]:
            try:
                validate_instance(value, branch, schema, location)
            except ProvenanceValidationError:
                continue
            passes += 1
        if passes != 1:
            raise ProvenanceValidationError(f"{location}: expected exactly one oneOf branch, got {passes}")
        return
    if "const" in rule and value != rule["const"]:
        raise ProvenanceValidationError(f"{location}: does not match const")
    if "enum" in rule and value not in rule["enum"]:
        raise ProvenanceValidationError(f"{location}: {value!r} is outside enum")
    expected_type = rule.get("type")
    if expected_type and not _type_ok(value, expected_type):
        raise ProvenanceValidationError(f"{location}: expected {expected_type}, got {type(value).__name__}")
    if isinstance(value, str):
        if len(value) < rule.get("minLength", 0):
            raise ProvenanceValidationError(f"{location}: string is too short")
        if "pattern" in rule and re.fullmatch(rule["pattern"], value) is None:
            raise ProvenanceValidationError(f"{location}: does not match {rule['pattern']}")
    if isinstance(value, int) and not isinstance(value, bool) and "minimum" in rule and value < rule["minimum"]:
        raise ProvenanceValidationError(f"{location}: below minimum")
    if isinstance(value, list):
        if len(value) < rule.get("minItems", 0):
            raise ProvenanceValidationError(f"{location}: too few items")
        if "items" in rule:
            for index, item in enumerate(value):
                validate_instance(item, rule["items"], schema, f"{location}[{index}]")
    if isinstance(value, dict):
        for key in rule.get("required", []):
            if key not in value:
                raise ProvenanceValidationError(f"{location}: missing required property {key}")
        properties = rule.get("properties", {})
        for key, item in value.items():
            if key in properties:
                validate_instance(item, properties[key], schema, f"{location}.{key}")
            elif rule.get("additionalProperties") is False:
                raise ProvenanceValidationError(f"{location}: unexpected property {key}")


def validate_checkpoint_provenance(root: Path) -> list[str]:
    checks: list[str] = []
    schema_path = root / "artifacts/vla_wam_shared_v3/prospective_tier_b/checkpoint_provenance.schema.json"
    output = root / "artifacts/vla_wam_shared_v3/prospective_tier_b/checkpoint_provenance"
    table_path = output / "checkpoint_provenance_table.json"
    manifest_path = output / "checkpoint_provenance_manifest.json"
    for path in (schema_path, table_path, output / "checkpoint_provenance_table.md", manifest_path):
        if not path.is_file():
            raise ProvenanceValidationError(f"required provenance artifact missing: {path.relative_to(root)}")
        checks.append(f"exists: {path.relative_to(root)}")
    schema = load(schema_path)
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ProvenanceValidationError("provenance schema is not Draft 2020-12")
    checks.append("committed schema declares JSON Schema Draft 2020-12")

    record_files = sorted(
        p for p in output.glob("*.json")
        if p.name not in {"checkpoint_provenance_table.json", "checkpoint_provenance_manifest.json"}
    )
    records = [load(path) for path in record_files]
    model_ids = {record.get("model_id") for record in records}
    if model_ids != EXPECTED_MODEL_IDS or len(records) != len(EXPECTED_MODEL_IDS):
        raise ProvenanceValidationError(f"checkpoint identity coverage mismatch: {sorted(model_ids)}")
    checks.append("exact ten-identity V3 provenance coverage")

    for path, record in zip(record_files, records, strict=True):
        validate_instance(record, schema, schema)
        if path.stem != record["model_id"]:
            raise ProvenanceValidationError(f"record filename/model mismatch: {path.name}")
        training = record["training_episode_multiset"]
        if training["episode_count"] != "not_disclosed" or training["trajectory_count"] != "not_disclosed":
            raise ProvenanceValidationError(f"{record['model_id']}: unverified training count claimed")
        captions = record["caption_exposure"]
        if captions["disclosure_status"] != "not_auditable" or any(
            captions[key] != "unknown" for key in ("left_right_tokens", "exact_probe_sentences", "synthetic_or_recaptioned_language")
        ):
            raise ProvenanceValidationError(f"{record['model_id']}: caption unknowns are not explicit")
        if record["preprocessing"].get("scope") != "v3 inference-time interface only" or record["preprocessing"].get("training_preprocessing") != "not_disclosed":
            raise ProvenanceValidationError(f"{record['model_id']}: inference/training preprocessing boundary missing")
        if not record.get("known_unknowns"):
            raise ProvenanceValidationError(f"{record['model_id']}: known_unknowns missing")
        for item in record["evidence"]:
            source = item["path_or_url"]
            if source.startswith(("http://", "https://")):
                continue
            source_path = root / source
            if not source_path.is_file() or sha256(source_path) != item["sha256_or_status"]:
                raise ProvenanceValidationError(f"{record['model_id']}: stale evidence hash for {source}")
        checks.append(f"schema/source boundary/hash evidence: {record['model_id']}")

    table = load(table_path)
    if table.get("record_count") != 10 or {row.get("model_id") for row in table.get("records", [])} != EXPECTED_MODEL_IDS:
        raise ProvenanceValidationError("provenance table does not match record identities")
    by_id = {record["model_id"]: record for record in records}
    for row in table["records"]:
        record = by_id[row["model_id"]]
        if row["content_sha256"] != record["checkpoint_identity"]["content_sha256"] or row["runtime_identity_sha256"] != record["checkpoint_identity"]["runtime_identity_sha256"]:
            raise ProvenanceValidationError(f"table hash mismatch: {row['model_id']}")
    checks.append("compact table agrees with individual records")

    manifest = load(manifest_path)
    if manifest.get("record_count") != 10 or manifest.get("schema_sha256") != sha256(schema_path):
        raise ProvenanceValidationError("provenance manifest schema/count mismatch")
    builder_path = root / manifest["builder_path"]
    if sha256(builder_path) != manifest["builder_sha256"]:
        raise ProvenanceValidationError("provenance builder hash is stale")
    expected_manifest_paths = {str(path.relative_to(root)) for path in record_files} | {
        str(table_path.relative_to(root)),
        str((output / "checkpoint_provenance_table.md").relative_to(root)),
    }
    manifest_paths = {item["path"] for item in manifest.get("files", [])}
    if manifest_paths != expected_manifest_paths:
        raise ProvenanceValidationError("provenance manifest has missing or stale file entries")
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ProvenanceValidationError(f"stale manifest entry: {item['path']}")
    checks.append("manifest binds schema, builder, records, and tables")

    historical = by_id["pi0_fast_droid_vla"]
    bridge = by_id["pi0_fast_old_name_config_v3a002"]
    if historical["study_status"] == bridge["study_status"] or "historical" not in historical["study_status"] or "compatibility" not in bridge["study_status"]:
        raise ProvenanceValidationError("historical and compatibility pi0-FAST identities are not separated")
    checks.append("historical and compatibility pi0-FAST identities remain distinct")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        checks = validate_checkpoint_provenance(args.root.resolve())
    except ProvenanceValidationError as exc:
        print(f"V3 checkpoint provenance validation failed: {exc}")
        return 1
    if not args.quiet:
        print(f"V3 checkpoint provenance validation passed: {len(checks)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
