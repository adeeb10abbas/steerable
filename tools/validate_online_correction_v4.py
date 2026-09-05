#!/usr/bin/env python3
"""Fail-closed validator for prospective V4 design/freeze artifacts.

Checks historical protocol integrity, deterministic queue regeneration, prompt
invariants, control reuse, seed-collision audit, block reasons, and absence of
launch-critical TODOs in any artifact that claims release.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "artifacts/online_correction_v4"
DEFAULT_CONFIG = ROOT / "docs/online_correction_v4/campaign.json"

SPEC = importlib.util.spec_from_file_location("online_correction_v4", ROOT / "tools/online_correction_v4.py")
v4 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(v4)

BUILD_SPEC = importlib.util.spec_from_file_location(
    "build_online_correction_v4_freeze", ROOT / "tools/build_online_correction_v4_freeze.py"
)
builder = importlib.util.module_from_spec(BUILD_SPEC)
assert BUILD_SPEC.loader is not None
BUILD_SPEC.loader.exec_module(builder)

TODO_RE = re.compile(r"(^|[^A-Za-z])(TODO|TBD|PLACEHOLDER|UNQUALIFIED)([^A-Za-z]|$)", re.I)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DUPLICATE_ARTICLE_RE = re.compile(r"\bthe the\b", re.I)

GENERATION_PARENT_COMMIT_KEYS = ("generation_parent_commit",)


def load_json(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: JSONL record must be an object")
        rows.append(value)
    return rows


def launch_critical_unresolved(value: Any, path: str = "") -> list[str]:
    errors: list[str] = []
    if value is None:
        errors.append(f"{path}: null in launch-critical artifact")
    elif isinstance(value, str):
        if not value.strip():
            errors.append(f"{path}: empty string")
        elif TODO_RE.search(value):
            errors.append(f"{path}: unresolved launch-critical text")
        elif "{UNRESOLVED" in value:
            errors.append(f"{path}: unresolved symbolic placeholder in launch-critical field")
    elif isinstance(value, dict):
        for key, child in value.items():
            errors.extend(launch_critical_unresolved(child, f"{path}.{key}" if path else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(launch_critical_unresolved(child, f"{path}[{index}]"))
    return errors


def validate_historical_protocols(ledger: dict, errors: list[str]) -> None:
    for entry in ledger.get("entries", []):
        rel = entry.get("path")
        expected = entry.get("sha256")
        if not rel or not HEX64.fullmatch(str(expected or "")):
            errors.append(f"historical ledger entry {entry.get('ledger_id')}: invalid path or sha256")
            continue
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"historical protocol missing: {rel}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(
                f"historical protocol bytes changed for {rel}: ledger={expected} current={actual}"
            )


def validate_queue(
    config: dict,
    config_sha256: str,
    queue_rows: list[dict],
    frozen_queue_sha256: str,
    queue_manifest: dict,
    errors: list[str],
) -> None:
    if queue_manifest.get("row_count") != len(queue_rows):
        errors.append("queue_manifest row_count does not match queue.jsonl")
    if queue_manifest.get("frozen_queue_sha256") != frozen_queue_sha256:
        errors.append("queue_manifest frozen_queue_sha256 does not match queue.jsonl bytes")
    if queue_manifest.get("queue_sha256") != frozen_queue_sha256:
        errors.append("queue_manifest queue_sha256 alias must match frozen_queue_sha256")
    base_rows = v4.build_manifest(config, config_sha256)
    planning_sha = v4.digest_bytes(v4.manifest_bytes(base_rows))
    if queue_manifest.get("planning_manifest_sha256") != planning_sha:
        errors.append("queue_manifest planning_manifest_sha256 does not match pre-enrichment inventory")
    if queue_manifest.get("expected_confirmatory_episodes") != config["expected_confirmatory_episodes"]:
        errors.append("queue_manifest expected episode count mismatch")
    if queue_manifest.get("release_status") == "RELEASED":
        errors.append("queue_manifest claims RELEASED but prospective freeze forbids release")
    if "total_control_reference_edges_by_family" not in queue_manifest:
        errors.append("queue_manifest missing total_control_reference_edges_by_family")
    if "unique_referenced_control_episode_ids_by_family" not in queue_manifest:
        errors.append("queue_manifest missing unique_referenced_control_episode_ids_by_family")
    if "unique_reused_control_references_by_family" in queue_manifest:
        errors.append("queue_manifest must not use deprecated unique_reused_control_references_by_family")

    episode_ids = {row["episode_id"] for row in queue_rows}
    if len(episode_ids) != len(queue_rows):
        errors.append("queue.jsonl episode_id values are not unique")

    base_rows_stripped = [{k: v for k, v in row.items() if k not in (
        "prompt_id", "prompt_text", "prompt_sha256", "prompt_physical_resolution",
        "launch_critical_names_resolved", "queue_row_kind", "reference_color",
        "physical_A_color", "reuse_episode_ids_meaning",
    )} for row in queue_rows]
    inventory_errors = v4.manifest_errors(base_rows_stripped, config, config_sha256)
    errors.extend(inventory_errors)

    c4_fast_new = [
        row for row in queue_rows
        if row["family"] == "C4"
        and row["factors"]["schedule"] == "fast_after_grasp"
        and row["factors"]["scenario"] in ("original_sham", "move_stop")
    ]
    if not c4_fast_new:
        errors.append("expected C4 fast-schedule sham/move rows as registered new episodes")
    for row in c4_fast_new:
        if row.get("queue_row_kind") != "new_episode":
            errors.append(f"{row['episode_id']}: C4 fast-schedule row must remain a new_episode")
        if not row.get("episode_id"):
            errors.append("C4 fast-schedule row missing episode_id")

    for row in queue_rows:
        if row.get("queue_row_kind") != "new_episode":
            errors.append(f"{row['episode_id']}: queue_row_kind must be new_episode (never reuse-only alias)")
        text, meta = builder.resolve_prompt_text(
            config, row["fixture"], row["factors"], row.get("counterbalance")
        )
        if DUPLICATE_ARTICLE_RE.search(row.get("prompt_text", "")):
            errors.append(f"{row['episode_id']}: prompt_text contains duplicate article 'the the'")
        if row.get("prompt_text") != text:
            errors.append(f"{row['episode_id']}: prompt_text drift from symbolic resolver")
        expected_hash = v4.digest_bytes(text.encode("utf-8"))
        if row.get("prompt_sha256") != expected_hash:
            errors.append(f"{row['episode_id']}: prompt_sha256 mismatch")
        expected_prompt_id = v4.digest(builder.prompt_identity(config, row))[:24]
        if row.get("prompt_id") != expected_prompt_id:
            errors.append(f"{row['episode_id']}: prompt_id mismatch")
        if row.get("launch_critical_names_resolved") is not meta["launch_critical_names_resolved"]:
            errors.append(f"{row['episode_id']}: launch_critical_names_resolved mismatch")
        if meta["launch_critical_names_resolved"]:
            errors.extend(launch_critical_unresolved(row.get("prompt_text"), f"{row['episode_id']}.prompt_text"))
        if row["fixture"] == "reference_binding":
            cb = row["counterbalance"]
            color_a = cb["physical_A_color"]
            expected_color = color_a if row["factors"]["named_reference"] == "A" else (
                "yellow" if color_a == "blue" else "blue"
            )
            if meta.get("reference_color") != expected_color:
                errors.append(
                    f"{row['episode_id']}: C2 reference_color {meta.get('reference_color')} != expected {expected_color}"
                )
            if row.get("reference_color") != expected_color:
                errors.append(f"{row['episode_id']}: queue reference_color mismatch vs counterbalance")
            if f"{expected_color} bowl" not in row.get("prompt_text", ""):
                errors.append(f"{row['episode_id']}: C2 prompt missing expected '{expected_color} bowl'")
        for eid in row.get("reuse_episode_ids", []):
            if eid not in episode_ids:
                errors.append(f"{row['episode_id']}: reuse target {eid} missing from queue")
            elif eid == row["episode_id"]:
                errors.append(f"{row['episode_id']}: row must not reuse itself")

    c3_rows = [r for r in queue_rows if r["family"] == "C3"]
    if c3_rows and any(r.get("queue_row_kind") != "new_episode" for r in c3_rows):
        errors.append("C3 rows must all be registered new episodes despite control links")
    if c3_rows and len({r["episode_id"] for r in c3_rows}) != len(c3_rows):
        errors.append("C3 episode IDs must remain unique new episodes")


def strip_generation_parent_commit(payload: dict) -> dict:
    cleaned = json.loads(json.dumps(payload))
    for key in GENERATION_PARENT_COMMIT_KEYS:
        cleaned.pop(key, None)
    return cleaned


def validate_seed_receipt_matches_audit(
    gate_report: dict,
    seed_manifest: dict,
    seed_manifest_sha256: str,
    errors: list[str],
) -> None:
    audit = seed_manifest.get("historical_collision_audit", {})
    receipt = gate_report.get("required_release_receipts", {}).get("historical_seed_collision_audit", {})
    env_collisions = audit.get("env_collisions", [])
    policy_collisions = audit.get("policy_collisions", [])
    computed_passed = len(env_collisions) == 0 and len(policy_collisions) == 0
    audit_passed = audit.get("passed") is True
    if audit_passed != computed_passed:
        errors.append(
            f"seed_manifest audit passed={audit_passed} != computed from collisions passed={computed_passed}"
        )
    expected_passed = computed_passed
    if receipt.get("passed") != expected_passed:
        errors.append(
            f"gate_report seed receipt passed={receipt.get('passed')} != seed audit passed={expected_passed}"
        )
    expected_status = "passed_at_freeze_build" if expected_passed else "failed_at_freeze_build"
    if receipt.get("status") != expected_status:
        errors.append(
            f"gate_report seed receipt status={receipt.get('status')!r} != expected {expected_status!r}"
        )
    if receipt.get("sha256") != seed_manifest_sha256:
        errors.append("gate_report historical_seed_collision_audit sha256 != seed_manifest bytes")
    if receipt.get("derived_from") != "seed_manifest.historical_collision_audit":
        errors.append("historical_seed_collision_audit receipt must declare derived_from seed audit")
    summary = receipt.get("audit_summary", {})
    if summary.get("env_collision_count") != len(env_collisions):
        errors.append("seed receipt env_collision_count mismatch vs seed_manifest audit")
    if summary.get("policy_collision_count") != len(policy_collisions):
        errors.append("seed receipt policy_collision_count mismatch vs seed_manifest audit")
    if summary.get("files_scanned") != audit.get("files_scanned"):
        errors.append("seed receipt files_scanned mismatch vs seed_manifest audit")


def validate_seed_manifest_vs_queue(
    seed_manifest: dict,
    queue_rows: list[dict],
    errors: list[str],
) -> None:
    by_episode = {row["episode_id"]: row for row in seed_manifest.get("confirmatory_rows", [])}
    if len(by_episode) != len(queue_rows):
        errors.append(
            f"seed_manifest confirmatory_rows count {len(by_episode)} != queue rows {len(queue_rows)}"
        )
    for row in queue_rows:
        entry = by_episode.get(row["episode_id"])
        if entry is None:
            errors.append(f"{row['episode_id']}: missing from seed_manifest confirmatory_rows")
            continue
        if entry.get("env_seed") != row.get("env_seed"):
            errors.append(f"{row['episode_id']}: seed_manifest env_seed mismatch vs queue")
        if entry.get("policy_seed") != row.get("policy_seed"):
            errors.append(f"{row['episode_id']}: seed_manifest policy_seed mismatch vs queue")
        if entry.get("fixture") != row.get("fixture"):
            errors.append(f"{row['episode_id']}: seed_manifest fixture mismatch vs queue")
        if entry.get("policy") != row["factors"]["policy"]:
            errors.append(f"{row['episode_id']}: seed_manifest policy mismatch vs queue")


def validate_prompt_sha256_semantics(
    prompt_manifest: dict,
    queue_rows: list[dict],
    errors: list[str],
) -> None:
    semantics = prompt_manifest.get("prompt_identity_semantics", {})
    if semantics.get("prompt_sha256_rule") != "sha256(utf8(prompt_text)); identical iff byte-identical resolved text":
        errors.append("prompt_manifest must declare prompt_sha256 byte-identity rule")
    if semantics.get("prompt_sha256_must_be_unique_when_text_differs") is not True:
        errors.append("prompt_manifest must require unique prompt_sha256 when resolved text differs")

    def _check_entries(entries: list[dict], label: str) -> None:
        text_to_sha: dict[str, str] = {}
        sha_to_texts: dict[str, set[str]] = defaultdict(set)
        for entry in entries:
            text = entry.get("prompt_text", "")
            sha = entry.get("prompt_sha256")
            expected = v4.digest_bytes(text.encode("utf-8"))
            if sha != expected:
                errors.append(f"{label} {entry.get('prompt_id', '?')}: prompt_sha256 mismatch vs utf8 text")
            prior_sha = text_to_sha.get(text)
            if prior_sha is not None and prior_sha != sha:
                errors.append(f"{label}: identical prompt_text maps to conflicting prompt_sha256 values")
            text_to_sha[text] = sha
            sha_to_texts[sha].add(text)
        for sha, texts in sha_to_texts.items():
            if len(texts) > 1:
                errors.append(
                    f"{label}: prompt_sha256 collision — one hash maps to {len(texts)} distinct prompt_text values"
                )

    _check_entries(prompt_manifest.get("prompts", []), "prompt_manifest")
    _check_entries(queue_rows, "queue")


def validate_deterministic_artifact_coverage(errors: list[str]) -> None:
    covered = set(builder.DETERMINISTIC_ARTIFACT_NAMES) | set(builder.GENERATION_PARENT_COMMIT_ARTIFACTS)
    for name in builder.ALL_GENERATED_FREEZE_ARTIFACTS:
        if name not in covered:
            errors.append(f"freeze artifact {name} has no deterministic regeneration policy (fail closed)")
    index_names = set(builder.ALL_GENERATED_FREEZE_ARTIFACTS) - {builder.FREEZE_MANIFEST_SELF_HASH_EXCLUDED}
    if index_names - covered:
        errors.append(
            f"freeze index artifacts missing regeneration policy: {sorted(index_names - covered)}"
        )


def validate_runtime_manifest_vs_campaign(runtime_manifest: dict, config: dict, errors: list[str]) -> None:
    if runtime_manifest.get("campaign_id") != config.get("campaign_id"):
        errors.append("runtime_manifest campaign_id != campaign.json")
    manifest_policies = set(runtime_manifest.get("policies", {}))
    campaign_policies = set(config.get("policies", {}))
    if manifest_policies != campaign_policies:
        errors.append(
            f"runtime_manifest policy keys {sorted(manifest_policies)} != campaign {sorted(campaign_policies)}"
        )
    for policy_id, spec in runtime_manifest.get("policies", {}).items():
        for field in (
            "checkpoint_uri", "checkpoint_sha256", "runtime_image_digest",
            "integration_commit", "native_control_dt_s",
        ):
            if spec.get(field) is not None:
                errors.append(f"runtime_manifest {policy_id}.{field} must remain null before release")


def validate_launch_matrix_vs_campaign(launch_matrix: dict, config: dict, errors: list[str]) -> None:
    if launch_matrix.get("campaign_id") != config.get("campaign_id"):
        errors.append("launch_matrix campaign_id != campaign.json")
    for field in ("cluster_context", "namespace", "lane_bundle_identity", "resource_budget"):
        if launch_matrix.get(field) is not None:
            errors.append(f"launch_matrix.{field} must remain null before release")


def validate_setup_manifest_vs_campaign(setup_manifest: dict, config: dict, errors: list[str]) -> None:
    if setup_manifest.get("campaign_id") != config.get("campaign_id"):
        errors.append("setup_manifest campaign_id != campaign.json")
    setup_fixtures = set(setup_manifest.get("fixtures", {}))
    campaign_fixtures = set(config.get("fixtures", {}))
    if setup_fixtures != campaign_fixtures:
        errors.append(
            f"setup_manifest fixture keys {sorted(setup_fixtures)} != campaign fixtures {sorted(campaign_fixtures)}"
        )
    for fixture_id, spec in setup_manifest.get("fixtures", {}).items():
        for field in ("geometry_uri", "geometry_sha256", "frame_transform_uri", "reset_registry_uri"):
            if spec.get(field) is not None:
                errors.append(f"setup_manifest {fixture_id}.{field} must remain null before release")
        if spec.get("calibration_scale") is not None or spec.get("D_cap_m") is not None:
            errors.append(f"setup_manifest {fixture_id} geometry calibration fields must remain null before release")


def validate_family_dispositions_cross_artifact(
    config: dict,
    protocol: dict,
    gate_report: dict,
    continuation: dict,
    errors: list[str],
) -> None:
    campaign_family_ids = {f["id"] for f in config.get("families", [])}
    protocol_family_ids = {f["id"] for f in protocol.get("families", [])}
    gate_family_ids = set(gate_report.get("families", {}))
    hard_blocked = set(gate_report.get("hard_blocked_families", {}))
    pending = set(gate_report.get("pending_not_released_families", {}))
    cont_hard = set(continuation.get("hard_blocked_families", {}))
    cont_pending = set(continuation.get("pending_not_released_families", {}))

    if protocol_family_ids != campaign_family_ids:
        errors.append("protocol.json family ids != campaign.json")
    if gate_family_ids != campaign_family_ids:
        errors.append("gate_report family ids != campaign.json")
    if hard_blocked | pending != campaign_family_ids:
        errors.append("gate_report hard_blocked + pending must partition all campaign families")
    if hard_blocked & pending:
        errors.append("gate_report family appears in both hard_blocked and pending")
    if cont_hard != hard_blocked:
        errors.append("continuation_state hard_blocked_families != gate_report")
    if cont_pending != pending:
        errors.append("continuation_state pending_not_released_families != gate_report")
    for fid in campaign_family_ids:
        gate = gate_report.get("families", {}).get(fid, {})
        expected = builder.family_gate_status(fid)
        if gate.get("disposition") != expected["disposition"]:
            errors.append(f"gate_report {fid} disposition != builder.family_gate_status")
        if gate.get("lifecycle_status") != expected["lifecycle_status"]:
            errors.append(f"gate_report {fid} lifecycle_status != builder.family_gate_status")
        if gate.get("release_state") != "NOT_RELEASED":
            errors.append(f"gate_report {fid} must remain NOT_RELEASED")
        cfg = next(f for f in config["families"] if f["id"] == fid)
        if gate.get("expected_new_episodes") != cfg["expected_new_episodes"]:
            errors.append(f"gate_report {fid} expected_new_episodes != campaign")


def validate_stub_artifacts_in_freeze_index(
    freeze_manifest: dict,
    continuation: dict,
    errors: list[str],
) -> None:
    for name in ("runtime_manifest.json", "setup_manifest.json", "launch_matrix.json"):
        if name not in freeze_manifest.get("artifact_sha256", {}):
            errors.append(f"freeze_manifest missing hash for required stub {name}")
        if not _authoritative_includes(continuation.get("authoritative_files", []), name):
            errors.append(f"continuation_state authoritative_files must include {name}")



def _authoritative_includes(authoritative_files: list[str], basename: str) -> bool:
    return any(entry == basename or entry.endswith(f"/{basename}") for entry in authoritative_files)


def validate_continuation_artifact_hashes(
    continuation: dict,
    freeze_manifest: dict,
    artifact_dir: Path,
    errors: list[str],
) -> None:
    cont_hashes = continuation.get("artifact_sha256", {})
    freeze_hashes = freeze_manifest.get("artifact_sha256", {})
    if not cont_hashes:
        errors.append("continuation_state missing artifact_sha256")
        return
    shared = set(cont_hashes) & set(freeze_hashes)
    for name in shared:
        if cont_hashes[name] != freeze_hashes[name]:
            errors.append(f"continuation_state artifact_sha256 mismatch for {name} vs freeze_manifest")
    for name, expected in cont_hashes.items():
        path = artifact_dir / name
        if not path.is_file():
            errors.append(f"continuation_state references missing artifact {name}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"continuation_state artifact_sha256 stale for {name}")
    if not _authoritative_includes(continuation.get("authoritative_files", []), "freeze_manifest.json"):
        errors.append("continuation_state authoritative_files must include freeze_manifest.json")
    if not _authoritative_includes(continuation.get("authoritative_files", []), "continuation_state.json"):
        errors.append("continuation_state authoritative_files must include continuation_state.json")


def validate_prompt_identity_semantics(prompt_manifest: dict, frozen_analysis: dict, errors: list[str]) -> None:
    semantics = prompt_manifest.get("prompt_identity_semantics", {})
    if semantics.get("primary_key") != "prompt_id":
        errors.append("prompt_manifest primary semantic key must be prompt_id")
    if semantics.get("prompt_sha256_may_map_to_multiple_prompt_ids") is not True:
        errors.append("prompt_manifest must allow prompt_sha256 to map to multiple prompt_ids")
    if "prompt_sha256" not in semantics.get("analysis_forbidden_primary_keys", []):
        errors.append("prompt_manifest must forbid analysis keyed on prompt_sha256 alone")
    binding = frozen_analysis.get("semantic_prompt_binding", {})
    if binding.get("primary_key") != "prompt_id":
        errors.append("frozen_analysis_manifest semantic primary key must be prompt_id")
    if "prompt_sha256" not in binding.get("forbidden_semantic_primary_keys", []):
        errors.append("frozen_analysis_manifest must forbid prompt_sha256 as semantic primary key")


def validate_deterministic_rebuild(
    config_path: Path,
    artifact_dir: Path,
    expected_hashes: dict[str, str],
    expected_normalized: dict[str, dict],
    errors: list[str],
) -> None:
    try:
        frozen_protocol = json.loads((artifact_dir / "protocol.json").read_text())
        frozen_parent = frozen_protocol.get("generation_parent_commit")
        if not isinstance(frozen_parent, str) or not frozen_parent:
            errors.append("protocol generation_parent_commit is required for deterministic rebuild")
            return
        with tempfile.TemporaryDirectory() as tmp:
            rebuilt = builder.build_freeze(
                config_path,
                Path(tmp),
                generation_parent_commit=frozen_parent,
            )
            rebuilt_hashes = rebuilt["artifact_sha256"]
            for name in builder.DETERMINISTIC_ARTIFACT_NAMES:
                expected = expected_hashes.get(name)
                actual = rebuilt_hashes.get(name)
                if expected != actual:
                    errors.append(f"deterministic rebuild hash mismatch for {name}")
            for name in builder.GENERATION_PARENT_COMMIT_ARTIFACTS:
                if name not in expected_normalized:
                    continue
                rebuilt_payload = json.loads((Path(tmp) / name).read_text())
                if strip_generation_parent_commit(rebuilt_payload) != expected_normalized[name]:
                    errors.append(
                        f"deterministic rebuild normalized content mismatch for {name} "
                        "(excluding generation_parent_commit)"
                    )
            freeze_path = Path(tmp) / builder.FREEZE_MANIFEST_SELF_HASH_EXCLUDED
            if freeze_path.is_file():
                outer = json.loads(freeze_path.read_text())
                inner = outer.get("artifact_sha256", {})
                if builder.FREEZE_MANIFEST_SELF_HASH_EXCLUDED in inner:
                    errors.append("freeze_manifest must not embed its own hash in artifact_sha256")
    except ValueError as exc:
        errors.append(f"deterministic rebuild failed: {exc}")


def validate_prompt_manifest(prompt_manifest: dict, queue_rows: list[dict], errors: list[str]) -> None:
    by_id = {item["prompt_id"]: item for item in prompt_manifest.get("prompts", [])}
    if len(by_id) != prompt_manifest.get("unique_prompt_count"):
        errors.append("prompt_manifest unique_prompt_count mismatch")
    for row in queue_rows:
        item = by_id.get(row["prompt_id"])
        if item is None:
            errors.append(f"{row['episode_id']}: prompt_id {row['prompt_id']} missing from prompt_manifest")
            continue
        if item["prompt_text"] != row["prompt_text"] or item["prompt_sha256"] != row["prompt_sha256"]:
            errors.append(f"{row['episode_id']}: prompt_manifest entry differs from queue row")
    if "second_stack" not in prompt_manifest.get("unresolved_physical_name_fixtures", []):
        errors.append("prompt_manifest must list second_stack as unresolved physical names")


def validate_seed_manifest(seed_manifest: dict, errors: list[str]) -> None:
    audit = seed_manifest.get("historical_collision_audit", {})
    if audit.get("required") and audit.get("passed") is not True:
        errors.append("seed_manifest historical collision audit failed or missing")
    if audit.get("collisions"):
        errors.append(f"seed_manifest reports {len(audit['collisions'])} historical seed collisions")
    if "method" not in audit or "limitations" not in audit:
        errors.append("seed_manifest collision audit must disclose method and limitations")
    if "policy_collisions" not in audit:
        errors.append("seed_manifest collision audit must include policy_collisions")
    confirmatory = seed_manifest.get("confirmatory_rows", [])
    block_env = {(row["fixture"], row["block_id"]): row["env_seed"] for row in confirmatory}
    for row in confirmatory:
        key = (row["fixture"], row["block_id"])
        if block_env.get(key) != row["env_seed"]:
            errors.append(
                f"seed_manifest: inconsistent env_seed for fixture={row['fixture']} block={row['block_id']}"
            )
    reserved = seed_manifest.get("confirmatory_unique_env_seeds", [])
    if len(reserved) != len(block_env):
        errors.append("seed_manifest confirmatory_unique_env_seeds count mismatch")
    policy_seeds = seed_manifest.get("confirmatory_unique_policy_seeds", [])
    if policy_seeds and len(policy_seeds) != len(set(policy_seeds)):
        errors.append("seed_manifest confirmatory_unique_policy_seeds must be unique")


def validate_gate_and_blocks(gate_report: dict, frozen_analysis: dict, errors: list[str]) -> None:
    if gate_report.get("release_status") == "RELEASED":
        errors.append("gate_report claims RELEASED")
    hard_blocked = gate_report.get("hard_blocked_families", {})
    pending = gate_report.get("pending_not_released_families", {})
    if set(hard_blocked) != {"C2", "C8"}:
        errors.append("hard_blocked_families must contain exactly C2 and C8")
    if set(pending) != {"C1", "C3", "C4", "C5", "C6", "C7"}:
        errors.append("pending_not_released_families must contain C1 and C3-C7")
    families = gate_report.get("families", {})
    c8 = families.get("C8", {})
    if c8.get("lifecycle_status") != "BLOCKED_RUNTIME":
        errors.append("C8 must be BLOCKED_RUNTIME")
    if c8.get("disposition") != "hard_blocked":
        errors.append("C8 disposition must be hard_blocked")
    if "GR00T" not in c8.get("block_reason", "") and "Bridge" not in c8.get("block_reason", ""):
        errors.append("C8 block_reason must cite second-stack runtime verification gap")
    c2 = families.get("C2", {})
    if c2.get("lifecycle_status") != "BLOCKED_SETUP":
        errors.append("C2 must be BLOCKED_SETUP")
    if c2.get("disposition") != "hard_blocked":
        errors.append("C2 disposition must be hard_blocked")
    if "common-prefix" not in c2.get("block_reason", "").lower() and "common prefix" not in c2.get("block_reason", "").lower():
        errors.append("C2 block_reason must cite verified common-prefix replay requirement")
    if c2.get("qualification_gates", {}).get("G5_trigger_and_branch_replay") != "blocked":
        errors.append("C2 G5_trigger_and_branch_replay must be blocked")
    c1 = families.get("C1", {})
    if c1.get("disposition") != "pending_qualification":
        errors.append("C1 disposition must be pending_qualification not hard_blocked")
    if frozen_analysis.get("C2_prefix_requirement", {}).get("status") != "blocked_pending_verification":
        errors.append("frozen_analysis_manifest C2 prefix requirement must remain blocked")
    for fid, reason in {**hard_blocked, **pending}.items():
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"family {fid} missing reason")
    receipts = gate_report.get("required_release_receipts", {})
    for name, receipt in receipts.items():
        if name == "historical_seed_collision_audit":
            if receipt.get("passed") is not True:
                errors.append("historical_seed_collision_audit receipt must pass at freeze")
            continue
        if receipt.get("passed") is True:
            errors.append(f"receipt {name}: must not fake-pass before runtime evidence exists")
        if receipt.get("uri") not in (None, "") and receipt.get("passed") is not True:
            continue
        if receipt.get("passed") is False and receipt.get("status") not in ("pending", "blocked"):
            errors.append(f"receipt {name}: unexpected non-pending status")


def validate_not_released_artifacts(artifact_dir: Path, protocol: dict, motion: dict, scoring: dict, errors: list[str]) -> None:
    if protocol.get("status") != "PROSPECTIVE_FROZEN_DESIGN_NOT_RELEASED":
        errors.append("protocol.json must remain PROSPECTIVE_FROZEN_DESIGN_NOT_RELEASED")
    for fixture, scale in motion.get("calibration", {}).get("selected_scale_by_fixture", {}).items():
        if scale is not None:
            errors.append(f"motion_manifest fixture {fixture}: calibration scale must remain null before geometry gate")
    for fixture, cap in scoring.get("D_cap_m_by_fixture", {}).items():
        if cap is not None:
            errors.append(f"scoring_manifest fixture {fixture}: D_cap_m must remain null before geometry receipt")
    runtime_lock = ROOT / "docs/online_correction_v4/runtime_lock.template.json"
    if runtime_lock.is_file():
        lock = json.loads(runtime_lock.read_text())
        if lock.get("release_status") != "NOT_RELEASED":
            errors.append("runtime_lock.template.json must remain NOT_RELEASED")


def validate_unreleased_stubs(artifact_dir: Path, errors: list[str]) -> None:
    for name in ("runtime_manifest.json", "setup_manifest.json", "launch_matrix.json"):
        path = artifact_dir / name
        if not path.is_file():
            errors.append(f"missing required stub artifact: {name}")
            continue
        payload = json.loads(path.read_text())
        if payload.get("release_status") != "NOT_RELEASED":
            errors.append(f"{name} must remain NOT_RELEASED")
        if name == "runtime_manifest.json" and payload.get("runner", {}).get("entrypoint") is not None:
            errors.append("runtime_manifest runner entrypoint must remain null before release")
        if name == "launch_matrix.json" and payload.get("qualified_lanes") != []:
            errors.append("launch_matrix qualified_lanes must remain empty before release")


def validate_freeze_manifest(freeze_manifest: dict, artifact_dir: Path, errors: list[str]) -> None:
    recorded = freeze_manifest.get("artifact_sha256", {})
    expected_in_index = set(builder.ALL_GENERATED_FREEZE_ARTIFACTS) - {builder.FREEZE_MANIFEST_SELF_HASH_EXCLUDED}
    if set(recorded) != expected_in_index:
        missing = sorted(expected_in_index - set(recorded))
        extra = sorted(set(recorded) - expected_in_index)
        if missing:
            errors.append(f"freeze_manifest missing artifact hashes: {missing}")
        if extra:
            errors.append(f"freeze_manifest unexpected artifact hashes: {extra}")
    if builder.FREEZE_MANIFEST_SELF_HASH_EXCLUDED in recorded:
        errors.append("freeze_manifest must not embed its own hash in artifact_sha256")
    for name, expected in recorded.items():
        path = artifact_dir / name
        if not path.is_file():
            errors.append(f"freeze_manifest missing artifact: {name}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"artifact hash mismatch for {name}")
    freeze_path = artifact_dir / builder.FREEZE_MANIFEST_SELF_HASH_EXCLUDED
    if freeze_path.is_file():
        outer_hash = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
        cont_path = artifact_dir / "continuation_state.json"
        if cont_path.is_file():
            continuation = json.loads(cont_path.read_text())
            indexed = continuation.get("artifact_sha256", {}).get(builder.FREEZE_MANIFEST_SELF_HASH_EXCLUDED)
            if indexed and indexed != outer_hash:
                errors.append("continuation_state freeze_manifest.json hash != on-disk bytes")


def validate_online_correction_v4(
    artifact_dir: Path = DEFAULT_DIR,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    errors: list[str] = []
    config, config_sha256 = v4.load_json(config_path)
    errors.extend(v4.config_errors(config))

    required_files = list(builder.ALL_GENERATED_FREEZE_ARTIFACTS)
    for name in required_files:
        if not (artifact_dir / name).is_file():
            errors.append(f"missing required artifact: {name}")

    validate_deterministic_artifact_coverage(errors)

    if errors:
        return {"ok": False, "errors": errors}

    ledger, _ = load_json(artifact_dir / "historical_protocol_ledger.json")
    protocol, protocol_sha = load_json(artifact_dir / "protocol.json")
    prompt_manifest, _ = load_json(artifact_dir / "prompt_manifest.json")
    motion_manifest, _ = load_json(artifact_dir / "motion_manifest.json")
    scoring_manifest, _ = load_json(artifact_dir / "scoring_manifest.json")
    seed_manifest, seed_manifest_sha256 = load_json(artifact_dir / "seed_manifest.json")
    queue_manifest, _ = load_json(artifact_dir / "queue_manifest.json")
    frozen_analysis, _ = load_json(artifact_dir / "frozen_analysis_manifest.json")
    gate_report, _ = load_json(artifact_dir / "gate_report.json")
    continuation, _ = load_json(artifact_dir / "continuation_state.json")
    freeze_manifest, _ = load_json(artifact_dir / "freeze_manifest.json")
    setup_manifest, _ = load_json(artifact_dir / "setup_manifest.json")
    runtime_manifest, _ = load_json(artifact_dir / "runtime_manifest.json")
    launch_matrix, _ = load_json(artifact_dir / "launch_matrix.json")

    queue_bytes = (artifact_dir / "queue.jsonl").read_bytes()
    frozen_queue_sha256 = hashlib.sha256(queue_bytes).hexdigest()
    queue_rows = read_jsonl(artifact_dir / "queue.jsonl")

    validate_historical_protocols(ledger, errors)
    validate_queue(config, config_sha256, queue_rows, frozen_queue_sha256, queue_manifest, errors)
    validate_prompt_manifest(prompt_manifest, queue_rows, errors)
    validate_prompt_identity_semantics(prompt_manifest, frozen_analysis, errors)
    validate_prompt_sha256_semantics(prompt_manifest, queue_rows, errors)
    validate_seed_manifest(seed_manifest, errors)
    validate_seed_manifest_vs_queue(seed_manifest, queue_rows, errors)
    validate_seed_receipt_matches_audit(gate_report, seed_manifest, seed_manifest_sha256, errors)
    validate_setup_manifest_vs_campaign(setup_manifest, config, errors)
    validate_runtime_manifest_vs_campaign(runtime_manifest, config, errors)
    validate_launch_matrix_vs_campaign(launch_matrix, config, errors)
    validate_family_dispositions_cross_artifact(config, protocol, gate_report, continuation, errors)
    validate_stub_artifacts_in_freeze_index(freeze_manifest, continuation, errors)
    validate_gate_and_blocks(gate_report, frozen_analysis, errors)
    validate_not_released_artifacts(artifact_dir, protocol, motion_manifest, scoring_manifest, errors)
    validate_unreleased_stubs(artifact_dir, errors)
    validate_freeze_manifest(freeze_manifest, artifact_dir, errors)
    validate_continuation_artifact_hashes(continuation, freeze_manifest, artifact_dir, errors)

    expected_normalized: dict[str, dict] = {}
    for name in builder.GENERATION_PARENT_COMMIT_ARTIFACTS:
        payload, _ = load_json(artifact_dir / name)
        expected_normalized[name] = strip_generation_parent_commit(payload)
    validate_deterministic_rebuild(
        config_path,
        artifact_dir,
        freeze_manifest.get("artifact_sha256", {}),
        expected_normalized,
        errors,
    )

    if protocol.get("config_sha256") != config_sha256:
        errors.append("protocol.config_sha256 does not match campaign.json")
    if protocol.get("frozen_queue_sha256") != frozen_queue_sha256:
        errors.append("protocol.frozen_queue_sha256 does not match queue.jsonl")
    if protocol.get("planning_manifest_sha256") != queue_manifest.get("planning_manifest_sha256"):
        errors.append("protocol planning_manifest_sha256 mismatch vs queue_manifest")
    if protocol.get("freeze_commit"):
        errors.append("protocol must not use deprecated freeze_commit; use generation_parent_commit")
    if "generation_parent_commit" not in protocol:
        errors.append("protocol must record generation_parent_commit")
    if continuation.get("release_status") == "RELEASED":
        errors.append("continuation_state claims RELEASED")
    if continuation.get("status") not in ("IMPLEMENTING", "QUALIFYING"):
        errors.append("continuation_state.status must be IMPLEMENTING or QUALIFYING")
    if continuation.get("policy_episodes_executed", 0) != 0:
        errors.append("continuation_state must record zero executed policy episodes at design freeze")
    if continuation.get("freeze_commit"):
        errors.append("continuation_state must not use deprecated freeze_commit")
    if freeze_manifest.get("build_commit"):
        errors.append("freeze_manifest must not use deprecated build_commit")

    if gate_report.get("runtime_lock_status") != "template_only_not_released":
        errors.append("gate_report runtime_lock_status must remain template_only_not_released")

    duplicate_article_count = sum(
        1 for row in queue_rows if DUPLICATE_ARTICLE_RE.search(row.get("prompt_text", ""))
    )
    c2_color_mismatches = sum(
        1
        for row in queue_rows
        if row["fixture"] == "reference_binding"
        and row.get("reference_color")
        != (
            row["counterbalance"]["physical_A_color"]
            if row["factors"]["named_reference"] == "A"
            else ("yellow" if row["counterbalance"]["physical_A_color"] == "blue" else "blue")
        )
    )

    return {
        "ok": not errors,
        "validation_scope": "prospective_design_freeze_only",
        "artifact_dir": str(artifact_dir),
        "planning_manifest_sha256": protocol.get("planning_manifest_sha256"),
        "frozen_queue_sha256": frozen_queue_sha256,
        "queue_sha256": frozen_queue_sha256,
        "row_count": len(queue_rows),
        "duplicate_article_prompt_count": duplicate_article_count,
        "c2_color_mismatch_count": c2_color_mismatches,
        "release_status": gate_report.get("release_status"),
        "continuation_status": continuation.get("status"),
        "seed_collision_audit_passed": seed_manifest.get("historical_collision_audit", {}).get("passed"),
        "errors": errors,
        "limitations": [
            "Does not verify remote checkpoint, geometry, or cluster receipt contents.",
            "Does not authorize policy inference or family release.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    report = validate_online_correction_v4(args.artifact_dir, args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
