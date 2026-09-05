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
    queue_sha256: str,
    queue_manifest: dict,
    errors: list[str],
) -> None:
    if queue_manifest.get("row_count") != len(queue_rows):
        errors.append("queue_manifest row_count does not match queue.jsonl")
    if queue_manifest.get("queue_sha256") != queue_sha256:
        errors.append("queue_manifest queue_sha256 does not match queue.jsonl bytes")
    if queue_manifest.get("expected_confirmatory_episodes") != config["expected_confirmatory_episodes"]:
        errors.append("queue_manifest expected episode count mismatch")
    if queue_manifest.get("release_status") == "RELEASED":
        errors.append("queue_manifest claims RELEASED but prospective freeze forbids release")
    base_rows = [{k: v for k, v in row.items() if k not in (
        "prompt_id", "prompt_text", "prompt_sha256", "prompt_physical_resolution",
        "launch_critical_names_resolved", "queue_row_kind",
    )} for row in queue_rows]
    inventory_errors = v4.manifest_errors(base_rows, config, config_sha256)
    errors.extend(inventory_errors)
    for row in queue_rows:
        text, meta = builder.resolve_prompt_text(config, row["fixture"], row["factors"])
        if row.get("prompt_text") != text:
            errors.append(f"{row['episode_id']}: prompt_text drift from symbolic resolver")
        expected_hash = v4.digest_bytes(text.encode("utf-8"))
        if row.get("prompt_sha256") != expected_hash:
            errors.append(f"{row['episode_id']}: prompt_sha256 mismatch")
        if row.get("launch_critical_names_resolved") is not meta["launch_critical_names_resolved"]:
            errors.append(f"{row['episode_id']}: launch_critical_names_resolved mismatch")
        if meta["launch_critical_names_resolved"]:
            errors.extend(launch_critical_unresolved(row.get("prompt_text"), f"{row['episode_id']}.prompt_text"))
        for eid in row.get("reuse_episode_ids", []):
            if eid not in {r["episode_id"] for r in queue_rows}:
                errors.append(f"{row['episode_id']}: reuse target {eid} missing from queue")


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


def validate_gate_and_blocks(gate_report: dict, frozen_analysis: dict, errors: list[str]) -> None:
    if gate_report.get("release_status") == "RELEASED":
        errors.append("gate_report claims RELEASED")
    blocked = gate_report.get("blocked_families", {})
    families = gate_report.get("families", {})
    c8 = families.get("C8", {})
    if c8.get("lifecycle_status") != "BLOCKED_RUNTIME":
        errors.append("C8 must be BLOCKED_RUNTIME")
    if "GR00T" not in c8.get("block_reason", "") and "Bridge" not in c8.get("block_reason", ""):
        errors.append("C8 block_reason must cite second-stack runtime verification gap")
    c2 = families.get("C2", {})
    if "common-prefix" not in c2.get("block_reason", "").lower() and "common prefix" not in c2.get("block_reason", "").lower():
        errors.append("C2 block_reason must cite verified common-prefix replay requirement")
    if c2.get("qualification_gates", {}).get("G5_trigger_and_branch_replay") != "blocked":
        errors.append("C2 G5_trigger_and_branch_replay must be blocked")
    if frozen_analysis.get("C2_prefix_requirement", {}).get("status") != "blocked_pending_verification":
        errors.append("frozen_analysis_manifest C2 prefix requirement must remain blocked")
    for fid, reason in blocked.items():
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"blocked family {fid} missing reason")
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


def validate_freeze_manifest(freeze_manifest: dict, artifact_dir: Path, errors: list[str]) -> None:
    recorded = freeze_manifest.get("artifact_sha256", {})
    for name, expected in recorded.items():
        path = artifact_dir / name
        if not path.is_file():
            errors.append(f"freeze_manifest missing artifact: {name}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"artifact hash mismatch for {name}")


def validate_online_correction_v4(
    artifact_dir: Path = DEFAULT_DIR,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    errors: list[str] = []
    config, config_sha256 = v4.load_json(config_path)
    errors.extend(v4.config_errors(config))

    required_files = [
        "historical_protocol_ledger.json",
        "protocol.json",
        "prompt_manifest.json",
        "motion_manifest.json",
        "scoring_manifest.json",
        "seed_manifest.json",
        "queue.jsonl",
        "queue_manifest.json",
        "frozen_analysis_manifest.json",
        "gate_report.json",
        "continuation_state.json",
        "freeze_manifest.json",
    ]
    for name in required_files:
        if not (artifact_dir / name).is_file():
            errors.append(f"missing required artifact: {name}")

    if errors:
        return {"ok": False, "errors": errors}

    ledger, _ = load_json(artifact_dir / "historical_protocol_ledger.json")
    protocol, protocol_sha = load_json(artifact_dir / "protocol.json")
    prompt_manifest, _ = load_json(artifact_dir / "prompt_manifest.json")
    motion_manifest, _ = load_json(artifact_dir / "motion_manifest.json")
    scoring_manifest, _ = load_json(artifact_dir / "scoring_manifest.json")
    seed_manifest, _ = load_json(artifact_dir / "seed_manifest.json")
    queue_manifest, _ = load_json(artifact_dir / "queue_manifest.json")
    frozen_analysis, _ = load_json(artifact_dir / "frozen_analysis_manifest.json")
    gate_report, _ = load_json(artifact_dir / "gate_report.json")
    continuation, _ = load_json(artifact_dir / "continuation_state.json")
    freeze_manifest, _ = load_json(artifact_dir / "freeze_manifest.json")

    queue_bytes = (artifact_dir / "queue.jsonl").read_bytes()
    queue_sha256 = hashlib.sha256(queue_bytes).hexdigest()
    queue_rows = read_jsonl(artifact_dir / "queue.jsonl")

    validate_historical_protocols(ledger, errors)
    validate_queue(config, config_sha256, queue_rows, queue_sha256, queue_manifest, errors)
    validate_prompt_manifest(prompt_manifest, queue_rows, errors)
    validate_seed_manifest(seed_manifest, errors)
    validate_gate_and_blocks(gate_report, frozen_analysis, errors)
    validate_not_released_artifacts(artifact_dir, protocol, motion_manifest, scoring_manifest, errors)
    validate_freeze_manifest(freeze_manifest, artifact_dir, errors)

    if protocol.get("config_sha256") != config_sha256:
        errors.append("protocol.config_sha256 does not match campaign.json")
    if protocol.get("queue_sha256") != queue_sha256:
        errors.append("protocol.queue_sha256 does not match queue.jsonl")
    if continuation.get("release_status") == "RELEASED":
        errors.append("continuation_state claims RELEASED")
    if continuation.get("status") not in ("IMPLEMENTING", "QUALIFYING"):
        errors.append("continuation_state.status must be IMPLEMENTING or QUALIFYING")
    if continuation.get("policy_episodes_executed", 0) != 0:
        errors.append("continuation_state must record zero executed policy episodes at design freeze")

    # Deterministic regeneration check (isolated temp dir; never overwrite committed freeze)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rebuilt = builder.build_freeze(config_path, Path(tmp))
            if rebuilt["queue_sha256"] != queue_sha256:
                errors.append("deterministic rebuild would change queue_sha256")
            if rebuilt["artifact_sha256"].get("queue_manifest.json") != freeze_manifest.get(
                "artifact_sha256", {}
            ).get("queue_manifest.json"):
                errors.append("deterministic rebuild would change queue_manifest hash")
    except ValueError as exc:
        errors.append(f"deterministic rebuild failed: {exc}")

    return {
        "ok": not errors,
        "validation_scope": "prospective_design_freeze_only",
        "artifact_dir": str(artifact_dir),
        "queue_sha256": queue_sha256,
        "row_count": len(queue_rows),
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
