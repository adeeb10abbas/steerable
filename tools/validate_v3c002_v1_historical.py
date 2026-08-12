#!/usr/bin/env python3
"""Audit the immutable, unexecuted V3-C002 V1 draft after V2 supersession."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound(record: object, label: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{label} binding is missing")
    path = Path(str(record.get("path", "")))
    path = path if path.is_absolute() else REPO_ROOT / path
    if not path.is_file() or record.get("bytes") != path.stat().st_size or record.get("sha256") != _sha(path):
        raise ValueError(f"{label} binding changed")


def validate(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    registration_path, queue_path = root / "registration.json", root / "queue.jsonl"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("schema_version") != "vla-wam-shared-v3c002-registration-v1":
        raise ValueError("historical V1 registration schema changed")
    if registration.get("registration_status") != "pre_registration_draft_pending_two_human_wording_agreements":
        raise ValueError("historical V1 draft was activated")
    if registration.get("model_request_count_before_registration") != 0 or registration.get("behavioral_episode_count_before_registration") != 0:
        raise ValueError("historical V1 records inference")
    if registration.get("model_requests_authorized") is not False or registration.get("behavioral_episodes_authorized") is not False:
        raise ValueError("historical V1 authorizes inference")
    rows = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 1364 or len({row.get("cell_id") for row in rows}) != 1364:
        raise ValueError("historical V1 queue changed")
    if any(row.get("release_status") != "pre_registration_draft_pending_two_human_wording_agreements" for row in rows):
        raise ValueError("historical V1 queue was released")
    release = json.loads((root / "release_gate.json").read_text(encoding="utf-8"))
    if release.get("schema_version") != "vla-wam-shared-v3c002-release-gate-v1" or release.get("passed") is not False:
        raise ValueError("historical V1 release gate changed")
    for name in ("registration", "queue", "wording_gate"):
        _bound(release.get(name), f"historical V1 {name}")
    if (root / "infrastructure_attempts.jsonl").read_bytes() != b"":
        raise ValueError("historical V1 has an infrastructure attempt")
    return {
        "status": "valid_immutable_unexecuted_superseded_v1_draft",
        "registration_sha256": _sha(registration_path),
        "queue_sha256": _sha(queue_path),
        "release_gate_sha256": _sha(root / "release_gate.json"),
        "queue_rows": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.root), indent=2, sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"historical V3-C002 V1 validation failed: {exc}") from exc


if __name__ == "__main__":
    main()
