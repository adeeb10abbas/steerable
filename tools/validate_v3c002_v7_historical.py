#!/usr/bin/env python3
"""Audit the immutable, unexecuted V3-C002 V7 draft and activation."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002"
DRAFT = BASE / "draft_v7"
ACTIVATION = BASE / "superseded_activation_v7"
def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def validate(draft_root: Path = DRAFT, activation_root: Path = ACTIVATION) -> dict:
    draft_root = Path(draft_root).resolve()
    activation_root = Path(activation_root).resolve()
    if not activation_root.is_dir() and (BASE / "active").is_dir():
        activation_root = (BASE / "active").resolve()
    draft_registration = json.loads((draft_root / "registration.json").read_text(encoding="utf-8"))
    if draft_registration.get("registration_status") != "pre_registration_draft_pending_two_human_wording_agreements": raise ValueError("historical V7 draft changed")
    registration_path = activation_root / "registration.json"; queue_path = activation_root / "queue.jsonl"; release_path = activation_root / "release_gate.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8")); release = json.loads(release_path.read_text(encoding="utf-8"))
    if registration.get("schema_version") != "vla-wam-shared-v3c002-registration-v4" or registration.get("registration_status") != "registered_after_two_human_wording_agreements": raise ValueError("historical V7 activation identity changed")
    if registration.get("model_requests_authorized") is not False or registration.get("behavioral_episodes_authorized") is not False: raise ValueError("historical V7 activation authorizes inference")
    if release.get("schema_version") != "vla-wam-shared-v3c002-release-gate-v4" or release.get("passed") is not False: raise ValueError("historical V7 release changed")
    rows = queue_path.read_text(encoding="utf-8").splitlines()
    if len(rows) != 1364 or len({json.loads(row).get("cell_id") for row in rows}) != 1364: raise ValueError("historical V7 queue changed")
    infrastructure = activation_root / "infrastructure_attempts.jsonl"
    if infrastructure.read_bytes() != b"": raise ValueError("historical V7 contains a behavioral/infrastructure attempt")
    return {"status": "valid_immutable_unexecuted_superseded_v7_activation", "draft_registration_sha256": _sha(draft_root / "registration.json"), "registration_sha256": _sha(registration_path), "queue_sha256": _sha(queue_path), "release_gate_sha256": _sha(release_path), "queue_rows": len(rows)}
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--draft-root", type=Path, default=DRAFT); parser.add_argument("--activation-root", type=Path, default=ACTIVATION); args = parser.parse_args()
    try: print(json.dumps(validate(args.draft_root, args.activation_root), indent=2, sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as exc: raise SystemExit(f"historical V3-C002 V7 validation failed: {exc}") from exc
if __name__ == "__main__": main()
