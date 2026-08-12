#!/usr/bin/env python3
"""Audit the immutable, unexecuted V3-C002 V5 draft after V6 supersession."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002/draft_v5"
def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def validate(root: Path = ROOT) -> dict:
    root = Path(root).resolve(); registration_path = root / "registration.json"; queue_path = root / "queue.jsonl"; release_path = root / "release_gate.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("schema_version") != "vla-wam-shared-v3c002-registration-v4" or registration.get("registration_status") != "pre_registration_draft_pending_two_human_wording_agreements": raise ValueError("historical V5 draft identity/status changed")
    if registration.get("model_requests_authorized") is not False or registration.get("behavioral_episodes_authorized") is not False: raise ValueError("historical V5 authorizes inference")
    rows = queue_path.read_text(encoding="utf-8").splitlines()
    if len(rows) != 1364 or len({json.loads(row).get("cell_id") for row in rows}) != 1364: raise ValueError("historical V5 queue changed")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("schema_version") != "vla-wam-shared-v3c002-release-gate-v4" or release.get("passed") is not False: raise ValueError("historical V5 release changed")
    if (root / "infrastructure_attempts.jsonl").read_bytes() != b"": raise ValueError("historical V5 has infrastructure attempts")
    return {"status": "valid_immutable_unexecuted_superseded_v5_draft", "registration_sha256": _sha(registration_path), "queue_sha256": _sha(queue_path), "release_gate_sha256": _sha(release_path), "queue_rows": len(rows)}
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=ROOT); args = parser.parse_args()
    try: print(json.dumps(validate(args.root), indent=2, sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as exc: raise SystemExit(f"historical V3-C002 V5 validation failed: {exc}") from exc
if __name__ == "__main__": main()
