#!/usr/bin/env python3
"""Bind the immutable original E006 and R001 exhaustion predecessors for R002."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


R001_COMMIT = "bbabac55dfd54f7a0b7d8a2693673a4b06409f21"
R001_PATH_MARKERS = (
    "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r001/",
    "experiments/v3/phase_e/canonical_stage_localization_v3e006_r001/",
)
R001_EXACT_FILES = {
    "tests/test_v3e006_r001.py",
    "tools/run_v3e006_r001_state_repair.py",
    "tools/validate_v3e006_r001.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.study_root.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite predecessor binding: {output}")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != R001_COMMIT:
        parser.error(f"predecessor binding must be built at {R001_COMMIT}, observed {head}")
    tracked = subprocess.check_output(["git", "-C", str(root), "ls-files"], text=True).splitlines()
    selected = sorted(
        row for row in tracked
        if row in R001_EXACT_FILES or any(row.startswith(marker) for marker in R001_PATH_MARKERS)
    )
    if len(selected) != 20:
        parser.error(f"expected exactly 20 frozen R001 files, observed {len(selected)}")
    files = [binding(root, relative) for relative in selected]
    result = {
        "schema_version": "vla-wam-shared-v3e006-r002-predecessor-closure-binding-v1",
        "repair_amendment_id": "V3-E006-R002",
        "status": "original_and_r001_predecessors_byte_identical_before_r002_live_candidate_or_model_request",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "r002_live_candidate_evaluation_count": 0,
        "required_repository_base": "18a2bf0200183647291cc7aeb1fe89997b3fb82f",
        "original_v3e006_closure_commit": "e13e7b22b048075bf0d3cf44a892c70853ce8a7e",
        "r001_exhaustion_closure_commit": R001_COMMIT,
        "r001_tree_files": files,
        "r001_tree_file_count": len(files),
        "r001_result_binding": next(row for row in files if row["path"].endswith("/results/results.json")),
        "r001_evidence_binding": next(row for row in files if row["path"].endswith("/results/evidence_manifest.json")),
        "r001_decision_memo_binding": next(row for row in files if row["path"].endswith("/results/DECISION_MEMO.md")),
        "claim_boundary": "The original V3-E006 and R001 remain failed/exhausted; this proof does not mark either candidate gate passed.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size, "sha256": sha256(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
