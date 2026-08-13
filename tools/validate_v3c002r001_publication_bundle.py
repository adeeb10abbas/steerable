#!/usr/bin/env python3
"""Additively validate the immutable C002 closure and its R001 repair."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/active"


def run(tool: str) -> dict:
    result = subprocess.run([sys.executable, str(ROOT / "tools" / tool)], cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise SystemExit(result.stderr or result.stdout)
    return {"tool": tool, "returncode": result.returncode, "stdout": result.stdout}


def main() -> None:
    runs = [run("validate_v3c002_publication_bundle.py"), run("validate_v3c002r001.py")]
    result_paths = [REPAIR / "raw/episodes.jsonl", REPAIR / "results/results.json", REPAIR / "results/evidence_manifest.json"]
    if any(path.exists() for path in result_paths):
        if not all(path.exists() for path in result_paths):
            raise SystemExit("partial V3-C002-R001 result bundle")
        runs.append(run("validate_v3c002r001_results.py"))
    print(json.dumps({"status": "valid_c002_closure_plus_r001", "runs": runs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
