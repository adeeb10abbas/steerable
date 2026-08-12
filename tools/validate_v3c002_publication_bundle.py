#!/usr/bin/env python3
"""Validate the frozen publication bundle plus the additive C002 closure."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(path: Path) -> dict:
    command = [sys.executable, str(path)]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    record = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        print(json.dumps({"status": "invalid", "run": record}, indent=2))
        raise SystemExit(completed.returncode)
    return record


def main() -> None:
    runs = [
        run(REPO_ROOT / "tools/validate_v3e_publication_bundle.py"),
        run(REPO_ROOT / "tools/validate_v3c002_isolation_closure.py"),
    ]
    print(
        json.dumps(
            {
                "status": "valid_with_c002_failed_isolation_closure",
                "c002_behaviorally_executed": False,
                "c002_semantic_result": False,
                "runs": runs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
