#!/usr/bin/env python3
"""Validate the base publication bundle and the non-mutating final merge audit."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], label: str) -> dict[str, object]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    record = {
        "label": label,
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
        run(
            [sys.executable, str(ROOT / "tools/validate_v3e_publication_bundle.py")],
            "pre-experiment canonical publication bundle",
        ),
        run(
            [sys.executable, str(ROOT / "tools/validate_v3_final_merge_reconciliation.py")],
            "C002/E006 frozen-validator reconciliation",
        ),
    ]
    print(
        json.dumps(
            {
                "status": "valid_final_publication_bundle_with_nonmutating_reconciliation",
                "runs": runs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
