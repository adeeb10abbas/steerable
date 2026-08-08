#!/usr/bin/env python3
"""Run the frozen V3 validator and all registered Phase-E validators."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    commands = [
        [sys.executable, str(ROOT / "tools/validate_vla_wam_v3_protocol.py")],
        [sys.executable, str(ROOT / "tools/validate_v3e001.py")],
        [sys.executable, str(ROOT / "tools/validate_v3e002.py")],
        [sys.executable, str(ROOT / "tools/validate_v3e003.py")],
        [sys.executable, str(ROOT / "tools/validate_v3e004.py")],
    ]
    outputs = []
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        outputs.append({"command": command, "returncode": completed.returncode,
                        "stdout": completed.stdout, "stderr": completed.stderr})
        if completed.returncode != 0:
            print(json.dumps({"status": "invalid", "runs": outputs}, indent=2))
            raise SystemExit(completed.returncode)
    print(json.dumps({"status": "valid", "runs": outputs}, indent=2))


if __name__ == "__main__":
    main()
