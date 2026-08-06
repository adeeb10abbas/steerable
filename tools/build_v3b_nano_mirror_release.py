#!/usr/bin/env python3
"""Build the calibration-gated V3-B001 Nano position-reflection release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v3.cosmos_nano_phase_b.design import (  # noqa: E402
    build_release,
    sha256_file,
    write_release,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Validate a completed model-blind RoboLab calibration report and "
            "emit the V3-B001 position-reflection amendment, 108-cell queue, "
            "and hash manifest."
        )
    )
    result.add_argument("--repo-root", type=Path, default=ROOT)
    result.add_argument("--calibration-report", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--recorded-at-utc", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payloads = build_release(
        args.repo_root,
        args.calibration_report,
        recorded_at_utc=args.recorded_at_utc,
    )
    paths = write_release(args.output_dir, payloads)
    print(
        json.dumps(
            {
                "status": "released",
                "behavioral_cells": len(payloads.rows),
                "outputs": {
                    name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in paths.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
