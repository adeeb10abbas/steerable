#!/usr/bin/env python3
"""Write a model-blind Nano position-mirror candidate outside inference evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.v3.cosmos_nano_phase_b.fixture_candidate import (
    ROBOLAB_COMMIT,
    canonical_bytes,
    derive_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-metadata", type=Path, required=True)
    parser.add_argument("--neutral-left-task", type=Path, required=True)
    parser.add_argument("--neutral-right-task", type=Path, required=True)
    parser.add_argument("--robolab-commit", default=ROBOLAB_COMMIT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite fixture candidate: {args.output}")
    value = derive_candidate(
        scene_metadata_path=args.scene_metadata,
        neutral_left_task_path=args.neutral_left_task,
        neutral_right_task_path=args.neutral_right_task,
        robolab_commit=args.robolab_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(value))
    print(args.output)


if __name__ == "__main__":
    main()

