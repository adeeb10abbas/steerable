#!/usr/bin/env python3
"""Create write-once OUTPUT_PARENT directories on the cluster PVC for a rendered bundle."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

OUTPUT_PARENT_RE = re.compile(
    r'name: OUTPUT_PARENT\n\s+value: "([^"]+)"'
)


def _collect_output_parents(bundle_root: Path) -> list[str]:
    parents: set[str] = set()
    for job_path in sorted(bundle_root.glob("s*-job.yaml")):
        text = job_path.read_text(encoding="utf-8")
        match = OUTPUT_PARENT_RE.search(text)
        if match is None:
            raise SystemExit(f"OUTPUT_PARENT not found in {job_path}")
        parents.add(match.group(1))
    return sorted(parents)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--kube-context", required=True)
    parser.add_argument("--namespace", default="211247-prod")
    parser.add_argument(
        "--publisher-pod",
        default="211247-sz5vjy-vla4-b200-4gpu",
        help="Pod with PVC mounted for mkdir",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    bundle_root = args.bundle_root.resolve()
    parents = _collect_output_parents(bundle_root)
    if not parents:
        raise SystemExit(f"no OUTPUT_PARENT paths found under {bundle_root}")

    quoted = " ".join(f'"{parent}"' for parent in parents)
    script = f"mkdir -p {quoted} && echo prepared={len(parents)}"
    cmd = [
        "kubectl",
        "--context",
        args.kube_context,
        "-n",
        args.namespace,
        "exec",
        args.publisher_pod,
        "--",
        "bash",
        "-lc",
        script,
    ]
    print(f"preparing {len(parents)} OUTPUT_PARENT directories", file=sys.stderr)
    if args.dry_run:
        print(script)
        return 0

    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stderr or completed.stdout, file=sys.stderr)
        return completed.returncode
    print(completed.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
