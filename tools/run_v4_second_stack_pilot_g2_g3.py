#!/usr/bin/env python3
"""Run C8 engineering-pilot G2 and G3 model-blind qualification locally on SimplerEnv."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

DEFAULT_PILOT_RESET = (
    ROOT
    / "artifacts/online_correction_v4/setup/second_stack_pilot_reset_registry.candidate.json"
)
DEFAULT_PLAN = (
    ROOT / "artifacts/online_correction_v4/setup/second_stack_g3_plan.candidate.json"
)
DEFAULT_INTEGRATION = Path("/data/users/ali/vla_wam/external/gr00t-bridge-integration")


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-reset-registry", type=Path, default=DEFAULT_PILOT_RESET)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--integration-root",
        type=Path,
        default=DEFAULT_INTEGRATION,
        help="GR00T Bridge / SimplerEnv integration checkout on cluster or laptop",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--max-seeds", type=int, default=24)
    parser.add_argument("--skip-g3-scripted", action="store_true")
    args = parser.parse_args(argv)

    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    g2_out = out_root / "pilot_g2_aggregate.json"
    scale_token = str(args.scale).replace(".", "p")
    g3_path_out = out_root / f"pilot_g3_path_scale_{scale_token}.json"
    g3_scripted_out = out_root / "pilot_g3_scripted_aggregate.json"

    _run(
        [
            sys.executable,
            str(TOOLS / "run_v4_second_stack_g2.py"),
            "--registry",
            str(args.pilot_reset_registry.resolve()),
            "--integration-root",
            str(args.integration_root.resolve()),
            "--output",
            str(g2_out),
            "--max-seeds",
            str(args.max_seeds),
        ]
    )

    _run(
        [
            sys.executable,
            str(TOOLS / "run_v4_second_stack_g3_path.py"),
            "--registry",
            str(args.pilot_reset_registry.resolve()),
            "--plan",
            str(args.plan.resolve()),
            "--integration-root",
            str(args.integration_root.resolve()),
            "--scale",
            str(args.scale),
            "--output",
            str(g3_path_out),
        ]
    )

    if not args.skip_g3_scripted:
        _run(
            [
                sys.executable,
                str(TOOLS / "run_v4_second_stack_g3_scripted.py"),
                "--registry",
                str(args.pilot_reset_registry.resolve()),
                "--plan",
                str(args.plan.resolve()),
                "--path-receipt",
                str(g3_path_out),
                "--integration-root",
                str(args.integration_root.resolve()),
                "--output",
                str(g3_scripted_out),
                "--max-checks",
                "8",
            ]
        )

    manifest = {
        "schema_version": "v4-second-stack-pilot-g2-g3-run-manifest-v1",
        "fixture_id": "second_stack",
        "family_id": "C8",
        "cohort": "engineering_pilot",
        "max_seeds": args.max_seeds,
        "scale": args.scale,
        "outputs": {
            "g2_aggregate": str(g2_out.relative_to(ROOT)),
            "g3_path_receipt": str(g3_path_out.relative_to(ROOT)),
            "g3_scripted_aggregate": None
            if args.skip_g3_scripted
            else str(g3_scripted_out.relative_to(ROOT)),
        },
        "execution_path": "local_simplerenv_not_k8s",
        "release_boundary": (
            "Engineering-pilot model-blind qualification only. Policy episodes remain "
            "blocked until repaired Isaac trigger path passes live controls and C8 "
            "policy trigger adapter is verified separately."
        ),
    }
    manifest_path = out_root / "pilot_g2_g3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
