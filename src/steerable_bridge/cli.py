from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audits import finalize_audits
from .constants import DEFAULT_SEED
from .io import download_inputs, write_json
from .pipeline import run_pipeline
from .visual_audit import prepare_visual_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steerable-res1",
        description="Audit and construct the matched Bridge 2 x 2 language ablation.",
    )
    parser.add_argument(
        "command",
        choices=("download", "run", "all", "visual-audit", "finalize-audits"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/res1"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command in {"download", "all"}:
        manifest = download_inputs(args.raw_dir, force=args.force_download)
        write_json(args.artifact_dir / "input_manifest.json", manifest)
        print(json.dumps(manifest, indent=2))
    if args.command in {"run", "all"}:
        summary = run_pipeline(args.raw_dir, args.artifact_dir, seed=args.seed)
        print(json.dumps(summary, indent=2))
    if args.command == "visual-audit":
        summary = prepare_visual_audit(
            args.artifact_dir / "visual_alignment_audit.csv",
            args.artifact_dir / "visual_audit",
        )
        print(json.dumps(summary, indent=2))
    if args.command == "finalize-audits":
        summary = finalize_audits(args.artifact_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
