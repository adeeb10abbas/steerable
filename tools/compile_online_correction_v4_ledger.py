#!/usr/bin/env python3
"""Compile finalized V4 attempt directories into the accepted ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.ledger import (  # noqa: E402
    LedgerError,
    compile_accepted_ledger_from_attempts,
    discover_finalized_attempts,
    load_queue_episode_ids,
    write_ledger_outputs,
)
from experiments.online_correction_v4.analysis import load_campaign_config, load_manifest  # noqa: E402

DEFAULT_CONFIG = ROOT / "docs/online_correction_v4/campaign.json"
DEFAULT_QUEUE = ROOT / "artifacts/online_correction_v4/queue.jsonl"
DEFAULT_PROTOCOL_SHA = "0" * 64


def _load_protocol_sha256(config_path: Path) -> str:
    protocol_path = ROOT / "artifacts/online_correction_v4/protocol.json"
    if protocol_path.is_file():
        payload = json.loads(protocol_path.read_text(encoding="utf-8"))
        digest = payload.get("protocol_sha256") or payload.get("content_sha256")
        if isinstance(digest, str) and len(digest) == 64:
            return digest
    return DEFAULT_PROTOCOL_SHA


def _load_scorer_sha256(config: dict) -> str:
    fixtures = config.get("fixtures", {})
    for fixture in fixtures.values():
        scorer = fixture.get("scorer_sha256")
        if isinstance(scorer, str) and len(scorer) == 64:
            return scorer
    return "1" * 64


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attempts-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--queue", type=Path, default=None)
    parser.add_argument("--protocol-sha256", type=str, default=None)
    parser.add_argument("--scorer-sha256", type=str, default=None)
    parser.add_argument(
        "--require-full-coverage",
        action="store_true",
        help="Fail unless every manifest episode has an accepted valid row.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compile and validate without writing output files.",
    )
    args = parser.parse_args(argv)
    try:
        config, _ = load_campaign_config(args.config)
        protocol_sha256 = args.protocol_sha256 or _load_protocol_sha256(args.config)
        scorer_sha256 = args.scorer_sha256 or _load_scorer_sha256(config)
        manifest = load_manifest(args.manifest)
        attempts = discover_finalized_attempts(args.attempts_root)
        queue_path = args.queue
        if queue_path is None and DEFAULT_QUEUE.is_file():
            queue_path = DEFAULT_QUEUE
        queue_ids = load_queue_episode_ids(queue_path) if queue_path is not None else None
        result = compile_accepted_ledger_from_attempts(
            manifest=manifest,
            attempts=attempts,
            attempts_root=args.attempts_root,
            protocol_sha256=protocol_sha256,
            scorer_sha256=scorer_sha256,
            config=config,
            queue_episode_ids=queue_ids,
            require_full_coverage=args.require_full_coverage,
        )
        report = {
            "ok": result.ok,
            "errors": result.errors,
            "warnings": result.warnings,
            "reconciliation": result.reconciliation,
            "validation_preview": result.manifest_payload.get("validation_preview"),
        }
        if not result.ok:
            print(json.dumps(report, indent=2, sort_keys=True))
            return 1
        if args.dry_run:
            report["dry_run"] = True
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        outputs = write_ledger_outputs(
            result,
            args.out,
            attempts_root=args.attempts_root,
            manifest_path=args.manifest,
        )
        report["outputs"] = outputs
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (LedgerError, ValueError, KeyError, OSError) as exc:
        print(json.dumps({"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
