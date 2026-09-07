#!/usr/bin/env python3
"""Provision isolated cluster study checkouts; refuse legacy re-pin while jobs are live."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import v4_study_checkout_isolation as isolation  # noqa: E402

DEFAULT_MANIFEST = isolation.DEFAULT_MANIFEST


def _resolve_pin_commit(explicit: str | None) -> str:
    if explicit:
        return explicit.lower()
    import subprocess

    completed = subprocess.run(
        ["git", "rev-parse", "origin/research/online-correction-v4"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if completed.returncode != 0:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
    return completed.stdout.strip().lower()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kube-context", required=True)
    parser.add_argument("--namespace", default="211247-prod")
    parser.add_argument("--publisher-pod", default="211247-sz5vjy-vla4-b200-4gpu")
    parser.add_argument("--pin-commit", default=None)
    parser.add_argument("--workstream", default=None, choices=sorted(isolation.WORKSTREAM_TEMPLATE_ROOTS))
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--list-live-attempts",
        action="store_true",
        help="Print active/pending V4 attempt ids and exit.",
    )
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    if args.list_live_attempts:
        live = sorted(
            isolation.list_live_attempt_ids(
                kube_context=args.kube_context,
                namespace=args.namespace,
            )
        )
        print(json.dumps({"live_attempt_ids": live}, indent=2, sort_keys=True))
        return 0

    pin_commit = _resolve_pin_commit(args.pin_commit)

    if args.workstream and args.attempt_id:
        report = isolation.provision_isolated_checkout(
            workstream_id=args.workstream,
            pin_commit=pin_commit,
            attempt_id=args.attempt_id,
            kube_context=args.kube_context,
            namespace=args.namespace,
            publisher_pod=args.publisher_pod,
            apply=args.apply,
            manifest_path=args.manifest_out,
        )
        manifest = isolation.load_manifest(args.manifest_out)
        manifest.update(
            {
                "observed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "branch": isolation.BRANCH,
                "pin_commit": pin_commit,
                "kube_context": args.kube_context,
                "namespace": args.namespace,
                "publisher_pod": args.publisher_pod,
                "last_provision": report,
            }
        )
        isolation.save_manifest(manifest, args.manifest_out)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("ok") else 1

    legacy_reports: list[dict[str, object]] = []
    blocked = False
    for study_root in sorted(isolation.LEGACY_SHARED_CHECKOUTS):
        try:
            isolation.assert_checkout_mutable(
                study_root=study_root,
                kube_context=args.kube_context,
                namespace=args.namespace,
                manifest_path=args.manifest_out,
            )
            legacy_reports.append(
                {
                    "study_root": study_root,
                    "mutable": True,
                    "note": "legacy shared path; do not re-pin — provision isolated checkout per wave",
                }
            )
        except isolation.CheckoutIsolationError as exc:
            blocked = True
            legacy_reports.append(
                {
                    "study_root": study_root,
                    "mutable": False,
                    "blocked_reason": str(exc),
                }
            )

    manifest = {
        "schema_version": "v4-cluster-study-checkout-manifest-v2",
        "observed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "branch": isolation.BRANCH,
        "pin_commit": pin_commit,
        "kube_context": args.kube_context,
        "namespace": args.namespace,
        "publisher_pod": args.publisher_pod,
        "applied": False,
        "legacy_shared_checkouts": legacy_reports,
        "isolated_checkouts": isolation.load_manifest(args.manifest_out).get("isolated_checkouts", []),
        "policy": (
            "Legacy shared study roots are frozen while live jobs exist. "
            "New waves must use provision_isolated_checkout via dispatch or "
            "--workstream/--attempt-id on this tool."
        ),
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if blocked and args.apply:
        print(
            "Refusing legacy --apply re-pin: use --workstream and --attempt-id to provision isolated checkouts.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
