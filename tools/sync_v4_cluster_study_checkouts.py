#!/usr/bin/env python3
"""Sync and verify cluster study checkouts before V4 dispatch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts/online_correction_v4/setup/cluster_study_checkouts.json"

KNOWN_CHECKOUTS: tuple[dict[str, str], ...] = (
    {
        "id": "g2_repair_v2",
        "study_root": "/data/users/ali/vla_wam/src/steerable-v4-g2-5874f2f",
    },
    {
        "id": "c2_g3",
        "study_root": "/data/users/ali/vla_wam/src/steerable-v4-c2-g3",
    },
    {
        "id": "g3_c5c6",
        "study_root": "/data/users/ali/vla_wam/src/steerable-v4-g3-c5c6-424a91b",
    },
)


def _resolve_pin_commit(explicit: str | None) -> str:
    if explicit:
        return explicit.lower()
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


def _kubectl_exec(
    *,
    kube_context: str,
    namespace: str,
    publisher_pod: str,
    script: str,
) -> str:
    completed = subprocess.run(
        [
            "kubectl",
            "--context",
            kube_context,
            "-n",
            namespace,
            "exec",
            publisher_pod,
            "--",
            "bash",
            "-lc",
            script,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stderr or completed.stdout or "kubectl exec failed")
    return completed.stdout


def sync_checkout(
    *,
    study_root: str,
    pin_commit: str,
    kube_context: str,
    namespace: str,
    publisher_pod: str,
    apply: bool,
) -> dict[str, Any]:
    before_script = (
        f'ROOT="{study_root}"; '
        f'[ -d "$ROOT/.git" ] || {{ echo "missing: $ROOT" >&2; exit 2; }}; '
        f'echo BEFORE=$(git -C "$ROOT" rev-parse HEAD); '
        f'echo DIRTY=$(git -C "$ROOT" status --porcelain | wc -l | tr -d " ")'
    )
    before_out = _kubectl_exec(
        kube_context=kube_context,
        namespace=namespace,
        publisher_pod=publisher_pod,
        script=before_script,
    )
    before_commit = None
    dirty_before = None
    for line in before_out.splitlines():
        if line.startswith("BEFORE="):
            before_commit = line.split("=", 1)[1].strip()
        if line.startswith("DIRTY="):
            dirty_before = int(line.split("=", 1)[1].strip())

    after_commit = before_commit
    dirty_after = dirty_before
    if apply:
        sync_script = (
            f'ROOT="{study_root}"; '
            f'git -C "$ROOT" fetch origin research/online-correction-v4; '
            f'git -C "$ROOT" checkout {pin_commit}; '
            f'git -C "$ROOT" reset --hard {pin_commit}; '
            f'git -C "$ROOT" clean -fd; '
            f'echo AFTER=$(git -C "$ROOT" rev-parse HEAD); '
            f'echo DIRTY=$(git -C "$ROOT" status --porcelain | wc -l | tr -d " ")'
        )
        sync_out = _kubectl_exec(
            kube_context=kube_context,
            namespace=namespace,
            publisher_pod=publisher_pod,
            script=sync_script,
        )
        for line in sync_out.splitlines():
            if line.startswith("AFTER="):
                after_commit = line.split("=", 1)[1].strip()
            if line.startswith("DIRTY="):
                dirty_after = int(line.split("=", 1)[1].strip())

    ok = after_commit == pin_commit and dirty_after == 0
    return {
        "study_root": study_root,
        "pin_commit": pin_commit,
        "before_commit": before_commit,
        "after_commit": after_commit,
        "dirty_before": dirty_before,
        "dirty_after": dirty_after,
        "ok": ok,
        "applied": apply,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kube-context", required=True)
    parser.add_argument("--namespace", default="211247-prod")
    parser.add_argument("--publisher-pod", default="211247-sz5vjy-vla4-b200-4gpu")
    parser.add_argument("--pin-commit", default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    pin_commit = _resolve_pin_commit(args.pin_commit)
    reports = []
    for entry in KNOWN_CHECKOUTS:
        reports.append(
            sync_checkout(
                study_root=entry["study_root"],
                pin_commit=pin_commit,
                kube_context=args.kube_context,
                namespace=args.namespace,
                publisher_pod=args.publisher_pod,
                apply=args.apply,
            )
            | {"id": entry["id"]}
        )

    manifest = {
        "schema_version": "v4-cluster-study-checkout-manifest-v1",
        "observed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "branch": "research/online-correction-v4",
        "pin_commit": pin_commit,
        "kube_context": args.kube_context,
        "namespace": args.namespace,
        "publisher_pod": args.publisher_pod,
        "applied": args.apply,
        "checkouts": reports,
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all(item["ok"] for item in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
