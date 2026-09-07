#!/usr/bin/env python3
"""Isolated, immutable cluster study checkouts for concurrent V4 workstreams."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts/online_correction_v4/setup/cluster_study_checkouts.json"
CLUSTER_SRC_PARENT = "/data/users/ali/vla_wam/src"
BRANCH = "research/online-correction-v4"

# Legacy shared paths that must never be re-pinned while jobs are live.
LEGACY_SHARED_CHECKOUTS: frozenset[str] = frozenset(
    {
        "/data/users/ali/vla_wam/src/steerable-v4-g2-5874f2f",
        "/data/users/ali/vla_wam/src/steerable-v4-c2-g3",
        "/data/users/ali/vla_wam/src/steerable-v4-g3-c5c6-424a91b",
    }
)

WORKSTREAM_TEMPLATE_ROOTS: dict[str, str] = {
    "g2_repair_v2": "/data/users/ali/vla_wam/src/steerable-v4-g2-5874f2f",
    "c2_g3": "/data/users/ali/vla_wam/src/steerable-v4-c2-g3",
    "g3_c5c6": "/data/users/ali/vla_wam/src/steerable-v4-g3-c5c6-424a91b",
}


class CheckoutIsolationError(RuntimeError):
    """Checkout allocation or mutation is unsafe."""


def isolated_study_root(*, workstream_id: str, pin_commit: str, attempt_id: str) -> str:
    safe_attempt = re.sub(r"[^a-zA-Z0-9_-]+", "-", attempt_id)[:32]
    return (
        f"{CLUSTER_SRC_PARENT}/steerable-v4-{workstream_id}-"
        f"{pin_commit[:8]}-{safe_attempt}"
    )


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": "v4-cluster-study-checkout-manifest-v2",
            "isolated_checkouts": [],
            "legacy_shared_checkouts": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version", "").endswith("v1"):
        payload["schema_version"] = "v4-cluster-study-checkout-manifest-v2"
        payload.setdefault("isolated_checkouts", [])
        payload.setdefault("legacy_shared_checkouts", payload.pop("checkouts", []))
    return payload


def save_manifest(payload: dict[str, Any], path: Path = DEFAULT_MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        raise CheckoutIsolationError(completed.stderr or completed.stdout or "kubectl exec failed")
    return completed.stdout


def list_live_attempt_ids(
    *,
    kube_context: str,
    namespace: str,
) -> set[str]:
    completed = subprocess.run(
        [
            "kubectl",
            "--context",
            kube_context,
            "-n",
            namespace,
            "get",
            "jobs",
            "-l",
            "app.kubernetes.io/part-of=vla-wam-v4",
            "-o",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CheckoutIsolationError(completed.stderr or completed.stdout or "kubectl get jobs failed")
    payload = json.loads(completed.stdout)
    live: set[str] = set()
    for job in payload.get("items", []):
        status = job.get("status") or {}
        active = int(status.get("active") or 0)
        succeeded = int(status.get("succeeded") or 0)
        failed = int(status.get("failed") or 0)
        if active > 0 or (succeeded == 0 and failed == 0):
            attempt = (job.get("metadata", {}).get("labels") or {}).get("v4-attempt-id")
            if attempt:
                live.add(str(attempt))
    return live


def bindings_for_study_root(
    manifest: Mapping[str, Any],
    study_root: str,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for entry in manifest.get("isolated_checkouts", []):
        if entry.get("study_root") == study_root:
            matches.append(dict(entry))
    for entry in manifest.get("legacy_shared_checkouts", []):
        if entry.get("study_root") == study_root:
            matches.append(dict(entry))
    return matches


def assert_checkout_mutable(
    *,
    study_root: str,
    kube_context: str,
    namespace: str,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> None:
    if study_root not in LEGACY_SHARED_CHECKOUTS:
        return
    manifest = load_manifest(manifest_path)
    live_attempts = list_live_attempt_ids(kube_context=kube_context, namespace=namespace)
    bound = bindings_for_study_root(manifest, study_root)
    blocking = [
        item
        for item in bound
        if item.get("attempt_id") in live_attempts and item.get("immutable", True)
    ]
    if blocking:
        attempts = sorted({str(item.get("attempt_id")) for item in blocking})
        raise CheckoutIsolationError(
            f"refusing to mutate shared checkout {study_root}: live jobs bound for attempts "
            f"{attempts}. Provision an isolated checkout for new waves instead."
        )
    # Also block if any live job still references the legacy path in-cluster bundles
    # even when manifest is stale.
    if live_attempts:
        raise CheckoutIsolationError(
            f"refusing to mutate legacy shared checkout {study_root} while "
            f"{len(live_attempts)} V4 attempt(s) have active or pending jobs. "
            "Use isolated checkout provisioning for new dispatches."
        )


def _checkout_state_on_cluster(
    *,
    study_root: str,
    kube_context: str,
    namespace: str,
    publisher_pod: str,
) -> dict[str, Any]:
    script = (
        f'ROOT="{study_root}"; '
        f'if [ ! -d "$ROOT/.git" ]; then echo "{{\\"exists\\":false}}"; exit 0; fi; '
        f'HEAD=$(git -C "$ROOT" rev-parse HEAD); '
        f'DIRTY=$(git -C "$ROOT" status --porcelain | wc -l | tr -d " "); '
        f'echo "{{\\"exists\\":true,\\"head\\":\\"$HEAD\\",\\"dirty_count\\":\\"$DIRTY\\"}}"'
    )
    return json.loads(
        _kubectl_exec(
            kube_context=kube_context,
            namespace=namespace,
            publisher_pod=publisher_pod,
            script=script,
        ).strip()
    )


def provision_isolated_checkout(
    *,
    workstream_id: str,
    pin_commit: str,
    attempt_id: str,
    kube_context: str,
    namespace: str,
    publisher_pod: str,
    apply: bool,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    pin_commit = pin_commit.lower()
    if workstream_id not in WORKSTREAM_TEMPLATE_ROOTS:
        raise CheckoutIsolationError(f"unknown workstream_id: {workstream_id}")
    study_root = isolated_study_root(
        workstream_id=workstream_id,
        pin_commit=pin_commit,
        attempt_id=attempt_id,
    )
    template_root = WORKSTREAM_TEMPLATE_ROOTS[workstream_id]
    before = _checkout_state_on_cluster(
        study_root=study_root,
        kube_context=kube_context,
        namespace=namespace,
        publisher_pod=publisher_pod,
    )
    after = dict(before)
    if apply:
        provision_script = (
            f'TARGET="{study_root}"; '
            f'TEMPLATE="{template_root}"; '
            f'PIN="{pin_commit}"; '
            f'if [ -d "$TARGET/.git" ]; then '
            f'  git -C "$TARGET" fetch origin {BRANCH}; '
            f'elif [ -d "$TEMPLATE/.git" ]; then '
            f'  git clone --local "$TEMPLATE" "$TARGET"; '
            f'  git -C "$TARGET" remote set-url origin "$(git -C "$TEMPLATE" remote get-url origin)"; '
            f'  git -C "$TARGET" fetch origin {BRANCH}; '
            f'else '
            f'  echo "missing template checkout: $TEMPLATE" >&2; exit 5; '
            f'fi; '
            f'git -C "$TARGET" checkout "$PIN"; '
            f'git -C "$TARGET" reset --hard "$PIN"; '
            f'git -C "$TARGET" clean -fd; '
            f'HEAD=$(git -C "$TARGET" rev-parse HEAD); '
            f'DIRTY=$(git -C "$TARGET" status --porcelain | wc -l | tr -d " "); '
            f'echo "{{\\"exists\\":true,\\"head\\":\\"$HEAD\\",\\"dirty_count\\":\\"$DIRTY\\"}}"'
        )
        after = json.loads(
            _kubectl_exec(
                kube_context=kube_context,
                namespace=namespace,
                publisher_pod=publisher_pod,
                script=provision_script,
            ).strip()
        )

    ok = after.get("exists") and str(after.get("head", "")).lower() == pin_commit and int(after.get("dirty_count", 1)) == 0
    entry = {
        "workstream_id": workstream_id,
        "attempt_id": attempt_id,
        "study_root": study_root,
        "pin_commit": pin_commit,
        "immutable": True,
        "template_root": template_root,
        "before_commit": before.get("head"),
        "after_commit": after.get("head"),
        "dirty_after": int(after.get("dirty_count") or 0),
        "ok": ok,
        "applied": apply,
    }
    if apply and ok:
        manifest = load_manifest(manifest_path)
        isolated = [
            item
            for item in manifest.get("isolated_checkouts", [])
            if item.get("attempt_id") != attempt_id
        ]
        isolated.append(entry)
        manifest["isolated_checkouts"] = isolated
        manifest["observed_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest["kube_context"] = kube_context
        manifest["namespace"] = namespace
        save_manifest(manifest, manifest_path)
    return entry


def resolve_workstream_id(study_root: str) -> str | None:
    for workstream_id, template_root in WORKSTREAM_TEMPLATE_ROOTS.items():
        if study_root == template_root or study_root.startswith(
            f"{CLUSTER_SRC_PARENT}/steerable-v4-{workstream_id}-"
        ):
            return workstream_id
    return None


def ensure_isolated_study_root_for_dispatch(
    spec: Mapping[str, Any],
    *,
    kube_context: str,
    namespace: str,
    publisher_pod: str,
    apply: bool,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    study_root = str(spec.get("study_root") or "")
    pin_commit = str(spec.get("expected_study_commit") or "")
    attempt_id = str(spec.get("attempt_id") or "")
    if not study_root or not pin_commit or not attempt_id:
        raise CheckoutIsolationError("render spec lacks study_root, expected_study_commit, or attempt_id")
    workstream_id = resolve_workstream_id(study_root)
    if workstream_id is None:
        return {
            "isolated": False,
            "study_root": study_root,
            "reason": "study_root is not a known shared V4 workstream template",
        }
    if study_root not in LEGACY_SHARED_CHECKOUTS:
        return {
            "isolated": True,
            "study_root": study_root,
            "workstream_id": workstream_id,
            "reason": "already using isolated checkout path",
        }
    report = provision_isolated_checkout(
        workstream_id=workstream_id,
        pin_commit=pin_commit,
        attempt_id=attempt_id,
        kube_context=kube_context,
        namespace=namespace,
        publisher_pod=publisher_pod,
        apply=apply,
        manifest_path=manifest_path,
    )
    report["isolated"] = True
    report["legacy_study_root"] = study_root
    return report
