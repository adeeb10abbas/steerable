#!/usr/bin/env python3
"""Provision C8 SimplerEnv/GR00T Bridge runtime paths on cluster PVC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

GR00T_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
INTEGRATION_SOURCE = "/data/users/ali/vla_wam/external/Isaac-GR00T-51d4c89"
INTEGRATION_LINK = "/data/users/ali/vla_wam/external/gr00t-bridge-integration"
CHECKPOINT_SOURCE = "/data/users/ali/vla_wam/checkpoints/gr00t_n17_simplerenv_bridge_940134b3"
CHECKPOINT_LINK = "/data/users/ali/vla_wam/checkpoints/groot-n1.7-simplerenv-bridge"
SIMPLER_PYTHON = (
    "/data/users/ali/vla_wam/external/Isaac-GR00T-51d4c89/"
    "gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
)
C8_BOOTSTRAP_BIN = "/data/users/ali/vla_wam/envs/c8-bootstrap/bin"
GROOT_RENDER_LIBS = "/data/users/ali/vla_wam/envs/groot-render-libs/lib"
DEFAULT_RECEIPT = (
    ROOT / "artifacts/online_correction_v4/qualification/20260908_c8_cluster_runtime_provision.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_remote(script: str, *, kube_context: str, namespace: str, pod: str) -> str:
    completed = subprocess.run(
        [
            "kubectl",
            "--context",
            kube_context,
            "-n",
            namespace,
            "exec",
            pod,
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
        raise RuntimeError(completed.stderr or completed.stdout or "remote command failed")
    return completed.stdout.strip()


def provision_on_cluster(
    *,
    kube_context: str,
    namespace: str,
    pod: str,
    apply: bool,
) -> dict[str, Any]:
    script = f"""
set -euo pipefail
INTEGRATION_SOURCE='{INTEGRATION_SOURCE}'
INTEGRATION_LINK='{INTEGRATION_LINK}'
CHECKPOINT_SOURCE='{CHECKPOINT_SOURCE}'
CHECKPOINT_LINK='{CHECKPOINT_LINK}'
SIMPLER_PYTHON='{SIMPLER_PYTHON}'
C8_BOOTSTRAP_BIN='{C8_BOOTSTRAP_BIN}'

if [ ! -d "$INTEGRATION_SOURCE" ]; then echo "missing integration source"; exit 1; fi
if [ ! -d "$CHECKPOINT_SOURCE" ]; then echo "missing checkpoint source"; exit 1; fi
if [ ! -x "$SIMPLER_PYTHON" ]; then echo "missing simpler python"; exit 1; fi

commit=$(git -C "$INTEGRATION_SOURCE" rev-parse HEAD)
if [ "$commit" != '{GR00T_COMMIT}' ]; then echo "integration commit mismatch: $commit"; exit 1; fi

if [ -e "$INTEGRATION_LINK" ] && [ ! -L "$INTEGRATION_LINK" ]; then echo "integration path exists and is not symlink"; exit 1; fi
if [ -L "$INTEGRATION_LINK" ]; then
  current=$(readlink -f "$INTEGRATION_LINK")
  if [ "$current" != "$INTEGRATION_SOURCE" ]; then echo "integration symlink target differs"; exit 1; fi
else
  ln -s "$INTEGRATION_SOURCE" "$INTEGRATION_LINK"
fi

if [ -e "$CHECKPOINT_LINK" ] && [ ! -L "$CHECKPOINT_LINK" ]; then echo "checkpoint path exists and is not symlink"; exit 1; fi
if [ -L "$CHECKPOINT_LINK" ]; then
  current=$(readlink -f "$CHECKPOINT_LINK")
  if [ "$current" != "$CHECKPOINT_SOURCE" ]; then echo "checkpoint symlink target differs"; exit 1; fi
else
  ln -s "$CHECKPOINT_SOURCE" "$CHECKPOINT_LINK"
fi

mkdir -p "$C8_BOOTSTRAP_BIN"
ln -sf "$SIMPLER_PYTHON" "$C8_BOOTSTRAP_BIN/python"
ln -sf "$SIMPLER_PYTHON" "$C8_BOOTSTRAP_BIN/python3"

export LD_LIBRARY_PATH='{GROOT_RENDER_LIBS}:/usr/lib/x86_64-linux-gnu'
export VK_ICD_FILENAMES="$INTEGRATION_SOURCE/external_dependencies/SimplerEnv/nvidia_icd.json"
"$SIMPLER_PYTHON" -c "import sapien; import numpy; print('runtime_ok', sapien.__version__, numpy.__version__)"
"""
    stdout = _run_remote(script, kube_context=kube_context, namespace=namespace, pod=pod) if apply else ""
    receipt: dict[str, Any] = {
        "schema_version": "v4-c8-cluster-runtime-provision-v1",
        "provisioned_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gr00t_commit": GR00T_COMMIT,
        "integration_source": INTEGRATION_SOURCE,
        "integration_link": INTEGRATION_LINK,
        "checkpoint_source": CHECKPOINT_SOURCE,
        "checkpoint_link": CHECKPOINT_LINK,
        "c8_bootstrap_python": f"{C8_BOOTSTRAP_BIN}/python",
        "simpler_python": SIMPLER_PYTHON,
        "ld_library_path": f"{GROOT_RENDER_LIBS}:/usr/lib/x86_64-linux-gnu",
        "remote_verify_stdout": stdout,
        "status": "provisioned" if apply else "dry_run",
    }
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kube-context", default="prod-dcwi-warrenq1-vmkub007")
    parser.add_argument("--namespace", default="211247-prod")
    parser.add_argument("--pod", default="211247-ali-b200-1gpu")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    receipt = provision_on_cluster(
        kube_context=args.kube_context,
        namespace=args.namespace,
        pod=args.pod,
        apply=args.apply,
    )
    if args.apply:
        if args.receipt.exists():
            raise FileExistsError(f"refusing to overwrite: {args.receipt}")
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        args.receipt.write_text(body, encoding="utf-8")
        receipt["receipt_path"] = str(args.receipt.relative_to(ROOT))
        receipt["receipt_sha256"] = sha256_file(args.receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
