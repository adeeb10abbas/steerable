#!/usr/bin/env python3
"""Run C7 FINAL ledger compile on cluster PVC via kubectl cp + exec."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KUBE = "prod-dcwi-warrenq1-vmkub007"
NS = "211247-prod"
POD = "211247-ali-b200-1gpu"
ATTEMPTS_ROOT = "/data/users/ali/vla_wam/raw/v4/c7-object-pair-main"
OUT_ROOT = f"{ATTEMPTS_ROOT}/compiled_ledger_20260908_FINAL"
REMOTE_DIR = "/tmp/v4-c7-final-compile"
MANIFEST = ROOT / "artifacts/online_correction_v4/setup/c7_confirmatory/queue.frozen.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--receipt-out",
        type=Path,
        default=ROOT
        / "artifacts/online_correction_v4/execution/c7_object_pair_20260906/compiled_ledger_20260908_FINAL_receipt.json",
    )
    args = parser.parse_args(argv)

    paths_to_copy = [
        ROOT / "tools/compile_online_correction_v4_ledger.py",
        ROOT / "tools/online_correction_v4.py",
        ROOT / "experiments/online_correction_v4",
        ROOT / "docs/online_correction_v4/campaign.json",
        MANIFEST,
        ROOT / "artifacts/online_correction_v4/protocol.json",
    ]
    for path in paths_to_copy:
        if not path.exists():
            raise SystemExit(f"missing required path: {path}")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tar_path = Path(tmp.name)
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in paths_to_copy:
            arc = path.relative_to(ROOT)
            tar.add(path, arcname=str(arc))

    remote_tar = f"{REMOTE_DIR}.tar.gz"
    if not args.dry_run:
        subprocess.run(
            ["kubectl", "cp", str(tar_path), f"{NS}/{POD}:{remote_tar}", "--context", KUBE],
            check=True,
        )
        setup = f"""
set -e
rm -rf {REMOTE_DIR}
mkdir -p {REMOTE_DIR}
tar xzf {remote_tar} -C {REMOTE_DIR}
cd {REMOTE_DIR}
PYTHONPATH=. python3 tools/compile_online_correction_v4_ledger.py \\
  --manifest artifacts/online_correction_v4/setup/c7_confirmatory/queue.frozen.jsonl \\
  --attempts-root {ATTEMPTS_ROOT} \\
  --out {OUT_ROOT} \\
  --queue artifacts/online_correction_v4/setup/c7_confirmatory/queue.frozen.jsonl \\
  --require-full-coverage \\
  --manifest-scoped-discovery \\
  --workers 4
"""
        proc = subprocess.run(
            ["kubectl", "exec", "-n", NS, "--context", KUBE, POD, "--", "bash", "-lc", setup],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(proc.returncode)

        sync = f"""
python3 - <<'PY'
import json, glob, os
from collections import Counter
ledger = "{OUT_ROOT}/accepted_ledger.jsonl"
comp = Counter()
rows = 0
for line in open(ledger):
    if not line.strip():
        continue
    rows += 1
    r = json.loads(line)
    comp[r.get("outcome", {{}}).get("failure_label", "?")] += 1
print(json.dumps({{"accepted_rows": rows, "composition": dict(comp)}}))
PY
"""
        summary_proc = subprocess.run(
            ["kubectl", "exec", "-n", NS, "--context", KUBE, POD, "--", "bash", "-lc", sync],
            capture_output=True,
            text=True,
            check=True,
        )
        summary = json.loads(summary_proc.stdout.strip())
    else:
        summary = {"dry_run": True}

    receipt = {
        "schema_version": "v4-c7-final-compile-receipt-v1",
        "compile_id": "compiled_ledger_20260908_FINAL",
        "pvc_path": OUT_ROOT,
        "require_full_coverage": True,
        "summary": summary,
        "agent_c_handoff_path": OUT_ROOT,
    }
    args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    tar_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
