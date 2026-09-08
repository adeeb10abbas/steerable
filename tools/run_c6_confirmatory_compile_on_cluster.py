#!/usr/bin/env python3
"""Run C6 confirmatory ledger compile on cluster PVC via kubectl cp + exec."""

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
ATTEMPTS_ROOT = "/data/users/ali/vla_wam/raw/v4/c6-containment-main"
REMOTE_DIR = "/tmp/v4-c6-final-compile"
MANIFEST = ROOT / "artifacts/online_correction_v4/setup/c6_confirmatory/queue.frozen.jsonl"
SCOPED_MANIFEST_ARC = "artifacts/online_correction_v4/setup/c6_confirmatory/queue.frozen.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--compile-id",
        default="compiled_ledger_20260908_FINAL",
        help="Output directory name under attempts root",
    )
    parser.add_argument("--require-full-coverage", action="store_true")
    parser.add_argument(
        "--receipt-out",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)

    out_root = f"{ATTEMPTS_ROOT}/{args.compile_id}"
    receipt_out = args.receipt_out or (
        ROOT
        / f"artifacts/online_correction_v4/execution/c6_containment_confirmatory_20260908/{args.compile_id}_receipt.json"
    )

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
    require_flag = " \\\n  --require-full-coverage" if args.require_full_coverage else ""
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
  --manifest {SCOPED_MANIFEST_ARC} \\
  --attempts-root {ATTEMPTS_ROOT} \\
  --out {out_root} \\
  --queue {SCOPED_MANIFEST_ARC} \\
  --manifest-scoped-discovery \\
  --workers 4{require_flag}
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
import json
from collections import Counter, defaultdict
ledger = "{out_root}/accepted_ledger.jsonl"
comp = Counter()
by_scenario = defaultdict(lambda: Counter())
rows = 0
for line in open(ledger):
    if not line.strip():
        continue
    rows += 1
    r = json.loads(line)
    label = r.get("outcome", {{}}).get("failure_label", "?")
    comp[label] += 1
    scenario = r.get("scenario") or r.get("factors", {{}}).get("scenario", "?")
    by_scenario[scenario][label] += 1
print(json.dumps({{"accepted_rows": rows, "composition": dict(comp), "by_scenario": {{k: dict(v) for k,v in by_scenario.items()}}}}))
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
        "schema_version": "v4-c6-confirmatory-compile-receipt-v1",
        "compile_id": args.compile_id,
        "pvc_path": out_root,
        "require_full_coverage": args.require_full_coverage,
        "summary": summary,
        "agent_c_handoff_path": out_root,
    }
    receipt_out.parent.mkdir(parents=True, exist_ok=True)
    receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    tar_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
