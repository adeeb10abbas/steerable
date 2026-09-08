#!/usr/bin/env python3
"""Clear stale C7 r6 simulator attempt locks and recreate failed sim jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / "artifacts/online_correction_v4/execution/c7_object_pair_20260906"
R6_RECEIPT = EXEC / "create-released-tail-reshard-20260908r6.json"
R6_RENDER = EXEC / "rendered-released-tail-reshard-20260908r6"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument(
        "--receipt-out",
        type=Path,
        default=EXEC / "c7_r6_lock_clear_recovery_20260908.json",
    )
    args = parser.parse_args(argv)

    import yaml

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    kube_context = "prod-dcwi-warrenq1-vmkub007"
    namespace = "211247-prod"
    publisher_pod = "211247-ali-b200-1gpu"
    pvc_root = "/data/users/ali/vla_wam/raw/v4/c7-object-pair-main"

    receipt = json.loads(R6_RECEIPT.read_text(encoding="utf-8"))
    lane_attempt_map = receipt["lane_attempt_map"]

    clear_script = f"""
import glob, json, os
root = {json.dumps(pvc_root)}
removed = []
for att in range(611, 625):
    pattern = root + f"/**/.simulator-lane-*-attempt-attempt{{att:04d}}.lock"
    for path in glob.glob(pattern, recursive=True):
        os.remove(path)
        removed.append(path)
print(json.dumps({{"removed_count": len(removed), "removed_paths": removed}}))
"""
    proc = subprocess.run(
        [
            "kubectl",
            "exec",
            "-n",
            namespace,
            "--context",
            kube_context,
            publisher_pod,
            "--",
            "python3",
            "-c",
            clear_script,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"lock clear failed: {proc.stderr}")
    lock_clear = json.loads(proc.stdout.strip())

    actions: list[dict] = []
    if args.dispatch:
        jobs_proc = subprocess.run(
            ["kubectl", "get", "jobs", "-n", namespace, "--context", kube_context, "-o", "name"],
            capture_output=True,
            text=True,
            check=True,
        )
        job_names = [line.removeprefix("job.batch/") for line in jobs_proc.stdout.splitlines()]

        for lane_id, attempt_id in sorted(lane_attempt_map.items()):
            for name in list(job_names):
                if not name.startswith(f"v4-{lane_id}-{attempt_id}"):
                    continue
                if name.endswith("-sim"):
                    subprocess.run(
                        [
                            "kubectl",
                            "--context",
                            kube_context,
                            "-n",
                            namespace,
                            "delete",
                            "job",
                            name,
                            "--ignore-not-found",
                        ],
                        check=False,
                    )
                    actions.append({"action": "delete_sim", "job": name})
                elif name.endswith("-policy"):
                    patch = subprocess.run(
                        [
                            "kubectl",
                            "--context",
                            kube_context,
                            "-n",
                            namespace,
                            "patch",
                            "job",
                            name,
                            "-p",
                            '{"spec":{"suspend":false}}',
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    actions.append(
                        {
                            "action": "unsuspend_policy",
                            "job": name,
                            "ok": patch.returncode == 0,
                        }
                    )

            matches = sorted(R6_RENDER.glob(f"{lane_id}-{attempt_id}-*"))
            if not matches:
                raise SystemExit(f"missing render bundle for {lane_id} {attempt_id}")
            sim_path = matches[0] / "simulator-job.yaml"
            doc = yaml.safe_load(sim_path.read_text(encoding="utf-8"))
            create = subprocess.run(
                ["kubectl", "--context", kube_context, "-n", namespace, "create", "-f", "-"],
                input=yaml.safe_dump(doc, sort_keys=False),
                capture_output=True,
                text=True,
                check=False,
            )
            ok = create.returncode == 0 or "AlreadyExists" in (create.stderr or "")
            actions.append(
                {
                    "action": "create_sim",
                    "job": str(doc.get("metadata", {}).get("name") or sim_path.name),
                    "ok": ok,
                    "message": (create.stderr or create.stdout or "").strip()[:200],
                }
            )

        from tools.run_v4_gpu_periodic_enforcement import enforce_c7_sim_first_gate

        gate_actions = enforce_c7_sim_first_gate(
            kube_context=kube_context,
            namespace=namespace,
            dry_run=False,
        )
    else:
        gate_actions = []

    out = {
        "schema_version": "v4-c7-r6-lock-clear-recovery-v1",
        "lane_count": len(lane_attempt_map),
        "lock_clear": lock_clear,
        "dispatch_actions": actions,
        "c7_gate_actions": gate_actions,
    }
    args.receipt_out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
