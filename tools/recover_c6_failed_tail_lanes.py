#!/usr/bin/env python3
"""Recover C6 wave-D tail lanes with Failed sims (sim-first policy deadlock)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / "artifacts/online_correction_v4/execution/c6_containment_confirmatory_20260908"
TAIL_RECEIPT = EXEC / "create-c6confirm20260908f-wave-d-tail.json"
TAIL_RENDER = EXEC / "rendered-c6confirm20260908f-wave-d-tail"


def _pod_phases(*, kube_context: str, namespace: str) -> dict[str, dict[str, list[str]]]:
    proc = subprocess.run(
        ["kubectl", "get", "pods", "-n", namespace, "--context", kube_context, "-o", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    import re

    by_lane: dict[str, dict[str, list[str]]] = {}
    for item in json.loads(proc.stdout)["items"]:
        name = item["metadata"]["name"]
        match = re.match(r"v4-(c6m\d+)-(attempt\d+)-[a-f0-9]+-(sim|policy)-", name)
        if not match:
            continue
        lane, role = match.group(1), match.group(3)
        phase = item.get("status", {}).get("phase", "?")
        by_lane.setdefault(lane, {}).setdefault(role, []).append(phase)
    return by_lane


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument(
        "--receipt-out",
        type=Path,
        default=EXEC / "c6_wave_d_tail_recovery_20260908.json",
    )
    args = parser.parse_args(argv)

    import yaml

    kube_context = "prod-dcwi-warrenq1-vmkub007"
    namespace = "211247-prod"
    publisher_pod = "211247-ali-b200-1gpu"
    pvc_root = "/data/users/ali/vla_wam/raw/v4/c6-containment-main"

    plan = json.loads(TAIL_RECEIPT.read_text(encoding="utf-8"))["plan"]
    lane_attempt = {row["lane_id"]: row["attempt_id"] for row in plan}
    phases = _pod_phases(kube_context=kube_context, namespace=namespace)

    targets: list[str] = []
    for lane_id, attempt_id in lane_attempt.items():
        lane_phases = phases.get(lane_id, {})
        sim_phases = lane_phases.get("sim", [])
        if any(p in {"Failed", "Error"} for p in sim_phases):
            targets.append(lane_id)

    clear_script = f"""
import glob, json, os
root = {json.dumps(pvc_root)}
removed = []
for att in range(226, 258):
    for path in glob.glob(root + f"/**/.simulator-lane-*-attempt-attempt{{att:04d}}.lock", recursive=True):
        os.remove(path)
        removed.append(path)
print(json.dumps({{"removed_count": len(removed)}}))
"""
    lock_clear = {}
    if args.dispatch:
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
        if proc.returncode == 0:
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

        for lane_id in sorted(targets):
            attempt_id = lane_attempt[lane_id]
            prefix = f"v4-{lane_id}-{attempt_id}"
            for name in list(job_names):
                if not name.startswith(prefix):
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
                    actions.append({"action": "delete_sim", "job": name, "lane_id": lane_id})
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
                            "lane_id": lane_id,
                            "ok": patch.returncode == 0,
                        }
                    )

            matches = sorted(TAIL_RENDER.glob(f"{lane_id}-{attempt_id}-*"))
            if not matches:
                raise SystemExit(f"missing bundle for {lane_id} {attempt_id}")
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
                    "job": doc["metadata"]["name"],
                    "lane_id": lane_id,
                    "ok": ok,
                    "message": (create.stderr or create.stdout or "").strip()[:200],
                }
            )

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from tools.v4_gpu_placement_enforce import enforce_gpu_placement

        gate = enforce_gpu_placement(
            policy_id="c6_a10040_spread",
            kube_context=kube_context,
            namespace=namespace,
            dry_run=False,
            protect_list_path=ROOT
            / "artifacts/online_correction_v4/execution/gpu_widen_20260908/gpu_sweep_protect_list_20260908_phase2.json",
            gates_only=True,
            settle_seconds=5,
        )
    else:
        gate = {}

    out = {
        "schema_version": "v4-c6-wave-d-tail-recovery-v1",
        "root_cause": "Sim-first deadlock on tail attempts 0226-0257: policies Suspended, sims timed out waiting for /healthz (same class as C7 r6). Gate fires but only after sim Running; failed sims never reached Running.",
        "failed_lane_count": len(targets),
        "failed_lanes": sorted(targets),
        "lock_clear": lock_clear,
        "recovery_actions": actions,
        "sim_first_gate": gate.get("actions") if isinstance(gate, dict) else [],
    }
    args.receipt_out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
