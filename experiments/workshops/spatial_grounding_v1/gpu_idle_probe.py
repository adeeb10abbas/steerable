"""Select only a visibly idle GPU inside the probe Job's allocation."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import time


def select_idle(gpus: list[dict], occupied_uuids: set[str], expected_count: int) -> dict:
    if expected_count not in range(1, 5) or len(gpus) != expected_count:
        raise ValueError("Visible GPU count differs from the bounded Job allocation")
    candidates = [
        gpu for gpu in gpus
        if gpu["uuid"] not in occupied_uuids
        and gpu["memory_used_mib"] <= 1024
        and gpu["utilization_percent"] == 0
        and gpu["memory_free_mib"] >= 16384
    ]
    if not candidates:
        raise RuntimeError("No allocated GPU is idle; no simulator or model may start")
    return min(candidates, key=lambda gpu: (gpu["memory_used_mib"], gpu["index"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ], text=True)
    gpus = []
    for row in csv.reader(io.StringIO(raw), skipinitialspace=True):
        index, uuid, name, used, free, utilization = row
        gpus.append({"index": int(index), "uuid": uuid, "name": name,
                     "memory_used_mib": int(used), "memory_free_mib": int(free),
                     "utilization_percent": int(utilization)})
    occupied = {row[0] for row in csv.reader(io.StringIO(processes), skipinitialspace=True) if row}
    receipt = {"study_id": "SGW-01", "observed_at_unix": time.time(),
               "expected_visible_gpu_count": args.expected_count, "gpus": gpus,
               "compute_occupied_uuids": sorted(occupied), "model_requests": 0}
    try:
        selected = select_idle(gpus, occupied, args.expected_count)
    except (ValueError, RuntimeError) as exc:
        receipt.update({"status": "blocked", "reason": str(exc)})
        with args.output.open("x") as stream:
            stream.write(json.dumps(receipt, indent=2) + "\n")
        raise
    receipt.update({"status": "passed_idle_snapshot_only", "selected_gpu": selected})
    with args.output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(selected["uuid"])


if __name__ == "__main__":
    main()
