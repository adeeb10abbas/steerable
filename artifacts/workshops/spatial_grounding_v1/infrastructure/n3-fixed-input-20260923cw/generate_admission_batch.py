"""Render a finite admission batch with one shared exclusive policy owner."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    base_path = Path("handoff/k8s/sgw01-ali-n3-fixedinput-20260923di.yaml")
    base = yaml.safe_load(base_path.read_text())
    container = base["spec"]["template"]["spec"]["containers"][0]
    environment = {row["name"]: row.get("value") for row in container["env"]}
    if (
        base["spec"]["backoffLimit"] != 0
        or container["resources"]["limits"]["nvidia.com/gpu"] != "1"
        or environment["CLAIM"] != "/data/users/ali/sgw-01/infrastructure/n3-fixedinput-20260923cw/request-owner/successor"
        or 'mkdir "$CLAIM"' not in container["args"][0]
    ):
        raise ValueError("base does not preserve one exclusive six-request owner")
    eligible = []
    for node in json.loads(args.nodes.read_text())["items"]:
        if (
            node["metadata"].get("labels", {}).get("nvidia.com/gpu.product") == "NVIDIA-A100-SXM4-80GB"
            and not node["spec"].get("taints")
            and not node["spec"].get("unschedulable")
            and int(node["status"]["allocatable"].get("nvidia.com/gpu", 0)) >= 1
            and any(c["type"] == "Ready" and c["status"] == "True" for c in node["status"]["conditions"])
            and node["metadata"]["name"] != "dcwipphai0055.edc.nam.gm.com"
        ):
            eligible.append(node["metadata"]["name"])
    nodes = sorted(eligible)[:8]
    if len(nodes) != 8:
        raise ValueError("expected eight eligible, previously untested 80GB A100 nodes")
    items, entries = [], []
    for index, node in enumerate(nodes):
        name = f"sgw01-ali-n3-fixedinput-20260923dk-{index:02d}"
        output = f"/data/users/ali/sgw-01/infrastructure/n3-fixedinput-20260923cw/run-dk-{index:02d}"
        job = copy.deepcopy(base)
        job["metadata"]["name"] = name
        job["metadata"]["labels"]["sgw-admission-batch"] = "20260923dk"
        job["spec"]["template"]["metadata"]["labels"]["sgw-admission-batch"] = "20260923dk"
        spec = job["spec"]["template"]["spec"]
        spec["nodeName"], spec["schedulerName"] = node, name
        for row in spec["containers"][0]["env"]:
            if row["name"] == "OUT":
                row["value"] = output
        items.append(job)
        entries.append({"job": name, "node": node, "output": output})
    plan = {
        "schema_version": "sgw-01-bounded-admission-batch-v1",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_id": "SGW-N3-FIXED-20260923CW",
        "authority": "SGW-OPS-002 and renewed user direction to find other GPUs while leaving SmolVLA running.",
        "maximum_candidate_gpu_leases": 8,
        "maximum_native_policy_owners": 1,
        "maximum_total_policy_requests": 6,
        "additional_request_allocation": 0,
        "behavioral_episodes": 0,
        "automatic_retries": 0,
        "entries": entries,
        "canonical_exclusive_claim": environment["CLAIM"],
        "selection_rule": "First candidate passing original asset/storage/physical-idle/UUID-lock gates and atomic mkdir owns the sole native start. Other candidates fail before model construction and must be released.",
        "nonpreemption": "Priority0 pre-bound nodes with dedicated schedulerName; no scheduler victim selection.",
        "claim_boundary": "Node readiness and advertised capacity are discovery facts, not physical idleness. No occupied GPU or unrelated workload may be used.",
        "existing_training_modified": False,
        "base_manifest": str(base_path),
        "base_manifest_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "node_inventory_sha256": hashlib.sha256(args.nodes.read_bytes()).hexdigest(),
    }
    for path, value in ((args.output, {"apiVersion": "v1", "kind": "List", "items": items}), (args.plan, plan)):
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps({"jobs": len(items), "maximum_policy_owners": 1, "maximum_total_requests": 6}))


if __name__ == "__main__":
    main()
