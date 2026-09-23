"""Reconcile retained BT boundary receipts without rescoring scientific outcomes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compile_boundaries(activation, archive, pods):
    members = {}
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            if member.isfile():
                require(member.name not in members, "duplicate archive member")
                members[member.name] = source.extractfile(member).read()

    def load(name):
        return json.loads(members[name])

    require(load("infrastructure-stop.json") == activation, "administrative sentinel changed")
    require(load("partition-binding.json") == activation["binding"], "partition binding changed")
    require(len(pods["items"]) == 4, "expected exactly four original Pods")
    ranks, completed_names, claim_names = [], set(), set()
    retained = {"infrastructure-stop.json", "partition-binding.json"}
    for initial in activation["ranks"]:
        rank = initial["rank"]
        receipt_name = f"{rank}/worker-receipt.json"
        completion_name = f"{rank}/worker-completion.json"
        receipt, completion = load(receipt_name), load(completion_name)
        require(receipt["rank"] == completion["rank"] == rank, "rank mismatch")
        binding_digest = digest(json.dumps(receipt["bindings"], sort_keys=True, separators=(",", ":")).encode())
        require(binding_digest == activation["binding"]["bindings_sha256"], "worker source binding changed")
        require(receipt["slot_order"] == initial["assigned"], "frozen assignments changed")
        require(completion["status"] == "stopped_before_next_slot" and completion["stopped_by_peer"] is True,
                "worker did not acknowledge its candidate boundary")
        completed = completion["completed_slots"]
        require(completed == initial["assigned"][:len(completed)], "completion is not an assigned prefix")
        expected_done = initial["completed_at_stop_request"] + initial["current_claims_to_finish"]
        key = lambda row: (row["family"], row["slot_index"])
        require(sorted(completed, key=key) == sorted(expected_done, key=key),
                "current candidate was not finished or a later candidate started")
        pending = initial["assigned"][len(completed):]
        require(pending == initial["not_yet_started_remain_pending"], "pending assignments changed")
        for slot in completed:
            suffix = f"{slot['family'].lower()}-{slot['slot_index']:03d}.json"
            completed_name, claim_name = f"{rank}/completed/{suffix}", f"slot-claims/{suffix}"
            done, claim = load(completed_name), load(claim_name)
            require(done["slot"] == claim["slot"] == slot and claim["rank"] == rank, "slot ownership mismatch")
            require(claim["bindings_sha256"] == binding_digest, "slot source binding mismatch")
            result = done["result"]
            require(result["status"] in {
                "externally_verified_candidate_slot_not_fixture_or_behavioral_release",
                "physical_geometry_rejection_accounted_slot_no_refill",
            }, "nonterminal slot outcome")
            require(result["model_request_count"] == result["behavioral_episode_count"] == 0
                    and result["release_permitted"] is False, "unexpected inference or release")
            require(claim_name not in claim_names, "duplicate slot")
            completed_names.add(completed_name)
            claim_names.add(claim_name)
        matches = [
            pod for pod in pods["items"]
            if pod["metadata"]["labels"].get("batch.kubernetes.io/job-completion-index") == str(rank)
        ]
        require(len(matches) == 1, "missing or duplicate rank Pod")
        pod = matches[0]
        require(any(owner["uid"] == activation["owned_job_uid"] and owner["name"] == activation["owned_job"]
                    for owner in pod["metadata"]["ownerReferences"]), "Pod has different owner")
        statuses = pod["status"]["containerStatuses"]
        require(pod["status"]["phase"] in {"Failed", "Succeeded"} and statuses
                and all("terminated" in item["state"] for item in statuses), "Pod still has an active container")
        ranks.append({
            "rank": rank, "pod": pod["metadata"]["name"], "pod_uid": pod["metadata"]["uid"],
            "node": pod["spec"]["nodeName"], "pod_phase": pod["status"]["phase"],
            "terminated_containers": [{"name": item["name"], **item["state"]["terminated"]} for item in statuses],
            "boundary_status": completion["status"],
            "completed_operational_slots": len(completed),
            "finished_current_candidates": initial["current_claims_to_finish"],
            "pending_unstarted_slots": pending,
        })
        retained.update({receipt_name, completion_name})
    actual_completed = {name for name in members if "/completed/" in name}
    actual_claims = {name for name in members if name.startswith("slot-claims/")}
    require(actual_completed == completed_names and actual_claims == claim_names, "extra or unfinished claimed slots")
    summary = {
        "schema_version": "sgw-01-administrative-boundary-completion-v1",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "all_four_owned_workers_stopped_after_current_complete_candidate",
        "cause": "ADMINISTRATIVE PRIORITY PAUSE",
        "activation": {"path": "activation.json", "recorded_at_utc": activation["recorded_at_utc"]},
        "raw_worker_root": "/data/users/ali/sgw-01/qualification/family-partition-20260923bt/workers",
        "owned_job": activation["owned_job"], "owned_job_uid": activation["owned_job_uid"],
        "binding": activation["binding"], "ranks": ranks,
        "assigned_slots": sum(len(row["assigned"]) for row in activation["ranks"]),
        "completed_operational_slots": len(completed_names),
        "unstarted_pending_slots": sum(len(row["pending_unstarted_slots"]) for row in ranks),
        "unfinished_claimed_slots": 0, "active_worker_containers": 0,
        "model_requests": 0, "behavioral_episodes": 0, "release_permitted": False,
        "claim_boundary": "Operational terminal receipts only. BU independently rechecks raw scientific evidence and videos.",
        "legacy_wrapper_caveat": activation["legacy_wrapper_caveat"],
        "reuse_gate": "Terminated ownership is verified; any new admission still requires fresh physical GPU idleness and ownership checks.",
        "source_archive_sha256": digest(archive.read_bytes()),
        "raw_receipt_manifest": [
            {"path": name, "sha256": digest(data), "bytes": len(data)}
            for name, data in sorted(members.items())
        ],
    }
    return summary, {name: members[name] for name in retained}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--pods", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, retained = compile_boundaries(
        json.loads(args.activation.read_text()), args.archive, json.loads(args.pods.read_text()),
    )
    summary["pod_snapshot_sha256"] = digest(args.pods.read_bytes())
    args.output.mkdir()
    for name, data in retained.items():
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    with (args.output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: summary[key] for key in (
        "status", "assigned_slots", "completed_operational_slots", "unstarted_pending_slots",
        "unfinished_claimed_slots", "active_worker_containers",
    )}))


if __name__ == "__main__":
    main()
