"""One authorized administrative boundary pause; never terminate a candidate."""
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

SOURCE = Path("/data/users/ali/sgw-01/source/68b9e100-bs")
WORKERS = Path("/data/users/ali/sgw-01/qualification/family-partition-20260923bt/workers")
OUTPUT = Path("/data/users/ali/sgw-01/infrastructure/administrative-pause-20260923cx")
worker_source = SOURCE / "experiments/workshops/spatial_grounding_v1/family_partition_worker.py"
if hashlib.sha256(worker_source.read_bytes()).hexdigest() != "778c6ad06bf270480f895535743b52cad0884d7373e2e6f0e6d93e0dd9881ab2":
    raise ValueError("deployed boundary-stop source differs")
sys.path.insert(0, str(SOURCE))
from experiments.workshops.spatial_grounding_v1.family_partition_worker import _locked
from experiments.workshops.spatial_grounding_v1.family_campaign_executor import _fsync_json

expected = {
    "bindings_sha256": "193561f1fed597e0a28b5fbb8fd951c6a0bb0ca8890e455990cff36c6a965e42",
    "config_sha256": "aa6413ae35cce7e8a1c9d5770a6b22260e9089d8301e45afe630f00ebf5d6ee9",
    "freeze_sha256": "55b2dbbce2376b1c297180630238b09a83e8f226697f8da6a1814d748eebefd3",
    "schema_version": "sgw-01-family-partition-worker-v1",
    "workers": 4,
}
with _locked(WORKERS):
    sentinel = WORKERS / "infrastructure-stop.json"
    if sentinel.exists():
        raise FileExistsError("preserve existing stop reason; administrative pause not applied")
    if json.loads((WORKERS / "partition-binding.json").read_bytes()) != expected:
        raise ValueError("owned partition binding differs")
    ranks = []
    for rank in range(4):
        root = WORKERS / str(rank)
        receipt = json.loads((root / "worker-receipt.json").read_bytes())
        completed = [json.loads(p.read_bytes())["slot"] for p in sorted((root / "completed").glob("*.json"))]
        claimed = [json.loads(p.read_bytes())["slot"] for p in (WORKERS / "slot-claims").glob("*.json")
                   if json.loads(p.read_bytes())["rank"] == rank]
        ranks.append({
            "rank": rank, "assigned": receipt["slot_order"], "completed_at_stop_request": completed,
            "current_claims_to_finish": [s for s in claimed if s not in completed],
            "not_yet_started_remain_pending": [s for s in receipt["slot_order"] if s not in claimed],
        })
    value = {
        "schema_version": "sgw-01-administrative-priority-pause-v1",
        "status": "administrative_priority_pause_requested_at_complete_candidate_boundary",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "SGW-OPS-003, renewed main-experiment priority; no mid-candidate kill",
        "user_approval": "Yes--prioritize the redesigned scene on our GPUs (Recommended)",
        "reason": "Prioritize the registered main-experiment N3 fixed-input/P progression. New B200 CW admission has no available device; existing ali B200 is occupied by unrelated training and is untouched.",
        "owned_job": "sgw01-ali-family-partition-20260923bt",
        "owned_job_uid": "7a70e5f9-b28a-41d2-88fc-7f13a18eff56",
        "binding": expected, "ranks": ranks,
        "infrastructure_failure_claimed": False, "scientific_failure_claimed": False,
        "model_requests": 0, "behavioral_episodes": 0,
        "worker_contract": "Existence checked under .partition.lock only before a new slot; current candidate completes before stopped_before_next_slot acknowledgement.",
        "legacy_wrapper_caveat": "Preserve original incomplete/infrastructure_stopped_partition launcher and collector labels. This separate receipt classifies the cause as ADMINISTRATIVE PRIORITY PAUSE, not infrastructure fault or model failure.",
        "reuse_gate": "Require worker boundary acknowledgement, process outcome and terminated Pod before re-admission; then independently check GPU idleness/ownership.",
        "resume_contract": "Retain completed outcomes and exact unstarted assignments. Never refill, rerun valid outcomes or count unfinished slots as failures.",
    }
    _fsync_json(OUTPUT / "activation.json", value)
    _fsync_json(sentinel, value)
    descriptor = os.open(WORKERS, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
print(json.dumps({"status": value["status"], "sentinel": str(sentinel),
                  "completed_counts": [len(r["completed_at_stop_request"]) for r in ranks],
                  "active_candidates_to_finish": [r["current_claims_to_finish"] for r in ranks]}, indent=2))
