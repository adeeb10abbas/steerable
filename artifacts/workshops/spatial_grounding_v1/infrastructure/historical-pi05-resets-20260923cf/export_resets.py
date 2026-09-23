"""Read only the final V3-B002 episode file and its 108 named reset attestations."""
import hashlib
import json
from pathlib import Path
import sys


PATH = Path(
    "/data/users/ali/vla_wam/raw/v3b/pi05_current_stack/position_reflection_v3b002/"
    "compiled_results_attempt04/pi05_v3b002_episodes.jsonl"
)
SHA = "7b89287a2b75e40cffc97cd6d2fea58c4189a09d2dc5a0fb9a6427df3d726e70"
raw = PATH.read_bytes()
if len(raw) != 7552700 or hashlib.sha256(raw).hexdigest() != SHA:
    raise ValueError("final episode file differs from the committed output manifest")
episodes = [json.loads(line) for line in raw.splitlines()]
expected = {
    f"v3b002:pi05:seed{seed}:{arm}:{relation}"
    for seed in range(9400, 9427)
    for arm in ("control", "position_mirrored")
    for relation in ("left", "right")
}
if len(episodes) != 108 or {row["registered_cell_id"] for row in episodes} != expected:
    raise ValueError("final episode population differs from the registered 108 cells")
entries = []
seen = set()
for episode in episodes:
    binding = episode["source_artifacts"]["reset_attestation"]
    path = Path(binding["path"])
    path.relative_to(PATH.parents[1])
    if path in seen:
        raise ValueError("reset attestation reused by multiple cells")
    seen.add(path)
    data = path.read_bytes()
    if len(data) != binding["bytes"] or hashlib.sha256(data).hexdigest() != binding["sha256"]:
        raise ValueError(f"historical reset attestation changed: {path}")
    attestation = json.loads(data)
    entries.append({
        "registered_cell_id": episode["registered_cell_id"],
        "arm": episode["phase_b_arm"],
        "environment_seed": episode["environment_seed"],
        "relation": episode["requested_relation"],
        "initial_state_sha256": episode["initial_state_sha256"],
        "measurement_frame": episode["measurement_frame"],
        "sample": episode["steps"][0],
        "reset_attestation": {"binding": binding, "value": attestation},
    })

json.dump({
    "schema_version": "sgw-01-pi05-v3b002-final-reset-export-v1",
    "source_episodes": {"path": str(PATH), "sha256": SHA, "bytes": len(raw)},
    "entries": sorted(entries, key=lambda row: row["registered_cell_id"]),
    "model_requests": 0,
    "behavioral_episodes": 0,
    "claim_boundary": (
        "108 named post-settle initial states for the final accepted behavioral cohort, with their "
        "original reset attestations rehashed. Not constructor, settle-window, model-blind-preflight "
        "or infrastructure-attempt population coverage; no new inference or cross-frame exclusion."
    ),
}, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
sys.stdout.write("\n")
