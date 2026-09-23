"""Audit V3-D001 reset records without inventing a historical producer anchor."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.v3.pi05_phase_b.contract import canonical_json_bytes
from experiments.workshops.spatial_grounding_v1.historical_root_nonmatch import _validated_position
from tools.audit_sgw_historical_lineage import require
from tools.audit_sgw_pi05_reset_population import _bound_bytes
from tools.prove_sgw_r005_reset_bounds import read_bound, sha
from tools.vla_wam_v3_episode_schema import derive_initial_state_sha256

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b"
EXPORT = ROOT / "artifacts/workshops/spatial_grounding_v1/infrastructure/historical-pi05-stochastic-20260923ch"
RECOVERY_BINDING = {
    "bytes": 2168898, "sha256": "685e5a913678500e52743fb5b6b31364b0af97ce2a8d5070101ec2872dc5ebf4",
}
EVIDENCE_SHA = "bdbff6a3a18d1894158dc731df245173405c5f4508bf5da4a9031d5cb975309c"
RUNTIME_SHA = "e73fe7a0cc22db09fa8fdc0babf80dd8ad3280d0502285c6ad1c4d822c7fa532"
BRIDGE = "experiments/v3/pi05_stochastic_v3d001/robolab_bridge.py"


def compile_audit() -> dict:
    recovered = read_bound(EXPORT / "resets.json", RECOVERY_BINDING)
    require(recovered["schema_version"] == "sgw-01-pi05-stochastic-final-reset-export-v1",
            "reset recovery schema differs")
    evidence_raw = (COHORT / "results/v3d001/evidence_manifest.json").read_bytes()
    require(sha(evidence_raw) == EVIDENCE_SHA, "final evidence manifest differs")
    evidence = json.loads(evidence_raw)
    episode_binding = evidence["files"]["pi05_v3d001_episodes.jsonl"]
    require(all(recovered["compact_episodes"][key] == episode_binding[key] for key in ("bytes", "sha256")),
            "source compact episode binding differs")
    raw = _bound_bytes(COHORT / "results/v3d001/pi05_v3d001_episodes.jsonl", episode_binding)
    episodes = [json.loads(line) for line in raw.splitlines()]
    episode_map = {row["registered_cell_id"]: row for row in episodes}
    release_raw = (COHORT / "releases/v3d001/release_manifest.json").read_bytes()
    require(sha(release_raw) == evidence["source_release"]["release_manifest_sha256"], "release differs")
    release = json.loads(release_raw)
    queue_binding = release["files"][0]
    raw = _bound_bytes(COHORT / "releases/v3d001/pi05_v3d001_stochastic_cells.jsonl", queue_binding)
    require(sha(raw) == evidence["source_release"]["queue_sha256"], "queue differs")
    queue = [json.loads(line) for line in raw.splitlines()]
    queue_map = {row["cell_id"]: row for row in queue}
    records = recovered["entries"]
    require(len(records) == len(episodes) == len(episode_map) == len(queue) == len(queue_map) == 432
            and len({r["registered_cell_id"] for r in records}) == 432
            and {r["registered_cell_id"] for r in records} == set(episode_map) == set(queue_map),
            "final-cell population differs")

    runtime = read_bound(EXPORT / "runtime_identity.json", recovered["runtime_identity"])
    require(recovered["runtime_identity"]["sha256"] == RUNTIME_SHA, "runtime identity differs")
    source_maps = {**runtime["adapter_source_sha256"], **runtime["frozen_v2_source_sha256"]}
    require(len(source_maps) == 8 and BRIDGE not in source_maps,
            "this bounded missing-producer finding no longer matches the runtime source maps")
    eligibility = read_bound(EXPORT / "eligibility_report.json", release["files"][2])
    probe_manifest = read_bound(EXPORT / "evidence_manifest.json", release["files"][3])
    require(probe_manifest["files"][0] == release["files"][2], "eligibility report binding differs")
    policy_raw = (EXPORT / "policy_runtime_attestation.json").read_bytes()
    require(sha(policy_raw) == eligibility["runtime_attestation"]["sha256"],
            "policy runtime attestation differs")
    policy = json.loads(policy_raw)
    body = {key: value for key, value in policy.items() if key != "runtime_attestation_sha256"}
    require(sha(canonical_json_bytes(body)) == policy["runtime_attestation_sha256"]
            == eligibility["runtime_attestation"]["runtime_attestation_sha256"],
            "policy runtime semantic identity differs")
    require(policy["server_source_sha256"] == source_maps["experiments/pi05_current_stack/v2a010_serve_policy.py"],
            "nested policy attestation server binding differs")

    points, initial_hashes, original_paths = [], set(), set()
    for row in records:
        cell_id = row["registered_cell_id"]
        episode, cell = episode_map[cell_id], queue_map[cell_id]
        require(row["raw_behavioral_result_valid"] is True
                and row["runtime_identity"]["sha256"] == cell["source_phase_a_runtime_identity_sha256"] == RUNTIME_SHA,
                "cell validity or runtime differs")
        require(row["environment_seed"] == episode["environment_seed"] == cell["environment_seed"]
                and row["requested_relation"] == episode["requested_relation"] == cell["requested_relation"]
                and row["sampling_index"] == episode["shared_policy_sampling_seed_index"]
                == cell["shared_policy_sampling_seed_index"], "registered condition differs")
        require(row["raw_episode"] == episode["raw_episode_jsonl"]
                and row["state_capture"] == episode["state_capture"]
                and row["raw_and_capture_samples_equal"] is True, "raw capture binding differs")
        frame, initial = row["measurement_frame"], row["initial_state_sha256"]
        require(frame == "robot_base_object_minus_reference_xyz_m"
                and initial == row["raw_initial_state_sha256"] == episode["initial_state_sha256"]
                == cell["source_phase_a_initial_state_sha256"]
                == derive_initial_state_sha256({"measurement_frame": frame, "steps": [row["initial_sample"]]}),
                "registered initial-state digest differs")
        attestations = row["reset_attestations"]
        require(len(attestations) == 2
                and [a["binding"] for a in attestations] == episode["pre_action_reset_attestations"],
                "reset attestation selection differs")
        for index, attestation in enumerate(attestations):
            binding, value = attestation["binding"], attestation["value"]
            original = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
            require(len(original) == binding["bytes"] and sha(original) == binding["sha256"],
                    "original reset attestation bytes differ")
            require(binding["path"] not in original_paths, "duplicate reset attestation path")
            original_paths.add(binding["path"])
            expected = {
                "schema_version": "vla-wam-shared-v3d001-pi05-reset-attestation-v1",
                "registered_cell_id": cell_id, "attempt_id": row["attempt_id"],
                "initial_state_sha256": initial, "pre_action_reset_index": index,
                "actions_executed_before_next_reset": 0,
            }
            require(all(type(value.get(key)) is type(wanted) and value[key] == wanted
                        for key, wanted in expected.items()), "attested reset identity differs")
            sample = value["sample"]
            require(type(sample["action_step"]) is int and sample["action_step"] == 0
                    and sample == row["initial_sample"]
                    and derive_initial_state_sha256({"measurement_frame": frame, "steps": [sample]}) == initial,
                    "retained reset sample differs")
            cube, bowl = (_validated_position(sample[key]) for key in ("object_xyz", "reference_xyz"))
            require(cube is not None and bowl is not None, "invalid root vectors")
            points.append((cube, bowl))
            initial_hashes.add(initial)
    distances = [math.dist(*point) for point in points]
    return {
        "schema_version": "sgw-01-pi05-stochastic-final-reset-audit-v1",
        "producer_sha256": sha(Path(__file__).read_bytes()), "recovery": RECOVERY_BINDING,
        "final_evidence_manifest_sha256": EVIDENCE_SHA,
        "release_manifest_sha256": sha(release_raw), "queue_sha256": queue_binding["sha256"],
        "runtime_identity_sha256": RUNTIME_SHA,
        "nested_policy_runtime_sha256": sha(policy_raw),
        "runtime_bound_source_files": source_maps,
        "retained_current_source": recovered["retained_current_source"],
        "final_cell_count": 432, "original_reset_attestations_reconstructed": len(original_paths),
        "registered_conditions": len({(r["environment_seed"], r["requested_relation"]) for r in records}),
        "distinct_initial_state_hashes": len(initial_hashes),
        "distinct_numerical_root_pairs": len(set(points)),
        "declared_frame_root_separation_range_m": [min(distances), max(distances)],
        "complete_final_manifest_cell_selection": True,
        "stochastic_bridge_in_recorded_runtime_source_maps": False,
        "independent_historical_producer_anchor_established": False,
        "cross_frame_comparison_qualified": False, "proved_historical_nonmatches_added": 0,
        "historical_population_coverage_complete": False,
        "new_model_requests": 0, "new_behavioral_episodes": 0, "release_permitted": False,
        "claim_boundary": (
            "All 864 named pre-action reset attestations from the 432 final V3-D001 cells are "
            "byte-reconstructed and reconciled with original raw/capture projections and registration. "
            "They contain one numerical root pair, not 864 independent layouts. The inherited "
            "runtime binds Phase-A sources; the nested stochastic policy attestation binds the server, "
            "not the stochastic simulator bridge. A current retained checkout or launch directory "
            "name does not supply its independently historical producer hash. No cross-frame "
            "nonmatches, constructor/settle/preflight/infrastructure coverage or SGW release."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compile_audit()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
