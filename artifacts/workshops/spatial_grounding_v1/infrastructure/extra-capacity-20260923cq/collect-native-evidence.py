"""Read-only compact export of SGW-ENG-008's retained native evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data/users/ali/sgw-01/qualification/paper-engineering-20260923cs")
RECHECK = Path("/data/users/ali/sgw-01/infrastructure/paper-engineering-recheck-20260923cu")
ATTEMPT = "SGW-ENG-008-LAT-057"


def record(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def load(path):
    return json.loads(path.read_bytes())


verification = load(RECHECK / "qualification-verification.json")
assert verification["candidate_id"] == ATTEMPT
assert len(verification["checks"]) == 6
assert verification["model_requests"] == verification["behavioral_episodes"] == 0
capture = load(ROOT / "evidence/translated/capture.json")
inventory = capture["collision_geometry_local_bounds"]
assert inventory["available"] is True
body_names = sorted(inventory["body_prim_paths"])
all_body_names = sorted(capture["robot_snapshot"]["body_frames"]["bodies"])
trial_root = ROOT / f"evidence/0-{ATTEMPT}/result"
trials = []
for check in verification["checks"]:
    sign, repeat = check["goal_sign"], check["reset_index"]
    trial = trial_root / f"trials/goal-{sign:+d}/reset-{repeat}"
    receipt = load(trial / "trial.json")
    assert record(trial / "trial.json")["sha256"] == check["trial_receipt_sha256"]
    assert receipt["score"] == check["score"]
    clearance_minima = {}
    clearance_unavailable_steps = []
    contact_rows = {}
    for step in range(451):
        path = trial / f"state-{step:04d}.json"
        binding = record(path)
        assert {k: binding[k] for k in ("bytes", "sha256")} == receipt["files"][path.name]
        state = load(path)
        robot = state["raw_snapshot"]["robot_snapshot"]
        clearance = robot["engineering_clearance"]
        if clearance["available"] is not True:
            clearance_unavailable_steps.append({"step": step, "reason": clearance.get("reason")})
        else:
            assert set(clearance["separations"]) == {
                f"{body}__{obj}" for body in body_names for obj in ("bowl", "banana", "table")
            }
            for key, row in clearance["separations"].items():
                distance = row["aabb_euclidean_separation_m"]
                assert np.isfinite(distance) and distance >= 0
                summary = clearance_minima.setdefault(key, {
                    "minimum_separation_m": distance, "first_minimum_step": step, "overlap_steps": 0,
                })
                if distance < summary["minimum_separation_m"]:
                    summary.update(minimum_separation_m=distance, first_minimum_step=step)
                summary["overlap_steps"] += int(row["aabb_overlap"])
        for name, row in robot["gripper_contact_forces"].items():
            summary = contact_rows.setdefault(name, {
                "available_steps": 0, "unavailable_steps": [], "nonzero_force_steps": 0,
                "maximum_vector_norm_n": None,
            })
            if row["available"] is not True:
                summary["unavailable_steps"].append({"step": step, "reason": row.get("reason")})
                continue
            forces = np.asarray(row["force_matrix_world_n"], dtype=float)
            assert forces.size and forces.shape[-1] == 3 and np.isfinite(forces).all()
            largest = float(np.linalg.norm(forces, axis=-1).max())
            summary["available_steps"] += 1
            summary["nonzero_force_steps"] += int(largest > 0)
            previous = summary["maximum_vector_norm_n"]
            summary["maximum_vector_norm_n"] = largest if previous is None else max(previous, largest)
    video = receipt["viewport_video"]
    assert {k: record(Path(video["path"]))[k] for k in ("bytes", "sha256")} == {
        k: video[k] for k in ("bytes", "sha256")
    }
    reset = receipt["reset_receipt"]
    trials.append({
        "goal_sign": sign, "reset_index": repeat, "score": check["score"], "passed": check["passed"],
        "reset_error": check["reset_error"], "trial_receipt": record(trial / "trial.json"),
        "state_count": 451, "command_count": 450, "viewport_video": video,
        "warmup_video": reset["render_only_warmup"]["viewport_video"],
        "clearance_minima": clearance_minima, "clearance_unavailable_steps": clearance_unavailable_steps,
        "gripper_filtered_contact_summary": contact_rows,
    })

records = [record(ROOT / name) for name in (
    "process_outcome.json", "evidence/infrastructure-failure.json",
    "evidence/reference/capture.json", "evidence/reference/capture-verification.json",
    "evidence/translated/capture.json", "evidence/translated/capture-verification.json",
    "evidence/registration-binding.json", "evidence/engineering-proposal.json",
    "evidence/translation-calculation.json", "evidence/robot-camera-comparison.json",
    "evidence/realization.json", "evidence/footprint-screen.json",
    "evidence/reference/render_diagnostic/render_only.mp4",
    "evidence/translated/render_diagnostic/render_only.mp4",
)]
records.extend(record(RECHECK / name) for name in ("intent.json", "qualification-verification.json"))
print(json.dumps({
    "schema_version": "sgw-01-native-engineering-evidence-v1", "attempt_id": ATTEMPT,
    "raw_root": str(ROOT), "recheck_root": str(RECHECK),
    "source_commit": "4f04e31ee0219dae052bba2ef95b9e2010dbceeb",
    "native_process_status": load(ROOT / "process_outcome.json")["status"],
    "verification_status": verification["status"],
    "passed_checks": verification["passed_checks"], "scripted_trials": 6,
    "scripted_actions": 2700, "recorded_control_states": 2706,
    "decoded_trial_frames": sum(row["decoded_trial_frames"] for row in verification["checks"]),
    "decoded_warmup_frames": sum(row["decoded_warmup_frames"] for row in verification["checks"]),
    "verified_trial_files": sum(row["verified_trial_files"] for row in verification["checks"]),
    "collision_inventory_body_names": body_names,
    "native_body_names_not_in_collision_inventory": sorted(set(all_body_names) - set(body_names)),
    "trials": trials, "records": records, "model_requests": 0, "behavioral_episodes": 0,
    "release_permitted": False, "outside_all_frozen_candidate_pools": True,
    "claim_boundary": "One independently verified engineering layout, not a benchmark or model release. "
                      "AABB bounds cover only the listed arm links; gripper collision geometry is not "
                      "certified by these bounds. Overlap is not a collision and missing evidence is not "
                      "zero. Filtered contact forces are recorded measurements, not full safety closure. "
                      "The original verifier-ordering failure and all raw files remain unchanged.",
}, indent=2, sort_keys=True))
