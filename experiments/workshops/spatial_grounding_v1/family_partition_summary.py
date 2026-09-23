"""Read-only compact accounting for a frozen family partition collection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .family_campaign_verifier import verify_design
from .family_partition_worker import (
    SCHEMA as WORKER_SCHEMA, _campaigns, _config_bindings, _freeze, _json, _sha256, partition_slots,
)

SCHEMA = "sgw-01-family-partition-summary-v1"


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("refusing to overwrite partition summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _counts(freeze: Mapping[str, Any], smoke: list[Mapping[str, Any]], results: list[Mapping[str, Any]]) -> dict[str, Any]:
    families = {}
    for family_row in freeze["families"]:
        family = family_row["family"]
        proposed = family_row["proposed_slots"]
        geometric = family_row["geometric_rejections"]
        smoke_rows = sum(row["family"] == family for row in smoke)
        family_results = [row for row in results if row["family"] == family]
        smoke_results = [row for row in smoke if row["family"] == family]
        families[family] = {
            "proposed": proposed, "geometric_rejections": geometric, "smoke": smoke_rows,
            "remaining": sum(row["family"] == family for row in freeze["remaining_capture_eligible_slots"]),
            "physical_accepted": sum(row["outcome"] == "full_six" and row["accepted"] for row in family_results + smoke_results),
            "physical_rejected": sum(row["outcome"] in {"full_six", "terminal_prefix"} and not row["accepted"] for row in family_results + smoke_results),
            "infrastructure_invalid": sum(row["outcome"] == "infrastructure_invalid" for row in family_results),
            "partial": sum(row["outcome"] == "partial" for row in family_results),
            "unstarted": sum(row["outcome"] == "unstarted" for row in family_results),
            "by_side": {
                side: {
                    "physical_accepted": sum(row["outcome"] == "full_six" and row["accepted"] and row.get("side") == side for row in family_results + smoke_results),
                    "physical_rejected": sum(row["outcome"] in {"full_six", "terminal_prefix"} and not row["accepted"] and row.get("side") == side for row in family_results + smoke_results),
                }
                for side in sorted({row.get("side") for row in family_results + smoke_results if isinstance(row.get("side"), str)})
            },
        }
    return families


def _smoke(
    *, config: Mapping[str, Any], freeze: Mapping[str, Any], campaigns: Mapping[str, Path],
    output: Path, verifier: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    entries = config.get("smoke_evidence")
    expected = {(row["family"], row["slot_index"]): row for row in freeze["native_smoke_slots"]}
    if not isinstance(entries, list) or len(entries) != 4:
        raise ValueError("partition config lacks four smoke roots")
    records = []
    for row in entries:
        if not isinstance(row, Mapping):
            raise ValueError("smoke entry is malformed")
        family, index, design = row.get("family"), row.get("slot_index"), row.get("design_id")
        root, receipt = Path(row.get("root", "")), Path(row.get("verification", ""))
        expected_file_sha, expected_digest = row.get("verification_file_sha256"), row.get("verification_sha256")
        frozen = expected.get((family, index))
        if (not isinstance(frozen, Mapping) or frozen.get("design_id") != design or not root.is_dir() or not receipt.is_file()
                or not isinstance(expected_file_sha, str) or not isinstance(expected_digest, str)
                or _sha256(receipt) != expected_file_sha):
            raise ValueError("smoke root/receipt differs from frozen config")
        value = _json(receipt)
        if value.get("verification_sha256") != expected_digest:
            raise ValueError("smoke receipt digest differs from config")
        recheck = output.parent / "smoke-rechecks" / f"{family.lower()}-{index:03d}.json"
        verified = verifier(campaign_path=campaigns[family], design_id=design, root=root, output=recheck)
        if verified.get("verification_sha256") != expected_digest:
            raise ValueError("smoke raw evidence differs from authoritative verifier")
        qualification_path = root / "qualification.json"
        qualification = _json(qualification_path)
        if qualification.get("status") not in {
            "accepted_model_blind_fixture_candidate", "rejected_model_blind_fixture_candidate",
        }:
            raise ValueError("smoke qualification outcome is not physically classified")
        records.append({
            **dict(row), "side": frozen.get("side"), "outcome": "terminal_prefix" if isinstance(verified.get("physical_geometry_rejection"), Mapping)
            else "full_six",
            "accepted": qualification["status"] == "accepted_model_blind_fixture_candidate",
            "verification": _record(receipt), "recheck": _record(recheck),
            "qualification": _record(qualification_path),
        })
    if {(row["family"], row["slot_index"]) for row in records} != set(expected):
        raise ValueError("smoke root coverage differs from freeze")
    return records


def _retained_trial_bindings(*, slot_root: Path, verified: Mapping[str, Any]) -> list[dict[str, Any]]:
    retained = []
    for row in verified.get("trials", []):
        receipt = Path(row.get("path", ""))
        if not receipt.is_file():
            raise ValueError("authoritative verification lacks retained trial receipt")
        value = _json(receipt)
        trial = receipt.parent
        reset = trial / "reset.json"
        video = trial / "viewport.mp4"
        retained.append({
            "goal_sign": row.get("goal_sign"), "reset_index": row.get("reset_index"),
            "trial": _record(receipt), "reset": _record(reset) if reset.is_file() else None,
            "viewport_video": _record(video) if video.is_file() else None,
            "render_only_warmup": value.get("reset_receipt", {}).get("render_only_warmup"),
        })
    return retained


def compile_summary(
    *, config_path: Path, workers_root: Path, output: Path,
    verifier: Callable[..., dict[str, Any]] = verify_design,
) -> dict[str, Any]:
    """Compile current state without reclassifying missing/partial evidence as failures."""
    config = _json(config_path)
    freeze_path = Path(config.get("freeze_receipt", ""))
    freeze = _freeze(freeze_path)
    campaigns = _campaigns(config, freeze)
    bindings = _config_bindings(config, freeze_path, campaigns)
    config_sha = _sha256(config_path)
    shared = _json(workers_root / "partition-binding.json")
    if (
        shared.get("schema_version") != WORKER_SCHEMA or shared.get("config_sha256") != config_sha
        or shared.get("bindings_sha256") != _digest(bindings) or shared.get("workers") != 4
    ):
        raise ValueError("shared worker identity differs from the hash-bound partition config")
    smoke = _smoke(config=config, freeze=freeze, campaigns=campaigns, output=output, verifier=verifier)
    all_slots = {(row["family"], row["slot_index"]): row for row in freeze["remaining_capture_eligible_slots"]}
    claims_dir = workers_root / "slot-claims"
    claimed_slots = set()
    if claims_dir.exists():
        for path in claims_dir.glob("*.json"):
            claim = _json(path)
            slot = claim.get("slot")
            if not isinstance(slot, Mapping):
                raise ValueError("slot claim is malformed")
            key = (slot.get("family"), slot.get("slot_index"))
            if key not in all_slots or slot != all_slots[key] or key in claimed_slots:
                raise ValueError("slot claim is duplicated or outside frozen registry")
            claimed_slots.add(key)
    results: list[dict[str, Any]] = []
    incomplete = (workers_root / "infrastructure-stop.json").exists()
    rank_claims = set()
    for rank in range(4):
        rank_root = workers_root / str(rank)
        claim = workers_root / "rank-claims" / f"rank-{rank}.json"
        expected = partition_slots(freeze, rank=rank, workers=4)
        if not rank_root.exists() or not claim.exists():
            incomplete = True
            for slot in expected:
                results.append({**slot, "outcome": "unstarted"})
            continue
        rank_claim = _json(claim)
        if (
            rank_claim.get("rank") != rank or rank_claim.get("root") != str(rank_root.resolve())
            or rank_claim.get("config_sha256") != config_sha or rank_claim.get("bindings_sha256") != _digest(bindings)
        ):
            raise ValueError("rank claim differs from worker root")
        if rank in rank_claims:
            raise ValueError("duplicate rank claim")
        rank_claims.add(rank)
        worker_receipt = _json(rank_root / "worker-receipt.json")
        if (
            worker_receipt.get("schema_version") != WORKER_SCHEMA or worker_receipt.get("rank") != rank
            or worker_receipt.get("slot_order") != expected or worker_receipt.get("bindings") != bindings
            or not isinstance(worker_receipt.get("smoke_verifications"), list)
        ):
            raise ValueError("worker receipt differs from frozen partition binding")
        completion_path = rank_root / "worker-completion.json"
        completed_slots = None
        if completion_path.is_file():
            completion = _json(completion_path)
            if completion.get("schema_version") != WORKER_SCHEMA or completion.get("rank") != rank:
                raise ValueError("worker completion receipt differs from rank")
            completed_slots = completion.get("completed_slots")
            if not isinstance(completed_slots, list) or any(row not in expected for row in completed_slots):
                raise ValueError("worker completion has an unexpected slot list")
        completed_paths = set((rank_root / "completed").glob("*.json")) if (rank_root / "completed").exists() else set()
        recorded_completed = []
        for slot in expected:
            key = (slot["family"], slot["slot_index"])
            claim_path = claims_dir / f"{slot['family'].lower()}-{slot['slot_index']:03d}.json"
            done_path = rank_root / "completed" / f"{slot['family'].lower()}-{slot['slot_index']:03d}.json"
            slot_root = rank_root / "slots" / f"{slot['family'].lower()}-{slot['slot_index']:03d}"
            if not claim_path.exists():
                incomplete = True
                results.append({**slot, "outcome": "unstarted"})
                continue
            slot_claim = _json(claim_path)
            if slot_claim.get("rank") != rank or slot_claim.get("slot") != slot or slot_claim.get("bindings_sha256") != _digest(bindings):
                raise ValueError("slot claim is duplicated or differs from deterministic partition")
            if not done_path.exists():
                failed = slot_root / "executor-failure.json"
                if failed.is_file():
                    failure = _json(failed)
                    if not isinstance(failure.get("error"), str) or not failure["error"]:
                        raise ValueError("terminal infrastructure receipt is malformed")
                    results.append({**slot, "outcome": "infrastructure_invalid", "claim": _record(claim_path),
                                    "infrastructure_receipt": _record(failed)})
                else:
                    incomplete = True
                    results.append({**slot, "outcome": "partial", "claim": _record(claim_path)})
                continue
            done = _json(done_path)
            if done.get("schema_version") != WORKER_SCHEMA or done.get("slot") != slot or not isinstance(done.get("result"), Mapping):
                raise ValueError("completed record differs from claimed slot")
            if done_path not in completed_paths:
                raise ValueError("completed file set differs from deterministic slot")
            completed_paths.remove(done_path)
            slot_root = Path(done.get("slot_root", ""))
            canonical_root = rank_root / "slots" / f"{slot['family'].lower()}-{slot['slot_index']:03d}"
            if slot_root.resolve() != canonical_root.resolve():
                raise ValueError("completed slot root differs from canonical rank evidence root")
            outcome = done["result"].get("status")
            if outcome not in {
                "externally_verified_candidate_slot_not_fixture_or_behavioral_release",
                "physical_geometry_rejection_accounted_slot_no_refill",
            }:
                raise ValueError("completion status cannot establish a physical outcome")
            verification_path = output.parent / "rechecks" / f"{slot['family'].lower()}-{slot['slot_index']:03d}.json"
            verified = verifier(campaign_path=campaigns[slot["family"]], design_id=slot["design_id"],
                                root=slot_root, output=verification_path)
            if verified.get("status") != "verified_evidence_not_fixture_release":
                raise ValueError("authoritative slot verification failed")
            if done["result"].get("verification_sha256") != verified.get("verification_sha256"):
                raise ValueError("completion verification chain differs from authoritative evidence")
            executor = slot_root / "executor-receipt.json"
            if not executor.is_file():
                raise ValueError("completed slot lacks executor receipt")
            executor_value = _json(executor)
            if executor_value.get("status") != outcome:
                raise ValueError("executor receipt differs from completed outcome")
            if completed_slots is not None and slot not in completed_slots:
                raise ValueError("completed slot differs from worker completion receipt")
            recorded_completed.append(slot)
            early = verified.get("physical_geometry_rejection")
            if outcome.startswith("physical_geometry") != isinstance(early, Mapping):
                raise ValueError("completion outcome differs from authoritative verification")
            qualification = _json(slot_root / "qualification.json")
            results.append({
                **slot, "outcome": "terminal_prefix" if isinstance(early, Mapping) else "full_six",
                "accepted": qualification.get("status") == "accepted_model_blind_fixture_candidate",
                "completion": _record(done_path), "claim": _record(claim_path),
                "verification": _record(verification_path), "qualification": _record(slot_root / "qualification.json"),
                "retained_trial_bindings": _retained_trial_bindings(slot_root=slot_root, verified=verified),
            })
        if completed_paths:
            raise ValueError("unexpected or duplicate completion files are present")
        if completed_slots is not None and completed_slots != recorded_completed:
            raise ValueError("worker completion slot list differs from completed records")
    if set((row["family"], row["slot_index"]) for row in results) != set(all_slots):
        raise ValueError("partition collection has duplicate or missing slot accounting")
    value = {
        "schema_version": SCHEMA, "status": "incomplete" if incomplete else "complete",
        "release_permitted": False, "fixture_release_permitted": False, "model_release_permitted": False,
        "freeze": _record(freeze_path), "config": _record(config_path), "workers_root": str(workers_root.resolve()),
        "smoke": smoke, "results": results, "families": _counts(freeze, smoke, results),
        "stop_sentinel": _record(workers_root / "infrastructure-stop.json")
        if (workers_root / "infrastructure-stop.json").is_file() else None,
        "raw_evidence_retained_on_pvc": True,
    }
    value["summary_sha256"] = _digest(value)
    _write(output, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compile_summary(config_path=args.config, workers_root=args.workers_root, output=args.output)))


if __name__ == "__main__":
    main()
