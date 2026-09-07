#!/usr/bin/env python3
"""Build the repaired horizontal C1/C3/C4 inventory and setup artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.horizontal_geometry_repair import (  # noqa: E402
    COHORT,
    FIXTURE_VERSION,
)
FREEZE_SPEC = importlib.util.spec_from_file_location(
    "build_online_correction_v4_freeze",
    ROOT / "tools/build_online_correction_v4_freeze.py",
)
freeze_builder = importlib.util.module_from_spec(FREEZE_SPEC)
assert FREEZE_SPEC.loader is not None
FREEZE_SPEC.loader.exec_module(freeze_builder)

SPEC = importlib.util.spec_from_file_location(
    "online_correction_v4", ROOT / "tools/online_correction_v4.py"
)
v4 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(v4)

from tools.build_v4_horizontal_g3_plan import build as build_g3_plan  # noqa: E402

RESET_SPEC = importlib.util.spec_from_file_location(
    "build_v4_horizontal_reset_registry",
    ROOT / "tools/build_v4_horizontal_reset_registry.py",
)
reset_builder = importlib.util.module_from_spec(RESET_SPEC)
assert RESET_SPEC.loader is not None
RESET_SPEC.loader.exec_module(reset_builder)

DEFAULT_CAMPAIGN = ROOT / "docs/online_correction_v4/campaign.json"
DEFAULT_HISTORICAL_QUEUE = ROOT / "artifacts/online_correction_v4/queue.jsonl"
DEFAULT_AMENDMENT = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_geometry_repair_amendment.candidate.json"
)
DEFAULT_QUEUE_OUT = (
    ROOT
    / "artifacts/online_correction_v4/queue_horizontal_geometry_repair_v1.jsonl"
)
DEFAULT_MANIFEST_OUT = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_geometry_repair_inventory_v1.json"
)
DEFAULT_RESET_OUT = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_reset_registry.geometry_repair_v1.candidate.json"
)
DEFAULT_G3_OUT = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_g3_plan.geometry_repair_v1.candidate.json"
)
AFFECTED_FAMILIES = ("C1", "C3", "C4")
EXPECTED_ROW_COUNT = 9728


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def artifact(path: Path) -> dict[str, object]:
    body = path.read_bytes()
    return {
        "path": portable_path(path),
        "bytes": len(body),
        "sha256": sha256_bytes(body),
    }


def _episode_id(
    campaign: str,
    family: str,
    fixture: str,
    block: int,
    factors: dict,
    *,
    fixture_version: str,
) -> str:
    suffix = v4.digest([fixture_version, fixture, block, factors])[:16]
    return f"{campaign}-{family}-{fixture_version}-b{block:03d}-{suffix}"


def _prefix_id(
    campaign: str,
    fixture: str,
    block: int,
    factors: dict,
    *,
    fixture_version: str,
) -> str:
    identity = {k: factors[k] for k in ("policy", "goal", "wording", "named_reference")}
    identity["initial_scene"] = (
        "destination" if factors["scenario"] == "destination_static" else "original"
    )
    identity["fixture_version"] = fixture_version
    return f"{campaign}-prefix-{fixture_version}-{v4.digest([fixture, block, identity])[:24]}"


def _block_key(campaign: str, fixture: str, block: int, *, fixture_version: str) -> str:
    return f"{campaign}:{fixture}:{fixture_version}:{block:03d}"


def build_repaired_rows(
    config: dict,
    *,
    config_sha256: str,
    fixture_version: str,
    cohort: str,
) -> list[dict]:
    seed = config["seed_reservation"]
    substitutions = v4.seed_substitution_map(config)
    families = {item["id"]: item for item in config["families"]}
    rows: list[dict] = []
    by_source: dict[tuple[str, str, int, str], list[dict]] = defaultdict(list)
    for family in config["families"]:
        if family["id"] not in AFFECTED_FAMILIES:
            continue
        names = sorted(family["factors"])
        fixture = family["fixture"]
        for block in range(family["blocks"]):
            nominal_env_seed = (
                seed["environment_base"]
                + config["fixtures"][fixture]["seed_slot"] * seed["fixture_stride"]
                + block
            )
            substitution = substitutions.get((fixture, block))
            env_seed = (
                substitution["replacement_seed"]
                if substitution is not None
                else nominal_env_seed
            )
            for values in itertools.product(*(family["factors"][name] for name in names)):
                factors = dict(family["fixed"], **dict(zip(names, values)))
                policy_seed = int(
                    v4.digest(
                        [
                            seed["policy_seed_namespace"],
                            fixture_version,
                            factors["policy"],
                            fixture,
                            block,
                        ]
                    )[:16],
                    16,
                ) % (2**31)
                row = {
                    "schema_version": 1,
                    "manifest_type": "geometry_repair_inventory_v1",
                    "runtime_bound": False,
                    "episode_id": _episode_id(
                        config["campaign_id"],
                        family["id"],
                        fixture,
                        block,
                        factors,
                        fixture_version=fixture_version,
                    ),
                    "campaign": config["campaign_id"],
                    "family": family["id"],
                    "fixture": fixture,
                    "fixture_version": fixture_version,
                    "block_id": block,
                    "block_key": _block_key(
                        config["campaign_id"], fixture, block, fixture_version=fixture_version
                    ),
                    "env_seed": env_seed,
                    "policy_seed": policy_seed,
                    "cohort": cohort,
                    "priority": family["priority"],
                    "factors": factors,
                    "prefix_group_id": _prefix_id(
                        config["campaign_id"],
                        fixture,
                        block,
                        factors,
                        fixture_version=fixture_version,
                    ),
                    "execution_group": f"{factors['policy']}:{fixture}:{fixture_version}",
                    "execution_order_key": v4.digest(
                        [
                            seed["policy_seed_namespace"],
                            "execution-order",
                            fixture_version,
                            fixture,
                            block,
                            factors,
                        ]
                    ),
                    "config_sha256": config_sha256,
                    "reuse_episode_ids": [],
                    "historical_layout": "original",
                    "repaired_layout_only": True,
                }
                if substitution is not None:
                    row["env_seed_substitution"] = {
                        "retired_seed": substitution["retired_seed"],
                        "replacement_seed": substitution["replacement_seed"],
                        "reason": substitution["reason"],
                        "evidence_path": substitution["evidence_path"],
                    }
                row["counterbalance"] = v4._counterbalance(config, fixture, block)
                row["prompt_recipe"] = v4._prompt_recipe(
                    config, fixture, factors, row["counterbalance"]
                )
                rows.append(row)
                by_source[(family["id"], fixture, block, factors["policy"])].append(row)
    for row in rows:
        row["reuse_episode_ids"] = v4._reuse_candidates(
            row, families[row["family"]], by_source
        )
    rows.sort(key=lambda item: (item["execution_group"], item["execution_order_key"]))
    order_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        row["execution_order"] = order_counts[row["execution_group"]]
        order_counts[row["execution_group"]] += 1
    return rows


def enrich_queue_rows(config: dict, rows: list[dict]) -> list[dict]:
    return freeze_builder.enrich_queue_rows(config, rows)


def build_inventory_manifest(
    *,
    rows: list[dict],
    amendment_path: Path,
    queue_path: Path,
    reset_registry_path: Path,
    g3_plan_path: Path,
    historical_queue_path: Path,
) -> dict:
    family_counts = {family: sum(row["family"] == family for row in rows) for family in AFFECTED_FAMILIES}
    return {
        "schema_version": "v4-horizontal-geometry-repair-inventory-v1",
        "campaign_id": "online_correction_v4",
        "fixture_id": "horizontal",
        "fixture_version": FIXTURE_VERSION,
        "cohort": COHORT,
        "row_count": len(rows),
        "expected_row_count": EXPECTED_ROW_COUNT,
        "family_row_counts": family_counts,
        "historical_authority_preserved": {
            "campaign_json": artifact(DEFAULT_CAMPAIGN),
            "historical_queue_jsonl": artifact(historical_queue_path),
        },
        "amendment": artifact(amendment_path),
        "queue": artifact(queue_path),
        "reset_registry": artifact(reset_registry_path),
        "g3_plan": artifact(g3_plan_path),
        "cross_layout_mixing_forbidden": True,
        "release_boundary": (
            "Repaired-layout inventory for C1/C3/C4 only. Qualification and runtime "
            "locks must bind this fixture_version before any repaired horizontal "
            "policy inference."
        ),
    }


def validate_rows(rows: list[dict], historical_queue_path: Path) -> None:
    if len(rows) != EXPECTED_ROW_COUNT:
        raise ValueError(f"repaired inventory row count {len(rows)} != {EXPECTED_ROW_COUNT}")
    historical_ids = {
        json.loads(line)["episode_id"]
        for line in historical_queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    repaired_ids = {row["episode_id"] for row in rows}
    if repaired_ids & historical_ids:
        raise ValueError("repaired episode_id collides with historical queue")
    for row in rows:
        if row.get("fixture_version") != FIXTURE_VERSION:
            raise ValueError("row fixture_version differs")
        if row.get("cohort") != COHORT:
            raise ValueError("row cohort differs")
        for reuse_id in row.get("reuse_episode_ids", []):
            if reuse_id not in repaired_ids:
                raise ValueError(f"{row['episode_id']} reuses non-repaired episode {reuse_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--historical-queue", type=Path, default=DEFAULT_HISTORICAL_QUEUE)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--queue-out", type=Path, default=DEFAULT_QUEUE_OUT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--reset-out", type=Path, default=DEFAULT_RESET_OUT)
    parser.add_argument("--g3-out", type=Path, default=DEFAULT_G3_OUT)
    args = parser.parse_args()

    config, config_sha256 = v4.load_json(args.campaign.resolve())
    rows = build_repaired_rows(
        config,
        config_sha256=config_sha256,
        fixture_version=FIXTURE_VERSION,
        cohort=COHORT,
    )
    validate_rows(rows, args.historical_queue.resolve())

    queue_rows = enrich_queue_rows(config, rows)
    queue_bytes = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in queue_rows
    ).encode("utf-8")
    args.queue_out.parent.mkdir(parents=True, exist_ok=True)
    args.queue_out.write_bytes(queue_bytes)

    reset_payload = reset_builder.build_registry(
        campaign_path=args.campaign.resolve(),
        queue_path=args.queue_out.resolve(),
        source_report_path=reset_builder.DEFAULT_SOURCE,
        geometry_repair_amendment_path=args.amendment.resolve(),
    )
    args.reset_out.write_bytes(reset_builder.canonical_json_bytes(reset_payload))

    build_g3_plan(
        campaign_path=args.campaign.resolve(),
        queue_path=args.queue_out.resolve(),
        motion_path=ROOT / "artifacts/online_correction_v4/motion_manifest.json",
        registry_path=args.reset_out.resolve(),
        g2_aggregate_path=None,
        output_path=args.g3_out.resolve(),
        fixture_id="horizontal",
        qualification_scope="confirmatory",
        geometry_repair_mode=True,
    )
    g3_payload = json.loads(args.g3_out.read_text(encoding="utf-8"))
    g3_payload["fixture_version"] = FIXTURE_VERSION
    g3_payload["geometry_repair_amendment"] = artifact(args.amendment.resolve())
    args.g3_out.write_bytes(canonical_json_bytes(g3_payload))

    manifest = build_inventory_manifest(
        rows=rows,
        amendment_path=args.amendment.resolve(),
        queue_path=args.queue_out.resolve(),
        reset_registry_path=args.reset_out.resolve(),
        g3_plan_path=args.g3_out.resolve(),
        historical_queue_path=args.historical_queue.resolve(),
    )
    args.manifest_out.write_bytes(canonical_json_bytes(manifest))

    print(
        json.dumps(
            {
                "row_count": len(rows),
                "queue": artifact(args.queue_out),
                "manifest": artifact(args.manifest_out),
                "reset_registry": artifact(args.reset_out),
                "g3_plan": artifact(args.g3_out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
