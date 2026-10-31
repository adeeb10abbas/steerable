#!/usr/bin/env python3
"""Authenticate and package retained H13 V1 evaluator-development evidence.

H13 is the historical Cosmos Edge V1 corpus registered in ABLATION_SPEC.md:
80 episodes, 752 action/forecast chunks, and four cached forecast frames per
chunk.  This adapter re-authenticates every upstream Git object, independently
replays the published row-level arithmetic, and verifies the frozen 160-item
historical-stratum annotation draw.  It does not recover media, align forecast
and execution time, validate image labels, run inference, or estimate forecast
fidelity.  All output is therefore evaluator-development evidence only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence


UPSTREAM_PIN = "ce561e66f82e95055e39d3d7711691982f6b2086"
AUDIT_COMMIT = "4eac280b73be3fcaf6fa4ab12697a153a485b422"
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
SPEC_PATH = PACKAGE / "docs" / "ABLATION_SPEC.md"
DEFAULT_JSON = PACKAGE / "paper" / "data" / "h13_v1_forecast_corpus.json"
DEFAULT_CSV = PACKAGE / "paper" / "data" / "h13_v1_forecast_corpus.csv"

AUDIT_PATH = "workshops/corl2026_world_models/results/audit_summary.json"
SOURCE_PATH = "workshops/corl2026_world_models/results/source_hashes.json"
ROWS_PATH = "workshops/corl2026_world_models/results/reconciled_chunks.jsonl"
SAMPLE_JSON_PATH = "workshops/corl2026_world_models/results/annotation_sample.json"
SAMPLE_CSV_PATH = "workshops/corl2026_world_models/results/annotation_sample.csv"
SAMPLE_SCRIPT_PATH = "workshops/corl2026_world_models/analysis/prepare_annotation_sample.py"
PROTOCOL_PATH = "workshops/corl2026_world_models/docs/ANALYSIS_PROTOCOL.md"

EXPECTED_BUNDLE: dict[str, dict[str, Any]] = {
    AUDIT_PATH: {
        "git_blob": "2e8bebb27cda6ba4d14aca4983564af44700ff3b",
        "sha256": "3d5752446c577f7049d02bb084c87800d5c9c001394e0782e4ab42eedd6ba0a5",
        "bytes": 7_913,
    },
    SOURCE_PATH: {
        "git_blob": "19f6b797bec3cca5e3f0a3078ea1324a2c98aa74",
        "sha256": "495bf12a236b46ac2b6dbb841324c729ee122e6a80b20d5132e6cf4173f35885",
        "bytes": 304_660,
    },
    ROWS_PATH: {
        "git_blob": "92d3c799d448eeb99306728e0a4206f4f40e6389",
        "sha256": "2b5c0067d1d3bc69f95fe0db4d9cbc1b1491e355043eaf918f1cf82e856c4dd7",
        "bytes": 476_042,
    },
    SAMPLE_JSON_PATH: {
        "git_blob": "e1ffa6ee09c5e16fbd47fb4c956d9d86c8d23d6c",
        "sha256": "f50cf289355c3dd5d509b98b089ff80ceff9ea34814b8841be8e1e91984995c5",
        "bytes": 144_051,
    },
    SAMPLE_CSV_PATH: {
        "git_blob": "7ae8b6f42f360167f16844f069901cc1e84ddb4b",
        "sha256": "b23bbd0f879f53489aed7cc534a7b29bafb96650aabf3477985ac6b8a1bb0eab",
        "bytes": 65_583,
    },
}

EXPECTED_SOURCE = {
    "files": 761,
    "bytes": 7_230_812,
    "lines": 222_796,
    "manifest_sha256": "ce7da897256d7583365f876e9697c08639e754dd30951bf476259ce7915b9490",
    "cache_files": 752,
    "cache_bytes": 5_269_817,
    "cache_lines": 210_592,
    "cache_manifest_sha256": "c40ef7924cbf6ebd5a879dab05dd4bbf47c6b2fa73d8f3969ec9cabd4c4c7a7c",
    "other_files": 9,
    "other_bytes": 1_960_995,
    "other_lines": 12_204,
}

WORDING_TIER = {"canonical": 6100, "short": 6100, "declarative": 7200, "contrastive": 7200}
TIER_WORDINGS = {6100: ("canonical", "short"), 7200: ("declarative", "contrastive")}
WORDING_SOURCE = {
    "canonical": "cosmos_canonical",
    "short": "cosmos_vague",
    "declarative": "cosmos_declarative",
    "contrastive": "cosmos_contrastive",
}
RELATION_TASK = {
    "left": "RubiksCubeLeftOfBowlMatchedTask",
    "right": "RubiksCubeRightOfBowlMatchedTask",
}
QUADRANTS = (
    "imagines_requested_executes_requested",
    "imagines_requested_executes_not_requested",
    "does_not_imagine_requested_executes_requested",
    "neither_imagines_nor_executes_requested",
    "uncertain_future",
)
EXPECTED_CELL_CHUNKS = {
    ("canonical", "left"): 99,
    ("canonical", "right"): 86,
    ("short", "left"): 125,
    ("short", "right"): 64,
    ("declarative", "left"): 66,
    ("declarative", "right"): 87,
    ("contrastive", "left"): 146,
    ("contrastive", "right"): 79,
}
EXPECTED_CELL_CERTAIN = {
    ("canonical", "left"): 51,
    ("canonical", "right"): 54,
    ("short", "left"): 71,
    ("short", "right"): 31,
    ("declarative", "left"): 34,
    ("declarative", "right"): 47,
    ("contrastive", "left"): 89,
    ("contrastive", "right"): 44,
}
STRATA = {
    "both_positive": (22, 22),
    "future_only_positive": (5, 5),
    "execution_only_positive": (3, 3),
    "both_negative": (391, 50),
    "forecast_abstention_execution_positive": (72, 40),
    "forecast_abstention_execution_negative": (259, 40),
}
ROW_KEYS = {
    "actual", "chunk_dir", "endpoint_sensitivity", "episode", "execution_end_step",
    "id", "legacy", "legacy_quadrant", "replan", "requested", "sampling_seed",
    "task_dir", "threshold_predictions", "wording",
}


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "--no-replace-objects", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    result = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(f"required ancestry is absent: {ancestor} -> {descendant}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON object key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _reject_nonfinite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number in {label}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{label}[{index}]")


def parse_json(raw: bytes, *, label: str) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {label}: {error}") from error
    _reject_nonfinite(value, label)
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return _sha256(raw)


def _typed_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _typed_label(value: Any, label: str) -> bool | None:
    if value is not None and type(value) is not bool:
        raise ValueError(f"{label} must be boolean or null")
    return value


def _quadrant(predicted: bool | None, actual: bool) -> str:
    if predicted is None:
        return QUADRANTS[4]
    if predicted and actual:
        return QUADRANTS[0]
    if predicted and not actual:
        return QUADRANTS[1]
    if not predicted and actual:
        return QUADRANTS[2]
    return QUADRANTS[3]


def _stratum(row: Mapping[str, Any]) -> str:
    legacy, actual = row["legacy"], row["actual"]
    mapping = {
        (True, True): "both_positive",
        (True, False): "future_only_positive",
        (False, True): "execution_only_positive",
        (False, False): "both_negative",
        (None, True): "forecast_abstention_execution_positive",
        (None, False): "forecast_abstention_execution_negative",
    }
    return mapping[(legacy, actual)]


def _read_bundle(root: Path) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    resolved = _git(root, "rev-parse", "--verify", AUDIT_COMMIT + "^{commit}").decode().strip()
    if resolved != AUDIT_COMMIT:
        raise ValueError("audit commit did not resolve exactly")
    upstream = _git(root, "rev-parse", "--verify", UPSTREAM_PIN + "^{commit}").decode().strip()
    if upstream != UPSTREAM_PIN:
        raise ValueError("upstream commit did not resolve exactly")
    _require_ancestor(root, UPSTREAM_PIN, AUDIT_COMMIT)
    raw_by_path: dict[str, bytes] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for path, expected in EXPECTED_BUNDLE.items():
        object_id = _git(root, "rev-parse", "--verify", f"{AUDIT_COMMIT}:{path}").decode().strip()
        if object_id != expected["git_blob"]:
            raise ValueError(f"audit-bundle blob mismatch: {path}")
        raw = _git(root, "cat-file", "blob", object_id)
        observed = {"git_blob": object_id, "sha256": _sha256(raw), "bytes": len(raw)}
        if observed != expected:
            raise ValueError(f"audit-bundle bytes mismatch: {path}")
        raw_by_path[path] = raw
        receipts[path] = observed
    return raw_by_path, receipts


def _authenticate_upstream_sources(root: Path, ledger: Any) -> dict[str, Any]:
    if not isinstance(ledger, list) or len(ledger) != EXPECTED_SOURCE["files"]:
        raise ValueError("source ledger must contain exactly 761 rows")
    observed: list[tuple[str, str, int, int, str]] = []
    seen: set[str] = set()
    for number, receipt in enumerate(ledger, 1):
        if not isinstance(receipt, dict) or set(receipt) != {
            "bytes", "matches_upstream_commit", "path", "sha256", "upstream_git_blob_sha1"
        }:
            raise ValueError(f"source-ledger schema mismatch at row {number}")
        path = receipt["path"]
        if not isinstance(path, str) or not path or path in seen:
            raise ValueError("invalid or duplicate source-ledger path")
        seen.add(path)
        if receipt["matches_upstream_commit"] is not True:
            raise ValueError(f"source was not authenticated upstream: {path}")
        object_id = _git(root, "rev-parse", "--verify", f"{UPSTREAM_PIN}:{path}").decode().strip()
        if object_id != receipt["upstream_git_blob_sha1"]:
            raise ValueError(f"upstream object identity changed: {path}")
        raw = _git(root, "cat-file", "blob", object_id)
        if len(raw) != receipt["bytes"] or _sha256(raw) != receipt["sha256"]:
            raise ValueError(f"upstream source bytes changed: {path}")
        observed.append((path, object_id, len(raw), len(raw.splitlines()), _sha256(raw)))
    observed.sort()
    manifest_raw = "".join(
        f"{path}\0{object_id}\0{size}\0{lines}\0{digest}\n"
        for path, object_id, size, lines, digest in observed
    ).encode("utf-8")
    cache = [row for row in observed if "/localization_cache/" in row[0]]
    other = [row for row in observed if "/localization_cache/" not in row[0]]
    cache_manifest = "".join(
        f"{path}\0{object_id}\0{size}\0{lines}\0{digest}\n"
        for path, object_id, size, lines, digest in cache
    ).encode("utf-8")
    actual = {
        "files": len(observed),
        "bytes": sum(row[2] for row in observed),
        "lines": sum(row[3] for row in observed),
        "manifest_sha256": _sha256(manifest_raw),
        "cache_files": len(cache),
        "cache_bytes": sum(row[2] for row in cache),
        "cache_lines": sum(row[3] for row in cache),
        "cache_manifest_sha256": _sha256(cache_manifest),
        "other_files": len(other),
        "other_bytes": sum(row[2] for row in other),
        "other_lines": sum(row[3] for row in other),
    }
    if actual != EXPECTED_SOURCE:
        raise ValueError(f"upstream aggregate identity changed: {actual}")
    return actual


def _parse_rows(raw: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    cell_counts: Counter[tuple[str, str]] = Counter()
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank reconciled row: {number}")
        row = parse_json(line, label=f"{ROWS_PATH}:{number}")
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise ValueError(f"reconciled-row schema mismatch: {number}")
        identifier = row["id"]
        if not isinstance(identifier, str) or identifier in seen:
            raise ValueError("invalid or duplicate reconciled ID")
        seen.add(identifier)
        wording, requested = row["wording"], row["requested"]
        if wording not in WORDING_TIER or requested not in RELATION_TASK:
            raise ValueError("unexpected wording or requested relation")
        episode = _typed_int(row["episode"], "episode")
        if episode > 9:
            raise ValueError("episode index exceeds registered ten-per-cell range")
        replan = _typed_int(row["replan"], "replan")
        sampling_seed = _typed_int(row["sampling_seed"], "sampling_seed")
        tier = WORDING_TIER[wording]
        if sampling_seed != (tier + episode) * 1000 + replan:
            raise ValueError("sampling-seed identity mismatch")
        expected_id = (
            f"{WORDING_SOURCE[wording]}/{RELATION_TASK[requested]}/"
            f"episode_{episode:03d}/chunk_{replan:03d}"
        )
        if identifier != expected_id:
            raise ValueError(f"canonical row identity mismatch: {identifier}")
        actual = row["actual"]
        if type(actual) is not bool:
            raise ValueError("execution label must be boolean")
        legacy = _typed_label(row["legacy"], "legacy")
        endpoint = _typed_label(row["endpoint_sensitivity"], "endpoint_sensitivity")
        if row["legacy_quadrant"] != _quadrant(legacy, actual):
            raise ValueError("legacy quadrant disagrees with labels")
        predictions = row["threshold_predictions"]
        if not isinstance(predictions, dict) or set(predictions) != {"0.1", "0.15", "0.2"}:
            raise ValueError("threshold-prediction schema mismatch")
        for threshold, value in predictions.items():
            _typed_label(value, f"threshold_predictions.{threshold}")
        if predictions["0.2"] != legacy:
            raise ValueError("legacy label differs from registered 0.20 threshold replay")
        _typed_int(row["execution_end_step"], "execution_end_step")
        if not all(isinstance(row[key], str) and row[key] for key in ("task_dir", "chunk_dir")):
            raise ValueError("missing historical recovery pointer")
        cell_counts[(wording, requested)] += 1
        rows.append(row)
    if len(rows) != 752 or cell_counts != Counter(EXPECTED_CELL_CHUNKS):
        raise ValueError("registered H13 cell counts changed")
    episodes = {(row["wording"], row["requested"], row["episode"]) for row in rows}
    if len(episodes) != 80:
        raise ValueError("registered H13 episode count changed")
    for tier, wordings in TIER_WORDINGS.items():
        for episode in range(10):
            block = {
                (row["wording"], row["requested"])
                for row in rows
                if WORDING_TIER[row["wording"]] == tier and row["episode"] == episode
            }
            if block != {(wording, relation) for wording in wordings for relation in RELATION_TASK}:
                raise ValueError("incomplete four-episode matched seed block")
    return rows


def _metrics(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    counts = Counter(_quadrant(row[label], row["actual"]) for row in rows)
    certain = sum(row[label] is not None for row in rows)
    positive = sum(row["actual"] for row in rows)
    covered_positive = sum(row["actual"] and row[label] is not None for row in rows)
    agreement = sum(row[label] is not None and row[label] == row["actual"] for row in rows)
    constant_negative = sum(row[label] is not None and not row["actual"] for row in rows)
    return {
        "n": len(rows),
        "certain": certain,
        "abstained": len(rows) - certain,
        "coverage": certain / len(rows),
        "executed_positive": positive,
        "covered_executed_positive": covered_positive,
        "positive_execution_coverage": covered_positive / positive,
        "quadrants": {key: counts[key] for key in QUADRANTS},
        "same_covered_subset_arithmetic": {
            "denominator": certain,
            "observed_agreement_numerator": agreement,
            "constant_negative_agreement_numerator": constant_negative,
            "observed_agreement": agreement / certain,
            "constant_negative_agreement": constant_negative / certain,
            "difference_numerator": agreement - constant_negative,
            "difference_percentage_points": 100 * (agreement - constant_negative) / certain,
            "interpretation": "same covered subset only; not validated fidelity or utility",
        },
    }


def _cluster_intervals(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    blocks: dict[int, dict[int, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        tier = WORDING_TIER[row["wording"]]
        blocks[tier][tier + row["episode"]].append(row)
    rng = random.Random(20260912)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(2_000):
        selected: list[Mapping[str, Any]] = []
        for tier in sorted(blocks):
            values = [blocks[tier][seed] for seed in sorted(blocks[tier])]
            for _ in values:
                selected.extend(rng.choice(values))
        metric = _metrics(selected, label)
        samples["coverage"].append(metric["coverage"])
        samples["positive_execution_coverage"].append(metric["positive_execution_coverage"])
    def interval(values: list[float]) -> list[float]:
        values = sorted(values)
        return [values[math.floor(0.025 * (len(values) - 1))], values[math.ceil(0.975 * (len(values) - 1))]]
    return {
        "method": "percentile matched-seed-block bootstrap within 6100 and 7200 tiers; four episodes per block, all chunks; pooled chunk ratio",
        "independent_seed_blocks": 20,
        "seed_blocks_by_tier": {"6100": 10, "7200": 10},
        "episodes_per_block": 4,
        "seed_identity_validation": "sampling_seed == (tier + episode_index) * 1000 + replan_index for every row; both wordings and both directions required per block",
        "replicates": 2_000,
        "seed": 20260912,
        "confidence": 0.95,
        "intervals": {key: interval(values) for key, values in samples.items()},
        "limitations": "Descriptive sampling uncertainty only; does not include evaluator error or selection bias.",
    }


def _cell_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for wording in WORDING_TIER:
        for relation in RELATION_TASK:
            selected = [row for row in rows if row["wording"] == wording and row["requested"] == relation]
            legacy = _metrics(selected, "legacy")
            endpoint = _metrics(selected, "endpoint_sensitivity")
            expected = EXPECTED_CELL_CERTAIN[(wording, relation)]
            if legacy["certain"] != expected:
                raise ValueError("cell-level legacy coverage changed")
            output.append({
                "wording": wording,
                "requested_relation": relation,
                "episodes": 10,
                "chunks": len(selected),
                "legacy_certain": legacy["certain"],
                "legacy_abstained": legacy["abstained"],
                "legacy_coverage": legacy["coverage"],
                "legacy_executed_positive": legacy["executed_positive"],
                "legacy_covered_executed_positive": legacy["covered_executed_positive"],
                "endpoint_certain": endpoint["certain"],
                "endpoint_coverage": endpoint["coverage"],
            })
    return output


def _verify_annotation_sample(root: Path, raw_by_path: Mapping[str, bytes], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    document = parse_json(raw_by_path[SAMPLE_JSON_PATH], label=SAMPLE_JSON_PATH)
    if not isinstance(document, dict) or set(document) != {
        "interpretation", "provenance", "release_gates", "sample", "sampling_design", "status", "visibility"
    }:
        raise ValueError("annotation sample document schema changed")
    if document["status"] != "PROVISIONAL PRE-ANNOTATION DESIGN":
        raise ValueError("annotation sample status changed")
    sample = document["sample"]
    if not isinstance(sample, list) or len(sample) != 160:
        raise ValueError("annotation sample must contain 160 rows")
    if len({item.get("source_id") for item in sample}) != 160:
        raise ValueError("annotation sample contains duplicate IDs")
    provenance = document["provenance"]
    if provenance.get("full_population_file_sha256") != EXPECTED_BUNDLE[ROWS_PATH]["sha256"]:
        raise ValueError("annotation draw is not bound to the reconciled population")
    if provenance.get("draw_sha256") != _canonical_sha(sample):
        raise ValueError("annotation draw digest mismatch")
    if provenance.get("csv_sha256") != _sha256(raw_by_path[SAMPLE_CSV_PATH]):
        raise ValueError("annotation CSV digest mismatch")
    script = _git(root, "show", f"{AUDIT_COMMIT}:{SAMPLE_SCRIPT_PATH}")
    protocol = _git(root, "show", f"{AUDIT_COMMIT}:{PROTOCOL_PATH}")
    if provenance.get("sampling_script_sha256") != _sha256(script):
        raise ValueError("annotation sampling script digest mismatch")
    if provenance.get("protocol_file_sha256") != _sha256(protocol):
        raise ValueError("annotation protocol digest mismatch")
    buckets: dict[str, list[Mapping[str, Any]]] = {key: [] for key in STRATA}
    for row in rows:
        buckets[_stratum(row)].append(row)
    rng = random.Random(20260912)
    expected_ids: list[str] = []
    for stratum, (population_n, sample_n) in STRATA.items():
        population = sorted(buckets[stratum], key=lambda row: row["id"])
        if len(population) != population_n:
            raise ValueError(f"annotation stratum population changed: {stratum}")
        expected_ids.extend(sorted(row["id"] for row in rng.sample(population, sample_n)))
    if [item.get("source_id") for item in sample] != expected_ids:
        raise ValueError("annotation draw no longer reproduces from the frozen algorithm")
    counts = Counter(item.get("stratum") for item in sample)
    if counts != Counter({key: values[1] for key, values in STRATA.items()}):
        raise ValueError("annotation sample stratum counts changed")
    for item in sample:
        if item.get("human_label_status") != "not_collected":
            raise ValueError("annotation sample unexpectedly claims a human label")
        if item.get("media_status") != "pending_recovery_and_verification":
            raise ValueError("annotation sample unexpectedly claims recovered media")
        if item.get("alignment_status") != "pending_recovery_and_verification":
            raise ValueError("annotation sample unexpectedly claims aligned evidence")
    csv_rows = list(csv.DictReader(io.StringIO(raw_by_path[SAMPLE_CSV_PATH].decode("utf-8"))))
    if len(csv_rows) != 160 or [row["source_id"] for row in csv_rows] != expected_ids:
        raise ValueError("annotation CSV disagrees with the frozen draw")
    return {
        "status": document["status"],
        "visibility": document["visibility"],
        "population_size": 752,
        "sample_size": 160,
        "seed": 20260912,
        "strata": [
            {
                "stratum": key,
                "population": population_n,
                "sample": sample_n,
                "inclusion_probability": sample_n / population_n,
                "population_weight": population_n / sample_n,
            }
            for key, (population_n, sample_n) in STRATA.items()
        ],
        "draw_sha256": provenance["draw_sha256"],
        "csv_sha256": provenance["csv_sha256"],
        "media_recovered_and_verified": 0,
        "alignment_verified": 0,
        "human_labels_collected": 0,
        "gate_advanced": False,
    }


def _compare_summary(report: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    checks = {
        "cohort": report["population"]["cohort"],
        "legacy_replay": report["legacy_replay"],
        "legacy_coverage_uncertainty": report["legacy_coverage_uncertainty"],
        "endpoint_metrics": report["endpoint_sensitivity"]["metrics"],
        "endpoint_uncertainty": report["endpoint_sensitivity"]["coverage_uncertainty"],
    }
    source = {
        "cohort": summary["cohort"],
        "legacy_replay": summary["legacy_replay"],
        "legacy_coverage_uncertainty": summary["legacy_coverage_uncertainty"],
        "endpoint_metrics": summary["endpoint_sensitivity"]["metrics"],
        "endpoint_uncertainty": summary["endpoint_sensitivity"]["coverage_uncertainty"],
    }
    # Our package adds explicit abstention and difference-numerator fields.  The
    # immutable audit remains the comparison oracle for every shared quantity.
    def shared_equal(new: Any, old: Any, label: str) -> None:
        if isinstance(old, dict):
            for key, value in old.items():
                new_key = "same_covered_subset_arithmetic" if key == "historical_label_arithmetic" else key
                if new_key not in new:
                    raise ValueError(f"missing reproduced summary field: {label}.{new_key}")
                shared_equal(new[new_key], value, f"{label}.{new_key}")
        elif isinstance(old, list):
            if len(new) != len(old):
                raise ValueError(f"summary list length changed: {label}")
            for index, value in enumerate(old):
                shared_equal(new[index], value, f"{label}[{index}]")
        elif isinstance(old, float):
            if not math.isclose(float(new), old, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"summary float changed: {label}")
        elif new != old:
            raise ValueError(f"summary value changed: {label}")
    for label in checks:
        shared_equal(checks[label], source[label], label)


def analyze(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    raw_by_path, bundle_receipts = _read_bundle(root)
    summary = parse_json(raw_by_path[AUDIT_PATH], label=AUDIT_PATH)
    ledger = parse_json(raw_by_path[SOURCE_PATH], label=SOURCE_PATH)
    if not isinstance(summary, dict):
        raise ValueError("audit summary must be an object")
    source_identity = _authenticate_upstream_sources(root, ledger)
    rows = _parse_rows(raw_by_path[ROWS_PATH])
    legacy = _metrics(rows, "legacy")
    legacy_agreement = legacy["same_covered_subset_arithmetic"]["observed_agreement_numerator"]
    legacy["unresolved_case_full_corpus_agreement_bounds"] = {
        "denominator": len(rows),
        "lower_numerator": legacy_agreement,
        "upper_numerator": legacy_agreement + legacy["abstained"],
        "lower": legacy_agreement / len(rows),
        "upper": (legacy_agreement + legacy["abstained"]) / len(rows),
        "interpretation": "historical-label missingness bound only; not a forecast-fidelity interval",
    }
    endpoint = _metrics(rows, "endpoint_sensitivity")
    legacy_uncertainty = _cluster_intervals(rows, "legacy")
    endpoint_uncertainty = _cluster_intervals(rows, "endpoint_sensitivity")
    thresholds: dict[str, Any] = {}
    for threshold in ("0.1", "0.15", "0.2"):
        replaced = [dict(row, threshold_label=row["threshold_predictions"][threshold]) for row in rows]
        thresholds[threshold] = _metrics(replaced, "threshold_label")
    sample = _verify_annotation_sample(root, raw_by_path, rows)
    report: dict[str, Any] = {
        "schema_version": "h13-v1-retained-evaluator-package-v1",
        "status": "RETAINED EVALUATOR DEVELOPMENT ONLY",
        "provenance": {
            "upstream_commit": UPSTREAM_PIN,
            "audit_commit": AUDIT_COMMIT,
            "audit_commit_descends_from_upstream": True,
            "bundle": bundle_receipts,
            "bundle_files": len(bundle_receipts),
            "bundle_bytes": sum(value["bytes"] for value in bundle_receipts.values()),
            "upstream_sources": source_identity,
        },
        "population": {
            "cohort": {
                "episodes": 80,
                "chunks": len(rows),
                "frames": summary["cohort"]["frames"],
                "reliable_frames": summary["cohort"]["reliable_frames"],
                "unreliable_frames": summary["cohort"]["frames"] - summary["cohort"]["reliable_frames"],
                "frame_rejection_reasons_overlapping": summary["cohort"]["frame_rejection_reasons_overlapping"],
            },
            "matched_seed_blocks": 20,
            "episodes_per_block": 4,
            "blocks_by_tier": {"6100": 10, "7200": 10},
            "independence_boundary": "20 matched seed blocks; 80 episodes and 752 chunks are not independent replicates",
            "cells": _cell_rows(rows),
        },
        "legacy_replay": legacy,
        "legacy_coverage_uncertainty": legacy_uncertainty,
        "threshold_replay": thresholds,
        "endpoint_sensitivity": {
            "status": "posthoc aggregation sensitivity; NOT corrected fidelity",
            "metrics": endpoint,
            "coverage_uncertainty": endpoint_uncertainty,
        },
        "annotation_sampling": sample,
        "corrected_fidelity_readiness": {
            "ready": False,
            "required_paths": 840,
            "missing_by_role": {
                "camera_and_coordinate_configuration": 8,
                "chunk_alignment_metadata": 752,
                "execution_trajectory": 80,
            },
            "supporting_generated_media_and_action_receipts_requiring_recovery": 2_256,
            "actual_execution_imagery_verified": False,
            "reason": "raw trajectories, alignment metadata, common geometry, and actual execution imagery remain unresolved",
        },
        "new_activity": {
            "annotation_media": 0,
            "behavioral_actions": 0,
            "behavioral_cells": 0,
            "confirmation_inference_cells": 0,
            "forecast_accuracy_estimates": 0,
            "human_responses": 0,
            "labels": 0,
            "model_requests": 0,
            "queue_descriptors": 0,
            "request_rows": 0,
            "request_selections": 0,
            "retrieval_jobs": 0,
            "runnable_manifests": 0,
        },
        "gate_state": {
            "annotation_media_ready": False,
            "camera_alignment_ready": False,
            "confirmation_release_ready": False,
            "current_GM_PVC_worker_state_reconciled": False,
            "development_annotation_gate_ready": False,
            "human_annotation_ready": False,
            "live_quiescence_inferred": False,
            "resource_freeze_ready": False,
            "safe_to_release_new_work": False,
        },
        "claim_boundary": {
            "new_model_inference": False,
            "new_behavioral_episodes": 0,
            "new_human_labels": 0,
            "forecast_fidelity_or_accuracy_estimated": False,
            "annotation_or_release_gate_advanced": False,
            "core_232_cell_accounting_changed": False,
            "maximum_core_behavioral_cells": 232,
            "allowed_interpretation": "authenticated retrospective evaluator coverage, missingness, and historical-label arithmetic",
            "prohibited_interpretations": [
                "validated forecast accuracy or fidelity",
                "incremental predictive utility over persistence",
                "752 independent trials or 80 independent scenes",
                "annotation completion or confirmation evidence",
                "cross-model ranking or a new Edge policy experiment",
            ],
        },
    }
    _compare_summary(report, summary)
    if summary["corrected_fidelity_readiness"]["ready"] is not False:
        raise ValueError("immutable audit unexpectedly claims corrected fidelity readiness")
    if summary["corrected_fidelity_readiness"]["missing_by_role"] != report["corrected_fidelity_readiness"]["missing_by_role"]:
        raise ValueError("corrected-fidelity missingness changed")
    spec = SPEC_PATH.read_text(encoding="utf-8")
    required_text = (
        "| H13 | Cosmos Edge V1 forecast corpus | 80 episodes, 752 action/forecast chunks and 3,008 cached frames.",
        "Evaluator development only; no validated forecast-accuracy result",
        "The V1 historical agreement score is not prediction fidelity",
    )
    if any(text not in spec for text in required_text):
        raise ValueError("ABLATION_SPEC H13 claim boundary changed")
    return report


def render_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_csv(report: Mapping[str, Any]) -> str:
    rows = report["population"]["cells"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--check", action="store_true", help="require checked-in outputs to match exactly")
    args = parser.parse_args(argv)
    report = analyze(ROOT)
    json_text = render_json(report)
    csv_text = render_csv(report)
    if args.check:
        if args.json_output.read_text(encoding="utf-8") != json_text:
            raise SystemExit(f"stale generated output: {args.json_output}")
        if args.csv_output.read_text(encoding="utf-8") != csv_text:
            raise SystemExit(f"stale generated output: {args.csv_output}")
    else:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json_text, encoding="utf-8")
        args.csv_output.write_text(csv_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
