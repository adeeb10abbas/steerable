"""Reproducible, fail-closed analysis from durable SGW-01 manifests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2_0260_922
SIGNFLIP_DRAWS = 100_000
SIGNFLIP_SEED = 2_0260_923


@dataclass(frozen=True)
class CompiledAnalysis:
    rows: tuple[Mapping[str, Any], ...]
    valid_rows: tuple[Mapping[str, Any], ...]
    technical_missing: int
    bootstrap_seed: int = BOOTSTRAP_SEED
    signflip_seed: int = SIGNFLIP_SEED


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifacts(manifest_path: Path, manifest: Mapping[str, Any]) -> None:
    listed = manifest.get("artifacts", {})
    for name, metadata in listed.items():
        artifact = manifest_path.parent / name
        if not artifact.is_file() or _sha256(artifact) != metadata.get("sha256"):
            raise ValueError(f"artifact hash mismatch: {artifact}")


def _load_manifest(path: Path, expected_release_hashes: Mapping[str, str]) -> Mapping[str, Any]:
    if path.suffix != ".json":
        raise ValueError(f"manifest is not JSON: {path}")
    raw = json.loads(path.read_text())
    # The recorder publishes cells/<cell>.complete.json as a pointer. Resolve
    # it only after verifying the pointer's immutable attempt manifest hash.
    if "manifest_path" not in raw:
        raise ValueError(f"compiler input must be a completion pointer: {path}")
    if "manifest_path" in raw:
        attempt_manifest = (path.parent.parent / raw["manifest_path"]).resolve()
        if not attempt_manifest.is_file():
            raise ValueError(f"completion pointer target is missing: {attempt_manifest}")
        if raw.get("manifest_sha256") != _sha256(attempt_manifest):
            raise ValueError(f"completion pointer hash mismatch: {path}")
        result_ref = raw.get("result", {})
        if isinstance(result_ref, Mapping) and result_ref.get("path"):
            result_path = (path.parent.parent / str(result_ref["path"])).resolve()
            if not result_path.is_file() or result_ref.get("sha256") != _sha256(result_path):
                raise ValueError(f"completion pointer result hash mismatch: {path}")
        path = attempt_manifest
        raw = json.loads(path.read_text())
    if raw.get("complete") is not True:
        raise ValueError(f"manifest is not durably complete: {path}")
    if raw.get("release_hashes") != dict(expected_release_hashes):
        raise ValueError(f"release provenance mismatch: {path}")
    _verify_artifacts(path, raw)
    result = raw.get("result")
    if not isinstance(result, Mapping):
        raise ValueError(f"manifest has no result: {path}")
    result_artifact = path.parent / "result.json"
    if result_artifact.is_file():
        if json.loads(result_artifact.read_text()) != result:
            raise ValueError(f"manifest result differs from result.json: {path}")
    status = result.get("status")
    if status not in {"valid_success", "valid_model_failure", "censored", "technical_invalid"}:
        raise ValueError(f"unknown result status: {status}")
    if status == "technical_invalid" and not str(result.get("technical_cause", "")).strip():
        raise ValueError(f"technical_invalid result lacks technical_cause: {path}")
    return result


def compile_manifests(
    manifest_paths: Iterable[str | Path],
    *,
    expected_release_hashes: Mapping[str, str],
) -> CompiledAnalysis:
    paths = [Path(path) for path in manifest_paths]
    if not paths:
        raise ValueError("no durable manifests supplied")
    rows = tuple(_load_manifest(path, expected_release_hashes) for path in paths)
    valid = tuple(row for row in rows if row.get("status") not in {"infrastructure_invalid", "technical_invalid"})
    return CompiledAnalysis(rows=rows, valid_rows=valid, technical_missing=len(rows) - len(valid))


def bootstrap_mean(values: Sequence[float], *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap empty values")
    rng = random.Random(seed)
    means = [sum(rng.choices(list(values), k=len(values))) / len(values) for _ in range(draws)]
    means.sort()
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def cluster_bootstrap_mean(
    values_by_layout: Mapping[str, Sequence[float]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bootstrap layouts, keeping all condition cells within a layout together."""
    if not values_by_layout or any(not values for values in values_by_layout.values()):
        raise ValueError("each layout must have at least one value")
    rng = random.Random(seed)
    layouts = list(values_by_layout)
    means = []
    for _ in range(draws):
        sampled = [layout for layout in rng.choices(layouts, k=len(layouts))]
        means.append(sum(sum(values_by_layout[l]) / len(values_by_layout[l]) for l in sampled) / len(sampled))
    means.sort()
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def paired_signflip(values: Sequence[float], *, draws: int = SIGNFLIP_DRAWS, seed: int = SIGNFLIP_SEED) -> float:
    if not values:
        raise ValueError("cannot sign-flip empty values")
    observed = abs(sum(values) / len(values))
    rng = random.Random(seed)
    exceed = 0
    for _ in range(draws):
        sample = sum(value if rng.getrandbits(1) else -value for value in values) / len(values)
        exceed += abs(sample) >= observed
    return (exceed + 1) / (draws + 1)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        adjusted[name] = running
    return adjusted


def censoring_bounds(
    observed: Sequence[float],
    missing_count: int,
    *,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    if missing_count < 0 or not lower <= upper:
        raise ValueError("invalid censoring bounds")
    denominator = len(observed) + missing_count
    if denominator == 0:
        raise ValueError("empty cohort")
    base = sum(observed)
    return ((base + missing_count * lower) / denominator, (base + missing_count * upper) / denominator)


def equivalence(
    success_ci: tuple[float, float],
    margin_ci: tuple[float, float],
    *,
    success_margin: float = 0.10,
    relation_margin: float = 0.02,
) -> bool:
    return (
        success_ci[0] >= -success_margin and success_ci[1] <= success_margin
        and margin_ci[0] >= -relation_margin and margin_ci[1] <= relation_margin
    )
