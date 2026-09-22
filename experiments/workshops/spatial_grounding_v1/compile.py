"""Reproducible, fail-closed analysis from durable SGW-01 manifests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import csv
import argparse
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2_0260_922
SIGNFLIP_DRAWS = 100_000
SIGNFLIP_SEED = 2_0260_923
PRIMARY_TEST_NAMES = (
    "N3-LAT", "N3-HEIGHT", "N3-DIST",
    "D1-LAT", "D1-HEIGHT", "D1-DIST",
)


@dataclass(frozen=True)
class CompiledAnalysis:
    rows: tuple[Mapping[str, Any], ...]
    valid_rows: tuple[Mapping[str, Any], ...]
    technical_missing: int
    incomplete_layouts: tuple[str, ...] = ()
    complete: bool = True


@dataclass(frozen=True)
class RegisteredCompilation:
    ledger: tuple[Mapping[str, Any], ...]
    confirmation_estimates: tuple[Mapping[str, Any], ...]
    complete: bool
    missing_cell_ids: tuple[str, ...]
    duplicate_cell_ids: tuple[str, ...] = ()
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
        artifact = (manifest_path.parent / name).resolve()
        if not artifact.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError(f"artifact path escapes attempt: {artifact}")
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
        release_root = path.parent.parent.resolve()
        attempt_manifest = (release_root / raw["manifest_path"]).resolve()
        if not attempt_manifest.is_relative_to(release_root):
            raise ValueError(f"manifest path escapes release: {path}")
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
    expected_layout_ids: Iterable[str] | None = None,
) -> CompiledAnalysis:
    paths = [Path(path) for path in manifest_paths]
    if not paths:
        raise ValueError("no durable manifests supplied")
    rows = tuple(_load_manifest(path, expected_release_hashes) for path in paths)
    valid = tuple(row for row in rows if row.get("status") not in {"infrastructure_invalid", "technical_invalid"})
    expected = set(expected_layout_ids or ())
    observed = {str(row["layout_id"]) for row in rows if row.get("layout_id") is not None}
    incomplete = tuple(sorted(expected - observed))
    return CompiledAnalysis(
        rows=rows, valid_rows=valid, technical_missing=len(rows) - len(valid),
        incomplete_layouts=incomplete, complete=not incomplete,
    )


def compile_registered_queue(
    queue_csv: str | Path,
    completion_pointers: Iterable[str | Path],
    *,
    expected_release_hashes: Mapping[str, str],
) -> RegisteredCompilation:
    """Compile the complete registered ledger; omitted cells remain not_run."""
    with Path(queue_csv).open(newline="") as stream:
        planned = list(csv.DictReader(stream))
    if not planned or any(not row.get("cell_id") for row in planned):
        raise ValueError("queue is empty or has an invalid cell_id")
    planned_by_id = {row["cell_id"]: row for row in planned}
    if len(planned_by_id) != len(planned):
        raise ValueError("queue contains duplicate cell_id")
    ledger = {cell_id: {**row, "analysis_status": "not_run"} for cell_id, row in planned_by_id.items()}
    seen: set[str] = set()
    duplicate: list[str] = []
    for pointer_path in completion_pointers:
        pointer = Path(pointer_path)
        raw_pointer = json.loads(pointer.read_text())
        cell_id = raw_pointer.get("cell_id")
        if cell_id not in planned_by_id:
            raise ValueError(f"completion pointer is outside registered queue: {pointer}")
        if cell_id in seen:
            duplicate.append(cell_id)
            raise ValueError(f"duplicate completion pointer for cell: {cell_id}")
        result = _load_manifest(pointer, expected_release_hashes)
        if result.get("cell_id") not in {None, cell_id}:
            raise ValueError(f"result cell_id mismatch: {pointer}")
        seen.add(cell_id)
        ledger[cell_id] = {**planned_by_id[cell_id], **result, "analysis_status": (
            "complete" if result.get("status") in {"valid_success", "valid_model_failure", "censored"}
            else "incomplete"
        )}
    missing = tuple(sorted(set(planned_by_id) - seen))
    for cell_id in missing:
        ledger[cell_id]["analysis_status"] = "not_run"
    rows = tuple(ledger[cell_id] for cell_id in planned_by_id)
    confirmation = _confirmation_estimates(rows)
    return RegisteredCompilation(
        ledger=rows,
        confirmation_estimates=confirmation,
        complete=not missing and not duplicate and not any(row["analysis_status"] == "incomplete" for row in rows),
        missing_cell_ids=missing,
        duplicate_cell_ids=tuple(sorted(duplicate)),
    )


def _confirmation_estimates(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("stage") == "C":
            groups.setdefault((str(row.get("model")), str(row.get("family")), str(row.get("layout_id"))), []).append(row)
    estimates: list[Mapping[str, Any]] = []
    for (model, family, layout), cells in sorted(groups.items()):
        by_form_goal = {(row.get("form"), row.get("physical_goal_sign")): row for row in cells}
        complete = len(by_form_goal) == 6 and all(
            row.get("analysis_status") == "complete" and row.get("S") is not None
            for row in cells
        )
        if not complete:
            estimates.append({"model": model, "family": family, "layout_id": layout, "status": "incomplete"})
            continue
        mean_by_form = {
            form: sum(float(by_form_goal[(form, sign)]["S"]) for sign in (-1, 1)) / 2
            for form in ("D", "C", "I")
        }
        margin_by_form = {
            form: [by_form_goal[(form, sign)].get("terminal_margin_m") for sign in (-1, 1)]
            for form in ("D", "C", "I")
        }
        margin_available = all(all(value is not None for value in values) for values in margin_by_form.values())
        estimates.append({
            "model": model, "family": family, "layout_id": layout, "status": "complete",
            "delta_I_C": mean_by_form["I"] - mean_by_form["C"],
            "delta_C_D": mean_by_form["C"] - mean_by_form["D"],
            "separation_by_form": {
                form: float(by_form_goal[(form, 1)]["S"]) - float(by_form_goal[(form, -1)]["S"])
                for form in ("D", "C", "I")
            },
            "margin_status": "conditional" if margin_available else "unavailable",
            "delta_M_I_C": (
                sum(float(v) for v in margin_by_form["I"]) / 2
                - sum(float(v) for v in margin_by_form["C"]) / 2
                if margin_available else None
            ),
        })
    return tuple(estimates)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap empty values")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    rng = random.Random(seed)
    means = [sum(rng.choices(list(values), k=len(values))) / len(values) for _ in range(draws)]
    means.sort()
    tail = (1 - confidence) / 2
    return means[int(tail * draws)], means[int((1 - tail) * draws) - 1]


def bootstrap_mean(values: Sequence[float], *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    return bootstrap_mean_ci(values, draws=draws, seed=seed, confidence=0.95)


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


def compile_primary_statistics(compiled: RegisteredCompilation) -> tuple[Mapping[str, Any], ...]:
    """Summarize only complete C-layout clusters; missing families stay unavailable."""
    result: list[Mapping[str, Any]] = []
    for model in ("N3", "D1"):
        for family in ("LAT", "HEIGHT", "DIST"):
            values = [
                float(row["delta_I_C"])
                for row in compiled.confirmation_estimates
                if row["model"] == model and row["family"] == family and row["status"] == "complete"
            ]
            if not values:
                result.append({"model": model, "family": family, "status": "unavailable"})
                continue
            layout_values = {
                str(index): [value] for index, value in enumerate(values)
            }
            ci95 = cluster_bootstrap_mean(layout_values)
            ci90 = bootstrap_mean_ci(values, confidence=0.90)
            result.append({
                "model": model, "family": family, "status": "complete",
                "n_layouts": len(values), "delta_I_C_ci95": ci95,
                "delta_I_C_ci90": ci90,
                "equivalent_success_relation": None,
                "p_unadjusted": paired_signflip(values),
            })
    adjusted = holm_adjust_primary({
        f"{row['model']}-{row['family']}": row["p_unadjusted"]
        for row in result if row["status"] == "complete"
    })
    return tuple({**row, "p_holm": adjusted[f"{row['model']}-{row['family']}"]}
                 for row in result)


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


def holm_adjust_primary(p_values: Mapping[str, float]) -> dict[str, float]:
    """Adjust all six prespecified model-family tests, filling absent tests as 1."""
    complete = {name: float(p_values.get(name, 1.0)) for name in PRIMARY_TEST_NAMES}
    return holm_adjust(complete)


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
        -success_margin < success_ci[0] and success_ci[1] < success_margin
        and -relation_margin < margin_ci[0] and margin_ci[1] < relation_margin
    )


def render_neutral_coverage_table(compiled: RegisteredCompilation) -> str:
    """Render a neutral, no-results table suitable for a generated artifact."""
    lines = ["model, family, stage, planned, complete, not_run, incomplete"]
    keys = sorted({(row.get("model"), row.get("family"), row.get("stage")) for row in compiled.ledger})
    for model, family, stage in keys:
        group = [row for row in compiled.ledger if (row.get("model"), row.get("family"), row.get("stage")) == (model, family, stage)]
        lines.append(f"{model},{family},{stage},{len(group)},"
                     f"{sum(row['analysis_status'] == 'complete' for row in group)},"
                     f"{sum(row['analysis_status'] == 'not_run' for row in group)},"
                     f"{sum(row['analysis_status'] == 'incomplete' for row in group)}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile SGW-01 hash-bound completion pointers")
    parser.add_argument("--release", type=Path, required=True, help="immutable release directory")
    parser.add_argument("--queue", type=Path, required=True, help="registered planned_cells.csv")
    parser.add_argument("--pointer", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    release = args.release.resolve()
    hashes_path = release / "hashes.json"
    if not hashes_path.is_file():
        parser.error(f"missing release hashes: {hashes_path}")
    hashes = json.loads(hashes_path.read_text())
    expected_hashes = hashes.get("hashes", hashes) if isinstance(hashes, Mapping) else None
    if not isinstance(expected_hashes, Mapping) or not expected_hashes:
        parser.error("hashes.json must contain a nonempty object")
    pointers = args.pointer if args.pointer is not None else sorted((release / "cells").glob("*.complete.json"))
    for pointer in pointers:
        resolved = pointer.resolve()
        if not resolved.is_relative_to(release):
            parser.error(f"completion pointer outside release: {pointer}")
    compiled = compile_registered_queue(args.queue, pointers, expected_release_hashes=expected_hashes)
    primary = compile_primary_statistics(compiled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema_version": "sgw-01-analysis-v1",
        "complete": compiled.complete,
        "missing_cell_ids": compiled.missing_cell_ids,
        "duplicate_cell_ids": compiled.duplicate_cell_ids,
        "ledger": compiled.ledger,
        "confirmation_estimates": compiled.confirmation_estimates,
        "primary_statistics": primary,
        "coverage_table": render_neutral_coverage_table(compiled),
    }, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
