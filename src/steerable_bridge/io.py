from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

from .constants import HOSTED_FILES, HostedFile


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(spec: HostedFile, raw_root: Path, force: bool = False) -> Path:
    destination = raw_root / spec.local_group / spec.path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == spec.size and not force:
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(spec.url, headers={"User-Agent": "steerable-res1/0.1"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    observed_size = partial.stat().st_size
    if observed_size != spec.size:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"Size mismatch for {spec.path}: expected {spec.size}, got {observed_size}"
        )
    os.replace(partial, destination)
    return destination


def download_inputs(raw_root: Path, force: bool = False) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for spec in HOSTED_FILES:
        path = download_file(spec, raw_root, force=force)
        records.append(
            {
                "repo": spec.repo,
                "revision": spec.revision,
                "remote_path": spec.path,
                "local_path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "url": spec.url,
            }
        )
    return {"schema_version": 1, "files": records}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_audit_template(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: list[str],
    *,
    review_fields: Iterable[str],
) -> str:
    """Write a deterministic audit template without erasing completed judgments.

    Pipeline reruns are expected while the data rules are being refined. Once a
    reviewer has entered any judgment, that sheet becomes human-owned and is
    preserved. A fresh generated template is written next to it for comparison.
    """

    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
        if any(
            str(row.get(field, "")).strip()
            for row in existing
            for field in review_fields
        ):
            generated = path.with_name(f"{path.stem}.generated{path.suffix}")
            write_csv(generated, rows, fieldnames)
            return f"preserved_reviewed_sheet_generated_template_at:{generated}"
    write_csv(path, rows, fieldnames)
    return "written"
