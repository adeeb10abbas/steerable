"""Finite CPU-only recovery of the seven hash-bound oversized bj sources."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tarfile
import time
import traceback

import ijson
from historical_layout_streaming import DEFAULT_LIMITS, OBJECT_FIELDS, extract_state_payload


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    out = Path(sys.argv[1])
    requests_path = Path("/inputs/requests.json")
    request = json.loads(requests_path.read_text())
    assert ijson.__version__ == "3.5.1" and ijson.backend == "yajl2_c"
    assert request["retention_limits"] == DEFAULT_LIMITS
    assert request["retained_object_fields"] == sorted(OBJECT_FIELDS)
    assert len(request["records"]) == 7
    assert request["release_authorization"] is False
    selected = out / "selected"
    selected.mkdir()
    manifest = {
        "schema_version": "sgw-01-bounded-historical-state-stream-export-v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_sha256": digest(requests_path),
        "extractor_sha256": digest(Path("/inputs/historical_layout_streaming.py")),
        "launcher_sha256": digest(Path(__file__)),
        "source_commit": os.environ["SOURCE_COMMIT"],
        "dependency_archive_sha256": digest(out / "parser.tar.gz"),
        "python": sys.version,
        "ijson_version": ijson.__version__,
        "ijson_backend": ijson.backend,
        "job_uid": os.environ["JOB_UID"],
        "pod_uid": os.environ["POD_UID"],
        "model_requests": 0,
        "behavioral_episodes": 0,
        "allocated_gpus": 0,
        "release_permitted": False,
        "coverage_status": "named_root_only_evidence_not_complete_comparable_geometry",
        "records": [],
    }
    write_json(out / "started.json", manifest)
    for index, original in enumerate(request["records"]):
        row = dict(original)
        path = Path(row["path"])
        assert path.is_absolute() and path.resolve().is_relative_to("/data/users/ali/vla_wam/raw")
        started = time.monotonic()
        print(f"START {row['cohort_id']} {row['expected_bytes']} bytes", flush=True)
        try:
            result = extract_state_payload(
                path, expected_sha256=row["expected_sha256"],
                expected_bytes=row["expected_bytes"], limits=request["retention_limits"],
            )
        except (OSError, ValueError, ijson.JSONError) as error:
            row.update({
                "export_status": "technical_extraction_failure",
                "error": str(error), "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
            })
            print(row["traceback"], file=sys.stderr, flush=True)
        else:
            assert result["selection_contract"] == request["selection_contract"]
            destination = selected / f"{index:02d}-state-fields.json"
            write_json(destination, result)
            row.update({
                "export_status": "exported_same_stream_hash_verified_fields",
                "export_path": destination.relative_to(out).as_posix(),
                "export_sha256": digest(destination),
                "export_bytes": destination.stat().st_size,
                "retained_record_counts": {
                    key: len(value) for key, value in result.items()
                    if isinstance(value, list)
                },
                "semantic_status": result["semantic_status"],
            })
        row["elapsed_seconds"] = time.monotonic() - started
        row["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(selected / f"{index:02d}-receipt.json", row)
        manifest["records"].append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(out / "manifest.json", manifest)
    archive = out / "fields.tar.gz"
    with tarfile.open(archive, "x:gz") as bundle:
        bundle.add(out / "manifest.json", arcname="manifest.json")
        for path in sorted(selected.iterdir()):
            bundle.add(path, arcname=path.relative_to(out).as_posix())
    with archive.open("rb") as stream:
        os.fsync(stream.fileno())
    receipt = {
        "path": str(archive), "bytes": archive.stat().st_size, "sha256": digest(archive),
        "failed_sources": sum(
            row["export_status"] == "technical_extraction_failure" for row in manifest["records"]
        ),
        "model_requests": 0, "behavioral_episodes": 0, "release_permitted": False,
    }
    write_json(out / "receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 1 if receipt["failed_sources"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
