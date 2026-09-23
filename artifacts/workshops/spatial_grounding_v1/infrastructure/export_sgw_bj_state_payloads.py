from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

request_bytes = sys.stdin.buffer.read(100001)
assert len(request_bytes) <= 100000
requests = json.loads(request_bytes)
assert len(requests["records"]) == 12
root = Path("/data/users/ali/sgw-01/infrastructure/historical-export-20260923bj/state-payloads")
root.mkdir(exist_ok=False)
allowed = Path("/data/users/ali/vla_wam/raw")
rows = []
total = 0
for index, request in enumerate(requests["records"]):
    path = Path(request["path"])
    assert path.is_absolute() and path.resolve().is_relative_to(allowed)
    assert path.name == "state_repair_result.json"
    row = dict(request)
    if not path.is_file():
        row["export_status"] = "source_missing"
    else:
        with path.open("rb") as stream:
            raw = stream.read(8 * 1024**2 + 1)
        if len(raw) > 8 * 1024**2 or total + len(raw) > 32 * 1024**2:
            row["export_status"] = "source_exceeds_bounded_export_size"
        else:
            digest = hashlib.sha256(raw).hexdigest()
            row.update({"actual_sha256": digest, "bytes": len(raw)})
            if digest != request["expected_sha256"] or len(raw) != request["expected_bytes"]:
                row["export_status"] = "source_hash_or_size_mismatch"
            else:
                name = f"{index:02d}-state.json"
                with (root / name).open("xb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                row.update({"export_status": "exported_hash_verified_payload", "export_path": name})
                total += len(raw)
    rows.append(row)
manifest = {
    "schema_version": "sgw-01-bounded-historical-child-payload-export-v1",
    "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
    "parent_archive_sha256": requests["parent_archive_sha256"],
    "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    "job_uid": os.environ["JOB_UID"],
    "pod_uid": os.environ["POD_UID"],
    "records": rows,
    "model_requests": 0,
    "behavioral_episodes": 0,
    "allocated_gpus": 0,
    "release_permitted": False,
}
for name, raw in (
    ("requests.json", request_bytes),
    ("manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()),
):
    with (root / name).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
archive = root / "payloads.tar.gz"
with tarfile.open(archive, "x:gz") as bundle:
    for name in ["requests.json", "manifest.json"] + [r["export_path"] for r in rows if "export_path" in r]:
        bundle.add(root / name, arcname=name)
with archive.open("rb") as stream:
    os.fsync(stream.fileno())
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
print(json.dumps({
    "path": str(archive), "bytes": archive.stat().st_size, "sha256": digest,
    "records": [{k: v for k, v in row.items() if k in ("cohort_id", "export_status", "bytes")} for row in rows],
}), flush=True)
