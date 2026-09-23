"""Export only hash-bound historical receipts; no simulator or model imports."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile


ROOT = Path("/data/users/ali/vla_wam/raw/v3e006_r005")
ATTEMPT = ROOT / "state_repair/98f0234-a40r06-attempt01"


def read(path, size, sha256):
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"historical input binding differs: {path}")
    return raw


failure_binding = {
    "path": str(ATTEMPT / "raw/state_construction_failure.json"),
    "bytes": 25614537,
    "sha256": "e858a4431bd0451a1e380d66c8c5d05c5486718f5b23388f874d24a7d271ae8e",
}
failure = json.loads(read(
    Path(failure_binding["path"]), failure_binding["bytes"], failure_binding["sha256"],
))
selected = {
    "schema_version": "sgw-01-r005-historical-controller-verification-v1",
    "failure_report": failure_binding,
    "json_pointer": "/controller_source_verification_before_AppLauncher",
    "value": failure["controller_source_verification_before_AppLauncher"],
}
files = {
    "controller-verification.json": (json.dumps(selected, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    "launch.json": read(
        ATTEMPT / "launch.json", 17730,
        "8781e2bff1b04162b4b72f1129f70e2862a40525c53e12e5235cbc3b404e3e10",
    ),
    "target-validation.json": read(
        ROOT / "validation/98f0234-a40r06-attempt01.json", 2458,
        "d72a0cde9145ea63c7f6c9def06ca2e91d3b665812677d0c6d126c2a29db8841",
    ),
}
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
    for name, raw in files.items():
        info = tarfile.TarInfo(name)
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
