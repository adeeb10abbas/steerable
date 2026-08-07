#!/usr/bin/env python3
"""Recover compile-only DreamZero evidence after concurrent manifest co-batching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.v3.dreamzero_droid.future_retention import (
    file_record,
    identify_and_partition_session,
    write_session_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--future-root", type=Path, required=True)
    args = parser.parse_args()

    export_path = args.export.resolve()
    export = json.loads(export_path.read_text())
    original_trace_path = Path(export["trace_manifest_path"]).resolve()
    original_trace = json.loads(original_trace_path.read_text())
    if original_trace.get("future_manifest"):
        raise ValueError("DreamZero trace already has a future manifest")
    raw_path = Path(original_trace["returned_raw_chunks"]["path"])
    chunks = np.load(raw_path, allow_pickle=False)
    session_manifest = identify_and_partition_session(
        args.future_root,
        prompt=str(original_trace["prompt"]),
        returned_raw_chunks=chunks,
    )

    stem = original_trace_path.name.removesuffix("_executed_actions.json")
    session_path = original_trace_path.with_name(f"{stem}_future_manifest_recovered.json")
    recovered_trace_path = original_trace_path.with_name(
        f"{stem}_executed_actions_recovered.json"
    )
    recovered_export_path = export_path.with_name("simulator_export_recovered.json")
    for output in (session_path, recovered_trace_path, recovered_export_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite recovered evidence: {output}")
    write_session_manifest(session_path, session_manifest)

    recovered_trace = dict(original_trace)
    recovered_trace["future_manifest"] = {
        **file_record(session_path),
        "request_count": len(session_manifest["requests"]),
        "official_decode_count": len(session_manifest["official_reset_decode"]),
    }
    recovered_trace["future_recovery"] = {
        "method": "exact_server_session_id_then_client_action_tensor_equality",
        "original_trace": file_record(original_trace_path),
        "preserved_infrastructure_failure": True,
        "session_id": session_manifest["session_id"],
    }
    recovered_trace_path.write_text(
        json.dumps(recovered_trace, indent=2, sort_keys=True) + "\n"
    )

    recovered_export = dict(export)
    recovered_export["trace_manifest_path"] = str(recovered_trace_path)
    recovered_export["future_recovery"] = {
        "original_export": file_record(export_path),
        "recovered_trace": file_record(recovered_trace_path),
    }
    recovered_export_path.write_text(
        json.dumps(recovered_export, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "recovered_export": str(recovered_export_path),
        "recovered_trace": str(recovered_trace_path),
        "session_future_manifest": str(session_path),
        "session_id": session_manifest["session_id"],
    }, indent=2))


if __name__ == "__main__":
    main()
