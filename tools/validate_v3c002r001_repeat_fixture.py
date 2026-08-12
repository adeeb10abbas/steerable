#!/usr/bin/env python3
"""Target-side zero-request validation of the retained R001 smoke fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import file_binding, sha256_file
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import ContractError
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.single_server_repeat import (
    REQUIRED_PACKED_KEYS,
    exact_pi05_request,
    reconstruct_native_fixture,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("--robolab-commit", required=True)
    parser.add_argument("--robolab-client-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ContractError(f"refusing to overwrite fixture receipt: {args.output}")
    observation, manifest = reconstruct_native_fixture(args.fixture, args.manifest)
    request = exact_pi05_request(
        observation,
        "Put the Rubik's cube to the left of the bowl.",
        robolab_root=args.robolab_root,
        robolab_commit=args.robolab_commit,
        client_sha256=args.robolab_client_sha256,
    )
    if set(request) != REQUIRED_PACKED_KEYS:
        raise ContractError("target-side π0.5 packed request keys changed")
    arrays = {}
    for key, value in request.items():
        if key == "prompt":
            continue
        array = np.ascontiguousarray(value)
        arrays[key] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "bytes": array.nbytes,
            "sha256": __import__("hashlib").sha256(array.tobytes(order="C")).hexdigest(),
        }
    value = {
        "schema_version": "vla-wam-shared-v3c002r001-repeat-fixture-target-rehash-v1",
        "status": "passed_native_tree_reconstruction_and_exact_pi05_request_pack",
        "passed": True,
        "fixture": file_binding(args.fixture),
        "fixture_manifest": file_binding(args.manifest),
        "observation_payload_sha256": manifest["observation_payload_sha256"],
        "observation_structure_sha256": manifest["observation_structure_sha256"],
        "observation_leaf_count": len(manifest["leaves"]),
        "packed_request_keys": sorted(request),
        "packed_array_bindings": arrays,
        "robolab_commit": args.robolab_commit,
        "robolab_client": file_binding(args.robolab_root / "policies/pi0_family/client.py"),
        "model_request_count": 0,
        "successful_response_count": 0,
        "behavioral_action_count": 0,
        "behavioral_episode_count": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
