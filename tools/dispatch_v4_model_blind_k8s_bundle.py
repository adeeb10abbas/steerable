#!/usr/bin/env python3
"""Render and kubectl-create one immutable V4 model-blind G2/G3 bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import v4_gpu_scheduling as gpu_scheduling  # noqa: E402
import v4_dispatch_gates as dispatch_gates  # noqa: E402


def _load_renderer(schema_version: str):
    if "g2" in schema_version:
        import render_v4_horizontal_g2_k8s_jobs as renderer

        return renderer
    if "g3" in schema_version and "scripted" in schema_version:
        import render_v4_horizontal_g3_scripted_k8s_jobs as renderer

        return renderer
    if "g3" in schema_version:
        import render_v4_horizontal_g3_k8s_jobs as renderer

        return renderer
    raise ValueError(f"unsupported schema_version: {schema_version}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _patch_spec(
    spec: dict[str, Any],
    *,
    spec_dir: Path,
    gpu_product: str | None,
    attempt_id: str | None,
) -> dict[str, Any]:
    patched = copy.deepcopy(spec)
    if gpu_product is not None:
        patched["gpu_product"] = gpu_product
        patched["expected_gpu_name"] = gpu_scheduling.display_name_for_product(
            gpu_product
        )
        patched.pop("gpu_product_allowlist", None)
    if attempt_id is not None:
        patched["attempt_id"] = attempt_id
    gpu_scheduling.resolve_model_blind_scheduling(patched)
    return patched


def _patch_spec_optional_fields(spec: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    patched = copy.deepcopy(spec)
    for key, value in overrides.items():
        if value is not None:
            patched[key] = value
    return patched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-product", default=None)
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--max-seed-jobs", type=int, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1-seed smoke dispatch; skips smoke-before-wave gate",
    )
    parser.add_argument("--publisher-pod", default="211247-sz5vjy-vla4-b200-4gpu")
    parser.add_argument("--skip-cluster-gates", action="store_true")
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args(argv)

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if args.max_seed_jobs is not None:
        spec = _patch_spec_optional_fields(spec, {"max_seed_jobs": args.max_seed_jobs})
    patched = _patch_spec(
        spec,
        spec_dir=args.spec.resolve().parent,
        gpu_product=args.gpu_product,
        attempt_id=args.attempt_id,
    )
    spec_path = args.spec.resolve().parent / (
        f".dispatch-{patched['attempt_id']}.render-spec.json"
    )
    spec_path.write_text(
        json.dumps(patched, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    renderer = _load_renderer(str(patched["schema_version"]))
    report = renderer.render(spec_path, args.output_root)
    bundle_root = Path(report["bundle_root"])
    print(json.dumps(report, indent=2, sort_keys=True))

    if not args.create:
        return 0

    is_smoke = bool(
        args.smoke
        or (args.max_seed_jobs is not None and int(args.max_seed_jobs) == 1)
    )
    fixture_id = str(patched.get("fixture_id") or "horizontal")
    gate = "G2" if "g2" in str(patched["schema_version"]) else "G3"
    gpu_product = str(patched["gpu_product"])
    if not args.skip_cluster_gates:
        try:
            gate_report = dispatch_gates.enforce_dispatch_gates(
                bundle_root=bundle_root,
                fixture_id=fixture_id,
                gate=gate,
                gpu_product=gpu_product,
                kube_context=str(patched["kube_context"]),
                namespace=str(patched["namespace"]),
                publisher_pod=str(args.publisher_pod),
                expected_study_commit=str(patched.get("expected_study_commit") or ""),
                is_smoke=is_smoke,
            )
        except dispatch_gates.DispatchGateError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps({"dispatch_gates": gate_report}, indent=2, sort_keys=True))

    kube_context = str(patched["kube_context"])
    completed = subprocess.run(
        ["kubectl", "--context", kube_context, "create", "-k", str(bundle_root)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(completed.stderr or completed.stdout, file=sys.stderr)
        return completed.returncode
    print(completed.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
