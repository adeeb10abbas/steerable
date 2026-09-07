#!/usr/bin/env python3
"""Replicate a rendered V4 k8s bundle onto a new homogeneous GPU stratum."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import v4_gpu_scheduling as gpu_scheduling  # noqa: E402

JOB_SHA_RE = re.compile(
    r'(name: LANE_LAUNCH_CONFIG_SHA256\n\s+value: ")([0-9a-f]{64})(")'
)


def _canonical_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _rewrite_bundle(
    dest: Path,
    *,
    from_attempt_id: str,
    to_attempt_id: str,
    from_gpu_product: str,
    to_gpu_product: str,
) -> None:
    to_display = gpu_scheduling.display_name_for_product(to_gpu_product)
    from_display = gpu_scheduling.display_name_for_product(from_gpu_product)

    seed_configmaps = sorted(
        path
        for path in dest.glob("s*-configmap.yaml")
        if re.fullmatch(r"s\d{3}-configmap\.yaml", path.name)
    )
    for config_path in seed_configmaps:
        text = config_path.read_text(encoding="utf-8")
        marker = "simulator-launch.json: |\n    "
        start = text.index(marker) + len(marker)
        end = text.index("\n  image.digest:", start)
        launch = json.loads(text[start:end])
        launch["gpu_product"] = to_gpu_product
        launch["expected_gpu_name"] = to_display
        launch["allowed_gpu_names"] = [to_display]
        launch_json = json.dumps(launch, sort_keys=True, separators=(",", ":")) + "\n"
        launch_sha = hashlib.sha256(launch_json.encode("utf-8")).hexdigest()
        launch_block = marker + launch_json
        updated = text[: text.index(marker)] + launch_block + text[end:]
        updated = updated.replace(from_attempt_id, to_attempt_id)
        config_path.write_text(updated, encoding="utf-8")

        suffix = config_path.name.replace("-configmap.yaml", "")
        job_path = dest / f"{suffix}-job.yaml"
        job_text = job_path.read_text(encoding="utf-8")
        job_text = job_text.replace(from_attempt_id, to_attempt_id)
        job_text = job_text.replace(from_gpu_product, to_gpu_product)
        job_text = job_text.replace(from_display, to_display)
        def _replace_sha(match: re.Match[str]) -> str:
            return f'{match.group(1)}{launch_sha}{match.group(3)}'

        job_text = JOB_SHA_RE.sub(_replace_sha, job_text, count=1)
        job_path.write_text(job_text, encoding="utf-8")

    for path in dest.glob("*.yaml"):
        if re.fullmatch(r"s\d{3}-(?:configmap|job)\.yaml", path.name):
            continue
        text = path.read_text(encoding="utf-8")
        updated = text.replace(from_attempt_id, to_attempt_id)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def _trim_to_max_seeds(dest: Path, max_seeds: int) -> None:
    if max_seeds < 1:
        raise ValueError("max_seeds must be positive")
    for path in sorted(dest.glob("s*-*.yaml")):
        match = re.fullmatch(r"s(\d{3})-(configmap|job)\.yaml", path.name)
        if match is None:
            continue
        if int(match.group(1)) >= max_seeds:
            path.unlink()
    resources = sorted(
        name
        for name in (p.name for p in dest.glob("*.yaml"))
        if name == "kustomization.yaml"
        or name == "scripts-configmap.yaml"
        or name == "bundle-manifest.json"
        or re.fullmatch(r"s\d{3}-(?:configmap|job)\.yaml", name)
    )
    kustom = dest / "kustomization.yaml"
    kustom.write_text(
        "apiVersion: kustomize.config.k8s.io/v1beta1\n"
        "kind: Kustomization\n"
        "resources:\n"
        + "".join(f"  - {name}\n" for name in resources if name != "kustomization.yaml"),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--from-attempt-id", required=True)
    parser.add_argument("--to-attempt-id", required=True)
    parser.add_argument("--from-gpu-product", default="NVIDIA-A40")
    parser.add_argument("--to-gpu-product", required=True)
    parser.add_argument("--max-seeds", type=int, default=None, help="Keep only first N seed jobs")
    parser.add_argument("--kube-context", default=None)
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args(argv)

    if args.create and not args.kube_context:
        parser.error("--kube-context is required with --create")

    source = args.source_bundle.resolve()
    bundle_name = source.name.replace(args.from_attempt_id, args.to_attempt_id)
    dest = args.output_root.resolve() / bundle_name
    if dest.exists():
        raise SystemExit(f"destination already exists: {dest}")
    shutil.copytree(source, dest)
    _rewrite_bundle(
        dest,
        from_attempt_id=args.from_attempt_id,
        to_attempt_id=args.to_attempt_id,
        from_gpu_product=args.from_gpu_product,
        to_gpu_product=args.to_gpu_product,
    )
    if args.max_seeds is not None:
        _trim_to_max_seeds(dest, args.max_seeds)

    manifest_path = dest / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["attempt_id"] = args.to_attempt_id
    manifest["gpu_product"] = args.to_gpu_product
    manifest["gpu_product_allowlist"] = [args.to_gpu_product]
    manifest["status"] = "rendered_not_created"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps({"bundle_root": str(dest), "attempt_id": args.to_attempt_id}, indent=2))
    if not args.create:
        return 0

    completed = subprocess.run(
        ["kubectl", "--context", args.kube_context, "create", "-k", str(dest)],
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
