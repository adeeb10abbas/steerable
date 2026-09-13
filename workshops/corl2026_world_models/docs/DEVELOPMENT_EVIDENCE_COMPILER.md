# Development evidence compiler

`analysis/compile_development_evidence.py` is a CPU-only adapter for the retained
development recordings. It authenticates each passed aggregate, cell receipt,
recorder completion, journal hash chain, recorder payload, official request
receipt, retained future artifact, and returned/executed action identity. It
does not import N3, DreamZero, RoboLab, PyTorch, NumPy, or a simulator, and it
does not issue a request, reset, or action.

Formal mode fails closed unless exactly the planned 16 N3 and 16 D1
development cells are supplied as four intact passed layout blocks per model.
`diagnostic_partial` still requires all four N3 blocks and at least one intact
passed D1 block. Its timing files deliberately use
`wmf-development-timing-request-inventory-diagnostic-v1`, which the timing
binder rejects, and every diagnostic receipt says it is unsafe for formal
release.

For D1, compilation independently authenticates all three exact input mappings
(`raw_inputs`, `converted_inputs`, and `normalized_model_inputs`) down to their
sorted keys, declared structure, exact tensor/array bytes, content digests, and
artifact descriptors. It also authenticates the recorder's packed model-request
artifact and binds its current/preceding observation IDs to the completion
inventory. For both models, the compiler checks the pinned model source and
checkpoint identities and the full per-episode reset/context chain before it
emits downstream evidence.

Formal D1 compilation authenticates the exact retained modulo-four decode
schedule established by the pinned source and the all-912-request cache
diagnostic: request indices divisible by four contain a three-latent
conditioning-origin tensor and decode to nine RGB frames; the intervening
requests contain two generated latents and decode to five RGB frames. Exact
latent, decoded-tensor, and decoded-RGB shapes are required. This acceptance is
structural only. The compiler never attaches boundary-3/6 timing semantics to a
standalone five-frame decode.

The signed D1 v2 timing sidecar is the sole request-level timing-eligibility
authority. It authenticates all 912 retained requests, maps only the 240
source-proven full conditioning-origin decodes, and marks all 672 incremental
standalone decodes timing-unmapped with zero target bindings. Of the mapped
requests, 224 have both native-clock targets inside the executed prefix and 16
final requests are prefix-truncated, yielding 448 matched target bindings and
32 explicit truncations. The release consumer preserves these missingness
counts and excludes every timing-unmapped request from alignment.

The compiler never creates resource measurements, camera crops, pixel-blindness
review receipts, rater labels, a movement threshold, or a confirmation release.
It records those omissions explicitly. The source viewport MP4 is retained only
as the episode-video identity; it is not used to infer physical time.

## Input contract

The manifest is immutable input with this exact top-level shape:

```json
{
  "schema_version": "wmf-development-evidence-compiler-input-v1",
  "study_id": "WMF-ABLATION-001",
  "mode": "formal_full",
  "raw_root": "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912",
  "camera_id": "over_shoulder_left_camera",
  "planned_cells": {"path": "...", "sha256": "...", "bytes": 0},
  "aggregate_receipts": [
    {"model_id": "N3", "layout_pair_id": "D01", "receipt": {"path": "...", "sha256": "...", "bytes": 0}},
    {"model_id": "D1", "layout_pair_id": "D01", "receipt": {"path": "...", "sha256": "...", "bytes": 0}}
  ]
}
```

There must be one row for every `(model_id, layout_pair_id)` in
`{N3,D1} x {D01,D02,D03,D04}` in formal mode. The planned-cell CSV is pinned to
SHA-256 `7d06120a56419877d1acdfdc498c6dc054bf6860ce6bda5b2a25f55ae3b4166e`.
Every behavioral receipt and artifact must remain beneath `raw_root`; symlinked
paths are rejected.

## Exact formal compiler commands on the PVC

Run this only after D01-D04 each have one terminal passed D1 aggregate. Use the
immutable staged source commit containing this compiler. The discovery code
fails if any model/layout has zero or more than one passed aggregate, so it
cannot silently choose among contradictory successes.

```bash
export WMF_RAW_ROOT=/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912
export WMF_SOURCE_ROOT="$WMF_RAW_ROOT/control/sources/REPLACE_WITH_COMPILER_SOURCE_COMMIT"
export WMF_COMPILER_RUN="$WMF_RAW_ROOT/derived/development-evidence-formal-001"
mkdir -p "$WMF_RAW_ROOT/derived"
mkdir "$WMF_COMPILER_RUN"

python3 - "$WMF_RAW_ROOT" "$WMF_SOURCE_ROOT" "$WMF_COMPILER_RUN/input_manifest.json" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

def reject_symlink_components(path, label):
    candidate = Path(path)
    lexical = candidate if candidate.is_absolute() else Path.cwd() / candidate
    cursor = lexical
    while True:
        if cursor.is_symlink():
            raise SystemExit(f"{label} contains symlink component: {cursor}")
        if cursor.parent == cursor:
            return
        cursor = cursor.parent

raw_root_supplied = Path(sys.argv[1])
source_root_supplied = Path(sys.argv[2])
reject_symlink_components(raw_root_supplied, "raw root")
reject_symlink_components(source_root_supplied, "source root")
raw_root = raw_root_supplied.resolve()
source_root = source_root_supplied.resolve()
output = Path(sys.argv[3])

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def descriptor(path):
    reject_symlink_components(path, "descriptor path")
    resolved = path.resolve()
    if not resolved.is_file():
        raise SystemExit(f"not an immutable regular file: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}

patterns = {
    "N3": "control/jobs/n3-development-d??-*/publish/n3_behavioral_development_receipt.json",
    "D1": "control/jobs/d1-development-d??-simulator-*/publish/d1_behavioral_development_receipt.json",
}
rows = []
for model in ("N3", "D1"):
    for layout in ("D01", "D02", "D03", "D04"):
        passed = []
        for path in sorted(raw_root.glob(patterns[model])):
            value = json.loads(path.read_text(encoding="utf-8"))
            if (value.get("status") == "passed" and value.get("exit_code") == 0
                    and value.get("study_id") == "WMF-ABLATION-001"
                    and value.get("model_config") == model
                    and value.get("layout_pair_id") == layout):
                passed.append(path)
        if len(passed) != 1:
            raise SystemExit(f"expected exactly one passed {model} {layout} aggregate, found {passed}")
        rows.append({"model_id": model, "layout_pair_id": layout, "receipt": descriptor(passed[0])})

planned = source_root / "workshops/corl2026_world_models/experiments/forecast_layout/planned_cells.csv"
manifest = {
    "schema_version": "wmf-development-evidence-compiler-input-v1",
    "study_id": "WMF-ABLATION-001",
    "mode": "formal_full",
    "raw_root": str(raw_root),
    "camera_id": "over_shoulder_left_camera",
    "planned_cells": descriptor(planned),
    "aggregate_receipts": rows,
}
with output.open("x", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

export WMF_MANIFEST_SHA256="$(sha256sum "$WMF_COMPILER_RUN/input_manifest.json" | awk '{print $1}')"
python3 "$WMF_SOURCE_ROOT/workshops/corl2026_world_models/analysis/compile_development_evidence.py" \
  --manifest "$WMF_COMPILER_RUN/input_manifest.json" \
  --manifest-sha256 "$WMF_MANIFEST_SHA256" \
  --output-dir "$WMF_COMPILER_RUN/bundle"
sha256sum "$WMF_COMPILER_RUN/bundle/compiler_receipt.json"
```

The successful receipt must say `formal_cohort_complete:true`,
`safe_for_timing_binding:true`, `compiled_cells:32`, source requests `1152`,
and source behavioral actions `14400`; all fields in `compiler_science_activity`
must remain zero. Here `safe_for_timing_binding` means the complete immutable
inventory may be presented to a timing validator; it does not assert that every
request has a valid timing mapping.

The canonical compiler timing inventories independently fix request order and
identity. N3 can be rebound directly from its canonical compiler inventory:

```bash
export WMF_TIMING_TOOL="$WMF_SOURCE_ROOT/workshops/corl2026_world_models/analysis/qualify_forecast_timing.py"

python3 "$WMF_TIMING_TOOL" bind-development \
  --authority "$WMF_RAW_ROOT/control/jobs/timing-n3-native-authority-002/raw/n3_timing_authority.json" \
  --authority-sha256 fe4bc0056489bbc752824212f7fed96df99a857c84736c767e1cca278f2adb22 \
  --request-inventory "$WMF_COMPILER_RUN/bundle/n3_development_timing_request_inventory.json" \
  --request-inventory-sha256 "$(sha256sum "$WMF_COMPILER_RUN/bundle/n3_development_timing_request_inventory.json" | awk '{print $1}')" \
  --output "$WMF_COMPILER_RUN/n3_development_timing.json"

python3 "$WMF_TIMING_TOOL" validate-development \
  --timing "$WMF_COMPILER_RUN/n3_development_timing.json" \
  --sha256 "$(sha256sum "$WMF_COMPILER_RUN/n3_development_timing.json" | awk '{print $1}')" \
  --model N3
```

For D1, use the passed missingness-aware attempt002 sidecar and its exact
signed inventory. The independently published cache diagnostic is hash-bound
to that signed inventory, so it must not be combined with the byte-different
canonical unsigned compiler inventory merely because their request lists are
expected to agree. The freeze consumer independently requires the sidecar's
ordered 912 request hashes to equal the compiler-derived request hashes.

```bash
export WMF_D1_TIMING_JOB="$WMF_RAW_ROOT/control/jobs/timing-d1-development-sidecar-002"
export WMF_D1_TIMING="$WMF_D1_TIMING_JOB/raw/d1_development_timing_sidecar.json"
export WMF_D1_TIMING_RECEIPT="$WMF_D1_TIMING_JOB/publish/timing_job_receipt.json"

test -f "$WMF_D1_TIMING"
test -f "$WMF_D1_TIMING_RECEIPT"
python3 "$WMF_TIMING_TOOL" validate-development \
  --timing "$WMF_D1_TIMING" \
  --sha256 "$(sha256sum "$WMF_D1_TIMING" | awk '{print $1}')" \
  --model D1
```

The validated D1 document must have schema
`wmf-native-generated-target-timing-v2`, status
`qualified_subset_from_native_runtime_metadata_with_explicit_missingness`, and
the exact coverage `912 total / 240 mapped / 672 timing-unmapped / 224 matched /
16 prefix-truncated / 448 matched targets / 32 truncated targets`. The cache
diagnostic at
`control/jobs/timing-d1-development-sidecar-cache-schedule-diagnostic-001/publish/diagnostic.json`
has SHA-256
`cb13bee1cb6aadbeef9acb8cad5417677b2c3c18ab9fc115ca1eb3fe1f2e0630`.

These commands close compilation and timing-sidecar provenance only. They do
not authorize a camera crop, invent a resource receipt, create a movement
threshold, substitute for either human rater, or release confirmation.

## Queue-wrapper validation boundary

A queue wrapper must treat a zero compiler exit code as necessary but not
sufficient. Before publishing a formal bundle, it must independently reload the
signed `compiler_receipt.json` and verify all of the following:

1. The receipt self-hash is valid and its schema, study, mode, status, cohort
   completion flags, model counts, and total counts are exactly the formal
   values: 32 cells, 1,152 source requests, and 14,400 source actions.
2. Every `compiler_science_activity` counter is zero, every unsupported output
   remains explicitly absent, and `safe_for_confirmation_release` is false.
3. `input_manifest`, `planned_cells`, `compiler_source`, and
   `freeze_validator_dependency` are exact regular-file descriptors. Their
   hashes and byte counts must match the immutable staged files, and no path or
   ancestor may be a symlink. The required dependency path is
   `workshops/corl2026_world_models/analysis/freeze_development_release.py`.
4. Each model's timing inventory, request-provenance inventory, freeze-cell
   fragment, aggregate receipts, per-cell action manifest, and per-cell
   recording receipt matches the descriptor embedded in the receipt. The
   wrapper must parse each file and validate its expected signed schema and
   cross-file identities; merely checking that the files exist is insufficient.
5. The formal timing inventories use
   `wmf-development-timing-request-inventory-v1`; the two signed provenance
   inventories use `wmf-development-request-provenance-v1`; the freeze
   fragments use `wmf-development-freeze-cell-evidence-fragment-v1`; and all
   model/status/count fields agree with the compiler receipt. Diagnostic schemas
   or `diagnostic_partial_only` status are never publishable as formal output.
6. Every emitted request binds to its model/cell action-manifest and recording
   descriptor, and each roster row binds to those same hashes and the retained
   source-video identity. The source request IDs and recording IDs must remain
   globally unique.

The current compiler receipt schema is intentionally stable at
`wmf-development-evidence-compiler-receipt-v1`; adding wrapper validation does
not require changing compiler output fields.

## Annotation inventory bridge remains separate

The compiler's `wmf-development-request-provenance-v1` output is not accepted
directly by `forecast_annotation_workflow.py select-requests`, which requires a
complete `wmf-forecast-request-inventory-v1`. A separate fail-closed adapter
must authenticate and join the compiler provenance, the signed per-request
development timing bindings, and the signed frozen alignment/camera-crop
contracts. That adapter must create and bind per-request alignment receipts and
derive only the objective timing/camera/technical fields required by the
annotation inventory. It must not create labels, visibility judgments, a
movement threshold, or human-review receipts. Until that bridge and its inputs
exist, request selection and confirmation release remain blocked.
