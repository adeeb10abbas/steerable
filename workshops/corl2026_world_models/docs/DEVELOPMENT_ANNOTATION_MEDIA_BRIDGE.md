# Development annotation media bridge

`prepare_development_annotation_media.py` is the fail-closed CPU bridge between
the formal development recorder/compiler evidence and the existing blinded
annotation workflow. It prepares real source pixels and provenance; it does not
create labels, certify pixel blindness, distribute a rater packet, or release
confirmation.

## Required immutable inputs

The input is one `wmf-development-annotation-media-input-v1` JSON object. Its
`raw_root` must be the absolute task PVC root
`/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912`. It names the passed
formal compiler job/receipt and, for each of `N3` and `D1`, descriptors for:

- complete request provenance;
- the passed missingness-aware timing sidecar and its queue receipt;
- a passed `wmf-camera-crop-contract-v1` witness;
- the frozen alignment contract; and
- the signed physical-alignment receipt.

Every descriptor is exactly `{path, sha256, bytes}`, is reopened beneath the
PVC root without symlinks, and must match its bytes and SHA-256. Internally the
bridge joins compiler and timing evidence only on
`(model_id, cell_id, request_index)`. It then reopens the recorder completion,
hash-chain journal, action manifest, original observation payload, official
request receipt, recorder response payload, and official generated-future
artifact. A mismatch is a technical failure, never a missing label.

The mapping self-hash is not treated as physical authority. The bridge reopens
the compiler freeze-cell fragments, native timing sidecars, original recording
clocks, and signed camera contracts and reruns the freeze implementation's
alignment derivation with resource release disabled. Both mappings (including
IDs, all clock extrema and residuals, and every source-lineage descriptor) and
both alignment contracts must equal that source replay byte for byte.

The full gate is fixed at 32 episodes and 1,152 request rows: N3 has 240 rows;
D1 has 912. Exactly 672 D1 rows remain
`forecast_timing_unavailable`, with null timing residual and no target binding.
The eligible population is 448 requests (224/model). The existing deterministic
selector draws 4 of 14 eligible requests per episode: 128 total, 64/model, with
inclusion probability `4/14 = 2/7`. Both the primary and early horizon must have
matched native camera/physics targets before a selected request is rendered.

## Exact pixel replay

Original-camera pixels come only from recorder observation artifacts. The
bridge calls `camera_crop_replay_witness.replay_original_camera_frame`; it does
not contain a resize approximation. One persistent isolated helper is started
per model, and `argv[0]` is the exact executable identity in that model's signed
`camera_crop_contract.runtime_dependencies.python`. Thus D1 replay runs in the
signed DreamZero environment even though the queue wrapper runs in RoboLab.
The parent and helper exchange only length-prefixed, signed headers and
hash-bound uint8 pixels; they bind ordinal, input, output, session and transcript
hashes, require a clean terminal handshake, and emit two signed runtime-session
receipts. A wrong interpreter, changed dependency, broken helper, truncated
body, or mismatched response fails closed. The helper executes and authenticates
the signed model-specific chain:

- N3: OpenPI bilinear resize-with-pad, torch bilinear downsample, mosaic
  placement, then the half-open 168x320 left-camera crop.
- D1: RoboLab bilinear resize-with-pad, the exact video tensor conversion,
  torchvision center crop and bilinear resize, evaluation-time identity color
  jitter, uint8 conversion, mosaic placement, then the half-open 176x320
  left-camera crop.

Generated frames are cropped only through
`camera_crop_replay_witness.extract_generated_crop`: N3 uses
`[33,528,640,3] -> y[360:528],x[0:320]`; D1 uses
`[9,352,640,3] -> y[176:352],x[0:320]`. The roles are `current`, `predicted`,
`executed`, `early_predicted`, `early_executed`, and `preceding` except at
request zero. No simulator-state render, video FPS, presentation-frame/action
equation, or generated-frame/action equation is permitted.

## Raw output and human gate

The atomic output directory contains:

- `request_inventory.json` and `request_selection.json`;
- `source_extraction_lineage.json`, binding each role to its parent
  file/member/frame/crop/hash and, for original-camera roles, the exact-runtime
  replay session/ordinal/input/output hashes;
- `camera_replay_runtime/{n3,d1}_session_receipt.json` and retained helper
  stderr logs;
- `rendered_png_manifest.json` and metadata-free PNGs in
  `annotation_media/`;
- canonical copies in `source_images/` and a signed receipt in
  `render_receipts/` for each render;
- `pixel_blindness_review_checklist.json`, an identity-blind list of unique
  asset hashes plus an explicitly unsigned human-receipt template;
- `image_inventory_pre_review.json`;
- `publish_tranche_index.json`; and
- `preparation_receipt.json`.

The checklist leaves reviewer, time, exclusion attestations, and receipt
signature null. It therefore cannot validate as a
`wmf-forecast-pixel-blindness-review-v1`. A named human must inspect every
unique PNG and its isolated presentation, attest that the required identity
and outcome cues are absent, and sign the exact asset population. Only then may
`finalize-reviewed-inventory` create the image inventory accepted by
`forecast_annotation_workflow.py package`.

The labeling rubric and illustrated examples are not invented or re-frozen by
this bridge. The packet builder must use the rubric/example hashes already
bound by the formal development freeze. Thus successful media preparation is
only `go_for_human_pixel_blindness_review_only`, not a rater-ready or
confirmation-release decision.

## Detached queue descriptors

Create the main descriptor without editing the active queue:

```bash
python workshops/corl2026_world_models/experiments/forecast_layout/development_annotation_media_jobs.py \
  emit-formal \
  --study-commit COMMIT \
  --inputs /absolute/path/annotation_media_inputs.json \
  --inputs-sha256 SHA256 \
  --output /absolute/path/development_annotation_media_wave.json
```

The queue runtime uses job ID `development-annotation-media-bridge-001`, role
`wmf-forecast-0912-worker-05`, a six-hour wall bound, and raw/PVC output. Its
only main-job published files are the compact job receipt and tranche index.
They are staged together and become visible through one directory rename; a
receipt-write failure cannot leave a success-looking tranche index beside a
failure receipt.

After those two files return on the results branch, generate deterministic
non-overlapping media-return jobs without PVC or Kubernetes access:

```bash
python workshops/corl2026_world_models/experiments/forecast_layout/development_annotation_media_jobs.py \
  emit-tranches \
  --study-commit COMMIT \
  --preparation-job-receipt RESULTS/development_annotation_media_job_receipt.json \
  --preparation-job-receipt-sha256 SHA256 \
  --tranche-index RESULTS/publish_tranche_index.json \
  --tranche-index-sha256 SHA256 \
  --output /absolute/path/development_annotation_media_tranche_wave.json
```

PNG assets are deduplicated globally by SHA-256 and hash-sorted. Each appears in
exactly one tranche. Every PNG is below the queue publisher's 16 MiB file cap;
each tranche allows at most 48 MiB of PNG bytes, preserving headroom below the
64 MiB job cap for its signed alias/provenance manifest and terminal receipt.
A tranche worker accepts only the exact passed main-job receipt at the canonical
PVC publication path and only the byte-identical raw tranche-index descriptor
named by that receipt. Independently signed or shape-valid replacement indexes,
nonzero science/label counts, and rater or confirmation authority fail closed.
A tranche stages and validates its entire publish tree before one atomic rename,
so failure cannot expose a partial successful asset population. The raw PVC
bundle remains canonical; returned media remain restricted pending human pixel
review and are not rater packets.

## Direct bridge commands

The queue normally invokes these, but the underlying operations are:

```bash
python workshops/corl2026_world_models/analysis/prepare_development_annotation_media.py \
  prepare --manifest INPUT.json --manifest-sha256 SHA256 --output-dir PVC_OUTPUT

python workshops/corl2026_world_models/analysis/prepare_development_annotation_media.py \
  finalize-reviewed-inventory \
  --preparation-dir PVC_OUTPUT \
  --review HUMAN_REVIEW.json --review-sha256 SHA256 \
  --output PVC_OUTPUT/image_inventory.json
```

Never fill or sign the human review automatically. Never run packet packaging
from `image_inventory_pre_review.json` or the checklist template.
