# Camera-crop replay witness

This slice defines the fail-closed pixel replay needed to qualify the geometry
used to compare an original selected camera with a generated decoded future.
The current attempt is diagnostic-only and does not qualify that geometry. It
does not run a model or simulator, infer physical time, evaluate behavior or
prediction skill, create a label, or release confirmation.

Attempt `camera-crop-replay-witness-001` ended technical-invalid after both
children exited 1. Its signed failure receipt SHA-256 is
`b8ba6bb884f7c2daaf6695bafd7fed97675d8bc9bc11a125e4353eb52df42550`;
the outer wrapper stderr SHA-256 is
`a2bf9ff0707832e9d1b4968ad6e954cd23290f791b449c29cf72834b39e85922`.
Its raw child logs remain on the PVC but were not published, so attempt 001
does not establish the common cause and remains preserved.

The fresh queue job is diagnostic-only `camera-crop-replay-witness-002` on
`wmf-forecast-0912-worker-06`. The queue controller already starts the wrapper
as a detached process-group leader. Both replay children deliberately inherit
that wrapper group, so controller shutdown, timeout, or interruption reaches
the wrapper and both children. The wrapper rejects execution unless its PID is
also its process-group ID, persists a signed launch receipt immediately after
each successful spawn, and locally terminates and reaps every started child on
partial launch, receipt-write, timeout, or other orchestration failure.

Attempt 002 uses the inherited queue environment and the same CPU/offline
overrides; only `PYTHONUNBUFFERED=1` is added so a crash cannot strand buffered
diagnostics. Before heavyweight imports, each child emits one interpreter,
module-spec, and path-only `PYTHONPATH`/`PYTHONHOME`/`VIRTUAL_ENV` preflight.
On failure it repeats that preflight immediately before the terminal traceback
so it normally remains inside the bounded tail. It never emits argv or
arbitrary/secret environment values. An SSH disconnect does not define the
wrapper lifetime; the shared queue controller remains the durable owner.

Attempt 002 can publish only `camera_crop_witness_job_failure.json`. Within
that signed job receipt, each child row binds the exit code, full retained-log
path/byte count/SHA-256, and at most the final 8,192 raw bytes by offset and SHA-256. It
publishes no arbitrary log text. Instead it drops a partial leading line and
projects only allowlisted preflight fields, traceback frame identities, error
types, and a missing-module identifier; all messages and unapproved paths are
represented only by byte counts and SHA-256. Generic wrapper exceptions use
the same structural-only rule and omit traceback source text. Full raw logs
remain only at their signed PVC paths. The queue descriptor sets its outer
stdout/stderr publication-tail budget to zero, so a chained interpreter
traceback cannot bypass the signed structural projection. Even if both
diagnostic children unexpectedly complete, raw contracts remain only on the
PVC and no crop contract is published. A later fresh attempt may apply only a
cause evidenced by this diagnostic.

## Fail-closed inputs

The descriptor builder accepts exactly five explicit files fetched from the
results branch, each with its expected SHA-256:

- N3 and D1 source-audit receipts;
- the passed six-request N3 generation qualification;
- the D1 generation normalization receipt; and
- the qualified six-request D1 job receipt.

It emits a deterministic `wmf-camera-crop-witness-wave-v1` descriptor and does
not edit or dispatch the queue. The cluster wrapper reopens the corresponding
immutable PVC copies. It also checks its queue descriptor, serialized claim,
worker role, pod UID, clean staged source commit, and the exact hashes of the
wrapper, replay tool, runtime contract, existing queue support, N3 runtime
contract, and D1 identity contract.

All source, checkpoint, capture, nested manifest, tensor, NumPy, and active
transform implementation paths are absolute and path-bound. Evidence paths and
every path component are rejected if they are symbolic links or if lexical and
resolved paths differ. N3 rehashes its 43-file checkpoint and D1 rehashes its
25-file checkpoint. Both rehash the exact tracked source tree and every
retained artifact they consume. The active Python, NumPy, torch, Pillow,
openpi, and (for D1) torchvision implementation/version chain is signed into
the result. A failed check produces only a technical-invalid zero-science
diagnostic receipt; it never fabricates or publishes a crop contract.

## N3 replay

The original `over_shoulder_left_camera` RGB frame is `720x1280x3` uint8.
The replay executes the active `openpi_client.image_tools.resize_with_pad` to
`360x640`, then the exact RoboLab client `torch.nn.functional.interpolate`
bilinear conversion to `180x320`. The latter is placed at half-open coordinates
`y=[360,540), x=[0,320)` in the `540x640` wrist/left/right wire mosaic. The
original comparison crop is `y=[360,528), x=[0,320)`, or `168x320`.

The witness requires byte equality with the retained client intermediates and
wire request. It then requires the retained transformed model input to have:

- one `video` item shaped `[1,3,33,544,736]` uint8;
- one `image_size` item shaped `[1,4]` float32 with exact value
  `[544,736,540,640]`;
- the wire image at time zero, zeros at the other 32 input frames, and exact
  bottom/right reflection padding;
- spatial compression factor 16 and retained latent shape
  `[1,48,9,33,40]`; and
- decoded RGB shape `[33,528,640,3]`.

The signed generated crop is the half-open THWC slice
`[:,360:528,0:320,:]`, shape `[33,168,320,3]`. It is explicitly not a
whole-frame identity.

## D1 replay

The same raw selected camera goes through the exact RoboLab PIL bilinear
`resize_with_pad` to `180x320`, and must equal the retained wire key
`observation/exterior_image_0_left`. The pinned server maps this to
`video.exterior_image_1_left` with a one-frame time axis.

The active checkpoint eval chain is replayed without approximation:

1. uint8 THWC to float32 TCHW divided by 255;
2. torchvision v2 center crop of `(171,304)`, whose active rounding yields
   half-open raw-wire bounds `y=[4,175), x=[8,312)`;
3. torchvision v2 resize to `(176,320)` with
   `InterpolationMode.BILINEAR` and `antialias=True`;
4. the eval-mode color-jitter identity; and
5. TCHW to THWC, multiply by 255, then `torch.to(uint8)`.

The left, right, and wrist views must reconstruct the retained DreamZero DROID
mosaic byte-for-byte: wrist repeated across the top row, selected left at the
bottom-left, and right at the bottom-right. Its exact normalized tensor shape is
`[1,1,352,640,3]` uint8. The retained latent must be
`[1,16,3,44,80]`, and decoded RGB must be `[9,352,640,3]`. The generated
selected crop is `[:,176:352,0:320,:]`, shape `[9,176,320,3]`; the original
comparison image is the exact normalized selected-left view, `176x320`.

## Candidate contract and API

Attempt 002 publishes none of the candidate crop-contract files. Once a later
fresh qualification attempt passes the exact replay without weakening a check,
its permitted compact outputs are:

- `n3_camera_crop_contract.json`;
- `d1_camera_crop_contract.json`; and
- a fresh-attempt camera witness job receipt.

Each model document is signed `wmf-camera-crop-contract-v1`. The executable
fields include top-level `crop_operation`, `image_width_px`, and
`image_height_px`, plus `original_camera_replay.source_camera_shape`, its
ordered `transform_chain`, `output_shape`, and the complete
`generated_decoded_crop` object (`operation`, `canvas_shape`, half-open `y` and
`x`, and `output_shape`). Artifact and pixel SHA-256 values bind this one
witness. Dependency, retained-artifact, and all-true replay-check records bind
how it was obtained.

Downstream code should load
`analysis/camera_crop_replay_witness.py` by its signed source hash and call:

```python
validate_camera_crop_contract(contract, expected_model="N3")
original_crop = replay_original_camera_frame(original_rgb, contract)
generated_crop = extract_generated_crop(decoded_rgb, contract)
```

`replay_original_camera_frame` reopens and verifies the signed active transform
dependencies before executing them. `extract_generated_crop` requires the
exact uint8 THWC decoded canvas and applies the signed half-open slice. Neither
API accepts simulator-state renders.

Every candidate contract and the diagnostic receipt fixes all science and label
counts to zero,
with `simulator_state_render_used`, `whole_frame_identity`,
`safe_to_release_confirmation`, and `confirmation_released` false.
