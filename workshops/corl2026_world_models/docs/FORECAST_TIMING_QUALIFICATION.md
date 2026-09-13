# Forecast timing qualification

This path is source/probe authority only. It does not infer physical time from
presentation FPS, conditioning FPS, equal frame/action ordinals, or
DreamZero's action/block ratio. It does not execute any model-returned action.

Current scientific decision: **NO-GO for confirmation** until the real GM
artifacts below pass and the immutable development receipts are covered by the
hash-bound request timing sidecar. The source integration and synthetic
fail-closed tests are GO; that is not a completed physical timing
qualification.

## Frozen source lineage

The machine contract is
`experiments/forecast_layout/forecast_timing_lineage_contract.json`. The
validator also hard-codes the pins so changing the JSON alone cannot weaken the
gate.

N3 is pinned to commit
`411d25b2e35bc441126f48c44a4b93e1c0564274`, tree
`1e77852f2240d6e00312342ab29164745042b1ae`, and checkpoint revision
`6706d7680581c255ff61e0f3bb49d90eac55c79e`. The audited chain is:

1. `droid_lerobot_dataset.py` selects 33 timestamped observation rows and 32
   transition-action rows.
2. `transforms.py` and `sequence_packing.py` retain the ordered sequence and
   explicitly place action 0 with vision frame 1.
3. `unified_3dmrope_utils.py` carries temporal positions through VAE packing;
   its FPS input is representation metadata, not physical-time authority.
4. `inference/action.py` constructs 33 vision positions for the 32-action
   chunk. The official RoboLab server adds the conditioning state row at frame
   0, retains the generated latent, decodes 33 RGB frames, and strips exactly
   that row before returning 32 actions.

This supports candidate pairs `(decoded frame, executed control boundary)`
`(0,0)` through `(32,32)`. Their numerals happen to match, but that equality is
not the evidence; the explicit action-to-vision placement and conditioning-row
strip are.

D1 is pinned to commit
`ab790c198fbce33503358efbbd4187ce9a89adf3`, tree
`6b7ba27f1af81e963a6507f1204c05c65a94098c`, and checkpoint revision
`96ad344138c66e82536422432ad742f015784942`. The audited chain is:

1. `lerobot.py` retrieves video pixels using each selected dataset timestamp.
2. `lerobot_sharded.py` literally selects video row offsets
   `[0,3,6,9,12,15,18,21,24]` and action offsets `[0,...,23]`.
3. `dreamzero_cotrain.py` retains video order and pads only the action channel
   dimension.
4. `wan_video_vae.py` causally encodes nine frames into three ordered latents
   and decodes three ordered latents into nine frames.
5. `wan_flow_matching_action_tf.py` conditions on the first latent, jointly
   generates the next two visual latents and 24 actions, then returns the
   conditioning plus generated latents. The 24:8 block assertion is not used as
   timing evidence.

This supports candidate pairs
`[(0,0),(1,3),(2,6),(3,9),(4,12),(5,15),(6,18),(7,21),(8,24)]`.
With D1's unchanged executed prefix of eight actions, only positive frames 1
and 2 (boundaries 3 and 6) are timing-eligible.

Every cited source file and its SHA-256 is enumerated in the contract. The
`audit-source` command verifies the pinned commit/tree, the Git blob bytes, and
the working bytes.

## Evidence gates

`analysis/qualify_forecast_timing.py` emits immutable, self-hashed JSON and
revalidates transitive files rather than trusting a copied summary.

- `audit-source` emits `wmf-forecast-source-timing-lineage-v1`; it explicitly
  sets `physical_time_qualified: false`.
- `prepare-n3-live-input` copies the exact P00 N3 wire arrays into an immutable
  `wmf-n3-fixed-observation-v1` bundle and emits
  `wmf-n3-live-zero-policy-input-v1`. It makes zero model requests and executes
  zero actions.
- The existing N3 six-request qualification must be rerun on that live bundle.
  The existing `n3-first-live-002` artifact is historical and is deliberately
  rejected for timing even though it is valid generation-runtime evidence.
- `normalize-generation` re-opens the exact raw N3 or D1 qualification,
  returned-action artifact, retained latent, decoded tensor/RGB, input capture,
  source and checkpoint identities. It accepts six requests and zero robot or
  behavioral episodes only.
- `qualify` revalidates a passed 450-action/451-observation recorder-only trace.
  Each action must be the exact proposed joint hold, tied to the immediately
  preceding proprioceptive observation, with identical executed-action bytes.
  The full journal chain, original-camera NPZ payload, native control identity,
  physics step/time and camera frame/time are checked.
- Physical target seconds are elapsed original-camera capture time from
  boundary zero. Physics elapsed time must agree within
  `min(half minimum positive physics/control interval, half minimum positive
  camera interval)`.
- `bind-development` does not require or pretend that timing metadata existed
  inside an old request receipt. It accepts sidecar inventory entries that name
  the exact immutable request, cell completion and full adapter journal. For
  each request it re-derives the request-start observation and every eligible
  source-mapped boundary from native original-camera and physics clocks,
  checking them against the model authority. Truncated-prefix targets are
  explicitly marked ineligible. It emits
  `binding_mode: immutable_request_receipt_native_clock_sidecar`,
  `native_runtime_field: request_timing_sidecar.generated_targets`, and
  `old_request_receipts_modified: false`.

The pure consumer API is:

```python
validate_timing_authority(
    path: pathlib.Path,
    expected_sha256: str,
    *,
    expected_model: str | None = None,
) -> dict
```

It rejects symlinks, duplicate JSON keys, hash/byte drift, signed-summary drift,
missing raw artifacts, altered action identities, omitted/reordered mapping
rows, and re-signed target values that disagree with the raw clocks.

## Concrete GM inputs and commands

Run these from one immutable staged study checkout on the PVC. Use the RoboLab
Python because authority validation opens the retained NPZ camera payloads;
the cluster system Python is not assumed to contain NumPy.

```bash
STUDY=<queue-state-dir>/sources/<staged-study-commit>
PY=/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python
TOOL=$STUDY/workshops/corl2026_world_models/analysis/qualify_forecast_timing.py
CONTRACT=$STUDY/workshops/corl2026_world_models/experiments/forecast_layout/forecast_timing_lineage_contract.json
CONTRACT_SHA=<sha256-of-that-staged-contract>
OUT=/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/timing_qualification/<unique-attempt-id>
RECORDER=/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control/jobs/recorder-qualification-p00-003/publish/recorder_qualification_receipt.json
RECORDER_SHA=0e3f02f37a2548e36ae3a45a38a1fac63c56cd8798732056f24b03d103991fde
CAPTURE=/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/fixed_observations/fixed-observation-p00-001/capture/capture_receipt.json
CAPTURE_SHA=8d42bc36fe57747f5ce008c54f14e64f03bbe27be756b80c2b431d50aef7f663
CAMERA=<camera-id-from-the-frozen-model-blind-crop-contract>
```

First audit both source checkouts:

```bash
$PY "$TOOL" audit-source --model N3 \
  --source-root /data/users/ali/vla_wam/external/v3-clean/cosmos-nano-411d25b \
  --contract "$CONTRACT" --contract-sha256 "$CONTRACT_SHA" \
  --output "$OUT/n3_source_audit.json"
$PY "$TOOL" audit-source --model D1 \
  --source-root /data/users/ali/vla_wam/external/DreamZero-v3e004-clean-ab790c1 \
  --contract "$CONTRACT" --contract-sha256 "$CONTRACT_SHA" \
  --output "$OUT/d1_source_audit.json"
```

Prepare and run the missing live N3 generation-only probe. The model command is
the already qualified N3 generation protocol; its input is changed only from
the historical packed fixture to the hash-bound live P00 fixture. It performs
six requests and executes no returned action.

```bash
$PY "$TOOL" prepare-n3-live-input \
  --capture "$CAPTURE" --capture-sha256 "$CAPTURE_SHA" \
  --camera-id "$CAMERA" --output-dir "$OUT/n3_live_input"

COSMOS_PY=/data/users/ali/vla_wam/envs/cosmos-nano-411d25b-v3-exact/bin/python
CUDA_VISIBLE_DEVICES=0 DS_IGNORE_CUDA_DETECTION=1 \
HF_HOME=/data/users/ali/vla_wam/cache/huggingface-cosmos \
PYTHONPATH=/data/users/ali/vla_wam/external/v3-clean/cosmos-nano-411d25b:$STUDY \
$COSMOS_PY "$STUDY/workshops/corl2026_world_models/experiments/forecast_layout/n3_first_live.py" \
  --source-root /data/users/ali/vla_wam/external/v3-clean/cosmos-nano-411d25b \
  --checkpoint-root /data/users/ali/vla_wam/checkpoints/cosmos3_nano_policy_droid \
  --observation-manifest "$OUT/n3_live_input/observation_manifest.json" \
  --output-dir "$OUT/n3_generation_raw" \
  --publish-dir "$OUT/n3_generation_publish" \
  --effective-seed 2026091000
```

Normalize the new N3 receipt and the already-live D1 receipt. The D1 receipt
path and hash are fixed by `d1-first-live-005`:

```bash
$PY "$TOOL" normalize-generation --model N3 \
  --qualification "$OUT/n3_generation_publish/n3_qualification.json" \
  --qualification-sha256 <new-n3-publish-receipt-sha256> \
  --contract "$CONTRACT" --contract-sha256 "$CONTRACT_SHA" \
  --output "$OUT/n3_generation_probe.json"

$PY "$TOOL" normalize-generation --model D1 \
  --qualification /data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control/jobs/d1-first-live-005/publish/d1_qualification_job_receipt.json \
  --qualification-sha256 3c856549999b9145dc07c30853a4c6d2968d09eb31c2db883f5eb1655d31627b \
  --contract "$CONTRACT" --contract-sha256 "$CONTRACT_SHA" \
  --output "$OUT/d1_generation_probe.json"
```

Then qualify each model against the native recorder clocks (replace only the
fresh output hashes):

```bash
$PY "$TOOL" qualify --model N3 \
  --contract "$CONTRACT" --contract-sha256 "$CONTRACT_SHA" \
  --source-audit "$OUT/n3_source_audit.json" --source-audit-sha256 <n3-source-audit-sha256> \
  --generation-probe "$OUT/n3_generation_probe.json" --generation-probe-sha256 <n3-generation-probe-sha256> \
  --recorder-receipt "$RECORDER" --recorder-receipt-sha256 "$RECORDER_SHA" \
  --camera-id "$CAMERA" --output "$OUT/n3_timing_authority.json"

$PY "$TOOL" qualify --model D1 \
  --contract "$CONTRACT" --contract-sha256 "$CONTRACT_SHA" \
  --source-audit "$OUT/d1_source_audit.json" --source-audit-sha256 <d1-source-audit-sha256> \
  --generation-probe "$OUT/d1_generation_probe.json" --generation-probe-sha256 <d1-generation-probe-sha256> \
  --recorder-receipt "$RECORDER" --recorder-receipt-sha256 "$RECORDER_SHA" \
  --camera-id "$CAMERA" --output "$OUT/d1_timing_authority.json"
```

These commands must run as a detached, immutable queue job in normal cluster
operation; this document is an exact argv/input specification, not authority to
bypass the queue. Publish the compact contracts/receipts and retain referenced
raw tensors, NPZ files and journals on the PVC.

## Immutable development sidecar; no rerun required

A passed authority proves the decoded-frame/control-boundary candidate and its
physical seconds under native clocks. The completed N3 development receipts and
the active D1 runtime do not contain a `forecast_timing` field; they must not be
mutated and the tooling never claims that field was present. Reruns are not
scientifically necessary if their retained native journals pass the sidecar
gate.

Construct one `wmf-development-timing-request-inventory-v1` per model. Its
`request_receipts` list must follow the exact flattened order used by the
development evidence manifest and contain one object per request:

```json
{
  "cell_id": "exact immutable cell_id",
  "request_index": 0,
  "request_receipt": {"path": "...", "sha256": "...", "bytes": 0},
  "adapter_completion": {"path": "...", "sha256": "...", "bytes": 0},
  "adapter_journal": {
    "path": "...",
    "sha256": "...",
    "bytes": 0,
    "event_count": 0,
    "tail_sha256": "..."
  }
}
```

Every path, SHA-256, byte count, event count and journal tail is mandatory; the
zero values above are schema placeholders, not usable evidence. For each entry,
the validator also reopens the hash-chained `model_response_received` payload
inside `adapter_journal` and requires its transported
`wmf_server_request_receipt` descriptor to point to that exact request file.
This supplies D1's cell/episode binding even though the immutable D1 request
receipt itself has no cell ID, and prevents substitution from another attempt.
Run:

```bash
$PY "$TOOL" bind-development \
  --authority "$OUT/n3_timing_authority.json" \
  --authority-sha256 <n3-timing-authority-sha256> \
  --request-inventory "$OUT/n3_development_request_inventory.json" \
  --request-inventory-sha256 <n3-development-inventory-sha256> \
  --output "$OUT/n3_development_timing_sidecar.json"

$PY "$TOOL" validate-development \
  --timing "$OUT/n3_development_timing_sidecar.json" \
  --sha256 <n3-development-sidecar-sha256> --model N3
```

Repeat for D1. `freeze_development_release.py` now invokes the same deep
validator, requires exact equality with its own ordered request hash inventory,
and bypasses the old embedded-field lookup only when this sidecar mode passes.
A re-signed sidecar with changed targets, ordering, clock values or descriptors
fails against the immutable files. Confirmation remains held until both model
sidecars and all other development/human-annotation release gates pass.
