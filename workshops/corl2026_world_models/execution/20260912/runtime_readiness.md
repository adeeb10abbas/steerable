# Runtime readiness — WMF-ABLATION-001

Recorded 12 September 2026. This is a source inspection and preparation record,
not runtime qualification. No GPU query, model request, policy episode or live
simulator execution was performed by this preparation work.

The new scientific matrix is fully specified. A new fixed-duration runner,
synchronized camera logger, physical scene manifest, forecast/time mapping and
qualified D1 path are still required. The existing success-terminated recordings
do not waive the new four-condition recording pilot.

## Source identities and restoration

Steerable source citations below refer to Git object
`ce561e66f82e95055e39d3d7711691982f6b2086`. The local workshop starts from
`e67e6c4`; these are different identities with different roles. Sparse files can
be read with `git show ce561e66f82e95055e39d3d7711691982f6b2086:PATH` without
changing the working tree. New material belongs only to the workshop namespace.

Read-only workstation inspection found these external repositories:

| Repository | Existing checkout | Required historical object |
| --- | --- | --- |
| `/home/ali/projects/RoboLab` | `11142d4319e44401e0464866bb5fedf7ec8a8927` | `0aef241fb088ca21bb4ebd24448940ed56620d17` exists locally |
| `/home/ali/projects/dreamzero` | `93a305435d11b71809af498488bdda98ebf80ac1` | `ab790c198fbce33503358efbbd4187ce9a89adf3` exists locally |
| `/home/ali/cosmos-framework` | `1439c1d5e45a23771e9b1a2ad8f40a5981ea86c0` | Historical identity still needs an isolated restoration and validation |

DreamZero and Cosmos working trees contain unrelated changes. Do not clean them
or treat their current contents as the pinned runtime. The original workstation
steerable checkouts lacked `ce561e6`; the coordinating task subsequently restored
the study source into `/home/ali/projects/steerable-forecast-layout-20260912`.
That restoration alone does not qualify external source, weights or execution.

## Required fixed-duration change

The historical Nano task config registers success as an Isaac termination term
in `experiments/v3/cosmos_nano_phase_b/fixture_tasks.py:116–142`. Its bridge
forwards `env.step` unchanged at `robolab_bridge.py:530–545`, and accepts an
episode as complete after either success or 450 actions at `:563` and `:587`.
DreamZero's V3-B003 bridge uses the same success-or-cap completion rule in
`experiments/v3/dreamzero_phase_b/robolab_bridge.py:388–389`.

At pinned RoboLab commit `0aef241fb088ca21bb4ebd24448940ed56620d17`,
`robolab/core/environments/env.py:69–102` manages internal termination/freezing.
`robolab/eval/episode.py:168–179` skips video frames for frozen environments and
`:185–186` ends the loop when all environments have terminated. Simply masking a
returned `done` flag would not prevent that internal behavior.

Implement new workshop task definitions with success removed from termination
before environment creation. Preserve the relation-and-release predicates as
independent per-action measurements; store first success as an event and continue
for exactly 450 executed actions unless a separately identified safety abort or
technical failure occurs. Keep actual partial traces and censor them explicitly.
Do not reuse a historical compiler whose valid-completion rule permits early
success. Record the final two-action truncation of the last chunk explicitly
for both 32-action Nano and eight-action DreamZero execution prefixes.

## Camera and physical-time logging

Historical Nano state samples in
`experiments/v3/cosmos_nano_phase_b/robolab_bridge.py:258–271` contain the action
index, robot-frame cube/bowl positions and release status. They contain neither
physical timestamps nor camera frames. The step hook at `:530–545` is a useful
integration point for a new recorder, but is not sufficient as recorded.

`experiments/cosmos/v2_robolab_client.py:57–84` retains each returned 32×8 action
chunk and 33-frame decoded future. `experiments/v3/cosmos_nano_phase_b/live_client.py:72–105`
adds request index, action-start index and reset identity. Neither establishes
generated-frame time or retains the two original timestamped observations needed
for the history baseline.

At pinned RoboLab `0aef241f`:

- `robolab/registrations/droid/auto_env_registrations_jointpos.py:113–115`
  sets physics `dt=1/120`, `decimation=8` and `render_interval=8`. These imply
  nominal 15 Hz control/render intervals in that registration, not verified
  generated-video alignment. Record actual runtime values and capture timing.
- Reset and step return original RGB tensors under
  `obs['image_obs']['over_shoulder_left_camera']`,
  `over_shoulder_right_camera` and `wrist_cam`, indexed by environment.
- `policies/cosmos3/client.py:69–106` pads/resizes camera images, downsamples
  exterior views and builds a mosaic with wrist above left/right views. Retain
  the original images and exact transformations used for model input/scoring.
- `policies/dreamzero/client.py:142–188` extracts and packs the three views.
  The frozen client overlay in steerable's
  `experiments/dreamzero_droid/v2_robolab_client.py:52–85` requires padded
  180×320 uint8 images and the right exterior camera in the second slot.
- `robolab/eval/episode.py:86–89` performs two pre-action resets and derives
  presentation video FPS from configuration. The Nano bridge already makes the
  duplicate reset idempotent after settling (`robolab_bridge.py:406–528`).

A workshop recorder should retain original observations at reset and every
control step, corresponding state measurements, physics/control counters,
measured sensor/capture timestamps, image hashes and camera identities. Request
records must reference the exact current and preceding observations, returned
action/latent identities and the actual executed prefix. Store physical-time
mapping, residual and eligibility explicitly. Video presentation FPS is not an
alignment certificate. No qualified horizon or tolerance has been invented here.

## N3 restoration and qualification

The existing source at
`experiments/v3/cosmos_nano_phase_b/live_support.py:75–122` validates the required
g3, four-step, shift-5, history-one, state-input, 15-conditioning-FPS,
resolution-480 and 32×8 contract. It also hardcodes historical path, port and
release identities. Rebind a new isolated namespace without editing that
validator or old registries. The external server source is
`/home/ali/cosmos-framework/cosmos_framework/scripts/action_policy_server_robolab.py`.
`experiments/v3/cosmos_nano_phase_b/serve_nano.py:59–108` shows the historical
request-bound seed override and authorization wrapper.

The six fixed-input decode/repeat requests remain required, including a verified
no-decode path that skips only rendering if the official interface supports it.
Neither 15 conditioning FPS nor a 33-frame array determines action-time mapping.
The bounded candidate-seed audit below does not demonstrate runtime acceptance.

## D1 official path and decoding

The V3 adapter is explicitly custom s2:
`experiments/v3/dreamzero_droid/adapter.py:27–79` binds
`V2-A015:dreamzero_action_cfg_s2`, guidance scale 2 and its patch identities.
The V3-B003 bridge repeats this identity at `robolab_bridge.py:359`. These
adapters cannot silently fill D1 cells.

An official-path starting point exists in
`experiments/dreamzero_droid/v2_instrumented_server.py:72–132`: it wraps official
inference without changing returned actions and saves the exact `video_pred`
latent beside the 24×8 action tensor. Its input retention is hashes only
(`:120–129`). Its reset hook (`:134–167`) calls the official reset decode, records
the resulting MP4 and clears its measurement records. The wrapper imports
`socket_test_optimized_AR` from the external DreamZero repository (`:27`).

Restore the official historical commit in an isolated checkout and add the new
namespace, raw observations, qualified reset evidence and measurement-only
decoding support. The current external checkout's
`socket_test_optimized_AR.py:309–360` contains accumulated-latent VAE decoding
and reset handling, but these current-checkout line numbers are only a location
pointer; they do not establish equivalence to the historical source. Inspect and
pin the restored implementation before qualification.

The official client reset endpoint is shown at pinned RoboLab
`policies/dreamzero/client.py:225–243`; invocation alone does not prove every
temporal buffer/cache was cleared. Use one isolated server context per serial
four-condition job and verify full reset before every episode. Historical
`experiments/v3/dreamzero_droid/future_retention.py:4–10` documents a global-state
multi-client retention problem; partitioned archives are not an isolation waiver.
Per-request latents and concatenated reset videos require a validated decoding
and physical-time mapping. If a patched s1 path is used, the three official
reference requests and action/latent equivalence checks remain mandatory.

## Prepared schedule and bounded seed audit

[parallel_schedule.json](parallel_schedule.json) contains 58 indivisible
four-condition jobs: 2 pilot, 8 development and 48 confirmation, covering all
232 unique core cells. D2 is excluded. Every job remains `NOT_RELEASED` and
`released: false`; no worker allocation or runtime command is present.

The namespace is `wmf_ablation_001_20260912`. Confirmation blocks are sorted by
SHA-256 of ASCII `namespace + ':' + block_id` and assigned the 24 lexicographic
condition permutations once each. N3 and D1 share each block's order. Pilot and
development orders use the complete digest modulo 24. The assignment consumes
only frozen inventory metadata and no outcomes. Dependencies require pilot
qualification before development and completion of all planned development jobs
plus a separate freeze before confirmation; any reduced study requires an
explicit subsequent design decision.

[nano_seed_audit.json](nano_seed_audit.json) records all 29 proposed Nano seeds,
the pinned artifact tree identity and an inventory hash for 1,848 JSON, JSONL and
CSV artifact files. No exact candidate seed value occurred in that bounded
corpus. This does not establish absence from raw external files, source code,
prose, other Git references or another study, and does not establish effective
RNG behavior. The script preserves input SHA-256 identities and refuses to
overwrite different prepared schedule evidence.

Remaining release blockers include restored source/weight/config identities,
model-blind physical layouts and rejected candidates, the new recorder and
fixed-duration pilot, qualified frame/action/camera correspondence, independent
annotation procedures, measured resource costs and an available authorized
execution lane. No old experiment is replaced or rerun by this preparation.
