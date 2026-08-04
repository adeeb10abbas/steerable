# VLA/WAM study continuation handoff

Updated: 4 August 2026, after all original bounded gates and both Cosmos3 base
interface probes completed, and after both six-cell V2-A015 behavioral arms
completed.

This document is the restart point for a human or coding model with no chat
context. The machine-readable companion is
[`continuation_state.json`](../artifacts/vla_wam_shared_v2/continuation_state.json).

## One-minute orientation

The study asks whether changing an episode-static language command changes the
requested physical outcome. It compares checkpoints inside two separate arenas:
DROID/RoboLab and RoboTwin. Raw success rates are never pooled across arenas.

Current direct-command evidence is:

| Arena | Checkpoint | LEFT | RIGHT | Frozen next step |
| --- | --- | ---: | ---: | --- |
| DROID | π0-FAST | 1/10 | 10/10 | historical wording blocked; V2-A008 release gate failed with zero behavioral cells |
| DROID | GR00T N1.7 | 0/3 | 0/3 | six-cell gate complete; no wording expansion |
| DROID | Cosmos3 Edge | 3/3 | 3/3 | six-cell gate complete; decoded futures retained |
| DROID | DreamZero | 2/3 | 1/3 | baseline complete/do not rerun; V2-A015 `s=2` complete separately at LEFT 1/3 and RIGHT 3/3 |
| DROID | π0.5 current-stack V2-A010 | 1/3 | 3/3 | six-cell gate complete; selected actual-rollout pair published separately |
| DROID | Cosmos3 Nano Policy DROID V2-A011 | 3/3 | 3/3 | baseline complete/do not rerun; V2-A015 `g=1` complete separately at LEFT 1/3 and RIGHT 3/3 |
| DROID | Cosmos3 Super base V2-A012/A014 | — | — | image-only action+video probe passed; zero behavioral cells released |
| DROID | Cosmos3 Edge base V2-A013 | — | — | three-request interface probe passed; behavior blocked by exact mapping audit |
| RoboTwin | Efficient-WAM-RT | 3/7 | 2/7 | pairs03–09 complete; do not rerun |
| RoboTwin | FastWAM | 1/7 | 1/7 | pairs03–09 complete; do not rerun |
| RoboTwin | LingBot-VA | 3/7 | 4/7 | pairs03–09 complete; do not rerun |
| RoboTwin | LingBot-VLA 4B | 1/3 | 0/3 | bounded six-cell gate complete |
| RoboTwin | Light-WAM | 1/3 | 0/3 | bounded six-cell gate complete |

All original bounded gates are complete. The sixty-cell historical π0-FAST
wording expansion remains blocked on missing exact OpenPI and RoboLab commits.
User-directed amendment `V2-A008` separately authorized the same prompt/seed
cells as a current-stack replication at the exact available revisions; it must
never be merged with or represented as the historical queue. Its fixed-
observation prompt-sensitivity gate failed: repeated LEFT was bit-identical,
but LEFT and RIGHT returned the same action tensor (RMS 0.0). It therefore has
zero behavioral episodes and is not runnable under the frozen protocol. User
amendment `V2-A009` withdrew LaWAM before any model request or behavioral
episode; it has zero remaining cells and is absent from the shareable gallery.
Amendment `V2-A010` is complete as a separate six-cell π0.5 current-stack gate:
LEFT 1/3, RIGHT 3/3, three aligned endpoint pairs, and three distinct executed-
action pairs. It is not recovered historical π0.5 footage and is never merged
with v1 evidence. `V2-A011` is complete as a separate six-cell Cosmos3 Nano
Policy DROID gate: all six cells succeeded, all three endpoint pairs were
aligned, all paired executed actions differed, and 37 exposed decoded futures
were retained. Its two pre-request setup attempts remain outside behavioral
denominators. Cosmos-Reason2 remains a completed static diagnostic, not an
action policy or robot rollout.

The Cosmos3 base-checkpoint arms remain nonbehavioral. Edge base V2-A013 passed
its exact three-request fixed-observation interface probe: repeat LEFT was
bit-identical, while RIGHT changed both the 10D action and generated video.
Its CuRobo branch is blocked because the exact RoboLab USD and pinned parser
disagree by 0.2644 m and about 90 degrees at the control frame, with no verified
mimic-joint or collision parity. No action reached the simulator. Super base
V2-A012/V2-A014 separately passed the image-only deterministic and
prompt-sensitivity diagnostic. Both arms keep six conditional cells
unreleased and remain distinct from the completed Edge-Policy-DROID result.

After compact evidence was copied and PVC persistence was verified, the three
task-created Cosmos3-Super A100/B200 pods were deleted. Their checkpoints,
environments, server logs, and raw outputs remain under the ali-owned PVC; no
pre-existing ali pod or process was changed.

All 42 prospective WAM episodes at pairs03–09 are now valid completed evidence
and must not be rerun. Efficient-WAM-RT produced 5/14 requested-direction
successes and 6/7 aligned endpoint pairs; FastWAM produced 2/14 and 3/7;
LingBot-VA produced 7/14 and 6/7. Every paired executed-action trace differed.
Infrastructure-invalid attempts remain outside model denominators: four for
Efficient-WAM-RT, eighteen for FastWAM, and five for LingBot-VA. The frozen
twenty-episode compilers still fail closed because historical pairs00–02 raw
files are absent on this PVC; the committed pairs03–09 slices are the current
claim boundary. π0-FAST has completed its separate DROID direct-only
confirmation. Post-result amendment `V2-A005` authorized the bounded GR00T and
LingBot-VLA gates, Cosmos behavioral replication, the non-behavioral Reason2
diagnostic, and the three remaining π0-FAST wording families. Every bounded
gate and diagnostic in that list is complete; only the wording expansion is
blocked on exact historical adapter provenance. This decision was made after
the completed three-WAM outcomes were known and is not presented as
preregistered.

User-directed post-result amendment `V2-A007` added DreamZero-DROID as a bounded
six-cell direct-command gate before DreamZero study inference began. That gate
is complete, including its simulator videos and official imagination archive.
Its behavioral simulator ran on the ali-owned RTX PRO 6000 lane and its policy
server ran on separate rechecked B200 GPUs. The unrelated pre-existing
DreamZero process on B200 GPU 0 and port 5000 was never used or modified.

The current article, figures, and videos are:

- [`VLA_VS_WAM_STEERABILITY_STUDY.md`](VLA_VS_WAM_STEERABILITY_STUDY.md)
- [`VLA_WAM_STEERABILITY_VIDEO_GALLERY.html`](VLA_WAM_STEERABILITY_VIDEO_GALLERY.html)
- [`figures_manifest.json`](../artifacts/vla_wam_shared_v2/figures/figures_manifest.json)
- [π0-FAST media manifest](../artifacts/vla_wam_shared_v2/media/droid_pi0_fast_pairs/media_index.json)
- [RoboTwin media manifest](../artifacts/vla_wam_shared_v2/media/robotwin_wam_pairs/media_index.json)

For a work-laptop Kubernetes/B200 continuation with no chat context, start at
[`WORK_LAPTOP_B200_HANDOFF.md`](WORK_LAPTOP_B200_HANDOFF.md). It includes the
portable external-repository bundle procedure and cluster/PVC evidence rules.

## Start-of-session checklist

Run these before changing code or launching a model:

```bash
cd /home/ali/projects/steerable
git status --short
git log -3 --oneline
nvidia-smi
df -h /home/ali/projects/steerable
ss -ltnp | rg ':8000|:5000' || true
python3 tools/validate_vla_wam_v2_protocol.py
```

Expected evidence baseline: validator status `valid`; use the check count in the
committed validation report. The live host
snapshot at handoff had two idle 24 GiB RTX 3090s and 464 GiB free. Treat that
as historical context and recheck it; it is not a guarantee.

Preserve unrelated dirt in external repositories. At handoff, the known items
were `/home/ali/projects/RoboLab/.cache/` and
`/home/ali/projects/Efficient-WAM/sapien_offscreen.png`.

<a id="experiment-1-pi0-fast-droid-directional-confirmation"></a>
## Experiment 1 — π0-FAST DROID directional confirmation

Priority: complete. Cost: 14 new episodes. Status: compiled; do not rerun.

Frozen registry:
[`pi0_fast_directional_expansion.json`](../artifacts/vla_wam_shared_v2/pilot/pi0_fast_directional_expansion.json).
Seeds 8300–8302 are the preserved pilot and 8303–8309 have now completed.
Do not rerun any seed unless hash validation shows corruption.

### Completion record

The compiled confirmation is
[`pi0_fast_direct_confirmation.json`](../artifacts/vla_wam_shared_v2/pilot/results/pi0_fast_direct_confirmation.json)
(74,549 bytes; SHA-256
`491c74812ed0e4d36c16f8e0ded17a70af3e69740c9bcb87af129bb6d9563073`),
with [CSV](../artifacts/vla_wam_shared_v2/pilot/results/pi0_fast_direct_confirmation.csv)
and [Markdown](../artifacts/vla_wam_shared_v2/pilot/results/pi0_fast_direct_confirmation.md)
companions. LEFT released requested placement is 1/10 and RIGHT is 10/10;
all 20 behavioral episodes are valid, all ten paired endpoint shifts align,
and one seed-8305 LEFT wall-latency measurement is excluded without changing
its behavioral failure. The hash-bearing intervention ledger is
[`pi0_fast_runtime_interventions.json`](../artifacts/vla_wam_shared_v2/pilot/directional_confirmation/pi0_fast_runtime_interventions.json).
The GPU-0 policy server, GPU-1 simulator container, and ports 8000/5000 were
released cleanly.

The historical launch recipe below is retained for provenance only.

### Policy server

Use physical GPU 0:

```bash
cd /home/ali/openpi-robolab
FSSPEC_GS_TOKEN=anon CUDA_VISIBLE_DEVICES=0 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.50 \
  .venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint --policy.config=pi0_fast_droid_jointpos \
  --policy.dir=/home/ali/.cache/openpi/openpi-assets-simeval/pi0_fast_droid_jointpos
```

Wait until port 8000 is listening before starting Isaac.

### Simulator cells

Use the RoboLab container recipe in
[`VLA_WAM_LOCAL_RUNBOOK.md`](VLA_WAM_LOCAL_RUNBOOK.md), exposing physical GPU 1
to the container. Run one process invocation per seed. The command inside the
container is:

```bash
for v2_seed in 8303 8304 8305 8306 8307 8308 8309; do
  /workspace/isaaclab/_isaac_sim/python.sh policies/pi0_family/run.py \
    --policy pi0_fast \
    --task RubiksCubeLeftOfBowlMatchedTask RubiksCubeRightOfBowlMatchedTask \
    --num-envs 1 --num-runs 1 --headless --device cuda:0 \
    --video-mode viewport --disable-subtask \
    --instruction-controller static --instruction-type default \
    --open-loop-horizon 10 \
    --environment-seed "${v2_seed}" --sampling-seed-base "${v2_seed}" \
    --output-folder-name "v2_pi0_fast_direct_seed${v2_seed}"
done
```

Start the Docker thermal guard immediately after each named simulator container
appears. Never run two Isaac cells concurrently on the same 3090.

### Completion gate

Extend `tools/compile_vla_wam_v2_droid_pilot.py` to compile all ten seeds while
preserving the original six-episode result. Produce a new confirmation artifact;
do not overwrite history and call it a pilot. Report:

- LEFT and RIGHT successes out of 10 with Wilson 95% intervals;
- pickup, requested-region entry, and release counts by direction;
- ten paired endpoint shifts and first-action-chunk RMS values;
- invalid attempts and thermal interventions separately;
- whether competence is now present in both directions.

Only if both directions show at least one valid direct-command success does the
wording grid become eligible. Record that decision explicitly before running a
wording cell.

<a id="experiment-2-three-wam-robotwin-directional-confirmation"></a>
## Experiment 2 — three-WAM RoboTwin directional confirmation

Priority: complete. Remaining cost: 0 episodes. Status: all pairs03–09 model
slices compiled; full twenty-episode compilers blocked on missing historical
pairs00–02 raw evidence.

Frozen registry:
[`directional_expansion.json`](../artifacts/vla_wam_shared_v2/pilot/directional_expansion.json).
Model-blind fixture validation:
[`directional_fixture_validation.json`](../artifacts/vla_wam_shared_v2/pilot/directional_fixture_validation.json).

Run pair04–pair09 for Efficient-WAM-RT and pair03–pair09 for FastWAM and
LingBot-VA. Each
anchor scene emits both the exact LEFT and RIGHT direct-command condition. Run
one scene per invocation to avoid accidental task/seed Cartesian products and
to keep thermal recovery manageable.

Scene mapping:

| Pair | Anchor task | Environment seed | Sampling seed |
| --- | --- | ---: | ---: |
| 03 | `place_a2b_right` | 4300003 | 8403 |
| 04 | `place_a2b_left` | 4300004 | 8404 |
| 05 | `place_a2b_right` | 4300005 | 8405 |
| 06 | `place_a2b_left` | 4300006 | 8406 |
| 07 | `place_a2b_right` | 4300007 | 8407 |
| 08 | `place_a2b_left` | 4300008 | 8408 |
| 09 | `place_a2b_right` | 4300009 | 8409 |

Use the native process-group thermal launcher for every pair. It starts the
checked-in model wrapper in a private session, records its PID/PGID and
temperatures, pauses only that group at 87 C, resumes at 80 C, and leaves it
held (not killed) at 90 C. Each pair-level process emits two cell-level ledger
records when a pause occurs; emergency or partial attempts go to the separate
invalid-attempt ledger.

### Efficient-WAM pair03 completion record

`robotwin_pair_03` completed locally with two valid behavioral failures and no
thermal intervention. Both cells ran 400 actions and retained viewport video,
executed-action traces, and a five-frame decoded future. The exact compact
record is
[`efficient_wam_rt_pair03_integration.json`](../artifacts/vla_wam_shared_v2/pilot/directional_confirmation/efficient_wam_rt_pair03_integration.json);
the hash-pinned thermal stream is
[`efficient_pair03.jsonl`](../artifacts/vla_wam_shared_v2/pilot/directional_confirmation/thermal/efficient_pair03.jsonl).
Raw model outputs remain outside ordinary Git at the path recorded in the
integration manifest. Do not rerun this pair or reinterpret its two valid
failures as infrastructure failures.

### Efficient-WAM work-laptop completion record

Pairs04–09 completed on the ali-owned B200 PVC on 2026-08-03. All twelve
episodes exited cleanly, retained simulator video, decoded future video,
trajectory, and executed-action traces, and are valid behavioral evidence.
The compact hash-bearing slice is
[`efficient_wam_rt_pairs04_09_slice.json`](../artifacts/vla_wam_shared_v2/pilot/directional_confirmation/efficient_wam_rt_pairs04_09_slice.json),
with its enclosing file manifest in
[`efficient_wam_rt_pairs04_09_evidence_manifest.json`](../artifacts/vla_wam_shared_v2/pilot/directional_confirmation/efficient_wam_rt_pairs04_09_evidence_manifest.json).
The selected compact publication clip is the matched pair05 LEFT/RIGHT
both-success execution in
[`robotwin_wam_confirmation`](../artifacts/vla_wam_shared_v2/media/robotwin_wam_confirmation/media_index.json).

| Pair | LEFT | RIGHT | Endpoint ordering | First-10 action RMS |
| --- | --- | --- | --- | ---: |
| 04 | success | failure | aligned | 0.001607 |
| 05 | success | success | aligned | 0.002990 |
| 06 | failure | failure | aligned | 0.003832 |
| 07 | failure | success | aligned | 0.003067 |
| 08 | success | failure | aligned | 0.002129 |
| 09 | failure | failure | aligned | 0.060466 |

The prospective B200 slice therefore has 5/12 requested-direction successes
(LEFT 3/6; RIGHT 2/6), six of six aligned endpoint pairs, and paired executed
actions that differ in every pair. Pair03 remains the separately committed
0/2, anti-aligned do-not-rerun result. Four pre-action infrastructure attempts
from pair04 and pair05 are retained separately in
[`invalid_attempts_efficient_wam_rt_robotwin.json`](../artifacts/vla_wam_shared_v2/pilot/directional_confirmation/invalid_attempts_efficient_wam_rt_robotwin.json)
and do not enter behavioral denominators. No valid run incurred a thermal
intervention.

The frozen twenty-episode compiler still fails closed on this work laptop. The
PVC has raw pairs04–09, but completed-pilot pairs00–02 and prospective pair03
are available here only through committed hash-bearing summaries; the compiler
requires their original `result.json`, trajectories, videos, and pair03 action
traces. Do not synthesize those files or weaken the compiler. Restore the raw
directories recorded by the existing artifacts, stage one input root with
pairs00–09, and then run the compiler command below. Until then the new slice
is evidence, not a model-level ten-scene confirmation claim.

### FastWAM work-laptop completion record

FastWAM pairs03–09 completed as fourteen valid behavioral episodes. The
hash-bearing slice is
[`fastwam_pairs03_09_slice.json`](../artifacts/vla_wam_shared_v2/pilot/directional_confirmation/fastwam_pairs03_09_slice.json),
with its enclosing
[`evidence manifest`](../artifacts/vla_wam_shared_v2/pilot/directional_confirmation/fastwam_pairs03_09_evidence_manifest.json).
FastWAM produced 2/14 requested-direction successes (LEFT 1/7; RIGHT 1/7),
3/7 aligned endpoint pairs, and 7/7 distinct paired executed-action traces.
Its released interface is action-only, so imagined-future evidence is marked
not applicable rather than zero. Eighteen pre-action infrastructure-invalid
cell attempts are retained separately, and no runtime intervention occurred.

| Pair | LEFT | RIGHT | Endpoint ordering | First-10 action RMS |
| --- | --- | --- | --- | ---: |
| 03 | failure | failure | anti-aligned | 0.010866 |
| 04 | failure | failure | anti-aligned | 0.002838 |
| 05 | failure | success | aligned | 0.003292 |
| 06 | success | failure | aligned | 0.002798 |
| 07 | failure | failure | anti-aligned | 0.003402 |
| 08 | failure | failure | aligned | 0.006010 |
| 09 | failure | failure | anti-aligned | 0.003468 |

### LingBot-VA work-laptop completion record

LingBot-VA pairs03–09 also completed as fourteen valid behavioral episodes.
The hash-bearing slice is
[`lingbot_va_pairs03_09_slice.json`](../artifacts/vla_wam_shared_v2/pilot/directional_confirmation/lingbot_va_pairs03_09_slice.json),
with its enclosing
[`evidence manifest`](../artifacts/vla_wam_shared_v2/pilot/directional_confirmation/lingbot_va_pairs03_09_evidence_manifest.json).
LingBot-VA produced 7/14 requested-direction successes (LEFT 3/7; RIGHT 4/7),
6/7 aligned endpoint pairs, and 7/7 distinct paired executed-action traces.
All fourteen exposed first-predicted latent tensors are retained as latent-only,
not decoded or scored as videos. Five infrastructure-invalid cell attempts are
retained separately, and no runtime intervention occurred.

| Pair | LEFT | RIGHT | Endpoint ordering | First-10 action RMS |
| --- | --- | --- | --- | ---: |
| 03 | failure | success | aligned | 0.002489 |
| 04 | success | failure | aligned | 0.000599 |
| 05 | failure | success | aligned | 0.000455 |
| 06 | success | failure | aligned | 0.002692 |
| 07 | failure | success | aligned | 0.000458 |
| 08 | success | failure | aligned | 0.000549 |
| 09 | failure | success | anti-aligned | 0.000508 |

Both models' frozen twenty-episode compilers fail closed because raw pilot
pairs00–02 are absent from this PVC. Do not synthesize those files, weaken the
compiler, or promote these slices to ten-scene confirmation claims. There is
no next authorized WAM cell. The commands below are retained for provenance
only; do not rerun any Efficient-WAM-RT, FastWAM, or LingBot-VA pair03–09 cell.

```bash
# Efficient-WAM-RT
cd /home/ali/projects/steerable
EFFICIENT_WAM_GPU=1 python3 tools/native_process_group_thermal_guard.py --launch \
  --gpu-index 1 \
  --output artifacts/vla_wam_shared_v2/pilot/directional_confirmation/thermal/efficient_pair03.jsonl \
  --ledger-output artifacts/vla_wam_shared_v2/pilot/directional_confirmation/runtime_interventions_efficient_wam_rt_robotwin.json \
  --invalid-attempts-output artifacts/vla_wam_shared_v2/pilot/directional_confirmation/invalid_attempts_efficient_wam_rt_robotwin.json \
  --model-id efficient_wam_rt_robotwin --pair-id robotwin_pair_03 \
  --environment-seed 4300003 --sampling-seed 8403 \
  --requested-relation left --requested-relation right -- \
  /home/ali/projects/Efficient-WAM/experiments/robotwin_language_gate/run_gate_3090.sh \
  --output-dir /home/ali/projects/Efficient-WAM/outputs/vla_wam_shared_v2/directional_confirmation/pair03 \
  --task place_a2b_right --seed 4300003 --sampling-seed 8403 \
  --prompt-family direct_command --max-actions 400 \
  --save-simulator-video --predicted-video-max-chunks 1

# FastWAM
cd /home/ali/projects/steerable
FASTWAM_GPU=0 python3 tools/native_process_group_thermal_guard.py --launch \
  --gpu-index 0 \
  --output artifacts/vla_wam_shared_v2/pilot/directional_confirmation/thermal/fastwam_pair03.jsonl \
  --ledger-output artifacts/vla_wam_shared_v2/pilot/directional_confirmation/runtime_interventions_fastwam_robotwin.json \
  --invalid-attempts-output artifacts/vla_wam_shared_v2/pilot/directional_confirmation/invalid_attempts_fastwam_robotwin.json \
  --model-id fastwam_robotwin --pair-id robotwin_pair_03 \
  --environment-seed 4300003 --sampling-seed 8403 \
  --requested-relation left --requested-relation right -- \
  /home/ali/projects/FastWAM/experiments/robotwin_language_gate/run_gate_3090.sh \
  --output-dir /home/ali/projects/FastWAM/outputs/vla_wam_shared_v2/directional_confirmation/pair03 \
  --cell place_a2b_right:4300003:8403 \
  --prompt-family direct_command --max-actions 400 \
  --action-horizon 32 --replan-steps 24 --num-inference-steps 10 \
  --text-cfg-scale 2.0 --save-simulator-video --resume

# LingBot-VA
cd /home/ali/projects/steerable
LINGBOT_GPU=0 VLA_WAM_V2_STUDY_ROOT=/home/ali/projects/steerable \
python3 tools/native_process_group_thermal_guard.py --launch --gpu-index 0 \
  --output artifacts/vla_wam_shared_v2/pilot/directional_confirmation/thermal/lingbot_pair03.jsonl \
  --ledger-output artifacts/vla_wam_shared_v2/pilot/directional_confirmation/runtime_interventions_lingbot_va_robotwin.json \
  --invalid-attempts-output artifacts/vla_wam_shared_v2/pilot/directional_confirmation/invalid_attempts_lingbot_va_robotwin.json \
  --model-id lingbot_va_robotwin --pair-id robotwin_pair_03 \
  --environment-seed 4300003 --sampling-seed 8403 \
  --requested-relation left --requested-relation right -- \
  /home/ali/projects/lerobot-lingbot/experiments/lingbot_language_gate/run_gate_3090.sh \
  --output-dir /home/ali/projects/lerobot-lingbot/outputs/vla_wam_shared_v2/directional_confirmation/robotwin_pair_03 \
  --task place_a2b_right --environment-seed 4300003 --sampling-seed 8403 \
  --prompt-family direct_command --condition correct --condition swapped \
  --max-actions 400 --guidance-scale 5.0 --action-guidance-scale 1.0 \
  --save-simulator-video --save-first-predicted-latent
```

Use the LingBot directional wrapper's `--dry-run` output to verify pair04–09,
then launch each printed scene command separately through the same guard form;
do not run the all-pairs `--run` mode outside the guard. Do not pass all seven
tasks or seeds directly to a runner: repeated CLI values form an unsafe
Cartesian product. Do not change model guidance, horizon, diffusion steps,
task config, or future-retention settings. A pause preserves behavior but the
generated ledger excludes both affected cells from wall-latency aggregates. An
emergency-held or otherwise partial pair remains an infrastructure attempt,
not a model failure; preserve its outputs and review before an exact-cell rerun.

For pair04–pair09, copy the corresponding model's pair03 command and replace
all of the following together from the scene-mapping table: `pair03` or
`robotwin_pair_03` in the raw-log/output paths, `--pair-id`, anchor task,
environment seed, and sampling seed. Keep both repeated `--requested-relation`
flags. Keep that model's ledger filenames unchanged across its seven pairs:

| Model | Runtime-intervention ledger | Invalid-attempt ledger |
| --- | --- | --- |
| Efficient-WAM-RT | `runtime_interventions_efficient_wam_rt_robotwin.json` | `invalid_attempts_efficient_wam_rt_robotwin.json` |
| FastWAM | `runtime_interventions_fastwam_robotwin.json` | `invalid_attempts_fastwam_robotwin.json` |
| LingBot-VA | `runtime_interventions_lingbot_va_robotwin.json` | `invalid_attempts_lingbot_va_robotwin.json` |

After all seven pairs for a model complete, compile its ten-scene confirmation
with both the historical pilot intervention ledger and only that model's new
confirmation ledger. The `--runtime-interventions` flag is intentionally
repeated; do not combine model ledgers:

```bash
cd /home/ali/projects/steerable

python3 tools/compile_vla_wam_v2_robotwin_confirmation.py \
  --input-root /home/ali/projects/Efficient-WAM/outputs/vla_wam_shared_v2 \
  --model-id efficient_wam_rt_robotwin \
  --runtime-interventions artifacts/vla_wam_shared_v2/pilot/runtime_interventions.json \
  --runtime-interventions artifacts/vla_wam_shared_v2/pilot/directional_confirmation/runtime_interventions_efficient_wam_rt_robotwin.json \
  --invalid-attempts artifacts/vla_wam_shared_v2/pilot/directional_confirmation/invalid_attempts_efficient_wam_rt_robotwin.json

python3 tools/compile_vla_wam_v2_robotwin_confirmation.py \
  --input-root /home/ali/projects/FastWAM/outputs/vla_wam_shared_v2 \
  --model-id fastwam_robotwin \
  --runtime-interventions artifacts/vla_wam_shared_v2/pilot/runtime_interventions.json \
  --runtime-interventions artifacts/vla_wam_shared_v2/pilot/directional_confirmation/runtime_interventions_fastwam_robotwin.json \
  --invalid-attempts artifacts/vla_wam_shared_v2/pilot/directional_confirmation/invalid_attempts_fastwam_robotwin.json

python3 tools/compile_vla_wam_v2_robotwin_confirmation.py \
  --input-root /home/ali/projects/lerobot-lingbot/outputs/vla_wam_shared_v2 \
  --model-id lingbot_va_robotwin \
  --runtime-interventions artifacts/vla_wam_shared_v2/pilot/runtime_interventions.json \
  --runtime-interventions artifacts/vla_wam_shared_v2/pilot/directional_confirmation/runtime_interventions_lingbot_va_robotwin.json \
  --invalid-attempts artifacts/vla_wam_shared_v2/pilot/directional_confirmation/invalid_attempts_lingbot_va_robotwin.json
```

The 42-cell queue is complete. Each full compiler was attempted and failed
closed on the documented missing historical raw inputs; the model-specific
prospective-slice compilers produced the committed claim boundary instead.
Keep decoded video, latent-only future, and action-only future interfaces
distinct in every downstream summary.

<a id="post-result-expansion-v2-a005"></a>
## Post-result expansion — V2-A005

Status: frozen before any newly authorized inference. Machine-readable source:
[`post_result_expansion_amendment.json`](../artifacts/vla_wam_shared_v2/pilot/post_result_expansion_amendment.json).

The completed three-WAM results were known when this queue was chosen. The
addition is therefore a disclosed post-result expansion, not a rewrite of the
original freeze. Static prompts, no oracle or coach, full video retention,
valid-failure preservation, infrastructure exclusions, and strict DROID versus
RoboTwin separation remain unchanged.

| Priority | Experiment | First gate | Authorized behavioral spend |
| ---: | --- | --- | ---: |
| 0 | GR00T N1.7 DROID VLA | exact-repeat fixed-observation probe | 6 direct cells |
| 1 | Cosmos3 Edge DROID WAM | existing-adapter repeat and action-contract audit | 6 direct cells |
| 2 | LingBot-VLA 4B RoboTwin | repository/checkpoint and observation/action audit | 6 direct cells |
| 3 | Cosmos-Reason2-2B | official-interface deterministic reasoning audit | 0; diagnostic only |
| 4 | π0-FAST wording expansion | verify frozen prompt registry | 60 non-direct cells |

GR00T and Cosmos3 Edge use DROID seeds 8300–8302 with the exact direct-command
LEFT/RIGHT prompts. LingBot-VLA uses RoboTwin pairs00–02. The π0-FAST direct
cells are already complete and must not be rerun; only short-command,
goal-as-outcome, and desired-plus-negated-opposite cells at seeds 8300–8309 are
new. Cosmos-Reason2 is not an action policy unless a separately frozen adapter
proves otherwise, so its outputs never enter robot-success denominators.

Cosmos3 Edge progress: the fixed-observation contract gate and the complete
six-cell corrected-neutral-scene direct gate are valid. LEFT was 3/3 and RIGHT
was 3/3; all three paired executed-action traces were distinct, all three
endpoint orderings were aligned, and all 47 exposed 33-frame decoded futures
were retained losslessly. The competence gate is `both_directions`. The
complete hash-bearing result is
[`cosmos3_edge_droid_direct_gate.json`](../artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_droid_direct_gate.json),
with immutable pair slices for seeds 8300–8302 alongside it. Do not rerun these
six valid cells. The deterministic seed-8302 paired publication video and the
three-pair endpoint scorecard are registered in
[`media_manifest.json`](../artifacts/vla_wam_shared_v2/media/cosmos3_edge_droid/media_manifest.json).
Cosmos-Reason2's
corrected neutral-fixture 12-call diagnostic is also complete, but remains a
non-behavioral text/point diagnostic outside every robot-success denominator.

V2-A005 initially placed Light-WAM, LaWAM, pi0 DROID, and DreamZero in an
audit-only conditional second wave capped at one checkpoint. The completed
official-release audit changed that decision before any conditional-candidate
download or inference; V2-A006 below discloses and freezes the revised choice.

### Selected second wave — V2-A006

The official-release audit selected two complementary RoboTwin WAMs before
their assets were downloaded or inference began. The machine-readable freeze is
[`post_result_second_wave_amendment.json`](../artifacts/vla_wam_shared_v2/pilot/post_result_second_wave_amendment.json).

| Priority | Model | Scientific role | Authorized spend |
| ---: | --- | --- | ---: |
| 0 | Light-WAM | lightweight state-fusion test of whether FastWAM's weak alignment persists | 6 direct cells at pairs00–02 |
| 1 | LaWAM | latent-visual-subgoal interface distinct from decoded-video and action-only WAMs | 6 direct cells at pairs00–02 |

Both must pass isolated environment, pinned asset, exact-repeat, prompt-only
action sensitivity, normalization, and native RoboTwin contract gates before
behavioral inference. Light-WAM's future interface is classified only after the
released inference path is audited. LaWAM's exposed visual subgoals remain
latent-only unless an official decoder is verified before inference.

#### Light-WAM completion

Light-WAM's six authorized direct-command cells are complete and must not be
rerun. The compact hash-bearing result is
[`light_wam_robotwin_direct_gate.json`](../artifacts/vla_wam_shared_v2/pilot/expansion/light_wam_robotwin_direct_gate.json)
(42,232 bytes; SHA-256
`f33e1ff8fdc82c4a035f2cc113b91b311d685d9747dbcbb1104453fc455745d6`).
All six cells are valid behavioral evidence: LEFT is 1/3 and RIGHT is 0/3,
for one requested-direction success and five valid failures. Pair00 LEFT
succeeded after 134 actions; each other cell exhausted 400 actions. Only
pair00 had aligned endpoint ordering, while all three exact pairs had distinct
executed-action traces. The frozen competence result is therefore LEFT-only,
so no wording grid is eligible.

The released RoboTwin interface is action-only. It exposes neither decoded
video nor a retained future latent, so imagined-future evidence is not
applicable and is never zero. Three successful guarded pair processes recorded
zero runtime interventions and a maximum temperature of 45 C. Six separate
infrastructure-invalid attempts are preserved outside the model denominator:
four pre-action ffmpeg failures from pair00/pair01 and two B200 OIDN partial
events from pair02. The final pair02 evidence came from the renderer-compatible
A100 retry; the B200 partial remains preserved and is not model evidence.

The preselected pair00 execution view is now published as
[`light_wam_pair00_left_success_right_failure.mp4`](../artifacts/vla_wam_shared_v2/media/light_wam_robotwin/light_wam_pair00_left_success_right_failure.mp4)
(1,634,716 bytes; SHA-256
`1830c4a0339ffc47eebb38eb4f14a00b7adb8a507cf130e3dc3c1700fa299386`).
It shows the LEFT-success episode beside its matched RIGHT failure; the shorter
LEFT episode holds its final frame. This post-hoc rendering changes no model
denominator or score. Its reproducible composition record is
[`media_manifest.json`](../artifacts/vla_wam_shared_v2/media/light_wam_robotwin/media_manifest.json)
(SHA-256
`4c4ad1f9487191291d8614ec6cbad9fd0294384f6c6ed3159569137d356ad398`).

#### LaWAM withdrawn before inference (`V2-A009`)

LaWAM is no longer active. The user withdrew its six cells before any model
action request or behavioral episode. It is not a failure, is not assigned a
zero, and is removed from the active result and media surfaces. The following
record is retained only as historical setup provenance; do not resume it.

LaWAM's multicomponent registry was frozen before any model action request and
is committed as
[`lawam_robotwin_registry.json`](../artifacts/vla_wam_shared_v2/pilot/expansion/lawam_robotwin_registry.json)
(SHA-256
`f2331f52f574a72ab26b9ab5c5bd54dbf41699f04e42abed4bdde7cc3ad332f0`).
The clean repository is pinned at
`1add20a376126eacab02f19a62d726072a322cae`. Qwen3-VL, the released LAM,
the RoboTwin SFT policy, its normalization statistics, and the official native
EEF adapter are present at the exact revisions and carry file-level hashes in
the registry. The audited native interface is a 50-step, 16-dimensional
absolute bimanual EEF action chunk at 30 Hz.

Setup is blocked before inference on
`facebook/dinov3-vitb16-pretrain-lvd1689m` revision
`5931719e67bbdb9737e363e781fb0c67687896bc`. The official model is manually
gated; anonymous exact-revision access returns HTTP 401, and no authorized
credential or exact-revision copy exists in the ali-owned lane. This consumes
zero model action requests and zero behavioral episodes and remains outside
every model denominator. Do not assign a zero or substitute another DINO
checkpoint.

An authenticated retry on 2026-08-04 used the ephemeral user-supplied
credential without retaining its value. Exact-revision payload access still
returned HTTP 401/GatedRepoError; only the public README and license were
downloaded, no model payload was obtained, and no model was loaded or queried.
The hash-bearing retry record is
[`lawam_dinov3_authenticated_access_retry.json`](../artifacts/vla_wam_shared_v2/pilot/expansion/lawam_dinov3_authenticated_access_retry.json).
The blocker now specifically requires accepting the DINOv3 terms for the
token-owning Hugging Face account, not another anonymous or ungranted retry.

The formerly proposed download command is retained for audit only and is no
longer authorized by the active queue:

```bash
/data/users/ali/vla_wam/envs/hf-tools/bin/hf download \
  facebook/dinov3-vitb16-pretrain-lvd1689m \
  --revision 5931719e67bbdb9737e363e781fb0c67687896bc \
  --local-dir /data/users/ali/vla_wam/checkpoints/dinov3-vitb16-pretrain-lvd1689m
```

Do not execute it for this study. No LaWAM inference started.

pi0 DROID remains deferred as the least architecturally distinct family
ablation and is not assigned a zero. DreamZero-DROID was subsequently selected
by the user in the separately disclosed `V2-A007` amendment below.

<a id="experiment-5-dreamzero-droid-direct-gate"></a>
## Experiment 5 — DreamZero DROID direct gate (`V2-A007`)

Status: complete. All six authorized V2-A007 cells are valid and MUST NOT be
rerun. Compiled evidence:
[`dreamzero_droid_direct_gate.json`](../artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_droid_direct_gate.json)
(SHA-256
`4c76cdc3ca9eaf227d21d160199408f22e1b3dd7a71176a5a5dbe22223714461`).
Machine-readable source:
[`post_result_dreamzero_amendment.json`](../artifacts/vla_wam_shared_v2/pilot/post_result_dreamzero_amendment.json)
(SHA-256
`785bf3a69409e231e3a78c7427089cbe653ceb37022ddfb16edd3f1bd152ee89`).
All completed V2-A005/V2-A006 results were known when this user-directed
addition was selected, so it is a post-result breadth expansion rather than
preregistration.

The final direct-command result is LEFT 2/3 and RIGHT 1/3: seed 8300
succeeded in both conditions, seed 8301 failed after 450 actions in both
conditions, and seed 8302 succeeded LEFT at action 341 but failed RIGHT after
450 actions. Every valid failure remains in the denominator. All three matched
pairs had distinct executed-action traces and endpoint ordering aligned with
the requested LEFT-to-RIGHT change. The competence gate is therefore
`both_directions`; a future wording grid is eligible but is not authorized by
this completed six-cell slice.

The fixed-observation gate passed before behavior: repeat LEFT actions and
latent futures were bit-identical, while LEFT versus RIGHT differed in actions
(RMS `0.03544579397992704`) and latent futures (RMS
`0.16675334252293472`). The complete server retention set contains 265
behavioral latent futures plus three probe futures, and six behavioral plus
three probe official reset-decode videos. Missing or unexposed futures were not
converted to zeros. Eleven setup-invalid attempts remain outside the model
denominator; no valid-run runtime intervention occurred.

The exact official sources are `dreamzero0/dreamzero` commit
`ab790c198fbce33503358efbbd4187ce9a89adf3` and
`GEAR-Dreams/DreamZero-DROID` revision
`96ad344138c66e82536422432ad742f015784942`. The observed release contains 25
files and 64,789,159,581 payload bytes; every payload used by inference must be
hashed before model load. The official policy jointly predicts actions and a
latent video future. Retain the official decoded future separately from the
executed RTX viewport video; unexposed or missing futures are not zeros.

The bounded behavioral queue is exactly six DROID cells: static direct-command
LEFT and RIGHT at environment/sampling seeds 8300, 8301, and 8302. Reuse the
frozen prompt bytes, neutral cube/bowl reset, official release-inside-cone
success predicate, and same-seed pairing used by the completed DROID gates. Do
not substitute one of DreamZero's canned simulator scenes.

Runtime topology is deliberately split:

- launch a fresh official distributed DreamZero server on two rechecked free
  GPUs in `lerobot-b200-4gpu-1-ali`;
- run exact-reset RoboLab/Isaac simulation and every viewport capture on
  `raytrace-rtxpro6000-ali`;
- use a new study port and prove cross-pod reachability before model load;
- keep all raw actions, futures, rollouts, and unbounded videos under
  `/data/users/ali/vla_wam` on the ali PVC.

The old process PID 25608 on B200 GPU 0/port 5000 predates this amendment. It
uses a different dirty checkout and is infrastructure outside the study. Never
send it a request, stop it, modify it, or claim evidence from it.

Before the first behavioral cell, all of these gates passed:

1. exact repository/checkpoint hashes and isolated environment;
2. fresh two-B200 official server and cross-pod transport;
3. RTX PRO 6000 Vulkan/Isaac viewport rendering and persistent video writing;
4. neutral matched reset and byte-identical static prompt injection;
5. official state/action normalization and eight-action open-loop execution;
6. exact executed-action trace and measurement-only decoded-future retention;
7. RNG-restored exact-repeat fixed-observation request;
8. LEFT/RIGHT prompt-only sensitivity request.

Infrastructure failures and partial attempts stay outside the model
denominator. Preserve every valid failure and all raw evidence on the ali PVC.
The three complete paired publication clips and their hashes are recorded in
[`media_manifest.json`](../artifacts/vla_wam_shared_v2/media/dreamzero_droid/media_manifest.json)
(SHA-256
`ce453dcf22a2761867eedf93200b792c0062711d2b0cd783a715ec61e3e76cb3`).
The complete bounded official imagination archive is separately committed in
[`imagination_media_manifest.json`](../artifacts/vla_wam_shared_v2/media/dreamzero_droid/imagination/imagination_media_manifest.json)
(SHA-256
`6eb087ad9bb56f89e480fc486e67bab2b9364c5916a267ca38a93e131e16374a`).
It includes all six behavioral-session reset-decode MP4s, all three
fixed-observation probe decodes, and three derived LEFT/RIGHT paired views;
there is no outcome-based selection. These files are explicitly labeled as
model-predicted imagined futures, not simulator executions, task outcomes, or
additional behavioral episodes. The 268 exact latent tensors remain on the
ali PVC and are hash-addressed by the compiled result rather than committed to
ordinary Git.

<a id="experiment-3-lingbot-vla-4b-robotwin-onboarding"></a>
## Experiment 3 — LingBot-VLA 4B RoboTwin onboarding

Priority: complete. Status: the bounded V2-A005 six-cell direct gate is
compiled and MUST NOT be rerun. Do not confuse this VLA with the already-tested
LingBot-VA WAM.

Checkpoint: `Robbyant/lingbot-vla-4b-posttrain-robotwin`.

The compiled result is
[`lingbot_vla_4b_direct_gate.json`](../artifacts/vla_wam_shared_v2/pilot/expansion/lingbot_vla_4b_direct_gate.json)
(SHA-256
`7c0ad19833d6cbb51bb5fbdac8f9546f0e311333e498a15594b55f68dc7b6534`).
All six pair00–02 cells are valid: LEFT is 1/3 and RIGHT is 0/3. Two of three
matched endpoint orderings align with the requested command change. The
released interface is action-only, so imagined-video evidence is not
applicable rather than zero. The bounded amendment authorizes no additional
LingBot-VLA cell; a wording grid is not eligible from this left-only result.

The historical sequence was:

1. Create an isolated checkout and environment; do not modify
   `/home/ali/projects/lerobot-lingbot` in place.
2. Record repository commit, checkpoint revision, byte count, auxiliary assets,
   and license/access status in `model_readiness.json`.
3. Audit the observation/action contract against the same RoboTwin fixtures.
4. Run an exact-repeat fixed-observation probe. A contract failure is technical,
   not a model result.
5. Run only the six direct-command pilot cells at pairs 00–02, with viewport
   video and no oracle.
6. Apply the frozen gate before spending on any remaining prompt family.

<a id="experiment-4-groot-n17-droid-onboarding"></a>
## Experiment 4 — GR00T N1.7 DROID onboarding

Priority: complete. Status: the exact-repeat probe and all six V2-A005 direct
cells are complete and MUST NOT be rerun. The official `Isaac-GR00T` checkout is
pinned at `b9955401d50c92a29258732e3ad6ccd579f1bdc0`; local model
directories contain `nvidia/GR00T-N1.7-DROID` at
`05e7cc97e40dbd33b0890c35cc0214fcb0547ab5`, `nvidia/GR00T-N1.7-3B` at
`2fc962b973bccdd5d8ce4f67cc63b264d6886495`, and
`nvidia/Cosmos-Reason2-2B` at `9ce19a195e423419c349abfc86fd07178b230561`.
The local DROID checkpoint loaded on GPU 0, bound `127.0.0.1:5555`, answered a
health ping, and was shut down cleanly. Use the no-weight-download preflight in
[`experiments/groot_droid/README.md`](../experiments/groot_droid/README.md)
before launching the server. Cosmos-Reason2 remains a diagnostic-only backbone:
no adapter audit or language probe has started, and it is not authorized for a
behavioral denominator. Cosmos3 Edge is the separately authorized action WAM.

The compiled result is
[`groot_n17_droid_v2_registry.json`](../artifacts/vla_wam_shared_v2/pilot/expansion/groot_n17_droid_v2_registry.json)
(SHA-256
`95077a42bb0115bc673ea13ae5acdc6fdef6f476627804662f73c219ebd88bc7`).
LEFT is 0/3 and RIGHT is 0/3: all six are valid behavioral failures. Every
matched pair produced different executed actions and all three endpoint
orderings align with the requested LEFT-to-RIGHT change. This is consistent
language-conditioned redirection without successful task completion. Under
the frozen zero-direct-success gate, no wording expansion is authorized.

The historical sequence was:

1. Run an exact-repeat language probe using the frozen v2 prompt bytes.
2. Run only the six DROID direct-command cells at seeds 8300–8302, recording
   viewport video for every cell.
3. Apply the same competence gate; never infer unsteerability from setup failure.

## Wording grid—π0-FAST three-family expansion authorized

The four exact prompt families are direct, short, goal-as-outcome, and desired
side plus negated opposite. The latter is the contrastive condition: it asks for
one side and explicitly negates the other. It is not contradictory.

Run the full wording grid for a checkpoint only after direct competence appears
in both directions under its confirmation. π0-FAST satisfies that frozen
eligibility condition, the three-WAM pairs03–09 slices are compiled, and
`V2-A005` authorizes the remaining three prompt families. Reuse the completed
direct cells rather than rerunning them. Run short-command, goal-as-outcome,
and contrastive cells at seeds 8300–8309 with exact same-seed LEFT/RIGHT pairs,
viewport video, and the target-last token-order diagnostic. Do not change the
prompt templates in `protocol.json`.

### π0-FAST wording pre-inference blocker

The exact sixty-cell prompt registry and checkpoint are ready, but behavioral
inference is blocked before model load. The hash-bearing readiness record is
[`pi0_fast_wording_readiness.json`](../artifacts/vla_wam_shared_v2/pilot/expansion/pi0_fast_wording_readiness.json)
(SHA-256
`47b38eb2f17be802c126ef0a7e93b16693823ee2df62b8007f51bb0514baf5c5`).
The public `pi0_fast_droid_jointpos` checkpoint is fully staged on the ali PVC
as 19 files and 10,844,314,410 payload bytes, with a SHA-256 for every file.
All twenty completed direct cells remain untouched.

The blocker is exact adapter provenance. The completed direct confirmation
records OpenPI commit `9e46d3aea26417bfb564227734b95d010aa827e5`
and RoboLab commit `11142d4319e44401e0464866bb5fedf7ec8a8927`.
Neither commit exists in its upstream repository, authenticated GitHub search,
any ali-owned PVC Git object store, or the work-laptop filesystem. The PVC
directory named `RoboLab-11142d4` actually resolves to
`0aef241fb088ca21bb4ebd24448940ed56620d17`, and the available OpenPI checkout
also differs. Do not substitute either checkout or reconstruct the missing
sampling-seed and simulator-checkpoint adapter by guesswork.

This setup consumed zero model loads, zero model action requests, and zero
behavioral episodes. All sixty non-direct cells remain authorized and unrun;
the blocker is outside every model denominator. On the original machine that
still contains the two historical repositories, the exact next recovery
command is:

```bash
git -C /home/ali/openpi-robolab bundle create /tmp/openpi-robolab-all.bundle --all && \
git -C /home/ali/projects/RoboLab bundle create /tmp/robolab-all.bundle --all
```

### V2-A008 current-stack replication

After the historical revisions proved unavailable, the user explicitly
requested completing the sixty prompt/seed cells with the code that is
actually present. Amendment
[`post_result_current_stack_replication_amendment.json`](../artifacts/vla_wam_shared_v2/pilot/post_result_current_stack_replication_amendment.json)
freezes that work before current-stack model load or behavioral inference.
It uses OpenPI `c23745b5ad24e98f66967ea795a07b2588ed6c79`, RoboLab
`0aef241fb088ca21bb4ebd24448940ed56620d17`, config
`pi0_fast_droid_jointpos_polaris`, and the already hash-pinned public
`pi0_fast_droid_jointpos` checkpoint.

This is a separate post-result replication, not a repair of the historical
queue. Its fixed-observation release probe is now committed at
[`pi0_fast_current_stack_v2a008_release_probe.json`](../artifacts/vla_wam_shared_v2/pilot/expansion/pi0_fast_current_stack_v2a008_release_probe.json).
The two LEFT requests were bit-identical, but the RIGHT request returned the
same 10×8 action tensor (RMS 0.0). The prompt-sensitivity gate therefore
failed after three model requests. No behavioral episode, simulator video, or
executed-action trace was produced; all sixty registered cells remain unrun
and are not behavioral failures or zeros. Do not launch them or the 60-cell
compiler under the frozen protocol. Keep the raw fixed fixture and probe arrays
on the ali PVC; only a new disclosed amendment could authorize a redesign or
more inference.

### V2-A010 π0.5 current-stack rollout/media gate — complete

The historical π0.5 result has trajectory evidence but no committed behavioral
MP4. `V2-A010` freezes a separate six-cell current-stack direct-command gate at
OpenPI `c23745b5ad24e98f66967ea795a07b2588ed6c79`, RoboLab
`0aef241fb088ca21bb4ebd24448940ed56620d17`, and config
`pi05_droid_jointpos_polaris`. Seeds 8300–8302 each run static LEFT and RIGHT
commands with horizon 15, viewport video, and executed-action traces. Label all
results and media “π0.5 current-stack V2-A010”; never merge them with the v1
80-episode π0.5 evidence or the π0-FAST V2-A008 replication. The compiled
result is [`pi05_current_stack_v2a010_direct_gate.json`](../artifacts/vla_wam_shared_v2/pilot/expansion/pi05_current_stack_v2a010_direct_gate.json)
(SHA-256 `0c54758fe316764dbca3299d7b665e3edf59412410d844a1f825abad92045f0c`):
all six cells are valid, LEFT is 1/3, RIGHT is 3/3, all three matched endpoint
pairs are aligned, and all three executed-action pairs differ. Two LEFT
failures remain in the denominator; there were no invalid attempts or runtime
interventions. The selected lowest-seed matched pair is actual simulator
execution only in the [V2-A010 media manifest](../artifacts/vla_wam_shared_v2/media/pi05_current_stack_v2a010/media_manifest.json);
it has no imagined-future counterpart. No V2-A010 behavioral cell remains.

### V2-A011 Cosmos3 Nano Policy DROID gate

NVIDIA's Cosmos3 collection contains two DROID action policies. Cosmos3 Edge
Policy DROID is already complete under its own six-cell gate. `V2-A011` adds the
other checkpoint, `nvidia/Cosmos3-Nano-Policy-DROID` revision
`6706d7680581c255ff61e0f3bb49d90eac55c79e`, as a separate six-cell direct
gate at seeds 8300–8302. Every valid episode must retain actual simulator video,
executed actions, and the checkpoint's exposed generated future. This is not
the completed Cosmos-Reason2 static diagnostic and the two identities must
never be conflated.

Complete. The fixed-observation gate passed: repeated LEFT actions and futures
were bit-identical, while LEFT and RIGHT differed (action RMS 0.0256503;
future pixel MAE 7.4659). The six static direct cells all succeeded (LEFT 3/3,
RIGHT 3/3); every pair had aligned endpoints and distinct executed actions.
The observed RIGHT-minus-LEFT endpoint shifts were +0.4060, +0.5255, and
+0.6058 m. All 37 exposed 33-frame RGB futures remain on the ali PVC. The
compact [result](../artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_direct_gate.json),
[provenance](../artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_policy_droid_provenance.json),
and bounded seed-8300 actual-versus-prediction media are committed separately.
Do not rerun these cells.

### V2-A012/V2-A014 Cosmos3 Super base interface probe

V2-A012 freezes `nvidia/Cosmos3-Super` revision
`e0262be9d8f7586bc24c069a2aed2b665bdff266` (88 files;
132,710,200,213 bytes). The checkpoint hash gate and two-A100 load gate passed.
V2-A014 replaces only the fixed-probe input with the implementation's image-only
action-and-video route; it does not add robot state, execution, or behavioral
claims. Its exact three requests passed: repeat LEFT was bit-identical and
RIGHT differed in both action and generated video. Six behavioral
cells remain conditional and unreleased, and 10D-to-8D execution remains
blocked pending a separately verified controller. See the [base amendment](../artifacts/vla_wam_shared_v2/pilot/post_result_cosmos3_super_droid_amendment.json),
[image-only amendment](../artifacts/vla_wam_shared_v2/pilot/post_result_cosmos3_super_image_only_v2a014_amendment.json),
[registry](../artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_droid_v2a012_registry.json),
[result](../artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_image_only_v2a014_result.json),
[prediction-and-actions media](../artifacts/vla_wam_shared_v2/media/cosmos3_super_base_v2a014/media_manifest.json),
and [runbook](../experiments/cosmos/COSMOS3_SUPER_V2A012.md).

### V2-A013 Cosmos3 Edge base feasibility probe

V2-A013 freezes `nvidia/Cosmos3-Edge` revision
`ff48d22144de52de296a7b4d3a78914831007212` (48 files;
9,173,855,122 bytes). Its exact three fixed-observation requests passed: all
returned finite bfloat16 `[16,10]` actions and decodable 17-frame futures;
repeat LEFT was bit-identical, while RIGHT differed in both outputs. This is
interface evidence, not behavior. The static CuRobo audit blocks execution:
the direct parser is 0.264417 m and 89.997 degrees away from the exact USD
control frame at zero state and lacks verified mimic-joint, collision, and
link-transform parity. Six behavioral cells remain unreleased, no simulator
action was sent, and no rerun is authorized. The completed native 8D Cosmos3
Edge-Policy-DROID result remains separate. See the [amendment](../artifacts/vla_wam_shared_v2/pilot/post_result_cosmos3_edge_base_amendment.json),
[registry](../artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_base_v2a013_registry.json),
[result](../artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_base_v2a013_fixed_observation.json),
[mapping audit](../artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_base_v2a013_curobo_usd_audit.json),
and [runbook](../experiments/cosmos/COSMOS3_EDGE_BASE_V2A013.md).

### V2-A015 post-result guidance ablation — complete

Amendment
[`post_result_cfg_ablation_v2a015_amendment.json`](../artifacts/vla_wam_shared_v2/pilot/post_result_cfg_ablation_v2a015_amendment.json)
was frozen after both baseline outcomes were known and before any ablation
model request. It authorizes exactly 12 new DROID/RoboLab behavioral cells,
with six cells and a separate denominator for each configuration:

| Configuration | New intervention | Preserved comparison |
| --- | --- | --- |
| Cosmos3 Nano Policy DROID | joint action-video CFG `g=1` | completed `g=3` V2-A011 result |
| DreamZero DROID | derived negative-branch action guidance `s=2`; video CFG fixed at `5` | completed conditional-action equivalent `s=1` V2-A007 result |

DreamZero's released second branch uses a fixed visual-quality negative prompt,
not a strict empty-text unconditional prompt. Report its intervention as
“CFG-style negative-branch action guidance,” not as an official DreamZero
action-CFG feature. The preserved Cosmos3 Nano `g=3` and DreamZero `s=1`
episodes are referenced by committed hashes; they are not rerun, relabeled, or
overwritten.

Each arm runs the exact static prompts “Put the Rubik's cube to the left of the
bowl.” and “Put the Rubik's cube to the right of the bowl.” at environment and
sampling seeds 8300, 8301, and 8302. Run the frozen fixed-observation gates
first: repeat determinism, LEFT/RIGHT sensitivity, finite action shape and
range, future retention, and DreamZero `s=1` overlay equivalence. A failed gate
releases no behavioral cells for that arm. After a passed gate, complete all
six cells without outcome-dependent stopping, retaining full simulator video,
executed actions, and every exposed future.

The DreamZero release path is now complete. At `s=1`, all three returned
actions and retained latent futures were bit-exact against the archived
official V2-A007 fixed-observation probe. Repeat LEFT was bit-exact, while the
exact LEFT and RIGHT prompts remained distinct in actions (RMS
`0.03544579397992704`) and latent futures (RMS `0.16675334252293472`). At
`s=2`, repeat LEFT actions and latent futures were again bit-exact; LEFT and
RIGHT differed in actions (RMS `0.031104018930907626`) and latent futures (RMS
`0.1741597991389692`); and all returned actions were finite `[24,8]` tensors
with finite retained futures. The `s=2` outputs also differed from `s=1`, so
the derived intervention was active.

All six `s=2` behavioral cells are now complete valid evidence. Seed 8300 was
LEFT failure at the 450-action cap and RIGHT success at action 217; seed 8301
was LEFT failure at the cap and RIGHT success at action 269; seed 8302 was LEFT
success at action 280 and RIGHT success at action 265. The `s=2` arm had LEFT
`1/3`, RIGHT `3/3`, total `4/6`, compared with the preserved conditional-action
equivalent `s=1` baseline of LEFT `2/3`, RIGHT `1/3`, total `3/6`. The observed
change is primarily a redistribution toward RIGHT, not evidence that guidance
uniformly improved competence. With six post-result episodes per setting, the
one-success aggregate difference is descriptive and is not a powered
improvement claim. All six simulator videos, action traces, and future
manifests are retained; all successful guards exited zero and no runtime
thermal intervention occurred.

The hash-bound compiled result is
[`dreamzero_v2a015_action_cfg_s2_result.json`](../artifacts/vla_wam_shared_v2/pilot/expansion/dreamzero_v2a015_action_cfg_s2_result.json)
(SHA-256
`273b7191cde61b51bf90b2eda04b1910e214f48d34f2347f2a81325069c32444`).
For the exact prompt “Put the Rubik's cube to the left of the bowl.”, requested
margin was `-0.0057327524`, `0.0075855255`, and `0.1533745974` m across seeds
8300–8302 (mean `0.0517424569` m; success `1/3`). For “Put the Rubik's cube to
the right of the bowl.”, the corresponding margins were `0.1198861320`,
`0.3106073290`, and `0.2200004235` m (mean `0.2168312948` m; success `3/3`).
The RIGHT-minus-LEFT mean-margin imbalance was therefore `0.1650888380` m.
All three matched pairs had aligned endpoints and distinct executed actions.
The two valid failures were qualitatively different: one never interacted with
the cube, while the other moved it without a verified pickup. These margins
show that the RIGHT successes also finished farther inside the requested region
on average; they do not turn the six-cell aggregate into a powered improvement
claim.

This `s=2` intervention remains derived CFG-style negative-branch action
guidance using DreamZero's fixed visual-quality negative prompt. It is not an
official DreamZero action-CFG feature, and the baseline-to-`s=2` change is
reported as descriptive directional redistribution.

The independent Cosmos3 Nano `g=1` release gate also passed. Repeat LEFT
actions and complete 33-frame RGB futures were bit-identical. The exact LEFT
and RIGHT prompts produced distinct finite `[32,8]` actions (RMS
`0.018947694945167855`) and distinct futures (pixel MAE
`2.9921545294325833`).

All six `g=1` behavioral cells are now complete valid evidence. Under the exact
prompt “Put the Rubik's cube to the left of the bowl.”, seed 8300 failed at the
450-action cap, seed 8301 succeeded at action 353, and seed 8302 failed at the
cap: LEFT `1/3`. Under “Put the Rubik's cube to the right of the bowl.”, seeds
8300, 8301, and 8302 succeeded at actions 253, 129, and 257: RIGHT `3/3`.
The `g=1` arm therefore had `4/6`, versus LEFT `3/3`, RIGHT `3/3`, total `6/6`
in the preserved `g=3` baseline. The contrast comprised two fewer LEFT
successes and no loss on RIGHT; success-count balance and robustness were worse
at `g=1` in this small sample. This is a six-cell post-result pilot, not a
powered or general performance claim, and the
two individual LEFT failures are not assigned a causal mechanism.

All six viewport videos, six executed-action traces, and 64 decoded 33-frame
request futures are retained. The six episodes contain 1,892 executed actions
and 64 behavioral model requests; per-cell request counts were LEFT/RIGHT
`15/8`, `12/5`, and `15/9` for seeds 8300–8302. All three pair guards exited
zero, and no runtime intervention occurred.

The final Cosmos result and paired cross-configuration comparison are
[`cosmos3_nano_v2a015_no_cfg_g1_result.json`](../artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_nano_v2a015_no_cfg_g1_result.json)
(SHA-256
`8796f4ab9ea9490ee5b78678bc689d6fd13f27a9551006f8b5d346e202d0cc5c`)
and
[`cfg_ablation_v2a015_comparison.json`](../artifacts/vla_wam_shared_v2/pilot/expansion/cfg_ablation_v2a015_comparison.json)
(SHA-256
`0e6daadd9391b7ad38ee316371df0235e34268cc4c32597c6597578ac9f9ed1e`).
Both were compiled at Git
`2f177333545e39e35d586e577a9a4496e4e97f24` against the immutable preflight
SHA-256
`472c09568713fc3e08dbb86d3c39a8da6f328b82b06ff5aadef4e02f1892dd4d`.
The publication figure is available as
[`v2a015_cfg_guidance_ablation.svg`](../artifacts/vla_wam_shared_v2/figures/v2a015_cfg_guidance_ablation.svg)
(SHA-256
`b2fdb9dbd743f435fae9cf008dcd9287f2c83aef0840f081a8faaa974b91435a`)
and
[`v2a015_cfg_guidance_ablation.png`](../artifacts/vla_wam_shared_v2/figures/v2a015_cfg_guidance_ablation.png)
(SHA-256
`0adef9257d4832923fc50fd28515091b4495488f4149675024324c185ff45e5b`).
It shows both exact prompts, every matched seed's requested-margin change, and
raw LEFT/RIGHT success counts; no confidence interval or significance claim is
implied for `n=3` per direction and setting.

Complete, hash-bound publication media retains every valid intervention cell
without outcome-based selection. Cosmos3 Nano `g=1` has an
[all-six-cell actual-execution composite](../artifacts/vla_wam_shared_v2/media/cfg_v2a015/cosmos3_nano_g1/cosmos3_nano_no_cfg_g1_all_seeds_actual.mp4)
and an
[all-request prediction composite](../artifacts/vla_wam_shared_v2/media/cfg_v2a015/cosmos3_nano_g1/cosmos3_nano_no_cfg_g1_all_seeds_local_predictions.mp4)
containing all 64 retained 33-frame local horizons, separated by request
slates. DreamZero `s=2` has an
[all-six-cell actual-execution composite](../artifacts/vla_wam_shared_v2/media/cfg_v2a015/dreamzero_action_cfg_s2/dreamzero_action_cfg_s2_all_seeds_actual.mp4)
and a
[complete official-decoder-output composite](../artifacts/vla_wam_shared_v2/media/cfg_v2a015/dreamzero_action_cfg_s2/dreamzero_action_cfg_s2_all_seeds_imagination.mp4)
containing all six official reset decodes. Predicted or imagined media is not
execution and contributes no additional episode. The corresponding manifests
are SHA-256 `c6d82722769f84023690e6a3cdeccb3bf7316ef0f4673fa50aa56680b8029e3b`
for Cosmos and
`48b8549f909a8c52500e56f035ae84a0cc0b090cef29019e8a5f4fe0f7ee3adc`
for DreamZero.

For Cosmos3 Nano, moving from `g=3` to `g=1` changed requested success from
`6/6` to `4/6`: LEFT fell from `3/3` to `1/3`, while RIGHT remained `3/3`.
The exact LEFT prompt was “Put the Rubik's cube to the left of the bowl.” and
the exact RIGHT prompt was “Put the Rubik's cube to the right of the bowl.”
Mean requested margin fell from `0.10322199` to `0.03614888` m on LEFT and
from `0.40921744` to `0.21589671` m on RIGHT; the all-cell paired mean change
was `-0.13019692` m. The paired transitions were two LEFT success-to-failure
changes and four unchanged successes.

The mean RIGHT-minus-LEFT margin gap narrowed from `0.305995` to `0.179748` m,
but this is not improved success balance: RIGHT margin fell even more than
LEFT, while the weaker-side mean margin itself worsened from `0.103222` to
`0.036149` m and LEFT lost two successes. Endpoint separation and weaker-side
competence must therefore be read as distinct diagnostics.

Trajectory quality was also mixed by direction. Mean cube path length changed
from `0.252732` to `0.496044` m on LEFT and from `1.095421` to `0.439394` m on
RIGHT. Mean joint-action total variation changed from `5.270188` to `10.041255`
on LEFT and from `10.487576` to `6.695867` on RIGHT. The `g=1` arm did not show
uniformly smoother trajectories; the contrast was direction-specific and
coincided with reduced LEFT robustness. These
paired observations do not establish a powered or general effect and do not
identify CFG as the cause of either individual failure.

For DreamZero, moving from the conditional-action-equivalent `s=1` baseline to
derived `s=2` changed success from `3/6` to `4/6`: LEFT `2/3` to `1/3`, RIGHT
`1/3` to `3/3`. Mean requested margin changed from `0.10978410` to
`0.05174246` m on LEFT and from `0.04008883` to `0.21683129` m on RIGHT; the
all-cell paired mean change was `+0.05935041` m. Signed RIGHT-minus-LEFT bias
flipped from `-0.069695` to `+0.165089` m. Exact transitions were two RIGHT
failure-to-success changes, one LEFT success-to-failure, one unchanged failure,
and two unchanged successes. This is descriptive redistribution toward RIGHT,
not a powered aggregate improvement. DreamZero `s=2` remains derived CFG-style
negative-branch action guidance using the fixed visual-quality negative
prompt—not an official DreamZero action-CFG feature.

The nine fixed-observation action requests across both arms are diagnostics,
not robot episodes, and contribute zero observations to every behavioral
success denominator. Sixteen setup-invalid attempts are also excluded. The
Cosmos VAE failures were registry-resolution failures: its S3-form Wan2.2 VAE
URI required the exact preserved V2-A011 compat-bin `uvx` wrapper and
`HF_HOME`, so testing root- and package-level links alone was insufficient.
Three further relaunches ended before any request while those settings were
still omitted; attempt06 succeeded only after restoring the exact wrapper and
cache. The separate cuDNN discovery failure and DreamZero probe launched
without the pinned RoboLab `policies` package also ended before any request or
executed behavior. Two subsequent DreamZero behavioral launches were likewise
setup-only: the first failed the arm-identity guard because its ledger
filenames omitted `dreamzero_droid_action_cfg`; the second reached the guarded
worker but omitted the already-authorized `OMNI_KIT_ACCEPT_EULA=YES` scope and
exited during Isaac import. A third passed the EULA gate but omitted the
readiness-recorded native-library prefix and NVIDIA Vulkan ICD, producing
missing `libGL`/`libX11` and `vkCreateInstance ERROR_INCOMPATIBLE_DRIVER` before
any policy request. Its exact failed process group was validated and stopped;
the later complete environment restored the combined native and GLVND paths.
None initialized behavior or sent a model request. Exact causes and recovery
boundaries are retained in
[`cfg_ablation_v2a015_preflight.json`](../artifacts/vla_wam_shared_v2/pilot/expansion/cfg_ablation_v2a015_preflight.json).

Two later seed-8300 launches were also setup-invalid. Attempt04 used only the
RoboLab native-library prefix, leaving `libGL` unavailable from the separate
ali GLVND bundle and Warp pointed at unwritable `/home/ali-lerobot`; exact PGID
50233 was validated and stopped before a model request. Attempt05 exited 127
before either guard or worker launch because the shell invoked nonexistent bare
`python`. Attempt06 used `/usr/bin/python3`, the combined native/GLVND/FastWAM
library path, and dedicated writable Warp/XDG/MPL caches; it became the valid
seed-8300 pair reported above. Attempts04 and 05 remain setup-invalid and are
not absorbed into the behavioral denominator.

Keep the two new six-cell denominators separate from one another and from their
historical baselines. Fixed probes, partial runs, and infrastructure-invalid
attempts remain outside all behavioral denominators. This small post-result
pilot can report paired discordances, directional margins, trajectory-quality
changes, and effect sizes; it cannot establish a powered improvement or a
general model-family effect. The optional higher-guidance arms remain
unauthorized.

V2-A015 is complete: all twelve authorized cells are valid, both result
artifacts and the paired comparison are compiled, the scientific figure is
rendered, and complete actual-versus-predicted publication media is verified.
Keep the two six-cell denominators separate. No inference, rerun, plot, or
media-render cell remains. Final repository validation is:

```bash
python3 tools/validate_vla_wam_v2_protocol.py
```

## Analysis and publication queue

After each coherent result slice:

1. Compile raw outputs into hash-bearing JSON/CSV/Markdown without overwriting
   earlier pilot artifacts.
2. Regenerate figures and both 16:9 and square videos.
3. Select successes and failures deterministically; show missing categories.
4. Update the gallery and article with exact prompts, expected region, actual
   path, endpoint, failure stage, latency, memory, and future interface.
5. Run `python3 tools/validate_vla_wam_v2_protocol.py` and `git diff --check`.
6. Commit the result and update `continuation_state.json`.

Separate backlog items that do not authorize new model claims:

- replay the six frozen v1 DROID media selections with endpoint validation;
- render synchronized Cosmos imagined-versus-executed positive-event examples;
- add cost/latency/VRAM panels only from measured operational records;
- draft X and LinkedIn excerpts after the confirmation results stabilize.

## Stop-and-handoff protocol

If compute, model access, or usage credits end mid-task:

- stop cleanly; do not delete partial output directories;
- label the cell `partial` or `technical_invalid`, never behavioral failure;
- record the last completed model/pair/direction and exact next command in
  `continuation_state.json`;
- record any process/container names and ports still active;
- preserve raw logs and thermal events;
- commit only if the repository still validates and the commit represents a
  coherent evidence slice;
- otherwise leave a short `work_in_progress` entry in the continuation state
  and do not claim completion.

The next agent should be able to resume from committed files alone. Chat text is
never the sole record of an experimental decision.
