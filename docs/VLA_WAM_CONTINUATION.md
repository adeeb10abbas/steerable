# VLA/WAM study continuation handoff

Updated: 3 August 2026, after the compiled π0-FAST directional confirmation
and the first prospective Efficient-WAM-RT RoboTwin pair.

This document is the restart point for a human or coding model with no chat
context. The machine-readable companion is
[`continuation_state.json`](../artifacts/vla_wam_shared_v2/continuation_state.json).

## One-minute orientation

The study asks whether changing an episode-static language command changes the
requested physical outcome. It compares checkpoints inside two separate arenas:
DROID/RoboLab and RoboTwin. Raw success rates are never pooled across arenas.

Four new-model direct gates are complete:

| Arena | Checkpoint | LEFT | RIGHT | Frozen next step |
| --- | --- | ---: | ---: | --- |
| DROID | π0-FAST | 1/10 | 10/10 | direct competence in both directions; wording eligible but deferred |
| RoboTwin | Efficient-WAM-RT | 2/3 pilot + 0/1 confirmation | 0/3 pilot + 0/1 confirmation | pair03 valid and complete; run pairs04–09 |
| RoboTwin | FastWAM | 1/3 | 0/3 | add paired scenes 03–09 under direct commands only |
| RoboTwin | LingBot-VA | 3/3 | 0/3 | add paired scenes 03–09 under direct commands only |

The three WAM gates still trigger the one-direction-only branch. Efficient-WAM
pair03 is valid completed evidence and must not be rerun: both requested
relations failed after 400 actions, the first ten executed actions differed
(RMS 0.0762), and the RIGHT-minus-LEFT endpoint shift was -0.0081 m
(anti-aligned). That leaves 40 prospective WAM episodes. π0-FAST has
now completed its direct-only confirmation: all 20 episodes are behaviorally
valid, all ten endpoint pairs align with the requested LEFT-to-RIGHT change,
and one thermal event excludes wall latency only. Its four-wording grid is
eligible under the frozen rule, but is explicitly **deferred and not
authorized** until the three WAM confirmations are compiled and a post-result
decision is recorded. No WAM wording sweep is authorized.

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

Priority: **first**. Remaining cost: 40 new episodes. Status: active next
confirmation; Efficient-WAM pair03 is complete and fixtures/registry are ready.

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

The exact next authorized model cells are FastWAM `robotwin_pair_03` and
LingBot-VA `robotwin_pair_03`; they may run concurrently in isolated model
environments and with separate intervention/invalid-attempt ledgers.

The pair03 commands below are retained as provenance and as templates. Do not
run the Efficient-WAM command or any Efficient-WAM pair03–09 cell again. For
FastWAM and LingBot-VA, replace the pair id, anchor task, environment seed,
sampling seed, and output/log path together using the table above.

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

After all 42 cells, run the compiler above for all three models, then regenerate
progression, endpoint, media, and article artifacts. Keep
decoded video, latent-only future, and action-only future interfaces distinct.

<a id="experiment-3-lingbot-vla-4b-robotwin-onboarding"></a>
## Experiment 3 — LingBot-VLA 4B RoboTwin onboarding

Priority: third. Status: blocked because the isolated repository and checkpoint
are not present. Do not confuse this VLA with the already-tested LingBot-VA WAM.

Checkpoint: `Robbyant/lingbot-vla-4b-posttrain-robotwin`.

Sequence:

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

Priority: fourth. Status: onboarding assets and server smoke complete; the
frozen direct-command pilot remains unrun. The official `Isaac-GR00T` checkout
is pinned at `b9955401d50c92a29258732e3ad6ccd579f1bdc0`; local model
directories contain `nvidia/GR00T-N1.7-DROID` at
`05e7cc97e40dbd33b0890c35cc0214fcb0547ab5`, `nvidia/GR00T-N1.7-3B` at
`2fc962b973bccdd5d8ce4f67cc63b264d6886495`, and
`nvidia/Cosmos-Reason2-2B` at `9ce19a195e423419c349abfc86fd07178b230561`.
The local DROID checkpoint loaded on GPU 0, bound `127.0.0.1:5555`, answered a
health ping, and was shut down cleanly. Use the no-weight-download preflight in
[`experiments/groot_droid/README.md`](../experiments/groot_droid/README.md)
before launching a future server.

Sequence:

1. Run an exact-repeat language probe using the frozen v2 prompt bytes.
2. Run only the six DROID direct-command cells at seeds 8300–8302, recording
   viewport video for every cell.
3. Apply the same competence gate; never infer unsteerability from setup failure.

## Wording grid—π0-FAST eligible but explicitly deferred

The four exact prompt families are direct, short, goal-as-outcome, and desired
side plus negated opposite. The latter is the contrastive condition: it asks for
one side and explicitly negates the other. It is not contradictory.

Run the full wording grid for a checkpoint only after direct competence appears
in both directions under its confirmation. π0-FAST now satisfies that frozen
eligibility condition, but its grid is explicitly deferred and **not
authorized** until the three-WAM directional confirmations are compiled and a
post-result decision records whether to spend on it. When eventually
authorized, retain exact same-seed LEFT/RIGHT pairs and the target-last
token-order diagnostic. Do not change the prompt templates in `protocol.json`.

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
