# VLA/WAM study continuation handoff

Updated: 3 August 2026, after commit `c4c67ad` and before any directional-
confirmation inference.

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
| DROID | π0-FAST | 0/3 | 3/3 | add seeds 8303–8309 under direct commands only |
| RoboTwin | Efficient-WAM-RT | 2/3 | 0/3 | add paired scenes 03–09 under direct commands only |
| RoboTwin | FastWAM | 1/3 | 0/3 | add paired scenes 03–09 under direct commands only |
| RoboTwin | LingBot-VA | 3/3 | 0/3 | add paired scenes 03–09 under direct commands only |

All four therefore triggered the one-direction-only branch. **No short,
declarative, or contrastive wording sweep is authorized yet.** This is the most
important decision to preserve.

The current article, figures, and videos are:

- [`VLA_VS_WAM_STEERABILITY_STUDY.md`](VLA_VS_WAM_STEERABILITY_STUDY.md)
- [`VLA_WAM_STEERABILITY_VIDEO_GALLERY.html`](VLA_WAM_STEERABILITY_VIDEO_GALLERY.html)
- [`figures_manifest.json`](../artifacts/vla_wam_shared_v2/figures/figures_manifest.json)
- [π0-FAST media manifest](../artifacts/vla_wam_shared_v2/media/droid_pi0_fast_pairs/media_index.json)
- [RoboTwin media manifest](../artifacts/vla_wam_shared_v2/media/robotwin_wam_pairs/media_index.json)

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

Expected evidence baseline: validator status `valid`, 243 checks. The live host
snapshot at handoff had two idle 24 GiB RTX 3090s and 464 GiB free. Treat that
as historical context and recheck it; it is not a guarantee.

Preserve unrelated dirt in external repositories. At handoff, the known items
were `/home/ali/projects/RoboLab/.cache/` and
`/home/ali/projects/Efficient-WAM/sapien_offscreen.png`.

<a id="experiment-1-pi0-fast-droid-directional-confirmation"></a>
## Experiment 1 — π0-FAST DROID directional confirmation

Priority: **first**. Cost: 14 new episodes. Status: ready.

Frozen registry:
[`pi0_fast_directional_expansion.json`](../artifacts/vla_wam_shared_v2/pilot/pi0_fast_directional_expansion.json).
Seeds 8300–8302 are complete. Run exactly 8303–8309, LEFT and RIGHT once per
seed. Do not rerun completed seeds unless hash validation shows corruption.

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

Priority: **second**. Cost: 42 new episodes. Status: fixtures and registry ready.

Frozen registry:
[`directional_expansion.json`](../artifacts/vla_wam_shared_v2/pilot/directional_expansion.json).
Model-blind fixture validation:
[`directional_fixture_validation.json`](../artifacts/vla_wam_shared_v2/pilot/directional_fixture_validation.json).

Run pair03–pair09 for each of Efficient-WAM-RT, FastWAM, and LingBot-VA. Each
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

Use the checked-in launch wrappers in the model repositories; they set the
correct Python path, Vulkan ICD, headless environment, and local virtualenv.
The pair03 commands illustrate the exact form:

```bash
# Efficient-WAM-RT
cd /home/ali/projects/Efficient-WAM
EFFICIENT_WAM_GPU=1 experiments/robotwin_language_gate/run_gate_3090.sh \
  --output-dir outputs/vla_wam_shared_v2/directional_confirmation/pair03 \
  --task place_a2b_right --seed 4300003 --sampling-seed 8403 \
  --prompt-family direct_command --max-actions 400 \
  --save-simulator-video --predicted-video-max-chunks 1

# FastWAM
cd /home/ali/projects/FastWAM
FASTWAM_GPU=0 experiments/robotwin_language_gate/run_gate_3090.sh \
  --output-dir outputs/vla_wam_shared_v2/directional_confirmation/pair03 \
  --cell place_a2b_right:4300003:8403 \
  --prompt-family direct_command --max-actions 400 \
  --action-horizon 32 --replan-steps 24 --num-inference-steps 10 \
  --text-cfg-scale 2.0 --save-simulator-video --resume

# LingBot-VA
cd /home/ali/projects/lerobot-lingbot
LINGBOT_GPU=0 VLA_WAM_V2_STUDY_ROOT=/home/ali/projects/steerable \
  experiments/lingbot_language_gate/run_v2_directional_confirmation.sh --dry-run

# After the study preflight passes, run the exact 14 new cells.
LINGBOT_GPU=0 VLA_WAM_V2_STUDY_ROOT=/home/ali/projects/steerable \
  experiments/lingbot_language_gate/run_v2_directional_confirmation.sh --run
```

The LingBot wrapper reads pair03–pair09 directly from the frozen registry and
launches one scene per invocation, each with its paired LEFT/RIGHT direct
commands. Do not pass all seven tasks or seeds directly to the runner: repeated
CLI values form an unsafe Cartesian product. Do not change model guidance,
horizon, diffusion steps, task config, or future-retention settings. LingBot-VA
previously required thermal pauses; exclude interrupted wall-time measurements
while retaining valid behavior.

After all 42 cells, extend the compiler to produce a ten-scene result per model,
then regenerate progression, endpoint, media, and article artifacts. Keep
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

Priority: fourth. Status: blocked on gated assets. The RoboLab client exists,
but the DROID checkpoint and `nvidia/Cosmos-Reason2-2B` dependency are absent.
Use the no-weight-download preflight in
[`experiments/groot_droid/README.md`](../experiments/groot_droid/README.md)
before installing anything.

Sequence:

1. Verify Hugging Face access without changing study files.
2. Install the server in an isolated Python 3.12 repository/environment and pin commits.
3. Download `nvidia/GR00T-N1.7-DROID` plus required assets; record revisions and
   exact byte counts.
4. Run a one-observation server/client contract smoke.
5. Run an exact-repeat language probe using the frozen v2 prompt bytes.
6. Run only the six DROID direct-command cells at seeds 8300–8302.
7. Apply the same competence gate; never infer unsteerability from setup failure.

## Wording grid—conditional, not currently runnable

The four exact prompt families are direct, short, goal-as-outcome, and desired
side plus negated opposite. The latter is the contrastive condition: it asks for
one side and explicitly negates the other. It is not contradictory.

Run the full wording grid for a checkpoint only after direct competence appears
in both directions under its confirmation. When authorized, retain exact
same-seed LEFT/RIGHT pairs and the target-last token-order diagnostic. Do not
change the prompt templates in `protocol.json`.

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
