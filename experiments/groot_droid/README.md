# GR00T N1.7 DROID controlled-access preflight

This is the onboarding gate for the exact DROID model in the v2 study:
`nvidia/GR00T-N1.7-DROID`. It does not replace the existing LIBERO CMI probe
in `experiments/groot_cmi`, and it is not a simulator rollout or a study
episode.

The DROID checkpoint metadata is public, but its official runtime loads the
gated `nvidia/Cosmos-Reason2-2B` backbone. Do not substitute the public Qwen
processor used by the separate CMI experiment: that substitution is not an
official DROID benchmark configuration.

## Fast preflight

Run this before downloading model tensors or starting a server:

```bash
cd /home/ali/projects/steerable
uv run --no-project --python 3.12 experiments/groot_droid/preflight.py
```

It requires CPython 3.12, matching the current official Isaac-GR00T runtime,
even though the primary study repository has a
different environment. The command requests only the small gated Cosmos
`config.json`, caps the read at 64 KiB, and exits before weight download or
server startup. Expected status on an unauthenticated or unapproved host is
`blocked_gated_cosmos_access` with exit code 20.

## The one external prerequisite

No local code can grant the required model access. In a browser, use the
Hugging Face account that will run the experiment to accept/request access at
[`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B).
Once it is approved, authenticate that account on this host once:

```bash
hf auth login
```

Do not put a token in source, shell history, an experiment card, or a command
line. Re-run the preflight after login. A passing preflight next checks that
the official `Isaac-GR00T` checkout contains
`gr00t/eval/run_gr00t_server.py`; clone it separately from this dirty study
workspace if it is absent:

```bash
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git \
  /home/ali/projects/Isaac-GR00T
```

## Post-access server contract smoke

This is a server/client integration smoke, not an authorized v2 pilot. Do not
run any frozen direct-command episode until the missing experiment card has
been restored and the study's required protocol validation has been run.

The official DROID server uses the matching embodiment tag and the RoboLab
client's validated horizon of eight:

```bash
cd /home/ali/projects/Isaac-GR00T
CUDA_VISIBLE_DEVICES=0 uv run --python 3.12 python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.7-DROID \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --device cuda --host 127.0.0.1 --port 5555 --use-sim-policy-wrapper
```

After the server reports `tcp://127.0.0.1:5555`, the only next check is the
official RoboLab smoke task:

```bash
cd /home/ali/projects/RoboLab
uv venv --python 3.12
uv sync --extra isaac50
export OMNI_KIT_ACCEPT_EULA=Y
uv run --python 3.12 python policies/gr00t/run.py \
  --task BananaOnPlateTask --remote-host 127.0.0.1 --remote-port 5555 \
  --open-loop-horizon 8
```

The adapter contract is fixed: two `180x320` HWC `uint8` images, DROID 9-D
EEF state plus joints/gripper, and no client-side letterboxing. Keep the
server and simulator on separate GPUs if a subsequent authorized study card
requires that allocation.
