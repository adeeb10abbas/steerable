# Incremental external-repository bundles

These are **incremental** Git bundles, not standalone source archives. Start
from the official repository and exact prerequisite below, verify the bundle,
fetch its named ref, then detach at the target commit. Do not apply these as
patches, rebase them, or substitute current upstream `main`.

| Bundle | Official prerequisite | Final checkout |
| --- | --- | --- |
| `efficient-wam-robotwin.bundle` | `https://github.com/RoboTwin-Platform/RoboTwin.git` at `c3ddfa8b97d5519efa828b075999bd0006778e5e` | `0bd8e76fde3afcffa4b30a3e3e8f92a206aa66cc` (`codex/efficient-wam-gate`) |
| `efficient-wam-steerability.bundle` | `https://github.com/jiajun613/Efficient-WAM.git` at `2bd75a8c56acfcd5754b98c7ed313176911ccae0` | `b0b6cfabcbd68d18888866e958c677ce640f0412` (`codex/left-right-gate`) |
| `fastwam-steerability.bundle` | `https://github.com/yuantianyuan01/FastWAM.git` at `45d8e1458921d83f8ad6cf9ce993d371208dabd0` | `068d3fd70c89df3726c09893f47b75a624b20c02` (`codex/language-steering-gate`) |
| `lingbot-va-steerability.bundle` | `https://github.com/huggingface/lerobot.git` at `adccdea1cfbec83ed98263feb7e59f7d047c5692` | `d42efbc04e502057dab4b18bb14770cc48e85131` (`codex/lingbot-language-gate`) |

The first bundle is the patched RoboTwin dependency used by Efficient-WAM. It
is independent of the Efficient-WAM bundle. FastWAM vendors its RoboTwin code
in its final bundled commit; LingBot's gate README pins its own dependency.

## Exact installation sequence

Set `BUNDLES` to this directory and `EXT` to a PVC-backed external-source
directory. `git bundle verify` must pass before `fetch`.

```bash
set -euo pipefail
: "${BUNDLES:?absolute path to steerable/handoff/repo_bundles}"
: "${EXT:?PVC-backed external source root}"
mkdir -p "$EXT"

git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$EXT/EfficientWAM-RoboTwin"
git -C "$EXT/EfficientWAM-RoboTwin" checkout --detach c3ddfa8b97d5519efa828b075999bd0006778e5e
git -C "$EXT/EfficientWAM-RoboTwin" bundle verify "$BUNDLES/efficient-wam-robotwin.bundle"
git -C "$EXT/EfficientWAM-RoboTwin" fetch "$BUNDLES/efficient-wam-robotwin.bundle" refs/heads/codex/efficient-wam-gate:refs/remotes/handoff/efficient-wam-gate
git -C "$EXT/EfficientWAM-RoboTwin" checkout --detach 0bd8e76fde3afcffa4b30a3e3e8f92a206aa66cc

git clone https://github.com/jiajun613/Efficient-WAM.git "$EXT/Efficient-WAM"
git -C "$EXT/Efficient-WAM" checkout --detach 2bd75a8c56acfcd5754b98c7ed313176911ccae0
git -C "$EXT/Efficient-WAM" bundle verify "$BUNDLES/efficient-wam-steerability.bundle"
git -C "$EXT/Efficient-WAM" fetch "$BUNDLES/efficient-wam-steerability.bundle" refs/heads/codex/left-right-gate:refs/remotes/handoff/left-right-gate
git -C "$EXT/Efficient-WAM" checkout --detach b0b6cfabcbd68d18888866e958c677ce640f0412

git clone https://github.com/yuantianyuan01/FastWAM.git "$EXT/FastWAM"
git -C "$EXT/FastWAM" checkout --detach 45d8e1458921d83f8ad6cf9ce993d371208dabd0
git -C "$EXT/FastWAM" bundle verify "$BUNDLES/fastwam-steerability.bundle"
git -C "$EXT/FastWAM" fetch "$BUNDLES/fastwam-steerability.bundle" refs/heads/codex/language-steering-gate:refs/remotes/handoff/language-steering-gate
git -C "$EXT/FastWAM" checkout --detach 068d3fd70c89df3726c09893f47b75a624b20c02

git clone https://github.com/huggingface/lerobot.git "$EXT/lerobot-lingbot"
git -C "$EXT/lerobot-lingbot" checkout --detach adccdea1cfbec83ed98263feb7e59f7d047c5692
git -C "$EXT/lerobot-lingbot" bundle verify "$BUNDLES/lingbot-va-steerability.bundle"
git -C "$EXT/lerobot-lingbot" fetch "$BUNDLES/lingbot-va-steerability.bundle" refs/heads/codex/lingbot-language-gate:refs/remotes/handoff/lingbot-language-gate
git -C "$EXT/lerobot-lingbot" checkout --detach d42efbc04e502057dab4b18bb14770cc48e85131

git -C "$EXT/EfficientWAM-RoboTwin" rev-parse HEAD
git -C "$EXT/Efficient-WAM" rev-parse HEAD
git -C "$EXT/FastWAM" rev-parse HEAD
git -C "$EXT/lerobot-lingbot" rev-parse HEAD
```

The four final lines must print, in order, `0bd8e76…`, `b0b6cfa…`,
`068d3fd…`, and `d42efbc…`. A missing prerequisite is a setup blocker; never
force a bundle into an unrelated checkout.

The three final model commits include prospective action-trace instrumentation.
Every pair03–pair09 `result.json` must declare `action_trace.path`, `sha256`,
`count`, and `shape`. Missing trace evidence is technically invalid, never a
zero action-distance value.
