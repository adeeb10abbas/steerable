# Historical model and runtime identity

Verified from the committed V1 evidence on 2026-09-12. These are historical
receipts, not a fresh inspection of model weights or the original execution host.

- Model: `nvidia/Cosmos3-Edge-Policy-DROID`.
- Hugging Face checkpoint revision: `3ea407af3e156c0af3b4bb6edd85842cc9a58777`.
- Visual localizer: `Qwen/Qwen3-VL-2B-Instruct`, revision
  `89644892e4d85e24eaac8bacfd4f463576704203`.
- Logged policy interface: joint-position action space, eight action dimensions,
  32-action chunks, decoded future enabled.

The source is `artifacts/vla_wam_shared_v1/checkpoint_provenance.json`, checked
against the study preregistration and confirmation operational snapshot.
"V1" means the study version; it does not mean Cosmos 1. The Cosmos 3 paper
citation in the manuscript describes the model family.

The checkpoint manifest's host aggregate hashes include absolute paths and
follow snapshot symlinks. They are historical host receipts, not portable model
distribution checksums. None of the model weights were downloaded or rehashed
for this workshop audit.

The repository metadata in
`artifacts/vla_wam_shared_v1/final_evidence/compiled_evidence.json`, under
`provenance.repositories`, records:

| Runtime repository | Historical base commit | Local modification status |
| --- | --- | --- |
| cosmos-framework | `1439c1d5e45a23771e9b1a2ad8f40a5981ea86c0` | Dirty, 6 paths |
| RoboLab | `992bc34eedb2b909888af8a334a2ac33b86c51d8` | Dirty, 1 path |

Exact runtime reproduction requires the historical local patches in addition
to those commits. Recover and hash the original modifications before claiming
an exact rerun. The current offline audit reproduces committed labels and cached
localization semantics; it does not reproduce model inference or simulation.
The audited evidence repository's pin is independently recorded in the package
README and source-hash inventory.
