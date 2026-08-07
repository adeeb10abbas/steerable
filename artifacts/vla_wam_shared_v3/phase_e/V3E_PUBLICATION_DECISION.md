# Phase-E publication decision

**Decision: do not add V3-E001 or V3-E002 to the evidence denominator.** Both
experiments were registered and pushed, but both stopped before their first
scientific request/episode at required infrastructure gates.

- V3-E001: no hash-bound pair of complete fixed observations for the exact
  V3-B001 control/reflected layouts; request count 0/336.
- V3-E002: Isaac/RTX renderer and CUDA gate failed on the ali-owned RTX PRO
  6000 pod; behavioral count 0/108 and learned-model request count 0.

The registrations, validators, fail-closed runners, compilers, and renderers
are committed on the Phase-E branch. The existing V3-B001–B003 results remain
closed and unchanged. No unsupported manuscript replacement text is emitted
from these controls.

## Validation

```text
python3 tools/validate_vla_wam_v3_protocol.py
V3 protocol validation passed: 628 checks

python3 tools/validate_v3e001.py
status: valid; requests: 336; behavioral_episodes: 0

python3 tools/validate_v3e002.py
status: valid; behavioral_episodes: 108; learned_model_requests: 0
```

The validator counts above describe the registered designs, not completed
evidence. The blocker memos contain the exact restart conditions and PVC log
identity.
