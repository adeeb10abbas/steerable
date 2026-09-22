# SGW-01 implementation and launch boundary

The immutable handoff is in [`spec/`](spec/README.md). Its 18 prompts, 174
six-cell blocks and 1,044 planned episodes were reproduced byte-for-byte.
Neither the imported specification nor historical V2/V3 protocols are edited
by this implementation.

**No SGW-01 learned-policy request, behavioral episode, or physically qualified
fixture exists yet.** Local synthetic tests are engineering checks, not study
evidence. Runtime factories, server trace provenance, full resets, physical-time
maps and the live simulator remain subject to qualification before release.

Current restart state and cluster evidence:

- [`STATUS.md`](../../../artifacts/workshops/spatial_grounding_v1/STATUS.md)
- [`continuation_state.json`](../../../artifacts/workshops/spatial_grounding_v1/continuation_state.json)

## Local development

From the repository root:

```bash
uv sync --frozen --extra dev --extra sgw --python 3.12
.venv/bin/python -m pytest -q tests/test_sgw_*.py
.venv/bin/python tools/validate_vla_wam_v3_protocol.py --quiet
.venv/bin/python tools/validate_vla_wam_v2_protocol.py
```

The `sgw` extra supplies the real viewport encoder/decoder. Raw arrays are
retained losslessly; an encoded video does not replace timestamped source
frames. Production encoder identity must be recorded in the runtime binding.

## Implementation surfaces

`contract`, `release`, `recorder`, and `worker` implement release validation,
bounded persistent attempts, locking, completion pointers and finite partitions.
`adapters`, `runtime`, and `trace` implement the policy interface and provenance
checks. `scoring`, `compile`, and `prediction_annotations` keep physical outcomes,
technical missingness, censoring and unobserved predictions separate.

`runtime_preflight` is a zero-model renderer probe, not a fixture qualification.
`lat_workspace_capture` records the actual scene after that probe passes;
`lat_candidate_generator`, `robolab_lat_qualification` and
`model_blind_qualification` are the model-blind LAT qualification path.
HEIGHT/DIST have contracts and selection checks, **not qualified physical
fixtures**. Do not substitute repeated LAT layouts for those branches.

## Execution order

1. Obtain a genuinely idle, authorized RTX/Vulkan-capable allocation. A
   Kubernetes GPU request or Ready pod is insufficient proof.
2. Run the immutable zero-model renderer preflight into a new persistent attempt
   directory. Preserve failures. Do not reuse or overwrite old receipts.
3. Capture actual assets/workspace, generate and qualify the family fixtures,
   and verify the real runtime, raw recorder, server trace and physical-time
   mappings. Keep simulator scoring state out of policy inputs.
4. Bind the direct fixed-input qualification and stage-specific receipts before
   releasing P, then D, then C. Advancement depends on technical correctness,
   never favorable task outcomes. Keep the supplied within-block order.
5. Run the finite released worker partition and regenerate analysis only from
   verified completion records. Do not promote synthetic records to a release.

The RTX PRO allocation was occupied, but a fresh A40 allocation passed the
idle guard. A bounded replacement renderer preflight passed with the
historically proven native-library order and actual three-camera scene
evidence. Fixture/runtime qualification and worker storage/budget guards
still block behavioral release. The existing B200 workload is not owned by
this task and must not be stopped. No policy server was started. The source manuscript remains a plan,
and Overleaf synchronization also requires an authenticated connection.
