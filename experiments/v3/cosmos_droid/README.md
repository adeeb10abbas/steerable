# Cosmos3 Edge/Nano DROID — v3 Phase A

This directory is a fail-closed prospective adapter for the 54 `authorized_new`
cells (27 matched seed pairs) of each Cosmos checkpoint. It never launches or
relabels seeds 8300–8302, never edits a v2 file, and never substitutes a newer
server, checkpoint, reset, prompt, predicate, or action/future contract.

## Order of operations

1. Run the repository-wide v3 validator and cluster preflight.
2. Restore the exact model-specific Cosmos checkout, RoboLab commit, checkpoint,
   and separate environment used by v2.
3. Create `runtime_identity.json` with `preflight.py`. It verifies the clean
   repository commits, registered checkpoint payload hashes, frozen task/client
   hashes, and the selected environment lock.
4. Run `fixed_observation_gate.py` against seed 8303. It issues LEFT, identical
   LEFT repeat, and RIGHT requests from the exact frozen observation. Exact
   repeat actions/futures must be identical; LEFT/RIGHT actions and futures must
   differ. Missing future output is an infrastructure failure, never a zero.
5. Run one registered pair with `run_pair.py`. Both directions are created from
   the same environment/sampling seed and frozen neutral reset; only the exact
   prompt and relation predicate differ.
6. Export one simulator JSON per direction and compile the pair with
   `compile_pair.py`. The compiler requires viewport video, the simulator-matched
   executed action trace, every 33-frame decoded future, an N+1 raw state stream,
   the separate scorer `final_detached_release` boolean, contact stream/status,
   and one identical initial-state fingerprint. It emits two validated behavioral
   JSONL rows plus a post-close hash manifest.

If a request becomes technical-invalid or partial after the video/action write
preflight, use `record_infrastructure.py` to write it to a separate validated
JSONL stream. Missing artifacts are never fabricated; failures before those
artifacts exist remain in the model-specific setup/intervention ledger.

Do not repair a technical attempt by changing seeds. Record it in the v3
infrastructure ledger and repair only the identical registered cell.

## Simulator export contract

`compile_pair.py` expects schema
`vla-wam-shared-v3-cosmos-simulator-export-v1`. Required identity fields are
checked byte-for-byte against the queue. `steps` contains the pre-action state
at `action_step=0` followed by one state after each executed action. Each sample
contains robot-base-frame `object_xyz`, `reference_xyz`, `grippers_open`, and
either a retained `contact_detected` boolean on every sample or an explicit
instrumentation-unavailable reason. `final_detached_release` must come from the
frozen scorer; it is not inferred from success or the gripper action.

Each `policy_requests` row points to the raw returned `[32,8]` action array and
decoded 33-frame RGB future. The raw arrays, HDF5/state export, videos, and model
weights stay on the PVC. Only compact JSONL, manifests, and selected media belong
in Git.

## Live prerequisites

- ali-owned pod/PVC with working Vulkan/SAPIEN rendering and sufficient free GPU
  memory; the cluster API must be reachable before any pod action;
- Cosmos3 Edge: `cosmos-framework` commit
  `a904d2d36b774a51dd06ff9ff906816b1a04f579`, checkpoint revision
  `3ea407af3e156c0af3b4bb6edd85842cc9a58777`, port 18010;
- Cosmos3 Nano: `cosmos-framework` commit
  `411d25b2e35bc441126f48c44a4b93e1c0564274`, checkpoint revision
  `6706d7680581c255ff61e0f3bb49d90eac55c79e`, port 18011;
- RoboLab commit `0aef241fb088ca21bb4ebd24448940ed56620d17`;
- the exact environment locks and existing compat-bin/HF cache used by the v2
  server stacks;
- a simulator-side export hook that retains N+1 state/contact samples and the
  frozen scorer's detached-release boolean. No inference is authorized until
  that output-write preflight and the fixed-observation gate pass.

Static verification:

```bash
.venv/bin/python -m unittest tests.test_v3_cosmos_droid_adapter -v
python3 -m py_compile experiments/v3/cosmos_droid/*.py
```
