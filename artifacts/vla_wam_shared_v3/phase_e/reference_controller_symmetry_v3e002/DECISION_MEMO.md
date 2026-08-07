# V3-E002 decision memo

Status: **blocked at the model-blind renderer/controller gate**. No learned
model request and no behavioral episode was issued.

The exact ali-owned RTX PRO 6000 pod was visible and the pinned RoboLab
checkout was clean. Isaac Sim accepted the EULA after an explicit terminal
acceptance, but the zero-request preflight did not produce a valid renderer or
controller gate. The captured log reports missing `libGL.so.1` and
`libXt.so.6`, failed RTX plugin loading, Warp CUDA error 36, and PhysX's
“no suitable CUDA GPU” diagnostic. `nvidia-smi` could see the GPU, but that is
not sufficient for the required Vulkan/Isaac capture and verified planning
stack. The attempt was stopped without launching the queue.

Attempt log on the PVC:

`/data/users/ali/vla_wam/raw/v3e/reference_controller_symmetry_v3e002/model_blind_gate/attempt02/run.log`

SHA-256: `864c553971623a5e1ea144d7ca886a3412fe52dd204f64c797e7530317683758`

## Counts

| quantity | registered | completed |
|---|---:|---:|
| model-blind episodes | 108 | 0 |
| learned-model requests | 0 | 0 |
| infrastructure-invalid attempts | — | 1 gate attempt |

## Safe restart condition

Repair or select an ali-owned image with the required GL/X11 runtime libraries
and CUDA/Vulkan-compatible Isaac stack, rerun the zero-request static planning
gate, and only then enable the deterministic reference-controller queue.
