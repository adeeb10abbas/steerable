# Cosmos3-Edge base DROID feasibility — V2-A013

This is a separate base-checkpoint feasibility arm for
`nvidia/Cosmos3-Edge` at revision
`ff48d22144de52de296a7b4d3a78914831007212`. It does not rerun or relabel the
completed six-cell `Cosmos3-Edge-Policy-DROID` native 8D result.

Current release state: three fixed-observation model requests are frozen but
not run; zero simulator cells are released. The base checkpoint has an official
generic 10D DROID action interface, not an official native 8D DROID execution
contract. Its only allowed behavioral branch is therefore explicitly labeled
derived-control CuRobo IK.

## 1. Rebuild and inspect the metadata freeze

This command downloads metadata and small public source files only. It does not
download weights, load a model, contact Kubernetes, or run inference.

```bash
python3 tools/build_v2a013_cosmos3_edge_base_registry.py
python3 tools/validate_vla_wam_v2_protocol.py >/tmp/v2_validation.json
git diff --check
```

The builder must resolve 48 files and 9,173,855,122 bytes at the exact model
revision. It also verifies pinned source-file hashes for Cosmos Framework,
vLLM-Omni, RoboLab, and CuRobo.

## 2. Artifact and software gates

On an ali-owned PVC, discover the live ali-owned environment and choose an
isolated model environment. Do not assume a pod, image, mount, GPU index, or
credential. Restore these commits and record their Git heads:

- NVIDIA/cosmos: `e494d734022ab0610061cdf57fa24c843e18767e`
- NVIDIA/cosmos-framework: `a904d2d36b774a51dd06ff9ff906816b1a04f579`
- vllm-project/vllm-omni: `900a7f0813d0482811b0e4dfd3cf7deabbe2429f`
- NVlabs/RoboLab: `0aef241fb088ca21bb4ebd24448940ed56620d17`
- NVlabs/curobo: `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`
  (`0.7.8`)

After PVC capacity and credentials pass, the exact checkpoint download command
is:

```bash
hf download nvidia/Cosmos3-Edge \
  --revision ff48d22144de52de296a7b4d3a78914831007212 \
  --local-dir /data/users/ali/vla_wam/checkpoints/cosmos3_edge_base_ff48d221
```

Before model load, hash every local file and compare it to the registry's byte,
Git-blob, and LFS records. Record an immutable environment/container digest,
package versions, source heads, visible GPUs, free memory, and model-load result.
An OOM or load failure is infrastructure evidence and authorizes no request.

## 3. Three-request fixed-observation probe

Only after the artifact and software gates pass, start the pinned vLLM-Omni
server with the exact base revision. Use the generic asynchronous video-policy
endpoint, not the RoboLab action-only websocket path. Every request uses the
same committed Edge diagnostic RGB bytes (raw RGB SHA-256
`6261ce5ab21383342c2012c14f7ff97d3dcd74e5f4202f2b3444355cc7ba3332`).

Issue exactly these conditions at sampling seed 8300:

1. `LEFT`: `Put the Rubik's cube to the left of the bowl.`
2. `LEFT exact repeat`: byte-identical multipart request
3. `RIGHT`: `Put the Rubik's cube to the right of the bowl.`

The frozen request body follows the pinned official action recipe:

```text
POST /v1/videos
model=nvidia/Cosmos3-Edge
input_reference=<frozen RGB image/jpeg>
size=640x480
num_frames=17
fps=5
num_inference_steps=30
guidance_scale=1.0
flow_shift=5.0
seed=8300
extra_params={"action_mode":"policy","domain_name":"droid_lerobot","raw_action_dim":10,"action_chunk_size":16}
```

Poll `/v1/videos/{id}` to completion. Save the full response JSON, top-level
action data, and `/v1/videos/{id}/content` MP4. The gate passes only if every
action is finite `[16,10]` with `raw_action_dim=10` and `domain_id=8`, every MP4
decodes to 17 frames, LEFT repeats are bit-identical, and LEFT/RIGHT actions and
videos differ. A missing, latent-only, corrupt, or substituted future is a
documented interface-gate failure, not a zero. These requests never send a
simulator action and never enter a behavioral denominator.

## 4. Exact 10D interpretation

The source-backed raw vector is:

```text
[delta_x, delta_y, delta_z,
 rot6d_col0_x, rot6d_col0_y, rot6d_col0_z,
 rot6d_col1_x, rot6d_col1_y, rot6d_col1_z,
 gripper]
```

For `backward_framewise`, `delta_T = inverse(T_i) @ T_i+1`; reconstruct with
`T_i+1 = T_i @ delta_T`. Rot6D is the first two rotation-matrix columns and is
projected to SO(3) during decode. No separate action normalizer is applied by
the pinned generic request route. Before behavior, a runtime source attestation
must prove that the DROID/RoboLab translation coordinates are meters and must
freeze base/world/control frames and quaternion ordering.

## 5. CuRobo gate — currently blocking all six cells

Do not use CuRobo's bundled Panda model as a substitute. The simulator uses:

- RoboLab asset: `assets/robots/franka_robotiq_2f_85_flattened.usd`
- LFS SHA-256: `f555695465687548a1bd31b5e3f30385182d476a67c17080b7820ad0ef747e41`
- control body: `Gripper/Robotiq_2F_85/base_link`

CuRobo's pinned `franka_panda.urdf` has SHA-256
`6a0044e6e72ee667927f17d1871ec3e2615a8bc5fe978882fc909e4094667967`
but models `panda_hand` and Panda fingers. The kinematic/collision/control-frame
mismatch blocks it.

Before the first behavioral action, supply and hash an exact Franka+Robotiq
URDF/collision description matched to that USD, independently verify the
`panda_link0` to Robotiq `base_link` transform and joint limits, and record the
entire collision world for the frozen reset. Then use the exact solver contract
in the registry: CuRobo `IKSolver`, 20 seeds, 5 mm position tolerance, 0.05 rad
rotation tolerance, self-collision check and optimization on, CUDA graph off,
and deterministic per-request seeding. Select the successful collision-free
candidate nearest the current seven arm joints, with candidate index as the
tie-break.

Derived output is `[panda_joint1..panda_joint7, gripper]` in radians plus the
RoboLab gripper scalar. Any missing solution, nonfinite value, tolerance,
limit, collision, stale-state, unit, frame, asset, or hash failure sends no
action and is written to the controller-rejection ledger.

## 6. Conditional behavioral cells

Only after every gate passes, run the six frozen static direct-command cells at
seeds 8300–8302, LEFT and RIGHT. For every valid cell retain the actual viewport
MP4, executed derived 8D trace, raw 10D model chunks, decoded imagined futures,
simulator state, solver records, and hashes. Preserve valid failures. Exclude
infrastructure failures and partial runs from denominators.

Always report these as `Cosmos3-Edge base DROID derived-control CuRobo IK`.
Never pool them with the completed native `Cosmos3-Edge-Policy-DROID` row.
