#!/usr/bin/env python3
"""Bind the E003 runtime identity to the registered queue and ali-owned lane."""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts/vla_wam_shared_v3/phase_e/bilateral_symmetry_null_control_v3e003"
REG=BASE/"registration.json"
CANDIDATE=BASE/"symmetry_gate/candidate.json"

def sha_bytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def sha(p:Path)->str:return sha_bytes(p.read_bytes())
def canon(v:object)->bytes:return json.dumps(v,allow_nan=False,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
def main()->None:
 reg_sha=sha(REG); commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 topology={"schema_version":"vla-wam-shared-v3b-pi05-live-topology-v1","policy_server":{"owner":"ali","pod":"lerobot-b200-4gpu-1-ali","pod_uid":"1e0f438c-6041-4cc3-af32-0c118963e54c","pod_ip":"10.244.103.110","gpu_uuid":"GPU-4ca76921-a7d2-e920-8555-47e0e8f105f7","gpu_model":"NVIDIA B200","driver_version":"580.105.08","gpu_index":2,"port":8001,"model_request_count_at_capture":0},"simulator_lanes":[{"owner":"ali","pod":"raytrace-rtxpro6000-ali","pod_uid":"d5b0405a-a9b1-4baa-a802-d5171e03c228","pod_ip":"10.244.222.28","gpu_uuid":"GPU-f28bd513-a38a-b768-7589-d2959f814ae8","gpu_model":"NVIDIA RTX PRO 6000 Blackwell Server Edition","driver_version":"580.105.08"}]}
 topology_sha=sha_bytes(canon(topology))
 runtime={"schema_version":"vla-wam-shared-v3e003-runtime-identity-v1","study_id":"vla_wam_language_steerability_v3","amendment_id":"V3-E003","model_id":"pi05_current_stack_droid","openpi_commit":"c23745b5ad24e98f66967ea795a07b2588ed6c79","robolab_commit":"0aef241fb088ca21bb4ebd24448940ed56620d17","openpi_config":"pi05_droid_jointpos_polaris","checkpoint_manifest_sha256":"f5a56d9565f9381ccdeeaa165b0495dab6d17a81836cc7b01c5fbc6ab89e74ca","checkpoint_sha256":"b193b28b05f9755e24d44a6f5cf3185ca23c2ad3da6c5913370379c82570fbf6","release_manifest_sha256":reg_sha,"study_git_commit":commit,"action_space":"joint_position_8d","action_chunk_shape":[15,8],"open_loop_horizon":15,"action_cap":450,"instruction_controller":"static_episode_prompt","future_interface":"actions_only","missing_future_policy":"action_only_interface_not_applicable_never_zero","renderer_backend":"Isaac Sim viewport realtime RTX Vulkan balanced","simulator_version":"Isaac Sim 5.0.0.0 / Isaac Lab 2.2.0 / RoboLab 0.2.1","live_topology":topology,"live_topology_sha256":topology_sha,"base_runtime_identity_sha256":"de55b8a73e8f27e0b67faf841595002578fcc1b3d7b583a151171114efead15f","environment_lock_sha256":"8ab95005e08716a16eefb18846bb57bf436e71a3a102d428cb38df1f30ddb29b","external_repository_diff_hash":"9cb1e3c8adc8fbeb420c605f07c472eb7c210fc61cc91f6b7aec9e88ca3d559f","openpi_dir_status_sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","robolab_dir_status_sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}
 runtime["runtime_identity_sha256"]=sha_bytes(canon(runtime)); BASE.joinpath("runtime_identity.json").write_text(json.dumps(runtime,indent=2,sort_keys=True)+"\n")
 gate={"schema_version":"vla-wam-shared-v3e003-release-gate-v1","study_id":runtime["study_id"],"amendment_id":"V3-E003","model_id":runtime["model_id"],"behavioral_release":True,"model_blind_symmetry_gate_passed":True,"model_blind_model_request_count":0,"model_blind_behavioral_episode_count":0,"registration_sha256":reg_sha,"runtime_identity_sha256":runtime["runtime_identity_sha256"],"live_topology_sha256":topology_sha,"symmetry_candidate_sha256":sha(CANDIDATE),"lane_bindings":[topology["simulator_lanes"][0]],"release_rule":"Only the registered 54 E003 cells may launch after this gate; infrastructure-invalid attempts remain outside denominators."}
 gate["release_gate_sha256"]=sha_bytes(canon(gate)); BASE.joinpath("release_gate.json").write_text(json.dumps(gate,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"registration_sha256":reg_sha,"runtime_identity_sha256":runtime["runtime_identity_sha256"],"release_gate_sha256":gate["release_gate_sha256"],"study_git_commit":commit},indent=2))
if __name__=="__main__":main()
