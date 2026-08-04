#!/usr/bin/env python3
"""V2-A010 pi0.5 exact-repeat and prompt-sensitivity release probe."""

from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from openpi_client import websocket_client_policy


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--fixture",type=Path,required=True);p.add_argument("--fixture-manifest",type=Path,required=True);p.add_argument("--registry",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--remote-host",required=True);p.add_argument("--remote-port",type=int,default=8001);p.add_argument("--sampling-seed",type=int,default=8300000);a=p.parse_args()
    manifest=json.loads(a.fixture_manifest.read_text())
    if manifest["fixture_sha256"]!=sha256(a.fixture):raise ValueError("fixture hash mismatch")
    registry=json.loads(a.registry.read_text());prompts={r["requested_relation"]:r["rendered_prompt"] for r in registry["cells"] if r["environment_seed"]==8300}
    with np.load(a.fixture,allow_pickle=False) as z:obs={k:z[k] for k in z.files}
    client=websocket_client_policy.WebsocketClientPolicy(a.remote_host,a.remote_port);a.output_dir.mkdir(parents=True,exist_ok=True)
    arrays={};records={}
    for label,relation in (("left_a","left"),("left_b","left"),("right","right")):
        response=client.infer({**obs,"prompt":prompts[relation],"sampling_seed":a.sampling_seed})
        if response.get("v2a010_sampling_seed")!=a.sampling_seed:raise ValueError("seed attestation mismatch")
        action=np.asarray(response["actions"],dtype=np.float32)
        if action.shape!=(15,8):raise ValueError(action.shape)
        path=a.output_dir/f"{label}_actions.npy";np.save(path,action,allow_pickle=False);arrays[label]=action;records[label]={"prompt":prompts[relation],"sampling_seed":a.sampling_seed,"action_path":str(path),"action_sha256":sha256(path),"shape":list(action.shape),"dtype":str(action.dtype)}
    repeat=bool(np.array_equal(arrays["left_a"],arrays["left_b"]));rms=float(np.sqrt(np.mean((arrays["left_a"]-arrays["right"])**2)));passed=repeat and rms>0
    result={"schema_version":"vla-wam-v2a010-pi05-current-release-probe-v1","fixture_manifest":str(a.fixture_manifest),"fixture_sha256":sha256(a.fixture),"registry_path":str(a.registry),"registry_sha256":sha256(a.registry),"records":records,"metrics":{"left_exact_repeat_bit_identical":repeat,"left_vs_right_action_rms":rms},"passed":passed}
    out=a.output_dir/"release_probe.json";out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2))
    if not passed:raise SystemExit(20)


if __name__=="__main__":main()
