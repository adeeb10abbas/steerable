"""Export a named DIST root capture and current native API contract, without imports."""
import ast
import base64
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile


SITE = Path("/data/users/ali/vla_wam/envs/robolab-v2-isaac50/lib/python3.11/site-packages")
LAB = SITE / "isaaclab/source/isaaclab/isaaclab"
BASE = Path("/data/users/ali/sgw-01/qualification/family-native-smoke-20260923br/2/evidence")
files = {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def binding(path, raw):
    return {"path": str(path), "sha256": sha(raw), "bytes": len(raw)}


def native_source(path, methods):
    raw = path.read_bytes()
    record = None
    for dist in importlib.metadata.distributions():
        if not dist.metadata["Name"].lower().startswith("isaac"):
            continue
        for member in dist.files or ():
            if Path(member).name != path.name:
                continue
            if Path(dist.locate_file(member)).resolve() != path.resolve():
                continue
            if record is not None or member.hash is None or member.hash.mode != "sha256":
                raise ValueError("native source lacks a unique SHA-256 distribution binding")
            expected = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
            if member.hash.value != expected:
                raise ValueError("native source differs from installed distribution RECORD")
            record = {"name": dist.metadata["Name"], "version": dist.version, "record_sha256": expected}
    if record is None:
        raise ValueError(f"native source has no distribution RECORD binding: {path}")
    tree = ast.parse(raw)
    selected = {}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for method in cls.body:
            key = f"{cls.name}.{getattr(method, 'name', '')}"
            if key not in methods:
                continue
            selected[key] = {
                "lines": [method.lineno, method.end_lineno],
                "ast_sha256": sha(ast.dump(method, include_attributes=False).encode()),
                "returns": [ast.unparse(node.value) for node in ast.walk(method) if isinstance(node, ast.Return)],
                "calls": sorted({ast.unparse(node.func) for node in ast.walk(method) if isinstance(node, ast.Call)}),
            }
    if set(selected) != set(methods):
        raise ValueError("native API methods differ")
    return tree, {"source": binding(path, raw), "distribution": record, "methods": selected}


verification_raw = (BASE / "family_verification.json").read_bytes()
if sha(verification_raw) != "c8a25186c24c2b8c5e016733bf0765bb76843365eacbaa2cb71de8714789c0b4":
    raise ValueError("DIST smoke verification differs")
verification = json.loads(verification_raw)
files["family_verification.json"] = verification_raw
for name, key in (("candidate.json", "materialized_candidate"), ("candidate_capture.json", "candidate_capture")):
    path = BASE / name
    raw = path.read_bytes()
    if binding(path, raw) != verification[key]:
        raise ValueError("native candidate/capture differs from original verification")
    files[name] = raw
candidate = json.loads(files["candidate.json"])
capture = json.loads(files["candidate_capture.json"])
if capture["environment_origin_world_xyz_m"] != [0, 0, 0]:
    raise ValueError("prospective root frame is not the recorded zero-origin frame")
for actor in ("rubiks_cube", "bowl"):
    if capture["objects"][actor]["root_position_env_local_xyz_m"] != candidate["object_poses"][actor]["position_m"]:
        raise ValueError("candidate roots differ from actual native capture")

native = {}
for relative, cls in (
    ("assets/rigid_object/rigid_object_data.py", "RigidObjectData"),
    ("assets/articulation/articulation_data.py", "ArticulationData"),
):
    _, native[cls] = native_source(LAB / relative, {
        f"{cls}.{name}" for name in ("root_pos_w", "root_quat_w", "root_link_pos_w", "root_link_quat_w", "root_link_pose_w")
    })
apis = list((SITE / "isaacsim").glob("exts*/omni.physics.tensors*/omni/physics/tensors/impl/api.py"))
if len(apis) != 1:
    raise ValueError("native tensor API is ambiguous")
tree, native["PhysX"] = native_source(apis[0], {
    "ArticulationView.get_root_transforms", "RigidBodyView.get_transforms",
})
for cls in tree.body:
    if isinstance(cls, ast.ClassDef) and cls.name in ("ArticulationView", "RigidBodyView"):
        name = "get_root_transforms" if cls.name == "ArticulationView" else "get_transforms"
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == "self._frontend.create_tensor"]
        if len(calls) != 1 or ast.unparse(calls[0].args[1]) != "float32":
            raise ValueError("native root tensor dtype is not explicit float32")

tree, native["SimulationContext"] = native_source(
    LAB / "sim/simulation_context.py", {"SimulationContext.__init__"},
)
units = [node.value for node in ast.walk(tree) if isinstance(node, ast.keyword)
         and node.arg == "stage_units_in_meters"]
if not units or any(not isinstance(value, ast.Constant) or type(value.value) is not float
                    or value.value != 1.0 for value in units):
    raise ValueError("native stage does not explicitly use metre units")

robolab = subprocess.check_output([
    "git", "-C", "/data/users/ali/vla_wam/external/RoboLab-pi05-v3-0aef241",
    "show", "0aef241fb088ca21bb4ebd24448940ed56620d17:robolab/core/world/world_state.py",
])
if sha(robolab) != "a4c12dc07673b0733c53990244e86577d16fa98888772a6c93934903f9699bbf":
    raise ValueError("pinned native pose getter changed")

value = {
    "schema_version": "sgw-01-native-metric-contract-export-v1",
    "native_sources": native,
    "native_root_tensor_dtype": "float32",
    "stage_units_in_meters": 1.0,
    "robolab_getter_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
    "robolab_getter_sha256": sha(robolab),
    "files": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in files.items()},
    "historical_native_import_bytes_independently_attested": False,
    "claim_boundary": (
        "Current distribution-verified native API implementation and a historically bound SGW DIST capture. "
        "The native API semantics match the registered Isaac Sim5/IsaacLab2.2 contract; this is not a "
        "historical import-byte attestation, global history coverage or fixture release."
    ),
    "model_requests": 0, "behavioral_episodes": 0, "release_permitted": False,
}
files["native_contract.json"] = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
    for name, raw in sorted(files.items()):
        entry = tarfile.TarInfo(name)
        entry.size = len(raw)
        archive.addfile(entry, io.BytesIO(raw))
