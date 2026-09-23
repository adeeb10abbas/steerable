import ast
import base64
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys


failure_path = Path(
    "/data/users/ali/vla_wam/raw/v3e006_r005/state_repair/"
    "98f0234-a40r06-attempt01/raw/state_construction_failure.json"
)
with failure_path.open("rb") as stream:
    raw = stream.read(32 * 1024 * 1024 + 1)
assert len(raw) == 25614537
failure_digest = hashlib.sha256(raw).hexdigest()
assert failure_digest == "e858a4431bd0451a1e380d66c8c5d05c5486718f5b23388f874d24a7d271ae8e"
failure = json.loads(raw)
argv = failure["invocation"]
root = Path(argv[argv.index("--robolab-root") + 1])
revision = argv[argv.index("--expected-robolab-commit") + 1]
assert revision == "0aef241fb088ca21bb4ebd24448940ed56620d17"
sources = []
for relative, expressions in (
    ("robolab/core/environments/base.py", (
        "record_initial_state: InitialStateRecorderCfg = InitialStateRecorderCfg()",
        "record_states: PostStepStatesRecorderCfg = PostStepStatesRecorderCfg()",
    )),
    ("robolab/core/events/basic_recorders.py", (
        "self.initial_state = self.extract_env_ids_values(self._env.scene.get_state(is_relative=True), env_ids)",
        'return "states", self._env.scene.get_state(is_relative=True)',
    )),
):
    blob = subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", f"{revision}:{relative}"])
    actual = (root / relative).read_bytes()
    assert actual == blob
    lines = actual.decode().splitlines()
    selected = []
    for expression in expressions:
        locations = [index for index, line in enumerate(lines, 1) if expression in line]
        assert len(locations) == 1
        selected.append({"line": locations[0], "expression": expression})
    sources.append({
        "path": str(root / relative), "revision": revision,
        "bytes": len(actual), "sha256": hashlib.sha256(actual).hexdigest(),
        "current_bytes_match_pinned_git_blob": True, "expressions": selected,
    })
dist = metadata.distribution("isaaclab")
assert dist.version == "2.2.0"
relative = "isaaclab/source/isaaclab/isaaclab/scene/interactive_scene.py"
entry = next(item for item in dist.files if str(item) == relative)
path = Path(dist.locate_file(entry))
raw = path.read_bytes()
assert entry.hash.mode == "sha256" and entry.size == len(raw)
assert base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=") == entry.hash.value
module = ast.parse(raw)
getter = next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == "get_state")
lines = raw.decode().splitlines()
assert 'asset_state["root_pose"] = rigid_object.data.root_pose_w.clone()' in "\n".join(lines[getter.lineno-1:getter.end_lineno])
assert 'asset_state["root_pose"][:, :3] -= self.env_origins' in "\n".join(lines[getter.lineno-1:getter.end_lineno])
result = {
    "schema_version": "sgw-01-r005-hdf5-recorder-source-contract-v1",
    "failure_report_sha256": failure_digest,
    "historical_robolab_root_argument": str(root),
    "historical_robolab_expected_revision_argument": revision,
    "robolab_sources": sources,
    "installed_isaaclab_getter": {
        "path": str(path), "version": dist.version, "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "distribution_record_sha256": entry.hash.value,
        "matches_distribution_record": True,
        "get_state_lines": [getter.lineno, getter.end_lineno],
    },
    "source_contract": {
        "initial_state_capture": "recorder reset, emitted by record_post_reset",
        "states_capture": "record_post_step",
        "is_relative": True,
        "rigid_root_position": "native world actor-root position minus environment origin",
        "rigid_root_quaternion": "world axes; no robot-base rotation",
    },
    "historical_imported_getter_bytes_independently_attested": False,
    "historical_population_coverage_complete": False,
    "release_permitted": False,
    "model_requests": 0,
    "behavioral_episodes": 0,
    "claim_boundary": (
        "The historical invocation pins RoboLab 0aef241 despite the directory's old 11142d4 name. "
        "Its recorder Git blobs request relative state. The currently installed IsaacLab2.2.0 "
        "getter matches its distribution RECORD and subtracts environment origins without "
        "rotation. This records the source contract, not an independent historical import-byte "
        "attestation, complete population recovery or a replacement for missing materialization states."
    ),
}
json.dump(result, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
sys.stdout.write("\n")
