"""Resume one pre-bound SGW Job after its recorded worker naturally succeeds.

The account needs GET on four named Pods and two Jobs, plus PATCH on the one
target Job. The only mutation is an atomic UID/spec-guarded change of suspend
from true to false. No Pod binding, eviction, deletion or node provisioning
operation is used; the target Pod is pre-bound at noncritical priority zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

NAMESPACE = "211247-prod"
SCHEMA = "sgw-01-native-successor-handoff-v2"
SERVICE_ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _owner(value: dict[str, Any], uid: str) -> bool:
    return any(
        row.get("kind") == "Job" and row.get("uid") == uid and row.get("controller") is True
        for row in value["metadata"].get("ownerReferences", [])
    )


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA or config.get("namespace") != NAMESPACE:
        raise ValueError("handoff schema or namespace differs")
    source, target = config["predecessor"], config["target"]
    pods = source["pods"]
    if (
        not 1 <= len(pods) <= 4
        or len({pod["node"] for pod in pods}) != len(pods)
        or len({pod["name"] for pod in pods}) != len(pods)
        or len({pod["uid"] for pod in pods}) != len(pods)
        or {pod["index"] for pod in pods} != set(range(len(pods)))
        or source["uid"] == target["uid"]
        or target["name"] in {pod["name"] for pod in pods}
        or not target["name"].startswith("sgw01-ali-")
        or target["node"] not in {pod["node"] for pod in pods}
        or type(config["timeout_seconds"]) is not int
        or not 1 <= config["timeout_seconds"] <= 172800
        or not re.fullmatch(r"[0-9a-f]{64}", target["spec_sha256"])
    ):
        raise ValueError("handoff does not bind a finite unique source/target set")
    for name in [source["name"], target["name"], *(pod["name"] for pod in pods)]:
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", name):
            raise ValueError("invalid API resource name")


def completed_indexes(value: str, count: int) -> set[int]:
    if not value:
        return set()
    result: set[int] = set()
    for term in value.split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", term):
            raise ValueError("invalid completedIndexes")
        bounds = [int(part) for part in term.split("-")]
        first, last = bounds[0], bounds[-1]
        if not 0 <= first <= last < count:
            raise ValueError("completedIndexes outside the registered predecessor")
        result.update(range(first, last + 1))
    return result


def select_node(
    config: dict[str, Any], job: dict[str, Any],
    source_pods: list[dict[str, Any]], target: dict[str, Any],
) -> str | None:
    """Fail closed on identity changes; never treat absent/failed work as capacity."""
    validate_config(config)
    source, expected = config["predecessor"], config["target"]
    count = len(source["pods"])
    if (
        job["metadata"]["uid"] != source["uid"]
        or job["metadata"]["name"] != source["name"]
        or job["metadata"]["namespace"] != NAMESPACE
        or job["spec"].get("completionMode") != "Indexed"
        or job["spec"].get("completions") != count
        or job["spec"].get("parallelism") != count
        or not (job["spec"].get("backoffLimit") == 0 or job["spec"].get("backoffLimitPerIndex") == 0)
        or job["spec"].get("podReplacementPolicy") != "Failed"
        or any(rule.get("action") == "Ignore" for rule in job["spec"].get("podFailurePolicy", {}).get("rules", []))
        or job["spec"].get("suspend", False)
        or len(source_pods) != count
    ):
        raise ValueError("predecessor identity or finite no-retry contract changed")
    if (
        target["metadata"]["uid"] != expected["uid"]
        or target["metadata"]["name"] != expected["name"]
        or target["metadata"]["namespace"] != NAMESPACE
        or target["metadata"].get("deletionTimestamp")
        or target["spec"].get("suspend") is not True
        or target["spec"].get("parallelism") != 1
        or target["spec"].get("completions") != 1
        or target["spec"].get("backoffLimit") != 0
        or target["spec"]["template"]["spec"].get("schedulerName") != expected["name"]
        or target["spec"]["template"]["spec"].get("priority") != 0
        or target["spec"]["template"]["spec"].get("priorityClassName")
        or target["spec"]["template"]["spec"].get("nodeName") != expected["node"]
        or any(target.get("status", {}).get(key) for key in ("startTime", "active", "succeeded", "failed", "terminating"))
        or digest(target["spec"]) != expected["spec_sha256"]
    ):
        raise ValueError("target is not the exact registered unstarted suspended Job")
    done = completed_indexes(job.get("status", {}).get("completedIndexes", ""), count)
    eligible: list[str] = []
    for record, pod in zip(source["pods"], source_pods):
        if (
            pod["metadata"]["name"] != record["name"]
            or pod["metadata"]["namespace"] != NAMESPACE
            or pod["metadata"]["uid"] != record["uid"]
            or pod["metadata"].get("deletionTimestamp")
            or not _owner(pod, source["uid"])
            or pod["metadata"].get("annotations", {}).get("batch.kubernetes.io/job-completion-index") != str(record["index"])
            or pod["spec"].get("nodeName") != record["node"]
            or pod["spec"].get("restartPolicy") != "Never"
            or len(pod["spec"]["containers"]) != 1
            or pod["spec"]["containers"][0]["resources"]["requests"].get("nvidia.com/gpu") != "1"
        ):
            raise ValueError("predecessor Pod identity or allocation changed")
        statuses = pod["status"].get("containerStatuses", [])
        if (
            record["index"] in done and pod["status"]["phase"] == "Succeeded"
            and len(statuses) == 1
            and statuses[0].get("state", {}).get("terminated", {}).get("exitCode") == 0
            and statuses[0].get("restartCount") == 0
        ):
            eligible.append(record["node"])
    return expected["node"] if expected["node"] in eligible else None


class Kubernetes:
    def __init__(self) -> None:
        host = os.environ["KUBERNETES_SERVICE_HOST"]
        if ":" in host:
            host = f"[{host}]"
        self.root = f"https://{host}:{os.environ['KUBERNETES_SERVICE_PORT_HTTPS']}"
        self.context = ssl.create_default_context(cafile=str(SERVICE_ACCOUNT / "ca.crt"))

    def request(self, method: str, path: str, payload: Any = None) -> dict[str, Any]:
        request = Request(
            self.root + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={
                "Authorization": "Bearer " + (SERVICE_ACCOUNT / "token").read_text().strip(),
                "Content-Type": "application/json-patch+json" if method == "PATCH" else "application/json",
            },
            method=method,
        )
        try:
            with urlopen(request, context=self.context, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            detail = error.read(32768).decode("utf-8", errors="replace")
            raise URLError(f"Kubernetes {method} {path} returned HTTP {error.code}: {detail}") from error


def api_path(resource: str, name: str) -> str:
    prefix = "/apis/batch/v1" if resource == "jobs" else "/api/v1"
    return f"{prefix}/namespaces/{NAMESPACE}/{resource}/{name}"


def write_receipt(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(config: dict[str, Any], output: Path, api: Kubernetes) -> None:
    validate_config(config)
    deadline = time.monotonic() + config["timeout_seconds"]
    while time.monotonic() < deadline:
        job = api.request("GET", api_path("jobs", config["predecessor"]["name"]))
        pods = [
            api.request("GET", api_path("pods", record["name"]))
            for record in config["predecessor"]["pods"]
        ]
        target_path = api_path("jobs", config["target"]["name"])
        target = api.request("GET", target_path)
        node = select_node(config, job, pods, target)
        print(json.dumps({
            "status": "eligible" if node else "waiting_for_natural_success",
            "source_phases": [pod["status"]["phase"] for pod in pods],
            "target_node": node, "observed_at_unix": time.time(),
        }), flush=True)
        if node:
            write_receipt(output / "resume-intent.json", {
                "config_sha256": digest(config), "node": node,
                "predecessor_job": job, "predecessor_pods": pods, "target_job": target,
            })
            patch = [
                {"op": "test", "path": "/metadata/uid", "value": config["target"]["uid"]},
                {"op": "test", "path": "/spec", "value": target["spec"]},
                {"op": "replace", "path": "/spec/suspend", "value": False},
            ]
            resumed_spec = {**target["spec"], "suspend": False}
            try:
                api.request("PATCH", target_path, patch)
            except URLError:
                # Reconcile a lost acknowledgement without submitting a second mutation.
                observed = api.request("GET", target_path)
                if observed["metadata"]["uid"] != config["target"]["uid"] or observed["spec"] != resumed_spec:
                    raise
            observed = api.request("GET", target_path)
            if observed["metadata"]["uid"] != config["target"]["uid"] or observed["spec"] != resumed_spec:
                raise RuntimeError("resume acknowledgement does not match the registered target")
            write_receipt(output / "result.json", {
                "status": "resumed_after_natural_predecessor_success", "node": node,
                "target_job": observed, "model_requests": 0, "release_permitted": False,
            })
            return
        time.sleep(30)
    raise TimeoutError("no registered worker naturally succeeded before the handoff deadline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError("handoff configuration bytes changed")
    args.output.mkdir()
    try:
        run(json.loads(raw), args.output, Kubernetes())
    except Exception as error:
        write_receipt(args.output / "infrastructure-failure.json", {
            "type": type(error).__name__, "message": str(error),
            "status": "handoff_failed_no_release", "model_requests": 0,
        })
        raise


if __name__ == "__main__":
    main()
