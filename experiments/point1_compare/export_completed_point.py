"""Export a completed E6 workpoint from immutable per-setting checkpoints.

This is a derived view of existing observations, never a new protocol run.
The original multi-point process may continue. Exact origin checkpoints,
manifest and archived code are preserved so the view can be reconciled.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from .artifacts import create_run, digest, dump_json, finalize_run
from .run_compare import ROOT, _new_run_id
from .run_performance import stats
from .validate_results import _inside, validate


def export_point(source, point):
    source = Path(source).resolve()
    origin_bytes = (source/"manifest.json").read_bytes()
    original = json.loads(origin_bytes)
    if original.get("schema_version") != 2 or original.get("experiment") != "e6":
        raise ValueError("Only schema-2 E6 complete workpoints are supported")
    selected = [s for s in original["settings"] if tuple(s[k] for k in ["N", "M", "d"]) == point]
    required = {"hvsa", "hvsa_three_mask_control", "hvsa_verification_off"}
    if len(selected) != 3 or {s["variant"] for s in selected} != required:
        raise ValueError("A point requires all three declared E6 controls exactly once")
    checkpoints, checkpoint_bytes = [], {}
    for spec in selected:
        relative = "checkpoints/"+spec["setting"]+".json"
        data = _inside(source, relative).read_bytes()
        checkpoint = json.loads(data)
        obs = checkpoint["observations"]
        expected = list(range(spec["warmup"]+spec["rounds"]))
        if (checkpoint["specification"] != spec or [o["round"] for o in obs] != expected):
            raise ValueError("Point is incomplete or specification changed: "+spec["setting"])
        if not all(o["valid"] for o in obs):
            raise ValueError("A failed control needs review; do not export a success-only projection")
        checkpoints.append(checkpoint)
        checkpoint_bytes[relative] = data
    for name, expected in original["source_sha256"].items():
        if digest(_inside(source, original["source_archive"]+"/"+name)) != expected:
            raise ValueError("Original archived code changed: "+name)
    run_id = _new_run_id("e6_complete_point_view")
    manifest = copy.deepcopy(original)
    manifest.update(run_id=run_id, settings=selected, execution_status="running",
                    artifact_kind="derived_complete_point", command="derived view: "+str(source)+" point="+str(point))
    manifest.pop("output_sha256", None)
    manifest["derivation"] = {"origin_run_id": original["run_id"], "origin_run_directory": str(source),
                              "origin_execution_status": original["execution_status"],
                              "point": list(point), "new_protocol_execution": False,
                              "origin_manifest": "origin/manifest.json", "origin_checkpoints": {},
                              "tool_source": "derivation/export_completed_point.py"}
    directory = Path(ROOT)/"results"/"point1_compare"/run_id
    create_run(directory, manifest, source/original["source_archive"])
    (directory/"origin"/"checkpoints").mkdir(parents=True)
    (directory/"origin"/"manifest.json").write_bytes(origin_bytes)
    manifest["derivation"]["origin_manifest_sha256"] = digest(directory/"origin"/"manifest.json")
    for relative, data in checkpoint_bytes.items():
        target = directory/"origin"/relative
        target.write_bytes(data)
        manifest["derivation"]["origin_checkpoints"]["origin/"+relative] = digest(target)
    (directory/"derivation").mkdir()
    tool_source = directory/"derivation"/"export_completed_point.py"
    tool_source.write_bytes(Path(__file__).read_bytes())
    manifest["derivation"]["tool_source_sha256"] = digest(tool_source)
    raw, messages, summary = [], [], []
    for checkpoint in checkpoints:
        raw.extend({**r, "run_id": run_id} for r in checkpoint["raw"])
        messages.extend({**r, "run_id": run_id} for r in checkpoint["messages"])
        measured = [o for o in checkpoint["observations"] if not o["warmup"]]
        spec = checkpoint["specification"]
        metrics = {"wall_ms": [o["wall_ms"] for o in measured],
                   "total_sent_bytes": [o["total_sent_bytes"] for o in measured]}
        metrics.update({r+"_ms": [o["role_ms"][r] for o in measured] for r in ["user", "edge", "ta", "cloud"]})
        for metric, values in metrics.items():
            summary.append({"setting": spec["setting"], "scheme": spec["variant"], "metric": metric, **stats(values)})
    correctness = {"experiment": "e6", "schemes": manifest["schemes"],
                   "observations": [o for c in checkpoints for o in c["observations"]],
                   "initialization": [{k: c[k] for k in ["specification", "init_ms", "init_sent_bytes", "cache"]}
                                      for c in checkpoints]}
    finalize_run(directory, manifest, raw, messages, correctness, summary, [])
    problems = validate(directory)
    if problems:
        raise ValueError("Derived view failed reconciliation: "+"; ".join(problems))
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--point", required=True, help="N,M,d")
    args = parser.parse_args(argv)
    point = tuple(map(int, args.point.split(",")))
    if len(point) != 3:
        parser.error("Use N,M,d")
    print(export_point(args.source, point))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
