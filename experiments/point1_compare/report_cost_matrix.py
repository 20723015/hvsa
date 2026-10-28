"""Validated E2 N/d/M matrix analysis; no new protocol observations.

Traverse the immutable batch lineage, preserve each source run, exclude pilots
and interrupted prefixes, and require the complete planned ten unique points.
Plots remain pending visual review until an assistant actually inspects them.
"""
from __future__ import annotations
import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from .artifacts import digest, dump_json
from .run_compare import ROOT
from .validate_results import validate


def batch_lineage(path):
    path = Path(path).resolve()
    seen, batches = set(), []
    while path not in seen:
        seen.add(path)
        batch = json.loads(path.read_text(encoding="utf-8"))
        batches.append((path, batch))
        parent = batch.get("continued_from")
        if not parent:
            return batches
        path = Path(parent["path"]).resolve()
        if digest(path) != parent["sha256"]:
            raise ValueError("Batch lineage changed: "+str(path))
    raise ValueError("Cyclic batch lineage")


def load_points(batch_path):
    lineage = batch_lineage(batch_path)
    if lineage[0][1]["status"] != "completed_hvsa_jobs_only":
        raise ValueError("The named batch must finish before matrix analysis")
    directories = {Path(ROOT)/"results"/"point1_compare"/lineage[0][1]["preserved_e2_n_scan"]}
    for _, batch in lineage:
        for job in batch["jobs"]:
            if job["name"].startswith("e2_") and job["status"] == "completed" and not job["name"].endswith("_pilot"):
                path = Path(job["run_dir"])
                if digest(path/"manifest.json") != job["manifest_sha256"]:
                    raise ValueError("Completed E2 manifest changed")
                directories.add(path)
    points, sources = {}, {}
    for directory in sorted(directories):
        problems = validate(directory)
        if problems:
            raise ValueError("Invalid source: "+str(directory)+": "+"; ".join(problems))
        manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        corr = json.loads((directory/"correctness.json").read_text(encoding="utf-8"))
        if manifest["experiment"] != "e2":
            raise ValueError("Only genuine E2 runs can enter the matrix")
        sources[str(directory)] = {"manifest_sha256": digest(directory/"manifest.json"),
                                  "root_source_sha256": {k:v for k,v in manifest["source_sha256"].items()
                                                        if k in ["user.py", "edge.py", "ta.py", "cloud.py", "utils.py"]}}
        for spec in manifest["settings"]:
            if (spec["variant"], spec["warmup"], spec["rounds"], spec["verification_enabled"], spec["mask_layers"]) != (
                    "hvsa", 3, 10, True, 1):
                raise ValueError("Unqualified E2 setting or pilot cannot be pooled")
            key = (spec["N"], spec["M"], spec["d"])
            if key in points:
                raise ValueError("Duplicate workpoint; select one source explicitly instead of pooling")
            observations = sorted([o for o in corr["observations"] if o["setting"] == spec["setting"]],
                                  key=lambda o:o["round"])
            if len(observations) != 13 or any(not o["valid"] for o in observations):
                raise ValueError("Complete correct workpoint required")
            points[key] = {"source_run": str(directory), "source_setting": spec["setting"],
                           "specification": spec, "observations": observations,
                           "initialization": next(i for i in corr["initialization"]
                                                  if i["specification"]["setting"] == spec["setting"])}
    expected = ({(n, 5, 7850) for n in [20, 50, 100, 200]}
                | {(50, 5, d) for d in [1000, 7850, 10000, 30000, 100000]}
                | {(100, m, 7850) for m in [2, 5, 10]})
    if set(points) != expected:
        raise ValueError("Planned matrix differs: missing="+repr(expected-set(points))+" extra="+repr(set(points)-expected))
    return points, sources, lineage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch")
    parser.add_argument("--output", required=True, help="New figure/analysis directory")
    parser.add_argument("--wait", action="store_true", help="One execution dependency on the named batch")
    args = parser.parse_args(argv)
    if args.wait:
        from .probe_ovp_full import await_terminal_batch
        await_terminal_batch(args.batch)
    points, sources, lineage = load_points(args.batch)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dump_json(output/"analysis.json", {"artifact_kind": "analysis_of_existing_evidence",
        "unique_workpoints": len(points), "points": list(points.values()), "sources": sources,
        "batch_lineage": [{"path": str(p), "sha256": digest(p)} for p, _ in lineage],
        "version_boundary": "The preserved N scan predates canonical-output/domain guard changes. Sources remain distinct.",
        "sample_count_note": "Ten unique points, each 3 warmup + 10 observations. Reusing a point across plots adds no observations."})
    from .plot_results import e2_or_e6
    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    axes = {"N": [(n, 5, 7850) for n in [20, 50, 100, 200]],
            "d": [(50, 5, d) for d in [1000, 7850, 10000, 30000, 100000]],
            "M": [(100, m, 7850) for m in [2, 5, 10]]}
    node = Path("C:/Users/20723/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe")
    marker = Path("C:/Users/20723/.codex/plugins/cache/openai-primary-runtime/pdf/26.1015.12040/skills/pdf/container_tools/mark_artifact_operation_started.mjs")
    # Required PDF authoring marker immediately before the first plot export.
    subprocess.run([str(node), str(marker), "--operation-kind", "create", "--expected-output-count", "6", "--output-format", "pdf"], check=True)
    for axis, keys in axes.items():
        specs, observations = [], []
        for index, key in enumerate(keys):
            point = points[key]
            spec = copy.deepcopy(point["specification"]); spec["setting"] = f"plot_{index}"
            specs.append(spec)
            for original in point["observations"]:
                observation = copy.deepcopy(original); observation["setting"] = spec["setting"]
                observations.append(observation)
        target = output/axis; target.mkdir()
        note = ("Includes archived d=7850 point from an earlier source version" if axis == "d" else
                "Includes archived M=5 point from an earlier source version" if axis == "M" else "")
        e2_or_e6({"experiment": "e2", "settings": specs, "provenance_note": note},
                 {"observations": observations}, target)
    lines = ["# HVSA E2完整规模矩阵", "", "十个唯一工作点，共130轮（30次预热、100次正式观测），均为完整验证。图中的重复工作点不是新增样本。", "",
        "N扫描沿用此前源版本；维度/M补充点使用后来加入输出域检查的版本。完整来源与根文件摘要保留在analysis.json；不将时段和源码差异隐藏。", "",
        "| N | M | d | 客户端 ms/人/轮 | 全部边缘 s/轮 | TA s/轮 | 串行回放 s/轮，中位数[Q25,Q75] | 封装 MiB/轮 | 原运行 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for key, point in sorted(points.items()):
        obs = [o for o in point["observations"] if not o["warmup"]]
        med = lambda values: float(np.median(values))
        q = np.percentile([o["wall_ms"]/1000 for o in obs], [25,50,75])
        n,m,d = key
        lines.append(f"| {n} | {m} | {d} | {med([o['role_ms']['user']/o['n_uploaded'] for o in obs]):.3f} | "
                     f"{med([o['role_ms']['edge']/1000 for o in obs]):.3f} | {med([o['role_ms']['ta']/1000 for o in obs]):.3f} | "
                     f"{q[1]:.3f} [{q[0]:.3f},{q[2]:.3f}] | {med([o['total_sent_bytes']/2**20 for o in obs]):.3f} | {Path(point['source_run']).name} |")
    lines += ["", "各列分别取正式10次观测的中位数；初始化在analysis.json另列。封装含逐接收者广播，不含网络头或估计RTT。当前Python高维哈希主导边缘/TA成本，不等于部署时延。", "",
              "本报告只分析HVSA。GPVO/VLMRS正式原文复现和跨论文排名仍未完成，不宣称SOTA。", "",
              "图表已生成，尚待人工目视检查。", ""]
    (output/"RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(__file__, output/"analysis_source.py")
    from . import plot_results
    shutil.copy2(plot_results.__file__, output/"plot_source.py")
    dump_json(output/"analysis_manifest.json", {"artifact_kind": "analysis_of_existing_evidence",
        "visual_review": "pending", "matplotlib": matplotlib.__version__, "numpy": np.__version__,
        "sources": sources, "output_sha256": {str(p.relative_to(output)): digest(p)
                                               for p in output.rglob("*") if p.is_file()}})
    print("ANALYSIS_COMPLETED="+str(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
