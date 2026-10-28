"""Join validated E6 evidence without rewriting runs or pooling control scopes.

The original three controls were interleaved. The corrected full-off control
was executed later on the same input sequence. This is analysis, not another
protocol run; every summarized observation retains its original identity.
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .artifacts import digest, dump_json
from .validate_results import validate
from .plot_results import save

FULL_OFF = "full_per_round_including_ta_mask_hash"
NAMES = {"hvsa": "HVSA (full)", "hvsa_three_mask_control": "3-mask control",
         "edge_off": "Edge checks off", "full_off": "All checks off"}


def load_controls(old_run, off_run):
    controls = []
    for path, corrected in [(Path(old_run).resolve(), False), (Path(off_run).resolve(), True)]:
        problems = validate(path)
        if problems:
            raise ValueError("Invalid source evidence: " + "; ".join(problems))
        manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
        corr = json.loads((path/"correctness.json").read_text(encoding="utf-8"))
        if manifest["experiment"] != "e6" or (manifest.get("verification_off_scope") == FULL_OFF) != corrected:
            raise ValueError("Expected separate original E6 and corrected full-off E6")
        for spec in manifest["settings"]:
            if (spec["N"], spec["M"], spec["d"], spec["warmup"], spec["rounds"]) not in (
                    (50, 5, 7850, 3, 10), (50, 5, 30000, 3, 10)):
                raise ValueError("Unexpected ablation specification")
            name = spec["variant"]
            if name == "hvsa_verification_off":
                name = "full_off" if corrected else "edge_off"
            elif corrected:
                raise ValueError("Corrected supplementary run must contain only the full-off control")
            if name not in NAMES:
                raise ValueError("Unknown control")
            obs = sorted([o for o in corr["observations"] if o["setting"] == spec["setting"]],
                         key=lambda o:o["round"])
            if len(obs) != 13 or any(not o["valid"] or not o["exact_match"] for o in obs):
                raise ValueError("Incomplete or failed control")
            if any(o["verified"] is not (True if spec["verification_enabled"] else None) for o in obs):
                raise ValueError("Incorrect verification scope")
            controls.append({"control": name, "d": spec["d"], "run_dir": str(path),
                             "manifest_sha256": digest(path/"manifest.json"),
                             "setting": spec["setting"], "observations": obs,
                             "initialization": next(i for i in corr["initialization"]
                                                    if i["specification"]["setting"] == spec["setting"])})
    expected = {(d, name) for d in [7850, 30000] for name in NAMES}
    if {(c["d"], c["control"]) for c in controls} != expected or len(controls) != 8:
        raise ValueError("Missing or duplicate E6 control")
    for d in [7850, 30000]:
        inputs = [[o["input_sha256"] for o in c["observations"]] for c in controls if c["d"] == d]
        if any(sequence != inputs[0] for sequence in inputs):
            raise ValueError("Control input sequences differ")
    return controls


def stats(values):
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {"median": float(median), "q25": float(q25), "q75": float(q75), "count": len(values)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_run")
    parser.add_argument("full_off_run")
    parser.add_argument("--output", required=True, help="New analysis directory")
    args = parser.parse_args(argv)
    controls = load_controls(args.original_run, args.full_off_run)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for c in controls:
        obs = [o for o in c["observations"] if not o["warmup"]]
        metrics = {"client_ms": [o["role_ms"]["user"]/o["n_uploaded"] for o in obs],
                   "wall_ms": [o["wall_ms"] for o in obs],
                   "sent_bytes": [o["total_sent_bytes"] for o in obs]}
        metrics.update({r+"_ms": [o["role_ms"][r] for o in obs] for r in ["user", "edge", "ta", "cloud"]})
        c["statistics"] = {metric: stats(values) for metric, values in metrics.items()}
        rows.extend({"d": c["d"], "control": c["control"], "source_run": Path(c["run_dir"]).name,
                     "setting": c["setting"], "metric": metric, **stat}
                    for metric, stat in c["statistics"].items())
    with (output/"summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    dump_json(output/"analysis.json", {"artifact_kind": "analysis_of_existing_evidence", "controls": controls,
              "limitations": ["No new protocol execution or independent samples in this artifact.",
                  "Original three controls interleaved; corrected full-off control collected later.",
                  "Per-role totals describe local serial replay, not deployed latency.",
                  "Three-mask control adds two DH and two PRG operations; it is not GPVO."]})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    for ax, d in zip(axes, [7850, 30000]):
        for i, name in enumerate(["hvsa", "hvsa_three_mask_control"]):
            c = next(c for c in controls if c["d"] == d and c["control"] == name)
            s = c["statistics"]["client_ms"]
            ax.errorbar(i, s["median"], yerr=[[s["median"]-s["q25"]], [s["q75"]-s["median"]]],
                        fmt="o", capsize=5, color="C0")
        ax.set_xticks([0, 1], [NAMES[n] for n in ["hvsa", "hvsa_three_mask_control"]])
        ax.set_xlim(-.5, 1.5); ax.set_title(f"d={d}, N=50, M=5")
        ax.set_ylabel("Mean client work (ms/client/round)"); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Mask ablation: 10 measured rounds per control; median and IQR", fontsize=10)
    save(fig, output, "mask_ablation")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for ax, d in zip(axes, [7850, 30000]):
        for i, name in enumerate(["hvsa", "edge_off", "full_off"]):
            c = next(c for c in controls if c["d"] == d and c["control"] == name)
            s = c["statistics"]["wall_ms"]
            ax.errorbar(i, s["median"]/1000,
                        yerr=[[(s["median"]-s["q25"])/1000], [(s["q75"]-s["median"])/1000]],
                        fmt="o", capsize=5, color="C1")
        ax.set_xticks([0, 1, 2], [NAMES[n] for n in ["hvsa", "edge_off", "full_off"]], fontsize=8)
        ax.set_xlim(-.5, 2.5); ax.set_yscale("log"); ax.grid(axis="y", alpha=.25)
        ax.set_title(f"d={d}, N=50, M=5"); ax.set_ylabel("Local serial replay time (s, log scale)")
    fig.suptitle("Verification cost controls; all-checks-off observed in a separate later run", fontsize=10)
    save(fig, output, "verification_ablation")
    shutil.copy2(__file__, output/"analysis_source.py")
    manifest = {"artifact_kind": "analysis_of_existing_evidence",
                "sources": {c["run_dir"]: c["manifest_sha256"] for c in controls},
                "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                "source_sha256": {str(Path(__file__).resolve()): digest(Path(__file__))},
                "output_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    dump_json(output/"analysis_manifest.json", manifest)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
