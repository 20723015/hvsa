"""Rebuild figures/tables from a validated completed run; never extrapolate.

Usage: python -m experiments.point1_compare.plot_results RUN_DIR
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .artifacts import digest, dump_json
from .run_compare import ROOT
from .validate_results import validate

ROLES = ["user", "edge", "ta", "cloud"]
ROLE_LABELS = {"user": "All clients", "edge": "All edges", "ta": "TA", "cloud": "Cloud"}
VARIANTS = {"hvsa": "HVSA, full verification", "hvsa_three_mask_control": "3-mask control",
            "hvsa_verification_off": "Verification off (cost control)"}
SHORT_VARIANTS = {"hvsa": "HVSA (full)", "hvsa_three_mask_control": "3-mask control",
                  "hvsa_verification_off": "Edge checks off"}


def interval(values):
    q = np.percentile(np.asarray(values, dtype=float), [25, 50, 75])
    return q[1], np.array([[q[1]-q[0]], [q[2]-q[1]]])


def save(fig, output, name):
    fig.tight_layout()
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(output / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def e2_or_e6(manifest, corr, output):
    observations = [o for o in corr["observations"] if not o["warmup"]]
    settings = manifest["settings"]
    varying = [k for k in ["N", "M", "d"] if len({s[k] for s in settings}) > 1]
    axis = varying[0] if len(varying) == 1 and manifest["experiment"] != "e6" else None
    fixed = ", ".join(f"{k}={settings[0][k]}" for k in ["N", "M", "d"]
                      if k != axis and len({s[k] for s in settings}) == 1)
    short_names = dict(SHORT_VARIANTS)
    if manifest.get("verification_off_scope") == "full_per_round_including_ta_mask_hash":
        short_names["hvsa_verification_off"] = "All checks off"
    labels = ([str(s[axis]) for s in settings] if axis else
              [f"{short_names.get(s['variant'], s['variant'])}\nN={s['N']}, M={s['M']}\nd={s['d']}"
               for s in settings])
    x = np.arange(len(settings))
    width = max(10, len(settings)*1.8) if manifest["experiment"] == "e6" else max(8, len(settings)*1.2)
    fig, axes = plt.subplots(1, 2, figsize=(width, 4))
    for i, s in enumerate(settings):
        obs = [o for o in observations if o["setting"] == s["setting"]]
        # First aggregate each round's complete client work, then divide by the
        # actual client count; never mix per-client and per-round units.
        client = [o["role_ms"]["user"]/o["n_uploaded"] for o in obs]
        value, error = interval(client)
        axes[0].errorbar(i, value, yerr=error, fmt="o", capsize=4, color="C0")
        for role in ROLES:
            value, error = interval([o["role_ms"][role] for o in obs])
            axes[1].errorbar(i, value, yerr=error, fmt="o", capsize=4,
                             label=ROLE_LABELS[role] if i == 0 else None,
                             color={"user": "C0", "edge": "C1", "ta": "C2", "cloud": "C3"}[role])
    for ax in axes:
        ax.set_xticks(x, labels, rotation=0, ha="center", fontsize=8)
        ax.set_xlabel(axis or "Measured setting")
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Mean client work (ms/client/round)")
    axes[1].set_ylabel("Role work (ms/round, log scale)")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    provenance = manifest.get("provenance_note", "")
    suffix = "\n"+provenance if provenance else ""
    fig.suptitle(f"Local serial replay ({fixed}); median and interquartile range\nInitialization separate"+suffix, fontsize=9)
    save(fig, output, "compute_cost")
    links = sorted({link for o in observations for link in o["link_bytes"]})
    fig, ax = plt.subplots(figsize=(max(7, len(settings)*1.2), 4))
    bottom = np.zeros(len(settings))
    for link in links:
        heights = np.array([np.median([o["link_bytes"].get(link, 0) for o in observations
                                       if o["setting"] == s["setting"]])/1024 for s in settings])
        ax.bar(x, heights, bottom=bottom, label=link)
        bottom += heights
    ax.set_xticks(x, labels, rotation=0, ha="center", fontsize=8)
    ax.set_xlabel(axis or "Measured setting")
    ax.set_ylabel("Serialized envelope bytes per round (KiB)")
    ax.set_title(f"{fixed}\nPer-link median bytes; transport headers excluded"+suffix, fontsize=9)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    save(fig, output, "communication")


def e3(manifest, corr, output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for s in manifest["settings"]:
        obs = sorted([o for o in corr["observations"] if o["setting"] == s["setting"]], key=lambda o:o["round"])
        init = next(i["init_ms"] for i in corr["initialization"] if i["specification"]["setting"] == s["setting"])
        total = init + np.cumsum([sum(o["role_ms"].values()) for o in obs])
        rounds = np.arange(1, len(obs)+1)
        counts = sorted({o["n_uploaded"] for o in obs})
        count_label = str(counts[0]) if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"
        label = f"{s['participation']}: {count_label} uploads"
        axes[0].plot(rounds, total/1000, label=label)
        axes[1].plot(rounds, total/rounds/1000, label=label)
    axes[0].set_ylabel("Cumulative measured work including init (s)")
    axes[1].set_ylabel("Amortized measured work per round (s)")
    for ax in axes:
        ax.set_xlabel("Actual completed round number")
        ax.grid(alpha=.25)
        ax.legend()
    axes[1].get_legend().remove()  # shared colors; keep the complete trajectory visible
    spec = manifest["settings"][0]
    fig.suptitle(f"HVSA serial execution: {spec['N']} registered clients, M={spec['M']}, d={spec['d']}", fontsize=9)
    save(fig, output, "multi_round")


def e4(manifest, corr, output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    stages = sorted({s["dropout_stage"] for s in manifest["settings"]})
    all_rates = sorted({s["dropout_rate"]*100 for s in manifest["settings"]})
    for stage_index, stage in enumerate(stages):
        specs = sorted([s for s in manifest["settings"] if s["dropout_stage"] == stage], key=lambda s:s["dropout_rate"])
        rates, success, times, lower, upper = [], [], [], [], []
        for s in specs:
            obs = [o for o in corr["observations"] if o["setting"] == s["setting"]]
            completed = [o["wall_ms"] for o in obs if o["valid"]]
            rates.append(s["dropout_rate"]*100)
            success.append(sum(o["valid"] for o in obs)/len(obs))
            times.append(np.median(completed)/1000 if completed else np.nan)
            lower.append(np.percentile(completed, 25)/1000 if completed else np.nan)
            upper.append(np.percentile(completed, 75)/1000 if completed else np.nan)
        centers = np.array([all_rates.index(rate) for rate in rates])
        width = .8/len(stages)
        offset = (stage_index-(len(stages)-1)/2)*width
        axes[0].bar(centers+offset, success, width=width, label=stage)
        axes[1].errorbar(rates, times, yerr=[np.array(times)-lower, np.array(upper)-times],
                         fmt="o-", capsize=3, label=stage)
    axes[0].set_xticks(range(len(all_rates)), [f"{r:g}" for r in all_rates])
    axes[0].set_ylabel("Correct, verified completion fraction")
    axes[0].set_ylim(0, 1.08)
    axes[1].set_ylabel("Successful local replay time, median (s)")
    for ax in axes:
        ax.set_xlabel("Unavailable clients (%)")
        ax.legend()
        ax.grid(alpha=.25)
    axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, fontsize=8)
    spec = manifest["settings"][0]
    fig.suptitle(f"HVSA: N={spec['N']} selected, M={spec['M']}, d={spec['d']}; {spec['rounds']} traces/point; time median and IQR", fontsize=9)
    save(fig, output, "dropout")


def e7(manifest, corr, output):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for seed in manifest["training_seeds"]:
        obs = sorted([o for o in corr["observations"] if o["seed"] == seed and o["valid"]], key=lambda o:o["round"])
        ax.plot([o["round"]+1 for o in obs], [o["test_accuracy"]*100 for o in obs], label=f"seed {seed}")
    ax.set_xlabel("Actual training round")
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.set_title("HVSA full verification; identical same-encoding plaintext reference", fontsize=9)
    ax.legend()
    ax.grid(alpha=.25)
    save(fig, output, "training")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--output", help="New output directory; existing artifacts are preserved")
    args = parser.parse_args(argv)
    directory = Path(args.run_dir).resolve()
    problems = validate(directory)
    if problems:
        raise ValueError("Refusing to plot incomplete/invalid evidence: " + "; ".join(problems))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    corr = json.loads((directory / "correctness.json").read_text(encoding="utf-8"))
    output = Path(args.output) if args.output else Path(ROOT) / "figures" / "point1_compare" / manifest["run_id"]
    if output.exists() and not args.output:
        from .run_compare import _new_run_id
        output = output.with_name(output.name + "_" + _new_run_id("plots"))
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    experiment = manifest["experiment"]
    if experiment in ("e2", "e6"):
        e2_or_e6(manifest, corr, output)
    elif experiment in ("e3", "e4", "e7"):
        {"e3": e3, "e4": e4, "e7": e7}[experiment](manifest, corr, output)
    with (directory / "summary.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    lines = [f"# {experiment.upper()} measured results", "",
             "Source: `"+str(directory)+"`", "",
             "Artifact validation establishes consistency, not security or cross-paper superiority.", ""]
    if manifest.get("artifact_kind") == "derived_complete_point":
        lines += ["This is a completed workpoint projected from archived checkpoints, not a new protocol execution.",
                  "Original run: `"+manifest["derivation"]["origin_run_id"]+"`; status at export: `"+
                  manifest["derivation"]["origin_execution_status"]+"`. The full matrix is not established by this view.", ""]
    if manifest.get("settings"):
        lines += ["| Setting | Variant | N | M | d | Warmup | Measured rounds |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        lines += [f"| {s['setting']} | {s['variant']} | {s['N']} | {s['M']} | {s['d']} | {s['warmup']} | {s['rounds']} |"
                  for s in manifest["settings"]]
        lines.append("")
    lines += ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines += ["| " + " | ".join(str(row[k]) for k in fields) + " |" for row in rows]
    (output / "RESULTS_SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    shutil.copy2(__file__, output / "plot_source.py")
    figure_manifest = {"source_run": str(directory), "source_manifest_sha256": digest(directory / "manifest.json"),
                       "plot_source_sha256": digest(Path(__file__)), "matplotlib": matplotlib.__version__,
                       "numpy": np.__version__, "measured_only": True,
                       "output_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    if manifest.get("artifact_kind") == "derived_complete_point":
        figure_manifest["derivation"] = manifest["derivation"]
    dump_json(output / "figure_manifest.json", figure_manifest)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
