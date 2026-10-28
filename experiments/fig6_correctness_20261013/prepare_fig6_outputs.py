"""Derive reviewer-friendly Fig. 6 data and audit counts from the run artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ATTACK_CASES = {"aggregate_coordinate_plus_1", "omit_last_client", "flip_signature_bit"}
NORMAL_CASES = {"verified_secure_fedavg", "plain_quantized_fedavg"}
WEIGHT_SIZE = 7840  # Tiny_MNIST fc.weight: [10, 784]


def main() -> None:
    with (HERE / "results.json").open(encoding="utf-8") as f:
        summary = json.load(f)
    with (HERE / "results.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    normals = [r for r in rows if r["case"] in NORMAL_CASES]
    attacks = [r for r in rows if r["case"] in ATTACK_CASES]
    secure_normals = [r for r in normals if r["case"] == "verified_secure_fedavg"]
    assert len(normals) == 80 and len(secure_normals) == 2 and len(attacks) == 6
    assert all(r["accepted"] == "0" and r["rejected"] == "1" for r in attacks)
    assert all(r["accepted"] == "3" and r["rejected"] == "0" for r in secure_normals)

    raw = np.load(HERE / "fig6_raw_parameters.npz")
    fc_raw: dict[str, np.ndarray] = {}
    epochs: dict[str, dict] = {}
    histogram_series: list[tuple[str, str, np.ndarray]] = []
    for epoch in (35, 79):
        suffix = f"epoch_{epoch}"
        local = raw[f"local_float_{suffix}"]
        secure = raw[f"secure_mean_{suffix}"]
        quantized_mean = raw[f"quantized_plain_mean_{suffix}"]
        raw_mean = raw[f"raw_float_mean_{suffix}"]
        local_w = local[:, :WEIGHT_SIZE].reshape(local.shape[0], 10, 784)
        secure_w = secure[:WEIGHT_SIZE].reshape(10, 784)
        quantized_w = quantized_mean[:WEIGHT_SIZE].reshape(10, 784)
        raw_mean_w = raw_mean[:WEIGHT_SIZE].reshape(10, 784)

        # Preserve all 12 real clients and a convenient four-client slice for the plot.
        fc_raw[f"local_fc_weight_all_clients_epoch_{epoch}"] = local_w
        fc_raw[f"local_fc_weight_first4_epoch_{epoch}"] = local_w[:4]
        fc_raw[f"secure_fc_weight_mean_epoch_{epoch}"] = secure_w
        fc_raw[f"quantized_plain_fc_weight_mean_epoch_{epoch}"] = quantized_w
        fc_raw[f"raw_float_fc_weight_mean_epoch_{epoch}"] = raw_mean_w
        histogram_series.extend(
            (f"epoch_{epoch}", f"client_{i + 1}", local_w[i].reshape(-1))
            for i in range(4)
        )
        histogram_series.append((f"epoch_{epoch}", "secure_mean", secure_w.reshape(-1)))
        histogram_series.append((f"epoch_{epoch}", "quantized_plain_mean", quantized_w.reshape(-1)))
        max_qerr = float(np.max(np.abs(secure - quantized_mean)))
        max_float_err = float(np.max(np.abs(secure - raw_mean)))
        epochs[str(epoch)] = {
            "clients_in_protocol": int(local.shape[0]),
            "clients_shown_in_plot": 4,
            "plot_client_ids": [f"U-0-{j}" for j in range(4)],
            "fc_weight_shape": [10, 784],
            "secure_mean_equals_quantized_plaintext_mean_elementwise": bool(np.array_equal(secure_w, quantized_w)),
            "max_abs_secure_vs_quantized_plaintext_fc_weight": float(np.max(np.abs(secure_w - quantized_w))),
            "max_abs_quantization_error_full_model_vs_raw_float_mean": max_float_err,
            "max_abs_quantization_error_fc_weight_vs_raw_float_mean": float(np.max(np.abs(secure_w - raw_mean_w))),
            "secure_fc_weight_mean": float(np.mean(secure_w)),
            "secure_fc_weight_std": float(np.std(secure_w)),
            "secure_fc_weight_min": float(np.min(secure_w)),
            "secure_fc_weight_max": float(np.max(secure_w)),
            "all_12_clients_local_fc_weight_mean": float(np.mean(local_w)),
            "all_12_clients_local_fc_weight_std": float(np.std(local_w)),
        }
        assert max_qerr == 0.0

    np.savez_compressed(HERE / "fig6_fc_weight_raw.npz", **fc_raw)

    # One shared set of edges for every displayed series and both epochs.
    values = np.concatenate([values.astype(np.float64, copy=False) for _, _, values in histogram_series])
    bin_edges = np.linspace(float(values.min()), float(values.max()), 61)
    bin_width = float(bin_edges[1] - bin_edges[0])
    with (HERE / "plot_histograms.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["epoch", "series", "bin_index", "bin_left", "bin_right", "count", "density"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch, series, data in histogram_series:
            counts, _ = np.histogram(data, bins=bin_edges)
            densities = counts / (len(data) * bin_width)
            for i, count in enumerate(counts):
                writer.writerow({
                    "epoch": epoch, "series": series, "bin_index": i,
                    "bin_left": bin_edges[i], "bin_right": bin_edges[i + 1],
                    "count": int(count), "density": float(densities[i]),
                })

    plot_json = {
        "description": "Shared-bin fc.weight histogram data derived from the actual protocol run.",
        "plot_layout": "2x2: columns epoch 35/79; top row first 4 local clients; bottom row safe mean and quantized plaintext mean.",
        "shared_bin_edges": bin_edges.tolist(),
        "shared_bin_count": len(bin_edges) - 1,
        "fc_weight_shape": [10, 784],
        "epochs": epochs,
        "source_npz": "fig6_fc_weight_raw.npz",
        "source_histogram_csv": "plot_histograms.csv",
    }
    with (HERE / "plot_data.json").open("w", encoding="utf-8") as f:
        json.dump(plot_json, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Correct the initial summary's selector bug without rewriting protocol.log.
    metrics = summary["results"]
    for stale_key in ("independent_expected_integer_sum_matches",
                      "exact_safe_mean_matches_quantized_plaintext_mean",
                      "attack_types", "attack_probes"):
        metrics.pop(stale_key, None)
    metrics["training_rounds_total"] = len(normals)
    metrics["normal_edge_verifications"] = sum(int(r["accepted"]) for r in secure_normals)
    metrics["normal_edge_verification_attempts"] = sum(int(r["es_checks"]) for r in secure_normals)
    metrics["normal_rejections"] = sum(int(r["rejected"]) for r in secure_normals)
    metrics["secure_protocol_sample_rounds"] = [35, 79]
    metrics["unverified_plain_fedavg_rounds"] = sum(r["case"] == "plain_quantized_fedavg" for r in normals)
    metrics["secure_integer_sum_matches_on_sample_rounds"] = sum(r["integer_sum_matches"] == "True" for r in secure_normals)
    metrics["secure_mean_matches_quantized_plaintext_mean_on_sample_rounds"] = sum(r["secure_mean_matches_quantized_mean"] == "True" for r in secure_normals)
    metrics["plain_quantized_fedavg_training_rounds"] = sum(r["case"] == "plain_quantized_fedavg" for r in normals)
    metrics["max_abs_float_quantization_error_across_rounds"] = max(float(r["max_abs_quantization_error_vs_float_mean"]) for r in normals)
    metrics["attack_cases"] = len(ATTACK_CASES)
    metrics["attack_probe_count"] = len(attacks)
    metrics["attack_edge_checks"] = sum(int(r["es_checks"]) for r in attacks)
    metrics["attack_rejections"] = sum(int(r["rejected"]) for r in attacks)
    metrics["attack_false_accepts"] = sum(int(r["accepted"]) for r in attacks)
    metrics["total_verifier_calls_including_attack_probes"] = (
        metrics["normal_edge_verification_attempts"] + metrics["attack_edge_checks"]
    )
    summary["sampled_fc_weight_metrics"] = epochs
    log_text = (HERE / "protocol.log").read_text(encoding="utf-8", errors="replace")
    stale_log_count = "attack_rejections=6/12" in log_text
    if stale_log_count:
        summary["audit_note"] = (
            "The retained protocol.log completion line used a stale selector and reported attack_rejections=6/12. "
            "The corrected counts below are recomputed from results.csv using only the three explicit attack names. "
            "Actual attack probes: 6 checks, 6 rejected, 0 accepted."
        )
    else:
        summary["audit_note"] = (
            "Counts are recomputed from results.csv using the explicit normal-training, sampled-secure, and attack case names. "
            "Actual attack probes: 6 checks, 6 rejected, 0 accepted."
        )
    with (HERE / "results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with (HERE / "summary_audit.txt").open("w", encoding="utf-8") as f:
        if stale_log_count:
            f.write("The retained initial protocol.log completion line reported attack checks as 12 because its summary selector included the two verified normal rounds.\n")
            f.write("This is a reporting bug only; results.csv rows show the underlying calls.\n")
        else:
            f.write("The run summary selector uses explicit cases; no stale attack count was found in protocol.log.\n")
        f.write("Recomputed explicit attack rows: 3 attack types x 2 sample rounds = 6 checks, 6 rejected, 0 accepted.\n")
        f.write("Normal secure verification: 6 checks, 6 accepted, 0 rejected (3 ES at each of epochs 35 and 79).\n")


if __name__ == "__main__":
    main()
