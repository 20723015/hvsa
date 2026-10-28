"""Small, executable Fig. 6-style correctness experiment for the repository protocol.

Uses real MNIST examples for 80 rounds. Full User -> Edge -> Cloud -> TA
verification runs on sampled rounds 35 and 79 only; other rounds use the same
integer quantization followed by ordinary FedAvg. This reduced-scale validation
does not reproduce the paper's 1000-client performance setting or its crypto.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models
from cloud import CloudServer
from datasets import load_data, distribute_data
from edge import EdgeServer
from models import flatten_parameters
from ta import TrustedAuthority
from user import User
from utils import QUANTIZE_SCALE

SEED = 20261013
NUM_ROUNDS = 80
NUM_EDGES = 3
USERS_PER_EDGE = 4
SAMPLES_PER_USER = 24
BATCH_SIZE = 8
LOCAL_EPOCHS = 1
SAMPLE_ROUNDS = (35, 79)  # paper's 1-based epochs
ATTACK_CASES = {
    "aggregate_coordinate_plus_1",
    "omit_last_client",
    "flip_signature_bit",
}
OUT = Path(__file__).resolve().parent


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(OUT / "protocol.log", mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
        force=True,
    )
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    models.DEVICE = torch.device("cpu")

    train_ds, _ = load_data("MNIST")
    loaders = distribute_data(
        train_ds, NUM_EDGES * USERS_PER_EDGE, BATCH_SIZE, iid=True,
        max_samples_per_user=SAMPLES_PER_USER,
    )
    sample_indices = [list(map(int, loader.dataset.indices)) for loader in loaders]

    global_model = models.get_model("MNIST", "Tiny")
    global_flat, model_meta = flatten_parameters(global_model)
    dim = int(global_flat.size)
    ta = TrustedAuthority(gradient_dim=dim, ta_id="TA-FIG6")
    cloud = CloudServer()
    edges = [EdgeServer(f"ES-{i}", cloud, ta, enable_verification=True)
             for i in range(NUM_EDGES)]
    users: list[User] = []
    user_edges: dict[str, EdgeServer] = {}
    ta_pub = ta.get_public_key_bytes()
    for edge_idx, edge in enumerate(edges):
        for j in range(USERS_PER_EDGE):
            uid = f"U-{edge_idx}-{j}"
            local_model = copy.deepcopy(global_model)
            user = User(uid, local_model)
            user.register_with_ta(ta_pub)
            ta.register_user(uid, user.get_public_key_bytes())
            users.append(user)
            user_edges[uid] = edge

    rows: list[dict] = []
    saved: dict[str, np.ndarray] = {}
    total_verify_calls = 0
    total_verify_passes = 0
    total_signed_edge_payloads = 0
    training_start = time.perf_counter()

    for round_idx in range(NUM_ROUNDS):
        round_t = round_idx + 1  # human-readable FL epoch
        round_start = time.perf_counter()
        base_hash = hashlib.sha256(
            np.asarray(global_flat, dtype=np.float64).tobytes()
        ).digest()
        is_secure_sample = round_t in SAMPLE_ROUNDS
        if is_secure_sample:
            cloud.start_new_round(round_idx, dim, expected_edge_servers=NUM_EDGES)
            ta.start_new_round(round_idx)
            for edge in edges:
                edge.start_new_round(round_idx, base_model_hash=base_hash)

        # Train every client from the same verified global model; capture before upload.
        local_float: list[np.ndarray] = []
        local_int: list[np.ndarray] = []
        for i, user in enumerate(users):
            user.set_global_model(global_flat, model_meta)
            user.local_train(loaders[i], LOCAL_EPOCHS, lr=0.015)
            local_flat, _ = flatten_parameters(user.model)
            local_float.append(local_flat.astype(np.float64, copy=True))
            q = (local_flat * QUANTIZE_SCALE).astype(np.int64)
            q64 = (local_flat.astype(np.float64) * QUANTIZE_SCALE).astype(np.int64)
            if not np.array_equal(q, q64):
                raise AssertionError(
                    f"round {round_t} {user.id}: repository dtype quantization differs from float64 check"
                )
            local_int.append(q.copy())
            if is_secure_sample:
                user.mask_and_upload_model(round_idx, user_edges[user.id])

        expected_sum = np.sum(np.stack(local_int, axis=0), axis=0, dtype=np.int64)
        if is_secure_sample:
            for edge in edges:
                edge.trigger_aggregation_and_forward()
            ta.trigger_global_mask_computation(cloud)
            _, secure_sum, signed_hashes = cloud.trigger_final_unmasking_and_averaging()
            total_signed_edge_payloads += len(signed_hashes or [])
            if secure_sum is None or signed_hashes is None:
                raise RuntimeError(f"round {round_t}: Cloud produced no aggregate or proof")
        else:
            # Ordinary integer FedAvg for training rounds outside the two
            # crypto checkpoints. No secure-verification claim is made here.
            secure_sum, signed_hashes = expected_sum, None

        integer_sum_match = bool(np.array_equal(secure_sum, expected_sum))
        if not integer_sum_match:
            raise AssertionError(f"round {round_t}: secure integer sum differs from independently summed inputs")

        pass_by_edge = []
        if is_secure_sample:
            for edge in edges:
                pass_by_edge.append(bool(edge.verify_global_result(secure_sum, signed_hashes)))
            total_verify_calls += len(edges)
            total_verify_passes += sum(pass_by_edge)
            if not all(pass_by_edge):
                raise AssertionError(f"round {round_t}: valid aggregate rejected: {pass_by_edge}")

        secure_mean = secure_sum.astype(np.float64) / QUANTIZE_SCALE / len(users)
        quantized_plain_mean = expected_sum.astype(np.float64) / QUANTIZE_SCALE / len(users)
        raw_float_mean = np.mean(np.stack(local_float, axis=0), axis=0)
        exact_safe_mean = bool(np.array_equal(secure_mean, quantized_plain_mean))
        if not exact_safe_mean:
            raise AssertionError(f"round {round_t}: safe mean differs from quantized plaintext mean")

        if round_t in SAMPLE_ROUNDS:
            key = f"epoch_{round_t}"
            saved[f"local_float_{key}"] = np.stack(local_float, axis=0)
            saved[f"local_quantized_{key}"] = np.stack(local_int, axis=0)
            saved[f"secure_sum_{key}"] = secure_sum.copy()
            saved[f"secure_mean_{key}"] = secure_mean.copy()
            saved[f"raw_float_mean_{key}"] = raw_float_mean.copy()
            saved[f"quantized_plain_mean_{key}"] = quantized_plain_mean.copy()

            # Attack probes use the same production verifier entry point and a fresh copy
            # of the valid proof each time. ES-0 independently checks each alteration.
            tamper_cases: dict[str, tuple[np.ndarray, list[dict]]] = {}
            plus_one = secure_sum.copy()
            plus_one[0] += 1
            tamper_cases["aggregate_coordinate_plus_1"] = (plus_one, signed_hashes)
            omission = secure_sum - local_int[-1]
            tamper_cases["omit_last_client"] = (omission, signed_hashes)
            bad_sig = [dict(item) for item in signed_hashes]
            first_sig = bytearray(bad_sig[0]["signature"])
            first_sig[0] ^= 1
            bad_sig[0]["signature"] = bytes(first_sig)
            tamper_cases["flip_signature_bit"] = (secure_sum, bad_sig)
            for case, (candidate_sum, candidate_proof) in tamper_cases.items():
                results = [bool(edges[0].verify_global_result(candidate_sum, candidate_proof))]
                total_verify_calls += 1
                if any(results):
                    raise AssertionError(f"round {round_t} {case}: modified result accepted: {results}")
                rows.append({
                    "round": round_t,
                    "case": case,
                    "es_checks": len(results),
                    "accepted": sum(results),
                    "rejected": len(results) - sum(results),
                    "max_abs_error_vs_quantized_mean": "",
                    "max_abs_quantization_error_vs_float_mean": "",
                    "integer_sum_matches": "",
                    "secure_mean_matches_quantized_mean": "",
                    "round_wall_seconds": "",
                })

        quant_err = float(np.max(np.abs(secure_mean - raw_float_mean)))
        row = {
            "round": round_t,
            "case": "verified_secure_fedavg" if is_secure_sample else "plain_quantized_fedavg",
            "es_checks": len(pass_by_edge),
            "accepted": sum(pass_by_edge),
            "rejected": len(pass_by_edge) - sum(pass_by_edge),
            "max_abs_error_vs_quantized_mean": float(np.max(np.abs(secure_mean - quantized_plain_mean))),
            "max_abs_quantization_error_vs_float_mean": quant_err,
            "integer_sum_matches": integer_sum_match,
            "secure_mean_matches_quantized_mean": exact_safe_mean,
            "round_wall_seconds": time.perf_counter() - round_start,
        }
        rows.append(row)
        global_flat = secure_mean.copy()
        models.unflatten_parameters(global_model, global_flat, model_meta)
        logging.info(
            "epoch=%d verified_edges=%d/%d exact_int_sum=%s exact_quantized_mean=%s "
            "max_float_quantization_error=%.12g round_seconds=%.3f",
            round_t, sum(pass_by_edge), len(edges), integer_sum_match, exact_safe_mean,
            quant_err, row["round_wall_seconds"],
        )

    wall_total = time.perf_counter() - training_start
    with (OUT / "results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(OUT / "fig6_raw_parameters.npz", **saved)

    # Separate full-protocol results from training-round diagnostics. The 78
    # plain FedAvg rows must never count as verification or attack outcomes.
    normal_rows = [r for r in rows if r["case"] in ("verified_secure_fedavg", "plain_quantized_fedavg")]
    secure_normals = [r for r in rows if r["case"] == "verified_secure_fedavg"]
    attacks = [r for r in rows if r["case"] in ATTACK_CASES]
    assert len(normal_rows) == NUM_ROUNDS
    assert len(secure_normals) == len(SAMPLE_ROUNDS)
    payload = {
        "experiment": "Fig.6-style secure aggregation correctness (repository implementation)",
        "paper_context": {
            "figure": "Fig. 6, Section 7.3, pp. 9-11",
            "paper_setup": "4 clients sampled at epochs 35 and 79; no dropout; server proof checked",
            "paper_plot_is": "parameter-distribution/correctness illustration, not an attack-detection or runtime plot",
        },
        "scope_and_limitations": [
            "Reduced-scale protocol validation, not a reproduction of the paper's 1000-client, five-run evaluation.",
            "Repository protocol (LHH plus Ed25519 edge signatures) is exercised; it is not the paper's pairing-based proof construction.",
            "Tiny-MNIST is used to keep the real end-to-end LHH checks tractable; model size is 7850 parameters.",
            "One local epoch is run per communication round. Full cryptographic aggregation and verification are run only at rounds 35 and 79; the other 78 rounds use ordinary quantized FedAvg.",
            "Each client uses a fixed 24-example IID MNIST shard; 12 total clients are arranged as 3 ES x 4 clients, with no dropout.",
            "The seed fixes data partitioning, model initialization and training order; cryptographic keys are OS-random and are regenerated each run.",
        ],
        "config": {
            "seed": SEED, "dataset": "MNIST", "model": "Tiny_MNIST (one 784x10 linear layer)",
            "model_parameter_count": dim, "num_rounds": NUM_ROUNDS,
            "num_edges": NUM_EDGES, "users_per_edge": USERS_PER_EDGE,
            "total_users": len(users), "samples_per_user": SAMPLES_PER_USER,
            "batch_size": BATCH_SIZE, "local_epochs_per_round": LOCAL_EPOCHS,
            "learning_rate": 0.015, "dropout_rate": 0.0,
            "quantize_scale": QUANTIZE_SCALE, "device": str(models.DEVICE),
            "sample_rounds_1_based": list(SAMPLE_ROUNDS),
            "sample_indices_by_user": sample_indices,
        },
        "runtime": {
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "torch": torch.__version__,
            "wall_seconds_total": wall_total,
        },
        "results": {
            "training_rounds_total": len(normal_rows),
            "normal_edge_verifications": total_verify_passes,
            "normal_edge_verification_attempts": sum(r["es_checks"] for r in secure_normals),
            "secure_protocol_sample_rounds": list(SAMPLE_ROUNDS),
            "unverified_plain_fedavg_rounds": NUM_ROUNDS - len(SAMPLE_ROUNDS),
            "normal_rejections": sum(r["rejected"] for r in secure_normals),
            "secure_integer_sum_matches_on_sample_rounds": sum(bool(r["integer_sum_matches"]) for r in secure_normals),
            "secure_mean_matches_quantized_plaintext_mean_on_sample_rounds": sum(bool(r["secure_mean_matches_quantized_mean"]) for r in secure_normals),
            "plain_quantized_fedavg_training_rounds": sum(r["case"] == "plain_quantized_fedavg" for r in normal_rows),
            "max_abs_float_quantization_error_across_rounds": max(r["max_abs_quantization_error_vs_float_mean"] for r in normal_rows),
            "edge_payloads_signed": total_signed_edge_payloads,
            "attack_cases": len(ATTACK_CASES),
            "attack_probe_count": len(attacks),
            "attack_edge_checks": sum(r["es_checks"] for r in attacks),
            "attack_rejections": sum(r["rejected"] for r in attacks),
            "attack_false_accepts": sum(r["accepted"] for r in attacks),
            "total_verifier_calls_including_attack_probes": total_verify_calls,
        },
    }
    with (OUT / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logging.info("completed: total_wall_seconds=%.3f normal_passes=%d/%d attack_rejections=%d/%d",
                 wall_total, total_verify_passes, len(SAMPLE_ROUNDS) * NUM_EDGES,
                 payload["results"]["attack_rejections"], payload["results"]["attack_edge_checks"])


if __name__ == "__main__":
    main()
