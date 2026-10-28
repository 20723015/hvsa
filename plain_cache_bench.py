"""Paired plaintext group-level screening of cached update calibration.

Both arms share the existing CloudServer FIFO/expiry selector, one event trace,
data partition, model initialization and each local minibatch RNG state. The
local trainer is unchanged. No cryptographic security or network latency is
measured by this script.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

import models
from cache_calibration import CacheCalibrationOracle
from cloud import CloudServer
from datasets import distribute_data, load_data
from main_point2 import DATASET_CONFIGS

ARMS = ("increment", "calibration")
SOURCE_FILES = ("plain_cache_bench.py", "cache_calibration.py", "models.py",
                "datasets.py", "cloud.py", "main_point2.py", "utils.py",
                "point2_types.py")


def source_digest():
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def rng_snapshot():
    return (torch.random.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    torch.random.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def paired_calls(calls):
    """Run each arm with identical RNG input; advance as one normal call."""
    before = rng_snapshot()
    values = []
    after_first = None
    for call in calls:
        restore_rng(before)
        values.append(call())
        after = rng_snapshot()
        if after_first is None:
            after_first = after
        elif (not torch.equal(after_first[0], after[0]) or
              any(not torch.equal(a, b) for a, b in zip(after_first[1], after[1]))):
            raise RuntimeError("paired calls consumed different RNG streams")
    restore_rng(after_first)
    return values


def run(args):
    started = time.perf_counter()
    code_hash = source_digest()
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = dict(DATASET_CONFIGS[args.dataset])
    if args.model:
        config["model"] = args.model
    if args.local_epochs is not None:
        config["local_epochs"] = args.local_epochs
    if args.samples_per_user is not None:
        config["samples_per_user"] = args.samples_per_user
    scientific = {k: v for k, v in vars(args).items() if k != "output_dir"}
    scientific.update(training_config=config, code_hash=code_hash,
                      deterministic=True, encoding="float64", protocol="plaintext_oracle")
    config_hash = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    meta_path = output / "pair_meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("config_hash") == config_hash and old.get("status") == "complete":
            print("skip complete pair with matching configuration", flush=True)
            return
        raise SystemExit("output already exists with another or incomplete configuration")

    train_ds, test_ds = load_data(args.dataset)
    total_users = args.num_edge_servers * args.users_per_edge
    loaders = distribute_data(train_ds, total_users, config["batch_size"],
                              iid=args.iid, max_samples_per_user=config["samples_per_user"],
                              dirichlet_alpha=args.alpha)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    initial_model = models.get_model(args.dataset, config["model"])
    initial_flat, model_meta = models.flatten_parameters(initial_model)
    weights = {f"U-{i}": len(loader.dataset) for i, loader in enumerate(loaders)}
    partition_hash = hashlib.sha256()
    for loader in loaders:
        indices = np.asarray(loader.dataset.indices, dtype=np.int64)
        partition_hash.update(len(indices).to_bytes(8, "little"))
        partition_hash.update(indices.tobytes())
    states = {arm: CacheCalibrationOracle(initial_flat, weights) for arm in ARMS}
    histories = {arm: {0: states[arm].model.copy()} for arm in ARMS}
    local_models = {arm: [copy.deepcopy(initial_model) for _ in loaders] for arm in ARMS}
    eval_models = {arm: copy.deepcopy(initial_model) for arm in ARMS}
    cloud = CloudServer()
    cloud.point2_reset(initial_flat)
    rng = np.random.default_rng(args.seed)
    edges = [f"edge_{i}" for i in range(args.num_edge_servers)]
    delays = {eid: int(rng.integers(args.min_delay, args.max_delay + 1)) for eid in edges}
    active = {}
    sequences = dict.fromkeys(edges, 0)
    trace = []
    records = {arm: [] for arm in ARMS}
    train_seconds = dict.fromkeys(ARMS, 0.0)
    evaluate_seconds = dict.fromkeys(ARMS, 0.0)
    counts = dict(executed_updates=0, executed_sgd_steps=0, executed_samples=0,
                  adopted_updates=0, adopted_sgd_steps=0, adopted_samples=0,
                  expired_packages=0, unsealed_packages=0, upload_lost_updates=0)
    package_work = {}
    vcm_max_abs_error = 0.0

    for tick in range(1, args.max_ticks + 1):
        for eid in edges:
            if eid not in active and not cloud.p2_has_pending_package(eid):
                token = f"{eid}:v{cloud.p2_current_version}:p{sequences[eid]}"
                sequences[eid] += 1
                active[eid] = (token, cloud.p2_current_version, tick + delays[eid])
                trace.append(("start", tick, token, cloud.p2_current_version))
        for edge_index, eid in enumerate(edges):
            if eid not in active or tick < active[eid][2]:
                continue
            token, base_version, _ = active.pop(eid)
            updates = {arm: {} for arm in ARMS}
            package_steps = package_samples = 0
            for i in range(edge_index * args.users_per_edge, (edge_index + 1) * args.users_per_edge):
                uid = f"U-{i}"
                dropped = bool(rng.random() < args.dropout_rate)
                trace.append(("drop", tick, token, uid, dropped))
                if dropped:
                    continue
                def train(arm):
                    model = local_models[arm][i]
                    base = histories[arm][base_version]
                    models.unflatten_parameters(model, base, model_meta)
                    t0 = time.perf_counter()
                    models.train_local(model, loaders[i], config["local_epochs"], config["lr"])
                    train_seconds[arm] += time.perf_counter() - t0
                    end, _ = models.flatten_parameters(model)
                    return end.astype(np.float64) - base
                deltas = paired_calls([lambda arm=arm: train(arm) for arm in ARMS])
                steps = config["local_epochs"] * len(loaders[i])
                samples = config["local_epochs"] * weights[uid]
                counts["executed_updates"] += 1
                counts["executed_sgd_steps"] += steps
                counts["executed_samples"] += samples
                lost = bool(rng.random() < args.upload_dropout_rate)
                trace.append(("upload_lost", tick, token, uid, lost))
                if lost:
                    counts["upload_lost_updates"] += 1
                    continue
                for arm, delta in zip(ARMS, deltas):
                    updates[arm][uid] = delta
                package_steps += steps
                package_samples += samples
            online = updates["increment"]
            if len(online) < args.n_min:
                counts["unsealed_packages"] += 1
                continue
            for arm in ARMS:
                states[arm].stage(token, base_version, updates[arm])
            package_work[token] = (len(online), package_steps, package_samples)
            package = SimpleNamespace(edge_id=eid, package_token=token,
                                      base_version=base_version, created_tick=tick,
                                      n_k=len(online), S_k=sum(weights[u] for u in online))
            cloud.p2_receive_group_package(package, arrival_tick=tick)
        old_tokens = {p.package_token for p in cloud.p2_package_buffer.values()}
        selected = cloud.p2_select_checkpoint(args.K, args.s_max, tick, policy="fifo")
        expired = old_tokens - {p.package_token for p in cloud.p2_package_buffer.values()}
        for token in expired:
            for arm in ARMS:
                states[arm].discard(token)
            package_work.pop(token)
            counts["expired_packages"] += 1
        if not selected:
            continue
        tokens = [p.package_token for p in selected]
        trace.append(("commit", tick, cloud.p2_current_version, tokens))
        proposals = {arm: states[arm].prepare(tokens, arm) for arm in ARMS}
        # Decisive algebra check: same endpoints/bases/weights, gamma=1 VCM.
        raw = np.zeros_like(initial_flat, dtype=np.float64)
        compensation = np.zeros_like(raw)
        mass = sum(p.S_k for p in selected)
        incremental = states["increment"]
        for package in selected:
            base = histories["increment"][package.base_version]
            pending = incremental.pending[package.package_token]
            for uid, delta in pending.deltas.items():
                raw += weights[uid] * (base + delta)
            compensation += package.S_k * (incremental.model - base)
        aligned = (raw + compensation) / mass
        error = float(np.max(np.abs(aligned - proposals["increment"].model)))
        vcm_max_abs_error = max(vcm_max_abs_error, error)
        if not np.allclose(aligned, proposals["increment"].model, atol=1e-12, rtol=1e-10):
            raise RuntimeError("explicit increment and default VCM differ")
        for token in tokens:
            updates_count, steps, samples = package_work.pop(token)
            counts["adopted_updates"] += updates_count
            counts["adopted_sgd_steps"] += steps
            counts["adopted_samples"] += samples
        receipts = {arm: states[arm].commit(proposals[arm]) for arm in ARMS}
        cloud.p2_current_version = states["increment"].version
        for package in selected:
            cloud.p2_package_buffer.pop(package.edge_id)
            cloud.p2_aggregated_ids.add(package.package_token)
        for arm in ARMS:
            histories[arm][cloud.p2_current_version] = states[arm].model.copy()
        def evaluate(arm):
            models.unflatten_parameters(eval_models[arm], states[arm].model, model_meta)
            t0 = time.perf_counter()
            result = models.evaluate(eval_models[arm], test_loader)
            evaluate_seconds[arm] += time.perf_counter() - t0
            return result
        scores = paired_calls([lambda arm=arm: evaluate(arm) for arm in ARMS])
        for arm, (accuracy, loss) in zip(ARMS, scores):
            row = dict(checkpoint=cloud.p2_current_version, tick=tick,
                       accuracy=accuracy, loss=loss, **counts,
                       **proposals[arm].diagnostics, **receipts[arm])
            row["tokens"] = tokens
            records[arm].append(row)
        print(f"cp={cloud.p2_current_version} tick={tick} "
              f"increment={scores[0][0]:.2f} calibration={scores[1][0]:.2f} "
              f"delta={scores[1][0]-scores[0][0]:+.2f}pp "
              f"age={proposals['calibration'].diagnostics['cache_age_mean']:.2f}", flush=True)
        if cloud.p2_current_version >= args.max_checkpoints:
            break
    if cloud.p2_current_version != args.max_checkpoints:
        raise RuntimeError("tick budget exhausted before requested checkpoints")
    if source_digest() != code_hash:
        raise RuntimeError("source files changed during the experiment")
    trace_hash = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
    (output / "event_trace.json").write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
    summary = {}
    for arm in ARMS:
        rows = records[arm]
        (output / f"{arm}_checkpoints.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        numeric = {key: np.asarray([row[key] for row in rows])
                   for key, value in rows[0].items() if isinstance(value, (int, float))}
        np.savez_compressed(output / f"{arm}.npz", **numeric)
        summary[arm] = dict(final_accuracy=rows[-1]["accuracy"],
                            last5_accuracy=float(np.mean([r["accuracy"] for r in rows[-5:]])),
                            training_seconds=train_seconds[arm],
                            evaluation_seconds=evaluate_seconds[arm],
                            max_invariant_error=max(r["invariant_max_abs_error"] for r in rows))
    metadata = dict(status="complete", config_hash=config_hash, scientific_config=scientific,
                    partition_hash=partition_hash.hexdigest(), event_trace_hash=trace_hash,
                    initial_model_hash=hashlib.sha256(initial_flat.tobytes()).hexdigest(),
                    shared_trace=True, paired_minibatch_rng=True,
                    edge_delays=delays, sample_sizes=weights, summary=summary,
                    max_vcm_equivalence_error=vcm_max_abs_error,
                    pair_wall_seconds=time.perf_counter() - started,
                    total_counts=counts, device=str(models.DEVICE),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    torch_version=torch.__version__,
                    limitations=["plaintext oracle; no security or network overhead",
                                 "sample-weighted candidate; no unbiasedness or convergence claim",
                                 "test accuracies are exploratory, not validation-tuned results"])
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASET_CONFIGS, default="FashionMNIST")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=.5)
    p.add_argument("--iid", action="store_true")
    p.add_argument("--model", choices=["CNN", "MLP", "Tiny"])
    p.add_argument("--local-epochs", type=int)
    p.add_argument("--samples-per-user", type=int)
    p.add_argument("--num-edge-servers", type=int, default=5)
    p.add_argument("--users-per-edge", type=int, default=4)
    p.add_argument("-K", type=int, default=3)
    p.add_argument("--s-max", type=int, default=3)
    p.add_argument("--n-min", type=int, default=2)
    p.add_argument("--min-delay", type=int, default=1)
    p.add_argument("--max-delay", type=int, default=5)
    p.add_argument("--dropout-rate", type=float, default=.1)
    p.add_argument("--upload-dropout-rate", type=float, default=0.0)
    p.add_argument("--max-checkpoints", type=int, default=20)
    p.add_argument("--max-ticks", type=int, default=300)
    p.add_argument("--output-dir", required=True)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    if (not 1 <= arguments.K <= arguments.num_edge_servers or
            not 1 <= arguments.n_min <= arguments.users_per_edge or
            not 0 <= arguments.dropout_rate < 1 or
            not 0 <= arguments.upload_dropout_rate < 1 or
            arguments.s_max < 0 or arguments.alpha <= 0 or
            arguments.min_delay <= 0 or arguments.max_delay < arguments.min_delay or
            arguments.max_checkpoints < 1 or arguments.max_ticks < 1 or
            (arguments.local_epochs is not None and arguments.local_epochs < 1) or
            (arguments.samples_per_user is not None and arguments.samples_per_user < 10)):
        raise SystemExit("invalid experimental configuration")
    logging.basicConfig(level=logging.WARNING)
    run(arguments)
