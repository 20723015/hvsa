"""E7: MNIST 7850-parameter linear model, real local training and full HVSA.

The same local updates are compared against an integer plaintext reference at
every round. Snapshots can later be replayed through qualified baselines. The
float comparison is the mean of the same updates, not an independent FedAvg run.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from .adapters.common import RoundResult, reference_sum
from .adapters.hvsa_point1 import HVSAProtocol
from .artifacts import create_run, digest, dump_json, finalize_run, message_rows, round_rows
from .run_compare import ROOT, _git_commit, _lib_versions, _new_run_id, _review_metadata
from .run_integrity import vector_digest


def set_parameters(model, vector):
    torch.nn.utils.vector_to_parameters(torch.tensor(vector, dtype=torch.float32), model.parameters())


def accuracy(model, dataset, batch_size=512):
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            predictions = model(images.flatten(1)).argmax(1)
            correct += int((predictions == labels).sum())
            total += labels.numel()
    return correct / total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--edges", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--data-root", default=str(Path(ROOT) / "data"))
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    if (len(set(seeds)) != len(seeds) or args.clients < 2*args.edges or args.rounds < 1
            or args.local_epochs < 1 or args.batch_size < 1 or args.lr <= 0):
        parser.error("Invalid training parameters")
    torch.set_num_threads(1)
    logging.disable(logging.CRITICAL)
    training = MNIST(args.data_root, train=True, download=False, transform=ToTensor())
    testing = MNIST(args.data_root, train=False, download=False, transform=ToTensor())
    dim = sum(p.numel() for p in nn.Linear(784, 10).parameters())
    run_id = _new_run_id("e7_hvsa_v2")
    data_hashes = {p.relative_to(Path(args.data_root)).as_posix(): digest(p)
                   for p in (Path(args.data_root) / "MNIST" / "raw").glob("*-ubyte")}
    manifest = {"run_id": run_id, "experiment": "e7", "command": " ".join(sys.argv),
                "seed": seeds[0], "training_seeds": seeds, "schemes": ["hvsa"],
                "git_commit": _git_commit(), "lib_versions": {**_lib_versions(), "torch": torch.__version__},
                "quant": {"scale": 1000000, "rule": "truncate to int64, equal client weight"},
                "protocol_mapping": "experiments/point1_compare/protocol_mapping.md",
                "paper_versions": {"hvsa": "real point-1 entities"}, "concurrency": "serial CPU, torch threads=1",
                "timing_scope": "training separate from local protocol replay", "verification": "full",
                "aggregation": "integer model sum then edge average", "model": "Linear(784,10)",
                "d": dim, "N": args.clients, "M": args.edges, "rounds": args.rounds,
                "dataset_sha256": data_hashes, "execution_status": "running"}
    manifest.update(_review_metadata(vars(args)))
    manifest["measurement_limitations"] = ["HVSA only; unresolved baselines not ranked.",
        "Float mean uses the same local updates; no independent float-FedAvg convergence claim.",
        "CPU serial replay, not deployed end-to-end latency."]
    directory = Path(ROOT) / "results" / "point1_compare" / run_id
    create_run(directory, manifest, Path(ROOT))
    (directory / "snapshots").mkdir()
    (directory / "checkpoints").mkdir()
    print(f"RUN_DIR={directory} d={dim}", flush=True)
    raw, messages, observations, failures, summaries = [], [], [], [], []
    for seed in seeds:
        torch.manual_seed(seed)
        model = nn.Linear(784, 10)
        rng = np.random.default_rng(seed)
        partitions = np.array_split(rng.permutation(len(training)), args.clients)
        groups = [[f"u{i}" for i in ids] for ids in np.array_split(np.arange(args.clients), args.edges)]
        state = HVSAProtocol(dim, groups)
        online = set(state.users)
        setup = RoundResult("hvsa", True, None, None, args.clients, 0, state.init_phases, state.init_messages)
        raw.extend(round_rows(setup, scheme="hvsa", run_id=run_id, seed=seed, group=f"seed_{seed}",
                              dimension=dim, m=args.edges, round_t=-1))
        messages.extend(message_rows(setup, scheme="hvsa", run_id=run_id, group=f"seed_{seed}", round_t=-1))
        seed_rows = []
        for round_t in range(args.rounds):
            tick = time.perf_counter()
            base = torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy().copy()
            vectors, floats = {}, []
            for i in range(args.clients):
                local = copy.deepcopy(model)
                local.train()
                optimizer = torch.optim.SGD(local.parameters(), lr=args.lr)
                generator = torch.Generator().manual_seed(seed*1000000+round_t*args.clients+i)
                loader = DataLoader(Subset(training, partitions[i].tolist()), batch_size=args.batch_size,
                                    shuffle=True, generator=generator)
                for _ in range(args.local_epochs):
                    for images, labels in loader:
                        optimizer.zero_grad()
                        nn.functional.cross_entropy(local(images.flatten(1)), labels).backward()
                        optimizer.step()
                flat = torch.nn.utils.parameters_to_vector(local.parameters()).detach().numpy().copy()
                floats.append(flat.astype(np.float64))
                vectors[f"u{i}"] = (flat*1000000).astype(np.int64)
            training_ms = (time.perf_counter()-tick)*1000
            reference = reference_sum([vectors[u] for u in sorted(online)])
            reference_model = reference.astype(np.float64) / 1000000 / args.clients
            snapshot = directory / "snapshots" / f"seed_{seed}_round_{round_t:03d}.npz"
            np.savez_compressed(snapshot, vectors=np.stack([vectors[f"u{i}"] for i in range(args.clients)]),
                                base_model=base, user_ids=np.array([f"u{i}" for i in range(args.clients)]))
            tick = time.perf_counter()
            result = state.run_round(vectors, online, round_t, base_model_vec=base)
            protocol_ms = (time.perf_counter()-tick)*1000
            exact = bool(result.completed and result.raw_sum is not None and np.array_equal(result.raw_sum, reference))
            group = f"seed_{seed}:round_{round_t}"
            raw.extend(round_rows(result, scheme="hvsa", run_id=run_id, seed=seed, group=group,
                                  dimension=dim, m=args.edges, round_t=round_t))
            messages.extend(message_rows(result, scheme="hvsa", run_id=run_id, group=group, round_t=round_t))
            valid = exact and result.verified is True
            row = {"seed": seed, "round": round_t, "group": group, "exact_match": exact,
                   "verified": result.verified, "valid": valid, "training_ms": training_ms,
                   "protocol_ms": protocol_ms, "input_sha256": vector_digest(vectors),
                   "snapshot": snapshot.relative_to(directory).as_posix(), "snapshot_sha256": digest(snapshot),
                   "raw_sum_sha256": hashlib.sha256(result.raw_sum.astype("<i8").tobytes()).hexdigest()
                   if result.raw_sum is not None else None}
            if valid:
                agreed = result.extra["model"]
                if not np.array_equal(agreed, reference_model):
                    raise ValueError("Edge model differs from the same-encoding reference")
                set_parameters(model, agreed)
                row["test_accuracy"] = accuracy(model, testing)
                float_mean = np.mean(np.stack(floats), axis=0)
                row["same_updates_float_mean_max_abs_difference"] = float(np.max(np.abs(float_mean-agreed)))
            else:
                failures.append({"scheme": "hvsa", "seed": seed, "round": round_t,
                                 "reason": result.failure_reason or "honest correctness/verification failed"})
            observations.append(row)
            seed_rows.append(row)
            dump_json(directory / "checkpoints" / f"seed_{seed}.json", seed_rows)
            print(f"seed={seed} round={round_t} valid={valid} accuracy={row.get('test_accuracy')} "
                  f"protocol_ms={protocol_ms:.1f}", flush=True)
            if not valid:
                break
        summaries.append({"scheme": "hvsa", "seed": seed, "rounds": len(seed_rows),
                          "exact_match": sum(r["exact_match"] for r in seed_rows),
                          "verified": sum(r["verified"] is True for r in seed_rows),
                          "last_test_accuracy": seed_rows[-1].get("test_accuracy"),
                          "training_ms": sum(r["training_ms"] for r in seed_rows),
                          "protocol_ms": sum(r["protocol_ms"] for r in seed_rows)})
    corr = {"experiment": "e7", "schemes": ["hvsa"], "observations": observations,
            "reference": "same local models, same integer encoding and equal-client mean"}
    finalize_run(directory, manifest, raw, messages, corr, summaries, failures)
    print(f"COMPLETED={directory} failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
