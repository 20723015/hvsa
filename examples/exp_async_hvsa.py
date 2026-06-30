import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import models
from agent_async_secure import AsyncAuditor, AsyncClient, AsyncCoordinator, AsyncEdge
from datasets import distribute_data, load_data
from rule_engine import RuleEngine
from ta import TrustedAuthority
from user import User

CONFIGS = {
    "MNIST": {
        "model": "CNN",
        "local_epochs": 1,
        "lr": 0.01,
        "samples_per_user": 300,
        "batch_size": 64,
    },
    "CIFAR10": {
        "model": "CNN",
        "local_epochs": 2,
        "lr": 0.01,
        "samples_per_user": 1000,
        "batch_size": 64,
    },
}


def evaluate_global_model(flat_model: np.ndarray, model_meta: list, test_loader, dataset_name: str, model_name: str):
    eval_model = models.get_model(dataset_name, model_name)
    models.unflatten_parameters(eval_model, flat_model, model_meta)
    return models.evaluate(eval_model, test_loader)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rule-driven semi-async HVSA experiment (no LLM agent dependency)"
    )
    parser.add_argument("--dataset", type=str, default="MNIST", choices=sorted(CONFIGS))
    parser.add_argument("--num-edge-servers", type=int, default=2)
    parser.add_argument("--users-per-edge", type=int, default=4)
    parser.add_argument("--target-batch-commits", type=int, default=12)
    parser.add_argument("--max-ticks", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--min-batch-size", type=int, default=2)
    parser.add_argument("--flush-timeout", type=int, default=2)
    parser.add_argument("--async-weight", type=float, default=0.6)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--max-staleness", type=int, default=3)
    parser.add_argument("--risk-threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noniid", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--disable-verification", action="store_true")
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--samples-per-user", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_cfg = CONFIGS[args.dataset]
    local_epochs = args.local_epochs if args.local_epochs is not None else dataset_cfg["local_epochs"]
    lr = args.lr if args.lr is not None else dataset_cfg["lr"]
    samples_per_user = (
        args.samples_per_user if args.samples_per_user is not None else dataset_cfg["samples_per_user"]
    )
    batch_size = args.batch_size if args.batch_size is not None else dataset_cfg["batch_size"]

    total_users = args.num_edge_servers * args.users_per_edge
    verification_enabled = not args.disable_verification
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    distribution_tag = f"noniid_a{args.alpha}" if args.noniid else "iid"
    log_filename = f"exp_async_hvsa_log_{args.dataset}_{distribution_tag}_{timestamp}.txt"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logging.getLogger("user").setLevel(logging.WARNING)
    logging.getLogger("ta").setLevel(logging.WARNING)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    logging.info("=== Rule-Driven Secure Semi-Async HVSA Experiment ===")
    logging.info(
        "Config: dataset=%s edges=%s users_per_edge=%s total_users=%s commits=%s max_ticks=%s",
        args.dataset,
        args.num_edge_servers,
        args.users_per_edge,
        total_users,
        args.target_batch_commits,
        args.max_ticks,
    )
    logging.info(
        "Policy: min_batch=%s timeout=%s async_weight=%.2f max_staleness=%s risk_threshold=%.2f verification=%s distribution=%s",
        args.min_batch_size,
        args.flush_timeout,
        args.async_weight,
        args.max_staleness,
        args.risk_threshold,
        "on" if verification_enabled else "off",
        distribution_tag,
    )

    train_dataset, test_dataset = load_data(args.dataset)
    train_loaders = distribute_data(
        train_dataset,
        total_users,
        batch_size,
        iid=not args.noniid,
        max_samples_per_user=samples_per_user,
        dirichlet_alpha=args.alpha,
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    initial_model = models.get_model(args.dataset, dataset_cfg["model"])
    initial_vector, model_meta = models.flatten_parameters(initial_model)

    ta = TrustedAuthority(gradient_dim=initial_vector.shape[0], ta_id="TA-01")
    rule_engine = RuleEngine(
        min_batch_for_flush=args.min_batch_size,
        max_staleness=args.max_staleness,
        reject_risk_threshold=args.risk_threshold,
    )
    auditor = AsyncAuditor()
    coordinator = AsyncCoordinator(
        ta=ta,
        initial_model_vector=initial_vector,
        async_weight=args.async_weight,
        llm_reasoner=None,
        decision_hook=None,
        rule_engine=rule_engine,
        max_staleness=args.max_staleness,
        reject_risk_threshold=args.risk_threshold,
    )
    edges = [
        AsyncEdge(
            edge_id=f"AE-{idx}",
            ta=ta,
            coordinator=coordinator,
            auditor=auditor,
            min_batch_size=args.min_batch_size,
            flush_timeout=args.flush_timeout,
            enable_verification=verification_enabled,
            llm_reasoner=None,
            decision_hook=None,
            rule_engine=rule_engine,
        )
        for idx in range(args.num_edge_servers)
    ]
    edge_map = {edge.id: edge for edge in edges}

    clients = []
    for idx in range(total_users):
        user = User(f"AU-{idx}", copy.deepcopy(initial_model))
        user.register_with_ta(ta.get_public_key_bytes())
        ta.register_user(user.id, user.get_public_key_bytes())

        client = AsyncClient(
            user=user,
            train_loader=train_loaders[idx],
            rng=rng,
            latency_range=(1, 3),
            dropout_rate=args.dropout_rate,
        )
        clients.append(client)

    history = []
    rejected_batches = 0

    for tick in range(args.max_ticks):
        global_vector, global_version = coordinator.get_global_snapshot()

        for idx, client in enumerate(clients):
            if not client.is_idle():
                continue

            edge = edges[idx // args.users_per_edge]
            window_token, base_version = edge.assign_window(global_version, tick)
            client.start_task(
                global_model_vector=global_vector,
                model_meta=model_meta,
                edge_id=edge.id,
                window_token=window_token,
                base_version=base_version,
                local_epochs=local_epochs,
                lr=lr,
            )

        for client in clients:
            status, upload, drop_info = client.tick()
            if status == "upload" and upload is not None:
                edge_map[upload.edge_id].receive_upload(upload, tick)
            elif status == "drop" and drop_info is not None:
                edge_id, window_token = drop_info
                edge_map[edge_id].note_dropout(window_token)

        for edge in edges:
            commits = edge.tick(tick)
            for commit in commits:
                if not commit.verification_passed:
                    rejected_batches += 1
                    logging.info(
                        "Tick %s | %s rejected batch token=%s staleness=%s risk=%.3f",
                        tick,
                        commit.edge_id,
                        commit.window_token,
                        commit.staleness,
                        commit.risk_score,
                    )
                    continue

                logging.info(
                    "Tick %s | %s committed batch size=%s staleness=%s weight=%.3f -> global_version=%s",
                    tick,
                    commit.edge_id,
                    commit.batch_size,
                    commit.staleness,
                    commit.merge_weight,
                    commit.applied_version,
                )

                if coordinator.commit_count % args.eval_every == 0:
                    acc, loss = evaluate_global_model(
                        coordinator.global_model_vector,
                        model_meta,
                        test_loader,
                        args.dataset,
                        dataset_cfg["model"],
                    )
                    history.append((coordinator.commit_count, acc, loss))
                    logging.info(
                        "Evaluation after %s commits | Acc: %.2f%% | Loss: %.4f",
                        coordinator.commit_count,
                        acc,
                        loss,
                    )

        if coordinator.commit_count >= args.target_batch_commits:
            break

    if not history:
        acc, loss = evaluate_global_model(
            coordinator.global_model_vector,
            model_meta,
            test_loader,
            args.dataset,
            dataset_cfg["model"],
        )
        history.append((coordinator.commit_count, acc, loss))

    data_file = f"exp_async_hvsa_data_{args.dataset}_{distribution_tag}_{timestamp}.npz"
    np.savez(
        data_file,
        commits=np.array([item[0] for item in history], dtype=np.int64),
        accuracy=np.array([item[1] for item in history], dtype=np.float64),
        loss=np.array([item[2] for item in history], dtype=np.float64),
        rejected_batches=np.array([rejected_batches], dtype=np.int64),
        accepted_batches=np.array([coordinator.commit_count], dtype=np.int64),
    )
    logging.info("Saved history to %s", data_file)

    plt.figure(figsize=(8, 6))
    plt.plot(
        [item[0] for item in history],
        [item[1] for item in history],
        marker="o",
        linewidth=2,
        label="Rule-Driven Semi-Async HVSA",
    )
    plt.xlabel("Committed Batches")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"Rule-Driven Semi-Async HVSA ({args.dataset})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plot_file = f"exp_async_hvsa_plot_{args.dataset}_{distribution_tag}_{timestamp}.png"
    plt.savefig(plot_file, dpi=300)
    logging.info("Saved plot to %s", plot_file)

    logging.info("Auditor summary: %s", auditor.summary())
    logging.info("Rejected batches: %s", rejected_batches)


if __name__ == "__main__":
    main()
