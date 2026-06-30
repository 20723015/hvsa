import argparse
import copy
import logging
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import models
from agent_async_secure import AsyncAuditor, AsyncClient, AsyncCoordinator, AsyncEdge
from datasets import distribute_data, load_data
from llm_reasoner import LLMReasoner
from rule_engine import RuleEngine
from ta import TrustedAuthority
from user import User

TARGET_DATASET = "MNIST"

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

CFG = CONFIGS[TARGET_DATASET]

NUM_EDGE_SERVERS = 2
USERS_PER_EDGE = 4
TOTAL_USERS = NUM_EDGE_SERVERS * USERS_PER_EDGE
TARGET_BATCH_COMMITS = 12
MAX_TICKS = 40
EVAL_EVERY = 2
CLIENT_LATENCY = (1, 3)
USER_DROPOUT_RATE = 0.1
MIN_BATCH_SIZE = 2
FLUSH_TIMEOUT = 2
ASYNC_WEIGHT = 0.6
ENABLE_VERIFICATION = True
SEED = 7


def evaluate_global_model(flat_model: np.ndarray, model_meta: list, test_loader, dataset_name: str, model_name: str):
    eval_model = models.get_model(dataset_name, model_name)
    models.unflatten_parameters(eval_model, flat_model, model_meta)
    return models.evaluate(eval_model, test_loader)


def build_reasoner(backend: str, api_key: Optional[str], model: Optional[str]) -> Optional[LLMReasoner]:
    if backend == "none":
        return None
    return LLMReasoner(
        backend=backend,
        api_key=api_key,
        model=model,
        fallback=True,
        cache_size=128,
        mock_seed=SEED,
    )


def main():
    parser = argparse.ArgumentParser(description="Secure semi-async HVSA experiment with optional agent advice")
    parser.add_argument("--backend", type=str, default="none", choices=["none", "mock", "anthropic", "openai"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--max-ticks", type=int, default=MAX_TICKS)
    parser.add_argument("--target-batch-commits", type=int, default=TARGET_BATCH_COMMITS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--min-batch-size", type=int, default=MIN_BATCH_SIZE)
    parser.add_argument("--flush-timeout", type=int, default=FLUSH_TIMEOUT)
    parser.add_argument("--async-weight", type=float, default=ASYNC_WEIGHT)
    parser.add_argument("--dropout-rate", type=float, default=USER_DROPOUT_RATE)
    parser.add_argument("--max-staleness", type=int, default=3)
    parser.add_argument("--risk-threshold", type=float, default=0.8)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = f"exp_async_secure_log_{TARGET_DATASET}_{args.backend}_{timestamp}.txt"

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

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    logging.info("=== Secure Semi-Async HVSA Experiment ===")
    logging.info(f"Dataset: {TARGET_DATASET}")
    logging.info(
        "Config: edges=%s users_per_edge=%s commits=%s latency=%s min_batch=%s timeout=%s async_weight=%.2f backend=%s",
        NUM_EDGE_SERVERS,
        USERS_PER_EDGE,
        args.target_batch_commits,
        CLIENT_LATENCY,
        args.min_batch_size,
        args.flush_timeout,
        args.async_weight,
        args.backend,
    )

    reasoner = build_reasoner(args.backend, args.api_key, args.model)
    if reasoner is not None:
        logging.info(f"LLM backend: {reasoner.backend_name}")
    else:
        logging.info("LLM backend: disabled (rule-only mode)")

    train_dataset, test_dataset = load_data(TARGET_DATASET)
    train_loaders = distribute_data(
        train_dataset,
        TOTAL_USERS,
        CFG["batch_size"],
        iid=True,
        max_samples_per_user=CFG["samples_per_user"],
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    initial_model = models.get_model(TARGET_DATASET, CFG["model"])
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
        llm_reasoner=reasoner,
        decision_hook=lambda dt, result: logging.getLogger("agent").info(
            "[%s] %s | confidence=%.2f | %s",
            dt.value,
            result.decision,
            result.confidence,
            result.reasoning,
        ),
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
            enable_verification=ENABLE_VERIFICATION,
            llm_reasoner=reasoner,
            decision_hook=lambda dt, result: logging.getLogger("agent").info(
                "[%s] %s | confidence=%.2f | %s",
                dt.value,
                result.decision,
                result.confidence,
                result.reasoning,
            ),
            rule_engine=rule_engine,
        )
        for idx in range(NUM_EDGE_SERVERS)
    ]
    edge_map = {edge.id: edge for edge in edges}

    clients = []
    for idx in range(TOTAL_USERS):
        user = User(f"AU-{idx}", copy.deepcopy(initial_model))
        user.register_with_ta(ta.get_public_key_bytes())
        ta.register_user(user.id, user.get_public_key_bytes())

        client = AsyncClient(
            user=user,
            train_loader=train_loaders[idx],
            rng=rng,
            latency_range=CLIENT_LATENCY,
            dropout_rate=args.dropout_rate,
        )
        clients.append(client)

    history = []
    for tick in range(args.max_ticks):
        global_vector, global_version = coordinator.get_global_snapshot()

        for idx, client in enumerate(clients):
            if not client.is_idle():
                continue

            edge = edges[idx // USERS_PER_EDGE]
            window_token, base_version = edge.assign_window(global_version, tick)
            client.start_task(
                global_model_vector=global_vector,
                model_meta=model_meta,
                edge_id=edge.id,
                window_token=window_token,
                base_version=base_version,
                local_epochs=CFG["local_epochs"],
                lr=CFG["lr"],
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
                    logging.info(
                        "Tick %s | %s rejected batch token=%s",
                        tick,
                        commit.edge_id,
                        commit.window_token,
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
                        TARGET_DATASET,
                        CFG["model"],
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
            TARGET_DATASET,
            CFG["model"],
        )
        history.append((coordinator.commit_count, acc, loss))

    data_file = f"exp_async_secure_data_{TARGET_DATASET}_{args.backend}_{timestamp}.npz"
    np.savez(
        data_file,
        commits=np.array([item[0] for item in history], dtype=np.int64),
        accuracy=np.array([item[1] for item in history], dtype=np.float64),
        loss=np.array([item[2] for item in history], dtype=np.float64),
    )
    logging.info("Saved history to %s", data_file)

    plt.figure(figsize=(8, 6))
    plt.plot(
        [item[0] for item in history],
        [item[1] for item in history],
        marker="o",
        linewidth=2,
        label=f"Secure Semi-Async ({args.backend})",
    )
    plt.xlabel("Committed Batches")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"Secure Semi-Async HVSA ({TARGET_DATASET})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plot_file = f"exp_async_secure_plot_{TARGET_DATASET}_{args.backend}_{timestamp}.png"
    plt.savefig(plot_file, dpi=300)
    logging.info("Saved plot to %s", plot_file)

    logging.info("Auditor summary: %s", auditor.summary())
    if reasoner is not None:
        logging.info("LLM stats: %s", reasoner.stats())


if __name__ == "__main__":
    main()
