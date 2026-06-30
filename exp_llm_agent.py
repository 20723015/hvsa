"""
实验：LLM Agent 驱动的半异步 HVSA。

展示如何使用 LLMReasoner 驱动 Agent 决策。
支持三种模式:
    python exp_llm_agent.py                  # 默认：mock (随机模拟 LLM)
    python exp_llm_agent.py --backend mock    # 显式 mock
    python exp_llm_agent.py --backend anthropic  # 真实 Claude API (需设置 ANTHROPIC_API_KEY)
    python exp_llm_agent.py --backend openai     # 真实 OpenAI API (需设置 OPENAI_API_KEY)
"""

import argparse
import copy
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import models
from agent_async import AgentAuditor, AgentClient, AgentCoordinator, AgentEdge
from datasets import distribute_data, load_data
from llm_reasoner import DecisionType, DecisionResult, LLMReasoner
from orchestrate_agent import OrchestrationRuleEngine, patch_auditor_for_orchestration
from ta import TrustedAuthority
from user import User

# ─── 配置 ──────────────────────────────────────────

TARGET_DATASET = "MNIST"

CONFIGS = {
    "MNIST": {
        "model": "MLP",
        "local_epochs": 1,
        "lr": 0.01,
        "samples_per_user": 100,
        "batch_size": 32,
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
ENABLE_VERIFICATION = False
SEED = 7


def evaluate_global_model(flat_model: np.ndarray, model_meta: list, test_loader, dataset_name: str, model_name: str):
    eval_model = models.get_model(dataset_name, model_name)
    models.unflatten_parameters(eval_model, flat_model, model_meta)
    return models.evaluate(eval_model, test_loader)


def main():
    parser = argparse.ArgumentParser(description="LLM Agent-driven PP-HVFL Experiment")
    parser.add_argument("--backend", type=str, default="mock",
                        choices=["mock", "anthropic", "openai", "deepseek"],
                        help="LLM backend (default: mock)")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model name (optional)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (default: from env)")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = f"exp_llm_agent_log_{TARGET_DATASET}_{args.backend}_{timestamp}.txt"

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

    # ─── 1. 初始化 LLM Reasoner ─────────────────────
    reasoner = LLMReasoner(
        backend=args.backend,
        api_key=args.api_key,
        model=args.model,
        fallback=True,      # LLM 失败时降级到 mock
        cache_size=128,     # 缓存减少 API 调用
        mock_seed=SEED,     # mock 模式下可复现
    )
    logging.info(f"LLM Reasoner initialized: backend={reasoner.backend_name}")
    logging.info(f"  ANTHROPIC_API_KEY={'set' if os.environ.get('ANTHROPIC_API_KEY') else 'not set'}")
    logging.info(f"  OPENAI_API_KEY={'set' if os.environ.get('OPENAI_API_KEY') else 'not set'}")

    # 可选：记录每次 LLM 决策
    decision_log: list[tuple[DecisionType, DecisionResult]] = []

    def log_decision(dt: DecisionType, result: DecisionResult):
        decision_log.append((dt, result))
        logger = logging.getLogger("llm")
        logger.info(f"[{dt.value}] {result.decision} | confidence={result.confidence:.2f} | {result.reasoning}")

    logging.getLogger("llm").setLevel(logging.DEBUG if args.backend != "mock" else logging.INFO)

    # ─── 2. 数据准备 ─────────────────────────────────
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

    # ─── 3. 系统初始化 ───────────────────────
    ta = TrustedAuthority(gradient_dim=initial_vector.shape[0], ta_id="TA-01")
    auditor = AgentAuditor()
    patch_auditor_for_orchestration(auditor)
    orchestrate_engine = OrchestrationRuleEngine(
        quorum_q_es=2, total_edges=NUM_EDGE_SERVERS) if args.backend != "mock" else None
    coordinator = AgentCoordinator(
        ta, initial_vector,
        orchestrator=orchestrate_engine, quorum_q_es=2,
        async_weight=ASYNC_WEIGHT,
        llm_reasoner=reasoner,
        decision_hook=log_decision,
    )
    edges = [
        AgentEdge(
            edge_id=f"AE-{idx}",
            ta=ta,
            coordinator=coordinator,
            auditor=auditor,
            min_batch_size=MIN_BATCH_SIZE,
            flush_timeout=FLUSH_TIMEOUT,
            enable_verification=ENABLE_VERIFICATION,
            llm_reasoner=reasoner,
            decision_hook=log_decision,
        )
        for idx in range(NUM_EDGE_SERVERS)
    ]
    edge_map = {edge.id: edge for edge in edges}

    rng = np.random.default_rng(SEED)
    clients = []
    for idx in range(TOTAL_USERS):
        user = User(f"AU-{idx}", copy.deepcopy(initial_model))
        ta_pub = ta.get_public_key_bytes()
        user.register_with_ta(ta_pub)
        ta.register_user(user.id, user.get_public_key_bytes())

        client = AgentClient(
            user=user,
            train_loader=train_loaders[idx],
            rng=rng,
            latency_range=CLIENT_LATENCY,
            dropout_rate=USER_DROPOUT_RATE,
            llm_reasoner=reasoner,   # ← 接入 LLM
            decision_hook=log_decision,
        )
        clients.append(client)

    # ─── 4. 训练循环 ─────────────────────────────────
    logging.info("=" * 60)
    logging.info(f"LLM Agent Experiment | dataset={TARGET_DATASET} | backend={args.backend}")
    logging.info("=" * 60)

    history = []
    for tick in range(MAX_TICKS):
        global_vector, global_version = coordinator.get_global_snapshot()

        # 4a. 分配任务
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

        # 4b. tick 客户端（LLM 决策是否掉线）
        for client in clients:
            status, upload, drop_info = client.tick()
            if status == "upload" and upload is not None:
                edge_map[upload.edge_id].receive_upload(upload, tick)
            elif status == "drop" and drop_info is not None:
                edge_id, window_token = drop_info
                edge_map[edge_id].note_dropout(window_token)

        # 4c. tick 边缘服务器（LLM 决策是否聚合）
        for edge in edges:
            edge.tick(tick)

        # Cloud quorum: 攒够 Q_ES 个 Edge 才统一处理
        commits = coordinator.process_quorum(tick, edges)
        for commit in commits:
                # commit.verification_passed = coordinator 内部已用 LLM 决策
                if not commit.verification_passed:
                    logging.info(
                        "Tick %s | %s batch rejected | staleness=%s risk=%.3f",
                        tick, commit.edge_id, commit.staleness, commit.risk_score,
                    )
                    continue

                logging.info(
                    "Tick %s | %s commit size=%s staleness=%s weight=%.3f -> v%s",
                    tick, commit.edge_id, commit.batch_size,
                    commit.staleness, commit.merge_weight, commit.applied_version,
                )

                if coordinator.commit_count % EVAL_EVERY == 0:
                    acc, loss = evaluate_global_model(
                        coordinator.global_model_vector,
                        model_meta,
                        test_loader,
                        TARGET_DATASET,
                        CFG["model"],
                    )
                    history.append((coordinator.commit_count, acc, loss))
                    logging.info(
                        "Eval after %s commits | Acc: %.2f%% | Loss: %.4f",
                        coordinator.commit_count, acc, loss,
                    )

                    # 把准确率传给各个客户端（LLM 上下文用）
                    for client in clients:
                        client.record_accuracy(acc)

        if coordinator.commit_count >= TARGET_BATCH_COMMITS:
            break

    # ─── 5. 结果 ─────────────────────────────────────
    if not history:
        acc, loss = evaluate_global_model(
            coordinator.global_model_vector, model_meta,
            test_loader, TARGET_DATASET, CFG["model"],
        )
        history.append((coordinator.commit_count, acc, loss))

    data_file = f"exp_llm_agent_data_{TARGET_DATASET}_{args.backend}_{timestamp}.npz"
    np.savez(
        data_file,
        commits=np.array([item[0] for item in history], dtype=np.int64),
        accuracy=np.array([item[1] for item in history], dtype=np.float64),
        loss=np.array([item[2] for item in history], dtype=np.float64),
    )

    # 统计 LLM 决策
    drops = sum(1 for dt, r in decision_log if dt == DecisionType.CLIENT_DROPOUT and r.decision == "drop")
    flushes = sum(1 for dt, r in decision_log if dt == DecisionType.EDGE_FLUSH and r.decision == "flush")
    rejects = sum(1 for dt, r in decision_log if dt == DecisionType.COORDINATOR_REJECT_BATCH and r.decision == "reject")
    total = len(decision_log)

    logging.info("\n" + "=" * 60)
    logging.info("Experiment Complete")
    logging.info(f"LLM stats: {reasoner.stats()}")
    logging.info(f"Decisions: total={total} drops={drops} flushes={flushes} rejects={rejects}")
    logging.info(f"Final accuracy: {history[-1][1]:.2f}%")
    logging.info(f"Auditor: {auditor.summary()}")

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(
        [item[0] for item in history],
        [item[1] for item in history],
        marker="o", linewidth=2,
        label=f"LLM Agent ({args.backend})",
    )
    plt.xlabel("Committed Batches")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"LLM Agent-driven Semi-Async HVSA ({TARGET_DATASET})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plot_file = f"exp_llm_agent_plot_{TARGET_DATASET}_{args.backend}_{timestamp}.png"
    plt.savefig(plot_file, dpi=300)
    logging.info(f"Plot saved: {plot_file}")
    plt.show()


if __name__ == "__main__":
    main()
