#!/usr/bin/env python3
"""
HVSA dropout overhead (Table tab:dropout) — measured, not formula-based.

Setup: N=300 users, M=30 edge servers, d=10,000, single TA.
Scope: user masking + cloud unmasking (no Edge LHH / verification).
Surviving users do not upload recovery shares; CS receives M edge aggregates + TA mask.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from baselines.crypto_common import (
    comm_server_hvsa_mb,
    comm_user_payload_kb,
    median_ms,
    server_aggregate_sum,
)
from baselines.hvsa_crypto import (
    NUM_TAS,
    _derive_ta_seeds,
    measure_user_encryption,
)
from utils import KA, PRG

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

N_TOTAL = 300
M_EDGE = 30
D_DIM = 10_000
DROPOUT_RATES = (0.0, 0.10, 0.20, 0.30)


def _online_count(sigma: float) -> int:
    return max(N_TOTAL - int(N_TOTAL * sigma), 0)


def _build_ta_user_secrets(n_users: int) -> dict[str, bytes]:
    _, ta_priv = KA.gen()
    secrets: dict[str, bytes] = {}
    for i in range(n_users):
        uid = f"user-{i}"
        _, u_pub = KA.gen()
        secrets[uid] = KA.agree(ta_priv, u_pub)
    return secrets


def _global_mask(secrets: dict[str, bytes], online_ids: list[str], d: int, round_t: int = 1) -> np.ndarray:
    total = np.zeros(d, dtype=np.int64)
    for uid in online_ids:
        total += PRG.mask_int(secrets[uid], uid, round_t, d)
    return total


def measure_user(d: int, repeats: int) -> float:
    ta_seeds = _derive_ta_seeds()
    ms, _ = median_ms(repeats, lambda: measure_user_encryption(d, ta_seeds))
    return ms


def measure_cs_comp(d: int, m_edge: int, repeats: int) -> float:
    def run():
        edge_vecs = [
            np.random.randint(-100, 100, size=d, dtype=np.int64)
            for _ in range(m_edge)
        ]
        summed = server_aggregate_sum(edge_vecs)
        mask = np.random.randint(-(2**30), 2**30, size=d, dtype=np.int64)
        return summed - mask

    ms, _ = median_ms(repeats, run)
    return ms


def comm_cs_mb(d: int, m_edge: int) -> float:
    """M edge pre-aggregates + one TA global mask (each ``d`` float32 params)."""
    return comm_server_hvsa_mb(d, m_edge, include_ta_mask=True)


def run_benchmark(repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    user_secrets = _build_ta_user_secrets(N_TOTAL)

    user_comp = measure_user(D_DIM, repeats)
    user_comm = comm_user_payload_kb(D_DIM)
    cs_comp_base = measure_cs_comp(D_DIM, M_EDGE, repeats)
    cs_comm = comm_cs_mb(D_DIM, M_EDGE)

    rows: list[dict] = []
    for sigma in DROPOUT_RATES:
        n_online = _online_count(sigma)
        online_ids = [f"user-{i}" for i in rng.choice(N_TOTAL, size=n_online, replace=False)]

        ta_mask_ms, _ = median_ms(
            repeats,
            lambda ids=online_ids: _global_mask(user_secrets, ids, D_DIM),
        )

        rows.append({
            "dropout_rate": sigma,
            "dropout_pct": int(sigma * 100),
            "n_online": n_online,
            "user_comm_kb": float(user_comm),
            "user_comp_ms": float(user_comp),
            "cs_comm_mb": float(cs_comm),
            "cs_comp_ms": float(cs_comp_base),
            "ta_mask_ms": float(ta_mask_ms),
            "note": "Comm = d×4B float32 payload only; comp measured; TA mask time logged separately.",
        })
        logger.info(
            f"sigma={sigma:.0%} n_online={n_online}  "
            f"user={user_comm:.2f}KB/{user_comp:.3f}ms  "
            f"CS={cs_comm:.2f}MB/{cs_comp_base:.3f}ms  "
            f"(TA mask {ta_mask_ms:.2f}ms, not in table)"
        )

    return {
        "config": {
            "N": N_TOTAL,
            "M": M_EDGE,
            "d": D_DIM,
            "repeats": repeats,
            "seed": seed,
            "num_tas": NUM_TAS,
        },
        "rows": rows,
    }


def latex_rows(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        pct = r["dropout_pct"]
        lines.append(
            f"\\textbf{{{pct}\\%}}  & "
            f"{r['user_comm_kb']:.2f} & {r['cs_comm_mb']:.2f} & "
            f"{r['user_comp_ms']:.2f} & {r['cs_comp_ms']:.2f} \\\\"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="HVSA dropout overhead benchmark")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results/exp_dropout_overhead.json"))
    args = parser.parse_args()

    payload = run_benchmark(args.repeats, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved {args.out}")
    logger.info("\nLaTeX rows:\n" + latex_rows(payload["rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
