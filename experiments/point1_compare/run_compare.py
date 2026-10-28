"""Point-1 protocol comparison runner: E1 correctness + E5 tamper (HVSA / GPVO).

计划: docs/POINT1_COMPARATIVE_EXPERIMENT_PLAN_20261017.md §E1/§E5;
映射与口径: experiments/point1_compare/protocol_mapping.md。

用法 (repo 根目录):
  python -m experiments.point1_compare.run_compare e1 --smoke
  python -m experiments.point1_compare.run_compare e1                  # 100 组
  python -m experiments.point1_compare.run_compare e1 --full-dim       # 加 10×d=7850 回放
  python -m experiments.point1_compare.run_compare e5                  # 篡改检查
  python -m experiments.point1_compare.run_compare e1 --schemes hvsa   # 单方案
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import logging
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.point1_compare.adapters.common import (  # noqa: E402
    MessageRecord,
    RoundResult,
    gen_client_vectors,
    partition_users,
    reference_sum,
    vector_digest,
)
from experiments.point1_compare.adapters.gpvo_full import (  # noqa: E402
    GPVOSession,
    com_decommit,
    com_param_gen,
    lhf_hash,
    run_gpvo_round,
)
from experiments.point1_compare.adapters.hvsa_point1 import HVSAProtocol  # noqa: E402
from utils import LHH  # noqa: E402
from experiments.point1_compare.artifacts import RAW_FIELDS, create_run, finalize_run  # noqa: E402

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "configs", "mvp.json")

def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _lib_versions() -> dict:
    import cryptography
    import ecdsa
    versions = {"python": platform.python_version(),
            "numpy": np.__version__,
            "cryptography": cryptography.__version__,
            "ecdsa": ecdsa.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "machine": platform.machine(), "logical_cpu_count": os.cpu_count()}
    try:
        import psutil
        versions["physical_cpu_count"] = psutil.cpu_count(logical=False)
        versions["memory_total_bytes"] = psutil.virtual_memory().total
    except ImportError:
        versions["memory_total_bytes"] = None
    return versions


def _review_metadata(cfg: dict) -> dict:
    """Snapshot future runs without rewriting historical manifests or their provenance."""
    source_paths = ["user.py", "edge.py", "ta.py", "cloud.py", "utils.py",
                    "tests/test_point1_compare.py"]
    experiment_root = os.path.join(ROOT, "experiments", "point1_compare")
    for current, dirs, files in os.walk(experiment_root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            if name.endswith((".py", ".json", ".md")):
                source_paths.append(os.path.relpath(os.path.join(current, name), ROOT))
    digests = {}
    # Include locally imported dependencies, not only the five entity files.
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and filename.endswith(".py"):
            absolute = os.path.abspath(filename)
            try:
                local = os.path.normcase(os.path.commonpath([ROOT, absolute])) == os.path.normcase(ROOT)
            except ValueError:
                local = False
            if local and os.path.isfile(absolute):
                source_paths.append(os.path.relpath(absolute, ROOT))
    for relative in sorted(set(source_paths)):
        with open(os.path.join(ROOT, relative), "rb") as source:
            digests[relative.replace(os.sep, "/")] = hashlib.sha256(source.read()).hexdigest()
    return {
        "config_snapshot": copy.deepcopy(cfg),
        "source_sha256": digests,
        "implementation_variants": {
            "hvsa": "real-point1-entities-exclusive-codec-receiver-v2",
            "gpvo": "Z_N2-LHF-fixed-session-compatibility-variant",
        },
        "performance_qualified": False,
        "measurement_limitations": [
            "GPVO LHF differs from the paper's stated P-256 implementation; security/parameter review pending.",
            "Fixed GPVO key lifecycle is a variant choice; per-training-round initialization must be evaluated.",
            "Legacy GPVO timing/traffic still requires alignment with the v2 HVSA measurement boundaries.",
            "E5 small-dimension checks do not complete the full-dimension or adaptive-tamper matrix.",
        ],
        "scope_review": "docs/POINT1_COMPARISON_E1_E5_20261018.md",
    }


def _new_run_id(tag: str) -> str:
    shanghai = _dt.timezone(_dt.timedelta(hours=8))
    return _dt.datetime.now(shanghai).strftime(f"%Y%m%d-%H%M%S-%f_{tag}")


def _write_run(run_dir: str, manifest: dict, raw_rows: List[dict],
               messages: List[dict], correctness: dict, failures: List[dict],
               summary: List[dict]) -> None:
    if manifest.get("schema_version") != 2:
        raise ValueError("New runs require archived schema-2 provenance")
    finalize_run(Path(run_dir), manifest, raw_rows, messages, correctness, summary, failures)



def _round_rows(res: RoundResult, *, run_id: str, seed: int, group: Any,
                repeat: int, m_edges: int, dim: int, round_t: int) -> List[dict]:
    rows = []
    base = {"scheme": res.scheme, "run_id": run_id, "seed": seed,
            "group": group, "repeat": repeat, "N_selected": res.n_selected,
            "N_uploaded": res.n_uploaded, "M": m_edges, "d": dim,
            "round": round_t, "success": res.completed,
            "verified": res.verified, "error_reason": res.failure_reason}
    for ph in res.phases:
        rows.append({**base, "phase": ph.phase, "role": ph.role,
                     "actor": ph.actor, "detail": ph.detail,
                     "elapsed_ms": round(ph.elapsed_ms, 6), "sent_bytes": None})
    return rows


def _message_rows(res: RoundResult, *, run_id: str, group: Any, round_t: int) -> List[dict]:
    return [{"scheme": res.scheme, "run_id": run_id, "group": group,
             "round": round_t, "link": m.link, "direction": m.direction,
             "sender": m.sender, "receiver": m.receiver, "phase": m.phase,
             "payload_kind": m.payload_kind, "num_bytes": m.num_bytes,
             "note": m.note} for m in res.messages]


# ==========================================================================
# E1: 完整聚合正确性与验证行为
# ==========================================================================

def _init_scheme(scheme: str, dim: int, groups: List[List[str]],
                 com_params, rsa_bits: int):
    if scheme == "hvsa":
        return HVSAProtocol(dim, groups), None
    if scheme == "gpvo":
        session = GPVOSession(dim, groups, com_params=com_params, rsa_bits=rsa_bits)
        return session, com_params
    raise ValueError(f"scheme {scheme} 未实现 (vlmrs 待原文冻结, 计划阶段 D)")


def _run_scheme_round(scheme: str, state, vectors, online: set, round_t: int,
                      base_vec: Optional[np.ndarray] = None) -> RoundResult:
    if scheme == "hvsa":
        return state.run_round(vectors, online, round_t, base_model_vec=base_vec)
    return run_gpvo_round(state, vectors, online, round_t)


def run_e1(cfg: dict, args) -> int:
    schemes: List[str] = args.schemes.split(",") if args.schemes else cfg["schemes"]
    e1 = cfg["e1"]
    seed = cfg["correctness_seed"]
    rng = np.random.default_rng(seed)
    run_id = _new_run_id("e1")
    rsa_bits = cfg.get("rsa_commit_bits", 3072)
    initial_metadata = _review_metadata(cfg)
    run_dir = os.path.join(ROOT, cfg.get("output_root", "results/point1_compare"), run_id)
    create_run(Path(run_dir), {"run_id": run_id, "experiment": "e1", "execution_status": "running",
                              **initial_metadata}, Path(ROOT))

    com_params = com_param_gen(rsa_bits) if "gpvo" in schemes else None
    if com_params is not None:
        print(f"  RSA 承诺参数已生成 (N={rsa_bits} bit, e>{rsa_bits}... bit, 一次/部署)")

    plan_groups: List[tuple] = []
    for _ in range(e1["groups"]):
        d = int(rng.choice(e1["dims"]))
        n = int(rng.choice(e1["n_values"]))
        m = int(rng.choice(e1["m_values"]))
        plan_groups.append((d, n, m))

    raw_rows: List[dict] = []
    msg_rows: List[dict] = []
    correctness: dict = {"experiment": "e1", "seed": seed, "schemes": schemes,
                         "groups": [], "counts": {}}
    failures: List[dict] = []
    tallies = {s: {"groups": 0, "exact_match": 0, "verified": 0,
                   "failed": 0, "avg_abs_err_max": 0.0} for s in schemes}

    print("=" * 88)
    print(f"E1 完整聚合正确性: {e1['groups']} 组, d∈{e1['dims']}, N∈{e1['n_values']}, "
          f"M∈{e1['m_values']}, schemes={schemes}")
    print("=" * 88)

    for gi, (d, n, m) in enumerate(plan_groups):
        idx_groups = partition_users(rng, n, m, e1.get("min_users_per_edge", 2))
        groups = [[f"u{i}" for i in g] for g in idx_groups]
        user_ids = [f"u{i}" for i in range(n)]
        vecs = gen_client_vectors(rng, n, d, e1["vector_mix"], e1["amplitude"])
        reference = reference_sum([vecs[uid] for uid in user_ids])
        online = set(user_ids)  # E1 无掉线
        n_online = len(online)

        for scheme in schemes:
            state, _ = _init_scheme(scheme, d, groups, com_params, rsa_bits)
            res = _run_scheme_round(scheme, state, vecs, online, round_t=gi)
            tallies[scheme]["groups"] += 1

            exact = (res.completed and res.raw_sum is not None
                     and np.array_equal(res.raw_sum, reference))
            verified = bool(res.verified) if res.verified is not None else False
            avg_abs_err = 0.0
            if res.completed and res.raw_sum is not None:
                ref_avg = reference.astype(np.float64) / n_online
                got_avg = res.raw_sum.astype(np.float64) / n_online
                avg_abs_err = float(np.max(np.abs(ref_avg - got_avg)))
                tallies[scheme]["avg_abs_err_max"] = max(
                    tallies[scheme]["avg_abs_err_max"], avg_abs_err)
            if exact:
                tallies[scheme]["exact_match"] += 1
            if verified:
                tallies[scheme]["verified"] += 1
            if not (res.completed and exact and verified):
                tallies[scheme]["failed"] += 1
                failures.append({"scheme": scheme, "group": gi, "seed": seed,
                                 "d": d, "N": n, "M": m, "round": gi,
                                 "completed": res.completed, "exact": exact,
                                 "verified": res.verified,
                                 "reason": res.failure_reason or "mismatch"})

            raw_rows.extend(_round_rows(res, run_id=run_id, seed=seed, group=gi,
                                        repeat=0, m_edges=m, dim=d, round_t=gi))
            msg_rows.extend(_message_rows(res, run_id=run_id, group=gi, round_t=gi))
            correctness["groups"].append({
                "group": gi, "scheme": scheme, "d": d, "N": n, "M": m,
                "n_online": n_online, "exact_match": bool(exact),
                "input_sha256": vector_digest(vecs),
                "verified": bool(res.verified) if res.verified is not None else None,
                "avg_abs_err": avg_abs_err,
                # user 总耗时按角色累计 (phase_ms 按 role 过滤, 两方案均为 role="user")
                "user_comp_ms_total": round(res.phase_ms("user"), 3),
            })
            print(f"  [{gi + 1:3d}/{e1['groups']}] {scheme:5s} d={d:6d} N={n:3d} M={m} "
                  f"exact={'Y' if exact else 'N'} verified={'Y' if verified else 'N'} "
                  f"({'FAIL: ' + res.failure_reason if not (exact and verified) else 'ok'})")

    for scheme in schemes:
        correctness["counts"][scheme] = tallies[scheme]

    # ---------------- independent full-dimension inputs (training snapshots when supplied) ----------------
    if args.full_dim:
        fr = e1["full_dim_replay"]
        d, n, m = fr["d"], fr["n"], fr["m"]
        online = {f"u{i}" for i in range(n)}
        print("-" * 88)
        snapshot_dir = getattr(args, "snapshot_dir", None)
        snapshot_paths = sorted(Path(snapshot_dir).glob("*.npz")) if snapshot_dir else []
        if snapshot_dir and len(snapshot_paths) < fr["groups"]:
            raise ValueError("Not enough training snapshots for the full-dimension schedule")
        print(f"E1 完整链路回放: {fr['groups']} 次 d={d} N={n} M={m} "
              f"({'真实训练快照' if snapshot_paths else '独立合成输入; 非训练快照'})")
        replay_rows = []
        for ri in range(fr["groups"]):
            idx_groups = partition_users(rng, n, m, e1.get("min_users_per_edge", 2))
            groups = [[f"u{i}" for i in g] for g in idx_groups]
            input_archive = None
            archive_hash = None
            base_vec = None
            if snapshot_paths:
                source = snapshot_paths[ri]
                input_path = Path(run_dir) / "inputs" / f"training_{ri:03d}.npz"
                input_path.parent.mkdir(parents=True, exist_ok=True)
                input_path.write_bytes(source.read_bytes())
                archive_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
                input_archive = input_path.relative_to(run_dir).as_posix()
                with np.load(input_path, allow_pickle=False) as snapshot:
                    values, ids = snapshot["vectors"], snapshot["user_ids"]
                    if values.shape != (n, d) or values.dtype != np.int64:
                        raise ValueError("Training snapshot shape/domain differs from configured E1 workpoint")
                    vecs = {str(uid): vector.copy() for uid, vector in zip(ids, values)}
                    base_vec = snapshot["base_model"].copy()
                    if set(vecs) != online:
                        raise ValueError("Training snapshot participants differ from configured E1 workpoint")
            else:
                vecs = gen_client_vectors(rng, n, d, {"uniform": 1., "zero": 0., "boundary": 0.}, e1["amplitude"])
            reference = reference_sum([vecs[f"u{i}"] for i in range(n)])
            for scheme in schemes:
                state, _ = _init_scheme(scheme, d, groups, com_params, rsa_bits)
                res = _run_scheme_round(scheme, state, vecs, online, round_t=1000 + ri,
                                        base_vec=base_vec)
                exact = (res.completed and res.raw_sum is not None
                         and np.array_equal(res.raw_sum, reference))
                verified = bool(res.verified) if res.verified is not None else False
                tallies[scheme]["groups"] += 1
                if exact:
                    tallies[scheme]["exact_match"] += 1
                if verified:
                    tallies[scheme]["verified"] += 1
                if not (res.completed and exact and verified):
                    tallies[scheme]["failed"] += 1
                    failures.append({"scheme": scheme, "group": f"replay_{ri}",
                                     "seed": seed, "d": d, "N": n, "M": m,
                                     "round": 1000 + ri, "completed": res.completed,
                                     "exact": exact, "verified": res.verified,
                                     "reason": res.failure_reason or "mismatch"})
                raw_rows.extend(_round_rows(res, run_id=run_id, seed=seed,
                                            group=f"replay_{ri}", repeat=0, m_edges=m,
                                            dim=d, round_t=1000 + ri))
                msg_rows.extend(_message_rows(res, run_id=run_id,
                                              group=f"replay_{ri}", round_t=1000 + ri))
                correctness["groups"].append({
                    "group": f"replay_{ri}", "scheme": scheme, "d": d, "N": n,
                    "M": m, "n_online": len(online), "exact_match": bool(exact),
                    "input_sha256": vector_digest(vecs),
                    "input_kind": "training_snapshot" if snapshot_paths else "independent_synthetic",
                    "input_archive": input_archive, "input_archive_sha256": archive_hash,
                    "verified": bool(res.verified) if res.verified is not None else None,
                    "avg_abs_err": 0.0,
                    "user_comp_ms_total": round(res.phase_ms("user"), 3),
                })
                print(f"  [replay {ri + 1:2d}/{fr['groups']}] {scheme:5s} "
                      f"exact={'Y' if exact else 'N'} verified={'Y' if verified else 'N'} "
                      f"({'FAIL: ' + res.failure_reason if not (exact and verified) else 'ok'})")

    summary = []
    for scheme in schemes:
        t = tallies[scheme]
        summary.append({"scheme": scheme, "groups": t["groups"],
                        "exact_match": t["exact_match"],
                        "verified": t["verified"], "failed": t["failed"],
                        "avg_abs_err_max": t["avg_abs_err_max"]})

    manifest = {
        "run_id": run_id, "experiment": "e1", "command": " ".join(sys.argv),
        "config": cfg.get("stage"), "seed": seed, "schemes": schemes,
        "git_commit": _git_commit(), "lib_versions": _lib_versions(),
        "quant": {"scale": 1_000_000, "prg_mask_range": [-1_000_000_000, 1_000_000_000],
                  "gpvo_compat_round_token": cfg.get("prg_round_token", "static_per_paper"),
                  "hvsa_prg_context": "user_id_and_round"},
        "rsa_commit_bits": rsa_bits,
        "rsa_commit_exp": cfg.get("rsa_commit_exp", "as_written_gt_N2"),
        "protocol_mapping": "experiments/point1_compare/protocol_mapping.md",
        "paper_versions": {"gpvo": "TNSM 2025, DOI 10.1109/TNSM.2025.3592122 (Algorithms 1-5)",
                           "hvsa": "chapter03/CODEX.md 单 TA 设计"},
        "concurrency": "single-process serial replay",
        "timing_scope": "本地协议回放耗时 (无 RTT/带宽建模)",
        "verification": cfg.get("verification", "full"),
        "aggregation": cfg.get("aggregation", "equal_weight_integer_sum_then_average"),
        "full_dim_replay": bool(args.full_dim),
        "input_digest_algorithm": "uid_zero_delimited_int64le_v1",
    }
    manifest.update(initial_metadata)
    manifest.update({"schema_version": 2, "source_archive": "sources"})
    _write_run(run_dir, manifest, raw_rows, msg_rows, correctness, failures, summary)
    print("-" * 88)
    for row in summary:
        print(f"  {row['scheme']:5s}: {row['groups']} 组, exact={row['exact_match']}, "
              f"verified={row['verified']}, failed={row['failed']}, "
              f"max|Δavg|={row['avg_abs_err_max']:.3e}")
    print(f"  结果已写入: {run_dir}")
    return 0 if all(t["failed"] == 0 for t in tallies.values()) else 1


# ==========================================================================
# E5: 恶意云篡改与上下文检查
# ==========================================================================

def run_e5(cfg: dict, args) -> int:
    """All new E5 execution uses individual-trial/schema-2 evidence."""
    from experiments.point1_compare.run_integrity import main as strict_e5
    selected = args.schemes.split(",") if args.schemes else cfg["schemes"]
    selected = ["gpvo_compat_fixed" if s == "gpvo" else s for s in selected]
    session = cfg["e5"].get("e5_session", {"n": 10, "m": 2})
    if session != {"n": 10, "m": 2}:
        raise ValueError("Strict E5 currently uses the documented small workpoint N=10,M=2")
    return strict_e5(["--schemes", ",".join(selected), "--small-trials",
                      str(args.e5_trials or cfg["e5"]["tamper_trials_per_type"]),
                      "--full-trials", str(cfg["e5"]["full_dim_checks"] if args.full_dim else 0),
                      "--small-dim", str(cfg["e5"]["small_dim"]),
                      "--full-dim", str(cfg["e5"]["full_dim_d"]),
                      "--rsa-bits", str(cfg.get("rsa_commit_bits", 3072)),
                      "--seed", str(cfg["correctness_seed"])])


# ------------------------------------------------------------------ helpers

def _honest_hash_list(state: HVSAProtocol):
    """Honest cloud-side hash/evidence list (deep-copied so tamper tests never
    corrupt protocol state). This is the real cloud._collected_hash_data."""
    return copy.deepcopy(state.cs._collected_hash_data)


def _gpvo_verify_only(session: GPVOSession, x_agg, c_agg: int, h_ta: int, r: int) -> bool:
    """Alg.5 verification-only entry (used after injecting tampered values).

    h_agg = LHF.Hash(x_agg); h = LHF.Aggregate(h_TA, h_agg) = h_TA·h_agg mod N²
    (整数乘法, 论文 Alg.5 / mapping §2.2-5); Decommit(c_agg, h, r) = [c_agg == h^e·g^r]."""
    h_agg = lhf_hash(session.com, x_agg)
    h = (h_ta * h_agg) % session.com.n2
    return com_decommit(session.com, c_agg, h, r)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("user").setLevel(logging.WARNING)
    logging.getLogger("edge").setLevel(logging.WARNING)
    logging.getLogger("cloud").setLevel(logging.WARNING)
    logging.getLogger("ta").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["e1", "e5"])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--schemes", default=None, help="逗号分隔, 如 hvsa,gpvo")
    ap.add_argument("--smoke", action="store_true", help="极小规模试跑")
    ap.add_argument("--full-dim", action="store_true", help="E1 加 d=7850 完整链路回放")
    ap.add_argument("--snapshot-dir", help="E1 full-dimension NPZ training snapshots; absent means synthetic inputs")
    ap.add_argument("--e1-groups", type=int, help="Explicit E1 small-input count (default 100)")
    ap.add_argument("--full-count", type=int, help="Explicit E1 full-dimension count (default 10)")
    ap.add_argument("--e5-trials", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    if args.smoke:
        cfg = copy.deepcopy(cfg)
        cfg["e1"]["groups"] = 3
        cfg["e1"]["full_dim_replay"] = {"n": 20, "m": 5, "d": 32, "groups": 1}
        cfg["e5"]["tamper_trials_per_type"] = 2
    if args.experiment == "e1":
        for option in [args.e1_groups, args.full_count]:
            if option is not None and option <= 0:
                ap.error("E1 counts must be positive")
        if args.e1_groups is not None:
            cfg["e1"]["groups"] = args.e1_groups
        if args.full_count is not None:
            cfg["e1"]["full_dim_replay"]["groups"] = args.full_count
    if args.experiment == "e1":
        sys.exit(run_e1(cfg, args))
    sys.exit(run_e5(cfg, args))
