"""GPVO-FL algebra-compatibility adapter (TNSM 2025, Algorithms 1-5).

Review 2026-10-18: this Z*_{N²}, fixed-session-key implementation is an
explicit compatibility variant, not a validated performance reproduction of
the paper's P-256 LHF implementation. Its aggregate commitment check accepts
coordinated changes to the result and commitment (see the regression test).
Fresh initialization per training round must be reviewed separately: the
paper's experimental section says keys are distributed each training round.

原文依据 (E:/Essay/DFL/GPVO-FL_...pdf, DOI 10.1109/TNSM.2025.3592122):
- §III.B/§III.C (p.4178-4179): DH / LHF / multiplicative homomorphic commitment
- Fig.3 (p.4183): Rounds 1-5 detailed process
- Alg.2-5 (p.4184): masking / pre-aggregation+proof / aggregation / verification
- Theorem 1 (p.4184): c_agg = Com.Commit(h, r), h = h_TA · h_agg = ∏_k h_k

解释决策见 protocol_mapping.md §2.2，要点:
1. Round 2 无秘密分享（原文无该步骤；baselines/gpvo_crypto.py:50-51 的额外
   Shamir 分享不进入本适配器）。
2. 掉线补偿取抵消正确符号 comp_i = -PRG(s_{i-1,i})·[i-1 在线] + PRG(s_{i,i+1})·[i+1 在线]
   （原文 Fig.3 Round 3 公式按字面执行会把残留翻倍，与定理 1 矛盾；由 E1/E4 裁决）。
3. Com.Commit(m,τ) = m^e·g^τ mod N²（原文式 (1) 变体，使定理 1 代数成立）；
   e > N² 且 gcd(e, φ(N²))=1（原文 ParamGen 原文照抄）；g 取 Z*_{N²} 均匀随机元素。
4. 本适配器使用固定部署密钥和PRG令牌；跨轮静态是此变体的选择，不能由原文
   省略PRG轮次参数就推定原文必然复用密钥。E3/E5须区分每轮初始化和复用。
5. LHF 实例化为 Z*_{N²} 上的乘法线性同态哈希 h(m)=∏g_i^{m_i} mod N²：定理 1 的
   承诺聚合代数要求 LHF 群律与承诺消息空间同构（群元素为 Z*_{N²} 中整数、群律为
   mod N² 乘法，原文 LHF 定义"each g_i ∈ G is coprime"即整数群语义）；§VIII.B 的
   "NISTP-256"表述与该代数冲突（EC 点编码不满足 enc(P+Q)=enc(P)·enc(Q)），
   此实例化差异随结果报告；本实现未给出碰撞抗性或整体协议安全性的独立归约。
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set

import numpy as np

from utils import KA, PRG

from .common import MessageRecord, RoundResult, Stopwatch, serialize

LHF_SEED = b"GPVO-LHF-public-seed-v1"
LHF_GENERATOR_DOMAIN = b"GPVO-LHF-generator-v1"
LHF_CACHE_MAXSIZE = 131072  # 记录于 mapping §3: 大维度下不假设全缓存


# --------------------------------------------------------------------------
# §III.C  Multiplicative homomorphic commitment (RSA-based, per the paper)
# --------------------------------------------------------------------------

@dataclass
class ComParams:
    n_mod: int   # N = p*q
    n2: int      # N^2
    g: int       # element of Z*_{N^2} (see mapping §2.2-3)
    e: int       # e > N^2, gcd(e, phi(N^2)) = 1 (paper §III.C as written)


def com_param_gen(bits: int = 3072) -> ComParams:
    """Com.ParamGen(1^k) -> {N, g, p, q, e} (paper §III.C item 1).

    Primes are drawn with the vetted ``cryptography`` RSA generator; only the
    modulus is used afterwards (no trapdoor is needed by the protocol).
    """
    from cryptography.hazmat.primitives.asymmetric import rsa as _rsa

    sk = _rsa.generate_private_key(public_exponent=65537, key_size=bits)
    numbers = sk.private_numbers()
    p, q = int(numbers.p), int(numbers.q)
    n_mod = p * q
    n2 = n_mod * n_mod
    phi_n2 = n_mod * (p - 1) * (q - 1)

    e_bytes = (n2.bit_length() + 15) // 8 + 2
    while True:
        e = int.from_bytes(os.urandom(e_bytes), "big") | 1
        if e > n2 and math.gcd(e, phi_n2) == 1:
            break

    g_bytes = (n2.bit_length() + 7) // 8
    while True:
        g = int.from_bytes(os.urandom(g_bytes), "big") % n2
        if g > 1 and math.gcd(g, n_mod) == 1:
            break

    return ComParams(n_mod=n_mod, n2=n2, g=g, e=e)


def com_commit(params: ComParams, m: int, tau: int) -> int:
    """Com.Commit(m, τ) = m^e · g^τ mod N² (paper §III.C item 3 / eq. (1))."""
    return (pow(m, params.e, params.n2) * pow(params.g, tau, params.n2)) % params.n2


def com_aggregate(params: ComParams, commitments: List[int]) -> int:
    """Com.Aggregate: ∏_k c_k mod N² (paper §III.C item 4 / Alg.4)."""
    out = 1
    for c in commitments:
        out = (out * c) % params.n2
    return out


def com_decommit(params: ComParams, c: int, m: int, r: int) -> bool:
    """Com.Decommit(c, m', r): recompute Com.Commit(m', r) and compare (§III.C item 5)."""
    return c == com_commit(params, m, r)


# --------------------------------------------------------------------------
# LHF over Z*_{N^2}: h(m) = ∏_i g_i^{m_i} mod N²  (mapping §2.2-5)
# --------------------------------------------------------------------------

@lru_cache(maxsize=LHF_CACHE_MAXSIZE)
def _lhf_generator(seed: bytes, index: int, n2: int, n_mod: int) -> int:
    """Deterministic hash-to-Z*_{N²} generator derivation (public LHF_pp)."""
    need = (n2.bit_length() + 7) // 8
    counter = 0
    while True:
        material = (LHF_GENERATOR_DOMAIN + len(seed).to_bytes(4, "big") + seed
                    + int(index).to_bytes(16, "big") + counter.to_bytes(8, "big"))
        digest = b""
        while len(digest) < need:
            digest += hashlib.sha256(
                material + len(digest).to_bytes(4, "big")).digest()
        g = int.from_bytes(digest, "big") % n2
        if g > 1 and math.gcd(g, n_mod) == 1:
            return g
        counter += 1


def lhf_hash(com: ComParams, vector: np.ndarray) -> int:
    """LHF.Hash(x) = ∏_i g_i^{x_i} mod N² (linear homomorphic over Z*_{N²}).

    LHF.Hash(a) · LHF.Hash(b) = LHF.Hash(a+b) (mod N²), negative exponents via
    modular inverse (Python pow handles them for units)."""
    flat = np.asarray(vector).reshape(-1)
    result = 1
    n2 = com.n2
    for i, val in enumerate(flat):
        v = int(val)
        if v == 0:
            continue
        g_i = _lhf_generator(LHF_SEED, i, n2, com.n_mod)
        result = (result * pow(g_i, v, n2)) % n2
    return result


# --------------------------------------------------------------------------
# PRG instantiation: ξ = PRG(s) keyed by the shared key only (paper Alg.2).
# utils.PRG additionally mixes a public label; both ends of a pair derive the
# SAME stream via the symmetric label below (cancellation requirement,
# mapping §2.2-4: this variant reuses keys and a fixed token; the paper's
# per-training-round initialization must not be inferred from PRG notation).
# --------------------------------------------------------------------------

_ROUND_TOKEN = "static_per_paper"  # Legacy identifier, not a verified paper key-lifecycle claim.


def _ta_mask(seed: bytes, user_id: str, dim: int) -> np.ndarray:
    return PRG.mask_int(seed, f"gpvo-ta|{user_id}", _ROUND_TOKEN, dim)


def _pair_mask(seed: bytes, group_idx: int, uid_a: str, uid_b: str, dim: int) -> np.ndarray:
    lo, hi = sorted((str(uid_a), str(uid_b)))
    return PRG.mask_int(seed, f"gpvo-pair|{group_idx}|{lo}|{hi}", _ROUND_TOKEN, dim)


# --------------------------------------------------------------------------
# Alg.1  Initialization
# --------------------------------------------------------------------------

class GPVOSession:
    """Alg.1/Round 1 state: TA-generated params, per-user DH keys, per-edge τ_k."""

    def __init__(self, dim: int, groups: List[List[str]],
                 com_params: Optional[ComParams] = None, rsa_bits: int = 3072):
        self.dim = dim
        self.groups = [list(g) for g in groups]
        t0 = time.perf_counter()
        self.com = com_params if com_params is not None else com_param_gen(rsa_bits)
        # TA generates every user's DH key pair and distributes (sk_i, pk_i)
        # (paper Round 1: "Compute key pair (sk_i^k, pk_i^k) for each user").
        self._user_priv: Dict[str, bytes] = {}
        self._user_pub: Dict[str, bytes] = {}
        for g in self.groups:
            for uid in g:
                pub, priv = KA.gen()
                self._user_pub[uid] = pub
                self._user_priv[uid] = priv
        self._ta_pub, self._ta_priv = KA.gen()
        # per-edge commitment keys τ_k ∈ Z*_{N²} (paper Com.KeyGen per committer)
        self._tau: List[int] = []
        for _ in self.groups:
            while True:
                tau = int.from_bytes(os.urandom((self.com.n2.bit_length() + 7) // 8), "big") % self.com.n2
                if tau > 0 and math.gcd(tau, self.com.n_mod) == 1:
                    self._tau.append(tau)
                    break
        # TA-side shared keys s_{i,TA} = DH.Agree(ta_priv, pk_i) (symmetric to user side)
        self._ta_shared = {uid: KA.agree(self._ta_priv, pub)
                           for uid, pub in self._user_pub.items()}
        self.init_ms = (time.perf_counter() - t0) * 1000.0

    def ring_neighbors(self, group_idx: int, uid: str) -> tuple:
        """Ring within the registered users of the group (paper Round 2:
        i-1 = m_k if i=1, i+1 = 1 if i=m_k). Returns (left_uid, right_uid)."""
        g = self.groups[group_idx]
        idx = g.index(uid)
        return g[(idx - 1) % len(g)], g[(idx + 1) % len(g)]

    def pair_key_user(self, uid: str, peer: str) -> bytes:
        """s_{i,j} = DH.Agree(sk_i, pk_j) computed on the user side."""
        return KA.agree(self._user_priv[uid], self._user_pub[peer])


def gpvo_user_mask(session: GPVOSession, group_idx: int, uid: str,
                   x_i: np.ndarray) -> np.ndarray:
    """Alg.2: x̃_i = x_i + ξ_{i,TA} − ξ_{i,i−1} + ξ_{i,i+1} (paper step 7)."""
    left, right = session.ring_neighbors(group_idx, uid)
    s_left = session.pair_key_user(uid, left)
    s_right = session.pair_key_user(uid, right)
    s_ta = KA.agree(session._user_priv[uid], session._ta_pub)
    d = session.dim
    masked = x_i.astype(np.int64).copy()
    masked += _ta_mask(s_ta, uid, d)
    masked -= _pair_mask(s_left, group_idx, uid, left, d)
    masked += _pair_mask(s_right, group_idx, uid, right, d)
    return masked


# --------------------------------------------------------------------------
# Rounds 2-5 execution
# --------------------------------------------------------------------------

def run_gpvo_round(session: GPVOSession, vectors: Dict[str, np.ndarray],
                   online: Set[str], round_t: int,
                   n_registered: Optional[int] = None) -> RoundResult:
    """Execute one GPVO-FL round (Alg.2-5) and return measurements + artifacts.

    ``online`` 是"实际上传且被边缘上报"的集合；未上传者按原文 Round 3 进入
    U_dropout 走补偿路径。E4 的"上传后离线"阶段由调用方把已上传者计入
    ``online``（其贡献照常聚合，成对掩码已对消）。
    """
    sw = Stopwatch()
    msgs: List[MessageRecord] = []
    d = session.dim
    n_uploaded = len(online)
    if n_uploaded == 0:
        return RoundResult(
            scheme="gpvo", completed=False, raw_sum=None, verified=None,
            n_selected=n_registered if n_registered is not None else sum(len(g) for g in session.groups),
            n_uploaded=0, phases=[], messages=[],
            failure_reason="no_online_users (TA 掩码为空, 回合无意义)")

    def record(link: str, phase: str, kind: str, sender: str, receiver: str,
               payload: Any, note: str = "") -> None:
        raw = serialize(payload)
        msgs.append(MessageRecord(link=link, direction="send", phase=phase,
                                  payload_kind=kind, sender=sender,
                                  receiver=receiver, num_bytes=len(raw), note=note))

    # ---------------- Round 2: user masking ----------------
    per_group_masked: Dict[int, Dict[str, np.ndarray]] = {k: {} for k in range(len(session.groups))}
    for k, g in enumerate(session.groups):
        for uid in g:
            if uid not in online:
                continue
            x_tilde = sw.time(
                "user_mask", "user",
                lambda uid=uid, k=k: gpvo_user_mask(session, k, uid, vectors[uid]),
                actor=uid)
            per_group_masked[k][uid] = x_tilde
            record("user->edge", "user_mask", "masked_upload", uid, f"edge_{k}",
                   {"user_id": uid, "group": k, "round": round_t, "x_tilde": x_tilde})

    # ---------------- Round 3: edge pre-aggregation + proof ----------------
    edge_state: Dict[int, Optional[Dict[str, Any]]] = {}
    for k, g in enumerate(session.groups):
        uploaded = per_group_masked[k]
        online_k = [u for u in g if u in online]
        dropouts = [u for u in g if u not in online]
        if not uploaded:
            # 组内无人上传: 该边缘本轮无数据可报 (全零聚合的 LHF 值为 1, 会令
            # c_k 与诚实聚合不可区分地退化); 原文语义为该组不参与本轮聚合
            edge_state[k] = None
            continue

        def aggregate_and_compensate(k=k, g=g, uploaded=uploaded, online_k=online_k, dropouts=dropouts):
            total = np.zeros(d, dtype=np.int64)
            for x_tilde in uploaded.values():
                total += x_tilde
            comp = np.zeros(d, dtype=np.int64)
            reveal_detail = []
            for d_uid in dropouts:
                left, right = session.ring_neighbors(k, d_uid)
                # left neighbor (if online) reveals the pair mask it ADDED -> subtract
                if left in online:
                    s = session.pair_key_user(left, d_uid)
                    comp -= _pair_mask(s, k, left, d_uid, d)
                    reveal_detail.append(f"{left}->-{d_uid}")
                # right neighbor (if online) reveals the pair mask it SUBTRACTED -> add back
                if right in online:
                    s = session.pair_key_user(right, d_uid)
                    comp += _pair_mask(s, k, right, d_uid, d)
                    reveal_detail.append(f"{right}->+{d_uid}")
            return total + comp, reveal_detail

        x_agg_k, reveal_detail = sw.time("edge_preagg", "edge",
                                         lambda: aggregate_and_compensate(),
                                         actor=f"edge_{k}")
        sw.records[-1].detail = ";".join(reveal_detail)
        # mask reveals are user->edge messages (compensation communication)
        for d_uid in dropouts:
            left, right = session.ring_neighbors(k, d_uid)
            if left in online:
                s = session.pair_key_user(left, d_uid)
                record("user->edge", "edge_preagg", "dropout_mask_reveal", left, f"edge_{k}",
                       {"user_id": left, "dropout_user": d_uid, "pair_mask": _pair_mask(s, k, left, d_uid, d)})
            if right in online:
                s = session.pair_key_user(right, d_uid)
                record("user->edge", "edge_preagg", "dropout_mask_reveal", right, f"edge_{k}",
                       {"user_id": right, "dropout_user": d_uid, "pair_mask": _pair_mask(s, k, right, d_uid, d)})

        h_k = sw.time("edge_evidence", "edge", lambda: lhf_hash(session.com, x_agg_k),
                      actor=f"edge_{k}")
        c_k = sw.time("edge_evidence", "edge",
                      lambda: com_commit(session.com, h_k, session._tau[k]),
                      actor=f"edge_{k}")
        record("edge->ta", "edge_evidence", "online_list", f"edge_{k}", "TA",
               {"edge_id": f"edge_{k}", "round": round_t, "U_online": online_k})
        record("edge->cloud", "edge_evidence", "pre_aggregation_with_proof", f"edge_{k}", "CS",
               {"edge_id": f"edge_{k}", "round": round_t, "x_agg_k": x_agg_k, "c_k": c_k})
        edge_state[k] = {"x_agg_k": x_agg_k, "h_k": h_k, "c_k": c_k,
                         "online_k": online_k, "dropouts": dropouts}

    # ---------------- Round 3 (TA): global mask, h_TA, r ----------------
    def ta_global_mask():
        xi = np.zeros(d, dtype=np.int64)
        for uid in sorted(online):
            xi += _ta_mask(session._ta_shared[uid], uid, d)
        return xi

    xi = sw.time("ta_mask", "ta", ta_global_mask, actor="TA")
    h_ta = sw.time("ta_mask", "ta", lambda: lhf_hash(session.com, xi), actor="TA")
    # r = Σ τ_k 只计入本轮实际参与聚合 (U_online 非空) 的边缘: 未提交承诺的
    # 边缘其 τ_k 不在乘积中 (原文假设每边缘都有数据; 空组语义见 mapping §2.2)
    active = [k for k in edge_state if edge_state[k] is not None]
    r = sw.time("ta_mask", "ta",
                lambda: sum(session._tau[k] for k in active), actor="TA")
    for k in range(len(session.groups)):
        record("ta->edge", "ta_mask", "verification_aid", "TA", f"edge_{k}",
               {"round": round_t, "h_TA": h_ta, "r": r})
    record("ta->cloud", "ta_mask", "global_mask", "TA", "CS",
           {"ta_id": "TA", "round": round_t, "xi": xi})

    # ---------------- Round 4: cloud aggregation ----------------
    def cloud_aggregate():
        x_agg = np.zeros(d, dtype=np.int64)
        for st in edge_state.values():
            if st is not None:
                x_agg += st["x_agg_k"]
        x_agg -= xi
        c_agg = com_aggregate(session.com,
                              [st["c_k"] for st in edge_state.values() if st is not None])
        return x_agg, c_agg

    x_agg, c_agg = sw.time("cloud_unmask", "cloud", cloud_aggregate, actor="CS")
    for k in range(len(session.groups)):
        record("cloud->edge", "cloud_unmask", "broadcast_aggregate", "CS", f"edge_{k}",
               {"round": round_t, "x_agg": x_agg, "c_agg": c_agg})

    # ---------------- Round 5: edge verification ----------------
    def verify_edge() -> bool:
        # h_agg = LHF.Hash(x_agg); h = LHF.Aggregate(h_TA, h_agg) = h_TA · h_agg
        # (mod N² 乘法, 论文 Alg.5); Decommit(c_agg, h, r) = [c_agg == h^e·g^r]
        h_agg = lhf_hash(session.com, x_agg)
        h = (h_ta * h_agg) % session.com.n2
        return com_decommit(session.com, c_agg, h, r)

    per_edge_ok = {}
    for k in range(len(session.groups)):
        per_edge_ok[k] = sw.time("edge_verify", "edge", verify_edge,
                                 actor=f"edge_{k}")
        record("edge->user", "edge_verify", "broadcast_result", f"edge_{k}", "users",
               {"round": round_t, "x_agg": x_agg})

    verified = all(per_edge_ok.values()) if per_edge_ok else None
    result = RoundResult(
        scheme="gpvo",
        completed=True,
        raw_sum=x_agg,
        verified=verified,
        n_selected=n_registered if n_registered is not None else sum(len(g) for g in session.groups),
        n_uploaded=n_uploaded,
        phases=sw.records,
        messages=msgs,
        extra={
            "per_edge_verified": {str(k): v for k, v in per_edge_ok.items()},
            "artifacts": {
                "edge": {str(k): {"x_agg_k": edge_state[k]["x_agg_k"],
                                  "c_k": edge_state[k]["c_k"],
                                  "h_k": edge_state[k]["h_k"]}
                         for k in edge_state if edge_state[k] is not None},
                "ta": {"xi": xi, "h_ta": h_ta, "r": r},
                "cloud": {"x_agg": x_agg, "c_agg": c_agg},
            },
            "init_ms": session.init_ms,
        },
    )
    return result
