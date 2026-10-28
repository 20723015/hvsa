import hashlib
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Union

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from ecdsa import NIST256p
from ecdsa.ellipticcurve import Point


QUANTIZE_SCALE = 1000000


def _hkdf_sha256(secret: bytes, length: int, info: bytes, salt: Optional[bytes] = None) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        info=info,
    ).derive(secret)


@dataclass(frozen=True)
class LHHPublicParams:
    """Public LHH parameters.

    The per-dimension generators are derived deterministically from this
    public seed, so we do not reuse a small fixed generator pool.
    """

    dim: int
    seed: bytes = b"HVSA-LHH-public-seed-v1"


class LHH:
    """
    Linear homomorphic hash over NIST P-256.

    Written in the multiplicative group-law notation used in the paper,
    H(m) = prod_i g_i^{m_i}. On this elliptic curve, each g_i^{m_i} is
    scalar multiplication and "prod" is point addition; LHH.Mul implements
    that group law, so LHH.Hash(a + b) == LHH.Mul(Hash(a), Hash(b)).

    Each per-dimension generator is derived by hash-to-curve
    (try-and-increment) from the public seed, i.e.
    g_i = HashToCurve("HVSA-LHH-generator-v1" || seed || i). The discrete log
    of g_i w.r.t. the base point G is therefore unknown to everyone (including
    the TA) in the random-oracle model, so collision resistance reduces to DLP
    while keeping O(1) parameter storage (no list of {g_i} is stored eagerly).
    """

    CURVE = NIST256p
    GENERATOR_DOMAIN = b"HVSA-LHH-generator-v1"

    @staticmethod
    def Gen(dim: int) -> LHHPublicParams:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"LHH.Gen requires a positive integer dimension, got {dim!r}")
        return LHHPublicParams(dim=dim)

    @staticmethod
    @lru_cache(maxsize=4096)
    def _generator_for(seed: bytes, index: int) -> Point:
        """hash-to-curve (try-and-increment) 派生第 index 个生成元.

        直接把 seed 哈希成曲线上一点, 该点的离散对数 (w.r.t. 基点 G) 在
        随机预言机模型下不可从公开种子派生。这与论文 Lemma 3.3 的 DLP 抗碰撞
        假设一致: 即使 `a_i` 不再被当作 TA 秘密, 敌手仍无法求 `g_i` 的离散对数。
        (旧实现把 seed 哈希成标量再乘 G, 等于公开了 `a_i`, 碰撞可平凡构造。)
        """
        curve = LHH.CURVE.curve
        p = curve.p()
        a = curve.a()
        b = curve.b()
        counter = 0
        while True:
            material = (
                LHH.GENERATOR_DOMAIN
                + len(seed).to_bytes(4, "big")
                + seed
                + int(index).to_bytes(16, "big", signed=False)
                + counter.to_bytes(8, "big")
            )
            x = int.from_bytes(hashlib.sha256(material).digest(), "big") % p
            # y² = x³ + a·x + b (mod p); P-256 满足 p ≡ 3 (mod 4)
            y2 = (pow(x, 3, p) + a * x + b) % p
            y = pow(y2, (p + 1) // 4, p)
            if y == 0 or (y * y) % p != y2:
                counter += 1
                continue
            return Point(curve, x, y, LHH.CURVE.order)

    @staticmethod
    def Hash(pp: Union[LHHPublicParams, list], vector: np.ndarray) -> Point:
        """Compute LHH.Hash(pp, vector) for an integer vector."""
        flat_vec = np.asarray(vector).reshape(-1)

        if isinstance(pp, LHHPublicParams):
            if flat_vec.size > pp.dim:
                raise ValueError(f"LHH input dimension {flat_vec.size} exceeds public dimension {pp.dim}")

            def get_generator(i: int) -> Point:
                return LHH._generator_for(pp.seed, i)
        else:
            # Compatibility for legacy tests that may pass an explicit generator list.
            if not pp:
                raise ValueError("LHH generator list cannot be empty")

            def get_generator(i: int) -> Point:
                if i >= len(pp):
                    raise ValueError("Legacy LHH generator list is shorter than the vector dimension")
                return pp[i]

        result_point = None
        for i, val in enumerate(flat_vec):
            scalar = int(val)
            if scalar == 0:
                continue

            generator = get_generator(i)
            if scalar < 0:
                term = -(generator * (-scalar))
            else:
                term = generator * scalar
            result_point = term if result_point is None else result_point + term

        if result_point is None:
            return LHH.CURVE.generator * 0
        return result_point

    @staticmethod
    def Mul(h1: Point, h2: Point) -> Point:
        """Group law of G (paper's multiplicative notation).

        LHH.Hash(a + b) == LHH.Mul(LHH.Hash(a), LHH.Hash(b)); on P-256 this
        is elliptic-curve point addition.
        """
        return h1 + h2

    @staticmethod
    def Exp(h: Point, scalar: int) -> Point:
        """H^k in multiplicative notation = k·H (EC 标量乘).

        用于加权验证: LHH.Hash(Σ_k w_k·x_k) == ∏_k LHH.Hash(x_k)^{w_k}。
        仅支持非负整数标量 (掩码域权重 w_k 为整数化结果)。
        """
        if not isinstance(scalar, int) or scalar < 0:
            raise ValueError(f"LHH.Exp requires a non-negative int scalar, got {scalar!r}")
        if scalar == 0:
            return LHH.CURVE.generator * 0
        return h * scalar


def staleness_int_weights(alphas, scale: int = 65536) -> list:
    """浮点陈旧度权重的掩码域整数化: w_k = round(α_k·scale).

    cloud (构造加权 M_Q / 加权聚合) 与 ES (独立复算比对) 必须共用本实现,
    否则整数化舍入差异会造成验证侧误拒。
    """
    return [int(round(float(a) * scale)) for a in alphas]


def staleness_alpha(metric: str, staleness: int, sigma: float,
                    dist: Optional[float] = None) -> float:
    """陈旧度补偿系数 α_k (公开确定性规则).

    仅 version (版本差 s_k=v−r_k) 与 rel2 (参数相对漂移) 可由 ES 依据
    共识量 (checkpoint 版本、签名绑定的 base_version、共识历史模型) 独立
    复算; wait 度量依赖云端缓冲调度状态, 不在本原语内 (策略绑定场景必须
    拒绝)。
    """
    if metric == 'version':
        return 1.0 / (staleness + 1.0) ** sigma
    if metric == 'rel2':
        if dist is None:
            raise ValueError("rel2 metric requires the parameter-drift distance")
        return math.exp(-sigma * dist)
    raise ValueError(f"staleness metric {metric!r} is not ES-recomputable")


class AlignmentSigmaPolicy:
    """Aσ-VCM 方向一致性 σ 档位选择 (公开规则, cloud 与 ES 共用同一实现).

    用前若干个具有非零补偿的 checkpoint 估计 cos(C^(0), Δtrain) 并一次锁定
    σ 档位; 全部输入均为公开量 (安全聚合后的 raw sum、共识版本历史、S_k、
    base_version、γ), 不读取单组明文。因此 ES 可在验证侧独立复算同一 σ_eff。

    两阶段语义 (与主循环/验证的时序对齐):
      - tentative_eff(cos): 输入本 checkpoint 的方向一致性 (无效轮传 None),
        返回本轮 σ_eff —— 探测期返回保守 σ, 达到 warmup 时立即按本次决策
        返回锁定档位; 不修改持久状态。
      - commit_observation(cos): checkpoint 成功提交后落地 EMA/探测/锁定
        状态。验证失败或未提交的候选不落地, 云端与 ES 状态不漂移。
    """

    def __init__(self, sigma_fallback: float, sigma_high: float,
                 ema_beta: float, warmup: int,
                 boost_threshold: float, mid_threshold: float):
        self.sigma_fallback = float(sigma_fallback)
        self.sigma_high = float(sigma_high)
        self.ema_beta = float(ema_beta)
        self.warmup = int(warmup)
        self.boost_threshold = float(boost_threshold)
        self.mid_threshold = float(mid_threshold)
        self.cos_ema: Optional[float] = None
        self.locked: Optional[float] = None
        self._probe_sum = 0.0
        self._probe_count = 0

    def tentative_eff(self, cos: Optional[float]) -> float:
        """本 checkpoint 的 σ_eff (含本次 cos 的试探性档位决策, 不落地)."""
        if self.locked is not None:
            return self.locked
        if cos is None:
            return self.sigma_fallback
        mean = (self._probe_sum + cos) / (self._probe_count + 1)
        if self._probe_count + 1 >= self.warmup:
            if mean >= self.boost_threshold:
                return self.sigma_high
            if mean >= self.mid_threshold:
                return 0.0
            return self.sigma_fallback
        return self.sigma_fallback

    def commit_observation(self, cos: Optional[float]) -> None:
        """checkpoint 成功提交后落地本次观测 (仅有效 cos 推进状态)."""
        if cos is None:
            return
        self.cos_ema = (cos if self.cos_ema is None
                        else self.ema_beta * self.cos_ema
                        + (1.0 - self.ema_beta) * cos)
        if self.locked is None:
            self._probe_sum += cos
            self._probe_count += 1
            if self._probe_count >= self.warmup:
                probe_mean = self._probe_sum / self._probe_count
                if probe_mean >= self.boost_threshold:
                    self.locked = self.sigma_high
                elif probe_mean >= self.mid_threshold:
                    self.locked = 0.0
                else:
                    self.locked = self.sigma_fallback


def vpda_weights(participation_counts, kappa=0.5, wmin=0.5, wmax=2.0,
                 scale=65536, all_counts=None):
    """V-PDA 逆参与权重 (公开规则, main/edge/测试共用同一实现).

    q_k = clip(( mean_j(A_j+1) / (A_k+1) )^κ, wmin, wmax)
    w_k = round(scale · q_k)   (整数化, 掩码域整数标量乘封闭)

    Args:
        participation_counts: 选中组的累计入选次数 A_k (当前 checkpoint 不计入).
        all_counts: 全部组的 A_j (含未选中组); 给定时均值在全部组上取
            (协议口径: 公平性基准是全体组, 而非被选中的子集)。缺省时
            退化为在 participation_counts 上取均值 (向后兼容)。

    Returns:
        (qs, ws): 浮点权重列表与整数权重列表 (与 participation_counts 一一对应)
    """
    n = len(participation_counts)
    if n == 0:
        return [], []
    ref = list(all_counts) if all_counts is not None else list(participation_counts)
    if not ref:
        return [], []
    mean_ap1 = sum(c + 1 for c in ref) / len(ref)
    qs = [min(wmax, max(wmin, (mean_ap1 / (c + 1.0)) ** kappa))
          for c in participation_counts]
    ws = [int(round(scale * q)) for q in qs]
    return qs, ws


class SIG:
    """Ed25519 signing helpers."""

    @staticmethod
    def gen(path: Optional[str] = None, nbits: Optional[int] = None) -> tuple[bytes, bytes]:
        del nbits  # Kept for backward-compatible call sites.
        priv_key = ed25519.Ed25519PrivateKey.generate()
        pub_key = priv_key.public_key()
        pub_bytes = pub_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        priv_bytes = priv_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

        if path is not None:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "pub.ed25519"), "wb") as f:
                f.write(pub_bytes)
            with open(os.path.join(path, "priv.ed25519"), "wb") as f:
                f.write(priv_bytes)

        return pub_bytes, priv_bytes

    @staticmethod
    def sign(msg: bytes, priv_key: bytes, hash_method: Optional[str] = None) -> bytes:
        del hash_method
        key = ed25519.Ed25519PrivateKey.from_private_bytes(priv_key)
        return key.sign(msg)

    @staticmethod
    def verify(msg: bytes, signature: bytes, pub_key: bytes) -> bool:
        try:
            key = ed25519.Ed25519PublicKey.from_public_bytes(pub_key)
            key.verify(signature, msg)
            return True
        except (InvalidSignature, ValueError, TypeError):
            return False


class AE:
    """AES-GCM authenticated encryption helpers kept for compatibility."""

    @staticmethod
    def gen(path: Optional[str] = None) -> tuple[bytes, bytes]:
        key = AESGCM.generate_key(bit_length=128)
        nonce = os.urandom(12)

        if path is not None:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "key"), "wb") as f:
                f.write(key)
            with open(os.path.join(path, "nonce"), "wb") as f:
                f.write(nonce)

        return key, nonce

    @staticmethod
    def encrypt(key: bytes, nonce: bytes, plaintext: bytes) -> bytes:
        return AESGCM(key).encrypt(nonce, plaintext, None)

    @staticmethod
    def decrypt(key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
        return AESGCM(key).decrypt(nonce, ciphertext, None)


class KA:
    """X25519 key agreement with HKDF-SHA256 seed derivation."""

    DOMAIN = b"HVSA-X25519-KA-v1"

    @staticmethod
    def gen() -> tuple[bytes, bytes]:
        priv_key = x25519.X25519PrivateKey.generate()
        pub_key = priv_key.public_key()
        pub_bytes = pub_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        priv_bytes = priv_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return pub_bytes, priv_bytes

    @staticmethod
    def agree(priv_key: bytes, pub_key: bytes) -> bytes:
        private = x25519.X25519PrivateKey.from_private_bytes(priv_key)
        public = x25519.X25519PublicKey.from_public_bytes(pub_key)
        shared_secret = private.exchange(public)
        return _hkdf_sha256(shared_secret, 32, KA.DOMAIN)


class PRG:
    """CSPRNG for additive integer masks.

    Key input: the user/TA shared seed s_i.
    Public context: domain label, user id, and round/window token.
    """

    DOMAIN = b"HVSA-mask-v1"
    MASK_LOW = -1_000_000_000
    MASK_HIGH = 1_000_000_000
    MASK_RANGE = MASK_HIGH - MASK_LOW
    UINT64_MODULUS = 1 << 64
    REJECTION_BOUND = UINT64_MODULUS - (UINT64_MODULUS % MASK_RANGE)

    @staticmethod
    def _context(user_id: str, round_token: Any) -> bytes:
        user_bytes = str(user_id).encode("utf-8")
        round_bytes = str(round_token).encode("utf-8")
        return (
            PRG.DOMAIN
            + len(user_bytes).to_bytes(4, "big")
            + user_bytes
            + len(round_bytes).to_bytes(4, "big")
            + round_bytes
        )

    @staticmethod
    def _derive_key_nonce(secret_seed: bytes, user_id: str, round_token: Any) -> tuple[bytes, bytes]:
        material = _hkdf_sha256(secret_seed, 48, PRG._context(user_id, round_token))
        return material[:32], material[32:48]

    @staticmethod
    def mask_int(secret_seed: bytes, user_id: str, round_token: Any, dim: int) -> np.ndarray:
        if secret_seed is None:
            raise ValueError("secret_seed is required")
        if not isinstance(dim, int) or dim < 0:
            raise ValueError(f"dim must be a non-negative integer, got {dim!r}")
        if dim == 0:
            return np.empty(0, dtype=np.int64)

        key, nonce = PRG._derive_key_nonce(secret_seed, user_id, round_token)
        encryptor = Cipher(algorithms.AES(key), modes.CTR(nonce)).encryptor()

        chunks = []
        remaining = dim
        while remaining > 0:
            # Rejection is extremely rare for a 2e9 range, so this is nearly exact.
            block_count = remaining + 8
            stream = encryptor.update(b"\x00" * (block_count * 8))
            raw = np.frombuffer(stream, dtype="<u8")
            accepted = raw[raw < PRG.REJECTION_BOUND]
            if accepted.size == 0:
                continue

            vals = (accepted % np.uint64(PRG.MASK_RANGE)).astype(np.int64)
            vals = vals + PRG.MASK_LOW
            take = min(remaining, vals.size)
            chunks.append(vals[:take])
            remaining -= take

        encryptor.finalize()
        return np.concatenate(chunks).astype(np.int64, copy=False)
