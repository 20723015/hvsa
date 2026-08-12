import hashlib
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

    For dimension i, the generator is deterministically derived as
    H("HVSA-LHH-generator-v1" || seed || i) * G. This gives each model
    coordinate its own public generator without storing all points eagerly.
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
    def Add(h1: Point, h2: Point) -> Point:
        return h1 + h2


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
