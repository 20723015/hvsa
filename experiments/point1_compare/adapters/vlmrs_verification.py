"""Executable VLMRS equations 4-10 on real BLS12-381 groups.

This component is not the complete privacy/dropout protocol. The keyed linear
HF below is an explicit vector instantiation choice, not claimed author code.
The coordinated-scaling diagnostic does not need the HF/PRF keys and relies
only on the paper's homomorphism plus a public invertible root exponent.
"""
from __future__ import annotations
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from functools import reduce
import numpy as np
from py_ecc.optimized_bls12_381 import (G1, G2, Z1, Z2, add, multiply, pairing, eq, curve_order)


def _prf_pair(key, label):
    return [int.from_bytes(hmac.new(key, label+bytes([i]), hashlib.sha256).digest(), "big") % curve_order
            for i in (0, 1)]


@dataclass
class Proof:
    A: tuple
    B: tuple
    L: tuple
    Q: tuple
    omega: int = 1


class VLMRSVerification:
    """The original scalar d is called root_exponent to avoid dimension confusion."""
    def __init__(self, dimension, root_exponent=7):
        if dimension < 1 or root_exponent <= 0 or root_exponent % curve_order == 0:
            raise ValueError("Positive dimension and invertible root exponent required")
        self.dimension, self.root_exponent = dimension, root_exponent
        self._delta, self._rho = secrets.token_bytes(32), secrets.token_bytes(32)
        self._k1, self._k2 = secrets.token_bytes(32), secrets.token_bytes(32)
        self._weights = [int.from_bytes(hmac.new(self._delta, self._rho+i.to_bytes(8, "big"),
                                                 hashlib.sha256).digest(), "big") % curve_order
                         for i in range(dimension)]
        self._pairing_base = pairing(G2, G1)

    def _hf(self, vector):
        vector = np.asarray(vector)
        if vector.shape != (self.dimension,) or vector.dtype.kind != "i":
            raise ValueError("Expected signed integer vector of the public dimension")
        return sum(w*int(x) for w, x in zip(self._weights, vector)) % curve_order

    def _tag(self, user, label):
        gamma_j, v_j = _prf_pair(self._k1, user.encode())
        gamma, v = _prf_pair(self._k2, label)
        return (gamma_j*gamma+v_j*v) % curve_order

    def create(self, user, vector, label=b"sum"):
        hashed = self._hf(vector)
        exponent = (self._tag(user, label)-hashed)*pow(self.root_exponent, -1, curve_order) % curve_order
        return Proof(multiply(G1, hashed), multiply(G2, hashed),
                     multiply(G1, exponent), multiply(G2, exponent))

    @staticmethod
    def aggregate(proofs):
        proofs = list(proofs)
        if not proofs:
            raise ValueError("Cannot aggregate an empty proof set")
        return Proof(reduce(add, (p.A for p in proofs), Z1),
                     reduce(add, (p.B for p in proofs), Z2),
                     reduce(add, (p.L for p in proofs), Z1),
                     reduce(add, (p.Q for p in proofs), Z2))

    def verify(self, vector, proof, users, label=b"sum"):
        hashed = self._hf(vector)
        hash_match = eq(proof.A, multiply(G1, hashed)) and eq(proof.B, multiply(G2, hashed))
        a_h, g_b = pairing(G2, proof.A), pairing(proof.B, G1)
        l_h, g_q = pairing(G2, proof.L), pairing(proof.Q, G1)
        target = self._pairing_base**(sum(self._tag(user, label) for user in users) % curve_order)
        checks = {"recomputed_hash": bool(hash_match), "pairing_A_B": a_h == g_b,
                  "pairing_L_Q": l_h == g_q,
                  "label_equation": target == a_h*(l_h**self.root_exponent)}
        return all(checks.values()), checks


def coordinated_scale(vector, proof, factor, public_root_exponent):
    """Cloud-only transformation: no access to VLMRSVerification secret state."""
    correction = (1-factor)*pow(public_root_exponent, -1, curve_order) % curve_order
    transformed = Proof(multiply(proof.A, factor % curve_order), multiply(proof.B, factor % curve_order),
                        add(proof.L, multiply(proof.A, correction)),
                        add(proof.Q, multiply(proof.B, correction)), proof.omega)
    return np.asarray(vector)*factor, transformed
