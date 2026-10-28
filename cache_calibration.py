"""Plaintext oracle for commit-driven cached update calibration.

This is an experimental state machine, not a secure protocol. Per-client
updates are visible here only so that the proposed invariants can be checked.
"""

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class PendingPackage:
    token: str
    base_version: int
    deltas: dict
    cache_sequences: dict


@dataclass(frozen=True)
class Proposal:
    version: int
    tokens: tuple
    mode: str
    model: np.ndarray
    reference: np.ndarray
    diagnostics: dict
    proposal_id: str


def _proposal_id(version, tokens, mode, model, reference):
    digest = hashlib.sha256(repr((version, tokens, mode)).encode())
    for value in (model, reference):
        digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
    return digest.hexdigest()


class CacheCalibrationOracle:
    def __init__(self, initial_model, sample_sizes):
        self.model = np.asarray(initial_model, dtype=np.float64).copy()
        if self.model.ndim != 1 or not np.isfinite(self.model).all():
            raise ValueError("initial model must be a finite vector")
        self.sizes = dict(sample_sizes)
        if not self.sizes or any(n <= 0 for n in self.sizes.values()):
            raise ValueError("sample weights must be positive")
        self.total_mass = sum(self.sizes.values())
        self.reference = np.zeros_like(self.model)
        self.caches = {uid: np.zeros_like(self.model) for uid in self.sizes}
        self.cache_sequences = dict.fromkeys(self.sizes, 0)
        self.cache_base_versions = dict.fromkeys(self.sizes, -1)
        self.version = 0
        self.pending = {}
        self.pending_users = set()
        self.terminal_tokens = set()
        self.receipts = {}

    def stage(self, token, base_version, deltas):
        """Receiving a package stages updates without changing any cache."""
        if token in self.pending or token in self.terminal_tokens:
            raise ValueError("duplicate or terminal token")
        if not 0 <= base_version <= self.version or not deltas:
            raise ValueError("invalid package base or empty updates")
        copied = {}
        for uid, delta in deltas.items():
            if uid not in self.sizes or uid in self.pending_users:
                raise ValueError("unknown client or outstanding client update")
            vector = np.asarray(delta, dtype=np.float64)
            if vector.shape != self.model.shape or not np.isfinite(vector).all():
                raise ValueError("invalid delta")
            copied[uid] = vector.copy()
        self.pending[token] = PendingPackage(
            token, base_version, copied,
            {uid: self.cache_sequences[uid] for uid in copied})
        self.pending_users.update(copied)

    def discard(self, token):
        """Only definitively aborted/expired packages are discarded."""
        package = self.pending.pop(token, None)
        if package is None:
            return False
        self.pending_users.difference_update(package.deltas)
        self.terminal_tokens.add(token)
        return True

    def prepare(self, tokens, mode):
        """Compute a joint transition; rejection leaves it unapplied."""
        if mode not in ("increment", "calibration"):
            raise ValueError("unknown aggregation mode")
        tokens = tuple(tokens)
        if not tokens or len(set(tokens)) != len(tokens):
            raise ValueError("empty or repeated selected tokens")
        delta_sum = np.zeros_like(self.model)
        residual_sum = np.zeros_like(self.model)
        old_sum = np.zeros_like(self.model)
        mass = 0
        residual_square = 0.0
        ages = []
        age_weights = []
        unseen_mass = 0
        users = set()
        for token in tokens:
            if token not in self.pending:
                raise ValueError("selected token has no pending package")
            package = self.pending[token]
            for uid, delta in package.deltas.items():
                if uid in users or package.cache_sequences[uid] != self.cache_sequences[uid]:
                    raise ValueError("duplicate client or stale cache sequence")
                users.add(uid)
                n = self.sizes[uid]
                old = self.caches[uid]
                residual = delta - old
                mass += n
                delta_sum += n * delta
                residual_sum += n * residual
                old_sum += n * old
                residual_square += n * float(np.dot(residual, residual))
                base = self.cache_base_versions[uid]
                if base < 0:
                    unseen_mass += n
                else:
                    ages.append(self.version - base)
                    age_weights.append(n)
        correction = self.reference - old_sum / mass
        direction = (delta_sum / mass if mode == "increment"
                     else self.reference + residual_sum / mass)
        next_model = self.model + direction
        next_reference = self.reference + residual_sum / self.total_mass
        all_seen = [uid for uid, base in self.cache_base_versions.items() if base >= 0]
        diagnostics = {
            "selected_users": len(users), "selected_mass": mass,
            "cache_age_mean": (float(np.average(ages, weights=age_weights)) if ages else -1.0),
            "cache_age_max": max(ages, default=-1),
            "global_cache_age_mean": (float(np.average(
                [self.version - self.cache_base_versions[u] for u in all_seen],
                weights=[self.sizes[u] for u in all_seen])) if all_seen else -1.0),
            "selected_unseen_mass_fraction": unseen_mass / mass,
            "global_unseen_mass_fraction": sum(
                self.sizes[u] for u in self.sizes if self.cache_base_versions[u] < 0) / self.total_mass,
            "residual_rms_norm": float(np.sqrt(residual_square / mass)),
            "aggregate_residual_norm": float(np.linalg.norm(residual_sum / mass)),
            "correction_norm": float(np.linalg.norm(correction)),
            "direction_norm": float(np.linalg.norm(direction)),
            "correction_to_increment_ratio": float(np.linalg.norm(correction) /
                                                   (np.linalg.norm(delta_sum / mass) + 1e-12)),
        }
        proposal_id = _proposal_id(self.version, tokens, mode, next_model, next_reference)
        return Proposal(self.version, tokens, mode, next_model, next_reference,
                        diagnostics, proposal_id)

    def commit(self, proposal):
        """Atomically apply the plaintext transition; replay is idempotent.

        In the future protocol this point must follow TA finalization. This
        oracle has no signatures, TA, delayed notifications or durable recovery.
        """
        actual_id = _proposal_id(proposal.version, proposal.tokens, proposal.mode,
                                 proposal.model, proposal.reference)
        if actual_id != proposal.proposal_id:
            raise ValueError("proposal content changed")
        if proposal.proposal_id in self.receipts:
            return self.receipts[proposal.proposal_id]
        if proposal.version != self.version:
            raise ValueError("proposal is based on an uncommitted state")
        expected = self.prepare(proposal.tokens, proposal.mode)
        if expected.proposal_id != proposal.proposal_id:
            raise ValueError("joint model/reference transition mismatch")
        # Validate the entire candidate before changing any authoritative state.
        updates = [(package, uid, delta)
                   for token in proposal.tokens
                   for package in [self.pending[token]]
                   for uid, delta in package.deltas.items()]
        self.model = expected.model.copy()
        self.reference = expected.reference.copy()
        self.version += 1
        for package, uid, delta in updates:
            self.caches[uid] = delta.copy()
            self.cache_sequences[uid] += 1
            self.cache_base_versions[uid] = package.base_version
        for token in proposal.tokens:
            package = self.pending.pop(token)
            self.pending_users.difference_update(package.deltas)
            self.terminal_tokens.add(token)
        recomputed = np.zeros_like(self.reference)
        for uid, value in self.caches.items():
            recomputed += self.sizes[uid] * value / self.total_mass
        error = float(np.max(np.abs(recomputed - self.reference)))
        if not np.allclose(recomputed, self.reference, rtol=1e-10, atol=1e-12):
            raise RuntimeError("committed cache invariant violated")
        receipt = {"version": self.version, "tokens": proposal.tokens,
                   "invariant_max_abs_error": error}
        self.receipts[proposal.proposal_id] = receipt
        return receipt
