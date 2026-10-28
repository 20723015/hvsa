"""Authorised GPVO instantiation: Z*_(N^2) LHF, fresh sessions, full wire costs.

This is a declared research reproduction, not author code or a claim that the
paper's P-256/commitment interface is uniquely determined. Public commitment
parameters are reused; DH keys and commitment randomisers are renewed each
round. No extra authenticated commitment channel is added. g is sampled
uniformly from units; maximal order is NOT certified. e is prime and > N^2.
"""
from __future__ import annotations
import math
import secrets
import numpy as np
import gmpy2
from .common import Stopwatch, WireLedger, RoundResult
from .gpvo_full import ComParams, GPVOSession, gpvo_user_mask, _pair_mask, _ta_mask, _lhf_generator, LHF_SEED

SCHEME = "gpvo_zn2_fresh_prime_e"


def parameter_gen(bits=3072):
    from cryptography.hazmat.primitives.asymmetric import rsa
    numbers = rsa.generate_private_key(public_exponent=65537, key_size=bits).private_numbers()
    p, q = int(numbers.p), int(numbers.q)
    n = p*q
    n2 = n*n
    phi = n*(p-1)*(q-1)
    e = int(gmpy2.next_prime(n2 + secrets.randbits(256)))
    while math.gcd(e, phi) != 1:
        e = int(gmpy2.next_prime(e))
    while True:
        g = secrets.randbelow(n2-2)+2
        if math.gcd(g, n) == 1:
            return ComParams(n, n2, g, e)


def hash_vector(params, vector):
    value = 1
    for i, x in enumerate(vector):
        if int(x):
            base = _lhf_generator(LHF_SEED, i, params.n2, params.n_mod)
            value = value*int(gmpy2.powmod(base, int(x), params.n2)) % params.n2
    return value


def commit(params, message, randomiser):
    return (int(gmpy2.powmod(message, params.e, params.n2))
            * int(gmpy2.powmod(params.g, randomiser, params.n2))) % params.n2


def verify_candidate(params, candidate, commitment, mask_hash, randomiser):
    if not isinstance(candidate, np.ndarray) or candidate.ndim != 1 or candidate.dtype != np.dtype("int64"):
        return False
    return commitment == commit(params, mask_hash*hash_vector(params, candidate) % params.n2, randomiser)


class GPVODecidedProtocol:
    def __init__(self, dimension, groups, rsa_bits=3072, params=None):
        self.dim, self.groups = dimension, [list(g) for g in groups]
        users = [u for g in self.groups for u in g]
        if dimension < 1 or any(len(g) < 2 for g in self.groups) or len(set(users)) != len(users):
            raise ValueError("Unique users, positive dimension and at least two users per ring required")
        self.users, self._used = set(users), set()
        _lhf_generator.cache_clear()
        sw = Stopwatch()
        wire = WireLedger(sw)
        self.params = params if params is not None else sw.time("public_parameter_generation", "ta",
                                                                lambda: parameter_gen(rsa_bits), "TA")
        public = {"n": self.params.n_mod, "g": self.params.g, "e": self.params.e,
                  "lhf_seed": LHF_SEED, "dimension": dimension, "backend": "gmpy2.powmod"}
        for recipient, role in [(f"edge_{i}", "edge") for i in range(len(groups))]+[("CS", "cloud")]:
            wire.transfer(public, link="ta->"+role, phase="init", kind="public_parameters", sender="TA",
                          receiver=recipient, sender_role="ta", receiver_role=role)
        self.init_phases, self.init_messages = sw.records, wire.messages
        self.init_ms = sum(p.elapsed_ms for p in sw.records)
        self.last_public_context = None
        self.session_public_fingerprints = []

    def run_round(self, vectors, uploaded, round_t, model_recipients=None):
        uploaded = set(uploaded)
        recipients = set(uploaded if model_recipients is None else model_recipients)
        if round_t in self._used or set(vectors) != self.users or not recipients <= uploaded <= self.users:
            raise ValueError("Invalid participant or reused round context")
        if any(v.shape != (self.dim,) or v.dtype != np.dtype("int64") for v in vectors.values()):
            raise ValueError("Canonical integer vectors required")
        self._used.add(round_t)
        sw = Stopwatch()
        wire = WireLedger(sw)
        # TA-side DH, shared-key generation and randomisers execute each round.
        session = sw.time("fresh_round_session", "ta", lambda: GPVOSession(
            self.dim, self.groups, com_params=self.params), "TA")
        self.session_public_fingerprints.append(tuple(session._user_pub[u] for u in sorted(self.users)))

        def send(value, sender, recipient, sr, rr, phase, kind):
            return wire.transfer(value, link=sr+"->"+rr, phase=phase, kind=kind, sender=sender,
                                 receiver=recipient, sender_role=sr, receiver_role=rr)
        for k, group in enumerate(self.groups):
            for uid in group:
                peers = session.ring_neighbors(k, uid)
                send({"round": round_t, "uid": uid, "private": session._user_priv[uid],
                      "public": session._user_pub[uid], "ta_public": session._ta_pub,
                      "neighbors": {p: session._user_pub[p] for p in peers}},
                     "TA", uid, "ta", "user", "round_init", "fresh_key_provisioning")
            send({"round": round_t, "tau": session._tau[k]}, "TA", f"edge_{k}", "ta", "edge",
                 "round_init", "fresh_commitment_randomiser")
        totals, commitments, active = [], [], []
        for k, group in enumerate(self.groups):
            online = sorted(set(group) & uploaded)
            missing = sorted(set(group)-uploaded)
            if not online:
                continue
            active.append(k)
            packets = []
            for uid in online:
                masked = sw.time("user_mask", "user", lambda uid=uid, k=k:
                                 gpvo_user_mask(session, k, uid, vectors[uid]), uid)
                packets.append(send({"round": round_t, "uid": uid, "vector": masked}, uid,
                                    f"edge_{k}", "user", "edge", "upload", "masked_upload")["vector"])
            total = sw.time("group_sum", "edge", lambda: np.sum(packets, axis=0, dtype=np.int64), f"edge_{k}")
            for uid in online:
                send((round_t, online, missing), f"edge_{k}", uid, "edge", "user", "dropout", "online_notice")
            # A missing user's left peer added its pair mask; right peer
            # subtracted it. Reveal computations belong to users, not edges.
            for missing_uid in missing:
                left, right = session.ring_neighbors(k, missing_uid)
                for peer, sign in [(left, -1), (right, 1)]:
                    if peer in uploaded:
                        mask = sw.time("dropout_mask", "user", lambda peer=peer, k=k, missing_uid=missing_uid:
                            _pair_mask(session.pair_key_user(peer, missing_uid), k, peer, missing_uid, self.dim), peer)
                        reveal = send((round_t, missing_uid, mask), peer, f"edge_{k}", "user", "edge",
                                      "dropout", "dropout_mask_reveal")[2]
                        total = sw.time("apply_dropout_correction", "edge", lambda: total+sign*reveal, f"edge_{k}")
            hashed = sw.time("group_hash", "edge", lambda: hash_vector(self.params, total), f"edge_{k}")
            evidence = sw.time("group_commitment", "edge", lambda: commit(self.params, hashed, session._tau[k]), f"edge_{k}")
            packet = send((round_t, total, evidence), f"edge_{k}", "CS", "edge", "cloud", "preaggregate", "group_sum_commitment")
            totals.append(packet[1]); commitments.append(packet[2])
            send((round_t, online), f"edge_{k}", "TA", "edge", "ta", "preaggregate", "online_list")
        if not active:
            return RoundResult(SCHEME, False, None, None, len(self.users), 0, sw.records, wire.messages, "no uploads")

        def ta_aid():
            masks = [_ta_mask(session._ta_shared[u], u, self.dim) for u in sorted(uploaded)]
            mask = np.sum(masks, axis=0, dtype=np.int64)
            return mask, hash_vector(self.params, mask), sum(session._tau[k] for k in active)
        mask, mask_hash, randomiser = sw.time("global_mask_hash_and_opening", "ta", ta_aid, "TA")
        cloud_mask = send((round_t, mask), "TA", "CS", "ta", "cloud", "ta_aid", "global_mask")[1]
        edge_aids = {}
        for k in range(len(self.groups)):
            edge_aids[k] = send((round_t, mask_hash, randomiser), "TA", f"edge_{k}", "ta", "edge", "ta_aid", "verification_aid")
        candidate = sw.time("cloud_unmask", "cloud", lambda: np.sum(totals, axis=0, dtype=np.int64)-cloud_mask, "CS")
        aggregate_commitment = sw.time("aggregate_commitments", "cloud", lambda: math.prod(commitments) % self.params.n2, "CS")
        verdicts = {}
        for k, group in enumerate(self.groups):
            value, proof = send((candidate, aggregate_commitment), "CS", f"edge_{k}", "cloud", "edge",
                                "verify", "aggregate_and_commitment")
            _, h, r = edge_aids[k]
            verdicts[str(k)] = sw.time("edge_verify", "edge", lambda: verify_candidate(self.params, value, proof, h, r), f"edge_{k}")
            if verdicts[str(k)]:
                for uid in sorted(set(group) & recipients):
                    send((round_t, value/len(uploaded)), f"edge_{k}", uid, "edge", "user", "publish", "verified_model")
        self.last_public_context = {"params": self.params, "candidate": candidate.copy(),
                                    "commitment": aggregate_commitment, "mask_hash": mask_hash, "randomiser": randomiser}
        return RoundResult(SCHEME, True, candidate, all(verdicts.values()), len(self.users), len(uploaded),
                           sw.records, wire.messages, extra={"per_edge_verified": verdicts,
                           "session_keys": "fresh_each_round", "qualified_paper_reproduction": False})
