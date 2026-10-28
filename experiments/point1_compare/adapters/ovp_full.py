"""OVP Fig.3 full-chain diagnostic with coordinate-wise real GT tags.

Explicit instantiation: BLS12-381 source groups and DH; SHAKE-256 expansion
into F_q; AES-GCM for the paper's abstract AE interface (its experiments use
CBC/1024-bit parameters). Public pairing-base precomputation is used through
bilinearity, not a scalar tag surrogate. Every coordinate is actually tagged
and every online client executes every coordinate's GT verification.

Long-term DH keys and PRNG(shared key) streams are reused; this is a literal
scope choice, not a claim about safe multi-round privacy. Alpha and service
beta/sharing randomness are fresh. This is not a qualified paper benchmark.
"""
from __future__ import annotations
import hashlib
import pickle
import secrets
from functools import reduce
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from py_ecc.optimized_bls12_381 import G1, G2, FQ12, curve_order, multiply, pairing
from py_ecc.bls.point_compression import compress_G1, decompress_G1
from .common import Stopwatch, WireLedger, RoundResult

SCHEME = "ovp_coordinatewise_bls_diagnostic"
Q = curve_order


def gt_bytes(value):
    return b"".join(int(c).to_bytes(48, "big") for c in value.coeffs)


def gt_read(raw):
    if len(raw) != 576:
        raise ValueError("Expected twelve full-width BLS12-381 target-field coefficients")
    return FQ12([int.from_bytes(raw[i:i+48], "big") for i in range(0, 576, 48)])


def g1_bytes(point):
    return int(compress_G1(point)).to_bytes(48, "big")


def shared_key(private, public):
    point = decompress_G1(int.from_bytes(public, "big"))
    return hashlib.sha256(b"OVP-DH|"+g1_bytes(multiply(point, private))).digest()


def prng_vector(seed, dimension):
    raw = hashlib.shake_256(b"OVP-field-stream|"+seed).digest(64*dimension)
    return [int.from_bytes(raw[i:i+64], "big") % Q for i in range(0, len(raw), 64)]


def field_sum(vectors, dimension):
    total = [0]*dimension
    for vector in vectors:
        if len(vector) != dimension:
            raise ValueError("Field vector dimension mismatch")
        total = [(a+int(b)) % Q for a, b in zip(total, vector)]
    return total


def additive_shares(total, count):
    if count < 1:
        raise ValueError("Positive additive-sharing count required")
    shares = [[secrets.randbelow(Q) for _ in total] for _ in range(count-1)]
    previous = field_sum(shares, len(total))
    return shares+[[ (a-b) % Q for a, b in zip(total, previous) ]]


def cloud_transform(candidate, tags, opposite_sums, public_round_authentication, attack):
    """Only public/cloud inputs; cannot access either VS secret beta or input."""
    if attack not in (None, "result_only", "coordinated"):
        raise ValueError("Unknown OVP diagnostic attack")
    candidate, tags = list(candidate), list(tags)
    opposite_sums = [list(v) for v in opposite_sums]
    if attack:
        candidate[0] = (candidate[0]+1) % Q
        if attack == "coordinated":
            tags[0] *= public_round_authentication
            for part in opposite_sums:
                part[0] = (part[0]+1) % Q
    return candidate, tags, opposite_sums


class OVPProtocol:
    def __init__(self, dimension, users):
        self.dim, self.users = dimension, sorted(users)
        if dimension < 1 or len(set(self.users)) != len(self.users) or len(self.users) < 2:
            raise ValueError("Positive dimension and at least two unique users required")
        sw, self._used = Stopwatch(), set()
        wire = WireLedger(sw)
        self.base = sw.time("public_pairing_precomputation", "ta", lambda: pairing(G2, G1), "TA")
        self.keys = {}
        for actor in self.users+["VS1", "VS2"]:
            def keygen():
                private = secrets.randbelow(Q-1)+1
                return private, g1_bytes(multiply(G1, private))
            private, public = sw.time("DH_keygen", "ta", keygen, "TA")
            role = "edge" if actor.startswith("VS") else "user"
            self.keys[actor] = wire.transfer((private, public), link="ta->"+role, phase="init",
                kind="entity_keypair", sender="TA", receiver=actor, sender_role="ta", receiver_role=role)
        params = ("BLS12-381", Q, dimension, gt_bytes(self.base), "SHA256", "SHAKE256", "AES-GCM")
        for actor in self.users+["VS1", "VS2", "AS"]:
            role = "cloud" if actor == "AS" else ("edge" if actor.startswith("VS") else "user")
            wire.transfer(params, link="ta->"+role, phase="init", kind="public_parameters",
                          sender="TA", receiver=actor, sender_role="ta", receiver_role=role)
        for uid in self.users:
            for server in ["VS1", "VS2"]:
                wire.transfer((uid, self.keys[uid][1]), link="user->edge", phase="init", kind="user_public_key",
                              sender=uid, receiver=server, sender_role="user", receiver_role="edge")
                wire.transfer((server, self.keys[server][1]), link="edge->user", phase="init", kind="service_public_key",
                              sender=server, receiver=uid, sender_role="edge", receiver_role="user")
        self.init_phases, self.init_messages = sw.records, wire.messages

    def run_round(self, vectors, uploaded_to, round_t, attack=None):
        if round_t in self._used:
            raise ValueError("Round already used")
        if set(vectors) != set(self.users) or len(uploaded_to) != 2:
            raise ValueError("Invalid selected users or two-server upload sets")
        received_sets = [set(s) for s in uploaded_to]
        if any(not s <= set(self.users) for s in received_sets):
            raise ValueError("Unknown uploaded user")
        for vector in vectors.values():
            if not isinstance(vector, np.ndarray) or vector.shape != (self.dim,) or vector.dtype != np.dtype("int64"):
                raise ValueError("Expected canonical int64 input vector")
            if any(abs(int(v)) > 10**12 for v in vector):
                raise ValueError("Diagnostic signed-input bound exceeded")
        self._used.add(round_t)
        sw = Stopwatch()
        wire = WireLedger(sw)
        roles = {**{u: "user" for u in self.users}, "VS1": "edge", "VS2": "edge", "AS": "cloud"}
        def send(payload, sender, receiver, phase, kind):
            return wire.transfer(payload, link=roles[sender]+"->"+roles[receiver], phase=phase,
                kind=kind, sender=sender, receiver=receiver, sender_role=roles[sender], receiver_role=roles[receiver])
        label = b"OVP-round|"+str(round_t).encode()
        hashed = int.from_bytes(hashlib.sha256(label).digest(), "big") % Q
        received = [{}, {}]
        public_auth = sw.time("public_round_authentication", "cloud", lambda: self.base**hashed, "AS")
        user_auth = {}
        for uid in sorted(set.union(*received_sets)):
            keys = [sw.time("DH", "user", lambda s=s: shared_key(self.keys[uid][0], self.keys[s][1]), uid)
                    for s in ["VS1", "VS2"]]
            def mask():
                streams = [prng_vector(key, self.dim) for key in keys]
                alpha = [secrets.randbelow(Q) for _ in range(self.dim)]
                combined = field_sum(streams, self.dim)
                return [[(int(x)-a+m) % Q for x, a, m in zip(vectors[uid], alpha, combined)],
                        [(a+m) % Q for a, m in zip(alpha, combined)]]
            parts = sw.time("split_and_two_mask_streams", "user", mask, uid)
            auth = sw.time("round_authentication", "user", lambda: gt_bytes(self.base**hashed), uid)
            user_auth[uid] = sw.time("cache_round_authentication", "user", lambda: gt_read(auth), uid)
            for v, server in enumerate(["VS1", "VS2"]):
                if uid not in received_sets[v]:
                    continue
                plain = sw.time("serialize_inner", "user", lambda v=v: pickle.dumps((parts[v], auth)), uid)
                nonce = secrets.token_bytes(12)
                aad = label+uid.encode()+server.encode()
                ciphertext = sw.time("AE_encrypt", "user", lambda: AESGCM(keys[v]).encrypt(nonce, plain, aad), uid)
                n, c = send((nonce, ciphertext), uid, server, "upload", "encrypted_vector_part_and_authentication")
                key = sw.time("DH", "edge", lambda: shared_key(self.keys[server][0], self.keys[uid][1]), server)
                decrypted = sw.time("AE_decrypt", "edge", lambda: AESGCM(key).decrypt(n, c, aad), server)
                vector, authentication = sw.time("deserialize_inner", "edge", lambda: pickle.loads(decrypted), server)
                if len(vector) != self.dim or authentication != gt_bytes(public_auth):
                    raise ValueError("Invalid honest-upload context")
                received[v][uid] = vector
        lists = [send(sorted(received[v]), f"VS{v+1}", f"VS{2-v}", "online", "received_user_list") for v in [0, 1]]
        online = sorted(set(lists[0]) & set(lists[1]))
        if len(online) < 2:
            return RoundResult(SCHEME, False, None, None, len(self.users), len(online), sw.records,
                               wire.messages, "fewer than two complete uploaders", {"intersection": online})
        service_parts, service_betas = [], []
        cloud_packets = [[], []]
        for v, server in enumerate(["VS1", "VS2"]):
            streams = []
            for uid in online:
                key = sw.time("DH_mask_recovery", "edge", lambda: shared_key(self.keys[server][0], self.keys[uid][1]), server)
                streams.append(sw.time("PRNG_mask_recovery", "edge", lambda: prng_vector(key, self.dim), server))
            shares = sw.time("additive_sharing", "edge", lambda: additive_shares(field_sum(streams, self.dim), len(online)), server)
            beta = secrets.randbelow(Q-1)+1
            service_betas.append(beta)
            corrected = []
            for uid, share in zip(online, shares):
                part = sw.time("correct_input_part", "edge",
                               lambda: [(x-2*s) % Q for x, s in zip(received[v][uid], share)], server)
                tags = sw.time("coordinate_tags", "edge",
                               lambda: [gt_bytes(self.base**(((hashed+beta)*x) % Q)) for x in part], server)
                packet = send((uid, part, tags), server, "AS", "corrected_parts", "field_vector_and_coordinate_GT_tags")
                cloud_packets[v].append(packet)
                corrected.append(part)
            service_parts.append(field_sum(corrected, self.dim))
        def aggregate():
            parts = [field_sum((p[1] for p in packets), self.dim) for packets in cloud_packets]
            total = field_sum(parts, self.dim)
            tags = [reduce(lambda a, b:a*b, (gt_read(p[2][j]) for packets in cloud_packets for p in packets), FQ12.one())
                    for j in range(self.dim)]
            return total, tags, [parts[1], parts[0]]
        candidate, tags, opposite = sw.time("aggregate_field_vectors_and_tags", "cloud", aggregate, "AS")
        candidate, tags, opposite = sw.time("cloud_transform", "cloud",
            lambda: cloud_transform(candidate, tags, opposite, public_auth, attack), "AS")
        feedback, own_plus_other_checks = [], []
        for v, server in enumerate(["VS1", "VS2"]):
            claimed, cross = send((candidate, opposite[v]), "AS", server, "feedback_request", "claimed_total_and_opposite_sum")
            own_plus_other_checks.append(field_sum([service_parts[v], cross], self.dim) == claimed)
            beta = service_betas[v]
            feedback.append(sw.time("coordinate_feedback", "edge", lambda:
                ([gt_bytes(self.base**((beta*x) % Q)) for x in cross],
                 [gt_bytes(self.base**((beta*x) % Q)) for x in claimed]), server))
        verdicts, accepted_coordinates = {}, {}
        for uid in online:
            claimed, mac = send((candidate, [gt_bytes(t) for t in tags]), "AS", uid, "result", "result_and_coordinate_aggregate_tags")
            received_feedback = [send(feedback[v], f"VS{v+1}", uid, "feedback", "coordinate_service_feedback") for v in [0, 1]]
            def verify():
                outcomes = []
                for j, x in enumerate(claimed):
                    left = gt_read(mac[j])*gt_read(received_feedback[0][0][j])*gt_read(received_feedback[1][0][j])
                    right = (user_auth[uid]**x)*gt_read(received_feedback[0][1][j])*gt_read(received_feedback[1][1][j])
                    outcomes.append(left == right)
                return outcomes
            checks = sw.time("verify_every_coordinate", "user", verify, uid)
            verdicts[uid], accepted_coordinates[uid] = all(checks), sum(checks)
        centered = [x if x <= Q//2 else x-Q for x in candidate]
        if any(not -(1 << 63) <= x < (1 << 63) for x in centered):
            raise ValueError("Decoded result outside signed-int64 contract")
        return RoundResult(SCHEME, True, np.asarray(centered, dtype=np.int64), all(verdicts.values()),
                           len(self.users), len(online), sw.records, wire.messages,
                           "" if all(verdicts.values()) else "coordinate verification rejected",
                           {"intersection": online, "received_sets": [sorted(s) for s in received_sets],
                            "client_verdicts": verdicts, "accepted_coordinates": accepted_coordinates,
                            "tags_per_server": self.dim*len(online),
                            "own_plus_claimed_other_equals_claimed_total": own_plus_other_checks,
                            "own_sum_check_scope": "recorded diagnostic, not an added Fig.3 rejection rule",
                            "attacker_uses_service_betas": False, "attack": attack})
