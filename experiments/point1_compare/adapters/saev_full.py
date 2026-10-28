"""SAEV-FL CRT/full-verification diagnostic with explicit field choices.

This is not author code or a qualified paper benchmark. The multiplicative
HH group is the prime-order subgroup of FFDHE3072. Arbitrary positive blinders
are nonzero field elements; proof multiplication/unblinding is modulo p.
The same large p bounds CRT moduli. Signed values use centered decoding, with
a_i>2*N*amplitude. All ciphertext coordinates remain full Python integers.
"""
from __future__ import annotations
import math
import secrets
from functools import reduce
import numpy as np
from utils import PRG
from .common import Stopwatch, WireLedger, RoundResult
from .finite_field import FiniteFieldHH, FFDHE3072_P


class SAEVProtocol:
    def __init__(self, dimension, groups, amplitude=1000):
        self.dim, self.groups, self.amplitude = dimension, [list(g) for g in groups], amplitude
        users = [u for group in self.groups for u in group]
        if len(users) != len(set(users)) or not users or amplitude < 1:
            raise ValueError("Unique nonempty user roster and positive bound required")
        self.users = sorted(users)
        sw, self.parameters = Stopwatch(), {}
        wire = WireLedger(sw)
        self.hh = sw.time("HH_parameters", "ta", lambda: FiniteFieldHH(dimension), "TA")
        def create_crt():
            moduli = []
            for uid in self.users:
                while True:
                    value = secrets.randbelow(FFDHE3072_P//2-1)+FFDHE3072_P//2+1
                    if value > 2*len(users)*amplitude and all(math.gcd(value, old) == 1 for old in moduli):
                        moduli.append(value)
                        break
            product = math.prod(moduli)
            blinders = set()
            for uid, modulus in zip(self.users, moduli):
                a_i = product//modulus
                coefficient = a_i*pow(a_i, -1, modulus)
                while True:
                    blinder = secrets.randbelow(FFDHE3072_P-1)+1
                    if blinder not in blinders:
                        blinders.add(blinder)
                        break
                self.parameters[uid] = (modulus, coefficient, blinder)
        sw.time("CRT_parameters", "ta", create_crt, "TA")
        public = {"p": self.hh.p, "q": self.hh.q, "generators": self.hh.generators, "dimension": dimension}
        for uid in self.users:
            wire.transfer((self.parameters[uid], public), link="ta->user", phase="init", kind="secret_CRT_and_public_HH",
                          sender="TA", receiver=uid, sender_role="ta", receiver_role="user")
        # HH parameters are public to the cloud as required by the diagnostic
        # attack; individual a_i/C_i/r_i and the TA's K/R are not delivered there.
        wire.transfer(public, link="ta->cloud", phase="init", kind="public_HH", sender="TA", receiver="CS",
                      sender_role="ta", receiver_role="cloud")
        self.init_phases, self.init_messages = sw.records, wire.messages
        self._used_rounds = set()

    def run_round(self, vectors, uploaded, round_t, attack=None):
        selected, uploaded = set(vectors), set(uploaded)
        if selected != set(self.users) or not uploaded or not uploaded <= selected or round_t in self._used_rounds:
            raise ValueError("Invalid roster or reused context")
        if attack not in (None, "result_only", "coordinated"):
            raise ValueError("Unknown diagnostic attack")
        for vector in vectors.values():
            if (not isinstance(vector, np.ndarray) or vector.shape != (self.dim,) or vector.dtype != np.dtype("int64")
                    or any(abs(int(x)) > self.amplitude for x in vector)):
                raise ValueError("Invalid input domain/bound")
        self._used_rounds.add(round_t)
        sw = Stopwatch()
        wire = WireLedger(sw)
        noises, local_uploads = {}, {}
        edge_packets = []
        online_lists = []
        p = self.hh.p
        for index, group in enumerate(self.groups):
            edge = f"edge_{index}"
            received = []
            for uid in sorted(uploaded & set(group)):
                def mask_and_prove():
                    seed_value = secrets.randbelow(p//2-1)+p//2+1
                    noise = PRG.mask_int(seed_value.to_bytes(384, "big"), "saev|"+uid, round_t, self.dim)
                    noises[uid] = noise
                    _, coefficient, blinder = self.parameters[uid]
                    ciphertext = [int(x)+coefficient*int(mask) for x, mask in zip(vectors[uid], noise)]
                    proof = blinder*self.hh.hash(vectors[uid]) % p
                    return ciphertext, proof
                packet = sw.time("mask_and_prove", "user", mask_and_prove, uid)
                received.append(wire.transfer(packet, link="user->edge", phase="upload", kind="masked_vector_and_proof",
                                               sender=uid, receiver=edge, sender_role="user", receiver_role="edge"))
            online = sorted(uploaded & set(group))
            online_lists.append(wire.transfer(online, link="edge->ta", phase="collect", kind="online_roster",
                                             sender=edge, receiver="TA", sender_role="edge", receiver_role="ta"))
            def preaggregate():
                ciphertext = [sum(packet[0][i] for packet in received) for i in range(self.dim)]
                proof = reduce(lambda a, b: a*b % p, (packet[1] for packet in received), 1)
                return ciphertext, proof
            partial = sw.time("preaggregate", "edge", preaggregate, edge)
            edge_packets.append(wire.transfer(partial, link="edge->cloud", phase="aggregate", kind="ciphertext_and_proof_sum",
                                               sender=edge, receiver="CS", sender_role="edge", receiver_role="cloud"))
        def auxiliary():
            online = {u for group in online_lists for u in group}
            k = sum(self.parameters[u][1] for u in online)
            r = math.prod(self.parameters[u][2] for u in online)
            return k, r
        aid = sw.time("auxiliary_parameters", "ta", auxiliary, "TA")
        def aggregate():
            ciphertext = [sum(packet[0][i] for packet in edge_packets) for i in range(self.dim)]
            proof = reduce(lambda a, b: a*b % p, (packet[1] for packet in edge_packets), 1)
            return ciphertext, proof
        aggregate_ciphertext, aggregate_proof = sw.time("aggregate", "cloud", aggregate, "CS")
        if attack is not None:
            # Cloud-only transformation uses ciphertext, proof and public HH.
            def tamper():
                changed = aggregate_ciphertext.copy()
                changed[0] += 1
                proof = aggregate_proof
                if attack == "coordinated":
                    delta = [1]+[0]*(self.dim-1)
                    proof = proof*self.hh.hash(delta) % p
                return changed, proof
            aggregate_ciphertext, aggregate_proof = sw.time("coordinated_tamper", "cloud", tamper, "CS")
        recovered, verdicts = {}, {}
        for uid in sorted(uploaded):
            k, r = wire.transfer(aid, link="ta->user", phase="verify", kind="secret_auxiliary_parameters",
                                 sender="TA", receiver=uid, sender_role="ta", receiver_role="user")
            ciphertext, proof = wire.transfer((aggregate_ciphertext, aggregate_proof), link="cloud->user", phase="verify",
                                              kind="aggregate_ciphertext_and_proof", sender="CS", receiver=uid,
                                              sender_role="cloud", receiver_role="user")
            def decrypt():
                modulus, coefficient, _ = self.parameters[uid]
                values = [(k*c-coefficient*coefficient*int(mask)) % modulus for c, mask in zip(ciphertext, noises[uid])]
                return np.array([v-modulus if v > modulus//2 else v for v in values], dtype=np.int64)
            recovered[uid] = sw.time("decrypt", "user", decrypt, uid)
            verdicts[uid] = sw.time("verify", "user", lambda: self.hh.hash(recovered[uid]) == proof*pow(r, -1, p) % p, uid)
        representative = recovered[sorted(uploaded)[0]]
        agree = all(np.array_equal(v, representative) for v in recovered.values())
        return RoundResult("saev_ffdhe3072_diagnostic", True, representative, all(verdicts.values()),
                           len(selected), len(uploaded), sw.records, wire.messages,
                           extra={"all_recovered_vectors_equal": agree, "verification_verdicts": verdicts,
                                  "ciphertext_coordinate_max_bits": max(abs(x).bit_length() for x in aggregate_ciphertext),
                                  "public_field_bits": p.bit_length(), "qualified_paper_reproduction": False,
                                  "attack": attack})
