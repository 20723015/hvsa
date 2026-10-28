"""Literal VLMRS all-neighbour diagnostic, not a qualified paper benchmark.

Explicit choices: epsilon=1/all selected peers; threshold ceil(2N/3); one
shared keyed linear HF/PRF instance; public invertible d; X25519/AES-GCM;
32-byte secrets shared over 2^521-1; pickle envelopes. Long-term masking and
encryption keys are reused, fresh beta and polynomial coefficients each round.
The set-size check uses N>=threshold, consistent with the stated t-out-of-N
reconstruction (the printed algorithm's opposite inequality is not followed).
This exposes the unresolved paper key-reuse scope rather than silently fixing
it with fresh masking keys. Real sharing, encryption, masks, reconstruction,
proof aggregation and each remaining client's pairings are all executed.
"""
from __future__ import annotations
import math
import pickle
import secrets
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from utils import KA, PRG
from .common import Stopwatch, WireLedger, RoundResult
from .sharing import share_secret, reconstruct_secret
from .vlmrs_verification import VLMRSVerification, coordinated_scale


class VLMRSFullGraphProtocol:
    def __init__(self, dimension, users, threshold=None, root_exponent=7):
        self.dim = dimension
        self.users = sorted(users)
        if len(set(self.users)) != len(self.users) or len(self.users) < 2:
            raise ValueError("At least two unique users required")
        self.threshold = threshold if threshold is not None else math.ceil(2*len(self.users)/3)
        self.root_exponent = root_exponent
        if not 1 <= self.threshold <= len(self.users):
            raise ValueError("Invalid reconstruction threshold")
        sw = Stopwatch()
        wire = WireLedger(sw)
        self.mask_keys, self.encryption_keys = {}, {}
        for uid in self.users:
            self.mask_keys[uid] = sw.time("mask_keygen", "user", KA.gen, uid)
            self.encryption_keys[uid] = sw.time("encryption_keygen", "user", KA.gen, uid)
            wire.transfer((uid, self.mask_keys[uid][0], self.encryption_keys[uid][0]),
                          link="user->cloud", phase="init", kind="public_keys", sender=uid,
                          receiver="server", sender_role="user", receiver_role="cloud")
        self.verifier = sw.time("verification_setup", "ta",
                                lambda: VLMRSVerification(dimension, root_exponent), "TA")
        wire.transfer(("BLS12-381", dimension, root_exponent), link="ta->cloud", phase="init",
                      kind="public_verification_parameters", sender="TA", receiver="server",
                      sender_role="ta", receiver_role="cloud")
        # These values are delivered only to clients under the ideal secure
        # channel assumption. They are not returned in public diagnostic output.
        setup_secrets = (self.verifier._delta, self.verifier._rho,
                         self.verifier._k1, self.verifier._k2, root_exponent)
        for uid in self.users:
            wire.transfer(setup_secrets, link="ta->user", phase="init", kind="verification_keys",
                          sender="TA", receiver=uid, sender_role="ta", receiver_role="user")
        self.init_phases, self.init_messages = sw.records, wire.messages
        self.cloud_history, self.cloud_reconstructed_keys = {}, {}
        self._used_rounds = set()

    @staticmethod
    def _pair_label(a, b):
        return "vlmrs-pair|"+"|".join(sorted((a, b)))

    def run_round(self, vectors, uploaded, round_t, recovery_clients=None, scale_attack=None):
        selected = sorted(vectors)
        uploaded = set(uploaded)
        recovery = set(uploaded if recovery_clients is None else recovery_clients)
        if round_t in self._used_rounds:
            raise ValueError("Round context must not be reused")
        if not set(selected) <= set(self.users) or not uploaded <= set(selected) or not recovery <= uploaded:
            raise ValueError("Invalid selected/uploaded/recovery sets")
        for vector in vectors.values():
            if not isinstance(vector, np.ndarray) or vector.shape != (self.dim,) or vector.dtype != np.dtype("int64"):
                raise ValueError("Expected canonical signed int64 vector")
            if np.any(np.abs(vector.astype(object)) > 10**12):
                raise ValueError("Diagnostic amplitude bound exceeded")
        self._used_rounds.add(round_t)
        sw, wire = Stopwatch(), None
        wire = WireLedger(sw)
        if len(selected) < self.threshold:
            return RoundResult("vlmrs_fullgraph_public_d", False, None, None, len(selected), len(uploaded),
                               sw.records, wire.messages, "selected set below reconstruction threshold")
        def send(value, sender, recipient, phase, kind):
            sr, rr = ("cloud" if sender == "server" else "user"), ("cloud" if recipient == "server" else "user")
            return wire.transfer(value, link=sr+"->"+rr, phase=phase, kind=kind,
                                 sender=sender, receiver=recipient, sender_role=sr, receiver_role=rr)

        roster = {u: (self.mask_keys[u][0], self.encryption_keys[u][0]) for u in selected}
        held_shares = {u: {} for u in selected}
        beta_secrets = {}
        for uid in selected:
            send((round_t, roster, "sum"), "server", uid, "roster", "public_roster")
            beta = secrets.token_bytes(32)
            beta_secrets[uid] = beta
            b_shares = sw.time("share_beta", "user", lambda: share_secret(beta, self.threshold, len(selected)), uid)
            k_shares = sw.time("share_mask_key", "user",
                               lambda: share_secret(self.mask_keys[uid][1], self.threshold, len(selected)), uid)
            for recipient, b_share, k_share in zip(selected, b_shares, k_shares):
                if recipient == uid:
                    held_shares[recipient][uid] = (b_share, k_share)
                    continue
                payload = sw.time("serialize", "user",
                                  lambda: pickle.dumps((round_t, uid, recipient, b_share, k_share)), uid)
                nonce = secrets.token_bytes(12)
                enc_key = sw.time("share_DH", "user",
                                  lambda: KA.agree(self.encryption_keys[uid][1], roster[recipient][1]), uid)
                ciphertext = sw.time("share_AE_encrypt", "user", lambda: AESGCM(enc_key).encrypt(nonce, payload, None), uid)
                envelope = send((nonce, ciphertext), uid, "server", "shares", "encrypted_share")
                nonce_r, ciphertext_r = send(envelope, "server", recipient, "shares", "relayed_encrypted_share")
                dec_key = sw.time("share_DH", "user",
                                  lambda: KA.agree(self.encryption_keys[recipient][1], roster[uid][1]), recipient)
                plaintext = sw.time("share_AE_decrypt", "user",
                                    lambda: AESGCM(dec_key).decrypt(nonce_r, ciphertext_r, None), recipient)
                context, origin, target, bs, ks = sw.time("deserialize", "user", lambda: pickle.loads(plaintext), recipient)
                if (context, origin, target) != (round_t, uid, recipient):
                    raise ValueError("Share envelope context mismatch")
                held_shares[recipient][uid] = (bs, ks)

        masked_uploads, proofs = {}, {}
        for uid in sorted(uploaded):
            def make_upload():
                masked = vectors[uid] + PRG.mask_int(beta_secrets[uid], "vlmrs-beta|"+uid, round_t, self.dim)
                for peer in selected:
                    if peer != uid:
                        shared = KA.agree(self.mask_keys[uid][1], roster[peer][0])
                        stream = PRG.mask_int(shared, self._pair_label(uid, peer), round_t, self.dim)
                        masked = masked + (1 if uid < peer else -1)*stream
                proof = self.verifier.create(uid, vectors[uid])
                return masked, proof
            packet = sw.time("mask_and_prove", "user", make_upload, uid)
            masked_uploads[uid], proofs[uid] = send(packet, uid, "server", "upload", "masked_vector_and_proof")

        if len(recovery) < self.threshold or len(selected) < self.threshold:
            return RoundResult("vlmrs_fullgraph_public_d", False, None, None, len(selected), len(uploaded),
                               sw.records, wire.messages, "insufficient recovery shares",
                               {"threshold": self.threshold, "recovery_clients": sorted(recovery)})
        recovery_packets = {}
        missing = set(selected)-uploaded
        for uid in sorted(recovery):
            requests = {origin: ("beta", held_shares[uid][origin][0]) if origin in uploaded
                        else ("mask_key", held_shares[uid][origin][1]) for origin in selected}
            send((round_t, sorted(uploaded), sorted(missing)), "server", uid, "recovery", "recovery_request")
            recovery_packets[uid] = send(requests, uid, "server", "recovery", "recovery_shares")

        def unmask():
            total = np.sum(list(masked_uploads.values()), axis=0, dtype=np.int64)
            recovered_beta = {}
            for origin in selected:
                shares = [recovery_packets[u][origin][1] for u in sorted(recovery)]
                secret = reconstruct_secret(shares, self.threshold)
                if origin in uploaded:
                    recovered_beta[origin] = secret
                    total -= PRG.mask_int(secret, "vlmrs-beta|"+origin, round_t, self.dim)
                else:
                    self.cloud_reconstructed_keys[origin] = secret
                    for uid in sorted(uploaded):
                        shared = KA.agree(secret, roster[uid][0])
                        stream = PRG.mask_int(shared, self._pair_label(uid, origin), round_t, self.dim)
                        total -= (1 if uid < origin else -1)*stream
            self.cloud_history[round_t] = {"roster": roster, "masked_uploads": masked_uploads,
                                           "recovered_beta": recovered_beta}
            return total
        total = sw.time("reconstruct_and_unmask", "cloud", unmask, "server")
        aggregate_proof = sw.time("aggregate_proof", "cloud", lambda: self.verifier.aggregate(proofs.values()), "server")
        if scale_attack is not None:
            total, aggregate_proof = sw.time("coordinated_scaling", "cloud",
                lambda: coordinated_scale(total, aggregate_proof, scale_attack, self.root_exponent), "server")
        verdicts = {}
        for uid in sorted(recovery):
            candidate, proof = send((total, aggregate_proof), "server", uid, "verify", "aggregate_and_proof")
            verdicts[uid] = sw.time("verify_all_equations", "user",
                                    lambda: self.verifier.verify(candidate, proof, sorted(uploaded))[0], uid)
        return RoundResult("vlmrs_fullgraph_public_d", True, total, all(verdicts.values()),
                           len(selected), len(uploaded), sw.records, wire.messages,
                           extra={"threshold": self.threshold, "verification_verdicts": verdicts,
                                  "recovery_clients": sorted(recovery),
                                  "mask_private_keys_reconstructed": sorted(missing),
                                  "fullgraph": True, "qualified_paper_reproduction": False})

    def recover_past_upload_with_later_key(self, uid, past_round):
        """Cloud state only: no reference vectors or verifier secrets are read."""
        history = self.cloud_history[past_round]
        private_key = self.cloud_reconstructed_keys[uid]
        recovered = history["masked_uploads"][uid].copy()
        recovered -= PRG.mask_int(history["recovered_beta"][uid], "vlmrs-beta|"+uid, past_round, self.dim)
        for peer, (mask_public, _) in history["roster"].items():
            if peer != uid:
                stream = PRG.mask_int(KA.agree(private_key, mask_public), self._pair_label(uid, peer), past_round, self.dim)
                recovered -= (1 if uid < peer else -1)*stream
        return recovered
