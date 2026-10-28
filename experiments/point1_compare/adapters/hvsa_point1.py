"""Measured replay of the real point-1 HVSA entities.

Only the adapter is instrumented: root entity code and protocol checks stay
unchanged. Codec work, receiver handling and sender computation have separate
exclusive timers. All broadcasts have one serialized envelope per recipient.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Set

import numpy as np

from user import User
from edge import EdgeServer
from cloud import CloudServer
from ta import TrustedAuthority
from utils import KA, PRG, QUANTIZE_SCALE
from .common import RoundResult, Stopwatch, WireLedger, trace_entity_pickle

QUANTIZE_SCALE_RP1 = QUANTIZE_SCALE


class HVSAProtocol:
    def __init__(self, dim: int, groups: List[List[str]], enable_verification: bool = True,
                 mask_layers: int = 1):
        if mask_layers not in (1, 3):
            raise ValueError("mask_layers must be 1 or the E6 three-mask control")
        self.dim = dim
        self.groups = [list(g) for g in groups]
        self.enable_verification = enable_verification
        self.mask_layers = mask_layers
        self._neighbors = {}
        init_sw = Stopwatch()
        setup = WireLedger(init_sw)
        with trace_entity_pickle(init_sw):
            self.ta = init_sw.time("init", "ta", lambda: TrustedAuthority(
                gradient_dim=dim, ta_id="TA-01", enable_verification=enable_verification), actor="TA")
            self.cs = init_sw.time("init", "cloud", CloudServer, actor="CS")
            original_register = self.ta.register_edge_server

            def timed_register(*args, **kwargs):
                return init_sw.time("init_signing_key", "ta",
                                    lambda: original_register(*args, **kwargs), actor="TA")

            self.ta.register_edge_server = timed_register
            self.edges = []
            for k in range(len(groups)):
                eid = f"edge_{k}"
                es = init_sw.time("init", "edge", lambda eid=eid: EdgeServer(
                    id=eid, cloud_server=self.cs, ta=self.ta,
                    enable_verification=enable_verification, n_min=2, rule_engine=None), actor=eid)
                self.edges.append(es)
            self.ta.register_edge_server = original_register
            self.users = {}
            self._user_edge = {}
            for k, group in enumerate(self.groups):
                for uid in group:
                    user = init_sw.time("init", "user", lambda uid=uid: User(id=uid, model=None), actor=uid)
                    self.users[uid] = user
                    self._user_edge[uid] = self.edges[k]
                    ta_pub = setup.transfer(self.ta.get_public_key_bytes(), link="ta->user", phase="init",
                                            kind="ta_public_key", sender="TA", receiver=uid,
                                            sender_role="ta", receiver_role="user")
                    init_sw.time("key_agreement", "user", lambda user=user, p=ta_pub:
                                 user.register_with_ta(p), actor=uid)
                    public = setup.transfer(user.get_public_key_bytes(), link="user->ta", phase="init",
                                            kind="user_public_key", sender=uid, receiver="TA",
                                            sender_role="user", receiver_role="ta")
                    init_sw.time("key_agreement", "ta", lambda uid=uid, p=public:
                                 self.ta.register_user(uid, p), actor="TA")
            for es in self.edges:
                # These are the actual materials provisioned via the root TA API.
                setup.transfer({"signing_private_key": es.my_sign_priv_key,
                                "lhh_pp": self.ta.get_lhh_pp(),
                                "edge_public_keys": self.ta.get_all_es_public_keys()},
                               link="ta->edge", phase="init", kind="edge_provisioning",
                               sender="TA", receiver=es.id, sender_role="ta", receiver_role="edge")
            if mask_layers == 3:
                # E6 control: same root verification, plus two cancelling ring
                # streams. This is not a GPVO implementation or a privacy claim.
                for k, group in enumerate(self.groups):
                    for index, uid in enumerate(group):
                        peers = (group[(index-1) % len(group)], group[(index+1) % len(group)])
                        self._neighbors[uid] = (k, peers)
                        setup.transfer({p: self.users[p].get_public_key_bytes() for p in peers},
                                       link="ta->user", phase="init", kind="control_peer_public_keys",
                                       sender="TA", receiver=uid, sender_role="ta", receiver_role="user")
        self.init_phases = list(init_sw.records)
        self.init_messages = list(setup.messages)
        self.init_ms = sum(p.elapsed_ms for p in self.init_phases)
        self._sw = None
        self._wire = None
        self._origin = ""
        self._phase = ""
        self._verifying_edge = None
        self._wrap_receive(self.cs, "receive_pre_aggregation", "edge->cloud", "pre_aggregation_with_evidence", "cloud", "CS")
        self._wrap_receive(self.cs, "receive_global_mask", "ta->cloud", "global_mask", "cloud", "CS")
        self._wrap_receive(self.ta, "receive_online_list", "edge->ta", "online_list", "ta", "TA")
        for es in self.edges:
            self._wrap_receive(es, "receive_masked_gradient", "user->edge", "masked_upload", "edge", es.id)
        self._wrap_ta_aid("get_all_es_public_keys", "edge_public_keys")
        self._wrap_ta_aid("get_verification_info", "verification_aid")

    def _wrap_receive(self, obj, attr, link, kind, role, receiver):
        original = getattr(obj, attr)

        def wrapped(raw):
            self._wire.capture(raw, link=link, phase=self._phase, kind=kind,
                               sender=self._origin, receiver=receiver)
            return self._sw.time("receive", role, lambda: original(raw), actor=receiver)
        setattr(obj, attr, wrapped)

    def _wrap_ta_aid(self, attr, kind):
        original = getattr(self.ta, attr)

        def wrapped():
            value = original()
            if self._verifying_edge is None:
                return value
            return self._wire.transfer(value, link="ta->edge", phase="edge_verify", kind=kind,
                                       sender="TA", receiver=self._verifying_edge,
                                       sender_role="ta", receiver_role="edge")
        setattr(self.ta, attr, wrapped)

    def run_round(self, vectors: Dict[str, np.ndarray], online: Set[str], round_t: int,
                  base_model_vec: Optional[np.ndarray] = None,
                  n_registered: Optional[int] = None,
                  model_recipients: Optional[Set[str]] = None) -> RoundResult:
        if not online.issubset(self.users):
            raise ValueError("Online set contains unregistered users")
        if self.mask_layers == 3 and online != set(self.users):
            raise ValueError("E6 ring control requires all clients; no recovery is specified")
        recipients = online if model_recipients is None else model_recipients
        if not recipients.issubset(online):
            raise ValueError("Model recipients must have uploaded in this round")
        if any(np.asarray(vectors[u]).size != self.dim for u in online):
            raise ValueError("Wrong update dimension")
        sw = self._sw = Stopwatch()
        wire = self._wire = WireLedger(sw)
        selected = n_registered if n_registered is not None else len(self.users)
        if base_model_vec is None:
            base_model_vec = np.zeros(self.dim, dtype=np.float64)
        base_hash = hashlib.sha256(np.asarray(base_model_vec, dtype=np.float64).tobytes()).digest()
        with trace_entity_pickle(sw):
            sw.time("round_setup", "cloud", lambda: self.cs.start_new_round(
                round_t, self.dim, expected_edge_servers=len(self.edges)), actor="CS")
            sw.time("round_setup", "ta", lambda: self.ta.start_new_round(round_t), actor="TA")
            for es in self.edges:
                sw.time("round_setup", "edge", lambda es=es: es.start_new_round(
                    round_t, base_model_hash=base_hash), actor=es.id)
            for uid in sorted(online):
                self._origin, self._phase = uid, "user_mask"

                def mask(uid=uid):
                    x = np.asarray(vectors[uid], dtype=np.int64)
                    masked = x + self.users[uid]._generate_ta_mask_int(round_t, self.dim)
                    if self.mask_layers == 3:
                        k, peers = self._neighbors[uid]
                        for sign, peer in zip((-1, 1), peers):
                            key = KA.agree(self.users[uid]._ka_priv_key,
                                           self.users[peer].get_public_key_bytes())
                            low, high = sorted((uid, peer))
                            label = f"hvsa-e6-ring|{k}|{low}|{high}"
                            masked += sign * PRG.mask_int(key, label, round_t, self.dim)
                    return masked

                masked = sw.time("user_mask", "user", mask, actor=uid)
                raw = wire.encode({"user_id": uid, "round": round_t,
                                   "masked_gradient": masked}, "user", uid)
                self._user_edge[uid].receive_masked_gradient(raw)
            for es in self.edges:
                self._origin, self._phase = es.id, "edge_aggregate_forward"
                sw.time("edge_aggregate_forward", "edge", es.trigger_aggregation_and_forward, actor=es.id)
            self._origin, self._phase = "TA", "ta_global_mask"
            sw.time("ta_global_mask", "ta", lambda: self.ta.trigger_global_mask_computation(self.cs), actor="TA")
            _, raw_sum, evidence = sw.time("cloud_unmask", "cloud",
                                          self.cs.trigger_final_unmasking_and_averaging, actor="CS")
            if raw_sum is None:
                return RoundResult("hvsa", False, None, None, selected, len(online),
                                   list(sw.records), list(wire.messages),
                                   "cloud_barrier_or_mask_not_ready",
                                   {"init_ms": self.init_ms, "communication_complete": True})
            per_edge = {}
            models = {}
            try:
                for es in self.edges:
                    packet = wire.transfer({"round": round_t, "raw_sum": raw_sum, "evidence": evidence},
                                           link="cloud->edge", phase="edge_verify", kind="aggregate_and_evidence",
                                           sender="CS", receiver=es.id, sender_role="cloud", receiver_role="edge")
                    self._verifying_edge = es.id
                    ok = sw.time("edge_verify", "edge", lambda es=es, p=packet:
                                 es.verify_global_result(p["raw_sum"], p["evidence"]), actor=es.id)
                    self._verifying_edge = None
                    per_edge[es.id] = bool(ok)
                    if ok:
                        models[es.id] = sw.time("model_reconstruct", "edge", lambda p=packet:
                                               p["raw_sum"].astype(np.float64) / QUANTIZE_SCALE / len(online), actor=es.id)
                        for uid in self.groups[self.edges.index(es)]:
                            if uid in recipients:
                                wire.transfer({"round": round_t, "model": models[es.id]},
                                              link="edge->user", phase="model_publish", kind="verified_model",
                                              sender=es.id, receiver=uid, sender_role="edge", receiver_role="user")
            finally:
                self._verifying_edge = None
        verified = all(per_edge.values()) if self.enable_verification else None
        return RoundResult("hvsa", True, raw_sum, verified, selected, len(online),
                           list(sw.records), list(wire.messages), extra={
                               "per_edge_verified": per_edge, "init_ms": self.init_ms,
                               "n_online_reported": len(online), "communication_complete": True,
                               "timing_version": "exclusive_codec_receiver_v2",
                               "verification_enabled": self.enable_verification,
                               "mask_layers": self.mask_layers,
                               "model": next(iter(models.values()), None),
                           })
