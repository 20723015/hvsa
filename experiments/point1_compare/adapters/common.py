"""Shared measurement scaffolding for point-1 protocol comparison adapters.

计量口径 (protocol_mapping.md §6): adapters return RoundResult with per-phase
timings and real serialized byte counts. No modeled network delay is charged
anywhere in this directory; wall-clock numbers are "本地协议回放耗时".
"""

from __future__ import annotations

import pickle
import hashlib
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

QUANTIZE_SCALE = 1_000_000  # real pipeline quantization (utils.QUANTIZE_SCALE)


def vector_digest(vectors) -> str:
    """Canonical participant-bound digest: sorted uid, NUL, little-endian int64."""
    h = hashlib.sha256()
    for uid in sorted(vectors):
        h.update(uid.encode() + b"\0")
        h.update(np.asarray(vectors[uid], dtype="<i8").tobytes())
    return h.hexdigest()


@dataclass
class PhaseRecord:
    phase: str
    role: str            # user | edge | ta | cloud
    elapsed_ms: float
    actor: str = ""      # concrete entity id, e.g. "edge_0" / "user 1001"
    detail: str = ""


@dataclass
class MessageRecord:
    link: str            # e.g. "user->edge", "edge->cloud", "ta->cloud"
    direction: str       # "send"
    phase: str
    payload_kind: str    # masked_upload | pre_aggregation | online_list | ...
    sender: str = ""
    receiver: str = ""
    num_bytes: int = 0
    note: str = ""


@dataclass
class RoundResult:
    scheme: str
    completed: bool
    raw_sum: Optional[np.ndarray]          # de-masked integer sum W_sum^raw
    verified: Optional[bool]               # None = verification not run
    n_selected: int
    n_uploaded: int
    phases: List[PhaseRecord] = field(default_factory=list)
    messages: List[MessageRecord] = field(default_factory=list)
    failure_reason: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def phase_ms(self, role: str) -> float:
        return sum(p.elapsed_ms for p in self.phases if p.role == role)

    def sent_bytes(self, link: str) -> int:
        return sum(m.num_bytes for m in self.messages
                   if m.link == link and m.direction == "send")


def serialize(payload: Any) -> bytes:
    """Real wire format used by the local implementation (pickle)."""
    return pickle.dumps(payload)


class Stopwatch:
    """Exclusive wall-clock costs: nested receivers/codecs are not charged twice."""

    def __init__(self) -> None:
        self.records: List[PhaseRecord] = []
        self._frames: List[dict] = []

    @property
    def current_role(self) -> str:
        return self._frames[-1]["role"] if self._frames else ""

    @property
    def current_actor(self) -> str:
        return self._frames[-1]["actor"] if self._frames else ""

    def time(self, phase: str, role: str, fn, actor: str = "", detail: str = ""):
        t0 = time.perf_counter()
        frame = {"role": role, "actor": actor, "children_ms": 0.0}
        self._frames.append(frame)
        try:
            return fn()
        finally:
            elapsed = (time.perf_counter() - t0) * 1000.0
            self._frames.pop()
            exclusive = max(0.0, elapsed - frame["children_ms"])
            self.records.append(PhaseRecord(phase, role, exclusive, actor, detail))
            if self._frames:
                self._frames[-1]["children_ms"] += elapsed


class WireLedger:
    """Serialized unicast protocol envelopes; no modeled network latency.

    A broadcast is explicitly one envelope per recipient. Setup and round
    ledgers are separate. These are bytes of the local protocol codec, not
    TCP/IP or TLS traffic measurements.
    """

    def __init__(self, stopwatch: Stopwatch):
        self.sw = stopwatch
        self.messages: List[MessageRecord] = []

    def capture(self, raw: bytes, *, link: str, phase: str, kind: str,
                sender: str, receiver: str, note: str = "") -> None:
        if not sender or not receiver or not raw:
            raise ValueError("A measured envelope needs sender, recipient and non-empty bytes")
        self.messages.append(MessageRecord(link, "send", phase, kind, sender,
                                           receiver, len(raw), note))

    def encode(self, payload, role: str, actor: str) -> bytes:
        return self.sw.time("serialize", role, lambda: serialize(payload), actor=actor)

    def transfer(self, payload, *, link: str, phase: str, kind: str,
                 sender: str, receiver: str, sender_role: str,
                 receiver_role: str):
        raw = self.encode(payload, sender_role, sender)
        self.capture(raw, link=link, phase=phase, kind=kind,
                     sender=sender, receiver=receiver)
        return self.sw.time("deserialize", receiver_role,
                            lambda: pickle.loads(raw), actor=receiver)


@contextmanager
def trace_entity_pickle(stopwatch: Stopwatch):
    """Time root entity codecs without modifying their source or global pickle.

    Only module-local ``pickle`` bindings are temporarily replaced. This
    harness is deliberately single-threaded; independent runs use processes.
    """
    import importlib

    class Proxy:
        def dumps(self, *args, **kwargs):
            role, actor = stopwatch.current_role, stopwatch.current_actor
            if not role:
                return pickle.dumps(*args, **kwargs)
            return stopwatch.time("serialize", role,
                                  lambda: pickle.dumps(*args, **kwargs), actor=actor)

        def loads(self, *args, **kwargs):
            role, actor = stopwatch.current_role, stopwatch.current_actor
            if not role:
                return pickle.loads(*args, **kwargs)
            return stopwatch.time("deserialize", role,
                                  lambda: pickle.loads(*args, **kwargs), actor=actor)

        def __getattr__(self, name):
            return getattr(pickle, name)

    saved = []
    try:
        for name in ("user", "edge", "ta", "cloud"):
            module = importlib.import_module(name)
            saved.append((module, module.pickle))
            module.pickle = Proxy()
        yield
    finally:
        for module, original in reversed(saved):
            module.pickle = original


def gen_client_vectors(rng: np.random.Generator, n_users: int, dim: int,
                       mix: Dict[str, float], amplitude: int) -> Dict[str, np.ndarray]:
    """E1 input generation: uniform / zero / boundary-heavy integer vectors.

    All values stay within the masked-domain amplitude so that per-dimension
    sums over N<=200 users cannot overflow int64 (reference sums additionally
    accumulated with Python ints and cross-checked)."""
    vecs: Dict[str, np.ndarray] = {}
    for i in range(n_users):
        kind = rng.choice(["uniform", "zero", "boundary"], p=[mix["uniform"], mix["zero"], mix["boundary"]])
        if kind == "zero":
            vec = np.zeros(dim, dtype=np.int64)
        elif kind == "boundary":
            signs = rng.choice([-1, 1], size=(dim,))
            # half the dims at the amplitude bound, the rest small negative values
            vals = np.where(rng.random(dim) < 0.5, amplitude, -rng.integers(1, 1000, size=dim))
            vec = (signs * vals).astype(np.int64)
            vec[rng.random(dim) < 0.2] *= -1
        else:
            vec = rng.integers(-amplitude, amplitude, size=dim, dtype=np.int64)
        vecs[f"u{i}"] = vec
    return vecs


def reference_sum(vectors: List[np.ndarray]) -> np.ndarray:
    """Plaintext reference: per-dimension integer sum with Python ints (no overflow)."""
    total = [0] * (vectors[0].shape[0] if vectors else 0)
    for v in vectors:
        for j, val in enumerate(v):
            total[j] += int(val)
    return np.array(total, dtype=np.int64)


def partition_users(rng: np.random.Generator, n_users: int, m_edges: int,
                    min_per_edge: int = 2) -> List[List[int]]:
    """Random user->edge assignment, every edge gets >= min_per_edge users."""
    if n_users < m_edges * min_per_edge:
        raise ValueError(f"n_users={n_users} too small for m_edges={m_edges}")
    perm = list(rng.permutation(n_users))
    base = n_users // m_edges
    rem = n_users % m_edges
    groups, idx = [], 0
    for k in range(m_edges):
        size = base + (1 if k < rem else 0)
        groups.append(perm[idx:idx + size])
        idx += size
    return groups
