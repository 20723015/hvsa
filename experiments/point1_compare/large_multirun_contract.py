"""Deterministic large-E3 design; counts and input pools are paired."""
from __future__ import annotations
import numpy as np
from .adapters.common import partition_users

GPVO, VLMRS = 'gpvo_zn2_fresh_prime_e', 'vlmrs_fullgraph_public_d'
SCHEMES = ['hvsa', GPVO, VLMRS]
ROLES = ['user', 'edge', 'ta', 'cloud']
STAGE = 'docs/POINT1_LARGE_MULTIRUN_STAGE_20261024.md'
LABELS = ['fixed', 'changing']


def shape(smoke=False):
    return dict(n=6 if smoke else 50, m=2 if smoke else 5,
        d=4 if smoke else 7850, uploaded=4 if smoke else 40,
        rounds=2 if smoke else 50, seed=42, smoke=smoke)


def design(s):
    groups = [[f'u{i}' for i in g] for g in partition_users(np.random.default_rng(s['seed']), s['n'], s['m'], 2)]
    return groups, sorted(u for g in groups for u in g)


def pool(s, index, users):
    rng = np.random.default_rng(s['seed']+100000+index)
    return {u:rng.integers(-10**9,10**9,s['d'],dtype=np.int64) for u in users}


def roster(s, label, index, users):
    trace_seed = s['seed']+10000+(index if label=='changing' else 0)
    missing = sorted(np.random.default_rng(trace_seed).permutation(users).tolist()[:s['n']-s['uploaded']])
    return missing, sorted(set(users)-set(missing)), trace_seed


def order(s):
    rows=[]
    for index in range(s['rounds']):
        for label in (LABELS if index%2==0 else LABELS[::-1]):
            offset=(index+LABELS.index(label))%len(SCHEMES)
            rows.extend((label,index,scheme) for scheme in SCHEMES[offset:]+SCHEMES[:offset])
    return rows


def summaries(entries,s):
    rows=[]
    for label in LABELS:
        for scheme in SCHEMES:
            init=next(e for e in entries if e['kind']=='init' and e['label']==label and e['scheme']==scheme)
            obs=[e for e in entries if e['kind']=='round' and e['label']==label and e['scheme']==scheme]
            for cutoff in [x for x in [1,10,50] if x<=s['rounds']]:
                prefix=obs[:cutoff]
                work=sum(init['role_ms'].values())+sum(sum(e['role_ms'].values()) for e in prefix)
                traffic=init['total_sent_bytes']+sum(e['total_sent_bytes'] for e in prefix)
                rows.append(dict(label=label,scheme=scheme,rounds=cutoff,
                    initialization_ms=sum(init['role_ms'].values()),
                    round_work_ms=work-sum(init['role_ms'].values()),cumulative_work_ms=work,
                    amortized_work_ms=work/cutoff,cumulative_bytes=traffic,
                    valid_aggregates=sum(e['valid'] for e in prefix)))
    return rows
