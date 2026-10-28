"""Predeclared N100 E4: nested dropout rosters and paired input pools."""
from __future__ import annotations
import math
import numpy as np
from .adapters.common import partition_users

GPVO, VLMRS = 'gpvo_zn2_fresh_prime_e', 'vlmrs_fullgraph_public_d'
SCHEMES = ['hvsa', GPVO, VLMRS]
ROLES = ['user', 'edge', 'ta', 'cloud']
STAGE = 'docs/POINT1_LARGE_DROPOUT_STAGE_20261024.md'
LABELS = [f'{stage}_{rate}' for stage in ['before_upload','after_upload'] for rate in [0,10,20,30]]


def shape(smoke=False):
    return dict(n=10 if smoke else 100, m=2 if smoke else 5,
        d=4 if smoke else 7850, rounds=1 if smoke else 10,
        rates=[0,30] if smoke else [0,10,20,30], seed=42, smoke=smoke)


def labels(s):
    return [label for label in LABELS if int(label.rsplit('_',1)[1]) in s['rates']]


def design(s):
    groups = [[f'u{i}' for i in g] for g in partition_users(np.random.default_rng(s['seed']),s['n'],s['m'],2)]
    return groups, sorted(u for g in groups for u in g)


def pool(s,index,users):
    rng=np.random.default_rng(s['seed']+400000+index)
    return {u:rng.integers(-10**9,10**9,s['d'],dtype=np.int64) for u in users}


def roster(s,label,index,users):
    stage,rate=label.rsplit('_',1);rate=int(rate)
    trace_seed=s['seed']+40000+index
    missing=sorted(np.random.default_rng(trace_seed).permutation(users).tolist()[:s['n']*rate//100])
    recipients=sorted(set(users)-set(missing))
    uploaded=recipients if stage=='before_upload' else list(users)
    return missing,uploaded,recipients,trace_seed


def order(s):
    rows=[];cells=labels(s)
    for index in range(s['rounds']):
        for label in (cells if index%2==0 else cells[::-1]):
            offset=(index+cells.index(label))%len(SCHEMES)
            rows.extend((label,index,scheme) for scheme in SCHEMES[offset:]+SCHEMES[:offset])
    return rows


def expectation(s,groups,scheme,uploaded,recipients):
    if scheme=='hvsa' and any(not set(g)&set(uploaded) for g in groups):
        return False,'cloud_barrier_or_mask_not_ready'
    if scheme==GPVO and not uploaded:return False,'no uploads'
    if scheme==VLMRS and len(recipients)<math.ceil(2*s['n']/3):
        return False,'insufficient recovery shares'
    return True,''


def reveals(groups,uploaded):
    online=set(uploaded);count=0
    for g in groups:
        for i,u in enumerate(g):
            if u not in online:
                count+=int(g[(i-1)%len(g)] in online)+int(g[(i+1)%len(g)] in online)
    return count


def summaries(entries,s):
    rows=[]
    for label in labels(s):
        for scheme in SCHEMES:
            init=next(e for e in entries if e['kind']=='init' and e['label']==label and e['scheme']==scheme)
            obs=[e for e in entries if e['kind']=='round' and e['label']==label and e['scheme']==scheme]
            setup=sum(init['role_ms'].values());work=sum(sum(e['role_ms'].values()) for e in obs)
            rows.append(dict(label=label,scheme=scheme,calls=len(obs),completed=sum(e['completed'] for e in obs),
                valid_aggregates=sum(e['valid'] for e in obs),unavailable=sum(not e['completed'] for e in obs),
                matches_expected=sum(e['matches_expected'] for e in obs),initialization_ms=setup,
                round_work_ms=work,cumulative_work_ms=setup+work,
                cumulative_bytes=init['total_sent_bytes']+sum(e['total_sent_bytes'] for e in obs)))
    return rows
