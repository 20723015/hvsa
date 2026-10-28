"""Exact original E2 grid, three reusable points, and seven remaining points."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .adapters.common import partition_users
from .directed_dropout_contract import frozen_sources, FREEZE, FREEZE_SHA
from .round_shards import file_sha

ROOT=Path(__file__).resolve().parents[2]
STAGE='docs/POINT1_FULL_SCALE_SCAN_STAGE_20261026.md'
GPVO='gpvo_zn2_fresh_prime_e';VLMRS='vlmrs_fullgraph_public_d'
SCHEMES=['hvsa',GPVO,VLMRS];ROLES=['user','edge','ta','cloud']
REUSED=[(20,5,7850),(50,5,1000),(50,5,7850)]
PENDING=[(100,5,7850),(200,5,7850),(100,2,7850),(100,10,7850),
         (50,5,10000),(50,5,30000),(50,5,100000)]
POINTS=REUSED+PENDING
SCANS={'N':[(n,5,7850) for n in [20,50,100,200]],
       'd':[(50,5,d) for d in [1000,7850,10000,30000,100000]],
       'M':[(100,m,7850) for m in [2,5,10]]}


def schemes(point):
    return SCHEMES if point[1]==5 else SCHEMES[:2]


def shape(point,smoke=False):
    n,m,d=(10,2,4) if smoke else tuple(point)
    if not smoke and tuple(point) not in PENDING:raise ValueError('Not a remaining full-scale point')
    return dict(n=n,m=m,d=d,seed=42,smoke=smoke,warmup=1 if smoke else 3,
                repeats=2 if smoke else 10,schemes=SCHEMES if smoke else schemes(point))


def design(s):
    groups=[[f'u{i}' for i in g] for g in partition_users(np.random.default_rng(s['seed']),s['n'],s['m'],2)]
    return groups,sorted(u for g in groups for u in g)


def pool(s,index,users):
    rng=np.random.default_rng(s['seed']+index)
    return {u:rng.integers(-10**9,10**9,s['d'],dtype=np.int64) for u in users}


def order(s):
    ss=s['schemes']
    return [(i,scheme) for i in range(s['warmup']+s['repeats'])
            for scheme in ss[i%len(ss):]+ss[:i%len(ss)]]


def summaries(entries,s):
    result=[]
    for scheme in s['schemes']:
        obs=[e for e in entries if e['kind']=='round' and e['scheme']==scheme]
        measured=[e for e in obs if not e['warmup']]
        result.append(dict(label='honest',scheme=scheme,calls=len(obs),warmup_calls=sum(e['warmup'] for e in obs),
            measured_calls=len(measured),valid_calls=sum(e['valid'] for e in obs),
            measured_total_work_ms=sum(sum(e['role_ms'].values()) for e in measured),
            measured_total_sent_bytes=sum(e['total_sent_bytes'] for e in measured)))
    return result


def reused_inventory(validate_runs=True):
    from .validate_results import validate
    from .run_decided_batch import cost_scope_problems
    path=ROOT/FREEZE
    if file_sha(path)!=FREEZE_SHA:raise ValueError('Old cost inventory changed')
    batch=json.loads(path.read_text(encoding='utf-8'));frozen=frozen_sources(ROOT);result=[]
    for point in REUSED:
        job=next(j for j in batch['jobs'] if tuple(j['point'])==point)
        root=Path(job['run_dir'])
        if job['status']!='completed' or file_sha(root/'manifest.json')!=job['manifest_sha256']:
            raise ValueError('Reused run changed')
        m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        c=json.loads((root/'correctness.json').read_text(encoding='utf-8'))
        if validate_runs:
            errors=validate(root)+cost_scope_problems(m,c,point)
            if errors:raise ValueError(str(errors))
            for rel,h in frozen.items():
                if m['source_sha256'].get(rel)!=h or file_sha(root/'sources'/rel)!=h:
                    raise ValueError('Reused critical source differs: '+rel)
            actors={o['group']:Counter() for o in c['observations']}
            with (root/'raw.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line)
                    if row['phase'] in ['edge_verify','verify_all_equations']:
                        if row['success'] is not True or row['verified'] is not True:
                            raise ValueError('Reused verification failed')
                        actors[row['group']][row['actor']]+=1
            for o in c['observations']:
                want=m['users'] if o['scheme']==VLMRS else [f'edge_{i}' for i in range(point[1])]
                if actors[o['group']]!=Counter(want):raise ValueError('Reused designated verifier missing')
        result.append(dict(point=list(point),run_dir=str(root),manifest_sha256=job['manifest_sha256'],
            calls=39,warmup_calls=9,measured_calls=30,source='preserved completed original cost point'))
    return result
