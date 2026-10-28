"""Predeclared functional cases, independent of the measured result claims."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .artifacts import digest

N, M, D, SEED, REPEATS, THRESHOLD = 20, 5, 257, 42, 3, 14
GPVO, VLMRS = 'gpvo_zn2_fresh_prime_e', 'vlmrs_fullgraph_public_d'
SCHEMES = ['hvsa', GPVO, VLMRS]
ROLES = ['user', 'edge', 'ta', 'cloud']
GROUPS = [[f'u{i}' for i in range(k*4, k*4+4)] for k in range(M)]
USERS = sorted(u for g in GROUPS for u in g)
STAGE = 'docs/POINT1_DIRECTED_DROPOUT_STAGE_20261024.md'
DECISIONS = 'experiments/point1_compare/configs/baseline_decisions_20261019.json'
FREEZE = 'results/point1_compare/20261020-212351-556444_decided_minimum_continuation/batch.json'
FREEZE_SHA = 'cce46db69060fb4656d6e50e58349addb9254dd0ad8b49a56b2efddb6dc2d696'
SIX = ['u0', 'u4', 'u8', 'u12', 'u16', 'u1']
CASES = [
    ('control', 'before_upload', []),
    ('whole_before', 'before_upload', GROUPS[0]),
    ('whole_after', 'after_upload', GROUPS[0]),
    ('adjacent_before', 'before_upload', ['u0', 'u1']),
    ('alternating_before', 'before_upload', ['u0', 'u2']),
    ('spread_before', 'before_upload', ['u0', 'u4']),
    ('at_threshold_before', 'before_upload', SIX),
    ('at_threshold_after', 'after_upload', SIX),
    ('below_threshold_before', 'before_upload', SIX+['u5']),
    ('below_threshold_after', 'after_upload', SIX+['u5']),
    ('all_after', 'after_upload', USERS),
    ('none_before', 'before_upload', USERS),
]


def specifications():
    result = []
    for name, stage, missing in CASES:
        recipients = sorted(set(USERS)-set(missing))
        uploaded = recipients if stage == 'before_upload' else USERS
        result.append(dict(case=name, stage=stage, missing=sorted(missing),
                           uploaded=list(uploaded), recipients=recipients))
    return result


def tasks():
    result = []
    specs = specifications()
    for index in range(REPEATS):
        for sp in (specs if index % 2 == 0 else specs[::-1]):
            offset = (index+specs.index(sp)) % len(SCHEMES)
            for scheme in SCHEMES[offset:]+SCHEMES[:offset]:
                result.append((sp['case'], index, scheme))
    return result


def vectors(index):
    rng = np.random.default_rng(SEED+300000+index)
    return {u: rng.integers(-10**9, 10**9, D, dtype=np.int64) for u in USERS}


def round_id(case, index):
    return 1000000+index*len(CASES)+[x[0] for x in CASES].index(case)


def expectation(sp, scheme):
    counts = [len(set(g) & set(sp['uploaded'])) for g in GROUPS]
    if scheme == 'hvsa' and 0 in counts:
        return False, 'cloud_barrier_or_mask_not_ready'
    if scheme == GPVO and not sp['uploaded']:
        return False, 'no uploads'
    if scheme == VLMRS and len(sp['recipients']) < THRESHOLD:
        return False, 'insufficient recovery shares'
    return True, ''


def expected_reveals(uploaded):
    online = set(uploaded)
    count = 0
    for g in GROUPS:
        for i, uid in enumerate(g):
            if uid not in online:
                count += int(g[(i-1) % 4] in online)+int(g[(i+1) % 4] in online)
    return count


def frozen_sources(root):
    source = Path(root)/FREEZE
    if digest(source) != FREEZE_SHA:
        raise ValueError('Frozen source inventory changed')
    return json.loads(source.read_text(encoding='utf-8'))['critical_source_sha256']


def summary(observations):
    rows = []
    for sp in specifications():
        for scheme in SCHEMES:
            part = [o for o in observations if o['case'] == sp['case'] and o['scheme'] == scheme]
            rows.append(dict(case=sp['case'], scheme=scheme, calls=len(part),
                completed=sum(o['completed'] for o in part),
                valid_aggregate=sum(o['valid_aggregate'] for o in part),
                unavailable=sum(not o['completed'] for o in part),
                matches_expected=sum(o['matches_expected_outcome'] for o in part)))
    return rows
