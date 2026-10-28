"""Fixed E7 control design and immutable historical HVSA evidence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'results/point1_compare/20261018-203010-459621_e7_hvsa_v2'
SOURCE_SHA = 'b50a882c9388287f709c14b660112d93d61b501037c622fc7071cba27bff978a'
STAGE = 'docs/POINT1_TRAINING_CONTROL_STAGE_20261026.md'
N, D, SCALE = 20, 7850, 1000000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def design(smoke=False):
    return dict(smoke=smoke, seeds=[42] if smoke else [42, 43, 44],
        rounds=1 if smoke else 20, clients=N, edges_in_historical_hvsa=5, d=D,
        model='Linear(784,10)', dataset='MNIST', train_size=60000, test_size=10000,
        lr=.01, local_epochs=1, batch_size=64, torch_threads=1, scale=SCALE,
        branches=['encoded_plaintext', 'independent_float64_mean'],
        participation='all clients; equal weights; IID equal partition',
        batch_seed_rule='seed*1000000+round*20+client',
        float_aggregation='own float32 local models -> float64 mean -> float32 next model')


def exact_sum(values):
    total = np.array([sum(int(x) for x in column) for column in values.T], dtype=np.int64)
    return total


def load_source(root=ROOT):
    p = Path(root)/SOURCE
    if sha(p/'manifest.json') != SOURCE_SHA:
        raise ValueError('Historical training manifest changed')
    m = json.loads((p/'manifest.json').read_text(encoding='utf-8'))
    for name, digest in m['output_sha256'].items():
        if sha(p/name) != digest:
            raise ValueError('Historical output changed: '+name)
    for name, digest in m['source_sha256'].items():
        if sha(p/'sources'/name) != digest:
            raise ValueError('Historical archived source changed: '+name)
    c = json.loads((p/'correctness.json').read_text(encoding='utf-8'))
    observations = {(o['seed'], o['round']): o for o in c['observations']}
    if len(observations) != 60 or len(c['observations']) != 60 or set(observations) != {
            (s, r) for s in [42, 43, 44] for r in range(20)}:
        raise ValueError('Historical training coverage differs')
    verifiers = {key: [] for key in observations}
    with (p/'raw.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['phase'] == 'edge_verify':
                if not (row['success'] is True and row['verified'] is True):
                    raise ValueError('Historical edge verification failed')
                verifiers[(row['seed'], row['round'])].append(row['actor'])
    for key, o in observations.items():
        if sorted(verifiers[key]) != [f'edge_{i}' for i in range(5)]:
            raise ValueError('Historical full-edge coverage differs')
        if not all(o[k] is True for k in ['valid', 'exact_match', 'verified']):
            raise ValueError('Historical round invalid')
        if sha(p/o['snapshot']) != o['snapshot_sha256']:
            raise ValueError('Historical snapshot changed')
    return p, m, observations


def source_arrays(source, observation):
    with np.load(source/observation['snapshot'], allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    if a['vectors'].dtype != np.int64 or a['vectors'].shape != (N, D):
        raise ValueError('Historical vector domain differs')
    if a['user_ids'].tolist() != [f'u{i}' for i in range(N)]:
        raise ValueError('Historical client order differs')
    expected = hashlib.sha256(exact_sum(a['vectors']).astype('<i8').tobytes()).hexdigest()
    if expected != observation['raw_sum_sha256']:
        raise ValueError('Historical integer sum differs')
    return a
