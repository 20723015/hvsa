"""Availability evidence must retain negative outcomes and complete verifiers."""
import json
import os
from pathlib import Path
import shutil
import pytest
from experiments.point1_compare.directed_dropout_contract import (
    specifications, tasks, SCHEMES, GPVO, VLMRS, expectation, expected_reveals)
from experiments.point1_compare.artifacts import dump_json, digest
from experiments.point1_compare.validate_directed_dropout import validate


def test_declared_boundaries_and_ring_cuts():
    specs = {s['case']:s for s in specifications()}
    assert len(tasks()) == len(set(tasks())) == 108
    assert sum(expectation(specs[case], scheme)[0] for case, _, scheme in tasks()) == 87
    assert expectation(specs['whole_before'], 'hvsa')[0] is False
    assert expectation(specs['whole_before'], GPVO)[0] is True
    assert expectation(specs['at_threshold_before'], VLMRS)[0] is True
    assert expectation(specs['below_threshold_before'], VLMRS)[0] is False
    assert [expected_reveals(specs[c]['uploaded']) for c in ['adjacent_before','alternating_before','spread_before']] == [2,4,4]


@pytest.fixture
def real_run():
    path = os.environ.get('POINT1_DIRECTED_RUN')
    if not path:
        pytest.skip('Actual completed diagnosis required for archive mutation tests')
    return Path(path)


def test_real_archive_reconciles(real_run):
    assert validate(real_run) == []


def _copy(source, target):
    # Hardlinks are read-only unless replaced atomically below; never edit linked bytes in place.
    shutil.copytree(source, target, copy_function=os.link)


@pytest.mark.parametrize('mutation', ['missing_verifier', 'false_recipient', 'hidden_unavailable',
    'changed_output', 'wrong_input', 'short_coverage', 'fake_completed'])
def test_rehashed_forgery_is_rejected(real_run, tmp_path, mutation):
    target = tmp_path/'run'
    _copy(real_run, target)
    corr = json.loads((target/'correctness.json').read_text(encoding='utf-8'))
    obs = corr['observations']
    honest = next(o for o in obs if o['scheme'] == 'hvsa' and o['case'] == 'control')
    if mutation == 'missing_verifier':
        honest['verdicts'].pop('edge_0')
    elif mutation == 'false_recipient':
        next(o for o in obs if o['scheme'] == 'hvsa' and o['case'] == 'all_after')['actual_recipients'] = ['u0']
    elif mutation == 'hidden_unavailable':
        (target/'failures.jsonl').unlink()
        (target/'failures.jsonl').write_text('', encoding='utf-8')
    elif mutation == 'changed_output':
        import numpy as np
        path = target/honest['output']
        actual = np.load(path, allow_pickle=False)
        actual[0] += 1
        temporary = path.with_suffix('.new.npy')
        np.save(temporary, actual, allow_pickle=False)
        temporary.replace(path)
        honest['output_sha256'] = digest(path)
    elif mutation == 'wrong_input':
        honest['snapshot'] = 'inputs/pool_1.npz'
        honest['snapshot_sha256'] = digest(target/honest['snapshot'])
    elif mutation == 'short_coverage':
        obs.pop()
    elif mutation == 'fake_completed':
        unavailable = next(o for o in obs if not o['completed'])
        unavailable.update(completed=True, verified=True, valid_aggregate=True)
    dump_json(target/'correctness.json', corr)
    manifest = json.loads((target/'manifest.json').read_text(encoding='utf-8'))
    manifest['output_sha256'] = {name:digest(target/name) for name in manifest['output_sha256']}
    dump_json(target/'manifest.json', manifest)
    assert validate(target), mutation
