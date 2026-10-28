"""Control-evidence tests: independent updates and semantic forgery checks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
from experiments.point1_compare.training_control_contract import (
    design, exact_sum, load_source, source_arrays, SOURCE_SHA)
from experiments.point1_compare.validate_training_controls import check_payload, validate


def test_fixed_scope_and_exact_integer_sum():
    assert design()['seeds'] == [42, 43, 44]
    assert design()['rounds'] == 20 and design(True)['rounds'] == 1
    values = np.array([[10**12, -10**12, 7], [-10**12, 10**12, -8]], dtype=np.int64)
    assert exact_sum(values).tolist() == [0, 0, -1]


@pytest.fixture(scope='module')
def run():
    value = os.environ.get('POINT1_TRAINING_CONTROL_TEST_RUN')
    if not value:
        pytest.fail('Set POINT1_TRAINING_CONTROL_TEST_RUN to an actual completed smoke/formal run')
    return Path(value)


def test_actual_archive_and_independent_sgd_sample(run):
    result = validate(run)
    assert result['valid'], result
    assert result['predictions_recomputed'] == 2*result['paired_rounds']
    assert result['independent_sampled_sgd_clients'] == 2*result['paired_rounds']
    assert result['historical_manifest_sha256'] == SOURCE_SHA


@pytest.fixture
def evidence(run):
    rows = json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    # Last row lets formal checks exercise branch continuity after divergence.
    row = copy.deepcopy(rows[-1])
    with np.load(run/row['payload'], allow_pickle=False) as archive:
        a = {k:archive[k].copy() for k in archive.files}
    source, _, old = load_source()
    historical_row = old[(row['seed'], row['round'])]
    historical = source_arrays(source, historical_row)
    with np.load(run/'initial'/f'seed_{row["seed"]}.npz', allow_pickle=False) as archive:
        initial = archive['model']
    previous = None
    if row['round']:
        prevrow = next(x for x in rows if x['seed'] == row['seed'] and x['round'] == row['round']-1)
        with np.load(run/prevrow['payload'], allow_pickle=False) as archive:
            previous = {k:archive[k] for k in archive.files}
    return a, row, previous, historical, historical_row, initial


@pytest.mark.parametrize('mutation', ['quantization', 'float_mean', 'next_state', 'batch_seed',
    'prediction_metric', 'base_reset', 'historical_update', 'local_update_reuse'])
def test_reject_semantic_forgery(evidence, mutation):
    a, e, previous, historical, historical_row, initial = evidence
    if mutation == 'quantization': a['integers'][0, 0] += 1
    elif mutation == 'float_mean': a['float_mean'][0] += 1e-4
    elif mutation == 'next_state': a['float_next'][0] += .01
    elif mutation == 'batch_seed': e['batch_seeds'][0] += 1
    elif mutation == 'prediction_metric': e['prediction_disagreements'] += 1
    elif mutation == 'base_reset': a['float_base'][0] += .1
    elif mutation == 'historical_update': historical['vectors'][0, 0] += 1
    elif mutation == 'local_update_reuse':
        # On round zero equality is expected; perturb a separate local update.
        if e['round'] == 0: a['float_locals'][0, 0] += .01
        else: a['float_locals'] = a['encoded_locals'].copy()
    with pytest.raises(ValueError):
        check_payload(a, e, previous, historical, historical_row, initial)
