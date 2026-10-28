"""Counterfactual trajectory and rehashed-evidence checks."""
import json
import os
from pathlib import Path
import shutil
import numpy as np
import pytest
from experiments.point1_compare.artifacts import digest,dump_json
from experiments.point1_compare.run_hvsa_paired40 import specifications,observe_results,validate,performance


def test_fixed_and_changing_have_equal_counts_and_explicit_trace_seeds():
    specs=specifications()
    assert len(specs)==2 and all(s['N']==50 and s['rounds']==50 and s['dropout_rate']==.2 for s in specs)
    assert specs[0]['trajectory_seeds']==[10042]*50
    assert specs[1]['participation']=='changing'
    with pytest.raises(ValueError):specifications(uploaded=50)
    with pytest.raises(ValueError):specifications(n=2,m=5)


def test_observer_does_not_replace_result_and_restores_on_failure(monkeypatch):
    result=object()
    class Stub:
        def run_round(self):return result
    monkeypatch.setattr(performance,'HVSAProtocol',Stub)
    active={'setting':'a'}
    with pytest.raises(RuntimeError):
        with observe_results(active) as states:
            state=performance.HVSAProtocol()
            assert state.run_round() is result and state.observed_result is result and states['a'] is state
            raise RuntimeError('test')
    assert performance.HVSAProtocol is Stub


@pytest.fixture
def actual_run(tmp_path):
    value=os.environ.get('POINT1_PAIRED40_TEST_RUN')
    if not value:pytest.skip('Set POINT1_PAIRED40_TEST_RUN to the actual smoke archive')
    path=Path(value);assert validate(path)==[]
    return shutil.copytree(path,tmp_path/'copied')


def test_actual_pairs_are_valid(actual_run):
    assert validate(actual_run)==[]


@pytest.mark.parametrize('forgery',['count','membership','input','input_domain','returned_sum','order'])
def test_rehashed_pair_forgeries_rejected(actual_run,forgery):
    mp=actual_run/'manifest.json';cp=actual_run/'correctness.json'
    m=json.loads(mp.read_text(encoding='utf-8'));c=json.loads(cp.read_text(encoding='utf-8'))
    o=c['observations'][0]
    if forgery=='count':o['n_uploaded']-=1
    elif forgery=='membership':o['unavailable']=['nonexistent']*len(o['unavailable'])
    elif forgery=='input':
        other=next(x for x in c['observations'] if x['round']!=o['round'])
        o['snapshot']=other['snapshot'];o['snapshot_sha256']=other['snapshot_sha256']
    elif forgery=='input_domain':
        path=actual_run/o['snapshot']
        with np.load(path,allow_pickle=False) as data:ids=data['user_ids'];vectors=data['vectors'].astype('float64')
        np.savez(path,user_ids=ids,vectors=vectors)
        for item in c['observations']:
            if item['snapshot']==o['snapshot']:item['snapshot_sha256']=digest(path)
    elif forgery=='returned_sum':
        path=actual_run/o['result_snapshot'];value=np.load(path,allow_pickle=False)
        value[0]+=1;np.save(path,value,allow_pickle=False);o['result_snapshot_sha256']=digest(path)
    else:m['execution_order']=list(reversed(m['execution_order']))
    dump_json(cp,c)
    m['output_sha256']={name:digest(actual_run/name) for name in m['output_sha256']}
    dump_json(mp,m)
    assert validate(actual_run)
