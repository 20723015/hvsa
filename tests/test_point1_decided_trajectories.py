"""Paired availability design and counterfactual evidence checks."""
import json
import os
from pathlib import Path
import shutil
import numpy as np
import pytest
from experiments.point1_compare.artifacts import digest,dump_json
from experiments.point1_compare.run_decided_trajectories import specifications,design,roster,case_inputs,expected_order
from experiments.point1_compare.validate_decided_trajectories import validate


def test_default_matrix_controls_counts_but_not_membership():
    shape=dict(n=20,m=5,d=1000,rounds=10,traces=10,seed=42,smoke=False)
    specs=specifications(**shape);_,users=design(shape)
    assert len(expected_order(specs))==300
    fixed=[roster(shape,specs[0],i,users)[1] for i in range(10)]
    changing=[roster(shape,specs[1],i,users)[1] for i in range(10)]
    assert fixed[0]==changing[0] and all(v==fixed[0] for v in fixed)
    assert len({tuple(v) for v in changing})>1 and all(len(v)==16 for v in fixed+changing)


def test_dropout_pairs_share_input_and_nested_trace_without_resampling():
    shape=dict(n=20,m=5,d=1000,rounds=10,traces=10,seed=42,smoke=False)
    specs=specifications(**shape);_,users=design(shape)
    for index in range(10):
        before=specs[2:6];after=specs[6:]
        previous=set()
        for b,a in zip(before,after):
            bm,bu,br,bs=roster(shape,b,index,users);am,au,ar,ss=roster(shape,a,index,users)
            assert bm==am and br==ar and bs==ss and previous<=set(bm)
            assert len(bu)==20-int(20*b['rate']) and len(au)==20 and len(ar)>=14
            assert all(np.array_equal(case_inputs(shape,b,index,users)[u],case_inputs(shape,a,index,users)[u]) for u in users)
            previous=set(bm)


@pytest.fixture
def actual_run(tmp_path):
    value=os.environ.get('POINT1_TRAJECTORIES_TEST_RUN')
    if not value:pytest.skip('Set POINT1_TRAJECTORIES_TEST_RUN to the actual smoke archive')
    source=Path(value);assert validate(source)==[]
    return shutil.copytree(source,tmp_path/'copied')


def test_actual_smoke_has_all_designated_verifiers(actual_run):
    assert validate(actual_run)==[]


@pytest.mark.parametrize('forgery',['count','roster','return','skip_verifier','omit_failure','order','summary'])
def test_rehashed_evidence_forgery_is_rejected(actual_run,forgery):
    mp=actual_run/'manifest.json';cp=actual_run/'correctness.json'
    m=json.loads(mp.read_text(encoding='utf-8'));c=json.loads(cp.read_text(encoding='utf-8'));o=c['observations'][0]
    if forgery=='count':o['n_uploaded']-=1
    elif forgery=='roster':o['recipients']=o['recipients'][:-1]
    elif forgery=='return':
        path=actual_run/o['result_snapshot'];a=np.load(path,allow_pickle=False);a[0]+=1
        np.save(path,a,allow_pickle=False);o['result_snapshot_sha256']=digest(path)
    elif forgery=='skip_verifier':o['verdicts'].pop(next(iter(o['verdicts'])))
    elif forgery=='omit_failure':c['execution_errors']+=1
    elif forgery=='order':m['execution_order']=m['execution_order'][::-1]
    else:
        path=actual_run/'summary.csv';s=path.read_text(encoding='utf-8');path.write_text(s.replace('wall_ms','invented_metric',1),encoding='utf-8')
    dump_json(cp,c);m['output_sha256']={name:digest(actual_run/name) for name in m['output_sha256']};dump_json(mp,m)
    assert validate(actual_run)
