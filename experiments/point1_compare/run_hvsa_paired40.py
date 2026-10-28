"""Paired HVSA trajectories: same inputs and contributor count, rotating order.

Reuses the frozen real performance generator. A passive subclass only retains
the returned result so inputs and raw sums can be archived outside round time.
"""
from __future__ import annotations
import argparse
import atexit
from contextlib import contextmanager
import csv
import json
import logging
import math
import os
from pathlib import Path
import sys
import numpy as np
from . import run_performance as performance
from .adapters.common import partition_users,gen_client_vectors,vector_digest,reference_sum
from .artifacts import create_run,digest,dump_json,finalize_run
from .run_compare import ROOT,_git_commit,_lib_versions,_new_run_id,_review_metadata
from .validate_results import validate as validate_base, _inside


def specifications(n=50,m=5,d=7850,uploaded=40,rounds=50,seed=42):
    if n<2*m or not 1<=uploaded<=n or d<1 or rounds<1:raise ValueError('Invalid positive trajectory shape')
    unavailable=n-uploaded
    if unavailable<1:raise ValueError('Choose fewer contributors than registered users')
    rate=unavailable/n
    if int(n*rate)!=unavailable:raise ValueError('Dropout rate must exactly encode contributor count')
    common={'N':n,'M':m,'d':d,'variant':'hvsa','mask_layers':1,'verification_enabled':True,
        'warmup':0,'rounds':rounds,'round_timeout_seconds':1800,'dropout_stage':'before_upload',
        'dropout_rate':rate,'prefix_cutoffs':[k for k in [1,10,50] if k<=rounds]}
    return [{**common,'setting':'fixed_count','participation':'dropout',
        'trajectory_seeds':[seed+10000]*rounds,'semantic_participation':'fixed_membership'},
        {**common,'setting':'changing_count','participation':'changing','semantic_participation':'changing_membership'}]


@contextmanager
def observe_results(active):
    original=performance.HVSAProtocol
    states={}
    class Observed(original):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);states[active['setting']]=self
        def run_round(self,*a,**kw):
            result=super().run_round(*a,**kw)
            self.observed_result=result
            return result
    performance.HVSAProtocol=Observed
    try:yield states
    finally:performance.HVSAProtocol=original


def validate(directory):
    root=Path(directory)
    problems=validate_base(root)
    if problems:return problems
    try:
        m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        c=json.loads((root/'correctness.json').read_text(encoding='utf-8'))
        expected=specifications(**m['paired_shape'])
        assert m['experiment']=='e3' and m['settings']==expected
        shape=m['paired_shape'];n=shape['n'];rounds=shape['rounds'];seed=shape['seed']
        rng=np.random.default_rng(seed)
        groups=[[f'u{i}' for i in group] for group in partition_users(rng,n,shape['m'],2)]
        users=sorted(u for group in groups for u in group)
        by={(o['setting'],o['round']):o for o in c['observations']}
        assert len(by)==2*rounds
        order=[]
        with (root/'messages.csv').open(encoding='utf-8',newline='') as f:messages=list(csv.DictReader(f))
        raw=[json.loads(line) for line in (root/'raw.jsonl').read_text(encoding='utf-8').splitlines() if line]
        initialization={i['specification']['setting']:i for i in c['initialization']}
        assert len(initialization)==2 and set(initialization)=={s['setting'] for s in expected}
        for label,item in initialization.items():
            total=sum(r['elapsed_ms'] for r in raw if r['group']==label and r['round']==-1)
            assert math.isclose(total,item['init_ms'],rel_tol=1e-9,abs_tol=1e-7)
            assert item['init_sent_bytes']==sum(int(r['num_bytes']) for r in messages if r['group']==label and int(r['round'])==-1)
        for index in range(rounds):
            vectors=gen_client_vectors(rng,n,shape['d'],{'uniform':1.,'zero':0.,'boundary':0.},1000000000)
            for sp in (expected if index%2==0 else list(reversed(expected))):
                order.append([index,sp['setting']]);o=by[sp['setting'],index]
                trace=np.random.default_rng(seed+10000+(index if sp['setting']=='changing_count' else 0))
                missing=sorted(trace.choice(users,size=n-shape['uploaded'],replace=False))
                assert o['unavailable']==missing and o['n_uploaded']==shape['uploaded']
                assert o['input_sha256']==vector_digest(vectors) and o['valid'] is True
                snapshot=_inside(root,o['snapshot']);assert digest(snapshot)==o['snapshot_sha256']
                with np.load(snapshot,allow_pickle=False) as data:
                    actual={str(u):v for u,v in zip(data['user_ids'],data['vectors'])}
                assert set(actual)==set(vectors)
                assert all(v.dtype==np.dtype('int64') and v.shape==(shape['d'],) for v in actual.values())
                assert vector_digest(actual)==vector_digest(vectors)
                result=_inside(root,o['result_snapshot']);assert digest(result)==o['result_snapshot_sha256']
                returned=np.load(result,allow_pickle=False)
                assert returned.shape==(shape['d'],) and returned.dtype==np.dtype('int64')
                assert np.array_equal(returned,reference_sum([vectors[u] for u in users if u not in missing]))
                recipients=sorted(x['receiver'] for x in messages if x['group']==o['group'] and x['payload_kind']=='verified_model')
                assert recipients==sorted(set(users)-set(missing))
        assert m['execution_order']==order,'Actual execution order differs from alternating pairs'
        fixed=[by['fixed_count',r]['unavailable'] for r in range(rounds)]
        assert all(x==fixed[0] for x in fixed)
        if rounds>1:assert len({tuple(by['changing_count',r]['unavailable']) for r in range(rounds)})>1
    except (AssertionError,KeyError,ValueError,OSError,TypeError) as error:return [type(error).__name__+': '+str(error)]
    return []


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registered',type=int,default=50);p.add_argument('--edges',type=int,default=5)
    p.add_argument('--uploaded',type=int,default=40);p.add_argument('--dimension',type=int,default=7850)
    p.add_argument('--rounds',type=int,default=50);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    shape={'n':args.registered,'m':args.edges,'d':args.dimension,'uploaded':args.uploaded,
           'rounds':args.rounds,'seed':args.seed}
    specs=specifications(**shape)
    import psutil
    workers={'experiments.point1_compare.run_decided_integrity','experiments.point1_compare.run_performance',
        'experiments.point1_compare.run_decided_comparison','experiments.point1_compare.run_hvsa_paired40'}
    for process in psutil.process_iter(['pid','name','cmdline']):
        if process.pid==os.getpid():continue
        try:
            if (process.info.get('name') or '').lower() in ['python.exe','python'] and workers.intersection(process.info.get('cmdline') or []):
                raise ValueError('Another protocol worker is active: '+str(process.pid))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    logging.disable(logging.CRITICAL)
    prior=Path(ROOT)/'results/point1_compare/20261020-212351-556444_decided_minimum_continuation/batch.json'
    old=json.loads(prior.read_text(encoding='utf-8'))
    for rel,expected in old['critical_source_sha256'].items():
        if digest(Path(ROOT)/rel)!=expected:raise ValueError('Frozen protocol changed: '+rel)
    run_id=_new_run_id('hvsa_paired_participation')
    m={'run_id':run_id,'experiment':'e3','command':' '.join(sys.argv),'seed':args.seed,'schemes':['hvsa'],
        'git_commit':_git_commit(),'lib_versions':_lib_versions(),'quant':{'scale':1000000},
        'protocol_mapping':'experiments/point1_compare/protocol_mapping.md','paper_versions':{'hvsa':'current real point-1 entities'},
        'concurrency':'serial alternating trajectory order each round','timing_scope':'local protocol replay',
        'verification':'all edges full verification','aggregation':'raw sum then canonical edge model',
        'settings':specs,'paired_shape':shape,'execution_status':'running','execution_order':[],'pid':os.getpid(),
        'measurement_limitations':['HVSA only; cross-scheme multi-round comparison still pending.',
            'Same inputs and contributor counts; local replay is not deployed network latency.',
            'Initialization once per trajectory, no warmup exclusion; actual cumulative prefix costs.']}
    m.update(_review_metadata(vars(args)))
    stage_relative='docs/POINT1_HVSA_PAIRED40_STAGE_20261023.md'
    m['source_sha256'][stage_relative]=digest(Path(ROOT)/stage_relative)
    m['measurement_limitations']=['HVSA only; cross-scheme multi-round comparison still pending.',
        'Same inputs and contributor counts; local replay is not deployed network latency.',
        'Initialization once per trajectory, no warmup exclusion; actual cumulative prefix costs.',
        'A passive subclass retains actual RoundResult references without changing packets or verdicts.']
    root=Path(ROOT)/'results/point1_compare'/run_id
    create_run(root,m,Path(ROOT))
    for name in ['inputs','returned','checkpoints']:(root/name).mkdir()
    print('RUN_DIR='+str(root),flush=True)
    def stopped():
        if m['execution_status']=='running':
            m['execution_status']='interrupted_or_failed';dump_json(root/'manifest.json',m)
    atexit.register(stopped)
    rng=np.random.default_rng(args.seed);partition_users(rng,args.registered,args.edges,2)
    generators={sp['setting']:performance.iter_setting(sp,m,root) for sp in specs}
    latest={};active={}
    with observe_results(active) as states:
        for index in range(args.rounds):
            vectors=gen_client_vectors(rng,args.registered,args.dimension,{'uniform':1.,'zero':0.,'boundary':0.},1000000000)
            snapshot=root/'inputs'/(str(index)+'.npz')
            np.savez(snapshot,user_ids=np.array(sorted(vectors)),vectors=np.stack([vectors[u] for u in sorted(vectors)]))
            for sp in (specs if index%2==0 else list(reversed(specs))):
                label=sp['setting'];active['setting']=label
                cp=next(generators[label]);result=states[label].observed_result
                output=root/'returned'/(label+'_'+str(index)+'.npy')
                np.save(output,result.raw_sum,allow_pickle=False)
                observation=cp['observations'][-1]
                if observation['input_sha256']!=vector_digest(vectors):raise ValueError('Pair input differs')
                observation.update(snapshot=snapshot.relative_to(root).as_posix(),snapshot_sha256=digest(snapshot),
                    result_snapshot=output.relative_to(root).as_posix(),result_snapshot_sha256=digest(output))
                latest[label]=cp;m['execution_order'].append([index,label])
                dump_json(root/'checkpoints'/(label+'.json'),cp)
                dump_json(root/'manifest.json',m)
            print('PAIRED_ROUND='+str(index+1)+'/'+str(args.rounds),flush=True)
    checkpoints=[latest[sp['setting']] for sp in specs];summary=[]
    for cp in checkpoints:
        for metric in ['wall_ms','total_sent_bytes','user_ms','edge_ms','ta_ms','cloud_ms']:
            values=[o['role_ms'][metric[:-3]] if metric in ['user_ms','edge_ms','ta_ms','cloud_ms'] else o[metric] for o in cp['observations']]
            summary.append({'setting':cp['specification']['setting'],'scheme':'hvsa','metric':metric,**performance.stats(values)})
    corr={'experiment':'e3','schemes':['hvsa'],'observations':[o for cp in checkpoints for o in cp['observations']],
        'initialization':[{k:cp[k] for k in ['specification','init_ms','init_sent_bytes','cache']} for cp in checkpoints]}
    failures=[f for cp in checkpoints for f in cp['failures']]
    finalize_run(root,m,[r for cp in checkpoints for r in cp['raw']],
                 [r for cp in checkpoints for r in cp['messages']],corr,summary,failures)
    problems=validate(root)
    if problems:raise ValueError(repr(problems))
    print('COMPLETED='+str(root),flush=True)
    return bool(failures)


if __name__=='__main__':raise SystemExit(main())
