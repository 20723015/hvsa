"""Declared three-scheme E3/E4 supplement; real paired inputs and rosters.

This is a small formal supplement, not the original N50/N100,d7850 matrix.
Protocol adapters and their lifecycle choices remain frozen at the cost run.
"""
from __future__ import annotations
import argparse
import atexit
from collections import defaultdict
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from .adapters.common import RoundResult, partition_users, reference_sum, vector_digest
from .adapters.hvsa_point1 import HVSAProtocol
from .adapters.gpvo_decided import GPVODecidedProtocol
from .artifacts import create_run, digest, dump_json, finalize_run, round_rows, message_rows
from .run_compare import ROOT, _git_commit, _lib_versions, _new_run_id, _review_metadata
from .run_decided_comparison import SCHEMES, GPVO, VLMRS, DECISIONS
from .run_performance import stats
from .watchdog import RoundDeadline
from utils import LHH

ROLES = ['user', 'edge', 'ta', 'cloud']
from .trajectory_segment_contract import STAGE,PREFIX_COUNT


def specifications(n=20, m=5, d=1000, rounds=10, traces=10, seed=42, smoke=False):
    if n < 2*m or m < 1 or d < 1 or rounds < 1 or traces < 1:
        raise ValueError('Positive dimensions, rounds/traces, N>=2M required')
    settings = []
    for membership in ['fixed', 'changing']:
        settings.append(dict(setting='e3_'+membership, experiment='e3', membership=membership,
            stage='before_upload', rate=.2, count=rounds))
    for stage in ['before_upload', 'after_upload']:
        for rate in ([0., .3] if smoke else [0., .1, .2, .3]):
            settings.append(dict(setting='e4_'+stage+'_'+str(int(rate*100)), experiment='e4',
                membership='independent_traces', stage=stage, rate=rate, count=traces))
    return settings


def design(shape):
    rng = np.random.default_rng(shape['seed'])
    groups = [[f'u{i}' for i in group] for group in partition_users(rng, shape['n'], shape['m'], 2)]
    return groups, sorted(u for group in groups for u in group)


def case_inputs(shape, spec, index, users):
    # E3 pairs and all E4 cells share their respective full input pool at index.
    offset = 100000 if spec['experiment'] == 'e3' else 200000
    rng = np.random.default_rng(shape['seed']+offset+index)
    return {u: rng.integers(-10**9, 10**9, shape['d'], dtype=np.int64) for u in users}


def roster(shape, spec, index, users):
    trace_index = 0 if spec['membership'] == 'fixed' else index
    offset = 10000 if spec['experiment'] == 'e3' else 20000
    trace_seed = shape['seed']+offset+trace_index
    permutation = np.random.default_rng(trace_seed).permutation(users).tolist()
    missing = sorted(permutation[:int(shape['n']*spec['rate'])])
    recipients = sorted(set(users)-set(missing))
    uploaded = recipients if spec['stage']=='before_upload' else users
    return missing, sorted(uploaded), recipients, trace_seed


def expected_order(specs):
    order = []
    # Keep two E3 states alive together, alternating trajectory and scheme order.
    e3 = [s for s in specs if s['experiment']=='e3']
    for i in range(e3[0]['count']):
        for sp in (e3 if i%2==0 else e3[::-1]):
            offset = (i+specs.index(sp))%3
            for scheme in SCHEMES[offset:]+SCHEMES[:offset]:
                order.append([sp['setting'], i, scheme])
    # Dropout cells are paired within each independent trace, no resampling.
    e4 = [s for s in specs if s['experiment']=='e4']
    for i in range(e4[0]['count']):
        for sp in (e4 if i%2==0 else e4[::-1]):
            offset = (i+specs.index(sp))%3
            for scheme in SCHEMES[offset:]+SCHEMES[:offset]:
                order.append([sp['setting'], i, scheme])
    return order


def summarize(observations, specs):
    rows=[]
    for sp in specs:
        for scheme in SCHEMES:
            obs=[o for o in observations if o['setting']==sp['setting'] and o['scheme']==scheme]
            for metric in ['wall_ms','total_sent_bytes','user_ms','edge_ms','ta_ms','cloud_ms']:
                values=[o['role_ms'][metric[:-3]] if metric in [r+'_ms' for r in ROLES] else o[metric] for o in obs]
                rows.append(dict(setting=sp['setting'],scheme=scheme,metric=metric,**stats(values)))
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-export',required=True)
    args=p.parse_args(argv)
    from .validate_trajectory_segments import validate
    parent_root=Path(args.parent_export).resolve()
    problems=validate(parent_root)
    if problems:raise ValueError('Parent export invalid: '+repr(problems))
    parent=json.loads((parent_root/'manifest.json').read_text(encoding='utf-8'))
    assert parent['experiment']=='trajectory_prefix_export'
    shape=parent['shape']
    specs=specifications(**shape);groups,users=design(shape)
    import psutil
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            if proc.pid!=os.getpid() and (proc.info['name'] or '').lower() in ['python.exe','python']:
                command=proc.info['cmdline'] or []
                if any(c.startswith('experiments.point1_compare.run_') or c=='experiments.point1_compare.continue_decided_trajectories' for c in command):
                    raise ValueError('Another protocol worker is active: '+str(proc.pid))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    prior=Path(ROOT)/'results/point1_compare/20261020-212351-556444_decided_minimum_continuation/batch.json'
    frozen=json.loads(prior.read_text(encoding='utf-8'))['critical_source_sha256']
    for rel,sha in frozen.items():
        if digest(Path(ROOT)/rel)!=sha:raise ValueError('Frozen protocol changed: '+rel)
    sys.path.insert(0,str(Path(ROOT)/'tmp/point1_pairing_deps'))
    import py_ecc
    import gmpy2
    from .adapters.vlmrs_fullgraph import VLMRSFullGraphProtocol
    logging.disable(logging.CRITICAL)
    run_id=_new_run_id('trajectory_suffix')
    manifest=dict(run_id=run_id,experiment='trajectory_suffix',command=' '.join(sys.argv),
        seed=shape['seed'],schemes=SCHEMES,git_commit=_git_commit(),
        lib_versions={**_lib_versions(),'py_ecc':py_ecc.__version__,'gmpy2':gmpy2.version()},
        quant=dict(input='signed int64',scale=1000000,amplitude=10**9),
        protocol_mapping='experiments/point1_compare/protocol_mapping.md',
        paper_versions={GPVO:'declared integer LHF/fresh round session',VLMRS:'declared fullgraph/public exponent/key reuse'},
        concurrency='one worker; serial schemes; rotating order',timing_scope='local full-chain replay',
        verification='all designated verifiers, no sampled pairing checks',aggregation='exact uploaded integer sum',
        shape=shape,settings=specs,groups=groups,users=users,pid=os.getpid(),
        execution_status='running',execution_order=[],frozen_critical_source_sha256=frozen,
        decisions=json.loads(DECISIONS.read_text(encoding='utf-8')),decisions_sha256=digest(DECISIONS),
        round_timeout_seconds=1800,performance_qualified=False,paper_reproduction_qualified=False,
        artifact_kind='new_process_suffix_only',epoch_reset=True,
        parent_export=dict(path=parent_root.as_posix(),manifest_sha256=digest(parent_root/'manifest.json')),
        measurement_limitations=[
            'Small formal supplement N20/M5/d1000; not original N50/N100,d7850 full matrix.',
            'E3 is not rerun; suffix executes only E4 indices 8/9, 48 actual calls.',
            'Suffix uses original common E4 inputs/rosters at indices 8/9; retained prefix contains indices 0..7. No favorable resampling.',
            'Fresh initialization per E4 cell/scheme in this new epoch; original epoch 8 traces plus new epoch 2 traces. No continuous 10-round key-state claim.',
            'E4 before-upload dropout happens after VLMRS roster/share distribution.',
            'All failed rounds and partial work retained; thresholds differ from cryptographic rejection.',
            'Declared choices, backend and cache differences remain; not equal-security/SOTA ranking.'])
    if manifest['lib_versions']!=parent['lib_versions']:
        raise ValueError('Runtime versions changed since the retained prefix')
    manifest.update(_review_metadata(vars(args)))
    manifest['source_sha256'][STAGE]=digest(Path(ROOT)/STAGE)
    root=Path(ROOT)/'results/point1_compare'/run_id
    create_run(root,manifest,Path(ROOT))
    for folder in ['inputs','returned','checkpoints']:(root/folder).mkdir()
    print('RUN_DIR='+str(root),flush=True)
    raw=[];messages=[];observations=[];initialization=[];failures=[]
    def checkpoint():
        dump_json(root/'checkpoints/completed_prefix.json',dict(raw=raw,messages=messages,
            observations=observations,initialization=initialization,failures=failures))
        dump_json(root/'manifest.json',manifest)
    def stopped():
        if manifest['execution_status']=='running':
            manifest['execution_status']='interrupted_or_failed';checkpoint()
    atexit.register(stopped)
    def timeout(context):
        manifest['execution_status']='timed_out';manifest['termination']=context;checkpoint()
    states={}
    specs_by={s['setting']:s for s in specs}
    schedule=expected_order(specs)[PREFIX_COUNT:]
    for label,index,scheme in schedule:
        sp=specs_by[label];key=(label,scheme)
        if key not in states:
            if scheme=='hvsa':LHH._generator_for.cache_clear()
            with RoundDeadline(1800,lambda:timeout(dict(setting=label,scheme=scheme,phase='init'))):
                state=(HVSAProtocol(shape['d'],groups) if scheme=='hvsa' else
                    GPVODecidedProtocol(shape['d'],groups) if scheme==GPVO else
                    VLMRSFullGraphProtocol(shape['d'],users,threshold=math.ceil(2*shape['n']/3),root_exponent=7))
            states[key]=state
            setup=RoundResult(scheme,True,None,None,shape['n'],0,state.init_phases,state.init_messages)
            init_group=label+':'+scheme+':init'
            raw+=round_rows(setup,scheme=scheme,run_id=run_id,seed=shape['seed'],group=init_group,
                dimension=shape['d'],m=None if scheme==VLMRS else shape['m'],round_t=-1)
            messages+=message_rows(setup,scheme=scheme,run_id=run_id,group=init_group,round_t=-1)
            item=dict(setting=label,scheme=scheme,group=init_group,role_ms={r:setup.phase_ms(r) for r in ROLES},
                sent_bytes=sum(m.num_bytes for m in setup.messages))
            if scheme==GPVO:
                item['gpvo_parameters']=dict(N=state.params.n_mod,g=state.params.g,e=state.params.e,
                    maximal_order_certified=False)
            initialization.append(item);checkpoint()
        vectors=case_inputs(shape,sp,index,users)
        input_name=sp['experiment']+'_'+str(index)+'.npz';snapshot=root/'inputs'/input_name
        if not snapshot.exists():np.savez(snapshot,user_ids=np.array(users),vectors=np.stack([vectors[u] for u in users]))
        missing,uploaded,recipients,trace_seed=roster(shape,sp,index,users)
        reference=reference_sum([vectors[u] for u in uploaded])
        context=dict(setting=label,scheme=scheme,round=index)
        state=states[key];tick=time.perf_counter()
        with RoundDeadline(1800,lambda:timeout(context)):
            result=(state.run_round(vectors,set(uploaded),index,recovery_clients=set(recipients)) if scheme==VLMRS else
                    state.run_round(vectors,set(uploaded),index,model_recipients=set(recipients)))
        wall_ms=(time.perf_counter()-tick)*1000
        group=label+':'+scheme+':'+str(index)
        raw+=round_rows(result,scheme=scheme,run_id=run_id,seed=shape['seed'],group=group,
            dimension=shape['d'],m=None if scheme==VLMRS else shape['m'],round_t=index,repeat=index)
        messages+=message_rows(result,scheme=scheme,run_id=run_id,group=group,round_t=index)
        exact=bool(result.raw_sum is not None and np.array_equal(result.raw_sum,reference))
        output=root/'returned'/(group.replace(':','_')+'.npy')
        if result.raw_sum is not None:np.save(output,result.raw_sum,allow_pickle=False)
        actors=defaultdict(float);links=defaultdict(int)
        for phase in result.phases:actors[phase.role+':'+phase.actor]+=phase.elapsed_ms
        for packet in result.messages:links[packet.link]+=packet.num_bytes
        o=dict(**context,group=group,experiment=sp['experiment'],stage=sp['stage'],rate=sp['rate'],
            n_selected=result.n_selected,n_uploaded=result.n_uploaded,missing=missing,uploaded=uploaded,
            recipients=recipients,trace_seed=trace_seed,input_sha256=vector_digest(vectors),
            snapshot=snapshot.relative_to(root).as_posix(),snapshot_sha256=digest(snapshot),
            result_snapshot=output.relative_to(root).as_posix() if result.raw_sum is not None else None,
            result_snapshot_sha256=digest(output) if result.raw_sum is not None else None,
            completed=result.completed,exact_match=exact,verified=result.verified,
            valid=bool(result.completed and exact and result.verified is True),failure_reason=result.failure_reason,
            wall_ms=wall_ms,role_ms={r:result.phase_ms(r) for r in ROLES},actor_ms=dict(actors),
            link_bytes=dict(links),total_sent_bytes=sum(links.values()),
            verdicts=result.extra.get('verification_verdicts',result.extra.get('per_edge_verified',{})))
        if scheme==GPVO:
            o['session_public_sha256']=hashlib.sha256(b''.join(state.session_public_fingerprints[-1])).hexdigest()
        observations.append(o)
        if not o['valid']:failures.append(dict(**context,reason=result.failure_reason or 'honest-chain mismatch'))
        manifest['execution_order'].append([label,index,scheme]);checkpoint()
        print(f"SUFFIX {len(observations)}/{len(schedule)} {label} {index} {scheme} valid={o['valid']} wall_ms={wall_ms:.2f}",flush=True)
    correctness=dict(experiment=manifest['experiment'],observations=observations,initialization=initialization,
                     execution_errors=len(failures))
    finalize_run(root,manifest,raw,messages,correctness,summarize(observations,[s for s in specs if s['experiment']=='e4']),failures)
    problems=validate(root)
    if problems:raise ValueError('Archive reconciliation failed: '+repr(problems))
    print('COMPLETED='+str(root),flush=True)
    # Known availability failures are data, not execution errors.
    return int(any(o['completed'] and not o['valid'] for o in observations))


if __name__=='__main__':raise SystemExit(main())
