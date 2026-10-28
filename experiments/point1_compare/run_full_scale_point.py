"""One E2 point: real frozen protocols, atomic round shards, 3+10 observations."""
from __future__ import annotations
import argparse
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
from .adapters.common import RoundResult,reference_sum,vector_digest
from .adapters.hvsa_point1 import HVSAProtocol
from .adapters.gpvo_decided import GPVODecidedProtocol
from . import artifacts,round_shards
from .artifacts import create_run,round_rows,message_rows
from .directed_dropout_contract import frozen_sources,DECISIONS
from .full_scale_contract import ROOT,STAGE,GPVO,VLMRS,ROLES,shape,design,pool,order,summaries
from .multirun_environment import snapshot as raw_environment,summarize as condition_summary
from .run_large_multirun import state_public_sha
from .run_compare import _git_commit,_lib_versions,_new_run_id,_review_metadata
from .windows_artifact_io import retry_writer
from .watchdog import RoundDeadline
from utils import LHH


def environment():
    frame=raw_environment()
    coordinator=int(os.environ.get('POINT1_FULL_SCALE_COORDINATOR_PID','0'))
    if coordinator and coordinator in frame['other_protocol_workers']:
        import psutil
        parent=psutil.Process(coordinator)
        if os.getppid()!=coordinator or 'experiments.point1_compare.run_full_scale_batch' not in parent.cmdline():
            raise ValueError('Unverified coordinator exclusion')
        frame['other_protocol_workers'].remove(coordinator)
    return frame


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point',default='100,5,7850')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(argv);s=shape(tuple(map(int,args.point.split(','))),args.smoke)
    groups,users=design(s);frozen=frozen_sources(ROOT)
    for rel,h in frozen.items():
        if round_shards.file_sha(ROOT/rel)!=h:raise ValueError('Frozen source changed: '+rel)
    if environment()['other_protocol_workers']:raise RuntimeError('Another protocol worker is active')
    sys.path.insert(0,str(ROOT/'tmp/point1_pairing_deps'))
    import py_ecc,gmpy2
    from .adapters.vlmrs_fullgraph import VLMRSFullGraphProtocol
    logging.disable(logging.CRITICAL)
    run_id=_new_run_id('full_scale_smoke' if args.smoke else 'full_scale_point')
    m=dict(run_id=run_id,experiment='full_scale_point',shape=s,groups=groups,users=users,
        schemes=s['schemes'],seed=s['seed'],pid=os.getpid(),git_commit=_git_commit(),
        lib_versions={**_lib_versions(),'py_ecc':py_ecc.__version__,'gmpy2':gmpy2.version()},
        stage=STAGE,planned_order=order(s),expected_rounds=len(order(s)),execution_status='running',
        frozen_critical_source_sha256=frozen,decisions_sha256=round_shards.file_sha(ROOT/DECISIONS),
        decisions=json.loads((ROOT/DECISIONS).read_text(encoding='utf-8')),round_timeout_seconds=1800,
        verification='all designated verifiers execute all checks',paper_reproduction_qualified=False,
        timing_scope='exclusive role work and local replay; setup, archive, reference separately',
        artifact_io_runtime_override=dict(modules=['artifacts','round_shards'],max_wait_seconds=3,
            affects_protocol_or_timers=False),state_epochs_per_scheme=1)
    m.update(_review_metadata(s));m['implementation_variants']={scheme:scheme for scheme in s['schemes']}
    m['measurement_limitations']=['Declared instances, not author/equal-security/SOTA certification.',
        'One key epoch per point/scheme, not independent key repetitions.',
        'Cross-point acquisition times differ; hardware temperature and frequency not locked.',
        'Synthetic int64 input; default zero HVSA base hash; not training convergence.',
        'Full checks executed in protocol; offline audit verifies records, sums and consistency.',
        'Backend, cache and trust facilities differ; local replay is not deployed network latency.']
    for rel in [STAGE,'tests/test_point1_full_scale.py']:
        m['source_sha256'][rel]=round_shards.file_sha(ROOT/rel)
    root=ROOT/'results/point1_compare'/run_id
    original=artifacts.dump_json
    def retried(path,value):
        def log(event):
            with (root/'io_retries.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(event)+'\n')
        return retry_writer(original,path,value,on_retry=log)
    artifacts.dump_json=round_shards.dump_json=retried
    create_run(root,m,ROOT)
    for folder in ['inputs','shards','checkpoints']:(root/folder).mkdir()
    journal=[];entries=[];retried(root/'journal.json',journal)
    print('RUN_DIR='+str(root),flush=True)
    context={}
    def progress():
        retried(root/'checkpoints/progress.json',dict(context=context,committed_entries=len(entries),
            completed_rounds=sum(e['kind']=='round' for e in entries),
            valid_rounds=sum(e.get('valid',False) for e in entries)))
    def stop(status,reason):
        m.update(execution_status=status,termination=dict(context=context,reason=reason))
        retried(root/'manifest.json',m);progress()
    states={}
    try:
        LHH._generator_for.cache_clear()
        for scheme in s['schemes']:
            context=dict(phase='initialization',scheme=scheme);progress()
            before=environment();tick=time.perf_counter()
            with RoundDeadline(1800,lambda:stop('timed_out','initialization deadline')):
                state=(HVSAProtocol(s['d'],groups) if scheme=='hvsa' else GPVODecidedProtocol(s['d'],groups)
                    if scheme==GPVO else VLMRSFullGraphProtocol(s['d'],users,threshold=math.ceil(2*s['n']/3),root_exponent=7))
            wall=(time.perf_counter()-tick)*1000;after=environment();states[scheme]=state
            result=RoundResult(scheme,True,None,None,s['n'],0,state.init_phases,state.init_messages)
            links=defaultdict(int)
            for packet in result.messages:links[packet.link]+=packet.num_bytes
            e=dict(kind='init',label='honest',scheme=scheme,round=-1,group=f'honest:{scheme}:init',
                state_public_sha256=state_public_sha(state,scheme,users),has_returned=False,
                completed=True,verified=None,failure_reason='',wall_ms=wall,
                role_ms={r:result.phase_ms(r) for r in ROLES},link_bytes=dict(links),
                total_sent_bytes=sum(links.values()),environment_before=before,environment_after=after)
            if scheme==GPVO:e['gpvo_parameters']=dict(N=state.params.n_mod,g=state.params.g,e=state.params.e,maximal_order_certified=False)
            raw=round_rows(result,scheme=scheme,run_id=run_id,seed=s['seed'],group=e['group'],dimension=s['d'],
                m=None if scheme==VLMRS else s['m'],round_t=-1,repeat=-1)
            wire=message_rows(result,scheme=scheme,run_id=run_id,group=e['group'],round_t=-1)
            entries.append(round_shards.commit(root,journal,e,raw,wire));progress()
            print(f'INITIALIZED={scheme} wall_s={wall/1000:.3f}',flush=True)
        snapshots={}
        for index,scheme in order(s):
            vectors=pool(s,index,users)
            if index not in snapshots:
                path=root/'inputs'/f'round_{index:03d}.npz'
                np.savez(path,user_ids=np.array(users),vectors=np.stack([vectors[u] for u in users]))
                snapshots[index]=dict(snapshot=path.relative_to(root).as_posix(),snapshot_sha256=round_shards.file_sha(path),input_sha256=vector_digest(vectors))
            reference=reference_sum([vectors[u] for u in users]);state=states[scheme]
            context=dict(phase='round',round=index,scheme=scheme,**snapshots[index]);progress()
            before=environment();tick=time.perf_counter()
            with RoundDeadline(1800,lambda:stop('timed_out','round deadline')):
                result=(state.run_round(vectors,set(users),index,recovery_clients=set(users)) if scheme==VLMRS else
                    state.run_round(vectors,set(users),index,model_recipients=set(users)))
            wall=(time.perf_counter()-tick)*1000;after=environment()
            if vector_digest(vectors)!=snapshots[index]['input_sha256']:raise ValueError('Shared input mutated')
            exact=bool(result.raw_sum is not None and np.array_equal(reference,result.raw_sum))
            verdicts=result.extra.get('verification_verdicts',result.extra.get('per_edge_verified',{}))
            verdicts={('edge_'+k if scheme==GPVO else k):bool(v) for k,v in verdicts.items()}
            links=defaultdict(int)
            for packet in result.messages:links[packet.link]+=packet.num_bytes
            e=dict(kind='round',label='honest',scheme=scheme,round=index,group=f'honest:{scheme}:{index}',
                **snapshots[index],warmup=index<s['warmup'],uploaded=users,recipients=users,
                n_selected=result.n_selected,n_uploaded=result.n_uploaded,
                group_upload_counts=list(map(len,groups)),state_public_sha256=state_public_sha(state,scheme,users),
                has_returned=result.raw_sum is not None,completed=result.completed,verified=result.verified,
                exact_match=exact,valid=bool(result.completed and exact and result.verified is True),
                failure_reason=result.failure_reason,verdicts=verdicts,wall_ms=wall,
                actual_recipients=sorted(p.receiver for p in result.messages if p.payload_kind==('aggregate_and_proof' if scheme==VLMRS else 'verified_model')),
                role_ms={r:result.phase_ms(r) for r in ROLES},link_bytes=dict(links),total_sent_bytes=sum(links.values()),
                environment_before=before,environment_after=after)
            if scheme==GPVO:e['session_public_sha256']=hashlib.sha256(b''.join(state.session_public_fingerprints[-1])).hexdigest()
            raw=round_rows(result,scheme=scheme,run_id=run_id,seed=s['seed'],group=e['group'],dimension=s['d'],
                m=None if scheme==VLMRS else s['m'],round_t=index,repeat=index-s['warmup'])
            wire=message_rows(result,scheme=scheme,run_id=run_id,group=e['group'],round_t=index)
            entries.append(round_shards.commit(root,journal,e,raw,wire,result.raw_sum));progress()
            done=sum(x['kind']=='round' for x in entries)
            print(f'PROGRESS={done}/{len(order(s))} round={index} {scheme} warmup={e["warmup"]} valid={e["valid"]} wall_s={wall/1000:.3f}',flush=True)
        m['observed_conditions']=condition_summary(entries)
        round_shards.seal(root,m,entries,summaries(entries,s))
        from .validate_full_scale import validate_point
        errors=validate_point(root)
        retried(root/'checkpoints/final_validation.json',dict(valid=not errors,errors=errors,journal_sha256=round_shards.file_sha(root/'journal.json')))
        m['archive_validation']='failed' if errors else 'passed';retried(root/'manifest.json',m)
        if errors:raise ValueError(str(errors))
        print('COMPLETED='+str(root),flush=True)
        return int(any(e['kind']=='round' and not e['valid'] for e in entries))
    except BaseException as error:
        stop('exception',repr(error));raise


if __name__=='__main__':raise SystemExit(main())
