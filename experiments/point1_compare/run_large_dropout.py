"""Actual N100 E4 with one continuous state per dropout cell and scheme."""
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
from .adapters.common import RoundResult,reference_sum,vector_digest
from .adapters.hvsa_point1 import HVSAProtocol
from .adapters.gpvo_decided import GPVODecidedProtocol
from .artifacts import create_run,dump_json,round_rows,message_rows
from .directed_dropout_contract import frozen_sources,DECISIONS
from .large_dropout_contract import shape,design,pool,roster,order,summaries,SCHEMES,ROLES,LABELS,GPVO,VLMRS,STAGE,labels,expectation
from .multirun_environment import snapshot as environment,summarize as condition_summary
from .round_shards import commit,seal,file_sha
from .run_compare import ROOT,_git_commit,_lib_versions,_new_run_id,_review_metadata
from .watchdog import RoundDeadline
from utils import LHH


def state_public_sha(state,scheme,users):
    h=hashlib.sha256(scheme.encode())
    if scheme=='hvsa':
        h.update(state.ta.get_public_key_bytes())
        for u in users:h.update(state.users[u].get_public_key_bytes())
    elif scheme==GPVO:
        h.update(json.dumps(dict(N=state.params.n_mod,g=state.params.g,e=state.params.e),sort_keys=True).encode())
    else:
        for u in users:
            h.update(state.mask_keys[u][0]);h.update(state.encryption_keys[u][0])
    return h.hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(argv);s=shape(args.smoke);groups,users=design(s)
    if environment()['other_protocol_workers']:raise RuntimeError('Another protocol worker is active')
    frozen=frozen_sources(ROOT)
    for rel,sha in frozen.items():
        if file_sha(Path(ROOT)/rel)!=sha:raise ValueError('Frozen source changed: '+rel)
    sys.path.insert(0,str(Path(ROOT)/'tmp/point1_pairing_deps'))
    import py_ecc
    import gmpy2
    from .adapters.vlmrs_fullgraph import VLMRSFullGraphProtocol
    logging.disable(logging.CRITICAL)
    run_id=_new_run_id('large_dropout_smoke' if args.smoke else 'large_dropout')
    manifest=dict(run_id=run_id,experiment='large_dropout',command=' '.join(sys.argv),
        seed=s['seed'],shape=s,groups=groups,users=users,schemes=SCHEMES,
        git_commit=_git_commit(),lib_versions={**_lib_versions(),'py_ecc':py_ecc.__version__,'gmpy2':gmpy2.version()},
        planned_order=order(s),expected_rounds=len(order(s)),pid=os.getpid(),
        stage=STAGE,execution_status='running',frozen_critical_source_sha256=frozen,
        decisions=json.loads((Path(ROOT)/DECISIONS).read_text(encoding='utf-8')),
        decisions_sha256=file_sha(Path(ROOT)/DECISIONS),round_timeout_seconds=1800,
        paper_reproduction_qualified=False,timing_scope='local sequential full protocol replay, exclusive role work',
        verification='all designated verifiers, no sampled equations',warmup_excluded=0,
        state_epochs_per_trajectory=1)
    manifest.update(_review_metadata(s))
    manifest['implementation_variants']={s:s for s in SCHEMES}
    manifest['measurement_limitations']=[
        'Declared baseline instances, not author code, equal-security or SOTA certification.',
        'One continuous state per scheme/dropout cell; one initialized key epoch, no restart/resume.',
        'Separate GPVO parameters per cell; setup variation is shown separately.',
        'Read-only power/frequency observations do not lock hardware conditions or temperature.',
        'Input generation, reference sums, environment snapshots and evidence I/O excluded from protocol timing.',
        'Local role work is not deployed distributed latency; backend/cache differences remain.',
        'All failed returned calls retained; interrupted in-flight work is not fabricated.']
    for rel in [STAGE,'tests/test_point1_large_dropout.py']:
        manifest['source_sha256'][rel]=file_sha(Path(ROOT)/rel)
    root=Path(ROOT)/'results/point1_compare'/run_id
    create_run(root,manifest,Path(ROOT))
    for folder in ['inputs','shards','checkpoints']:(root/folder).mkdir()
    journal=[];entries=[];dump_json(root/'journal.json',journal)
    context=dict(phase='setup')

    def progress():
        obs=[e for e in entries if e['kind']=='round']
        dump_json(root/'checkpoints/progress.json',dict(context=context,committed_entries=len(journal),
            completed_rounds=len(obs),valid_aggregates=sum(e['valid'] for e in obs),
            initialization_count=sum(e['kind']=='init' for e in entries),
            journal_sha256=file_sha(root/'journal.json'),execution_status=manifest['execution_status']))

    def stop(status,reason):
        if manifest['execution_status']=='running':
            manifest.update(execution_status=status,termination=dict(context=context,reason=reason))
            dump_json(root/'manifest.json',manifest);progress()

    atexit.register(lambda:stop('interrupted','driver exited before all scheduled calls completed'))
    print('RUN_DIR='+str(root),flush=True)
    states={}
    try:
        LHH._generator_for.cache_clear()
        for label in labels(s):
            for scheme in SCHEMES:
                context=dict(phase='initialization',label=label,scheme=scheme);progress()
                before=environment();tick=time.perf_counter()
                with RoundDeadline(1800,lambda:stop('timed_out','initialization deadline')):
                    state=(HVSAProtocol(s['d'],groups) if scheme=='hvsa' else
                        GPVODecidedProtocol(s['d'],groups) if scheme==GPVO else
                        VLMRSFullGraphProtocol(s['d'],users,threshold=math.ceil(2*s['n']/3),root_exponent=7))
                wall_ms=(time.perf_counter()-tick)*1000;after=environment()
                states[label,scheme]=state
                result=RoundResult(scheme,True,None,None,s['n'],0,state.init_phases,state.init_messages)
                group=f'{label}:{scheme}:init';links=defaultdict(int)
                for packet in result.messages:links[packet.link]+=packet.num_bytes
                entry=dict(kind='init',label=label,scheme=scheme,round=-1,group=group,
                    state_public_sha256=state_public_sha(state,scheme,users),
                    has_returned=False,completed=True,verified=None,failure_reason='',
                    wall_ms=wall_ms,role_ms={r:result.phase_ms(r) for r in ROLES},
                    link_bytes=dict(links),total_sent_bytes=sum(links.values()),
                    environment_before=before,environment_after=after)
                if scheme==GPVO:
                    entry['gpvo_parameters']=dict(N=state.params.n_mod,g=state.params.g,e=state.params.e,maximal_order_certified=False)
                raw=round_rows(result,scheme=scheme,run_id=run_id,seed=s['seed'],group=group,
                    dimension=s['d'],m=None if scheme==VLMRS else s['m'],round_t=-1,repeat=-1)
                wire=message_rows(result,scheme=scheme,run_id=run_id,group=group,round_t=-1)
                entries.append(commit(root,journal,entry,raw,wire));progress()
                print(f'INITIALIZED={label} {scheme} wall_s={wall_ms/1000:.3f}',flush=True)
        snapshots={}
        for label,index,scheme in order(s):
            if index not in snapshots:
                vectors=pool(s,index,users)
                path=root/'inputs'/f'round_{index:03d}.npz'
                np.savez(path,user_ids=np.array(users),vectors=np.stack([vectors[u] for u in users]))
                snapshots[index]=dict(snapshot=path.relative_to(root).as_posix(),snapshot_sha256=file_sha(path),input_sha256=vector_digest(vectors))
            vectors=pool(s,index,users)
            missing,uploaded,recipients,trace_seed=roster(s,label,index,users)
            reference=reference_sum([vectors[u] for u in uploaded])
            state=states[label,scheme]
            context=dict(phase='round',label=label,round=index,scheme=scheme);progress()
            before=environment();tick=time.perf_counter()
            with RoundDeadline(1800,lambda:stop('timed_out','round deadline')):
                result=(state.run_round(vectors,set(uploaded),index,recovery_clients=set(recipients)) if scheme==VLMRS else
                    state.run_round(vectors,set(uploaded),index,model_recipients=set(recipients)))
            wall_ms=(time.perf_counter()-tick)*1000;after=environment()
            if vector_digest(vectors)!=snapshots[index]['input_sha256']:raise ValueError('Adapter mutated shared input pool')
            exact=bool(result.raw_sum is not None and np.array_equal(result.raw_sum,reference))
            group=f'{label}:{scheme}:{index}';links=defaultdict(int)
            for packet in result.messages:links[packet.link]+=packet.num_bytes
            verdicts=result.extra.get('verification_verdicts',result.extra.get('per_edge_verified',{}))
            verdicts={('edge_'+k if scheme==GPVO else k):bool(v) for k,v in verdicts.items()}
            kind='aggregate_and_proof' if scheme==VLMRS else 'verified_model'
            entry=dict(kind='round',label=label,scheme=scheme,round=index,group=group,
                **snapshots[index],trace_seed=trace_seed,missing=missing,uploaded=uploaded,recipients=recipients,
                n_selected=result.n_selected,n_uploaded=result.n_uploaded,
                group_upload_counts=[len(set(g)&set(uploaded)) for g in groups],
                state_public_sha256=state_public_sha(state,scheme,users),
                has_returned=result.raw_sum is not None,completed=result.completed,verified=result.verified,
                exact_match=exact,valid=bool(result.completed and exact and result.verified is True),
                failure_reason=result.failure_reason,verdicts=verdicts,
                actual_recipients=sorted(p.receiver for p in result.messages if p.payload_kind==kind),
                wall_ms=wall_ms,role_ms={r:result.phase_ms(r) for r in ROLES},
                link_bytes=dict(links),total_sent_bytes=sum(links.values()),
                environment_before=before,environment_after=after)
            expected,reason=expectation(s,groups,scheme,uploaded,recipients)
            entry.update(expected_completed=expected,expected_reason=reason,
                matches_expected=bool(entry['valid'] if expected else not result.completed and result.failure_reason==reason))
            if scheme==VLMRS:
                entry['reconstructed_mask_keys']=result.extra.get('mask_private_keys_reconstructed',[])
            if scheme==GPVO:
                entry['session_public_sha256']=hashlib.sha256(b''.join(state.session_public_fingerprints[-1])).hexdigest()
            raw=round_rows(result,scheme=scheme,run_id=run_id,seed=s['seed'],group=group,
                dimension=s['d'],m=None if scheme==VLMRS else s['m'],round_t=index,repeat=index)
            wire=message_rows(result,scheme=scheme,run_id=run_id,group=group,round_t=index)
            entries.append(commit(root,journal,entry,raw,wire,result.raw_sum));progress()
            completed=sum(e['kind']=='round' for e in entries)
            print(f'PROGRESS={completed}/{len(order(s))} {label} round={index} {scheme} valid={entry["valid"]} wall_s={wall_ms/1000:.3f}',flush=True)
            if completed==3:
                from .validate_large_dropout import validate
                errors=validate(root,complete=False)
                dump_json(root/'checkpoints/first_cell_validation.json',dict(journal_sha256=file_sha(root/'journal.json'),valid=not errors,errors=errors))
                if errors:raise ValueError('First cell did not reconcile: '+repr(errors))
        manifest['observed_conditions']=condition_summary(entries)
        seal(root,manifest,entries,summaries(entries,s))
        from .validate_large_dropout import validate
        errors=validate(root)
        dump_json(root/'checkpoints/final_validation.json',dict(valid=not errors,errors=errors,journal_sha256=file_sha(root/'journal.json')))
        manifest['archive_validation']='failed' if errors else 'passed'
        dump_json(root/'manifest.json',manifest);progress()
        if errors:raise ValueError('Complete evidence did not reconcile: '+repr(errors))
        print('COMPLETED='+str(root),flush=True)
        return int(any(e['kind']=='round' and not e['matches_expected'] for e in entries))
    except BaseException as error:
        stop('exception',repr(error));raise


if __name__=='__main__':raise SystemExit(main())
