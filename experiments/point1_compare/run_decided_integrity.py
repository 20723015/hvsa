"""Production-parameter verifier diagnostics for the declared baseline choices.

Honest contexts execute the complete protocol at every designated verifier.
Each cloud-side mutation is then tested at one representative verifier, using
the same production verification function. This is E5 behavior, not timing.
"""
from __future__ import annotations
import argparse
import atexit
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import pickle
import sys
import time
import gmpy2
import numpy as np
from .adapters.common import (PhaseRecord, RoundResult, WireLedger, reference_sum,
                              vector_digest)
from .adapters.gpvo_decided import (GPVODecidedProtocol, SCHEME as GPVO,
    parameter_gen, verify_candidate, hash_vector)
from .artifacts import create_run, digest, dump_json, finalize_run, round_rows, message_rows
from .run_compare import ROOT, _git_commit, _lib_versions, _new_run_id, _review_metadata
from .watchdog import RoundDeadline

VLMRS='vlmrs_fullgraph_public_d'
TYPES=['result_only','proof_only','coordinated_result_proof',
       'replay_same_set','replay_changed_set','zero_extended_dimension']
DECISIONS=Path(ROOT)/'experiments/point1_compare/configs/baseline_decisions_20261019.json'


def payload_digest(payload):
    return hashlib.sha256(pickle.dumps(payload)).hexdigest()


def make_schedule(small=100,full=10):
    if small<0 or full<0 or small+full<1:
        raise ValueError('At least one nonnegative independent trial required')
    specs=[]
    for index,(d,count,n,m) in enumerate([(32,(small+1)//2,10,2),(257,small//2,10,2),(7850,full,20,5)]):
        if count:
            specs.append({'setting':f'setting_{index:02d}','d':d,'count':count,
                          'N':n,'M':m,'size':'full' if d==7850 else 'small'})
    return specs


@contextmanager
def capture_vlmrs_delivery(module):
    """Observe an actually serialized public delivery, without changing it."""
    saved=module.WireLedger
    captured={}
    class Capture(saved):
        def transfer(self,value,**kwargs):
            delivered=super().transfer(value,**kwargs)
            if kwargs.get('kind')=='aggregate_and_proof' and 'payload' not in captured:
                captured['payload']=delivered
            return delivered
    module.WireLedger=Capture
    try:
        yield captured
    finally:
        module.WireLedger=saved


def mutate(scheme,kind,public,anchors):
    candidate=public['candidate'].copy()
    proof=public['proof']
    if kind.startswith('replay_'):
        old=anchors[kind.removeprefix('replay_')]
        return {'candidate':old['candidate'].copy(),'proof':old['proof']}
    if kind=='result_only':
        candidate[0]+=1
    elif kind=='proof_only':
        if scheme==GPVO:
            proof=proof*2 % public['params'].n2
        else:
            from dataclasses import replace
            from py_ecc.optimized_bls12_381 import G1,add
            proof=replace(proof,A=add(proof.A,G1))
    elif kind=='coordinated_result_proof':
        if scheme==GPVO:
            delta=np.zeros_like(candidate);delta[0]=1
            candidate+=delta
            params=public['params']
            proof=proof*int(gmpy2.powmod(hash_vector(params,delta),params.e,params.n2)) % params.n2
        else:
            from .adapters.vlmrs_verification import coordinated_scale
            candidate,proof=coordinated_scale(candidate,proof,2,7)
    elif kind=='zero_extended_dimension':
        candidate=np.concatenate([candidate,np.zeros(1,dtype=np.int64)])
    else:
        raise ValueError('Unknown mutation '+kind)
    return {'candidate':candidate,'proof':proof}


def check_candidate(scheme,state,public,payload,uploaded):
    if scheme==GPVO:
        return bool(verify_candidate(state.params,payload['candidate'],payload['proof'],
                                    public['mask_hash'],public['randomiser'])),{}
    return state.verifier.verify(payload['candidate'],payload['proof'],sorted(uploaded))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--small-trials',type=int,default=100)
    p.add_argument('--full-trials',type=int,default=10)
    p.add_argument('--schemes',default=GPVO+','+VLMRS)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--round-timeout',type=float,default=1800)
    args=p.parse_args(argv)
    specs=make_schedule(args.small_trials,args.full_trials)
    schemes=args.schemes.split(',')
    if not schemes or len(set(schemes))!=len(schemes) or not set(schemes)<={GPVO,VLMRS}:
        p.error('Use unique declared baseline scheme names')
    if args.round_timeout<=0 or not math.isfinite(args.round_timeout):p.error('Positive finite deadline required')
    import psutil
    workers={'experiments.point1_compare.run_decided_integrity',
        'experiments.point1_compare.run_decided_comparison','experiments.point1_compare.run_performance',
        'experiments.point1_compare.run_integrity','experiments.point1_compare.run_training'}
    for process in psutil.process_iter(['pid','name','cmdline']):
        if process.pid==os.getpid():continue
        try:
            if (process.info.get('name') or '').lower() in ['python.exe','python'] and workers.intersection(process.info.get('cmdline') or []):
                raise ValueError('Another protocol worker is active: '+str(process.pid))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    sys.path.insert(0,str(Path(ROOT)/'tmp/point1_pairing_deps'))
    import py_ecc
    from .adapters import vlmrs_fullgraph as vlmrs
    logging.disable(logging.CRITICAL)
    decisions=json.loads(DECISIONS.read_text(encoding='utf-8'))
    run_id=_new_run_id('decided_integrity')
    manifest={'run_id':run_id,'experiment':'decided_integrity','command':' '.join(sys.argv),
        'seed':args.seed,'schemes':schemes,'git_commit':_git_commit(),
        'lib_versions':{**_lib_versions(),'gmpy2':gmpy2.version(),'py_ecc':py_ecc.__version__},
        'quant':{'input':'signed int64','amplitude':1000000000},
        'protocol_mapping':'experiments/point1_compare/protocol_mapping.md',
        'paper_versions':{'gpvo':'declared Z_N2/fresh-session','vlmrs':'declared fullgraph/public-root/reuse'},
        'concurrency':'one serial cryptographic worker','timing_scope':'integrity diagnostics, not performance observations',
        'verification':'all honest verifiers; one representative verifier per mutation',
        'aggregation':'integer reference from actual uploaded vectors',
        'settings':specs,'tamper_types':TYPES,'decisions':decisions,'decisions_sha256':digest(DECISIONS),
        'tamper_verifiers':1,'execution_status':'running','pid':os.getpid(),
        'anchor_policy':'two full honest contexts per scheme/dimension, reused only as public replay payloads',
        'measurement_limitations':['No author-code or paper-security-equivalence certification.',
            'Mutations use cloud-visible data and public parameters; no verifier keys or honest inputs.',
            'Replays reuse one archived anchor per membership condition; current tested inputs are independent.',
            'Six classes are not an exhaustive attack catalogue; one representative tamper verifier.',
            'Output dimension acceptance is an implementation-interface result, not a paper-level theorem.']}
    manifest.update(_review_metadata(vars(args)))
    stage_relative='docs/POINT1_DECLARED_INTEGRITY_STAGE_20261023.md'
    manifest['source_sha256'][stage_relative]=digest(Path(ROOT)/stage_relative)
    manifest['implementation_variants']={s:s for s in schemes}
    manifest['performance_qualified']=False
    prior=Path(ROOT)/'results/point1_compare/20261020-212351-556444_decided_minimum_continuation/batch.json'
    cost=json.loads(prior.read_text(encoding='utf-8'))
    if cost['status']!='completed_decided_minimum_matrix':raise ValueError('Prior cost batch not complete')
    for relative,expected in cost['critical_source_sha256'].items():
        if digest(Path(ROOT)/relative)!=expected:raise ValueError('Frozen baseline source changed: '+relative)
    manifest['baseline_cost_batch']={'path':str(prior),'sha256':digest(prior),
        'critical_source_sha256':cost['critical_source_sha256']}
    directory=Path(ROOT)/'results/point1_compare'/run_id
    create_run(directory,manifest,Path(ROOT))
    (directory/'baseline_cost_batch.json').write_bytes(prior.read_bytes())
    for name in ['inputs','public','checkpoints']:(directory/name).mkdir()
    print('RUN_DIR='+str(directory),flush=True)
    raw,messages,honest,trials,failures,lifecycle=[],[],[],[],[],[]
    def checkpoint():
        dump_json(directory/'checkpoints/completed_prefix.json',{'honest':honest,'trials':trials,
            'lifecycle':lifecycle,'raw':raw,'messages':messages,'failures':failures})
    def timeout(context):
        checkpoint();manifest.update(execution_status='timed_out',termination=context)
        dump_json(directory/'manifest.json',manifest)
    def stopped():
        if manifest['execution_status']=='running':
            checkpoint();manifest['execution_status']='interrupted_or_failed'
            dump_json(directory/'manifest.json',manifest)
    atexit.register(stopped)
    params=None
    if GPVO in schemes:
        with RoundDeadline(args.round_timeout,lambda:timeout({'phase':'parameter_generation'})):
            params=parameter_gen(3072)
        manifest['gpvo_public_parameters']={'N':params.n_mod,'g':params.g,'e':params.e,
            'modulus_bits':params.n_mod.bit_length(),'e_bits':params.e.bit_length(),
            'prime_e':bool(gmpy2.is_prime(params.e)),'maximal_order_certified':False}
        dump_json(directory/'manifest.json',manifest)
    for si,spec in enumerate(specs):
        users=[f'u{i}' for i in range(spec['N'])]
        groups=[users[k::spec['M']] for k in range(spec['M'])]
        for scheme in schemes:
            state=(GPVODecidedProtocol(spec['d'],groups,params=params) if scheme==GPVO else
                   vlmrs.VLMRSFullGraphProtocol(spec['d'],users,threshold=math.ceil(2*len(users)/3),root_exponent=7))
            setup=RoundResult(scheme,True,None,None,len(users),0,state.init_phases,state.init_messages)
            raw+=round_rows(setup,scheme=scheme,run_id=run_id,seed=args.seed,group='init:'+spec['setting'],
                            dimension=spec['d'],m=None if scheme==VLMRS else spec['M'],round_t=-1)
            messages+=message_rows(setup,scheme=scheme,run_id=run_id,group='init:'+spec['setting'],round_t=-1)
            anchors={}
            first_vectors=None
            for index in range(spec['count']+2):
                round_t=si*100000+index
                context='same_set' if index==0 else 'changed_set' if index==1 else 'trial'
                # Same NPZ values for both schemes; independent trial inputs.
                rng=np.random.default_rng(args.seed+si*100000+index)
                vectors={u:rng.integers(-10**9,10**9,spec['d'],dtype=np.int64) for u in users}
                vectors[users[0]][0]=123456+index
                uploaded=set(users)-({users[0]} if index==1 else set())
                snapshot=directory/'inputs'/(spec['setting']+'_'+str(index)+'.npz')
                if not snapshot.exists():np.savez(snapshot,user_ids=np.array(users),vectors=np.stack([vectors[u] for u in users]))
                reference=reference_sum([vectors[u] for u in sorted(uploaded)])
                group=spec['setting']+':'+str(index)
                with RoundDeadline(args.round_timeout,lambda:timeout({'scheme':scheme,'group':group,'phase':'honest'})):
                    if scheme==GPVO:
                        result=state.run_round(vectors,uploaded,round_t)
                        public={**state.last_public_context,'proof':state.last_public_context['commitment']}
                    else:
                        with capture_vlmrs_delivery(vlmrs) as captured:
                            result=state.run_round(vectors,uploaded,round_t,recovery_clients=uploaded)
                        value,proof=captured['payload']
                        public={'candidate':value.copy(),'proof':proof}
                exact=bool(result.completed and result.raw_sum is not None and np.array_equal(result.raw_sum,reference))
                if not (exact and result.verified is True):
                    raise RuntimeError('Honest context failed '+scheme+' '+group)
                public_path=directory/'public'/(scheme+'_'+group.replace(':','_')+'.pkl')
                public_path.write_bytes(pickle.dumps(public))
                honest.append({'scheme':scheme,'setting':spec['setting'],'group':group,'index':index,
                    'round':round_t,'context':context,'snapshot':str(snapshot.relative_to(directory)),
                    'snapshot_sha256':digest(snapshot),'input_sha256':vector_digest(vectors),
                    'uploaded':sorted(uploaded),'reference_sum':reference.tolist(),
                    'returned_sum':result.raw_sum.tolist(),'exact_match':exact,'verified':result.verified,
                    'public_transcript':str(public_path.relative_to(directory)),'public_sha256':digest(public_path)})
                raw+=round_rows(result,scheme=scheme,run_id=run_id,seed=args.seed,group=group,
                    dimension=spec['d'],m=None if scheme==VLMRS else spec['M'],round_t=round_t)
                messages+=message_rows(result,scheme=scheme,run_id=run_id,group=group,round_t=round_t)
                if context!='trial':
                    anchors[context]=public
                    if index==0:first_vectors=vectors
                    if index==1 and scheme==VLMRS:
                        recovered=state.recover_past_upload_with_later_key(users[0],si*100000)
                        lifecycle.append({'scheme':scheme,'setting':spec['setting'],'user':users[0],
                            'past_round':si*100000,'reconstruction_round':round_t,
                            'reference':first_vectors[users[0]].tolist(),'cloud_recovered':recovered.tolist(),
                            'exact_match':bool(np.array_equal(recovered,first_vectors[users[0]])),
                            'attacker_reads_verifier_keys_or_reference':False})
                else:
                    for kind in TYPES:
                        payload=mutate(scheme,kind,public,anchors)
                        tick=time.perf_counter();exception=''
                        with RoundDeadline(args.round_timeout,lambda:timeout({'scheme':scheme,'group':group,'phase':kind})):
                            try:accepted,checks=check_candidate(scheme,state,public,payload,uploaded)
                            except ValueError as error:
                                if kind!='zero_extended_dimension':raise
                                accepted,checks,exception=False,{},str(error)
                        elapsed=(time.perf_counter()-tick)*1000
                        trial_path=directory/'public'/(scheme+'_'+group.replace(':','_')+'_'+kind+'.pkl')
                        trial_path.write_bytes(pickle.dumps(payload))
                        same_shape=payload['candidate'].shape==reference.shape
                        trials.append({'scheme':scheme,'setting':spec['setting'],'group':group,'index':index,
                            'round':round_t,'tamper_type':kind,'accepted':bool(accepted),
                            'same_shape':same_shape,'exact_match':bool(same_shape and np.array_equal(payload['candidate'],reference)),
                            'verification_ms':elapsed,'checks':checks,'input_rejection':exception,
                            'payload':str(trial_path.relative_to(directory)),'payload_sha256':digest(trial_path),
                            'source_context':kind.removeprefix('replay_') if kind.startswith('replay_') else 'current'})
                        role='edge' if scheme==GPVO else 'user'
                        probe=RoundResult(scheme,True,None,bool(accepted),len(users),len(uploaded),
                            [PhaseRecord('tamper_verify',role,elapsed,'edge_0' if role=='edge' else users[0],kind)])
                        raw+=round_rows(probe,scheme=scheme,run_id=run_id,seed=args.seed,group=group,
                            dimension=spec['d'],m=None if scheme==VLMRS else spec['M'],round_t=round_t)
                checkpoint()
                print(f"{scheme} {spec['setting']} {context} {max(0,index-1)}/{spec['count']} honest=True attacks={len(trials)}",flush=True)
    counts=Counter((t['scheme'],t['setting'],t['tamper_type'],t['accepted']) for t in trials)
    summary=[{'scheme':s,'setting':sp['setting'],'tamper_type':kind,'trials':sp['count'],
        'rejected':counts[s,sp['setting'],kind,False],'accepted':counts[s,sp['setting'],kind,True]}
        for sp in specs for s in schemes for kind in TYPES]
    corr={'experiment':'decided_integrity','honest_contexts':honest,'trials':trials,
        'lifecycle_diagnostics':lifecycle,'execution_errors':len(failures),
        'all_tested_attacks_rejected':not any(t['accepted'] for t in trials),'paper_reproduction_qualified':False}
    finalize_run(directory,manifest,raw,messages,corr,summary,failures)
    from .validate_decided_integrity import validate
    problems=validate(directory)
    if problems:raise ValueError('; '.join(problems))
    print('COMPLETED='+str(directory),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
