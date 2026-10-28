"""Validate the six-class declared-baseline E5 evidence contract.

Checks archived inputs, public payload constructions and recorded execution.
Does not infer rejection, rerun secret-key checks or certify protocol security.
"""
from __future__ import annotations
from collections import Counter
import csv
import json
import math
from pathlib import Path
import pickle
import sys
import numpy as np
from .adapters.common import reference_sum, vector_digest
from .artifacts import digest, RAW_FIELDS, MESSAGE_FIELDS


def _inside(root,relative):
    path=(root/relative).resolve()
    if root.resolve() not in path.parents:raise ValueError('Path escapes archive')
    return path


def _validate(root):
    from .run_decided_integrity import TYPES, GPVO, VLMRS, DECISIONS
    from .run_compare import ROOT
    # The standalone verifier must load the same isolated pairing runtime as
    # the execution driver before decoding archived public curve points.
    pairing_directory=str(Path(ROOT)/'tmp/point1_pairing_deps')
    if pairing_directory not in sys.path:sys.path.insert(0,pairing_directory)
    m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    c=json.loads((root/'correctness.json').read_text(encoding='utf-8'))
    assert m['experiment']==c['experiment']=='decided_integrity'
    assert m['schema_version']==2 and m['execution_status']=='completed'
    assert m['tamper_verifiers']==1 and m['tamper_types']==TYPES
    assert m['schemes'] and len(set(m['schemes']))==len(m['schemes']) and set(m['schemes'])<={GPVO,VLMRS}
    for rel,expected in m['source_sha256'].items():
        assert digest(_inside(root,m['source_archive']+'/'+rel))==expected,'Source '+rel
    required={'raw.jsonl','messages.csv','correctness.json','summary.csv','failures.jsonl'}
    assert set(m['output_sha256'])==required
    for rel,expected in m['output_sha256'].items():assert digest(_inside(root,rel))==expected,'Output '+rel
    prior=root/'baseline_cost_batch.json'
    assert digest(prior)==m['baseline_cost_batch']['sha256']
    old=json.loads(prior.read_text(encoding='utf-8'))
    assert old['status']=='completed_decided_minimum_matrix'
    assert old['critical_source_sha256']==m['baseline_cost_batch']['critical_source_sha256']
    for rel,expected in old['critical_source_sha256'].items():
        assert m['source_sha256'][rel]==expected,'Protocol differs from costs: '+rel
    config_rel='experiments/point1_compare/configs/baseline_decisions_20261019.json'
    archived=_inside(root,m['source_archive']+'/'+config_rel)
    assert digest(archived)==m['decisions_sha256']==old['decisions_sha256']
    assert json.loads(archived.read_text(encoding='utf-8'))==m['decisions']
    if GPVO in m['schemes']:
        import gmpy2
        pp=m['gpvo_public_parameters']
        assert pp['N'].bit_length()==pp['modulus_bits']==3072
        assert pp['e']>pp['N']**2 and bool(gmpy2.is_prime(pp['e']))
        assert math.gcd(pp['g'],pp['N'])==1 and pp['prime_e'] is True
        assert pp['maximal_order_certified'] is False
    raw=[json.loads(line) for line in (root/'raw.jsonl').read_text(encoding='utf-8').splitlines() if line]
    with (root/'messages.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f);assert set(MESSAGE_FIELDS)<=set(reader.fieldnames);messages=list(reader)
    with (root/'summary.csv').open(encoding='utf-8',newline='') as f:summary=list(csv.DictReader(f))
    assert raw and messages and summary
    for r in raw:
        assert set(RAW_FIELDS)<=set(r) and r['scheme'] in m['schemes'] and r['run_id']==m['run_id']
        assert isinstance(r['elapsed_ms'],(int,float)) and math.isfinite(r['elapsed_ms']) and r['elapsed_ms']>=0
    for r in messages:assert int(r['num_bytes'])>0 and r['sender'] and r['receiver']
    specs={s['setting']:s for s in m['settings']}
    assert len(specs)==len(m['settings'])
    for s in specs.values():
        assert s['count']>=1
        assert (s['N'],s['M'])==((20,5) if s['size']=='full' else (10,2))
        assert s['d']==7850 if s['size']=='full' else s['d'] in [32,257]
    hs=c['honest_contexts'];ts=c['trials']
    expected={(s,k,i) for s in m['schemes'] for k,sp in specs.items() for i in range(sp['count']+2)}
    by={(h['scheme'],h['setting'],h['index']):h for h in hs}
    assert set(by)==expected and len(by)==len(hs),'Honest coverage'
    public={};inputs={}
    for key,h in by.items():
        scheme,k,i=key;sp=specs[k]
        path=_inside(root,h['snapshot']);assert digest(path)==h['snapshot_sha256']
        with np.load(path,allow_pickle=False) as data:vectors={str(u):v for u,v in zip(data['user_ids'],data['vectors'])}
        users=[f'u{j}' for j in range(sp['N'])]
        assert set(vectors)==set(users) and all(v.dtype==np.dtype('int64') and v.shape==(sp['d'],) for v in vectors.values())
        assert vector_digest(vectors)==h['input_sha256']
        assert h['context']==('same_set' if i==0 else 'changed_set' if i==1 else 'trial')
        assert set(h['uploaded'])==set(users)-({users[0]} if i==1 else set())
        ref=reference_sum([vectors[u] for u in h['uploaded']])
        assert np.array_equal(ref,h['reference_sum']) and np.array_equal(ref,h['returned_sum'])
        assert h['exact_match'] is True and h['verified'] is True
        if (k,i) in inputs:assert inputs[k,i]==h['input_sha256'],'Paired input mismatch'
        inputs[k,i]=h['input_sha256']
        transcript=_inside(root,h['public_transcript']);assert digest(transcript)==h['public_sha256']
        payload=pickle.loads(transcript.read_bytes())
        assert np.array_equal(payload['candidate'],ref)
        public[key]=payload
        measured=[r for r in raw if r['scheme']==scheme and r['group']==h['group'] and r['round']==h['round'] and r['phase']!='tamper_verify']
        assert measured and all(r['success'] is True and r['verified'] is True for r in measured)
        phase='edge_verify' if scheme==GPVO else 'verify_all_equations'
        assert len([r for r in measured if r['phase']==phase])==(sp['M'] if scheme==GPVO else len(h['uploaded'])),'Not all honest verifiers ran'
        assert any(r['scheme']==scheme and r['group']==h['group'] and int(r['round'])==h['round'] for r in messages)
    for scheme in m['schemes']:
        hashes=[h['input_sha256'] for h in hs if h['scheme']==scheme]
        assert len(hashes)==len(set(hashes)),'Repeated independent input'
    expected_t={(s,k,i,t) for s,k,i in expected if i>=2 for t in TYPES}
    identities={(t['scheme'],t['setting'],t['index'],t['tamper_type']) for t in ts}
    assert identities==expected_t and len(identities)==len(ts),'Trial coverage'
    counts=Counter()
    for t in ts:
        key=t['scheme'],t['setting'],t['index'];h=by[key];sp=specs[t['setting']]
        assert t['group']==h['group'] and t['round']==h['round']
        path=_inside(root,t['payload']);assert digest(path)==t['payload_sha256']
        payload=pickle.loads(path.read_bytes());candidate=payload['candidate'];kind=t['tamper_type']
        same_shape=candidate.shape==(sp['d'],)
        assert t['same_shape'] is same_shape
        exact=bool(same_shape and np.array_equal(candidate,h['reference_sum']))
        assert t['exact_match'] is exact and isinstance(t['accepted'],bool)
        current=public[key];same_proof=lambda x,y:pickle.dumps(x)==pickle.dumps(y)
        if kind.startswith('replay_'):
            old=public[t['scheme'],t['setting'],0 if kind=='replay_same_set' else 1]
            assert t['source_context']==kind.removeprefix('replay_')
            assert np.array_equal(candidate,old['candidate']) and same_proof(payload['proof'],old['proof'])
        else:
            assert t['source_context']=='current'
            if kind=='proof_only':
                assert np.array_equal(candidate,current['candidate']) and not same_proof(payload['proof'],current['proof'])
            elif kind=='zero_extended_dimension':
                assert np.array_equal(candidate,np.append(current['candidate'],np.int64(0)))
                assert same_proof(payload['proof'],current['proof'])
            elif kind=='result_only':
                wrong=current['candidate'].copy();wrong[0]+=1
                assert np.array_equal(candidate,wrong) and same_proof(payload['proof'],current['proof'])
            else:
                if t['scheme']==GPVO:
                    wrong=current['candidate'].copy();wrong[0]+=1
                else:wrong=current['candidate']*2
                assert np.array_equal(candidate,wrong) and not same_proof(payload['proof'],current['proof'])
        measured=[r for r in raw if r['scheme']==t['scheme'] and r['group']==t['group'] and r['phase']=='tamper_verify' and r['detail']==kind]
        assert len(measured)==1 and measured[0]['verified'] is t['accepted'] and measured[0]['elapsed_ms']==t['verification_ms']
        assert t['input_rejection']=='' or (kind=='zero_extended_dimension' and t['accepted'] is False)
        counts[t['scheme'],t['setting'],kind,t['accepted']]+=1
    assert len(summary)==len(specs)*len(m['schemes'])*len(TYPES)
    summary_keys=set()
    for r in summary:
        key=r['scheme'],r['setting'],r['tamper_type'];assert key not in summary_keys;summary_keys.add(key)
        assert int(r['trials'])==specs[r['setting']]['count']
        assert int(r['accepted'])==counts[key+(True,)] and int(r['rejected'])==counts[key+(False,)]
    assert c['all_tested_attacks_rejected'] is (not any(t['accepted'] for t in ts))
    assert c['execution_errors']==0 and (root/'failures.jsonl').read_text(encoding='utf-8')==''
    ls=c['lifecycle_diagnostics']
    assert len(ls)==(len(specs) if VLMRS in m['schemes'] else 0)
    for item in ls:
        h=by[VLMRS,item['setting'],0]
        assert item['past_round']==h['round'] and item['reconstruction_round']==by[VLMRS,item['setting'],1]['round']
        with np.load(_inside(root,h['snapshot']),allow_pickle=False) as data:
            vectors={str(u):v for u,v in zip(data['user_ids'],data['vectors'])}
        assert item['reference']==vectors[item['user']].tolist()
        assert item['exact_match'] is bool(np.array_equal(item['reference'],item['cloud_recovered']))
        assert item['attacker_reads_verifier_keys_or_reference'] is False


def validate(directory):
    try:_validate(Path(directory).resolve())
    except (AssertionError,KeyError,ValueError,TypeError,OSError,IndexError) as e:return [type(e).__name__+': '+str(e)]
    return []


if __name__=='__main__':
    problems=validate(sys.argv[1]);print(json.dumps({'problems':problems}));raise SystemExit(bool(problems))
