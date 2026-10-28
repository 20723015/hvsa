"""Recompute trajectory coverage, inputs, rosters, sums, timing and wire records."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np
from .artifacts import digest, RAW_FIELDS, MESSAGE_FIELDS
from .adapters.common import reference_sum, vector_digest
from .run_decided_trajectories import specifications, design, case_inputs, roster, expected_order, summarize, ROLES
from .run_decided_comparison import SCHEMES, GPVO, VLMRS
from .validate_results import _inside, _finite, REQUIRED_MANIFEST, OUTPUTS
from .trajectory_segment_contract import PARENT_MANIFEST_SHA,PARENT_PREFIX_SHA,PREFIX_COUNT


def validate(directory):
    try:
        _validate(Path(directory))
        return []
    except (AssertionError, OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        return [f'{type(exc).__name__}: {exc}']


def _validate(root):
    m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    c=json.loads((root/'correctness.json').read_text(encoding='utf-8'))
    assert all(k in m for k in REQUIRED_MANIFEST), 'Missing manifest metadata'
    assert m['experiment']==c['experiment'] and m['experiment'] in ['trajectory_prefix_export','trajectory_suffix']
    prefix=m['experiment']=='trajectory_prefix_export'
    assert m['execution_status']=='completed' and m['schema_version']==2
    assert set(m['output_sha256'])==OUTPUTS
    for name,sha in m['output_sha256'].items():assert digest(_inside(root,name))==sha, 'Output hash '+name
    assert m['source_sha256'], 'Missing source snapshots'
    for rel,sha in m['source_sha256'].items():
        assert digest(_inside(root,m['source_archive']+'/'+rel))==sha, 'Source hash '+rel
    if prefix:
        parent_path=_inside(root,'provenance/parent_manifest.json')
        payload_path=_inside(root,'provenance/parent_prefix.json')
        assert digest(parent_path)==PARENT_MANIFEST_SHA and digest(payload_path)==PARENT_PREFIX_SHA, 'Pinned parent lineage'
        parent=json.loads(parent_path.read_text(encoding='utf-8'))
        original=json.loads(payload_path.read_text(encoding='utf-8'))
        assert parent['execution_status']=='paused_by_user' and m['run_id']==parent['run_id']
        assert m['parent_manifest_sha256']==PARENT_MANIFEST_SHA and m['parent_prefix_sha256']==PARENT_PREFIX_SHA
        assert m['planned_scope_complete'] is False and m['artifact_kind']=='completed_export_of_user_paused_prefix'
        assert m['source_sha256']==parent['source_sha256'] and m['shape']==parent['shape']
        assert c['observations']==original['observations'] and c['initialization']==original['initialization'], 'Parent observations/init altered'
        for rel,sha in m['derivation_sources'].items():assert digest(_inside(root,'derivation_sources/'+rel))==sha
    else:
        assert m['artifact_kind']=='new_process_suffix_only' and m['epoch_reset'] is True
        parent_root=Path(m['parent_export']['path'])
        assert digest(parent_root/'manifest.json')==m['parent_export']['manifest_sha256'], 'Parent export changed'
        pm=json.loads((parent_root/'manifest.json').read_text(encoding='utf-8'))
        assert pm['experiment']=='trajectory_prefix_export' and pm['parent_prefix_sha256']==PARENT_PREFIX_SHA
        assert m['shape']==pm['shape'] and m['frozen_critical_source_sha256']==pm['frozen_critical_source_sha256']
        _validate(parent_root)
    for rel,sha in m['frozen_critical_source_sha256'].items():assert m['source_sha256'][rel]==sha, 'Frozen source '+rel
    rel='experiments/point1_compare/configs/baseline_decisions_20261019.json'
    assert m['source_sha256'][rel]==m['decisions_sha256']
    assert json.loads(_inside(root,m['source_archive']+'/'+rel).read_text(encoding='utf-8'))==m['decisions']
    assert m['schemes']==SCHEMES
    shape=m['shape'];specs=specifications(**shape);groups,users=design(shape)
    assert m['settings']==specs and m['groups']==groups and m['users']==users
    schedule=expected_order(specs)
    schedule=schedule[:PREFIX_COUNT] if prefix else schedule[PREFIX_COUNT:]
    assert m['execution_order']==schedule, 'Execution order'
    observations=c['observations'];initialization=c['initialization']
    identities=[(o['setting'],o['round'],o['scheme']) for o in observations]
    assert identities==[tuple(k) for k in schedule], 'Observation coverage/order'
    raw=[json.loads(s) for s in (root/'raw.jsonl').read_text(encoding='utf-8').splitlines() if s]
    failures=[json.loads(s) for s in (root/'failures.jsonl').read_text(encoding='utf-8').splitlines() if s]
    with (root/'messages.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==MESSAGE_FIELDS;messages=list(reader)
    with (root/'summary.csv').open(encoding='utf-8',newline='') as f:summary=list(csv.DictReader(f))
    if prefix:
        assert raw==original['raw'] and failures==original['failures'], 'Parent raw/failure records altered'
        normalized=[{k:int(v) if k in ['round','num_bytes'] else v for k,v in row.items()} for row in messages]
        assert normalized==original['messages'], 'Parent wire records altered'
        del original
    phases=defaultdict(list);wire=defaultdict(list)
    for row in raw:
        assert set(row)==set(RAW_FIELDS) and _finite(row['elapsed_ms']) and row['role'] in ROLES
        assert row['run_id']==m['run_id'] and row['scheme'] in SCHEMES
        phases[row['group']].append(row)
    for row in messages:
        assert row['direction']=='send' and int(row['num_bytes'])>0 and row['sender'] and row['receiver']
        assert row['run_id']==m['run_id'] and row['scheme'] in SCHEMES
        wire[row['group']].append(row)
    active_settings={o['setting'] for o in observations}
    expected_init={(s['setting'],scheme) for s in specs if s['setting'] in active_settings for scheme in SCHEMES}
    assert len(initialization)==len(expected_init) and {(i['setting'],i['scheme']) for i in initialization}==expected_init
    valid_groups=set()
    def reconcile(item,initial=False):
        group=item['group'];valid_groups.add(group);rr=phases[group];mm=wire[group]
        assert rr and mm, 'Missing measured phase/message records'
        assert all(r['scheme']==item['scheme'] for r in rr+mm)
        for role in ROLES:
            assert math.isclose(sum(r['elapsed_ms'] for r in rr if r['role']==role),item['role_ms'][role],rel_tol=1e-9,abs_tol=1e-7), 'Role time mismatch'
        assert sum(int(r['num_bytes']) for r in mm)==item['sent_bytes' if initial else 'total_sent_bytes'], 'Wire byte mismatch'
        if initial:
            assert all(r['round']==-1 and r['success'] is True and r['verified'] is None for r in rr)
            assert all(int(r['round'])==-1 for r in mm)
        else:
            assert all(r['round']==item['round'] and r['repeat']==item['round'] and r['N_selected']==shape['n'] and r['N_uploaded']==len(item['uploaded']) and r['success']==item['completed'] and r['verified']==item['verified'] for r in rr), 'Phase round/count/verdict mismatch'
            assert all(int(r['round'])==item['round'] for r in mm)
            actors=defaultdict(float);links=defaultdict(int)
            for r in rr:actors[r['role']+':'+r['actor']]+=r['elapsed_ms']
            for r in mm:links[r['link']]+=int(r['num_bytes'])
            assert dict(links)==item['link_bytes'] and set(actors)==set(item['actor_ms'])
            assert all(math.isclose(v,item['actor_ms'][k],rel_tol=1e-9,abs_tol=1e-7) for k,v in actors.items())
            assert _finite(item['wall_ms']) and item['wall_ms']+1e-5>=sum(item['role_ms'].values()), 'Exclusive role time exceeds wall time'
        return rr,mm
    for item in initialization:
        assert item['group']==item['setting']+':'+item['scheme']+':init'
        reconcile(item,True)
        if item['scheme']==GPVO:
            p=item['gpvo_parameters'];assert p['N'].bit_length()==3072 and p['e']>p['N']**2 and math.gcd(p['g'],p['N'])==1
            assert p['maximal_order_certified'] is False
    specs_by={sp['setting']:sp for sp in specs};session_ids=defaultdict(list);invalid=[]
    for o in observations:
        sp=specs_by[o['setting']];index=o['round'];scheme=o['scheme']
        vectors=case_inputs(shape,sp,index,users)
        missing,uploaded,recipients,trace_seed=roster(shape,sp,index,users)
        assert o['missing']==missing and o['uploaded']==uploaded and o['recipients']==recipients and o['trace_seed']==trace_seed, 'Roster differs from seeded design'
        assert o['n_selected']==shape['n'] and o['n_uploaded']==len(uploaded), 'Actual participation count'
        assert o['experiment']==sp['experiment'] and o['stage']==sp['stage'] and o['rate']==sp['rate']
        assert o['group']==o['setting']+':'+scheme+':'+str(index)
        assert o['input_sha256']==vector_digest(vectors), 'Input differs from independent seed'
        path=_inside(root,o['snapshot']);assert digest(path)==o['snapshot_sha256'], 'Input file hash'
        with np.load(path,allow_pickle=False) as snap:
            assert snap['user_ids'].tolist()==users and snap['vectors'].shape==(shape['n'],shape['d']) and snap['vectors'].dtype==np.dtype('int64'), 'Input shape/domain'
            actual=dict(zip(snap['user_ids'].tolist(),snap['vectors']))
        assert vector_digest(actual)==vector_digest(vectors), 'Input pool mismatch'
        assert type(o['completed']) is bool and type(o['exact_match']) is bool and type(o['valid']) is bool
        if o['result_snapshot'] is not None:
            path=_inside(root,o['result_snapshot']);assert digest(path)==o['result_snapshot_sha256']
            returned=np.load(path,allow_pickle=False)
            assert returned.dtype==np.dtype('int64') and returned.shape==(shape['d'],), 'Return shape/domain'
            exact=bool(np.array_equal(returned,reference_sum([vectors[u] for u in uploaded])))
        else:
            assert o['result_snapshot_sha256'] is None
            exact=False
        assert o['exact_match']==exact
        assert o['valid']==bool(o['completed'] and exact and o['verified'] is True), 'Validity not derived from sum/verification'
        rr,mm=reconcile(o)
        if o['completed']:
            assert type(o['verified']) is bool
            expected_verifiers=recipients if scheme==VLMRS else ([str(i) for i in range(shape['m'])] if scheme==GPVO else ['ES_'+str(i) for i in range(shape['m'])])
            # HVSA edge IDs are obtained from actual registration metadata below.
            if scheme=='hvsa':expected_verifiers=sorted(r['actor'] for r in rr if r['phase']=='edge_verify')
            assert len(expected_verifiers)==(len(recipients) if scheme==VLMRS else shape['m']) and len(set(expected_verifiers))==len(expected_verifiers)
            assert set(o['verdicts'])==set(expected_verifiers) and all(type(v) is bool for v in o['verdicts'].values())
            assert o['verified']==all(o['verdicts'].values())
            phase='verify_all_equations' if scheme==VLMRS else 'edge_verify'
            checks=[r for r in rr if r['phase']==phase]
            actors=recipients if scheme==VLMRS else (['edge_'+str(i) for i in range(shape['m'])] if scheme==GPVO else expected_verifiers)
            assert Counter(r['actor'] for r in checks)==Counter(actors), 'Missing designated verifier calls'
            if scheme!=VLMRS:
                delivered=sorted(r['receiver'] for r in mm if r['payload_kind']=='verified_model')
                assert delivered==recipients if o['verified'] else not delivered, 'Actual model recipients'
            else:
                assert sorted(r['receiver'] for r in mm if r['payload_kind']=='aggregate_and_proof')==recipients
        else:
            assert o['verified'] is None and not o['verdicts'] and o['failure_reason']
        if scheme==GPVO:
            session_ids[o['setting']].append(o['session_public_sha256'])
        if not o['valid']:invalid.append(dict(setting=o['setting'],scheme=scheme,round=index,reason=o['failure_reason'] or 'honest-chain mismatch'))
    assert set(phases)==valid_groups and set(wire)==valid_groups, 'Extraneous measured contexts'
    assert all(len(v)==len(set(v)) and all(len(x)==64 for x in v) for v in session_ids.values()), 'Fresh session public keys reused'
    assert failures==invalid and c['execution_errors']==len(failures), 'Failures omitted/relabelled'
    expected_summary=summarize(observations,[s for s in specs if s['setting'] in active_settings])
    assert len(summary)==len(expected_summary)
    for actual,expected in zip(summary,expected_summary):
        assert set(actual)==set(expected)
        for k,v in expected.items():
            assert actual[k]==v if isinstance(v,str) else math.isclose(float(actual[k]),v,rel_tol=1e-9,abs_tol=1e-7), 'Summary differs from observations'


if __name__=='__main__':
    problems=validate(sys.argv[1]);print('\n'.join(problems) if problems else 'OK: trajectory evidence reconciles')
    raise SystemExit(bool(problems))
