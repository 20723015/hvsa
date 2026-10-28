"""Bounded-memory checks of complete or journaled-prefix large-E3 evidence."""
from __future__ import annotations
from collections import Counter,defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import numpy as np
from .adapters.common import reference_sum,vector_digest
from .artifacts import RAW_FIELDS,MESSAGE_FIELDS
from .directed_dropout_contract import frozen_sources,DECISIONS
from .large_multirun_contract import shape,design,pool,roster,order,summaries,SCHEMES,ROLES,LABELS,GPVO,VLMRS,STAGE
from .multirun_environment import summarize as condition_summary
from .round_shards import load,file_sha


def require(test,message):
    if not test:raise ValueError(message)


def validate(root,complete=True):
    try:return _validate(Path(root),complete)
    except (KeyError,ValueError,OSError,TypeError,AssertionError,IndexError) as error:return [str(error) or type(error).__name__]


def _validate(root,complete):
    m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    require(m['experiment']=='large_multirun' and type(m['shape']['smoke']) is bool,'Wrong experiment')
    s=shape(m['shape']['smoke']);groups,users=design(s)
    require(m['shape']==s and m['groups']==groups and m['users']==users and m['schemes']==SCHEMES,'Changed design')
    require(m['performance_qualified'] is False and m['paper_reproduction_qualified'] is False,'Overstated qualification')
    require(m['stage']==STAGE and STAGE in m['source_sha256'],'Missing archived stage')
    frozen=frozen_sources(Path(__file__).resolve().parents[2])
    require(m['frozen_critical_source_sha256']==frozen,'Changed frozen source inventory')
    for rel,sha in frozen.items():require(m['source_sha256'].get(rel)==sha,'Protocol hash mismatch: '+rel)
    for rel,sha in m['source_sha256'].items():
        target=(root/'sources'/rel).resolve()
        require((root/'sources').resolve() in target.parents,'Source path escapes run')
        require(file_sha(target)==sha,'Archived source hash: '+rel)
    require(m['decisions_sha256']==frozen[DECISIONS] and m['decisions']==json.loads((root/'sources'/DECISIONS).read_text(encoding='utf-8')),'Changed baseline choices')
    journal,entries=load(root)
    init=[e for e in entries if e['kind']=='init'];obs=[e for e in entries if e['kind']=='round']
    initial_order=[(label,scheme) for label in LABELS for scheme in SCHEMES]
    require([(e['label'],e['scheme']) for e in init]==initial_order[:len(init)],'Initialization order/count')
    require(len(init)<=6 and (not obs or len(init)==6),'Missing or repeated initialization')
    require([e['kind'] for e in entries]==['init']*len(init)+['round']*len(obs),'Interleaved initialization/restart')
    require([(e['label'],e['round'],e['scheme']) for e in obs]==order(s)[:len(obs)],'Journal observation order/coverage')
    require(m['expected_rounds']==len(order(s)) and m['planned_order']==[list(v) for v in order(s)],'Changed planned coverage')
    if complete:
        require(m['execution_status']=='completed' and len(obs)==len(order(s)) and len(init)==6,'Incomplete trajectory')
        names={'raw.jsonl','messages.csv','correctness.json','summary.csv','failures.jsonl','journal.json'}
        require(set(m['output_sha256'])==names,'Missing output hashes')
        for name,sha in m['output_sha256'].items():require(file_sha(root/name)==sha,'Final output hash: '+name)
    elif m['execution_status']=='completed':require(len(obs)==len(order(s)),'False completed status')
    init_by={(e['label'],e['scheme']):e for e in init}
    require(len({e['state_public_sha256'] for e in init})==len(init),'Duplicate initialized state identity')
    pools={};snapshots={};sessions=defaultdict(list)
    raw_hash=hashlib.sha256();wire_hash=hashlib.sha256()
    for e in entries:
        scheme=e['scheme'];label=e['label'];is_init=e['kind']=='init'
        require(scheme in SCHEMES and label in LABELS and type(e['has_returned']) is bool,'Entry identity/domain')
        r=-1 if is_init else e['round']
        require(e['group']==f'{label}:{scheme}:'+('init' if is_init else str(r)),'Entry group identity')
        require(len(e['state_public_sha256'])==64 and all(c in '0123456789abcdef' for c in e['state_public_sha256']),'State fingerprint domain')
        for key in ['environment_before','environment_after']:
            f=e[key]
            require(set(f)=={'wall_utc','monotonic_seconds','battery','cpu_frequency','system_cpu_times','process_cpu_times','process_rss_bytes','other_protocol_workers'},'Missing environment frame')
            require(math.isfinite(f['monotonic_seconds']) and f['process_rss_bytes']>0,'Invalid environment frame')
        require(e['environment_after']['monotonic_seconds']>=e['environment_before']['monotonic_seconds'],'Environment frame order')
        if is_init:
            require(e['round']==-1 and e['completed'] is True and e['verified'] is None and not e['has_returned'],'Invalid initialization result')
            if scheme==GPVO:
                import gmpy2
                p=e['gpvo_parameters'];require(p['N'].bit_length()==3072 and p['e']>p['N']**2 and bool(gmpy2.is_prime(p['e'])) and math.gcd(p['g'],p['N'])==1 and p['maximal_order_certified'] is False,'GPVO public parameters')
                fingerprint=hashlib.sha256(scheme.encode())
                fingerprint.update(json.dumps({k:p[k] for k in ['N','g','e']},sort_keys=True).encode())
                require(e['state_public_sha256']==fingerprint.hexdigest(),'GPVO public-parameter fingerprint mismatch')
        else:
            require(e['state_public_sha256']==init_by[label,scheme]['state_public_sha256'],'Initialized public keys/parameters changed')
            if r not in pools:
                vectors=pool(s,r,users);path=root/'inputs'/f'round_{r:03d}.npz'
                with np.load(path,allow_pickle=False) as data:
                    require(data.files==['user_ids','vectors'] and data['user_ids'].tolist()==users and data['vectors'].dtype==np.dtype('int64') and data['vectors'].shape==(s['n'],s['d']),'Input pool domain')
                    require(np.array_equal(data['vectors'],np.stack([vectors[u] for u in users])),'Input differs from seeded pool')
                pools[r]=vectors;snapshots[r]=file_sha(path)
            vectors=pools[r];missing,uploaded,trace_seed=roster(s,label,r,users)
            require(e['missing']==missing and e['uploaded']==uploaded and e['recipients']==uploaded and e['trace_seed']==trace_seed,'Unpaired membership roster')
            require(e['snapshot']==f'inputs/round_{r:03d}.npz' and e['snapshot_sha256']==snapshots[r] and e['input_sha256']==vector_digest(vectors),'Input ownership/hash')
            reference=reference_sum([vectors[u] for u in uploaded]);exact=False
            if e['has_returned']:
                returned=np.load(root/e['shard']/'returned.npy',allow_pickle=False)
                require(returned.dtype==np.dtype('int64') and returned.shape==(s['d'],),'Returned sum domain')
                exact=bool(np.array_equal(returned,reference))
            require(e['exact_match'] is exact and type(e['completed']) is bool and e['valid'] is bool(e['completed'] and exact and e['verified'] is True),'False exactness/validity')
            require(e['n_selected']==s['n'] and e['n_uploaded']==s['uploaded'],'Wrong actual participation count')
            require(e['group_upload_counts']==[len(set(g)&set(uploaded)) for g in groups],'Group participation mismatch')
            if scheme==GPVO:sessions[label].append(e['session_public_sha256'])
        roles=defaultdict(float);actors=defaultdict(float);verifiers=Counter()
        with (root/e['shard']/'raw.jsonl').open('rb') as stream:
            for line in stream:
                raw_hash.update(line);row=json.loads(line)
                require(set(row)==set(RAW_FIELDS) and row['run_id']==m['run_id'] and row['scheme']==scheme and row['group']==e['group'] and row['seed']==s['seed'],'Phase ownership/fields')
                require(row['round']==r and row['repeat']==r and row['d']==s['d'] and row['M']==(None if scheme==VLMRS else s['m']) and row['N_selected']==s['n'] and row['N_uploaded']==(0 if is_init else s['uploaded']),'Phase round/count')
                require(row['success']==e['completed'] and row['verified']==e['verified'] and row['error_reason']==e['failure_reason'],'Phase result')
                require(row['role'] in ROLES and row['actor'] and math.isfinite(row['elapsed_ms']) and row['elapsed_ms']>=0,'Phase timing/actor')
                roles[row['role']]+=row['elapsed_ms'];actors[row['role']+':'+row['actor']]+=row['elapsed_ms']
                phase='verify_all_equations' if scheme==VLMRS else 'edge_verify'
                if row['phase']==phase and row['role']==('user' if scheme==VLMRS else 'edge'):verifiers[row['actor']]+=1
        require(set(e['role_ms'])==set(ROLES) and all(math.isclose(roles[x],e['role_ms'][x],rel_tol=1e-9,abs_tol=1e-6) for x in ROLES),'Role work mismatch')
        require(math.isfinite(e['wall_ms']) and e['wall_ms']+.1>=sum(e['role_ms'].values()),'Exclusive work exceeds wall time')
        data=(root/e['shard']/'messages.csv').read_bytes()
        wire_hash.update(data if e['sequence']==0 else data.split(b'\n',1)[1])
        reader=csv.DictReader(io.StringIO(data.decode('utf-8')))
        require(reader.fieldnames==MESSAGE_FIELDS,'Message fields')
        links=defaultdict(int);received=[];uploads=Counter();evidence=Counter()
        for row in reader:
            require(row['run_id']==m['run_id'] and row['scheme']==scheme and row['group']==e['group'] and int(row['round'])==r and row['direction']=='send' and int(row['num_bytes'])>0 and row['sender'] and row['receiver'],'Message ownership/domain')
            links[row['link']]+=int(row['num_bytes'])
            kind=row['payload_kind']
            if kind==('aggregate_and_proof' if scheme==VLMRS else 'verified_model'):received.append(row['receiver'])
            if kind==('masked_vector_and_proof' if scheme==VLMRS else 'masked_upload'):uploads[row['sender']]+=1
            if kind==('pre_aggregation_with_evidence' if scheme=='hvsa' else 'group_sum_commitment'):evidence[row['sender']]+=1
        require(e['link_bytes']==dict(links) and e['total_sent_bytes']==sum(links.values()),'Wire counts mismatch')
        if not is_init:
            require(Counter(uploaded)==uploads,'Actual upload senders mismatch')
            if scheme!=VLMRS:
                require(evidence==Counter(f'edge_{i}' for i,g in enumerate(groups) if set(g)&set(uploaded)),'Active-group evidence mismatch')
            designated=uploaded if scheme==VLMRS else [f'edge_{i}' for i in range(s['m'])]
            if e['completed']:
                require(set(e['verdicts'])==set(designated) and all(type(v) is bool for v in e['verdicts'].values()) and e['verified'] is all(e['verdicts'].values()),'Designated verdicts missing')
                require(verifiers==Counter(designated),'Designated verification calls missing')
                delivery=uploaded if scheme==VLMRS else sorted(u for i,g in enumerate(groups) if e['verdicts'][f'edge_{i}'] for u in set(g)&set(uploaded))
            else:
                require(e['verified'] is None and not e['verdicts'] and not verifiers and not e['has_returned'],'Unavailable round claims verification/output')
                delivery=[]
            require(sorted(received)==delivery and e['actual_recipients']==sorted(received),'Actual recipients mismatch')
    require(all(len(v)==len(set(v)) and all(len(x)==64 for x in v) for v in sessions.values()),'GPVO session key reuse')
    if complete:
        require(raw_hash.hexdigest()==m['output_sha256']['raw.jsonl'] and wire_hash.hexdigest()==m['output_sha256']['messages.csv'],'Flattened records differ from journal shards')
        c=json.loads((root/'correctness.json').read_text(encoding='utf-8'))
        require(c['experiment']==m['experiment'] and c['observations']==obs and c['initialization']==init and c['journal_sha256']==file_sha(root/'journal.json'),'Correctness differs from shard journal')
        failures=[dict(label=e['label'],scheme=e['scheme'],round=e['round'],reason=e['failure_reason'] or 'incorrect or unverified aggregate') for e in obs if not e['valid']]
        require([json.loads(l) for l in (root/'failures.jsonl').read_text(encoding='utf-8').splitlines()]==failures and c['execution_errors']==len(failures),'Failures hidden')
        with (root/'summary.csv').open(encoding='utf-8',newline='') as stream:actual=list(csv.DictReader(stream))
        expected=summaries(entries,s)
        require(len(actual)==len(expected),'Missing prefix summary')
        for row,want in zip(actual,expected):
            require(set(row)==set(want),'Summary columns')
            for key,value in want.items():
                require(row[key]==value if isinstance(value,str) else math.isclose(float(row[key]),value,rel_tol=1e-9,abs_tol=1e-6),'Cumulative cost not derived from real prefix')
        require(m['observed_conditions']==condition_summary(entries),'Environment summary mismatch')
        require({p.name for p in (root/'inputs').glob('*.npz')}=={f'round_{i:03d}.npz' for i in range(s['rounds'])},'Missing/extra input pool')
    return []


if __name__=='__main__':
    errors=validate(sys.argv[1],complete='--prefix' not in sys.argv)
    print(json.dumps(dict(valid=not errors,errors=errors)))
    raise SystemExit(int(bool(errors)))
