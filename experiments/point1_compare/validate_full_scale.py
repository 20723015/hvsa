"""Independent streaming checks for point evidence and exact full E2 coverage."""
from __future__ import annotations
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .artifacts import RAW_FIELDS,MESSAGE_FIELDS
from .adapters.common import reference_sum,vector_digest
from .directed_dropout_contract import frozen_sources,DECISIONS
from .full_scale_contract import ROOT,STAGE,GPVO,VLMRS,ROLES,PENDING,POINTS,schemes,shape,design,pool,order,summaries,reused_inventory
from .multirun_environment import summarize as condition_summary
from .round_shards import file_sha,load


def require(test,message):
    if not test:raise ValueError(message)


def validate_point(root,complete=True):
    try:return _point(Path(root),complete)
    except (KeyError,ValueError,OSError,TypeError,AssertionError,IndexError) as error:return [str(error) or type(error).__name__]


def _point(root,complete):
    m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    require(m['experiment']=='full_scale_point' and type(m['shape']['smoke']) is bool,'Wrong experiment')
    s=shape((m['shape']['n'],m['shape']['m'],m['shape']['d']),m['shape']['smoke']);groups,users=design(s)
    require(m['shape']==s and m['schemes']==s['schemes'] and m['groups']==groups and m['users']==users,'Changed point/design')
    require(m['performance_qualified'] is False and m['paper_reproduction_qualified'] is False,'Overstated qualification')
    require(m['stage']==STAGE and STAGE in m['source_sha256'],'Missing stage')
    frozen=frozen_sources(ROOT)
    require(m['frozen_critical_source_sha256']==frozen,'Changed frozen inventory')
    for rel,h in frozen.items():require(m['source_sha256'].get(rel)==h,'Changed protocol: '+rel)
    for rel,h in m['source_sha256'].items():
        target=(root/'sources'/rel).resolve()
        require((root/'sources').resolve() in target.parents and file_sha(target)==h,'Source archive changed: '+rel)
    require(m['decisions_sha256']==frozen[DECISIONS] and m['decisions']==json.loads((root/'sources'/DECISIONS).read_text(encoding='utf-8')),'Changed choices')
    require(m['artifact_io_runtime_override']==dict(modules=['artifacts','round_shards'],max_wait_seconds=3,affects_protocol_or_timers=False),'Unrecorded artifact override')
    journal,entries=load(root);ss=s['schemes'];init=[e for e in entries if e['kind']=='init'];obs=[e for e in entries if e['kind']=='round']
    require([e['kind'] for e in entries]==['init']*len(init)+['round']*len(obs),'State restart or initialization interleaved')
    require([e['scheme'] for e in init]==ss[:len(init)] and len(init)<=len(ss) and (not obs or len(init)==len(ss)),'Initialization coverage')
    require([(e['round'],e['scheme']) for e in obs]==order(s)[:len(obs)],'Round order/coverage')
    require(m['expected_rounds']==len(order(s)) and m['planned_order']==[list(x) for x in order(s)],'Changed expected coverage')
    if complete:
        require(m['execution_status']=='completed' and len(obs)==len(order(s)) and len(init)==len(ss),'Incomplete point')
        require(set(m['output_sha256'])=={'raw.jsonl','messages.csv','correctness.json','summary.csv','failures.jsonl','journal.json'},'Output inventory')
        for name,h in m['output_sha256'].items():require(file_sha(root/name)==h,'Output hash: '+name)
    init_by={e['scheme']:e for e in init};sessions=[];cached_index=None
    raw_hash=hashlib.sha256();wire_hash=hashlib.sha256()
    for e in entries:
        scheme=e['scheme'];is_init=e['kind']=='init';r=-1 if is_init else e['round']
        require(e['label']=='honest' and e['group']==f'honest:{scheme}:'+('init' if is_init else str(r)),'Entry ownership')
        require(type(e['has_returned']) is bool and len(e['state_public_sha256'])==64,'Entry domain')
        require(e['environment_after']['monotonic_seconds']>=e['environment_before']['monotonic_seconds'],'Environment order')
        if is_init:
            require(e['round']==-1 and e['completed'] is True and e['verified'] is None and not e['has_returned'],'Initialization result')
            if scheme==GPVO:
                import gmpy2
                p=e['gpvo_parameters']
                require(p['N'].bit_length()==3072 and p['e']>p['N']**2 and bool(gmpy2.is_prime(p['e'])) and math.gcd(p['N'],p['g'])==1 and p['maximal_order_certified'] is False,'GPVO parameters')
                h=hashlib.sha256(scheme.encode());h.update(json.dumps({k:p[k] for k in ['N','g','e']},sort_keys=True).encode())
                require(h.hexdigest()==e['state_public_sha256'],'GPVO state fingerprint')
        else:
            require(e['state_public_sha256']==init_by[scheme]['state_public_sha256'],'State changed/reset')
            require(e['warmup'] is (r<s['warmup']),'Warmup misclassified')
            require(e['uploaded']==users and e['recipients']==users and e['n_uploaded']==e['n_selected']==s['n'],'Participation/recipients')
            require(e['group_upload_counts']==list(map(len,groups)),'Group upload count')
            if cached_index!=r:
                vectors=pool(s,r,users);path=root/'inputs'/f'round_{r:03d}.npz'
                with np.load(path,allow_pickle=False) as data:
                    require(data.files==['user_ids','vectors'] and data['user_ids'].tolist()==users and data['vectors'].dtype==np.int64,'Input domain/IDs')
                    require(np.array_equal(data['vectors'],np.stack([vectors[u] for u in users])),'Seeded input differs')
                snap_sha=file_sha(path);input_sha=vector_digest(vectors)
                reference=reference_sum([vectors[u] for u in users]);cached_index=r
            require(e['snapshot']==f'inputs/round_{r:03d}.npz' and e['snapshot_sha256']==snap_sha and e['input_sha256']==input_sha,'Input linkage')
            exact=False
            if e['has_returned']:
                returned=np.load(root/e['shard']/'returned.npy',allow_pickle=False)
                require(returned.dtype==np.int64 and returned.shape==(s['d'],),'Returned sum domain')
                exact=bool(np.array_equal(returned,reference))
            require(e['exact_match'] is exact and e['valid'] is bool(e['completed'] and exact and e['verified'] is True),'Incorrect validity')
            if scheme==GPVO:sessions.append(e['session_public_sha256'])
        roles=defaultdict(float);verifiers=Counter()
        with (root/e['shard']/'raw.jsonl').open('rb') as stream:
            for line in stream:
                raw_hash.update(line);row=json.loads(line)
                require(set(row)==set(RAW_FIELDS) and row['run_id']==m['run_id'] and row['scheme']==scheme and row['group']==e['group'] and row['seed']==s['seed'],'Phase ownership')
                require(row['round']==r and row['repeat']==(-1 if is_init else r-s['warmup']) and row['N_selected']==s['n'] and row['N_uploaded']==(0 if is_init else s['n']) and row['d']==s['d'] and row['M']==(None if scheme==VLMRS else s['m']),'Phase geometry/count')
                require(row['success']==e['completed'] and row['verified']==e['verified'] and row['error_reason']==e['failure_reason'],'Phase outcome')
                require(row['role'] in ROLES and row['actor'] and math.isfinite(row['elapsed_ms']) and row['elapsed_ms']>=0,'Phase timing')
                roles[row['role']]+=row['elapsed_ms']
                if row['phase']==('verify_all_equations' if scheme==VLMRS else 'edge_verify') and row['role']==('user' if scheme==VLMRS else 'edge'):
                    verifiers[row['actor']]+=1
        require(set(e['role_ms'])==set(ROLES) and all(math.isclose(roles[k],e['role_ms'][k],rel_tol=1e-9,abs_tol=1e-6) for k in ROLES),'Role work differs')
        require(math.isfinite(e['wall_ms']) and e['wall_ms']+.1>=sum(e['role_ms'].values()),'Invalid wall time')
        message_path=root/e['shard']/'messages.csv'
        with message_path.open('rb') as stream:
            if e['sequence']:stream.readline()
            for block in iter(lambda:stream.read(2**20),b''):wire_hash.update(block)
        links=defaultdict(int);received=[];uploads=Counter();evidence=Counter()
        with message_path.open(encoding='utf-8',newline='') as stream:
            reader=csv.DictReader(stream);require(reader.fieldnames==MESSAGE_FIELDS,'Message fields')
            for row in reader:
                require(row['run_id']==m['run_id'] and row['scheme']==scheme and row['group']==e['group'] and int(row['round'])==r and row['direction']=='send' and int(row['num_bytes'])>0 and row['sender'] and row['receiver'],'Message ownership/domain')
                links[row['link']]+=int(row['num_bytes']);kind=row['payload_kind']
                if kind==('aggregate_and_proof' if scheme==VLMRS else 'verified_model'):received.append(row['receiver'])
                if kind==('masked_vector_and_proof' if scheme==VLMRS else 'masked_upload'):uploads[row['sender']]+=1
                if kind==('pre_aggregation_with_evidence' if scheme=='hvsa' else 'group_sum_commitment'):evidence[row['sender']]+=1
        require(dict(links)==e['link_bytes'] and sum(links.values())==e['total_sent_bytes'],'Communication differs')
        if not is_init:
            require(uploads==Counter(users),'Actual uploader ledger differs')
            designated=users if scheme==VLMRS else [f'edge_{i}' for i in range(s['m'])]
            require(set(e['verdicts'])==set(designated) and all(type(v) is bool for v in e['verdicts'].values()) and e['verified'] is all(e['verdicts'].values()),'Missing designated verdict')
            require(verifiers==Counter(designated),'Incomplete designated verification')
            require(e['actual_recipients']==sorted(received)==users,'Wrong actual recipients')
            if scheme!=VLMRS:require(evidence==Counter(f'edge_{i}' for i in range(s['m'])),'Missing edge evidence')
    require(len(sessions)==len(set(sessions)) and all(len(x)==64 for x in sessions),'GPVO session reused')
    if complete:
        require(raw_hash.hexdigest()==m['output_sha256']['raw.jsonl'] and wire_hash.hexdigest()==m['output_sha256']['messages.csv'],'Flattened records differ')
        c=json.loads((root/'correctness.json').read_text(encoding='utf-8'))
        require(c['experiment']==m['experiment'] and c['observations']==obs and c['initialization']==init and c['journal_sha256']==file_sha(root/'journal.json'),'Correctness summary differs')
        failures=[dict(label=e['label'],scheme=e['scheme'],round=e['round'],reason=e['failure_reason'] or 'incorrect or unverified aggregate') for e in obs if not e['valid']]
        require([json.loads(x) for x in (root/'failures.jsonl').read_text(encoding='utf-8').splitlines()]==failures and c['execution_errors']==len(failures),'Failures hidden')
        with (root/'summary.csv').open(encoding='utf-8',newline='') as stream:actual=list(csv.DictReader(stream))
        require(actual==[{k:str(v) for k,v in row.items()} for row in summaries(entries,s)],'Cost summary differs')
        require(m['observed_conditions']==condition_summary(entries),'Environment summary differs')
        require({p.name for p in (root/'inputs').iterdir()}=={f'round_{i:03d}.npz' for i in range(s['warmup']+s['repeats'])},'Input pool coverage differs')
    return []


def validate_batch(path,complete=True):
    try:
        path=Path(path);b=json.loads(path.read_text(encoding='utf-8'))
        require(b['kind']=='full_scale_scan' and b['reused']==reused_inventory(),'Reused points differ')
        require([tuple(j['point']) for j in b['jobs']]==PENDING,'Missing/changed full-grid jobs')
        require(b['required_point_scheme_cells']==28 and b['required_calls']==364,'Narrowed full scope')
        cells={(tuple(p),s) for p in POINTS for s in schemes(p)}
        require(len(cells)==28,'Full-grid duplicate geometry')
        for rel,h in b['execution_source_sha256'].items():
            require(file_sha(path.parent/'sources'/rel)==h,'Batch archived source changed')
        for job in b['jobs']:
            if job['status']!='completed':
                require(not complete,'Incomplete full-grid job: '+job['name']);continue
            root=Path(job['run_dir']);require(file_sha(root/'manifest.json')==job['manifest_sha256'],'Point manifest changed')
            errors=validate_point(root);require(not errors,str(errors))
            m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
            require(not m['shape']['smoke'] and (m['shape']['n'],m['shape']['m'],m['shape']['d'])==tuple(job['point']),'Wrong point substituted')
        if complete:require(b['status']=='completed_full_scale_scan','Incomplete full batch')
        return []
    except (KeyError,ValueError,OSError,TypeError,AssertionError,IndexError) as error:return [str(error) or type(error).__name__]


if __name__=='__main__':
    import sys
    errors=validate_batch(sys.argv[1]) if '--batch' in sys.argv else validate_point(sys.argv[1],complete='--prefix' not in sys.argv)
    print(json.dumps(dict(valid=not errors,errors=errors)))
    raise SystemExit(int(bool(errors)))
