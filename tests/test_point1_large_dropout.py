"""Atomic evidence and staged dropout invariants, plus actual archive forgeries."""
import json
import os
from pathlib import Path
import shutil
import numpy as np
import pytest
from experiments.point1_compare.artifacts import dump_json,MESSAGE_FIELDS,RAW_FIELDS
from experiments.point1_compare.round_shards import commit,load,file_sha
from experiments.point1_compare.large_dropout_contract import shape,design,roster,order,GPVO
from experiments.point1_compare.validate_large_dropout import validate


def test_nested_staged_schedule_preserves_uploaded_contributions():
    from experiments.point1_compare.large_dropout_contract import pool,SCHEMES,labels,expectation
    s=shape();groups,users=design(s)
    assert len(order(s))==len(set(order(s)))==240 and len(labels(s))==8
    assert sum(len(g) for g in groups)==100
    for i in range(10):
        previous=set()
        for rate in [0,10,20,30]:
            a=roster(s,f'before_upload_{rate}',i,users)
            b=roster(s,f'after_upload_{rate}',i,users)
            assert a[0]==b[0] and a[2]==b[2] and a[3]==b[3]
            assert len(a[1])==len(a[2])==len(b[2])==100-rate and b[1]==users
            assert previous<=set(a[0]);previous=set(a[0])
            for scheme in SCHEMES:
                assert expectation(s,groups,scheme,a[1],a[2])==(True,'')
                assert expectation(s,groups,scheme,b[1],b[2])==(True,'')
        assert all(len([t for t in order(s) if t[0]==label and t[2]==scheme])==10
                   for label in labels(s) for scheme in SCHEMES)


def test_commit_refuses_overwrite_and_detects_orphans(tmp_path):
    (tmp_path/'shards').mkdir();journal=[]
    entry=dict(kind='round',has_returned=True)
    commit(tmp_path,journal,entry,[],[],np.zeros(4,dtype=np.int64))
    assert len(load(tmp_path)[1])==1
    (tmp_path/'shards/0001').mkdir()
    with pytest.raises(FileExistsError):commit(tmp_path,journal,entry,[],[])
    with pytest.raises(ValueError,match='Unregistered'):load(tmp_path)


def test_commit_hash_detects_modified_payload(tmp_path):
    (tmp_path/'shards').mkdir();journal=[]
    commit(tmp_path,journal,dict(kind='round',has_returned=False),[],[])
    (tmp_path/'shards/0000/raw.jsonl').write_text('changed\n',encoding='utf-8')
    with pytest.raises(ValueError,match='content mismatch'):load(tmp_path)


@pytest.fixture
def actual_run():
    value=os.environ.get('POINT1_LARGE_DROPOUT_TEST_RUN')
    if not value:pytest.skip('Actual smoke or completed formal archive required')
    return Path(value)


def test_actual_archive(actual_run):
    assert validate(actual_run)==[]


@pytest.mark.parametrize('mutation',['missing_verifier','returned_sum','state_reset','wrong_roster','short_coverage','false_recipient','discard_after_upload','wrong_reconstruction'])
def test_rehashed_evidence_cannot_hide_inconsistency(actual_run,tmp_path,mutation):
    root=tmp_path/'copy';shutil.copytree(actual_run,root,copy_function=os.link)
    journal=json.loads((root/'journal.json').read_text(encoding='utf-8'))
    def chosen(j):
        e=json.loads((root/j['shard']/'entry.json').read_text(encoding='utf-8'))
        if e['kind']!='round':return False
        if mutation=='discard_after_upload':return e['label']=='after_upload_30' and e['scheme']=='hvsa'
        if mutation=='wrong_reconstruction':return e['label']=='before_upload_30' and e['scheme']=='vlmrs_fullgraph_public_d'
        return True
    first=next(j for j in journal if chosen(j));path=root/first['shard']/'entry.json'
    e=json.loads(path.read_text(encoding='utf-8'))
    if mutation=='missing_verifier':e['verdicts'].pop(next(iter(e['verdicts'])))
    elif mutation=='returned_sum':
        f=root/first['shard']/'returned.npy';a=np.load(f,allow_pickle=False);a[0]+=1
        temporary=f.with_suffix('.new.npy');np.save(temporary,a,allow_pickle=False);temporary.replace(f)
        e['content_sha256']['returned.npy']=file_sha(f)
    elif mutation=='state_reset':e['state_public_sha256']='0'*64
    elif mutation=='wrong_roster':e['uploaded']=e['uploaded'][:-1]
    elif mutation=='short_coverage':journal.pop()
    elif mutation=='false_recipient':e['actual_recipients']=[]
    elif mutation=='discard_after_upload':e['uploaded']=e['recipients']
    elif mutation=='wrong_reconstruction':e['reconstructed_mask_keys']=[]
    dump_json(path,e);first['entry_sha256']=file_sha(path)
    dump_json(root/'journal.json',journal)
    m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    m['output_sha256']['journal.json']=file_sha(root/'journal.json')
    dump_json(root/'manifest.json',m)
    assert validate(root),mutation
