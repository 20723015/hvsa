"""Atomic evidence and paired trajectory invariants, plus actual archive forgeries."""
import json
import os
from pathlib import Path
import shutil
import numpy as np
import pytest
from experiments.point1_compare.artifacts import dump_json,MESSAGE_FIELDS,RAW_FIELDS
from experiments.point1_compare.round_shards import commit,load,file_sha
from experiments.point1_compare.large_multirun_contract import shape,design,roster,order,GPVO
from experiments.point1_compare.validate_large_multirun import validate


def test_paired_schedule_is_equal_count_without_roster_resampling():
    s=shape();groups,users=design(s)
    assert len(order(s))==len(set(order(s)))==300
    fixed=[];changing=[]
    for i in range(50):
        a=roster(s,'fixed',i,users);b=roster(s,'changing',i,users)
        fixed.append(tuple(a[0]));changing.append(tuple(b[0]))
        assert len(a[1])==len(b[1])==40
    assert len(set(fixed))==1 and len(set(changing))==50 and fixed[0]==changing[0]


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
    value=os.environ.get('POINT1_LARGE_MULTIRUN_TEST_RUN')
    if not value:pytest.skip('Actual smoke or completed formal archive required')
    return Path(value)


def test_actual_archive(actual_run):
    assert validate(actual_run)==[]


@pytest.mark.parametrize('mutation',['missing_verifier','returned_sum','state_reset','wrong_roster','short_coverage','false_recipient'])
def test_rehashed_evidence_cannot_hide_inconsistency(actual_run,tmp_path,mutation):
    root=tmp_path/'copy';shutil.copytree(actual_run,root,copy_function=os.link)
    journal=json.loads((root/'journal.json').read_text(encoding='utf-8'))
    first=journal[6];path=root/first['shard']/'entry.json'
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
    dump_json(path,e);first['entry_sha256']=file_sha(path)
    dump_json(root/'journal.json',journal)
    m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    m['output_sha256']['journal.json']=file_sha(root/'journal.json')
    dump_json(root/'manifest.json',m)
    assert validate(root),mutation
