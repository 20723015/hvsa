"""Full-grid scope and actual sharded point semantic tampering checks."""
import json
import os
from pathlib import Path
import shutil
import numpy as np
import pytest
from experiments.point1_compare.artifacts import dump_json
from experiments.point1_compare.full_scale_contract import SCANS,POINTS,REUSED,PENDING,GPVO,VLMRS,schemes,shape,order,reused_inventory
from experiments.point1_compare.round_shards import file_sha
from experiments.point1_compare.validate_full_scale import validate_point


def test_full_original_grid_is_not_narrowed():
    union=set(p for points in SCANS.values() for p in points)
    assert union==set(POINTS) and len(union)==10
    assert len(REUSED)==3 and len(PENDING)==7 and not set(REUSED)&set(PENDING)
    assert sum(len(schemes(p)) for p in POINTS)==28
    assert sum(len(order(shape(p))) for p in PENDING)==247
    assert sum(len(schemes(p))*13 for p in REUSED)==117
    assert [len([p for p in POINTS if s in schemes(p)]) for s in ['hvsa',GPVO,VLMRS]]==[10,10,8]


def test_reused_actual_points():
    r=reused_inventory()
    assert len(r)==3 and sum(x['calls'] for x in r)==117


@pytest.fixture
def actual():
    value=os.environ.get('POINT1_FULL_SCALE_TEST_RUN')
    if not value:pytest.fail('POINT1_FULL_SCALE_TEST_RUN must reference a completed actual point')
    return Path(value)


def test_actual_point(actual):
    assert validate_point(actual)==[]


@pytest.mark.parametrize('mutation',['missing_verifier','returned_sum','state_reset','wrong_roster',
    'short_coverage','false_recipient','warmup_relabel','changed_geometry'])
def test_rehashed_payload_does_not_hide_semantic_error(actual,tmp_path,mutation):
    root=tmp_path/'evidence';shutil.copytree(actual,root,copy_function=os.link)
    journal=json.loads((root/'journal.json').read_text(encoding='utf-8'))
    init_count=len(json.loads((root/'manifest.json').read_text(encoding='utf-8'))['schemes'])
    item=journal[init_count];path=root/item['shard']/'entry.json'
    e=json.loads(path.read_text(encoding='utf-8'))
    if mutation=='missing_verifier':e['verdicts'].pop(next(iter(e['verdicts'])))
    elif mutation=='returned_sum':
        f=root/item['shard']/'returned.npy';a=np.load(f,allow_pickle=False);a[0]+=1
        temp=f.with_suffix('.new.npy');np.save(temp,a,allow_pickle=False);temp.replace(f)
        e['content_sha256']['returned.npy']=file_sha(f)
    elif mutation=='state_reset':e['state_public_sha256']='0'*64
    elif mutation=='wrong_roster':e['uploaded']=e['uploaded'][:-1]
    elif mutation=='short_coverage':journal.pop()
    elif mutation=='false_recipient':e['actual_recipients']=[]
    elif mutation=='warmup_relabel':e['warmup']=not e['warmup']
    elif mutation=='changed_geometry':e['group_upload_counts'][0]+=1
    dump_json(path,e);item['entry_sha256']=file_sha(path);dump_json(root/'journal.json',journal)
    m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    m['output_sha256']['journal.json']=file_sha(root/'journal.json');dump_json(root/'manifest.json',m)
    assert validate_point(root),mutation
