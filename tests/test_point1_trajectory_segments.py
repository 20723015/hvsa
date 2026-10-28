"""Paused lineage, frozen traces and rehashed-prefix counterfactual checks."""
import json
import os
from pathlib import Path
import shutil
import pytest
from experiments.point1_compare.artifacts import digest,dump_json
from experiments.point1_compare.run_decided_trajectories import specifications,expected_order
from experiments.point1_compare.trajectory_segment_contract import PREFIX_COUNT
from experiments.point1_compare.validate_trajectory_segments import validate


def test_suffix_is_exactly_48_original_cases_and_does_not_repeat_e3():
    order=expected_order(specifications())
    suffix=order[PREFIX_COUNT:]
    assert len(order)==300 and len(suffix)==48
    assert all(label.startswith('e4_') and index in [8,9] for label,index,scheme in suffix)
    assert len({(label,index,scheme) for label,index,scheme in suffix})==48


@pytest.fixture(scope='module')
def original_export():
    value=os.environ.get('POINT1_TRAJECTORY_PREFIX_TEST_RUN')
    if not value:pytest.skip('Set actual immutable prefix export')
    path=Path(value);assert validate(path)==[]
    return path


def test_actual_export_keeps_60_e3_and_192_e4(original_export):
    c=json.loads((original_export/'correctness.json').read_text(encoding='utf-8'))
    assert len(c['observations'])==252
    assert sum(o['experiment']=='e3' for o in c['observations'])==60
    assert all(o['valid'] for o in c['observations'])


@pytest.mark.parametrize('forgery',['lineage','roster','missing_verifier','initialization','false_completion'])
def test_rehashed_prefix_forgeries_rejected(original_export,tmp_path,forgery):
    # Immutable payloads may be hard-linked; all test changes use atomic replace,
    # so no original inode is written. No image/NPY/raw/provenance mutation.
    copied=shutil.copytree(original_export,tmp_path/'copy',copy_function=os.link)
    mp=copied/'manifest.json';cp=copied/'correctness.json'
    m=json.loads(mp.read_text(encoding='utf-8'));c=json.loads(cp.read_text(encoding='utf-8'))
    if forgery=='lineage':m['parent_prefix_sha256']='0'*64
    elif forgery=='roster':c['observations'][0]['uploaded']=c['observations'][0]['uploaded'][:-1]
    elif forgery=='missing_verifier':c['observations'][0]['verdicts'].pop(next(iter(c['observations'][0]['verdicts'])))
    elif forgery=='initialization':c['initialization'][0]['role_ms']['ta']+=1
    else:m['planned_scope_complete']=True
    dump_json(cp,c);m['output_sha256']={name:digest(copied/name) for name in m['output_sha256']};dump_json(mp,m)
    assert validate(copied)
