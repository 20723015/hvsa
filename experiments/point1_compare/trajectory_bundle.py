"""Read-only analysis bundle of preserved 252 calls plus fresh 48 calls."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from .artifacts import digest
from .run_decided_trajectories import expected_order
from .trajectory_segment_contract import PREFIX_COUNT,PARENT_PREFIX_SHA
from .validate_trajectory_segments import validate


def collect(prefix,suffix):
    prefix=Path(prefix).resolve();suffix=Path(suffix).resolve()
    pm=json.loads((prefix/'manifest.json').read_text(encoding='utf-8'))
    sm=json.loads((suffix/'manifest.json').read_text(encoding='utf-8'))
    assert pm['experiment']=='trajectory_prefix_export' and sm['experiment']=='trajectory_suffix'
    assert sm['parent_export']==dict(path=prefix.as_posix(),manifest_sha256=digest(prefix/'manifest.json'))
    problems=validate(suffix)  # Includes strict parent validation and pinned lineage.
    if problems:raise ValueError(repr(problems))
    assert pm['shape']==sm['shape'] and pm['frozen_critical_source_sha256']==sm['frozen_critical_source_sha256']
    assert pm['lib_versions']==sm['lib_versions'] and sm['epoch_reset'] is True
    pc=json.loads((prefix/'correctness.json').read_text(encoding='utf-8'))
    sc=json.loads((suffix/'correctness.json').read_text(encoding='utf-8'))
    assert len(pc['observations'])==252 and len(sc['observations'])==48
    observations=[];initialization=[]
    for epoch,manifest,corr in [(0,pm,pc),(1,sm,sc)]:
        observations.extend([dict(o,epoch=epoch,origin_run_id=manifest['run_id']) for o in corr['observations']])
        initialization.extend([dict(i,epoch=epoch,origin_run_id=manifest['run_id']) for i in corr['initialization']])
    assert [[o['setting'],o['round'],o['scheme']] for o in observations]==expected_order(pm['settings'])
    assert len({(o['setting'],o['round'],o['scheme']) for o in observations})==300
    assert len(initialization)==54
    bundle=dict(kind='analysis_bundle_not_a_continuous_protocol_run',
        prefix=dict(path=prefix.as_posix(),manifest_sha256=digest(prefix/'manifest.json'),observations=252),
        suffix=dict(path=suffix.as_posix(),manifest_sha256=digest(suffix/'manifest.json'),observations=48),
        pinned_original_prefix_sha256=PARENT_PREFIX_SHA,shape=pm['shape'],
        e3=dict(rounds_per_trajectory=10,epochs=1,observations=60,rerun_observations=0),
        e4=dict(trace_indices_per_epoch=[list(range(8)),[8,9]],initializations_per_cell=2,
            continuous_key_state_across_epochs=False,observations=240),
        total_observations=300,valid=sum(o['valid'] for o in observations),
        note='Original paused run unchanged; prefix export and suffix retain their actual run IDs and ledgers.')
    virtual=copy.deepcopy(pm)
    virtual['artifact_kind']='read_only_two_segment_analysis'
    return bundle,virtual,dict(observations=observations,initialization=initialization)
