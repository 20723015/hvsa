"""Immutable 252-call export: planned 300 is not marked completed."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
from .artifacts import create_run,digest,dump_json,finalize_run
from .run_compare import ROOT,_new_run_id
from .run_decided_trajectories import summarize,expected_order
from .trajectory_segment_contract import load_pinned_parent,PARENT,PARENT_MANIFEST_SHA,PARENT_PREFIX_SHA,STAGE


def export():
    parent,c=load_pinned_parent();m=copy.deepcopy(parent)
    export_id=_new_run_id('trajectory_prefix_export')
    m.update(experiment='trajectory_prefix_export',export_id=export_id,
        artifact_kind='completed_export_of_user_paused_prefix',planned_scope_complete=False,
        parent_manifest_sha256=PARENT_MANIFEST_SHA,parent_prefix_sha256=PARENT_PREFIX_SHA,
        derivation_sources={})
    assert m['execution_order']==expected_order(m['settings'])[:252]
    root=Path(ROOT)/'results/point1_compare'/export_id
    create_run(root,m,PARENT/'sources')
    for folder in ['inputs','returned']:shutil.copytree(PARENT/folder,root/folder)
    (root/'provenance').mkdir()
    shutil.copyfile(PARENT/'manifest.json',root/'provenance/parent_manifest.json')
    shutil.copyfile(PARENT/'checkpoints/user_paused_prefix.json',root/'provenance/parent_prefix.json')
    for rel in ['experiments/point1_compare/export_trajectory_prefix.py',
                'experiments/point1_compare/validate_trajectory_segments.py',
                'experiments/point1_compare/trajectory_segment_contract.py',STAGE]:
        target=root/'derivation_sources'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(ROOT)/rel,target);m['derivation_sources'][rel]=digest(target)
    corr=dict(experiment=m['experiment'],observations=c['observations'],initialization=c['initialization'],execution_errors=len(c['failures']))
    finalize_run(root,m,c['raw'],c['messages'],corr,summarize(c['observations'],m['settings']),c['failures'])
    return root


if __name__=='__main__':
    root=export()
    from .validate_trajectory_segments import validate
    problems=validate(root)
    if problems:raise ValueError(repr(problems))
    print('EXPORTED_PREFIX='+str(root),flush=True)
