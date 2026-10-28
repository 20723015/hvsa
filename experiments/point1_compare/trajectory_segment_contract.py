"""Pinned interrupted source and explicit two-epoch suffix contract."""
from pathlib import Path
import json
from .artifacts import digest
from .run_compare import ROOT

PARENT=Path(ROOT)/'results/point1_compare/20261023-232909-034709_decided_trajectories'
PARENT_MANIFEST_SHA='1a21400a04e2e594f44eca238a3deed833fef9d1da7c0ea5a9ec7e0747e7e8f9'
PARENT_PREFIX_SHA='3b54c258c2dbde1b70eb3707b6b2dbd793520fb5f0afefa4972772be4eda4248'
PREFIX_COUNT=252
STAGE='docs/POINT1_TRAJECTORY_CONTINUATION_STAGE_20261024.md'


def load_pinned_parent(root=PARENT):
    root=Path(root)
    assert digest(root/'manifest.json')==PARENT_MANIFEST_SHA, 'Paused manifest changed'
    p=root/'checkpoints/user_paused_prefix.json'
    assert digest(p)==PARENT_PREFIX_SHA, 'Paused prefix changed'
    m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    c=json.loads(p.read_text(encoding='utf-8'))
    assert m['execution_status']=='paused_by_user' and len(c['observations'])==PREFIX_COUNT
    assert m['shape']==dict(n=20,m=5,d=1000,rounds=10,traces=10,seed=42,smoke=False)
    return m,c
