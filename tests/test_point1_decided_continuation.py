"""A continuation preserves complete points and refuses changed provenance."""
import json
from pathlib import Path
import pytest
from experiments.point1_compare.artifacts import digest
from experiments.point1_compare.continue_decided_batch import continuation_plan
from experiments.point1_compare.run_decided_batch import POINTS,SCHEMES


@pytest.fixture
def parent_fixture(tmp_path):
    source=tmp_path/"source.py"
    source.write_text("# frozen\n",encoding="utf-8")
    run=tmp_path/"done";run.mkdir()
    manifest={"experiment":"decided_comparison","pilot":False,"schemes":list(SCHEMES),"decisions_sha256":"decision",
        "settings":[{"variant":s,"N":20,"M":None if s=="vlmrs_fullgraph_public_d" else 5,"d":1000,
                     "warmup":3,"rounds":10,"case":"honest"} for s in SCHEMES]}
    (run/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    corr={"observations":[{"valid":True,"warmup":i<3} for s in SCHEMES for i in range(13)]}
    (run/"correctness.json").write_text(json.dumps(corr),encoding="utf-8")
    jobs=[{"name":f"n{n}_d{d}","point":[n,m,d],"status":"pending"} for n,m,d in POINTS]
    jobs[0].update(status="completed",run_dir=str(run),manifest_sha256=digest(run/"manifest.json"),observations=39)
    jobs[2].update(status="interrupted_by_user",pid=123,completed_prefix_observations=2)
    parent={"kind":"one_shot_decided_cost_batch","status":"paused_by_user","decisions_sha256":"decision",
            "critical_source_sha256":{"source.py":digest(source)},"jobs":jobs}
    path=tmp_path/"batch.json";path.write_text(json.dumps(parent),encoding="utf-8")
    return path,tmp_path,parent


def test_preserve_completed_drop_interrupted_runtime_state(parent_fixture):
    path,root,parent=parent_fixture
    original=path.read_bytes()
    _,jobs=continuation_plan(path,root,validator=lambda p:[])
    assert jobs[0]["status"]=="completed" and jobs[0]["run_dir"]==parent["jobs"][0]["run_dir"]
    assert jobs[2]["status"]=="pending" and "pid" not in jobs[2]
    assert path.read_bytes()==original


@pytest.mark.parametrize("change",["running","source","manifest"])
def test_refuse_unreliable_continuation(parent_fixture,change):
    path,root,parent=parent_fixture
    if change=="running":
        parent["status"]="running";path.write_text(json.dumps(parent),encoding="utf-8")
    elif change=="source":(root/"source.py").write_text("# changed",encoding="utf-8")
    else:Path(parent["jobs"][0]["run_dir"],"manifest.json").write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError):continuation_plan(path,root,validator=lambda p:[])
