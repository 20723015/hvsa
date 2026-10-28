"""The formal batch cannot substitute a pilot or an incomplete workpoint."""
import copy
import pytest
from experiments.point1_compare.run_decided_batch import cost_scope_problems, SCHEMES


def complete():
    manifest={"experiment":"decided_comparison","pilot":False,"schemes":list(SCHEMES),
        "settings":[{"variant":s,"N":20,"M":None if s=="vlmrs_fullgraph_public_d" else 5,
                     "d":1000,"warmup":3,"rounds":10,"case":"honest"} for s in SCHEMES]}
    corr={"observations":[{"valid":True,"warmup":i<3} for s in SCHEMES for i in range(13)]}
    return manifest,corr


def test_complete_cost_schedule_admitted():
    m,c=complete()
    assert cost_scope_problems(m,c,(20,5,1000))==[]


@pytest.mark.parametrize("change",["pilot","partial","wrong_point","bad_round"])
def test_invalid_cost_schedule_refused(change):
    m,c=complete()
    if change=="pilot": m["pilot"]=True
    elif change=="partial": c["observations"].pop()
    elif change=="wrong_point": m["settings"][0]["d"]=257
    else: c["observations"][0]["valid"]=False
    assert cost_scope_problems(m,c,(20,5,1000))
