"""A baseline comparison must use the same inputs and declared assumptions."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.point1_compare.artifacts import digest
from experiments.point1_compare.adapters.common import vector_digest
from experiments.point1_compare.validate_results import _decided_comparison


@pytest.fixture
def evidence(tmp_path):
    decisions = {"status":"decided", "security_equivalence":False}
    relative = "experiments/point1_compare/configs/baseline_decisions_20261019.json"
    archived = tmp_path/"sources"/relative
    archived.parent.mkdir(parents=True)
    archived.write_text(json.dumps(decisions),encoding="utf-8")
    schemes = ["hvsa","gpvo_zn2_fresh_prime_e","vlmrs_fullgraph_public_d"]
    manifest = {"schemes":schemes,"users":["u0","u1"],"source_archive":"sources",
                "decisions":decisions,"decisions_sha256":digest(archived),"performance_qualified":False}
    values = np.array([[1,-1],[2,3]],dtype=np.int64)
    path = tmp_path/"input.npz"
    np.savez(path,user_ids=np.array(["u0","u1"]),vectors=values)
    observations,raw,messages,initialization = [],[],[],[]
    for i,scheme in enumerate(schemes):
        group = "setting_"+str(i)+":0"
        observations.append({"scheme":scheme,"case":"honest","round":0,"protocol_round":0,"group":group,
            "N":2,"d":2,"snapshot":"input.npz","snapshot_sha256":digest(path),
            "input_sha256":vector_digest(dict(zip(["u0","u1"],values))),"uploaded":["u0","u1"],
            "recipients":["u0","u1"],"n_uploaded":2,"reference_sum":[3,2],"returned_sum":[3,2],
            "completed":True,"exact_match":True,"verified":True,"valid":True})
        row = {"scheme":scheme,"group":group,"round":0,"success":True,"verified":True,"role":"user","elapsed_ms":1.}
        raw.extend([row,{**row,"group":"init:"+scheme,"round":-1}])
        kind = "aggregate_and_proof" if scheme=="vlmrs_fullgraph_public_d" else "verified_model"
        messages.extend([{"group":group,"payload_kind":kind,"receiver":u,"num_bytes":"5"} for u in ["u0","u1"]])
        messages.append({"group":"init:"+scheme,"payload_kind":"setup","receiver":"u0","num_bytes":"5"})
        initialization.append({"scheme":scheme,"role_ms":{"user":1.},"sent_bytes":5})
    corr = {"observations":observations,"initialization":initialization,"paper_reproduction_qualified":False,"execution_errors":0}
    (tmp_path/"failures.jsonl").write_text("",encoding="utf-8")
    return manifest,corr,raw,messages,tmp_path


def problems(evidence):
    found=[]
    _decided_comparison(*evidence,found)
    return found


def test_paired_comparison_contract(evidence):
    assert problems(evidence)==[]


@pytest.mark.parametrize("change",["different_valid_input","undisclosed_decision","missing_receiver","wrong_init_cost"])
def test_comparison_rejects_semantic_skew(evidence,change):
    manifest,corr,raw,messages,root=evidence
    first=corr["observations"][0]
    if change=="different_valid_input":
        values=np.array([[2,-1],[2,3]],dtype=np.int64)
        path=root/"different.npz"
        np.savez(path,user_ids=np.array(["u0","u1"]),vectors=values)
        first.update(snapshot=path.name,snapshot_sha256=digest(path),
                     input_sha256=vector_digest(dict(zip(["u0","u1"],values))),reference_sum=[4,2],returned_sum=[4,2])
    elif change=="undisclosed_decision":
        manifest["decisions"]["security_equivalence"]=True
    elif change=="missing_receiver":
        messages[:]=[m for m in messages if not(m["group"]==first["group"] and m["receiver"]=="u1")]
    else:
        corr["initialization"][0]["role_ms"]["user"]=.5
    assert problems(evidence)
