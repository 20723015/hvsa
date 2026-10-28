"""Evidence-level falsification tests against a real new-driver smoke run."""
import json
import os
from pathlib import Path
import shutil
import types
import pytest
from experiments.point1_compare.artifacts import digest,dump_json
from experiments.point1_compare.run_decided_integrity import make_schedule,capture_vlmrs_delivery
from experiments.point1_compare.validate_decided_integrity import validate


def test_sample_schedule_and_negative_counts():
    specs=make_schedule()
    assert sum(s['count'] for s in specs)==110
    assert sum(s['count']+2 for s in specs)==116
    assert [s['d'] for s in specs]==[32,257,7850]
    assert [s['count'] for s in specs]==[50,50,10]
    with pytest.raises(ValueError):make_schedule(-1,10)
    with pytest.raises(ValueError):make_schedule(0,0)


def test_capture_observes_delivered_packet_and_restores_after_error():
    class Ledger:
        def transfer(self,value,**kwargs):return ('actually_decoded',value)
    module=types.SimpleNamespace(WireLedger=Ledger)
    with pytest.raises(RuntimeError):
        with capture_vlmrs_delivery(module) as captured:
            ledger=module.WireLedger()
            ledger.transfer('upload',kind='masked_vector_and_proof')
            ledger.transfer('first',kind='aggregate_and_proof')
            ledger.transfer('second',kind='aggregate_and_proof')
            assert captured['payload']==('actually_decoded','first')
            raise RuntimeError('test restoration')
    assert module.WireLedger is Ledger


@pytest.fixture
def actual_run(tmp_path):
    path=os.environ.get('POINT1_DECIDED_INTEGRITY_TEST_RUN')
    if not path:pytest.skip('Set POINT1_DECIDED_INTEGRITY_TEST_RUN to the completed smoke archive')
    source=Path(path)
    assert validate(source)==[]
    return shutil.copytree(source,tmp_path/'copied_run')


def rehash(run):
    path=run/'manifest.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    manifest['output_sha256']={name:digest(run/name) for name in manifest['output_sha256']}
    dump_json(path,manifest)


def test_valid_archive_retains_accepted_attacks(actual_run):
    c=json.loads((actual_run/'correctness.json').read_text(encoding='utf-8'))
    assert any(t['accepted'] and not t['exact_match'] for t in c['trials'])
    assert c['all_tested_attacks_rejected'] is False
    assert validate(actual_run)==[]


@pytest.mark.parametrize('forgery',['verdict','payload','honest_reference','missing_verifier','summary'])
def test_rehashed_forged_evidence_is_rejected(actual_run,forgery):
    c=json.loads((actual_run/'correctness.json').read_text(encoding='utf-8'))
    if forgery=='verdict':
        c['trials'][0]['accepted']=not c['trials'][0]['accepted']
    elif forgery=='payload':
        trial=c['trials'][0]
        # Updating the claimed checksum must not make an arbitrary payload valid.
        other=next(t for t in c['trials'] if t['tamper_type']=='proof_only')
        trial['payload']=other['payload'];trial['payload_sha256']=other['payload_sha256']
    elif forgery=='honest_reference':
        c['honest_contexts'][0]['reference_sum'][0]+=1
        c['honest_contexts'][0]['returned_sum'][0]+=1
    elif forgery=='missing_verifier':
        path=actual_run/'raw.jsonl'
        raw=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        index=next(i for i,r in enumerate(raw) if r['phase']=='edge_verify')
        raw.pop(index)
        path.write_text(''.join(json.dumps(r)+'\n' for r in raw),encoding='utf-8')
    else:
        path=actual_run/'summary.csv'
        text=path.read_text(encoding='utf-8')
        lines=text.splitlines();columns=lines[0].split(',');row=lines[1].split(',')
        row[columns.index('accepted')]=str(int(row[columns.index('accepted')])+1)
        lines[1]=','.join(row);path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump_json(actual_run/'correctness.json',c)
    rehash(actual_run)
    assert validate(actual_run)
