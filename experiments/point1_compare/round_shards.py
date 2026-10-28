"""Atomic per-call evidence, bounded-memory merging and independently hashed files."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import shutil
from .artifacts import RAW_FIELDS, MESSAGE_FIELDS, dump_json, write_csv


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def commit(root, journal, entry, raw, messages, returned=None):
    """Commit shard first, then journal; a crash may leave a detectable orphan."""
    import numpy as np
    root=Path(root);seq=len(journal)
    relative=f'shards/{seq:04d}'
    folder=root/relative;pending=root/f'shards/.{seq:04d}.pending'
    if folder.exists() or pending.exists():raise FileExistsError('Shard cannot be overwritten')
    pending.mkdir()
    with (pending/'raw.jsonl').open('w',encoding='utf-8') as stream:
        for row in raw:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    write_csv(pending/'messages.csv',MESSAGE_FIELDS,messages)
    if returned is not None:np.save(pending/'returned.npy',returned,allow_pickle=False)
    payload={**entry,'sequence':seq,'shard':relative}
    names=['raw.jsonl','messages.csv']+(['returned.npy'] if returned is not None else [])
    payload['content_sha256']={name:file_sha(pending/name) for name in names}
    dump_json(pending/'entry.json',payload)
    pending.rename(folder)
    item=dict(sequence=seq,shard=relative,entry_sha256=file_sha(folder/'entry.json'))
    journal.append(item)
    dump_json(root/'journal.json',journal)
    return payload


def load(root):
    root=Path(root);journal=json.loads((root/'journal.json').read_text(encoding='utf-8'))
    entries=[]
    expected=set()
    for seq,item in enumerate(journal):
        relative=f'shards/{seq:04d}'
        if item['sequence']!=seq or item['shard']!=relative:raise ValueError('Noncontiguous shard journal')
        folder=root/relative;expected.add(folder.name)
        if file_sha(folder/'entry.json')!=item['entry_sha256']:raise ValueError('Shard entry hash mismatch')
        entry=json.loads((folder/'entry.json').read_text(encoding='utf-8'))
        if entry['sequence']!=seq or entry['shard']!=relative:raise ValueError('Shard entry ownership mismatch')
        allowed={'raw.jsonl','messages.csv'}|({'returned.npy'} if entry.get('has_returned') else set())
        if set(entry['content_sha256'])!=allowed:raise ValueError('Missing shard payload')
        for name,sha in entry['content_sha256'].items():
            if file_sha(folder/name)!=sha:raise ValueError('Shard content mismatch: '+name)
        if {p.name for p in folder.iterdir()}!=allowed|{'entry.json'}:raise ValueError('Unexpected shard file')
        entries.append(entry)
    if {p.name for p in (root/'shards').iterdir()}!=expected:raise ValueError('Unregistered or unfinished shard retained outside journal')
    return journal,entries


def seal(root, manifest, entries, summary):
    root=Path(root)
    with (root/'raw.jsonl').open('wb') as raw, (root/'messages.csv').open('wb') as wire:
        for i,e in enumerate(entries):
            with (root/e['shard']/'raw.jsonl').open('rb') as part:shutil.copyfileobj(part,raw)
            with (root/e['shard']/'messages.csv').open('rb') as part:
                if i:part.readline()
                shutil.copyfileobj(part,wire)
    obs=[e for e in entries if e['kind']=='round'];init=[e for e in entries if e['kind']=='init']
    failures=[dict(label=e['label'],scheme=e['scheme'],round=e['round'],reason=e['failure_reason'] or 'incorrect or unverified aggregate') for e in obs if not e['valid']]
    dump_json(root/'correctness.json',dict(experiment=manifest['experiment'],observations=obs,
        initialization=init,execution_errors=len(failures),journal_sha256=file_sha(root/'journal.json')))
    with (root/'failures.jsonl').open('w',encoding='utf-8') as stream:
        for e in failures:stream.write(json.dumps(e,ensure_ascii=False,allow_nan=False)+'\n')
    write_csv(root/'summary.csv',list(summary[0]),summary)
    names=['raw.jsonl','messages.csv','correctness.json','summary.csv','failures.jsonl','journal.json']
    manifest['output_sha256']={name:file_sha(root/name) for name in names}
    manifest['execution_status']='completed'
    dump_json(root/'manifest.json',manifest)
