"""Windows evidence I/O only; protocol arithmetic and timed calls are untouched."""
from __future__ import annotations
import json
from pathlib import Path
import re
import time


def retry_writer(original,path,value,*,max_wait=3.0,on_retry=None):
    deadline=time.monotonic()+max_wait
    attempt=0
    while True:
        try:return original(path,value)
        except PermissionError as error:
            if getattr(error,'winerror',None) not in (5,32,33) or time.monotonic()>=deadline:raise
            attempt+=1
            if on_retry:on_retry(dict(path=str(path),attempt=attempt,winerror=error.winerror))
            time.sleep(min(.02*attempt,.15,max(0,deadline-time.monotonic())))


def read_stdout_progress(path):
    """Observe only append-only stdout; never open live replace-target JSON."""
    data=Path(path).read_bytes()
    if not data.endswith(b'\n'):data=data.rsplit(b'\n',1)[0] if b'\n' in data else b''
    lines=data.decode('utf-8',errors='replace')
    rows=re.findall(r'^PROGRESS=(\d+)/(\d+) ([^\r\n]+)\r?$',lines,re.MULTILINE)
    return dict(completed_rounds=int(rows[-1][0]) if rows else 0,
        failed_calls=sum('valid=False' in row[2] for row in rows),
        initialization_count=len(re.findall(r'^INITIALIZED=',lines,re.MULTILINE)),
        last_call=rows[-1][2] if rows else None,completed_marker='COMPLETED=' in lines)


def install_writer_retry(on_retry=None):
    """Explicit artifact-only alias override; frozen original writer stays archived."""
    from . import artifacts,round_shards,run_large_dropout
    modules=[artifacts,round_shards,run_large_dropout]
    previous=[module.dump_json for module in modules]
    original=artifacts.dump_json
    def write(path,value):return retry_writer(original,path,value,on_retry=on_retry)
    for module in modules:module.dump_json=write
    def restore():
        for module,old in zip(modules,previous):module.dump_json=old
    return write,restore
