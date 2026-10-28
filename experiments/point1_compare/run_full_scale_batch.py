"""Sequential seven-point E2 completion; child output goes to physical files."""
from __future__ import annotations
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .artifacts import create_run,dump_json as original_dump
from .directed_dropout_contract import frozen_sources
from .full_scale_contract import ROOT,STAGE,PENDING,schemes,reused_inventory
from .round_shards import file_sha
from .run_compare import _new_run_id
from .validate_full_scale import validate_point,validate_batch
from .windows_artifact_io import retry_writer

EXECUTION_FILES=[STAGE,'experiments/point1_compare/full_scale_contract.py',
    'experiments/point1_compare/run_full_scale_point.py','experiments/point1_compare/run_full_scale_batch.py',
    'experiments/point1_compare/validate_full_scale.py','experiments/point1_compare/round_shards.py',
    'experiments/point1_compare/multirun_environment.py','experiments/point1_compare/windows_artifact_io.py',
    'experiments/point1_compare/run_large_multirun.py','tests/test_point1_full_scale.py']


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    import psutil
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            if p.pid!=os.getpid() and (p.info['name'] or '').lower() in ['python.exe','python'] and any(
                'run_full_scale_batch' in x or 'run_full_scale_point' in x for x in (p.info['cmdline'] or [])):
                raise RuntimeError('Existing full-scale worker: '+str(p.pid))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    frozen=frozen_sources(ROOT)
    for rel,h in frozen.items():
        if file_sha(ROOT/rel)!=h:raise ValueError('Frozen source changed: '+rel)
    reused=reused_inventory()
    hashes={rel:file_sha(ROOT/rel) for rel in set(EXECUTION_FILES)|set(frozen)}
    root=ROOT/'results/point1_compare'/_new_run_id('full_scale_batch')
    root.mkdir(parents=True,exist_ok=False)
    for rel,h in hashes.items():
        target=root/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/rel).read_bytes())
        if file_sha(target)!=h:raise ValueError('Source changed during archive')
    b=dict(kind='full_scale_scan',status='running',pid=os.getpid(),started_utc=now(),
        execution_source_sha256=hashes,reused=reused,required_point_scheme_cells=28,required_calls=364,
        newly_planned_calls=247,warmup_per_cell=3,measured_per_cell=10,
        jobs=[dict(name=f'n{n}_m{m}_d{d}',point=[n,m,d],schemes=schemes((n,m,d)),status='pending') for n,m,d in PENDING],
        monitor='Append-only stdout and actual process; no live JSON reads',
        scope='Complete original E2 N/d/M grid; VLMRS M not applicable; preserved 117 + new 247 calls')
    def dump():retry_writer(original_dump,root/'batch.json',b)
    dump();print('BATCH_DIR='+str(root),flush=True)
    for job in b['jobs']:
        changed=[rel for rel,h in hashes.items() if file_sha(ROOT/rel)!=h]
        if changed:
            b.update(status='stopped_source_changed',changed=changed);dump();return 2
        folder=root/job['name'];folder.mkdir()
        command=[sys.executable,'-u','-m','experiments.point1_compare.run_full_scale_point','--point',','.join(map(str,job['point']))]
        env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
        env['POINT1_FULL_SCALE_COORDINATOR_PID']=str(os.getpid())
        with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
            child=subprocess.Popen(command,cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                creationflags=0x00000208,close_fds=True)
        job.update(status='running',pid=child.pid,started_utc=now(),command=command,
            process_create_time=psutil.Process(child.pid).create_time(),log_dir=str(folder),
            child_creationflags='0x00000208: detached group; no second breakaway request');dump()
        print('START='+job['name']+' PID='+str(child.pid),flush=True)
        offset=0;pending=b'';run=None
        while True:
            with (folder/'stdout.log').open('rb') as stream:
                stream.seek(offset);chunk=stream.read();offset=stream.tell()
            pending+=chunk
            lines=pending.split(b'\n');pending=lines.pop()
            for raw in lines:
                line=raw.decode('utf-8',errors='replace').rstrip('\r')
                print(job['name']+': '+line,flush=True)
                if line.startswith('RUN_DIR='):
                    run=Path(line.split('=',1)[1]);job['run_dir']=str(run);dump()
            code=child.poll()
            if code is not None:
                # Final log may have become visible after this iteration's read.
                with (folder/'stdout.log').open('rb') as stream:
                    stream.seek(offset);tail=stream.read()
                for line in (pending+tail).decode('utf-8',errors='replace').splitlines():
                    if line:
                        print(job['name']+': '+line,flush=True)
                        if line.startswith('RUN_DIR='):run=Path(line.split('=',1)[1]);job['run_dir']=str(run)
                break
            time.sleep(5)
        job.update(return_code=code,finished_utc=now())
        errors=validate_point(run) if run and code==0 else ['Point process did not complete normally']
        job.update(validation_errors=errors,status='completed' if code==0 and not errors else 'incomplete')
        if run:job['manifest_sha256']=file_sha(run/'manifest.json')
        dump()
        if job['status']!='completed':
            b['status']='stopped_needs_review';dump()
            print('STOPPED_NEEDS_REVIEW='+job['name'],flush=True);return code or 2
        print('VALIDATED='+job['name'],flush=True)
    b.update(status='completed_full_scale_scan',finished_utc=now());dump()
    errors=validate_batch(root/'batch.json')
    retry_writer(original_dump,root/'validation.json',dict(valid=not errors,errors=errors,batch_sha256=file_sha(root/'batch.json')))
    if errors:raise ValueError(str(errors))
    print('BATCH_COMPLETED='+str(root),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
