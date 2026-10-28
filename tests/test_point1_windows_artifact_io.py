"""Actual Windows sharing violations and bounded, artifact-only recovery."""
import ctypes
import json
import os
from pathlib import Path
import threading
import time
import pytest
from experiments.point1_compare.artifacts import dump_json
from experiments.point1_compare.windows_artifact_io import retry_writer,read_stdout_progress,install_writer_retry


def lock_without_delete(path):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
    kernel.CreateFileW.restype=ctypes.c_void_p;kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.CreateFileW(str(path),0x80000000,3,None,3,0x80,None)
    assert handle!=ctypes.c_void_p(-1).value
    return lambda:kernel.CloseHandle(handle)


@pytest.mark.skipif(os.name!='nt',reason='Real Windows sharing semantics')
def test_transient_destination_lock_recovers_atomically(tmp_path):
    path=tmp_path/'progress.json';dump_json(path,{'old':1});close=lock_without_delete(path)
    observed=threading.Event();events=[]
    def release():
        assert observed.wait(2);time.sleep(.07);close()
    thread=threading.Thread(target=release);thread.start()
    def record(event):events.append(event);observed.set()
    retry_writer(dump_json,path,{'new':2},on_retry=record)
    thread.join(2)
    assert events and not thread.is_alive() and json.loads(path.read_text())=={'new':2}
    assert not path.with_suffix('.json.partial').exists()


@pytest.mark.skipif(os.name!='nt',reason='Real Windows sharing semantics')
def test_permanent_lock_fails_with_original_target_intact(tmp_path):
    path=tmp_path/'progress.json';dump_json(path,{'old':1});close=lock_without_delete(path)
    tick=time.monotonic()
    try:
        with pytest.raises(PermissionError):retry_writer(dump_json,path,{'new':2},max_wait=.06)
    finally:close()
    assert .05<=time.monotonic()-tick<1
    assert json.loads(path.read_text())=={'old':1}


def test_other_errors_are_not_retried(tmp_path):
    calls=[]
    def failing(*args):calls.append(1);raise ValueError('serialization failure')
    with pytest.raises(ValueError):retry_writer(failing,tmp_path/'x.json',{})
    assert len(calls)==1


@pytest.mark.parametrize('newline',[b'\n',b'\r\n'])
def test_append_only_monitor_ignores_incomplete_lines(tmp_path,newline):
    path=tmp_path/'stdout.log'
    with path.open('wb',buffering=0) as writer:
        writer.write(b'INITIALIZED=cell hvsa wall_s=0.1'+newline+b'PROGRESS=1/240 cell round=0 hvsa valid=True'+newline+b'PROGRESS=2/240 cell valid=Fal')
        first=read_stdout_progress(path)
        assert first['completed_rounds']==1 and first['failed_calls']==0 and first['initialization_count']==1
        writer.write(b'se'+newline)
        second=read_stdout_progress(path)
        assert second['completed_rounds']==2 and second['failed_calls']==1


def test_hook_changes_only_artifact_aliases_and_restores_them(tmp_path):
    from experiments.point1_compare import artifacts,round_shards,run_large_dropout
    from experiments.point1_compare.adapters.hvsa_point1 import HVSAProtocol
    protocol=HVSAProtocol.run_round;old=artifacts.dump_json
    write,restore=install_writer_retry()
    try:
        assert artifacts.dump_json is round_shards.dump_json is run_large_dropout.dump_json is write
        write(tmp_path/'sample.json',{'value':3})
        assert HVSAProtocol.run_round is protocol
    finally:restore()
    assert artifacts.dump_json is round_shards.dump_json is run_large_dropout.dump_json is old
