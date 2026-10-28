"""One-shot completion action for the two already running experiment processes.

No clock-based shutdown: both captured Windows process handles must exit with
code zero, the named batch and its OVP dependent must reconcile, and a summary
must be saved first. Any uncertainty leaves the machine on. The user can cancel
by setting armed=false in control.json. No /f or implicit forced timeout.
"""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone, timedelta
import psutil

ZONE = timezone(timedelta(hours=8))


def now():
    return datetime.now(ZONE).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class CapturedProcess:
    """Hold the original OS handle so exit codes cannot be confused by PID reuse."""
    def __init__(self, pid, expected_module):
        process = psutil.Process(pid)
        if expected_module not in process.cmdline():
            raise ValueError(f"PID {pid} does not run the expected module")
        self.pid, self.created = pid, process.create_time()
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.GetExitCodeProcess.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.OpenProcess(0x1000 | 0x00100000, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if psutil.Process(pid).create_time() != self.created:
            self.close()
            raise ValueError("Process identity changed during capture")

    def exit_code(self):
        code = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return None if code.value == 259 else code.value

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def terminal_batch(batch, original_identity):
    if (batch["pid"], batch["started"]) != original_identity:
        raise ValueError("Batch identity changed")
    if batch["status"] == "running":
        return False
    if batch["status"] != "completed_hvsa_jobs_only":
        raise ValueError("Batch did not finish successfully: "+batch["status"])
    if any(j["status"] != "completed" or j.get("returncode") != 0
           or j.get("invalid_observations") != 0 or j.get("validation_problems")
           for j in batch["jobs"]):
        raise ValueError("A batch job was skipped, failed or needs review")
    return True


def validate_and_summarize(batch_path, prior_ovp_runs, report_path):
    from .validate_results import validate
    batch_path, report_path = Path(batch_path).resolve(), Path(report_path).resolve()
    batch = read(batch_path)
    jobs = batch.get("continued_from", {}).get("completed_jobs", []) + batch["jobs"]
    sources, lines = [], ["# 本轮实验执行完成记录", "", "记录时间："+now(), "",
        "此记录覆盖已排队的HVSA性能批次及OVP条件诊断。基础三方案复现、正式跨论文排名和全部论文材料仍未完成，不能据此宣称SOTA。", "",
        "| 任务 | 实际回合（含预热） | 正确回合 | 启用验证并通过 | 原始运行 |",
        "|---|---:|---:|---:|---|"]
    for job in jobs:
        directory = Path(job["run_dir"])
        if sha(directory/"manifest.json") != job["manifest_sha256"]:
            raise ValueError("Completed manifest changed: "+str(directory))
        problems = validate(directory)
        if problems:
            raise ValueError("Result reconciliation failed: "+"; ".join(problems))
        corr = read(directory/"correctness.json")
        obs = corr["observations"]
        sources.append({"task": job["name"], "run_dir": str(directory), "manifest_sha256": job["manifest_sha256"]})
        lines.append(f"| {job['name']} | {len(obs)} | {sum(o['exact_match'] for o in obs)} | "
                     f"{sum(o['verified'] is True for o in obs)} | {directory.name} |")
    matches = []
    for directory in batch_path.parent.parent.glob("*_ovp_full_diagnostic"):
        if str(directory.resolve()) in prior_ovp_runs:
            continue
        manifest = read(directory/"manifest.json")
        dep = manifest.get("prior_batch_dependency") or {}
        if (Path(dep.get("path", "")).resolve() == batch_path and dep.get("sha256") == sha(batch_path)
                and manifest.get("N") == 20 and manifest.get("d") == 257
                and manifest.get("config_snapshot", {}).get("check_components") is True):
            matches.append(directory)
    if len(matches) != 1:
        raise ValueError("Exactly one completed dependent OVP run is required")
    ovp = matches[0]
    problems = validate(ovp)
    if problems:
        raise ValueError("OVP artifact reconciliation failed: "+"; ".join(problems))
    corr = read(ovp/"correctness.json")
    if corr["execution_errors"] != 0:
        raise ValueError("OVP honest execution failed")
    observations = corr["observations"]
    if any(not (o["completed"] and o["exact_match"] and o["verified"])
           for o in observations if o["case"] in ("honest", "before_upload", "partial_upload")):
        raise ValueError("OVP honest/dropout chain requires review")
    sources.append({"task": "ovp_full_diagnostic", "run_dir": str(ovp), "manifest_sha256": sha(ovp/"manifest.json")})
    lines += ["", "OVP为明确实例化条件下的准入诊断，不是合格原文性能复现。攻击案例的接受/拒绝是观察结果，不把错误接受自动写成原论文漏洞。", "",
              "| OVP案例 | 完整上传客户端数 | 聚合正确 | 验证接受 |", "|---|---:|---|---|"]
    for o in observations:
        lines.append(f"| {o['case']} | {len(o['uploaded'])} | {o['exact_match']} | {o['verified']} |")
    lines += ["", "原始数据、源码与输出摘要均已保存并复核；后续图表合并和人工目视检查尚待完成。", "",
              "用户已授权：本轮执行并保存结果后正常关机。此次关机不强制结束其他应用。", ""]
    if report_path.exists():
        raise ValueError("Refusing to overwrite an existing completion report")
    with report_path.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines)); stream.flush(); os.fsync(stream.fileno())
    return {"report": str(report_path), "report_sha256": sha(report_path), "sources": sources}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--ovp-pid", required=True, type=int)
    parser.add_argument("--control-dir", required=True)
    args = parser.parse_args(argv)
    if os.name != "nt":
        raise ValueError("This completion action requires Windows")
    control = Path(args.control_dir).resolve()
    control.mkdir(parents=True, exist_ok=False)
    settings, state_path = control/"control.json", control/"state.json"
    save(settings, {"armed": True, "batch": str(Path(args.batch).resolve()),
                    "user_authorization": "2026-10-19: 执行好了帮我关机", "created": now()})
    state = {"status": "initializing", "pid": os.getpid(), "updated": now(), "script_sha256": sha(__file__)}
    handles = []
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    kernel.SetThreadExecutionState.restype = wintypes.DWORD
    try:
        batch_path = Path(args.batch).resolve()
        first = read(batch_path)
        identity = (first["pid"], first["started"])
        handles = [CapturedProcess(first["pid"], "experiments.point1_compare.run_hvsa_batch"),
                   CapturedProcess(args.ovp_pid, "experiments.point1_compare.probe_ovp_full")]
        prior = {str(p.resolve()) for p in batch_path.parent.parent.glob("*_ovp_full_diagnostic")}
        if not kernel.SetThreadExecutionState(0x80000001):
            raise ctypes.WinError(ctypes.get_last_error())
        state.update(status="waiting_for_experiments", batch=str(batch_path),
                     processes=[{"pid": h.pid, "created": h.created} for h in handles])
        save(state_path, state)
        print("ARMED="+str(settings), flush=True)
        while True:
            if read(settings).get("armed") is not True:
                state["status"] = "cancelled"; break
            batch = read(batch_path)
            done = terminal_batch(batch, identity)
            codes = [h.exit_code() for h in handles]
            if any(code is not None and code != 0 for code in codes):
                raise ValueError("An experiment process failed: "+repr(codes))
            if codes[0] == 0 and not done:
                raise ValueError("Batch process exited without a complete journal")
            if codes[1] == 0 and not done:
                raise ValueError("OVP exited before the required batch completion")
            if done and codes == [0, 0]:
                state.update(status="validating_and_saving", updated=now()); save(state_path, state)
                report = control.parents[2]/"docs"/"POINT1_EXECUTION_COMPLETION_20261019.md"
                state.update(validate_and_summarize(batch_path, prior, report))
                state.update(status="ready_for_shutdown", updated=now()); save(state_path, state)
                # A Python grace period avoids shutdown.exe /t>0, which implies /f.
                for _ in range(30):
                    if read(settings).get("armed") is not True:
                        state["status"] = "cancelled"; break
                    time.sleep(1)
                if state["status"] == "cancelled":
                    break
                state.update(status="shutdown_requested", updated=now()); save(state_path, state)
                shutdown = Path(os.environ["SystemRoot"])/"System32"/"shutdown.exe"
                subprocess.run([str(shutdown), "/s", "/t", "0", "/d", "p:0:0", "/c",
                                "HVSA experiment jobs completed and verified; user requested shutdown."], check=True)
                break
            time.sleep(10)
    except Exception as exc:
        state.update(status="stopped_without_shutdown", error=f"{type(exc).__name__}: {exc}", updated=now())
        print(state["error"], flush=True)
    finally:
        for handle in handles:
            handle.close()
        kernel.SetThreadExecutionState(0x80000000)
        state["updated"] = now()
        save(state_path, state)
    return int(state["status"] == "stopped_without_shutdown")


if __name__ == "__main__":
    raise SystemExit(main())
