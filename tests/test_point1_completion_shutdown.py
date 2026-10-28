"""Shutdown is fully mocked; exercise failure, cancellation and completion gates."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from experiments.point1_compare import finish_and_shutdown as completion


@pytest.mark.parametrize("mode", ["success", "failed_process", "failed_batch", "failed_report", "cancelled"])
def test_shutdown_requires_success_saved_report_and_live_authorization(tmp_path, monkeypatch, mode):
    batch = tmp_path/"results"/"point1_compare"/"batch"/"batch.json"
    batch.parent.mkdir(parents=True)
    control = batch.parent.parent/"shutdown"
    value = {"pid": 11, "started": "original", "status": "completed_hvsa_jobs_only", "jobs": [
        {"status": "completed", "returncode": 0, "invalid_observations": 0, "validation_problems": []}]}
    if mode == "failed_batch":
        value["status"] = "stopped_for_review"
    completion.save(batch, value)
    class Process:
        def __init__(self, pid, expected_module):
            self.pid, self.created = pid, 1
        def exit_code(self):
            return 1 if mode == "failed_process" and self.pid == 22 else 0
        def close(self):
            pass
    monkeypatch.setattr(completion, "CapturedProcess", Process)
    kernel = SimpleNamespace(SetThreadExecutionState=Mock(return_value=1))
    monkeypatch.setattr(completion.ctypes, "WinDLL", Mock(return_value=kernel))
    events = []
    def report(*args):
        events.append("report")
        if mode == "failed_report":
            raise ValueError("source checksum mismatch")
        return {"report": "saved_report.md", "report_sha256": "test"}
    monkeypatch.setattr(completion, "validate_and_summarize", report)
    def sleep(seconds):
        if mode == "cancelled":
            completion.save(control/"control.json", {"armed": False})
    monkeypatch.setattr(completion.time, "sleep", sleep)
    calls = []
    def shutdown(arguments, **kwargs):
        events.append("shutdown")
        calls.append(arguments)
    monkeypatch.setattr(completion.subprocess, "run", shutdown)
    code = completion.main(["--batch", str(batch), "--ovp-pid", "22", "--control-dir", str(control)])
    state = json.loads((control/"state.json").read_text(encoding="utf-8"))
    if mode == "success":
        assert code == 0 and events == ["report", "shutdown"]
        assert calls[0][1:4] == ["/s", "/t", "0"] and "/f" not in calls[0]
        assert state["status"] == "shutdown_requested"
    else:
        assert not calls
        assert state["status"] == ("cancelled" if mode == "cancelled" else "stopped_without_shutdown")


def test_incomplete_or_replaced_journal_cannot_qualify():
    value = {"pid": 11, "started": "original", "status": "running", "jobs": []}
    assert completion.terminal_batch(value, (11, "original")) is False
    with pytest.raises(ValueError, match="identity changed"):
        completion.terminal_batch(value, (12, "original"))
    value.update(status="completed_hvsa_jobs_only", jobs=[{"status": "pending"}])
    with pytest.raises(ValueError, match="skipped, failed or needs review"):
        completion.terminal_batch(value, (11, "original"))
