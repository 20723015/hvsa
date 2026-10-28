"""Deadlines must actually stop work and preserve the interrupted context."""
import json
import subprocess
import sys
from pathlib import Path
from experiments.point1_compare.watchdog import RoundDeadline


def test_deadline_terminates_child_after_preserving_context(tmp_path):
    target = tmp_path/"timeout.json"
    code = """
import json,sys,time
from pathlib import Path
from experiments.point1_compare.watchdog import RoundDeadline
p=Path(sys.argv[1])
with RoundDeadline(.05, lambda: p.write_text(json.dumps({'round':7,'completed':False}))):
    time.sleep(5)
raise SystemExit(9)
"""
    child = subprocess.run([sys.executable, "-c", code, str(target)], timeout=10,
                           cwd=Path(__file__).resolve().parents[1], capture_output=True)
    assert child.returncode == 124
    assert json.loads(target.read_text()) == {"round": 7, "completed": False}


def test_finished_round_cancels_deadline():
    called = []
    with RoundDeadline(10, lambda: called.append(True)) as deadline:
        pass
    assert deadline._closed and not called
