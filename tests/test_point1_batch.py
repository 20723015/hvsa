"""A continuation must preserve completed work and refuse duplicate live jobs."""
import json
import pytest
from experiments.point1_compare.artifacts import digest
from experiments.point1_compare.run_hvsa_batch import continuation_jobs


def test_continuation_refuses_a_live_parent():
    with pytest.raises(ValueError, match="live worker"):
        continuation_jobs({"status": "running", "jobs": []})


@pytest.mark.parametrize("full_off", [False, True])
def test_continuation_reuses_done_jobs_and_fills_only_missing_off_control(tmp_path, full_off):
    manifest = ({"verification_off_scope": "full_per_round_including_ta_mask_hash", "settings": [
        {"N": 50, "M": 5, "d": d, "variant": "hvsa_verification_off", "verification_enabled": False}
        for d in [7850, 30000]]} if full_off else {})
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    parent = {"status": "stopped_source_changed", "jobs": [
        {"name": "e6_formal", "arguments": ["e6"], "status": "completed", "run_dir": str(tmp_path),
         "manifest_sha256": digest(path)},
        {"name": "e3_50_rounds", "arguments": ["e3"], "status": "pending"},
        {"name": "e4_dropouts", "arguments": ["e4"], "status": "completed_with_protocol_failures"}]}
    queue = continuation_jobs(parent)
    assert [name for name, _ in queue] == (["e3_50_rounds"] if full_off else ["e6_full_off_control", "e3_50_rounds"])
    if not full_off:
        assert queue[0][1] == ["e6", "--variants", "hvsa_verification_off"]
    path.write_text("{} changed", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        continuation_jobs(parent)
