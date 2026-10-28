"""Evidence selection must not cross a modified or unfinished batch lineage."""
import json
import pytest
from experiments.point1_compare.artifacts import digest
from experiments.point1_compare.report_cost_matrix import batch_lineage, load_points


def test_batch_lineage_retains_parent_and_rejects_tampering(tmp_path):
    parent, child = tmp_path/"parent.json", tmp_path/"child.json"
    parent.write_text(json.dumps({"status": "paused_by_user", "jobs": []}), encoding="utf-8")
    child.write_text(json.dumps({"status": "running", "jobs": [], "continued_from": {
        "path": str(parent), "sha256": digest(parent)}}), encoding="utf-8")
    assert len(batch_lineage(child)) == 2
    with pytest.raises(ValueError, match="must finish"):
        load_points(child)
    parent.write_text(json.dumps({"status": "completed_hvsa_jobs_only", "jobs": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="lineage changed"):
        batch_lineage(child)
