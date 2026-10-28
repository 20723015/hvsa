"""A completed-point view must match the actual archived checkpoints exactly."""
import json
from pathlib import Path
import pytest
from experiments.point1_compare import artifacts, export_completed_point, run_performance
from experiments.point1_compare.validate_results import validate


@pytest.fixture
def small_completed_point(tmp_path, monkeypatch):
    repository = Path(run_performance.ROOT)
    original_create = artifacts.create_run
    monkeypatch.setattr(run_performance, "ROOT", str(tmp_path))
    monkeypatch.setattr(export_completed_point, "ROOT", str(tmp_path))
    monkeypatch.setattr(run_performance, "create_run", lambda directory, manifest, ignored:
                        original_create(directory, manifest, repository))
    assert run_performance.main(["e6", "--point", "6,2,8", "--warmup", "1", "--repeats", "2"]) == 0
    source = next((tmp_path/"results"/"point1_compare").iterdir())
    return source


def test_derived_records_cannot_be_changed_even_with_recomputed_output_hashes(small_completed_point):
    directory = export_completed_point.export_point(small_completed_point, (6, 2, 8))
    assert validate(directory) == []
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    raw = [json.loads(line) for line in (directory/"raw.jsonl").read_text(encoding="utf-8").splitlines()]
    raw[0]["elapsed_ms"] += 10
    (directory/"raw.jsonl").write_text("\n".join(json.dumps(r) for r in raw)+"\n", encoding="utf-8")
    manifest["output_sha256"]["raw.jsonl"] = artifacts.digest(directory/"raw.jsonl")
    artifacts.dump_json(directory/"manifest.json", manifest)
    assert any("Derived records differ" in p for p in validate(directory))


def test_point_export_rejects_an_incomplete_control(small_completed_point):
    path = small_completed_point/"checkpoints"/"setting_001.json"
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    checkpoint["observations"].pop()
    artifacts.dump_json(path, checkpoint)
    with pytest.raises(ValueError, match="Point is incomplete"):
        export_completed_point.export_point(small_completed_point, (6, 2, 8))
