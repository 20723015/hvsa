"""The measured runner must preserve real rounds and both dropout stages."""
import argparse
import json
from pathlib import Path

import pytest

from experiments.point1_compare import artifacts, run_performance
from experiments.point1_compare.validate_results import validate


def test_e2_deduplicates_overlapping_n_d_m_workpoints():
    config = json.loads((Path(run_performance.__file__).parent / "configs" / "full.json").read_text(encoding="utf-8"))
    args = argparse.Namespace(experiment="e2", point=None, scan="all", warmup=3, repeats=10)
    specs = run_performance.specifications(config, args)
    points = [(s["N"], s["M"], s["d"]) for s in specs]
    assert len(points) == len(set(points)) == 10
    assert all(s["warmup"] == 3 and s["rounds"] == 10 for s in specs)


@pytest.mark.parametrize("experiment", ["e3", "e4"])
def test_real_rounds_dropout_boundaries_and_persisted_contract(tmp_path, monkeypatch, experiment):
    repository = Path(run_performance.ROOT)
    monkeypatch.setattr(run_performance, "ROOT", str(tmp_path))
    original_create = artifacts.create_run
    monkeypatch.setattr(run_performance, "create_run", lambda directory, manifest, ignored:
                        original_create(directory, manifest, repository))
    config = {"round_scan": {"n": 6, "m": 2, "d": 8, "values": [1, 2, 3]},
              "dropout": {"n_selected": 6, "m": 2, "d": 8,
                          "phases": ["before_upload", "after_upload"], "rates": [0, .5, 1.],
                          "trajectory_seeds": [42, 43]}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    status = run_performance.main([experiment, "--config", str(config_path)])
    # All-before-upload dropouts are real failures; after-upload contributions remain.
    assert status == (1 if experiment == "e4" else 0)
    directory = next((tmp_path / "results" / "point1_compare").iterdir())
    assert validate(directory) == []
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    observations = json.loads((directory / "correctness.json").read_text(encoding="utf-8"))["observations"]
    if experiment == "e3":
        assert len(observations) == 6
        assert all(o["valid"] for o in observations)
        for s in manifest["settings"]:
            assert [o["round"] for o in observations if o["setting"] == s["setting"]] == [0, 1, 2]
    else:
        for s in manifest["settings"]:
            if s["dropout_rate"] == 1.:
                subset = [o for o in observations if o["setting"] == s["setting"]]
                if s["dropout_stage"] == "before_upload":
                    assert all(not o["valid"] and o["n_uploaded"] == 0 for o in subset)
                else:
                    assert all(o["valid"] and o["n_uploaded"] == 6 for o in subset)
