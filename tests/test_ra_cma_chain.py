from pathlib import Path

import analyze_tier1
import run_tier1_batch


def _write_log(path: Path, mode: str, raw: float, shadow: float):
    path.write_text(
        "Accuracy history: [(1, 70.0), (2, %.2f)]\n" % raw
        + f"  最终准确率: {raw:.2f}%\n"
        + f"  最终准确率(影子 {mode}): {shadow:.2f}%\n",
        encoding="utf-8",
    )


def test_parse_fixed_and_risk_shadow_logs(tmp_path):
    fixed = tmp_path / "fixed.txt"
    risk = tmp_path / "risk.txt"
    _write_log(fixed, "fixed", 80.0, 81.0)
    _write_log(risk, "risk", 80.0, 81.2)

    fixed_row = analyze_tier1.parse_log(fixed)
    risk_row = analyze_tier1.parse_log(risk)

    assert fixed_row["ema_acc"] == 81.0
    assert fixed_row["cma_acc"] == 81.0
    assert risk_row["ema_acc"] is None
    assert risk_row["cma_acc"] == 81.2


def test_promotion_contract():
    passed = analyze_tier1.promotion_result([0.9, 1.0, 0.8], 0.95)
    failed_positive = analyze_tier1.promotion_result([2.6, -0.1, -0.1], 0.7)

    assert passed["promote"] is True
    assert failed_positive["positive_ok"] is False
    assert failed_positive["promote"] is False


def test_runner_command_has_unique_tag():
    first = run_tier1_batch.build_command("RISK50_s42", "--dataset MNIST")
    second = run_tier1_batch.build_command("RISK50_s43", "--dataset MNIST")

    assert first[-2:] == ["--tag", "RISK50_s42"]
    assert second[-2:] == ["--tag", "RISK50_s43"]
    assert first != second


def test_screening_report_promotes_qualified_mock_logs(tmp_path, monkeypatch,
                                                       capsys):
    monkeypatch.chdir(tmp_path)

    for seed in analyze_tier1.SEEDS:
        _write_log(
            tmp_path / f"point2_log_SIG_NI_0.5_s{seed}_MNIST_mock.txt",
            "none", 80.0, 80.0,
        )
        for decay in analyze_tier1.EMA_DECAYS:
            gain = 0.9 if decay == 0.5 else 0.5
            _write_log(
                tmp_path / f"point2_log_EMA{decay}_s{seed}_MNIST_mock.txt",
                "fixed", 80.0, 80.0 + gain,
            )

    for tag, _ in analyze_tier1.CMA_SCREEN:
        for seed in analyze_tier1.CMA_SCREEN_SEEDS:
            gain = 0.95 if tag == "RISK50" else 0.2
            _write_log(
                tmp_path / f"point2_log_{tag}_s{seed}_MNIST_mock.txt",
                "risk", 80.0, 80.0 + gain,
            )

    analyze_tier1.main()
    output = capsys.readouterr().out

    risk50_line = next(line for line in output.splitlines()
                       if line.strip().startswith("RISK50"))
    assert "=> 晋级" in risk50_line
    assert "晋级排序: RISK50(+0.95pp)" in output
