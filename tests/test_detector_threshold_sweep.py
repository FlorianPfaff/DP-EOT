from pathlib import Path

from dpeot.experiments.export_detector_threshold_sweep import (
    THRESHOLDS,
    format_latex_table,
    format_markdown_table,
    run_threshold_sweep,
    write_threshold_sweep_artifacts,
)


def test_default_thresholds_match_detector_calibration_grid() -> None:
    assert THRESHOLDS == (-10.0, -5.0, 0.0, 5.0, 10.0)


def test_threshold_sweep_runs_true_and_no_merge_cases() -> None:
    rows = run_threshold_sweep(
        num_trials=1,
        base_seed=17,
        thresholds=(-5.0, 0.0),
        scenario_names=("true_merge", "near_miss_no_merge"),
    )

    assert len(rows) == 4
    assert {row["scenario"] for row in rows} == {"true_merge", "near_miss_no_merge"}
    assert {float(row["threshold"]) for row in rows} == {-5.0, 0.0}
    for row in rows:
        assert 0.0 <= float(row["group_detection_recall"]) <= 1.0
        assert 0.0 <= float(row["false_group_scan_rate"]) <= 1.0


def test_threshold_sweep_exports_tables(tmp_path: Path) -> None:
    rows = [
        {
            "scenario": "true_merge",
            "threshold": 0.0,
            "num_trials": 1,
            "group_detection_precision": 1.0,
            "group_detection_recall": 1.0,
            "group_detection_f1": 1.0,
            "false_group_scan_rate": 0.0,
            "false_group_scans": 0.0,
            "missed_group_scans": 0.0,
            "wrong_membership_scans": 0.0,
            "active_group_scan_rate": 0.17,
            "merge_onset_delay": 0.0,
            "split_release_delay": 0.0,
            "runtime_ms_per_scan": 0.2,
        },
        {
            "scenario": "near_miss_no_merge",
            "threshold": 0.0,
            "num_trials": 1,
            "group_detection_precision": 0.0,
            "group_detection_recall": 0.0,
            "group_detection_f1": 0.0,
            "false_group_scan_rate": 0.01,
            "false_group_scans": 0.4,
            "missed_group_scans": 0.0,
            "wrong_membership_scans": 0.0,
            "active_group_scan_rate": 0.01,
            "merge_onset_delay": 0.0,
            "split_release_delay": 0.0,
            "runtime_ms_per_scan": 0.2,
        },
    ]

    markdown = format_markdown_table(rows)
    latex = format_latex_table(rows)

    assert "Merge Detector Threshold Sweep" in markdown
    assert "Near miss" in markdown
    assert "False scans" in markdown
    assert "\\begin{table*}" in latex
    assert "False-rate" in latex

    write_threshold_sweep_artifacts(
        rows=rows,
        output_dir=tmp_path,
        figure_dir=tmp_path,
        num_trials=1,
        base_seed=17,
        thresholds=(0.0,),
        write_figures=False,
    )

    assert (tmp_path / "detector_threshold_sweep.json").exists()
    assert (tmp_path / "detector_threshold_sweep.csv").exists()
    assert (tmp_path / "detector_threshold_sweep.md").exists()
    assert (tmp_path / "detector_threshold_sweep_table.tex").exists()
