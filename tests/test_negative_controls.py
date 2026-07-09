from pathlib import Path

from dpeot.experiments.export_negative_controls import (
    format_latex_table,
    format_markdown_table,
    run_negative_controls,
    write_negative_control_artifacts,
)


def test_negative_controls_run_true_and_no_merge_cases() -> None:
    rows = run_negative_controls(
        num_trials=1,
        base_seed=13,
    )

    assert [row["scenario"] for row in rows] == [
        "true_merge",
        "near_miss_no_merge",
        "parallel_close_tracks",
        "single_large_extended_target",
    ]
    assert float(rows[0]["group_detection_recall"]) >= 0.0
    assert float(rows[3]["false_group_scan_rate"]) == 0.0
    assert float(rows[3]["active_group_scan_rate"]) == 0.0


def test_negative_control_exports_tables(tmp_path: Path) -> None:
    rows = [
        {
            "scenario": "true_merge",
            "num_trials": 1,
            "group_detection_precision": 1.0,
            "group_detection_recall": 1.0,
            "group_detection_f1": 1.0,
            "false_group_scan_rate": 0.0,
            "false_group_scans": 0.0,
            "missed_group_scans": 0.0,
            "wrong_membership_scans": 0.0,
            "active_group_scan_rate": 0.17,
            "true_group_scans": 7.0,
            "runtime_ms_per_scan": 0.25,
        },
        {
            "scenario": "near_miss_no_merge",
            "num_trials": 1,
            "group_detection_precision": 0.0,
            "group_detection_recall": 0.0,
            "group_detection_f1": 0.0,
            "false_group_scan_rate": 0.01,
            "false_group_scans": 0.4,
            "missed_group_scans": 0.0,
            "wrong_membership_scans": 0.0,
            "active_group_scan_rate": 0.01,
            "true_group_scans": 0.0,
            "runtime_ms_per_scan": 0.2,
        },
    ]

    markdown = format_markdown_table(rows)
    latex = format_latex_table(rows)

    assert "True merge" in markdown
    assert "Near miss" in markdown
    assert "False-rate" in markdown
    assert "\\begin{table*}" in latex
    assert "Group-F1" in latex

    write_negative_control_artifacts(
        rows=rows,
        output_dir=tmp_path,
        num_trials=1,
        base_seed=13,
    )

    assert (tmp_path / "negative_controls.json").exists()
    assert (tmp_path / "negative_controls.md").exists()
    assert (tmp_path / "negative_controls_table.tex").exists()
