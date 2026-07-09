from pathlib import Path

from dpeot.experiments.summarize_stress_sweep import (
    format_method_summary_latex,
    format_method_summary_markdown,
    method_summary,
    proposed_worst_cases,
    write_summary_artifacts,
)


ROWS = [
    {
        "method": "proposed_group_labels",
        "merge_duration": 3,
        "clutter_rate": 0.0,
        "noise_level": "low",
        "extent_similarity": "same",
        "rate_asymmetry": "equal",
        "crossing_angle": "shallow",
        "label_recovery_post_split": 1.0,
        "group_detection_f1": 0.9,
        "position_error": 0.1,
        "runtime_ms_per_scan": 0.5,
    },
    {
        "method": "proposed_group_labels",
        "merge_duration": 5,
        "clutter_rate": 10.0,
        "noise_level": "high",
        "extent_similarity": "asymmetric",
        "rate_asymmetry": "asymmetric",
        "crossing_angle": "steep",
        "label_recovery_post_split": 0.8,
        "group_detection_f1": 0.6,
        "position_error": 0.2,
        "runtime_ms_per_scan": 0.7,
    },
    {
        "method": "distance_collapse",
        "merge_duration": 3,
        "clutter_rate": 0.0,
        "noise_level": "low",
        "extent_similarity": "same",
        "rate_asymmetry": "equal",
        "crossing_angle": "shallow",
        "label_recovery_post_split": 0.0,
        "group_detection_f1": 0.0,
        "position_error": 0.4,
        "runtime_ms_per_scan": 0.2,
    },
]


def test_method_summary_counts_failures_and_runtime_distribution() -> None:
    rows = method_summary(ROWS, failure_threshold=0.95)
    by_method = {row["method"]: row for row in rows}

    proposed = by_method["proposed_group_labels"]
    assert proposed["num_settings"] == 2
    assert proposed["mean_rec_post"] == 0.9
    assert proposed["mean_group_f1"] == 0.75
    assert proposed["worst_rec_post"] == 0.8
    assert proposed["failure_count"] == 1
    assert proposed["runtime_p50_ms_per_scan"] == 0.6

    distance = by_method["distance_collapse"]
    assert distance["failure_count"] == 1


def test_worst_cases_select_proposed_rows() -> None:
    worst = proposed_worst_cases(ROWS, limit=1)

    assert len(worst) == 1
    assert worst[0]["method"] == "proposed_group_labels"
    assert worst[0]["label_recovery_post_split"] == 0.8


def test_summary_export_writes_paper_artifacts(tmp_path: Path) -> None:
    summary = method_summary(ROWS, failure_threshold=0.95)
    markdown = format_method_summary_markdown(summary)
    latex = format_method_summary_latex(summary, failure_threshold=0.95)

    assert "Full Stress Sweep Summary" in markdown
    assert "Failure" in markdown
    assert "\\begin{table*}" in latex
    assert "Worst Rec." in latex

    write_summary_artifacts(
        rows=ROWS,
        output_dir=tmp_path,
        figure_dir=tmp_path,
        failure_threshold=0.95,
        write_figures=False,
    )

    assert (tmp_path / "stress_full_summary.json").exists()
    assert (tmp_path / "stress_full_method_summary.csv").exists()
    assert (tmp_path / "stress_full_method_summary.md").exists()
    assert (tmp_path / "stress_full_method_summary_table.tex").exists()
    assert (tmp_path / "stress_full_proposed_worst_cases.csv").exists()
    assert (tmp_path / "stress_full_proposed_worst_cases.md").exists()
    assert (tmp_path / "stress_full_heatmaps.md").exists()
