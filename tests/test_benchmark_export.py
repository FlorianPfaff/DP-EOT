from pathlib import Path

from dpeot.experiments.export_two_target_benchmark import (
    format_latex_table,
    format_markdown_table,
    write_benchmark_artifacts,
)


ROWS = [
    {
        "method": "proposed_group_labels",
        "id_switches": 0.0,
        "id_switches_total": 2.0,
        "id_switches_pre_merge": 0.0,
        "id_switches_during_unresolved": 2.0,
        "id_switches_post_split": 0.0,
        "id_switches_resolved": 0.0,
        "label_recovery": 1.0,
        "label_recovery_post_split": 1.0,
        "split_delay": 0.0,
        "split_recovery_delay": 0.0,
        "group_membership": 1.0,
        "group_membership_during_unresolved": 1.0,
        "position_error": 0.12345,
        "runtime_ms_per_scan": 0.98765,
    },
    {
        "method": "mfm_x_order",
        "id_switches": 2.0,
        "id_switches_total": 2.0,
        "id_switches_pre_merge": 0.0,
        "id_switches_during_unresolved": 0.0,
        "id_switches_post_split": 2.0,
        "id_switches_resolved": 2.0,
        "label_recovery": 0.0,
        "label_recovery_post_split": 0.0,
        "split_delay": 18.0,
        "split_recovery_delay": 18.0,
        "group_membership": 0.0,
        "group_membership_during_unresolved": 0.0,
        "position_error": 0.54321,
        "runtime_ms_per_scan": 1.23456,
    },
    {
        "method": "labeled_split_hypothesis",
        "id_switches": 0.0,
        "id_switches_total": 2.0,
        "id_switches_pre_merge": 0.0,
        "id_switches_during_unresolved": 2.0,
        "id_switches_post_split": 0.0,
        "id_switches_resolved": 0.0,
        "label_recovery": 1.0,
        "label_recovery_post_split": 1.0,
        "split_delay": 0.0,
        "split_recovery_delay": 0.0,
        "group_membership": 0.0,
        "group_membership_during_unresolved": 0.0,
        "position_error": 0.23456,
        "runtime_ms_per_scan": 1.11111,
    },
]


def test_markdown_export_contains_human_readable_method_names() -> None:
    table = format_markdown_table(ROWS)

    assert "Proposed group labels" in table
    assert "MFM x-order" in table
    assert "Labeled split hypothesis" in table
    assert "IDsw-post" in table
    assert "0.123" in table


def test_latex_export_is_complete_table() -> None:
    table = format_latex_table(ROWS)

    assert "\\begin{table}" in table
    assert "Proposed group labels" in table
    assert "MFM x-order" in table
    assert "Labeled split hypothesis" in table
    assert "IDsw-post" in table
    assert "\\end{table}" in table


def test_write_benchmark_artifacts(tmp_path: Path) -> None:
    write_benchmark_artifacts(ROWS, tmp_path, num_trials=1, base_seed=7)

    assert (tmp_path / "two_target_benchmark.json").exists()
    assert (tmp_path / "two_target_benchmark.md").exists()
    assert (tmp_path / "two_target_benchmark_table.tex").exists()
