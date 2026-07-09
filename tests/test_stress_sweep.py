from pathlib import Path

from dpeot.experiments.export_stress_sweep import (
    format_heatmap_markdown,
    run_stress_sweep,
    write_stress_artifacts,
)


def test_stress_sweep_runs_reduced_grid(tmp_path: Path) -> None:
    rows = run_stress_sweep(
        num_trials=1,
        merge_durations=(3,),
        clutter_rates=(0.0,),
        noise_levels=("low",),
        extent_modes=("same",),
        rate_modes=("equal",),
        crossing_angles=("shallow",),
        methods=("distance_collapse", "proposed_group_labels"),
    )

    assert {row["method"] for row in rows} == {"distance_collapse", "proposed_group_labels"}
    assert all(row["merge_duration"] == 3 for row in rows)
    assert all(row["clutter_rate"] == 0.0 for row in rows)

    write_stress_artifacts(
        rows,
        output_dir=tmp_path,
        figure_dir=tmp_path,
        num_trials=1,
        base_seed=7000,
        profile="heatmap",
        write_figures=False,
    )

    assert (tmp_path / "stress_sweep.json").exists()
    assert (tmp_path / "stress_sweep.csv").exists()
    assert (tmp_path / "stress_label_recovery_heatmaps.md").exists()


def test_stress_heatmap_markdown_lists_methods() -> None:
    rows = run_stress_sweep(
        num_trials=1,
        merge_durations=(3,),
        clutter_rates=(0.0,),
        noise_levels=("low",),
        extent_modes=("same",),
        rate_modes=("equal",),
        crossing_angles=("shallow",),
        methods=("proposed_group_labels",),
    )

    table = format_heatmap_markdown(rows)

    assert "proposed_group_labels" in table
    assert "3" in table
