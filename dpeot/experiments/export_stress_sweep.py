"""Export synthetic stress sweeps for the merge/split benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any

import numpy as np

from dpeot.experiments.run_two_target_crossing import _method_factories, _summarize_result
from dpeot.scenarios.two_target_merge_split import ScenarioConfig, generate_two_target_merge_split


MERGE_DURATIONS = (3, 5, 7, 10, 15)
CLUTTER_RATES = (0.0, 2.0, 5.0, 10.0)
NOISE_LEVELS = {
    "low": 0.04,
    "medium": 0.08,
    "high": 0.16,
}
EXTENT_MODES = {
    "same": ((0.8, 0.25), (0.8, 0.25)),
    "asymmetric": ((0.8, 0.25), (1.15, 0.18)),
}
RATE_MODES = {
    "equal": (12.0, 12.0),
    "asymmetric": (16.0, 8.0),
}
CROSSING_ANGLES = {
    "shallow": 0.35,
    "steep": 2.0,
}
STRESS_METHODS = (
    "distance_collapse",
    "dp_x_order",
    "mfm_x_order",
    "labeled_split_hypothesis",
    "proposed_group_labels",
)
CSV_FIELDS = (
    "method",
    "merge_duration",
    "clutter_rate",
    "noise_level",
    "extent_similarity",
    "rate_asymmetry",
    "crossing_angle",
    "num_trials",
    "label_recovery_post_split",
    "split_recovery_delay",
    "group_membership_during_unresolved",
    "group_detection_precision",
    "group_detection_recall",
    "group_detection_f1",
    "merge_onset_delay",
    "split_release_delay",
    "false_group_scans",
    "missed_group_scans",
    "wrong_membership_scans",
    "id_switches_post_split",
    "position_error",
    "runtime_ms_per_scan",
)


@dataclass(frozen=True)
class StressFactor:
    """One synthetic stress setting."""

    merge_duration: int
    clutter_rate: float
    noise_level: str
    extent_similarity: str
    rate_asymmetry: str
    crossing_angle: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=7000)
    parser.add_argument("--profile", choices=("heatmap", "full"), default="heatmap")
    parser.add_argument("--output-dir", type=Path, default=Path("results/stress_sweep"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    rows = run_stress_sweep(
        num_trials=args.num_trials,
        base_seed=args.base_seed,
        profile=args.profile,
        workers=args.workers,
    )
    write_stress_artifacts(
        rows=rows,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        num_trials=args.num_trials,
        base_seed=args.base_seed,
        profile=args.profile,
        write_figures=not args.no_figures,
    )
    print(format_heatmap_markdown(rows))


def run_stress_sweep(
    num_trials: int = 5,
    base_seed: int = 7000,
    profile: str = "heatmap",
    methods: Sequence[str] = STRESS_METHODS,
    merge_durations: Sequence[int] = MERGE_DURATIONS,
    clutter_rates: Sequence[float] = CLUTTER_RATES,
    noise_levels: Sequence[str] | None = None,
    extent_modes: Sequence[str] | None = None,
    rate_modes: Sequence[str] | None = None,
    crossing_angles: Sequence[str] | None = None,
    workers: int = 1,
) -> list[dict[str, float | int | str]]:
    """Run a synthetic stress sweep and return aggregate rows."""

    if num_trials <= 0:
        raise ValueError("num_trials must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")

    factors = list(
        iter_stress_factors(
            profile=profile,
            merge_durations=merge_durations,
            clutter_rates=clutter_rates,
            noise_levels=noise_levels,
            extent_modes=extent_modes,
            rate_modes=rate_modes,
            crossing_angles=crossing_angles,
        )
    )

    jobs = [
        (factor_index, factor, num_trials, base_seed, tuple(methods))
        for factor_index, factor in enumerate(factors)
    ]
    if workers == 1:
        factor_rows = [_run_stress_factor(job) for job in jobs]
    else:
        with mp.Pool(processes=workers) as pool:
            factor_rows = pool.map(_run_stress_factor, jobs)

    rows: list[dict[str, float | int | str]] = []
    for rows_for_factor in factor_rows:
        rows.extend(rows_for_factor)
    return rows


def iter_stress_factors(
    profile: str,
    merge_durations: Sequence[int] = MERGE_DURATIONS,
    clutter_rates: Sequence[float] = CLUTTER_RATES,
    noise_levels: Sequence[str] | None = None,
    extent_modes: Sequence[str] | None = None,
    rate_modes: Sequence[str] | None = None,
    crossing_angles: Sequence[str] | None = None,
) -> Iterable[StressFactor]:
    """Yield stress factors for either the heatmap slice or full factorial grid."""

    if profile not in {"heatmap", "full"}:
        raise ValueError("profile must be 'heatmap' or 'full'")

    if profile == "heatmap":
        noise_levels = noise_levels or ("medium",)
        extent_modes = extent_modes or ("same",)
        rate_modes = rate_modes or ("equal",)
        crossing_angles = crossing_angles or ("shallow",)
    else:
        noise_levels = noise_levels or tuple(NOISE_LEVELS)
        extent_modes = extent_modes or tuple(EXTENT_MODES)
        rate_modes = rate_modes or tuple(RATE_MODES)
        crossing_angles = crossing_angles or tuple(CROSSING_ANGLES)

    for merge_duration in merge_durations:
        for clutter_rate in clutter_rates:
            for noise_level in noise_levels:
                for extent_mode in extent_modes:
                    for rate_mode in rate_modes:
                        for crossing_angle in crossing_angles:
                            _validate_factor_values(
                                noise_level, extent_mode, rate_mode, crossing_angle
                            )
                            yield StressFactor(
                                merge_duration=int(merge_duration),
                                clutter_rate=float(clutter_rate),
                                noise_level=noise_level,
                                extent_similarity=extent_mode,
                                rate_asymmetry=rate_mode,
                                crossing_angle=crossing_angle,
                            )


def stress_scenario_config(factor: StressFactor, seed: int) -> ScenarioConfig:
    """Build a scenario config from one stress factor."""

    merge_start, merge_end = _merge_window(factor.merge_duration)
    extent_a, extent_b = EXTENT_MODES[factor.extent_similarity]
    rates = RATE_MODES[factor.rate_asymmetry]
    return ScenarioConfig(
        merge_start=merge_start,
        merge_end=merge_end,
        measurement_rate=rates[0],
        measurement_rates=rates,
        clutter_rate=factor.clutter_rate,
        extent_axes=extent_a,
        extent_axes_b=extent_b,
        measurement_noise_std=NOISE_LEVELS[factor.noise_level],
        crossing_y_offset=CROSSING_ANGLES[factor.crossing_angle],
        seed=seed,
    )


def write_stress_artifacts(
    rows: list[dict[str, float | int | str]],
    output_dir: Path,
    figure_dir: Path,
    num_trials: int,
    base_seed: int,
    profile: str,
    write_figures: bool = True,
) -> None:
    """Write JSON, CSV, Markdown, and optional heatmap PNG artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "benchmark": "two_target_stress_sweep",
        "num_trials": num_trials,
        "base_seed": base_seed,
        "profile": profile,
        "methods": list(STRESS_METHODS),
        "axes": _active_axes(rows),
        "available_axes": {
            "merge_duration": list(MERGE_DURATIONS),
            "clutter_rate": list(CLUTTER_RATES),
            "noise_level": list(NOISE_LEVELS),
            "extent_similarity": list(EXTENT_MODES),
            "rate_asymmetry": list(RATE_MODES),
            "crossing_angle": list(CROSSING_ANGLES),
        },
        "rows": rows,
    }
    (output_dir / "stress_sweep.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "stress_sweep.csv", rows)
    (output_dir / "stress_label_recovery_heatmaps.md").write_text(
        format_heatmap_markdown(rows) + "\n",
        encoding="utf-8",
    )
    if write_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_label_recovery_heatmaps(
            rows,
            figure_dir / "stress_label_recovery_heatmaps.png",
        )


def _run_stress_factor(
    job: tuple[int, StressFactor, int, int, tuple[str, ...]],
) -> list[dict[str, float | int | str]]:
    factor_index, factor, num_trials, base_seed, methods = job
    method_summaries: dict[str, list[dict[str, float]]] = {method: [] for method in methods}
    for trial in range(num_trials):
        seed = base_seed + 1000 * factor_index + trial
        config = stress_scenario_config(factor, seed=seed)
        scenario = generate_two_target_merge_split(config)
        factories = _method_factories(scenario)
        for method in methods:
            start = perf_counter()
            result = factories[method]()
            elapsed = perf_counter() - start
            method_summaries[method].append(_summarize_result(scenario, result, elapsed))

    return [
        _stress_row(factor, method, summaries, num_trials)
        for method, summaries in method_summaries.items()
    ]


def format_heatmap_markdown(rows: Sequence[dict[str, float | int | str]]) -> str:
    """Return one Markdown heatmap table per method."""

    durations = sorted({int(row["merge_duration"]) for row in rows})
    clutter_rates = sorted({float(row["clutter_rate"]) for row in rows})
    lines = [
        "# Stress Sweep: Post-Split Label Recovery",
        "",
        "Rows are clutter rates; columns are merge durations.",
    ]
    for method in STRESS_METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        values = _heatmap_values(method_rows, durations, clutter_rates)
        lines.extend(["", f"## {method}", ""])
        lines.append("| clutter \\ duration | " + " | ".join(str(d) for d in durations) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in durations) + " |")
        for clutter_rate, row_values in zip(clutter_rates, values):
            formatted = " | ".join(f"{value:.2f}" for value in row_values)
            lines.append(f"| {clutter_rate:g} | {formatted} |")
    return "\n".join(lines)


def plot_label_recovery_heatmaps(
    rows: Sequence[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    """Plot one post-split label-recovery heatmap per stress method."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    durations = sorted({int(row["merge_duration"]) for row in rows})
    clutter_rates = sorted({float(row["clutter_rate"]) for row in rows})
    fig, axes = plt.subplots(
        1,
        len(STRESS_METHODS),
        figsize=(3.1 * len(STRESS_METHODS), 3.4),
        constrained_layout=True,
    )
    if len(STRESS_METHODS) == 1:
        axes = [axes]

    image = None
    for axis, method in zip(axes, STRESS_METHODS):
        method_rows = [row for row in rows if row["method"] == method]
        values = np.asarray(_heatmap_values(method_rows, durations, clutter_rates))
        image = axis.imshow(values, vmin=0.0, vmax=1.0, origin="lower", cmap="viridis")
        axis.set_title(method.replace("_", "\n"))
        axis.set_xticks(range(len(durations)), labels=[str(d) for d in durations])
        axis.set_yticks(range(len(clutter_rates)), labels=[f"{rate:g}" for rate in clutter_rates])
        axis.set_xlabel("merge duration")
        axis.set_ylabel("clutter rate")
        for y_index, row_values in enumerate(values):
            for x_index, value in enumerate(row_values):
                axis.text(
                    x_index,
                    y_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.55 else "black",
                    fontsize=7,
                )

    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="post-split recovery")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _stress_row(
    factor: StressFactor,
    method: str,
    summaries: Sequence[dict[str, float]],
    num_trials: int,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        **asdict(factor),
        "method": method,
        "num_trials": num_trials,
    }
    for key in (
        "label_recovery_post_split",
        "split_recovery_delay",
        "group_membership_during_unresolved",
        "group_detection_precision",
        "group_detection_recall",
        "group_detection_f1",
        "merge_onset_delay",
        "split_release_delay",
        "false_group_scans",
        "missed_group_scans",
        "wrong_membership_scans",
        "id_switches_post_split",
        "position_error",
        "runtime_per_scan",
    ):
        row[key] = mean(summary[key] for summary in summaries)
    row["runtime_ms_per_scan"] = 1000.0 * float(row["runtime_per_scan"])
    row["label_recovery_post_split_std"] = _std(summaries, "label_recovery_post_split")
    return row


def _write_csv(path: Path, rows: Sequence[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _active_axes(rows: Sequence[dict[str, float | int | str]]) -> dict[str, list[float | int | str]]:
    return {
        "merge_duration": sorted({int(row["merge_duration"]) for row in rows}),
        "clutter_rate": sorted({float(row["clutter_rate"]) for row in rows}),
        "noise_level": sorted({str(row["noise_level"]) for row in rows}),
        "extent_similarity": sorted({str(row["extent_similarity"]) for row in rows}),
        "rate_asymmetry": sorted({str(row["rate_asymmetry"]) for row in rows}),
        "crossing_angle": sorted({str(row["crossing_angle"]) for row in rows}),
    }


def _heatmap_values(
    rows: Sequence[dict[str, float | int | str]],
    durations: Sequence[int],
    clutter_rates: Sequence[float],
) -> list[list[float]]:
    values: list[list[float]] = []
    for clutter_rate in clutter_rates:
        row_values: list[float] = []
        for duration in durations:
            matching = [
                float(row["label_recovery_post_split"])
                for row in rows
                if int(row["merge_duration"]) == duration
                and float(row["clutter_rate"]) == clutter_rate
            ]
            row_values.append(mean(matching) if matching else float("nan"))
        values.append(row_values)
    return values


def _merge_window(duration: int, num_steps: int = 41) -> tuple[int, int]:
    if duration < 1:
        raise ValueError("merge duration must be positive")
    center = (num_steps - 1) // 2
    start = center - (duration - 1) // 2
    end = start + duration - 1
    if start < 0 or end >= num_steps:
        raise ValueError("merge duration does not fit into the scenario")
    return start, end


def _std(summaries: Sequence[dict[str, float]], key: str) -> float:
    values = [summary[key] for summary in summaries]
    return pstdev(values) if len(values) > 1 else 0.0


def _validate_factor_values(
    noise_level: str,
    extent_mode: str,
    rate_mode: str,
    crossing_angle: str,
) -> None:
    if noise_level not in NOISE_LEVELS:
        raise ValueError(f"unknown noise level: {noise_level}")
    if extent_mode not in EXTENT_MODES:
        raise ValueError(f"unknown extent mode: {extent_mode}")
    if rate_mode not in RATE_MODES:
        raise ValueError(f"unknown rate mode: {rate_mode}")
    if crossing_angle not in CROSSING_ANGLES:
        raise ValueError(f"unknown crossing angle: {crossing_angle}")


if __name__ == "__main__":
    main()
