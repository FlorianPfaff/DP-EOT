"""Export merge-detector threshold calibration artifacts.

The proposed group-label tracker declares an unresolved group when the group
hypothesis beats the resolved-target hypothesis by a threshold. This exporter
sweeps that threshold on one true-merge scenario and several no-merge controls
so that the detector can be evaluated as a detector rather than only as a
tracker.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np

from dpeot.experiments.export_negative_controls import SCENARIO_FACTORIES, SCENARIO_LABELS
from dpeot.metrics.group_detection import group_detection_metrics
from dpeot.partitions.distance_partition import distance_partition
from dpeot.scenarios.two_target_merge_split import ScenarioConfig
from dpeot.tracking.merge_split_filter import FilterConfig, run_detected_group_filter


THRESHOLDS = (-25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0)
SCENARIOS = tuple(SCENARIO_FACTORIES)
CSV_FIELDS = (
    "scenario",
    "threshold",
    "num_trials",
    "group_detection_precision",
    "group_detection_recall",
    "group_detection_f1",
    "false_group_scan_rate",
    "false_group_scans",
    "missed_group_scans",
    "wrong_membership_scans",
    "active_group_scan_rate",
    "merge_onset_delay",
    "split_release_delay",
    "runtime_ms_per_scan",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=13000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/detector_threshold_sweep"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=list(THRESHOLDS),
        help="Merge-score thresholds to sweep.",
    )
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    rows = run_threshold_sweep(
        num_trials=args.num_trials,
        base_seed=args.base_seed,
        thresholds=tuple(args.thresholds),
    )
    write_threshold_sweep_artifacts(
        rows=rows,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        num_trials=args.num_trials,
        base_seed=args.base_seed,
        thresholds=tuple(args.thresholds),
        write_figures=not args.no_figures,
    )
    print(format_markdown_table(rows))


def run_threshold_sweep(
    num_trials: int = 50,
    base_seed: int = 13000,
    thresholds: Sequence[float] = THRESHOLDS,
    scenario_names: Sequence[str] = SCENARIOS,
) -> list[dict[str, float | int | str]]:
    """Run detector-threshold calibration over true and no-merge scenarios."""

    if num_trials <= 0:
        raise ValueError("num_trials must be positive")
    if not thresholds:
        raise ValueError("at least one threshold is required")

    rows: list[dict[str, float | int | str]] = []
    for scenario_index, scenario_name in enumerate(scenario_names):
        if scenario_name not in SCENARIO_FACTORIES:
            raise ValueError(f"unknown calibration scenario: {scenario_name}")
        for threshold_index, threshold in enumerate(thresholds):
            summaries = [
                _run_trial(
                    scenario_name=scenario_name,
                    threshold=float(threshold),
                    seed=base_seed + 10000 * scenario_index + 1000 * threshold_index + trial,
                )
                for trial in range(num_trials)
            ]
            rows.append(_aggregate_row(scenario_name, float(threshold), summaries, num_trials))
    return rows


def write_threshold_sweep_artifacts(
    rows: list[dict[str, float | int | str]],
    output_dir: Path,
    figure_dir: Path,
    num_trials: int,
    base_seed: int,
    thresholds: Sequence[float],
    write_figures: bool = True,
) -> None:
    """Write JSON, CSV, Markdown, LaTeX, and optional calibration figure."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "benchmark": "merge_detector_threshold_sweep",
        "num_trials": num_trials,
        "base_seed": base_seed,
        "thresholds": list(thresholds),
        "scenarios": list(SCENARIOS),
        "rows": rows,
    }
    (output_dir / "detector_threshold_sweep.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "detector_threshold_sweep.csv", rows)
    (output_dir / "detector_threshold_sweep.md").write_text(
        format_markdown_table(rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "detector_threshold_sweep_table.tex").write_text(
        format_latex_table(rows) + "\n",
        encoding="utf-8",
    )
    if write_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_threshold_sweep(rows, figure_dir / "detector_threshold_sweep.png")


def format_markdown_table(rows: list[dict[str, float | int | str]]) -> str:
    """Return compact Markdown calibration summary at the swept thresholds."""

    lines = [
        "# Merge Detector Threshold Sweep",
        "",
        "Recall should be high for the true merge. False-rate should be low for no-merge controls.",
        "",
        "| Scenario | Threshold | Recall | Group-F1 | False-rate | Active-rate | Onset | Release | ms/scan |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {} | {:.1f} | {:.2f} | {:.2f} | {:.3f} | {:.3f} | {:.2f} | {:.2f} | {:.3f} |".format(
                _scenario_label(str(row["scenario"])),
                float(row["threshold"]),
                float(row["group_detection_recall"]),
                float(row["group_detection_f1"]),
                float(row["false_group_scan_rate"]),
                float(row["active_group_scan_rate"]),
                float(row["merge_onset_delay"]),
                float(row["split_release_delay"]),
                float(row["runtime_ms_per_scan"]),
            )
        )
    return "\n".join(lines)


def format_latex_table(rows: list[dict[str, float | int | str]]) -> str:
    """Return a compact LaTeX table at the default threshold if present."""

    selected_threshold = 0.0 if any(float(row["threshold"]) == 0.0 for row in rows) else float(rows[0]["threshold"])
    selected = [row for row in rows if float(row["threshold"]) == selected_threshold]
    line_break = r"\\"
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Merge-detector threshold calibration at the operating threshold. True merge should have high recall; no-merge controls should have low false group rates.}",
        "\\label{tab:detector-threshold-calibration}",
        "\\begin{tabular}{@{}lrrrrrrr@{}}",
        "\\toprule",
        f"Scenario & $\\tau$ & Recall & Group-F1 & False-rate & Active-rate & Onset & Release {line_break}",
        "\\midrule",
    ]
    for row in selected:
        lines.append(
            "{} & {:.1f} & {:.2f} & {:.2f} & {:.3f} & {:.3f} & {:.2f} & {:.2f} {}".format(
                _scenario_label(str(row["scenario"])),
                float(row["threshold"]),
                float(row["group_detection_recall"]),
                float(row["group_detection_f1"]),
                float(row["false_group_scan_rate"]),
                float(row["active_group_scan_rate"]),
                float(row["merge_onset_delay"]),
                float(row["split_release_delay"]),
                line_break,
            )
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


def plot_threshold_sweep(rows: Sequence[dict[str, float | int | str]], output_path: Path) -> None:
    """Plot true-merge recall and no-merge false activation rates by threshold."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = sorted({float(row["threshold"]) for row in rows})
    fig, axis = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)

    true_rows = [row for row in rows if row["scenario"] == "true_merge"]
    if true_rows:
        axis.plot(
            thresholds,
            [_metric_at(true_rows, threshold, "group_detection_recall") for threshold in thresholds],
            marker="o",
            label="true merge recall",
        )

    for scenario in SCENARIOS:
        if scenario == "true_merge":
            continue
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        if not scenario_rows:
            continue
        axis.plot(
            thresholds,
            [_metric_at(scenario_rows, threshold, "false_group_scan_rate") for threshold in thresholds],
            marker="x",
            label=f"{_scenario_label(scenario)} false-rate",
        )

    axis.set_xlabel("merge-score threshold")
    axis.set_ylabel("rate")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _run_trial(scenario_name: str, threshold: float, seed: int) -> dict[str, float]:
    scenario = SCENARIO_FACTORIES[scenario_name](ScenarioConfig(seed=seed))
    start = perf_counter()
    result = run_detected_group_filter(
        scenario,
        partitioner=lambda scan: distance_partition(scan.measurements, threshold=1.25),
        config=FilterConfig(merge_score_threshold=threshold),
    )
    elapsed = perf_counter() - start

    true_members = [scan.unresolved_members for scan in scenario.scans]
    detection = group_detection_metrics(result.group_membership_trace, true_members)
    true_group_scans = sum(bool(members) for members in true_members)
    resolved_scans = len(scenario.scans) - true_group_scans
    estimated_group_scans = sum(bool(members) for members in result.group_membership_trace)

    return {
        "group_detection_precision": detection.precision,
        "group_detection_recall": detection.recall,
        "group_detection_f1": detection.f1,
        "false_group_scans": float(detection.false_group_scans),
        "missed_group_scans": float(detection.missed_group_scans),
        "wrong_membership_scans": float(detection.wrong_membership_scans),
        "false_group_scan_rate": _safe_divide(detection.false_group_scans, resolved_scans),
        "active_group_scan_rate": estimated_group_scans / len(scenario.scans),
        "merge_onset_delay": float(detection.merge_onset_delay or 0),
        "split_release_delay": float(detection.split_release_delay),
        "runtime_per_scan": elapsed / len(scenario.scans),
    }


def _aggregate_row(
    scenario_name: str,
    threshold: float,
    summaries: Sequence[dict[str, float]],
    num_trials: int,
) -> dict[str, float | int | str]:
    return {
        "scenario": scenario_name,
        "threshold": threshold,
        "num_trials": num_trials,
        "group_detection_precision": _avg(summaries, "group_detection_precision"),
        "group_detection_recall": _avg(summaries, "group_detection_recall"),
        "group_detection_f1": _avg(summaries, "group_detection_f1"),
        "false_group_scan_rate": _avg(summaries, "false_group_scan_rate"),
        "false_group_scans": _avg(summaries, "false_group_scans"),
        "missed_group_scans": _avg(summaries, "missed_group_scans"),
        "wrong_membership_scans": _avg(summaries, "wrong_membership_scans"),
        "active_group_scan_rate": _avg(summaries, "active_group_scan_rate"),
        "merge_onset_delay": _avg(summaries, "merge_onset_delay"),
        "split_release_delay": _avg(summaries, "split_release_delay"),
        "runtime_ms_per_scan": 1000.0 * _avg(summaries, "runtime_per_scan"),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _metric_at(
    rows: Sequence[dict[str, float | int | str]],
    threshold: float,
    key: str,
) -> float:
    matching = [float(row[key]) for row in rows if float(row["threshold"]) == threshold]
    return mean(matching) if matching else float("nan")


def _scenario_label(scenario_name: str) -> str:
    return SCENARIO_LABELS.get(scenario_name, scenario_name.replace("_", " "))


def _avg(summaries: Sequence[dict[str, float]], key: str) -> float:
    return mean(summary[key] for summary in summaries)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
