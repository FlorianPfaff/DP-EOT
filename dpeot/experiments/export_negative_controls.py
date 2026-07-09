"""Export merge-detector negative controls as paper-ready artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from dpeot.metrics.group_detection import group_detection_metrics
from dpeot.partitions.distance_partition import distance_partition
from dpeot.scenarios.two_target_merge_split import (
    Scenario,
    ScenarioConfig,
    generate_near_miss_no_merge,
    generate_parallel_close_tracks,
    generate_single_large_extended_target,
    generate_two_target_merge_split,
)
from dpeot.tracking.merge_split_filter import run_detected_group_filter


ScenarioFactory = Callable[[ScenarioConfig | None], Scenario]

SCENARIO_FACTORIES: dict[str, ScenarioFactory] = {
    "true_merge": generate_two_target_merge_split,
    "near_miss_no_merge": generate_near_miss_no_merge,
    "parallel_close_tracks": generate_parallel_close_tracks,
    "single_large_extended_target": generate_single_large_extended_target,
}

SCENARIO_LABELS = {
    "true_merge": "True merge",
    "near_miss_no_merge": "Near miss",
    "parallel_close_tracks": "Parallel close",
    "single_large_extended_target": "Single large target",
}

METRIC_COLUMNS = [
    ("scenario", "Scenario"),
    ("group_detection_recall", "Recall"),
    ("group_detection_f1", "Group-F1"),
    ("false_group_scan_rate", "False-rate"),
    ("false_group_scans", "False"),
    ("active_group_scan_rate", "Active-rate"),
    ("runtime_ms_per_scan", "ms/scan"),
]

LATEX_LINE_BREAK = r"\\"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=11000)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = run_negative_controls(num_trials=args.num_trials, base_seed=args.base_seed)
    write_negative_control_artifacts(
        rows=rows,
        output_dir=args.output_dir,
        num_trials=args.num_trials,
        base_seed=args.base_seed,
    )
    print(format_markdown_table(rows))


def run_negative_controls(
    num_trials: int = 100,
    base_seed: int = 11000,
    scenario_names: Sequence[str] = tuple(SCENARIO_FACTORIES),
) -> list[dict[str, float | int | str]]:
    """Run the proposed detector on true-merge and no-merge controls."""

    if num_trials <= 0:
        raise ValueError("num_trials must be positive")

    rows: list[dict[str, float | int | str]] = []
    for scenario_index, scenario_name in enumerate(scenario_names):
        if scenario_name not in SCENARIO_FACTORIES:
            raise ValueError(f"unknown negative-control scenario: {scenario_name}")

        summaries = [
            _run_trial(
                scenario_name,
                seed=base_seed + 1000 * scenario_index + trial,
            )
            for trial in range(num_trials)
        ]
        rows.append(_aggregate_row(scenario_name, summaries, num_trials))
    return rows


def write_negative_control_artifacts(
    rows: list[dict[str, float | int | str]],
    output_dir: Path,
    num_trials: int,
    base_seed: int,
) -> None:
    """Write JSON, Markdown, and LaTeX negative-control artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "benchmark": "merge_detector_negative_controls",
        "num_trials": num_trials,
        "base_seed": base_seed,
        "rows": rows,
    }
    (output_dir / "negative_controls.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "negative_controls.md").write_text(
        format_markdown_table(rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "negative_controls_table.tex").write_text(
        format_latex_table(rows) + "\n",
        encoding="utf-8",
    )


def format_markdown_table(rows: list[dict[str, float | int | str]]) -> str:
    """Return a Markdown table for detector negative controls."""

    header = "| " + " | ".join(label for _, label in METRIC_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] + ["---:"] * (len(METRIC_COLUMNS) - 1)) + " |"
    body = [
        "| " + " | ".join(_format_cell(row, key) for key, _ in METRIC_COLUMNS) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def format_latex_table(rows: list[dict[str, float | int | str]]) -> str:
    """Return a complete LaTeX table for detector negative controls."""

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Merge-detector negative controls. True merge should have high group recall; the no-merge controls should have low false group scan rates.}",
        "\\label{tab:negative-controls}",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        f"Scenario & Recall & Group-F1 & False-rate & False & Active-rate & ms/scan {LATEX_LINE_BREAK}",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {:.2f} & {:.2f} & {:.3f} & {:.2f} & {:.3f} & {:.3f} {}".format(
                _scenario_label(str(row["scenario"])),
                float(row["group_detection_recall"]),
                float(row["group_detection_f1"]),
                float(row["false_group_scan_rate"]),
                float(row["false_group_scans"]),
                float(row["active_group_scan_rate"]),
                float(row["runtime_ms_per_scan"]),
                LATEX_LINE_BREAK,
            )
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


def _run_trial(scenario_name: str, seed: int) -> dict[str, float]:
    scenario = SCENARIO_FACTORIES[scenario_name](ScenarioConfig(seed=seed))
    start = perf_counter()
    result = run_detected_group_filter(
        scenario,
        partitioner=lambda scan: distance_partition(scan.measurements, threshold=1.25),
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
        "true_group_scans": float(true_group_scans),
        "runtime_per_scan": elapsed / len(scenario.scans),
    }


def _aggregate_row(
    scenario_name: str,
    summaries: Sequence[dict[str, float]],
    num_trials: int,
) -> dict[str, float | int | str]:
    return {
        "scenario": scenario_name,
        "num_trials": num_trials,
        "group_detection_precision": _avg(summaries, "group_detection_precision"),
        "group_detection_recall": _avg(summaries, "group_detection_recall"),
        "group_detection_f1": _avg(summaries, "group_detection_f1"),
        "false_group_scan_rate": _avg(summaries, "false_group_scan_rate"),
        "false_group_scans": _avg(summaries, "false_group_scans"),
        "missed_group_scans": _avg(summaries, "missed_group_scans"),
        "wrong_membership_scans": _avg(summaries, "wrong_membership_scans"),
        "active_group_scan_rate": _avg(summaries, "active_group_scan_rate"),
        "true_group_scans": _avg(summaries, "true_group_scans"),
        "runtime_ms_per_scan": 1000.0 * _avg(summaries, "runtime_per_scan"),
    }


def _format_cell(row: dict[str, float | int | str], key: str) -> str:
    value = row[key]
    if key == "scenario":
        return _scenario_label(str(value))
    if key in {"false_group_scan_rate", "active_group_scan_rate"}:
        return f"{float(value):.3f}"
    return f"{float(value):.2f}"


def _scenario_label(scenario_name: str) -> str:
    return SCENARIO_LABELS.get(scenario_name, scenario_name.replace("_", " "))


def _avg(summaries: Sequence[dict[str, float]], key: str) -> float:
    return mean(summary[key] for summary in summaries)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
