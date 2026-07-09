"""Export detector-threshold calibration sweeps.

The sweep varies the resolved-vs-group log-likelihood threshold used by the
non-oracle unresolved-group detector. It evaluates both a true merge scenario and
negative controls where the detector should avoid creating a group.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any

import numpy as np

from dpeot.metrics.group_detection import group_detection_metrics
from dpeot.metrics.identity import (
    count_post_split_identity_switches,
    label_recovery_accuracy,
    split_recovery_delay,
)
from dpeot.partitions.distance_partition import distance_partition
from dpeot.scenarios.two_target_merge_split import (
    Scenario,
    ScenarioConfig,
    generate_near_miss_no_merge,
    generate_parallel_close_tracks,
    generate_single_large_extended_target,
    generate_two_target_merge_split,
)
from dpeot.tracking.merge_split_filter import (
    FilterConfig,
    mean_unlabeled_position_error,
    run_detected_group_filter,
)


THRESHOLDS = (-30.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
SCENARIO_FACTORIES: dict[str, Callable[[ScenarioConfig], Scenario]] = {
    "true_merge": generate_two_target_merge_split,
    "near_miss_no_merge": generate_near_miss_no_merge,
    "parallel_close_tracks": generate_parallel_close_tracks,
    "single_large_extended_target": generate_single_large_extended_target,
}
CSV_FIELDS = (
    "scenario",
    "threshold",
    "num_trials",
    "group_detection_precision",
    "group_detection_recall",
    "group_detection_f1",
    "false_group_scans",
    "false_group_rate",
    "missed_group_scans",
    "missed_group_rate",
    "wrong_membership_scans",
    "merge_onset_delay",
    "split_release_delay",
    "label_recovery_post_split",
    "id_switches_post_split",
    "position_error",
    "runtime_ms_per_scan",
)


@dataclass(frozen=True)
class ThresholdSweepConfig:
    """Configuration for the threshold sweep exporter."""

    num_trials: int = 100
    base_seed: int = 9000
    thresholds: tuple[float, ...] = THRESHOLDS
    scenarios: tuple[str, ...] = tuple(SCENARIO_FACTORIES)

    def validate(self) -> None:
        if self.num_trials <= 0:
            raise ValueError("num_trials must be positive")
        unknown = set(self.scenarios).difference(SCENARIO_FACTORIES)
        if unknown:
            raise ValueError(f"unknown threshold-sweep scenarios: {sorted(unknown)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=9000)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=list(THRESHOLDS),
        help="Merge detector thresholds to evaluate.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=list(SCENARIO_FACTORIES),
        choices=list(SCENARIO_FACTORIES),
        help="Scenario names to include in the sweep.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/threshold_sweep"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    config = ThresholdSweepConfig(
        num_trials=args.num_trials,
        base_seed=args.base_seed,
        thresholds=tuple(args.thresholds),
        scenarios=tuple(args.scenarios),
    )
    rows = run_threshold_sweep(config)
    write_threshold_artifacts(
        rows=rows,
        config=config,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        write_figures=not args.no_figures,
    )
    print(format_threshold_markdown(rows))


def run_threshold_sweep(config: ThresholdSweepConfig) -> list[dict[str, float | int | str]]:
    """Run detector-threshold sweeps and return aggregate rows."""

    config.validate()
    rows: list[dict[str, float | int | str]] = []
    for scenario_name in config.scenarios:
        for threshold in config.thresholds:
            summaries = [
                _run_one_trial(
                    scenario_name=scenario_name,
                    threshold=threshold,
                    seed=config.base_seed
                    + 100000 * _scenario_index(scenario_name)
                    + 1000 * _threshold_index(config.thresholds, threshold)
                    + trial,
                )
                for trial in range(config.num_trials)
            ]
            rows.append(_aggregate_row(scenario_name, threshold, summaries, config.num_trials))
    return rows


def write_threshold_artifacts(
    rows: Sequence[dict[str, float | int | str]],
    config: ThresholdSweepConfig,
    output_dir: Path,
    figure_dir: Path,
    write_figures: bool = True,
) -> None:
    """Write JSON, CSV, Markdown, and optional detector-calibration figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "benchmark": "detector_threshold_sweep",
        "num_trials": config.num_trials,
        "base_seed": config.base_seed,
        "thresholds": list(config.thresholds),
        "scenarios": list(config.scenarios),
        "rows": list(rows),
    }
    (output_dir / "threshold_sweep.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "threshold_sweep.csv", rows)
    (output_dir / "threshold_sweep.md").write_text(
        format_threshold_markdown(rows) + "\n",
        encoding="utf-8",
    )
    if write_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_threshold_sweep(rows, figure_dir / "threshold_sweep_precision_recall.png")


def format_threshold_markdown(rows: Sequence[dict[str, float | int | str]]) -> str:
    """Return a compact Markdown report for the threshold sweep."""

    scenarios = sorted({str(row["scenario"]) for row in rows})
    lines = [
        "# Detector Threshold Sweep",
        "",
        "Each row is averaged over trials. False and missed rates are normalized by scan count.",
    ]
    for scenario in scenarios:
        scenario_rows = sorted(
            [row for row in rows if row["scenario"] == scenario],
            key=lambda row: float(row["threshold"]),
        )
        lines.extend(
            [
                "",
                f"## {scenario}",
                "",
                "| tau | precision | recall | F1 | false rate | missed rate | onset | release | rec-post | pos |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in scenario_rows:
            lines.append(
                "| {threshold:.1f} | {precision:.2f} | {recall:.2f} | {f1:.2f} | "
                "{false_rate:.3f} | {missed_rate:.3f} | {onset:.2f} | {release:.2f} | "
                "{recovery:.2f} | {pos:.3f} |".format(
                    threshold=float(row["threshold"]),
                    precision=float(row["group_detection_precision"]),
                    recall=float(row["group_detection_recall"]),
                    f1=float(row["group_detection_f1"]),
                    false_rate=float(row["false_group_rate"]),
                    missed_rate=float(row["missed_group_rate"]),
                    onset=float(row["merge_onset_delay"]),
                    release=float(row["split_release_delay"]),
                    recovery=float(row["label_recovery_post_split"]),
                    pos=float(row["position_error"]),
                )
            )
    return "\n".join(lines)


def plot_threshold_sweep(rows: Sequence[dict[str, float | int | str]], output_path: Path) -> None:
    """Plot precision/recall/F1 and false-group rate across thresholds."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = sorted({str(row["scenario"]) for row in rows})
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(6.4, 2.8 * len(scenarios)), sharex=True)
    if len(scenarios) == 1:
        axes = [axes]

    for axis, scenario in zip(axes, scenarios):
        scenario_rows = sorted(
            [row for row in rows if row["scenario"] == scenario],
            key=lambda row: float(row["threshold"]),
        )
        thresholds = [float(row["threshold"]) for row in scenario_rows]
        precision = [float(row["group_detection_precision"]) for row in scenario_rows]
        recall = [float(row["group_detection_recall"]) for row in scenario_rows]
        f1 = [float(row["group_detection_f1"]) for row in scenario_rows]
        false_rate = [float(row["false_group_rate"]) for row in scenario_rows]

        axis.plot(thresholds, precision, marker="o", label="precision")
        axis.plot(thresholds, recall, marker="o", label="recall")
        axis.plot(thresholds, f1, marker="o", label="F1")
        axis.plot(thresholds, false_rate, marker="o", label="false rate")
        axis.set_title(scenario)
        axis.set_ylim(-0.05, 1.05)
        axis.set_ylabel("metric")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", ncol=4, fontsize=8)

    axes[-1].set_xlabel("merge detector threshold tau")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _run_one_trial(scenario_name: str, threshold: float, seed: int) -> dict[str, float]:
    scenario = _build_scenario(scenario_name, seed)
    config = FilterConfig(merge_score_threshold=threshold)
    start = perf_counter()
    result = run_detected_group_filter(
        scenario,
        partitioner=lambda scan: distance_partition(scan.measurements, threshold=1.25),
        config=config,
    )
    elapsed = perf_counter() - start

    labels = scenario.labels
    split_index = _split_index_for_metrics(scenario)
    post_split_assignments = result.assignments[split_index:]
    split_delay = _safe_split_delay(result.assignments, labels, split_index)
    true_members = [scan.unresolved_members for scan in scenario.scans]
    detection = group_detection_metrics(result.group_membership_trace, true_members)

    return {
        "group_detection_precision": detection.precision,
        "group_detection_recall": detection.recall,
        "group_detection_f1": detection.f1,
        "false_group_scans": float(detection.false_group_scans),
        "false_group_rate": detection.false_group_scans / len(scenario.scans),
        "missed_group_scans": float(detection.missed_group_scans),
        "missed_group_rate": detection.missed_group_scans / len(scenario.scans),
        "wrong_membership_scans": float(detection.wrong_membership_scans),
        "merge_onset_delay": float(
            detection.merge_onset_delay if detection.merge_onset_delay is not None else 0
        ),
        "split_release_delay": float(detection.split_release_delay),
        "label_recovery_post_split": label_recovery_accuracy(post_split_assignments, labels),
        "id_switches_post_split": float(
            count_post_split_identity_switches(result.assignments, split_index)
        ),
        "split_recovery_delay": float(split_delay),
        "position_error": mean_unlabeled_position_error(scenario, result),
        "runtime_per_scan": elapsed / len(scenario.scans),
    }


def _aggregate_row(
    scenario: str,
    threshold: float,
    summaries: Sequence[dict[str, float]],
    num_trials: int,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "scenario": scenario,
        "threshold": threshold,
        "num_trials": num_trials,
    }
    for key in (
        "group_detection_precision",
        "group_detection_recall",
        "group_detection_f1",
        "false_group_scans",
        "false_group_rate",
        "missed_group_scans",
        "missed_group_rate",
        "wrong_membership_scans",
        "merge_onset_delay",
        "split_release_delay",
        "label_recovery_post_split",
        "id_switches_post_split",
        "split_recovery_delay",
        "position_error",
        "runtime_per_scan",
    ):
        row[key] = mean(summary[key] for summary in summaries)
    row["runtime_ms_per_scan"] = 1000.0 * float(row["runtime_per_scan"])
    row["group_detection_f1_std"] = _std(summaries, "group_detection_f1")
    return row


def _build_scenario(scenario_name: str, seed: int) -> Scenario:
    factory = SCENARIO_FACTORIES[scenario_name]
    return factory(ScenarioConfig(seed=seed))


def _split_index_for_metrics(scenario: Scenario) -> int:
    true_group_indices = [index for index, scan in enumerate(scenario.scans) if scan.is_unresolved]
    if true_group_indices:
        return min(true_group_indices[-1] + 1, len(scenario.scans) - 1)
    return 0


def _safe_split_delay(assignments, labels: Sequence[str], split_index: int) -> int:
    if not labels:
        return 0
    delay = split_recovery_delay(assignments, labels, split_index)
    if delay is None:
        return len(assignments) - split_index
    return delay


def _write_csv(path: Path, rows: Sequence[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _std(summaries: Sequence[dict[str, float]], key: str) -> float:
    values = [summary[key] for summary in summaries]
    return pstdev(values) if len(values) > 1 else 0.0


def _scenario_index(scenario_name: str) -> int:
    return list(SCENARIO_FACTORIES).index(scenario_name)


def _threshold_index(thresholds: Sequence[float], threshold: float) -> int:
    return list(thresholds).index(threshold)


if __name__ == "__main__":
    main()
