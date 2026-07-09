"""Summarize the full synthetic stress sweep for paper-facing results."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from dpeot.experiments.export_stress_sweep import CLUTTER_RATES, MERGE_DURATIONS


FAILURE_RECOVERY_THRESHOLD = 0.95
METHOD_LABELS = {
    "distance_collapse": "Distance collapse",
    "dp_x_order": "DP x-order",
    "mfm_x_order": "MFM x-order",
    "labeled_split_hypothesis": "Labeled split hypothesis",
    "proposed_group_labels": "Proposed group labels",
}
SUMMARY_FIELDS = (
    "method",
    "num_settings",
    "mean_rec_post",
    "mean_group_f1",
    "worst_rec_post",
    "failure_count",
    "runtime_mean_ms_per_scan",
    "runtime_p50_ms_per_scan",
    "runtime_p90_ms_per_scan",
    "runtime_max_ms_per_scan",
)
WORST_CASE_FIELDS = (
    "method",
    "merge_duration",
    "clutter_rate",
    "noise_level",
    "extent_similarity",
    "rate_asymmetry",
    "crossing_angle",
    "label_recovery_post_split",
    "group_detection_f1",
    "position_error",
    "runtime_ms_per_scan",
)
LATEX_LINE_BREAK = r"\\"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/stress_sweep_full/stress_sweep.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/stress_sweep_full"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--failure-threshold", type=float, default=FAILURE_RECOVERY_THRESHOLD)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    rows = load_stress_rows(args.input)
    write_summary_artifacts(
        rows=rows,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        failure_threshold=args.failure_threshold,
        write_figures=not args.no_figures,
    )
    print(
        format_method_summary_markdown(
            method_summary(rows, args.failure_threshold),
            failure_threshold=args.failure_threshold,
        )
    )


def load_stress_rows(path: Path) -> list[dict[str, float | int | str]]:
    """Load stress rows from the JSON artifact written by export_stress_sweep."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    return [dict(row) for row in rows]


def write_summary_artifacts(
    rows: list[dict[str, float | int | str]],
    output_dir: Path,
    figure_dir: Path,
    failure_threshold: float = FAILURE_RECOVERY_THRESHOLD,
    write_figures: bool = True,
) -> None:
    """Write compact full-grid summaries and optional heatmap figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = method_summary(rows, failure_threshold)
    worst_rows = proposed_worst_cases(rows)
    payload: dict[str, Any] = {
        "benchmark": "stress_sweep_full_summary",
        "failure_threshold": failure_threshold,
        "method_summary": summary_rows,
        "proposed_worst_cases": worst_rows,
    }
    (output_dir / "stress_full_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "stress_full_method_summary.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(
        output_dir / "stress_full_proposed_worst_cases.csv",
        worst_rows,
        WORST_CASE_FIELDS,
    )
    (output_dir / "stress_full_method_summary.md").write_text(
        format_method_summary_markdown(summary_rows, failure_threshold) + "\n",
        encoding="utf-8",
    )
    (output_dir / "stress_full_method_summary_table.tex").write_text(
        format_method_summary_latex(summary_rows, failure_threshold) + "\n",
        encoding="utf-8",
    )
    (output_dir / "stress_full_proposed_worst_cases.md").write_text(
        format_worst_cases_markdown(worst_rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "stress_full_heatmaps.md").write_text(
        format_heatmap_markdown(rows) + "\n",
        encoding="utf-8",
    )
    if write_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_metric_heatmaps(
            rows,
            metric="label_recovery_post_split",
            output_path=figure_dir / "stress_full_rec_post_heatmaps.png",
            colorbar_label="post-split recovery",
        )
        plot_metric_heatmaps(
            rows,
            metric="group_detection_f1",
            output_path=figure_dir / "stress_full_group_f1_heatmaps.png",
            colorbar_label="group-detection F1",
        )


def method_summary(
    rows: Sequence[dict[str, float | int | str]],
    failure_threshold: float = FAILURE_RECOVERY_THRESHOLD,
) -> list[dict[str, float | int | str]]:
    """Return method-level summary statistics over the full factorial grid."""

    summary: list[dict[str, float | int | str]] = []
    for method in _method_order(rows):
        method_rows = [row for row in rows if row["method"] == method]
        rec_values = [float(row["label_recovery_post_split"]) for row in method_rows]
        group_values = [float(row["group_detection_f1"]) for row in method_rows]
        runtime_values = [float(row["runtime_ms_per_scan"]) for row in method_rows]
        summary.append(
            {
                "method": method,
                "num_settings": len(method_rows),
                "mean_rec_post": mean(rec_values),
                "mean_group_f1": mean(group_values),
                "worst_rec_post": min(rec_values),
                "failure_count": sum(value < failure_threshold for value in rec_values),
                "runtime_mean_ms_per_scan": mean(runtime_values),
                "runtime_p50_ms_per_scan": _percentile(runtime_values, 50),
                "runtime_p90_ms_per_scan": _percentile(runtime_values, 90),
                "runtime_max_ms_per_scan": max(runtime_values),
            }
        )
    return summary


def proposed_worst_cases(
    rows: Sequence[dict[str, float | int | str]],
    limit: int = 12,
) -> list[dict[str, float | int | str]]:
    """Return the worst proposed-method settings by post-split recovery."""

    proposed = [row for row in rows if row["method"] == "proposed_group_labels"]
    ranked = sorted(
        proposed,
        key=lambda row: (
            float(row["label_recovery_post_split"]),
            float(row["group_detection_f1"]),
            -float(row["clutter_rate"]),
            -int(row["merge_duration"]),
        ),
    )
    return [
        {field: row[field] for field in WORST_CASE_FIELDS}
        for row in ranked[:limit]
    ]


def format_method_summary_markdown(
    rows: Sequence[dict[str, float | int | str]],
    failure_threshold: float = FAILURE_RECOVERY_THRESHOLD,
) -> str:
    """Return the method-level full-grid summary as Markdown."""

    lines = [
        "# Full Stress Sweep Summary",
        "",
        f"Failure count uses Rec-post < {failure_threshold:.2f}.",
        "",
        "| Method | Mean Rec-post | Mean Group-F1 | Worst Rec-post | Failures | Runtime mean | Runtime p50 | Runtime p90 | Runtime max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {}/{} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                _method_label(str(row["method"])),
                float(row["mean_rec_post"]),
                float(row["mean_group_f1"]),
                float(row["worst_rec_post"]),
                int(row["failure_count"]),
                int(row["num_settings"]),
                float(row["runtime_mean_ms_per_scan"]),
                float(row["runtime_p50_ms_per_scan"]),
                float(row["runtime_p90_ms_per_scan"]),
                float(row["runtime_max_ms_per_scan"]),
            )
        )
    return "\n".join(lines)


def format_method_summary_latex(
    rows: Sequence[dict[str, float | int | str]],
    failure_threshold: float = FAILURE_RECOVERY_THRESHOLD,
) -> str:
    """Return a LaTeX table for the method-level full-grid summary."""

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Full factorial stress-grid summary. Failure count is the number of settings with post-split label recovery below "
        f"{failure_threshold:.2f}. Runtime columns report ms/scan across stress settings.}}",
        "\\label{tab:full-stress-summary}",
        "\\begin{tabular}{@{}lrrrrrrrr@{}}",
        "\\toprule",
        "Method & Mean Rec. & Mean Group-F1 & Worst Rec. & Failures & "
        f"Runtime mean & p50 & p90 & max {LATEX_LINE_BREAK}",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {:.3f} & {:.3f} & {:.3f} & {}/{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} {}".format(
                _method_label(str(row["method"])),
                float(row["mean_rec_post"]),
                float(row["mean_group_f1"]),
                float(row["worst_rec_post"]),
                int(row["failure_count"]),
                int(row["num_settings"]),
                float(row["runtime_mean_ms_per_scan"]),
                float(row["runtime_p50_ms_per_scan"]),
                float(row["runtime_p90_ms_per_scan"]),
                float(row["runtime_max_ms_per_scan"]),
                LATEX_LINE_BREAK,
            )
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


def format_worst_cases_markdown(rows: Sequence[dict[str, float | int | str]]) -> str:
    """Return a compact Markdown table of proposed-method worst cases."""

    lines = [
        "# Proposed Group Labels: Worst Full-Grid Settings",
        "",
        "| Merge | Clutter | Noise | Extent | Rate | Angle | Rec-post | Group-F1 | Pos. | ms/scan |",
        "| ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {} | {:g} | {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                int(row["merge_duration"]),
                float(row["clutter_rate"]),
                row["noise_level"],
                row["extent_similarity"],
                row["rate_asymmetry"],
                row["crossing_angle"],
                float(row["label_recovery_post_split"]),
                float(row["group_detection_f1"]),
                float(row["position_error"]),
                float(row["runtime_ms_per_scan"]),
            )
        )
    return "\n".join(lines)


def format_heatmap_markdown(rows: Sequence[dict[str, float | int | str]]) -> str:
    """Return aggregate Rec-post and Group-F1 heatmap tables."""

    lines = ["# Full Stress Sweep Heatmaps", ""]
    for metric, title in (
        ("label_recovery_post_split", "Post-Split Label Recovery"),
        ("group_detection_f1", "Group-Detection F1"),
    ):
        lines.extend([f"## {title}", ""])
        for method in _method_order(rows):
            method_rows = [row for row in rows if row["method"] == method]
            values = _heatmap_values(method_rows, metric)
            lines.extend(["", f"### {_method_label(method)}", ""])
            lines.append("| clutter \\ duration | " + " | ".join(str(d) for d in MERGE_DURATIONS) + " |")
            lines.append("| --- | " + " | ".join("---:" for _ in MERGE_DURATIONS) + " |")
            for clutter_rate, row_values in zip(CLUTTER_RATES, values):
                formatted = " | ".join(f"{value:.2f}" for value in row_values)
                lines.append(f"| {clutter_rate:g} | {formatted} |")
        lines.append("")
    return "\n".join(lines).rstrip()


def plot_metric_heatmaps(
    rows: Sequence[dict[str, float | int | str]],
    metric: str,
    output_path: Path,
    colorbar_label: str,
) -> None:
    """Plot one aggregate heatmap per method over merge duration and clutter."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = _method_order(rows)
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(3.1 * len(methods), 3.4),
        constrained_layout=True,
    )
    if len(methods) == 1:
        axes = [axes]

    image = None
    for axis, method in zip(axes, methods):
        method_rows = [row for row in rows if row["method"] == method]
        values = np.asarray(_heatmap_values(method_rows, metric))
        image = axis.imshow(values, vmin=0.0, vmax=1.0, origin="lower", cmap="viridis")
        axis.set_title(_method_label(method).replace(" ", "\n"))
        axis.set_xticks(
            range(len(MERGE_DURATIONS)),
            labels=[str(duration) for duration in MERGE_DURATIONS],
        )
        axis.set_yticks(
            range(len(CLUTTER_RATES)),
            labels=[f"{rate:g}" for rate in CLUTTER_RATES],
        )
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
        fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label=colorbar_label)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _heatmap_values(
    rows: Sequence[dict[str, float | int | str]],
    metric: str,
) -> list[list[float]]:
    values: list[list[float]] = []
    for clutter_rate in CLUTTER_RATES:
        row_values: list[float] = []
        for duration in MERGE_DURATIONS:
            matching = [
                float(row[metric])
                for row in rows
                if int(row["merge_duration"]) == duration
                and float(row["clutter_rate"]) == clutter_rate
            ]
            row_values.append(mean(matching) if matching else float("nan"))
        values.append(row_values)
    return values


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, float | int | str]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _method_order(rows: Sequence[dict[str, float | int | str]]) -> list[str]:
    present = {str(row["method"]) for row in rows}
    preferred = [method for method in METHOD_LABELS if method in present]
    remaining = sorted(present.difference(preferred))
    return preferred + remaining


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


if __name__ == "__main__":
    main()
