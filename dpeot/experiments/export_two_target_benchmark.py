"""Export the two-target merge/split benchmark as paper-ready artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dpeot.experiments.run_two_target_crossing import run_benchmark


METRIC_COLUMNS = [
    ("method", "Method"),
    ("id_switches", "IDsw"),
    ("label_recovery", "Rec."),
    ("split_delay", "Delay"),
    ("group_membership", "Group"),
    ("position_error", "Pos."),
    ("runtime_ms_per_scan", "ms/scan"),
]

METHOD_LABELS = {
    "distance_collapse": "Distance collapse",
    "dp_x_order": "DP x-order",
    "proposed_group_labels": "Proposed group labels",
    "oracle_identity": "Oracle identity",
}

LATEX_LINE_BREAK = r"\\"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = run_benchmark(num_trials=args.num_trials, base_seed=args.base_seed)
    write_benchmark_artifacts(
        rows=rows,
        output_dir=args.output_dir,
        num_trials=args.num_trials,
        base_seed=args.base_seed,
    )
    print(format_markdown_table(rows))


def write_benchmark_artifacts(
    rows: list[dict[str, float | str]],
    output_dir: Path,
    num_trials: int,
    base_seed: int,
) -> None:
    """Write JSON, Markdown, and LaTeX benchmark artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "benchmark": "two_target_merge_split",
        "num_trials": num_trials,
        "base_seed": base_seed,
        "rows": rows,
    }

    (output_dir / "two_target_benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "two_target_benchmark.md").write_text(
        format_markdown_table(rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "two_target_benchmark_table.tex").write_text(
        format_latex_table(rows) + "\n",
        encoding="utf-8",
    )


def format_markdown_table(rows: list[dict[str, float | str]]) -> str:
    """Return a Markdown table for the benchmark rows."""

    header = "| " + " | ".join(label for _, label in METRIC_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] + ["---:"] * (len(METRIC_COLUMNS) - 1)) + " |"
    body = [
        "| " + " | ".join(_format_cell(row, key) for key, _ in METRIC_COLUMNS) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def format_latex_table(rows: list[dict[str, float | str]]) -> str:
    """Return a complete LaTeX table for the benchmark rows."""

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Initial two-target merge/split benchmark. Lower is better for identity switches, split delay, position error, and runtime. Higher is better for label recovery and group-membership accuracy.}",
        "\\label{tab:initial-benchmark}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        f"Method & IDsw & Rec. & Delay & Group & Pos. & ms {LATEX_LINE_BREAK}",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.3f} & {:.3f} {}".format(
                _method_label(str(row["method"])),
                float(row["id_switches"]),
                float(row["label_recovery"]),
                float(row["split_delay"]),
                float(row["group_membership"]),
                float(row["position_error"]),
                float(row["runtime_ms_per_scan"]),
                LATEX_LINE_BREAK,
            )
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def _format_cell(row: dict[str, float | str], key: str) -> str:
    value = row[key]
    if key == "method":
        return _method_label(str(value))
    if key == "position_error":
        return f"{float(value):.3f}"
    return f"{float(value):.2f}"


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


if __name__ == "__main__":
    main()
