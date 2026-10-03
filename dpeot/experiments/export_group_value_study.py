"""Run the versioned, paired unresolved-group falsification study."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np

from dpeot.metrics.group_value import paired_interval, summarize_trace
from dpeot.scenarios.group_value import CONDITIONS, PHASES, ROOT_SEED, STUDY_VERSION, generate_study_instance, random_stream
from dpeot.tracking.group_value import METHODS, StudyFilterConfig, run_study_filter


THRESHOLDS = (-10.0, -5.0, 0.0, 5.0, 10.0)
DEFAULT_TRIALS = {"pilot": 20, "calibrate": 100, "confirm": 200, "runtime": 20, "stress": 200}
DETECTOR_METHODS = tuple(m for m in METHODS if m not in {"prediction_only", "oracle_mode"})
METRICS = ("labeled_position_rmse", "recovery", "group_f1", "coverage_95", "position_nll", "false_group_scan_rate", "merge_onset_delay", "split_release_delay", "retained_mass_min", "runtime_ms_per_scan")


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    files = sorted(root.joinpath("dpeot").rglob("*.py")) + [root / "pyproject.toml"]
    protocol = root / "docs/group_value_protocol.md"
    if protocol.exists():
        files.append(protocol)
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _revision() -> str | None:
    if os.environ.get("DPEOT_SOURCE_REVISION"):
        return os.environ["DPEOT_SOURCE_REVISION"]
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _trial_job(job: tuple) -> list[dict]:
    phase, condition_index, trial, methods, thresholds, hypotheses = job
    condition = CONDITIONS[condition_index]
    instance = generate_study_instance(condition, phase, condition_index, trial)
    if phase == "runtime":
        methods = tuple(random_stream(phase, condition_index, trial, 3).permutation(methods))
    rows = []
    for method in methods:
        values = THRESHOLDS if phase == "calibrate" else (float(thresholds.get(method, 0.0)),)
        if method not in DETECTOR_METHODS:
            values = (0.0,)
        for threshold in values:
            start = perf_counter()
            trace = run_study_filter(
                instance.inputs, method, StudyFilterConfig(threshold, hypotheses),
                oracle_groupings=instance.truth.groupings if method == "oracle_mode" else None,
            )
            elapsed = perf_counter() - start
            rows.append({
                "phase": phase, "condition": condition.name, "condition_index": condition_index,
                "trial": trial, "method": method, "threshold": threshold,
                "max_hypotheses": hypotheses,
                **summarize_trace(instance, trace),
                "runtime_ms_per_scan": 1000 * elapsed / len(instance.inputs.scans),
            })
    return rows


def select_thresholds(rows: list[dict]) -> dict:
    negative = {condition.name for condition in CONDITIONS if condition.negative_control}
    result = {}
    for method in DETECTOR_METHODS:
        candidates = []
        for threshold in THRESHOLDS:
            selected = [r for r in rows if r["method"] == method and r["threshold"] == threshold]
            if not selected:
                continue
            controls = [r for r in selected if r["condition"] in negative]
            if not controls:
                raise ValueError("calibration requires both negative-control conditions")
            resolved = sum(r["resolved_scans"] for r in controls)
            false_rate = sum(r["false_group_scans"] for r in controls) / resolved if resolved else 1.0
            tp = sum(r["group_tp"] for r in selected)
            fp = sum(r["group_fp"] for r in selected)
            fn = sum(r["group_fn"] for r in selected)
            f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            candidates.append({"threshold": threshold, "group_f1": f1, "recall": recall,
                               "false_group_scan_rate": false_rate, "false_rate_feasible": false_rate <= 0.01})
        if not candidates:
            continue
        feasible = [r for r in candidates if r["false_rate_feasible"]]
        if feasible:
            chosen = max(feasible, key=lambda r: (r["group_f1"], -r["false_group_scan_rate"], -abs(r["threshold"]), r["threshold"]))
        else:
            chosen = min(candidates, key=lambda r: (r["false_group_scan_rate"], -r["group_f1"], abs(r["threshold"])))
        result[method] = {**chosen, "useful_operating_point": chosen["false_rate_feasible"] and chosen["recall"] >= 0.8,
                          "candidates": candidates}
    return result


def summarize_rows(rows: list[dict]) -> list[dict]:
    result = []
    for condition in sorted({r["condition"] for r in rows}):
        for method in METHODS:
            selected = [r for r in rows if r["condition"] == condition and r["method"] == method]
            if not selected:
                continue
            thresholds = {r["threshold"] for r in selected}
            for threshold in sorted(thresholds):
                matching = [r for r in selected if r["threshold"] == threshold]
                summary = {"condition": condition, "method": method, "threshold": threshold, "trials": len(matching)}
                for metric in METRICS:
                    values = [r[metric] for r in matching if r[metric] is not None]
                    summary[metric] = float(np.mean(values)) if values else None
                    summary[metric + "_available"] = len(values)
                summary["trials_with_merge"] = sum(r["merge_events"] > 0 for r in matching)
                summary["censored_recovery_events"] = sum(r["censored_recovery_events"] for r in matching)
                summary["proposal_failure_scans"] = sum(r["proposal_failure_scans"] for r in matching)
                result.append(summary)
    return result


def compare_methods(rows: list[dict], candidate: str, baseline: str) -> list[dict]:
    comparisons = []
    for condition in sorted({r["condition"] for r in rows}):
        a = {r["trial"]: r for r in rows if r["condition"] == condition and r["method"] == candidate}
        b = {r["trial"]: r for r in rows if r["condition"] == condition and r["method"] == baseline}
        paired = sorted(set(a) & set(b))
        if not paired:
            continue
        comparison = {"condition": condition, "candidate": candidate, "baseline": baseline}
        for metric, relative in (("labeled_position_rmse", True), ("recovery", False), ("position_nll", False), ("coverage_95", False)):
            available = [trial for trial in paired if a[trial][metric] is not None and b[trial][metric] is not None]
            comparison[metric] = paired_interval([a[i][metric] for i in available], [b[i][metric] for i in available], relative=relative) if available else None
        comparisons.append(comparison)
    return comparisons


def study_verdict(comparisons: list[dict], calibration: dict | None, phase: str) -> dict:
    difficult = {c.name for c in CONDITIONS if c.difficult}
    coast = [r for r in comparisons if r["baseline"] == "labeled_coast"]
    qualifying, regression, uncertain_noninferiority = [], [], []
    for row in coast:
        error, recovery = row["labeled_position_rmse"], row["recovery"]
        benefit = (error["estimate"] >= 0.10 and error["low"] > 0) or (
            recovery is not None and recovery["estimate"] >= 0.05 and recovery["low"] > 0
        )
        if row["condition"] in difficult and benefit:
            qualifying.append(row["condition"])
        if recovery is not None and recovery["estimate"] < -0.02:
            regression.append(row["condition"])
        if recovery is not None and recovery["low"] < -0.02:
            uncertain_noninferiority.append(row["condition"])
    matched = [r for r in comparisons if r["baseline"] == "labeled_joint"]
    equivalent = bool(matched) and all(abs(r["labeled_position_rmse"]["estimate"]) < 1e-10 and (
        r["recovery"] is None or abs(r["recovery"]["estimate"]) < 1e-10
    ) for r in matched)
    mechanism = len(qualifying) >= 2 and not regression and not uncertain_noninferiority
    complete = phase == "confirm" and {r["condition"] for r in coast} == {c.name for c in CONDITIONS} and all(
        r["labeled_position_rmse"]["pairs"] >= DEFAULT_TRIALS["confirm"] for r in coast
    )
    useful = calibration is not None and calibration.get("coupled_group", {}).get("useful_operating_point", False)
    return {
        "phase": phase, "confirmatory": complete,
        "mechanism_signal": mechanism,
        "mechanism_gate": mechanism and complete, "qualifying_conditions": qualifying,
        "recovery_regressions": regression, "noninferiority_unresolved": uncertain_noninferiority,
        "matched_labeled_equivalent": equivalent, "calibrated_detector_useful": useful,
        "contribution_gate": False,
        "decision": ("reframe" if mechanism else "stop_current_claim") if complete else "provisional_only",
        "reason": "A representation-specific contribution is not established. Aggregate observations use the same joint Gaussian update in a labeled representation; published-method novelty must be assessed before any expansion.",
        "conditional_stress_and_dp_authorized": False,
    }


def _number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_reports(output: Path, rows: list[dict], metadata: dict, calibration: dict | None) -> None:
    summaries = summarize_rows(rows)
    _write_json(output / "summary.json", summaries)
    comparisons = [] if metadata["stage"] == "calibrate" else [
        row for baseline in ("labeled_coast", "labeled_hypothesis", "labeled_joint", "oracle_mode")
        for row in compare_methods(rows, "coupled_group", baseline)
    ]
    _write_json(output / "paired_comparisons.json", comparisons)
    if metadata["stage"] != "calibrate":
        _write_json(output / "verdict.json", study_verdict(comparisons, calibration, metadata["stage"]))
    lines = [f"# {STUDY_VERSION}: {metadata['stage']}", "", "Means are across independent trials. n/a means no eligible events, not zero error.", "", "| Condition | Method | tau | RMSE | Rec-post | Group-F1 | Coverage | Merge trials |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        lines.append(f"| {row['condition']} | {row['method']} | {row['threshold']:g} | {_number(row['labeled_position_rmse'])} | {_number(row['recovery'])} | {_number(row['group_f1'])} | {_number(row['coverage_95'])} | {row['trials_with_merge']}/{row['trials']} |")
    lines += ["", "Timing from parallel runs is diagnostic only. Use the dedicated single-worker runtime stage.", "Retained mass is conditional on the gated candidate bank; it is not total posterior mass."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if metadata["stage"] != "calibrate":
        _write_figures(output, summaries, comparisons)
    elif calibration is not None:
        _write_calibration_figure(output, calibration)
    if metadata["stage"] == "runtime":
        _write_json(output / "runtime_distribution.json", [
            {"method": method, "trials": len(values), "mean_ms": float(np.mean(values)),
             "p50_ms": float(np.quantile(values, .5)), "p90_ms": float(np.quantile(values, .9)),
             "max_ms": float(np.max(values))}
            for method in METHODS
            if (values := [r["runtime_ms_per_scan"] for r in rows if r["method"] == method])
        ])
    _write_table(output, summaries)


def _write_table(output: Path, summaries: list[dict]) -> None:
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Corrected finite-resolution study. Entries average condition means; recovery excludes trials without eligible post-split scans.}", r"\label{tab:group-value}", r"\begin{tabular}{lrrrr}", r"\toprule", r"Method & RMSE & Rec. & F1 & Cov. \\", r"\midrule"]
    labels = {"prediction_only": "Prediction", "labeled_coast": "Labeled coast", "labeled_hypothesis": "Labeled hypotheses", "coupled_group": "Coupled group", "labeled_joint": "Labeled joint", "oracle_mode": "Oracle mode"}
    for method in METHODS:
        selected = [r for r in summaries if r["method"] == method]
        if selected:
            values = []
            for metric in ("labeled_position_rmse", "recovery", "group_f1", "coverage_95"):
                available = [r[metric] for r in selected if r[metric] is not None]
                values.append(_number(float(np.mean(available)) if available else None))
            lines.append(labels[method] + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output / "method_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figures(output: Path, summaries: list[dict], comparisons: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paired = [r for r in comparisons if r["baseline"] == "labeled_coast"]
    if paired:
        fig, ax = plt.subplots(figsize=(8, 5))
        y = np.arange(len(paired))
        values = [100 * r["labeled_position_rmse"]["estimate"] for r in paired]
        low = [100 * (r["labeled_position_rmse"]["estimate"] - r["labeled_position_rmse"]["low"]) for r in paired]
        high = [100 * (r["labeled_position_rmse"]["high"] - r["labeled_position_rmse"]["estimate"]) for r in paired]
        ax.errorbar(values, y, xerr=np.maximum([low, high], 0), fmt="o", color="#00796b", capsize=3)
        ax.axvline(0, color="#555555", linewidth=1)
        ax.axvline(10, color="#b45533", linestyle="--", linewidth=1)
        ax.set_yticks(y, [r["condition"].replace("_", " ") for r in paired])
        ax.set_xlabel("RMSE reduction relative to labeled coast (%)")
        ax.set_title("Paired trial bootstrap: 95% intervals")
        fig.tight_layout()
        fig.savefig(output / "paired_rmse.png", dpi=180)
        plt.close(fig)
    _write_detection_figure(output, summaries)


def _write_calibration_figure(output: Path, calibration: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, result in calibration.items():
        candidates = result["candidates"]
        ax.plot([r["false_group_scan_rate"] for r in candidates], [r["recall"] for r in candidates], marker="o", label=method.replace("_", " "))
        for row in candidates:
            ax.annotate(f"{row['threshold']:g}", (row["false_group_scan_rate"], row["recall"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axvline(.01, color="#555555", linestyle="--")
    ax.set_xlabel("False-group rate on resolved negative-control scans")
    ax.set_ylabel("Group recall")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "calibration_tradeoff.png", dpi=180)
    plt.close(fig)


def _write_detection_figure(output: Path, summaries: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [r for r in summaries if r["method"] in {"labeled_hypothesis", "coupled_group", "labeled_joint"}]
    if selected:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for method, color in (("labeled_hypothesis", "#b45533"), ("coupled_group", "#00796b"), ("labeled_joint", "#514c9b")):
            data = [r for r in selected if r["method"] == method]
            for ax, metric in zip(axes, ("group_f1", "coverage_95")):
                ax.plot(range(len(data)), [r[metric] if r[metric] is not None else np.nan for r in data], marker="o", markersize=3, label=method.replace("_", " "), color=color)
                ax.set_xticks(range(len(data)), [r["condition"].replace("_", " ") for r in data], rotation=75, ha="right", fontsize=7)
                ax.set_ylim(0, 1.03)
        axes[0].set_ylabel("Membership F1")
        axes[1].set_ylabel("Moment-matched 95% ellipse coverage")
        axes[1].axhline(0.95, color="#555555", linestyle="--", linewidth=1)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "detection_coverage.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(PHASES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--num-trials", type=int)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--hypotheses", type=int, default=8)
    parser.add_argument("--conditions", nargs="+", choices=[c.name for c in CONDITIONS])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    trials = args.num_trials if args.num_trials is not None else DEFAULT_TRIALS[args.stage]
    if trials < 1 or not 1 <= args.workers <= 32 or args.hypotheses < 1:
        parser.error("trials/hypotheses must be positive and workers must be between 1 and 32")
    if args.stage == "stress":
        parser.error("stress expansion is gated on a distinct contribution; the current representation-equivalent methods do not authorize it")
    if args.stage == "runtime" and args.workers != 1:
        parser.error("runtime measurements require exactly one worker")
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(name) != "1":
            parser.error(f"set {name}=1 before starting Python")
    fingerprint = source_fingerprint()
    calibration, thresholds = None, {}
    if args.calibration:
        frozen = json.loads(args.calibration.read_text())
        if frozen["source_fingerprint"] != fingerprint:
            parser.error("source changed since calibration; rerun calibration before confirmation")
        if not frozen.get("complete_protocol", False):
            parser.error("calibration is incomplete; all conditions and detector methods need at least 100 trials")
        calibration = frozen["methods"]
        thresholds = {method: value["threshold"] for method, value in calibration.items()}
    if args.stage in {"confirm", "runtime"} and calibration is None:
        parser.error("confirmation/runtime requires a frozen --calibration file")
    conditions = [i for i, c in enumerate(CONDITIONS) if args.conditions is None or c.name in args.conditions]
    if args.stage == "calibrate" and any(c.name not in [CONDITIONS[i].name for i in conditions] for c in CONDITIONS if c.negative_control):
        parser.error("calibration must contain both negative controls")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "trials").mkdir(exist_ok=True)
    specification = {"study_version": STUDY_VERSION, "stage": args.stage, "trials": trials,
                     "conditions": [asdict(CONDITIONS[i]) for i in conditions], "methods": args.methods,
                     "source_fingerprint": fingerprint, "thresholds": thresholds,
                     "max_hypotheses": args.hypotheses, "root_seed": ROOT_SEED, "phase_seed_id": PHASES[args.stage]}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text())
        if metadata["specification"] != specification:
            parser.error("output directory belongs to a different frozen study configuration")
    else:
        metadata = {**specification, "specification": specification, "git_revision": _revision(),
                    "python": sys.version, "numpy": np.__version__, "host": platform.node(),
                    "platform": platform.platform(), "workers": args.workers,
                    "blas_threads": {name: os.environ[name] for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")}}
        _write_json(manifest_path, metadata)
    jobs = []
    for condition_index in conditions:
        for trial in range(trials):
            path = output / "trials" / f"{condition_index:02d}-{trial:04d}.json"
            if not path.exists():
                jobs.append((args.stage, condition_index, trial, tuple(args.methods), thresholds, args.hypotheses))
    if args.summarize_only and jobs:
        parser.error(f"study incomplete: {len(jobs)} trial jobs missing")
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_trial_job, job): job for job in jobs}
            for number, future in enumerate(as_completed(futures), 1):
                job = futures[future]
                _write_json(output / "trials" / f"{job[1]:02d}-{job[2]:04d}.json", future.result())
                if number % 10 == 0 or number == len(jobs):
                    print(f"{args.stage}: {number}/{len(jobs)} new trial jobs completed", flush=True)
    rows = [row for path in sorted((output / "trials").glob("*.json")) for row in json.loads(path.read_text())]
    _write_json(output / "trials.json", rows)
    if args.stage == "calibrate":
        calibration = select_thresholds(rows)
        complete_protocol = trials >= DEFAULT_TRIALS["calibrate"] and len(conditions) == len(CONDITIONS) and set(DETECTOR_METHODS).issubset(args.methods)
        _write_json(output / "calibration.json", {"study_version": STUDY_VERSION, "source_fingerprint": fingerprint, "methods": calibration,
                                                 "num_trials": trials, "conditions": [CONDITIONS[i].name for i in conditions], "complete_protocol": complete_protocol})
    write_reports(output, rows, metadata, calibration)
    print(f"Artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
