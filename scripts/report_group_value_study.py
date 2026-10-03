"""Audit frozen study artifacts and prepare the paper's research decision.

This postprocessing script is outside the frozen simulation package. It does not
change calibration, select additional settings, or feed results back to trackers.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from dpeot.metrics.group_value import paired_interval, summarize_trace
from dpeot.scenarios.group_value import CONDITIONS, generate_study_instance
from dpeot.tracking.group_value import METHODS, StudyFilterConfig, run_study_filter


PHASE_SIZES = {"pilot": 20, "calibrate": 100, "confirm": 200, "beam16": 20, "runtime": 20}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def audit_phase(directory: Path, name: str) -> tuple[dict, list[dict]]:
    manifest = read_json(directory / "manifest.json")
    rows = read_json(directory / "trials.json")
    assert manifest["trials"] == PHASE_SIZES[name], (name, "trial count")
    assert len(manifest["conditions"]) == len(CONDITIONS), (name, "condition count")
    assert set(manifest["methods"]) == set(METHODS), (name, "methods")
    keys = [(r["condition_index"], r["trial"], r["method"], r["threshold"]) for r in rows]
    expected_per_trial = 22 if name == "calibrate" else 6
    assert len(set(keys)) == len(keys) == len(CONDITIONS) * PHASE_SIZES[name] * expected_per_trial, (name, "duplicate or missing rows")
    assert {r["trial"] for r in rows} == set(range(PHASE_SIZES[name]))
    for index, condition in enumerate(CONDITIONS):
        selected = [r for r in rows if r["condition_index"] == index]
        assert {r["condition"] for r in selected} == {condition.name}
        assert len(selected) == PHASE_SIZES[name] * expected_per_trial
        for method in METHODS:
            values = [r for r in selected if r["method"] == method]
            multiplier = 5 if name == "calibrate" and method not in {"prediction_only", "oracle_mode"} else 1
            assert len(values) == PHASE_SIZES[name] * multiplier
    assert all(np.isfinite(r["labeled_position_rmse"]) and np.isfinite(r["position_nll"]) for r in rows)
    assert all(0 <= r["coverage_95"] <= 1 for r in rows)
    if name == "runtime":
        assert manifest["workers"] == 1
    assert set(manifest["blas_threads"].values()) == {"1"}
    return manifest, rows


def matched_differences(rows: list[dict]) -> dict:
    left = {(r["condition_index"], r["trial"]): r for r in rows if r["method"] == "coupled_group"}
    right = {(r["condition_index"], r["trial"]): r for r in rows if r["method"] == "labeled_joint"}
    assert left.keys() == right.keys()
    metrics = ("labeled_position_rmse", "recovery", "group_f1", "coverage_95", "position_nll",
               "post_split_identity_switches", "false_group_scans", "retained_mass_min")
    differences = {}
    for metric in metrics:
        eligible = [key for key in left if left[key][metric] is not None and right[key][metric] is not None]
        assert all((left[key][metric] is None) == (right[key][metric] is None) for key in left)
        differences[metric] = max((abs(left[k][metric] - right[k][metric]) for k in eligible), default=0)
    return {"paired_trials": len(left), "maximum_absolute_difference": differences,
            "equivalent_to_tolerance": all(value <= 1e-10 for value in differences.values())}


def beam_comparison(confirm: list[dict], beam: list[dict]) -> list[dict]:
    lookup = {(r["condition"], r["trial"], r["method"]): r for r in confirm}
    result = []
    for condition in CONDITIONS:
        selected = [r for r in beam if r["condition"] == condition.name and r["method"] == "coupled_group"]
        reference = [lookup[(r["condition"], r["trial"], r["method"])] for r in selected]
        recovery = [(a["recovery"], b["recovery"]) for a, b in zip(selected, reference)
                    if a["recovery"] is not None and b["recovery"] is not None]
        result.append({
            "condition": condition.name,
            "rmse_reduction_beam16_vs8": paired_interval([r["labeled_position_rmse"] for r in selected],
                                                       [r["labeled_position_rmse"] for r in reference], relative=True),
            "recovery_change": paired_interval([a for a, _ in recovery], [b for _, b in recovery]) if recovery else None,
            "minimum_retained_mass_beam8": min(r["retained_mass_min"] for r in reference),
            "minimum_retained_mass_beam16": min(r["retained_mass_min"] for r in selected),
        })
    return result


def pooled_detection(rows: list[dict], method: str) -> dict:
    selected = [r for r in rows if r["method"] == method]
    negative_names = {c.name for c in CONDITIONS if c.negative_control}
    controls = [r for r in selected if r["condition"] in negative_names]
    tp, fp, fn = (sum(r[key] for r in selected) for key in ("group_tp", "group_fp", "group_fn"))
    resolved = sum(r["resolved_scans"] for r in controls)
    false = sum(r["false_group_scans"] for r in controls)
    return {"recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
            "negative_control_false_rate": false / resolved if resolved else None, "negative_control_false_scans": false,
            "negative_control_resolved_scans": resolved}


def diagnostic_cases(rows: list[dict], destination: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proposed = [r for r in rows if r["method"] == "coupled_group"]
    eligible = [r for r in proposed if r["recovery"] is not None]
    worst_rmse = max(proposed, key=lambda r: r["labeled_position_rmse"])
    worst_recovery = min(eligible, key=lambda r: (r["recovery"], -r["labeled_position_rmse"]))
    unique = {(r["condition_index"], r["trial"]): r for r in (worst_rmse, worst_recovery)}
    result = []
    for (index, trial), row in unique.items():
        instance = generate_study_instance(CONDITIONS[index], "confirm", index, trial)
        trace = run_study_filter(instance.inputs, "coupled_group", StudyFilterConfig(row["threshold"], 8))
        regenerated = summarize_trace(instance, trace)
        for metric, value in regenerated.items():
            if value is None:
                assert row[metric] is None
            else:
                np.testing.assert_allclose(value, row[metric], atol=1e-10, rtol=1e-10)
        labeled = run_study_filter(instance.inputs, "labeled_joint", StudyFilterConfig(row["threshold"], 8))
        np.testing.assert_allclose(trace.means, labeled.means, atol=1e-10)
        np.testing.assert_allclose(trace.covariances, labeled.covariances, atol=1e-10)
        assert trace.groupings == labeled.groupings
        for a, b in zip(trace.hypothesis_weights, labeled.hypothesis_weights):
            np.testing.assert_allclose(a, b, atol=1e-10)
        name = f"{row['condition']}-{trial:04d}"
        n = len(instance.inputs.labels)
        estimates = trace.means.reshape(-1, n, 4)
        np.savez_compressed(destination / f"{name}.npz", truth=instance.truth.states, estimates=trace.means,
                            covariance=trace.covariances, detector_score=trace.detector_scores,
                            retained_mass=trace.retained_mass, initial_mean=instance.inputs.initial_mean,
                            initial_covariance=instance.inputs.initial_covariance,
                            observations=np.vstack([s.measurements for s in instance.inputs.scans]),
                            observation_offsets=np.cumsum([0] + [len(s.measurements) for s in instance.inputs.scans]))
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for i, color in enumerate(("#00796b", "#b45533", "#514c9b")[:n]):
            axes[0].plot(instance.truth.states[:, i, 0], instance.truth.states[:, i, 1], color=color, label=f"{instance.inputs.labels[i]} truth")
            axes[0].plot(estimates[:, i, 0], estimates[:, i, 1], "--", color=color, label=f"{instance.inputs.labels[i]} estimate")
            errors = np.linalg.norm(estimates[:, i, :2] - instance.truth.states[:, i, :2], axis=1)
            axes[1].plot(errors, color=color, label=instance.inputs.labels[i])
        active = [any(len(g) > 1 for g in grouping) for grouping in instance.truth.groupings]
        for k, grouped in enumerate(active):
            if grouped:
                axes[1].axvspan(k - .5, k + .5, color="#777777", alpha=.12, linewidth=0)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].legend(fontsize=6)
        axes[1].set_xlabel("Scan (gray: true unresolved)")
        axes[1].set_ylabel("Labeled position error")
        fig.suptitle(name.replace("_", " "))
        fig.tight_layout()
        fig.savefig(destination / f"{name}.png", dpi=180)
        plt.close(fig)
        result.append({"condition": row["condition"], "trial": trial, "threshold": row["threshold"],
                       "recovery": row["recovery"], "rmse": row["labeled_position_rmse"],
                       "trace_regeneration_verified": True, "groupings": trace.groupings,
                       "true_groupings": instance.truth.groupings, "files": [f"{name}.npz", f"{name}.png"]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifests, phases = {}, {}
    for name in PHASE_SIZES:
        manifests[name], phases[name] = audit_phase(args.input_root / name, name)
    assert len({m["source_fingerprint"] for m in manifests.values()}) == 1
    assert len({manifests[name]["phase_seed_id"] for name in ("pilot", "calibrate", "confirm", "runtime")}) == 4
    assert manifests["beam16"]["phase_seed_id"] == manifests["confirm"]["phase_seed_id"]
    assert manifests["beam16"]["max_hypotheses"] == 16
    calibration = read_json(args.input_root / "calibrate/calibration.json")
    assert calibration["complete_protocol"]
    frozen_thresholds = {m: value["threshold"] for m, value in calibration["methods"].items()}
    for name in ("confirm", "beam16", "runtime"):
        assert manifests[name]["thresholds"] == frozen_thresholds
        for row in phases[name]:
            assert row["threshold"] == frozen_thresholds.get(row["method"], 0.)
    comparisons = read_json(args.input_root / "confirm/paired_comparisons.json")
    verdict = read_json(args.input_root / "confirm/verdict.json")
    assert verdict["confirmatory"]
    equivalence = matched_differences(phases["confirm"])
    if not equivalence["equivalent_to_tolerance"]:
        raise RuntimeError("matched-representation control failed; inspect differences before writing the paper")
    sensitivity = beam_comparison(phases["confirm"], phases["beam16"])
    detection = {m: pooled_detection(phases["confirm"], m) for m in METHODS}
    summary = read_json(args.input_root / "confirm/summary.json")
    cases_dir = output / "failure_cases"
    cases_dir.mkdir(exist_ok=True)
    cases = diagnostic_cases(phases["confirm"], cases_dir)
    audit = {"complete": True, "source_fingerprint": manifests["confirm"]["source_fingerprint"],
             "simulation_revision": manifests["confirm"]["git_revision"],
             "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "rows_by_phase": {name: len(rows) for name, rows in phases.items()},
             "matched_equivalence": equivalence, "beam_sensitivity": sensitivity, "pooled_detection": detection,
             "failure_cases": cases, "verdict": verdict,
             "proposal_failure_scans": {name: sum(r["proposal_failure_scans"] for r in rows) for name, rows in phases.items()}}
    write_json(output / "audit.json", audit)
    for name in PHASE_SIZES:
        target = output / name
        target.mkdir(exist_ok=True)
        for path in (args.input_root / name).iterdir():
            if path.is_file() and path.name != "trials.json":
                shutil.copy2(path, target / path.name)
        raw = (args.input_root / name / "trials.json").read_bytes()
        (target / "trials.json.gz").write_bytes(gzip.compress(raw, mtime=0))
        write_json(target / "trial_archive.json", {"uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
                                                "rows": len(phases[name]), "archive": "trials.json.gz"})
    proposed_summary = {r["condition"]: r for r in summary if r["method"] == "coupled_group"}
    coast_comparisons = {r["condition"]: r for r in comparisons if r["baseline"] == "labeled_coast"}
    lines = ["# Research Decision", "",
             "The corrected experiment does not establish a new unresolved-group tracking method. "
             "Aggregate observations can help relative to coasting, but a labeled joint-state filter uses the same information and gives equivalent results.",
             "", f"Protocol decision: **{verdict['decision']}**. Mechanism gate: **{verdict['mechanism_gate']}**. "
             "The contribution gate is not passed; broad stress and DP experiments are therefore not run.", "",
             "## Evidence", "",
             f"- {len(phases['confirm'])} method runs on 2,400 independent held-out scenarios (200 per condition).",
             f"- Matched group/labeled equality verified across {equivalence['paired_trials']} paired trials; maximum differences: "
             + json.dumps(equivalence["maximum_absolute_difference"]) + ".",
             f"- Group recall {detection['coupled_group']['recall']:.3f}, pooled F1 {detection['coupled_group']['f1']:.3f}, "
             f"negative-control false scan rate {detection['coupled_group']['negative_control_false_rate']:.4f}.",
             f"- Qualifying difficult conditions: {', '.join(verdict['qualifying_conditions']) or 'none'}.",
             f"- Recovery noninferiority unresolved: {', '.join(verdict['noninferiority_unresolved']) or 'none'}.",
             "", "| Condition | Group RMSE | RMSE reduction vs coast (95% CI) | Rec-post | Coverage | Eligible recovery trials |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for condition in CONDITIONS:
        row = proposed_summary[condition.name]
        effect = coast_comparisons[condition.name]["labeled_position_rmse"]
        recovery = "n/a" if row["recovery"] is None else f"{row['recovery']:.3f}"
        lines.append(f"| {condition.name} | {row['labeled_position_rmse']:.3f} | {100*effect['estimate']:.1f}% [{100*effect['low']:.1f}, {100*effect['high']:.1f}] | {recovery} | {row['coverage_95']:.3f} | {row['recovery_available']}/200 |")
    lines += ["", "## Interpretation and Limits", "",
              "The matched comparator shares the numerical core and assembles the same observation operator in labeled coordinates. "
              "Equality is an algebraic/control result, not an independently reproduced state-of-the-art baseline. "
              "The 2014 Beard/Vo/Vo merged-measurement GLMB paper already demonstrates labeled Bayesian treatment of merging; "
              "our joint Gaussian update does not establish a new mechanism relative to that literature.",
              "Known extents/rates and established tracks remain assumptions. The detector uses a truncated candidate bank and a best-mode decision; "
              "covariance-matched ellipses are a diagnostic, not guaranteed posterior credible regions. Oracle mode is not a performance upper bound.",
              "", "## Beam and Runtime Checks", ""]
    for row in sensitivity:
        effect = row["rmse_reduction_beam16_vs8"]
        lines.append(f"- {row['condition']}: beam-16 RMSE change as reduction {100*effect['estimate']:.2f}% [{100*effect['low']:.2f}, {100*effect['high']:.2f}], paired on the first 20 confirmation trials.")
    runtime = read_json(args.input_root / "runtime/runtime_distribution.json")
    lines += ["", "| Method | Mean ms/scan | Median | P90 | Max trial mean |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in runtime:
        lines.append(f"| {row['method']} | {row['mean_ms']:.3f} | {row['p50_ms']:.3f} | {row['p90_ms']:.3f} | {row['max_ms']:.3f} |")
    lines += ["", "Runtime is measured separately with one worker/BLAS thread; p90/max refer to trial-average times, not individual scan latency.",
              "", "## Provenance", "", f"Simulation revision: {manifests['confirm']['git_revision']}.",
              f"Source fingerprint: {manifests['confirm']['source_fingerprint']}.",
              "All phase manifests, compressed raw trial records, bootstrap contrasts, calibration decisions, beam checks, and regenerated worst-case traces are included. "
              "Legacy artifacts are excluded from the analysis.",
              "", "Recommendation: stop the current novel-method/DP superiority claim. Preserve the corrected study as an audit/research note. "
              "Reopen a methods-paper effort only after specifying a mechanism that is not equivalent to matched labeled joint inference."]
    (output / "research_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    group = detection["coupled_group"]
    worst = min(proposed_summary.values(), key=lambda r: r["coverage_95"])
    qualified = ", ".join(name.replace("_", " ") for name in verdict["qualifying_conditions"]) or "none"
    noninferior = ", ".join(name.replace("_", " ") for name in verdict["noninferiority_unresolved"]) or "none"
    finding = (
        f"The held-out study comprises 2,400 independent scenarios and 14,400 method runs. "
        f"The coupled group and matched labeled-joint implementations agree across all 2,400 paired trials "
        f"to the $10^{{-10}}$ audit tolerance on tracking, membership, and uncertainty metrics. "
        f"This supports the representation-equivalence control, not an advantage for group labels.\n\n"
        f"The coupled method's pooled membership recall is {group['recall']:.3f} and pooled F1 is {group['f1']:.3f}. "
        f"On resolved negative-control scans its false-group rate is {100*group['negative_control_false_rate']:.2f}\\%. "
        f"These pooled detector metrics differ from the condition-averaged F1 in Table~\\ref{{tab:group-value}}.\n\n"
        f"The predeclared difficult conditions meeting the benefit screen are {qualified}. "
        f"Recovery noninferiority remains unresolved in: {noninferior}. "
        f"The full mechanism gate {'passes' if verdict['mechanism_gate'] else 'does not pass'}. "
        f"The lowest condition-mean 95\\% ellipse coverage is {worst['coverage_95']:.3f} "
        f"({worst['condition'].replace('_', ' ')}), exposing a limitation that localization averages alone conceal.\n\n"
        "The doubled-beam sensitivity results and separately measured runtime distributions are included in the "
        "reproducibility artifacts, together with regenerated worst-error and worst-recovery traces. "
        "Regardless of the coast comparison, the representation-specific contribution gate does not pass: "
        "the same information is available to matched labeled joint inference. We therefore stop the current "
        "new-method claim and do not run the conditional broad stress or DP expansion.\n"
    )
    (output / "findings.tex").write_text(finding, encoding="utf-8")
    print(json.dumps({"audit_complete": True, "decision": verdict["decision"], "equivalent": equivalence["equivalent_to_tolerance"]}, indent=2))


if __name__ == "__main__":
    main()
