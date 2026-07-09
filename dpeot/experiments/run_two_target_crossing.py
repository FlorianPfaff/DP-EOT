"""Run the initial two-target merge/split benchmark.

The experiment is intentionally small: it compares clustering-only baselines with
an identity-aware unresolved-group model on controlled two-target crossings.
The goal is to obtain the first falsifiable identity-preservation result before
building a full PMBM/GLMB-style tracker.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from statistics import mean, pstdev
from time import perf_counter

from dpeot.metrics.group_detection import group_detection_metrics
from dpeot.metrics.identity import (
    count_identity_switches,
    count_post_split_identity_switches,
    count_resolved_period_identity_switches,
    group_membership_accuracy,
    label_recovery_accuracy,
    split_recovery_delay,
)
from dpeot.partitions.distance_partition import distance_partition
from dpeot.partitions.dp_partition import DPPartitionConfig, dp_partition
from dpeot.partitions.mfm_partition import MFMPartitionConfig, mfm_partition
from dpeot.scenarios.two_target_merge_split import (
    Scenario,
    ScenarioConfig,
    generate_two_target_merge_split,
)
from dpeot.tracking.merge_split_filter import (
    FilterRunResult,
    mean_unlabeled_position_error,
    run_detected_group_filter,
    run_distance_collapse_baseline,
    run_identity_aware_group_filter,
    run_labeled_split_hypothesis_baseline,
    run_oracle_identity_baseline,
    run_x_order_clustering_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=7)
    args = parser.parse_args()

    rows = run_benchmark(num_trials=args.num_trials, base_seed=args.base_seed)
    _print_rows(rows)


def run_benchmark(num_trials: int = 100, base_seed: int = 7) -> list[dict[str, float | str]]:
    if num_trials <= 0:
        raise ValueError("num_trials must be positive")

    method_summaries: dict[str, list[dict[str, float]]] = {}

    for trial in range(num_trials):
        config = ScenarioConfig(seed=base_seed + trial)
        scenario = generate_two_target_merge_split(config)

        methods = _method_factories(scenario)
        for name, method in methods.items():
            start = perf_counter()
            result = method()
            elapsed = perf_counter() - start
            method_summaries.setdefault(name, []).append(
                _summarize_result(scenario, result, elapsed)
            )

    rows: list[dict[str, float | str]] = []
    for name, summaries in method_summaries.items():
        rows.append(
            {
                "method": name,
                "id_switches": _avg(summaries, "id_switches_total"),
                "id_switches_total": _avg(summaries, "id_switches_total"),
                "id_switches_pre_merge": _avg(summaries, "id_switches_pre_merge"),
                "id_switches_during_unresolved": _avg(
                    summaries, "id_switches_during_unresolved"
                ),
                "id_switches_post_split": _avg(summaries, "id_switches_post_split"),
                "id_switches_resolved": _avg(summaries, "id_switches_resolved"),
                "label_recovery": _avg(summaries, "label_recovery_post_split"),
                "label_recovery_post_split": _avg(summaries, "label_recovery_post_split"),
                "split_delay": _avg(summaries, "split_recovery_delay"),
                "split_recovery_delay": _avg(summaries, "split_recovery_delay"),
                "group_membership": _avg(summaries, "group_membership_during_unresolved"),
                "group_membership_during_unresolved": _avg(
                    summaries, "group_membership_during_unresolved"
                ),
                "group_detection_precision": _avg(summaries, "group_detection_precision"),
                "group_detection_recall": _avg(summaries, "group_detection_recall"),
                "group_detection_f1": _avg(summaries, "group_detection_f1"),
                "merge_onset_delay": _avg(summaries, "merge_onset_delay"),
                "split_release_delay": _avg(summaries, "split_release_delay"),
                "false_group_scans": _avg(summaries, "false_group_scans"),
                "missed_group_scans": _avg(summaries, "missed_group_scans"),
                "wrong_membership_scans": _avg(summaries, "wrong_membership_scans"),
                "position_error": _avg(summaries, "position_error"),
                "runtime_ms_per_scan": 1000.0 * _avg(summaries, "runtime_per_scan"),
                "id_switches_total_std": _std(summaries, "id_switches_total"),
                "id_switches_post_split_std": _std(summaries, "id_switches_post_split"),
            }
        )
    return rows


def _method_factories(scenario: Scenario) -> dict[str, Callable[[], FilterRunResult]]:
    return {
        "distance_collapse": lambda: run_distance_collapse_baseline(
            scenario,
            partitioner=_distance_partitioner,
        ),
        "dp_x_order": lambda: run_x_order_clustering_baseline(
            scenario,
            partitioner=_dp_partitioner_factory(scenario),
        ),
        "mfm_x_order": lambda: run_x_order_clustering_baseline(
            scenario,
            partitioner=_mfm_partitioner_factory(scenario),
        ),
        "labeled_split_hypothesis": lambda: run_labeled_split_hypothesis_baseline(
            scenario,
            partitioner=_distance_partitioner,
        ),
        "proposed_group_labels": lambda: run_detected_group_filter(
            scenario,
            partitioner=_distance_partitioner,
        ),
        "oracle_group_labels": lambda: run_identity_aware_group_filter(
            scenario,
            partitioner=_distance_partitioner,
        ),
        "oracle_identity": lambda: run_oracle_identity_baseline(scenario),
    }


def _distance_partitioner(scan) -> list[list[int]]:
    return distance_partition(scan.measurements, threshold=1.25)


def _dp_partitioner_factory(scenario: Scenario) -> Callable:
    def partitioner(scan) -> list[list[int]]:
        return dp_partition(
            scan.measurements,
            DPPartitionConfig(
                alpha=1.0,
                covariance_scale=0.7,
                num_sweeps=3,
                seed=7919 * scenario.config.seed + scan.k,
            ),
        )

    return partitioner


def _mfm_partitioner_factory(scenario: Scenario) -> Callable:
    def partitioner(scan) -> list[list[int]]:
        return mfm_partition(
            scan.measurements,
            MFMPartitionConfig(
                min_components=1,
                max_components=4,
                num_initializations=4,
                num_iterations=8,
                penalty_per_component=2.0,
                seed=104729 * scenario.config.seed + scan.k,
            ),
        )

    return partitioner


def _summarize_result(
    scenario: Scenario,
    result: FilterRunResult,
    elapsed: float,
) -> dict[str, float]:
    labels = scenario.labels
    merge_start = scenario.config.merge_start
    split_index = scenario.config.merge_end + 1
    post_split_assignments = result.assignments[split_index:]
    split_delay = split_recovery_delay(result.assignments, labels, split_index)
    unresolved_horizon = len(result.assignments) - split_index + 1
    unresolved_mask = [scan.is_unresolved for scan in scenario.scans]
    unresolved_indices = [i for i, scan in enumerate(scenario.scans) if scan.is_unresolved]
    true_members = [scan.unresolved_members for scan in scenario.scans]
    detection = group_detection_metrics(result.group_membership_trace, true_members)

    return {
        "id_switches_total": float(count_identity_switches(result.assignments)),
        "id_switches_pre_merge": float(
            count_identity_switches(result.assignments, end=merge_start)
        ),
        "id_switches_during_unresolved": float(
            count_identity_switches(
                result.assignments,
                start=merge_start,
                end=split_index,
            )
        ),
        "id_switches_post_split": float(
            count_post_split_identity_switches(result.assignments, split_index)
        ),
        "id_switches_resolved": float(
            count_resolved_period_identity_switches(result.assignments, unresolved_mask)
        ),
        "label_recovery_post_split": label_recovery_accuracy(post_split_assignments, labels),
        "split_recovery_delay": float(
            split_delay if split_delay is not None else unresolved_horizon
        ),
        "group_membership_during_unresolved": group_membership_accuracy(
            [result.group_membership_trace[i] for i in unresolved_indices],
            [scenario.scans[i].unresolved_members for i in unresolved_indices],
        ),
        "group_detection_precision": detection.precision,
        "group_detection_recall": detection.recall,
        "group_detection_f1": detection.f1,
        "merge_onset_delay": float(
            detection.merge_onset_delay if detection.merge_onset_delay is not None else 0
        ),
        "split_release_delay": float(detection.split_release_delay),
        "false_group_scans": float(detection.false_group_scans),
        "missed_group_scans": float(detection.missed_group_scans),
        "wrong_membership_scans": float(detection.wrong_membership_scans),
        "position_error": mean_unlabeled_position_error(scenario, result),
        "runtime_per_scan": elapsed / len(scenario.scans),
    }


def _avg(summaries: list[dict[str, float]], key: str) -> float:
    return mean(summary[key] for summary in summaries)


def _std(summaries: list[dict[str, float]], key: str) -> float:
    values = [summary[key] for summary in summaries]
    return pstdev(values) if len(values) > 1 else 0.0


def _print_rows(rows: list[dict[str, float | str]]) -> None:
    print("DP-EOT two-target merge/split benchmark")
    print()
    print(
        f"{'method':24s} "
        f"{'IDtot':>7s} "
        f"{'IDpost':>7s} "
        f"{'recov':>7s} "
        f"{'delay':>7s} "
        f"{'gF1':>7s} "
        f"{'onset':>7s} "
        f"{'release':>7s} "
        f"{'false':>7s} "
        f"{'poserr':>8s} "
        f"{'ms/scan':>8s}"
    )
    print("-" * 104)
    for row in rows:
        print(
            f"{str(row['method']):24s} "
            f"{float(row['id_switches_total']):7.2f} "
            f"{float(row['id_switches_post_split']):7.2f} "
            f"{float(row['label_recovery']):7.2f} "
            f"{float(row['split_delay']):7.2f} "
            f"{float(row['group_detection_f1']):7.2f} "
            f"{float(row['merge_onset_delay']):7.2f} "
            f"{float(row['split_release_delay']):7.2f} "
            f"{float(row['false_group_scans']):7.2f} "
            f"{float(row['position_error']):8.3f} "
            f"{float(row['runtime_ms_per_scan']):8.3f}"
        )


if __name__ == "__main__":
    main()
