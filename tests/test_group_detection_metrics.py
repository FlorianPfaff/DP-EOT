from math import isclose

from dpeot.metrics.group_detection import group_detection_metrics


def test_group_detection_metrics_count_scan_errors() -> None:
    truth = [
        frozenset(),
        frozenset({"A", "B"}),
        frozenset({"A", "B"}),
        frozenset({"A", "B"}),
        frozenset(),
        frozenset(),
    ]
    estimated = [
        frozenset({"A", "B"}),
        frozenset(),
        frozenset({"A"}),
        frozenset({"A", "B"}),
        frozenset({"A", "B"}),
        frozenset(),
    ]

    metrics = group_detection_metrics(estimated, truth)

    assert metrics.precision == 1 / 4
    assert metrics.recall == 1 / 3
    assert isclose(metrics.f1, 2 / 7)
    assert metrics.merge_onset_delay == 2
    assert metrics.split_release_delay == 1
    assert metrics.false_group_scans == 2
    assert metrics.missed_group_scans == 1
    assert metrics.wrong_membership_scans == 1


def test_group_detection_metrics_report_perfect_detection() -> None:
    truth = [frozenset(), frozenset({"A", "B"}), frozenset()]

    metrics = group_detection_metrics(truth, truth)

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.merge_onset_delay == 0
    assert metrics.split_release_delay == 0
    assert metrics.false_group_scans == 0
    assert metrics.missed_group_scans == 0
    assert metrics.wrong_membership_scans == 0
