"""Scan-level unresolved-group detection metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


Label = str


@dataclass(frozen=True)
class GroupDetectionMetrics:
    """Operational diagnostics for unresolved-group detection."""

    precision: float
    recall: float
    f1: float
    merge_onset_delay: int | None
    split_release_delay: int
    false_group_scans: int
    missed_group_scans: int
    wrong_membership_scans: int


def group_detection_metrics(
    estimated_members: Sequence[frozenset[Label]],
    true_members: Sequence[frozenset[Label]],
) -> GroupDetectionMetrics:
    """Compare estimated unresolved-group member sets against truth by scan.

    Precision, recall, and F1 award credit only for exact member-set recovery.
    The scan-count diagnostics then separate false group activations, missed
    group activations, and wrong member sets.
    """

    if len(estimated_members) != len(true_members):
        raise ValueError("estimated_members and true_members must have the same length")

    correct_group_scans = 0
    estimated_group_scans = 0
    true_group_scans = 0
    false_group_scans = 0
    missed_group_scans = 0
    wrong_membership_scans = 0

    for estimated, truth in zip(estimated_members, true_members):
        estimated_active = bool(estimated)
        truth_active = bool(truth)
        estimated_group_scans += int(estimated_active)
        true_group_scans += int(truth_active)

        if estimated_active and truth_active and estimated == truth:
            correct_group_scans += 1
        elif estimated_active and not truth_active:
            false_group_scans += 1
        elif not estimated_active and truth_active:
            missed_group_scans += 1
        elif estimated_active and truth_active and estimated != truth:
            wrong_membership_scans += 1

    precision = _safe_divide(correct_group_scans, estimated_group_scans)
    recall = _safe_divide(correct_group_scans, true_group_scans)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    return GroupDetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        merge_onset_delay=_merge_onset_delay(estimated_members, true_members),
        split_release_delay=_split_release_delay(estimated_members, true_members),
        false_group_scans=false_group_scans,
        missed_group_scans=missed_group_scans,
        wrong_membership_scans=wrong_membership_scans,
    )


def _merge_onset_delay(
    estimated_members: Sequence[frozenset[Label]],
    true_members: Sequence[frozenset[Label]],
) -> int | None:
    truth_indices = [index for index, members in enumerate(true_members) if members]
    if not truth_indices:
        return None

    truth_start = truth_indices[0]
    truth_end = truth_indices[-1]
    for index in range(truth_start, truth_end + 1):
        if estimated_members[index] == true_members[index] and true_members[index]:
            return index - truth_start
    return truth_end - truth_start + 1


def _split_release_delay(
    estimated_members: Sequence[frozenset[Label]],
    true_members: Sequence[frozenset[Label]],
) -> int:
    truth_indices = [index for index, members in enumerate(true_members) if members]
    if not truth_indices:
        return sum(bool(members) for members in estimated_members)

    truth_end = truth_indices[-1]
    delay = 0
    for members in estimated_members[truth_end + 1 :]:
        if not members:
            break
        delay += 1
    return delay


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
