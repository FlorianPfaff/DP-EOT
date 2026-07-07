"""Identity-preservation metrics for merge/split benchmarks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


Label = str


def count_identity_switches(assignments: Iterable[Mapping[Label, Label]]) -> int:
    """Count changes in estimated identity for each true label over time.

    Parameters
    ----------
    assignments:
        Sequence of mappings ``true_label -> estimated_label``. Missing labels
        are ignored for that scan. A switch is counted when a true label was
        assigned to one estimated label in the previous visible scan and to a
        different estimated label in the current scan.
    """

    previous: dict[Label, Label] = {}
    switches = 0
    for assignment in assignments:
        for true_label, estimated_label in assignment.items():
            if true_label in previous and previous[true_label] != estimated_label:
                switches += 1
            previous[true_label] = estimated_label
    return switches


def label_recovery_accuracy(
    assignments: Iterable[Mapping[Label, Label]],
    true_labels: Sequence[Label],
) -> float:
    """Return the fraction of labels that are recovered without permutation."""

    total = 0
    correct = 0
    for assignment in assignments:
        for label in true_labels:
            if label not in assignment:
                continue
            total += 1
            correct += int(assignment[label] == label)
    return correct / total if total else 0.0


def split_recovery_delay(
    assignments: Sequence[Mapping[Label, Label]],
    true_labels: Sequence[Label],
    split_index: int,
) -> int | None:
    """Return scans after split until all labels are correctly recovered.

    Returns ``None`` if the labels are never simultaneously recovered after the
    split index.
    """

    if split_index < 0 or split_index >= len(assignments):
        raise ValueError("split_index must point to a valid scan")

    for delay, assignment in enumerate(assignments[split_index:]):
        if all(assignment.get(label) == label for label in true_labels):
            return delay
    return None


def group_membership_accuracy(
    estimated_members: Iterable[frozenset[Label]],
    true_members: Iterable[frozenset[Label]],
) -> float:
    """Average Jaccard accuracy for unresolved-group member sets."""

    scores: list[float] = []
    for estimated, truth in zip(estimated_members, true_members):
        union = estimated | truth
        if not union:
            scores.append(1.0)
        else:
            scores.append(len(estimated & truth) / len(union))
    return sum(scores) / len(scores) if scores else 0.0
