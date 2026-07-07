"""Oracle partition helper for controlled experiments."""

from __future__ import annotations

from dpeot.scenarios.two_target_merge_split import Scan, oracle_cells


def oracle_partition(scan: Scan, group_during_unresolved: bool = True) -> list[list[int]]:
    """Return ground-truth measurement cells for upper-bound diagnostics."""

    return oracle_cells(scan, group_during_unresolved=group_during_unresolved)
