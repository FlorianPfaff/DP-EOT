import numpy as np

from dpeot.partitions.mfm_partition import MFMPartitionConfig, mfm_partition


def test_mfm_partition_returns_nonempty_cells_covering_all_points() -> None:
    measurements = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [4.0, 4.0],
            [4.1, 4.0],
        ]
    )

    cells = mfm_partition(
        measurements,
        MFMPartitionConfig(max_components=3, num_initializations=3, seed=5),
    )

    flattened = sorted(index for cell in cells for index in cell)
    assert flattened == [0, 1, 2, 3]
    assert all(cell for cell in cells)


def test_mfm_partition_handles_empty_scan() -> None:
    cells = mfm_partition(np.empty((0, 2)))

    assert cells == []
