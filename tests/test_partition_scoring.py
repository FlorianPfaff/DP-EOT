import numpy as np

from dpeot.tracking.partition_scoring import cell_centroid, log_cell_likelihood


def test_cell_centroid_matches_mean() -> None:
    measurements = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0]])

    assert np.allclose(cell_centroid(measurements, [0, 1]), np.array([1.0, 0.0]))


def test_log_cell_likelihood_prefers_nearby_cell() -> None:
    measurements = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    covariance = 0.5 * np.eye(2)

    near_score = log_cell_likelihood(
        measurements,
        cell=[0, 1],
        predicted_position=np.array([0.0, 0.0]),
        expected_rate=2.0,
        covariance=covariance,
    )
    far_score = log_cell_likelihood(
        measurements,
        cell=[2, 3],
        predicted_position=np.array([0.0, 0.0]),
        expected_rate=2.0,
        covariance=covariance,
    )

    assert near_score > far_score
