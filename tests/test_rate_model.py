from dpeot.models.rate import GammaPoissonRate


def test_gamma_poisson_update_preserves_shape_rate_conjugacy() -> None:
    prior = GammaPoissonRate(shape=2.0, rate=1.0)
    posterior = prior.update(count=5)

    assert posterior.shape == 7.0
    assert posterior.rate == 2.0
    assert posterior.mean == 3.5


def test_gamma_poisson_predictive_is_finite() -> None:
    model = GammaPoissonRate(shape=2.0, rate=1.0)

    assert model.log_predictive_pmf(0) < 0.0
    assert model.log_predictive_pmf(3) < 0.0
