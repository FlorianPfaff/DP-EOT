"""Gamma-Poisson measurement-rate model for extended objects."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log


@dataclass(frozen=True)
class GammaPoissonRate:
    """Conjugate Gamma-Poisson model using shape/rate parameterization.

    The latent measurement rate ``gamma`` has prior

        gamma ~ Gamma(shape, rate)

    and the per-scan measurement count follows

        n | gamma ~ Poisson(gamma).

    This model is intentionally separate from any DP/CRP partition proposal so
    that physical measurement counts are not inferred from occupied clusters.
    """

    shape: float
    rate: float

    def __post_init__(self) -> None:
        if self.shape <= 0:
            raise ValueError("shape must be positive")
        if self.rate <= 0:
            raise ValueError("rate must be positive")

    @property
    def mean(self) -> float:
        return self.shape / self.rate

    @property
    def variance(self) -> float:
        return self.shape / (self.rate * self.rate)

    def update(self, count: int, exposure: float = 1.0) -> "GammaPoissonRate":
        """Return the posterior after observing a nonnegative count."""

        if count < 0:
            raise ValueError("count must be nonnegative")
        if exposure <= 0:
            raise ValueError("exposure must be positive")
        return GammaPoissonRate(shape=self.shape + count, rate=self.rate + exposure)

    def log_predictive_pmf(self, count: int, exposure: float = 1.0) -> float:
        """Negative-binomial log predictive probability for a future count."""

        if count < 0:
            raise ValueError("count must be nonnegative")
        if exposure <= 0:
            raise ValueError("exposure must be positive")

        # p(n) = Gamma(a+n)/(Gamma(a)n!) * (b/(b+t))^a * (t/(b+t))^n
        a = self.shape
        b = self.rate
        t = exposure
        return (
            lgamma(a + count)
            - lgamma(a)
            - lgamma(count + 1)
            + a * log(b / (b + t))
            + count * log(t / (b + t))
        )
