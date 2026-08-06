"""LORD procedure for a sequence of locally dependent hypotheses."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class LORDLocalDependence:
    """LORD with delayed rewards for local dependence.

    Parameters
    ----------
    alpha:
        Target online false discovery rate.
    number_of_hypotheses:
        Length of the p-value sequence.
    lag:
        Local dependence lag. A rejection becomes available for future alpha
        allocation only after this number of steps.
    start_fraction:
        Fraction of ``alpha`` used as the initial wealth.
    gamma_exponent:
        Exponent of the normalized power-law gamma sequence.
    """

    def __init__(
        self,
        *,
        alpha: float,
        number_of_hypotheses: int,
        lag: int,
        start_fraction: float = 0.1,
        gamma_exponent: float = 1.6,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in the open interval (0, 1).")
        if number_of_hypotheses <= 0:
            raise ValueError("number_of_hypotheses must be positive.")
        if lag < 0:
            raise ValueError("lag must be non-negative.")
        if not 0.0 < start_fraction < 1.0:
            raise ValueError(
                "start_fraction must be in the open interval (0, 1)."
            )
        if gamma_exponent <= 1.0:
            raise ValueError(
                "gamma_exponent must be greater than 1 so the power-law "
                "sequence is summable."
            )

        self.alpha = float(alpha)
        self.number_of_hypotheses = int(number_of_hypotheses)
        self.lag = int(lag)
        self.start_fraction = float(start_fraction)
        self.gamma_exponent = float(gamma_exponent)

        sequence_length = max(
            10_000,
            self.number_of_hypotheses + self.lag + 2,
        )
        positions = np.arange(1, sequence_length + 1, dtype=np.float64)
        gamma = 1.0 / np.power(positions, self.gamma_exponent)
        self.gamma = gamma / gamma.sum()

        self.initial_wealth = self.start_fraction * self.alpha
        self.alpha_levels_: NDArray[np.float64] | None = None

    def run(self, pvalues: ArrayLike) -> NDArray[np.bool_]:
        """Evaluate a complete p-value sequence and return rejections."""
        pvalue_array = np.asarray(pvalues, dtype=np.float64).reshape(-1)

        if len(pvalue_array) != self.number_of_hypotheses:
            raise ValueError(
                "The p-value vector length does not match "
                "number_of_hypotheses."
            )
        if not np.isfinite(pvalue_array).all():
            raise ValueError("p-values must be finite.")
        if np.any((pvalue_array < 0.0) | (pvalue_array > 1.0)):
            raise ValueError("p-values must lie in [0, 1].")

        alpha_levels = np.zeros(self.number_of_hypotheses + 1)
        alpha_levels[1] = self.gamma[0] * self.initial_wealth

        delayed_rejections = np.zeros(
            self.number_of_hypotheses + 1 + self.lag,
            dtype=np.int64,
        )
        available_rejection_times: list[int] = []

        for index in range(self.number_of_hypotheses):
            testing_level = alpha_levels[index + 1]
            completion_time = index + 1 + self.lag

            rejected = pvalue_array[index] < testing_level
            delayed_rejections[completion_time] = int(rejected)

            if delayed_rejections[index + 1] >= 1:
                available_rejection_times.extend(
                    [index + 1] * int(delayed_rejections[index + 1])
                )

            if available_rejection_times:
                first_time = available_rejection_times[0]
                first_gamma = (
                    self.gamma[index + 1 - first_time]
                    if first_time <= index + 1
                    else 0.0
                )

                if len(available_rejection_times) >= 2:
                    remaining_times = np.asarray(
                        available_rejection_times[1:],
                        dtype=np.int64,
                    )
                    available_mask = remaining_times <= index + 1
                    offsets = index + 1 - remaining_times[available_mask]
                    gamma_sum = float(self.gamma[offsets].sum())
                else:
                    gamma_sum = 0.0

                next_alpha = (
                    self.gamma[index + 1] * self.initial_wealth
                    + (self.alpha - self.initial_wealth) * first_gamma
                    + self.alpha * gamma_sum
                )
            else:
                next_alpha = self.gamma[index + 1] * self.initial_wealth

            if index < self.number_of_hypotheses - 1:
                alpha_levels[index + 2] = next_alpha

        self.alpha_levels_ = alpha_levels[1:]
        return delayed_rejections[1 + self.lag :].astype(bool)
