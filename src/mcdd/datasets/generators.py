"""Synthetic data-stream generators used in the MCDD experiments.

The module supports abrupt, gradual, and incremental drift for normal,
exponential, and gamma distributions. Gradual transitions mix observations
from the old and new concepts according to a sigmoid probability. Incremental
transitions interpolate the parameters of the selected distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

Distribution: TypeAlias = Literal["normal", "exponential", "gamma"]
DriftType: TypeAlias = Literal["abrupt", "gradual", "incremental"]

SUPPORTED_DISTRIBUTIONS: tuple[Distribution, ...] = (
    "normal",
    "exponential",
    "gamma",
)
SUPPORTED_DRIFT_TYPES: tuple[DriftType, ...] = (
    "abrupt",
    "gradual",
    "incremental",
)
DEFAULT_TRANSITION_SIZES: tuple[int, ...] = (1_000, 2_000, 3_000)


@dataclass(frozen=True, slots=True)
class GeneratedStream:
    """A generated stream and the metadata required to interpret it."""

    values: NDArray[np.float64]
    drift_start: int
    drift_end: int
    transition_size: int
    parameter_names: tuple[str, ...]
    before_parameters: tuple[float, ...]
    after_parameters: tuple[float, ...]


def generate_stream(
    *,
    drift_type: DriftType,
    distribution: Distribution,
    seed: int,
    n_samples: int = 70_000,
    drift_point: int = 40_000,
    noise_standard_deviation: float = 0.1,
    transition_sizes: tuple[int, ...] = DEFAULT_TRANSITION_SIZES,
) -> GeneratedStream:
    """Generate one synthetic data stream.

    Parameters
    ----------
    drift_type:
        Temporal drift pattern: ``abrupt``, ``gradual``, or ``incremental``.
    distribution:
        Continuous distribution used before, during, and after the drift:
        ``normal``, ``exponential``, or ``gamma``.
    seed:
        Random seed associated with the stream.
    n_samples:
        Total number of observations.
    drift_point:
        Index at which the drift begins.
    noise_standard_deviation:
        Standard deviation of the additive Gaussian noise.
    transition_sizes:
        Candidate transition lengths for gradual and incremental drift.
    """
    _validate_inputs(
        drift_type=drift_type,
        distribution=distribution,
        n_samples=n_samples,
        drift_point=drift_point,
        noise_standard_deviation=noise_standard_deviation,
        transition_sizes=transition_sizes,
    )

    rng = np.random.RandomState(seed)
    parameter_names, before_parameters, after_parameters = _draw_parameters(
        distribution, rng
    )

    if drift_type == "abrupt":
        transition_size = 0
        before_values = _sample_distribution(
            distribution, before_parameters, drift_point, rng
        )
        after_values = _sample_distribution(
            distribution, after_parameters, n_samples - drift_point, rng
        )
        values = np.concatenate((before_values, after_values))
        drift_end = drift_point
    else:
        transition_size = int(rng.choice(np.asarray(transition_sizes)))
        if drift_point + transition_size > n_samples:
            raise ValueError(
                "drift_point plus transition_size must not exceed n_samples."
            )

        before_values = _sample_distribution(
            distribution, before_parameters, drift_point, rng
        )
        after_values = _sample_distribution(
            distribution,
            after_parameters,
            n_samples - drift_point - transition_size,
            rng,
        )

        if drift_type == "gradual":
            transition_values = _generate_gradual_transition(
                distribution=distribution,
                before_parameters=before_parameters,
                after_parameters=after_parameters,
                transition_size=transition_size,
                rng=rng,
            )
        else:
            transition_values = _generate_incremental_transition(
                distribution=distribution,
                before_parameters=before_parameters,
                after_parameters=after_parameters,
                transition_size=transition_size,
                rng=rng,
            )

        values = np.concatenate((before_values, transition_values, after_values))
        drift_end = drift_point + transition_size

    values = _add_noise(
        values=values,
        distribution=distribution,
        noise_standard_deviation=noise_standard_deviation,
        rng=rng,
    )

    return GeneratedStream(
        values=values,
        drift_start=drift_point,
        drift_end=drift_end,
        transition_size=transition_size,
        parameter_names=parameter_names,
        before_parameters=before_parameters,
        after_parameters=after_parameters,
    )


def _draw_parameters(
    distribution: Distribution,
    rng: np.random.RandomState,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    before_first = float(rng.uniform(0.0, 5.0))
    before_scale = float(rng.uniform(0.5, 1.5))
    after_first = float(rng.uniform(0.0, 5.0))
    after_scale = float(rng.uniform(0.5, 1.5))

    if distribution == "normal":
        return (
            ("mean", "standard_deviation"),
            (before_first, before_scale),
            (after_first, after_scale),
        )
    if distribution == "exponential":
        return (("scale",), (before_scale,), (after_scale,))
    return (
        ("shape", "scale"),
        (before_first, before_scale),
        (after_first, after_scale),
    )


def _sample_distribution(
    distribution: Distribution,
    parameters: tuple[float, ...],
    size: int,
    rng: np.random.RandomState,
) -> NDArray[np.float64]:
    if distribution == "normal":
        values = rng.normal(
            loc=parameters[0],
            scale=parameters[1],
            size=size,
        )
    elif distribution == "exponential":
        values = rng.exponential(scale=parameters[0], size=size)
    else:
        values = rng.gamma(
            shape=parameters[0],
            scale=parameters[1],
            size=size,
        )

    return np.asarray(values, dtype=np.float64)


def _generate_gradual_transition(
    *,
    distribution: Distribution,
    before_parameters: tuple[float, ...],
    after_parameters: tuple[float, ...],
    transition_size: int,
    rng: np.random.RandomState,
) -> NDArray[np.float64]:
    positions = np.arange(transition_size, dtype=np.float64)
    midpoint = transition_size / 2.0
    slope = 8.0 / transition_size
    new_concept_probability = 1.0 / (
        1.0 + np.exp(-slope * (positions - midpoint))
    )

    values = np.empty(transition_size, dtype=np.float64)
    for index, probability in enumerate(new_concept_probability):
        use_new_concept = bool(
            rng.choice((False, True), p=(1.0 - probability, probability))
        )
        parameters = after_parameters if use_new_concept else before_parameters
        values[index] = _sample_distribution(
            distribution, parameters, 1, rng
        )[0]

    return values


def _generate_incremental_transition(
    *,
    distribution: Distribution,
    before_parameters: tuple[float, ...],
    after_parameters: tuple[float, ...],
    transition_size: int,
    rng: np.random.RandomState,
) -> NDArray[np.float64]:
    weights = np.linspace(0.0, 1.0, num=transition_size)

    if distribution == "normal":
        means = _interpolate(
            before_parameters[0], after_parameters[0], weights
        )
        standard_deviations = _interpolate(
            before_parameters[1], after_parameters[1], weights
        )
        values = rng.normal(loc=means, scale=standard_deviations)
    elif distribution == "exponential":
        scales = _interpolate(
            before_parameters[0], after_parameters[0], weights
        )
        values = rng.exponential(scale=scales)
    else:
        shapes = _interpolate(
            before_parameters[0], after_parameters[0], weights
        )
        scales = _interpolate(
            before_parameters[1], after_parameters[1], weights
        )
        values = rng.gamma(shape=shapes, scale=scales)

    return np.asarray(values, dtype=np.float64)


def _interpolate(
    initial_value: float,
    final_value: float,
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    return (1.0 - weights) * initial_value + weights * final_value


def _add_noise(
    *,
    values: NDArray[np.float64],
    distribution: Distribution,
    noise_standard_deviation: float,
    rng: np.random.RandomState,
) -> NDArray[np.float64]:
    noisy_values = values + rng.normal(
        loc=0.0,
        scale=noise_standard_deviation,
        size=values.size,
    )

    if distribution in {"exponential", "gamma"}:
        noisy_values = np.clip(noisy_values, a_min=0.0, a_max=None)

    return np.asarray(noisy_values, dtype=np.float64)


def _validate_inputs(
    *,
    drift_type: str,
    distribution: str,
    n_samples: int,
    drift_point: int,
    noise_standard_deviation: float,
    transition_sizes: tuple[int, ...],
) -> None:
    if drift_type not in SUPPORTED_DRIFT_TYPES:
        raise ValueError(
            f"Unsupported drift type {drift_type!r}. "
            f"Choose from {SUPPORTED_DRIFT_TYPES}."
        )
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(
            f"Unsupported distribution {distribution!r}. "
            f"Choose from {SUPPORTED_DISTRIBUTIONS}."
        )
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than zero.")
    if not 0 < drift_point < n_samples:
        raise ValueError("drift_point must lie inside the stream.")
    if noise_standard_deviation < 0:
        raise ValueError("noise_standard_deviation must be non-negative.")
    if not transition_sizes or any(size <= 0 for size in transition_sizes):
        raise ValueError(
            "transition_sizes must contain positive integer values."
        )
