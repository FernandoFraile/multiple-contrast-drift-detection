"""Tests for the synthetic concept-drift generators."""

from __future__ import annotations

import numpy as np
import pytest

from mcdd.datasets import generate_stream


@pytest.mark.parametrize(
    ("drift_type", "distribution"),
    [
        (drift_type, distribution)
        for drift_type in ("abrupt", "gradual", "incremental")
        for distribution in ("normal", "exponential", "gamma")
    ],
)
def test_generated_stream_has_expected_shape_and_metadata(
    drift_type: str,
    distribution: str,
) -> None:
    """Every supported configuration should produce a valid stream."""
    stream = generate_stream(
        drift_type=drift_type,
        distribution=distribution,
        seed=42,
        n_samples=300,
        drift_point=180,
        transition_sizes=(40,),
    )

    assert stream.values.shape == (300,)
    assert stream.values.dtype == np.float64
    assert np.isfinite(stream.values).all()
    assert stream.drift_start == 180

    if drift_type == "abrupt":
        assert stream.transition_size == 0
        assert stream.drift_end == 180
    else:
        assert stream.transition_size == 40
        assert stream.drift_end == 220


def test_same_seed_reproduces_the_same_stream() -> None:
    """The same inputs and seed must reproduce the same values and metadata."""
    first = generate_stream(
        drift_type="incremental",
        distribution="gamma",
        seed=137,
        n_samples=500,
        drift_point=300,
        transition_sizes=(80,),
    )
    second = generate_stream(
        drift_type="incremental",
        distribution="gamma",
        seed=137,
        n_samples=500,
        drift_point=300,
        transition_sizes=(80,),
    )

    np.testing.assert_array_equal(first.values, second.values)
    assert first.drift_start == second.drift_start
    assert first.drift_end == second.drift_end
    assert first.transition_size == second.transition_size
    assert first.parameter_names == second.parameter_names
    assert first.before_parameters == second.before_parameters
    assert first.after_parameters == second.after_parameters


@pytest.mark.parametrize(
    ("drift_type", "distribution"),
    [
        ("gradual", "exponential"),
        ("gradual", "gamma"),
        ("incremental", "exponential"),
        ("incremental", "gamma"),
    ],
)
def test_positive_support_distributions_remain_non_negative(
    drift_type: str,
    distribution: str,
) -> None:
    """Noise correction must preserve the support of exponential and gamma data."""
    stream = generate_stream(
        drift_type=drift_type,
        distribution=distribution,
        seed=52,
        n_samples=400,
        drift_point=240,
        noise_standard_deviation=0.1,
        transition_sizes=(60,),
    )

    assert np.all(stream.values >= 0.0)


def test_transition_size_must_fit_inside_the_stream() -> None:
    """Invalid transition limits should be rejected explicitly."""
    with pytest.raises(ValueError):
        generate_stream(
            drift_type="gradual",
            distribution="normal",
            seed=42,
            n_samples=100,
            drift_point=80,
            transition_sizes=(30,),
        )
