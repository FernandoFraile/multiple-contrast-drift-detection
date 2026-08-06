"""Tests for the Multiple Contrast Drift Detection implementation."""

from __future__ import annotations

import numpy as np
import pytest

from mcdd.detectors import MCDD


class SequentialPValueTest:
    """Return a predefined sequence of p-values."""

    def __init__(self, pvalues: list[float]) -> None:
        self._pvalues = iter(pvalues)

    def __call__(
        self,
        first_sample: np.ndarray,
        second_sample: np.ndarray,
    ) -> float:
        del first_sample, second_sample
        return next(self._pvalues)


def test_detection_remains_latched_until_reset() -> None:
    """A later non-rejection must not clear an earlier MCDD alarm."""
    hypothesis_test = SequentialPValueTest([0.0, 1.0])
    detector = MCDD(
        hypothesis_test,
        window_size=4,
        n_subwindows=2,
        alpha=0.01,
    )

    initial = detector.update([0.0, 0.0])
    detected = detector.update([1.0, 1.0])
    later_update = detector.update([1.0, 1.0])

    assert initial["ready"] is False
    assert initial["drift"] is False

    assert detected["new_detection"] is True
    assert detected["drift"] is True
    assert detector.drift_detected is True

    assert later_update["new_detection"] is False
    assert later_update["drift"] is True
    assert detector.drift_detected is True


def test_reset_clears_buffer_history_and_detection_state() -> None:
    """Reset must make the detector ready for a new independent stream."""
    detector = MCDD(
        lambda first, second: 0.0,
        window_size=4,
        n_subwindows=2,
        alpha=0.01,
    )

    detector.update([0.0, 0.0])
    detector.update([1.0, 1.0])

    assert detector.drift_detected is True
    assert detector.history
    assert detector.buffer

    detector.reset()

    assert detector.drift_detected is False
    assert detector.history == []
    assert len(detector.buffer) == 0
    assert detector.last_result() is None


def test_update_requires_the_configured_batch_size() -> None:
    """MCDD should reject batches that do not match its input step."""
    detector = MCDD(
        lambda first, second: 1.0,
        window_size=6,
        n_subwindows=3,
    )

    assert detector.batch_size == 2

    with pytest.raises(ValueError, match="Expected a batch of 2"):
        detector.update([1.0])


def test_growing_fixed_uses_the_most_recent_complete_subwindows() -> None:
    """An incomplete remainder must be removed from the oldest observations."""
    detector = MCDD(
        lambda first, second: 1.0,
        window_size=6,
        n_subwindows=3,
        window_mode="growing_fixed",
        min_window_size=6,
        max_window_size=11,
    )

    values = np.arange(11, dtype=np.float64)
    analysis_window, subwindow_size, number_of_subwindows = (
        detector._prepare_analysis_window(values)
    )

    assert subwindow_size == 2
    assert number_of_subwindows == 5
    np.testing.assert_array_equal(
        analysis_window,
        np.arange(1, 11, dtype=np.float64),
    )


def test_minimum_number_of_rejections_controls_the_alarm() -> None:
    """The alarm should require at least min_rejections rejected contrasts."""
    detector_with_two_required = MCDD(
        lambda first, second: 0.0,
        window_size=6,
        n_subwindows=3,
        min_rejections=2,
    )
    detector_with_three_required = MCDD(
        lambda first, second: 0.0,
        window_size=6,
        n_subwindows=3,
        min_rejections=3,
    )

    for batch in ([0.0, 0.0], [0.0, 0.0], [1.0, 1.0]):
        result_two = detector_with_two_required.update(batch)
        result_three = detector_with_three_required.update(batch)

    assert result_two["rejection_count"] == 2
    assert result_two["new_detection"] is True
    assert detector_with_two_required.drift_detected is True

    assert result_three["rejection_count"] == 2
    assert result_three["new_detection"] is False
    assert detector_with_three_required.drift_detected is False
