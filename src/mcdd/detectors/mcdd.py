"""Multiple Contrast Drift Detection (MCDD).

MCDD evaluates several statistical contrasts inside a data window and applies
a false discovery rate correction to the resulting p-values. Three window
modes are supported:

- ``sliding``: fixed-size sliding window;
- ``growing_dynamic``: growing window with a fixed number of subwindows;
- ``growing_fixed``: growing window with a fixed subwindow size and an
  increasing number of contrasts.

The detector is intentionally latched. Once a drift is detected,
``drift_detected`` remains ``True`` until :meth:`MCDD.reset` is called.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Literal, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray
from statsmodels.stats.multitest import multipletests

WindowMode: TypeAlias = Literal[
    "sliding",
    "growing_dynamic",
    "growing_fixed",
]
FDRCorrection: TypeAlias = Literal["fdr_by", "fdr_bh"]


def _as_one_dimensional_array(values: Sequence[float]) -> NDArray[np.float64]:
    """Convert input values to a finite one-dimensional NumPy array."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)

    if array.size == 0:
        raise ValueError("Input data must contain at least one observation.")
    if not np.isfinite(array).all():
        raise ValueError("Input data must contain only finite observations.")

    return array


class MCDD:
    """Multiple Contrast Drift Detection.

    Parameters
    ----------
    test:
        Hypothesis test receiving two one-dimensional samples. The test may
        return a p-value directly, a ``(statistic, p_value)`` tuple, a mapping
        containing a p-value, or an object with a ``pvalue`` attribute.
    window_size:
        Full window size in ``sliding`` mode. It also acts as the default
        minimum and maximum window size when those values are omitted.
    n_subwindows:
        Number of equal subwindows used in ``sliding`` and
        ``growing_dynamic`` modes. The latest subwindow is compared with every
        preceding subwindow.
    alpha:
        False discovery rate control level.
    window_mode:
        ``sliding``, ``growing_dynamic``, or ``growing_fixed``.
    min_window_size:
        Initial window size in a growing mode.
    max_window_size:
        Maximum window size in a growing mode. Once this size is reached, the
        window starts sliding.
    correction:
        ``fdr_by`` for Benjamini--Yekutieli or ``fdr_bh`` for
        Benjamini--Hochberg. Benjamini--Yekutieli is the default because the
        contrasts share the latest subwindow and may be dependent.
    min_rejections:
        Minimum number of rejected hypotheses required to trigger an alarm.
    buffer_dtype:
        Data type used when the internal deque is converted to a NumPy array.

    Notes
    -----
    Detection is latched. After the first alarm, ``drift_detected`` remains
    ``True`` in later updates. Call :meth:`reset` before using the same detector
    to identify another drift.
    """

    def __init__(
        self,
        test: Any,
        *,
        window_size: int = 6_000,
        n_subwindows: int = 10,
        alpha: float = 0.01,
        window_mode: WindowMode = "sliding",
        min_window_size: int | None = None,
        max_window_size: int | None = None,
        correction: FDRCorrection = "fdr_by",
        min_rejections: int = 1,
        buffer_dtype: Any = np.float64,
    ) -> None:
        self._validate_configuration(
            window_size=window_size,
            n_subwindows=n_subwindows,
            alpha=alpha,
            window_mode=window_mode,
            min_window_size=min_window_size,
            max_window_size=max_window_size,
            correction=correction,
            min_rejections=min_rejections,
        )

        self.test = test
        self.window_size = int(window_size)
        self.n_subwindows = int(n_subwindows)
        self.alpha = float(alpha)
        self.window_mode = window_mode
        self.correction = correction
        self.min_rejections = int(min_rejections)
        self.buffer_dtype = buffer_dtype

        if window_mode == "sliding":
            self.min_window_size = self.window_size
            self.max_window_size = self.window_size
        else:
            self.min_window_size = int(
                min_window_size
                if min_window_size is not None
                else self.window_size
            )
            self.max_window_size = int(
                max_window_size
                if max_window_size is not None
                else self.window_size
            )

        if self.min_window_size % self.n_subwindows != 0:
            raise ValueError(
                "min_window_size must be divisible by n_subwindows so that "
                "the fixed input batch size is well defined."
            )

        if (
            self.window_mode == "sliding"
            and self.window_size % self.n_subwindows != 0
        ):
            raise ValueError(
                "window_size must be divisible by n_subwindows in sliding mode."
            )

        self._batch_size = self.min_window_size // self.n_subwindows
        self._initial_subwindow_size = self._batch_size

        if self.window_mode == "sliding":
            self.buffer: deque[float] = deque(maxlen=self.window_size)
        else:
            self.buffer = deque()

        self.history: list[dict[str, Any]] = []
        self._drift_detected = False

    @staticmethod
    def _validate_configuration(
        *,
        window_size: int,
        n_subwindows: int,
        alpha: float,
        window_mode: str,
        min_window_size: int | None,
        max_window_size: int | None,
        correction: str,
        min_rejections: int,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if n_subwindows < 2:
            raise ValueError("n_subwindows must be at least 2.")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in the open interval (0, 1).")
        if window_mode not in {
            "sliding",
            "growing_dynamic",
            "growing_fixed",
        }:
            raise ValueError(
                "window_mode must be 'sliding', 'growing_dynamic', or "
                "'growing_fixed'."
            )
        if correction not in {"fdr_by", "fdr_bh"}:
            raise ValueError("correction must be 'fdr_by' or 'fdr_bh'.")
        if min_rejections <= 0:
            raise ValueError("min_rejections must be a positive integer.")

        if window_mode != "sliding":
            resolved_minimum = (
                int(min_window_size)
                if min_window_size is not None
                else int(window_size)
            )
            resolved_maximum = (
                int(max_window_size)
                if max_window_size is not None
                else int(window_size)
            )
            if resolved_minimum <= 0:
                raise ValueError("min_window_size must be positive.")
            if resolved_maximum < resolved_minimum:
                raise ValueError(
                    "max_window_size must be greater than or equal to "
                    "min_window_size."
                )

    @property
    def drift_detected(self) -> bool:
        """Whether this detector has raised an alarm since its last reset."""
        return self._drift_detected

    @property
    def batch_size(self) -> int:
        """Fixed number of observations expected by each update."""
        return self._batch_size

    def update(self, batch: Sequence[float]) -> dict[str, Any]:
        """Add one batch and evaluate the current window.

        The returned ``drift`` field is latched: after an alarm, it remains
        ``True`` until :meth:`reset` is called. ``new_detection`` indicates
        whether the current evaluation itself met the rejection rule.
        """
        batch_array = _as_one_dimensional_array(batch)

        if batch_array.size != self.batch_size:
            raise ValueError(
                f"Expected a batch of {self.batch_size} observations, "
                f"received {batch_array.size}."
            )

        self.buffer.extend(float(value) for value in batch_array)

        if self.window_mode != "sliding":
            while len(self.buffer) > self.max_window_size:
                self.buffer.popleft()

        if len(self.buffer) < self.min_window_size:
            record = {
                "drift": self._drift_detected,
                "new_detection": False,
                "raw_pvalues": [],
                "adjusted_pvalues": [],
                "reject": [],
                "n_subwindows": self.n_subwindows,
                "subwindow_size": self._initial_subwindow_size,
                "current_window_size": len(self.buffer),
                "ready": False,
            }
            self.history.append(record)
            return record

        window = np.fromiter(self.buffer, dtype=self.buffer_dtype)
        analysis_window, subwindow_size, number_of_subwindows = (
            self._prepare_analysis_window(window)
        )

        subwindows = [
            analysis_window[
                index * subwindow_size : (index + 1) * subwindow_size
            ]
            for index in range(number_of_subwindows)
        ]
        latest = subwindows[-1]

        raw_pvalues = [
            self._call_test(reference, latest)
            for reference in subwindows[:-1]
        ]

        reject_array, adjusted_array, _, _ = multipletests(
            raw_pvalues,
            alpha=self.alpha,
            method=self.correction,
        )

        rejection_count = int(np.count_nonzero(reject_array))
        new_detection = rejection_count >= self.min_rejections

        if new_detection:
            self._drift_detected = True

        record = {
            "drift": self._drift_detected,
            "new_detection": new_detection,
            "raw_pvalues": [float(value) for value in raw_pvalues],
            "adjusted_pvalues": [
                float(value) for value in adjusted_array
            ],
            "reject": [bool(value) for value in reject_array],
            "rejection_count": rejection_count,
            "n_subwindows": number_of_subwindows,
            "subwindow_size": subwindow_size,
            "current_window_size": len(window),
            "analysed_window_size": len(analysis_window),
            "ready": True,
        }
        self.history.append(record)
        return record

    def _prepare_analysis_window(
        self,
        window: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], int, int]:
        current_size = len(window)

        if self.window_mode == "growing_fixed":
            subwindow_size = self._initial_subwindow_size
            number_of_subwindows = current_size // subwindow_size
            number_of_subwindows = max(2, number_of_subwindows)
            usable_size = number_of_subwindows * subwindow_size

            # Keep the most recent complete subwindows. For example, a
            # 20,000-sample window with 600-sample subwindows uses the latest
            # 19,800 observations rather than discarding the newest 200.
            analysis_window = window[-usable_size:]
            return analysis_window, subwindow_size, number_of_subwindows

        number_of_subwindows = self.n_subwindows
        subwindow_size = current_size // number_of_subwindows
        usable_size = subwindow_size * number_of_subwindows

        # In the paper configurations the window sizes are divisible by the
        # number of subwindows. Retaining the most recent complete portion also
        # makes the behavior well defined for other valid sizes.
        analysis_window = window[-usable_size:]
        return analysis_window, subwindow_size, number_of_subwindows

    def _call_test(
        self,
        reference: NDArray[np.float64],
        latest: NDArray[np.float64],
    ) -> float:
        if hasattr(self.test, "test") and callable(self.test.test):
            result = self.test.test(reference, latest)
        elif callable(self.test):
            result = self.test(reference, latest)
        else:
            raise ValueError(
                "test must be callable or expose a callable 'test' method."
            )

        if hasattr(result, "pvalue"):
            pvalue = float(result.pvalue)
        elif isinstance(result, (float, int, np.floating, np.integer)):
            pvalue = float(result)
        elif isinstance(result, tuple) and len(result) >= 2:
            pvalue = float(result[1])
        elif isinstance(result, dict):
            for key in ("pvalue", "p_val", "p"):
                if key in result:
                    pvalue = float(result[key])
                    break
            else:
                raise ValueError(
                    "A mapping returned by the test must contain a p-value."
                )
        else:
            raise ValueError(
                "The test must return a p-value, a (statistic, p-value) tuple, "
                "a mapping containing a p-value, or an object with a "
                "'pvalue' attribute."
            )

        if not 0.0 <= pvalue <= 1.0:
            raise ValueError(f"Invalid p-value returned by test: {pvalue}.")

        return pvalue

    def reset(self) -> None:
        """Clear the window, history, and latched detection state."""
        self.buffer.clear()
        self.history.clear()
        self._drift_detected = False

    def last_result(self) -> dict[str, Any] | None:
        """Return the most recent evaluation record, if available."""
        return self.history[-1] if self.history else None
