from __future__ import annotations

"""Real-time filtering for the six normalized hand-joint values.

This module provides two causal filters with a shared ``update/reset`` API:

- ``JointValueFilter``: median + One Euro adaptive low-pass
- ``MovingAverageJointFilter``: fixed-window moving average

The default factory now returns the moving-average variant.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


DEFAULT_LIMITS = np.asarray(
    [
        [0.0, 1.0],   # thumb CMC yaw
        [0.0, 1.0],   # thumb CMC pitch
        [0.0, 1.0],   # index MCP
        [0.0, 1.0],   # middle MCP
        [0.0, 1.0],   # ring MCP
        [0.0, 1.0],   # pinky MCP
    ],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class JointFilterConfig:
    """Configuration for :class:`JointValueFilter`.

    Parameters
    ----------
    min_cutoff:
        One Euro cutoff frequency while the hand is nearly stationary.
        Smaller values remove more jitter but add more lag.
    beta:
        How quickly the cutoff rises during movement. Larger values make rapid
        movements more responsive.
    derivative_cutoff:
        Low-pass cutoff for the estimated joint velocity.
    median_window:
        Odd causal median-window size. ``3`` rejects single-frame spikes with
        little delay. Set to ``1`` to disable the median stage.
    reset_gap_s:
        Reset the filter after this much time without a valid hand estimate.
    nominal_frequency_hz:
        Used only when timestamps are missing or invalid.
    """

    min_cutoff: float | Sequence[float] = 1.0
    beta: float | Sequence[float] = 0.45
    derivative_cutoff: float | Sequence[float] = 1.0
    median_window: int = 3
    reset_gap_s: float = 0.25
    nominal_frequency_hz: float = 30.0


@dataclass(frozen=True, slots=True)
class MovingAverageFilterConfig:
    """Configuration for :class:`MovingAverageJointFilter`."""

    strength: float = 0.286
    jitter_enabled: bool = True
    jitter_ratio: float = 0.02
    min_window_size: int = 1
    max_window_size: int = 15
    reset_gap_s: float = 0.25
    nominal_frequency_hz: float = 30.0


class JointValueFilter:
    """Causal robust One Euro filter for a vector of joint values.

    The default configuration expects the estimator's six values in this order::

        [thumb_yaw, thumb_pitch, index_mcp, middle_mcp, ring_mcp, pinky_mcp]

    ``update`` returns ``None`` when the estimator has no valid hand. A short
    detection loss preserves the internal state, while a gap longer than
    ``reset_gap_s`` resets it to prevent stale-state jumps on reacquisition.
    """

    def __init__(
        self,
        config: JointFilterConfig | None = None,
        value_limits: np.ndarray | Sequence[Sequence[float]] = DEFAULT_LIMITS,
    ) -> None:
        self.config = config or JointFilterConfig()

        limits = np.asarray(value_limits, dtype=np.float64)
        if limits.ndim != 2 or limits.shape[1] != 2:
            raise ValueError("value_limits must have shape (num_joints, 2).")
        if np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("Each lower joint limit must be <= its upper limit.")

        if self.config.median_window < 1 or self.config.median_window % 2 == 0:
            raise ValueError("median_window must be a positive odd integer.")
        if self.config.reset_gap_s < 0.0:
            raise ValueError("reset_gap_s must be non-negative.")
        if self.config.nominal_frequency_hz <= 0.0:
            raise ValueError("nominal_frequency_hz must be positive.")

        self._limits = limits
        self._num_joints = limits.shape[0]
        self._min_cutoff = self._parameter_array(
            self.config.min_cutoff, "min_cutoff", strictly_positive=True
        )
        self._beta = self._parameter_array(
            self.config.beta, "beta", strictly_positive=False
        )
        self._derivative_cutoff = self._parameter_array(
            self.config.derivative_cutoff,
            "derivative_cutoff",
            strictly_positive=True,
        )

        self._raw_history: deque[np.ndarray] = deque(
            maxlen=self.config.median_window
        )
        self._previous_raw: Optional[np.ndarray] = None
        self._filtered_value: Optional[np.ndarray] = None
        self._filtered_derivative: Optional[np.ndarray] = None
        self._last_timestamp_s: Optional[float] = None
        self._last_valid_timestamp_s: Optional[float] = None

    @property
    def initialized(self) -> bool:
        return self._filtered_value is not None

    @property
    def last_value(self) -> Optional[np.ndarray]:
        if self._filtered_value is None:
            return None
        return self._filtered_value.astype(np.float32, copy=True)

    def reset(self) -> None:
        """Clear all temporal state."""
        self._raw_history.clear()
        self._previous_raw = None
        self._filtered_value = None
        self._filtered_derivative = None
        self._last_timestamp_s = None
        self._last_valid_timestamp_s = None

    def update(
        self,
        joint_values: Optional[Sequence[float] | np.ndarray],
        timestamp_s: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Filter one estimator output.

        Parameters
        ----------
        joint_values:
            Six raw estimator values, or ``None`` when no hand is detected.
        timestamp_s:
            Monotonic frame timestamp. Passing ``HandEstimate.timestamp_s`` is
            recommended. If omitted, the configured nominal frequency is used.
        """
        timestamp = self._resolve_timestamp(timestamp_s)

        if joint_values is None:
            self._handle_missing_measurement(timestamp)
            return None

        measurement = np.asarray(joint_values, dtype=np.float64)
        if measurement.shape != (self._num_joints,):
            raise ValueError(
                f"Expected {self._num_joints} joint values, got "
                f"shape {measurement.shape}."
            )
        if not np.all(np.isfinite(measurement)):
            self._handle_missing_measurement(timestamp)
            return None

        measurement = np.clip(
            measurement,
            self._limits[:, 0],
            self._limits[:, 1],
        )
        self._last_valid_timestamp_s = timestamp

        self._raw_history.append(measurement.copy())
        robust_measurement = np.median(
            np.stack(tuple(self._raw_history), axis=0),
            axis=0,
        )

        if self._filtered_value is None:
            self._previous_raw = robust_measurement.copy()
            self._filtered_value = robust_measurement.copy()
            self._filtered_derivative = np.zeros_like(robust_measurement)
            self._last_timestamp_s = timestamp
            return self._filtered_value.astype(np.float32, copy=True)

        dt = timestamp - float(self._last_timestamp_s)
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1.0 / self.config.nominal_frequency_hz

        raw_derivative = (
            robust_measurement - self._previous_raw
        ) / dt

        derivative_alpha = self._alpha(
            self._derivative_cutoff,
            dt,
        )
        self._filtered_derivative = self._low_pass(
            raw_derivative,
            self._filtered_derivative,
            derivative_alpha,
        )

        adaptive_cutoff = self._min_cutoff + self._beta * np.abs(
            self._filtered_derivative
        )
        value_alpha = self._alpha(adaptive_cutoff, dt)
        self._filtered_value = self._low_pass(
            robust_measurement,
            self._filtered_value,
            value_alpha,
        )
        self._filtered_value = np.clip(
            self._filtered_value,
            self._limits[:, 0],
            self._limits[:, 1],
        )

        self._previous_raw = robust_measurement.copy()
        self._last_timestamp_s = timestamp
        return self._filtered_value.astype(np.float32, copy=True)

    def update_estimate(self, estimate) -> Optional[np.ndarray]:
        """Convenience adapter for a ``HandEstimate``-like object."""
        if not hasattr(estimate, "joint_values"):
            raise TypeError("estimate must have a joint_values attribute.")
        timestamp_s = getattr(estimate, "timestamp_s", None)
        return self.update(estimate.joint_values, timestamp_s)

    def _handle_missing_measurement(self, timestamp_s: float) -> None:
        if self._last_valid_timestamp_s is None:
            return
        if timestamp_s - self._last_valid_timestamp_s > self.config.reset_gap_s:
            self.reset()

    def _resolve_timestamp(self, timestamp_s: Optional[float]) -> float:
        if timestamp_s is None or not np.isfinite(timestamp_s):
            if self._last_timestamp_s is None:
                return 0.0
            return self._last_timestamp_s + (
                1.0 / self.config.nominal_frequency_hz
            )
        return float(timestamp_s)

    def _parameter_array(
        self,
        value: float | Sequence[float],
        name: str,
        *,
        strictly_positive: bool,
    ) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = np.full(self._num_joints, float(array), dtype=np.float64)
        elif array.shape != (self._num_joints,):
            raise ValueError(
                f"{name} must be a scalar or have shape "
                f"({self._num_joints},)."
            )

        if strictly_positive:
            invalid = array <= 0.0
            requirement = "greater than zero"
        else:
            invalid = array < 0.0
            requirement = "non-negative"
        if np.any(invalid) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite and {requirement}.")
        return array

    @staticmethod
    def _alpha(cutoff_hz: np.ndarray, dt: float) -> np.ndarray:
        tau = 1.0 / (2.0 * np.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / dt)

    @staticmethod
    def _low_pass(
        current: np.ndarray,
        previous: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        return alpha * current + (1.0 - alpha) * previous


class MovingAverageJointFilter:
    """Causal moving-average filter for the six-joint estimator output."""

    def __init__(
        self,
        config: MovingAverageFilterConfig | None = None,
        value_limits: np.ndarray | Sequence[Sequence[float]] = DEFAULT_LIMITS,
    ) -> None:
        self.config = config or MovingAverageFilterConfig()

        limits = np.asarray(value_limits, dtype=np.float64)
        if limits.ndim != 2 or limits.shape[1] != 2:
            raise ValueError("value_limits must have shape (num_joints, 2).")
        if np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("Each lower joint limit must be <= its upper limit.")
        if not 0.0 <= self.config.strength <= 1.0:
            raise ValueError("strength must be in the range [0, 1].")
        if not 0.0 <= self.config.jitter_ratio <= 1.0:
            raise ValueError("jitter_ratio must be in the range [0, 1].")
        if self.config.min_window_size < 1:
            raise ValueError("min_window_size must be a positive integer.")
        if self.config.max_window_size < self.config.min_window_size:
            raise ValueError(
                "max_window_size must be >= min_window_size."
            )
        if self.config.reset_gap_s < 0.0:
            raise ValueError("reset_gap_s must be non-negative.")
        if self.config.nominal_frequency_hz <= 0.0:
            raise ValueError("nominal_frequency_hz must be positive.")

        self._limits = limits
        self._num_joints = limits.shape[0]
        self._jitter_threshold = (
            (self._limits[:, 1] - self._limits[:, 0]) * self.config.jitter_ratio
        )
        self._history: deque[np.ndarray] = deque(
            maxlen=self.config.max_window_size
        )
        self._filtered_value: Optional[np.ndarray] = None
        self._last_timestamp_s: Optional[float] = None
        self._last_valid_timestamp_s: Optional[float] = None
        self._strength = float(self.config.strength)
        self._active_window_size = self._window_size_from_strength(self._strength)

    @property
    def initialized(self) -> bool:
        return self._filtered_value is not None

    @property
    def strength(self) -> float:
        return self._strength

    @property
    def window_size(self) -> int:
        return self._active_window_size

    @property
    def jitter_enabled(self) -> bool:
        return self.config.jitter_enabled

    @property
    def last_value(self) -> Optional[np.ndarray]:
        if self._filtered_value is None:
            return None
        return self._filtered_value.astype(np.float32, copy=True)

    def reset(self) -> None:
        """Clear all temporal state."""
        self._history.clear()
        self._filtered_value = None
        self._last_timestamp_s = None
        self._last_valid_timestamp_s = None

    def set_strength(self, strength: float) -> None:
        """Set smoothing strength in [0, 1]. Larger means more smoothing."""
        strength = float(strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in the range [0, 1].")
        self._strength = strength
        self._active_window_size = self._window_size_from_strength(strength)

    def set_jitter_enabled(self, enabled: bool) -> None:
        """Enable or disable per-joint jitter deadband suppression."""
        self.config = MovingAverageFilterConfig(
            strength=self.config.strength,
            jitter_enabled=bool(enabled),
            jitter_ratio=self.config.jitter_ratio,
            min_window_size=self.config.min_window_size,
            max_window_size=self.config.max_window_size,
            reset_gap_s=self.config.reset_gap_s,
            nominal_frequency_hz=self.config.nominal_frequency_hz,
        )

    def update(
        self,
        joint_values: Optional[Sequence[float] | np.ndarray],
        timestamp_s: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        timestamp = self._resolve_timestamp(timestamp_s)

        if joint_values is None:
            self._handle_missing_measurement(timestamp)
            return None

        measurement = np.asarray(joint_values, dtype=np.float64)
        if measurement.shape != (self._num_joints,):
            raise ValueError(
                f"Expected {self._num_joints} joint values, got "
                f"shape {measurement.shape}."
            )
        if not np.all(np.isfinite(measurement)):
            self._handle_missing_measurement(timestamp)
            return None

        measurement = np.clip(
            measurement,
            self._limits[:, 0],
            self._limits[:, 1],
        )
        self._last_valid_timestamp_s = timestamp
        self._last_timestamp_s = timestamp

        self._history.append(measurement.copy())
        history_values = tuple(self._history)[-self._active_window_size:]
        candidate_value = np.mean(
            np.stack(history_values, axis=0),
            axis=0,
        )
        candidate_value = np.clip(
            candidate_value,
            self._limits[:, 0],
            self._limits[:, 1],
        )

        if self._filtered_value is None:
            self._filtered_value = candidate_value
        elif not self.config.jitter_enabled:
            self._filtered_value = candidate_value
        else:
            delta = np.abs(candidate_value - self._filtered_value)
            self._filtered_value = np.where(
                delta <= self._jitter_threshold,
                self._filtered_value,
                candidate_value,
            )
        return self._filtered_value.astype(np.float32, copy=True)

    def update_estimate(self, estimate) -> Optional[np.ndarray]:
        """Convenience adapter for a ``HandEstimate``-like object."""
        if not hasattr(estimate, "joint_values"):
            raise TypeError("estimate must have a joint_values attribute.")
        timestamp_s = getattr(estimate, "timestamp_s", None)
        return self.update(estimate.joint_values, timestamp_s)

    def _handle_missing_measurement(self, timestamp_s: float) -> None:
        if self._last_valid_timestamp_s is None:
            self._last_timestamp_s = timestamp_s
            return
        self._last_timestamp_s = timestamp_s
        if timestamp_s - self._last_valid_timestamp_s > self.config.reset_gap_s:
            self.reset()

    def _resolve_timestamp(self, timestamp_s: Optional[float]) -> float:
        if timestamp_s is None or not np.isfinite(timestamp_s):
            if self._last_timestamp_s is None:
                return 0.0
            return self._last_timestamp_s + (
                1.0 / self.config.nominal_frequency_hz
            )
        return float(timestamp_s)

    def _window_size_from_strength(self, strength: float) -> int:
        span = self.config.max_window_size - self.config.min_window_size
        if span <= 0:
            return self.config.min_window_size
        return self.config.min_window_size + int(round(strength * span))


# Recommended starting point for the optional One Euro filter.
DEFAULT_HAND_FILTER_CONFIG = JointFilterConfig(
    min_cutoff=(0.65, 0.80, 0.90, 0.90, 0.90, 0.90),
    beta=(2.00, 1.80, 1.50, 1.50, 1.50, 1.50),
    derivative_cutoff=2.0,
    median_window=3,
    reset_gap_s=0.25,
    nominal_frequency_hz=30.0,
)

DEFAULT_MOVING_AVERAGE_FILTER_CONFIG = MovingAverageFilterConfig(
    strength=0.286,
    jitter_enabled=True,
    jitter_ratio=0.02,
    min_window_size=1,
    max_window_size=15,
    reset_gap_s=0.25,
    nominal_frequency_hz=30.0,
)

def make_default_hand_filter() -> MovingAverageJointFilter:
    """Create the default moving-average filter for the current 30 Hz estimator."""
    return MovingAverageJointFilter(config=DEFAULT_MOVING_AVERAGE_FILTER_CONFIG)
