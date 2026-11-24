"""Time-domain and frequency-domain analysis tools for analog circuits.

This module provides functions to:

* Compute Bode-plot data (magnitude & phase vs. frequency).
* Compute step and impulse responses.
* Locate −3 dB cutoff and resonant-peak frequencies.
* Extract poles, zeros, and assess BIBO stability.

All functions accept transfer-function coefficient arrays in **descending**
power order, matching :class:`scipy.signal.TransferFunction` conventions.

Typical usage::

    >>> from analog_circuit_simulator.circuits import RCCircuit
    >>> from analog_circuit_simulator.analysis import (
    ...     analyze_frequency_response,
    ...     analyze_time_response,
    ... )
    >>> rc = RCCircuit(1e3, 1e-6)
    >>> num, den = rc.transfer_function()
    >>> freq_resp = analyze_frequency_response(num, den)
    >>> time_resp = analyze_time_response(num, den, response_type="step")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

# ======================================================================
# Data containers
# ======================================================================


@dataclass
class FrequencyResponse:
    """Container for frequency-domain analysis results.

    Attributes:
        frequencies: Frequency vector in Hz.
        magnitude_db: Magnitude in decibels (20·log₁₀|H|).
        phase_deg: Phase in degrees.
    """

    frequencies: np.ndarray
    magnitude_db: np.ndarray
    phase_deg: np.ndarray

    def to_csv(self, filepath: str | Path) -> None:
        """Export frequency response to a CSV file.

        Args:
            filepath: Destination file path.
        """
        data = np.column_stack((self.frequencies, self.magnitude_db, self.phase_deg))
        np.savetxt(
            filepath,
            data,
            delimiter=",",
            header="frequency_hz,magnitude_db,phase_deg",
            comments="",
        )

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert results to a dictionary of numpy arrays."""
        return {
            "frequency": self.frequencies,
            "magnitude_db": self.magnitude_db,
            "phase_deg": self.phase_deg,
        }


@dataclass
class TimeResponse:
    """Container for time-domain analysis results.

    Attributes:
        time: Time vector in seconds.
        output: System output signal.
        response_type: ``"step"`` or ``"impulse"``.
    """

    time: np.ndarray
    output: np.ndarray
    response_type: str

    def to_csv(self, filepath: str | Path) -> None:
        """Export time response to a CSV file.

        Args:
            filepath: Destination file path.
        """
        data = np.column_stack((self.time, self.output))
        header_name = f"output_{self.response_type}"
        np.savetxt(
            filepath,
            data,
            delimiter=",",
            header=f"time_s,{header_name}",
            comments="",
        )

    def to_dict(self) -> dict[str, np.ndarray | str]:
        """Convert results to a dictionary."""
        return {
            "time": self.time,
            "output": self.output,
            "response_type": self.response_type,
        }


# ======================================================================
# Core helpers
# ======================================================================


def compute_transfer_function(
    num: np.ndarray,
    den: np.ndarray,
) -> signal.TransferFunction:
    """Create a continuous-time :class:`scipy.signal.TransferFunction`.

    Args:
        num: Numerator polynomial coefficients (descending power).
        den: Denominator polynomial coefficients (descending power).

    Returns:
        A :class:`scipy.signal.TransferFunction` (continuous-time, ``s``
        domain).

    Raises:
        ValueError: If *num* or *den* are empty.

    Example:
        >>> import numpy as np
        >>> tf = compute_transfer_function(np.array([1.0]), np.array([1e-3, 1.0]))
        >>> tf
        TransferFunctionContinuous(
        array([1.]),
        array([0.001, 1.   ]),
        dt: None
        )
    """
    num = np.atleast_1d(np.asarray(num, dtype=float))
    den = np.atleast_1d(np.asarray(den, dtype=float))
    if num.size == 0:
        raise ValueError("Numerator coefficient array must not be empty.")
    if den.size == 0:
        raise ValueError("Denominator coefficient array must not be empty.")
    return signal.TransferFunction(num, den)


# ======================================================================
# Frequency-domain analysis
# ======================================================================


def analyze_frequency_response(
    num: np.ndarray,
    den: np.ndarray,
    freq_start: float = 1.0,
    freq_stop: float = 1e6,
    num_points: int = 1000,
) -> FrequencyResponse:
    """Compute Bode-plot data (magnitude and phase vs. frequency).

    Internally uses :func:`scipy.signal.bode` and returns the results wrapped
    in a :class:`FrequencyResponse` dataclass.

    Args:
        num: Numerator polynomial coefficients (descending power).
        den: Denominator polynomial coefficients (descending power).
        freq_start: Start frequency in Hz.  Must be positive.
        freq_stop: Stop frequency in Hz.  Must be greater than *freq_start*.
        num_points: Number of logarithmically spaced frequency points.

    Returns:
        A :class:`FrequencyResponse` with *frequencies* in Hz,
        *magnitude_db* in dB, and *phase_deg* in degrees.

    Raises:
        ValueError: If frequency limits are invalid or *num_points* < 2.

    Example:
        >>> import numpy as np
        >>> resp = analyze_frequency_response(
        ...     np.array([1.0]), np.array([1e-3, 1.0]),
        ...     freq_start=10.0, freq_stop=1e5, num_points=500,
        ... )
        >>> resp.frequencies.shape
        (500,)
        >>> resp.magnitude_db[0] > resp.magnitude_db[-1]  # low-pass roll-off
        True
    """
    if freq_start <= 0:
        raise ValueError(f"freq_start must be positive, got {freq_start}")
    if freq_stop <= freq_start:
        raise ValueError(
            f"freq_stop ({freq_stop}) must be greater than freq_start ({freq_start})"
        )
    if num_points < 2:
        raise ValueError(f"num_points must be >= 2, got {num_points}")

    # scipy.signal.bode expects angular frequencies
    w_start = 2.0 * np.pi * freq_start
    w_stop = 2.0 * np.pi * freq_stop
    w = np.logspace(np.log10(w_start), np.log10(w_stop), num_points)

    sys_tf = compute_transfer_function(num, den)
    w_out, mag_db, phase_deg = signal.bode(sys_tf, w=w)

    frequencies_hz = w_out / (2.0 * np.pi)
    return FrequencyResponse(
        frequencies=frequencies_hz,
        magnitude_db=mag_db,
        phase_deg=phase_deg,
    )


# ======================================================================
# Time-domain analysis
# ======================================================================


def _estimate_settling_time(den: np.ndarray, multiplier: float = 5.0) -> float:
    """Estimate a reasonable simulation end-time from denominator poles.

    Uses the slowest pole's time constant times *multiplier*.

    Args:
        den: Denominator polynomial coefficients.
        multiplier: Safety factor for settling (default 5 time constants).

    Returns:
        Estimated settling time in seconds.
    """
    poles = np.roots(den)
    real_parts = np.abs(np.real(poles))
    # Avoid division by zero for poles at the origin
    real_parts = real_parts[real_parts > 0]
    if real_parts.size == 0:
        return 1.0  # fallback
    slowest_tau = 1.0 / np.min(real_parts)
    return float(slowest_tau * multiplier)


def analyze_time_response(
    num: np.ndarray,
    den: np.ndarray,
    response_type: str = "step",
    t_end: float | None = None,
    num_points: int = 1000,
) -> TimeResponse:
    """Compute the step or impulse response of a system.

    If *t_end* is ``None`` the function automatically estimates an appropriate
    simulation horizon from the system's poles.

    Args:
        num: Numerator polynomial coefficients (descending power).
        den: Denominator polynomial coefficients (descending power).
        response_type: ``"step"`` or ``"impulse"``.
        t_end: End time in seconds.  ``None`` for automatic estimation.
        num_points: Number of uniformly spaced time samples.

    Returns:
        A :class:`TimeResponse` dataclass.

    Raises:
        ValueError: If *response_type* is not ``"step"`` or ``"impulse"``,
            or if *t_end* ≤ 0, or if *num_points* < 2.

    Example:
        >>> import numpy as np
        >>> tr = analyze_time_response(
        ...     np.array([1.0]), np.array([1e-3, 1.0]),
        ...     response_type="step",
        ... )
        >>> tr.response_type
        'step'
        >>> tr.output[-1]  # DC gain = 1 for this system
        1.0
    """
    response_type = response_type.lower().strip()
    if response_type not in ("step", "impulse"):
        raise ValueError(
            f"response_type must be 'step' or 'impulse', got '{response_type}'"
        )
    if t_end is not None and t_end <= 0:
        raise ValueError(f"t_end must be positive, got {t_end}")
    if num_points < 2:
        raise ValueError(f"num_points must be >= 2, got {num_points}")

    if t_end is None:
        t_end = _estimate_settling_time(np.asarray(den, dtype=float))

    t = np.linspace(0.0, t_end, num_points)
    sys_tf = compute_transfer_function(num, den)

    if response_type == "step":
        t_out, y_out = signal.step(sys_tf, T=t)
    else:
        t_out, y_out = signal.impulse(sys_tf, T=t)

    return TimeResponse(time=t_out, output=y_out, response_type=response_type)


# ======================================================================
# Feature extraction
# ======================================================================


def find_cutoff_frequency(freq_response: FrequencyResponse) -> float:
    """Find the −3 dB cutoff frequency from measured frequency response data.

    The cutoff is defined as the first frequency where the magnitude drops to
    −3 dB relative to the DC (lowest-frequency) magnitude.

    Args:
        freq_response: A :class:`FrequencyResponse` instance.

    Returns:
        The −3 dB cutoff frequency in Hz.

    Raises:
        ValueError: If the −3 dB point is not found within the data range
            (i.e., the magnitude never drops 3 dB below DC).

    Example:
        >>> import numpy as np
        >>> resp = analyze_frequency_response(
        ...     np.array([1.0]), np.array([1e-3, 1.0]),
        ... )
        >>> fc = find_cutoff_frequency(resp)
        >>> abs(fc - 159.15) < 1.0  # ≈ 1/(2π·1kΩ·1µF)
        True
    """
    mag = freq_response.magnitude_db
    freqs = freq_response.frequencies

    dc_level = mag[0]
    threshold = dc_level - 3.0

    # Find first index where magnitude falls below threshold
    below = np.where(mag <= threshold)[0]
    if below.size == 0:
        raise ValueError(
            "The −3 dB cutoff frequency was not found within the provided "
            "frequency range.  Try increasing freq_stop."
        )

    idx = below[0]
    if idx == 0:
        return float(freqs[0])

    # Linear interpolation for better accuracy
    f_lo, f_hi = freqs[idx - 1], freqs[idx]
    m_lo, m_hi = mag[idx - 1], mag[idx]
    fraction = (threshold - m_lo) / (m_hi - m_lo)
    # Interpolate in log-frequency space for better accuracy on log sweeps
    fc = f_lo * (f_hi / f_lo) ** fraction
    return float(fc)


def find_resonant_frequency(
    freq_response: FrequencyResponse,
) -> float | None:
    """Find the resonant peak frequency from frequency response data.

    The resonant frequency is the frequency at which the magnitude is
    maximized, provided the peak is above the DC level by at least 0.1 dB
    (to avoid false positives in overdamped systems).

    Args:
        freq_response: A :class:`FrequencyResponse` instance.

    Returns:
        Resonant peak frequency in Hz, or ``None`` if no resonance peak is
        detected.

    Example:
        >>> import numpy as np
        >>> # Underdamped RLC → has a resonance peak
        >>> resp = analyze_frequency_response(
        ...     np.array([1.0]), np.array([1e-8, 1e-4, 1.0]),
        ...     freq_start=100.0, freq_stop=1e5,
        ... )
        >>> fr = find_resonant_frequency(resp)
        >>> fr is not None
        True
    """
    mag = freq_response.magnitude_db
    freqs = freq_response.frequencies

    peak_idx = int(np.argmax(mag))
    dc_level = mag[0]

    # Require the peak to be meaningfully above DC and not at the boundary
    if mag[peak_idx] - dc_level < 0.1:
        return None
    if peak_idx == 0 or peak_idx == len(mag) - 1:
        return None

    # Parabolic interpolation around the peak for sub-bin accuracy
    if peak_idx >= 1 and peak_idx < len(mag) - 1:
        log_f = np.log10(freqs)
        alpha = mag[peak_idx - 1]
        beta = mag[peak_idx]
        gamma = mag[peak_idx + 1]
        denom = 2.0 * (2.0 * beta - alpha - gamma)
        if abs(denom) > 1e-15:
            delta = (alpha - gamma) / denom
            log_f_peak = log_f[peak_idx] + delta * (
                log_f[peak_idx + 1] - log_f[peak_idx]
            )
            return float(10.0**log_f_peak)

    return float(freqs[peak_idx])


# ======================================================================
# Poles, zeros, and stability
# ======================================================================


def compute_poles_zeros(
    num: np.ndarray,
    den: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the poles and zeros of a transfer function.

    Args:
        num: Numerator polynomial coefficients (descending power).
        den: Denominator polynomial coefficients (descending power).

    Returns:
        A tuple ``(poles, zeros)`` of complex numpy arrays.

    Example:
        >>> import numpy as np
        >>> poles, zeros = compute_poles_zeros(
        ...     np.array([1.0]),
        ...     np.array([1e-3, 1.0]),
        ... )
        >>> poles
        array([-1000.])
        >>> zeros.size
        0
    """
    num = np.atleast_1d(np.asarray(num, dtype=float))
    den = np.atleast_1d(np.asarray(den, dtype=float))

    zeros = np.roots(num)
    poles = np.roots(den)

    # Filter out trivial zeros introduced by leading coefficients
    # (e.g., num=[1] has no meaningful roots)
    if num.size <= 1:
        zeros = np.array([], dtype=complex)

    return poles, zeros


def compute_stability(den: np.ndarray) -> bool:
    """Check BIBO (bounded-input bounded-output) stability.

    A continuous-time LTI system is BIBO stable if and only if all poles of
    its transfer function have strictly negative real parts.

    Args:
        den: Denominator polynomial coefficients (descending power).

    Returns:
        ``True`` if the system is BIBO stable, ``False`` otherwise.

    Example:
        >>> import numpy as np
        >>> compute_stability(np.array([1e-3, 1.0]))  # pole at s = −1000
        True
        >>> compute_stability(np.array([1.0, -1.0]))  # pole at s = +1
        False
        >>> compute_stability(np.array([1.0, 0.0]))   # pole at s = 0 (marginal)
        False
    """
    den = np.atleast_1d(np.asarray(den, dtype=float))
    poles = np.roots(den)

    if poles.size == 0:
        return True  # static gain, trivially stable

    return bool(np.all(np.real(poles) < 0))
