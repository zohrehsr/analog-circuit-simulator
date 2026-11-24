"""Analog circuit models: RC, RL, and RLC configurations.

Provides first-order (RC, RL) and second-order (RLC) circuit models with
analytical transfer functions compatible with :mod:`scipy.signal`.

Each circuit class exposes:

* ``transfer_function()`` → ``(num, den)`` coefficient arrays for
  :class:`scipy.signal.TransferFunction`.
* Characteristic properties such as cutoff/natural frequency, time constant,
  quality factor, and damping ratio.

Typical usage::

    >>> from analog_circuit_simulator.circuits import RCCircuit
    >>> rc = RCCircuit(resistance=1e3, capacitance=1e-6)
    >>> rc.cutoff_frequency
    159.1549430918953
    >>> num, den = rc.transfer_function()
"""

from __future__ import annotations

import math

import numpy as np

from .components import Capacitor, Inductor, Resistor


def _validate_positive(name: str, value: float) -> None:
    """Raise :class:`ValueError` if *value* is not positive and finite.

    Args:
        name: Human-readable parameter name for the error message.
        value: The value to validate.

    Raises:
        ValueError: If *value* ≤ 0 or is not finite.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value}")


def _format_eng(value: float, unit: str) -> str:
    """Format *value* with an SI engineering prefix and *unit*.

    Args:
        value: Numeric value to format.
        unit: SI unit string (e.g. ``"Ω"``, ``"F"``).

    Returns:
        Compact engineering string such as ``"4.7 kΩ"`` or ``"100 nF"``.
    """
    prefixes = [
        (1e-12, "p"),
        (1e-9, "n"),
        (1e-6, "µ"),
        (1e-3, "m"),
        (1e0, ""),
        (1e3, "k"),
        (1e6, "M"),
    ]
    for scale, prefix in reversed(prefixes):
        if abs(value) >= scale:
            scaled = value / scale
            return f"{scaled:.4g} {prefix}{unit}"
    return f"{value:.4g} {unit}"


class RCCircuit:
    """First-order RC low-pass filter.

    The voltage transfer function (output across *C*) is:

    .. math::

        H(s) = \\frac{1}{R C s + 1}

    Attributes:
        resistor: :class:`Resistor` instance.
        capacitor: :class:`Capacitor` instance.

    Args:
        resistance: Resistance in Ohms (Ω).
        capacitance: Capacitance in Farads (F).

    Raises:
        ValueError: If any parameter is non-positive or non-finite.

    Example:
        >>> rc = RCCircuit(1e3, 1e-6)
        >>> f"{rc.cutoff_frequency:.2f} Hz"
        '159.15 Hz'
    """

    def __init__(self, resistance: float, capacitance: float) -> None:
        _validate_positive("resistance", resistance)
        _validate_positive("capacitance", capacitance)

        self.resistor = Resistor("R", resistance)
        self.capacitor = Capacitor("C", capacitance)
        self._R = resistance
        self._C = capacitance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def cutoff_frequency(self) -> float:
        """Cutoff (−3 dB) frequency in Hz.

        .. math::

            f_c = \\frac{1}{2\\pi R C}

        Returns:
            Cutoff frequency in Hz.
        """
        return 1.0 / (2.0 * math.pi * self._R * self._C)

    @property
    def cutoff_frequency_rad(self) -> float:
        """Cutoff frequency in rad/s.

        Returns:
            Cutoff angular frequency ω_c = 1/(RC).
        """
        return 1.0 / (self._R * self._C)

    @property
    def time_constant(self) -> float:
        """Time constant τ = R·C in seconds.

        Returns:
            Time constant in seconds.
        """
        return self._R * self._C

    # ------------------------------------------------------------------
    # Transfer function
    # ------------------------------------------------------------------
    def transfer_function(self) -> tuple[np.ndarray, np.ndarray]:
        """Return transfer-function coefficient arrays.

        The continuous-time transfer function is::

            H(s) = 1 / (RC·s + 1)

        Returns:
            Tuple ``(num, den)`` where *num* and *den* are 1-D numpy arrays of
            descending-power polynomial coefficients suitable for
            :class:`scipy.signal.TransferFunction`.

        Example:
            >>> rc = RCCircuit(1e3, 1e-6)
            >>> num, den = rc.transfer_function()
            >>> num
            array([1.])
            >>> den
            array([1.e-03, 1.e+00])
        """
        tau = self._R * self._C
        num = np.array([1.0])
        den = np.array([tau, 1.0])
        return num, den

    def __repr__(self) -> str:
        return (
            f"RCCircuit(R={_format_eng(self._R, 'Ω')}, "
            f"C={_format_eng(self._C, 'F')}, "
            f"fc={_format_eng(self.cutoff_frequency, 'Hz')})"
        )


class RLCircuit:
    """First-order RL high-pass filter.

    The voltage transfer function (output across *L*) is:

    .. math::

        H(s) = \\frac{\\frac{L}{R} s}{\\frac{L}{R} s + 1}

    This is a high-pass filter with the same cutoff form as an RC circuit.

    Args:
        resistance: Resistance in Ohms (Ω).
        inductance: Inductance in Henries (H).

    Raises:
        ValueError: If any parameter is non-positive or non-finite.

    Example:
        >>> rl = RLCircuit(1e3, 100e-3)
        >>> f"{rl.cutoff_frequency:.2f} Hz"
        '1591.55 Hz'
    """

    def __init__(self, resistance: float, inductance: float) -> None:
        _validate_positive("resistance", resistance)
        _validate_positive("inductance", inductance)

        self.resistor = Resistor("R", resistance)
        self.inductor = Inductor("L", inductance)
        self._R = resistance
        self._L = inductance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def cutoff_frequency(self) -> float:
        """Cutoff (−3 dB) frequency in Hz.

        .. math::

            f_c = \\frac{R}{2\\pi L}

        Returns:
            Cutoff frequency in Hz.
        """
        return self._R / (2.0 * math.pi * self._L)

    @property
    def cutoff_frequency_rad(self) -> float:
        """Cutoff frequency in rad/s.

        Returns:
            Angular cutoff frequency ω_c = R/L.
        """
        return self._R / self._L

    @property
    def time_constant(self) -> float:
        """Time constant τ = L/R in seconds.

        Returns:
            Time constant in seconds.
        """
        return self._L / self._R

    # ------------------------------------------------------------------
    # Transfer function
    # ------------------------------------------------------------------
    def transfer_function(self) -> tuple[np.ndarray, np.ndarray]:
        """Return transfer-function coefficient arrays.

        The continuous-time transfer function is::

            H(s) = (L/R)·s / ((L/R)·s + 1)

        Returns:
            Tuple ``(num, den)`` of 1-D numpy arrays (descending power).

        Example:
            >>> rl = RLCircuit(1e3, 100e-3)
            >>> num, den = rl.transfer_function()
            >>> num
            array([0.0001, 0.    ])
            >>> den
            array([1.e-04, 1.e+00])
        """
        tau = self._L / self._R
        num = np.array([tau, 0.0])
        den = np.array([tau, 1.0])
        return num, den

    def __repr__(self) -> str:
        return (
            f"RLCircuit(R={_format_eng(self._R, 'Ω')}, "
            f"L={_format_eng(self._L, 'H')}, "
            f"fc={_format_eng(self.cutoff_frequency, 'Hz')})"
        )


class RLCCircuit:
    """Second-order series RLC circuit.

    The voltage transfer function (output across *C*) is:

    .. math::

        H(s) = \\frac{1}{L C s^2 + R C s + 1}

    Args:
        resistance: Resistance in Ohms (Ω).
        inductance: Inductance in Henries (H).
        capacitance: Capacitance in Farads (F).

    Raises:
        ValueError: If any parameter is non-positive or non-finite.

    Example:
        >>> rlc = RLCCircuit(100.0, 10e-3, 1e-6)
        >>> f"{rlc.natural_frequency:.2f} Hz"
        '1591.55 Hz'
        >>> f"{rlc.quality_factor:.2f}"
        '3.16'
    """

    def __init__(
        self,
        resistance: float,
        inductance: float,
        capacitance: float,
    ) -> None:
        _validate_positive("resistance", resistance)
        _validate_positive("inductance", inductance)
        _validate_positive("capacitance", capacitance)

        self.resistor = Resistor("R", resistance)
        self.inductor = Inductor("L", inductance)
        self.capacitor = Capacitor("C", capacitance)
        self._R = resistance
        self._L = inductance
        self._C = capacitance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def natural_frequency(self) -> float:
        """Undamped natural frequency in Hz.

        .. math::

            f_0 = \\frac{1}{2\\pi\\sqrt{LC}}

        Returns:
            Natural frequency in Hz.
        """
        return 1.0 / (2.0 * math.pi * math.sqrt(self._L * self._C))

    @property
    def natural_frequency_rad(self) -> float:
        """Undamped natural frequency in rad/s.

        Returns:
            ω₀ = 1/√(LC).
        """
        return 1.0 / math.sqrt(self._L * self._C)

    @property
    def quality_factor(self) -> float:
        """Quality factor Q (dimensionless).

        .. math::

            Q = \\frac{1}{R} \\sqrt{\\frac{L}{C}}

        Higher Q means a sharper resonance peak.

        Returns:
            Quality factor.
        """
        return (1.0 / self._R) * math.sqrt(self._L / self._C)

    @property
    def damping_ratio(self) -> float:
        """Damping ratio ζ (dimensionless).

        .. math::

            \\zeta = \\frac{R}{2} \\sqrt{\\frac{C}{L}}

        Returns:
            Damping ratio.  ζ < 1 → underdamped, ζ = 1 → critically damped,
            ζ > 1 → overdamped.
        """
        return (self._R / 2.0) * math.sqrt(self._C / self._L)

    @property
    def bandwidth(self) -> float:
        """3-dB bandwidth in Hz.

        .. math::

            BW = \\frac{f_0}{Q}

        Returns:
            Bandwidth in Hz.
        """
        return self.natural_frequency / self.quality_factor

    @property
    def circuit_type(self) -> str:
        """Classify the circuit based on damping ratio.

        Returns:
            One of ``"underdamped"``, ``"critically_damped"``, or
            ``"overdamped"``.

        Example:
            >>> RLCCircuit(100.0, 10e-3, 1e-6).circuit_type
            'underdamped'
        """
        zeta = self.damping_ratio
        if math.isclose(zeta, 1.0, rel_tol=1e-9):
            return "critically_damped"
        elif zeta < 1.0:
            return "underdamped"
        else:
            return "overdamped"

    # ------------------------------------------------------------------
    # Transfer function
    # ------------------------------------------------------------------
    def transfer_function(self) -> tuple[np.ndarray, np.ndarray]:
        """Return transfer-function coefficient arrays.

        The continuous-time transfer function is::

            H(s) = 1 / (LC·s² + RC·s + 1)

        Returns:
            Tuple ``(num, den)`` of 1-D numpy arrays (descending power).

        Example:
            >>> rlc = RLCCircuit(100.0, 10e-3, 1e-6)
            >>> num, den = rlc.transfer_function()
            >>> num
            array([1.])
            >>> den
            array([1.e-08, 1.e-04, 1.e+00])
        """
        lc = self._L * self._C
        rc = self._R * self._C
        num = np.array([1.0])
        den = np.array([lc, rc, 1.0])
        return num, den

    def __repr__(self) -> str:
        return (
            f"RLCCircuit(R={_format_eng(self._R, 'Ω')}, "
            f"L={_format_eng(self._L, 'H')}, "
            f"C={_format_eng(self._C, 'F')}, "
            f"f0={_format_eng(self.natural_frequency, 'Hz')}, "
            f"Q={self.quality_factor:.3g}, "
            f"ζ={self.damping_ratio:.3g}, "
            f"type={self.circuit_type})"
        )
