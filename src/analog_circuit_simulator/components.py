"""Electrical component models for analog circuit simulation.

Provides dataclass-based models for fundamental passive electrical components
(Resistor, Capacitor, Inductor) with complex impedance calculations over
frequency arrays.

Typical usage::

    >>> import numpy as np
    >>> r = Resistor("R1", 1000.0)
    >>> c = Capacitor("C1", 1e-6)
    >>> freqs = np.logspace(0, 6, 100)
    >>> z_r = r.impedance(freqs)
    >>> z_c = c.impedance(freqs)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Component(abc.ABC):
    """Abstract base class for all electrical components.

    Attributes:
        name: Human-readable identifier for the component (e.g. ``"R1"``).
        value: Component value in SI units (Ω, F, or H).
        unit: SI unit symbol, set automatically by concrete subclasses.
    """

    name: str
    value: float
    unit: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate that the component value is positive and finite."""
        if not np.isfinite(self.value):
            raise ValueError(
                f"{self.__class__.__name__} '{self.name}': "
                f"value must be finite, got {self.value}"
            )
        if self.value <= 0:
            raise ValueError(
                f"{self.__class__.__name__} '{self.name}': "
                f"value must be positive, got {self.value}"
            )

    @abc.abstractmethod
    def impedance(self, frequency: np.ndarray) -> np.ndarray:
        """Compute complex impedance as a function of frequency.

        Args:
            frequency: Array of frequencies in Hz.  All elements must be
                non-negative.

        Returns:
            Complex-valued numpy array of impedances in Ohms, with the same
            shape as *frequency*.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_eng(value: float) -> str:
        """Return *value* with an SI engineering prefix (p … M).

        Args:
            value: The numeric value to format.

        Returns:
            A compact string such as ``"4.7 k"`` or ``"100 n"``.

        Example:
            >>> Component._format_eng(4700.0)
            '4.7 k'
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
                # Remove trailing zeros
                formatted = f"{scaled:.4g}"
                return f"{formatted} {prefix}".rstrip()
        return f"{value:.4g}"


@dataclass
class Resistor(Component):
    """Ideal resistor model.

    The impedance of a resistor is purely real and frequency-independent:

    .. math::

        Z(f) = R

    Attributes:
        name: Component identifier.
        value: Resistance in Ohms (Ω).  Must be positive.

    Example:
        >>> import numpy as np
        >>> r = Resistor("R1", 1000.0)
        >>> r.impedance(np.array([100.0, 1000.0]))
        array([1000.+0.j, 1000.+0.j])
    """

    def __post_init__(self) -> None:
        self.unit = "Ω"
        super().__post_init__()

    def impedance(self, frequency: np.ndarray) -> np.ndarray:
        """Compute resistor impedance (constant, real-valued).

        Args:
            frequency: Array of frequencies in Hz (unused for resistors but
                accepted for interface consistency).

        Returns:
            Complex array filled with the resistance value.

        Raises:
            ValueError: If any frequency is negative.

        Example:
            >>> import numpy as np
            >>> Resistor("R1", 470.0).impedance(np.array([1e3]))
            array([470.+0.j])
        """
        freq = np.asarray(frequency, dtype=float)
        if np.any(freq < 0):
            raise ValueError("Frequencies must be non-negative.")
        return np.full_like(freq, self.value, dtype=complex)

    def __repr__(self) -> str:
        return f"Resistor('{self.name}', {self._format_eng(self.value)}{self.unit})"


@dataclass
class Capacitor(Component):
    """Ideal capacitor model.

    The impedance of a capacitor is:

    .. math::

        Z(f) = \\frac{1}{j \\cdot 2\\pi f \\cdot C}

    At *f = 0* the impedance is infinite (open circuit).

    Attributes:
        name: Component identifier.
        value: Capacitance in Farads (F).  Must be positive.

    Example:
        >>> import numpy as np
        >>> c = Capacitor("C1", 1e-6)
        >>> z = c.impedance(np.array([1000.0]))
    """

    def __post_init__(self) -> None:
        self.unit = "F"
        super().__post_init__()

    def impedance(self, frequency: np.ndarray) -> np.ndarray:
        """Compute capacitor impedance.

        Args:
            frequency: Array of frequencies in Hz.  Must be non-negative.

        Returns:
            Complex array of impedances.  Elements corresponding to *f = 0*
            are set to ``np.inf + 0j`` (open circuit).

        Raises:
            ValueError: If any frequency is negative.

        Example:
            >>> import numpy as np
            >>> Capacitor("C1", 1e-6).impedance(np.array([0.0, 1000.0]))
            array([inf+0.j, 0.-159.15494309j])
        """
        freq = np.asarray(frequency, dtype=float)
        if np.any(freq < 0):
            raise ValueError("Frequencies must be non-negative.")

        result = np.empty_like(freq, dtype=complex)
        zero_mask = freq == 0.0
        nonzero_mask = ~zero_mask

        result[zero_mask] = np.inf + 0j
        omega = 2.0 * np.pi * freq[nonzero_mask]
        result[nonzero_mask] = 1.0 / (1j * omega * self.value)
        return result

    def __repr__(self) -> str:
        return f"Capacitor('{self.name}', {self._format_eng(self.value)}{self.unit})"


@dataclass
class Inductor(Component):
    """Ideal inductor model.

    The impedance of an inductor is:

    .. math::

        Z(f) = j \\cdot 2\\pi f \\cdot L

    At *f = 0* the impedance is zero (short circuit).

    Attributes:
        name: Component identifier.
        value: Inductance in Henries (H).  Must be positive.

    Example:
        >>> import numpy as np
        >>> l = Inductor("L1", 10e-3)
        >>> z = l.impedance(np.array([1000.0]))
    """

    def __post_init__(self) -> None:
        self.unit = "H"
        super().__post_init__()

    def impedance(self, frequency: np.ndarray) -> np.ndarray:
        """Compute inductor impedance.

        Args:
            frequency: Array of frequencies in Hz.  Must be non-negative.

        Returns:
            Complex array of impedances.  At *f = 0* the impedance is ``0+0j``
            (short circuit).

        Raises:
            ValueError: If any frequency is negative.

        Example:
            >>> import numpy as np
            >>> Inductor("L1", 10e-3).impedance(np.array([1000.0]))
            array([0.+62.83185307j])
        """
        freq = np.asarray(frequency, dtype=float)
        if np.any(freq < 0):
            raise ValueError("Frequencies must be non-negative.")

        omega = 2.0 * np.pi * freq
        return 1j * omega * self.value

    def __repr__(self) -> str:
        return f"Inductor('{self.name}', {self._format_eng(self.value)}{self.unit})"
