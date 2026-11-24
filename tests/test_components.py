"""Tests for the components module."""

from __future__ import annotations

import numpy as np
import pytest

from analog_circuit_simulator.components import Capacitor, Inductor, Resistor

# ---------------------------------------------------------------------------
# Resistor
# ---------------------------------------------------------------------------


class TestResistor:
    """Test suite for the Resistor component."""

    def test_impedance_is_constant(self) -> None:
        """Resistor impedance must equal R at all frequencies."""
        r = Resistor("R1", 1_000.0)
        freqs = np.logspace(0, 6, 50)
        z = r.impedance(freqs)
        np.testing.assert_allclose(np.real(z), 1_000.0)

    def test_impedance_is_real(self) -> None:
        """Resistor impedance must have zero imaginary part."""
        r = Resistor("R1", 470.0)
        freqs = np.logspace(1, 5, 20)
        np.testing.assert_array_equal(np.imag(r.impedance(freqs)), 0.0)

    def test_impedance_shape_matches_input(self) -> None:
        """Output shape must match input shape."""
        r = Resistor("R1", 100.0)
        freqs = np.array([10.0, 100.0, 1_000.0])
        assert r.impedance(freqs).shape == freqs.shape

    def test_negative_frequency_raises(self) -> None:
        """Negative frequencies must raise ValueError."""
        r = Resistor("R1", 100.0)
        with pytest.raises(ValueError, match="non-negative"):
            r.impedance(np.array([-1.0, 100.0]))

    def test_negative_value_raises(self) -> None:
        """Non-positive resistance must raise ValueError."""
        with pytest.raises(ValueError):
            Resistor("R1", -100.0)

    def test_zero_value_raises(self) -> None:
        """Zero resistance must raise ValueError."""
        with pytest.raises(ValueError):
            Resistor("R1", 0.0)

    def test_repr(self) -> None:
        """repr must include the component name and value."""
        r = Resistor("R1", 4_700.0)
        assert "R1" in repr(r)
        assert "Ω" in repr(r)

    @pytest.mark.parametrize("r_val", [1.0, 100.0, 1e6])
    def test_various_resistances(self, r_val: float) -> None:
        """Impedance must equal the resistance value."""
        r = Resistor("R", r_val)
        z = r.impedance(np.array([1_000.0]))
        np.testing.assert_allclose(np.real(z), r_val)


# ---------------------------------------------------------------------------
# Capacitor
# ---------------------------------------------------------------------------


class TestCapacitor:
    """Test suite for the Capacitor component."""

    def test_impedance_decreases_with_frequency(self) -> None:
        """Capacitor impedance magnitude must decrease as frequency increases."""
        c = Capacitor("C1", 1e-6)
        freqs = np.logspace(1, 6, 100)
        mag = np.abs(c.impedance(freqs))
        assert np.all(np.diff(mag) < 0)

    def test_impedance_at_zero_frequency(self) -> None:
        """Impedance at DC (f=0) must be infinite (open circuit)."""
        c = Capacitor("C1", 1e-6)
        z = c.impedance(np.array([0.0]))
        assert np.isinf(np.real(z[0])) or np.isinf(np.abs(z[0]))

    def test_impedance_formula(self) -> None:
        """Verify Z = 1/(j·2π·f·C) at a specific frequency."""
        c = Capacitor("C1", 1e-6)
        f = 1_000.0
        expected = 1.0 / (1j * 2 * np.pi * f * 1e-6)
        z = c.impedance(np.array([f]))
        np.testing.assert_allclose(z[0], expected, rtol=1e-10)

    def test_imaginary_part_is_negative(self) -> None:
        """Capacitor impedance imaginary part must be negative."""
        c = Capacitor("C1", 1e-9)
        freqs = np.logspace(3, 9, 50)
        assert np.all(np.imag(c.impedance(freqs)) < 0)

    def test_negative_value_raises(self) -> None:
        with pytest.raises(ValueError):
            Capacitor("C1", -1e-6)

    def test_negative_frequency_raises(self) -> None:
        c = Capacitor("C1", 1e-6)
        with pytest.raises(ValueError):
            c.impedance(np.array([-100.0]))

    def test_repr(self) -> None:
        c = Capacitor("C1", 1e-9)
        assert "C1" in repr(c)
        assert "F" in repr(c)


# ---------------------------------------------------------------------------
# Inductor
# ---------------------------------------------------------------------------


class TestInductor:
    """Test suite for the Inductor component."""

    def test_impedance_increases_with_frequency(self) -> None:
        """Inductor impedance magnitude must increase with frequency."""
        l = Inductor("L1", 10e-3)
        freqs = np.logspace(1, 6, 100)
        mag = np.abs(l.impedance(freqs))
        assert np.all(np.diff(mag) > 0)

    def test_impedance_at_zero_frequency(self) -> None:
        """Impedance at DC (f=0) must be zero (short circuit)."""
        l = Inductor("L1", 10e-3)
        z = l.impedance(np.array([0.0]))
        np.testing.assert_allclose(np.abs(z[0]), 0.0)

    def test_impedance_formula(self) -> None:
        """Verify Z = j·2π·f·L at a specific frequency."""
        l = Inductor("L1", 10e-3)
        f = 1_000.0
        expected = 1j * 2 * np.pi * f * 10e-3
        z = l.impedance(np.array([f]))
        np.testing.assert_allclose(z[0], expected, rtol=1e-10)

    def test_imaginary_part_is_positive(self) -> None:
        """Inductor impedance imaginary part must be positive."""
        l = Inductor("L1", 1e-3)
        freqs = np.logspace(1, 6, 50)
        assert np.all(np.imag(l.impedance(freqs)) > 0)

    def test_negative_value_raises(self) -> None:
        with pytest.raises(ValueError):
            Inductor("L1", -10e-3)

    def test_repr(self) -> None:
        l = Inductor("L1", 10e-3)
        assert "L1" in repr(l)
        assert "H" in repr(l)
