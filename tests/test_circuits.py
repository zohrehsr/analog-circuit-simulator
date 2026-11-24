"""Tests for the circuits module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from analog_circuit_simulator.circuits import RCCircuit, RLCCircuit, RLCircuit

# ---------------------------------------------------------------------------
# RCCircuit
# ---------------------------------------------------------------------------


class TestRCCircuit:
    """Test suite for RCCircuit."""

    def test_cutoff_frequency_known_value(self) -> None:
        """RC cutoff frequency: R=1kΩ, C=1µF → fc ≈ 159.15 Hz."""
        rc = RCCircuit(1e3, 1e-6)
        expected = 1.0 / (2 * math.pi * 1e3 * 1e-6)  # ≈ 159.155
        np.testing.assert_allclose(rc.cutoff_frequency, expected, rtol=1e-10)

    def test_time_constant(self) -> None:
        """Time constant τ = R·C."""
        rc = RCCircuit(1e3, 1e-6)
        np.testing.assert_allclose(rc.time_constant, 1e-3, rtol=1e-12)

    def test_cutoff_frequency_rad(self) -> None:
        """Angular cutoff: ωc = 1/(RC)."""
        rc = RCCircuit(1e3, 1e-6)
        np.testing.assert_allclose(rc.cutoff_frequency_rad, 1e3, rtol=1e-12)

    def test_transfer_function_coefficients(self) -> None:
        """H(s) = 1/(RC·s + 1) → num=[1], den=[RC, 1]."""
        rc = RCCircuit(1e3, 1e-6)
        num, den = rc.transfer_function()
        np.testing.assert_allclose(num, [1.0])
        np.testing.assert_allclose(den, [1e-3, 1.0])

    def test_negative_resistance_raises(self) -> None:
        with pytest.raises(ValueError):
            RCCircuit(-1e3, 1e-6)

    def test_zero_capacitance_raises(self) -> None:
        with pytest.raises(ValueError):
            RCCircuit(1e3, 0.0)

    def test_repr_contains_components(self) -> None:
        rc = RCCircuit(1e3, 1e-6)
        r = repr(rc)
        assert "RC" in r or "Ω" in r

    @pytest.mark.parametrize(
        "R, C, expected_fc",
        [
            (1e3, 1e-6, 1 / (2 * math.pi * 1e3 * 1e-6)),
            (10e3, 100e-9, 1 / (2 * math.pi * 10e3 * 100e-9)),
            (100.0, 1e-9, 1 / (2 * math.pi * 100 * 1e-9)),
        ],
    )
    def test_cutoff_parametrized(self, R: float, C: float, expected_fc: float) -> None:
        rc = RCCircuit(R, C)
        np.testing.assert_allclose(rc.cutoff_frequency, expected_fc, rtol=1e-10)


# ---------------------------------------------------------------------------
# RLCircuit
# ---------------------------------------------------------------------------


class TestRLCircuit:
    """Test suite for RLCircuit."""

    def test_cutoff_frequency_known_value(self) -> None:
        """RL cutoff: R=1kΩ, L=100mH → fc ≈ 1591.55 Hz."""
        rl = RLCircuit(1e3, 100e-3)
        expected = 1e3 / (2 * math.pi * 100e-3)
        np.testing.assert_allclose(rl.cutoff_frequency, expected, rtol=1e-10)

    def test_time_constant(self) -> None:
        """Time constant τ = L/R."""
        rl = RLCircuit(1e3, 100e-3)
        np.testing.assert_allclose(rl.time_constant, 1e-4, rtol=1e-12)

    def test_transfer_function_coefficients(self) -> None:
        """H(s) = (L/R)·s / ((L/R)·s + 1) → num=[L/R, 0], den=[L/R, 1]."""
        rl = RLCircuit(1e3, 100e-3)
        num, den = rl.transfer_function()
        tau = 100e-3 / 1e3  # 1e-4
        np.testing.assert_allclose(num, [tau, 0.0])
        np.testing.assert_allclose(den, [tau, 1.0])

    def test_negative_inductance_raises(self) -> None:
        with pytest.raises(ValueError):
            RLCircuit(1e3, -1e-3)


# ---------------------------------------------------------------------------
# RLCCircuit
# ---------------------------------------------------------------------------


class TestRLCCircuit:
    """Test suite for RLCCircuit."""

    def test_natural_frequency(self) -> None:
        """f0 = 1/(2π√(LC)) for R=100, L=10mH, C=1µF → ≈ 1591.55 Hz."""
        rlc = RLCCircuit(100.0, 10e-3, 1e-6)
        expected = 1.0 / (2 * math.pi * math.sqrt(10e-3 * 1e-6))
        np.testing.assert_allclose(rlc.natural_frequency, expected, rtol=1e-10)

    def test_quality_factor(self) -> None:
        """Q = (1/R)·√(L/C) for R=100, L=10mH, C=1µF."""
        rlc = RLCCircuit(100.0, 10e-3, 1e-6)
        expected = (1 / 100.0) * math.sqrt(10e-3 / 1e-6)
        np.testing.assert_allclose(rlc.quality_factor, expected, rtol=1e-10)

    def test_damping_ratio(self) -> None:
        """ζ = (R/2)·√(C/L)."""
        rlc = RLCCircuit(100.0, 10e-3, 1e-6)
        expected = (100.0 / 2.0) * math.sqrt(1e-6 / 10e-3)
        np.testing.assert_allclose(rlc.damping_ratio, expected, rtol=1e-10)

    def test_bandwidth(self) -> None:
        """BW = f0/Q."""
        rlc = RLCCircuit(100.0, 10e-3, 1e-6)
        np.testing.assert_allclose(
            rlc.bandwidth,
            rlc.natural_frequency / rlc.quality_factor,
            rtol=1e-10,
        )

    def test_underdamped_classification(self) -> None:
        """High Q → underdamped (ζ < 1)."""
        rlc = RLCCircuit(100.0, 10e-3, 1e-6)
        assert rlc.damping_ratio < 1.0
        assert rlc.circuit_type == "underdamped"

    def test_overdamped_classification(self) -> None:
        """Low Q → overdamped (ζ > 1)."""
        rlc = RLCCircuit(10_000.0, 10e-3, 1e-6)
        assert rlc.damping_ratio > 1.0
        assert rlc.circuit_type == "overdamped"

    def test_critically_damped_classification(self) -> None:
        """ζ ≈ 1 → critically damped."""
        L = 10e-3
        C = 1e-6
        # R_crit = 2·√(L/C)
        R_crit = 2.0 * math.sqrt(L / C)
        rlc = RLCCircuit(R_crit, L, C)
        assert rlc.circuit_type == "critically_damped"

    def test_transfer_function_coefficients(self) -> None:
        """H(s) = 1/(LC·s² + RC·s + 1)."""
        rlc = RLCCircuit(100.0, 10e-3, 1e-6)
        num, den = rlc.transfer_function()
        np.testing.assert_allclose(num, [1.0])
        np.testing.assert_allclose(den, [10e-3 * 1e-6, 100.0 * 1e-6, 1.0], rtol=1e-10)

    def test_negative_resistance_raises(self) -> None:
        with pytest.raises(ValueError):
            RLCCircuit(-1.0, 10e-3, 1e-6)

    def test_nonfinite_inductance_raises(self) -> None:
        with pytest.raises(ValueError):
            RLCCircuit(100.0, float("inf"), 1e-6)

    def test_quality_damping_reciprocal(self) -> None:
        """ζ = 1/(2Q) must hold for all series RLC circuits."""
        rlc = RLCCircuit(200.0, 50e-3, 10e-9)
        np.testing.assert_allclose(
            rlc.damping_ratio, 1.0 / (2.0 * rlc.quality_factor), rtol=1e-9
        )
