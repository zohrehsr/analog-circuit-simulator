"""Tests for the analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from analog_circuit_simulator.analysis import (
    FrequencyResponse,
    TimeResponse,
    analyze_frequency_response,
    analyze_time_response,
    compute_poles_zeros,
    compute_stability,
    compute_transfer_function,
    find_cutoff_frequency,
    find_resonant_frequency,
)
from analog_circuit_simulator.circuits import RCCircuit, RLCCircuit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rc_tf(R: float = 1e3, C: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    return RCCircuit(R, C).transfer_function()


def _rlc_tf(
    R: float = 100.0, L: float = 10e-3, C: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    return RLCCircuit(R, L, C).transfer_function()


# ---------------------------------------------------------------------------
# compute_transfer_function
# ---------------------------------------------------------------------------


class TestComputeTransferFunction:
    def test_returns_transfer_function_object(self) -> None:
        from scipy.signal import TransferFunction

        tf = compute_transfer_function(np.array([1.0]), np.array([1e-3, 1.0]))
        assert isinstance(tf, TransferFunction)

    def test_empty_num_raises(self) -> None:
        with pytest.raises(ValueError, match="Numerator"):
            compute_transfer_function(np.array([]), np.array([1.0]))

    def test_empty_den_raises(self) -> None:
        with pytest.raises(ValueError, match="Denominator"):
            compute_transfer_function(np.array([1.0]), np.array([]))


# ---------------------------------------------------------------------------
# analyze_frequency_response
# ---------------------------------------------------------------------------


class TestAnalyzeFrequencyResponse:
    def test_output_shape(self) -> None:
        num, den = _rc_tf()
        resp = analyze_frequency_response(num, den, num_points=200)
        assert resp.frequencies.shape == (200,)
        assert resp.magnitude_db.shape == (200,)
        assert resp.phase_deg.shape == (200,)

    def test_rc_low_pass_rolls_off(self) -> None:
        """Magnitude must decrease from low to high frequency for RC LPF."""
        num, den = _rc_tf()
        resp = analyze_frequency_response(num, den, freq_start=1.0, freq_stop=1e6)
        assert resp.magnitude_db[0] > resp.magnitude_db[-1]

    def test_dc_gain_is_zero_db_for_rc(self) -> None:
        """DC gain (at freq_start → 0) should approach 0 dB for RC LPF."""
        num, den = _rc_tf()
        resp = analyze_frequency_response(
            num, den, freq_start=0.01, freq_stop=1e6, num_points=1000
        )
        np.testing.assert_allclose(resp.magnitude_db[0], 0.0, atol=0.1)

    def test_invalid_freq_start_raises(self) -> None:
        num, den = _rc_tf()
        with pytest.raises(ValueError, match="freq_start"):
            analyze_frequency_response(num, den, freq_start=-1.0)

    def test_freq_stop_less_than_start_raises(self) -> None:
        num, den = _rc_tf()
        with pytest.raises(ValueError):
            analyze_frequency_response(num, den, freq_start=1e5, freq_stop=1e3)

    def test_num_points_too_small_raises(self) -> None:
        num, den = _rc_tf()
        with pytest.raises(ValueError, match="num_points"):
            analyze_frequency_response(num, den, num_points=1)

    def test_returns_dataclass(self) -> None:
        num, den = _rc_tf()
        resp = analyze_frequency_response(num, den)
        assert isinstance(resp, FrequencyResponse)


# ---------------------------------------------------------------------------
# analyze_time_response
# ---------------------------------------------------------------------------


class TestAnalyzeTimeResponse:
    def test_step_final_value_unity(self) -> None:
        """Step response of RC unity-gain system must settle to 1 (within 1% at 5*tau)."""
        num, den = _rc_tf()
        tr = analyze_time_response(num, den, response_type="step")
        np.testing.assert_allclose(tr.output[-1], 1.0, atol=1e-2)

    def test_step_starts_at_zero(self) -> None:
        """Step response must start at (or near) zero."""
        num, den = _rc_tf()
        tr = analyze_time_response(num, den, response_type="step")
        np.testing.assert_allclose(tr.output[0], 0.0, atol=1e-6)

    def test_impulse_returns_dataclass(self) -> None:
        num, den = _rc_tf()
        tr = analyze_time_response(num, den, response_type="impulse")
        assert isinstance(tr, TimeResponse)
        assert tr.response_type == "impulse"

    def test_impulse_decays_to_zero(self) -> None:
        """Impulse response of a stable system must decay towards zero (< 1% of peak)."""
        num, den = _rc_tf()
        tr = analyze_time_response(num, den, response_type="impulse")
        peak = float(np.max(np.abs(tr.output)))
        np.testing.assert_allclose(tr.output[-1], 0.0, atol=0.01 * peak)

    def test_invalid_response_type_raises(self) -> None:
        num, den = _rc_tf()
        with pytest.raises(ValueError, match="response_type"):
            analyze_time_response(num, den, response_type="ramp")

    def test_negative_t_end_raises(self) -> None:
        num, den = _rc_tf()
        with pytest.raises(ValueError, match="t_end"):
            analyze_time_response(num, den, t_end=-1.0)

    def test_custom_t_end(self) -> None:
        """Custom t_end must be respected."""
        num, den = _rc_tf()
        tr = analyze_time_response(num, den, t_end=0.01, num_points=100)
        np.testing.assert_allclose(tr.time[-1], 0.01, rtol=1e-6)

    def test_num_points_respected(self) -> None:
        num, den = _rc_tf()
        tr = analyze_time_response(num, den, num_points=250)
        assert tr.time.shape == (250,)


# ---------------------------------------------------------------------------
# find_cutoff_frequency
# ---------------------------------------------------------------------------


class TestFindCutoffFrequency:
    @pytest.mark.parametrize(
        "R, C",
        [(1e3, 1e-6), (10e3, 100e-9), (1e3, 10e-9)],
    )
    def test_cutoff_accuracy(self, R: float, C: float) -> None:
        """Detected cutoff must be within 1 % of the analytical value."""
        rc = RCCircuit(R, C)
        num, den = rc.transfer_function()
        resp = analyze_frequency_response(num, den, freq_start=0.01, freq_stop=1e8)
        fc_found = find_cutoff_frequency(resp)
        fc_expected = rc.cutoff_frequency
        np.testing.assert_allclose(fc_found, fc_expected, rtol=0.01)


# ---------------------------------------------------------------------------
# find_resonant_frequency
# ---------------------------------------------------------------------------


class TestFindResonantFrequency:
    def test_underdamped_has_resonance(self) -> None:
        """Underdamped RLC must have a detectable resonance peak."""
        num, den = _rlc_tf(R=10.0, L=10e-3, C=1e-6)
        resp = analyze_frequency_response(num, den, freq_start=100.0, freq_stop=1e5)
        fr = find_resonant_frequency(resp)
        assert fr is not None

    def test_overdamped_no_resonance(self) -> None:
        """Overdamped RLC must NOT have a resonance peak."""
        num, den = _rlc_tf(R=10_000.0, L=10e-3, C=1e-6)
        resp = analyze_frequency_response(num, den)
        fr = find_resonant_frequency(resp)
        assert fr is None

    def test_rc_no_resonance(self) -> None:
        """Simple RC filter must NOT have a resonance peak."""
        num, den = _rc_tf()
        resp = analyze_frequency_response(num, den)
        fr = find_resonant_frequency(resp)
        assert fr is None


# ---------------------------------------------------------------------------
# compute_poles_zeros
# ---------------------------------------------------------------------------


class TestComputePolesZeros:
    def test_rc_has_one_pole_no_zeros(self) -> None:
        """RC circuit: 1 pole at s = -1/(RC), 0 zeros."""
        num, den = _rc_tf(R=1e3, C=1e-6)
        poles, zeros = compute_poles_zeros(num, den)
        assert poles.size == 1
        assert zeros.size == 0
        np.testing.assert_allclose(np.real(poles[0]), -1e3, rtol=1e-6)

    def test_rlc_has_two_poles(self) -> None:
        """RLC circuit: 2 poles."""
        num, den = _rlc_tf()
        poles, zeros = compute_poles_zeros(num, den)
        assert poles.size == 2

    def test_poles_shape(self) -> None:
        num, den = _rc_tf()
        poles, zeros = compute_poles_zeros(num, den)
        assert poles.ndim == 1


# ---------------------------------------------------------------------------
# compute_stability
# ---------------------------------------------------------------------------


class TestComputeStability:
    def test_rc_is_stable(self) -> None:
        """RC circuit is BIBO stable."""
        _, den = _rc_tf()
        assert compute_stability(den) is True

    def test_rlc_stable(self) -> None:
        _, den = _rlc_tf()
        assert compute_stability(den) is True

    def test_unstable_pole_detected(self) -> None:
        """Positive real pole → unstable."""
        # H(s) = 1 / (s - 1)  → pole at +1
        den = np.array([1.0, -1.0])
        assert compute_stability(den) is False

    def test_marginal_pole_at_origin(self) -> None:
        """Pole at s=0 → not BIBO stable."""
        den = np.array([1.0, 0.0])
        assert compute_stability(den) is False

    def test_static_gain_is_stable(self) -> None:
        """H(s) = k (constant) has no poles → stable."""
        den = np.array([2.0])
        assert compute_stability(den) is True


# ---------------------------------------------------------------------------
# Data export tests
# ---------------------------------------------------------------------------


class TestDataExport:
    def test_frequency_response_to_csv(self, tmp_path) -> None:
        num, den = _rc_tf()
        resp = analyze_frequency_response(num, den, num_points=10)
        csv_path = tmp_path / "freq_resp.csv"
        resp.to_csv(csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "frequency_hz,magnitude_db,phase_deg" in content
        lines = content.strip().splitlines()
        assert len(lines) == 11  # 1 header + 10 data rows

    def test_time_response_to_csv(self, tmp_path) -> None:
        num, den = _rc_tf()
        tr = analyze_time_response(num, den, response_type="step", num_points=20)
        csv_path = tmp_path / "time_resp.csv"
        tr.to_csv(csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "time_s,output_step" in content
        lines = content.strip().splitlines()
        assert len(lines) == 21

    def test_frequency_response_to_dict(self) -> None:
        num, den = _rc_tf()
        resp = analyze_frequency_response(num, den, num_points=10)
        d = resp.to_dict()
        assert "frequency" in d and "magnitude_db" in d and "phase_deg" in d
        assert len(d["frequency"]) == 10

    def test_time_response_to_dict(self) -> None:
        num, den = _rc_tf()
        tr = analyze_time_response(num, den, response_type="impulse", num_points=15)
        d = tr.to_dict()
        assert d["response_type"] == "impulse"
        assert len(d["time"]) == 15
