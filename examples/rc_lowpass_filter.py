"""Example: RC Low-Pass Filter Analysis
=======================================

Demonstrates complete analysis of a first-order RC low-pass filter:
  - Computing cutoff frequency and time constant
  - Bode plot (magnitude and phase)
  - Step and impulse responses
  - Parameter sweep: comparing multiple R values
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from analog_circuit_simulator.analysis import (
    analyze_frequency_response,
    analyze_time_response,
    find_cutoff_frequency,
)
from analog_circuit_simulator.circuits import RCCircuit
from analog_circuit_simulator.plotting import plot_bode, plot_comparison, plot_time_response

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("=" * 60)
    print("  RC Low-Pass Filter Analysis")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Create circuit
    # -----------------------------------------------------------------------
    R = 1_000.0   # 1 kΩ
    C = 1e-6      # 1 µF
    rc = RCCircuit(R, C)

    print(f"\nCircuit: {rc}")
    print(f"  Cutoff frequency  : {rc.cutoff_frequency:.4f} Hz")
    print(f"  Time constant (τ) : {rc.time_constant * 1e3:.4f} ms")
    print(f"  Cutoff (rad/s)    : {rc.cutoff_frequency_rad:.4f} rad/s")

    # -----------------------------------------------------------------------
    # 2. Frequency response & Bode plot
    # -----------------------------------------------------------------------
    num, den = rc.transfer_function()
    freq_resp = analyze_frequency_response(num, den, freq_start=1.0, freq_stop=1e5, num_points=1000)
    fc_detected = find_cutoff_frequency(freq_resp)

    print(f"\n  Detected cutoff (from Bode data): {fc_detected:.4f} Hz")
    print(f"  Error vs. analytical            : {abs(fc_detected - rc.cutoff_frequency) / rc.cutoff_frequency * 100:.4f} %")

    bode_path = str(OUTPUT_DIR / "rc_bode.png")
    plot_bode(
        freq_resp.frequencies,
        freq_resp.magnitude_db,
        freq_resp.phase_deg,
        title="RC Low-Pass Filter — Bode Plot (R=1 kΩ, C=1 µF)",
        cutoff_freq=rc.cutoff_frequency,
        save_path=bode_path,
        show=False,
    )
    print(f"\n  Bode plot saved → {bode_path}")

    # -----------------------------------------------------------------------
    # 3. Step response
    # -----------------------------------------------------------------------
    step_resp = analyze_time_response(num, den, response_type="step")
    step_path = str(OUTPUT_DIR / "rc_step_response.png")
    plot_time_response(
        step_resp.time,
        step_resp.output,
        response_type="step",
        title="RC Low-Pass Filter — Step Response (R=1 kΩ, C=1 µF)",
        save_path=step_path,
        show=False,
    )
    print(f"  Step response saved → {step_path}")

    # -----------------------------------------------------------------------
    # 4. Impulse response
    # -----------------------------------------------------------------------
    impulse_resp = analyze_time_response(num, den, response_type="impulse")
    impulse_path = str(OUTPUT_DIR / "rc_impulse_response.png")
    plot_time_response(
        impulse_resp.time,
        impulse_resp.output,
        response_type="impulse",
        title="RC Low-Pass Filter — Impulse Response (R=1 kΩ, C=1 µF)",
        save_path=impulse_path,
        show=False,
    )
    print(f"  Impulse response saved → {impulse_path}")

    # -----------------------------------------------------------------------
    # 5. Parameter sweep: compare R = 1 kΩ, 10 kΩ, 100 kΩ
    # -----------------------------------------------------------------------
    print("\n  Parameter sweep: varying R value")
    resistances = [1e3, 10e3, 100e3]
    comparison_data = []

    for r_val in resistances:
        rc_sweep = RCCircuit(r_val, C)
        n, d = rc_sweep.transfer_function()
        fr = analyze_frequency_response(n, d, freq_start=0.1, freq_stop=1e6)
        comparison_data.append(
            {
                "frequencies": fr.frequencies,
                "magnitude_db": fr.magnitude_db,
                "label": f"R = {r_val/1e3:.0f} kΩ  (fc = {rc_sweep.cutoff_frequency:.1f} Hz)",
            }
        )
        print(f"    R = {r_val/1e3:.0f} kΩ → fc = {rc_sweep.cutoff_frequency:.2f} Hz")

    comparison_path = str(OUTPUT_DIR / "rc_parameter_sweep.png")
    plot_comparison(
        comparison_data,
        title="RC Low-Pass Filter — Parameter Sweep (C = 1 µF)",
        save_path=comparison_path,
        show=False,
    )
    print(f"\n  Comparison plot saved → {comparison_path}")
    print("\nDone! ✓")


if __name__ == "__main__":
    main()
