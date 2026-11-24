"""Example: RLC Bandpass / Low-Pass Filter Analysis
===================================================

Demonstrates complete analysis of a second-order series RLC circuit:
  - Natural frequency, Q factor, damping ratio, bandwidth
  - Bode plot with resonance marker
  - Pole-zero map
  - Step responses: underdamped, critically damped, overdamped comparison
  - Q factor sweep
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from analog_circuit_simulator.analysis import (
    analyze_frequency_response,
    analyze_time_response,
    compute_poles_zeros,
    find_resonant_frequency,
)
from analog_circuit_simulator.circuits import RLCCircuit
from analog_circuit_simulator.plotting import (
    plot_bode,
    plot_comparison,
    plot_pole_zero,
    plot_time_response,
)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("=" * 60)
    print("  RLC Series Circuit Analysis")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Create underdamped RLC circuit
    # -----------------------------------------------------------------------
    R, L, C = 100.0, 10e-3, 1e-6
    rlc = RLCCircuit(R, L, C)

    print(f"\nCircuit: {rlc}")
    print(f"  Natural frequency  : {rlc.natural_frequency:.4f} Hz")
    print(f"  Quality factor (Q) : {rlc.quality_factor:.4f}")
    print(f"  Damping ratio (ζ)  : {rlc.damping_ratio:.6f}")
    print(f"  Bandwidth          : {rlc.bandwidth:.4f} Hz")
    print(f"  Circuit type       : {rlc.circuit_type}")

    # -----------------------------------------------------------------------
    # 2. Bode plot with resonance detection
    # -----------------------------------------------------------------------
    num, den = rlc.transfer_function()
    freq_resp = analyze_frequency_response(num, den, freq_start=10.0, freq_stop=1e5, num_points=2000)
    f_resonant = find_resonant_frequency(freq_resp)

    print(f"\n  Resonant frequency (detected) : {f_resonant:.4f} Hz" if f_resonant else "\n  No resonance peak detected.")

    bode_path = str(OUTPUT_DIR / "rlc_bode.png")
    plot_bode(
        freq_resp.frequencies,
        freq_resp.magnitude_db,
        freq_resp.phase_deg,
        title=f"RLC Circuit — Bode Plot (R={R}Ω, L={L*1e3:.0f}mH, C={C*1e6:.0f}µF)",
        cutoff_freq=f_resonant,
        save_path=bode_path,
        show=False,
    )
    print(f"  Bode plot saved → {bode_path}")

    # -----------------------------------------------------------------------
    # 3. Pole-zero map
    # -----------------------------------------------------------------------
    poles, zeros = compute_poles_zeros(num, den)
    print(f"\n  Poles: {poles}")
    print(f"  Zeros: {zeros}")

    pz_path = str(OUTPUT_DIR / "rlc_pole_zero.png")
    plot_pole_zero(poles, zeros, title="RLC Circuit — Pole-Zero Map", save_path=pz_path, show=False)
    print(f"  Pole-zero map saved → {pz_path}")

    # -----------------------------------------------------------------------
    # 4. Step response comparison: underdamped / critically damped / overdamped
    # -----------------------------------------------------------------------
    print("\n  Step response comparison (varying R):")

    # Critically damped: R_crit = 2·√(L/C)
    R_crit = 2.0 * math.sqrt(L / C)
    R_over = R_crit * 3   # overdamped

    configs = [
        (R, "Underdamped"),
        (R_crit, "Critically Damped"),
        (R_over, "Overdamped"),
    ]
    colors = ["#2563EB", "#16A34A", "#DC2626"]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("RLC Circuit — Step Response Comparison", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.axhline(1.0, color="#6B7280", linestyle="--", linewidth=1.0, alpha=0.5, label="Final value")

    for (r_val, label), color in zip(configs, colors):
        c = RLCCircuit(r_val, L, C)
        n, d = c.transfer_function()
        tr = analyze_time_response(n, d, response_type="step", num_points=1000)
        ax.plot(tr.time, tr.output, color=color, linewidth=2.0,
                label=f"{label} (R={r_val:.0f}Ω, ζ={c.damping_ratio:.2f})")
        print(f"    {label}: R={r_val:.1f}Ω, ζ={c.damping_ratio:.3f}, Q={c.quality_factor:.3f}")

    ax.legend()
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    step_cmp_path = str(OUTPUT_DIR / "rlc_step_comparison.png")
    fig.savefig(step_cmp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Step comparison saved → {step_cmp_path}")

    # -----------------------------------------------------------------------
    # 5. Q factor sweep — Bode comparison
    # -----------------------------------------------------------------------
    print("\n  Q factor sweep (Bode comparison):")
    q_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    comparison_data = []
    for q_target in q_values:
        # R = (1/Q)·√(L/C)
        r_val = (1 / q_target) * math.sqrt(L / C)
        c_circ = RLCCircuit(r_val, L, C)
        n, d = c_circ.transfer_function()
        fr = analyze_frequency_response(n, d, freq_start=100.0, freq_stop=1e5, num_points=1000)
        comparison_data.append(
            {"frequencies": fr.frequencies, "magnitude_db": fr.magnitude_db, "label": f"Q = {q_target}"}
        )
        print(f"    Q = {q_target} → R = {r_val:.2f} Ω, f0 = {c_circ.natural_frequency:.2f} Hz")

    q_sweep_path = str(OUTPUT_DIR / "rlc_q_sweep.png")
    plot_comparison(
        comparison_data,
        title=f"RLC Circuit — Q Factor Sweep (L={L*1e3:.0f}mH, C={C*1e6:.0f}µF)",
        save_path=q_sweep_path,
        show=False,
    )
    print(f"\n  Q-sweep plot saved → {q_sweep_path}")
    print("\nDone! ✓")


if __name__ == "__main__":
    main()
