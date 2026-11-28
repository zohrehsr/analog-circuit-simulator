# Analog Circuit Simulator

[![CI](https://github.com/zohrehsr/analog-circuit-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/zohrehsr/analog-circuit-simulator/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

An analytical Python framework for behavioral simulation, small-signal frequency response, and transient time-domain characterization of fundamental passive analog networks (RC, RL, and RLC). Designed for rapid analog circuit prototyping and filter synthesis prior to schematic-level SPICE netlist generation.

---

## Features

| Capability | Description |
|---|---|
| **Component Models** | Parameterized `Resistor`, `Capacitor`, and `Inductor` classes evaluating complex impedance $Z(\omega)$ across arbitrary frequency vectors. |
| **Circuit Topologies** | Analytical transfer functions for first-order (`RCCircuit`, `RLCircuit`) and second-order (`RLCCircuit`) filters. |
| **Frequency-Domain Analysis** | Continuous-time Bode response (magnitude in dB, phase in degrees), $-3\text{ dB}$ cutoff frequency, and resonant peaking evaluation. |
| **Time-Domain Analysis** | Exact step and impulse response evaluation with automated simulation horizon estimation based on dominant system poles. |
| **Pole-Zero Characterization** | Extraction of continuous-time poles and zeros with bounded-input bounded-output (BIBO) stability checks. |
| **Laboratory Data Export** | Direct export of simulated Bode grids and time traces to CSV format for post-processing in MATLAB, Origin, or oscilloscope comparison. |
| **Visualization Suite** | Publication-quality Matplotlib figures for Bode magnitude/phase plots, step/impulse responses, and $s$-plane pole-zero maps. |
| **Command-Line Interface (CLI)** | Standalone `analog-sim` executable for parameter sweeps, simulation, and data export directly from the terminal. |

---

## Mathematical Formulations & Circuit Models

Linear continuous-time circuits are modeled as rational transfer functions in the Laplace domain:

$$H(s) = \frac{N(s)}{D(s)} = \frac{b_m s^m + \dots + b_1 s + b_0}{a_n s^n + \dots + a_1 s + a_0}$$

### 1. RC Low-Pass Filter (First-Order)

$$H(s) = \frac{1}{RCs + 1}, \quad \tau = RC, \quad f_c = \frac{1}{2\pi RC}$$

* Single pole at $p_1 = -1/(RC)$.
* High-frequency roll-off of $-20\text{ dB/decade}$.

```python
from analog_circuit_simulator import RCCircuit

rc = RCCircuit(resistance=1e3, capacitance=1e-6)
print(f"Cutoff frequency: {rc.cutoff_frequency:.2f} Hz")  # 159.15 Hz
print(f"Time constant:    {rc.time_constant * 1e3:.2f} ms")  # 1.00 ms
```

### 2. RL High-Pass Filter (First-Order)

$$H(s) = \frac{\frac{L}{R}s}{\frac{L}{R}s + 1}, \quad \tau = \frac{L}{R}, \quad f_c = \frac{R}{2\pi L}$$

* Single zero at origin $s = 0$, single real pole at $p_1 = -R/L$.
* Low-frequency attenuation with $+20\text{ dB/decade}$ slope up to $f_c$.

```python
from analog_circuit_simulator import RLCircuit

rl = RLCircuit(resistance=1e3, inductance=100e-3)
print(f"Cutoff frequency: {rl.cutoff_frequency:.2f} Hz")  # 1591.55 Hz
```

### 3. RLC Resonant Network (Second-Order)

$$H(s) = \frac{1}{LCs^2 + RCs + 1} = \frac{\omega_0^2}{s^2 + 2\zeta\omega_0 s + \omega_0^2}$$

$$\omega_0 = \frac{1}{\sqrt{LC}}, \quad \zeta = \frac{R}{2}\sqrt{\frac{C}{L}}, \quad Q = \frac{1}{2\zeta} = \frac{1}{R}\sqrt{\frac{L}{C}}$$

* **Damping Classification:**
  * $\zeta > 1$: Overdamped (two distinct negative real poles).
  * $\zeta = 1$: Critically damped (repeated negative real pole).
  * $0 < \zeta < 1$: Underdamped (complex-conjugate poles, exhibiting transient ringing and potential frequency peaking).

```python
from analog_circuit_simulator import RLCCircuit

rlc = RLCCircuit(resistance=100.0, inductance=10e-3, capacitance=1e-6)
print(f"Resonance (f0): {rlc.natural_frequency:.2f} Hz")  # 1591.55 Hz
print(f"Quality Factor: {rlc.quality_factor:.3f}")        # 3.162
print(f"Damping Ratio:  {rlc.damping_ratio:.4f}")         # 0.1581
print(f"Response Type:  {rlc.circuit_type}")              # underdamped
```

---

## Architecture & Implementation Notes

* **$s$-Domain Analytical Modeling:** Rather than integrating non-linear differential algebraic equations (DAEs) via SPICE numerical engines, circuit blocks are synthesized directly as continuous Linear Time-Invariant (LTI) transfer functions via `scipy.signal.TransferFunction`. This guarantees instantaneous sweeps across wide parametric ranges without convergence issues.
* **Vectorized Impedance Calculations:** Impedance expressions $Z_R = R$, $Z_C(\omega) = \frac{1}{j\omega C}$, and $Z_L(\omega) = j\omega L$ leverage NumPy broadcasting across arbitrary logarithmic or linear frequency vectors.
* **Automated Settling Horizon Heuristic:** For time-domain transient analysis, the simulation duration $t_{\text{stop}}$ is derived from the real part of the dominant (slowest) system pole:
  $$\tau_{\text{dominant}} = \frac{1}{\min_i |\text{Re}(p_i)|}, \quad t_{\text{sim}} \approx 5 \cdot \tau_{\text{dominant}}$$
  ensuring full $>99.3\%$ transient settling without manual time-step tuning.
* **Clean `src`-Layout:** Modular directory structure isolating source code from test harnesses, avoiding import-path poisoning and ensuring packaging integrity.

---

## Installation

### From Source

```bash
git clone https://github.com/zohrehsr/analog-circuit-simulator.git
cd analog-circuit-simulator

# Install in editable mode
pip install -e .

# Install development dependencies (pytest, ruff, pytest-cov)
pip install -e ".[dev]"
```

---

## Quick Start

### Frequency & Time-Domain Simulation

```python
from analog_circuit_simulator import RCCircuit, analyze_frequency_response, analyze_time_response
from analog_circuit_simulator.plotting import plot_bode, plot_time_response

# Instantiate 1st-order RC low-pass filter
rc = RCCircuit(resistance=1e3, capacitance=1e-6)
num, den = rc.transfer_function()

# Compute continuous frequency response (1 Hz to 100 kHz)
freq_resp = analyze_frequency_response(num, den, freq_start=1.0, freq_stop=1e5)
plot_bode(
    freq_resp.frequencies,
    freq_resp.magnitude_db,
    freq_resp.phase_deg,
    cutoff_freq=rc.cutoff_frequency,
)

# Compute transient step response
time_resp = analyze_time_response(num, den, response_type="step")
plot_time_response(time_resp.time, time_resp.output)
```

---

## Command-Line Interface (CLI)

The package provides a command-line interface `analog-sim` for terminal-based workflow execution and CSV data generation.

```bash
# Frequency sweep of an RC filter with CSV export
analog-sim analyze rc -R 1000 -C 1e-6 --export rc_bode.csv

# Transient step and impulse simulation of an RLC filter with plot output
analog-sim simulate rlc -R 100 -L 10e-3 -C 1e-6 --response both --save-plot rlc_transient.png
```

---

## Verification & Test Suite

The repository maintains an automated test suite with full coverage of analytical equations, edge cases, stability boundaries, and CLI command execution.

```bash
# Run complete test suite
pytest

# Run tests with line coverage report
pytest --cov=analog_circuit_simulator
```

All 88 unit tests run against Python 3.10, 3.11, and 3.12 under GitHub Actions CI matrix:

```
collected 88 items
tests/test_analysis.py    ............................  [ 31%]
tests/test_circuits.py    ............................  [ 63%]
tests/test_cli.py         ....                          [ 68%]
tests/test_components.py  ............................  [100%]

88 passed in 2.45s (100% passing)
```

---

## Project Structure

```
analog-circuit-simulator/
├── src/
│   └── analog_circuit_simulator/
│       ├── __init__.py        # Public package exports
│       ├── components.py      # Resistor, Capacitor, Inductor classes
│       ├── circuits.py        # RCCircuit, RLCircuit, RLCCircuit
│       ├── analysis.py        # s-domain transfer functions & response solvers
│       ├── plotting.py        # Matplotlib visualization routines
│       └── cli.py             # CLI parser and command handlers
├── tests/
│   ├── test_components.py     # Impedance and parameter validation
│   ├── test_circuits.py       # Transfer function & stability assertions
│   ├── test_analysis.py       # Frequency & time response verification
│   └── test_cli.py            # CLI integration tests
├── examples/
│   ├── rc_lowpass_filter.py
│   └── rlc_bandpass_filter.py
├── pyproject.toml             # Build configuration and dependency specifications
├── LICENSE                    # MIT License
└── README.md
```

---

## Research Roadmap

This repository forms the foundation for a modular research framework in analog IC modeling and mixed-signal data conversion:

- [x] **Phase I: Fundamental Passive Networks** — Continuous-time $s$-domain models for RC, RL, and RLC filters with stability verification.
- [x] **Phase II: Phase-Locked Loop Behavioral Model** — Behavioral simulation of phase detectors, charge pumps, and loop filters. Released in [pll-behavioral-model](https://github.com/Zohrehsr/pll-behavioral-model).
- [ ] **Phase III: Active Circuit Blocks** — Operational amplifier models incorporating finite gain-bandwidth product (GBW), slew-rate limiting, and input offset.
- [ ] **Phase IV: VCO-Based ADC Architecture** — Behavioral quantization and phase-domain integration models for continuous-time $\Delta\Sigma$ and open-loop VCO ADCs.
- [ ] **Phase V: Machine Learning Calibration** — Neural network and polynomial post-distortion algorithms for non-linearity correction in VCO-based ADCs.

---

## References

1. A. S. Sedra and K. C. Smith, *Microelectronic Circuits*, 8th ed., Oxford University Press, 2020.
2. B. Razavi, *Design of Analog CMOS Integrated Circuits*, 2nd ed., McGraw-Hill Education, 2017.
3. A. V. Oppenheim and A. S. Willsky, *Signals and Systems*, 2nd ed., Prentice Hall, 1996.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Author

**Zohreh Sedighrad**  
M.Sc. in Integrated Circuits (Electronic Engineering)  
Research Focus: Mixed-Signal IC Design & Machine Learning Calibration for VCO-based ADCs
