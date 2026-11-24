# Analog Circuit Simulator

[![CI](https://github.com/zohrehsr/analog-circuit-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/zohrehsr/analog-circuit-simulator/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A Python library for **behavioral simulation and frequency/time-domain analysis** of fundamental analog circuits — RC, RL, and RLC.  
Built for electronics engineers, students, and researchers who want to quickly prototype, visualize, and validate analog filter behavior without leaving Python.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔌 **Component models** | `Resistor`, `Capacitor`, `Inductor` with complex impedance over frequency |
| 📐 **Circuit models** | `RCCircuit`, `RLCircuit`, `RLCCircuit` with analytical transfer functions |
| 📊 **Frequency analysis** | Bode plot data (magnitude & phase), −3 dB cutoff, resonant frequency |
| ⏱️ **Time analysis** | Step & impulse responses with auto-estimated simulation horizon |
| 🗺️ **Pole-Zero analysis** | Poles, zeros, BIBO stability check |
| 💾 **Data Export** | Export Bode and time-response curves to CSV for laboratory/MATLAB post-processing |
| 📈 **Plotting** | Publication-quality Bode, time-response, pole-zero, and comparison plots |
| 🖥️ **CLI** | `analog-sim simulate` and `analog-sim analyze` with optional CSV export |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/zohrehsr/analog-circuit-simulator.git
cd analog-circuit-simulator

# Install in editable mode
pip install -e .

# Install with dev tools (for running tests)
pip install -e ".[dev]"
```

### Basic Usage

```python
from analog_circuit_simulator import RCCircuit, analyze_frequency_response, analyze_time_response
from analog_circuit_simulator.plotting import plot_bode, plot_time_response

# Create an RC low-pass filter: R = 1 kΩ, C = 1 µF
rc = RCCircuit(resistance=1e3, capacitance=1e-6)

print(f"Cutoff frequency: {rc.cutoff_frequency:.2f} Hz")  # → 159.15 Hz
print(f"Time constant  : {rc.time_constant*1e3:.2f} ms")  # → 1.00 ms

# Frequency response
num, den = rc.transfer_function()
freq_resp = analyze_frequency_response(num, den, freq_start=1.0, freq_stop=1e5)
plot_bode(freq_resp.frequencies, freq_resp.magnitude_db, freq_resp.phase_deg,
          cutoff_freq=rc.cutoff_frequency)

# Step response
time_resp = analyze_time_response(num, den, response_type="step")
plot_time_response(time_resp.time, time_resp.output)
```

---

## 📐 Circuit Models

### RC Circuit (Low-Pass Filter)

$$H(s) = \frac{1}{RCs + 1}, \quad f_c = \frac{1}{2\pi RC}$$

```python
from analog_circuit_simulator import RCCircuit

rc = RCCircuit(resistance=1e3, capacitance=1e-6)
print(rc.cutoff_frequency)   # 159.15 Hz
print(rc.time_constant)      # 0.001 s
```

### RL Circuit (High-Pass Filter)

$$H(s) = \frac{\frac{L}{R}s}{\frac{L}{R}s + 1}, \quad f_c = \frac{R}{2\pi L}$$

```python
from analog_circuit_simulator import RLCircuit

rl = RLCircuit(resistance=1e3, inductance=100e-3)
print(rl.cutoff_frequency)   # 1591.55 Hz
```

### RLC Circuit (Second-Order)

$$H(s) = \frac{1}{LCs^2 + RCs + 1}, \quad f_0 = \frac{1}{2\pi\sqrt{LC}}, \quad Q = \frac{1}{R}\sqrt{\frac{L}{C}}$$

```python
from analog_circuit_simulator import RLCCircuit

rlc = RLCCircuit(resistance=100.0, inductance=10e-3, capacitance=1e-6)
print(f"f0  = {rlc.natural_frequency:.2f} Hz")   # 1591.55 Hz
print(f"Q   = {rlc.quality_factor:.3f}")          # 3.162
print(f"ζ   = {rlc.damping_ratio:.4f}")           # 0.1581
print(f"BW  = {rlc.bandwidth:.2f} Hz")            # 503.29 Hz
print(f"Type: {rlc.circuit_type}")                # underdamped
```

---

## 🖥️ Command-Line Interface

```bash
# Analyze frequency response of an RC circuit and export data to CSV
analog-sim analyze rc -R 1000 -C 1e-6 --export bode_data.csv

# Simulate step and impulse response of an RLC circuit
analog-sim simulate rlc -R 100 -L 10e-3 -C 1e-6 --response both --save-plot rlc_sim.png
```

---

## 🏛️ Architecture & Design Decisions

* **Continuous-Time Transfer Functions ($s$-domain):** Rather than evaluating slow numerical SPICE netlists, linear analog topologies (RC, RL, RLC) are represented as continuous LTI transfer functions $H(s) = \frac{N(s)}{D(s)}$ using `scipy.signal.TransferFunction`. This achieves instantaneous parameter sweeps and pole-zero extractions.
* **Vectorized Impedance Models:** All fundamental passive components calculate complex impedance $Z(\omega)$ using NumPy array broadcasting ($1 / (j \omega C)$, $j \omega L$), enabling high-throughput frequency analysis across arbitrary spectrum grids.
* **Settling Time Heuristic:** Automated time-domain horizon calculation locates the slowest system pole ($\tau = 1 / |\text{Re}(p_{\text{slowest}})|$) and scales by $5\tau$ ($99.3\%$ settling criterion), avoiding manual simulation window adjustments.
* **Modern `src-layout`:** Implements `src/` directory packaging to prevent import-shadowing during unit testing and enforce installation integrity.

---

## 🧪 Running Tests

```bash
pytest                                  # run all 88 unit tests
pytest --cov=analog_circuit_simulator  # with line coverage report
```

Expected output:
```
collected 88 items
tests/test_analysis.py    ............................  [ 31%]
tests/test_circuits.py    ............................  [ 63%]
tests/test_cli.py         ....                          [ 68%]
tests/test_components.py  ............................  [100%]

88 passed in 2.50s (100% passing)
```

---

## 📂 Project Structure

```
analog-circuit-simulator/
├── src/
│   └── analog_circuit_simulator/
│       ├── __init__.py        # Public API
│       ├── components.py      # Resistor, Capacitor, Inductor
│       ├── circuits.py        # RCCircuit, RLCircuit, RLCCircuit
│       ├── analysis.py        # Frequency & time domain analysis
│       ├── plotting.py        # Matplotlib visualization
│       └── cli.py             # Command-line interface
├── tests/
│   ├── test_components.py
│   ├── test_circuits.py
│   ├── test_analysis.py
│   └── test_cli.py
├── examples/
│   ├── rc_lowpass_filter.py
│   ├── rlc_bandpass_filter.py
│   
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 🔭 Roadmap

This project is part of a larger portfolio in **analog IC design** and **ML for analog circuits**:

- [ ] **v0.2** — Operational amplifier models (ideal and non-ideal)
- [ ] **v0.3** — PLL behavioral model (Phase Detector, Loop Filter, VCO)
- [ ] **v0.4** — VCO-based ADC simulator (open-loop and closed-loop)
- [ ] **v0.5** — ADC performance metrics (ENOB, SFDR, SNDR, INL/DNL)
- [ ] **v1.0** — ML-based calibration for VCO-based ADC 

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome!  
Please open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👩‍💻 Author

**Zohreh Sedighrad** — M.Sc. in Integrated Circuits (Electronic Engineering)  
Research focus: VCO-based ADC calibration with Machine Learning