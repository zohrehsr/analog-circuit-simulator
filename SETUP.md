# Quick Setup & Execution Guide

This guide walks you through setting up the virtual environment, installing dependencies, running the test suite, and executing example simulations.

---

## 1. Environment Setup

### Create and activate virtual environment:

**Windows (PowerShell):**
```powershell
# Navigate to the project directory
cd analog-circuit-simulator

# Create virtual environment (using standard Python or uv)
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS (Bash / Zsh):**
```bash
# Navigate to the project directory
cd analog-circuit-simulator

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

---

## 2. Install Dependencies

Install the package in editable mode along with development and testing dependencies:

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 3. Run Test Suite

Run all 88 unit tests with verbose output:

```bash
pytest -v
```

To run tests with a code coverage report:

```bash
pytest --cov=analog_circuit_simulator --cov-report=term-missing
```

---

## 4. Run Example Scripts

Execute the pre-built circuit analysis and parameter sweep scripts:

```bash
python examples/rc_lowpass_filter.py
python examples/rlc_bandpass_filter.py
```

Generated plots will be automatically saved to `examples/output/`.

---

## 5. Command-Line Interface (CLI)

Use the built-in `analog-sim` CLI tool for fast terminal-based simulation and data export:

```bash
# Frequency-domain Bode analysis (with optional CSV data export)
analog-sim analyze rc -R 1000 -C 1e-6 --export bode_data.csv

# Time-domain step and impulse response simulation
analog-sim simulate rlc -R 100 -L 10e-3 -C 1e-6 --response both --save-plot rlc_response.png
```

---

## 6. Git & Version Control

To track changes and synchronize with your GitHub repository:

```bash
git add .
git commit -m "feat: your descriptive commit message"
git push origin main
```
