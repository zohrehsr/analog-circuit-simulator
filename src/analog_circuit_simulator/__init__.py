"""Analog Circuit Simulator — behavioral simulation and analysis of analog circuits.

A Python library for time-domain and frequency-domain analysis of RC, RL,
and RLC circuits.
"""

__version__ = "0.1.0"

from .analysis import (
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
from .circuits import RCCircuit, RLCCircuit, RLCircuit
from .components import Capacitor, Component, Inductor, Resistor

__all__ = [
    "__version__",
    # Components
    "Component",
    "Resistor",
    "Capacitor",
    "Inductor",
    # Circuits
    "RCCircuit",
    "RLCircuit",
    "RLCCircuit",
    # Analysis data containers
    "FrequencyResponse",
    "TimeResponse",
    # Analysis functions
    "analyze_frequency_response",
    "analyze_time_response",
    "compute_transfer_function",
    "find_cutoff_frequency",
    "find_resonant_frequency",
    "compute_poles_zeros",
    "compute_stability",
]
