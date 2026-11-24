"""Publication-quality plotting utilities for analog circuit analysis.

Provides functions to generate Bode diagrams, time-domain response plots,
pole-zero maps, and multi-circuit comparison plots using Matplotlib.

All functions accept numpy arrays produced by the :mod:`analysis` module
and return a :class:`matplotlib.figure.Figure` instance so the caller can
further customise or save the figure.

Typical usage::

    >>> from analog_circuit_simulator.circuits import RCCircuit
    >>> from analog_circuit_simulator.analysis import (
    ...     analyze_frequency_response, analyze_time_response,
    ... )
    >>> from analog_circuit_simulator.plotting import plot_bode, plot_time_response
    >>> rc = RCCircuit(1e3, 1e-6)
    >>> num, den = rc.transfer_function()
    >>> freq_resp = analyze_frequency_response(num, den)
    >>> fig = plot_bode(freq_resp.frequencies, freq_resp.magnitude_db,
    ...                 freq_resp.phase_deg, show=False)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Style configuration – applied once at import time
# ---------------------------------------------------------------------------
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False


def _check_mpl() -> None:
    """Raise ImportError if matplotlib is not installed."""
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting.  "
            "Install it with: pip install matplotlib"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_bode(
    frequencies: np.ndarray,
    magnitude_db: np.ndarray,
    phase_deg: np.ndarray,
    title: str = "Bode Plot",
    cutoff_freq: float | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> Figure:
    """Plot a Bode diagram (magnitude and phase vs. frequency).

    Creates a two-subplot figure:

    * **Top** – magnitude in dB vs. frequency (Hz), semi-log x-axis.
    * **Bottom** – phase in degrees vs. frequency (Hz), semi-log x-axis.

    Args:
        frequencies: Frequency array in Hz.
        magnitude_db: Magnitude array in dB.
        phase_deg: Phase array in degrees.
        title: Figure title.
        cutoff_freq: If given, draw a vertical dashed marker at this
            frequency and annotate the −3 dB point on the magnitude axis.
        save_path: If given, save the figure to this file path.
        show: If ``True``, call :func:`plt.show()` before returning.

    Returns:
        The :class:`matplotlib.figure.Figure` instance.

    Example:
        >>> import numpy as np
        >>> from analog_circuit_simulator.analysis import analyze_frequency_response
        >>> from analog_circuit_simulator.circuits import RCCircuit
        >>> rc = RCCircuit(1e3, 1e-6)
        >>> num, den = rc.transfer_function()
        >>> fr = analyze_frequency_response(num, den)
        >>> fig = plot_bode(fr.frequencies, fr.magnitude_db, fr.phase_deg, show=False)
    """
    _check_mpl()
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # -- Magnitude --
    ax_mag.semilogx(
        frequencies, magnitude_db, color="#2563EB", linewidth=2.0, label="Magnitude"
    )
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_ylim(bottom=min(magnitude_db) - 5)
    ax_mag.axhline(
        -3, color="#DC2626", linewidth=1.0, linestyle=":", alpha=0.7, label="−3 dB"
    )

    if cutoff_freq is not None:
        ax_mag.axvline(
            cutoff_freq,
            color="#16A34A",
            linewidth=1.5,
            linestyle="--",
            label=f"fc = {cutoff_freq:.1f} Hz",
        )
        ax_phase.axvline(
            cutoff_freq,
            color="#16A34A",
            linewidth=1.5,
            linestyle="--",
            label=f"fc = {cutoff_freq:.1f} Hz",
        )

    ax_mag.legend(loc="lower left")

    # -- Phase --
    ax_phase.semilogx(frequencies, phase_deg, color="#7C3AED", linewidth=2.0)
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (degrees)")
    ax_phase.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])

    if cutoff_freq is not None:
        ax_phase.legend(loc="lower left")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()

    return fig


def plot_time_response(
    time: np.ndarray,
    output: np.ndarray,
    response_type: str = "step",
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> Figure:
    """Plot a step or impulse response.

    Args:
        time: Time vector in seconds.
        output: System output signal.
        response_type: ``"step"`` or ``"impulse"`` — used for axis label and
            automatic title.
        title: Override the auto-generated title.
        save_path: If given, save the figure to this file path.
        show: If ``True``, call :func:`plt.show()` before returning.

    Returns:
        The :class:`matplotlib.figure.Figure` instance.

    Example:
        >>> import numpy as np
        >>> from analog_circuit_simulator.analysis import analyze_time_response
        >>> from analog_circuit_simulator.circuits import RCCircuit
        >>> rc = RCCircuit(1e3, 1e-6)
        >>> num, den = rc.transfer_function()
        >>> tr = analyze_time_response(num, den, response_type="step")
        >>> fig = plot_time_response(tr.time, tr.output, show=False)
    """
    _check_mpl()
    if title is None:
        label = response_type.capitalize()
        title = f"{label} Response"

    fig, ax = plt.subplots(figsize=(10, 5))
    color = "#2563EB" if response_type == "step" else "#7C3AED"
    ax.plot(time, output, color=color, linewidth=2.0, label=response_type.capitalize())

    if response_type == "step":
        # Mark final value
        final_val = output[-1]
        ax.axhline(
            final_val,
            color="#6B7280",
            linewidth=1.0,
            linestyle="--",
            alpha=0.6,
            label=f"Final value ≈ {final_val:.4g}",
        )
        # Mark 63.2% (one time-constant) point
        target_63 = 0.632 * final_val
        idx_63 = np.searchsorted(output, target_63) if final_val > 0 else None
        if idx_63 is not None and 0 < idx_63 < len(time):
            ax.axvline(
                time[idx_63],
                color="#DC2626",
                linewidth=1.0,
                linestyle=":",
                alpha=0.7,
                label=f"τ ≈ {time[idx_63]:.4g} s (63.2%)",
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()

    return fig


def plot_pole_zero(
    poles: np.ndarray,
    zeros: np.ndarray,
    title: str = "Pole-Zero Map",
    save_path: str | None = None,
    show: bool = True,
) -> Figure:
    """Plot poles and zeros in the complex s-plane.

    Poles are drawn as red ``×`` markers; zeros as blue ``○`` markers.

    Args:
        poles: Complex array of pole locations.
        zeros: Complex array of zero locations.
        title: Figure title.
        save_path: If given, save the figure to this file path.
        show: If ``True``, call :func:`plt.show()` before returning.

    Returns:
        The :class:`matplotlib.figure.Figure` instance.

    Example:
        >>> import numpy as np
        >>> from analog_circuit_simulator.analysis import compute_poles_zeros
        >>> from analog_circuit_simulator.circuits import RLCCircuit
        >>> rlc = RLCCircuit(100.0, 10e-3, 1e-6)
        >>> num, den = rlc.transfer_function()
        >>> p, z = compute_poles_zeros(num, den)
        >>> fig = plot_pole_zero(p, z, show=False)
    """
    _check_mpl()
    fig, ax = plt.subplots(figsize=(8, 7))

    # Axes lines through origin
    ax.axhline(0, color="#374151", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="#374151", linewidth=0.8, alpha=0.5)

    # Plot poles
    if poles.size:
        ax.scatter(
            np.real(poles),
            np.imag(poles),
            marker="x",
            s=120,
            color="#DC2626",
            linewidths=2.5,
            zorder=5,
            label="Poles (×)",
        )

    # Plot zeros
    if zeros.size:
        ax.scatter(
            np.real(zeros),
            np.imag(zeros),
            marker="o",
            s=90,
            facecolors="none",
            edgecolors="#2563EB",
            linewidths=2.0,
            zorder=5,
            label="Zeros (○)",
        )

    ax.set_xlabel("Real Axis (σ)")
    ax.set_ylabel("Imaginary Axis (jω)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()

    # Shade unstable region
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_betweenx(
        ylim,
        0,
        max(xlim[1], 1.0),
        alpha=0.07,
        color="#DC2626",
        label="Unstable (Re > 0)",
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()

    return fig


def plot_comparison(
    responses: list[dict],
    title: str = "Frequency Response Comparison",
    save_path: str | None = None,
    show: bool = True,
) -> Figure:
    """Compare multiple frequency responses on a single magnitude plot.

    Args:
        responses: List of dicts, each with keys:

            * ``"frequencies"`` (:class:`numpy.ndarray`) – Hz.
            * ``"magnitude_db"`` (:class:`numpy.ndarray`) – dB.
            * ``"label"`` (str) – legend label.

        title: Figure title.
        save_path: If given, save the figure to this file path.
        show: If ``True``, call :func:`plt.show()` before returning.

    Returns:
        The :class:`matplotlib.figure.Figure` instance.

    Raises:
        ValueError: If *responses* is empty or any dict is missing required keys.

    Example:
        >>> import numpy as np
        >>> from analog_circuit_simulator.circuits import RCCircuit
        >>> from analog_circuit_simulator.analysis import analyze_frequency_response
        >>> responses = []
        >>> for r in [1e3, 10e3, 100e3]:
        ...     rc = RCCircuit(r, 1e-6)
        ...     num, den = rc.transfer_function()
        ...     fr = analyze_frequency_response(num, den)
        ...     responses.append({"frequencies": fr.frequencies,
        ...                       "magnitude_db": fr.magnitude_db,
        ...                       "label": f"R={r/1e3:.0f} kΩ"})
        >>> fig = plot_comparison(responses, show=False)
    """
    _check_mpl()
    if not responses:
        raise ValueError("responses list must not be empty.")

    required_keys = {"frequencies", "magnitude_db", "label"}
    for i, resp in enumerate(responses):
        missing = required_keys - set(resp.keys())
        if missing:
            raise ValueError(f"Response[{i}] is missing keys: {missing}")

    # Colour palette
    colors = [
        "#2563EB",
        "#DC2626",
        "#16A34A",
        "#7C3AED",
        "#EA580C",
        "#0891B2",
        "#DB2777",
        "#65A30D",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, resp in enumerate(responses):
        color = colors[i % len(colors)]
        ax.semilogx(
            resp["frequencies"],
            resp["magnitude_db"],
            color=color,
            linewidth=1.8,
            label=resp["label"],
        )

    ax.axhline(
        -3, color="#6B7280", linewidth=1.0, linestyle=":", alpha=0.6, label="−3 dB"
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()

    return fig
