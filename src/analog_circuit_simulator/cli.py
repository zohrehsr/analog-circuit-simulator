"""Command-line interface for the Analog Circuit Simulator.

Provides ``simulate`` and ``analyze`` subcommands for quick circuit
analysis directly from the terminal.

Usage Examples::

    analog-sim simulate rc --resistance 1000 --capacitance 1e-6
    analog-sim analyze rlc --resistance 100 --inductance 0.01 --capacitance 1e-6
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__

# ─── Banner ───────────────────────────────────────────────────────────

_BANNER = rf"""
    ╔═══════════════════════════════════════════════╗
    ║   Analog Circuit Simulator  v{__version__:<15s}║
    ║   Time-domain & Frequency-domain Analysis     ║
    ╚═══════════════════════════════════════════════╝
"""


# ─── Argument parsing ────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="analog-sim",
        description="Analog Circuit Simulator — simulate and analyze RC/RL/RLC circuits.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── simulate ──────────────────────────────────────────────────
    sim_parser = subparsers.add_parser(
        "simulate",
        help="Simulate step/impulse response of a circuit.",
    )
    _add_circuit_args(sim_parser)
    sim_parser.add_argument(
        "--duration",
        type=float,
        default=0.01,
        help="Simulation duration in seconds (default: 0.01).",
    )
    sim_parser.add_argument(
        "--response",
        choices=["step", "impulse", "both"],
        default="both",
        help="Type of response to plot (default: both).",
    )
    sim_parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Optional path to export simulation data as CSV.",
    )
    sim_parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Optional path to save plot image.",
    )
    sim_parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display plot window.",
    )

    # ── analyze ───────────────────────────────────────────────────
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Show Bode plot (magnitude & phase) of a circuit.",
    )
    _add_circuit_args(analyze_parser)
    analyze_parser.add_argument(
        "--fmin",
        type=float,
        default=1.0,
        help="Minimum frequency in Hz for Bode plot (default: 1).",
    )
    analyze_parser.add_argument(
        "--fmax",
        type=float,
        default=1e6,
        help="Maximum frequency in Hz for Bode plot (default: 1e6).",
    )
    analyze_parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Optional path to export Bode data as CSV.",
    )
    analyze_parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Optional path to save plot image.",
    )
    analyze_parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display plot window.",
    )

    return parser


def _add_circuit_args(parser: argparse.ArgumentParser) -> None:
    """Add shared circuit-type and component-value arguments to *parser*.

    Args:
        parser: The sub-parser to augment.
    """
    parser.add_argument(
        "circuit_type",
        choices=["rc", "rl", "rlc"],
        help="Circuit topology to use.",
    )
    parser.add_argument(
        "-R",
        "--resistance",
        type=float,
        required=True,
        help="Resistance in ohms.",
    )
    parser.add_argument(
        "-C",
        "--capacitance",
        type=float,
        default=None,
        help="Capacitance in farads (required for rc, rlc).",
    )
    parser.add_argument(
        "-L",
        "--inductance",
        type=float,
        default=None,
        help="Inductance in henrys (required for rl, rlc).",
    )


# ─── Validation helpers ──────────────────────────────────────────────


def _validate_components(args: argparse.Namespace) -> None:
    """Ensure the required component values were supplied.

    Args:
        args: Parsed CLI arguments.

    Raises:
        SystemExit: If a required component value is missing.
    """
    ct = args.circuit_type
    if ct in ("rc", "rlc") and args.capacitance is None:
        sys.exit(f"Error: --capacitance is required for '{ct}' circuit.")
    if ct in ("rl", "rlc") and args.inductance is None:
        sys.exit(f"Error: --inductance is required for '{ct}' circuit.")


def _build_circuit(args: argparse.Namespace):
    """Instantiate the appropriate circuit object from CLI args.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A circuit instance (``RCCircuit``, ``RLCircuit``, or ``RLCCircuit``).
    """
    from .circuits import RCCircuit, RLCCircuit, RLCircuit

    ct = args.circuit_type
    if ct == "rc":
        return RCCircuit(resistance=args.resistance, capacitance=args.capacitance)
    if ct == "rl":
        return RLCircuit(resistance=args.resistance, inductance=args.inductance)
    return RLCCircuit(
        resistance=args.resistance,
        inductance=args.inductance,
        capacitance=args.capacitance,
    )


# ─── Subcommand handlers ─────────────────────────────────────────────


def _handle_simulate(args: argparse.Namespace) -> None:
    """Execute the ``simulate`` subcommand.

    Computes and plots the step and/or impulse response of the selected
    circuit using :func:`~analog_circuit_simulator.analysis.analyze_time_response`.

    Args:
        args: Parsed CLI arguments.
    """
    import matplotlib.pyplot as plt

    from .analysis import analyze_time_response

    _validate_components(args)
    circuit = _build_circuit(args)
    num, den = circuit.transfer_function()

    step_resp = None
    imp_resp = None

    if args.response in ("step", "both"):
        step_resp = analyze_time_response(
            num, den, response_type="step", t_end=args.duration
        )
    if args.response in ("impulse", "both"):
        imp_resp = analyze_time_response(
            num, den, response_type="impulse", t_end=args.duration
        )

    # Export to CSV if requested
    if args.export:
        if args.response == "step" and step_resp is not None:
            step_resp.to_csv(args.export)
        elif args.response == "impulse" and imp_resp is not None:
            imp_resp.to_csv(args.export)
        elif args.response == "both" and step_resp is not None and imp_resp is not None:
            combined = np.column_stack(
                (step_resp.time, step_resp.output, imp_resp.output)
            )
            np.savetxt(
                args.export,
                combined,
                delimiter=",",
                header="time_s,step_output,impulse_output",
                comments="",
            )
        print(f"Simulation data exported to '{args.export}'")

    if not args.no_plot:
        nrows = 2 if args.response == "both" else 1
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            figsize=(10, 6),
            squeeze=False,
        )

        idx = 0
        if step_resp is not None:
            ax = axes[idx, 0]
            ax.plot(
                step_resp.time * 1e3, step_resp.output, linewidth=1.5, color="#2563EB"
            )
            ax.set_title(f"Step Response — {args.circuit_type.upper()} Circuit")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            idx += 1

        if imp_resp is not None:
            ax = axes[idx, 0]
            ax.plot(
                imp_resp.time * 1e3, imp_resp.output, linewidth=1.5, color="#DC2626"
            )
            ax.set_title(f"Impulse Response — {args.circuit_type.upper()} Circuit")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if args.save_plot:
            fig.savefig(args.save_plot, dpi=300)
            print(f"Plot saved to '{args.save_plot}'")
        plt.show()

    print("Simulation complete.")


def _handle_analyze(args: argparse.Namespace) -> None:
    """Execute the ``analyze`` subcommand.

    Computes and plots the Bode plot (magnitude & phase) of the selected
    circuit using :func:`~analog_circuit_simulator.analysis.analyze_frequency_response`.

    Args:
        args: Parsed CLI arguments.
    """
    import matplotlib.pyplot as plt

    from .analysis import analyze_frequency_response

    _validate_components(args)
    circuit = _build_circuit(args)
    num, den = circuit.transfer_function()

    resp = analyze_frequency_response(
        num, den, freq_start=args.fmin, freq_stop=args.fmax, num_points=1000
    )

    if args.export:
        resp.to_csv(args.export)
        print(f"Frequency response exported to '{args.export}'")

    if not args.no_plot:
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ax_mag.semilogx(
            resp.frequencies, resp.magnitude_db, linewidth=1.5, color="#2563EB"
        )
        ax_mag.set_title(f"Bode Plot — {args.circuit_type.upper()} Circuit")
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.grid(True, which="both", alpha=0.3)

        ax_phase.semilogx(
            resp.frequencies, resp.phase_deg, linewidth=1.5, color="#059669"
        )
        ax_phase.set_xlabel("Frequency (Hz)")
        ax_phase.set_ylabel("Phase (°)")
        ax_phase.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        if args.save_plot:
            fig.savefig(args.save_plot, dpi=300)
            print(f"Plot saved to '{args.save_plot}'")
        plt.show()

    print("Analysis complete.")


# ─── Entry point ──────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for ``analog-sim``.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    print(_BANNER)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "simulate": _handle_simulate,
        "analyze": _handle_analyze,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
