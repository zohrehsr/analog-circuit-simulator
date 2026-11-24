"""Tests for the command-line interface (CLI)."""

from __future__ import annotations

import pytest

from analog_circuit_simulator.cli import main


class TestCLI:
    def test_cli_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Analog Circuit Simulator" in captured.out

    def test_cli_analyze_rc_with_export(
        self, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        export_file = tmp_path / "bode_out.csv"
        main(
            [
                "analyze",
                "rc",
                "-R",
                "1000",
                "-C",
                "1e-6",
                "--no-plot",
                "--export",
                str(export_file),
            ]
        )
        assert export_file.exists()
        content = export_file.read_text()
        assert "frequency_hz,magnitude_db,phase_deg" in content
        captured = capsys.readouterr()
        assert "Analysis complete." in captured.out

    def test_cli_simulate_rlc_with_export(
        self, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        export_file = tmp_path / "sim_out.csv"
        main(
            [
                "simulate",
                "rlc",
                "-R",
                "100",
                "-L",
                "10e-3",
                "-C",
                "1e-6",
                "--response",
                "both",
                "--no-plot",
                "--export",
                str(export_file),
            ]
        )
        assert export_file.exists()
        content = export_file.read_text()
        assert "time_s,step_output,impulse_output" in content
        captured = capsys.readouterr()
        assert "Simulation complete." in captured.out

    def test_cli_missing_component_raises(self) -> None:
        with pytest.raises(SystemExit):
            main(["analyze", "rc", "-R", "1000"])  # missing -C
