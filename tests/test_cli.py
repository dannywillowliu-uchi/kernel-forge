"""Tests for the CLI entry point."""

from __future__ import annotations

from click.testing import CliRunner

from kernel_forge import __version__
from kernel_forge.cli import main


class TestCLI:
	def test_help(self) -> None:
		runner = CliRunner()
		result = runner.invoke(main, ["--help"])
		assert result.exit_code == 0
		assert "Kernel Forge" in result.output
		assert "optimize" in result.output
		assert "list-problems" in result.output
		assert "report" in result.output

	def test_version(self) -> None:
		runner = CliRunner()
		result = runner.invoke(main, ["--version"])
		assert result.exit_code == 0
		assert __version__ in result.output

	def test_optimize_help(self) -> None:
		runner = CliRunner()
		result = runner.invoke(main, ["optimize", "--help"])
		assert result.exit_code == 0
		assert "PROBLEM_NAME" in result.output
		assert "--goal" in result.output
		assert "--dry-run" in result.output

	def test_optimize_dry_run(self) -> None:
		runner = CliRunner()
		result = runner.invoke(main, [
			"optimize", "matmul_basic", "--dry-run", "--max-attempts", "1",
		])
		# Dry run executes the loop with DryRunExecutor; check it completes
		assert result.exit_code == 0

	def test_list_problems_help(self) -> None:
		runner = CliRunner()
		result = runner.invoke(main, ["list-problems", "--help"])
		assert result.exit_code == 0
		assert "--difficulty" in result.output
		assert "--problems-dir" in result.output

	def test_list_problems_no_dir(self) -> None:
		runner = CliRunner()
		result = runner.invoke(main, ["list-problems"])
		assert result.exit_code == 0
		assert "No problems directory" in result.output

	def test_list_problems_with_dir(self, tmp_path: object) -> None:
		"""list-problems with a real directory containing sample problems."""
		from pathlib import Path

		assert isinstance(tmp_path, Path)
		level1 = tmp_path / "level1"
		level1.mkdir()
		(level1 / "matmul.py").write_text(
			"class Model: pass\n"
			"def get_inputs(): return []\n"
			"def get_init_inputs(): return []\n"
		)
		runner = CliRunner()
		result = runner.invoke(main, ["list-problems", "--problems-dir", str(tmp_path)])
		assert result.exit_code == 0
		assert "matmul" in result.output
		assert "Total: 1" in result.output

	def test_report_help(self) -> None:
		runner = CliRunner()
		result = runner.invoke(main, ["report", "--help"])
		assert result.exit_code == 0
		assert "PROBLEM_NAME" in result.output

	def test_report_command(self) -> None:
		runner = CliRunner()
		result = runner.invoke(main, ["report", "matmul_basic"])
		assert result.exit_code == 0
		assert "matmul_basic" in result.output
