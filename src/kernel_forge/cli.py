"""CLI entry point for kernel-forge."""

from __future__ import annotations

from pathlib import Path

import click

from kernel_forge import __version__


@click.group()
@click.version_option(version=__version__, prog_name="kernel-forge")
def main() -> None:
	"""Kernel Forge: Autonomous GPU kernel optimization system."""


@main.command()
@click.argument("problem_name")
@click.option("--goal", default="latency", type=click.Choice(["latency", "throughput", "memory", "balanced"]), help="Optimization goal.")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True, path_type=Path), help="Path to TOML config file.")
@click.option("--dry-run", is_flag=True, default=False, help="Run in dry-run mode without GPU access.")
@click.option("--difficulty", default=None, type=int, help="Filter problems by difficulty level.")
def optimize(
	problem_name: str,
	goal: str,
	config_path: Path | None,
	dry_run: bool,
	difficulty: int | None,
) -> None:
	"""Optimize a kernel problem for maximum performance."""
	click.echo(f"Optimizing {problem_name} for {goal} (dry_run={dry_run})")


@main.command("list-problems")
@click.option("--difficulty", default=None, type=int, help="Filter by difficulty level.")
@click.option("--problems-dir", default=None, type=click.Path(exists=True, path_type=Path), help="Path to KernelBench problems directory.")
def list_problems(
	difficulty: int | None,
	problems_dir: Path | None,
) -> None:
	"""List available KernelBench problems."""
	if problems_dir is None:
		click.echo("No problems directory specified. Use --problems-dir.")
		return

	from kernel_forge.harness.kernelbench import KernelBenchAdapter

	adapter = KernelBenchAdapter(problems_dir)
	problems = adapter.list_problems(difficulty=difficulty)

	if not problems:
		click.echo("No problems found.")
		return

	for p in problems:
		level = f"L{p.difficulty_level}" if p.difficulty_level else "L?"
		click.echo(f"  [{level}] {p.name}")

	click.echo(f"\nTotal: {len(problems)} problems")


@main.command()
@click.argument("problem_name")
def report(problem_name: str) -> None:
	"""Show optimization report for a problem."""
	click.echo(f"Report for {problem_name} (not yet implemented)")
