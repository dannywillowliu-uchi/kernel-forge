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
@click.option(
	"--goal",
	default="latency",
	type=click.Choice(["latency", "throughput", "memory", "balanced"]),
	help="Optimization goal.",
)
@click.option(
	"--config",
	"config_path",
	default=None,
	type=click.Path(exists=True, path_type=Path),
	help="Path to TOML config file.",
)
@click.option(
	"--dry-run",
	is_flag=True,
	default=False,
	help="Run in dry-run mode without GPU access.",
)
@click.option(
	"--difficulty",
	default=None,
	type=int,
	help="Filter problems by difficulty level.",
)
@click.option(
	"--max-attempts",
	default=None,
	type=int,
	help="Override max attempts per problem.",
)
def optimize(
	problem_name: str,
	goal: str,
	config_path: Path | None,
	dry_run: bool,
	difficulty: int | None,
	max_attempts: int | None,
) -> None:
	"""Optimize a kernel problem for maximum performance."""
	import asyncio
	asyncio.run(_run_optimize(
		problem_name, goal, config_path, dry_run, difficulty, max_attempts,
	))


async def _run_optimize(
	problem_name: str,
	goal: str,
	config_path: Path | None,
	dry_run: bool,
	difficulty: int | None,
	max_attempts: int | None,
) -> None:
	"""Async implementation of optimize command."""
	import logging

	from kernel_forge.agents.claude import ClaudeCodeAgent
	from kernel_forge.config import default_config, load_config
	from kernel_forge.core.orchestrator import Orchestrator
	from kernel_forge.core.types import KernelProblem, OptimizationGoal
	from kernel_forge.knowledge.db import KnowledgeDB
	from kernel_forge.knowledge.learnings import LearningsManager
	from kernel_forge.knowledge.query import KnowledgeQuery
	from kernel_forge.remote.dry_run import DryRunExecutor
	from kernel_forge.remote.ssh import SSHExecutor

	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
		datefmt="%H:%M:%S",
	)

	# Config
	if config_path:
		config = load_config(config_path)
	else:
		config = default_config()
	config.dry_run = dry_run
	if max_attempts:
		config.termination.max_attempts = max_attempts

	# Executor
	if dry_run:
		executor = DryRunExecutor()
	else:
		executor = SSHExecutor(
			ssh_host=config.hardware.ssh_host,
			ssh_user=config.hardware.ssh_user,
			remote_workspace=config.hardware.remote_workspace,
			cuda_visible_devices=config.hardware.cuda_visible_devices,
		)

	# DB
	config.db_path.parent.mkdir(parents=True, exist_ok=True)
	db = KnowledgeDB(config.db_path)
	await db.initialize()

	# Knowledge
	learnings = LearningsManager(config.knowledge_dir)
	query = KnowledgeQuery(db, learnings)

	# Agent
	agent = ClaudeCodeAgent(model="sonnet")

	# Build problem
	problem = KernelProblem(
		name=problem_name,
		reference_source="",
		input_shapes={},
		benchmark_suite="kernelbench",
		difficulty_level=difficulty or 1,
	)

	opt_goal = OptimizationGoal(primary=goal)

	# Prepare run dir
	from datetime import datetime, timezone
	timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
	run_dir = config.runs_dir / f"{problem_name}_{timestamp}"
	run_dir.mkdir(parents=True, exist_ok=True)

	# Run
	orchestrator = Orchestrator(
		executor=executor,
		db=db,
		learnings=learnings,
		query=query,
		config=config,
		agent=agent,
	)

	try:
		summary = await orchestrator.run(problem, opt_goal, run_dir)
		click.echo(f"\nBest speedup: {summary.get('best_speedup', 0):.3f}x")
		click.echo(f"Total attempts: {summary.get('total_attempts', 0)}")
		click.echo(f"Run dir: {run_dir}")
	finally:
		await db.close()


@main.command("list-problems")
@click.option(
	"--difficulty",
	default=None,
	type=int,
	help="Filter by difficulty level.",
)
@click.option(
	"--problems-dir",
	default=None,
	type=click.Path(exists=True, path_type=Path),
	help="Path to KernelBench problems directory.",
)
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
