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

	# Agent
	agent = ClaudeCodeAgent(model="opus")

	# Load problem
	from kernel_forge.harness.kernelbench import KernelBenchAdapter
	adapter = KernelBenchAdapter(config.knowledge_dir / "kernelbench")
	problem = adapter.get_problem(problem_name)
	if problem is None:
		problem = KernelProblem(
			name=problem_name,
			reference_source="",
			input_shapes={},
			benchmark_suite="kernelbench",
			difficulty_level=difficulty or 1,
		)
	if difficulty is not None:
		problem.difficulty_level = difficulty

	opt_goal = OptimizationGoal(primary=goal)

	# Prepare run dir
	from datetime import datetime, timezone
	timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
	run_dir = config.runs_dir / f"{problem_name}_{timestamp}"
	run_dir.mkdir(parents=True, exist_ok=True)

	# Run
	orchestrator = Orchestrator(
		executor=executor,
		config=config,
		agent=agent,
	)

	summary = await orchestrator.run(problem, opt_goal, run_dir)
	click.echo(f"\nBest speedup: {summary.get('best_speedup', 0):.3f}x")
	click.echo(f"Approach: {summary.get('approach', 'N/A')}")
	click.echo(f"Run dir: {run_dir}")


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


@main.command()
@click.argument("problem_name")
@click.option("--gpu", default=3, help="GPU ID on B200.")
@click.option(
	"--problem-file", default=None,
	help="Path to problem file (on B200 or local).",
)
@click.option(
	"--instructions", default="",
	help="Additional instructions for the agent.",
)
@click.option(
	"--config", "config_path", default=None,
	type=click.Path(exists=True, path_type=Path),
)
@click.option(
	"--print-only", is_flag=True,
	help="Print the prompt instead of saving to file.",
)
def solve(
	problem_name: str,
	gpu: int,
	problem_file: str | None,
	instructions: str,
	config_path: Path | None,
	print_only: bool,
) -> None:
	"""Generate a complete optimization prompt for any kernel problem.

	The output prompt can be used as a subagent prompt in Claude Code,
	or run via: claude -p "$(cat prompt.md)"

	Examples:
	    kernel-forge solve 36_RMSNorm_ --gpu 3
	    kernel-forge solve my_problem --problem-file path/to/kernel.py
	"""
	from kernel_forge.config import default_config, load_config
	from kernel_forge.solve import build_solve_prompt, save_prompt

	if config_path:
		config = load_config(config_path)
	else:
		config = default_config()

	# Try to load problem source from KernelBench
	problem_source = ""
	if not problem_file:
		from kernel_forge.harness.kernelbench import KernelBenchAdapter
		adapter = KernelBenchAdapter(
			config.knowledge_dir / "kernelbench"
		)
		problem = adapter.get_problem(problem_name)
		if problem:
			problem_source = problem.reference_source

	prompt = build_solve_prompt(
		problem_name=problem_name,
		problem_source=problem_source,
		problem_file=problem_file or "",
		gpu_id=gpu,
		config=config,
		custom_instructions=instructions,
	)

	if print_only:
		click.echo(prompt)
	else:
		path = save_prompt(prompt, problem_name)
		click.echo(f"Prompt saved to: {path}")
		click.echo(f"Prompt size: {len(prompt)} chars")
		click.echo("")
		click.echo("To run as subagent in Claude Code:")
		click.echo(
			f"  Use Agent tool with the content of {path}"
		)
		click.echo("")
		click.echo("To run via CLI:")
		click.echo(
			f"  claude --permission-mode bypassPermissions "
			f"-p \"$(cat {path})\" --model opus "
			f"--max-turns 30 --output-format text"
		)


@main.command()
def scorecard() -> None:
	"""Show the evaluation scorecard vs baselines."""
	from kernel_forge.config import default_config
	from kernel_forge.eval.scorecard import (
		compute_scorecard,
		format_scorecard,
		load_baselines,
		load_our_results,
	)

	config = default_config()
	baselines = load_baselines(
		config.knowledge_dir / "baselines_b200.json"
	)
	results = load_our_results(
		Path("knowledge/experience/records.jsonl")
	)
	card = compute_scorecard(baselines, results)
	click.echo(format_scorecard(card))
