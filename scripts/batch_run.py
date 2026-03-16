#!/usr/bin/env python3
"""Batch run kernel-forge across multiple KernelBench problems.

Usage:
	uv run python scripts/batch_run.py --problems 1,2,19,23,26,50,67,80,90,100
	uv run python scripts/batch_run.py --level 1 --count 10
	uv run python scripts/batch_run.py --level 1 --all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kernel_forge.agents.claude import ClaudeCodeAgent
from kernel_forge.config import ForgeConfig, default_config
from kernel_forge.core.orchestrator import Orchestrator
from kernel_forge.core.types import OptimizationGoal
from kernel_forge.harness.kernelbench import KernelBenchAdapter
from kernel_forge.knowledge.db import KnowledgeDB
from kernel_forge.knowledge.learnings import LearningsManager
from kernel_forge.knowledge.query import KnowledgeQuery
from kernel_forge.remote.ssh import SSHExecutor


async def run_batch(
	problem_names: list[str],
	max_attempts: int = 5,
	goal: str = "latency",
) -> dict:
	config = default_config()
	config.termination.max_attempts = max_attempts

	executor = SSHExecutor(
		ssh_host=config.hardware.ssh_host,
		ssh_user=config.hardware.ssh_user,
		remote_workspace=config.hardware.remote_workspace,
		cuda_visible_devices=config.hardware.cuda_visible_devices,
	)

	config.db_path.parent.mkdir(parents=True, exist_ok=True)
	db = KnowledgeDB(config.db_path)
	await db.initialize()

	learnings = LearningsManager(config.knowledge_dir)
	query = KnowledgeQuery(db, learnings)
	agent = ClaudeCodeAgent(model="sonnet")
	adapter = KernelBenchAdapter(config.knowledge_dir / "kernelbench")

	results: list[dict] = []
	batch_start = time.time()

	for i, name in enumerate(problem_names):
		logging.info(
			"\n{'='*60}\n[BATCH %d/%d] %s\n{'='*60}",
			i + 1, len(problem_names), name,
		)

		problem = adapter.get_problem(name)
		if problem is None:
			logging.error("Problem not found: %s", name)
			results.append({"problem": name, "error": "not found"})
			continue

		opt_goal = OptimizationGoal(primary=goal)
		timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
		run_dir = config.runs_dir / f"{name}_{timestamp}"
		run_dir.mkdir(parents=True, exist_ok=True)

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
			results.append(summary)
			logging.info(
				"[BATCH RESULT] %s: %.3fx speedup (%d attempts)",
				name,
				summary.get("best_speedup", 0),
				summary.get("total_attempts", 0),
			)
		except Exception as e:
			logging.error("[BATCH ERROR] %s: %s", name, e)
			results.append({"problem": name, "error": str(e)})

		# Clean up remote kernels between problems
		await executor.run(f"rm -rf kernels/{name}_*")

	await db.close()

	# Compute fast_p metrics
	batch_time = time.time() - batch_start
	correct_results = [r for r in results if r.get("best_speedup", 0) > 0]
	total = len(problem_names)

	fast_1 = sum(1 for r in correct_results if r["best_speedup"] > 1.0) / total
	fast_2 = sum(1 for r in correct_results if r["best_speedup"] > 2.0) / total
	fast_5 = sum(1 for r in correct_results if r["best_speedup"] > 5.0) / total
	fast_10 = sum(1 for r in correct_results if r["best_speedup"] > 10.0) / total

	report = {
		"total_problems": total,
		"correct": len(correct_results),
		"failed": total - len(correct_results),
		"fast_1": round(fast_1, 3),
		"fast_2": round(fast_2, 3),
		"fast_5": round(fast_5, 3),
		"fast_10": round(fast_10, 3),
		"avg_speedup": round(
			sum(r["best_speedup"] for r in correct_results) / max(len(correct_results), 1),
			3,
		),
		"batch_time_seconds": round(batch_time, 1),
		"results": results,
	}

	# Save report
	report_path = Path("runs") / f"batch_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
	report_path.parent.mkdir(parents=True, exist_ok=True)
	report_path.write_text(json.dumps(report, indent=2))

	print(f"\n{'='*60}")
	print(f"BATCH RESULTS: {total} problems")
	print(f"{'='*60}")
	print(f"Correct: {len(correct_results)}/{total}")
	print(f"fast_1  (>1x):  {fast_1:.1%}")
	print(f"fast_2  (>2x):  {fast_2:.1%}")
	print(f"fast_5  (>5x):  {fast_5:.1%}")
	print(f"fast_10 (>10x): {fast_10:.1%}")
	print(f"Avg speedup: {report['avg_speedup']:.2f}x")
	print(f"Total time: {batch_time/60:.1f} min")
	print(f"Report: {report_path}")

	for r in results:
		name = r.get("problem", "?")
		speedup = r.get("best_speedup", 0)
		attempts = r.get("total_attempts", 0)
		err = r.get("error", "")
		if err:
			print(f"  {name}: ERROR - {err}")
		elif speedup > 1.0:
			print(f"  {name}: {speedup:.2f}x ({attempts} attempts)")
		else:
			print(f"  {name}: no speedup ({attempts} attempts)")

	return report


if __name__ == "__main__":
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s: %(message)s",
		datefmt="%H:%M:%S",
	)

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--problems", type=str, default=None,
		help="Comma-separated problem names (stems without .py)",
	)
	parser.add_argument("--level", type=int, default=1)
	parser.add_argument("--count", type=int, default=10)
	parser.add_argument("--all", action="store_true")
	parser.add_argument("--max-attempts", type=int, default=3)
	parser.add_argument("--goal", default="latency")
	args = parser.parse_args()

	if args.problems:
		names = [n.strip() for n in args.problems.split(",")]
	else:
		adapter = KernelBenchAdapter(Path("knowledge/kernelbench"))
		problems = adapter.list_problems(difficulty=args.level)
		if args.all:
			names = [p.name for p in problems]
		else:
			# Pick a diverse sample
			step = max(1, len(problems) // args.count)
			names = [p.name for p in problems[::step]][:args.count]

	print(f"Running {len(names)} problems with max {args.max_attempts} attempts each")
	print(f"Problems: {names}")

	asyncio.run(run_batch(names, max_attempts=args.max_attempts, goal=args.goal))
