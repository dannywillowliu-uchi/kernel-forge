"""Orchestrator: manages budget, experience, and agent lifecycle.

The agent does the actual optimization (profiling, kernel generation,
benchmarking, iteration). The orchestrator handles:
- Problem setup (baseline, trait analysis)
- Experience loading and recording
- Budget/termination management
- Run directory and logging
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from kernel_forge.agents.claude import ClaudeCodeAgent
from kernel_forge.config import ForgeConfig
from kernel_forge.core.types import (
	KernelProblem,
	OptimizationGoal,
)
from kernel_forge.knowledge.classifier import analyze_traits
from kernel_forge.knowledge.experience import ExperienceRecord, ExperienceStore
from kernel_forge.knowledge.learnings import LearningsManager
from kernel_forge.remote.executor import CommandResult, Executor
from kernel_forge.remote.gpu_guard import GpuGuard

logger = logging.getLogger(__name__)

HARNESS_DIR = "harness"
KERNELBENCH_DIR = "harness/KernelBench/KernelBench"


class Orchestrator:
	"""Thin wrapper that sets up context and lets the agent optimize."""

	def __init__(
		self,
		executor: Executor,
		config: ForgeConfig,
		agent: ClaudeCodeAgent | None = None,
		experience: ExperienceStore | None = None,
		learnings: LearningsManager | None = None,
	) -> None:
		self._executor = executor
		self._config = config
		self._agent = agent or ClaudeCodeAgent()
		self._experience = experience or ExperienceStore(
			config.knowledge_dir / "experience"
		)
		self._learnings = learnings or LearningsManager(config.knowledge_dir)
		self._gpu_guard = GpuGuard(
			executor,
			gpu_id=config.hardware.gpu_id,
			memory_threshold_mib=config.hardware.gpu_memory_threshold_mib,
		)

	async def _check_gpu(self) -> bool:
		status = await self._gpu_guard.check()
		if not status.available:
			logger.error(
				"GPU %d unavailable: %s",
				self._config.hardware.gpu_id,
				status.message,
			)
		return status.available

	async def _run_harness(self, cmd: str) -> CommandResult:
		full_cmd = f"python3 {HARNESS_DIR}/forge_harness.py {cmd}"
		return await self._executor.run(full_cmd, timeout=300)

	def _problem_path(self, problem: KernelProblem) -> str:
		level = problem.difficulty_level or 1
		return f"{KERNELBENCH_DIR}/level{level}/{problem.name}.py"

	async def run(
		self,
		problem: KernelProblem,
		goal: OptimizationGoal,
		run_dir: Path,
	) -> dict:
		"""Run the agent on a kernel problem.

		1. Check GPU, baseline, analyze traits
		2. Load experience as advisory context
		3. Let the agent optimize autonomously
		4. Record experience from the run
		"""
		logger.info(
			"=== %s (goal=%s) ===",
			problem.name, goal.primary,
		)

		# GPU check
		if not await self._check_gpu():
			return {"problem": problem.name, "error": "GPU unavailable"}

		# Baseline
		logger.info("[BASELINE] Benchmarking reference...")
		prob_path = self._problem_path(problem)
		result = await self._run_harness(f"baseline {prob_path}")
		if not result.success:
			logger.error("Baseline failed: %s", result.stderr)
			return {"problem": problem.name, "error": "Baseline failed"}

		try:
			baseline_ms = float(json.loads(result.stdout.strip())["baseline_ms"])
		except (json.JSONDecodeError, KeyError):
			logger.error("Could not parse baseline: %s", result.stdout)
			return {"problem": problem.name, "error": "Baseline parse failed"}

		logger.info("[BASELINE] %.4f ms", baseline_ms)

		# Trait analysis
		traits = analyze_traits(
			problem.name,
			problem.reference_source,
			problem.input_shapes,
		)
		logger.info("[TRAITS] %s", traits.summary())

		# Experience (advisory)
		experience_ctx = self._experience.build_advisory_context(
			traits, max_tokens=4000
		)

		# Let the agent optimize
		logger.info("[AGENT] Launching autonomous optimization...")
		agent_result = await self._agent.optimize(
			problem_name=problem.name,
			problem_source=problem.reference_source,
			baseline_ms=baseline_ms,
			experience_context=experience_ctx,
			traits_summary=traits.summary(),
		)

		# Save agent output
		(run_dir / "agent_output.txt").write_text(agent_result.raw_output)

		# Log result
		if agent_result.success:
			logger.info(
				"[RESULT] %.3fx speedup via: %s",
				agent_result.speedup,
				agent_result.approach,
			)
		else:
			logger.info("[RESULT] No speedup achieved")

		# Record experience
		self._experience.record(ExperienceRecord(
			problem_name=problem.name,
			dominant_ops=traits.dominant_ops,
			strategy_name=agent_result.approach[:50] if agent_result.approach else "unknown",
			approach_notes=agent_result.approach,
			outcome="success" if agent_result.success else "no_improvement",
			speedup=agent_result.speedup,
			baseline_ms=baseline_ms,
			optimized_ms=(
				baseline_ms / agent_result.speedup
				if agent_result.speedup > 0 else 0
			),
			bottleneck_type=traits.estimated_bottleneck,
			roofline_utilization_pct=0.0,
			root_cause=agent_result.approach,
			has_data_reuse=traits.has_data_reuse,
			shape_category=traits.shape_category,
			estimated_bottleneck=traits.estimated_bottleneck,
			input_shapes=problem.input_shapes,
		))

		# Summary
		summary = {
			"problem": problem.name,
			"goal": goal.primary,
			"best_speedup": agent_result.speedup,
			"baseline_ms": baseline_ms,
			"approach": agent_result.approach,
			"kernel_path": agent_result.kernel_path,
			"success": agent_result.success,
		}
		(run_dir / "summary.json").write_text(
			json.dumps(summary, indent=2)
		)

		return summary
