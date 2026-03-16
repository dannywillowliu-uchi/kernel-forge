"""Full optimization loop orchestrator.

Wires together: agent, tools, knowledge, remote executor, evaluation.
Runs the profile -> diagnose -> strategize -> implement -> validate ->
benchmark -> evaluate loop until convergence or budget exhaustion.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from kernel_forge.agents.claude import ClaudeCodeAgent
from kernel_forge.config import ForgeConfig
from kernel_forge.core.evaluate import (
	classify_failure,
	compute_roofline,
)
from kernel_forge.core.loop import LoopState
from kernel_forge.core.types import (
	Attempt,
	Diagnosis,
	FailureReport,
	FailureType,
	KernelProblem,
	OptimizationGoal,
	ProfileData,
	RooflineAnalysis,
)
from kernel_forge.knowledge.db import KnowledgeDB
from kernel_forge.knowledge.learnings import LearningsManager
from kernel_forge.knowledge.query import KnowledgeQuery
from kernel_forge.remote.executor import CommandResult, Executor
from kernel_forge.remote.gpu_guard import GpuGuard

logger = logging.getLogger(__name__)

HARNESS_DIR = "harness"
KERNELBENCH_DIR = "harness/KernelBench/KernelBench"
KERNELS_DIR = "kernels"


@dataclass
class AttemptResult:
	"""Result of a single optimization attempt."""

	correct: bool = False
	speedup: float = 0.0
	baseline_ms: float = 0.0
	optimized_ms: float = 0.0
	max_abs_diff: float = 0.0
	failure: FailureReport | None = None
	kernel_source: str = ""
	strategy_name: str = ""
	approach_notes: str = ""
	roofline: RooflineAnalysis | None = None


class Orchestrator:
	"""Runs the full kernel optimization loop."""

	def __init__(
		self,
		executor: Executor,
		db: KnowledgeDB,
		learnings: LearningsManager,
		query: KnowledgeQuery,
		config: ForgeConfig,
		agent: ClaudeCodeAgent | None = None,
	) -> None:
		self._executor = executor
		self._db = db
		self._learnings = learnings
		self._query = query
		self._config = config
		self._agent = agent or ClaudeCodeAgent()
		self._gpu_guard = GpuGuard(
			executor,
			gpu_id=config.hardware.gpu_id,
			memory_threshold_mib=config.hardware.gpu_memory_threshold_mib,
		)

	async def _check_gpu(self) -> bool:
		status = await self._gpu_guard.check()
		if not status.available:
			logger.error("GPU %d unavailable: %s", self._config.hardware.gpu_id, status.message)
		return status.available

	def _problem_path(self, problem: KernelProblem) -> str:
		level = problem.difficulty_level or 1
		return f"{KERNELBENCH_DIR}/level{level}/{problem.name}.py"

	def _kernel_path(self, problem: KernelProblem, attempt_num: int) -> str:
		return f"{KERNELS_DIR}/{problem.name}_v{attempt_num}.py"

	async def _run_harness(
		self, cmd: str, timeout: int = 300
	) -> CommandResult:
		full_cmd = f"python3 {HARNESS_DIR}/forge_harness.py {cmd}"
		return await self._executor.run(full_cmd, timeout=timeout)

	async def baseline(self, problem: KernelProblem) -> float:
		"""Benchmark the reference implementation. Returns runtime in ms."""
		prob_path = self._problem_path(problem)
		result = await self._run_harness(f"baseline {prob_path}")
		if not result.success:
			logger.error("Baseline failed: %s", result.stderr)
			return -1.0
		try:
			data = json.loads(result.stdout.strip())
			return float(data["baseline_ms"])
		except (json.JSONDecodeError, KeyError):
			logger.error("Could not parse baseline: %s", result.stdout)
			return -1.0

	async def test_kernel(
		self,
		problem: KernelProblem,
		kernel_source: str,
		attempt_num: int,
		baseline_ms: float = 0.0,
	) -> AttemptResult:
		"""Upload kernel, validate correctness, benchmark if correct."""
		kernel_path = self._kernel_path(problem, attempt_num)
		prob_path = self._problem_path(problem)

		# Upload kernel source to remote via temp file + rsync
		import tempfile
		with tempfile.NamedTemporaryFile(
			mode="w", suffix=".py", delete=False
		) as tmp:
			tmp.write(kernel_source)
			tmp_path = tmp.name

		# Ensure remote dir exists
		await self._executor.run(f"mkdir -p {KERNELS_DIR}")

		# Upload
		remote_path = f"{self._config.hardware.remote_workspace}/{kernel_path}"
		await self._executor.upload(tmp_path, remote_path)

		# Cleanup local temp
		import os
		os.unlink(tmp_path)

		# Verify upload
		verify = await self._executor.run(f"test -f {kernel_path} && echo ok")
		if "ok" not in verify.stdout:
			return AttemptResult(
				failure=FailureReport(
					failure_type=FailureType.COMPILATION_ERROR,
					error_output="Failed to upload kernel to remote",
					kernel_source=kernel_source,
					strategy_name="",
				)
			)

		# Run test (correctness + benchmark)
		baseline_arg = f" --baseline-ms {baseline_ms}" if baseline_ms > 0 else ""
		test_result = await self._run_harness(
			f"test {prob_path} {kernel_path}{baseline_arg}",
			timeout=self._config.hardware.command_timeout_seconds,
		)

		if not test_result.success:
			ft = classify_failure(
				test_result.exit_code,
				test_result.stderr,
				test_result.stdout,
			)
			return AttemptResult(
				failure=FailureReport(
					failure_type=ft,
					error_output=test_result.stderr + "\n" + test_result.stdout,
					kernel_source=kernel_source,
					strategy_name="",
				),
				kernel_source=kernel_source,
			)

		try:
			data = json.loads(test_result.stdout.strip())
		except json.JSONDecodeError:
			return AttemptResult(
				failure=FailureReport(
					failure_type=FailureType.COMPILATION_ERROR,
					error_output=f"Unparseable output: {test_result.stdout}",
					kernel_source=kernel_source,
					strategy_name="",
				),
				kernel_source=kernel_source,
			)

		return AttemptResult(
			correct=data.get("correct", False),
			speedup=data.get("speedup", 0.0),
			baseline_ms=data.get("baseline_ms", 0.0),
			optimized_ms=data.get("optimized_ms", 0.0),
			max_abs_diff=data.get("max_abs_diff", 0.0),
			kernel_source=kernel_source,
		)

	def _estimate_flops(self, problem: KernelProblem) -> float:
		"""Rough FLOPs estimate from problem shapes."""
		shapes = problem.input_shapes
		if not shapes:
			return 0.0
		# For matmul-like: 2*M*N*K
		dims = list(shapes.values())
		if len(dims) >= 2:
			a_shape = dims[0]
			b_shape = dims[1]
			if len(a_shape) == 2 and len(b_shape) == 2:
				m, k = a_shape
				_, n = b_shape
				return 2.0 * m * k * n
		# Fallback: product of first shape dims
		if dims and len(dims[0]) >= 1:
			total = 1
			for d in dims[0]:
				total *= d
			return float(total)
		return 0.0

	def _estimate_bytes(self, problem: KernelProblem) -> float:
		"""Rough bytes estimate from problem shapes."""
		shapes = problem.input_shapes
		if not shapes:
			return 0.0
		total = 0.0
		for shape in shapes.values():
			elems = 1
			for d in shape:
				elems *= d
			total += elems * 4  # FP32
		# Add output (assume same as first input)
		if shapes:
			first = list(shapes.values())[0]
			out_elems = 1
			for d in first:
				out_elems *= d
			total += out_elems * 4
		return total

	async def run(
		self,
		problem: KernelProblem,
		goal: OptimizationGoal,
		run_dir: Path,
	) -> dict:
		"""Execute the full optimization loop.

		Returns a summary dict with best_speedup, attempts, etc.
		"""
		logger.info("=== Starting optimization: %s (goal=%s) ===", problem.name, goal.primary)

		# Check GPU availability
		if not await self._check_gpu():
			return {"error": "GPU not available", "best_speedup": 0.0}

		# Step 1: BASELINE
		logger.info("[BASELINE] Benchmarking reference implementation...")
		baseline_ms = await self.baseline(problem)
		if baseline_ms < 0:
			return {"error": "Baseline benchmark failed", "best_speedup": 0.0}
		logger.info("[BASELINE] Reference: %.4f ms", baseline_ms)

		# Initialize loop state
		state = LoopState(
			problem=problem, goal=goal, config=self._config
		)
		state.best_speedup = 1.0

		# Open attempts log
		attempts_log = run_dir / "attempts.jsonl"

		# Roofline for baseline
		flops = self._estimate_flops(problem)
		bytes_moved = self._estimate_bytes(problem)
		if flops > 0 and bytes_moved > 0:
			baseline_roofline = compute_roofline(
				baseline_ms, flops, bytes_moved, precision="fp32"
			)
			logger.info(
				"[ROOFLINE] Baseline: %.1f TFLOPS, %.1f%% of FP32 peak",
				baseline_roofline.achieved_tflops, baseline_roofline.utilization_pct,
			)

		# Step 2: ANALYZE -- query knowledge base
		kernel_type = problem.name.split("_")[0] if "_" in problem.name else problem.name
		knowledge_ctx = await self._query.build_context(
			kernel_problem=problem.name,
			kernel_type=kernel_type,
			bottleneck_type="",
			max_tokens=8000,
		)

		# Initial diagnosis (before any profiling, infer from problem)
		diagnosis: Diagnosis | None = None
		current_strategy = "tf32_tensor_cores"  # default first strategy

		# === MAIN LOOP ===
		while not state.should_stop:
			attempt_num = state.attempt_count + 1
			logger.info(
				"\n[ATTEMPT %d] strategy=%s, best_so_far=%.3fx",
				attempt_num, current_strategy, state.best_speedup,
			)

			# Step 5: GENERATE kernel via agent
			logger.info("[GENERATE] Invoking Claude agent...")
			candidate = await self._agent.generate_kernel(
				problem=problem,
				goal=goal,
				diagnosis=diagnosis,
				strategy_name=current_strategy,
				prior_attempts=state.attempts,
				knowledge_context=knowledge_ctx,
			)

			if candidate is None:
				logger.warning("[GENERATE] Agent failed to produce kernel")
				state.record_attempt(0.0, False, 0.0)
				state.attempts.append(Attempt(
					kernel_problem=problem.name,
					strategy_name=current_strategy,
					speedup=0.0,
					correct=False,
					hardware="b200",
					optimization_goal=goal.primary,
					failure_report=FailureReport(
						failure_type=FailureType.COMPILATION_ERROR,
						error_output="Agent returned no kernel",
						kernel_source="",
						strategy_name=current_strategy,
					),
				))
				continue

			# Step 6-8: TEST (upload, validate, benchmark)
			logger.info("[TEST] Testing kernel on GPU %d...", self._config.hardware.gpu_id)
			if not await self._check_gpu():
				logger.error("GPU became unavailable mid-run")
				break

			result = await self.test_kernel(
				problem, candidate.source, attempt_num, baseline_ms=baseline_ms
			)
			result.strategy_name = current_strategy
			result.approach_notes = candidate.approach_notes

			# Record attempt
			attempt = Attempt(
				kernel_problem=problem.name,
				strategy_name=current_strategy,
				speedup=result.speedup,
				correct=result.correct,
				hardware="b200",
				optimization_goal=goal.primary,
				kernel_source_hash=LoopState.kernel_hash(candidate.source),
				failure_report=result.failure,
			)
			state.attempts.append(attempt)
			state.record_attempt(result.speedup, result.correct, 0.0)
			await self._db.insert_attempt(attempt)

			# Log attempt
			log_entry = {
				"attempt": attempt_num,
				"strategy": current_strategy,
				"correct": result.correct,
				"speedup": result.speedup,
				"baseline_ms": result.baseline_ms,
				"optimized_ms": result.optimized_ms,
				"approach": candidate.approach_notes,
			}
			if result.failure:
				log_entry["failure_type"] = result.failure.failure_type.value
				log_entry["error"] = result.failure.error_output[:500]

			with open(attempts_log, "a") as f:
				f.write(json.dumps(log_entry) + "\n")

			# Step 9: EVALUATE
			if result.correct:
				logger.info(
					"[RESULT] CORRECT: %.4f ms (%.3fx speedup)",
					result.optimized_ms, result.speedup,
				)

				# Roofline analysis
				if flops > 0 and result.optimized_ms > 0:
					roofline = compute_roofline(
						result.optimized_ms, flops, bytes_moved, precision="tf32"
					)
					result.roofline = roofline
					logger.info(
						"[ROOFLINE] %.1f TFLOPS, %.1f%% utilization, %s",
						roofline.achieved_tflops,
						roofline.utilization_pct,
						"WORTH PUSHING" if roofline.worth_optimizing else "NEAR OPTIMAL",
					)
					logger.info("[ROOFLINE] %s", roofline.explanation)

					if not roofline.worth_optimizing:
						logger.info("[CONVERGED] Near peak utilization, stopping.")
						break

				# Save best kernel
				if result.speedup >= state.best_speedup:
					state.best_kernel_source = candidate.source
					(run_dir / "best_kernel.py").write_text(candidate.source)
			else:
				failure_desc = ""
				if result.failure:
					failure_desc = f" ({result.failure.failure_type.value})"
				logger.info("[RESULT] FAILED%s", failure_desc)

			# Step 4: DIAGNOSE for next iteration
			if result.correct and result.optimized_ms > 0:
				profile = ProfileData(
					runtime_us=result.optimized_ms * 1000,
					profiling_tier=state.profiling_tier,
					metrics={
						"speedup": result.speedup,
						"baseline_ms": result.baseline_ms,
						"optimized_ms": result.optimized_ms,
					},
				)
				diagnosis = await self._agent.diagnose_bottleneck(
					profile=profile,
					kernel_source=candidate.source,
					problem=problem,
				)
				if diagnosis:
					logger.info(
						"[DIAGNOSE] %s: %s",
						diagnosis.bottleneck_type.value,
						diagnosis.explanation[:100],
					)

			# Step 5: STRATEGIZE for next iteration
			if diagnosis:
				strategies = await self._db.get_strategies_for_bottleneck(
					diagnosis.bottleneck_type
				)
				if strategies:
					suggested = await self._agent.suggest_strategies(
						diagnosis=diagnosis,
						available_strategies=strategies,
						prior_attempts=state.attempts,
					)
					if suggested:
						current_strategy = suggested[0]
						logger.info("[STRATEGIZE] Next: %s", current_strategy)

			# Check escalation
			if state.should_escalate and state.profiling_tier == "cuda_events":
				state.profiling_tier = "ncu"
				logger.info("[ESCALATE] Switching to ncu profiling tier")

		# === REPORT ===
		logger.info("\n=== Optimization complete ===")
		logger.info("Attempts: %d", state.attempt_count)
		logger.info("Best speedup: %.3fx", state.best_speedup)
		logger.info("Total cost: $%.2f", state.total_cost)

		summary = {
			"problem": problem.name,
			"goal": goal.primary,
			"best_speedup": state.best_speedup,
			"total_attempts": state.attempt_count,
			"total_cost_usd": state.total_cost,
			"elapsed_seconds": state.elapsed_seconds,
		}
		(run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

		# Record learnings
		if state.best_speedup > 1.0 and state.best_kernel_source:
			self._learnings.write(
				"insights",
				f"Problem {problem.name}: achieved {state.best_speedup:.2f}x speedup "
				f"in {state.attempt_count} attempts. "
				f"Best strategy landed on after trying {state.attempt_count} approaches.",
				problem.name,
			)

		return summary
