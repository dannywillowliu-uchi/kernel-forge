"""Orchestrator: manages budget, experience, telemetry, and agent lifecycle.

The agent does the actual optimization (profiling, kernel generation,
benchmarking, iteration). The orchestrator handles:
- Problem setup (baseline, trait analysis)
- Experience loading and recording
- Telemetry tracking (timing, tokens, GPU time)
- Run directory and logging
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from kernel_forge.agents.claude import AgentResult, ClaudeCodeAgent
from kernel_forge.config import ForgeConfig
from kernel_forge.core.telemetry import RunTracker
from kernel_forge.core.types import (
	KernelProblem,
	OptimizationGoal,
)
from kernel_forge.eval.scorecard import get_gap_context_for_problem, load_baselines
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
		self._agent = agent or ClaudeCodeAgent(
			model=config.agent.model,
			max_turns=config.agent.max_turns,
		)
		self._experience = experience or ExperienceStore(
			config.experience.store_path
		)
		self._learnings = learnings or LearningsManager(
			config.knowledge_dir
		)
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
		"""Run the agent on a kernel problem with full telemetry."""
		tracker = RunTracker(problem.name)

		logger.info(
			"=== %s (goal=%s) ===",
			problem.name, goal.primary,
		)

		# GPU check
		async with tracker.span("gpu_check"):
			if not await self._check_gpu():
				return {
					"problem": problem.name,
					"error": "GPU unavailable",
				}

		# Baseline
		async with tracker.span("baseline") as s:
			logger.info("[BASELINE] Benchmarking reference...")
			prob_path = self._problem_path(problem)
			result = await self._run_harness(
				f"baseline {prob_path}"
			)
			if not result.success:
				logger.error("Baseline failed: %s", result.stderr)
				return {
					"problem": problem.name,
					"error": "Baseline failed",
				}
			try:
				baseline_ms = float(
					json.loads(result.stdout.strip())["baseline_ms"]
				)
			except (json.JSONDecodeError, KeyError):
				logger.error(
					"Could not parse baseline: %s", result.stdout
				)
				return {
					"problem": problem.name,
					"error": "Baseline parse failed",
				}
			s.set("baseline_ms", baseline_ms)
			tracker.record_gpu_time(baseline_ms)

		logger.info("[BASELINE] %.4f ms", baseline_ms)

		# Trait analysis
		async with tracker.span("trait_analysis") as s:
			traits = analyze_traits(
				problem.name,
				problem.reference_source,
				problem.input_shapes,
			)
			s.set("traits", traits.summary())

		logger.info("[TRAITS] %s", traits.summary())

		# Roofline analysis on baseline
		async with tracker.span("roofline_analysis") as s:
			roofline_ctx = self._compute_roofline_context(
				problem, traits, baseline_ms
			)
			s.set("roofline", roofline_ctx[:200])

		if roofline_ctx:
			logger.info("[ROOFLINE] %s", roofline_ctx[:200])

		# Load distilled knowledge for this op type
		async with tracker.span("load_knowledge") as s:
			distilled_guide, triton_examples = (
				self._load_distilled_knowledge(traits)
			)
			s.set("guide_chars", len(distilled_guide))
			s.set("examples_chars", len(triton_examples))
			if distilled_guide:
				logger.info(
					"[KNOWLEDGE] Loaded distilled guide "
					"(%d chars) + examples (%d chars)",
					len(distilled_guide),
					len(triton_examples),
				)

		# Load baselines for gap context
		async with tracker.span("load_baselines") as s:
			baselines_path = (
				self._config.knowledge_dir / "baselines_b200.json"
			)
			baselines = load_baselines(baselines_path)
			gap_ctx = get_gap_context_for_problem(
				problem.name, baselines
			)
			s.set("has_gap_context", bool(gap_ctx))
			if gap_ctx:
				logger.info("[GAP] %s", gap_ctx[:150])

		# Experience (advisory)
		async with tracker.span("experience_query") as s:
			experience_ctx = self._experience.build_advisory_context(
				traits,
				max_tokens=self._config.experience.max_context_tokens,
			)
			s.set("has_context", bool(experience_ctx))

		# Agent optimization -- full context injection
		async with tracker.span("agent_optimize") as s:
			logger.info("[AGENT] Launching autonomous optimization...")

			# Combine roofline + gap into one context block
			full_roofline = roofline_ctx
			if gap_ctx:
				full_roofline += "\n\n" + gap_ctx

			agent_result = await self._agent.optimize(
				problem_name=problem.name,
				problem_source=problem.reference_source,
				baseline_ms=baseline_ms,
				experience_context=experience_ctx,
				traits_summary=traits.summary(),
				roofline_context=full_roofline,
				distilled_guide=distilled_guide,
				triton_examples=triton_examples,
				gpu_id=self._config.hardware.gpu_id,
			)
			s.set("speedup", agent_result.speedup)
			s.set("success", agent_result.success)
			s.set("approach", agent_result.approach)
			tracker.record_agent_call(
				s.duration_ms if s.end_time > 0 else 0
			)

		# Save outputs
		async with tracker.span("save_results"):
			(run_dir / "agent_output.txt").write_text(
				agent_result.raw_output
			)
			self._log_result(agent_result, run_dir)
			self._record_experience(
				agent_result, problem, traits, baseline_ms
			)

		# Save telemetry
		tracker.finish()
		tracker.save(run_dir / "telemetry.json")
		logger.info(tracker.report())

		summary = {
			"problem": problem.name,
			"goal": goal.primary,
			"best_speedup": agent_result.speedup,
			"baseline_ms": baseline_ms,
			"approach": agent_result.approach,
			"kernel_path": agent_result.kernel_path,
			"success": agent_result.success,
			"telemetry": tracker.summary(),
		}
		(run_dir / "summary.json").write_text(
			json.dumps(summary, indent=2)
		)
		return summary

	def _load_distilled_knowledge(
		self, traits: object
	) -> tuple[str, str]:
		"""Load distilled guide + Triton examples for this op type.

		Layer 1: Distilled guide from knowledge/distilled/<op>.md
		Layer 2: Triton code examples from external/triton_examples_index.json
		"""
		distilled_dir = self._config.knowledge_dir / "distilled"
		external_dir = self._config.knowledge_dir / "external"

		# Layer 1: Find best matching distilled guide
		guide = ""
		if hasattr(traits, "dominant_ops") and traits.dominant_ops:
			# Map trait ops to distilled file names
			op_to_file = {
				"matmul": "matmul",
				"conv": "conv",
				"attention": "attention",
				"softmax": "softmax",
				"norm": "norm",
				"elementwise": "elementwise",
				"reduction": "reduce",
				"loss": "other",
				"pooling": "pooling",
				"indexing": "reduce",
				"cumulative": "reduce",
			}
			for op in traits.dominant_ops:
				fname = op_to_file.get(op, op)
				guide_path = distilled_dir / f"{fname}.md"
				if guide_path.exists():
					content = guide_path.read_text()
					if len(content) > 100:
						guide = content[:8000]  # cap at ~2K tokens
						break

		# Layer 2: Triton examples
		examples = ""
		index_path = external_dir / "triton_examples_index.json"
		if index_path.exists() and hasattr(traits, "dominant_ops"):
			import json
			index = json.loads(index_path.read_text())
			for op in traits.dominant_ops:
				if op in index:
					ex_list = index[op]
					parts = []
					for ex in ex_list[:2]:  # max 2 examples
						code = ex["code"][:2000]
						parts.append(
							f"### {ex['name']}\n```python\n"
							f"{code}\n```"
						)
					if parts:
						examples = "\n\n".join(parts)
						break

		return guide, examples

	def _compute_roofline_context(
		self,
		problem: KernelProblem,
		traits: object,
		baseline_ms: float,
	) -> str:
		"""Compute roofline analysis and format as agent context."""
		from kernel_forge.core.evaluate import compute_roofline
		from kernel_forge.core.types import B200_PEAKS

		# Estimate FLOPs and bytes from shapes
		shapes = problem.input_shapes
		flops = 0.0
		bytes_moved = 0.0

		if shapes:
			dims = list(shapes.values())
			# Matmul-like: 2*M*N*K
			if (
				len(dims) >= 2
				and len(dims[0]) == 2
				and len(dims[1]) == 2
			):
				m, k = dims[0]
				_, n = dims[1]
				flops = 2.0 * m * k * n
				bytes_moved = (m * k + k * n + m * n) * 4.0
			else:
				# Elementwise/reduction estimate
				total_elems = 0
				for shape in dims:
					elems = 1
					for d in shape:
						elems *= d
					total_elems += elems
				flops = float(total_elems)
				bytes_moved = total_elems * 4.0 * 2  # read + write

		if flops <= 0 or bytes_moved <= 0:
			return (
				f"Baseline: {baseline_ms:.4f} ms. "
				f"Could not estimate FLOPs/bytes from shapes. "
				f"Use forge_roofline.py after computing FLOPs."
			)

		arith_intensity = flops / bytes_moved
		lines = []

		# Analyze at multiple precision tiers
		for precision, label in [
			("fp32", "FP32 (CUDA cores)"),
			("tf32", "TF32 (tensor cores)"),
			("bf16", "BF16 (tensor cores)"),
		]:
			r = compute_roofline(
				baseline_ms, flops, bytes_moved, precision
			)
			lines.append(
				f"- {label}: {r.achieved_tflops:.1f} / "
				f"{r.peak_tflops:.0f} TFLOPS = "
				f"**{r.utilization_pct:.1f}%** utilization, "
				f"{r.headroom_pct:.1f}% headroom"
			)

		# Compute theoretical best times
		peak_tf32 = B200_PEAKS.tf32_tflops * 1e12
		peak_bf16 = B200_PEAKS.bf16_tflops * 1e12
		peak_bw = B200_PEAKS.hbm_bandwidth_tb_s * 1e12

		compute_limit_tf32 = flops / peak_tf32 * 1000
		compute_limit_bf16 = flops / peak_bf16 * 1000
		bw_limit = bytes_moved / peak_bw * 1000

		bound = "compute-bound" if arith_intensity > 120 else "memory-bound"

		context = (
			f"Baseline: {baseline_ms:.4f} ms\n"
			f"FLOPs: {flops:.2e}, Bytes: {bytes_moved:.2e}\n"
			f"Arithmetic intensity: {arith_intensity:.1f} FLOPs/byte "
			f"(ridge=120.5) -> **{bound}**\n\n"
			f"Utilization at current baseline:\n"
			+ "\n".join(lines)
			+ f"\n\nTheoretical limits:\n"
			f"- Compute (TF32): {compute_limit_tf32:.4f} ms "
			f"(potential {baseline_ms/compute_limit_tf32:.1f}x)\n"
			f"- Compute (BF16): {compute_limit_bf16:.4f} ms "
			f"(potential {baseline_ms/compute_limit_bf16:.1f}x)\n"
			f"- Memory BW: {bw_limit:.4f} ms "
			f"(potential {baseline_ms/bw_limit:.1f}x)\n"
			f"- Achievable: min(compute, memory) = "
			f"{max(compute_limit_tf32, bw_limit):.4f} ms for TF32"
		)
		return context

	def _log_result(
		self, result: AgentResult, run_dir: Path
	) -> None:
		if result.success:
			logger.info(
				"[RESULT] %.3fx speedup via: %s",
				result.speedup, result.approach,
			)
			if result.why_it_worked:
				logger.info("[WHY] %s", result.why_it_worked)
		else:
			logger.info("[RESULT] No speedup achieved")

		if result.tool_requests:
			logger.info("[TOOL REQUESTS] Agent wants:")
			for req in result.tool_requests:
				logger.info("  -> %s", req)
			(run_dir / "tool_requests.txt").write_text(
				"\n".join(result.tool_requests)
			)

	def _record_experience(
		self,
		result: AgentResult,
		problem: KernelProblem,
		traits: object,
		baseline_ms: float,
	) -> None:
		self._experience.record(ExperienceRecord(
			problem_name=problem.name,
			dominant_ops=traits.dominant_ops,
			strategy_name=(
				result.approach[:50] if result.approach else "unknown"
			),
			approach_notes=result.approach,
			outcome="success" if result.success else "no_improvement",
			speedup=result.speedup,
			baseline_ms=baseline_ms,
			optimized_ms=(
				baseline_ms / result.speedup
				if result.speedup > 0 else 0
			),
			bottleneck_type=traits.estimated_bottleneck,
			roofline_utilization_pct=0.0,
			root_cause=result.approach,
			has_data_reuse=traits.has_data_reuse,
			shape_category=traits.shape_category,
			estimated_bottleneck=traits.estimated_bottleneck,
			input_shapes=problem.input_shapes,
		))
