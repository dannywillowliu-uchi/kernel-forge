"""Claude Code CLI agent implementation."""

from __future__ import annotations

import asyncio
import logging

from kernel_forge.agents.prompts import (
	build_diagnose_prompt,
	build_generate_prompt,
	build_suggest_strategies_prompt,
	parse_diagnosis_output,
	parse_kernel_output,
	parse_strategies_output,
)
from kernel_forge.core.types import (
	Attempt,
	Diagnosis,
	KernelCandidate,
	KernelProblem,
	OptimizationGoal,
	ProfileData,
	Strategy,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sonnet"
DEFAULT_MAX_TURNS = 50


class ClaudeCodeAgent:
	"""KernelAgent implementation that wraps the Claude CLI subprocess."""

	def __init__(
		self,
		model: str = DEFAULT_MODEL,
		max_turns: int = DEFAULT_MAX_TURNS,
	) -> None:
		self._model = model
		self._max_turns = max_turns

	async def _invoke_claude(self, prompt: str) -> str:
		"""Run claude CLI subprocess and return its text output."""
		cmd = [
			"claude",
			"--permission-mode", "auto",
			"--max-turns", str(self._max_turns),
			"--output-format", "text",
			"--model", self._model,
			"-p", prompt,
		]
		logger.debug("Invoking claude CLI with model=%s", self._model)

		proc = await asyncio.create_subprocess_exec(
			*cmd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		stdout_bytes, stderr_bytes = await proc.communicate()

		stdout = stdout_bytes.decode("utf-8", errors="replace")
		stderr = stderr_bytes.decode("utf-8", errors="replace")

		if proc.returncode != 0:
			logger.warning(
				"Claude CLI returned exit code %d: %s",
				proc.returncode,
				stderr[:500],
			)

		return stdout

	async def generate_kernel(
		self,
		problem: KernelProblem,
		goal: OptimizationGoal,
		diagnosis: Diagnosis | None,
		strategy_name: str,
		prior_attempts: list[Attempt],
		knowledge_context: str,
	) -> KernelCandidate | None:
		"""Generate a kernel candidate via Claude CLI.

		On parse failure, retries once with a format reminder.
		"""
		prompt = build_generate_prompt(
			problem=problem,
			goal=goal,
			diagnosis=diagnosis,
			strategy_name=strategy_name,
			prior_attempts=prior_attempts,
			knowledge_context=knowledge_context,
		)

		raw = await self._invoke_claude(prompt)
		candidate = parse_kernel_output(raw)

		if candidate is None:
			logger.info("First parse failed, retrying with format reminder")
			retry_prompt = (
				prompt
				+ "\n\nIMPORTANT: Your previous response did not include the required markers. "
				"Please format your response with:\n"
				"KERNEL_SOURCE_START\n<kernel code>\nKERNEL_SOURCE_END\n"
				"APPROACH_NOTES: <explanation>"
			)
			raw = await self._invoke_claude(retry_prompt)
			candidate = parse_kernel_output(raw)

		if candidate is not None:
			candidate.strategy_name = strategy_name

		return candidate

	async def diagnose_bottleneck(
		self,
		profile: ProfileData,
		kernel_source: str,
		problem: KernelProblem,
	) -> Diagnosis | None:
		"""Diagnose the performance bottleneck via Claude CLI."""
		prompt = build_diagnose_prompt(
			profile=profile,
			kernel_source=kernel_source,
			problem=problem,
		)

		raw = await self._invoke_claude(prompt)
		diagnosis = parse_diagnosis_output(raw)

		if diagnosis is not None:
			diagnosis.profiling_tier = profile.profiling_tier

		return diagnosis

	async def suggest_strategies(
		self,
		diagnosis: Diagnosis,
		available_strategies: list[Strategy],
		prior_attempts: list[Attempt],
	) -> list[str]:
		"""Suggest strategy names via Claude CLI.

		Falls back to ["shared_mem_tiling"] if parsing fails.
		"""
		prompt = build_suggest_strategies_prompt(
			diagnosis=diagnosis,
			available_strategies=available_strategies,
			prior_attempts=prior_attempts,
		)

		raw = await self._invoke_claude(prompt)
		names = parse_strategies_output(raw)

		if not names:
			logger.warning("Strategy parsing failed, using fallback")
			return ["shared_mem_tiling"]

		return names
