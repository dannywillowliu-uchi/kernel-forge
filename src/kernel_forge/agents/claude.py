"""Claude Code agent with tool access for autonomous kernel optimization."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

AGENT_PROMPT_PATH = Path(__file__).parent / "agent_prompt.md"
DEFAULT_MODEL = "opus"
DEFAULT_MAX_TURNS = 30


class ClaudeCodeAgent:
	"""Autonomous kernel optimization agent using Claude Code CLI.

	Unlike v1 (single-turn text generator), this agent has tools:
	it can SSH to the B200, compile kernels, benchmark them, profile
	with ncu, and iterate on its own. The orchestrator manages
	budget and termination; the agent manages the optimization.
	"""

	def __init__(
		self,
		model: str = DEFAULT_MODEL,
		max_turns: int = DEFAULT_MAX_TURNS,
	) -> None:
		self._model = model
		self._max_turns = max_turns
		self._agent_prompt = ""
		if AGENT_PROMPT_PATH.exists():
			self._agent_prompt = AGENT_PROMPT_PATH.read_text()

	async def optimize(
		self,
		problem_name: str,
		problem_source: str,
		baseline_ms: float,
		experience_context: str,
		traits_summary: str,
		max_attempts: int = 5,
	) -> AgentResult:
		"""Run the agent autonomously on a kernel problem.

		The agent has full tool access: SSH to B200, benchmark,
		profile, write kernels. It iterates until it finds the
		best kernel it can within the turn budget.
		"""
		problem_section = (
			f"## Problem: {problem_name}\n\n"
			f"Reference implementation:\n```python\n{problem_source}\n```\n\n"
			f"Baseline runtime: {baseline_ms:.4f} ms\n"
		)

		traits_section = (
			f"## Trait Analysis (advisory)\n{traits_summary}\n"
		)

		experience_section = ""
		if experience_context:
			experience_section = f"\n{experience_context}\n"

		task_section = (
			f"\n## Your Task\n\n"
			f"Optimize the kernel above. The reference runs at "
			f"{baseline_ms:.4f} ms on B200 GPU 2.\n\n"
			f"1. Profile the baseline with ncu to understand the bottleneck\n"
			f"2. Write an optimized kernel in kernels/{problem_name}_opt.py\n"
			f"3. Test correctness and benchmark\n"
			f"4. Profile your kernel to see where you are on the roofline\n"
			f"5. Iterate to improve\n"
			f"6. When done, output BEST_KERNEL_PATH, BEST_SPEEDUP, APPROACH\n"
		)

		full_prompt = (
			self._agent_prompt + "\n\n"
			+ problem_section
			+ traits_section
			+ experience_section
			+ task_section
		)

		# Invoke Claude Code with tool access
		output = await self._invoke(full_prompt)

		# Parse results from agent output
		return self._parse_result(output, problem_name)

	async def _invoke(self, prompt: str) -> str:
		"""Run Claude Code CLI with tools enabled.

		Writes the prompt to a temp file and passes via -p to avoid
		shell argument length limits and reduce prompt processing time.
		"""
		import os
		import tempfile

		# Write prompt to temp file
		with tempfile.NamedTemporaryFile(
			mode="w", suffix=".md", delete=False, prefix="forge_prompt_"
		) as f:
			f.write(prompt)
			prompt_file = f.name

		# Use a concise -p that references the file
		short_prompt = (
			f"Read the task file at {prompt_file} and execute it. "
			f"Follow the methodology exactly. Output BEST_KERNEL_PATH, "
			f"BEST_SPEEDUP, APPROACH when done."
		)

		cmd = [
			"claude",
			"--permission-mode", "bypassPermissions",
			"--max-turns", str(self._max_turns),
			"--output-format", "text",
			"--model", self._model,
			"-p", short_prompt,
		]
		logger.info(
			"Invoking Claude agent (model=%s, max_turns=%d)",
			self._model, self._max_turns,
		)

		timeout_s = 900  # 15 minutes for full optimization run

		proc = await asyncio.create_subprocess_exec(
			*cmd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		try:
			stdout_bytes, stderr_bytes = await asyncio.wait_for(
				proc.communicate(), timeout=timeout_s
			)
		except asyncio.TimeoutError:
			logger.warning(
				"Claude agent timed out after %ds", timeout_s
			)
			try:
				proc.kill()
			except ProcessLookupError:
				pass
			return ""
		finally:
			os.unlink(prompt_file)

		stdout = stdout_bytes.decode("utf-8", errors="replace")
		stderr = stderr_bytes.decode("utf-8", errors="replace")

		if proc.returncode != 0:
			logger.warning(
				"Claude agent exit code %d: %s",
				proc.returncode, stderr[:300],
			)

		return stdout

	def _parse_result(
		self, output: str, problem_name: str
	) -> AgentResult:
		"""Parse the agent's output for best kernel info."""
		# Look for BEST_KERNEL_PATH
		path_match = re.search(
			r"BEST_KERNEL_PATH:\s*(.+)", output
		)
		speedup_match = re.search(
			r"BEST_SPEEDUP:\s*([\d.]+)", output
		)
		approach_match = re.search(
			r"APPROACH:\s*(.+)", output
		)

		kernel_path = (
			path_match.group(1).strip() if path_match else ""
		)
		speedup = (
			float(speedup_match.group(1))
			if speedup_match else 0.0
		)
		approach = (
			approach_match.group(1).strip()
			if approach_match else ""
		)

		return AgentResult(
			kernel_path=kernel_path,
			speedup=speedup,
			approach=approach,
			raw_output=output,
			success=speedup > 1.0,
		)


class AgentResult:
	"""Result from an autonomous agent optimization run."""

	def __init__(
		self,
		kernel_path: str = "",
		speedup: float = 0.0,
		approach: str = "",
		raw_output: str = "",
		success: bool = False,
	) -> None:
		self.kernel_path = kernel_path
		self.speedup = speedup
		self.approach = approach
		self.raw_output = raw_output
		self.success = success
