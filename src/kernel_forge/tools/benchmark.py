"""Correctness checking tool."""

from __future__ import annotations

import json
from typing import Any

from kernel_forge.remote.executor import Executor
from kernel_forge.tools.registry import ToolResult


class CorrectnessTool:
	"""Run correctness check: torch.allclose against reference on randomized inputs."""

	def __init__(self, executor: Executor) -> None:
		self._executor = executor

	@property
	def name(self) -> str:
		return "correctness_check"

	@property
	def description(self) -> str:
		return "Validate kernel correctness via torch.allclose against reference implementation"

	async def run(self, **kwargs: Any) -> ToolResult:
		kernel_path: str = kwargs.get("kernel_path", "")
		problem_name: str = kwargs.get("problem_name", "")
		rtol: float = kwargs.get("rtol", 1e-3)
		atol: float = kwargs.get("atol", 1e-3)

		if not kernel_path:
			return ToolResult(success=False, error="kernel_path is required")

		cmd = (
			f"python correctness_check.py "
			f"--kernel {kernel_path} "
			f"--problem {problem_name} "
			f"--rtol {rtol} "
			f"--atol {atol} "
			f"--output json"
		)
		result = await self._executor.run(cmd, timeout=300)

		if not result.success:
			return ToolResult(
				success=False,
				error=f"Correctness check failed to run: {result.stderr}",
				output=result.stdout,
			)

		try:
			data = json.loads(result.stdout.strip())
		except json.JSONDecodeError:
			return ToolResult(
				success=False,
				error=f"Could not parse correctness output: {result.stdout!r}",
				output=result.stdout,
			)

		is_correct = data.get("correct", False)
		return ToolResult(
			success=is_correct,
			data=data,
			output=f"Correct: {is_correct}, max_abs_diff: {data.get('max_abs_diff', 'N/A')}",
			error="" if is_correct else "Kernel output does not match reference",
		)
