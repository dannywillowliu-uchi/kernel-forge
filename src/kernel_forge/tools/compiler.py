"""Kernel compilation tool."""

from __future__ import annotations

from typing import Any

from kernel_forge.remote.executor import Executor
from kernel_forge.tools.registry import ToolResult


class KernelCompiler:
	"""Compile a CUDA/Triton kernel source on the remote host."""

	def __init__(self, executor: Executor) -> None:
		self._executor = executor

	@property
	def name(self) -> str:
		return "kernel_compile"

	@property
	def description(self) -> str:
		return "Compile CUDA or Triton kernel source on the remote GPU host"

	async def run(self, **kwargs: Any) -> ToolResult:
		kernel_source: str = kwargs.get("kernel_source", "")
		kernel_path: str = kwargs.get("kernel_path", "kernels/kernel.cu")

		if not kernel_source:
			return ToolResult(success=False, error="kernel_source is required")

		# Upload kernel source to the remote path
		await self._executor.upload(kernel_source, kernel_path)

		# Attempt compilation
		cmd = f"nvcc -o /dev/null -c {kernel_path} 2>&1"
		result = await self._executor.run(cmd, timeout=120)

		if not result.success:
			return ToolResult(
				success=False,
				data={"exit_code": result.exit_code},
				output=result.stdout,
				error=result.stderr or result.stdout,
			)

		return ToolResult(
			success=True,
			data={"kernel_path": kernel_path},
			output="Compilation successful",
		)
