"""Profiling tools: GPU status, CUDA events benchmark, and NCU profiling."""

from __future__ import annotations

import json
from typing import Any

from kernel_forge.remote.executor import Executor
from kernel_forge.remote.gpu_guard import GpuGuard
from kernel_forge.tools.registry import ToolResult


class GpuStatusTool:
	"""Check GPU availability via nvidia-smi."""

	def __init__(
		self, executor: Executor, gpu_id: int = 2, memory_threshold_mib: int = 100
	) -> None:
		self._guard = GpuGuard(
			executor, gpu_id=gpu_id, memory_threshold_mib=memory_threshold_mib
		)

	@property
	def name(self) -> str:
		return "gpu_status"

	@property
	def description(self) -> str:
		return "Check GPU availability and memory usage via nvidia-smi"

	async def run(self, **kwargs: Any) -> ToolResult:
		status = await self._guard.check()
		return ToolResult(
			success=status.available,
			data={
				"available": status.available,
				"memory_used_mib": status.memory_used_mib,
			},
			output=status.message,
			error="" if status.available else status.message,
		)


class CudaEventsBench:
	"""Run CUDA events-based benchmark via the remote executor."""

	def __init__(self, executor: Executor) -> None:
		self._executor = executor

	@property
	def name(self) -> str:
		return "cuda_events_bench"

	@property
	def description(self) -> str:
		return "Fast wall-clock timing using CUDA events (warmup + timed reps)"

	async def run(self, **kwargs: Any) -> ToolResult:
		kernel_path: str = kwargs.get("kernel_path", "")
		problem_name: str = kwargs.get("problem_name", "")
		warmup_reps: int = kwargs.get("warmup_reps", 10)
		timed_reps: int = kwargs.get("timed_reps", 100)

		if not kernel_path:
			return ToolResult(success=False, error="kernel_path is required")

		cmd = (
			f"python benchmark_harness.py "
			f"--kernel {kernel_path} "
			f"--problem {problem_name} "
			f"--warmup {warmup_reps} "
			f"--reps {timed_reps} "
			f"--output json"
		)
		result = await self._executor.run(cmd, timeout=300)

		if not result.success:
			return ToolResult(
				success=False,
				error=f"Benchmark failed: {result.stderr}",
				output=result.stdout,
			)

		try:
			data = json.loads(result.stdout.strip())
		except json.JSONDecodeError:
			return ToolResult(
				success=False,
				error=f"Could not parse benchmark JSON output: {result.stdout!r}",
				output=result.stdout,
			)

		return ToolResult(
			success=True,
			data=data,
			output=f"Runtime: {data.get('runtime_us', 'N/A')} us, "
			f"Speedup: {data.get('speedup', 'N/A')}x",
		)


class NcuProfile:
	"""Run Nsight Compute profiling via the remote executor."""

	DEFAULT_METRICS = [
		"sm__throughput.avg.pct_of_peak_sustained_elapsed",
		"gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
		"sm__warps_active.avg.pct_of_peak_sustained_active",
	]

	def __init__(self, executor: Executor) -> None:
		self._executor = executor

	@property
	def name(self) -> str:
		return "ncu_profile"

	@property
	def description(self) -> str:
		return "Deep kernel profiling via Nsight Compute with configurable metrics"

	async def run(self, **kwargs: Any) -> ToolResult:
		kernel_path: str = kwargs.get("kernel_path", "")
		problem_name: str = kwargs.get("problem_name", "")
		metrics: list[str] = kwargs.get("metrics", self.DEFAULT_METRICS)

		if not kernel_path:
			return ToolResult(success=False, error="kernel_path is required")

		metrics_str = ",".join(metrics)
		cmd = (
			f"ncu --metrics {metrics_str} "
			f"python run_kernel.py --kernel {kernel_path} --problem {problem_name}"
		)
		result = await self._executor.run(cmd, timeout=600)

		if not result.success:
			return ToolResult(
				success=False,
				error=f"ncu profiling failed: {result.stderr}",
				output=result.stdout,
			)

		return ToolResult(
			success=True,
			data={"raw_output": result.stdout},
			output=result.stdout,
		)
