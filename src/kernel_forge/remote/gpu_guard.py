"""GPU availability checking and contention detection."""

from __future__ import annotations

from dataclasses import dataclass

from kernel_forge.remote.executor import CommandResult, Executor


@dataclass
class GpuStatus:
	"""Result of a GPU availability check."""

	available: bool
	memory_used_mib: int
	message: str


class GpuGuard:
	"""Checks GPU availability before running benchmarks."""

	def __init__(
		self,
		executor: Executor,
		gpu_id: int = 2,
		memory_threshold_mib: int = 100,
	) -> None:
		self._executor = executor
		self._gpu_id = gpu_id
		self._memory_threshold_mib = memory_threshold_mib

	async def check(self) -> GpuStatus:
		"""Check if the target GPU is available (memory below threshold)."""
		cmd = (
			f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {self._gpu_id}"
		)
		result: CommandResult = await self._executor.run(cmd, timeout=30)

		if not result.success:
			return GpuStatus(
				available=False,
				memory_used_mib=0,
				message=f"nvidia-smi failed: {result.stderr}",
			)

		try:
			memory_used = int(result.stdout.strip())
		except ValueError:
			return GpuStatus(
				available=False,
				memory_used_mib=0,
				message=f"Could not parse nvidia-smi output: {result.stdout!r}",
			)

		if memory_used > self._memory_threshold_mib:
			return GpuStatus(
				available=False,
				memory_used_mib=memory_used,
				message=(
					f"GPU {self._gpu_id} in use: {memory_used} MiB > "
					f"{self._memory_threshold_mib} MiB threshold"
				),
			)

		return GpuStatus(
			available=True,
			memory_used_mib=memory_used,
			message=f"GPU {self._gpu_id} available: {memory_used} MiB used",
		)
