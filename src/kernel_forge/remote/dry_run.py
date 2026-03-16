"""Dry-run executor for local development without GPU access."""

from __future__ import annotations

import json

from kernel_forge.remote.executor import CommandResult


class DryRunExecutor:
	"""Mock executor that returns synthetic responses for development/testing."""

	def __init__(self) -> None:
		self._upload_log: list[tuple[str, str]] = []
		self._download_log: list[tuple[str, str]] = []
		self._command_log: list[str] = []

	async def run(self, command: str, timeout: int = 300) -> CommandResult:
		"""Return synthetic results based on command patterns."""
		self._command_log.append(command)

		if "nvidia-smi" in command:
			return self._nvidia_smi_response()
		if "nvcc" in command or "compile" in command.lower():
			return self._compile_response()
		if "ncu" in command:
			return self._ncu_response()
		if "bench" in command.lower() or "benchmark" in command.lower():
			return self._benchmark_response()
		if "correctness" in command.lower() or "allclose" in command.lower():
			return self._correctness_response()

		return CommandResult(stdout="", stderr="", exit_code=0)

	async def upload(self, local_path: str, remote_path: str) -> None:
		"""No-op upload; logs the paths."""
		self._upload_log.append((local_path, remote_path))

	async def download(self, remote_path: str, local_path: str) -> None:
		"""No-op download; logs the paths."""
		self._download_log.append((remote_path, local_path))

	def _nvidia_smi_response(self) -> CommandResult:
		return CommandResult(stdout="0\n", stderr="", exit_code=0)

	def _compile_response(self) -> CommandResult:
		return CommandResult(
			stdout="Compilation successful\n",
			stderr="",
			exit_code=0,
		)

	def _ncu_response(self) -> CommandResult:
		output = (
			"==PROF== Profiling complete\n"
			"Metric: sm__throughput.avg.pct_of_peak_sustained_elapsed = 45.2\n"
			"Metric: gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed = 78.3\n"
			"Metric: sm__warps_active.avg.pct_of_peak_sustained_active = 62.1\n"
		)
		return CommandResult(stdout=output, stderr="", exit_code=0)

	def _benchmark_response(self) -> CommandResult:
		result = {
			"runtime_us": 150.5,
			"speedup": 1.25,
			"correct": True,
			"warmup_reps": 10,
			"timed_reps": 100,
		}
		return CommandResult(stdout=json.dumps(result) + "\n", stderr="", exit_code=0)

	def _correctness_response(self) -> CommandResult:
		result = {
			"correct": True,
			"max_abs_diff": 1e-5,
			"max_rel_diff": 1e-4,
			"all_close": True,
		}
		return CommandResult(stdout=json.dumps(result) + "\n", stderr="", exit_code=0)
