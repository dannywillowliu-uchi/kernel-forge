"""Loop state tracking for the kernel optimization loop."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from kernel_forge.config import ForgeConfig
from kernel_forge.core.evaluate import should_escalate_profiling, should_terminate
from kernel_forge.core.types import Attempt, KernelProblem, OptimizationGoal


@dataclass
class LoopState:
	"""Tracks state across iterations of the optimization loop."""

	problem: KernelProblem
	goal: OptimizationGoal
	config: ForgeConfig
	attempt_count: int = 0
	best_speedup: float = 0.0
	best_kernel_source: str | None = None
	total_cost: float = 0.0
	consecutive_failures: int = 0
	profiling_tier: str = "cuda_events"
	start_time: float = field(default_factory=time.time)
	attempts: list[Attempt] = field(default_factory=list)

	def record_attempt(self, speedup: float, correct: bool, cost: float) -> None:
		"""Record the outcome of an optimization attempt."""
		self.attempt_count += 1
		self.total_cost += cost

		if correct and speedup > self.best_speedup:
			self.best_speedup = speedup
			self.consecutive_failures = 0
		elif not correct:
			self.consecutive_failures += 1
		else:
			# Correct but not an improvement
			self.consecutive_failures = 0

	@property
	def should_stop(self) -> bool:
		"""Check if the loop should terminate."""
		elapsed = time.time() - self.start_time
		return should_terminate(
			attempt_count=self.attempt_count,
			total_cost=self.total_cost,
			elapsed_seconds=elapsed,
			consecutive_failures=self.consecutive_failures,
			config=self.config.termination,
		)

	@property
	def elapsed_seconds(self) -> float:
		"""Seconds since loop started."""
		return time.time() - self.start_time

	@property
	def should_escalate(self) -> bool:
		"""Check if profiling should be escalated to ncu."""
		return should_escalate_profiling(
			recent_attempts=self.attempts,
			config=self.config.termination,
		)

	@staticmethod
	def kernel_hash(source: str) -> str:
		"""Compute a truncated SHA-256 hash of kernel source code."""
		return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
