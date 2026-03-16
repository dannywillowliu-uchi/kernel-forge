"""KernelAgent protocol for LLM-driven kernel optimization."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kernel_forge.core.types import (
	Attempt,
	Diagnosis,
	KernelCandidate,
	KernelProblem,
	OptimizationGoal,
	ProfileData,
	Strategy,
)


@runtime_checkable
class KernelAgent(Protocol):
	"""Interface for any LLM that can generate/optimize kernels."""

	async def generate_kernel(
		self,
		problem: KernelProblem,
		goal: OptimizationGoal,
		diagnosis: Diagnosis | None,
		strategy_name: str,
		prior_attempts: list[Attempt],
		knowledge_context: str,
	) -> KernelCandidate | None: ...

	async def diagnose_bottleneck(
		self,
		profile: ProfileData,
		kernel_source: str,
		problem: KernelProblem,
	) -> Diagnosis | None: ...

	async def suggest_strategies(
		self,
		diagnosis: Diagnosis,
		available_strategies: list[Strategy],
		prior_attempts: list[Attempt],
	) -> list[str]: ...
