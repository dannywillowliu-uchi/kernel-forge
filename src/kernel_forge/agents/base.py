"""Agent protocol for autonomous kernel optimization."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class KernelAgent(Protocol):
	"""Protocol for an autonomous kernel optimization agent.

	The agent has tools and iterates on its own. It receives a problem
	with advisory context and returns its best result.
	"""

	async def optimize(
		self,
		problem_name: str,
		problem_source: str,
		baseline_ms: float,
		experience_context: str,
		traits_summary: str,
		max_attempts: int = 5,
	) -> object: ...
