"""Knowledge query layer for context injection from DB + learnings."""

from __future__ import annotations

from kernel_forge.core.types import BottleneckType
from kernel_forge.knowledge.db import KnowledgeDB
from kernel_forge.knowledge.learnings import LearningsManager


class KnowledgeQuery:
	"""Builds agent context from the knowledge DB and markdown learnings."""

	def __init__(self, db: KnowledgeDB, learnings: LearningsManager) -> None:
		self._db = db
		self._learnings = learnings

	async def build_context(
		self,
		kernel_problem: str,
		kernel_type: str | None = None,
		bottleneck_type: BottleneckType | None = None,
		max_tokens: int = 8000,
	) -> str:
		"""Build agent context string from DB + learnings.

		Priority order (trimmed from bottom if over budget):
		1. Best strategies from DB for this kernel type
		2. Strategies matching bottleneck type
		3. Prior attempts on this problem (last 5)
		4. Relevant learnings from markdown
		"""
		sections: list[str] = []
		total_chars = 0
		max_chars = max_tokens * 4  # ~4 chars per token

		# 1. Best strategies for this kernel type
		if kernel_type:
			best = await self._db.get_best_strategies_for_kernel_type(kernel_type, limit=5)
			if best:
				lines = ["## Best strategies for this kernel type"]
				for entry in best:
					lines.append(
						f"- {entry['strategy_name']}: "
						f"avg speedup {entry['avg_speedup']:.2f}x "
						f"({entry['attempt_count']} attempts)"
					)
				section = "\n".join(lines)
				section_len = len(section)
				if total_chars + section_len <= max_chars:
					sections.append(section)
					total_chars += section_len

		# 2. Strategies matching bottleneck type
		if bottleneck_type:
			strategies = await self._db.get_strategies_for_bottleneck(bottleneck_type)
			if strategies:
				lines = [f"## Strategies for {bottleneck_type.value} bottleneck"]
				for s in strategies[:5]:
					lines.append(f"- **{s.name}**: {s.description}")
					if s.expected_impact:
						lines.append(f"  Expected impact: {s.expected_impact}")
				section = "\n".join(lines)
				section_len = len(section)
				if total_chars + section_len <= max_chars:
					sections.append(section)
					total_chars += section_len

		# 3. Prior attempts on this problem (last 5)
		attempts = await self._db.get_attempts_for_problem(kernel_problem)
		if attempts:
			last_5 = attempts[-5:]
			lines = [f"## Prior attempts on {kernel_problem}"]
			for a in last_5:
				status = "correct" if a.correct else "INCORRECT"
				lines.append(
					f"- {a.strategy_name}: {a.speedup:.2f}x ({status}, "
					f"tier: {a.profiling_tier})"
				)
			section = "\n".join(lines)
			section_len = len(section)
			if total_chars + section_len <= max_chars:
				sections.append(section)
				total_chars += section_len

		# 4. Relevant learnings from markdown
		remaining_tokens = (max_chars - total_chars) // 4
		if remaining_tokens > 50 and kernel_type:
			relevant = self._learnings.read_relevant(kernel_type, max_tokens=remaining_tokens)
			if relevant:
				lines = ["## Relevant learnings"]
				lines.extend(relevant)
				section = "\n".join(lines)
				section_len = len(section)
				if total_chars + section_len <= max_chars:
					sections.append(section)
					total_chars += section_len

		if not sections:
			return ""

		return "\n\n".join(sections)
