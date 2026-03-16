"""Structured experience records for cross-problem learning.

Records WHY strategies work or fail. Uses trait similarity
for cross-problem matching instead of rigid categories.

All experience is ADVISORY -- the agent uses it as context
but makes its own decisions based on profiling data.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kernel_forge.knowledge.classifier import KernelTraits

logger = logging.getLogger(__name__)


@dataclass
class ExperienceRecord:
	"""A structured learning from one optimization attempt."""

	# What problem
	problem_name: str
	dominant_ops: list[str]  # from traits: ["matmul"], ["elementwise", "reduction"]

	# What was tried
	strategy_name: str
	approach_notes: str

	# What happened
	outcome: str  # "success", "no_improvement", "correctness_failure", etc.
	speedup: float
	baseline_ms: float
	optimized_ms: float

	# WHY it worked or failed
	bottleneck_type: str  # from profiling or estimated
	roofline_utilization_pct: float
	root_cause: str  # human-readable explanation

	# Traits for similarity matching
	has_data_reuse: bool = False
	shape_category: str = "unknown"
	estimated_bottleneck: str = "unknown"
	input_shapes: dict = field(default_factory=dict)
	precision_constraint: str = "fp32_strict"


@dataclass
class SimilarExperience:
	"""Experience from a similar problem, with similarity score."""

	record: ExperienceRecord
	similarity: float


class ExperienceStore:
	"""Persists and queries structured experience records.

	Uses trait-based similarity matching for cross-problem learning.
	All returned data is ADVISORY context for the agent.
	"""

	def __init__(self, path: Path | str) -> None:
		self._path = Path(path)
		self._path.mkdir(parents=True, exist_ok=True)
		self._records_file = self._path / "records.jsonl"

	def record(self, exp: ExperienceRecord) -> None:
		"""Append an experience record."""
		with open(self._records_file, "a") as f:
			f.write(json.dumps(asdict(exp)) + "\n")

	def get_all_records(self) -> list[ExperienceRecord]:
		"""Read all records."""
		if not self._records_file.exists():
			return []
		records: list[ExperienceRecord] = []
		with open(self._records_file) as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				try:
					data = json.loads(line)
					records.append(ExperienceRecord(**data))
				except (json.JSONDecodeError, TypeError):
					continue
		return records

	def find_similar(
		self,
		traits: KernelTraits,
		min_similarity: float = 0.3,
		limit: int = 20,
	) -> list[SimilarExperience]:
		"""Find experience records from problems with similar traits.

		Returns records sorted by similarity (highest first).
		"""
		records = self.get_all_records()
		scored: list[SimilarExperience] = []

		for rec in records:
			# Build a pseudo-traits from the record for comparison
			rec_traits = KernelTraits(
				dominant_ops=rec.dominant_ops,
				estimated_bottleneck=rec.estimated_bottleneck,
				has_data_reuse=rec.has_data_reuse,
				shape_category=rec.shape_category,
			)
			sim = traits.similarity(rec_traits)
			if sim >= min_similarity:
				scored.append(SimilarExperience(
					record=rec, similarity=sim
				))

		scored.sort(key=lambda x: x.similarity, reverse=True)
		return scored[:limit]

	def build_advisory_context(
		self,
		traits: KernelTraits,
		max_tokens: int = 4000,
	) -> str:
		"""Build LLM-readable advisory context from similar experiences.

		This is SUGGESTIONS, not rules. The agent should override
		these based on actual profiling data for the current problem.
		"""
		similar = self.find_similar(traits, min_similarity=0.3)
		if not similar:
			return ""

		sections: list[str] = []
		sections.append(
			"## Experience from similar problems "
			"(advisory -- override based on profiling)"
		)

		# Aggregate strategy stats from similar experiences
		strategy_stats: dict[str, dict] = {}
		failure_stats: dict[str, int] = {}

		for se in similar:
			rec = se.record
			sname = rec.strategy_name
			if sname not in strategy_stats:
				strategy_stats[sname] = {
					"speedups": [],
					"successes": 0,
					"total": 0,
				}
			strategy_stats[sname]["total"] += 1
			if rec.outcome == "success":
				strategy_stats[sname]["successes"] += 1
				strategy_stats[sname]["speedups"].append(rec.speedup)
			elif rec.outcome != "no_improvement":
				key = f"{sname}: {rec.outcome}"
				failure_stats[key] = failure_stats.get(key, 0) + 1

		# What tends to work
		working = []
		for name, stats in strategy_stats.items():
			if stats["speedups"]:
				avg = sum(stats["speedups"]) / len(stats["speedups"])
				rate = stats["successes"] / max(stats["total"], 1)
				working.append((name, avg, rate, stats["total"]))
		working.sort(key=lambda x: x[1], reverse=True)

		if working:
			lines = ["### Strategies that worked on similar problems:"]
			for name, avg, rate, total in working[:5]:
				lines.append(
					f"- **{name}**: {avg:.2f}x avg speedup "
					f"({rate:.0%} success, {total} tries)"
				)
			sections.append("\n".join(lines))

		# What tends to fail
		if failure_stats:
			lines = ["### Common failures on similar problems:"]
			sorted_fails = sorted(
				failure_stats.items(),
				key=lambda x: x[1],
				reverse=True,
			)
			for desc, count in sorted_fails[:5]:
				lines.append(f"- {desc} ({count} times)")
			sections.append("\n".join(lines))

		# Root cause insights from successes
		successes = [
			se.record for se in similar if se.record.outcome == "success"
		]
		if successes:
			# Bottleneck distribution
			bottlenecks = [r.bottleneck_type for r in successes]
			bt_counts = Counter(bottlenecks)
			if bt_counts:
				dominant = bt_counts.most_common(1)[0]
				sections.append(
					f"### Typical bottleneck: {dominant[0]} "
					f"({dominant[1]}/{len(successes)} successful attempts)"
				)

		# Specific root causes
		root_causes = [
			r.root_cause for r in successes
			if r.root_cause and len(r.root_cause) > 20
		]
		if root_causes:
			lines = ["### Why strategies worked:"]
			seen: set[str] = set()
			for cause in root_causes[:3]:
				short = cause[:150]
				if short not in seen:
					lines.append(f"- {short}")
					seen.add(short)
			sections.append("\n".join(lines))

		ctx = "\n\n".join(sections)
		char_budget = max_tokens * 4
		return ctx[:char_budget]
