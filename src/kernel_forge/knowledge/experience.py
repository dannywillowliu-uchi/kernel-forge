"""Structured experience records for cross-problem learning.

Records WHY strategies work or fail, not just WHAT happened.
Enables generalization: "TF32 works on compute-bound matmul problems
because baselines don't enable tensor cores."
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExperienceRecord:
	"""A structured learning from one optimization attempt."""

	# What problem
	problem_name: str
	kernel_class: str  # matmul, elementwise, reduction, conv, etc.

	# What was tried
	strategy_name: str
	approach_notes: str

	# What happened
	outcome: str  # "success", "correctness_failure", "compilation_error", etc.
	speedup: float
	baseline_ms: float
	optimized_ms: float

	# WHY it worked or failed
	bottleneck_type: str  # compute_bound, memory_bound, etc.
	roofline_utilization_pct: float
	root_cause: str  # human-readable explanation of why

	# Context that matters for generalization
	input_shapes: dict
	precision_constraint: str  # "fp32_strict", "bf16_ok", etc.


@dataclass
class ClassPattern:
	"""Generalized pattern for a kernel class.

	Derived from multiple ExperienceRecords for the same kernel_class.
	"""

	kernel_class: str
	total_problems: int
	total_attempts: int

	# What works
	best_strategies: list[dict]  # [{name, avg_speedup, success_rate}]

	# What fails
	common_failures: list[dict]  # [{strategy, failure_type, frequency}]

	# Key insights
	insights: list[str]  # Human-readable generalizations


class ExperienceStore:
	"""Persists and queries structured experience records."""

	def __init__(self, path: Path | str) -> None:
		self._path = Path(path)
		self._path.mkdir(parents=True, exist_ok=True)
		self._records_file = self._path / "experience_records.jsonl"
		self._patterns_file = self._path / "class_patterns.json"

	def record(self, exp: ExperienceRecord) -> None:
		"""Append an experience record."""
		with open(self._records_file, "a") as f:
			f.write(json.dumps(asdict(exp)) + "\n")
		logger.info(
			"Recorded experience: %s/%s -> %s (%.2fx)",
			exp.kernel_class, exp.strategy_name,
			exp.outcome, exp.speedup,
		)

	def get_records(
		self, kernel_class: str | None = None
	) -> list[ExperienceRecord]:
		"""Read all records, optionally filtered by kernel class."""
		if not self._records_file.exists():
			return []
		records: list[ExperienceRecord] = []
		with open(self._records_file) as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				data = json.loads(line)
				rec = ExperienceRecord(**data)
				if kernel_class is None or rec.kernel_class == kernel_class:
					records.append(rec)
		return records

	def get_pattern(self, kernel_class: str) -> ClassPattern | None:
		"""Get the generalized pattern for a kernel class."""
		records = self.get_records(kernel_class)
		if not records:
			return None
		return self._derive_pattern(kernel_class, records)

	def build_context_for_class(
		self, kernel_class: str, max_tokens: int = 4000
	) -> str:
		"""Build LLM-readable context from experience with this kernel class."""
		pattern = self.get_pattern(kernel_class)
		if pattern is None:
			return ""

		sections: list[str] = []
		sections.append(
			f"## Experience with {kernel_class} kernels "
			f"({pattern.total_problems} problems, "
			f"{pattern.total_attempts} attempts)"
		)

		if pattern.best_strategies:
			lines = ["### What works:"]
			for s in pattern.best_strategies[:5]:
				lines.append(
					f"- **{s['name']}**: {s['avg_speedup']:.2f}x avg "
					f"({s['success_rate']:.0%} success rate)"
				)
			sections.append("\n".join(lines))

		if pattern.common_failures:
			lines = ["### What fails:"]
			for f in pattern.common_failures[:5]:
				lines.append(
					f"- {f['strategy']}: {f['failure_type']} "
					f"({f['frequency']} times)"
				)
			sections.append("\n".join(lines))

		if pattern.insights:
			lines = ["### Key insights:"]
			for insight in pattern.insights[:5]:
				lines.append(f"- {insight}")
			sections.append("\n".join(lines))

		ctx = "\n\n".join(sections)
		char_budget = max_tokens * 4
		return ctx[:char_budget]

	def _derive_pattern(
		self, kernel_class: str, records: list[ExperienceRecord]
	) -> ClassPattern:
		"""Derive generalized patterns from records."""
		problems = set(r.problem_name for r in records)

		# Best strategies by avg speedup
		strategy_stats: dict[str, dict] = {}
		for r in records:
			if r.strategy_name not in strategy_stats:
				strategy_stats[r.strategy_name] = {
					"speedups": [],
					"successes": 0,
					"total": 0,
				}
			stats = strategy_stats[r.strategy_name]
			stats["total"] += 1
			if r.outcome == "success":
				stats["successes"] += 1
				stats["speedups"].append(r.speedup)

		best_strategies = []
		for name, stats in strategy_stats.items():
			if stats["speedups"]:
				best_strategies.append({
					"name": name,
					"avg_speedup": sum(stats["speedups"]) / len(stats["speedups"]),
					"success_rate": stats["successes"] / max(stats["total"], 1),
				})
		best_strategies.sort(key=lambda x: x["avg_speedup"], reverse=True)

		# Common failures
		failure_counts: dict[str, dict] = {}
		for r in records:
			if r.outcome != "success":
				key = f"{r.strategy_name}|{r.outcome}"
				if key not in failure_counts:
					failure_counts[key] = {
						"strategy": r.strategy_name,
						"failure_type": r.outcome,
						"frequency": 0,
						"example_cause": r.root_cause,
					}
				failure_counts[key]["frequency"] += 1
		common_failures = sorted(
			failure_counts.values(),
			key=lambda x: x["frequency"],
			reverse=True,
		)

		# Derive insights
		insights: list[str] = []
		successes = [r for r in records if r.outcome == "success"]
		failures = [r for r in records if r.outcome != "success"]

		if successes:
			avg_speedup = sum(r.speedup for r in successes) / len(successes)
			best = max(successes, key=lambda r: r.speedup)
			insights.append(
				f"Average speedup on {kernel_class}: {avg_speedup:.2f}x "
				f"(best: {best.speedup:.2f}x via {best.strategy_name})"
			)

		# Bottleneck consistency
		bottlenecks = [r.bottleneck_type for r in records if r.bottleneck_type]
		if bottlenecks:
			from collections import Counter
			bt_counts = Counter(bottlenecks)
			dominant = bt_counts.most_common(1)[0]
			if dominant[1] / len(bottlenecks) > 0.6:
				insights.append(
					f"{kernel_class} kernels are typically "
					f"{dominant[0]} ({dominant[1]}/{len(bottlenecks)} cases)"
				)

		# Precision failures
		prec_failures = [
			r for r in failures if "correctness" in r.outcome
		]
		if prec_failures:
			insights.append(
				f"Correctness failures common on {kernel_class} "
				f"({len(prec_failures)}/{len(records)} attempts). "
				f"Common cause: {prec_failures[0].root_cause[:100]}"
			)

		return ClassPattern(
			kernel_class=kernel_class,
			total_problems=len(problems),
			total_attempts=len(records),
			best_strategies=best_strategies,
			common_failures=common_failures,
			insights=insights,
		)
