"""Telemetry for the optimization loop.

Wraps each step with timing spans and tracks token usage,
GPU time, and bottleneck identification. Inspired by NVIDIA
AIQ Toolkit's callback-injection profiling pattern.

Usage:
    tracker = RunTracker(problem_name="matmul")
    with tracker.span("baseline") as s:
        baseline_ms = await run_baseline()
        s.set("baseline_ms", baseline_ms)
    tracker.report()  # prints structured summary
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Span:
	"""A single timed operation in the optimization pipeline."""

	name: str
	start_time: float = 0.0
	end_time: float = 0.0
	metadata: dict = field(default_factory=dict)
	children: list[Span] = field(default_factory=list)

	@property
	def duration_ms(self) -> float:
		return (self.end_time - self.start_time) * 1000

	def set(self, key: str, value: object) -> None:
		self.metadata[key] = value

	def to_dict(self) -> dict:
		d: dict = {
			"name": self.name,
			"duration_ms": round(self.duration_ms, 1),
		}
		if self.metadata:
			d["metadata"] = self.metadata
		if self.children:
			d["children"] = [c.to_dict() for c in self.children]
		return d


class SpanContext:
	"""Context manager for timing a span."""

	def __init__(self, span: Span) -> None:
		self._span = span

	def __enter__(self) -> Span:
		self._span.start_time = time.monotonic()
		return self._span

	def __exit__(self, *args: object) -> None:
		self._span.end_time = time.monotonic()

	async def __aenter__(self) -> Span:
		self._span.start_time = time.monotonic()
		return self._span

	async def __aexit__(self, *args: object) -> None:
		self._span.end_time = time.monotonic()


class RunTracker:
	"""Tracks timing and metadata across an optimization run.

	Produces a structured trace showing where time is spent:
	agent calls, GPU benchmarks, experience queries, etc.
	"""

	def __init__(self, problem_name: str) -> None:
		self.problem_name = problem_name
		self.root = Span(name=f"optimize:{problem_name}")
		self.root.start_time = time.monotonic()
		self._stack: list[Span] = [self.root]

		# Aggregate counters
		self.total_agent_calls: int = 0
		self.total_agent_tokens_in: int = 0
		self.total_agent_tokens_out: int = 0
		self.total_gpu_time_ms: float = 0.0
		self.total_agent_time_ms: float = 0.0

	def span(self, name: str) -> SpanContext:
		"""Create a new child span under the current span."""
		child = Span(name=name)
		self._stack[-1].children.append(child)
		return SpanContext(child)

	def record_agent_call(
		self,
		duration_ms: float,
		tokens_in: int = 0,
		tokens_out: int = 0,
	) -> None:
		self.total_agent_calls += 1
		self.total_agent_tokens_in += tokens_in
		self.total_agent_tokens_out += tokens_out
		self.total_agent_time_ms += duration_ms

	def record_gpu_time(self, duration_ms: float) -> None:
		self.total_gpu_time_ms += duration_ms

	def finish(self) -> None:
		self.root.end_time = time.monotonic()

	@property
	def total_ms(self) -> float:
		return self.root.duration_ms

	def summary(self) -> dict:
		"""Structured summary of the run."""
		self.finish()
		total = self.total_ms
		return {
			"problem": self.problem_name,
			"total_ms": round(total, 1),
			"total_seconds": round(total / 1000, 1),
			"agent_calls": self.total_agent_calls,
			"agent_time_ms": round(self.total_agent_time_ms, 1),
			"agent_time_pct": (
				round(self.total_agent_time_ms / total * 100, 1)
				if total > 0 else 0
			),
			"gpu_time_ms": round(self.total_gpu_time_ms, 1),
			"gpu_time_pct": (
				round(self.total_gpu_time_ms / total * 100, 1)
				if total > 0 else 0
			),
			"tokens_in": self.total_agent_tokens_in,
			"tokens_out": self.total_agent_tokens_out,
			"overhead_pct": round(
				100 - (self.total_agent_time_ms + self.total_gpu_time_ms)
				/ max(total, 1) * 100, 1
			),
		}

	def report(self) -> str:
		"""Human-readable report."""
		s = self.summary()
		lines = [
			f"=== Run: {s['problem']} ({s['total_seconds']}s) ===",
			f"Agent: {s['agent_calls']} calls, "
			f"{s['agent_time_ms']:.0f}ms ({s['agent_time_pct']}%)",
			f"GPU:   {s['gpu_time_ms']:.0f}ms ({s['gpu_time_pct']}%)",
			f"Overhead: {s['overhead_pct']}%",
			f"Tokens: {s['tokens_in']} in, {s['tokens_out']} out",
		]
		return "\n".join(lines)

	def save(self, path: Path) -> None:
		"""Save full trace to JSON."""
		data = {
			"summary": self.summary(),
			"trace": self.root.to_dict(),
		}
		path.write_text(json.dumps(data, indent=2))
