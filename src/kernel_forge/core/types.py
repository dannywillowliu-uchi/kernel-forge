"""Core data types for kernel problems, strategies, attempts, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class StrategyCategory(Enum):
	"""Categories of optimization strategies."""

	MEMORY_OPT = "memory_opt"
	COMPUTE_OPT = "compute_opt"
	PRECISION_OPT = "precision_opt"
	PIPELINE_OPT = "pipeline_opt"
	EXECUTION_OPT = "execution_opt"
	ALGORITHMIC = "algorithmic"


class BottleneckType(Enum):
	"""Types of performance bottlenecks."""

	MEMORY_BOUND = "memory_bound"
	COMPUTE_BOUND = "compute_bound"
	LAUNCH_OVERHEAD = "launch_overhead"
	OCCUPANCY_LIMITED = "occupancy_limited"
	MIXED = "mixed"


class FailureType(Enum):
	"""Types of kernel execution failures."""

	COMPILATION_ERROR = "compilation_error"
	LINK_ERROR = "link_error"
	RUNTIME_SEGFAULT = "runtime_segfault"
	RUNTIME_OOM = "runtime_oom"
	CORRECTNESS_FAILURE = "correctness_failure"
	NUMERICAL_INSTABILITY = "numerical_instability"
	TIMEOUT = "timeout"
	PERFORMANCE_REGRESSION = "performance_regression"


@dataclass
class GoalConstraints:
	"""Constraints for optimization goals."""

	correctness_rtol: float = 1e-3
	correctness_atol: float = 1e-3
	max_memory_mb: int | None = None
	batch_sizes: list[int] = field(default_factory=lambda: [1])
	precision: str = "bf16"


@dataclass
class OptimizationGoal:
	"""Optimization goal for a kernel problem."""

	primary: Literal["latency", "throughput", "memory", "balanced"] = "latency"
	constraints: GoalConstraints = field(default_factory=GoalConstraints)


@dataclass
class KernelProblem:
	"""A kernel optimization problem definition."""

	name: str
	reference_source: str
	input_shapes: dict[str, list[int]]
	benchmark_suite: str = "kernelbench"
	difficulty_level: int | None = None


@dataclass
class Strategy:
	"""An optimization strategy from the knowledge base."""

	id: str
	name: str
	category: StrategyCategory
	description: str
	applicability: str
	expected_impact: str


@dataclass
class Attempt:
	"""A single optimization attempt record."""

	kernel_problem: str
	strategy_name: str
	speedup: float
	correct: bool
	hardware: str
	optimization_goal: str
	kernel_source_hash: str | None = None
	input_tokens: int | None = None
	output_tokens: int | None = None
	cost_usd: float | None = None
	profiling_tier: str = "cuda_events"
	failure_report: FailureReport | None = None


@dataclass
class KernelCandidate:
	"""A generated kernel candidate from an agent."""

	source: str
	approach_notes: str
	strategy_name: str


@dataclass
class ProfileData:
	"""Profiling data from a kernel run."""

	runtime_us: float
	profiling_tier: str
	metrics: dict[str, float] = field(default_factory=dict)
	raw_output: str | None = None


@dataclass
class Diagnosis:
	"""Bottleneck diagnosis from profiling data."""

	bottleneck_type: BottleneckType
	explanation: str
	evidence: dict[str, Any]
	profiling_tier: str


@dataclass
class FailureReport:
	"""Report of a kernel execution failure."""

	failure_type: FailureType
	error_output: str
	kernel_source: str
	strategy_name: str


@dataclass
class TerminationConfig:
	"""Configuration for loop termination conditions."""

	plateau_threshold: float = 0.02
	plateau_window_cuda_events: int = 3
	plateau_window_ncu: int = 3
	max_attempts: int = 25
	max_cost_usd: float = 5.0
	max_wall_time_seconds: int = 1800
	max_consecutive_failures: int = 5
