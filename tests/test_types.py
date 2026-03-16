"""Tests for core data types."""

from kernel_forge.core.types import (
	Attempt,
	BottleneckType,
	Diagnosis,
	FailureReport,
	FailureType,
	GoalConstraints,
	KernelCandidate,
	KernelProblem,
	OptimizationGoal,
	ProfileData,
	Strategy,
	StrategyCategory,
	TerminationConfig,
)


class TestKernelProblem:
	def test_creation(self) -> None:
		problem = KernelProblem(
			name="matmul_tiled",
			reference_source="def model(a, b): return a @ b",
			input_shapes={"a": [512, 512], "b": [512, 512]},
		)
		assert problem.name == "matmul_tiled"
		assert problem.reference_source == "def model(a, b): return a @ b"
		assert problem.input_shapes == {"a": [512, 512], "b": [512, 512]}

	def test_defaults(self) -> None:
		problem = KernelProblem(
			name="test",
			reference_source="pass",
			input_shapes={},
		)
		assert problem.benchmark_suite == "kernelbench"
		assert problem.difficulty_level is None


class TestOptimizationGoal:
	def test_creation(self) -> None:
		goal = OptimizationGoal(primary="latency")
		assert goal.primary == "latency"
		assert isinstance(goal.constraints, GoalConstraints)

	def test_with_constraints(self) -> None:
		constraints = GoalConstraints(
			correctness_rtol=1e-2,
			max_memory_mb=4096,
			batch_sizes=[1, 8, 16],
			precision="fp32",
		)
		goal = OptimizationGoal(primary="throughput", constraints=constraints)
		assert goal.constraints.correctness_rtol == 1e-2
		assert goal.constraints.max_memory_mb == 4096
		assert goal.constraints.batch_sizes == [1, 8, 16]
		assert goal.constraints.precision == "fp32"


class TestGoalConstraints:
	def test_defaults(self) -> None:
		c = GoalConstraints()
		assert c.correctness_rtol == 1e-3
		assert c.correctness_atol == 1e-3
		assert c.max_memory_mb is None
		assert c.batch_sizes == [1]
		assert c.precision == "bf16"


class TestStrategy:
	def test_creation(self) -> None:
		s = Strategy(
			id="shared_mem_tiling",
			name="Shared Memory Tiling",
			category=StrategyCategory.MEMORY_OPT,
			description="Use shared memory to reduce global memory accesses",
			applicability="memory-bound matmul kernels",
			expected_impact="1.5x-3x for memory-bound kernels",
		)
		assert s.id == "shared_mem_tiling"
		assert s.category == StrategyCategory.MEMORY_OPT
		assert s.name == "Shared Memory Tiling"


class TestStrategyCategory:
	def test_all_categories_exist(self) -> None:
		categories = [
			StrategyCategory.MEMORY_OPT,
			StrategyCategory.COMPUTE_OPT,
			StrategyCategory.PRECISION_OPT,
			StrategyCategory.PIPELINE_OPT,
			StrategyCategory.EXECUTION_OPT,
			StrategyCategory.ALGORITHMIC,
		]
		assert len(categories) == 6

	def test_values(self) -> None:
		assert StrategyCategory.MEMORY_OPT.value == "memory_opt"
		assert StrategyCategory.ALGORITHMIC.value == "algorithmic"


class TestAttempt:
	def test_creation(self) -> None:
		attempt = Attempt(
			kernel_problem="matmul_tiled",
			strategy_name="shared_mem_tiling",
			speedup=1.5,
			correct=True,
			hardware="b200",
			optimization_goal="latency",
		)
		assert attempt.speedup == 1.5
		assert attempt.correct is True

	def test_defaults(self) -> None:
		attempt = Attempt(
			kernel_problem="test",
			strategy_name="test_strategy",
			speedup=1.0,
			correct=True,
			hardware="b200",
			optimization_goal="latency",
		)
		assert attempt.kernel_source_hash is None
		assert attempt.input_tokens is None
		assert attempt.output_tokens is None
		assert attempt.cost_usd is None
		assert attempt.profiling_tier == "cuda_events"
		assert attempt.failure_report is None


class TestKernelCandidate:
	def test_creation(self) -> None:
		candidate = KernelCandidate(
			source="__global__ void kernel() {}",
			approach_notes="Basic kernel implementation",
			strategy_name="shared_mem_tiling",
		)
		assert candidate.source == "__global__ void kernel() {}"
		assert candidate.approach_notes == "Basic kernel implementation"
		assert candidate.strategy_name == "shared_mem_tiling"


class TestProfileData:
	def test_creation(self) -> None:
		profile = ProfileData(
			runtime_us=150.0,
			profiling_tier="cuda_events",
		)
		assert profile.runtime_us == 150.0
		assert profile.profiling_tier == "cuda_events"
		assert profile.metrics == {}
		assert profile.raw_output is None

	def test_with_metrics(self) -> None:
		profile = ProfileData(
			runtime_us=150.0,
			profiling_tier="ncu",
			metrics={"memory_throughput_pct": 85.0, "compute_utilization_pct": 30.0},
			raw_output="ncu output here",
		)
		assert profile.metrics["memory_throughput_pct"] == 85.0
		assert profile.raw_output == "ncu output here"


class TestDiagnosis:
	def test_creation(self) -> None:
		diag = Diagnosis(
			bottleneck_type=BottleneckType.MEMORY_BOUND,
			explanation="High memory throughput, low compute utilization",
			evidence={"memory_throughput_pct": 85.0},
			profiling_tier="ncu",
		)
		assert diag.bottleneck_type == BottleneckType.MEMORY_BOUND
		assert diag.explanation == "High memory throughput, low compute utilization"
		assert diag.evidence["memory_throughput_pct"] == 85.0


class TestBottleneckType:
	def test_all_types_exist(self) -> None:
		types = [
			BottleneckType.MEMORY_BOUND,
			BottleneckType.COMPUTE_BOUND,
			BottleneckType.LAUNCH_OVERHEAD,
			BottleneckType.OCCUPANCY_LIMITED,
			BottleneckType.MIXED,
		]
		assert len(types) == 5

	def test_values(self) -> None:
		assert BottleneckType.MEMORY_BOUND.value == "memory_bound"
		assert BottleneckType.MIXED.value == "mixed"


class TestFailureReport:
	def test_creation(self) -> None:
		report = FailureReport(
			failure_type=FailureType.COMPILATION_ERROR,
			error_output="error: expected ';' at end",
			kernel_source="__global__ void kernel() {}",
			strategy_name="shared_mem_tiling",
		)
		assert report.failure_type == FailureType.COMPILATION_ERROR
		assert report.error_output == "error: expected ';' at end"


class TestFailureType:
	def test_all_types_exist(self) -> None:
		types = [
			FailureType.COMPILATION_ERROR,
			FailureType.LINK_ERROR,
			FailureType.RUNTIME_SEGFAULT,
			FailureType.RUNTIME_OOM,
			FailureType.CORRECTNESS_FAILURE,
			FailureType.NUMERICAL_INSTABILITY,
			FailureType.TIMEOUT,
			FailureType.PERFORMANCE_REGRESSION,
		]
		assert len(types) == 8

	def test_values(self) -> None:
		assert FailureType.COMPILATION_ERROR.value == "compilation_error"
		assert FailureType.PERFORMANCE_REGRESSION.value == "performance_regression"


class TestTerminationConfig:
	def test_defaults(self) -> None:
		config = TerminationConfig()
		assert config.plateau_threshold == 0.02
		assert config.plateau_window_cuda_events == 3
		assert config.plateau_window_ncu == 3
		assert config.max_attempts == 25
		assert config.max_cost_usd == 5.0
		assert config.max_wall_time_seconds == 1800
		assert config.max_consecutive_failures == 5

	def test_custom_values(self) -> None:
		config = TerminationConfig(
			plateau_threshold=0.05,
			max_attempts=50,
			max_cost_usd=10.0,
		)
		assert config.plateau_threshold == 0.05
		assert config.max_attempts == 50
		assert config.max_cost_usd == 10.0
