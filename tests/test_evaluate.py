"""Tests for evaluation functions: scoring, escalation, termination, failure classification."""

from __future__ import annotations

from kernel_forge.core.evaluate import (
	classify_failure,
	compute_roofline,
	compute_score,
	should_escalate_profiling,
	should_terminate,
)
from kernel_forge.core.types import (
	B200_PEAKS,
	Attempt,
	FailureType,
	TerminationConfig,
)


class TestComputeScore:
	def test_correct_returns_speedup(self) -> None:
		assert compute_score(1.5, True) == 1.5

	def test_incorrect_returns_zero(self) -> None:
		assert compute_score(2.0, False) == 0.0

	def test_zero_speedup_correct(self) -> None:
		assert compute_score(0.0, True) == 0.0

	def test_negative_speedup_correct(self) -> None:
		"""Edge case: regression should still return raw speedup if correct."""
		assert compute_score(0.5, True) == 0.5


class TestShouldEscalateProfiling:
	def _make_attempt(
		self, speedup: float, correct: bool = True, tier: str = "cuda_events"
	) -> Attempt:
		return Attempt(
			kernel_problem="test",
			strategy_name="test",
			speedup=speedup,
			correct=correct,
			hardware="b200",
			optimization_goal="latency",
			profiling_tier=tier,
		)

	def test_not_enough_attempts(self) -> None:
		config = TerminationConfig(plateau_window_cuda_events=3)
		attempts = [self._make_attempt(1.1), self._make_attempt(1.11)]
		assert should_escalate_profiling(attempts, config) is False

	def test_plateau_triggers_escalation(self) -> None:
		config = TerminationConfig(plateau_threshold=0.02, plateau_window_cuda_events=3)
		attempts = [
			self._make_attempt(1.5),
			self._make_attempt(1.51),
			self._make_attempt(1.515),
		]
		assert should_escalate_profiling(attempts, config) is True

	def test_improvement_prevents_escalation(self) -> None:
		config = TerminationConfig(plateau_threshold=0.02, plateau_window_cuda_events=3)
		attempts = [
			self._make_attempt(1.5),
			self._make_attempt(1.51),
			self._make_attempt(1.6),  # >2% improvement
		]
		assert should_escalate_profiling(attempts, config) is False

	def test_ignores_incorrect_attempts(self) -> None:
		config = TerminationConfig(plateau_threshold=0.02, plateau_window_cuda_events=3)
		attempts = [
			self._make_attempt(1.5),
			self._make_attempt(0.0, correct=False),
			self._make_attempt(1.51),
			self._make_attempt(1.515),
		]
		assert should_escalate_profiling(attempts, config) is True

	def test_ignores_ncu_tier_attempts(self) -> None:
		config = TerminationConfig(plateau_threshold=0.02, plateau_window_cuda_events=3)
		attempts = [
			self._make_attempt(1.5),
			self._make_attempt(1.51, tier="ncu"),
			self._make_attempt(1.505),
			self._make_attempt(1.508),
		]
		# Only 3 correct cuda_events: 1.5, 1.505, 1.508 -> plateau
		assert should_escalate_profiling(attempts, config) is True

	def test_empty_attempts(self) -> None:
		config = TerminationConfig()
		assert should_escalate_profiling([], config) is False


class TestShouldTerminate:
	def test_max_attempts(self) -> None:
		config = TerminationConfig(max_attempts=10)
		assert should_terminate(10, 0.0, 0.0, 0, config) is True
		assert should_terminate(9, 0.0, 0.0, 0, config) is False

	def test_max_cost(self) -> None:
		config = TerminationConfig(max_cost_usd=5.0)
		assert should_terminate(0, 5.0, 0.0, 0, config) is True
		assert should_terminate(0, 4.99, 0.0, 0, config) is False

	def test_max_wall_time(self) -> None:
		config = TerminationConfig(max_wall_time_seconds=1800)
		assert should_terminate(0, 0.0, 1800.0, 0, config) is True
		assert should_terminate(0, 0.0, 1799.0, 0, config) is False

	def test_max_consecutive_failures(self) -> None:
		config = TerminationConfig(max_consecutive_failures=5)
		assert should_terminate(0, 0.0, 0.0, 5, config) is True
		assert should_terminate(0, 0.0, 0.0, 4, config) is False

	def test_no_triggers(self) -> None:
		config = TerminationConfig()
		assert should_terminate(0, 0.0, 0.0, 0, config) is False


class TestClassifyFailure:
	def test_timeout(self) -> None:
		assert classify_failure(-1, "", "") == FailureType.TIMEOUT
		assert classify_failure(1, "timed out", "") == FailureType.TIMEOUT

	def test_compilation_error(self) -> None:
		result = classify_failure(
			1, "error: expected ';' nvcc compile failed", ""
		)
		assert result == FailureType.COMPILATION_ERROR

	def test_compilation_error_cu_file(self) -> None:
		result = classify_failure(1, "kernel.cu:10: error: syntax error", "")
		assert result == FailureType.COMPILATION_ERROR

	def test_link_error(self) -> None:
		result = classify_failure(1, "undefined symbol: _Z3foov", "")
		assert result == FailureType.LINK_ERROR

	def test_link_error_undefined_reference(self) -> None:
		result = classify_failure(1, "undefined reference to `cudaMalloc'", "")
		assert result == FailureType.LINK_ERROR

	def test_runtime_oom(self) -> None:
		result = classify_failure(1, "CUDA error: out of memory", "")
		assert result == FailureType.RUNTIME_OOM

	def test_runtime_segfault(self) -> None:
		result = classify_failure(1, "SIGSEGV: segmentation fault", "")
		assert result == FailureType.RUNTIME_SEGFAULT

	def test_runtime_segfault_illegal_access(self) -> None:
		result = classify_failure(
			1, "an illegal memory access was encountered", ""
		)
		assert result == FailureType.RUNTIME_SEGFAULT

	def test_correctness_failure(self) -> None:
		result = classify_failure(1, "", "allclose failed")
		assert result == FailureType.CORRECTNESS_FAILURE

	def test_numerical_instability(self) -> None:
		result = classify_failure(1, "output contains nan values", "")
		assert result == FailureType.NUMERICAL_INSTABILITY

	def test_performance_regression(self) -> None:
		result = classify_failure(0, "performance regression detected", "")
		assert result == FailureType.PERFORMANCE_REGRESSION

	def test_unknown_nonzero_defaults_to_compilation(self) -> None:
		result = classify_failure(1, "something unknown happened", "")
		assert result == FailureType.COMPILATION_ERROR


class TestComputeRoofline:
	"""Test roofline gap analysis."""

	def test_matmul_4096_tf32(self) -> None:
		"""Real result from our first KernelBench run on B200."""
		n = 4096
		flops = 2 * n**3  # 137.4 GFLOP
		bytes_moved = 3 * n * n * 4  # A + B read + C write, FP32
		result = compute_roofline(
			runtime_ms=0.1828,
			flops=flops,
			bytes_moved=bytes_moved,
			precision="tf32",
		)
		assert result.achieved_tflops > 700
		assert result.utilization_pct > 70
		assert result.headroom_pct < 30
		assert result.roofline_bound == "compute_bound"
		assert result.worth_optimizing is True  # 22% headroom > 10% threshold

	def test_near_peak_not_worth_optimizing(self) -> None:
		"""When utilization is >90%, declare near-optimal."""
		result = compute_roofline(
			runtime_ms=0.15,
			flops=2 * 4096**3,
			bytes_moved=3 * 4096 * 4096 * 4,
			precision="tf32",
			headroom_threshold=10.0,
		)
		assert result.utilization_pct > 90
		assert result.worth_optimizing is False
		assert "near-optimal" in result.explanation.lower() or "only" in result.explanation.lower()

	def test_memory_bound_kernel(self) -> None:
		"""Elementwise kernel with low arithmetic intensity."""
		n = 1024 * 1024
		flops = n  # 1 FLOP per element
		bytes_moved = 2 * n * 4  # read + write, FP32
		result = compute_roofline(
			runtime_ms=0.01,
			flops=flops,
			bytes_moved=bytes_moved,
			precision="tf32",
		)
		assert result.roofline_bound == "memory_bound"
		assert result.arithmetic_intensity < 1.0

	def test_bf16_precision(self) -> None:
		"""BF16 should use BF16 peak."""
		result = compute_roofline(
			runtime_ms=0.0906,
			flops=2 * 4096**3,
			bytes_moved=3 * 4096 * 4096 * 2,  # BF16 = 2 bytes
			precision="bf16",
		)
		assert result.peak_tflops == B200_PEAKS.bf16_tflops
		assert result.utilization_pct > 70

	def test_slow_kernel_has_high_headroom(self) -> None:
		"""A very slow kernel should have high headroom and be worth optimizing."""
		result = compute_roofline(
			runtime_ms=100.0,
			flops=2 * 4096**3,
			bytes_moved=3 * 4096 * 4096 * 4,
			precision="tf32",
		)
		assert result.headroom_pct > 99
		assert result.worth_optimizing is True

	def test_b200_peaks_values(self) -> None:
		"""Verify B200 peak constants."""
		assert B200_PEAKS.bf16_tflops == 1929.0
		assert B200_PEAKS.fp32_tflops == 481.0
		assert B200_PEAKS.hbm_bandwidth_tb_s == 8.0
