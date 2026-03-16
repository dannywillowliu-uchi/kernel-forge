"""Tests for evaluation functions: scoring, escalation, termination, failure classification."""

from __future__ import annotations

from kernel_forge.core.evaluate import (
	classify_failure,
	compute_score,
	should_escalate_profiling,
	should_terminate,
)
from kernel_forge.core.types import Attempt, FailureType, TerminationConfig


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
