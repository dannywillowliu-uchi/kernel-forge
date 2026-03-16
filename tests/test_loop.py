"""Tests for LoopState: state tracking, should_stop, should_escalate, kernel_hash."""

from __future__ import annotations

import time

from kernel_forge.config import ForgeConfig
from kernel_forge.core.loop import LoopState
from kernel_forge.core.types import (
	Attempt,
	KernelProblem,
	OptimizationGoal,
	TerminationConfig,
)


def _make_state(**overrides: object) -> LoopState:
	"""Create a LoopState with sensible defaults for testing."""
	defaults: dict[str, object] = {
		"problem": KernelProblem(
			name="test_problem",
			reference_source="pass",
			input_shapes={},
		),
		"goal": OptimizationGoal(primary="latency"),
		"config": ForgeConfig(),
	}
	defaults.update(overrides)
	return LoopState(**defaults)  # type: ignore[arg-type]


class TestLoopStateCreation:
	def test_default_values(self) -> None:
		state = _make_state()
		assert state.attempt_count == 0
		assert state.best_speedup == 0.0
		assert state.best_kernel_source is None
		assert state.total_cost == 0.0
		assert state.consecutive_failures == 0
		assert state.profiling_tier == "cuda_events"
		assert state.attempts == []

	def test_problem_and_goal_stored(self) -> None:
		state = _make_state()
		assert state.problem.name == "test_problem"
		assert state.goal.primary == "latency"


class TestRecordAttempt:
	def test_increments_attempt_count(self) -> None:
		state = _make_state()
		state.record_attempt(1.5, True, 0.10)
		assert state.attempt_count == 1
		state.record_attempt(1.6, True, 0.10)
		assert state.attempt_count == 2

	def test_accumulates_cost(self) -> None:
		state = _make_state()
		state.record_attempt(1.5, True, 0.10)
		state.record_attempt(1.6, True, 0.15)
		assert abs(state.total_cost - 0.25) < 1e-9

	def test_updates_best_speedup(self) -> None:
		state = _make_state()
		state.record_attempt(1.5, True, 0.10)
		assert state.best_speedup == 1.5
		state.record_attempt(2.0, True, 0.10)
		assert state.best_speedup == 2.0

	def test_does_not_update_best_on_lower_speedup(self) -> None:
		state = _make_state()
		state.record_attempt(2.0, True, 0.10)
		state.record_attempt(1.5, True, 0.10)
		assert state.best_speedup == 2.0

	def test_incorrect_attempt_does_not_update_best(self) -> None:
		state = _make_state()
		state.record_attempt(1.0, True, 0.10)
		state.record_attempt(5.0, False, 0.10)
		assert state.best_speedup == 1.0

	def test_incorrect_increments_consecutive_failures(self) -> None:
		state = _make_state()
		state.record_attempt(0.0, False, 0.10)
		assert state.consecutive_failures == 1
		state.record_attempt(0.0, False, 0.10)
		assert state.consecutive_failures == 2

	def test_correct_resets_consecutive_failures(self) -> None:
		state = _make_state()
		state.record_attempt(0.0, False, 0.10)
		state.record_attempt(0.0, False, 0.10)
		assert state.consecutive_failures == 2
		state.record_attempt(1.5, True, 0.10)
		assert state.consecutive_failures == 0

	def test_correct_no_improvement_resets_failures(self) -> None:
		"""A correct but non-improving attempt should still reset failures."""
		state = _make_state()
		state.record_attempt(2.0, True, 0.10)
		state.record_attempt(0.0, False, 0.10)
		state.record_attempt(0.0, False, 0.10)
		assert state.consecutive_failures == 2
		state.record_attempt(1.0, True, 0.10)  # Correct but lower than best
		assert state.consecutive_failures == 0


class TestShouldStop:
	def test_stops_on_max_attempts(self) -> None:
		config = ForgeConfig(termination=TerminationConfig(max_attempts=3))
		state = _make_state(config=config)
		state.record_attempt(1.0, True, 0.0)
		state.record_attempt(1.0, True, 0.0)
		state.record_attempt(1.0, True, 0.0)
		assert state.should_stop is True

	def test_stops_on_max_cost(self) -> None:
		config = ForgeConfig(termination=TerminationConfig(max_cost_usd=1.0))
		state = _make_state(config=config)
		state.record_attempt(1.0, True, 0.5)
		assert state.should_stop is False
		state.record_attempt(1.0, True, 0.5)
		assert state.should_stop is True

	def test_stops_on_consecutive_failures(self) -> None:
		config = ForgeConfig(
			termination=TerminationConfig(max_consecutive_failures=2)
		)
		state = _make_state(config=config)
		state.record_attempt(0.0, False, 0.0)
		assert state.should_stop is False
		state.record_attempt(0.0, False, 0.0)
		assert state.should_stop is True

	def test_does_not_stop_normally(self) -> None:
		state = _make_state()
		state.record_attempt(1.5, True, 0.10)
		assert state.should_stop is False

	def test_stops_on_wall_time(self) -> None:
		config = ForgeConfig(
			termination=TerminationConfig(max_wall_time_seconds=0)
		)
		state = _make_state(config=config, start_time=time.time() - 1)
		assert state.should_stop is True


class TestShouldEscalate:
	def _make_attempt(self, speedup: float) -> Attempt:
		return Attempt(
			kernel_problem="test",
			strategy_name="test",
			speedup=speedup,
			correct=True,
			hardware="b200",
			optimization_goal="latency",
			profiling_tier="cuda_events",
		)

	def test_escalates_on_plateau(self) -> None:
		config = ForgeConfig(
			termination=TerminationConfig(
				plateau_threshold=0.02,
				plateau_window_cuda_events=3,
			)
		)
		state = _make_state(config=config)
		state.attempts = [
			self._make_attempt(1.5),
			self._make_attempt(1.51),
			self._make_attempt(1.515),
		]
		assert state.should_escalate is True

	def test_no_escalation_with_improvement(self) -> None:
		config = ForgeConfig(
			termination=TerminationConfig(
				plateau_threshold=0.02,
				plateau_window_cuda_events=3,
			)
		)
		state = _make_state(config=config)
		state.attempts = [
			self._make_attempt(1.5),
			self._make_attempt(1.6),
			self._make_attempt(1.7),
		]
		assert state.should_escalate is False


class TestKernelHash:
	def test_produces_16_char_hex(self) -> None:
		h = LoopState.kernel_hash("void kernel() {}")
		assert len(h) == 16
		assert all(c in "0123456789abcdef" for c in h)

	def test_deterministic(self) -> None:
		h1 = LoopState.kernel_hash("void kernel() {}")
		h2 = LoopState.kernel_hash("void kernel() {}")
		assert h1 == h2

	def test_different_source_different_hash(self) -> None:
		h1 = LoopState.kernel_hash("void kernel_a() {}")
		h2 = LoopState.kernel_hash("void kernel_b() {}")
		assert h1 != h2
