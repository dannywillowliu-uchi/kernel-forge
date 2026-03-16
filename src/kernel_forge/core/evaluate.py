"""Scoring, termination logic, and failure classification."""

from __future__ import annotations

from kernel_forge.core.types import Attempt, FailureType, TerminationConfig


def compute_score(speedup: float, correct: bool) -> float:
	"""Compute the optimization score for an attempt.

	Returns speedup if correct, 0.0 otherwise (matching KernelBench scoring).
	"""
	return speedup if correct else 0.0


def should_escalate_profiling(
	recent_attempts: list[Attempt],
	config: TerminationConfig,
) -> bool:
	"""Determine if profiling should be escalated from cuda_events to ncu.

	Returns True if the last N correct attempts at cuda_events tier all have
	less than the plateau_threshold relative improvement.
	"""
	window = config.plateau_window_cuda_events

	# Filter to correct attempts at cuda_events tier
	correct_ce = [
		a for a in recent_attempts
		if a.correct and a.profiling_tier == "cuda_events"
	]

	if len(correct_ce) < window:
		return False

	last_n = correct_ce[-window:]
	for i in range(1, len(last_n)):
		improvement = (last_n[i].speedup - last_n[i - 1].speedup) / max(last_n[i - 1].speedup, 1e-9)
		if improvement >= config.plateau_threshold:
			return False

	return True


def should_terminate(
	attempt_count: int,
	total_cost: float,
	elapsed_seconds: float,
	consecutive_failures: int,
	config: TerminationConfig,
) -> bool:
	"""Determine if the optimization loop should terminate.

	Returns True if any termination trigger is hit:
	- Max attempts exceeded
	- Cost budget exceeded
	- Wall time exceeded
	- Max consecutive failures exceeded
	"""
	if attempt_count >= config.max_attempts:
		return True
	if total_cost >= config.max_cost_usd:
		return True
	if elapsed_seconds >= config.max_wall_time_seconds:
		return True
	if consecutive_failures >= config.max_consecutive_failures:
		return True
	return False


def classify_failure(exit_code: int, stderr: str, stdout: str) -> FailureType:
	"""Classify a kernel failure based on exit code and error output.

	Uses pattern matching on common error signals to determine failure type.
	"""
	combined = (stderr + " " + stdout).lower()

	# Timeout
	if exit_code == -1 or "timed out" in combined or "timeout" in combined:
		return FailureType.TIMEOUT

	# Compilation errors
	if "error:" in combined and ("nvcc" in combined or "compile" in combined):
		return FailureType.COMPILATION_ERROR
	if "error:" in combined and ".cu" in combined:
		return FailureType.COMPILATION_ERROR

	# Link errors
	if "undefined symbol" in combined or "undefined reference" in combined:
		return FailureType.LINK_ERROR
	if "cannot find -l" in combined:
		return FailureType.LINK_ERROR

	# Runtime OOM
	if "out of memory" in combined or "cuda_error_out_of_memory" in combined:
		return FailureType.RUNTIME_OOM
	if "oom" in combined.split():
		return FailureType.RUNTIME_OOM

	# Runtime segfault
	if "segfault" in combined or "sigsegv" in combined or "segmentation fault" in combined:
		return FailureType.RUNTIME_SEGFAULT
	if "illegal memory access" in combined or "illegal address" in combined:
		return FailureType.RUNTIME_SEGFAULT

	# Correctness failures
	if "allclose" in combined or "not close" in combined:
		return FailureType.CORRECTNESS_FAILURE
	if "correctness" in combined and "fail" in combined:
		return FailureType.CORRECTNESS_FAILURE

	# Numerical instability
	if "nan" in combined.split() or "inf" in combined.split():
		return FailureType.NUMERICAL_INSTABILITY
	if "numerical" in combined and "instab" in combined:
		return FailureType.NUMERICAL_INSTABILITY

	# Performance regression (correct but slow)
	if "regression" in combined or "slower" in combined:
		return FailureType.PERFORMANCE_REGRESSION

	# Default to compilation error for non-zero exit
	if exit_code != 0:
		return FailureType.COMPILATION_ERROR

	return FailureType.COMPILATION_ERROR
