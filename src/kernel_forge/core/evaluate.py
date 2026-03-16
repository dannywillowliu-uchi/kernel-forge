"""Scoring, termination logic, failure classification, and roofline analysis."""

from __future__ import annotations

from kernel_forge.core.types import (
	B200_PEAKS,
	Attempt,
	FailureType,
	HardwarePeaks,
	RooflineAnalysis,
	TerminationConfig,
)


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


def compute_roofline(
	runtime_ms: float,
	flops: float,
	bytes_moved: float,
	precision: str = "tf32",
	peaks: HardwarePeaks = B200_PEAKS,
	headroom_threshold: float = 10.0,
) -> RooflineAnalysis:
	"""Compute roofline analysis: how far from peak and whether to keep optimizing.

	Args:
		runtime_ms: Kernel runtime in milliseconds.
		flops: Total floating-point operations in the kernel.
		bytes_moved: Total bytes moved to/from HBM (for arithmetic intensity).
		precision: Precision being used ("fp32", "tf32", "bf16", "fp8", "fp4").
		peaks: Hardware peak performance specs.
		headroom_threshold: Minimum headroom % to consider worth optimizing further.
	"""
	achieved_tflops = flops / (runtime_ms / 1000) / 1e12

	peak_map = {
		"fp32": peaks.fp32_tflops,
		"tf32": peaks.tf32_tflops,
		"bf16": peaks.bf16_tflops,
		"fp8": peaks.fp8_tflops,
		"fp4": peaks.fp4_tflops,
	}
	peak_tflops = peak_map.get(precision, peaks.tf32_tflops)

	utilization = (achieved_tflops / peak_tflops * 100) if peak_tflops > 0 else 0
	headroom = 100.0 - utilization

	# Arithmetic intensity = FLOPs / bytes
	arith_intensity = flops / bytes_moved if bytes_moved > 0 else float("inf")

	# Roofline ridge point: peak_compute / peak_bandwidth
	peak_compute_flops_s = peak_tflops * 1e12
	peak_bw_bytes_s = peaks.hbm_bandwidth_tb_s * 1e12
	ridge_point = peak_compute_flops_s / peak_bw_bytes_s if peak_bw_bytes_s > 0 else 0

	if arith_intensity < ridge_point * 0.8:
		roofline_bound = "memory_bound"
	elif arith_intensity > ridge_point * 1.2:
		roofline_bound = "compute_bound"
	else:
		roofline_bound = "balanced"

	# Decision: is it worth pushing further?
	ai_str = f"AI={arith_intensity:.1f}"
	ridge_str = f"ridge={ridge_point:.1f}"

	worth_it = headroom > headroom_threshold
	if worth_it:
		if roofline_bound == "memory_bound":
			explanation = (
				f"{headroom:.1f}% headroom, memory-bound "
				f"({ai_str} < {ridge_str}). "
				f"Optimize memory: tiling, coalescing, "
				f"vectorized loads, fusion."
			)
		elif roofline_bound == "compute_bound":
			explanation = (
				f"{headroom:.1f}% headroom, compute-bound "
				f"({ai_str} > {ridge_str}). "
				f"Optimize compute: tensor cores, warp "
				f"primitives, register blocking."
			)
		else:
			explanation = (
				f"{headroom:.1f}% headroom, balanced "
				f"({ai_str} ~ {ridge_str}). "
				f"Both memory and compute may help. "
				f"Profile with ncu for bottleneck."
			)
	else:
		prec = precision.upper()
		explanation = (
			f"Only {headroom:.1f}% headroom at "
			f"{utilization:.1f}% of {prec} peak. "
			f"Near-optimal. Consider higher precision "
			f"tier or accept current result."
		)

	return RooflineAnalysis(
		achieved_tflops=round(achieved_tflops, 2),
		runtime_ms=round(runtime_ms, 4),
		peak_tflops=peak_tflops,
		precision=precision,
		utilization_pct=round(utilization, 1),
		headroom_pct=round(headroom, 1),
		arithmetic_intensity=round(arith_intensity, 1),
		roofline_bound=roofline_bound,
		worth_optimizing=worth_it,
		explanation=explanation,
	)
