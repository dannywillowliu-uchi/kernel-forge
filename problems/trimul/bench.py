#!/usr/bin/env python3
"""Self-contained test + benchmark for TriMul submissions.

Usage:
	python bench.py                    # correctness tests only
	python bench.py --benchmark        # correctness + benchmark (geomean)
	python bench.py --benchmark-only   # skip correctness, just time
	python bench.py --profile          # torch profiler on largest case
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import torch

from reference import generate_input, check_implementation, ref_kernel
from submission import custom_kernel

# -- Competition test specs ------------------------------------------------
TESTS = [
	{"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 9371, "nomask": True, "distribution": "normal"},
	{"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 1092, "nomask": False, "distribution": "normal"},
	{"seqlen": 64, "bs": 2, "dim": 256, "hiddendim": 128, "seed": 2291, "nomask": True, "distribution": "normal"},
	{"seqlen": 64, "bs": 2, "dim": 256, "hiddendim": 128, "seed": 210284, "nomask": False, "distribution": "normal"},
	{"seqlen": 128, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 81934, "nomask": True, "distribution": "normal"},
	{"seqlen": 256, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 1932, "nomask": True, "distribution": "normal"},
	{"seqlen": 256, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 10432, "nomask": False, "distribution": "normal"},
	{"seqlen": 768, "bs": 2, "dim": 128, "hiddendim": 128, "seed": 731, "nomask": True, "distribution": "normal"},
	{"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 53121, "nomask": False, "distribution": "normal"},
	{"seqlen": 1024, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 31, "nomask": True, "distribution": "normal"},
	{"seqlen": 1024, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 4921, "nomask": False, "distribution": "normal"},
	# Cauchy distribution tests
	{"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 937321, "nomask": True, "distribution": "cauchy"},
	{"seqlen": 64, "bs": 2, "dim": 256, "hiddendim": 128, "seed": 2291, "nomask": True, "distribution": "cauchy"},
	{"seqlen": 128, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 8134, "nomask": True, "distribution": "cauchy"},
	{"seqlen": 256, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 932, "nomask": True, "distribution": "cauchy"},
	{"seqlen": 768, "bs": 2, "dim": 128, "hiddendim": 128, "seed": 31, "nomask": True, "distribution": "cauchy"},
	{"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 5321, "nomask": False, "distribution": "cauchy"},
	{"seqlen": 1024, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 491, "nomask": False, "distribution": "cauchy"},
]

# Competition benchmark specs (scored by geomean of these)
BENCHMARKS = [
	{"seqlen": 256, "bs": 2, "dim": 128, "hiddendim": 128, "seed": 9371, "nomask": True, "distribution": "normal"},
	{"seqlen": 768, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 381, "nomask": True, "distribution": "cauchy"},
	{"seqlen": 256, "bs": 2, "dim": 384, "hiddendim": 128, "seed": 2301, "nomask": False, "distribution": "normal"},
	{"seqlen": 512, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 12819, "nomask": True, "distribution": "normal"},
	{"seqlen": 1024, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 381, "nomask": True, "distribution": "cauchy"},
	{"seqlen": 768, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 481, "nomask": False, "distribution": "normal"},
	{"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 23291, "nomask": True, "distribution": "normal"},
]


def clone_data(data):
	"""Recursively clone all tensors in data."""
	if isinstance(data, tuple):
		return tuple(clone_data(x) for x in data)
	elif isinstance(data, list):
		return [clone_data(x) for x in data]
	elif isinstance(data, dict):
		return {k: clone_data(v) for k, v in data.items()}
	elif isinstance(data, torch.Tensor):
		return data.clone()
	else:
		return data


def run_correctness_tests():
	"""Run all correctness tests. Returns True if all pass."""
	print("=" * 70)
	print("CORRECTNESS TESTS")
	print("=" * 70)
	all_pass = True
	for i, spec in enumerate(TESTS):
		label = f"seq={spec['seqlen']:>4d} bs={spec['bs']} dim={spec['dim']:>3d} mask={'no' if spec['nomask'] else 'yes':>3s} dist={spec['distribution']:>7s}"
		try:
			data = generate_input(**spec)
			torch.cuda.synchronize()
			output = custom_kernel(clone_data(data))
			torch.cuda.synchronize()
			good, msg = check_implementation(data, output)
			if good:
				print(f"  [{i+1:2d}/{len(TESTS)}] PASS  {label}")
			else:
				print(f"  [{i+1:2d}/{len(TESTS)}] FAIL  {label}")
				print(f"         {msg}")
				all_pass = False
		except Exception as e:
			print(f"  [{i+1:2d}/{len(TESTS)}] ERROR {label}")
			print(f"         {e}")
			all_pass = False
		finally:
			torch.cuda.empty_cache()

	if all_pass:
		print(f"\nAll {len(TESTS)} tests PASSED")
	else:
		print(f"\nSome tests FAILED")
	return all_pass


def time_one(spec, warmup=3, repeats=20):
	"""Time a single benchmark spec. Returns mean time in ms."""
	data = generate_input(**spec)
	torch.cuda.synchronize()

	# Warmup
	for _ in range(warmup):
		_ = custom_kernel(clone_data(data))
		torch.cuda.synchronize()

	# Timed runs
	times_ms = []
	for _ in range(repeats):
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)
		start.record()
		_ = custom_kernel(clone_data(data))
		end.record()
		torch.cuda.synchronize()
		times_ms.append(start.elapsed_time(end))

	torch.cuda.empty_cache()
	return sorted(times_ms)[len(times_ms) // 4]  # Q1 (robust to outliers)


def time_reference(spec, warmup=3, repeats=10):
	"""Time the reference implementation for speedup comparison."""
	data = generate_input(**spec)
	torch.cuda.synchronize()

	for _ in range(warmup):
		_ = ref_kernel(clone_data(data))
		torch.cuda.synchronize()

	times_ms = []
	for _ in range(repeats):
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)
		start.record()
		_ = ref_kernel(clone_data(data))
		end.record()
		torch.cuda.synchronize()
		times_ms.append(start.elapsed_time(end))

	torch.cuda.empty_cache()
	return sorted(times_ms)[len(times_ms) // 4]


def run_benchmarks(include_reference=True):
	"""Run benchmark suite. Returns geomean in ms."""
	print("\n" + "=" * 70)
	print("BENCHMARKS")
	print("=" * 70)

	sub_times = []
	ref_times = []

	for i, spec in enumerate(BENCHMARKS):
		label = f"seq={spec['seqlen']:>4d} bs={spec['bs']} dim={spec['dim']:>3d}"
		t_sub = time_one(spec)
		sub_times.append(t_sub)

		if include_reference:
			t_ref = time_reference(spec)
			ref_times.append(t_ref)
			speedup = t_ref / t_sub if t_sub > 0 else 0
			print(f"  [{i+1}/{len(BENCHMARKS)}] {label}  sub={t_sub:8.3f}ms  ref={t_ref:8.3f}ms  speedup={speedup:.2f}x")
		else:
			print(f"  [{i+1}/{len(BENCHMARKS)}] {label}  sub={t_sub:8.3f}ms")

	# Geomean
	log_sum = sum(math.log(t) for t in sub_times)
	geomean = math.exp(log_sum / len(sub_times))

	print(f"\n{'=' * 70}")
	print(f"GEOMEAN: {geomean:.3f} ms ({geomean * 1000:.1f} us)")

	if include_reference and ref_times:
		ref_log_sum = sum(math.log(t) for t in ref_times)
		ref_geomean = math.exp(ref_log_sum / len(ref_times))
		overall_speedup = ref_geomean / geomean
		print(f"REF GEOMEAN: {ref_geomean:.3f} ms ({ref_geomean * 1000:.1f} us)")
		print(f"OVERALL SPEEDUP: {overall_speedup:.2f}x")

	print(f"{'=' * 70}")

	# Auto-write checkpoint
	per_results = [
		{"config": f"seq={spec['seqlen']} bs={spec['bs']} dim={spec['dim']}", "time_ms": round(sub_times[i], 4)}
		for i, spec in enumerate(BENCHMARKS)
	]
	_write_checkpoint(geomean, per_results)

	return geomean


def _write_checkpoint(geomean_ms, per_benchmark_results):
	"""Append checkpoint to checkpoint.jsonl in the current directory."""
	checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint.jsonl")

	iteration = 0
	best_geomean = geomean_ms
	if os.path.exists(checkpoint_path):
		with open(checkpoint_path) as f:
			lines = f.readlines()
			if lines:
				last = json.loads(lines[-1])
				iteration = last.get("iteration", 0)
				best_geomean = min(last.get("best_geomean_ms", geomean_ms), geomean_ms)

	iteration += 1
	best_geomean = min(best_geomean, geomean_ms)

	checkpoint = {
		"iteration": iteration,
		"geomean_ms": round(geomean_ms, 4),
		"best_geomean_ms": round(best_geomean, 4),
		"per_benchmark": per_benchmark_results,
		"timestamp": datetime.now(timezone.utc).isoformat(),
	}

	with open(checkpoint_path, "a") as f:
		f.write(json.dumps(checkpoint) + "\n")


def _write_profile(prof_table, spec):
	"""Write structured profile data to profile_latest.json."""
	base_dir = os.path.dirname(os.path.abspath(__file__))
	profile_path = os.path.join(base_dir, "profile_latest.json")
	checkpoint_path = os.path.join(base_dir, "checkpoint.jsonl")

	iteration = 0
	if os.path.exists(checkpoint_path):
		with open(checkpoint_path) as f:
			lines = f.readlines()
			if lines:
				iteration = json.loads(lines[-1]).get("iteration", 0)

	profile = {
		"iteration": iteration,
		"source": "torch_profiler",
		"config": f"seq={spec['seqlen']} bs={spec['bs']} dim={spec['dim']}",
		"raw_output": prof_table,
		"timestamp": datetime.now(timezone.utc).isoformat(),
	}

	with open(profile_path, "w") as f:
		json.dump(profile, f, indent=2)


def run_profile():
	"""Profile the largest benchmark case."""
	from torch.profiler import profile, ProfilerActivity

	# Use the largest benchmark spec
	spec = BENCHMARKS[-1]  # 1024, dim=384
	print(f"\nProfiling: seq={spec['seqlen']} bs={spec['bs']} dim={spec['dim']}")

	data = generate_input(**spec)
	torch.cuda.synchronize()

	# Warmup
	for _ in range(3):
		_ = custom_kernel(clone_data(data))
		torch.cuda.synchronize()

	with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
		_ = custom_kernel(clone_data(data))
		torch.cuda.synchronize()

	table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
	print(table)
	_write_profile(table, spec)


def main():
	parser = argparse.ArgumentParser(description="TriMul benchmark harness")
	parser.add_argument("--benchmark", action="store_true", help="Run benchmarks after correctness tests")
	parser.add_argument("--benchmark-only", action="store_true", help="Skip correctness, just benchmark")
	parser.add_argument("--profile", action="store_true", help="Run torch profiler on largest case")
	parser.add_argument("--no-ref", action="store_true", help="Skip reference timing (faster)")
	args = parser.parse_args()

	if args.profile:
		run_profile()
		return

	if not args.benchmark_only:
		passed = run_correctness_tests()
		if not passed:
			print("\nAborting: fix correctness first.")
			sys.exit(1)

	if args.benchmark or args.benchmark_only:
		run_benchmarks(include_reference=not args.no_ref)


if __name__ == "__main__":
	main()
