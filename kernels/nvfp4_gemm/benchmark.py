"""Benchmark and verify NVFP4 GEMM submission."""
import torch
import sys
import importlib

def main():
	# Import submission
	mod_name = sys.argv[1] if len(sys.argv) > 1 else "submission"
	mod = importlib.import_module(mod_name)
	custom_kernel = mod.custom_kernel

	from reference import generate_input, check_implementation

	# Verify correctness
	print("=== Correctness Check ===")
	for m, n, k in [(128, 256, 256), (128, 7168, 16384), (128, 4096, 7168), (128, 7168, 2048)]:
		data = generate_input(m=m, n=n, k=k, l=1, seed=1111)
		output = custom_kernel(data)
		ok, msg = check_implementation(data, output)
		status = "PASS" if ok else "FAIL"
		print(f"  {m}x{n}x{k}: {status} {msg}")
		if not ok:
			print(f"    ERROR: {msg}")

	# Benchmark
	print("\n=== Benchmark ===")
	targets = {
		(128, 7168, 16384): 8.994,
		(128, 4096, 7168): 2.354,
		(128, 7168, 2048): 1.333,
	}

	times = []
	for (m, n, k), target in targets.items():
		data = generate_input(m=m, n=n, k=k, l=1, seed=1111)

		# Warmup
		for _ in range(20):
			custom_kernel(data)
		torch.cuda.synchronize()

		# Benchmark
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)
		start.record()
		for _ in range(100):
			custom_kernel(data)
		end.record()
		torch.cuda.synchronize()
		us = start.elapsed_time(end) / 100 * 1000

		ratio = us / target
		times.append(us)
		print(f"  {m}x{n}x{k}: {us:.3f} us (target: {target} us, ratio: {ratio:.2f}x)")

	# Geomean
	geomean = (times[0] * times[1] * times[2]) ** (1/3)
	target_geomean = (8.994 * 2.354 * 1.333) ** (1/3)
	print(f"\n  GEOMEAN: {geomean:.3f} us (target: {target_geomean:.3f} us)")
	print(f"  VS_SPEED_OF_LIGHT: {geomean / target_geomean:.2f}x")

	# Reference benchmark for comparison
	print("\n=== Reference Benchmark ===")
	from reference import ref_kernel
	ref_times = []
	for (m, n, k), target in targets.items():
		data = generate_input(m=m, n=n, k=k, l=1, seed=1111)
		for _ in range(5):
			ref_kernel(data)
		torch.cuda.synchronize()
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)
		start.record()
		for _ in range(50):
			data2 = generate_input(m=m, n=n, k=k, l=1, seed=1111)
			ref_kernel(data2)
		end.record()
		torch.cuda.synchronize()
		us = start.elapsed_time(end) / 50 * 1000
		ref_times.append(us)
		print(f"  {m}x{n}x{k}: {us:.3f} us")
	ref_geomean = (ref_times[0] * ref_times[1] * ref_times[2]) ** (1/3)
	print(f"  REF GEOMEAN: {ref_geomean:.3f} us")
	print(f"  SPEEDUP vs ref: {ref_geomean / geomean:.2f}x")


if __name__ == "__main__":
	main()
