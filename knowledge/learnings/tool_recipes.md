# Tool Recipes

## [2026-03-15 01:00] cuda_events_timing
Standard CUDA events benchmarking pattern: 500 warmup iterations, 100 timed reps using CUDA events (start.record(), end.record(), torch.cuda.synchronize()). Use CyclingCallable or similar to evict L2 cache between reps for representative cold-cache timing. Expected variance: +/- 5-10%.

## [2026-03-15 01:00] ncu_key_metrics
Key Nsight Compute metrics for bottleneck diagnosis:
- sm__throughput.avg.pct_of_peak_sustained_elapsed (compute utilization)
- dram__throughput.avg.pct_of_peak_sustained_elapsed (memory throughput)
- sm__warps_active.avg.pct_of_peak_sustained_elapsed (occupancy)
- l2__throughput.avg.pct_of_peak_sustained_elapsed (L2 cache pressure)
Use: ncu --metrics <above> --target-processes all python kernel.py

## [2026-03-15 01:00] kernelbench_harness
KernelBench problem format: Model class (reference), ModelNew class (optimized), get_inputs() for test data, get_init_inputs() for constructor args. Solutions use torch.utils.cpp_extension.load_inline() to embed CUDA/HIP kernels inline. Correctness via torch.allclose(rtol=1e-3, atol=1e-3).
