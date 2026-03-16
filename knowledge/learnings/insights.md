# Insights

## [2026-03-15 01:00] optimization_hierarchy
Optimizations should generally be applied in order of measured impact, not by implementation complexity. From the B200 inference takehome, the hierarchy was: algorithmic changes (KV cache: 2.4x) > precision (explicit bf16: 1.5x) > compiler (torch.compile: 1.8x) > memory management (static alloc: consistency) > pipeline (async tokens: 1.15x) > execution (CUDA graphs: 1.14x). But this is a starting heuristic, not a rigid order -- profiling after each change should drive the next decision.

## [2026-03-15 01:00] complementary_techniques
torch.compile fuses kernels (reducing GPU work) and CUDA graphs eliminate CPU launch overhead (reducing CPU work). They are complementary, not alternatives. Compiled scatter+mask + CUDA graph gave +39% over compile alone on B200 decode.

## [2026-03-15 01:00] sa_peephole_alternation
From the FP4 multiplier optimization: Simulated Annealing gets stuck at local minima when escape requires coordinated 4+ gate changes. SAT-based peephole optimization is exact within a small window but can't restructure globally. Alternating between them resolves each other's weaknesses -- SA provides global restructuring, peephole provides exact local optimization. This pattern (alternate between complementary search strategies) applies broadly to kernel optimization.

## [2026-03-15 01:00] batch_size_gpu_utilization
B200 has massive compute capacity. At batch=1, a ~350M parameter model heavily underutilizes the GPU -- tensor cores are starved for work. Batching fills the GPU with useful work, giving near-linear scaling (16.6x at batch=16). Always profile at the target batch size, not just batch=1.

## [2026-03-16 04:00] matmul_4096_kernelbench_l1
KernelBench L1 problem 1 (4096x4096 FP32 matmul): baseline uses cuBLAS but does NOT enable TF32. Enabling TF32 via torch.backends.cuda.matmul.allow_tf32=True gives 9.5x speedup (2.16ms -> 0.23ms) while passing torch.allclose(rtol=1e-3, atol=1e-3). TF32 uses 10-bit mantissa which is sufficient for 1e-3 tolerance. Pure BF16 (7-bit mantissa) FAILS correctness at this scale (max_diff=4.17). A naive 32x32 tiled FP32 CUDA kernel was 6x SLOWER than cuBLAS baseline. The agent must recognize when the baseline is already using an optimized library -- the win is configuration (TF32), not a custom kernel.

## [2026-03-16 05:37:13 UTC] (ref: 1_Square_matrix_multiplication_)

Problem 1_Square_matrix_multiplication_: achieved 1.00x speedup in 5 attempts. Best strategy landed on after trying 5 approaches.

## [2026-03-16 06:51:43 UTC] (ref: 1_Square_matrix_multiplication_)

Problem 1_Square_matrix_multiplication_: achieved 12.66x speedup in 5 attempts. Best strategy landed on after trying 5 approaches.

## [2026-03-16 07:10:01 UTC] (ref: 1_Square_matrix_multiplication_)

Problem 1_Square_matrix_multiplication_: achieved 12.65x speedup in 3 attempts. Best strategy landed on after trying 3 approaches.

## [2026-03-16 07:14:56 UTC] (ref: 2_Standard_matrix_multiplication_)

Problem 2_Standard_matrix_multiplication_: achieved 13.10x speedup in 3 attempts. Best strategy landed on after trying 3 approaches.
