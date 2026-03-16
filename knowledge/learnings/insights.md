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

## [2026-03-16 07:24:43 UTC] (ref: 19_ReLU)

Problem 19_ReLU: achieved 1.01x speedup in 3 attempts. Best strategy landed on after trying 3 approaches.

## [2026-03-16 08:43:08 UTC] (ref: 1_Square_matrix_multiplication_)

Problem 1_Square_matrix_multiplication_: 12.62x speedup via strategy 'tf32_tensor_cores'. Tried 3 approaches total. Kernel type: 1.

## [2026-03-16 08:49:33 UTC] (ref: 19_ReLU)

Problem 19_ReLU: 1.01x speedup via strategy 'tf32_tensor_cores'. Tried 3 approaches total. Kernel type: 19.

## [2026-03-16 08:54:01 UTC] (ref: 23_Softmax)

Problem 23_Softmax: 1.00x speedup via strategy 'tf32_tensor_cores'. Tried 3 approaches total. Kernel type: 23.

## [2026-03-16 08:59:23 UTC] (ref: 26_GELU_)

Problem 26_GELU_: 1.05x speedup via strategy 'tf32_tensor_cores'. Tried 3 approaches total. Kernel type: 26.

## [2026-03-16 14:32:42 UTC] (ref: 100_HingeLoss)

Problem 100_HingeLoss: 2.46x speedup via strategy 'tf32_tensor_cores'. Tried 3 approaches total. Kernel type: 100.

## [2026-03-16 19:00] elementwise_bandwidth_ceiling
KernelBench L1 elementwise ops (ReLU, Sigmoid, Tanh, GELU) on B200 with 4096x393216 FP32 tensors: both PyTorch native and custom Triton kernels achieve ~3.2 TB/s effective HBM bandwidth. This is the bandwidth ceiling. Custom kernels cannot exceed it. The ONLY win on isolated memory-bound ops is kernel FUSION (Swish: 2.50x by replacing 2-kernel x*sigmoid(x) with 1-kernel F.silu).

## [2026-03-16 19:00] b200_clock_variance
B200 GPU shows significant clock state variation between process invocations (baselines range 1.86ms to 4.03ms at different times). In-process measurements are stable. Always use --baseline-ms with in-process measurement for accurate speedup comparison.

## [2026-03-16 19:00] transposed_matmul_contiguous
Removing .contiguous() on transposed matmul inputs gives ~5x extra speedup (10.4x -> 15.3x). cuBLAS handles transposes natively via transa/transb flags. The contiguous copy is pure overhead.

## [2026-03-16 19:00] irregular_shape_padding
Padding irregular matrix dimensions to multiples of 128 before matmul boosted speedup from 2.1x to 8.9x on problem 8 (8205x2949x5921). Tensor cores work on fixed tiles; irregular shapes waste computation at boundaries.

## [2026-03-16 21:00] triton_beats_pytorch_elementwise
Triton elementwise kernels achieve ~1.51x over PyTorch native on B200 for large FP32 tensors (4096x393216). PyTorch native reaches ~6ms, Triton reaches ~4ms (actual bandwidth ceiling ~3.2 TB/s). PyTorch's eager kernels are NOT at the HBM bandwidth ceiling on B200 -- this contradicts the earlier batch 4 finding.

## [2026-03-16 21:00] kernel_fusion_dominant_strategy
For memory-bound ops, kernel fusion is the dominant optimization: Softsign 5.24x (3->1 kernel), RMSNorm 5.98x (4->1), GroupNorm 4.00x, InstanceNorm 3.29x, Softplus 2.21x, BatchNorm 2.52x. Look for baselines that launch multiple CUDA kernels for what should be one operation.

## [2026-03-16 21:00] batchnorm_eval_mode_trick
BatchNorm in eval mode can be optimized by precomputing scale=weight*rsqrt(var+eps) and shift=bias-mean*scale, then running a single x*scale+shift kernel. 2.52x speedup.
