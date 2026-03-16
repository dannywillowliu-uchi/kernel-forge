# Insights

## [2026-03-15 01:00] optimization_hierarchy
Optimizations should generally be applied in order of measured impact, not by implementation complexity. From the B200 inference takehome, the hierarchy was: algorithmic changes (KV cache: 2.4x) > precision (explicit bf16: 1.5x) > compiler (torch.compile: 1.8x) > memory management (static alloc: consistency) > pipeline (async tokens: 1.15x) > execution (CUDA graphs: 1.14x). But this is a starting heuristic, not a rigid order -- profiling after each change should drive the next decision.

## [2026-03-15 01:00] complementary_techniques
torch.compile fuses kernels (reducing GPU work) and CUDA graphs eliminate CPU launch overhead (reducing CPU work). They are complementary, not alternatives. Compiled scatter+mask + CUDA graph gave +39% over compile alone on B200 decode.

## [2026-03-15 01:00] sa_peephole_alternation
From the FP4 multiplier optimization: Simulated Annealing gets stuck at local minima when escape requires coordinated 4+ gate changes. SAT-based peephole optimization is exact within a small window but can't restructure globally. Alternating between them resolves each other's weaknesses -- SA provides global restructuring, peephole provides exact local optimization. This pattern (alternate between complementary search strategies) applies broadly to kernel optimization.

## [2026-03-15 01:00] batch_size_gpu_utilization
B200 has massive compute capacity. At batch=1, a ~350M parameter model heavily underutilizes the GPU -- tensor cores are starved for work. Batching fills the GPU with useful work, giving near-linear scaling (16.6x at batch=16). Always profile at the target batch size, not just batch=1.
