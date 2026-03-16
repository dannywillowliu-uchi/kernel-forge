# Compute Optimization Strategies

### loop_unrolling
- **Category:** compute_opt
- **Applicability:** inner loops with small fixed trip counts. GEMM accumulation, reduction trees, stencils.
- **Expected Impact:** 10-30% instruction throughput improvement. Key enabler for ILP.
- **Description:** #pragma unroll replicates loop body N times, eliminating counter/branch instructions, enabling compile-time index resolution (keeping data in registers vs local memory), and exposing independent instructions for ILP. #pragma unroll (no arg) = compiler decides. #pragma unroll 1 = disable (reduces register pressure). Over-unrolling increases register pressure and may reduce occupancy. Use template parameters for compile-time known trip counts.

### warp_level_primitives
- **Category:** compute_opt
- **Applicability:** warp-level reductions, prefix scans, broadcast, warp-aggregated atomics
- **Expected Impact:** 2-5x faster than equivalent shared memory reduction. 32x less atomic contention.
- **Description:** Register-level data exchange within a warp without SMEM. __shfl_down_sync for tree reduction (5 rounds for 32 threads). __shfl_sync for arbitrary lane-to-lane movement. Always use _sync variants (CUDA 9+). Compute mask BEFORE divergent branch: __ballot_sync(0xFFFFFFFF, condition). CUB's WarpReduce/BlockReduce wrap these correctly for production.

### register_blocking
- **Category:** compute_opt
- **Applicability:** GEMM, batched GEMM, convolution. Core technique for 40%->90% cuBLAS efficiency.
- **Expected Impact:** 2-4x. Benchmark: 1D tiling 8,475 GFLOPs -> 2D tiling (TM=TN=8) 15,972 GFLOPs on A100.
- **Description:** Each thread computes TM x TN output tile, keeping accumulators in registers across inner loop. Amortizes SMEM load cost across TM*TN results. FLOPs/(SMEM loads) = 2*TM*TN/(TM+TN); for TM=TN=8: ratio = 8. Load regA and regB once per k-step, perform TM*TN FMAs. TM/TN are template parameters for compile-time unrolling. On Blackwell: tile alignment to 128x128 for tcgen05.

### instruction_level_parallelism
- **Category:** compute_opt
- **Applicability:** all compute-bound kernels. Hides arithmetic latency (4-20 cycles for FP32 FMA).
- **Expected Impact:** 1.5-3x for properly ILP-structured inner loops.
- **Description:** Issue multiple independent instructions per thread so GPU scheduler can overlap execution latencies. Pattern: compute 2-4 independent accumulations (acc0 += a0*b0; acc1 += a1*b1; etc). GEMM register blocking naturally creates ILP via TM*TN independent accumulators. Avoid read-after-write dependencies within 4-6 instructions. Too many accumulators -> register pressure -> occupancy drop.

### cooperative_groups
- **Category:** compute_opt
- **Applicability:** warp/block/grid reductions, fine-grained producer-consumer, multi-block algorithms
- **Expected Impact:** Primarily correctness/portability. Grid-level sync enables algorithms that previously required multiple launches.
- **Description:** Portable API for thread synchronization at any granularity (warp, tile, block, grid). Replaces implicit warp-synchronous assumptions. cg::tiled_partition<32>(block) for warp-sized tile. cg::this_grid() for grid-level sync (requires cudaLaunchCooperativeKernel). On Blackwell: cg::this_cluster() for cluster-scope via distributed SMEM.

### thread_coarsening
- **Category:** compute_opt
- **Applicability:** kernels where per-thread overhead is comparable to computation. Small reductions, elementwise.
- **Expected Impact:** 1.5-2x for register-cache workloads. Can hurt if occupancy drops too much.
- **Description:** Each thread processes multiple output elements via grid-stride loop: for (int i = tid; i < N; i += stride). Reduces total threads, amortizes per-thread overhead. Coarsening factor must be tuned. Often combined with register blocking (coarsening = outer, blocking = inner).
