# Pipeline and Execution Optimization Strategies

### cuda_graphs
- **Category:** execution_opt
- **Applicability:** iterative workloads (training loops, inference) with same kernel sequence, fixed shapes. Most impactful when kernels < 1ms.
- **Expected Impact:** 1.2-1.5x for inference with many small kernels. Optimal batching 50-100 nodes per graph.
- **Description:** Capture CUDA operations as static DAG, replay with single CPU call. Eliminates per-kernel CPU launch overhead (~5-20us). Capture via cudaStreamBeginCapture/EndCapture, replay via cudaGraphLaunch. Constraints: no host-device sync inside, no dynamic shapes. PyTorch: torch.cuda.CUDAGraph() or torch.compile(mode="reduce-overhead"). Conditional nodes (CUDA 12.4+) allow scalar param updates without re-capture.

### kernel_fusion
- **Category:** execution_opt
- **Applicability:** sequences of elementwise/pointwise ops (bias+activation+dropout after GEMM), attention computation, any chain where intermediates can live in registers/SMEM.
- **Expected Impact:** 1.5-4x for fused vs unfused chains. FlashAttention: 7-20x memory efficiency over naive.
- **Description:** Combine multiple kernels into one that reads input once, writes output once. Eliminates intermediate GMEM round-trips and launch overhead. Manual: single kernel computing all ops inline. Auto: torch.compile with Triton backend. CUTLASS epilogue visitor tree (EVT) composes complex epilogues as compile-time tree in SMEM->register->GMEM store path. Gotcha: fused kernels have higher register pressure.

### persistent_kernels
- **Category:** execution_opt
- **Applicability:** GEMM, convolution, attention with many output tiles.
- **Expected Impact:** 5-15% for large problems; 30-50% for small problems where launch overhead dominates.
- **Description:** Launch exactly SM-count thread blocks, each processing dynamic work queue of tiles. Blocks run full duration, eliminating per-tile launch overhead. atomicAdd to shared tile counter for next tile. On Blackwell: enables epilogue-TMA/next-tile-MMA overlap. Grid = SM count. CUTLASS 3.x uses persistent kernels by default for large GEMM on Hopper/Blackwell.

### warp_specialization
- **Category:** execution_opt
- **Applicability:** high-performance GEMM and attention on Hopper/Blackwell. Required for >70% tensor core utilization.
- **Expected Impact:** Difference between 40-50% and 80-90%+ tensor core utilization. FlashAttention-4: 71% B200 utilization.
- **Description:** Partition CTA warps into producers (TMA copies, barrier management) and consumers (MMA compute). Producers sleep during memory wait; consumers sleep during data wait. True concurrency between memory and compute within single CTA. Typical: 1 producer warpgroup + 2-4 consumer warpgroups. Ping-pong: consumer 0 does MMA while consumer 1 does epilogue, swap. On Blackwell: additional TMEM barriers needed. FlashAttention-4: second warpgroup handles softmax while first handles MMA.

### stream_overlapping
- **Category:** execution_opt
- **Applicability:** pipelines where data prep and computation can overlap; independent sub-problems.
- **Expected Impact:** Up to 2x when memcpy fully overlaps compute.
- **Description:** Multiple CUDA streams for concurrent execution. Minimum 2 streams. Async copies with pinned host memory (cudaHostAlloc mandatory). 2 copy engines (H2D + D2H) on modern GPUs; use separate streams for each direction. CUDA events for cross-stream dependencies. Profile with Nsight Systems timeline.

### torch_compile_integration
- **Category:** execution_opt
- **Applicability:** PyTorch models before investing in custom CUDA. Best for transformer workloads.
- **Expected Impact:** 1.5-3x vs eager. reduce-overhead adds 1.2-1.5x more.
- **Description:** JIT-compiles Python/ATen to Triton/CUDA kernels via inductor backend. Auto-fuses elementwise ops, reduces launch overhead. Modes: default (reliable fusion), reduce-overhead (CUDA graphs), max-autotune (Triton search). Custom Triton kernels integrate via torch.library.triton_op. Gotchas: dynamic shapes break graphs, 30-120s first-call compile, not as fast as hand-tuned CUDA. Ensure PyTorch >= 2.5 with sm100 support for B200.

### occupancy_tuning
- **Category:** execution_opt
- **Applicability:** all kernels. First check after writing a kernel.
- **Expected Impact:** 2-4x for memory-latency-bound going 25%->100%. Less for compute-bound (occupancy above ~50% often no benefit).
- **Description:** Maximize active warps per SM via three-way tradeoff: threads/block, registers/thread, SMEM/block. B200: 64K registers, 228 KB SMEM, 64 max warps per SM. Use __launch_bounds__(maxThreads, minBlocks) to cap registers. nvcc -Xptxas -v prints register count. The "occupancy paradox": 60-70% often outperforms 100% because more registers per thread reduces spilling. Use cudaOccupancyMaxActiveBlocksPerMultiprocessor API.

### thread_block_clusters
- **Category:** execution_opt
- **Applicability:** GEMM with TMA multicast, all-reduce within GPC. Hopper/Blackwell only.
- **Expected Impact:** Up to cluster_size x reduction in global memory traffic for multicast inputs.
- **Description:** Group CTAs into cluster co-scheduled on adjacent SMs. CTAs access each other's SMEM (distributed shared memory). TMA multicast: one op feeds all cluster CTAs simultaneously. B200: max portable 8, non-portable 16. Cluster dims specified via cudaLaunchKernelEx. All CTAs must be concurrently resident. DSMEM adds ~2-5ns vs local SMEM.
