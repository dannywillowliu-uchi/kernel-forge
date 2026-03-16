# Memory Optimization Strategies

### shared_memory_tiling
- **Category:** memory_opt
- **Applicability:** memory_bound matrix operations, convolutions, stencils, any kernel with high data reuse
- **Expected Impact:** 3-10x for arithmetic-heavy kernels. Benchmark: naive GEMM 309 GFLOPs -> tiled 2,980 GFLOPs on A100.
- **Description:** Divide computation into tiles sized for SMEM. Load tiles from GMEM once, compute from SMEM. Reduces global memory transactions by tile factor. On B200: up to 228 KB SMEM. Tile BM=BN=128, BK=16 typical. Pad arrays +1 to avoid bank conflicts. Must use __syncthreads() between load/compute. Tile size must be multiple of 128 for M/N on Blackwell for full tensor core throughput.

### global_memory_coalescing
- **Category:** memory_opt
- **Applicability:** all kernels with global memory access. Most important optimization on any GPU.
- **Expected Impact:** 5-50x effective bandwidth improvement. Non-coalesced = 1/32 efficiency.
- **Description:** Arrange thread-to-memory access so consecutive thread IDs access consecutive addresses within each warp. Hardware combines 32 individual accesses into 1-4 cache-line transactions (128 bytes). Thread index mapping must ensure threadIdx.x maps to fastest-varying dimension. Stride > 1 degrades proportionally (stride=2 = ~50% efficiency). Check "Global Memory Load Efficiency" in ncu.

### bank_conflict_avoidance
- **Category:** memory_opt
- **Applicability:** any kernel using shared memory with non-unit stride access patterns
- **Expected Impact:** 2-16x SMEM throughput improvement when eliminating high-degree conflicts.
- **Description:** SMEM has 32 banks (4-byte width). Simultaneous access to same bank from multiple warp threads serializes. Fix: pad arrays (float A[M][K+1]) to shift bank alignment. Bank index = (byte_address / 4) % 32. Broadcast (all threads same address) is free. Use ncu "Shared Memory Bank Conflicts" metric.

### l2_cache_persistence
- **Category:** memory_opt
- **Applicability:** kernels repeatedly reading same weights/embeddings across kernel calls. Ampere+ only.
- **Expected Impact:** 10-30% latency reduction for pinned data. B200 L2 = 126 MB.
- **Description:** Designate global memory region as "persisting" to occupy reserved L2 portion across kernel launches. API: cudaStreamAttrValue with accessPolicyWindow. Set hitRatio < 1.0 to prevent thrashing when data exceeds set-aside. Reset with cudaAccessPropertyNormal before different working sets.

### vectorized_loads
- **Category:** memory_opt
- **Applicability:** bulk data reads/writes, elementwise ops, GEMM tile load phase
- **Expected Impact:** 1.3-2x throughput for load-dominated kernels. +44% in GEMM load phase.
- **Description:** Replace scalar 32-bit loads with 128-bit float4/int4/uint4. Halves instruction count, reduces address overhead, enables wider transactions. Data must be 16-byte aligned. Dimension must be divisible by 4. Reinterpret pointer: float4 tmp = reinterpret_cast<float4*>(&A[idx])[0]. For BF16/FP16: float4 loads 8 elements. For FP8: int4 loads 16 values.

### async_memcpy_pipelining
- **Category:** memory_opt
- **Applicability:** any kernel with distinct load and compute phases. Essential for peak tensor core utilization.
- **Expected Impact:** 1.5-3x by hiding memory latency. Difference between 40% and 80%+ tensor core utilization.
- **Description:** Use cuda::memcpy_async (Ampere) or TMA (Hopper/Blackwell) for non-blocking GMEM->SMEM copies. Overlap copy latency for stage N+1 with computation on stage N. Multi-stage pattern: pre-fill S stages, then main loop consumes stage[i%S] while issuing copy for stage[(i+S)%S]. Double-buffering minimum; 4-5 stages often needed on B200 (420-cycle TMEM miss).

### tma_tensor_memory_accelerator
- **Category:** memory_opt
- **Applicability:** Hopper/Blackwell kernels staging tiles through SMEM. Mandatory for peak performance.
- **Expected Impact:** ~50% instruction overhead reduction for load phase. Enables warp specialization.
- **Description:** Hardware unit for bulk async multi-dimensional tile copies. Single thread issues entire tile copy via descriptor (CUTensorMap). Supports 1D-5D, auto-predication at boundaries. On Blackwell: TMA multicast feeds all CTAs in cluster (up to 16) from one operation. Synchronization via mbarrier. Replaces cp.async + manual address calculation.

### texture_memory
- **Category:** memory_opt
- **Applicability:** 2D spatially-local access (image processing, fluid sim). Less useful for 1D streaming.
- **Expected Impact:** 1.2-2x for 2D spatially-local workloads. Near-zero benefit for sequential 1D on Ampere+.
- **Description:** Dedicated read-only cache with 2D spatial locality optimization, hardware filtering, automatic boundary handling. Use cudaTextureObject_t (modern API). Separate from L1/L2 cache. Data must be read-only during kernel. For writes use cudaSurfaceObject_t.
