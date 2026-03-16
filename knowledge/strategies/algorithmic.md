# Algorithmic Optimization Strategies

### parallel_reduction
- **Category:** algorithmic
- **Applicability:** sum, max, min, any associative reduction. Softmax, layer norm, loss computation.
- **Expected Impact:** Near-peak bandwidth when memory-bound. 5-20x vs naive sequential. CUB DeviceReduce: ~95% theoretical bandwidth.
- **Description:** Tree-structured reduction: N elements to 1 in O(log N) parallel steps. Phase 1: thread coarsening (4-8 elements/thread). Phase 2: warp-level via __shfl_down_sync (5 rounds). Phase 3: warp leaders write SMEM, one warp does final reduction. Phase 4: block leaders atomicAdd to global OR second kernel. Use stride-1 sequential access. Production: use CUB cub::DeviceReduce::Sum or cub::BlockReduce. For per-row: one warp per row.

### parallel_scan_prefix_sum
- **Category:** algorithmic
- **Applicability:** compaction, radix sort, histogram normalization, dynamic GPU memory allocation
- **Expected Impact:** Near-peak bandwidth for memory-bound inputs. CUB DeviceScan: ~85-90% HBM bandwidth.
- **Description:** Running prefix sum in O(N/P * log P). Up-sweep (build tree of partial sums) then down-sweep (propagate prefix sums). Bank-conflict-free addressing via offset: int ai = 2*tid + (2*tid >> 5). For large arrays: multi-block with auxiliary sum array + second scan. Use CUB cub::DeviceScan::ExclusiveSum.

### flash_attention_tiling
- **Category:** algorithmic
- **Applicability:** self-attention, cross-attention, multi-head attention for any sequence length.
- **Expected Impact:** 7-20x HBM reduction vs naive attention. O(N^2)->O(N) memory. FlashAttention-4 on B200: 1,613 TFLOPS (71% utilization).
- **Description:** Tile over sequence length so full QK^T score matrix never materializes in GMEM. Online softmax (Milakov & Gimelshein 2018) enables correct normalization without full matrix. Outer loop over K,V blocks; inner over Q. Maintain running max m_i and normalizer l_i; rescale output when m_new > m_old. FlashAttention-4 on B200: skip rescale when |m_new - m_old| < threshold (~15% non-matmul reduction); FMA-based polynomial exp() approximation instead of hardware EXP unit; TMEM stores output accumulator; 2-CTA backward pass with distributed SMEM.

### tiled_algorithms
- **Category:** algorithmic
- **Applicability:** any algorithm with reuse: GEMM, convolution, correlation, FFT, sparse ops.
- **Expected Impact:** Foundational technique underlying 80-90% of high-performance GPU kernels. Without tiling: virtually nothing reaches >20% peak.
- **Description:** Subdivide problem into tiles sized to memory hierarchy levels. Three-level hierarchy: L1-tile (SMEM/registers), L2-tile (fits L2), HBM-tile (streaming). For GEMM: outermost = L2 tile (BM x BN x BK), middle = register tile (TM x TN), innermost = tensor core shape. On B200: SMEM 228 KB, L2 126 MB. Tile size selection: arithmetic intensity I = FLOPs/bytes must exceed HBM_BW/peak_TFLOPS (roofline model).
