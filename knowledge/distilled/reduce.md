Looking at these 5 kernels, I can identify clear patterns. Let me analyze them systematically.

**Key observations before writing the guide:**

- Examples 1-4: All use `triton_poi` (pointwise) with manual unrolling for tiny reductions (size=4). No reduction loop at all.
- Example 5: Uses `triton_red` with proper RBLOCK loop for larger reductions (rnumel=484).
- Examples 2/3/4: Use `eviction_policy='evict_last'` for broadcast loads (same element loaded by multiple work items).
- Examples 2/3: Fuse reduction + elementwise ops (normalize, scale, bias) into single kernel.
- Two-pass in Example 3: Mean subtraction → variance normalization (LayerNorm pattern).

---

## REDUCE Optimization Guide

### What makes fast reduce kernels fast

**1. Unroll tiny reductions instead of looping (most impactful)**
When the reduction dimension is ≤32 elements, emit N explicit `tl.load()` calls and add them manually (examples 1, 2, 3, 4 all do this). No RBLOCK loop, no synchronization overhead, no partial accumulation. The compiler can schedule these loads freely. Switch to `triton_red` with an RBLOCK loop only when reduction dim exceeds ~64 elements and a loop becomes cheaper than code bloat.

**2. Kernel fusion eliminates HBM round-trips**
Examples 2 and 3 fuse the reduction with downstream elementwise ops (L2 normalize → affine transform; mean-subtract → variance normalize → scale/bias). Without fusion, each intermediate result writes to HBM and gets re-read. On B200 with ~8 TB/s HBM bandwidth, a pure reduce is latency-bound by kernel launch overhead at small sizes. Fusion amortizes that cost by doing more compute per byte moved.

**3. `eviction_policy='evict_last'` for broadcast loads**
When multiple work items load the same address (e.g., all elements in a row loading the row's sum), mark those loads with `eviction_policy='evict_last'`. This hints L1/L2 to keep the cache line alive. Without it, cache thrashes when XBLOCK > cache line reuse distance. Examples 2, 3, 4 all apply this pattern.

---

### Common code patterns

**Block sizes seen:**
- XBLOCK=64, num_warps=1 for xnumel=64 (example 1): 1 warp handles 64 elements
- XBLOCK=128, num_warps=4 for xnumel=256 (example 4): 4 warps, 32 elements/warp
- XBLOCK=N, RBLOCK=M for 2D tiling (example 5): X axis = output elements, R axis = reduction elements

**Tiling pattern for `triton_red`:**
```python
for roffset in range(0, rnumel, RBLOCK):
    rindex = roffset + rbase
    rmask = rindex < rnumel
    _acc += tl.load(...)  # accumulate
tmp = tl.sum(_acc, 1)[:, None]  # warp-level reduce
```

**Memory access for small reduce (unrolled pattern):**
```python
# Reduction over last dim of size 4 — no loop
tmp0 = tl.load(in_ptr0 + 4 * x0,     xmask, eviction_policy='evict_last')
tmp1 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
tmp2 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
tmp3 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
result = tmp0 + tmp1 + tmp2 + tmp3
```

**Two-pass normalization (LayerNorm/RMSNorm):**
- Pass 1: Compute mean, write `x - mean` to intermediate buffer
- Pass 2: Load that buffer, compute variance via unrolled sum-of-squares, apply `rsqrt`, scale/bias
- On B200 with 126 MB L2: if the intermediate fits, pass 2 reads from L2, not HBM. Size threshold ~31M fp32 elements.

---

### B200/Blackwell considerations

**228KB shared memory:** For `triton_red` kernels, increase RBLOCK aggressively. 228KB ÷ 4 bytes = 57K fp32 elements per SM. Can tile reduction dims up to ~16K before splitting across thread blocks. Prefer RBLOCK=4096+ for large reductions (layer norm over long sequences).

**126MB L2:** Two-pass algorithms become cheaper on B200. The intermediate buffer from pass 1 stays in L2 for pass 2 if total size < ~60MB (conservative half-L2 estimate). For batch sizes up to ~15M elements, two-pass layer norm avoids all HBM re-reads on pass 2.

**Tensor cores for reductions:** Standard scalar reductions don't use tensor cores. But fused patterns like attention (QK^T + softmax + V) can restructure as GEMM + reduce to hit tensor cores. For pure sum/mean/norm reductions, tensor cores are irrelevant — stay on FP32 CUDA cores.

**Occupancy:** B200 has 132 SMs. For small kernels (xnumel < 1000), you'll underutilize. Batch multiple independent reductions into one kernel launch, or increase XBLOCK to pack more work per SM.

---

### Typical speedup range

| Scenario | Over PyTorch baseline |
|---|---|
| Small reduce (dim ≤ 32), unrolled Triton | 2–4x (kernel launch overhead dominates baseline) |
| Medium reduce (dim 64–1024), fused ops | 1.5–3x |
| Large reduce (dim > 4096), memory-bound | 1.2–1.8x (both limited by HBM bandwidth) |
| Multi-op fusion (e.g., norm + scale + bias) | 3–6x vs unfused PyTorch ops |

The biggest wins come from fusion, not from the reduction algorithm itself.

---

### Common mistakes

**1. Using `triton_red` for tiny reductions.** If rnumel ≤ 32, the loop overhead + RBLOCK partial accumulation is slower than unrolled loads. Use `triton_poi` with explicit load-and-add.

**2. Missing `evict_last` on broadcast loads.** Every output element in a row re-loads the same row-sum. Without `evict_last`, these thrash L1 cache. This is especially bad when XBLOCK >> cache associativity.

**3. Wrong XBLOCK/num_warps ratio.** XBLOCK should be a multiple of 32 (warp size). num_warps = XBLOCK // 32 for compute-dense kernels, or lower if the kernel is memory-bound and you want more concurrent warps for latency hiding.

**4. Not fusing.** Writing separate kernels for "compute mean" and "normalize" forces two HBM passes. Always inspect whether reduction + subsequent elementwise ops can be merged.

**5. Naive two-pass when one-pass is feasible.** For small reductions (dim ≤ 32), one-pass Welford (online mean/variance) is possible and avoids the intermediate write. For large reductions on B200, two-pass is fine since L2 absorbs the intermediate.

**6. Ignoring the output stride.** Example 1 removes one dimension during reduction but keeps strides matching (`(16, 4, 1)` → `(4, 1, ...)`). Mismatched output strides force non-coalesced writes. Always verify output memory layout post-reduction.

---

### Memory-bound vs compute-bound

**Memory-bound regime (most reductions):**
- When reduction dim > ~256 and no heavy compute in the loop
- On B200: arithmetic intensity < ~300 FLOP/byte puts you in memory-bound territory
- Symptoms: occupancy ≥ 50%, SM utilization low, HBM bandwidth near peak
- Fix: increase RBLOCK to amortize HBM load cost, fuse more ops

**Compute-bound regime:**
- Fused kernels with many ops per loaded element (e.g., example 2's sqrt + division + multiply + add on every element)
- When reduction dim is small but the fused computation is heavy
- On B200: FP32 throughput is ~2.25 PFLOPS — hard to saturate with scalar math; mainly hit when using tensor cores

**Crossover point (empirical rule of thumb):**
- Reduction dim ≤ 64: always memory-bound; unrolled `triton_poi` wins
- Reduction dim 64–1024: memory-bound unless ≥5 fused ops per element
- Reduction dim > 4096: bandwidth-limited; focus on vectorized loads (`tl.load` with vector width 4) and maximizing in-flight memory requests
