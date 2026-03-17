## SOFTMAX Optimization Guide

### What makes fast softmax kernels fast

**1. Explicit unrolling for fixed small reduction dimensions**
When the softmax dimension is statically known and small (4 in most examples), kernels load each element individually with separate `tl.load` calls rather than using a loop. This eliminates loop overhead and lets the compiler schedule loads optimally. The `eviction_policy='evict_last'` hint on "broadcast" loads (the row's max/sum reused across threads) keeps those values warm in L2 rather than evicting them early.

**2. Two-pass decomposition with kernel fusion**
Every example splits softmax into: (1) a pointwise pass computing max-subtract-exp, (2) a reduction pass computing sum-divide. This is fused with adjacent operations (loss computation, normalization) to avoid a standalone softmax entirely. The "free" computation is absorbed into the loss kernel's reduction, saving one full HBM round-trip.

**3. Numerically stable online reduction without extra passes**
All kernels use `triton_helpers.maximum` (not naive `>`) for the max reduction, which is numerically safe. For small fixed sizes (4 elements), the max is computed by chaining `maximum` calls — no warp shuffle needed. For large dims, the online softmax algorithm (FlashAttention-style: update running max+sum in one pass) eliminates a second HBM read entirely.

---

### Common code patterns

**Block sizes:** All examples use XBLOCK as a `constexpr` with small values (typically 16–256 for the outer dimension). The reduction dimension is handled entirely within a single program via explicit loads, not tiled.

**Memory access pattern:**
```python
x0 = xindex % dim_size      # inner (reduction) index
x2 = xindex // row_stride   # outer (batch) index
# Load row max via explicit unroll:
tmp1 = tl.load(ptr + (x0 + stride * x2), eviction_policy='evict_last')
tmp2 = tl.load(ptr + (offset + x0 + stride * x2), eviction_policy='evict_last')
```
The current element load is `tl.load(ptr + x3, xmask)` — no eviction hint needed since it's only read once.

**Tiling pattern:** For the reduction pass (`triton_per_*`), the outer dim is `XBLOCK` (handled by program_id) and the reduction is `RBLOCK` (a constexpr handled with `tl.arange`). The standard shape is `[XBLOCK, RBLOCK]` for 2D tiling.

---

### B200/Blackwell considerations

**Memory-bound by default.** Softmax arithmetic intensity ≈ 4 FLOPs / 6 bytes ≈ 0.67 FLOPs/byte. B200 ridge point is ~250 FLOPs/byte. You are 375x below compute saturation — every optimization should target HBM bandwidth, not compute.

**228KB SMEM allows larger tiles.** Use RBLOCK up to 4096 for the reduction dimension before spilling. This enables online softmax to hold an entire sequence in SMEM on B200, avoiding multi-pass for sequences up to ~4K tokens in BF16.

**126MB L2 benefits batched softmax.** If computing softmax over many small tensors (batch of logits vectors), the reuse pattern across warps fits comfortably in L2. Set `eviction_policy='evict_last'` on any value read by multiple threads in the same program.

**TMA for async prefetching.** For multi-pass softmax on long sequences (>4K), use `tl.make_tensor_descriptor` + async copies to overlap the second pass's loads with the first pass's compute. Not needed for small kernels.

**BF16 storage, FP32 accumulation.** B200's BF16 throughput is ~2x FP32, but accumulating softmax denominators in BF16 loses ~2 decimal digits of precision. Load/store BF16, promote to FP32 for the `exp` and `sum`.

---

### Typical speedup range

- **Small fixed-size dims (4–64):** 1.5–2.5x over PyTorch via fusion. Most gain comes from eliminating the standalone softmax kernel launch and merging into the loss reduction.
- **Medium dims (512–4096):** 2–4x via online softmax (single HBM pass) and BF16 throughput.
- **Large dims (>8K, e.g. LM head):** 4–8x via persistent kernels + tiling + fusion with preceding matmul output.
- **Attention softmax (FlashAttention-style):** 8–15x due to eliminating the full attention matrix materialization.

---

### Common mistakes

**1. Two-kernel softmax without fusion.** Writing separate max-kernel → exp-kernel → sum-kernel → divide-kernel incurs 3 extra HBM round-trips. Merge into one or two kernels.

**2. Reducing along the wrong layout dimension.** If your tensor is row-major `[B, N]` and you reduce over `N`, threads in a warp must stride-access memory if N is the contiguous dim — kills memory coalescing. Ensure the reduction dim is contiguous (innermost in memory) or transpose before computing.

**3. Skipping the max-subtraction.** Computing `exp(x)` directly before subtracting the row max causes overflow for logits > 88 (FP32) or ~9 (FP16). Always do `exp(x - max(x))`.

**4. Under-blocking the outer dimension.** Using XBLOCK=1 for the outer loop means one thread block processes one row. For small rows, this wastes occupancy. Tile 8–32 rows per block.

**5. Using `tl.sum` on the wrong axis for 2D tiles.** `tl.sum(x, axis=1)` reduces columns, `axis=0` reduces rows. Transposing the indexing logic silently computes the wrong reduction without numerical errors — hard to catch.

**6. Not specializing BLOCK_SIZE for occupancy.** A BLOCK_SIZE of 1024 with no autotuning often leaves 50% of SMs idle on B200. Autotune `XBLOCK` and `RBLOCK` independently with `triton.autotune`.

---

### Memory-bound vs compute-bound

| Shape | Bottleneck | Notes |
|---|---|---|
| Seq len < 512 | Launch overhead | Persistent kernels dominate |
| Seq len 512–8K | HBM bandwidth | One-pass online softmax is optimal |
| Seq len > 8K, BF16 | HBM bandwidth (98%+) | TMA async prefetch helps marginally |
| Vocab softmax (>64K) | HBM bandwidth | Exp throughput becomes secondary concern at 100K+ |
| Fused attention softmax | SRAM bandwidth | Tiling to avoid HBM materialization is critical |

The bottleneck **never** shifts to compute for standard softmax on B200 with realistic shapes. The only exception is fused softmax+matmul (e.g., attention) where tensor cores become the bottleneck and the softmax is essentially free inside the tile.
