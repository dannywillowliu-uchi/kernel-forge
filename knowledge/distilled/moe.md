## MOE Optimization Guide

### What makes fast MoE kernels fast

**1. Token reordering before GEMM (biggest win)**
Random scatter patterns destroy memory bandwidth. All fast implementations (moe_v3, both AMD examples) first sort/reindex tokens by expert assignment into a contiguous buffer (`t_input_reorder`). This converts irregular scatter into sequential reads for the GEMM phase. The GEMM then sees a dense `[total_routed_tokens, d_hidden]` layout instead of strided/indexed accesses. Typically **2-4x** of the total speedup comes from this alone.

**2. Two-phase separation (routing vs. compute)**
Split the kernel into: (1) a lightweight routing phase that computes `expert2token_count`, `within_expert_index`, and `token2index` maps; (2) a grouped/segmented GEMM that consumes the reordered buffer. moe_v3's Triton kernels and the AMD moe1/moe2 split both do this. Separation allows the routing phase to be tuned for memory bandwidth independently from the GEMM which is compute-bound.

**3. Warp-coherent cumsum for within-expert indexing**
moe_v3 uses `tl.cumsum` + `tl.sum` per BLOCK iteration to compute within-expert offsets. This avoids global atomics (which serialize under high contention with many tokens per expert). The cumsum approach is a prefix-sum over a boolean mask — O(tokens) work with log-depth latency in hardware.

---

### Common code patterns

| Pattern | Values seen | Notes |
|---|---|---|
| Block size | 512 tokens per block | Fits 512 × 2B (fp16 indices) in registers; good for H100/B200 occupancy |
| Grid shape | `(n_experts,)` — one program per expert | Expert-parallel; each program owns one row of `expert2token_count` |
| num_warps | 4 | Standard for memory-bandwidth-bound counting kernels |
| num_stages | 1 | No pipelining needed for pure load-reduce patterns |
| Expert buffer | Pre-allocated `[n_experts × max_tokens_per_expert, d_hidden]` | Must assume a max capacity; leads to padding waste |
| Reduction | `tl.sum(is_current_expert)` after cumsum | Gets total count without atomic add |

**Memory access pattern**: topk_indices are read in `BLOCK_SIZE`-sized chunks across the full `[tokens × k]` flat array. Load → mask → compare → cumsum → conditional store. The store mask (`is_current_expert.to(tl.int1)`) avoids writing -1 sentinel values.

---

### B200/Blackwell considerations

- **Large L2 (126MB)**: Token routing buffers (indices, counts, reordered activations) for batch sizes up to ~8K tokens at bf16/fp8 will fit in L2. This makes repeat-access patterns in the scatter/gather phase significantly faster than on H100.
- **228KB SMEM**: Use for expert weight tiles in the GEMM phase. With fp8, a 128×128 tile fits comfortably. Exploit double-buffering for async loads.
- **TMA (Tensor Memory Accelerator)**: Replace manual `tl.load`/`tl.store` with TMA-backed async copies for the expert weight loads. TMA handles non-contiguous strides natively, which helps with grouped GEMM where each expert may have different token counts.
- **wgmma / MMA async**: Use warpgroup-level matrix multiply for the expert GEMM tiles rather than per-warp mma. Each warpgroup handles one expert's tile.
- **FP8 tensor cores**: MoE experts are pure GEMMs — FP8 with hardware scaling gives ~2x throughput over BF16. The router logits still need BF16/FP32 for numerical stability.
- **Thread block clusters**: For very large expert counts, cluster-level collective operations can reduce cross-SM communication in the counting/prefix-sum phase.

---

### Typical speedup range

| Baseline | Optimized | Notes |
|---|---|---|
| PyTorch naive (loop over experts) | **3–6x** | Most of the gain is eliminating Python-level expert dispatch |
| PyTorch with `torch.index_select` routing | **1.5–3x** | Still pays for scatter/gather kernel launches |
| Custom Triton (no reordering) | **1.3–2x** | Mainly from fusing routing ops |

Top GPU MODE / POPCORN leaderboard entries typically show **4–8x** over the reference implementation on the AMD MI300X task (which uses a naive softmax + loop dispatch baseline).

---

### Common mistakes

1. **Naive selection sort for top-k**: The AMD examples use O(n×k) selection sort per token. For k=2-6 and n=256 experts this is tolerable (256 comparisons), but for n=1024+ experts it becomes a bottleneck. Use radix sort or partial bitonic sort instead.

2. **Global atomics for token counting**: Using `atomicAdd` into `expert2token_count` from all threads serializes under contention. The cumsum approach in moe_v3 is strictly better.

3. **Fixed max-capacity buffers assuming uniform load**: `max_tokens_per_expert = ceil(tokens × k / n_experts) × capacity_factor` — if experts are imbalanced (which they frequently are during training), this wastes memory and causes silent dropped tokens. Add a guard/overflow check.

4. **Not fusing top-k + softmax**: Router scores need softmax normalization anyway. Computing softmax first, then finding top-k with a partial sort fuses two passes into one. Many implementations do them separately.

5. **Launching many small expert GEMMs**: The naive approach launches N separate `torch.matmul` calls for N experts. Even with expert parallelism, launch overhead + poor SM utilization kills throughput. Batch them as a grouped GEMM (cutlass `GemmUniversal` or Triton segmented kernel).

6. **Over-autotuning with too many configs**: The moe_v3 autotuner has only one config (BLOCK_SIZE=512). This is actually fine for the counting kernel — the real tuning should happen on the GEMM block sizes (M, N, K tiles), not the routing.

---

### Memory-bound vs. compute-bound

```
Transition point ≈ when GEMM FLOPs / HBM bytes > GPU arithmetic intensity ceiling

For B200: ~3.5 TFLOP/s BF16 per SM, ~8 TB/s HBM3e total
```

| Regime | Token count | Bottleneck | What to optimize |
|---|---|---|---|
| Memory-bound | < 64 tokens | Routing scatter/gather, expert weight loads dominate | Coalescing, TMA, L2 reuse |
| Transitional | 64–256 tokens | Mixed; routing overhead + partial GEMM utilization | Fuse routing + first GEMM layer |
| Compute-bound | > 256 tokens | Expert GEMMs dominate | Tile sizes, FP8, wgmma, occupancy |

For **DeepSeek-V3 style** (d_hidden=7168, d_expert=2048, n_experts=256, k=8): transition is around 128–256 tokens. At inference with batch=1 (single token decode), the system is almost entirely memory-bound — expert weight loads, not GEMM throughput, are the ceiling. This is why continuous batching and expert caching matter for production MoE inference.
