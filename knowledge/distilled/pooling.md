# GPU Pooling Kernel Optimization Guide
## Based on 85 Community Kernels (Triton/CUDA)

---

## 1. What Makes Fast Pooling Kernels Fast

### Memory Bandwidth Utilization
Pooling is almost universally **memory-bound**, not compute-bound. The arithmetic intensity is near-zero: you read K×K values, do K×K-1 comparisons or additions, write 1 value. On B200 (10 TB/s HBM3e), the ceiling is high — but most kernels leave significant bandwidth on the table.

**Fast kernels share these traits:**

| Trait | Why It Matters |
|-------|----------------|
| Coalesced reads along the W dimension | L1/L2 cache line = 128 bytes; misaligned access wastes 50-75% of bandwidth |
| XBLOCK sized to warp multiples (32, 64, 128) | Avoids partial warp execution; full SM utilization |
| `eviction_policy='evict_last'` on reused data | Prevents polluting L1 with data only read once |
| Fused activation post-pool | Eliminates a separate kernel launch and memory round-trip |
| No redundant index recomputation | Integer division/modulo is ~20 cycles on GPU; minimize it |

### From the Examples

**Example 1 (MultiLevelPooling)** — degenerate case. When pool window = output size, inductor emits a direct copy. Zero wasted work. This is the ceiling: arithmetic intensity = 0, pure bandwidth.

**Example 2 (MaxPool2dDynamicSamePadding)** — boundary-check penalty. The `tmp0 >= tmp1` / `>= tmp3` guards add predicated loads. Every thread evaluates the bounds even when 90% of outputs are interior pixels. This is a **common 15-30% slowdown source**.

**Example 3 (SppPooling)** — stride pattern `16 * x0`. Adaptive pool where output is much smaller than input. Non-unit strides cause **non-coalesced reads** — adjacent threads read locations 16×sizeof(float) apart instead of adjacent.

---

## 2. Common Patterns Across the 85 Kernels

### Pattern A: Flat 1D Launch (Most Common)
```python
# All kernels use xnumel = N*C*H_out*W_out, flat grid
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:]
```
**Implication:** Output-centric decomposition. Each thread owns one output element. Simple but doesn't exploit spatial locality of the input pool window.

### Pattern B: Index Decomposition
```python
x1 = xindex // 4 % 4   # H dimension
x0 = xindex % 4         # W dimension  
x4 = xindex             # flat index for output
```
The `//` and `%` operations recur constantly. On B200, integer division is expensive. **Optimization:** precompute strides at launch or use bit-shift when dimensions are powers of 2.

### Pattern C: Boundary-Guarded Loads (Same-Padding)
```python
tmp0 = -1 + x1          # offset by padding
tmp2 = tmp0 >= tmp1     # lower bound check
# ... mask all loads
tmp_val = tl.where(mask, tl.load(ptr), -float('inf'))
```
For `padding > 0`, every load is predicated. The `tl.where` with `-inf` for max-pool or `0` for avg-pool is correct but serializes the reduction in hardware.

**Faster alternative:** Separate interior vs. boundary tiles. 80-90% of tiles for typical 3×3 pooling on large feature maps are interior — handle them in a fast path with no guards.

### Pattern D: Strided Access in Adaptive Pooling
```python
tl.load(in_ptr0 + 16 * x0, ...)  # stride = output_ratio
```
When input/output ratio is not 1, adjacent output threads read non-adjacent input elements. **This is the #1 performance killer** in SppPooling / RoIPool variants.

### Pattern E: `reinterpret_tensor` for Layout Changes
Both Example 1 and 3 use `reinterpret_tensor`. This is a zero-copy view change. Fast kernels use this to **avoid explicit transposes** — pass the problem to the stride interpretation.

---

## 3. B200-Specific Considerations

### Hardware Profile (B200)
- HBM3e bandwidth: ~10 TB/s
- L2 cache: 96 MB (vs. 50MB on H100) — larger L2 dramatically helps small feature map pooling
- SM count: 160
- FP32 throughput: ~80 TFLOPS (irrelevant for pooling — you're bandwidth-bound before this matters)

### B200-Specific Tuning Knobs

**XBLOCK selection:**
- B200 warp size = 32 (same as all NVIDIA)
- Target occupancy: 2-4 warps/SM minimum
- For pooling: `XBLOCK=256` is almost always optimal — matches L1 cache line behavior and fills the SM

```python
# B200 autotune config for pooling
@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}),
        triton.Config({'XBLOCK': 256}),
        triton.Config({'XBLOCK': 512}),
    ],
    key=['xnumel']
)
```

**L2 reuse on B200:**
For 96MB L2, tensors up to ~12M float32 elements fit entirely. For typical inference batch sizes (B=1 to 16), most pooling inputs fit in L2. This means **second-pass reads are effectively free** — exploit this in two-pass avg-pool (sum then divide) without penalty.

**TMA (Tensor Memory Accelerator) opportunity:**
B200 has TMA hardware. Triton 3.x supports `tl.make_tensor_descriptor`. For 2D pooling, TMA async copies of the input window give ~20% bandwidth improvement over standard loads on B200 specifically.

### Memory Hierarchy for Pooling on B200

```
Register file  → hold accumulator (max or sum): 1 cycle
L1 cache       → 256KB/SM, ~5 cycles: pool window fits here for kernel ≤7×7
L2 cache       → 96MB total, ~30 cycles: entire input tensor for small batches
HBM3e          → ~10 TB/s, ~300 cycles: fallback for large inputs
```

**Key insight:** For ResNet-style 3×3 max-pool on 112×112 feature maps with 64 channels, the input is 112×112×64×4B = 3.2MB per batch element. B=8 = 25.6MB — fits in L2. Pool once, all subsequent operations hit L2. Don't fight the hardware.

---

## 4. Typical Speedup Ranges

| Kernel Type | Baseline (PyTorch eager) | Typical Triton | Optimized Triton | Notes |
|-------------|--------------------------|----------------|------------------|-------|
| 2×2 MaxPool, stride 2 | 1.0× | 0.9-1.1× | 1.2-1.5× | Inductor already close to optimal |
| 3×3 MaxPool, same pad | 1.0× | 0.8-1.0× | 1.3-2.0× | Padding overhead, big win removing guards |
| Adaptive AvgPool (global) | 1.0× | 1.5-2.5× | 3-5× | Parallel reduction wins big |
| SPP (multi-scale) | 1.0× | 1.2-1.8× | 2-4× | Fusing all scales into one kernel |
| RoIPool / RoIAlign | 1.0× | 1.5-2.0× | 2.5-4× | Non-contiguous access benefits from TMA |
| 1D AvgPool (audio/NLP) | 1.0× | 1.8-3.0× | 4-6× | Often badly unoptimized baseline |

**Reality check:** For large batch inference on B200, you're often within 10% of bandwidth ceiling already with `torch.compile`. The big wins are in:
1. Fusing pool + activation + next conv input prep
2. Eliminating separate pooling kernels for SPP (fuse all levels)
3. Persistent kernels that avoid re-launching for sequence of pools

---

## 5. Common Mistakes

### Mistake 1: Fighting the Scheduler with Too-Small XBLOCK
```python
# BAD: underutilizes SM
XBLOCK: tl.constexpr = 32  # one warp only

# GOOD: 8 warps/SM, fills pipeline
XBLOCK: tl.constexpr = 256
```
A single warp/SM means 7/8 of the SM's warp slots idle while memory latency is being hidden.

### Mistake 2: Uniform Boundary Checking for Interior-Heavy Kernels
```python
# BAD: checks bounds on every load, even interior pixels
for kh in range(kernel_h):
    tmp = kh + h_start - pad
    valid = (tmp >= 0) & (tmp < H)
    val = tl.load(ptr + tmp * W + kw, mask=valid, other=-inf)
```
For a 224×224 feature map with 3×3 pool and 1px padding, 99.2% of output positions are fully interior. Splitting into two kernels (interior fast path + border slow path) gives 1.5-2× on large feature maps.

### Mistake 3: Non-Coalesced Write Pattern
```python
# BAD: threads write to non-adjacent output locations
out_idx = batch * C * H_out * W_out + c * H_out * W_out + h * W_out + w
```
This is correct for NCHW layout — writes are coalesced along W. **The mistake** is using NHWC layout for pooling without transposing: then C-adjacent writes are strided by `H_out * W_out`. Verify your memory layout assumption.

### Mistake 4: Ignoring the `eviction_policy` for Single-Pass Data
```python
# BAD: loads go into L1, evicting useful data
tmp = tl.load(in_ptr0 + idx)

# GOOD: streaming data that won't be reused
tmp = tl.load(in_ptr0 + idx, eviction_policy='evict_first')
```
For global avg-pool, you read every input element exactly once. Using `evict_first` tells the hardware not to pollute L1/L2 — critical when running multiple ops in sequence.

### Mistake 5: Separate Kernels for Multi-Scale Pooling
```python
# BAD: 3 kernel launches for SPP
out1 = F.adaptive_avg_pool2d(x, (1,1))   # launch 1
out2 = F.adaptive_avg_pool2d(x, (2,2))   # launch 2  
out3 = F.adaptive_avg_pool2d(x, (4,4))   # launch 3
```
All three read the same input tensor. Fusing into one kernel reads input once, writes to three output buffers — 3× reduction in HBM reads.

### Mistake 6: Integer Arithmetic Explosion in Index Computation
The `xindex // width % height` pattern appears in almost every kernel. At XBLOCK=256 with 256 threads, this is 256 integer divides per thread. On B200, `idiv` throughput is ~4 cycles vs. 1 cycle for add. **Precompute or use modular arithmetic reformulation:**
```python
# Instead of recomputing per-element
h = xindex // W_out % H_out
w = xindex % W_out
# Use incremental counting in the loop (when iterating over output positions)
```

---

## 6. Memory vs. Compute Bound Analysis

### Classification Framework

```
Arithmetic Intensity (FLOP/byte) for Pooling:

MaxPool (k×k):    (k²-1) comparisons / (k²+1) * 4 bytes
                  ≈ (k²-1) / (4*(k²+1)) FLOP/byte

k=2: ~0.06 FLOP/byte   (pure memory bound)
k=3: ~0.10 FLOP/byte   (pure memory bound)
k=7: ~0.12 FLOP/byte   (pure memory bound)

AvgPool (k×k):    k² adds + 1 divide / (k²+1) * 4 bytes
                  ≈ same order

B200 ridge point: ~80 TFLOPS / 10 TB/s = 8 FLOP/byte
```

**Every standard pooling operation is 80× below the ridge point.** You are always memory-bound. Period. Optimizing for compute efficiency (reducing ops) is irrelevant — optimizing for memory access patterns is everything.

### The Roofline for B200 Pooling

```
Theoretical peak bandwidth: 10 TB/s
Achievable bandwidth (practical): ~7-8 TB/s (efficiency factor)

For MaxPool2d 3×3, 64 channels, 112×112 input:
  Input bytes:  64 * 112 * 112 * 4 = 3.2 MB
  Output bytes: 64 * 56 * 56 * 4 = 0.8 MB
  Total I/O:    ~4 MB

Theoretical minimum time: 4MB / 7.5TB/s = 0.53 μs
Typical PyTorch eager:    ~2-5 μs (4-10× off peak)
Good Triton kernel:       ~0.8-1.2 μs (1.5-2× off peak)
```

### Profiling Decision Tree

```
nsys/ncu shows:
├── Memory throughput < 50% of peak?
│   ├── Check: Is stride non-unit? → Fix coalescing
│   ├── Check: Is XBLOCK too small? → Increase to 256+
│   └── Check: Bounds checking overhead? → Split interior/border
├── Memory throughput 50-80% of peak?
│   ├── Check: Cache hit rate → eviction_policy tuning
│   └── Check: Kernel launch overhead → persistent kernel
└── Memory throughput > 80% of peak?
    → You're at the hardware limit; focus on fusion instead
```

### When Compute Actually Matters

Three cases where pooling becomes non-trivial compute:
1. **Fractional MaxPool** — probabilistic index sampling adds random number generation overhead
2. **Dilated MaxPool** — large dilation with small kernel can hit instruction throughput limits
3. **Mixed-precision** — BF16/FP16 max-pool requires upcast for comparison in some implementations; verify your kernel isn't doing F32 compares on F16 data

---

## 7. Actionable Optimization Checklist

For each pooling kernel, check in order:

- [ ] **Layout audit**: Is data NCHW? Are output writes coalesced along W?
- [ ] **XBLOCK=256**: If not, benchmark 128/256/512 and pick best
- [ ] **Padding separation**: If `padding > 0`, is there an interior fast path?
- [ ] **Eviction policy**: For single-pass reads, use `evict_first`
- [ ] **Stride check**: Are input strides unit along the innermost access dimension?
- [ ] **Fusion opportunity**: Can pool + activation be fused? Is there a following operation?
- [ ] **Multi-scale fusion**: If SPP/pyramid, fuse all scales into one kernel
- [ ] **Profile**: Run `ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l2_read_throughput` — if L2 throughput is near 10 TB/s, you're done
