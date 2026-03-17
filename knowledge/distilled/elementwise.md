```markdown
# Elementwise GPU Kernel Optimization Guide

B200 + Triton / CUDA — derived from 61 community kernels

---

## 1. Mental Model: Roofline First

Before writing a single line, classify your kernel:

```
Arithmetic Intensity (AI) = FLOPs / Bytes transferred

Memory-bound:  AI < ~200 FLOP/byte  →  chase bandwidth
Compute-bound: AI > ~200 FLOP/byte  →  chase throughput
```

**B200 roofline numbers (one GPC, practical):**
| Resource | Peak |
|---|---|
| HBM3e bandwidth | ~8 TB/s |
| FP32 throughput | ~60 TFLOP/s |
| BF16/FP16 (non-TC) | ~120 TFLOP/s |
| BF16 Tensor Core | ~2.25 PFLOP/s |
| FP8 Tensor Core | ~4.5 PFLOP/s |

**AI of common elementwise ops:**
| Op | AI (FP32) | Bound |
|---|---|---|
| `y = a + b` | 0.25 | Memory |
| `y = relu(x)` | 0.125 | Memory |
| `y = x * scale + bias` | 0.33 | Memory |
| `y = gelu(x)` | ~4–8 | Memory (barely) |
| `y = exp(x)` | ~6–10 | Compute (on CPU equiv), still mem on GPU |
| `y = softmax(row)` | ~2–3 | Memory |
| Flash-attn inner loop | ~10–50 | Compute (Tensor Core) |

**Rule of thumb:** If your kernel is just reading and writing tensors with cheap ops, it's memory-bound. Almost every fused activation is memory-bound on B200. Accept it and optimize bandwidth utilization.

---

## 2. Block Size Selection

### Triton (BLOCK_SIZE)

```python
# For 1D elementwise — start here
BLOCK_SIZE = 1024  # covers 4KB of float32 = one L2 fetch
```

| Scenario | Recommended BLOCK_SIZE |
|---|---|
| Simple unary (relu, neg, abs) | 1024–2048 |
| Binary elementwise (add, mul) | 1024–2048 |
| Expensive unary (exp, log, sqrt) | 512–1024 |
| Fused chains (norm + act + mul) | 512–1024 |
| Very small tensors (<4K elements) | 256–512 |

**Why BLOCK_SIZE matters:**
- Controls how many elements a thread block processes per launch
- Larger = better arithmetic intensity amortization, fewer kernel launches
- Too large = register spill, occupancy drop
- Always use powers of 2

**Autotune template:**
```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [512, 1024, 2048, 4096]
        for nw in [4, 8]
    ],
    key=["n_elements"],
)
```

### CUDA (threads per block)

```
256 threads/block → safe default, good occupancy
512 threads/block → good for register-light kernels  
1024 threads/block → maximum, risk occupancy drop if register use > 32/thread
```

---

## 3. Vectorized Loads: The Single Biggest Win

On B200, HBM bandwidth is massive but **each load transaction is 128 bytes**. Wasting it on scalar loads leaves 4–16× bandwidth on the table.

### Triton: Use `tl.load` on contiguous blocks
Triton handles vectorization automatically when accessing contiguous `pid`-strided blocks. Just ensure:
1. Input pointer is aligned (128 bytes = 32 float32s)
2. BLOCK_SIZE is ≥ 32
3. Access is contiguous (no gather/scatter)

```python
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(x_ptr + offs, mask=offs < n_elements)
```

Triton will emit 128-bit vectorized loads automatically. ✓

### CUDA: Explicit vectorization
```cpp
// Bad: scalar
float val = input[idx];

// Good: 4-element vector load
float4 val = reinterpret_cast<float4*>(input)[idx / 4];
```

**Speedup from vectorization:** 1.5–4× for memory-bound kernels.

### FP16/BF16: Double the vector width
For half-precision, load `float4` (which is 8 × fp16) or use `half8` custom types.

---

## 4. Memory Access Patterns

### Coalescing is mandatory

All threads in a warp must access contiguous memory addresses:

```
Warp threads: [0, 1, 2, ..., 31]
BAD:  thread i → input[i * stride]   ← strided, uncoalesced
GOOD: thread i → input[base + i]     ← coalesced, single transaction
```

**Detecting uncoalesced access:** `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` — if sectors >> requests, you have strided access.

### Layout matters: NCHW vs NHWC

For elementwise ops that broadcast across spatial dims:
- NHWC → spatial dims are contiguous → coalesced ✓
- NCHW → channel dim is outermost → may be uncoalesced for channel-wise ops

### Shared memory: Usually unnecessary for pure elementwise

Don't add shared memory to a pure elementwise kernel. It:
- Adds sync barriers
- Reduces occupancy
- Gains nothing (no data reuse)

**Exception:** If you're doing a reduction followed by elementwise (e.g., layernorm), shared memory is essential for the reduction pass.

### Cache pollution

For large tensors that won't fit in L2 (B200 L2 = 50MB), use streaming loads:
```cpp
// CUDA: hint non-temporal (bypasses L1)
__ldg(ptr)                    // streaming load
__stwt(ptr, val)              // streaming store
```
In Triton, this is automatic for out-of-cache tensors.

---

## 5. B200-Specific Considerations

### Memory bandwidth is the floor, not the ceiling

B200 HBM3e = ~8 TB/s. A kernel that fully saturates it and does nothing clever is still fast. Don't over-optimize for compute when you're memory-bound.

**Practical bandwidth utilization target:** 70–85% of peak. If you're below 50%, you have an access pattern problem.

### L2 cache = 50MB

For tensors < 50MB, the second kernel invocation will see L2 hits. This biases microbenchmarks. Always measure with tensors larger than L2 or flush cache between runs.

### Tensor Cores require specific shapes and dtypes

Elementwise kernels **cannot use Tensor Cores** (they're for matrix multiply). If a community kernel is claiming Tensor Core speedup for elementwise, it's fused with a matmul.

### NVLink / multi-GPU (B200 NVL)

For multi-GPU elementwise, the bottleneck shifts to NVLink bandwidth (~1.8 TB/s per link). Prefer all-reduce + local elementwise over per-element cross-GPU ops.

### FP8 for elementwise

B200 supports FP8 (e4m3, e5m2). For inference kernels:
```python
# Quantize input to FP8 elementwise
x_fp8 = x.to(torch.float8_e4m3fn)
```
FP8 halves bandwidth vs FP16. For memory-bound kernels, this is a **2× speedup** before any compute savings.

### SM count awareness

B200 has 132 SMs. Grid size should be a multiple of 132 (or 264, 396, ...) to ensure full utilization:
```python
grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
# For n=1M, BLOCK_SIZE=1024 → grid=977 → 977/132 = 7.4 waves → ok
# For n=128, BLOCK_SIZE=1024 → grid=1 → 131 SMs idle → bad
```

**Small tensor problem:** For tensors with < 132K elements at BLOCK_SIZE=1024, you won't fill all SMs. Consider smaller BLOCK_SIZE or persistent kernels.

---

## 6. Memory vs Compute Bound: Decision Tree

```
Is AI < 50 FLOP/byte?
├── YES → Memory-bound
│   ├── Fuse multiple ops (eliminate round-trips to HBM)
│   ├── Use vectorized loads (float4)
│   ├── Use lower precision (FP16/BF16/FP8)
│   └── Measure: bandwidth / peak_bandwidth → target > 70%
└── NO → Potentially compute-bound
    ├── Is dtype FP32?
    │   ├── YES → Can you use BF16? (2× throughput)
    │   └── NO → Already at target precision
    ├── Is there data parallelism within element?
    │   └── YES → Expose it (vectorize, unroll)
    └── Measure: FLOP/s / peak_FLOP_s → target > 50%
```

**Fusion is the #1 lever for memory-bound kernels.** Every HBM round-trip costs ~12ns. Fusing 3 elementwise ops (e.g., `out = gelu(x * scale + bias)`) into one kernel eliminates 2 extra round-trips.

**Speedup from fusion:**
| Fused ops | Expected speedup |
|---|---|
| 2 elementwise | 1.5–1.9× |
| 3 elementwise | 1.8–2.7× |
| 4 elementwise | 2.0–3.5× |
| Norm + activation + scale | 2–4× |

---

## 7. Common Mistakes

### M1: Wrong grid launch for small tensors
```python
# Bad: always launches N/BLOCK_SIZE blocks
grid = (n_elements // BLOCK_SIZE,)
# This silently drops the last partial block!

# Good: ceil division + masked load
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
# ...
mask = offs < n_elements
x = tl.load(ptr + offs, mask=mask, other=0.0)
```

### M2: Forgetting to handle non-contiguous inputs
```python
# Bad: assumes contiguous layout
x = tl.load(x_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))

# Good: use strides
x = tl.load(x_ptr + pid * BLOCK_SIZE * stride + tl.arange(0, BLOCK_SIZE) * stride)
```
Non-contiguous tensors (transposed, expanded, sliced) break vectorization and coalescing silently.

### M3: Numerically unstable reductions inside elementwise
```python
# Bad: naive softmax in single pass
exp_sum = tl.sum(tl.exp(x))  # overflows for large x

# Good: subtract max first
x_max = tl.max(x)
exp_sum = tl.sum(tl.exp(x - x_max))
```

### M4: Excessive atomics
```python
# Bad: atomic add in elementwise loop (serializes warps)
tl.atomic_add(out_ptr + idx, val)

# Good: only use atomics at reduction boundaries, not per-element
```

### M5: Register spill from too-large BLOCK_SIZE
Symptoms: occupancy drops to 1–2 warps/SM, performance worse than smaller BLOCK_SIZE.
Fix: reduce BLOCK_SIZE or split registers across multiple passes.

### M6: Not unrolling inner loops
```python
# Bad: loop inside kernel
for i in range(4):
    acc += tl.load(ptr + offs + i * stride)

# Good: unroll manually or use tl.constexpr loop count
UNROLL: tl.constexpr = 4
acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
for i in tl.static_range(UNROLL):
    acc += tl.load(ptr + offs + i * stride)
```

### M7: Comparing against PyTorch eager on CPU-bottlenecked shapes
If your tensor is < 10K elements, kernel launch overhead dominates. Profile with tensors > 1M elements for meaningful comparison.

### M8: Missing `num_warps` tuning
```python
# Bad: hardcoded 4 warps for compute-heavy kernel
@triton.jit
def kernel(...):  # default num_warps=4

# Good: autotune num_warps too
triton.Config({"BLOCK_SIZE": 1024}, num_warps=8)  # for complex ops
triton.Config({"BLOCK_SIZE": 1024}, num_warps=4)  # for simple ops
```

### M9: Unnecessary dtype casts inside the kernel
```python
# Bad: FP32 cast for intermediate of a BF16 kernel
x = tl.load(ptr).to(tl.float32)  # doubles register pressure
y = compute(x)
tl.store(out_ptr, y.to(tl.bfloat16))

# Good: only upcast for numerically sensitive accumulations (e.g., sum)
```

### M10: Writing to output before all threads have read input
When input and output alias (in-place ops), ensure warp-level ordering:
```python
x = tl.load(x_ptr + offs, mask=mask)  # load first
y = f(x)
tl.store(y_ptr + offs, y, mask=mask)  # then store
# If x_ptr == y_ptr: safe only because load completes before store
```

---

## 8. Speedup Ranges (Empirical)

From community kernels on B200-class hardware:

| Optimization | Typical Speedup | When It Applies |
|---|---|---|
| Fusion (2–3 ops) | 1.5–3× | Any memory-bound chain |
| Vectorized loads (scalar→float4) | 1.3–2.5× | Non-vectorized baseline |
| FP32 → BF16 | 1.6–2× | Memory-bound, precision allows |
| FP16 → FP8 | 1.5–2× | Quantization-friendly kernels |
| BLOCK_SIZE tuning | 1.1–1.5× | Wrong-sized baseline |
| Persistent kernel | 1.1–1.4× | Launch-overhead-heavy workloads |
| `num_warps` tuning | 1.05–1.3× | Compute-heavy unary ops |
| Eliminating branch divergence | 1.05–1.2× | Kernels with per-element conditionals |
| Tiling for cache reuse | 1.0–1.1× | Elementwise (usually negligible) |

**Total reachable speedup** for a naive PyTorch eager implementation of a fused pattern: **3–8×** is common. 10–15× is possible when baseline uses separate kernels + FP32.

---

## 9. Profiling Checklist

Use Nsight Compute (`ncu`) on B200:

```bash
ncu --metrics \
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
  l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  gpu__time_duration.sum \
  python run_kernel.py
```

| Metric | Good | Investigate if |
|---|---|---|
| `dram__bytes_read` bandwidth / 8TB/s | > 70% | < 40% |
| Achieved occupancy | > 50% | < 25% |
| Warp efficiency | > 85% | < 70% |
| L2 hit rate | > 80% for small tensors | < 20% for repeated access |
| Instructions per warp | baseline-matched | 2× baseline = unnecessary work |

---

## 10. Quick Reference: B200 Elementwise Template

```python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [512, 1024, 2048]
        for nw in [4, 8]
    ],
    key=["n_elements"],
)
@triton.jit
def elementwise_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Vectorized loads (Triton emits 128-bit instructions automatically)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)

    # Fused computation (eliminates intermediate HBM round-trips)
    out = tl.sigmoid(x) * y  # example: SiLU gate

    tl.store(out_ptr + offs, out, mask=mask)


def launch(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    elementwise_kernel[grid](x, y, out, n)
    return out
```

---

## 11. When to Stop Optimizing

For a given elementwise kernel, you're done when:
1. **Memory-bound:** Achieved bandwidth > 75% of HBM peak
2. **Compute-bound:** Achieved TFLOP/s > 60% of FP32 peak (or 50% BF16 non-TC)
3. **Further fusion impossible:** All adjacent ops are already in this kernel
4. **Diminishing returns:** Last optimization was < 5% improvement

If you're below these thresholds, the problem is always one of: uncoalesced access, insufficient vectorization, low occupancy, or unfused ops.
```
