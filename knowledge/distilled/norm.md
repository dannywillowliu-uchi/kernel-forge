# NVIDIA B200 GPU Kernel Optimization Guide

## I. What Makes Fast Kernels Fast: 3 Core Techniques

### 1. **Memory Bandwidth Saturation** (The #1 lever)

**Why it matters:** B200 HBM bandwidth is 8TB/s—the single highest-throughput resource on the chip. If your kernel doesn't saturate it, you're leaving 2-4x speedup on the table.

**The physics:**
- Modern GPU kernels are memory-bound, not compute-bound (see § VI for identification)
- Your kernel's performance = (data_volume / latency) = bandwidth utilization
- At 100% bandwidth utilization, a kernel moving 1GB of data takes ~125 µs on B200
- At 25% utilization, that same kernel takes 500 µs—4x slower

**How to saturate it:**
- **Coalesce memory access:** Ensure consecutive threads read/write consecutive memory locations. Bank conflicts (§ V) kill bandwidth.
- **Hide latency:** Issue many outstanding memory requests before blocking on results. In Triton, this is free—use large block sizes (see § III). In CUDA, explicitly manage register buffers and warp scheduling.
- **Maximize compute-to-memory ratio:** Reuse loaded data via tiling (§ III). A matrix multiply tiling fetches each element K times; unoptimized versions fetch it once.

**Metric:** Check `FBWDTH` (framebuffer bandwidth utilization) in NVIDIA Profiler. Target >70%.

---

### 2. **Instruction-Level Parallelism (ILP) & Latency Hiding**

**Why it matters:** B200 memory latency is ~200-400 cycles. If a warp stalls waiting for one load, it wastes GPU slots. Keeping warp schedulers fed requires scheduling independent operations in parallel.

**The physics:**
- Each SM has 4-6 warp schedulers and can issue ~2 instructions per clock per scheduler
- If you issue dependent instructions (`load → use`), stall cycles dominate
- If you interleave independent operations, schedulers feed continuously
- Example: Matrix multiply with K=256. If you load once and multiply once sequentially, you stall 100+ cycles. If you unroll the loop—loading multiple elements and multiplying with different accumulators in parallel—you hide stalls.

**How to exploit it:**
- **Unroll loops:** Process multiple iterations' worth of data per thread. Instead of `for i in range(N): acc += A[i] * B[i]`, do `for i in range(0, N, UNROLL): acc0 += A[i] * B[i]; acc1 += A[i+1] * B[i+1]; ...`
- **Use multiple accumulators:** Let the hardware reorder operations. 4-8 independent accumulators per thread are typical.
- **Increase occupancy:** More threads = more warps ready to run while others stall. 75%+ occupancy is good; aim for 90%+ (requires modest register pressure).

**Metric:** Check `SM Utilization` in profiler. >80% indicates good ILP. If <50%, you're leaving parallelism on the table.

---

### 3. **Hierarchical Computation via Shared Memory + Tiling**

**Why it matters:** Shared memory (SMEM) is 228KB per SM and 80x faster than HBM (~10 cycles vs. 200+). Loading data into SMEM once and reusing it reduces HBM bandwidth pressure by 10-100x.

**The physics:**
- Tiling = load a block of data into SMEM, do lots of computation, write results
- Reuse ratio = (FLOPs on block) / (HBM bytes moved for block)
- Unoptimized matmul: 2MK FLOPs, MK+MN+NK HBM bytes = reuse ~2. Tiled matmul: same FLOPs, but load M×K into SMEM, reuse K times across N outputs = reuse ~2K.
- At K=256, that's 500x less HBM pressure for the same work

**How to tile effectively:**
- **Calculate SMEM budget:** Each thread block computes a (BLOCK_M × BLOCK_N) tile of output. Load (BLOCK_M × BLOCK_K) into SMEM, multiply, accumulate. For FP32: ~1KB per 256 elements. (128×128×2) elements of A + B = ~200KB, fits in 228KB easily.
- **Plan SMEM layout:** Use `SharedMemory(torch.Size([BLOCK_M, BLOCK_K]))` in Triton to statically allocate. In CUDA, use `__shared__ float A[BLOCK_M][BLOCK_K];` and ensure no bank conflicts (§ V).
- **Synchronize barriers:** After loading, sync threads in the block (`tl.cdiv` in Triton, `__syncthreads()` in CUDA) before reading.

**Metric:** Track `SMEM Utilization`. If <30%, you're underutilizing cache. If >90%, you're register-pressure limited; reduce block size.

---

## II. B200 Architecture: Quick Reference

| Resource | Spec | Impact |
|----------|------|--------|
| **HBM Bandwidth** | 8 TB/s | Saturate via coalesced access; 128-256 element blocks per warp |
| **Shared Memory** | 228 KB / SM | Tile blocks to fit; enable 80x speedup vs. HBM |
| **Peak TF32 Throughput** | 964 TFLOPS | Theoretical max; realistic sustained is 60-80% at FP32 |
| **Memory Latency** | ~200-400 cycles | Hide via ILP (unroll, multiple accumulators) |
| **Warp Size** | 32 threads | Coalesce in multiples of 32; thread divergence kills 50%+ throughput |
| **Max Threads/Block** | 1024 | Use 256-512 for good occupancy + SMEM budget |
| **L1 Cache** | 128 KB / SM | Auto-managed; re-read same address → free cache hit |

**Key constraint:** 8TB/s bandwidth is shared across all SMs. A single kernel using all SMs can saturate it. But occupancy (# blocks) scales linearly until hitting SMEM/register limits.

---

## III. Common Code Patterns: Triton & CUDA

### Pattern 1: Tiled Matrix Multiplication (FP32)

**Triton (recommended for fast iteration):**

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 32,
):
    pid = tl.program_id(0)
    pidm = pid // tl.cdiv(N, BLOCK_N)
    pidn = pid % tl.cdiv(N, BLOCK_N)
    
    # Thread block ID → tile position
    rm = pidm * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rn = pidn * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate K dimension with BLOCK_K steps
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        rk = k * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # Load A[rm, rk], B[rk, rn] into registers (Triton auto-manages SMEM)
        a = tl.load(A + rm * stride_am[:, None] + rk[None, :] * stride_ak, mask=rm[:, None] < M)
        b = tl.load(B + rk[:, None] * stride_bk + rn[None, :] * stride_bn, mask=rn[None, :] < N)
        
        # Accumulate
        acc = tl.dot(a, b, acc=acc, allow_tf32=True)
    
    # Store C[rm, rn]
    tl.store(C + rm * stride_cm[:, None] + rn[None, :] * stride_cn, acc.to(tl.float32), mask=rm[:, None] < M)

# Launch
def matmul(A, B, C, M, N, K):
    grid = (tl.cdiv(M, 128) * tl.cdiv(N, 128),)
    matmul_kernel[grid](A, B, C, M, N, K, ...)
```

**Key design choices:**
- `BLOCK_M=128, BLOCK_N=128, BLOCK_K=32`: Fits ~200KB SMEM. 128×32 float32 = 16KB per matrix.
- `tl.dot()`: Calls `mma` (tensor cores) directly on B200. Saturates compute.
- `mask=...`: Handles non-square matrices without performance penalty (branch per thread, not per warp).

**CUDA equivalent (manual SMEM):**

```cuda
__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float a_shared[128][32];
    __shared__ float b_shared[32][128];
    
    int bm = blockIdx.x * 128;
    int bn = blockIdx.y * 128;
    int tm = threadIdx.x / 4;  // 32 threads; map to row/col
    int tn = threadIdx.x % 4;
    
    float acc = 0.0f;
    
    for (int k = 0; k < K; k += 32) {
        // Load A[bm+tm][k+tn] → a_shared[tm][tn] (coalesced, 4 threads per row)
        a_shared[tm][tn * 8] = (bm + tm < M && k + tn < K) ? A[(bm + tm) * K + k + tn * 8] : 0.0f;
        // Load B[k+tm][bn+tn] → b_shared[tm][tn] (transposed layout for coalesced reads)
        b_shared[tm][tn * 4] = (k + tm < K && bn + tn < N) ? B[(k + tm) * N + bn + tn * 4] : 0.0f;
        
        __syncthreads();
        
        // Multiply a_shared[tm][32] × b_shared[32][128], accumulate in acc
        for (int kk = 0; kk < 32; kk++) {
            acc += a_shared[tm][kk] * b_shared[kk][tn];
        }
        
        __syncthreads();
    }
    
    if (bm + tm < M && bn + tn < N) {
        C[(bm + tm) * N + bn + tn] = acc;
    }
}
```

**Optimization details:**
- `__shared__` is statically sized. Total = 128×32 + 32×128 = 8192 floats = 32KB (well under 228KB).
- Loop unrolling: `#pragma unroll 8` before the inner `for (kk...)` hides latency and increases ILP.
- Bank conflicts (§ V): Layout is conflict-free because each warp reads from different SMEM addresses.

---

### Pattern 2: Fused Reduction (Sum, Max, etc.)

**Triton:**

```python
@triton.jit
def reduce_sum_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr = 1024):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    idx = offset + tl.arange(0, BLOCK_SIZE)
    
    x = tl.load(X + idx, mask=idx < N, other=0.0)
    y = tl.sum(x)
    
    if tl.first(idx % WARP_SIZE == 0):  # One thread per warp
        tl.atomic_add(Y, y)
```

**Key pattern:**
- `tl.sum()`: Warp-level reduction (8 cycles, fully pipelined). Efficient if BLOCK_SIZE = multiple of warp size.
- `tl.atomic_add()`: Thread-safe global accumulation. Single bottleneck on Y, but fast enough for reductions.

---

### Pattern 3: Elementwise with SMEM (Activation + Bias Fusion)

**Triton (fused activation):**

```python
@triton.jit
def fused_activation(X, Bias, Y, N, BLOCK_SIZE: tl.constexpr = 1024):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    x = tl.load(X + idx, mask=idx < N)
    b = tl.load(Bias + idx, mask=idx < N)
    
    y = tl.nn.gelu(x + b)  # Fused: 1 HBM read per element vs. 3 separate ops
    
    tl.store(Y + idx, y, mask=idx < N)
```

**Why fused?**
- Naive: load X, write temp; load temp, load Bias, write temp; load temp, apply GELU, write Y. = 5 HBM round-trips.
- Fused: load X, load Bias, apply GELU, write Y. = 2 HBM round-trips. 2.5x bandwidth savings.

---

## IV. Speedup Expectations vs. PyTorch

| Kernel Type | B200 vs PyTorch | Reasoning |
|-------------|-----------------|-----------|
| **Matrix Multiply (FP32, M=N=K=4096)** | 1.3–1.8x | PyTorch cuBLAS is near-optimal; hand-written kernels match but rarely exceed |
| **Batched GEMMs** | 1.5–3x | Batch overhead in cuBLAS; custom kernels reduce launch cost |
| **Fused Activations (GELU, LayerNorm, etc.)** | 2–4x | PyTorch fuses some ops; you eliminate intermediate writes |
| **Reductions (sum, max, argmax)** | 1.2–2.5x | cuTensor (used by PyTorch) is solid; you gain via job-specific optimizations |
| **Sparse Operations** | 2–10x | PyTorch sparse kernels are conservative; structured sparsity enables big wins |
| **Custom Attention** | 1.5–3x | Flash Attention is already optimized; hand-rolled attention lags unless you innovate |

**Realistic guidance:**
- **Start at 1.2–1.5x expected gain** from a hand-optimized kernel. If you're seeing 1.05x, it means PyTorch is already very good (or your kernel has a bug).
- **2–3x is achievable** for kernels that fuse 2-3 operations or exploit problem-specific structure.
- **>5x requires innovation:** either exploiting hardware features PyTorch doesn't (tensor cores in a new way), or eliminating a fundamental PyTorch overhead (communication, synchronization).

---

## V. Common Mistakes & How to Avoid Them

### Mistake 1: Bank Conflicts (Kills 50%+ of SMEM bandwidth)

**What it is:** Shared memory is 32 banks. If two threads in the same warp access the same bank in the same clock, one stalls.

**Bad code:**
```cuda
__shared__ float data[32];
float x = data[threadIdx.x];  // If threadIdx.x = {0, 1, 2, ..., 31}, each accesses a different bank: OK
float x = data[threadIdx.x / 2];  // Threads 0,1 both read bank 0: CONFLICT
```

**Fix:**
- Pad SMEM: `float data[32 + 1];` (adds dummy element to shift banks)
- Or use Triton (Triton auto-avoids conflicts via smart memory layout)

**In Triton:** Not an issue; Triton handles layout. In CUDA, audit SMEM declarations and consider padding.

---

### Mistake 2: Warp Divergence (Kills 25–50% of throughput)

**What it is:** If threads in a warp follow different code paths, the GPU serializes them. A 32-wide warp becomes 16-wide + 16-wide.

**Bad code:**
```cuda
if (threadIdx.x < 16) {
    // Path A (16 threads)
} else {
    // Path B (16 threads)
}
// Both paths execute sequentially, not in parallel
```

**Fix:**
- Use predication (all threads execute both paths, masked by `if`), not branching
- Or ensure branch predicates are warp-aligned: `if (threadIdx.x / warpSize == 0)` (all or nothing per warp)

**In Triton:** Triton generates masked code automatically. Divergence is minimal.

---

### Mistake 3: Underutilized Shared Memory

**Problem:** You allocate 228KB but only use 50KB. Wastes block capacity; fewer blocks → lower occupancy → lower ILP.

**Fix:**
- Increase BLOCK_M or BLOCK_N to use more SMEM
- Target 80–90% occupancy. For matmul: 256×256×32 float32 ≈ 512KB (too much); 128×128×32 ≈ 128KB (good)

---

### Mistake 4: Inefficient Global Memory Striding

**Bad pattern:**
```cuda
// Reading every 32nd element (non-coalesced)
float x = data[threadIdx.x * 32];  // Thread 0 → addr 0, Thread 1 → addr 32, ... 
// Not coalesced; each thread fetches a different 128-byte cache line
```

**Good pattern:**
```cuda
// Consecutive elements (coalesced)
float x = data[blockIdx.x * blockDim.x + threadIdx.x];
// Threads 0–31 fetch elements 0–31 from the same cache line
```

**Metric:** Use NVIDIA's `achieve_occupancy` from nsys profiler. If <80%, you have memory stalls due to poor coalescing.

---

### Mistake 5: Over-unrolling (Register Pressure)

**Problem:** Too many accumulators or loop unrolls → spill to local memory (registers on disk, 100x slower than registers).

**Symptom:** Profiler shows high local memory usage + sudden perf cliff.

**Fix:**
- Target 50–60% register utilization per thread. B200 has 256 registers per thread. If you need 200 (78%), you're at risk.
- Reduce BLOCK_K or number of accumulators.

---

## VI. Identifying Memory-Bound vs. Compute-Bound

**Memory-Bound:** Kernel spends most cycles waiting for data, not computing.  
**Compute-Bound:** Kernel issues compute (tensor cores, FMA) at full rate.

### How to tell (using NVIDIA Profiler):

1. **Measure SM Utilization:**
   ```bash
   ncu --metrics sm__utilization.avg,l1tex__throughput_pipe_tensor_operations.sum <your_kernel>
   ```
   - If `sm__utilization` < 50% and `tensor_op throughput` is low → **Memory-bound**
   - If `sm__utilization` > 75% and `tensor_op throughput` is high → **Compute-bound**

2. **Roofline Model:**
   - B200 roofline = max(8TB/s bandwidth, 964 TFLOPS compute) — whichever you hit first
   - Arithmetic intensity = FLOPs / bytes moved
   - If `AI < 8` (bytes to FLOPs ratio high), you're memory-bound
   - Example: matmul with K=256, you reuse K times → AI ≈ 256, compute-bound

3. **Metric-based check:**
   ```
   Achieved TFLOPS = (FLOPs) / (kernel_time)
   Peak TFLOPS = 964 (TF32) or 1928 (TF32 with sparsity)
   If Achieved < 200 TFLOPS → Memory-bound
   If Achieved > 500 TFLOPS → Compute-bound
   ```

### What to optimize:

| Status | Optimize For |
|--------|--------------|
| **Memory-bound** | Reduce HBM traffic via tiling, caching, fusion. Increase AI via more SMEM reuse. |
| **Compute-bound** | Increase ILP, reduce branch divergence, maximize occupancy. You're doing compute well; don't add memory ops. |

---

## VII. Practical Checklist: Optimize a Kernel in 5 Steps

1. **Profile baseline:**
   ```bash
   pytorch_benchmark = 100.0  # ms, PyTorch reference
   ncu --metrics sm__utilization.avg,fbwdth.sum your_kernel_binary > baseline.csv
   ```

2. **Identify bottleneck (§ VI):**
   - Memory-bound? → Add tiling, increase BLOCK_K
   - Compute-bound? → Increase block size, unroll loops

3. **Implement optimization (§ III pattern):**
   - Triton: faster iteration. CUDA: finer control.
   - Allocate SMEM: (BLOCK_M × BLOCK_K + BLOCK_K × BLOCK_N) × element_size

4. **Verify improvement:**
   ```bash
   your_kernel_time = 50.0  # ms, your kernel
   speedup = pytorch_benchmark / your_kernel_time = 2.0x
   ```
   - If speedup < 1.2x, debug or try different BLOCK_M/N/K

5. **Check saturation (§ VI metric):**
   - Target SM utilization >75%
   - FBWDTH >70% (if memory-bound)
   - TFLOPS >500 (if compute-bound)

---

## VIII. Quick Reference: BLOCK_* Configuration

For a 128-thread warp × N blocks (common setup):

| Kernel | BLOCK_M | BLOCK_N | BLOCK_K | SMEM Est. | Occupancy |
|--------|---------|---------|---------|-----------|-----------|
| Matmul (FP32) | 128 | 128 | 32 | 32 KB | 90% |
| Batched GEMM | 64 | 64 | 16 | 8 KB | 100% |
| Reduction | 1024 | 1 | 1 | <1 KB | 50% |
| Activation (fusion) | 256 | 1 | 1 | <1 KB | 100% |

**Rule of thumb:**
- 128×128 tiles saturate HBM on B200 for matmul
- Reductions prefer large blocks (>256 threads) to amortize global synchronization
- Fused ops (activation) benefit from max occupancy (1024 threads) because SMEM cost is negligible

---

## IX. Additional Resources

- **NVIDIA B200 Whitepaper:** Architecture details, tensor core specs
- **Triton Tutorial:** https://triton-lang.org/main/getting-started/tutorials/
- **CUDA C++ Best Practices Guide:** https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **NVIDIA Profiler (ncu):** https://developer.nvidia.com/tools-downloads/nsight-compute

---

**TL;DR:** B200 kernels are fast when they (1) saturate HBM bandwidth via coalesced tiling, (2) hide latency via ILP, and (3) reuse data in SMEM. Start with Triton for 1.3–2x speedup over PyTorch; push to 3–5x via problem-specific optimization. Profile early, avoid bank conflicts and divergence, and keep occupancy >75%.
