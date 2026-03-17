# Distilled GPU Optimization Knowledge

Generated from 4000+ community kernels (GPU MODE KernelBook + kernelbot).


---

# GPU Kernel Optimization Guide for NVIDIA B200

A comprehensive reference for achieving fast kernels on NVIDIA B200 through understanding hardware, patterns, and common pitfalls.

---

## Table of Contents

1. [What Makes Fast Kernels Fast](#what-makes-fast-kernels-fast)
2. [B200 Hardware Specs & Implications](#b200-hardware-specs--implications)
3. [Memory-Bound vs Compute-Bound Analysis](#memory-bound-vs-compute-bound-analysis)
4. [Common Triton/CUDA Patterns](#common-triton-cuda-patterns)
5. [Typical Speedup Ranges Over PyTorch](#typical-speedup-ranges-over-pytorch)
6. [Common Mistakes](#common-mistakes)
7. [Decision Tree: Optimization Strategy](#decision-tree-optimization-strategy)

---

## What Makes Fast Kernels Fast

### 1. **Memory Coalescing & Data Movement (Primary Impact)**

**Why it matters:** Memory bandwidth is the bottleneck on most GPU workloads. On B200, uncoalesced memory access can waste 5–10x bandwidth.

**The technique:**
- Threads in a warp (32 consecutive threads) must access consecutive memory addresses when reading from global memory
- If thread 0 reads addr[0], thread 1 reads addr[1], ..., thread 31 reads addr[31], this is **coalesced** → 1 transaction
- If threads read random addresses or skip, this is **uncoalesced** → 32 separate transactions → 32x slower

**Example: Row-major vs column-major access**

```python
# GOOD: Coalesced (Triton)
# Thread i reads C[i, 0:BLOCK_SIZE] contiguously
tl.arange(0, BLOCK_N)[None, :] + tl.arange(0, BLOCK_M)[:, None] * stride_c

# BAD: Uncoalesced
# Thread i reads C[0:BLOCK_M, i] with stride
tl.arange(0, BLOCK_M)[:, None] * stride_c + tl.arange(0, BLOCK_N)[None, :]
```

**B200 consequence:** 8 TB/s HBM bandwidth is only achieved with perfect coalescing. Uncoalesced access drops to ~400 GB/s.

**How to verify:**
- Triton: Use `triton.profiler` to measure L1 cache hit rates and memory transactions
- CUDA: Use `ncu` (NVIDIA Nsight Compute) to measure `dram_read_throughput` vs peak

---

### 2. **Arithmetic Intensity (FLOP/Byte Ratio)**

**Why it matters:** Modern GPUs can do 10x more computation than memory bandwidth allows. If your kernel doesn't do enough math per byte, you're memory-bound forever.

**The formula:**
```
Arithmetic Intensity = FLOPs per kernel / Bytes loaded from global memory

For B200:
- Peak throughput: 964 TFLOPS (TF32)
- Peak bandwidth: 8 TB/s = 8,000 GB/s
- Compute ceiling: 964 TFLOPS / 8,000 GB/s = 0.12 FLOPS/byte before hitting memory limit
```

**Technique: Increase data reuse via tiling & shared memory**

```python
# Low intensity: Read A[i,k] once, use once
# C[i,j] += A[i,k] * B[k,j]  → AI ~= (2 * 128 * 128) / (128*4 + 128*4) ≈ 16

# High intensity: Load A into SMEM, use 128 times
# Tile size 128x128x64:
# Load: 128*64*4 + 64*128*4 = 65 KB into SMEM
# Compute: 128*128*64*2 = 2M FLOPs
# AI = 2M / (65K * 2) = ~15 FLOPS/byte → still compute-bound on B200!
```

**Why B200 is special:**
- 228 KB shared memory per SM (vs 96 KB on H100)
- Allows larger tiles → higher arithmetic intensity
- Strategy: Use **all 228 KB** via block sizes 256x128 or larger

---

### 3. **Occupancy & Latency Hiding**

**Why it matters:** Warps must be in flight to hide memory latency. If occupancy is low, you wait for memory.

**The technique:**
- **Occupancy** = (active warps) / (max warps per SM)
- B200: 2,560 max threads per SM = 80 warps
- Target: 50–75% occupancy (40–60 warps) to hide ~300-400 cycle memory latency

**Resource limiting factors:**
1. **Registers per thread**: Deep kernels (many variables) use more registers → fewer threads fit
2. **Shared memory**: Large tiles use SMEM → fewer blocks fit
3. **Warps launched**: Larger block size = fewer blocks in flight

**Example calculation:**

```
Block size: 256 (8 warps per block)
Shared memory: 128 KB per block
Registers: 64 per thread → 256 * 64 = 16 KB per block

Max blocks = min(
    SM_SMEM / 128KB = 228 / 128 ≈ 1 block,    ← SMEM-limited!
    SM_REGS / 16KB = 65536 / 16 = 4 blocks,
    SM_THREADS / 256 = 2560 / 256 = 10 blocks
)

Occupancy = 1 block * 8 warps / 80 max = 10% → TERRIBLE, memory latency NOT hidden
```

**Fix: Smaller block size or more aggressive register spilling**

```
Block size: 128 (4 warps per block)
SMEM: 64 KB per block
Regs: 64 per thread → 8 KB per block

Max blocks = min(228 / 64 = 3, 65536 / 8 = 8192, 2560 / 128 = 20) = 3 blocks
Occupancy = 3 * 4 / 80 = 15% → still bad
```

**Better solution: Reduce register footprint**

Use blocking loops instead of unrolled code. Let the compiler spill to registers/L1 cache rather than keeping everything hot.

---

## B200 Hardware Specs & Implications

### Memory Hierarchy

| Level | Size | Bandwidth | Latency | Notes |
|-------|------|-----------|---------|-------|
| **Registers** | 64 per thread (128 KB per SM) | N/A | 1 cycle | L0 cache |
| **L1 Cache** | 128 KB per SM | per-thread | ~4 cycles | Write-through to L2 |
| **L2 Cache** | 50 MB (shared) | ~3 TB/s | ~20 cycles | Coherent |
| **HBM3** | 192 GB | 8 TB/s | ~300 cycles | Device memory |

### Compute Units

| Metric | Value | Implication |
|--------|-------|-------------|
| **SMs** | 272 | 272 parallel kernels or high occupancy |
| **SMEM per SM** | 228 KB | +137% vs H100 → larger tiles possible |
| **Threads per SM** | 2,560 | Max block size is 1024; stay ≤ 512 for dual blocks |
| **FP32 Tensor Cores** | 2 per SM (TensorFloat32) | 964 TFLOPS TF32 |
| **FP8 Tensor Cores** | 8 per SM | 3.85 PFLOPS FP8 |
| **Memory Bandwidth** | 8 TB/s | Peak; requires 100% coalescing |

### Roofline Analysis for B200

```
For a kernel with arithmetic intensity AI (FLOPS/byte):

Peak Throughput = min(
    Peak Compute: 964 TFLOPS (TF32),
    Bandwidth Ceiling: 8 TB/s * AI
)

Examples:
- AI = 1 FLOP/byte → 8 TFLOPS → 0.8% peak (memory-bound!)
- AI = 4 FLOP/byte → 32 TFLOPS → 3.3% peak (memory-bound)
- AI = 16 FLOP/byte → 128 TFLOPS → 13% peak (starting to compute-bound)
- AI = 128 FLOP/byte → 964 TFLOPS → 100% peak (compute-bound, max throughput)

Implication: Most kernels are memory-bound unless they're matmuls or reduction ops.
```

---

## Memory-Bound vs Compute-Bound Analysis

### How to Diagnose

**1. Calculate theoretical arithmetic intensity:**

```python
def estimate_ai(kernel_flops, global_reads_bytes):
    """Arithmetic Intensity = FLOPs / Bytes from global memory"""
    return kernel_flops / global_reads_bytes

# Example: Softmax
# Input: (M, N), Output: (M, N)
# FLOPs: M * (N + 3*N) = 4*M*N (read, subtract, exp, divide)
# Reads: M*N*4 bytes (input) + M*4 bytes (max value) = ~M*N*4 bytes
# AI ≈ 4 FLOP/byte → MEMORY-BOUND

# Example: Matmul (M=256, N=256, K=256, no tiling)
# FLOPs: 2*256*256*256 = 33M
# Reads: 256*256 + 256*256 = 128K elements = 512 KB
# AI = 33M / 512K ≈ 64 FLOP/byte → COMPUTE-BOUND
```

**2. Measure empirically using Triton profiler:**

```python
from triton.profiler import measure

# Measure memory footprint
with measure():
    kernel[grid](args...)
    
# Triton report will show:
# - Bandwidth utilization (% of 8 TB/s)
# - FLOPS achieved
# - L1 cache hit rate
```

**3. Use NVIDIA Nsight Compute:**

```bash
ncu -k kernel_name --metrics dram_read_throughput,sm_efficiency /path/to/binary

# Look for:
# dram_read_throughput / 8000 GB/s = % of bandwidth used
# sm_efficiency = % of SM cycles active
```

### Optimization Strategy by Type

#### **Memory-Bound Kernels** (Optimize bandwidth first)

**Goal:** Minimize data movement, maximize coalescing, saturate memory bus.

| Problem | Solution | Example |
|---------|----------|---------|
| Uncoalesced access | Transpose data or use warp-level shuffle | Change loop order from `[i, j]` to `[j, i]` |
| Redundant loads | Share memory + local reduction | Load once to SMEM, reduce across block |
| Wrong data layout | Match kernel access pattern | NCHW vs NHWC; batch matrix layout |
| Too many passes | Fuse operations | Don't separate norm + bias; do in one pass |

**Example: Bandwidth-optimized softmax**

```python
@triton.jit
def softmax_fused(
    output_ptr, input_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_start = tl.arange(0, BLOCK_SIZE)
    
    # Single coalesced load from global
    input = tl.load(input_ptr + row_idx * N + block_start)
    
    # Compute max (reduction in registers first, then atomic)
    row_max = tl.max(input, axis=0)
    
    # Subtract and exp (fused into one pass)
    exp_input = tl.exp(input - row_max)
    
    # Single coalesced store
    tl.store(output_ptr + row_idx * N + block_start, exp_input)
```

#### **Compute-Bound Kernels** (Optimize arithmetic intensity)

**Goal:** Maximize data reuse per byte loaded, use tensor cores, reduce instruction latency.

| Problem | Solution | Example |
|---------|----------|---------|
| Low AI | Increase tile size or unroll loops | Use `BLOCK_M=256, BLOCK_N=256, BLOCK_K=64` |
| Register pressure | Use shared memory for data reuse | Load A into SMEM once, use many times |
| Tensor core underutilization | Ensure tile dims are multiples of 16 | Use 128x128 blocks, not 100x100 |
| Long latency chains | Prefetch next iteration while computing | Software pipelining in CUDA |

**Example: Compute-optimized matmul**

```python
@triton.jit
def matmul_compute_optimized(
    A, B, C, M, N, K,
    BLOCK_M: tl.constexpr = 256,
    BLOCK_N: tl.constexpr = 256,
    BLOCK_K: tl.constexpr = 64,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Accumulator in registers (use 228 KB SMEM to its max)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Tile loop: maximize data reuse
    for k in range(0, K, BLOCK_K):
        # Load tiles into SMEM (coalesced)
        a_tile = tl.load(A + (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * K + 
                          (k + tl.arange(0, BLOCK_K)[None, :]))
        b_tile = tl.load(B + (k + tl.arange(0, BLOCK_K)[:, None]) * N + 
                          (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]))
        
        # Matmul in registers (high AI within loop)
        acc += tl.dot(a_tile, b_tile)
    
    # Single coalesced store
    tl.store(C + (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * N +
               (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]), acc)
```

---

## Common Triton/CUDA Patterns

### Pattern 1: Coalesced Loading (CRITICAL)

```python
# ✅ CORRECT: Threads read consecutive memory
block_n = tl.program_id(0) * BLOCK_SIZE
offsets = block_n + tl.arange(0, BLOCK_SIZE)
data = tl.load(ptr + offsets)

# ❌ WRONG: Threads read with stride (uncoalesced)
offsets = (block_n + tl.arange(0, BLOCK_SIZE)) * N  # stride=N, every thread jumps
data = tl.load(ptr + offsets)
```

### Pattern 2: 2D Tile Access

```python
# ✅ CORRECT: Load 2D region as row-major stripes
block_m = tl.program_id(0) * BLOCK_M
block_n = tl.program_id(1) * BLOCK_N

m_offsets = block_m + tl.arange(0, BLOCK_M)[:, None]
n_offsets = block_n + tl.arange(0, BLOCK_N)[None, :]

# Row-major: contiguous reads across rows
tile = tl.load(ptr + m_offsets * N + n_offsets)

# ❌ WRONG: Column-major reads (strided)
tile = tl.load(ptr + m_offsets + n_offsets * N)  # Each column read has stride=N
```

### Pattern 3: Shared Memory for Data Reuse

```python
# Load once, use many times (reduce global memory traffic)
def tiled_reduction(input_ptr, output_ptr, N, BLOCK_SIZE=512):
    block_idx = tl.program_id(0)
    block_start = block_idx * BLOCK_SIZE
    
    # Coalesced load to SMEM
    smem = tl.load(input_ptr + block_start + tl.arange(0, BLOCK_SIZE))
    
    # Reduce within block (efficient warp shuffles)
    for s in range(BLOCK_SIZE // 2, 0, s //= 2):
        smem += tl.shuffle_down(smem, s)
    
    # Write final result
    if tl.thread_idx_x == 0:
        tl.store(output_ptr + block_idx, smem[0])
```

### Pattern 4: Block-Level Synchronization

```python
# Synchronize threads within a block (for SMEM consistency)
tl.dot(a, b)  # Triton automatically syncs
# or in CUDA:
__syncthreads();
```

### Pattern 5: Warp-Level Operations (Fast!)

```python
# Warp-level shuffle: no SMEM needed, ~10 cycles
value = tl.shuffle_down(x, 1)  # Shift within 32-thread warp

# Warp-level reduction: 5x faster than atomic
def warp_reduce(x):
    for s in range(1, 32, s *= 2):
        x += tl.shuffle_down(x, s)
    return x
```

### Pattern 6: Block Size Selection

```python
# B200 optimal block sizes (balance SMEM, occupancy, coalescing)

# For memory-bound (softmax, norm, bias):
BLOCK_SIZE = 1024  # max threads per block, 32 KB SMEM per block

# For compute-bound (matmul):
BLOCK_M, BLOCK_N = 256, 256  # fits in 228 KB SMEM
BLOCK_SIZE = 256 * 256 / 64 = 1024  # effectively tiles of size 256x256x64

# For occupancy-sensitive:
BLOCK_SIZE = 256 or 512  # allows 2-4 blocks per SM
```

---

## Typical Speedup Ranges Over PyTorch

### Memory-Bound Operations

| Kernel | PyTorch Baseline | Custom Kernel | Speedup | Why |
|--------|------------------|---------------|---------|-----|
| **Softmax** | 100% (very optimized) | 1.0-1.2x | 1.0-1.2x | PyTorch is already near-optimal (cuDNN) |
| **Layer Norm** | 100% | 1.1-1.3x | 1.1-1.3x | Fusing bias + norm helps; PyTorch does this |
| **GELU** | 100% | 1.2-1.5x | 1.2-1.5x | Custom fused GELU + bias; PyTorch less optimized |
| **Attention (QKV proj)** | 100% | 1.5-2.0x | 1.5-2.0x | Kernel fusion; PyTorch uses separate ops |
| **Attention (Softmax)** | 100% | 1.0-1.1x | 1.0-1.1x | PyTorch uses cuDNN flash attention |

### Compute-Bound Operations

| Kernel | PyTorch Baseline | Custom Kernel | Speedup | Why |
|--------|------------------|---------------|---------|-----|
| **Matmul (no TensorCore)** | 100% | 2-3x | 2-3x | cuBLAS is well-optimized; custom uses TF32 |
| **Matmul (TensorCore)** | 100% | 1.1-1.3x | 1.1-1.3x | cuBLAS already uses TC; custom overhead |
| **Grouped Matmul** | 60-80% | 1.0-1.5x | 1.2-2.5x | PyTorch launches separate kernels; custom is one |
| **Fused Matmul + Bias** | 90% | 1.1-1.3x | 1.1-1.3x | PyTorch fusion already good |
| **Fused Matmul + GELU** | 80% | 1.5-2.5x | 1.5-2.5x | Multi-op fusion pays off |

### Bandwidth-Limited Sweet Spot

```
Typical fusion payoff:
- Separate ops: 3-5 kernel launches, each 20-40% peak bandwidth
  Total throughput: 1 op every 50-100 cycles

- Fused ops: 1 kernel launch, 60-80% peak bandwidth
  Total throughput: 3 ops every 50-100 cycles

Speedup ≈ (3 ops, high BW) / (1 op, low BW) ≈ 2-3x
```

### Conditions for Highest Speedup

1. **Fused multi-op (matmul + activation + bias)**: 3-5x
2. **Grouped operations (batched matmul, attention heads)**: 2-3x
3. **Unusual memory access (sparse, grouped)**: 2-4x
4. **Simple ops with poor PyTorch coverage**: 1.5-2x
5. **Already well-optimized in PyTorch (softmax, norm)**: 1.0-1.2x

---

## Common Mistakes

### Mistake 1: Ignoring Coalescing (CRITICAL)

```python
# ❌ WRONG: Uncoalesced access (5-10x slower)
for i in range(M):
    for j in range(N):
        data[j * M + i]  # Threads read with stride=M
        
# ✅ CORRECT: Coalesced access
for i in range(M):
    for j in range(N):
        data[i * N + j]  # Threads read consecutive memory
```

**Cost:** Drops bandwidth from 8 TB/s to ~400 GB/s. One mistake here can wipe out all other optimizations.

---

### Mistake 2: Excessive Shared Memory Usage

```python
# ❌ WRONG: 256 KB SMEM per block (doesn't fit; B200 has 228 KB)
BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 256
# SMEM = 256*256*4 + 256*256*4 = 512 KB > 228 KB limit
# Kernel will fail or run at 1 block per SM (0% occupancy)

# ✅ CORRECT: Use 200 KB SMEM per block
BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
# SMEM = 256*64*4 + 64*256*4 = 130 KB < 228 KB
# Allows 2+ blocks per SM, better occupancy
```

**Diagnostic:**
```bash
nvcc -Xptxas -v,--stats dummy.cu 2>&1 | grep "smem"
# or in Triton:
triton.compiler.compile(...).kernel.smem
```

---

### Mistake 3: Register Pressure (Occupancy Death Spiral)

```python
# ❌ WRONG: Too many local variables
def bad_kernel(x):
    acc1 = x + 1; acc2 = x + 2; acc3 = x + 3; ...  # 100 vars
    acc50 = x + 50
    # Compiler spills to memory → 10-100x slower
    return sum([acc1, ..., acc50])

# ✅ CORRECT: Reuse variables, avoid explosion
def good_kernel(x):
    acc = x + 1
    for i in range(2, 51):
        acc += x + i
    return acc
```

**How to check:**
```bash
# CUDA: ptxas -v flag shows register usage
nvcc -ptxas -v kernel.cu

# Triton: check kernel IR or use profiler
```

---

### Mistake 4: Misaligned Memory Access

```python
# ❌ WRONG: Unaligned base pointer (8-byte vs 16-byte)
data = torch.randn(1000).data_ptr() + 1  # misaligned!
kernel(data)  # hardware doesn't coalesce

# ✅ CORRECT: Ensure 16-byte alignment
data = torch.randn(1000).data_ptr()  # 16-byte aligned by default
kernel(data)
```

**B200 requirement:** Pointers should be 16-byte aligned for 128-bit transactions. Triton auto-aligns; CUDA requires manual care.

---

### Mistake 5: Divergent Branching (Warp Serialization)

```python
# ❌ WRONG: Warp divergence (threads take different paths)
if tl.program_id(0) % 2 == 0:
    tl.store(ptr_a, value)
else:
    tl.store(ptr_b, value)
# Warp executes both paths sequentially → 2x slower

# ✅ CORRECT: Uniform branching per warp
if tl.program_id(0) % WARP_SIZE == 0:
    # All threads in warp take same path
    tl.store(ptr_a, value)
```

---

### Mistake 6: Atomic Contention

```python
# ❌ WRONG: All blocks atomic-add to same location
tl.atomic_add(global_counter, 1)  # 272 blocks contending!
# Serializes, effective throughput: 1 atomic per block

# ✅ CORRECT: Use warp-level or block-level reduction
block_result = warp_reduce_sum(local_value)  # no atomics
if tl.thread_idx_x == 0:
    tl.atomic_add(global_counter, block_result)  # 1 atomic per block
```

---

### Mistake 7: Missing Synchronization

```python
# ❌ WRONG: Read from SMEM without __syncthreads()
tl.store(smem + idx, value)
result = tl.load(smem + other_idx)  # May read stale SMEM!

# ✅ CORRECT: Synchronize
tl.store(smem + idx, value)
tl.wait(smem)  # or __syncthreads() in CUDA
result = tl.load(smem + other_idx)
```

---

### Mistake 8: Assuming Linear Speedup

```
Speedup is NOT proportional to throughput improvement:
- Bandwidth improvement 2x → speedup ~1.8x (Amdahl's law: other latencies matter)
- Latency improvement 2x → speedup ~1.5x (critical path matters, not all cycles)

Always measure wall-clock time, not just utilization.
```

---

## Decision Tree: Optimization Strategy

```
START: "I want to speed up kernel X"
│
├─ Step 1: MEASURE BASELINE
│  ├─ Time on B200: torch.cuda.Event() or nsys
│  ├─ PyTorch vs custom: compare absolute ms
│  └─ Current utilization: ncu --metrics sm_efficiency
│
├─ Step 2: CLASSIFY OPERATION
│  ├─ Is it matmul-like (A @ B)?
│  │  └─ GO TO: Matmul Strategy (page below)
│  │
│  ├─ Is it element-wise (softmax, norm, GELU)?
│  │  └─ GO TO: Memory-Bound Strategy
│  │
│  ├─ Is it a reduction (sum, max, mean)?
│  │  └─ GO TO: Reduction Strategy
│  │
│  └─ Is it a custom fusion (multi-op)?
│     └─ GO TO: Fusion Strategy
│
├─ Step 3: MEMORY-BOUND STRATEGY
│  │  Target: Saturate bandwidth, minimize global memory traffic
│  │
│  ├─ Analyze memory pattern:
│  │  ├─ Are reads coalesced? ncu --metrics l1cache_hit_rate
│  │  ├─ Are there redundant loads? Add intermediate reduction?
│  │  └─ Wrong data layout? Transpose before kernel?
│  │
│  ├─ If not coalesced:
│  │  └─ Reorder loops to match memory layout
│  │
│  ├─ If high L1 cache miss:
│  │  └─ Reduce data working set via blocking
│  │
│  └─ If multiple passes (norm + bias):
│     └─ Fuse into single kernel
│
│  Expected gain: 1.2-2.0x (PyTorch already good)
│
├─ Step 4: COMPUTE-BOUND STRATEGY
│  │  Target: Maximize arithmetic intensity, use tensor cores
│  │
│  ├─ Calculate AI (FLOPs / bytes):
│  │  ├─ AI < 4? → Still memory-bound, don't optimize here
│  │  ├─ 4 < AI < 16? → Borderline, increase tile size
│  │  └─ AI > 16? → Compute-bound, focus on throughput
│  │
│  ├─ Increase data reuse:
│  │  ├─ Load tile to SMEM, reuse 64-128 times
│  │  └─ Block size: 256x256 for B200 (228 KB SMEM)
│  │
│  ├─ Use tensor cores:
│  │  ├─ Block dims multiple of 16 (for tensor core alignment)
│  │  ├─ For Triton: tl.dot() uses TC automatically
│  │  └─ For CUDA: mma.sync (Volta+) or cuBLAS
│  │
│  └─ Prefetch next iteration:
│     └─ Load A[k+1], B[k+1] while computing A[k], B[k]
│
│  Expected gain: 1.3-3.0x (higher for grouped matmul)
│
├─ Step 5: REDUCTION STRATEGY
│  │  Target: Minimize latency, use warp-level ops
│  │
│  ├─ Single block? (size < 1024)
│  │  └─ Warp-shuffle reduction (10 cycles per level)
│  │
│  ├─ Multi-block? (size > 1024)
│  │  ├─ Block-level reduction (parallel)
│  │  └─ Global atomic-add to single output
│  │
│  └─ If contention on atomic:
│     └─ Use multiple atomics + grid-stride loop
│
│  Expected gain: 1.2-1.8x
│
└─ Step 6: VERIFY & BENCHMARK
   ├─ Run on B200 for 100+ iterations
   ├─ Compare vs PyTorch (same input sizes)
   ├─ Profile with nsys or ncu
   ├─ Check: bandwidth utilization, occupancy, cache hit rates
   └─ If no improvement: Re-measure, debug assumption
```

---

## Matmul Strategy (Deep Dive)

```
For matmul (C = A @ B):

Step 1: Choose tile size BLOCK_M, BLOCK_N, BLOCK_K
├─ B200 SMEM: 228 KB
├─ Need: 2 * BLOCK_M * BLOCK_K * 4 + 2 * BLOCK_N * BLOCK_K * 4 < 228 KB
│  (factor 2 for input + output)
├─ For square tiles: BLOCK_M = BLOCK_N = 256, BLOCK_K = 64
│  → SMEM = 256*64*4 + 64*256*4 = 130 KB ✓
│
└─ Arithmetic intensity:
   ├─ FLOPs: 2 * BLOCK_M * BLOCK_N * BLOCK_K (per tile iteration)
   ├─ Bytes: BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K (unique loads)
   ├─ AI per iteration: 2*256*256*64 / (256*64*4 + 64*256*4) ≈ 128 FLOP/byte
   └─ Compute-bound? YES (128 >> 0.12 threshold)

Step 2: Block organization
├─ Grid: (M / BLOCK_M, N / BLOCK_N)
├─ Block size: min(256, BLOCK_M * BLOCK_N / warp_size)
│  → For 256x256 tile: 256x1 or 128x2 or 64x4 (thread organization)
└─ Occupancy: Want 2+ blocks per SM
   └─ SMEM per block: 130 KB → allows 228 / 130 ≈ 1-2 blocks

Step 3: Implement tiling loop
├─ Load A[m, :], B[:, n] tiles into SMEM (coalesced)
├─ Compute partial C[m, n] += A[m, k] @ B[k, n]
├─ Accumulate in registers (no spill)
└─ Move to next tile k

Step 4: Optimize K loop
├─ Prefetch next tile while computing current
├─ Use pipelined loads (separate load thread vs compute thread)
└─ Unroll innermost loop 2-4x to hide latency

Expected speedup: 1.3-2.0x vs cuBLAS (cuBLAS is already very good)
```

---

## Quick Reference: B200 Optimization Checklist

- [ ] **Coalescing**: Verify threads read consecutive memory (ncu L1 hit rate > 90%)
- [ ] **Arithmetic Intensity**: Calculate FLOPs/bytes; if < 4, focus on memory
- [ ] **SMEM Usage**: Keep per-block SMEM < 228 KB; target 150-200 KB
- [ ] **Block Size**: 256-512 threads for memory-bound; 256-1024 for compute-bound
- [ ] **Occupancy**: Target 50-75% (40-60 warps active per SM)
- [ ] **Register Pressure**: Check ptxas output; ideal < 64 registers per thread
- [ ] **Data Layout**: Row-major for coalesced reads; avoid column-major access
- [ ] **Synchronization**: Use `__syncthreads()` or Triton `tl.wait()` for SMEM consistency
- [ ] **Warp Divergence**: Ensure all threads in warp take same control path
- [ ] **Atomic Contention**: Avoid global atomics in innermost loops; use reduction
- [ ] **Profiling**: Always measure on B200; don't extrapolate from simulations

---

## Conclusion

Fast kernels on B200 come from:

1. **Memory coalescing** → must-have, 5-10x impact if wrong
2. **Arithmetic intensity** → tile and reuse data, 2-4x impact
3. **Occupancy & latency hiding** → choose block sizes carefully, 1.5-2x impact

Start with measurement, classify the operation, then apply the appropriate strategy. PyTorch is already very good; custom kernels shine in fusion, unusual access patterns, and grouped operations.


---

## CONV Optimization Guide

---

### What makes fast conv kernels fast

**1. Delegate conv to cuDNN/CUTLASS, fuse everything else in Triton**

All 5 examples use `extern_kernels.convolution` (cuDNN under the hood) for the actual weight × input multiply, then handle bias-add and activations in separate Triton pointwise kernels. The conv itself is near-optimal via cuDNN's autotuned GEMM paths (Winograd for 3×3, FFT for larger kernels, direct GEMM for 1×1). Your optimization leverage is in fusing the post-ops — every extra kernel launch reads/writes the full output tensor from HBM.

**2. Fuse bias + activation into a single in-place pass**

Example 1 fuses `bias_add + ReLU + threshold_backward` into one kernel (`in_out_ptr0` is read, modified, and stored in place). Example 4 fuses `cat + tanh * sigmoid` (GLU gate). Each un-fused elementwise op costs a full HBM round-trip on the output tensor. On B200 with 8 TB/s HBM3e, a 128MB activation tensor costs ~16µs per read — fusing 3 ops saves ~32µs.

**3. `eviction_policy='evict_last'` on broadcast operands**

Every example loads the bias vector with `eviction_policy='evict_last'`. This pins the bias in L2 across warps since it's tiny (C floats, e.g. 16 bytes for C=4) but accessed once per output element. Without this hint, L2 eviction under pressure would force repeated HBM reads.

---

### Common code patterns

**Block sizes:**
- `xnumel ≤ 32`: `XBLOCK=16-32`, `num_warps=1` — don't over-provision for tiny tensors
- `xnumel 64-512`: `XBLOCK=128`, `num_warps=4` — standard for medium bias-add fusions
- `xnumel > 4096`: `XBLOCK=256-1024`, `num_warps=8` — benchmark; larger blocks hide memory latency

**Bias broadcast pattern (all 5 examples):**
```python
x1 = xindex // spatial_size % C  # extract channel dim
bias = tl.load(in_ptr0 + x1, eviction_policy='evict_last')
out = conv_out + bias
```
This is the canonical pattern. `xindex // 16 % 4` means spatial=16, C=4. Adjust divisors for your shape.

**In-place fusion:**
```python
tmp = tl.load(in_out_ptr0 + x)  # read conv output
tmp = tmp + bias                  # bias add
tmp = triton_helpers.maximum(0, tmp)  # ReLU
tl.store(in_out_ptr0 + x, tmp)   # write back in-place
```
Use a separate `out_ptr` only when you need to preserve the pre-activation value (e.g., for backward mask in Example 1).

**Padding as a separate Triton kernel (Example 2):**
The `constant_pad_nd` kernel pads by materializing a larger buffer and zero-filling boundaries using conditional loads. This is correct but adds an extra kernel launch + HBM allocation. For fixed padding, prefer passing `padding=` directly to `extern_kernels.convolution` — cuDNN handles it in-register.

**CausalConv "over-pad then slice" (Example 3):**
```python
buf0 = extern_kernels.convolution(..., padding=(dilation*(kernel_size-1),))
return reinterpret_tensor(buf0, (B, C, T), strides, offset=0)  # slice off tail
```
`reinterpret_tensor` is zero-cost (metadata-only). Valid only when the slice is contiguous in the underlying buffer.

---

### B200/Blackwell considerations

**TMA (Tensor Memory Accelerator):** Use `tl.experimental_descriptor_load` (Triton 3.x) for async 2D tile loads into shared memory. Critical for conv kernels that tile over spatial dims — eliminates index calculation overhead and enables pipelining of memory and compute.

**228KB SMEM:** Blackwell doubles H100's SMEM. For a direct convolution tile of size `(Bx × Kx × BLOCK_N × BLOCK_K)` in fp16, you can hold ~56K elements per tile — enabling `BLOCK_M=128, BLOCK_N=256` tiles without spilling. Size your tiles to fill SMEM, not H100-era 164KB.

**FP8 tensor cores:** B200 has 4 PFLOPS FP8 vs 2 PFLOPS BF16. For inference, cast weights to `e4m3` and activations to `e4m3`/`e5m2`. None of the 5 examples use FP8 — all run in FP32 on toy shapes. Real production kernels should target BF16 minimum, FP8 where accuracy permits.

**126MB L2:** The full weight tensor of a ResNet conv layer (e.g., 512×512×3×3 FP16 = 4.7MB) fits in L2. For weight-stationary kernel loops, if weights fit in L2, subsequent output tiles read weights from L2 not HBM — bias the tiling to maximize weight reuse.

**Warp group size = 4 warps (128 threads):** Blackwell's MMA instruction operates on 4-warp groups. Set `num_warps` to a multiple of 4 for tensor core ops.

---

### Typical speedup range

| Scenario | Speedup over `torch.nn.Conv2d` |
|---|---|
| Bias-add fusion only (what these examples do) | 1.05–1.15× for small tensors, negligible for large |
| Bias + activation fusion | 1.1–1.3× |
| Custom tiled direct conv (small kernels, 1×1, 3×3) | 1.2–2.5× |
| FP8 vs FP32 | 3–4× |
| Full custom kernel (TMA + FP8 + tuned tiling) | 2–5× over cuDNN FP16 |

The examples are all tiny (batch=4, C=4, spatial=4×4) — speedup numbers at these sizes are noise-dominated. Real measurements require `N≥64, C≥64, HW≥32`.

---

### Common mistakes

**1. Writing a Triton conv from scratch when cuDNN is available**
For standard conv (stride=1, dilation=1, groups=1), cuDNN's autotuned paths (e.g., IMPLICIT_PRECOMP_GEMM, WINOGRAD_NONFUSED) beat hand-written Triton for most shapes. Write custom kernels only for: fused sequences cuDNN can't express, non-standard layouts, or shapes cuDNN heuristics mishandle.

**2. Padding in a separate kernel**
Example 2 materializes the padded input as a new buffer before calling convolution. This doubles HBM traffic for the input. Pass padding directly to `extern_kernels.convolution` and let cuDNN absorb it.

**3. XBLOCK too large for tiny tensors**
Setting `XBLOCK=256` with `num_warps=4` on `xnumel=16` wastes 240/256 threads. TorchInductor correctly uses `XBLOCK=16, num_warps=1` for Example 1. Always match block size to problem size.

**4. Ignoring `num_stages` for pipelining**
All examples use `num_stages=1`. For memory-bound kernels with predictable access patterns, `num_stages=3-4` enables software pipelining (prefetch next tile while computing current). At B200 memory bandwidth, this can recover 20–40% of latency for large tensors.

**5. FP32 accumulation when BF16 is sufficient**
These examples all operate in FP32. For inference, BF16 halves memory traffic and doubles tensor core throughput with negligible accuracy impact on most conv layers.

---

### Memory-bound vs compute-bound

The crossover depends on arithmetic intensity (FLOPs / bytes read):

| Shape | Intensity | Bottleneck |
|---|---|---|
| 1×1 conv, C≤64 | < 10 FLOPs/byte | Memory-bound (HBM) |
| 3×3 conv, C=64–256 | 10–50 FLOPs/byte | Balanced — tune tiling carefully |
| 3×3 conv, C≥512 | > 50 FLOPs/byte | Compute-bound (tensor cores) |
| Depthwise conv (groups=C) | 1–5 FLOPs/byte | Always memory-bound |
| 1×1 conv, large batch | > 100 FLOPs/byte | Compute-bound |

**Rule of thumb for B200:** With 8 TB/s HBM and 4 PFLOPS FP8, the ridge point is at ~500 FLOPs/byte. Most single-layer convolutions in practice are memory-bound unless batch size is very large (≥256) or channels are very wide (≥1024). Depthwise is always memory-bound — optimize for memory coalescing and vectorized loads (`tl.load` with aligned offsets) rather than tiling depth.

For memory-bound ops: maximize vectorization (load 4× fp32 per thread), minimize kernel launches (fuse aggressively), use `evict_last` on reused small tensors.

For compute-bound ops: maximize tensor core utilization (BF16/FP8, aligned `BLOCK_M/N/K`), use TMA for async loading, tune tile shapes to fill SMEM.


---

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


---

## MATMUL Optimization Guide

### What makes fast matmul kernels fast

**1. TF32 tensor cores (10-15x, single line)**
`torch.backends.cuda.matmul.allow_tf32 = True` routes FP32 matmul through B200 tensor cores at 964 TFLOPS instead of CUDA cores at 481 TFLOPS. TF32 uses 10-bit mantissa, sufficient for rtol=1e-3. The baseline doesn't enable it.

**2. Avoiding unnecessary copies (1.5-5x)**
Removing `.contiguous()` on transposed inputs gives ~5x extra because cuBLAS handles transposes natively via transa/transb flags.

**3. Shape alignment to 128 multiples (1.5-4x)**
Padding irregular dimensions to multiples of 128. Tensor cores work on fixed tiles; irregular shapes waste computation at boundaries. Problem 8 (8205x2949): 2.1x -> 8.9x after padding.

### Common patterns
- Block 128x128 on B200. TF32 always enabled.
- 4D tensor matmul: reshape to 2D, matmul, reshape back (8.13x)
- Diagonal matmul: elementwise `diag * B` not matmul (2.28x)
- cuBLAS > custom CUDA for standard shapes

### B200 specifics
- TF32: 964 TFLOPS. BF16: 1929 TFLOPS. SMEM: 228KB. L2: 126MB.
- Tile alignment to 128 mandatory for full throughput
- cuBLAS selects sm100-native cutlass3x_sm100_tensorop kernels

### Typical speedups: 2.8-15.3x (TF32). Matrix-vector: 1.0x (memory-bound).

### Common mistakes
- BF16 fails correctness at large sizes (7-bit mantissa)
- Custom CUDA slower than cuBLAS for standard matmul
- torch.compile on bare matmul: no benefit


---

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


---

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


---

## OTHER Optimization Guide

### What makes fast other kernels fast

**1. Extern kernels for heavy compute, Triton for elementwise glue**
All examples delegate matmul (`mm`, `addmm`) and convolution to `extern_kernels` (cuBLAS/cuDNN), then use Triton *only* for fusing the lightweight ops (activations, bias add, residual add). This is correct because cuBLAS has hand-tuned Tensor Core paths that no Triton kernel at production shapes will beat for dense GEMM. The Triton layer exists to eliminate the extra memory roundtrips between ops.

**2. Elementwise fusion collapses memory traffic**
`fused_add_relu_threshold_backward` (SimpleBlock, BottleneckBlock) computes residual add + forward ReLU + backward mask in a single pass. Without fusion, this is 3 kernel launches and 3×N reads + 2×N writes. Fused: 1 launch, 2N reads + 2N writes. On B200 with 8 TB/s HBM, the bandwidth savings directly translate to speedup for these memory-bound ops.

**3. `evict_last` for broadcast loads**
When a small tensor (bias vector of size C) is broadcast across a large tensor (B×C×H×W), loading the same value N times would trash L1. `eviction_policy='evict_last'` pins those elements in cache. See MLP's `fused_gelu_0` — the bias (`in_ptr1`, size 4) is loaded with `evict_last` while the main buffer is streamed normally.

---

### Common code patterns

**Block sizes:**
- Tiny elementwise (≤64 elements): `XBLOCK=16`, 1 warp
- Small elementwise (64–512 elements): `XBLOCK=128`, `num_warps=4`
- Medium pointwise (512–4096): `XBLOCK=512`, `num_warps=8`
- Large pointwise: `XBLOCK=1024`, `num_warps=8`, `num_stages=1`

**In-place activation pattern** (zero extra buffer):
```python
# in_out_ptr0: reads and writes same buffer
tmp0 = tl.load(in_out_ptr0 + x0, xmask)
tmp1 = triton_helpers.maximum(tl.full([1], 0, tl.int32), tmp0)
tl.store(in_out_ptr0 + x0, tmp1, xmask)
```

**Dual output forward+backward in one pass:**
```python
# Compute relu output AND backward mask simultaneously
tmp4 = triton_helpers.maximum(tmp3, tmp2)   # forward: relu(x+residual)
tmp6 = tmp4 <= 0.0                           # backward: dead neuron mask
tl.store(out_ptr0 + x2, tmp4, xmask)        # save activation
tl.store(out_ptr1 + x2, tmp6, xmask)        # save grad mask
```
Eliminates a separate backward kernel launch during training.

**Conditional load for gather/cat ops (BehlerAngular pattern):**
```python
tmp4 = tmp0 < tmp3                    # branch on element index
tmp5 = tl.load(ptr + x1, tmp4 & mask, other=0.0)   # predicated load
```
Avoids warp divergence by using predicated loads instead of if/else branches.

---

### B200/Blackwell considerations

**Tensor Cores:** Only activated by `extern_kernels.mm`/`addmm`/`convolution`. For the Triton elementwise layer, Tensor Cores are irrelevant — these are pure memory-bandwidth ops. Don't try to use Tensor Cores for activation kernels.

**228KB SMEM:** Irrelevant for pure elementwise kernels (no tiling needed). Relevant only if you're reimplementing the matmul yourself — in that case tile to fit A and B tiles in SMEM. But default: use cuBLAS, not hand-rolled SMEM tiling.

**126MB L2:** "Other" ops at production batch sizes (B≥32, large hidden dims) will fit activation tensors in L2 across kernel launches. Design kernel launch sequences so consumers of a buffer run immediately after producers — L2 residency degrades with unrelated work between launches.

**NVLink/SM count:** B200 has 132 SMs. Kernels with fewer than ~132 blocks are underutilizing the GPU. At batch=4, hidden=4 (toy sizes in these examples), you're launching 1–4 blocks. At production: ensure `grid = ceil(N / XBLOCK)` yields ≥528 blocks (4× SM count) for full occupancy.

---

### Typical speedup range

| Pattern | Baseline (PyTorch eager) | Fused Triton | Notes |
|---|---|---|---|
| Bias + activation (ReLU/GELU/ELU) | 1.0× | 1.8–2.5× | Eliminates 1 read+write roundtrip |
| Add + ReLU + backward mask | 1.0× | 2.5–3.5× | 3-op → 1-pass |
| Cat + transform (BehlerAngular) | 1.0× | 1.5–2.0× | Avoids intermediate concat buffer |
| Full block (Conv + BN + ReLU + residual) | 1.0× | 1.2–1.6× | cuDNN handles conv; gains from elementwise tail |

At toy sizes (≤512 elements), measured speedup is unreliable due to launch overhead dominating. Real gains appear at B≥32.

---

### Common mistakes

**1. Reimplementing matmul in Triton** — `extern_kernels.mm` already dispatches to cuBLAS with Tensor Core paths tuned for B200. A custom Triton GEMM will be 0.5–0.8× the speed unless you're doing something exotic (e.g., fused attention with custom masking). Don't replace it.

**2. Separate kernels for chained elementwise ops** — `relu(add(conv(x), residual))` as three separate launches is 3× the memory traffic. The pattern in all examples is to fuse them into one Triton kernel.

**3. Forgetting `evict_last` for broadcast dimensions** — Loading a bias of size C into a (B×C×H×W) loop without pinning it causes repeated L1 evictions. Always annotate broadcast dimensions.

**4. Wrong XBLOCK for small tensors** — Using XBLOCK=1024 for a 16-element kernel launches 1/64th of a warp, wasting SM resources. Match XBLOCK to problem size: use the smallest power-of-2 ≥ problem size, capped at warp size (32) for tiny problems.

**5. Not fusing backward mask computation** — Computing the ReLU dead-neuron mask in a separate backward pass re-reads the activation buffer. The `threshold_backward` fusion pattern (BottleneckBlock, SimpleBlock) does this in the forward pass at zero extra cost.

---

### Memory-bound vs compute-bound

| Regime | Bottleneck | Crossover |
|---|---|---|
| Activation functions (ReLU, GELU, ELU) | Always memory-bound | N/A |
| Bias add, residual add, cat | Always memory-bound | N/A |
| Small matmul (M,N,K ≤ 256) | Memory-bound (GEMM overhead) | — |
| Medium matmul (M,N,K ~ 512–1024) | Transitioning | Depends on batch |
| Large matmul (M,N,K ≥ 2048) | Compute-bound (Tensor Cores) | — |
| Convolution (large spatial) | Compute-bound via cuDNN | Kernel size 3×3+ at C≥64 |

**Rule of thumb for "other" ops:** If the op is not a matmul or convolution, assume memory-bound and optimize for bandwidth: minimize reads/writes through fusion, use `evict_last` for broadcasts, keep kernel count low. Only investigate compute-boundedness if profiling shows >70% math throughput utilization.


---

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


---

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


---

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
