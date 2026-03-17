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
