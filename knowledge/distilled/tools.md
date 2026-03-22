# GPU Optimization Toolchain Guide

Available tools on B200 and when to use each one.

## Kernel Writing (by launch overhead, lowest first)

### 1. CuPy RawKernel (lowest launch overhead)
```python
import cupy as cp

kernel = cp.RawKernel(r'''
extern "C" __global__ void grayscale(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = i * 3;
        out[i] = 0.2989f * in[j] + 0.5870f * in[j+1] + 0.1140f * in[j+2];
    }
}
''', 'grayscale')

# Launch with minimal Python overhead
kernel((grid,), (block,), (in_ptr, out_ptr, n))
```
**Use when:** Kernel is fast but launch overhead dominates (small tensors, simple ops).
**Why:** CuPy compiles once and caches. Launch path is shorter than PyTorch's load_inline.

### 2. Pre-compiled CUDA via ctypes (zero Python dispatch)
```python
import ctypes, torch

# Compile once: nvcc -shared -o kernel.so kernel.cu -O3
lib = ctypes.CDLL('./kernel.so')
lib.launch_grayscale.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

# Launch with raw pointer -- minimal overhead
lib.launch_grayscale(x.data_ptr(), y.data_ptr(), n)
```
**Use when:** Absolute minimum launch overhead needed. Kernel changes are rare.
**Why:** No Python framework in the dispatch path at all.

### 3. Triton (@triton.jit)
```python
import triton
import triton.language as tl

@triton.jit
def kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < N)
    tl.store(Y + offs, x, mask=offs < N)
```
**Use when:** Memory-bound kernels, reductions, elementwise fusion. Good balance of control and iteration speed.
**Compile time:** ~1-3 seconds first call, cached after.
**Launch overhead:** ~10-20us per call.

### 4. Raw CUDA via load_inline
```python
from torch.utils.cpp_extension import load_inline
module = load_inline(name="op", cuda_sources=src, cpp_sources=cpp,
                     functions=["op"], extra_cuda_cflags=["-O3", "--use_fast_math"])
```
**Use when:** Need full CUDA control -- shared memory, warp intrinsics, vectorized loads.
**Compile time:** ~30-90 seconds first call.
**Launch overhead:** ~15-25us per call (through PyTorch C++ extension).

### 5. torch.compile
```python
@torch.compile(mode="max-autotune-no-cudagraphs", fullgraph=True)
def fn(x): return x * 2 + 1
```
**Use when:** Fusing elementwise chains, reducing kernel launches for compound operations.
**Limitation:** Cannot fuse across matmul boundaries. Cannot fuse LayerNorm + Linear.
**Compile time:** 1-30 seconds depending on graph complexity.

## Optimization Libraries

### Liger Kernel
```python
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
# Or: from liger_kernel.transformers import LigerRMSNorm
```
**Use when:** Fused LayerNorm/RMSNorm, fused cross-entropy, fused SwiGLU.
**Speedup:** 2-8x over PyTorch for normalization ops.

### cuEquivariance
```python
import cuequivariance_torch as cuet
from cuequivariance_ops_torch.triangle_multiplicative_update import triangle_multiplicative_update
```
**Use when:** Triangle multiplicative update (AlphaFold3), equivariant operations.
**Caveat:** Architecture must match exactly (check gate ordering). Needs `cuequivariance-ops-torch-cu12`.

### CUTLASS (via cutlass_library)
```python
import cutlass_library
```
**Use when:** Custom GEMM tiles, split-K, grouped GEMM. For shapes where cuBLAS picks suboptimal configs.

## Profiling Tools

### torch.profiler (quick overview)
```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    fn()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="self_cuda_time_total"))
```

### ncu (detailed kernel analysis)
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed python3 script.py
```

### nsys (timeline)
```bash
nsys profile -o out python3 script.py && nsys stats out.nsys-rep
```

## Common Patterns

### Reducing kernel launch overhead
When your kernel runs in <50us but launch overhead is 7-15us:
1. **Batch multiple ops into one kernel** -- process all sizes in a single launch
2. **Use CuPy RawKernel** instead of load_inline or Triton
3. **CUDA graphs** -- capture and replay the launch sequence (requires fixed tensor addresses)
4. **Persistent kernel** -- one launch that processes all work items via grid-stride loop

### Maximizing HBM bandwidth
When your kernel is memory-bound:
1. **Vectorized loads** -- `float4` reads 16 bytes per thread (128-bit transactions)
2. **Coalesced access** -- consecutive threads read consecutive addresses
3. **`__ldg()`** -- read-only texture cache path
4. **Minimize read/write ratio** -- fuse operations to reduce intermediate materializations

### Breaking torch.compile ceilings
When torch.compile can't fuse across boundaries:
1. Write **custom Triton kernels** for the fused operation
2. Use **library kernels** (Liger, cuEquivariance) that are pre-fused
3. Write **raw CUDA** for maximum control over memory layout
4. Profile to verify the custom kernel actually beats torch.compile (often it doesn't)
