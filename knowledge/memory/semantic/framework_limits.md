# Framework Limitations (Verified)

## torch.compile
- CANNOT fuse across matmul boundaries (LN + Linear stays as 2 kernels)
- CANNOT fuse LayerNorm + projection into one kernel
- CAN fuse elementwise chains (sigmoid, mask, multiply, cast, permute)
- max-autotune runs internal autotuning benchmark (~0.3-1s per matmul shape)
- dynamic=True avoids recompilation but generates slower generic code
- dynamic=False recompiles per shape (16 shapes = 16 compilations)
- reduce-overhead mode: skips autotuning, still fuses elementwise

## Triton
- Good for memory-bound kernels (within 5% of CUDA for simple patterns)
- tl.dot uses tensor cores for matmul
- Strided loads (offs * 3) are slower than float4 vectorized loads
- JIT compiles on first call (~1-3s), cached after
- @triton.autotune interferes with CUDA event timing (unreliable measurements)

## CuPy
- RawKernel launch overhead is same as PyTorch load_inline (NOT lower)
- Useful for simpler API, not for performance advantage

## cuBLAS
- Near-optimal for standard GEMM shapes on B200
- Custom Triton matmul is typically 30% slower than cuBLAS
- auto-selects CUTLASS sm100 tensorop kernels

## SOL-ExecBench specific
- "stream" keyword banned in Python source (even in comments)
- torch.compile triggers false "stream injection" detection for fast kernels
- Their eval server has ~2.5ms sync overhead (kills kernels < 0.05ms)
