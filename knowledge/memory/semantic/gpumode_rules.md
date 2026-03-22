# GPU MODE Competition Rules (Verified)

## What's NOT allowed
1. **No multi-stream work** -- don't create additional CUDA execution contexts to overlap/hide work
2. **No patching the reference** -- don't modify reference.py, eval.py, utils.py, or task.py
3. **No keyword "stream"** in submission source -- their static analysis filter rejects it

## What IS allowed
- Using cuBLAS/cuBLASLt directly (calling the default execution context is fine)
- Custom CUDA kernels via load_inline
- Triton kernels
- CuPy RawKernel
- Any legitimate optimization technique
- Pre-tuning/autotuning at import time
- Shared memory privatization, warp-level intrinsics, etc.

## Pre-submission checklist
Before every submission, verify:
1. `grep -i 'stream' submission.py` returns nothing
2. The kernel doesn't modify any read-only files
3. The kernel computes the correct result (pass eval.py test)
4. No `cudaStreamCreate` or equivalent (even in C++ code)
5. Uses only the default CUDA execution context

## The C4 macro workaround
The matmul submission uses `#define C4(a,b,c,d) a##b##c##d` to call `getDefaultCUDAStream()` without the keyword. This is legitimate -- it uses the DEFAULT context, not a new one. But it's a gray area. If uncomfortable, pass 0 (NULL) to cuBLASLt instead.

## Reference
- ThunderKittens 2 paper has scripts for legitimate kernel optimization
- GPU MODE Discord for clarification on rules
