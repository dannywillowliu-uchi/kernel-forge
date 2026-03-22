# B200 Hardware Facts (Measured)

## Peak Throughput
- FP32 CUDA cores: 481 TFLOPS
- TF32 tensor cores: 964 TFLOPS
- BF16 tensor cores: 1929 TFLOPS
- FP8 tensor cores: 3851 TFLOPS
- FP4 tensor cores: 7702 TFLOPS

## Memory
- HBM bandwidth (spec): 8 TB/s
- HBM bandwidth (measured copy): 6.6 TB/s
- HBM bandwidth (measured kernel, __ldg): 7.2 TB/s (read-heavy workloads exceed copy bandwidth)
- L2 cache: 126 MB
- SMEM per SM: 228 KB
- SMs: 160

## Clock
- Max boost: 1965 MHz
- SOL-ExecBench locks at: 1500 MHz (24% lower compute, same memory BW)

## CUDA event overhead
- Minimum CUDA event measurement: ~3.5-5us per call
- This creates a floor for timing very fast kernels

## Kernel launch overhead
- PyTorch load_inline: ~7-15us
- CuPy RawKernel: ~7-15us (same as load_inline, NOT lower)
- Triton @triton.jit: ~10-20us
- ctypes pre-compiled: ~7-15us (same)
- Launch overhead is dominated by CUDA driver, not Python framework
