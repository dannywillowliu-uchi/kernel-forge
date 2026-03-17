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
