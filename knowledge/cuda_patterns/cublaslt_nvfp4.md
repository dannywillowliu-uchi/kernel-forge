# cuBLASLt NVFP4 GEMM API Reference

Requires: CUDA 12.8+, cuBLAS 12.9+, Blackwell (sm100/sm120)

## Key types
- `CUDA_R_4F_E2M1` -- packed FP4, 2 elements per byte
- `__nv_fp8_e4m3` (UE4M3) -- block scale factors, one per 16 FP4 elements
- Scale mode: `CUBLASLT_MATMUL_MATRIX_SCALE_NVFP4_BLOCKWISE_16`

## Performance
- FP4 on GB200: 6787 TFLOPS (4.6x faster than H200 FP8)
- NVFP4 ~1.25x over MXFP8, ~1.65x over BF16

## Working sample
See: github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtNvfp4Matmul

## Scale factor layout
NVFP4 group_size = 16 (vs MXFP4 = 32)
5D interleaved tile layout: [M//128, K//16//4, 32, 16]
