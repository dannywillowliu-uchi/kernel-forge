# Precision Optimization Strategies

### fp16_bf16_computation
- **Category:** precision_opt
- **Applicability:** training and inference. BF16 default for LLM training (matches FP32 range). FP16 common in inference.
- **Expected Impact:** 2x bandwidth reduction vs FP32. 4-8x tensor core throughput vs FP32 CUDA cores. On B200: BF16 = 1,929 TFLOPS vs FP32 core = ~37 TFLOPS (51x via tensor cores).
- **Description:** Replace FP32 with 16-bit. BF16 (8-bit exp, 7-bit mantissa) = same range as FP32. FP16 (5-bit exp, 10-bit mantissa) = higher precision, narrower range. Always accumulate in FP32. On Blackwell: FP16 CUDA cores run at SAME rate as FP32; must use tensor cores for throughput. Use __half / __nv_bfloat16 types. For PyTorch: explicit model.bfloat16() preferred over autocast for inference (+47% throughput by removing dispatch overhead).

### fp8_quantization
- **Category:** precision_opt
- **Applicability:** LLM inference (H100+), training with careful scaling. Most beneficial for large batch inference.
- **Expected Impact:** 2x throughput vs BF16. 4x memory vs FP32. On B200: 3,851 TFLOPS.
- **Description:** E4M3 (higher precision) or E5M2 (wider range for gradients). Requires per-tensor or per-channel FP32 scaling factors. Blackwell supports MXFP8 (per-block-of-32 scale as E8M0). CUTLASS 3.x and Transformer Engine provide FP8 GEMM APIs. D = A(FP8) @ B(FP8) + C with FP32 accumulation. Activation quantization harder than weight quantization.

### nvfp4_quantization
- **Category:** precision_opt
- **Applicability:** LLM inference on Blackwell ONLY. Not for training.
- **Expected Impact:** 7,702 TFLOPS on B200 (2x FP8). 3.5x memory vs FP16. <1% accuracy degradation from FP8 with PTQ.
- **Description:** Blackwell-exclusive 4-bit format (E2M1: 1 sign, 2 exp, 1 mantissa, range [-6,6]). Two-level scaling: one E4M3 FP8 scale per 16-value block + one FP32 per-tensor. Tensor cores handle grouping/dequant/scaling in hardware during MMA. Stored as packed int8 (2 weights/byte). NVFP4-FP8-Dynamic (W4A8) optimal at large batch; NVFP4-W4A16 better at small batch. Use TensorRT-LLM or vLLM with Blackwell FP4 support.

### mixed_precision_accumulation
- **Category:** precision_opt
- **Applicability:** all neural network GEMM operations. Standard since Volta.
- **Expected Impact:** Maintains FP32 accuracy while achieving tensor core throughput.
- **Description:** Compute matmul in reduced precision (FP8/FP16/BF16) but accumulate partial sums in FP32. In CUTLASS: ElementAccumulator = float regardless of input type. Epilogue: FP32 accumulator -> elementwise ops -> convert to storage type. Loss scaling required for FP16 training (not BF16). Transformer Engine handles automatically.

### tensor_core_utilization
- **Category:** precision_opt
- **Applicability:** GEMM, batched GEMM, convolution, attention. Any computation expressible as matmul.
- **Expected Impact:** 4-16x vs CUDA cores same precision. On B200: BF16 tensor = 1,929 TFLOPS vs CUDA core = ~74 TFLOPS (26x).
- **Description:** Hardware MMA units computing D = A*B + C on fixed tile sizes. Hopper: wgmma.mma_async, 4 warpgroups (128 threads), max tile 64x256x16 BF16, results in registers. Blackwell: tcgen05.mma, single-thread dispatch, max tile 128x256x16 (1 SM) or 256x256x16 (2-SM CTA pair), results in TMEM. Tile alignment: M,N must be multiples of 128 on Blackwell. Use CUTLASS TiledMma. Must explicitly alloc/dealloc TMEM.
