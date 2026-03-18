// Pattern: Fused RMSNorm + Quantization (from WaferBench)
// Use for: inference pipelines where norm is followed by quantization
// Speedup: 1.3-2x by eliminating intermediate BF16 tensor materialization
// Key: compute norm and quantize in the same kernel, one HBM read

// From Gemini-3.1-Pro's WaferBench submission:
template<int BLOCK_SIZE>
__global__ void fused_add_rmsnorm_fp4quant(
    const __nv_bfloat16* __restrict__ input,    // [B, H]
    const __nv_bfloat16* __restrict__ residual,  // [B, H]
    const __nv_bfloat16* __restrict__ weight,    // [H]
    uint8_t* __restrict__ fp4_output,            // [B, H/2] packed
    __nv_fp8_e4m3* __restrict__ block_scales,    // [B, H/BLOCK_SIZE]
    float global_scale,
    int B, int H, float eps
) {
    int row = blockIdx.x;
    extern __shared__ float smem[];

    // Step 1: Fused add + compute sum-of-squares for RMS
    float thread_sq_sum = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float x = __bfloat162float(input[row * H + i])
                + __bfloat162float(residual[row * H + i]);
        smem[i] = x;  // store in shared memory for reuse
        thread_sq_sum += x * x;
    }

    // Step 2: Block-level reduction for RMS
    float rms_sq = block_reduce_sum<BLOCK_SIZE>(thread_sq_sum);
    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) {
        s_rms_inv = rsqrtf(rms_sq / H + eps);
    }
    __syncthreads();

    // Step 3: Fused normalize + quantize to FP4
    float rms_inv = s_rms_inv;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float normalized = smem[i] * rms_inv
                        * __bfloat162float(weight[i]);
        // Quantize to FP4 in-place (no intermediate write)
        // ... (use fp4_encoding pattern)
    }
}

// Key optimization: shared memory holds the add+residual result
// so we read input+residual from HBM once, write FP4 output once.
// Without fusion: read input+residual, write BF16 norm output,
// read BF16 norm output, write FP4 output = 2x the memory traffic.
